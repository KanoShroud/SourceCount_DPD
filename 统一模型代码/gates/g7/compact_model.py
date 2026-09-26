"""Size-explicit P2-B adaptation and G7 local interfaces; historical code stays frozen."""
import itertools

import numpy as np
import torch
from torch.nn import functional as F
from scipy.ndimage import maximum_filter

from 统一模型代码.gates.g7.compact_data import EDGE, FINE_N, STEP, network_inputs
from 统一模型代码.models.e2e_latent_fusion import forward_ch3_features
from 统一模型代码.gates.g5.e2e_g5_model import build_context, g4
from 统一模型代码.gates.g6.g6_p2_model import SplitPhysical
from 统一模型代码.gates.g6.g6_p1_speed import physical_maps
from 统一模型代码.gates.g7.g7_model import LocalRefiner, LocalFrequencySelector, decode_local, local_loss


def candidates(heat, top_k=5):
    """Per-slot predicted grid centers, never offsets or GT; deterministic tie order."""
    if heat.shape[1:] != (3, FINE_N, FINE_N) or top_k not in (5, 8):
        raise ValueError('Wrong candidate grid/top-k')
    if not torch.isfinite(heat).all():
        raise ValueError('Nonfinite candidate heatmap')
    records = []
    for h in heat.detach().sigmoid().cpu().numpy():
        queries = []
        for q in range(3):
            mask = (h[q] == maximum_filter(h[q], size=7, mode='constant')) & (h[q] > 0)
            ids = np.flatnonzero(mask)
            ids = ids[np.argsort(-h[q].ravel()[ids], kind='stable')[:top_k]]
            if not len(ids):
                raise ValueError('No finite positive candidate')
            queries.append(dict(query=q, flat_indices=ids.tolist(), scores=h[q].ravel()[ids].tolist()))
        records.append(dict(candidates=queries))
    return records


def windows(record, top_k=5, size=41):
    if size != 41 or top_k not in (5, 8):
        raise ValueError('Unapproved local configuration')
    centers = torch.zeros(3, top_k, 2, dtype=torch.float64)
    exists = torch.zeros(3, top_k, dtype=torch.bool)
    for q, row in enumerate(record['candidates']):
        for j, flat in enumerate(row['flat_indices'][:top_k]):
            y, x = divmod(int(flat), FINE_N)
            if not (0 <= x < FINE_N and 0 <= y < FINE_N):
                raise ValueError('Flat candidate index outside compact grid')
            centers[q, j] = torch.tensor([x*STEP-EDGE, y*STEP-EDGE])
            exists[q, j] = True
    axis = (torch.arange(size, dtype=torch.float64) - (size-1)/2)*STEP
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    points = centers[..., None, None, :] + torch.stack((x, y), -1)
    valid = (points.abs() <= EDGE).all(-1) & exists[..., None, None]
    unique, inverse = torch.unique(points.clamp(-EDGE, EDGE).reshape(-1, 2), dim=0, return_inverse=True)
    return centers, valid, unique, inverse.reshape(3, top_k, size, size)


def batch_inputs(records, device):
    pairs = [network_inputs(r['coarse'].numpy(), r['fine'].numpy()) for r in records]
    return tuple(torch.stack([p[i] for p in pairs]).to(device) for i in range(2))


class CompactModel:
    """Use identical P2 modules/weights; recompute prefixes on small raw inputs."""
    def __init__(self, out, manifest, seed, initial, device='cuda'):
        from 统一模型代码.gates.g6.g6_p2_train import restore
        self.context = build_context(out, manifest, seed, torch.device(device))
        self.physical = SplitPhysical('b').to(device)
        restore(self.context, self.physical, initial['state'])
        self.device = torch.device(device)
        torch.manual_seed(seed + 2000)
        self.selector = LocalFrequencySelector().to(device)
        self.selector.selector.load_state_dict(self.physical.selector.state_dict())
        self.refiner = LocalRefiner().to(device)

    def state(self):
        return dict(base=g4.state_payload(self.context), physical=self.physical.state_dict(),
                    selector=self.selector.state_dict(), refiner=self.refiner.state_dict())

    def restore(self, state):
        g4.load_state(self.context, state['base'])
        for name in ('physical', 'selector', 'refiner'):
            getattr(self, name).load_state_dict(state[name], strict=True)

    def mode(self, training):
        g4.set_mode(self.context, training=training)
        for m in (self.physical, self.selector, self.refiner):
            m.train(training)

    def optimizer(self, phase, manifest):
        groups = list(self.context.parameter_groups)
        if phase == 'baseline':
            groups.append(dict(params=list(self.physical.parameters()), lr=1e-4, name='physical'))
        else:
            # D8/splitter are absent from local forward; do not maintain optimizer state for them.
            groups = [dict(params=list(self.context.ch3.band_heads[:3].parameters())+
                           list(self.context.query_builder.parameters()), lr=1e-4, name='query'),
                      dict(params=list(self.context.ch3.cross_attn.parameters()), lr=1e-5, name='ch3_tail'),
                      dict(params=list(self.selector.parameters())+list(self.refiner.parameters()),
                           lr=1e-4, name='local')]
        return torch.optim.AdamW(groups, weight_decay=manifest['config']['weight_decay'])

    def query(self, records):
        pairs = [network_inputs(r['coarse'].numpy(), r['fine'].numpy())[0] for r in records]
        features = forward_ch3_features(self.context.ch3, torch.stack(pairs).to(self.device))
        query, logits = self.context.query_builder(features)
        return features, query, logits

    def baseline(self, records, physics):
        c, f = batch_inputs(records, self.device)
        current = forward_ch3_features(self.context.ch3, c)
        query, logits = self.context.query_builder(current)
        spatial, attention = self.context.splitter(current.spatial, query.detach(), logits.detach())
        with torch.no_grad():
            e1, d2 = g4.d8_prefix(self.context.d8, f)
            up1 = self.context.d8.decoder.up1(d2)
        d1 = self.context.d8.decoder.c1(torch.cat([up1, e1], 1))
        d0 = self.context.d8.decoder.up0(d1)[..., :FINE_N, :FINE_N]
        heat, offset = self.context.source_head(d0, spatial, query.detach())
        weights = self.physical.weights(query, logits, 'b')
        maps = physical_maps(physics, weights, [r['statistics'] for r in records])
        b, q, h, w = maps.shape
        residual = self.physical.physical.net(maps.float().reshape(b*q, 1, h, w))
        heat = heat + F.interpolate(residual, size=(FINE_N, FINE_N), mode='bilinear',
                                    align_corners=True).reshape(b, q, FINE_N, FINE_N)
        return dict(query=query, band_logits=logits, heat=heat, offset=offset, weights=weights)

    def local(self, records, physics, stats, arm):
        _, query, logits = self.query(records)
        weights = self.selector.weights(query, logits, arm)
        maps = (torch.stack([r['full_maps'] for r in stats]).to(self.device) if arm == 'f'
                else local_maps(physics, weights, stats))
        centers = torch.stack([r['centers_m'] for r in stats]).to(self.device).float()
        valid = torch.stack([r['valid_mask'] for r in stats]).to(self.device)
        output = self.refiner(maps, query, valid)
        return dict(query=query, band_logits=logits, centers_m=centers, valid_mask=valid, output=output)


def local_maps(physics, probabilities, records, full=False):
    """Original FP64 G7 matrix/EVD, with candidate count inferred from index shape."""
    from 统一模型代码.gates.g6.coherent_dpd import spatial_evd
    from 统一模型代码.gates.g7.g7_physics import normalize
    output = []
    for probabilities_i, record in zip(probabilities, records):
        device = probabilities_i.device
        energy, coherent, inverse = (record[k].to(device) for k in ('energy', 'coherent', 'inverse'))
        if full:
            weights = torch.ones(3, len(energy), dtype=torch.float64, device=device)
        else:
            weights = physics.weights(probabilities_i.double())
            if len(energy) > weights.shape[-1]:
                weights = torch.cat((weights, torch.zeros(3, 1, device=device, dtype=torch.float64)), -1)
        maps = []
        for q in range(3):
            unique, gather = torch.unique(inverse[q].flatten(), return_inverse=True)
            a = weights[q]
            e = a @ energy
            den = (e/physics.geo.N0**2).clamp_min(1e-20).sqrt()
            cross = torch.einsum('r,rgp->gp', a.to(torch.complex128), coherent[:, unique])
            matrix = torch.diag((e/den.square()).to(torch.complex128))[None].expand(len(unique), -1, -1).clone()
            for p, (m, n) in enumerate(physics.geo.pairs):
                v = cross[:, p]/(den[m]*den[n])
                matrix[:, m, n], matrix[:, n, m] = v, v.conj()
            matrix = matrix+1e-6*torch.eye(4, dtype=torch.complex128, device=device)
            values = spatial_evd(matrix[None], chunk_size=256)[0]
            maps.append(values[gather].reshape(inverse[q].shape))
        output.append(normalize(torch.stack(maps), record['valid_mask']))
    return torch.stack(output).float()


def baseline_loss(result, records):
    """Original G2-R1 set loss, changing only grid dimensions/origin."""
    logits, heat, offset = (result[k] for k in ('band_logits', 'heat', 'offset'))
    bt, bv = torch.zeros_like(logits), torch.ones_like(logits)
    et, ht = torch.zeros_like(logits[..., 0]), torch.zeros_like(heat)
    offsets = []
    axis = torch.arange(FINE_N, device=heat.device, dtype=heat.dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    for bi, r in enumerate(records):
        k = r['count']
        bands, ignore = r['band'].to(logits), r['ignore'].to(logits)
        xy = (r['positions'].to(heat)+EDGE)/STEP
        grid = xy.round().long().clamp(0, FINE_N-1)
        cost = np.zeros((3, k))
        for q in range(3):
            for t in range(k):
                valid = ignore[t] < .5
                bc = F.binary_cross_entropy_with_logits(logits[bi, q, valid], bands[t, valid])
                cost[q, t] = float((bc+1-heat[bi, q, grid[t, 1], grid[t, 0]].sigmoid()).detach())
        order = min(itertools.permutations(range(3), k), key=lambda p: sum(cost[p[t], t] for t in range(k)))
        for t, q in enumerate(order):
            et[bi, q] = 1
            bt[bi, q], bv[bi, q] = bands[t], 1-ignore[t]
            target = torch.exp(-((xx-xy[t, 0]).square()+(yy-xy[t, 1]).square())/(2*g4.g1.GAUSS_SIGMA**2))
            ht[bi, q] = target/target.max().clamp_min(1e-12)
            ix, iy = grid[t]
            offsets.append((offset[bi, q, :, iy, ix]-(xy[t]-grid[t])).abs().sum())
    parts = dict(exist=F.binary_cross_entropy_with_logits(logits.amax(-1), et),
        band=(F.binary_cross_entropy_with_logits(logits, bt, reduction='none')*bv).sum()/bv.sum().clamp_min(1),
        heatmap=g4.r1.focal_loss_hm(heat.reshape(-1, 1, FINE_N, FINE_N), ht.reshape(-1, 1, FINE_N, FINE_N)),
        offset=torch.stack(offsets).mean() if offsets else offset.sum()*0)
    return sum(parts.values()), parts


def loss(result, records, phase):
    if phase == 'baseline':
        return baseline_loss(result, records)
    fields = [[r[name][:r['count']] for r in records] for name in ('positions', 'band', 'ignore')]
    total, parts, _ = local_loss(result['output'], result['band_logits'], result['centers_m'],
                                 result['valid_mask'], *fields, config=dict(step_m=10., gaussian_sigma=2.))
    return total, parts


def decode(result, phase, top_k=8):
    if phase != 'baseline':
        return decode_local(result['output'], result['centers_m'], result['valid_mask'], result['band_logits'])
    records = candidates(result['heat'], top_k)
    b = len(records)
    heat = result['heat'].new_full((b, 3, top_k, 1, 1), -100.)
    offset = heat.new_zeros(b, 3, top_k, 2, 1, 1)
    centers = heat.new_zeros(b, 3, top_k, 2)
    valid = torch.zeros_like(heat, dtype=torch.bool)
    # Decode global offsets directly; local decoder's one-pixel clipping would discard them.
    for bi, r in enumerate(records):
        for q, row in enumerate(r['candidates']):
            for j, flat in enumerate(row['flat_indices']):
                y, x = divmod(flat, FINE_N)
                delta = result['offset'][bi, q, :, y, x].clamp(-1., 1.)
                centers[bi, q, j] = ((delta+delta.new_tensor([x, y]))*STEP-EDGE).clamp(-EDGE, EDGE)
                heat[bi, q, j, 0, 0] = result['heat'][bi, q, y, x]
                valid[bi, q, j] = True
    output = dict(heat_logits=heat, offset=offset, candidate_logits=heat.new_zeros(b, 3, top_k))
    return decode_local(output, centers, valid, result['band_logits'])
