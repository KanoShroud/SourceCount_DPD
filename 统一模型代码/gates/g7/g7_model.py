"""G7 shared local refiner; x/y in metres, tensor row/column = y/x.

Only the selector's physical path can backpropagate localization into queries.
Candidate windows are externally supplied predictions, never GT-centred.
"""
from __future__ import annotations

import itertools

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.nn import functional as F


DEFAULT_LOSS_WEIGHTS = dict(exist=1., band=1., heatmap=1., offset=1., candidate=1.)


class LocalFrequencySelector(nn.Module):
    """Load ``selector`` from P2-B; zero init is only an initialization fallback."""

    def __init__(self):
        super().__init__()
        self.selector = nn.Linear(128, 19)
        nn.init.zeros_(self.selector.weight)
        nn.init.zeros_(self.selector.bias)

    def weights(self, query, band_logits, arm):
        arm = arm.upper()
        if arm == "F":
            return torch.ones_like(band_logits)
        if arm not in ("S", "E"):
            raise ValueError(arm)
        source = query.detach() if arm == "S" else query
        return (band_logits.detach() + self.selector(source)).sigmoid()


class LocalRefiner(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2, channels, 3, padding=1),
            nn.ReLU(), nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.condition = nn.Linear(128, channels * 2)
        self.fusion = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU())
        self.heatmap = nn.Conv2d(channels, 1, 1)
        self.offset = nn.Conv2d(channels, 2, 1)
        self.candidate = nn.Linear(channels, 1)
        nn.init.constant_(self.heatmap.bias, -2.19)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def forward(self, maps, query, valid_mask):
        b, q, p, h, w = maps.shape
        if valid_mask.shape != maps.shape or query.shape != (b, q, 128):
            raise ValueError("Local map, mask or query shape mismatch")
        valid = valid_mask.to(maps.dtype)
        x = torch.stack([maps * valid, valid], dim=3).reshape(-1, 2, h, w)
        feature = self.encoder(x)
        gamma, beta = self.condition(query.detach()).chunk(2, -1)
        gamma = gamma[:, :, None].expand(-1, -1, p, -1).reshape(-1, feature.shape[1], 1, 1)
        beta = beta[:, :, None].expand(-1, -1, p, -1).reshape_as(gamma)
        feature = self.fusion(feature * (1 + gamma) + beta)
        flat_valid = valid.reshape(-1, 1, h, w)
        pooled = (feature * flat_valid).sum((-2, -1)) / flat_valid.sum((-2, -1)).clamp_min(1)
        return dict(heat_logits=self.heatmap(feature).reshape(b, q, p, h, w),
                    offset=self.offset(feature).reshape(b, q, p, 2, h, w),
                    candidate_logits=self.candidate(pooled).reshape(b, q, p))


def _target_location(position, centers, valid, step_m):
    h, w = valid.shape[-2:]
    xy = (position[None] - centers) / step_m
    xy = xy + xy.new_tensor([(w - 1) / 2, (h - 1) / 2])
    rounded = xy.round().long()
    covered = ((xy[:, 0] >= 0) & (xy[:, 0] <= w - 1)
               & (xy[:, 1] >= 0) & (xy[:, 1] <= h - 1))
    ix, iy = rounded[:, 0].clamp(0, w - 1), rounded[:, 1].clamp(0, h - 1)
    covered &= valid[torch.arange(len(centers), device=valid.device), iy, ix].bool()
    return xy, ix, iy, covered


def local_loss(outputs, band_logits, centers_m, valid_mask, target_positions,
               target_bands, target_ignore=None, config=None):
    """Targets are lists of Kx2/Kx19 tensors (unpadded); returns total, parts, info.

    config: loss_weights (five keys), step_m=10, gaussian_sigma=2.0.
    Assignment follows existing band BCE + GT-location heat confidence cost.
    Matched sources outside every window retain band/existence supervision only.
    """
    config = config or {}
    step = float(config.get("step_m", 10.))
    sigma = float(config.get("gaussian_sigma", 2.0))
    weights = config.get("loss_weights", DEFAULT_LOSS_WEIGHTS)
    if set(weights) != set(DEFAULT_LOSS_WEIGHTS):
        raise ValueError("G7 loss weights must include exist/band/heatmap/offset/candidate")
    heat, offset, scores = (outputs[k] for k in ("heat_logits", "offset", "candidate_logits"))
    device = band_logits.device
    b, nq, npatch, h, w = heat.shape
    valid = valid_mask.bool()
    centers_m = centers_m.to(device)
    band_target = torch.zeros_like(band_logits)
    band_valid = torch.ones_like(band_logits)
    exist_target = torch.zeros_like(band_logits[..., 0])
    heat_target = torch.zeros_like(heat)
    heat_valid = valid.clone()
    score_target = torch.zeros_like(scores)
    score_valid = valid.flatten(-2).any(-1)
    offset_losses, mappings = [], []
    covered_sources, total_sources = 0, 0
    yy, xx = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    for bi in range(b):
        pos = torch.as_tensor(target_positions[bi], device=device, dtype=heat.dtype).reshape(-1, 2)
        bands = torch.as_tensor(target_bands[bi], device=device, dtype=band_logits.dtype).reshape(-1, 19)
        ignores = (torch.zeros_like(bands) if target_ignore is None else
                   torch.as_tensor(target_ignore[bi], device=device, dtype=bands.dtype).reshape(-1, 19))
        k = len(pos)
        if k > nq or len(bands) != k:
            raise ValueError("Invalid target source count")
        total_sources += k
        locations = {}
        cost = np.zeros((nq, k), dtype=np.float64)
        with torch.no_grad():
            for qi in range(nq):
                for ti in range(k):
                    xy, ix, iy, covered = _target_location(pos[ti], centers_m[bi, qi], valid[bi, qi], step)
                    locations[qi, ti] = xy, ix, iy, covered
                    bv = (ignores[ti] < .5).to(bands.dtype)
                    bc = (F.binary_cross_entropy_with_logits(band_logits[bi, qi], bands[ti], reduction="none") * bv).sum() / bv.sum().clamp_min(1)
                    confidence = heat[bi, qi, torch.arange(npatch, device=device), iy, ix].sigmoid()
                    lc = 1 - confidence[covered].max() if covered.any() else heat.new_tensor(1.)
                    cost[qi, ti] = float(bc + lc)
        rows, cols = linear_sum_assignment(cost)
        mapping = dict(zip(rows.tolist(), cols.tolist()))
        mappings.append(mapping)
        for qi, ti in mapping.items():
            exist_target[bi, qi] = 1
            band_target[bi, qi] = bands[ti]
            band_valid[bi, qi] = 1 - ignores[ti]
            xy, ix, iy, covered = locations[qi, ti]
            if not covered.any():
                heat_valid[bi, qi] = False
                score_valid[bi, qi] = False
                continue
            covered_sources += 1
            score_target[bi, qi] = covered.to(scores.dtype)
            for pi in covered.nonzero().flatten().tolist():
                gaussian = torch.exp(-((xx - xy[pi, 0]) ** 2 + (yy - xy[pi, 1]) ** 2) / (2 * sigma ** 2))
                gaussian = gaussian * valid[bi, qi, pi]
                heat_target[bi, qi, pi] = gaussian / gaussian.max().clamp_min(1e-12)
                offset_losses.append((offset[bi, qi, pi, :, iy[pi], ix[pi]] - (xy[pi] - torch.stack([ix[pi], iy[pi]]))).abs().sum())
    p = heat.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    focal = -(heat_target * (1-p).square() * p.log()
              + (1-heat_target).pow(4) * p.square() * (1-p).log())
    npos = ((heat_target > .5) & heat_valid).sum()
    heat_den = npos if int(npos) else heat_valid.sum().clamp_min(1)
    band_bce = F.binary_cross_entropy_with_logits(band_logits, band_target, reduction="none")
    score_bce = F.binary_cross_entropy_with_logits(scores, score_target, reduction="none")
    parts = dict(exist=F.binary_cross_entropy_with_logits(band_logits.amax(-1), exist_target),
                 band=(band_bce * band_valid).sum() / band_valid.sum().clamp_min(1),
                 heatmap=(focal * heat_valid).sum() / heat_den,
                 offset=torch.stack(offset_losses).mean() if offset_losses else offset.sum() * 0,
                 candidate=(score_bce * score_valid).sum() / score_valid.sum().clamp_min(1))
    total = sum(float(weights[name]) * value for name, value in parts.items())
    return total, parts, dict(mappings=mappings, covered_sources=covered_sources, total_sources=total_sources)


@torch.no_grad()
def decode_local(outputs, centers_m, valid_mask, band_logits, step_m=10., separation_m=30., threshold=.5, counts=None):
    """One local maximum/window; exhaustive joint choice for active semantic slots.

    Score = log(sigmoid(candidate)) + log(sigmoid(local peak)); no new K head.
    When no separation-feasible tuple exists, preserve independent predictions
    with an explicit fallback flag, as in the historical joint decoder.
    """
    heat, offset, scores = (outputs[k] for k in ("heat_logits", "offset", "candidate_logits"))
    b, nq, npatch, h, w = heat.shape
    masked = heat.masked_fill(~valid_mask.bool(), -torch.inf)
    flat_idx = masked.flatten(-2).argmax(-1)
    iy, ix = flat_idx // w, flat_idx % w
    all_rows = []
    for bi in range(b):
        active = (band_logits[bi].sigmoid().amax(-1) >= threshold).nonzero().flatten().tolist()
        if counts is not None:
            active = sorted(torch.argsort(band_logits[bi].amax(-1), descending=True,
                stable=True)[:int(counts[bi])].tolist())
        candidates = []
        for qi in active:
            points, values, patch_indices = [], [], []
            for pi in range(npatch):
                if not valid_mask[bi, qi, pi].any():
                    continue
                y, x = int(iy[bi, qi, pi]), int(ix[bi, qi, pi])
                delta = offset[bi, qi, pi, :, y, x].clamp(-.5, .5)
                local_xy = delta + delta.new_tensor([x, y])
                # Bound subpixel outputs to the valid rectangular world slice.
                vy, vx = valid_mask[bi, qi, pi].nonzero(as_tuple=True)
                local_xy[0].clamp_(float(vx.min()), float(vx.max()))
                local_xy[1].clamp_(float(vy.min()), float(vy.max()))
                point = centers_m[bi, qi, pi] + (local_xy - local_xy.new_tensor([(w-1)/2, (h-1)/2])) * step_m
                value = F.logsigmoid(scores[bi, qi, pi]) + F.logsigmoid(heat[bi, qi, pi, y, x])
                points.append(point.cpu().tolist())
                values.append(float(value))
                patch_indices.append(pi)
            if not points:
                raise ValueError("Active slot has no valid candidate window")
            candidates.append(dict(positions=points, scores=np.exp(values).tolist(),
                                   log_scores=values, patch_indices=patch_indices))
        independent = [int(np.argmax(c['log_scores'])) for c in candidates]
        best = None
        for ranks in itertools.product(*(range(len(c['positions'])) for c in candidates)):
            points = np.asarray([c['positions'][r] for c, r in zip(candidates, ranks)], dtype=np.float64).reshape(-1, 2)
            if any(np.linalg.norm(a-z) < separation_m for a, z in itertools.combinations(points, 2)):
                continue
            value = sum(c['log_scores'][r] for c, r in zip(candidates, ranks))
            if best is None or value > best[0]:
                best = value, list(ranks)
        ranks = best[1] if best is not None else independent
        points = [c['positions'][r] for c, r in zip(candidates, ranks)]
        patches = [c['patch_indices'][r] for c, r in zip(candidates, ranks)]
        all_rows.append(dict(active=active, active_slots=active, joint=points, positions_m=points,
            candidates=candidates, selected_ranks=[r+1 for r in ranks], candidate_indices=patches,
            scores=[c['scores'][r] for c, r in zip(candidates, ranks)],
            original=[c['positions'][r] for c, r in zip(candidates, independent)], fallback=best is None))
    return all_rows
