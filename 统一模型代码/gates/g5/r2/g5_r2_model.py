"""R2唯一新模块；候选与坐标不求导，身份特征保持可微。"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.ndimage import maximum_filter
import torch
from torch import nn
from torch.nn import functional as F

from 统一模型代码.gates.g5.e2e_g5_model import forward, g4
from 统一模型代码.gates.g5.r1.g5_r1_decode import positions_from_indices


class AssociationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(128, 16)
        self.mlp = nn.Sequential(nn.Linear(451, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, spatial, query, logits, points):
        # CH3: 5/s2/p2, 3/s2/p1, 3/s2/p1. Centers 0,8,...,80
        # in the 81-grid; physical centers -2000,-1600,...,2000 m.
        if spatial.shape != (19, 128, 11, 11):
            raise ValueError('Unexpected CH3 feature geometry')
        local = sample_centers(spatial, points)  # [N,19,128]
        n = len(points)
        joined = torch.cat((self.project(local).flatten(1), query.expand(n, -1),
                            logits.sigmoid().expand(n, -1)), dim=-1)
        return self.mlp(joined).squeeze(-1)


def sample_centers(spatial, points):
    grid = points.detach().to(spatial).div(2000).view(1, -1, 1, 2)
    sampled = F.grid_sample(spatial, grid.expand(len(spatial), -1, -1, -1),
                            mode='bilinear', padding_mode='border', align_corners=True)
    return sampled[..., 0].permute(2, 0, 1)


def candidate_batch(heat, offset):
    heat = heat.detach().sigmoid().cpu().numpy()
    offset = offset.detach().cpu().numpy()
    if not np.isfinite(heat).all() or not np.isfinite(offset).all():
        raise ValueError('Nonfinite candidate input')
    records = []
    # Same R1 candidates, but no unnecessary 8^3 joint enumeration in training.
    for h,o in zip(heat,offset):
        candidates,original = [],[]
        for q in range(3):
            original.append(positions_from_indices([h[q].argmax()],o[q],401)[0].tolist())
            mask = (h[q] == maximum_filter(h[q],size=7,mode='constant')) & (h[q]>0)
            ids = np.flatnonzero(mask)
            ids = ids[np.argsort(-h[q].ravel()[ids],kind='stable')[:8]]
            candidates.append({'query':q,'flat_indices':ids.tolist(),
                               'scores':h[q].ravel()[ids].astype(float).tolist(),
                               'positions':positions_from_indices(ids,o[q],401).tolist()})
        records.append({'candidates':candidates,'original':original})
    return records


def forward_r2(context, head, features, ids, device, arm):
    query, logits, attention, heat, offset = forward(
        context, features, ids, device, stop_gradient=arm != 'c2')
    candidates = [] if arm == 'c0' and torch.is_grad_enabled() else candidate_batch(heat, offset)
    scores = []
    if head is not None:
        spatial = g4.numpy_batch(features.spatial, ids, device)
        q, b = (query, logits) if arm == 'c2' else (query.detach(), logits.detach())
        for i, record in enumerate(candidates):
            scores.append([head(spatial[i], q[i, c['query']], b[i, c['query']],
                                torch.tensor(c['positions'], device=device, dtype=spatial.dtype))
                           for c in record['candidates']])
    return (query, logits, attention, heat, offset), candidates, scores


def positive_mask(points, truth, source):
    distances = torch.cdist(points, truth)
    nearest = distances.min(dim=1).values
    tied = (distances == nearest[:, None]).sum(dim=1) > 1
    positive = (distances[:, source] <= 100) & (distances[:, source] == nearest) & ~tied
    return positive, ~tied


def association_loss(scores, candidates, mappings, targets, ids, zero):
    losses, supervised, positives, negatives, ambiguous = [], 0, 0, 0, 0
    for i, mapping in enumerate(mappings):
        truth = targets.positions[ids[i], :int(targets.counts[ids[i]])].to(zero.device)
        for q, source in mapping.items():
            supervised += 1
            s = scores[i][q]
            if not len(s):
                continue
            points = torch.tensor(candidates[i]['candidates'][q]['positions'], device=zero.device, dtype=truth.dtype)
            pos, valid = positive_mask(points, truth, source)
            positives += int(pos.sum())
            negatives += int((valid & ~pos).sum())
            ambiguous += int((~valid).sum())
            if pos.any():
                losses.append(torch.logsumexp(s[valid], 0)-torch.logsumexp(s[pos], 0))
    loss = torch.stack(losses).mean() if losses else zero.sum()*0
    return loss, {'matched_slots': supervised, 'covered_slots': len(losses),
                  'positive_candidates': positives, 'negative_candidates': negatives, 'ambiguous': ambiguous}


def compute_loss(outputs, candidates, scores, targets, ids, config, arm):
    _, logits, _, heat, offset = outputs
    original, components, mappings = g4.r1.compute_losses(
        logits, heat, offset, g4.as_cached_targets(targets), ids, config)
    assoc, stats = logits.sum()*0, {}
    if arm != 'c0':
        assoc, stats = association_loss(scores, candidates, mappings, targets, ids, logits)
    return original + .2*assoc, {**components, 'association': assoc}, stats


def decode_r2(logits, candidate_record, scores=None):
    active = np.flatnonzero(logits.max(-1) >= 0).tolist()
    candidates = [candidate_record['candidates'][q] for q in active]
    original = np.asarray(candidate_record['original'], dtype=np.float32)[active].reshape(-1, 2)
    if scores is None:
        log_assoc = [np.zeros(len(c['scores'])) for c in candidates]
    else:
        log_assoc = [scores[q].detach().log_softmax(0).cpu().numpy() for q in active]
        # A slot-constant term cannot affect the mathematical argmax; remove it
        # exactly at initialization to avoid changing floating-point tie breaks.
        log_assoc = [np.zeros_like(a) if len(a) and np.all(a == a[0]) else a for a in log_assoc]
    best = None
    # R1 K=0/1 behavior is kept; association changes only multisource choices.
    if len(active) >= 2:
        for ranks in itertools.product(*(range(len(c['scores'])) for c in candidates)):
            points = np.asarray([c['positions'][r] for c, r in zip(candidates, ranks)])
            if any(np.linalg.norm(a-b) < 30 for a, b in itertools.combinations(points, 2)):
                continue
            # Use float64 constants to preserve the R1 ordering at zero initialization.
            value = sum(np.log(max(c['scores'][r], 1e-20)) + float(a[r])
                        for c, a, r in zip(candidates, log_assoc, ranks))
            if best is None or value > best[0]:
                best = value, points, list(ranks)
    predicted = best[1] if best is not None else original
    return {'active': active, 'joint': np.asarray(predicted, dtype=np.float32).reshape(-1, 2).tolist(),
            'candidates': candidates, 'original': original.tolist(),
            'association_logits': [scores[q].detach().cpu().tolist() for q in active] if scores is not None else None,
            'selected_ranks': [x+1 for x in best[2]] if best is not None else ([1]*len(active) if len(active)<2 else None),
            'fallback': len(active)>=2 and best is None}
