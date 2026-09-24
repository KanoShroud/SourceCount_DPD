"""G6-P1: frozen P0 physics and zero-initialized heatmap residual."""
import itertools

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from 统一模型代码.gates.g5.e2e_g5_model import forward as base_forward
from 统一模型代码.gates.g6.g6_p1_speed import physical_maps


class PhysicalResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, maps):
        b, q, h, w = maps.shape
        small = self.net(maps.float().reshape(b*q, 1, h, w))
        return F.interpolate(small, size=(401, 401), mode='bilinear',
                             align_corners=True).reshape(b, q, 401, 401)


def forward(context, head, features, ids, arm, physics, stats):
    # Old latent path is stopped in ALL tracks; auxiliary band loss remains live.
    query, logits, attention, heat, offset = base_forward(
        context, features, ids, torch.device('cuda:0'), stop_gradient=True)
    if head is None:
        return (query, logits, attention, heat, offset), torch.zeros_like(heat)
    probabilities = logits.sigmoid()
    if arm == 'c1':
        probabilities = probabilities.detach()
    residual = head(physical_maps(physics, probabilities, stats))
    return (query, logits, attention, heat+residual, offset), residual


def identity_hits(truth, predicted, logits, active, bands, ignore):
    """Pure spatial maximum-cardinality matching, min distance, deterministic tie."""
    truth = np.asarray(truth).reshape(-1, 2)
    predicted = np.asarray(predicted).reshape(-1, 2)
    if not len(truth) or not len(predicted):
        return {'spatial_identity_denominator': 0, 'spatial_identity_numerator': 0}
    distances = np.linalg.norm(truth[:, None]-predicted[None], axis=-1)
    best = None
    for choices in itertools.product(range(-1, len(predicted)), repeat=len(truth)):
        used = [p for p in choices if p >= 0]
        if len(set(used)) != len(used):
            continue
        pairs = [(t, p) for t, p in enumerate(choices) if p >= 0]
        if any(distances[t, p] > 100 for t, p in pairs):
            continue
        key = (-len(pairs), sum(distances[t, p] for t, p in pairs), choices)
        if best is None or key < best[0]:
            best = key, pairs
    hits = 0
    for t, p in best[1]:
        valid = np.asarray(ignore[t]) < .5
        a = np.asarray(bands[t])[valid] > .5
        b = np.asarray(logits[active[p]])[valid] >= 0
        f1 = 2*np.sum(a & b)/max(int(a.sum()+b.sum()), 1)
        hits += int(f1 >= .8)
    return {'spatial_identity_denominator': len(best[1]), 'spatial_identity_numerator': hits}
