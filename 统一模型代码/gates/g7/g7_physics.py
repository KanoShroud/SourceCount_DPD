"""Exact local-grid FP64 sufficient statistics; no coarse-map interpolation."""
from __future__ import annotations

import math

import torch

from 统一模型代码.gates.g6.coherent_dpd import geometry, spatial_evd

SIZE, STEP, HALF = 41, 10., 20


def windows(record):
    """Centers are predicted grid cells, not donor offsets or ground truth."""
    centers = torch.zeros(3, 8, 2, dtype=torch.float64)
    exists = torch.zeros(3, 8, dtype=torch.bool)
    for q, row in enumerate(record['candidates']):
        for j, flat in enumerate(row['flat_indices']):
            y, x = divmod(flat, 401)
            centers[q, j] = torch.tensor([x*STEP-2000, y*STEP-2000])
            exists[q, j] = True
    axis = (torch.arange(SIZE, dtype=torch.float64)-HALF)*STEP
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    points = centers[..., None, None, :]+torch.stack((x, y), -1)
    valid = (points.abs() <= 2000).all(-1) & exists[..., None, None]
    # Invalid pixels have a harmless valid index but can never affect normalization/loss.
    clipped = points.clamp(-2000, 2000)
    unique, inverse = torch.unique(clipped.reshape(-1, 2), dim=0, return_inverse=True)
    return centers, valid, unique, inverse.reshape(3, 8, SIZE, SIZE)


@torch.no_grad()
def statistics(physics, signal, points, chunk=256, guard=None):
    """Bound phase workspace; aggregate each FFT bin once per atomic interval."""
    geo = geometry(points, device=physics.geo.device)
    x = torch.fft.fftshift(torch.fft.fft(torch.as_tensor(
        signal, dtype=torch.complex128, device=geo.device), dim=-1), dim=-1)
    if x.shape != (4, geo.N0):
        raise ValueError('Expected four-station IQ')
    masks = physics.masks
    # Keep uncovered frequencies as an extra interval for the genuine all-FFT F input.
    outside = ~physics.support
    if outside.any():
        masks = torch.cat((masks, outside[None]))
    energy = masks.double() @ x.abs().square().T
    cross = torch.stack([x[m]*x[n].conj() for m, n in geo.pairs])
    result = torch.empty(len(masks), len(points), 6, dtype=torch.complex128, device='cpu')
    masks_complex = masks.to(torch.complex128)
    for start in range(0, len(points), chunk):
        if guard:
            guard()
        stop = min(start+chunk, len(points))
        phase = torch.exp(2j*math.pi*geo.dtaus[start:stop].T[..., None]*geo.f_full)
        block = torch.einsum('pgf,rf->rgp', phase*cross[:, None], masks_complex)
        result[:, start:stop] = block.cpu()
    return {'energy':energy.cpu(), 'coherent':result,
            'conditional_intervals':len(physics.masks)}


def raw_maps(physics, probabilities, record, full=False):
    """Only per-query required points enter EVD; return [3,8,41,41]."""
    device = probabilities.device
    energy = record['energy'].to(device)
    coherent = record['coherent'].to(device)
    inverse = record['inverse'].to(device)
    if full:
        weights = torch.ones(3, len(energy), dtype=torch.float64, device=device)
    else:
        weights = physics.weights(probabilities.double())
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
            matrix[:, m, n] = v
            matrix[:, n, m] = v.conj()
        matrix = matrix+1e-6*torch.eye(4, dtype=torch.complex128, device=device)
        values = spatial_evd(matrix[None], chunk_size=256)[0]
        maps.append(values[gather].reshape(8, SIZE, SIZE))
    return torch.stack(maps)


def normalize(raw, valid):
    valid = valid.to(device=raw.device, dtype=torch.bool)
    x = raw.log1p()
    n = valid.sum((-1, -2), keepdim=True).clamp_min(1)
    mean = (x*valid).sum((-1, -2), keepdim=True)/n
    var = ((x-mean).square()*valid).sum((-1, -2), keepdim=True)/n
    return ((x-mean)/(var.clamp_min(1e-24).sqrt()+1e-6)).masked_fill(~valid, 0)


def local_maps(physics, probabilities, records, full=False):
    return torch.stack([normalize(raw_maps(physics, p, r, full), r['valid_mask'])
                        for p, r in zip(probabilities, records)]).float()
