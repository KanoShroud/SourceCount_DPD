"""FP64 atomic-interval statistics; preserve the registered DPD definition."""
from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from 统一模型代码.physics.fine_dpd_autograd import compute_fine_dpd_autograd


def geometry(points, n_fft=4096, fs=100e6, device='cpu', shape=None):
    points = torch.as_tensor(points, dtype=torch.float64, device=device)
    angles = np.arange(4) * 2 * np.pi / 4
    # Same receiver coordinate conversion as the historical DPDGeometry.
    receivers = torch.tensor(np.stack([500*np.cos(angles), 500*np.sin(angles)], 1),
                             dtype=torch.float32, device=device).double()
    taus = ((points[:, None]-receivers).square().sum(-1)).sqrt()/299792458.
    pairs = list(itertools.combinations(range(4), 2))
    dtaus = torch.stack([taus[:, m]-taus[:, n] for m, n in pairs], -1)
    h, w = shape or (1, len(points))
    return SimpleNamespace(device=torch.device(device), rcv_num=4, N0=n_fft, fs=fs,
                           f_full=torch.linspace(-fs/2, fs/2-fs/n_fft, n_fft,
                                                 dtype=torch.float64, device=device),
                           dtaus=dtaus, pairs=pairs, num_grid=len(points),
                           num_y=h, num_x=w)


def spatial_evd(matrix, chunk_size=64):
    """Bound CUDA eigensolver workspace without changing per-grid matrices."""
    return torch.cat([torch.linalg.eigvalsh(matrix[:, start:start+chunk_size].contiguous())
                      .abs().amax(-1) for start in range(0, matrix.shape[1], chunk_size)], dim=1)


def grid_points():
    axis = torch.linspace(-2000, 2000, 81, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return torch.stack([x.flatten(), y.flatten()], -1)


class AtomicDPD:
    def __init__(self, lo, hi, geo, precompute_phase=True):
        self.geo = geo
        self.lo = torch.as_tensor(lo, dtype=torch.float64, device=geo.device)
        self.hi = torch.as_tensor(hi, dtype=torch.float64, device=geo.device)
        if self.lo.shape != self.hi.shape or not (self.hi > self.lo).all():
            raise ValueError('Invalid frequency boundaries')
        f = geo.f_full
        bands = (f[:, None] >= self.lo) & (f[:, None] < self.hi)
        edges = torch.unique(torch.cat([self.lo, self.hi]), sorted=True)
        masks, coverage, intervals = [], [], []
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (f >= a) & (f < b)
            if not mask.any() or not bands[mask][0].any():
                continue
            if not (bands[mask] == bands[mask][0]).all():
                raise AssertionError('Nonconstant band coverage in atomic interval')
            masks.append(mask)
            coverage.append(bands[mask][0])
            intervals.append([a.item(), b.item()])
        self.masks = torch.stack(masks)
        self.coverage = torch.stack(coverage)
        self.intervals = intervals
        self.support = bands.any(-1)
        if not torch.equal(self.masks.sum(0), self.support.long()):
            raise AssertionError('Frequency bins must be covered exactly once')
        self.phase = None
        if precompute_phase:
            # Avoid stacking six full temporaries: CUDA/WDDM staging also costs RAM.
            self.phase = torch.empty((6, geo.num_grid, geo.N0), dtype=torch.complex128,
                                     device=geo.device)
            for p in range(6):
                for start in range(0, geo.num_grid, 256):
                    end = min(start+256, geo.num_grid)
                    self.phase[p, start:end] = torch.exp(
                        2j*math.pi*geo.dtaus[start:end, p, None]*f[None])

    def weights(self, probabilities):
        p = probabilities.to(device=self.geo.device, dtype=torch.float64)
        if not torch.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
            raise ValueError('Band probabilities outside [0,1]')
        return 1-torch.prod(1-p[..., None, :]*self.coverage, dim=-1)

    def fft_weights(self, weights):
        return weights @ self.masks.double()

    def statistics(self, signal):
        x = torch.fft.fftshift(torch.fft.fft(torch.as_tensor(
            signal, dtype=torch.complex128, device=self.geo.device), dim=-1), dim=-1)
        if x.shape != (4, self.geo.N0):
            raise ValueError('Expected four-station IQ')
        cross = torch.stack([x[m]*x[n].conj() for m, n in self.geo.pairs], 1)
        energy, coherent = [], []
        for mask in self.masks:
            energy.append(x[:, mask].abs().square().sum(-1))
            if self.phase is None:
                phase = torch.stack([torch.exp(2j*math.pi*self.geo.dtaus[:, p, None]
                                              * self.geo.f_full[mask][None]) for p in range(6)])
            else:
                phase = self.phase[:, :, mask]
            coherent.append(torch.einsum('pgf,fp->gp', phase, cross[mask]))
        return {'energy': torch.stack(energy), 'coherent': torch.stack(coherent)}

    def evaluate(self, stats, weights):
        """Return [slot,y,x]; energy normalization follows Parseval E/N^2."""
        a = weights.to(device=self.geo.device, dtype=torch.float64)
        if a.ndim == 1:
            a = a[None]
        energy = a @ stats['energy']
        denom = (energy / self.geo.N0**2).clamp_min(1e-20).sqrt()
        coherent = torch.einsum('qr,rgp->qgp', a.to(torch.complex128), stats['coherent'])
        diag = energy/denom.square()
        matrix = torch.diag_embed(diag.to(torch.complex128))[:, None].expand(
            -1, self.geo.num_grid, -1, -1).clone()
        for p, (m, n) in enumerate(self.geo.pairs):
            value = coherent[..., p]/(denom[:, m]*denom[:, n])[:, None]
            matrix[:, :, m, n] = value
            matrix[:, :, n, m] = value.conj()
        eye = torch.eye(4, dtype=torch.complex128, device=a.device)
        result = spatial_evd(matrix+1e-6*eye)
        return result.reshape(len(a), self.geo.num_y, self.geo.num_x)

    def direct(self, signal, weights, geo=None):
        geo = geo or self.geo
        return compute_fine_dpd_autograd(signal, geo, self.fft_weights(weights),
            fixed_support=self.support, grid_chunk_size=1024, frequency_chunk_size=256,
            eig_device='cuda' if geo.device.type == 'cuda' else 'cpu',
            checkpoint_mode='off', real_dtype=torch.float64)


def normalize_maps(raw):
    x = raw.log1p()
    mean = x.mean((-1, -2), keepdim=True)
    std = x.std((-1, -2), unbiased=False, keepdim=True) + 1e-6
    return (x-mean)/std, mean, std


def sample_map(maps, points):
    points = torch.as_tensor(points, dtype=maps.dtype, device=maps.device)
    return F.grid_sample(maps[:, None], (points/2000)[None, :, None].expand(
        len(maps), -1, -1, -1), mode='bilinear', padding_mode='border',
        align_corners=True)[:, 0, :, 0]


def band_assignments(probabilities, truth, ignore):
    """All minimum band-only BCE permutations; spatial information is excluded."""
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1-1e-12)
    truth, ignore = np.asarray(truth), np.asarray(ignore)
    k = len(truth)
    perms = np.asarray(list(itertools.permutations(range(k))), dtype=int).reshape(-1, k)
    cost = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            valid = ignore[j] < .5
            if not valid.any():
                raise ValueError('No valid band labels')
            cost[i, j] = (-truth[j]*np.log(p[i])-(1-truth[j])*np.log1p(-p[i]))[valid].mean()
    values = cost[np.arange(k), perms].mean(1)
    correct = np.isclose(values, values.min(), atol=1e-10, rtol=0)
    return perms, correct, cost


def identity_scores(response, perms, correct):
    response = np.asarray(response, dtype=float)
    k = len(response)
    values = response[np.arange(k), perms].mean(1)
    top = np.isclose(values, values.max(), atol=1e-9, rtol=0)
    margins = []
    # Average over all band-equivalent correct mappings, excluding same target.
    for p in perms[correct]:
        for i, own in enumerate(p):
            margins += [float(response[i, own]-response[i, other])
                        for other in range(k) if other != own]
    wrong = values[~correct]
    return {'response': response.tolist(), 'permutation_scores': values.tolist(),
            'correct_permutations': np.flatnonzero(correct).tolist(),
            'winning_permutations': np.flatnonzero(top).tolist(),
            'accuracy': float(correct[top].mean()), 'chance': float(correct.mean()),
            'informative': bool((~correct).any()),
            'joint_margin': float(values[correct].max()-wrong.max()) if len(wrong) else None,
            'pair_margins': margins, 'pair_positive_fraction': float(np.mean(np.array(margins) > 1e-9))}
