"""FP64物理谱；DPD-HR与原NumPy实现同式，显式传递观测长度。

输入IQ[station,time]、points[point,xy]，输出每点线性谱。
仅分块/向量化，不抽频、不截短记录。fc=0遵循项目基带时延模型。
"""
from __future__ import annotations

import itertools
import numpy as np
import torch
from scipy.ndimage import maximum_filter

FS = 100e6
C = 299792458.
RECEIVERS = np.stack([500*np.cos(np.arange(4)*np.pi/2),
                      500*np.sin(np.arange(4)*np.pi/2)], axis=1).astype(np.float32).astype(float)
LO = np.arange(19)*5e6-50e6
HI = LO+10e6


def grid(step=50., edge=1000.):
    axis = np.arange(-edge, edge+step/2, step)
    x, y = np.meshgrid(axis, axis, indexing='xy')
    return np.stack([x.ravel(), y.ravel()], axis=-1), (len(axis), len(axis))


class Spectrum:
    def __init__(self, iq, method='dpd', *, fs=FS, receivers=RECEIVERS,
                 segments=4, loading=1e-4, device='cuda', guard=lambda: None):
        iq = np.asarray(iq, dtype=np.complex128)
        if iq.ndim != 2 or iq.shape[0] != 4 or not np.isfinite(iq).all():
            raise ValueError('Expected finite [4,N] IQ')
        if method not in ('dpd', 'hr') or iq.shape[1] < 2:
            raise ValueError('Invalid method/N')
        self.device, self.method, self.guard = device, method, guard
        self.fs, self.n = float(fs), iq.shape[1]
        self.receivers = torch.as_tensor(receivers, dtype=torch.float64, device=device)
        self.pairs = list(itertools.combinations(range(4), 2))
        self.info = dict(method=method, N_total=self.n, fs_hz=fs, precision='complex128')
        if method == 'hr':
            if segments not in (2, 4, 8) or loading <= 0:
                raise ValueError('Unregistered HR parameters')
            nf = self.n//segments
            x = iq[:, :nf*segments].reshape(4, segments, nf)
            x = np.fft.fftshift(np.fft.fft(x, axis=-1), axes=-1)
            covariance = np.einsum('mjf,njf->fmn', x, x.conj())/segments
            covariance = (covariance+covariance.conj().transpose(0, 2, 1))/2
            raw_eig = np.linalg.eigvalsh(covariance)
            loads = loading*np.trace(covariance, axis1=1, axis2=2).real/4
            loaded = covariance+loads[:, None, None]*np.eye(4)
            ev = np.linalg.eigvalsh(loaded)
            ratios = ev[:, 0]/ev[:, -1]
            if not np.all(ratios > 64*np.finfo(float).eps):
                raise ValueError('HR covariance is not numerically positive definite')
            self.inverse = torch.as_tensor(np.linalg.inv(loaded), device=device)
            self.freq = torch.as_tensor(np.fft.fftshift(np.fft.fftfreq(nf, 1/fs)), device=device)
            baseline = np.linalg.norm(np.asarray(receivers)[:, None]-receivers, axis=-1).max()
            self.info.update(J=segments, loading=loading, N_fft=nf, frequencies=nf,
                N_used=nf*segments, N_discarded=self.n-nf*segments,
                segment_margin=(nf/fs)/(baseline/C), required_margin=40.,
                rank_bound=min(4, segments),
                effective_rank_min=int((raw_eig > raw_eig[:, -1:]*64*np.finfo(float).eps).sum(-1).min()),
                loaded_condition_max=float(1/ratios.min()))
        else:
            self.x = torch.as_tensor(np.fft.fftshift(np.fft.fft(iq, axis=-1), axes=-1), device=device)
            self.freq = torch.as_tensor(np.fft.fftshift(np.fft.fftfreq(self.n, 1/fs)), device=device)

    @torch.no_grad()
    def evaluate(self, points, mask=None, chunk=128):
        points = torch.as_tensor(points, dtype=torch.float64, device=self.device)
        if points.ndim != 2 or points.shape[1] != 2 or not torch.isfinite(points).all():
            raise ValueError('Expected finite [P,2] points')
        if mask is not None and self.method != 'dpd':
            raise ValueError('HR uses the registered full receiver bandwidth')
        if self.method == 'dpd':
            mask = torch.ones(self.n, dtype=torch.bool, device=self.device) if mask is None else torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            if mask.shape != (self.n,):
                raise ValueError('Frequency mask length differs from actual IQ')
            if not mask.any():
                return np.zeros(len(points))
            x, freq = self.x[:, mask], self.freq[mask]
            e = x.abs().square().sum(-1)
            den = (e/self.n**2).clamp_min(1e-20).sqrt()
            cross = torch.stack([x[m]*x[n].conj()/(den[m]*den[n]) for m, n in self.pairs])
        else:
            freq = self.freq
        outputs = []
        for start in range(0, len(points), chunk):
            self.guard()
            tau = torch.linalg.vector_norm(points[start:start+chunk, None]-self.receivers, dim=-1)/C
            matrix = torch.zeros(len(tau), 4, 4, dtype=torch.complex128, device=self.device)
            if self.method == 'dpd':
                matrix += torch.diag((e/den.square()).to(torch.complex128))
            else:
                for m in range(4):
                    matrix[:, m, m] = self.inverse[:, m, m].sum()
            for p, (m, n) in enumerate(self.pairs):
                value = torch.zeros(len(tau), dtype=torch.complex128, device=self.device)
                for f0 in range(0, len(freq), 1024):
                    phase = torch.exp(2j*np.pi*(tau[:, m]-tau[:, n])[:, None]*freq[None, f0:f0+1024])
                    v = cross[p, f0:f0+1024] if self.method == 'dpd' else self.inverse[f0:f0+1024, m, n]
                    value += phase @ v
                matrix[:, m, n], matrix[:, n, m] = value, value.conj()
            if self.method == 'dpd':
                matrix += torch.eye(4, device=self.device, dtype=torch.complex128)*1e-6
            matrix = (matrix+matrix.mH)/2
            ev = torch.linalg.eigvalsh(matrix)
            if self.method == 'hr':
                if not (ev[:, 0] > 64*np.finfo(float).eps*ev[:, -1]).all():
                    raise ValueError('HR spectrum is not positive definite')
                values = 1/ev[:, 0]
            else:
                values = ev.abs().amax(-1)
            outputs.append(values.cpu().numpy())
        result = np.concatenate(outputs) if outputs else np.empty(0)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite spectrum')
        return result


def peaks(points, values, count, separation=30.):
    selected = []
    for i in np.argsort(-values, kind='stable'):
        if all(np.linalg.norm(points[i]-points[j]) >= separation for j in selected):
            selected.append(int(i))
            if len(selected) == count:
                break
    return selected


def search(spectrum, count):
    if count == 0:
        return dict(positions=[], candidates=[], scores=[])
    if not 0 < count <= 10:
        raise ValueError('K must be in [0,10]')
    points, shape = grid()
    values = spectrum.evaluate(points)
    local = values.reshape(shape) == maximum_filter(values.reshape(shape), size=3, mode='constant', cval=-np.inf)
    ids = np.flatnonzero(local.ravel())
    chosen = ids[peaks(points[ids], values[ids], max(5, count))]
    axis = np.arange(-50., 51., 10.)
    xx, yy = np.meshgrid(axis, axis)
    offsets = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    candidates, scores = [], []
    for i in chosen:
        window = points[i]+offsets
        window = window[(abs(window) <= 1000).all(-1)]
        v = spectrum.evaluate(window)
        side_x = np.unique(window[:, 0]).size
        side_y = np.unique(window[:, 1]).size
        lm = v.reshape(side_y, side_x) == maximum_filter(v.reshape(side_y, side_x), 3, mode='constant', cval=-np.inf)
        candidates.extend(window[lm.ravel()].tolist())
        scores.extend(v[lm.ravel()].tolist())
        candidates.append(points[i].tolist())
        scores.append(float(values[i]))
    candidates, scores = np.asarray(candidates), np.asarray(scores)
    ids = peaks(candidates, scores, count)
    return dict(positions=candidates[ids].tolist(), candidates=candidates.tolist(), scores=scores.tolist())
