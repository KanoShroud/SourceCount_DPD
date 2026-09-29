"""冻结G7的只读适配器；不构造会写入旧目录的Runtime。"""
from __future__ import annotations

import io
import json
from pathlib import Path
import sys
import time
import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .storage import ROOT, read, identity, checked_bytes
from .physics import Spectrum, LO, HI, grid, RECEIVERS, C

for folder in (ROOT, ROOT/'第三章代码', ROOT/'第四章代码'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from 统一模型代码.common.g5_verified_io import VerifiedFile  # noqa: E402
from 统一模型代码.gates.g7.formal_model import FormalFoundation, FormalCandidate  # noqa: E402
from 统一模型代码.gates.g7.compact_foundation import ch3_input, d8_input  # noqa: E402
from 统一模型代码.gates.g7.formal_hard import decode_hard, predicted_selection  # noqa: E402
from 统一模型代码.gates.g7.compact_model import decode  # noqa: E402
from 统一模型代码.gates.g6.coherent_dpd import geometry, AtomicDPD  # noqa: E402


class Sources:
    def __init__(self, out):
        self.out = Path(out)
        # Resolve the sole completed audited formal run, rather than guess a checkpoint.
        runs = []
        for audit_path in (ROOT/'outputs_e2e/unified/e2e_g7_compact').glob('*/evaluation/final_audit_report.json'):
            audit = read(audit_path)
            if audit.get('status') == 'PASS' and audit.get('scope') == 'scratch_training_and_development_comparison':
                runs.append((audit_path, audit))
        if len(runs) != 1:
            raise RuntimeError('Expected one audited formal G7 run; resolve ambiguity explicitly')
        audit_path, audit = runs[0]
        self.run = audit_path.parents[1]
        contract = json.loads(checked_bytes(audit['contract']))
        rows = contract['files']
        manifest_row = next(r for r in rows if Path(r['path']).name == 'manifest.json')
        self.manifest = json.loads(checked_bytes(manifest_row))
        self.inputs = [identity(audit_path), audit['contract'], manifest_row]
        self.checkpoints = {}
        self.seeds = contract['config']['seeds']
        for seed in self.seeds:
            for kind in ('ch3', 'd8', 'candidate'):
                marker = self.run/f'training/{seed}/{kind}/completed.json'
                row = read(marker)['best']['checkpoint']
                path = Path(row['path']).resolve(strict=True)
                if not path.is_relative_to(self.run):
                    raise RuntimeError('G7 checkpoint is outside its registered run')
                checked_bytes(row)
                self.checkpoints[(seed, kind)] = row
                self.inputs.extend([identity(marker), row])
        files = self.manifest['inputs']['files']
        self.raw = next(r for r in files if Path(r['path']).name.endswith('_val_data.mat'))
        self.coarse = next(r for r in files if Path(r['path']).name.endswith('_val_select.mat'))
        for row in (self.raw, self.coarse):
            path = Path(row['path']).resolve(strict=True)
            if not path.is_relative_to(ROOT/'outputs_e2e') or 'input_snapshot' not in path.parts:
                raise ValueError('Old IQ must be a registered local snapshot')
        self.inputs.extend([self.raw, self.coarse])

    def state(self, seed, kind):
        return torch.load(io.BytesIO(checked_bytes(self.checkpoints[(seed, kind)])),
                          map_location='cpu', weights_only=False)['state']

    def selected(self):
        rng = np.random.default_rng(2026092708)
        rows = self.manifest['subsets']['val_select']
        out = []
        for k in range(4):
            ids = sorted((i for i, r in enumerate(rows) if r['true_k'] == k),
                         key=lambda i: rows[i]['raw_index'])
            chosen = rng.choice(ids, 64, replace=False).tolist()
            out.extend(dict(index=i, role='calibration' if j < 16 else 'check', count=k)
                       for j, i in enumerate(chosen))
        return out

    def old_scene(self, entry):
        idx = entry['index']
        mapping = self.manifest['subsets']['val_select'][idx]
        ri, ci = int(mapping['raw_index']), int(mapping['local_index'])
        with VerifiedFile(self.raw, self.out/'anomalies') as stream, h5py.File(stream, 'r') as raw:
            profile = ''.join(chr(int(c)) for c in np.asarray(raw['band_label_profile_val']).reshape(-1))
            if profile != 'hard19_actual_t020':
                raise ValueError('Raw label profile differs from registered G7 interface')
            iq = np.asarray(raw['sig_rcv_real_all'][:, :, ri], dtype=float).T+1j*np.asarray(raw['sig_rcv_imag_all'][:, :, ri], dtype=float).T
            positions = np.asarray(raw['src_pos_all'][:, :, ri], dtype=float).T[:entry['count']]
            extra = {}
            for name in ('Pt_W_all', 'snr_each_all', 'BW_actual_all', 'fc_offset_all', 'symbolRate_all'):
                if name in raw:
                    extra[name] = np.asarray(raw[name][..., ri]).tolist()
        with VerifiedFile(self.coarse, self.out/'anomalies') as stream, h5py.File(stream, 'r') as coarse:
            k = int(coarse['src_count_all'][0, ci])
            if k != entry['count']:
                raise ValueError('Old sample count mismatch')
            band = np.asarray(coarse['band_mask_all'][:, :, ci]).T[:k]
            ignore = np.asarray(coarse['ignore_mask_all'][:, :, ci]).T[:k]
            cached = np.asarray(coarse['mtr_sub_all'][:, :, :, ci], dtype=np.float32).transpose(2, 1, 0)[:, 20:61, 20:61].copy()
            for field, expected in [('fs_val', 100e6), ('thresh_val', .2)]:
                if field in coarse and not np.isclose(float(np.asarray(coarse[field]).item()), expected):
                    raise ValueError(f'旧数据接口参数不符：{field}')
        if 'Pt_W_all' in extra and len(positions):
            tx = np.asarray(extra['Pt_W_all']).reshape(-1)[:k]
            distance = np.linalg.norm(positions[:, None]-RECEIVERS, axis=-1)
            received = tx[:, None]*(C/(4*np.pi*5.8e9*distance))**2
            bandwidth = np.asarray(extra.get('BW_actual_all', [])).reshape(-1)[:k]
            extra['rx_power'] = received.tolist()
            extra['power_reconstruction'] = 'main31 free-space fc5800MHz noise-90dBm; model, not measured'
            if len(bandwidth) == k:
                extra['model_snr_db'] = (10*np.log10(received/(1e-12*bandwidth[:, None]/1e8))).tolist()
        return dict(iq=iq, cached_coarse=cached, metadata=dict(origin='old', group=idx,
            role=entry['role'], index=idx, count=k, positions=positions.tolist(),
            bands=band.tolist(), ignore=ignore.tolist(), raw_index=ri,
            fs_hz=100e6, samples=iq.shape[1], components_available=False, **extra))


def atomic_statistics(calc, physics, points):
    """N-aware bounded adapter; no call to the old fixed4096 statistics function."""
    geo = geometry(points, n_fft=calc.n, fs=calc.fs, device=calc.device, shape=(41, 41))
    masks = physics.masks
    energy = masks.double() @ calc.x.abs().square().T
    cross = torch.stack([calc.x[m]*calc.x[n].conj() for m, n in geo.pairs])
    coherent = torch.empty(len(masks), len(points), 6, dtype=torch.complex128, device='cpu')
    for start in range(0, len(points), 64):
        calc.guard()
        phase = torch.exp(2j*np.pi*geo.dtaus[start:start+64].T[..., None]*geo.f_full)
        block = torch.einsum('pgf,rf->rgp', phase*cross[:, None], masks.to(torch.complex128))
        coherent[:, start:start+64] = block.cpu()
    return dict(energy=energy.cpu(), coherent=coherent)


class Models:
    def __init__(self, sources, seed, device='cuda'):
        self.device = device
        self.native = FormalFoundation(None, None, seed, 'ch3', device)
        self.d8 = FormalFoundation(None, None, seed, 'd8', device)
        self.candidate = FormalCandidate(None, None, seed, device=device)
        for kind, model in [('ch3', self.native), ('d8', self.d8), ('candidate', self.candidate)]:
            model.restore(sources.state(seed, kind))
            model.mode(False)

    @torch.no_grad()
    def infer(self, iq, guard=lambda: None):
        if self.device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        calc = Spectrum(iq, device=self.device, guard=guard)
        points, shape = grid()
        # Native MATLAB cache stores [sub,x,y]; preserve frozen CH3 input convention.
        # D8 fine maps and physical candidate maps remain [y,x].
        coarse = np.stack([calc.evaluate(points, (calc.freq >= lo) & (calc.freq < hi)).reshape(shape).T for lo, hi in zip(LO, HI)]).astype(np.float32)
        coarse_seconds = time.perf_counter()-start
        logits = self.native.model(ch3_input(dict(coarse=coarse))[None].to(self.device))[0]
        torch.cuda.synchronize() if self.device == 'cuda' else None
        ch3_seconds = time.perf_counter()-start
        active, selected = predicted_selection(logits)
        fine_points, fine_shape = grid(10)
        if len(active):
            union = ((calc.freq[:, None] >= torch.as_tensor(LO, device=self.device)) &
                     (calc.freq[:, None] < torch.as_tensor(HI, device=self.device)) & selected.any(0)).any(-1)
            fine = calc.evaluate(fine_points, union).reshape(fine_shape).astype(np.float32)
            heat, offset = self.d8.model(d8_input(dict(oracle_fine=fine))[None].to(self.device))
            positions, _ = decode_hard(heat[0], offset[0], len(active))
        else:
            positions = np.empty((0, 2))
        torch.cuda.synchronize() if self.device == 'cuda' else None
        hard_seconds = time.perf_counter()-start
        candidate_start = time.perf_counter()
        fine = calc.evaluate(fine_points).reshape(fine_shape).astype(np.float32)
        geo = geometry(points, n_fft=calc.n, fs=calc.fs, device=self.device, shape=shape)
        physics = AtomicDPD(LO, HI, geo, precompute_phase=False)
        physics.p1_batch_mode, physics.p1_chunk = 'batched', 256
        stats = atomic_statistics(calc, physics, points)
        record = dict(coarse=torch.from_numpy(coarse), fine=torch.from_numpy(fine), statistics=stats)
        output = self.candidate.baseline([record], physics)
        decoded = decode(output, 'baseline', top_k=8)[0]
        torch.cuda.synchronize() if self.device == 'cuda' else None
        return dict(hard=dict(positions=positions.tolist(), logits=logits.cpu().tolist(),
                    active=active.cpu().tolist(), seconds=hard_seconds),
            candidate=dict(positions=decoded['positions_m'], logits=output['band_logits'][0].cpu().tolist(),
                active=decoded['active'], candidates=decoded['candidates'],
                seconds=coarse_seconds+time.perf_counter()-candidate_start),
            coarse_seconds=coarse_seconds, ch3_seconds=ch3_seconds, coarse=coarse)


def bind(iq, positions, probabilities, device='cuda', guard=lambda: None, iterations=0, temperature=1.):
    """Fixed-position one-to-one binding; no truth, no learned scorer."""
    positions = np.asarray(positions).reshape(-1, 2)
    probabilities = np.asarray(probabilities).reshape(-1, 19)
    if not len(positions) or not len(probabilities):
        return dict(slot_indices=[], positions=[], scores=[])
    calc = Spectrum(iq, device=device, guard=guard)
    response = np.stack([calc.evaluate(positions, (calc.freq >= lo) & (calc.freq < hi)) for lo, hi in zip(LO, HI)], -1)
    z = np.log1p(response)
    z = (z-z.mean(0, keepdims=True))/(z.std(0, keepdims=True)+1e-6)
    prior = (probabilities > .5).astype(float)
    score = prior @ z.T/np.maximum(prior.sum(-1, keepdims=True), 1)
    original = score.copy()
    for _ in range(iterations):
        rows, cols = linear_sum_assignment(-score)
        evidence = np.zeros_like(prior)
        evidence[rows] = z[cols]
        exp = np.exp(np.clip(evidence/temperature, -30, 30))*prior
        weights = exp/(1+exp.sum(0, keepdims=True))  # fixed zero-logit background
        updated = weights @ z.T/np.maximum(weights.sum(-1, keepdims=True), 1e-12)
        score = updated if np.any(updated > 0) else original
    rows, cols = linear_sum_assignment(-score)
    return dict(slot_indices=rows.tolist(), positions=positions[cols].tolist(), scores=score.tolist(),
                response=response.tolist())
