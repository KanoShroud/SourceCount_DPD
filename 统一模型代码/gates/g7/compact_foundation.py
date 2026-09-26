"""Full-network native-task adaptation to the centred 41/201 small ROI.

Stored ``coarse`` and ``oracle_fine`` are LINEAR physical DPD, not log maps.
Only this module applies log1p and native per-sample z-score before the model.
The oracle D8 input uses the BW_actual >=0.2 coverage subband union,
not actual continuous bandwidth and not the unfiltered all-frequency input.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from 统一模型代码.gates.g5.e2e_g5_features import build_models, g4
from 统一模型代码.gates.g6.coherent_dpd import geometry
from 统一模型代码.gates.g7.compact_data import EDGE, COARSE_N, FINE_N, STEP
from 统一模型代码.physics.fine_dpd_autograd import compute_fine_dpd_autograd
from train_v26 import compute_loss as native_ch3_loss
from yolo_model import focal_loss_hm


NATIVE_CONFIG = dict(ch3_gamma=2., d8_gaussian_sigma_px=2., d8_dice_weight=0.,
    d8_offset_weight=1., d8_conf_weight_offset=False, amp=False,
    learning_rate=1e-4, ch3_weight_decay=5e-4, d8_weight_decay=5e-3,
    d8_input='GT_BW_actual_hard19_coverage_ge_0.2_union', stored_maps='linear_float32',
    foundation_parameters='all', state_includes_buffers=True)


@lru_cache(maxsize=2)
def _small_geometry(device):
    axis = torch.linspace(-EDGE, EDGE, FINE_N, dtype=torch.float64)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    return geometry(torch.stack([x.flatten(), y.flatten()], -1), device=device,
                    shape=(FINE_N, FINE_N))


def oracle_fft_mask(record, edges, geo):
    """Exact >=lo,<hi union using the registered D8 hard_actual input rule.

    Must supply oracle_slots, or actual_fc/actual_bw/b_win. The semantic CH3
    band labels use a different definition and are deliberately never used.
    """
    count = int(record['count'])
    lo, hi = (torch.as_tensor(v, device=geo.device, dtype=torch.float64) for v in edges)
    if lo.shape != (19,) or hi.shape != (19,) or not (hi > lo).all():
        raise ValueError('Expected 19 valid subband edges')
    if 'oracle_slots' in record:
        bands = torch.as_tensor(record['oracle_slots'], device=geo.device)
    elif count == 0:
        bands = torch.zeros(0, 19, device=geo.device)
    else:
        fc = torch.as_tensor(record['actual_fc'], dtype=torch.float64, device=geo.device)[:count]
        bw = torch.as_tensor(record['actual_bw'], dtype=torch.float64, device=geo.device)[:count]
        b_win = float(record['b_win'])
        if len(fc) != count or len(bw) != count or not torch.isfinite(fc).all() or not torch.isfinite(bw).all() or (bw <= 0).any() or not np.isfinite(b_win) or b_win <= 0:
            raise ValueError('Invalid BW_actual metadata')
        overlap = (torch.minimum((fc+bw/2)[:, None], hi) - torch.maximum((fc-bw/2)[:, None], lo)).clamp_min(0)
        # Native actual_slot_coverages stores float32 BEFORE the >=0.2 test.
        bands = (overlap / b_win).float() >= .2
    if bands.ndim != 2 or bands.shape[-1] != 19 or not 0 <= count <= len(bands):
        raise ValueError('Malformed oracle slots/count record')
    union = (bands[:count] > .5).any(0)
    mask = ((geo.f_full[:, None] >= lo) & (geo.f_full[:, None] < hi) & union).any(-1)
    if count and not mask.any():
        raise ValueError('Nonzero-source oracle has empty positive-band union')
    return mask


@torch.no_grad()
def build_oracle_fine(iq, record, edges, device='cuda'):
    """Return CPU float32 LINEAR [201,201]; physics is FP64/complex128.

    Empty-source union produces a constant-zero map, hence a zero normalized
    input. No raw IQ is modified, and no full-band fallback is used.
    """
    geo = _small_geometry(str(torch.device(device)))
    mask = oracle_fft_mask(record, edges, geo)
    if not mask.any():
        return torch.zeros(FINE_N, FINE_N, dtype=torch.float32)
    raw = compute_fine_dpd_autograd(iq, geo, mask.double(), fixed_support=mask,
        grid_chunk_size=2048, frequency_chunk_size=512, real_dtype=torch.float64,
        eig_device='cuda' if geo.device.type == 'cuda' else 'cpu',
        use_checkpoint=False, checkpoint_mode='off')
    if raw.shape != (FINE_N, FINE_N) or not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Invalid oracle fine DPD')
    return raw.float().cpu()


def ch3_input(record):
    coarse = torch.as_tensor(record['coarse']).cpu().numpy().astype(np.float32, copy=False)
    if coarse.shape != (19, COARSE_N, COARSE_N) or not np.isfinite(coarse).all() or (coarse < 0).any():
        raise ValueError('Expected finite nonnegative raw CH3 map')
    # Native train_v26 uses np.log(x+1), population std over all 19 channels.
    value = np.log(coarse + np.float32(1.))
    return torch.from_numpy(((value - value.mean()) / (value.std() + 1e-6)).copy())


def d8_input(record):
    raw = torch.as_tensor(record['oracle_fine'], dtype=torch.float32)
    if raw.shape != (FINE_N, FINE_N) or not torch.isfinite(raw).all() or (raw < 0).any():
        raise ValueError('Expected finite nonnegative oracle_fine raw map')
    value = torch.log(raw + 1.)
    # Native dataset receives already-logged maps and uses sample std.
    return ((value - value.mean()) / (value.std() + 1e-6))[None]


def d8_targets(records, device):
    """Native max-combined Gaussian then ONE image normalization, sigma=2px."""
    axis = torch.arange(FINE_N, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    maps, positions = [], []
    for r in records:
        pos = torch.as_tensor(r['positions'], device=device, dtype=torch.float32)[:int(r['count'])]
        if pos.ndim != 2 or pos.shape[1] != 2 or not torch.isfinite(pos).all() or (pos.abs() > EDGE).any():
            raise ValueError('Native D8 labels outside centred small ROI')
        xy = (pos + EDGE) / STEP
        target = torch.zeros(FINE_N, FINE_N, device=device)
        for px, py in xy:
            target = torch.maximum(target, torch.exp(-((xx-px).square()+(yy-py).square()) / 8.))
        target /= target.max().clamp_min(1e-12)
        maps.append(target)
        positions.append(xy)
    return torch.stack(maps)[:, None], positions


def d8_loss(heat, offset, records):
    """Native dualhead_std: focal heatmap + unweighted source-average L1 offset."""
    target, positions = d8_targets(records, heat.device)
    if heat.shape != target.shape or offset.shape != (len(records), 2, FINE_N, FINE_N):
        raise ValueError('D8 output/target shape mismatch')
    loc = []
    for bi, xy in enumerate(positions):
        ij = xy.round().long().clamp(0, FINE_N-1)
        for point, (ix, iy) in zip(xy, ij):
            loc.append((offset[bi, :, iy, ix] - (point - torch.stack([ix, iy]))).abs().sum())
    parts = dict(heatmap=focal_loss_hm(heat.float(), target),
                 offset=torch.stack(loc).mean() if loc else offset.sum()*0.)
    return parts['heatmap']+parts['offset'], parts


def _semantic_targets(records, logits):
    target, ignore = torch.zeros_like(logits), torch.zeros_like(logits)
    for i, record in enumerate(records):
        b = torch.as_tensor(record['band'], device=logits.device, dtype=logits.dtype)
        ig = torch.as_tensor(record['ignore'], device=logits.device, dtype=logits.dtype)
        if b.shape != ig.shape or b.ndim != 2 or b.shape[-1] != logits.shape[-1]:
            raise ValueError('CH3 label dimensions differ')
        n = min(len(b), logits.shape[1])
        if int(record['count']) > n or (b[n:] > .5).any():
            raise ValueError('Cannot truncate positive CH3 slots')
        target[i, :n], ignore[i, :n] = b[:n], ig[:n]
    return target, ignore


class FoundationModel:
    """Only one native network is retained; every parameter and BN is adapted."""
    def __init__(self, out, manifest, seed, kind, device='cuda'):
        if kind not in ('ch3', 'd8'):
            raise ValueError(kind)
        g4.set_deterministic(seed)
        self.kind, self.device = kind, torch.device(device)
        ch3, d8 = build_models(Path(out), manifest, self.device)
        self.model = ch3 if kind == 'ch3' else d8
        del ch3, d8
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.mode(True)

    def state(self):
        # Clone: state_dict() alone aliases live parameters and BN running buffers.
        return dict(kind=self.kind, model={k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()}, native_config=dict(NATIVE_CONFIG))

    def restore(self, state):
        if state['kind'] != self.kind or state['native_config'] != NATIVE_CONFIG:
            raise ValueError('Foundation kind/loss contract differs')
        self.model.load_state_dict(state['model'], strict=True)

    def mode(self, training):
        self.model.train(training)

    def optimizer(self, manifest=None, lr=None):
        return torch.optim.AdamW(self.model.parameters(), lr=NATIVE_CONFIG['learning_rate'] if lr is None else float(lr),
            weight_decay=NATIVE_CONFIG[f'{self.kind}_weight_decay'])

    def _forward(self, records):
        maker = ch3_input if self.kind == 'ch3' else d8_input
        inputs = torch.stack([maker(r) for r in records]).to(self.device)
        return self.model(inputs)

    def forward_loss(self, records):
        output = self._forward(records)
        if self.kind == 'd8':
            return d8_loss(*output, records)
        truth, ignore = _semantic_targets(records, output)
        loss = native_ch3_loss(output, truth, ignore, gamma=NATIVE_CONFIG['ch3_gamma'])
        return loss, dict(band=loss)

    @torch.no_grad()
    def validation_statistics(self, records):
        """Additive native-task diagnostics; caller aggregates loss by batch size.

        Checkpoint choice remains native validation loss, not E2E/test metrics.
        D8 peak diagnostic uses native 7x7 local maxima, oracle K and offsets.
        """
        was_training = self.model.training
        self.mode(False)
        try:
            output = self._forward(records)
            if self.kind == 'ch3':
                truth, ignore = _semantic_targets(records, output)
                loss = native_ch3_loss(output, truth, ignore, gamma=2.)
                pred = output.sigmoid() > .5
                count = pred.any(-1).sum(-1)
                expected = torch.tensor([int(r['count']) for r in records], device=self.device)
                positive_slots = torch.arange(output.shape[1], device=self.device)[None] < expected[:, None]
                valid = (ignore < .5) & positive_slots[..., None]
                return dict(loss=float(loss), samples=len(records),
                    count_correct=int((count == expected).sum()),
                    band_correct=int(((pred == (truth > .5)) & valid).sum()), band_total=int(valid.sum()))
            heat, offset = output
            loss, _ = d8_loss(heat, offset, records)
            from scipy.ndimage import maximum_filter
            from scipy.optimize import linear_sum_assignment
            hits10, hits100, ntrue, square_error, nmatched = 0, 0, 0, 0., 0
            for bi, record in enumerate(records):
                count = int(record['count'])
                ntrue += count
                if count == 0:
                    continue
                p = heat[bi, 0].sigmoid().cpu().numpy()
                peaks = np.flatnonzero(p == maximum_filter(p, size=7, mode='constant'))
                peaks = peaks[np.argsort(-p.ravel()[peaks], kind='stable')[:count]]
                points = []
                for flat in peaks:
                    iy, ix = divmod(int(flat), FINE_N)
                    delta = offset[bi, :, iy, ix].cpu().numpy()
                    points.append((np.array([ix, iy])+delta)*STEP-EDGE)
                truth = torch.as_tensor(record['positions']).cpu().numpy()[:count]
                distances = np.linalg.norm(np.asarray(points)[:, None]-truth[None], axis=-1)
                r, c = linear_sum_assignment(distances)
                error = distances[r, c]
                hits10 += int((error <= 10).sum())
                hits100 += int((error <= 100).sum())
                square_error += float(np.square(error).sum())
                nmatched += len(error)
            return dict(loss=float(loss), samples=len(records), true_sources=ntrue,
                oracle_k_hits10=hits10, oracle_k_hits100=hits100,
                matched_squared_error_m2=square_error, matched_sources=nmatched)
        finally:
            self.mode(was_training)
