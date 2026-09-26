"""PB-BASE hard cascade of NEW formal native CH3/D8, without oracle inputs.

This baseline predicts a position SET from the union of all predicted bands.
It does not define slot-position identity, so joint band-position metrics are
not reported. Ground truth is read only by metric helpers after inference.
"""
from __future__ import annotations

import gc
import itertools
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from 统一模型代码.gates.g7.compact_data import EDGE, FINE_N, STEP, load
from 统一模型代码.gates.g7.compact_foundation import build_oracle_fine, ch3_input, d8_input
from 统一模型代码.gates.g4 import e2e_g4 as g4
from yolo_config import PEAK_SIZE
from yolo_model import nms_heatmap


TIMING_SCOPE = 'CH3_forward+predicted_band_union+IQ_read+fine_DPD+D8_forward+decode; excludes common coarse_DPD preparation, model loading and metric computation'


def predicted_selection(logits):
    if logits.shape != (10, 19) or not torch.isfinite(logits).all():
        raise ValueError('PB-BASE requires finite native 10x19 logits')
    slots = logits.sigmoid() > .5
    active = slots.any(-1).nonzero().flatten()
    return active, slots[active]


@torch.no_grad()
def build_predicted_fine(iq, predicted_slots, edges, device):
    """Use the shared physical calculator with PREDICTED slots only.

    ``oracle_slots`` is merely the lower-level calculator's argument name;
    no truth-band metadata is supplied by this inference wrapper.
    """
    if predicted_slots.ndim != 2 or predicted_slots.shape[-1] != 19:
        raise ValueError('Malformed predicted frequency slots')
    return build_oracle_fine(iq, dict(count=len(predicted_slots),
        oracle_slots=predicted_slots), edges, device=device)


@torch.no_grad()
def decode_hard(heat, offset, count):
    """Native PB-BASE NMS/top-K/offset rule, only grid origin becomes -1000m."""
    if heat.shape != (1, FINE_N, FINE_N) or offset.shape != (2, FINE_N, FINE_N):
        raise ValueError('Wrong compact D8 output shape')
    if not 0 <= count <= 10:
        raise ValueError('Native count must be in [0,10]')
    if not torch.isfinite(heat).all() or not torch.isfinite(offset).all():
        raise ValueError('Nonfinite native D8 output')
    if count == 0:
        return np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)
    suppressed = nms_heatmap(heat[None].sigmoid(), PEAK_SIZE)[0, 0]
    scores, indices = suppressed.flatten().topk(count)
    x, y = (indices % FINE_N).float(), (indices // FINE_N).float()
    for rank in range(count):
        ix, iy = int(x[rank]), int(y[rank])
        x[rank] += offset[0, iy, ix].clamp(-1, 1)
        y[rank] += offset[1, iy, ix].clamp(-1, 1)
    positions = torch.stack([x, y], -1)*STEP-EDGE
    return positions.cpu().numpy().astype(np.float32), scores.cpu().numpy().astype(np.float32)


def frequency_metrics(logits, record):
    """Frequency-only Hungarian assignment, unrelated to the predicted positions."""
    k = int(record['count'])
    if not k:
        return [], []
    logits = logits.detach().cpu()
    bands = torch.as_tensor(record['band']).cpu()[:k]
    valid = torch.as_tensor(record['ignore']).cpu()[:k] < .5
    cost = np.empty((10, k), dtype=np.float64)
    for q in range(10):
        for t in range(k):
            bce = F.binary_cross_entropy_with_logits(logits[q], bands[t].to(logits), reduction='none')
            cost[q, t] = float((bce*valid[t]).sum()/valid[t].sum().clamp_min(1))
    pred, truth = linear_sum_assignment(cost)
    f1, iou = [], []
    for q, t in zip(pred, truth):
        p, z = logits[q, valid[t]].sigmoid() > .5, bands[t, valid[t]] > .5
        tp, fp, fn = int((p & z).sum()), int((p & ~z).sum()), int((~p & z).sum())
        f1.append(2*tp/max(2*tp+fp+fn, 1))
        iou.append(tp/max(tp+fp+fn, 1))
    return f1, iou


def spatial_row(record, predicted, logits, active, inference_seconds):
    truth = torch.as_tensor(record['positions']).cpu().numpy()[:int(record['count'])]
    pred = np.asarray(predicted, dtype=np.float32).reshape(-1, 2)
    g = g4.g1.gospa_sample(truth, pred)
    band_f1, band_iou = frequency_metrics(logits, record)
    row = dict(record['metadata'], index=record['index'], true_count=len(truth),
        predicted_count=len(active), truth=truth.tolist(), predicted_positions_m=pred.tolist(),
        band_logits=logits.detach().cpu().tolist(), active_slots=list(active),
        gospa_m=float(g['value_m']), matched_errors_m=g4.distance_errors(truth, pred),
        duplicate30=any(np.linalg.norm(a-b) < 30 for a, b in itertools.combinations(pred, 2)),
        band_only_f1=band_f1, band_only_iou=band_iou,
        inference_seconds=float(inference_seconds), slot_position_binding=None)
    for component in ('localization', 'missed', 'false'):
        row[f'gospa_{component}_p_sum'] = float(g[f'{component}_p_sum'])
    for threshold in (10, 30, 50, 100):
        row[f'tp_at_{threshold}m'] = g4.g1.maximum_matches_within(truth, pred, threshold)
    return row


def summarize_hard(rows):
    errors = [e for row in rows for e in row['matched_errors_m']]
    truth = sum(r['true_count'] for r in rows)
    pred = sum(r['predicted_count'] for r in rows)
    result = dict(scenes=len(rows), true_sources=truth, predicted_sources=pred,
        gospa_m=float(np.mean([r['gospa_m'] for r in rows])) if rows else None,
        matched_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        matched_coverage=len(errors)/truth if truth else None,
        matched_median_p90_p95_m=np.quantile(errors, [.5, .9, .95]).tolist() if errors else None,
        count_accuracy=float(np.mean([r['true_count'] == r['predicted_count'] for r in rows])) if rows else None,
        duplicate30_scenes=sum(r['duplicate30'] for r in rows),
        tail500_scenes=sum(max(r['matched_errors_m'], default=0) > 500 for r in rows),
        joint_recall100_f1_08=None, joint_precision100_f1_08=None,
        joint_metric_reason='PB-BASE predicts an unbound position set; no slot-position identity',
        inference_seconds_total=sum(r['inference_seconds'] for r in rows), inference_timing_scope=TIMING_SCOPE,
        inference_seconds_mean=float(np.mean([r['inference_seconds'] for r in rows])) if rows else None)
    for threshold in (10, 30, 50, 100):
        tp = sum(r[f'tp_at_{threshold}m'] for r in rows)
        result[f'recall{threshold}'] = tp/truth if truth else None
        result[f'precision{threshold}'] = tp/pred if pred else None
    for name in ('band_only_f1', 'band_only_iou'):
        values = [v for r in rows for v in r[name]]
        result[name] = float(np.mean(values)) if values else None
    for component in ('localization', 'missed', 'false'):
        result[f'gospa_{component}_mean_p_sum'] = float(np.mean(
            [r[f'gospa_{component}_p_sum'] for r in rows])) if rows else None
    return result


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate_hard(rt, seed, split):
    if split not in ('val_select', 'val_compare'):
        raise ValueError('Hard cascade evaluation accepts development validation only, never test')
    ch3, d8 = rt.model(seed, 'ch3'), rt.model(seed, 'd8')
    ch3.restore(load(rt.best(seed, 'ch3'), rt.out)['state'])
    d8.restore(load(rt.best(seed, 'd8'), rt.out)['state'])
    ch3.mode(False)
    d8.mode(False)
    ids = list(rt.ids('ch3', split))
    progress = rt.progress(f'{seed}/hard_native {split}', len(ids))
    rows = []
    try:
        for n, i in enumerate(ids):
            rt.guard()
            record = rt.data('ch3', seed, split, [i])[0]
            _sync(ch3.device)
            started = time.perf_counter()
            logits = ch3.model(ch3_input(record)[None].to(ch3.device))[0]
            active, selected = predicted_selection(logits)
            if len(active):
                raw = build_predicted_fine(rt.signal(split, i), selected, (rt.lo, rt.hi), d8.device)
                heat, offset = d8.model(d8_input(dict(oracle_fine=raw))[None].to(d8.device))
                positions, scores = decode_hard(heat[0], offset[0], len(active))
            else:
                positions, scores = np.empty((0, 2), dtype=np.float32), np.empty(0, dtype=np.float32)
            _sync(ch3.device)
            elapsed = time.perf_counter()-started
            row = spatial_row(record, positions, logits, active.cpu().tolist(), elapsed)
            row['position_scores'] = scores.tolist()
            rows.append(row)
            progress.update(n+1)
    finally:
        del ch3, d8
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return summarize_hard(rows), rows
