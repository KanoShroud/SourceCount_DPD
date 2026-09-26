"""Batch-one online latency of frozen NEW formal models, without learned caches.

The caller preselects the same 32 validation scenes (8 per K) for every arm.
Truth is not passed to any inference function. This profiles the implemented
online path, not a theoretical minimum or the cached training throughput.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import torch

from 统一模型代码.gates.g7.compact_data import EDGE, COARSE_N, FINE_N, load
from 统一模型代码.gates.g7.compact_foundation import _small_geometry
from 统一模型代码.gates.g7.compact_model import candidates, windows, local_maps, decode
from 统一模型代码.gates.g7.g7_physics import statistics
from 统一模型代码.physics.fine_dpd_autograd import compute_fine_dpd_autograd


TIMING_SCOPE = (
    'IQ_read+all_frequency_201_DPD+41_coherent_statistics+frozen_candidate_network'
    '+candidate_decode_or_predicted_local_windows+local_coherent_statistics'
    '+local_physical_maps+local_network+final_decode; '
    'excludes_common_given_19x41x41_coarse_DPD_generation,model_loading,metric_computation; '
    'batch_one_no_learned_feature_or_DPD_cache; F_local_stats_are_recomputed_not_optimized_away'
)


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


@torch.no_grad()
def profile_online(rt, seed, phase, ids):
    """Return timing-only summary plus per-index times; never read the test set.

    One unreported warmup uses the first selected scene. The default final-report
    caller must supply the SAME preregistered 32 val_compare ids to all phases.
    Smaller id lists are allowed for explicitly labelled engineering checks.
    """
    if phase not in ('candidate', 'f', 's', 'e'):
        raise ValueError('Unknown online timing phase')
    ids = [int(i) for i in ids]
    if not ids or len(set(ids)) != len(ids) or any(i < 0 or i >= rt.config['counts']['val_compare'] for i in ids):
        raise ValueError('Online timing requires unique valid val_compare indices')
    candidate = rt.model(seed, 'candidate')
    candidate.restore(load(rt.best(seed, 'candidate'), rt.out)['state'])
    candidate.mode(False)
    local = None
    if phase != 'candidate':
        local = rt.model(seed, phase)
        local.restore(load(rt.best(seed, phase), rt.out)['state'])
        local.mode(False)
    device = candidate.device
    geo = _small_geometry(str(device))
    fft_ones = torch.ones(geo.N0, dtype=torch.float64, device=device)
    support = torch.ones(geo.N0, dtype=torch.bool, device=device)
    axis = torch.linspace(-EDGE, EDGE, COARSE_N, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    coarse_points = torch.stack([xx.flatten(), yy.flatten()], -1)
    interval_count = len(rt.physics.masks)
    top_k = int(rt.config['top_k'])
    if top_k not in (5, 8):
        raise ValueError('Unexpected local candidate count')

    def infer_one(index):
        rt.guard()
        source = rt.data('ch3', seed, 'val_compare', [index])[0]
        # Only the shared measured coarse input crosses into inference.
        coarse = source['coarse']
        _sync(device)
        started = time.perf_counter()
        iq = rt.signal('val_compare', index)
        fine = compute_fine_dpd_autograd(iq, geo, fft_ones, fixed_support=support,
            grid_chunk_size=2048, frequency_chunk_size=512, real_dtype=torch.float64,
            eig_device='cuda' if device.type == 'cuda' else 'cpu',
            use_checkpoint=False, checkpoint_mode='off').float().cpu()
        if fine.shape != (FINE_N, FINE_N) or not torch.isfinite(fine).all():
            raise RuntimeError('Invalid online all-frequency DPD')
        global_stats = statistics(rt.physics, iq, coarse_points, guard=rt.guard)
        # The global conditional evaluator has no outside-support coefficient.
        global_stats = {name:global_stats[name][:interval_count] for name in ('energy', 'coherent')}
        record = dict(coarse=coarse, fine=fine, statistics=global_stats)
        global_output = candidate.baseline([record], rt.physics)
        if phase == 'candidate':
            decoded = decode(global_output, 'baseline', top_k=8)
        else:
            proposed = candidates(global_output['heat'], top_k=8)[0]
            centers, valid, points, inverse = windows(proposed, top_k=top_k)
            local_stats = statistics(rt.physics, iq, points, guard=rt.guard)
            local_stats.update(centers_m=centers, valid_mask=valid, inverse=inverse)
            if phase == 'f':
                local_stats['full_maps'] = local_maps(rt.physics,
                    torch.ones(1, 3, 19, device=device), [local_stats], full=True)[0].cpu()
            output = local.local([record], rt.physics, [local_stats], phase)
            decoded = decode(output, phase)
        _sync(device)
        elapsed = time.perf_counter()-started
        if len(decoded) != 1:
            raise RuntimeError('Online decoder batch changed')
        # No accuracy metric or GT-dependent matching is performed here.
        return dict(index=index, raw_index=int(source['metadata']['raw_index']),
                    seconds=float(elapsed), predicted_count=len(decoded[0]['active']))

    rows = []
    try:
        warmup = infer_one(ids[0])
        progress = rt.progress(f'{seed}/{phase} 在线全链耗时', len(ids))
        for ordinal, index in enumerate(ids, start=1):
            rows.append(infer_one(index))
            progress.update(ordinal)
        times = np.asarray([r['seconds'] for r in rows], dtype=np.float64)
        return dict(seed=seed, phase=phase, split='val_compare', samples=len(rows),
            indices=ids, warmup_excluded=dict(index=ids[0], seconds=warmup['seconds']),
            mean_seconds=float(times.mean()), median_seconds=float(np.median(times)),
            p95_seconds=float(np.quantile(times, .95)), total_seconds=float(times.sum()),
            timing_scope=TIMING_SCOPE, test_read=False, accuracy_evaluated=False,
            batch_size=1, rows=rows)
    finally:
        candidate = None
        local = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
