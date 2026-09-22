"""Approved G6-P0: 128 fixed development scenes, no network training or test."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONUTF8', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8:replace')

import h5py
import numpy as np
import psutil
import torch

from 统一模型代码.common.g5_verified_io import VerifiedFile, verified_read
from 统一模型代码.gates.g6.coherent_dpd import (
    AtomicDPD, band_assignments, geometry, grid_points, identity_scores,
    normalize_maps, sample_map,
)

ROOT = Path(__file__).resolve().parents[3]
G5 = ROOT/'outputs_e2e/unified/e2e_g5/20260919_approved'
R5 = ROOT/'outputs_e2e/unified/e2e_g5_r5/20260922_152822'
BASE = ROOT/'outputs_e2e/unified/e2e_g6_p0'
CONFIG = {'gate': 'E2E-G6-P0', 'selection_seed': 20260922,
          'counts': [8, 8, 56, 56], 'seeds': [20260921, 20260922],
          'wall_seconds': 3600, 'ram_stop_percent': 85, 'gpu_limit_gib': 14,
          'disk_floor_gib': 50, 'cache_precision': 'complex128/float64', 'evd_grid_chunk': 64,
          'map_relative_l2_tolerance': 1e-10, 'map_max_relative_tolerance': 1e-10,
          'gradient_relative_tolerance': .005, 'difference_steps': [.001, .0003],
          'band_assignment_atol': 1e-10, 'physical_score_tie_atol': 1e-9,
          'bootstrap_seed': 20260922, 'bootstrap_repeats': 2000,
          'training_executed': False, 'test_executed': False, 'val_compare_executed': False}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            digest.update(block)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def announce(message):
    enc = sys.stdout.encoding or 'utf-8'
    print(message.encode(enc, errors='replace').decode(enc), flush=True)


def sync():
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()


class Run:
    def __init__(self, out):
        self.out = out.resolve(strict=True)
        if self.out.parent != BASE.resolve() or any(self.out.iterdir()):
            raise ValueError('New empty isolated output required')
        self.start = time.perf_counter()
        self.inputs = {}
        self.peak_ram = 0.

    def guard(self):
        self.peak_ram = max(self.peak_ram, psutil.virtual_memory().percent)
        if psutil.virtual_memory().percent >= CONFIG['ram_stop_percent']:
            raise RuntimeError('RAM85 stop')
        if time.perf_counter()-self.start > CONFIG['wall_seconds']:
            raise RuntimeError('G6-P0 budget expired')
        if shutil.disk_usage(self.out).free < CONFIG['disk_floor_gib']*1024**3:
            raise RuntimeError('Disk below 50 GiB')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > CONFIG['gpu_limit_gib']*1024**3:
            raise RuntimeError('GPU above approved 14 GiB')

    def get(self, row):
        self.guard()
        p = Path(row['path']).resolve(strict=True)
        if not p.is_relative_to(ROOT/'outputs_e2e') or p.is_relative_to(self.out):
            raise ValueError('Unexpected input or overlapping output')
        self.inputs[str(p)] = row
        return verified_read(row, self.out/'anomalies')

    def audit_mat(self, row):
        self.guard()
        path = Path(row['path']).resolve(strict=True)
        if path.parent != (G5/'input_snapshot').resolve() or path.stat().st_size != row['size_bytes']:
            raise ValueError('Snapshot identity/location failure')
        h = hashlib.sha256()
        for i, part in enumerate(row['blocks']):
            self.guard()
            h.update(verified_read({'path': str(path), **part}, self.out/'anomalies',
                                  offset=i*row['block_size'], length=part['size_bytes']))
        if h.hexdigest() != row['sha256']:
            raise ValueError('Snapshot full SHA mismatch')


def overlap_category(band):
    if len(band) < 2:
        return 'K_LE_1'
    if all(np.array_equal(a, b) for a, b in itertools.combinations(band, 2)):
        return 'identical'
    if not any(np.logical_and(a > .5, b > .5).any() for a, b in itertools.combinations(band, 2)):
        return 'distinct'
    return 'partial'


def select_scenes(records, labels, predictions):
    rng = np.random.default_rng(CONFIG['selection_seed'])
    selected = []
    for k, count in enumerate(CONFIG['counts']):
        bins = defaultdict(list)
        for r in records:
            if r['true_k'] != k:
                continue
            snr = predictions[(CONFIG['seeds'][0], r['raw_index'])]['metrics'][0]['snr_db']
            group = (labels[r['raw_index']]['overlap'], int(np.digitize(snr, [-10, 0, 10])))
            bins[group].append(r)
        for group in bins:
            ids = rng.permutation(len(bins[group]))
            bins[group] = [bins[group][i] for i in ids]
        taken = []
        while len(taken) < count:
            grew = False
            for group in sorted(bins):
                if bins[group] and len(taken) < count:
                    taken.append(bins[group].pop())
                    grew = True
            if not grew:
                raise ValueError('Not enough development samples')
        selected += taken
    return sorted(selected, key=lambda r: r['raw_index'])


def summarize_identity(rows):
    result = {}
    for mode in ['label']+[f'pred_{s}' for s in CONFIG['seeds']]:
        subset = [r for r in rows if r['mode'] == mode and r.get('eligible')]
        result[mode] = {}
        for group in ('all', 'distinct', 'partial', 'identical'):
            group_rows = [r for r in subset if group == 'all' or r['overlap'] == group]
            output = {'scenes': len(group_rows)}
            for space in ('grid', 'exact'):
                data = [r[space] for r in group_rows if r[space]['informative']]
                entry = {'informative_scenes': len(data)}
                if data:
                    acc = np.array([v['accuracy'] for v in data])
                    chance = np.array([v['chance'] for v in data])
                    rng = np.random.default_rng(CONFIG['bootstrap_seed'])
                    boot = rng.choice(acc-chance, (CONFIG['bootstrap_repeats'], len(data))).mean(1)
                    margins = [m for v in data for m in v['pair_margins']]
                    entry.update(accuracy=float(acc.mean()), chance=float(chance.mean()),
                        excess_over_chance=float((acc-chance).mean()),
                        excess_95ci=np.quantile(boot, [.025, .975]).tolist(),
                        joint_margin_median=float(np.median([v['joint_margin'] for v in data])),
                        pair_margin_median=float(np.median(margins)),
                        pair_positive_fraction=float(np.mean(np.array(margins) > 1e-9)))
                output[space] = entry
            result[mode][group] = output
        result[mode]['ineligible_counts'] = dict((reason, sum(r.get('reason') == reason for r in rows if r['mode'] == mode))
                                               for reason in ('K_LE_1', 'COUNT_MISMATCH', 'EMPTY_LABEL_BAND'))
    return result


def compare_map(cached, direct):
    diff = cached-direct
    rel = float(diff.norm()/direct.norm().clamp_min(1e-20))
    maxrel = float(diff.abs().max()/direct.abs().max().clamp_min(1e-20))
    ia, ib = int(cached.argmax()), int(direct.argmax())
    w = cached.shape[-1]
    peak_distance = float(np.linalg.norm(np.array(divmod(ia, w))-np.array(divmod(ib, w)))*50)
    return {'relative_l2': rel, 'max_abs': float(diff.abs().max()), 'max_relative': maxrel,
            'peak_difference_m': peak_distance,
            'pass': rel <= CONFIG['map_relative_l2_tolerance'] and maxrel <= CONFIG['map_max_relative_tolerance']}


def gradient_check(physics, stats, signal, probabilities, raw_index):
    """Compare cached and original direct-IQ scalar objectives on nine fixed grid points."""
    ix = torch.linspace(0, physics.geo.num_grid-1, 9, device=physics.geo.device).long()
    small_geo = geometry(grid_points()[ix.cpu()], device=physics.geo.device,
                         n_fft=physics.geo.N0, fs=physics.geo.fs)
    small = AtomicDPD(physics.lo, physics.hi, small_geo, precompute_phase=False)
    small_stats = {'energy': stats['energy'], 'coherent': stats['coherent'][:, ix]}
    gen = torch.Generator(device='cpu').manual_seed(CONFIG['selection_seed']+raw_index)
    # Random interior probabilities avoid the nondifferentiable all-zero boundary.
    z = torch.logit(probabilities.detach().double().clamp(.05, .95)).requires_grad_(True)
    objective_weights = torch.linspace(-.4, .8, 9, dtype=torch.float64, device=z.device)

    def scalar(value, direct=False):
        a = small.weights(value.sigmoid())
        dpd = small.direct(signal, a) if direct else small.evaluate(small_stats, a)[0]
        return (dpd.flatten().log1p()*objective_weights).mean()

    grad = torch.autograd.grad(scalar(z), z)[0]
    direct_grad = torch.autograd.grad(scalar(z, True), z)[0]
    relative = float((grad-direct_grad).norm()/direct_grad.norm().clamp_min(1e-12))
    checks = []
    for _ in range(2):
        direction = torch.randn(z.shape, generator=gen, dtype=torch.float64).to(z.device)
        direction /= direction.norm()
        auto = float((grad*direction).sum())
        for h in CONFIG['difference_steps']:
            with torch.no_grad():
                finite = float((scalar(z+h*direction)-scalar(z-h*direction))/(2*h))
            error = abs(auto-finite)/max(abs(auto), abs(finite), 1e-10)
            checks.append({'step': h, 'autograd': auto, 'finite_difference': finite, 'relative_error': error,
                           'pass': error <= CONFIG['gradient_relative_tolerance'] or abs(auto-finite) <= 1e-9})
    return {'raw_index': raw_index, 'cached_vs_direct_gradient_relative': relative,
            'directions': checks, 'pass': relative <= 1e-7 and all(c['pass'] for c in checks)}


def execute(runtime):
    out = runtime.out
    # R5 registers exactly the logits consumed here and the G5 input manifest.
    audit5 = json.loads(runtime.get(identity(R5/'final_audit.json')))
    if audit5['status'] != 'PASS':
        raise ValueError('R5 audit not PASS')
    registered = {Path(r['path']).name: r for r in audit5['files']}
    audit_inputs = json.loads(runtime.get(registered['input_audit.json']))
    manifest_rows = [r for r in audit_inputs['files'] if Path(r['path']).resolve() == (G5/'manifest.json').resolve()]
    if len(manifest_rows) != 1:
        raise ValueError('Missing registered G5 manifest')
    manifest = json.loads(runtime.get(manifest_rows[0]))
    predictions = {(r['seed'], r['raw_index']): r for r in
                   map(json.loads, runtime.get(registered['samples.jsonl']).decode('utf-8').splitlines())}
    mats = {Path(r['path']).name: r for r in manifest['inputs']['files']}
    coarse, raw = mats['files_01_val_select.mat'], mats['files_03_val_data.mat']
    for row in (coarse, raw):
        runtime.audit_mat(row)
    labels = {}
    with VerifiedFile(coarse, out/'anomalies') as stream, h5py.File(stream, 'r') as h:
        lo, hi = (np.asarray(h[key]).reshape(-1) for key in ('sub_f_lo_val', 'sub_f_hi_val'))
        fs = float(np.asarray(h['fs_val']).reshape(-1)[0])
        for r in manifest['subsets']['val_select']:
            i, k = r['local_index'], r['true_k']
            if int(h['src_count_all'][0, i]) != k:
                raise ValueError('Count identity mismatch')
            band = np.asarray(h['band_mask_all'][:, :, i], dtype=float).T[:k]
            ignore = np.asarray(h['ignore_mask_all'][:, :, i], dtype=float).T[:k]
            points = np.asarray(h['src_pos_all'][:, :, i], dtype=float).T[:k]
            labels[r['raw_index']] = {'band': band, 'ignore': ignore, 'points': points,
                                      'overlap': overlap_category(band)}
    selected = select_scenes(manifest['subsets']['val_select'], labels, predictions)
    if len(selected) != 128 or fs != 100e6:
        raise ValueError('Unexpected P0 data contract')
    records = [{**r, 'overlap': labels[r['raw_index']]['overlap'],
                'snr_db': predictions[(CONFIG['seeds'][0], r['raw_index'])]['metrics'][0]['snr_db']} for r in selected]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    code = sorted(set([Path(__file__).resolve(), Path(__file__).with_name('coherent_dpd.py'),
                       Path(__file__).with_name('test_g6_p0.py'),
                       ROOT/'统一模型代码/physics/fine_dpd_autograd.py',
                       ROOT/'统一模型代码/common/g5_verified_io.py',
                       ROOT/'统一模型代码/common/verified_feature_loader.py']))
    audit_ids = [next(r['raw_index'] for r in selected if r['true_k'] == k) for k in range(4)]
    contract = {'config': CONFIG, 'scenes': records, 'audited_raw_ids': audit_ids,
                'inputs': list(runtime.inputs.values()), 'mat_inputs': [coarse, raw],
                'code': [identity(p) for p in code], 'device': device,
                'software': {'torch': torch.__version__, 'numpy': np.__version__, 'h5py': h5py.__version__, 'python': sys.version},
                'frequency_mapping': 'Per-slot noisy union on half-open atomic intervals; no slot union',
                'normalization': 'Energy/Nfft^2 power, floor1e-20; diagonal1e-6; absolute max eigenvalue',
                'identity': 'Label hard19 positives only; predicted band-only BCE assignment ignores ignore labels',
                'stats': 'Descriptive scene bootstrap per mode; not independent evidence or a new rejection gate',
                'limits': 'Physical representation probe only; no updated network/system performance'}
    write(out/'contract.json', contract)
    write(out/'progress.json', {'stage': 'PREFLIGHT_COMPLETE', 'scenes': 128})
    announce('阶段1完成：128条场景及两seed已固定；开始物理缓存与数值验证。')
    t = time.perf_counter()
    physics = AtomicDPD(lo, hi, geometry(grid_points(), device=device, shape=(81, 81)))
    sync()
    geometry_seconds = time.perf_counter()-t
    runtime.guard()
    write(out/'physical_contract.json', {'intervals_hz': physics.intervals,
          'coverage': physics.coverage.cpu().tolist(), 'grid_order': 'y,x',
          'interval_bin_counts': physics.masks.sum(1).cpu().tolist(),
          'phase_precompute_seconds': geometry_seconds})
    rows, numerical, gradients, cache_index, timings, profiles = [], [], [], [], [], []
    (out/'cache').mkdir()
    (out/'maps').mkdir()
    process_start = time.perf_counter()
    with VerifiedFile(raw, out/'anomalies') as stream, h5py.File(stream, 'r') as h:
        if h['sig_rcv_real_all'].shape[:2] != (4096, 4):
            raise ValueError('IQ shape mismatch')
        for index, record in enumerate(records):
            runtime.guard()
            began = time.perf_counter()
            i, k = record['raw_index'], record['true_k']
            label = labels[i]
            if int(h['src_count_all'][0, i]) != k:
                raise ValueError('IQ source count mismatch')
            raw_points = np.asarray(h['src_pos_all'][:, :, i], dtype=float).T[:k]
            raw_band = np.asarray(h['band_mask_all'][:, :, i], dtype=float).T[:k]
            np.testing.assert_array_equal(raw_points, label['points'])
            np.testing.assert_array_equal(raw_band, label['band'])
            signal = (np.asarray(h['sig_rcv_real_all'][:, :, i], dtype=np.float64).T
                      + 1j*np.asarray(h['sig_rcv_imag_all'][:, :, i], dtype=np.float64).T)
            read_seconds = time.perf_counter()-began
            t = time.perf_counter()
            with torch.no_grad():
                stats = physics.statistics(signal)
            sync()
            precompute_seconds = time.perf_counter()-t
            path = out/f'cache/{i:05d}.pt'
            t = time.perf_counter()
            torch.save({key: value.cpu() for key, value in stats.items()}, path)
            cache_row = identity(path)
            cache_index.append(cache_row)
            # The decoded cache is exactly the SHA-verified byte stream.
            cpu_stats = torch.load(io.BytesIO(verified_read(cache_row, out/'anomalies')),
                                   map_location='cpu', weights_only=True)
            stats = {key: value.to(device) for key, value in cpu_stats.items()}
            cache_seconds = time.perf_counter()-t
            modes = [('label', torch.as_tensor(label['band'], dtype=torch.float64, device=device), None)]
            for seed in CONFIG['seeds']:
                source = predictions[(seed, i)]
                logits = torch.tensor(source['band_logits'], dtype=torch.float32, device=device)
                active = torch.nonzero(logits.amax(-1) >= 0).flatten()
                modes.append((f'pred_{seed}', logits[active].sigmoid().double(), seed))
            if i in audit_ids:
                rng = np.random.default_rng(CONFIG['selection_seed']+i)
                cases = [('one_interval', torch.eye(len(physics.intervals), dtype=torch.float64, device=device)[0]),
                         ('hard_band', physics.weights(torch.tensor((rng.random(19) > .5).astype(float), device=device))),
                         ('soft_band', physics.weights(torch.tensor(rng.uniform(.1, .9, 19), device=device))),
                         ('predicted_band', physics.weights(torch.tensor(predictions[(CONFIG['seeds'][0], i)]['band_logits'][0],
                                                                         dtype=torch.float32, device=device).sigmoid().double()))]
                for name, weights in cases:
                    runtime.guard()
                    with torch.no_grad():
                        direct = physics.direct(signal, weights)
                        cached = physics.evaluate(stats, weights)[0]
                        item = {'raw_index': i, 'case': name, **compare_map(cached, direct)}
                    numerical.append(item)
                    if not item['pass']:
                        write(out/'numerical_audit.json', numerical)
                        raise RuntimeError('Cached/direct DPD equivalence failed')
                if k:
                    g = gradient_check(physics, stats, signal, modes[1][1][0] if len(modes[1][1]) else
                                       torch.full((19,), .5, dtype=torch.float64, device=device), i)
                    gradients.append(g)
                    if not g['pass']:
                        write(out/'gradient_audit.json', gradients)
                        raise RuntimeError('Physical gradient audit failed')
            if len(profiles) < 8 and k >= 2:
                probe = torch.tensor(predictions[(CONFIG['seeds'][0], i)]['band_logits'],
                                     dtype=torch.float64, device=device, requires_grad=True)
                sync()
                t = time.perf_counter()
                maps_probe = physics.evaluate(stats, physics.weights(probe.sigmoid()))
                sync()
                forward_time = time.perf_counter()-t
                ramp = torch.linspace(-1, 1, 81, dtype=torch.float64, device=device)
                objective = (maps_probe.log1p()*ramp[None, :, None]).mean()
                t = time.perf_counter()
                objective.backward()
                sync()
                profiles.append({'raw_index': i, 'three_slot_forward_seconds': forward_time,
                                 'backward_seconds': time.perf_counter()-t,
                                 'gradient_norm': float(probe.grad.norm())})
                del maps_probe, objective, probe
            saved_maps = {'points_m': label['points'], 'band_truth': label['band']}
            t = time.perf_counter()
            for mode, probabilities, seed in modes:
                entry = {'raw_index': i, 'k': k, 'mode': mode, 'seed': seed,
                         'overlap': label['overlap'], 'snr_db': record['snr_db'],
                         'predicted_count': len(probabilities), 'eligible': False}
                if not len(probabilities):
                    entry['reason'] = 'K_LE_1' if k < 2 else 'COUNT_MISMATCH'
                    rows.append(entry)
                    continue
                with torch.no_grad():
                    weights = physics.weights(probabilities)
                    maps = physics.evaluate(stats, weights)
                    normalized, mean, std = normalize_maps(maps)
                saved_maps[mode] = normalized.cpu().numpy()
                if k < 2:
                    entry['reason'] = 'K_LE_1'
                elif len(probabilities) != k:
                    entry['reason'] = 'COUNT_MISMATCH'
                elif (label['band'].sum(1) == 0).any():
                    entry['reason'] = 'EMPTY_LABEL_BAND'
                else:
                    perms, correct, cost = band_assignments(probabilities.cpu().numpy(), label['band'], label['ignore'])
                    if mode == 'label':
                        correct = np.array([all(np.array_equal(label['band'][q], label['band'][s])
                                                for q, s in enumerate(p)) for p in perms])
                    exact_geo = geometry(label['points'], device=device)
                    with torch.no_grad():
                        exact = torch.stack([physics.direct(signal, w, exact_geo).flatten() for w in weights])
                        exact = (exact.log1p()-mean.flatten()[:, None])/std.flatten()[:, None]
                        sampled = sample_map(normalized, label['points'])
                    entry.update(eligible=True, band_cost=cost.tolist(), permutations=perms.tolist(),
                        grid=identity_scores(sampled.cpu().numpy(), perms, correct),
                        exact=identity_scores(exact.cpu().numpy(), perms, correct))
                rows.append(entry)
            sync()
            forward_seconds = time.perf_counter()-t
            np.savez(out/f'maps/{i:05d}.npz', **saved_maps)
            timings.append({'raw_index': i, 'read_seconds': read_seconds,
                            'cache_precompute_seconds': precompute_seconds,
                            'cache_roundtrip_seconds': cache_seconds,
                            'all_modes_grid_and_exact_seconds': forward_seconds,
                            'total_seconds': time.perf_counter()-began})
            if index == 3:
                elapsed = time.perf_counter()-process_start
                estimate = elapsed/4*128*1.5 + 120
                write(out/'pilot.json', {'scenes': 4, 'seconds': elapsed, 'conservative_total_seconds': estimate})
                if time.perf_counter()-runtime.start+estimate > CONFIG['wall_seconds']:
                    raise RuntimeError('Pilot exceeds approved 1h budget')
            if (index+1) % 32 == 0:
                write(out/'progress.json', {'stage': 'PHYSICS', 'scenes': index+1, 'total': 128})
                announce(f'物理阶段：{index+1}/128，已用{(time.perf_counter()-runtime.start)/60:.1f}分钟。')
    write(out/'cache_index.json', cache_index)
    write(out/'numerical_audit.json', numerical)
    write(out/'gradient_audit.json', gradients)
    write(out/'timings.json', timings)
    write(out/'physical_profile.json', {'scope': 'Isolated three-slot physics only; not a full training epoch estimate',
                                      'samples': profiles})
    with (out/'samples.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    summary = summarize_identity(rows)
    report = {'status': 'COMPLETED', 'config': CONFIG, 'identity': summary,
              'numerical_pass': all(r['pass'] for r in numerical),
              'gradient_pass': all(r['pass'] for r in gradients),
              'cache_bytes': sum(r['size_bytes'] for r in cache_index),
              'phase_precompute_seconds': geometry_seconds,
              'timing_medians_seconds': {key: float(np.median([r[key] for r in timings]))
                                        for key in timings[0] if key != 'raw_index'},
              'scope': 'Only physical identity evidence at true positions; not system joint recall or RMSE'}
    announce('阶段2完成：128条物理证据已计算；开始输入复核、缓存回读和结果汇总。')
    for row in (coarse, raw):
        runtime.audit_mat(row)
    for row in list(runtime.inputs.values()):
        runtime.get(row)
    for row in contract['code']:
        verified_read(row, out/'anomalies')
    for row in cache_index:
        runtime.guard()
        verified_read(row, out/'anomalies')
    report.update(wall_seconds=time.perf_counter()-runtime.start, peak_ram_percent=runtime.peak_ram,
                  peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/1024**3 if torch.cuda.is_initialized() else 0,
                  inputs_pre_post_verified=True)
    write(out/'report.json', report)
    lines = ['# E2E-G6-P0运行结论', '', '本轮仅检查物理表征；未训练、未读取val_compare/test。', '',
             '| 输入 | 可区分场景 | 网格配对正确率 | 精确坐标配对正确率 | 随机排列参考 |', '|---|---:|---:|---:|---:|']
    for mode, values in summary.items():
        g, e = values['all']['grid'], values['all']['exact']
        lines.append(f"| {mode} | {g['informative_scenes']} | {g.get('accuracy', 0):.4%} | {e.get('accuracy', 0):.4%} | {g.get('chance', 0):.4%} |")
    lines += ['', f"总墙钟{report['wall_seconds']:.2f}秒；缓存{report['cache_bytes']/1024**3:.3f} GiB。",
              '这些是已知真实位置间的身份判别，不是端到端系统性能或新独立测试结果。后续训练另行审批。']
    (out/'运行结论.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(out/'input_audit.json', {'status': 'PASS', 'files': list(runtime.inputs.values()),
                                 'mat_inputs': [coarse, raw], 'same_bytes_verified_consumed': True})
    write(out/'progress.json', {'stage': 'COMPLETED', 'scenes': 128})
    files = [identity(p) for p in sorted(out.rglob('*')) if p.is_file() and 'anomalies' not in p.parts]
    for row in files:
        runtime.guard()
        verified_read(row, out/'anomalies')
    write(out/'final_audit.json', {'status': 'PASS', 'files': files,
                                  'wall_seconds_including_output_readback': time.perf_counter()-runtime.start})
    announce(f'G6-P0完成：{out}')


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out = BASE/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    runtime = Run(out)
    announce(f'G6-P0开始：{out}')
    try:
        execute(runtime)
    except Exception:
        write(out/'failure.json', {'status': 'FAILED', 'traceback': traceback.format_exc(),
                                  'wall_seconds': time.perf_counter()-runtime.start,
                                  'peak_ram_percent': runtime.peak_ram,
                                  'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else 0})
        raise


if __name__ == '__main__':
    main()
