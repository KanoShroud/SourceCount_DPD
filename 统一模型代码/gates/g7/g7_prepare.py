"""Approved G7 step one only: input evidence, gradients, replay and measured budget."""
import copy
import gc
import io
import shutil
import time
import traceback

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng, save_checkpoint
from 统一模型代码.gates.g5.r2.g5_r2_prepare import same
from 统一模型代码.gates.g6.coherent_dpd import AtomicDPD, geometry
from 统一模型代码.gates.g6.g6_p0 import overlap_category
from 统一模型代码.gates.g6.g6_p2_runtime import digest_state, g4
from 统一模型代码.gates.g7 import g7_physics as local
from 统一模型代码.gates.g7.g7_runtime import (
    BASE, CONFIG, ARMS, SEEDS, Runtime, Progress, identity, read, write, safe_print)
from 统一模型代码.gates.g7.g7_train import payload, restore, train_step, evaluate


def balanced_ids(targets, n=32):
    generator = torch.Generator().manual_seed(20260924)
    groups = []
    for k in range(4):
        ids = torch.nonzero(targets.counts == k).flatten()
        groups.append(ids[torch.randperm(len(ids), generator=generator)[:n//4]])
    return torch.stack(groups, dim=1).flatten()


def band_mapping(logits, bands, ignore, k):
    cost = np.empty((3, k))
    for q in range(3):
        for t in range(k):
            valid = ignore[t] < .5
            cost[q, t] = float(torch.nn.functional.binary_cross_entropy_with_logits(
                logits[q, valid], bands[t, valid]))
    a, b = linear_sum_assignment(cost)
    return dict(zip(a.tolist(), b.tolist()))


@torch.no_grad()
def input_comparison(runtime, bundle, seed, ids):
    rows = []
    targets, metadata = bundle[1], bundle[2]
    progress = Progress(f'局部输入对照 {seed}', len(ids), runtime.out/'progress.json')
    for n, i in enumerate(ids.tolist(), 1):
        runtime.guard()
        r = runtime.stats(seed, 'val_select', torch.tensor([i]))[0]
        k = int(targets.counts[i])
        mapping = band_mapping(r['frozen_logits'], targets.band[i], targets.ignore[i], k)
        predicted = r['predicted_weights'].cuda()
        truth_weights = predicted.clone()
        for q, t in mapping.items():
            truth_weights[q] = targets.band[i, t].cuda()
        maps = {name:local.raw_maps(runtime.physics, w, r, full=full).cpu()
                for name, w, full in [('full', predicted, True), ('predicted', predicted, False),
                                      ('gt_band', truth_weights, False)]}
        axis = (torch.arange(41)-20)*10
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        sources = []
        for q, t in mapping.items():
            truth = targets.positions[i, t].double()
            centers = r['centers_m'][q]
            covered = (centers-truth).abs().amax(-1) <= 200
            covered &= r['valid_mask'][q].flatten(1).any(-1)
            source = dict(query=q, target=t, covered=bool(covered.any()))
            if covered.any():
                # GT chooses the diagnostic window only, identically for all three inputs.
                dist = (centers-truth).square().sum(-1).masked_fill(~covered, torch.inf)
                j = int(dist.argmin())
                points = centers[j]+torch.stack((xx, yy), -1)
                distance = (points-truth).square().sum(-1).sqrt()
                valid = r['valid_mask'][q, j]
                source.update(window=j, truth=truth.tolist(), center=centers[j].tolist())
                for name, raw in maps.items():
                    z = local.normalize(raw[q, j], valid)
                    flat = int(z.masked_fill(~valid, -torch.inf).argmax())
                    y, x = divmod(flat, 41)
                    near, far = valid & (distance <= 30), valid & (distance > 100)
                    source[name] = dict(peak_error_m=float(distance[y, x]),
                        target_minus_distractor=float(z[near].max()-z[far].max()) if near.any() and far.any() else None)
            sources.append(source)
        union = (targets.band[i, :k] > .5).any(0).double().cuda() if k else torch.zeros(19, device='cuda', dtype=torch.float64)
        occupied = runtime.physics.fft_weights(runtime.physics.weights(union)) > .5
        rows.append(dict(seed=seed, index=i, raw_index=r['raw_index'], count=k,
            snr_db=metadata[i].get('snr_db'), occupancy=float(occupied.double().mean()),
            overlap=overlap_category(targets.band[i, :k].numpy()),
            unique_grid_points=len(r['points']), sources=sources))
        progress.update(n)
    write(runtime.out/f'{seed}_input_samples.json', rows)
    result = summarize_input(rows)
    return result, rows


def summarize_input(rows):
    sources = [s for r in rows for s in r['sources']]
    covered = [s for s in sources if s['covered']]
    result = dict(scenes=len(rows), sources=len(sources), covered_sources=len(covered),
                  coverage=len(covered)/max(len(sources), 1), by_input={})
    for name in ('full', 'predicted', 'gt_band'):
        errors = np.array([s[name]['peak_error_m'] for s in covered])
        gaps = [s[name]['target_minus_distractor'] for s in covered if s[name]['target_minus_distractor'] is not None]
        result['by_input'][name] = dict(mean_peak_error_m=float(errors.mean()) if len(errors) else None,
            median_peak_error_m=float(np.median(errors)) if len(errors) else None,
            **{f'recall{d}':float(np.mean(errors <= d)) if len(errors) else None for d in (10, 30, 50, 100)},
            mean_target_minus_distractor=float(np.mean(gaps)) if gaps else None)
    return result


def physics_probe(runtime, record):
    """Check independent direct FFT evaluation and a small FP64 directional derivative."""
    with g4.g1.SampleStore(record['split']) as store:
        raw, j = store._raw(record['raw_index'])
        signal = (np.asarray(raw['sig_rcv_real_all'][:, :, j], dtype=np.float64).T
                  +1j*np.asarray(raw['sig_rcv_imag_all'][:, :, j], dtype=np.float64).T)
    flat = torch.arange(0, 41*41, 100)[:17]
    ids = record['inverse'][0, 0].flatten()[flat]
    points = record['points'][ids]
    physics = AtomicDPD(runtime.physics.lo, runtime.physics.hi, geometry(points, device='cuda'), False)
    independent = physics.statistics(signal)
    cached = dict(energy=record['energy'][:len(physics.masks)].cuda(),
                  coherent=record['coherent'][:len(physics.masks), ids].cuda())
    for key in cached:
        torch.testing.assert_close(cached[key], independent[key], atol=1e-6, rtol=1e-10)
    p = record['predicted_weights'][0].double().cuda().clamp(.02, .98).requires_grad_(True)
    atom = physics.weights(p)
    actual = physics.evaluate(cached, atom).flatten()
    direct = physics.direct(signal, atom).flatten()
    torch.testing.assert_close(actual, direct, atol=1e-5, rtol=1e-9)
    g7 = local.raw_maps(runtime.physics, p.expand(3, -1), record)[0, 0].flatten()[flat]
    torch.testing.assert_close(g7, actual, atol=1e-5, rtol=1e-9)
    coefficients = torch.linspace(-1, 1, len(actual), device='cuda', dtype=torch.float64)
    def scalar(prob):
        return (physics.evaluate(cached, physics.weights(prob)).flatten().log1p()*coefficients).sum()
    direction = torch.linspace(.1, 1., 19, device='cuda', dtype=torch.float64)
    direction /= direction.norm()
    analytical = (torch.autograd.grad(scalar(p), p)[0]*direction).sum()
    eps = 1e-4
    numerical = (scalar(p.detach()+eps*direction)-scalar(p.detach()-eps*direction))/(2*eps)
    relative = float((analytical-numerical).abs()/torch.maximum(analytical.abs(), numerical.abs()).clamp_min(1e-10))
    if not np.isfinite(relative) or relative > .01:
        raise RuntimeError(f'FP64 local physical gradient mismatch {relative}')
    return dict(direct_max_absolute_error=float((actual-direct).abs().max().detach()),
        local_max_absolute_error=float((g7-actual).abs().max().detach()), directional_relative_error=relative)


def gradient_probe(runtime, context, head, batch, ids, stats, targets, arm):
    result = runtime.forward(context, head, batch, ids, arm, stats)
    _, parts, _ = runtime.loss(result, targets, ids)
    params = list(context.query_builder.parameters())+list(context.ch3.cross_attn.parameters())
    selector = list(head.selector.parameters())
    values = torch.autograd.grad(parts['heatmap']+parts['offset']+parts['candidate'],
        params+selector+[result['band_logits']], allow_unused=True)
    def norm(vs):
        return sum(float(v.square().sum()) for v in vs if v is not None)**.5
    report = dict(query_ch3=norm(values[:len(params)]), selector=norm(values[len(params):-1]),
                  band_logits=norm(values[-1:]))
    if not all(np.isfinite(v) for v in report.values()) or report['band_logits'] != 0:
        raise RuntimeError('Nonfinite gradient or semantic-logit leakage')
    if arm in ('f', 's') and report['query_ch3'] != 0:
        raise RuntimeError('SG localization gradient leaked upstream')
    if arm == 'e' and report['query_ch3'] <= 0:
        raise RuntimeError('E physical feedback is missing')
    if arm in ('s', 'e') and report['selector'] <= 0:
        raise RuntimeError('Selector localization gradient is missing')
    return report


def prepare():
    begin = time.perf_counter()
    out = BASE/f'engineering/{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(out, deadline=time.time()+3600)
    try:
        runtime.preflight()
        safe_print('G7第一步：核对冻结输入与固定样本，不执行正式训练。')
        runtime.audit_raw()
        train, val = runtime.features('train'), runtime.features('val_select')
        train_ids, val_ids = balanced_ids(train[1]), balanced_ids(val[1])
        write(out/'plan.json', dict(config=CONFIG, train_ids=train_ids.tolist(), val_select_ids=val_ids.tolist(),
            contract=identity(BASE/'contract.json'), formal_training_executed=False, test_executed=False))
        cache_reports, inputs, all_rows = {}, {}, []
        for seed in SEEDS:
            cache_reports[f'{seed}_val'] = runtime.ensure_local_cache(seed, 'val_select', val_ids, val)
            inputs[str(seed)], rows = input_comparison(runtime, val, seed, val_ids)
            all_rows += rows
        cache_reports['train'] = runtime.ensure_local_cache(SEEDS[0], 'train', train_ids, train)
        write(out/'input_report.json', dict(seeds=inputs, pooled=summarize_input(all_rows),
            unique_scenes=len({r['raw_index'] for r in all_rows}),
            repeated_seed_observations=True,
            scope='Diagnostic GT-selected nearest covering predicted window; not network localization metrics',
            by_count={str(k):summarize_input([r for r in all_rows if r['count'] == k]) for k in (1, 2, 3)},
            by_occupancy={name:summarize_input([r for r in all_rows if predicate(r['occupancy'])]) for name, predicate in
                [('le_half', lambda x:x <= .5), ('gt_half', lambda x:x > .5)]},
            by_snr={name:summarize_input([r for r in all_rows if r['count'] > 0 and predicate(r['snr_db'])])
                for name, predicate in [('below_minus5', lambda x:x < -5), ('ge_minus5', lambda x:x >= -5)]},
            by_overlap={str(key):summarize_input([r for r in all_rows if r['overlap'] == key])
                for key in sorted({r['overlap'] for r in all_rows})}))
        safe_print('G7第二阶段：FP64局部物理值/梯度、SG/E梯度隔离与断点恢复。')
        nonzero = next(int(i) for i in train_ids if int(train[1].counts[i]) > 0)
        physical = physics_probe(runtime, runtime.stats(SEEDS[0], 'train', torch.tensor([nonzero]))[0])
        reports, first = {}, None
        for arm in ARMS:
            context, head, optimizer, params = runtime.context(SEEDS[0], arm)
            probe_ids = train_ids[:4]
            iterator = runtime.batches(train[0], [probe_ids], 'train', SEEDS[0])
            ids, batch, stats = next(iterator); iterator.close()
            runtime.consumed(train[3], ids)
            g4.set_mode(context, training=False); head.eval()
            with torch.no_grad():
                value = runtime.forward(context, head, batch, ids, arm, stats)
                signature = {key:v.cpu() for key, v in value['output'].items()}
                if arm == 's':
                    first = signature
                elif arm == 'e' and not same(first, signature):
                    raise RuntimeError('S/E initial local outputs differ')
            g4.set_mode(context, training=True); head.train()
            gradients = gradient_probe(runtime, context, head, batch, ids, stats, train[1], arm)
            losses = [train_step(runtime, context, head, optimizer, params, arm, batch, train[1], ids, stats)[0]
                      for _ in range(4)]
            cp = dict(state=payload(context, head), optimizer=optimizer.state_dict(), rng=rng_state())
            path = out/arm/'resume.pt'
            path.parent.mkdir()
            row = save_checkpoint(path, cp)
            del cp
            expected_loss = train_step(runtime, context, head, optimizer, params, arm, batch, train[1], ids, stats)[0]
            expected = digest_state(payload(context, head))
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            saved = torch.load(io.BytesIO(verified_read(row, out/'anomalies')), map_location='cpu', weights_only=False)
            restore(context, head, saved['state']); optimizer.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
            actual_loss = train_step(runtime, context, head, optimizer, params, arm, batch, train[1], ids, stats)[0]
            if actual_loss != expected_loss or digest_state(payload(context, head)) != expected or not same(expected_optimizer, optimizer.state_dict()):
                raise RuntimeError('Local model checkpoint replay mismatch')
            del saved, expected_optimizer, batch, stats, value
            torch.cuda.synchronize()
            start = time.perf_counter()
            iterator = runtime.batches(train[0], list(train_ids.split(4)), 'train', SEEDS[0])
            try:
                for ids, batch, stats in iterator:
                    runtime.consumed(train[3], ids)
                    train_step(runtime, context, head, optimizer, params, arm, batch, train[1], ids, stats)
            finally:
                iterator.close()
            torch.cuda.synchronize()
            training_seconds = time.perf_counter()-start
            start = time.perf_counter()
            evaluate(runtime, context, head, arm, val, SEEDS[0], indices=val_ids.tolist(), label=f'G7/{arm}短验证')
            validation_seconds = time.perf_counter()-start
            reports[arm] = dict(gradient=gradients, short_losses=losses, resume_exact=True,
                train_32_seconds=training_seconds, val_32_seconds=validation_seconds)
            safe_print(f'{arm} 短测：32条训练{training_seconds:.1f}s，验证{validation_seconds:.1f}s，恢复一致。')
            del context, head, optimizer, params, batch, stats
            gc.collect(); torch.cuda.empty_cache()
        runtime.postcheck('preparation')
        runtime.audit_raw()
        elapsed = time.perf_counter()-begin
        previous_seconds = sum(read(p).get('seconds', 0) for p in (BASE/'engineering').glob('*/report.json'))
        previous_seconds += sum(read(p).get('seconds', 0) for p in (BASE/'engineering').glob('*/failure.json'))
        generated_n = sum(r['count'] for r in cache_reports.values())
        generated_bytes = sum(r['bytes'] for r in cache_reports.values())
        per_cache_seconds = sum(r['seconds'] for r in cache_reports.values())/max(generated_n, 1)
        all_n = (4096+512+1024)*len(SEEDS)
        cache_remaining = max(all_n-generated_n, 0)*per_cache_seconds
        storage_remaining = max(all_n-generated_n, 0)*generated_bytes/max(generated_n, 1)
        def forecast(epochs):
            training = sum(r['train_32_seconds']/32*4096*epochs*2 for r in reports.values())
            validation = sum(r['val_32_seconds']/32*(512*(epochs//2+1)+1024)*2 for r in reports.values())
            return elapsed+previous_seconds+1.25*(cache_remaining+training+validation)+1200
        disk_required = int(1.2*storage_remaining+55*1024**3)
        disk_free = shutil.disk_usage(BASE).free
        # GT and predicted are judged by spatial usefulness, not image similarity.
        pool = summarize_input(all_rows)
        f, p, g = [pool['by_input'][k] for k in ('full', 'predicted', 'gt_band')]
        input_pass = (p['recall30'] > f['recall30'] or g['recall30'] > f['recall30']
                      or p['mean_peak_error_m'] < f['mean_peak_error_m'] or g['mean_peak_error_m'] < f['mean_peak_error_m'])
        resources_pass = disk_free >= disk_required
        report = dict(status='ENGINEERING_PASS', physics=physical, arms=reports, inputs=inputs,
            input_check_pass=input_pass, resources_pass=resources_pass,
            forecast_within_budget=forecast(24) <= CONFIG['wall_seconds'],
            forecast_base_seconds=forecast(20), forecast_max_seconds=forecast(24),
            cache_reports=cache_reports, estimated_cache_remaining_seconds=cache_remaining,
            estimated_cache_remaining_bytes=storage_remaining, disk_required_free_bytes=disk_required,
            disk_free_bytes=disk_free, preparation_consumed_seconds=elapsed+previous_seconds, seconds=elapsed,
            peak_ram_percent=runtime.peak_ram, peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            contract_sha256=identity(BASE/'contract.json')['sha256'],
            formal_training_executed=False, val_compare_executed=False, test_executed=False)
        report['ready_for_formal'] = all(report[k] for k in ('input_check_pass', 'resources_pass', 'forecast_within_budget'))
        write(out/'report.json', report)
        write(BASE/'preparation_report.json', {**report, 'report':identity(out/'report.json')})
        summary = ['# G7第一步运行摘要', '', '## Material Passport', '',
            '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
            '- Scope: approved G7 short preparation; no formal training or test', '',
            f'工程检查通过；正式入口就绪：{report["ready_for_formal"]}。',
            f'20轮预计{forecast(20)/3600:.2f}小时；24轮预计{forecast(24)/3600:.2f}小时（含缓存与预留）。',
            f'预计尚需局部缓存{storage_remaining/1024**3:.1f} GiB，可用{disk_free/1024**3:.1f} GiB。', '',
            '输入对照为同一预测窗口内的物理峰诊断，不是整网准确率；详细分层见input_report.json。']
        (out/'运行摘要.md').write_text('\n'.join(summary)+'\n', encoding='utf-8')
        evidence = [identity(p) for p in sorted(out.glob('*.json'))]
        evidence.append(identity(out/'运行摘要.md'))
        write(out/'final_audit_report.json', dict(status='PASS', contract=identity(BASE/'contract.json'),
            outputs=evidence, formal_training_executed=False, test_executed=False,
            scientific_status='INPUT_SUPPORTED_RESOURCE_HOLD' if input_pass and not report['ready_for_formal'] else
                'PREPARATION_READY' if report['ready_for_formal'] else 'INPUT_HOLD'))
        safe_print(f'G7第一步完成：正式就绪={report["ready_for_formal"]}；20/24轮预计{forecast(20)/3600:.2f}/{forecast(24)/3600:.2f}h。')
        return report
    except BaseException:
        write(out/'failure.json', dict(status='FAILED', traceback=traceback.format_exc(), seconds=time.perf_counter()-begin))
        raise
