"""G7 paired F/S/E training; verified epoch commits and development-only evaluation."""
import gc
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.e2e_g5_model import band_values, g4
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng, save_checkpoint
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run
from 统一模型代码.gates.g5.r1.g5_r1_report import paired_many, strata, summarize as spatial_joint_summary
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import summarize, auxiliary_pairs
from 统一模型代码.gates.g5.r2.g5_r2_train import selection_key, must_extend
from 统一模型代码.gates.g6.g6_p0 import overlap_category
from 统一模型代码.gates.g7.g7_runtime import ARMS, SEEDS, CONFIG, Progress, read, write, safe_print


def load_registered(row, out):
    return torch.load(io.BytesIO(verified_read(row, Path(out)/'anomalies')),
                      map_location='cpu', weights_only=False)


def payload(context, head):
    return {'base': g4.state_payload(context), 'local': head.state_dict()}


def restore(context, head, state):
    g4.load_state(context, state['base'])
    head.load_state_dict(state['local'], strict=True)


def train_step(runtime, context, head, optimizer, params, arm, batch, targets, ids, stats):
    runtime.guard()
    result = runtime.forward(context, head, batch, ids, arm, stats)
    loss, parts, diagnostics = runtime.loss(result, targets, ids)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, 10.)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise RuntimeError('Nonfinite G7 loss/gradient')
    optimizer.step()
    return float(loss.detach()), {k: float(v.detach()) if torch.is_tensor(v) else float(v)
                                  for k, v in parts.items()}, diagnostics


@torch.no_grad()
def evaluate(runtime, context, head, arm, bundle, seed, split='val_select', indices=None, label='验证'):
    features, targets, metadata, index, _ = bundle
    indices = list(range(len(targets.counts))) if indices is None else list(indices)
    size = CONFIG['batch_size']
    order = [torch.tensor(indices[i:i+size]) for i in range(0, len(indices), size)]
    g4.set_mode(context, training=False)
    head.eval()
    progress = Progress(label, len(indices), runtime.out/'progress.json')
    iterator = runtime.batches(features, order, split, seed)
    rows = []
    try:
        for ids, batch, stats in iterator:
            runtime.guard()
            runtime.consumed(index, ids)
            result = runtime.forward(context, head, batch, ids, arm, stats)
            decoded = runtime.decode(result)
            oracle = runtime.decode(result, counts=targets.counts[ids]) if split == 'val_compare' else None
            logits = result['band_logits'].detach().cpu()
            for local, i in enumerate(ids.tolist()):
                k = int(targets.counts[i])
                truth = targets.positions[i, :k].numpy()
                bands, ignore = targets.band[i].numpy(), targets.ignore[i].numpy()
                logit, dec = logits[local].numpy(), decoded[local]
                row = R1Run.metric(None, truth, dec['joint'], logit, dec['active'], bands, ignore, metadata[i])
                mapping = {}
                if k:
                    cost = np.empty((3, k))
                    for q in range(3):
                        for t in range(k):
                            valid = targets.ignore[i, t] < .5
                            cost[q, t] = float(torch.nn.functional.binary_cross_entropy_with_logits(
                                logits[local, q, valid], targets.band[i, t, valid]))
                    a, b = linear_sum_assignment(cost)
                    mapping = dict(zip(a.tolist(), b.tolist()))
                row['band_only_f1'], row['band_only_iou'] = band_values(logits[local], targets, i, mapping)
                row.update(index=i, truth=truth.tolist(), band_logits=logit.tolist(), decode=dec,
                           overlap_category=overlap_category(bands[:k]))
                if oracle is not None:
                    d = oracle[local]
                    row['oracle_count_metrics'] = R1Run.metric(None, truth, d['joint'], logit,
                        d['active'], bands, ignore, metadata[i])
                rows.append(row)
            progress.update(len(rows))
    finally:
        iterator.close()
        g4.set_mode(context, training=True)
        head.train()
    return summarize(rows), rows


def initial_signature(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest()


def train_track(runtime, seed, arm, end):
    root = runtime.out/f'training/{seed}/{arm}'
    root.mkdir(parents=True, exist_ok=True)
    bundle = runtime.features('train')
    validation = runtime.features('val_select', bundle[-1])
    context, head, optimizer, params = runtime.context(seed, arm)
    generator = torch.Generator().manual_seed(seed)

    def commit(history, best, steps):
        folder = root/f'checkpoints/epoch{history[-1]["epoch"]:03d}_{time.time_ns()}'
        folder.mkdir(parents=True)
        checkpoint = save_checkpoint(folder/'state.pt', dict(state=payload(context, head),
            optimizer=optimizer.state_dict(), generator=generator.get_state(), rng=rng_state(),
            history=history, best=best, steps=steps, seed=seed, arm=arm))
        write(root/'completed.json', dict(checkpoint=checkpoint, epoch=history[-1]['epoch'], steps=steps))

    try:
        marker_path = root/'completed.json'
        if marker_path.exists():
            saved = load_registered(read(marker_path)['checkpoint'], runtime.out)
            if (saved['seed'], saved['arm']) != (seed, arm):
                raise RuntimeError('G7 recovery track mismatch')
            restore(context, head, saved['state'])
            optimizer.load_state_dict(saved['optimizer'])
            generator.set_state(saved['generator'])
            restore_rng(saved['rng'])
            history, best, steps = saved['history'], saved['best'], saved['steps']
            del saved
        else:
            metrics, rows = evaluate(runtime, context, head, arm, validation, seed, label=f'{seed}/{arm} epoch0')
            if arm in ('s', 'e'):
                marker = root.parent/'se_initial_predictions.json'
                signature = initial_signature(rows)
                if marker.exists() and read(marker)['sha256'] != signature:
                    raise RuntimeError('S/E initialization predictions disagree')
                if not marker.exists():
                    write(marker, {'sha256': signature})
            row = save_checkpoint(root/'initial.pt', {'state': payload(context, head), 'epoch': 0, 'metrics': metrics})
            best = {'epoch': 0, 'metrics': metrics, 'checkpoint': row}
            history, steps = [{'epoch': 0, 'validation': metrics}], 0
            commit(history, best, steps)
        for epoch in range(history[-1]['epoch']+1, end+1):
            g4.set_mode(context, training=True)
            head.train()
            order = torch.randperm(len(bundle[1].counts), generator=generator).split(CONFIG['batch_size'])
            iterator = runtime.batches(bundle[0], order, 'train', seed)
            progress = Progress(f'{seed}/{arm} epoch {epoch}/{end}', len(order), runtime.out/'progress.json')
            start = time.perf_counter()
            losses, sums = [], {}
            try:
                for n, (ids, batch, stats) in enumerate(iterator, 1):
                    runtime.consumed(bundle[3], ids)
                    loss, parts, _ = train_step(runtime, context, head, optimizer, params, arm,
                                               batch, bundle[1], ids, stats)
                    losses.append(loss)
                    steps += 1
                    for key, value in parts.items():
                        sums[key] = sums.get(key, 0)+value
                    progress.update(n)
            finally:
                iterator.close()
            elapsed = time.perf_counter()-start
            metrics, val_seconds = None, 0
            if epoch % CONFIG['evaluate_every'] == 0:
                start = time.perf_counter()
                metrics, _ = evaluate(runtime, context, head, arm, validation, seed, label=f'{seed}/{arm} 验证{epoch}')
                val_seconds = time.perf_counter()-start
                if selection_key(metrics, epoch) < selection_key(best['metrics'], best['epoch']):
                    folder = root/f'selections/epoch{epoch:03d}_{time.time_ns()}'
                    folder.mkdir(parents=True)
                    row = save_checkpoint(folder/'state.pt', {'state': payload(context, head), 'epoch': epoch, 'metrics': metrics})
                    best = {'epoch': epoch, 'metrics': metrics, 'checkpoint': row}
            history.append(dict(epoch=epoch, validation=metrics, training_seconds=elapsed,
                validation_seconds=val_seconds, loss=float(np.mean(losses)),
                loss_components={k: v/len(losses) for k, v in sums.items()}))
            commit(history, best, steps)
            write(root/'history.json', history)
            safe_print(f'{seed}/{arm} 完成{epoch}轮，训练{elapsed:.1f}s，验证{val_seconds:.1f}s；best={best["epoch"]}')
        runtime.postcheck(f'{seed}_{arm}_{end}')
        result = dict(seed=seed, arm=arm, completed_epoch=history[-1]['epoch'], optimizer_steps=steps,
            best=best, extend=must_extend(history, end=CONFIG['base_epochs']),
            budget_unresolved=end == CONFIG['extension_epochs'] and best['epoch'] in (end-2, end),
            total_training_seconds=sum(r.get('training_seconds', 0) for r in history))
        write(root/f'report_epoch{end}.json', result)
        return result
    finally:
        context = head = optimizer = params = bundle = validation = None
        gc.collect()
        torch.cuda.empty_cache()


def train_all(runtime):
    results = {}
    for seed in SEEDS:
        current = {}
        for arm in ARMS:
            current[arm] = train_track(runtime, seed, arm, CONFIG['base_epochs'])
        if any(row['extend'] for row in current.values()):
            for arm in ARMS:
                current[arm] = train_track(runtime, seed, arm, CONFIG['extension_epochs'])
        results[str(seed)] = current
        write(runtime.out/'training_report.json', {'status': 'RUNNING', 'seeds': results})
    report = {'status': 'PASS', 'seeds': results, 'test_executed': False}
    write(runtime.out/'training_report.json', report)
    return report


def build_report(results, out, training, references, hard_references=None):
    report = {'status': 'COMPLETE_FOR_REVIEW', 'training': training, 'tracks': {}, 'paired': {},
              'scope': 'Reused development data, conditional on two seeds; no test', 'test_executed': False}
    for (seed, arm), rows in results.items():
        report['tracks'][f'{seed}_{arm}'] = {'overall': summarize(rows), 'strata': strata(rows),
            'frequency_overlap': {kind: summarize([r for r in rows if r['overlap_category'] == kind])
                for kind in sorted({r['overlap_category'] for r in rows})},
            'oracle_count':spatial_joint_summary([r['oracle_count_metrics'] for r in rows])}
    lines = ['# G7运行摘要', '', '六轨训练和开发集比较完成；test未读取。', '',
             '| 比较 | 联合Recall差/pp | GOSPA差/m | RMSE差/m |', '|---|---:|---:|---:|']
    def compare(pairs):
        combined = paired_many(pairs, seed=20260921, repeats=CONFIG['bootstrap_repeats'])
        combined.update(auxiliary_pairs(pairs, 20260921, CONFIG['bootstrap_repeats']))
        return {'combined': combined, 'per_seed': {str(s): {
            **paired_many([p], seed=s, repeats=CONFIG['bootstrap_repeats']),
            **auxiliary_pairs([p], s, CONFIG['bootstrap_repeats'])} for s, p in zip(SEEDS, pairs)}}
    for name, a, b in [('selected_input_s_minus_f', 'f', 's'), ('feedback_e_minus_s', 's', 'e')]:
        row = compare([(results[s, a], results[s, b]) for s in SEEDS])
        report['paired'][name] = row
        c = row['combined']
        lines.append(f'| {name} | {100*c["joint_recall100_f1_08"]["mean"]:.3f} | '
                     f'{c["gospa_m"]["mean"]:.3f} | {c["matched_rmse_m"]["mean"]:.3f} |')
    report['frozen_p2_b_reference'] = {
        arm: compare([(references[s], results[s, arm]) for s in SEEDS]) for arm in ARMS}
    report['reference_scope'] = 'Frozen P2-B is an absolute reference, not a matched continuation-training contrast'
    if hard_references is not None:
        def spatial_summary(rows):
            errors = [e for r in rows for e in r['matched_errors_m']]
            count = sum(r['true_count'] for r in rows)
            return dict(scenes=len(rows), gospa_m=float(np.mean([r['gospa_m'] for r in rows])),
                matched_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                matched_coverage=len(errors)/max(count, 1),
                **{f'recall{d}':sum(r[f'tp_at_{d}m'] for r in rows)/max(count, 1) for d in (10, 30, 50, 100)})
        report['hard_matched_spatial'] = {}
        for name in ('hard_base', 'r5a_selected'):
            old = hard_references[name]['samples']
            keys = [r['raw_index'] for r in old]
            group = dict(reference=spatial_summary(old), tracks={})
            for (seed, arm), rows in results.items():
                by_id = {r['raw_index']:r for r in rows}
                selected = [by_id[k] for k in keys]
                for a, b in zip(old, selected):
                    np.testing.assert_array_equal(np.asarray(a['true_positions_m']).reshape(-1, 2),
                                                  np.asarray(b['truth']).reshape(-1, 2))
                group['tracks'][f'{seed}_{arm}'] = spatial_summary(selected)
            report['hard_matched_spatial'][name] = group
        report['hard_matched_scope'] = 'Only the aligned 512 multisource scenes; no invented hard-cascade band-binding metrics'
    write(out/'comparison_report.json', report)
    (out/'运行摘要.md').write_text('\n'.join(lines)+'\n\n完整区间、分层Recall、计数频带及尾部见JSON；RMSE须结合覆盖率。\n'
        '工程完成不代表科学假设通过；不设置95%@10m否决条件。\n', encoding='utf-8')
    return report
