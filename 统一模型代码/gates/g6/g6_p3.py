"""G6-P3: approved frozen-model candidate sufficiency and selection diagnosis."""
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run
from 统一模型代码.gates.g5.r1.g5_r1_report import summarize
from 统一模型代码.gates.g5.r2.g5_r2_model import candidate_batch, decode_r2
from 统一模型代码.gates.g6.g6_p1_speed import physical_batches
from 统一模型代码.gates.g6.g6_p2_diagnose import load_model, representations
from 统一模型代码.gates.g6.g6_p2_runtime import (
    BASE as P2, ROOT, Runtime as P2Runtime, SEEDS, identity, read,
    safe_print, setup_environment, write)
from 统一模型代码.gates.g6.g6_p3_candidates import (
    THRESHOLDS, ceilings, decode, missing_ranks, rescore_base, summarize_candidates)

BASE = ROOT/'outputs_e2e/unified/e2e_g6_p3'
MODES = ('base', 'fixed_physical', 'final')


class Runtime(P2Runtime):
    def __init__(self, out, deadline):
        self.out = Path(out).resolve(strict=True)
        if not self.out.is_relative_to(BASE.resolve()) or self.out.is_relative_to(P2.resolve()):
            raise ValueError('P3 output isolation violation')
        self.deadline = deadline
        self.inputs, self.ranges, self.physical_consumed = {}, {}, {}
        self.peak_ram = 0
        self.manifest = self.fm = self.physics = None
        self._cache_indexes = {}


def load_json(row, out):
    return json.loads(verified_read(row, out/'anomalies'))


def read_inputs(out):
    audit_path = P2/'run/evaluation/final_audit_report.json'
    audit = read(audit_path)
    if audit['status'] != 'PASS' or not audit['six_tracks_complete']:
        raise RuntimeError('P2 incomplete')
    inputs = [identity(audit_path), audit['contract']]
    comp_row = next(r for r in audit['outputs'] if Path(r['path']).name == 'comparison_report.json')
    comparison = load_json(comp_row, out); inputs.append(comp_row)
    historical = {}
    for seed in SEEDS:
        for arm in ('b', 'c'):
            key = f'{seed}_{arm}'
            marker_row = next(r for r in audit['outputs'] if Path(r['path']).name == f'{key}_complete.json')
            marker = load_json(marker_row, out)
            raw = verified_read(marker['samples'], out/'anomalies').decode('utf-8')
            historical[key] = {r['raw_index']: r for r in map(json.loads, raw.splitlines())}
            inputs.extend([marker_row, marker['samples'], marker['checkpoint']])
    r1contract = ROOT/'outputs_e2e/unified/e2e_g5_r1/20260920_approved/contract.json'
    contract = read(r1contract); inputs.append(identity(r1contract))
    hard_row = next(r for r in contract['files'] if Path(r['path']).name == 'hard_reference_report.json')
    hard = load_json(hard_row, out); inputs.append(hard_row)
    if hard['status'] != 'PASS' or hard['test_executed']:
        raise RuntimeError('Hard reference audit failed')
    references = hard['references']
    ids = {r['raw_index'] for r in references['hard_base']['samples']}
    assert len(ids) == 512
    assert ids == {r['raw_index'] for r in references['r5a_selected']['samples']}
    for row in inputs:
        verified_read(row, out/'anomalies')
    return comparison['training'], historical, references, ids, inputs


@torch.no_grad()
def run_samples(runtime, bundle, split, indices, context, head, arm, historical=None):
    features, targets, metadata, index, _ = bundle
    rows = []
    iterator = physical_batches(runtime, features, [torch.tensor(indices[i:i+4])
        for i in range(0, len(indices), 4)], split, True)
    try:
        for ids, batch, stats in iterator:
            runtime.guard(); runtime.consumed(index, ids)
            base, _, residual = representations(context, head, batch, ids, arm, runtime.physics, stats)
            final_heat = base[3]+residual['full']
            records = {'base': candidate_batch(base[3], base[4]), 'final': candidate_batch(final_heat, base[4])}
            probabilities = {'base': base[3].sigmoid().cpu().numpy(), 'final': final_heat.sigmoid().cpu().numpy()}
            offsets = base[4].cpu().numpy(); logits = base[1].cpu().numpy()
            for j, i in enumerate(ids.tolist()):
                k = int(targets.counts[i]); truth = targets.positions[i, :k].numpy()
                bands, ignore = targets.band[i].numpy(), targets.ignore[i].numpy()
                z = logits[j]; candidates = {s: records[s][j] for s in ('base', 'final')}
                candidates['fixed_physical'] = rescore_base(candidates['base'], probabilities['final'][j])
                decoded = {m: decode(z, candidates[m]) for m in MODES}
                for m in ('base', 'final'):
                    native = decode_r2(z, candidates[m], None)
                    np.testing.assert_array_equal(decoded[m]['joint'], native['joint'])
                for q, point in zip(decoded['fixed_physical']['active'], decoded['fixed_physical']['joint']):
                    p = np.asarray(candidates['base']['candidates'][q]['positions'], dtype=np.float32)
                    assert (p == np.asarray(point, dtype=np.float32)).all(1).any()
                metrics = {m: R1Run.metric(None, truth, d['joint'], z, d['active'], bands, ignore, metadata[i])
                           for m, d in decoded.items()}
                for m in MODES:
                    metrics[m]['fallback'] = decoded[m]['fallback']
                upper = {s: ceilings(truth, candidates[s], z, bands, ignore) for s in ('base', 'final')}
                for m, s in [('base', 'base'), ('fixed_physical', 'base'), ('final', 'final')]:
                    if not decoded[m]['fallback']:
                        assert metrics[m]['joint_tp'] <= upper[s]['joint100']
                        for d in THRESHOLDS:
                            assert metrics[m][f'tp_at_{d}m'] <= upper[s]['feasible'][str(d)] <= upper[s]['union'][str(d)]
                if historical is not None:
                    old = historical[metadata[i]['raw_index']]
                    assert old['true_count'] == k
                    np.testing.assert_allclose(truth, old['truth'], rtol=0, atol=0)
                    np.testing.assert_array_equal(z, old['band_logits'])
                    np.testing.assert_allclose(metrics['final']['predicted_positions_m'], old['predicted_positions_m'], rtol=0, atol=1e-4)
                    for field in ('gospa_m', 'joint_tp', 'predicted_count', 'tp_at_10m', 'tp_at_30m', 'tp_at_50m', 'tp_at_100m'):
                        np.testing.assert_allclose(metrics['final'][field], old[field], rtol=0, atol=1e-5)
                rows.append(dict(index=i, raw_index=metadata[i]['raw_index'], true_count=k,
                    predicted_count=len(decoded['final']['active']), truth=truth.tolist(),
                    logits=z.tolist(), band=bands.tolist(), ignore=ignore.tolist(),
                    metrics=metrics, decoded=decoded, candidates={s:candidates[s] for s in ('base', 'final')},
                    ceilings=upper, missing_ranks={s: missing_ranks(truth, candidates[s], probabilities[s][j], offsets[j])
                                                for s in ('base', 'final')}))
    finally:
        iterator.close()
    return rows


def summarize_run(rows):
    return dict(actual={m: summarize([r['metrics'][m] for r in rows]) for m in MODES},
        candidates=summarize_candidates(rows),
        fallback={m: sum(r['metrics'][m]['fallback'] for r in rows) for m in MODES})


def markdown(report):
    lines = ['# G6-P3诊断结果', '', '冻结checkpoint、无训练；候选上限使用GT，只作诊断。两组数据均为开发证据。', '']
    for split in ('val_select', 'hard_matched'):
        lines += [f'## {split}', '', '以下为两个seed指标的算术平均，RMSE未合并重算。', '',
            '|模型/候选|union10|union30|union50|union100|feasible10|feasible100|joint上限|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ('b','c'):
            for source in ('base','final'):
                values = [report['tracks'][f'{seed}_{arm}'][split]['candidates'][source] for seed in SEEDS]
                numbers = [np.mean([v['union'][str(d)] for v in values])*100 for d in THRESHOLDS]
                numbers += [np.mean([v['feasible'][str(d)] for v in values])*100 for d in (10,100)]
                numbers += [np.mean([v['joint100'] for v in values])*100]
                lines.append(f'|{arm}/{source}|'+'|'.join(f'{n:.4f}%' for n in numbers)+'|')
        lines += ['', '|模型/方式|联合Recall|GOSPA/m|RMSE/m|覆盖率|R10|R100|', '|---|---:|---:|---:|---:|---:|---:|']
        for arm in ('b','c'):
            for mode in MODES:
                values = [report['tracks'][f'{seed}_{arm}'][split]['actual'][mode] for seed in SEEDS]
                names = ('joint_recall100_f1_08','gospa_m','matched_rmse_m','matched_coverage','recall10','recall100')
                numbers = [np.mean([v[n] for v in values])*(1 if n in ('gospa_m','matched_rmse_m') else 100) for n in names]
                lines.append(f'|{arm}/{mode}|'+'|'.join(f'{n:.4f}' for n in numbers)+'|')
        lines.append('')
    lines += ['## 同样本硬级联参考', '', '仅K=2/3空间指标；预训练与输入条件不同，不作架构因果比较。', '',
              '|参考|GOSPA/m|RMSE/m|覆盖率|R10|R100|', '|---|---:|---:|---:|---:|---:|']
    for name, v in report['hard_reference'].items():
        lines.append(f"|{name}|{v['gospa_mean_m']:.4f}|{v['matched_errors_m']['rmse']:.4f}|{v['matched_pair_coverage_of_true']*100:.4f}%|{v['recall_at_10m']*100:.4f}%|{v['recall_at_100m']*100:.4f}%|")
    return '\n'.join(lines)+'\n'


def main():
    setup_environment(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = BASE/time.strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter(); runtime = Runtime(out, time.time()+3600)
    report = dict(status='RUNNING', training_executed=False, test_executed=False, tracks={})
    try:
        training, historical, hard, hard_ids, inputs = read_inputs(out)
        code_paths = {Path(__file__).resolve(), Path(__file__).with_name('g6_p3_candidates.py').resolve(),
                      Path(__file__).with_name('test_g6_p3.py').resolve()}
        for module in tuple(sys.modules.values()):
            value = getattr(module, '__file__', None)
            if isinstance(value, str) and Path(value).is_absolute():
                p = Path(value).resolve()
                if p.is_relative_to(ROOT/'统一模型代码') and p.suffix == '.py':
                    code_paths.add(p)
        codes = [identity(p) for p in sorted(code_paths)]
        runtime.preflight(); runtime.audit_raw()
        select = runtime.features('val_select'); compare = runtime.features('val_compare', select[-1])
        compare_ids = [i for i,r in enumerate(compare[2]) if r['raw_index'] in hard_ids]
        assert len(select[1].counts) == 512 and len(compare_ids) == 512
        # Preserve historical 4-sample batch composition for numerical replay.
        assert all(compare_ids[i:i+4] == list(range(compare_ids[i], compare_ids[i]+4))
                   and compare_ids[i] % 4 == 0 for i in range(0,512,4))
        hard_by_id = {r['raw_index']: r for r in hard['hard_base']['samples']}
        for i in compare_ids:
            h = hard_by_id[compare[2][i]['raw_index']]; k = int(compare[1].counts[i])
            assert k == h['true_count'] and k in (2,3)
            np.testing.assert_allclose(compare[1].positions[i,:k].numpy(), h['true_positions_m'], rtol=0, atol=1e-4)
        plan = dict(gate='G6-P3', seeds=list(SEEDS), arms=['b','c'], modes=list(MODES),
            inputs=inputs, code=codes, torch=torch.__version__, numpy=np.__version__, budget_seconds=3600,
            subsets={'val_select':list(range(512)), 'hard_matched':compare_ids},
            score='sum(log(max(sigmoid(base_logit + residual_at_candidate_grid),1e-20)))',
            thresholds=list(THRESHOLDS), top_k=8, local_max=7, exclusion_m=30,
            union='all three slots, exact-coordinate dedup, one-to-one matching',
            no_training=True, no_test=True, no_parameter_search=True)
        write(out/'plan.json', plan)
        safe_print('阶段：输入预检通过，进行32条平衡K短测。')
        context, head, _ = load_model(runtime, SEEDS[0], 'b', training)
        pilot_ids = []
        for k in range(4):
            selected = [i for i,v in enumerate(select[1].counts.tolist()) if int(v) == k][:8]
            assert all(selected[i] % 4 == 0 for i in (0,4))
            pilot_ids.extend(selected)
        t = time.perf_counter()
        pilot = run_samples(runtime, select, 'val_select', pilot_ids, context, head, 'b')
        elapsed = time.perf_counter()-t
        estimate = elapsed/32*4096*1.25+120
        write(out/'pilot.json', dict(status='PASS', samples=len(pilot), seconds=elapsed,
                                     estimated_remaining_seconds=estimate))
        if time.time()+estimate > runtime.deadline:
            raise RuntimeError('Pilot predicts runtime above approved one-hour budget')
        del context, head, pilot; gc.collect(); torch.cuda.empty_cache()
        for seed in SEEDS:
            for arm in ('b','c'):
                key = f'{seed}_{arm}'; context, head, best = load_model(runtime, seed, arm, training)
                report['tracks'][key] = {'epoch':best['epoch']}
                for label, split, bundle, ids in [('val_select','val_select',select,list(range(512))),
                                                  ('hard_matched','val_compare',compare,compare_ids)]:
                    safe_print(f'阶段：{key} / {label}，固定512条候选与解码诊断。')
                    t = time.perf_counter()
                    rows = run_samples(runtime, bundle, split, ids, context, head, arm,
                                       historical[key] if split == 'val_compare' else None)
                    result = summarize_run(rows)
                    if label == 'val_select':
                        for name,value in result['actual']['final'].items():
                            if name in best['metrics']:
                                np.testing.assert_allclose(value,best['metrics'][name],rtol=0,atol=1e-12)
                    result.update(seconds=time.perf_counter()-t, replay_matches=True)
                    report['tracks'][key][label] = result
                    write(out/f'{key}_{label}_samples.json',rows)
                    write(out/'report.json',report)
                    runtime.postcheck(f'{key}_{label}')
                    del rows
                del context,head; gc.collect(); torch.cuda.empty_cache()
        runtime.audit_raw()
        for row in inputs+codes:
            verified_read(row,out/'anomalies')
        report.update(status='PASS', seconds=time.perf_counter()-start,
            peak_ram_percent=runtime.peak_ram, peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            hard_reference={name:value['overall'] for name,value in hard.items()},
            val_compare_scope='only frozen hard-matched 512 K=2/3; development diagnosis')
        write(out/'report.json',report)
        (out/'运行摘要.md').write_text(markdown(report),encoding='utf-8')
        write(out/'final_audit.json',dict(status='PASS',inputs_unchanged=True,
            files=[identity(p) for p in sorted(out.rglob('*')) if p.is_file()],
            training_executed=False,test_executed=False))
        safe_print(f'G6-P3完成：{out}')
    except BaseException:
        write(out/'failure.json',dict(status='FAILED',seconds=time.perf_counter()-start,traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
