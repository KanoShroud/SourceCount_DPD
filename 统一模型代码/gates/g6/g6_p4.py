"""Approved G6-P4: frozen shared-candidate and factorized-score comparison."""
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
from 统一模型代码.gates.g5.r2.g5_r2_model import candidate_batch
from 统一模型代码.gates.g6.g6_p1_speed import physical_batches
from 统一模型代码.gates.g6.g6_p2_diagnose import load_model, representations
from 统一模型代码.gates.g6.g6_p2_runtime import (
    ROOT, Runtime as P2Runtime, SEEDS, identity, read, safe_print, setup_environment, write)
from 统一模型代码.gates.g6.g6_p3 import read_inputs
from 统一模型代码.gates.g6.g6_p3_candidates import THRESHOLDS
from 统一模型代码.gates.g6.g6_p4_decode import MODES, decode_shared, prepare, upper_bounds

BASE = ROOT/'outputs_e2e/unified/e2e_g6_p4'
P3 = ROOT/'outputs_e2e/unified/e2e_g6_p3/20260924_182847'


class Runtime(P2Runtime):
    def __init__(self, out, deadline):
        self.out = Path(out).resolve(strict=True)
        if not self.out.is_relative_to(BASE.resolve()) or self.out.is_relative_to(Path('F:/SourceCount_DPD/outputs').resolve()):
            raise ValueError('P4 output isolation violation')
        self.deadline = deadline
        self.inputs, self.ranges, self.physical_consumed = {}, {}, {}
        self.peak_ram = 0
        self.manifest = self.fm = self.physics = None
        self._cache_indexes = {}


def p3_inputs(out):
    audit_path = (P3/'final_audit.json').resolve(strict=True)
    audit = read(audit_path)
    if audit['status'] != 'PASS' or audit['test_executed'] or audit['training_executed']:
        raise RuntimeError('P3 audit mismatch')
    by_name = {Path(r['path']).name:r for r in audit['files']}
    inputs, previous = [identity(audit_path)], {}
    for name in ['plan.json', 'report.json'] + [f'{s}_{a}_{d}_samples.json'
            for s in SEEDS for a in ('b','c') for d in ('val_select','hard_matched')]:
        row = by_name[name]
        path = Path(row['path']).resolve(strict=True)
        if path.parent != P3.resolve():
            raise RuntimeError('P3 evidence path mismatch')
        previous[name] = json.loads(verified_read(row,out/'anomalies'))
        inputs.append(row)
    if previous['report.json']['status'] != 'PASS':
        raise RuntimeError('P3 report incomplete')
    return previous, inputs


@torch.no_grad()
def run_samples(runtime, bundle, split, indices, context, head, arm, old_rows):
    features, targets, metadata, index, _ = bundle
    old_by_id = {r['raw_index']:r for r in old_rows}
    rows = []
    iterator = physical_batches(runtime, features, [torch.tensor(indices[i:i+4])
        for i in range(0,len(indices),4)], split, True)
    try:
        for ids,batch,stats in iterator:
            runtime.guard(); runtime.consumed(index,ids)
            base,_,residual = representations(context,head,batch,ids,arm,runtime.physics,stats)
            heat = base[3]+residual['full']
            records = candidate_batch(heat,base[4])
            probabilities = heat.sigmoid().cpu().numpy()
            correction = residual['full'].cpu().numpy()
            logits = base[1].cpu().numpy()
            for j,i in enumerate(ids.tolist()):
                raw_id = metadata[i]['raw_index']; old = old_by_id[raw_id]
                k = int(targets.counts[i]); truth = targets.positions[i,:k].numpy()
                bands,ignore = targets.band[i].numpy(),targets.ignore[i].numpy()
                z,record = logits[j],records[j]
                assert old['true_count'] == k
                np.testing.assert_array_equal(truth,np.asarray(old['truth']).reshape(-1,2))
                np.testing.assert_array_equal(z,old['logits'])
                assert record == old['candidates']['final'], 'P3 candidate replay mismatch'
                prepared = prepare(z,record,probabilities[j],correction[j])
                decoded = {m:decode_shared(z,record,prepared,m) for m in MODES}
                metrics = {m:R1Run.metric(None,truth,d['joint'],z,d['active'],bands,ignore,metadata[i])
                           for m,d in decoded.items()}
                for m,d in decoded.items():
                    metrics[m]['fallback'] = d['fallback']
                    assert d['active'] == decoded['native']['active']
                    if d['selected_pool'] is not None:
                        for p,point in zip(d['selected_pool'],d['joint']):
                            np.testing.assert_array_equal(point,prepared['pool'][p]['position'])
                np.testing.assert_array_equal(decoded['native']['joint'],old['decoded']['final']['joint'])
                for field in ('gospa_m','joint_tp','predicted_count','matched_errors_m',
                              'tp_at_10m','tp_at_30m','tp_at_50m','tp_at_100m'):
                    np.testing.assert_array_equal(metrics['native'][field],old['metrics']['final'][field])
                upper = upper_bounds(truth,z,bands,ignore,prepared,old['ceilings']['final'])
                for m in MODES:
                    if not decoded[m]['fallback']:
                        level = 'native_feasible' if m == 'native' else 'shared_separated'
                        joint = 'native_joint100' if m == 'native' else 'shared_joint100'
                        assert metrics[m]['joint_tp'] <= upper[joint]
                        for d in THRESHOLDS:
                            assert metrics[m][f'tp_at_{d}m'] <= upper[level][str(d)]
                if len(prepared['active']) >= 2 and not decoded['native']['fallback']:
                    native_value = sum(np.log(max(c['scores'][r-1],1e-20)) for c,r in
                        zip(decoded['native']['candidates'],decoded['native']['selected_ranks']))
                    assert decoded['shared_final']['score'] >= native_value-1e-12
                rows.append(dict(index=i,raw_index=raw_id,true_count=k,truth=truth.tolist(),
                    logits=z.tolist(),band=bands.tolist(),ignore=ignore.tolist(),pool=prepared['pool'],
                    final_probabilities=prepared['final_probabilities'].tolist(),
                    physical_residual=prepared['residual'].tolist(),
                    scores={m:v.tolist() for m,v in prepared['scores'].items()},
                    decoded=decoded,metrics=metrics,ceilings=upper))
    finally:
        iterator.close()
    return rows


def summarize_run(rows):
    total = max(sum(r['true_count'] for r in rows),1)
    levels = ('union','pred_k_union','shared_separated','native_feasible')
    upper = {level:{str(d):sum(r['ceilings'][level][str(d)] for r in rows)/total
                    for d in THRESHOLDS} for level in levels}
    upper.update({level:sum(r['ceilings'][level] for r in rows)/total
                  for level in ('shared_joint100','native_joint100')})
    actual = {m:summarize([r['metrics'][m] for r in rows]) for m in MODES}
    pairs = {}
    for after,before in [('shared_final','native'),('shared_factorized','shared_final'),('shared_factorized','native')]:
        name = f'{after}_minus_{before}'
        pairs[name] = {field:actual[after][field]-actual[before][field] for field in
            ('gospa_m','matched_rmse_m','joint_recall100_f1_08','recall10','recall30','recall50','recall100')}
        pairs[name].update(
            gospa_better_scenes=sum(r['metrics'][after]['gospa_m'] < r['metrics'][before]['gospa_m']-1e-9 for r in rows),
            gospa_worse_scenes=sum(r['metrics'][after]['gospa_m'] > r['metrics'][before]['gospa_m']+1e-9 for r in rows),
            joint_net_true_sources=sum(r['metrics'][after]['joint_tp']-r['metrics'][before]['joint_tp'] for r in rows))
    return dict(actual=actual,ceilings=upper,paired=pairs,
        cross_slot={m:sum(r['decoded'][m]['cross_slot_count'] for r in rows) for m in MODES},
        cross_slot_scenes={m:sum(r['decoded'][m]['cross_slot_count']>0 for r in rows) for m in MODES},
        fallback={m:sum(r['decoded'][m]['fallback'] for r in rows) for m in MODES},
        shared_no_feasible_scenes=sum(r['ceilings']['shared_feasible_combinations']==0 for r in rows))


def markdown(report):
    lines = ['# G6-P4共享候选与评分对照', '', '## Material Passport', '',
        '- Origin Skill: experiment-agent', '- Origin Mode: run',
        '- Origin Date: '+time.strftime('%Y-%m-%d'),
        '- Verification Status: VERIFIED (native replay and input/output integrity)',
        '- Version Label: g6_p4_v1', '',
        '无训练、无test。native=原解码；shared_final=共享候选原评分；shared_factorized=共享候选因子化评分。',
        '下表为两seed算术平均；RMSE先按各seed计算，不将上限当成实际收益。', '']
    fields = ('joint_recall100_f1_08','gospa_m','matched_rmse_m','matched_coverage','recall10','recall30','recall50','recall100')
    for split in ('val_select','hard_matched'):
        lines += [f'## {split}', '', '|模型/方式|联合Recall%|GOSPA/m|RMSE/m|coverage%|R10%|R30%|R50%|R100%|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for arm in ('b','c'):
            values = [report['tracks'][f'{s}_{arm}'][split] for s in SEEDS]
            for mode in MODES:
                numbers = [np.mean([v['actual'][mode][f] for v in values])*(1 if f in ('gospa_m','matched_rmse_m') else 100) for f in fields]
                lines.append(f'|{arm}/{mode}|'+'|'.join(f'{n:.4f}' for n in numbers)+'|')
        lines += ['', '|模型/上限|10m%|30m%|50m%|100m%|','|---|---:|---:|---:|---:|']
        for arm in ('b','c'):
            values = [report['tracks'][f'{s}_{arm}'][split]['ceilings'] for s in SEEDS]
            for level in ('union','pred_k_union','shared_separated','native_feasible'):
                numbers = [np.mean([v[level][str(d)] for v in values])*100 for d in THRESHOLDS]
                lines.append(f'|{arm}/{level}|'+'|'.join(f'{n:.4f}' for n in numbers)+'|')
            lines += ['', f"{arm} 联合上限 native/shared："+'/'.join(f"{np.mean([v[f] for v in values])*100:.4f}%" for f in ('native_joint100','shared_joint100')), '']
    lines += ['## 执行', '', f"状态：{report['status']}；耗时：{report['seconds']:.2f}s；RAM峰值：{report['peak_ram_percent']:.1f}%；GPU峰值：{report['peak_gpu_gib']:.3f}GiB。",
              '逐seed结果、配对变化、尾部错误与外槽候选使用次数见report.json；完整候选和评分见各samples.json。']
    return '\n'.join(lines)+'\n'


def main():
    setup_environment(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    out = BASE/time.strftime('%Y%m%d_%H%M%S'); out.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter(); runtime = Runtime(out,time.time()+3600)
    report = dict(status='RUNNING',training_executed=False,test_executed=False,tracks={})
    try:
        previous,p3_rows = p3_inputs(out)
        training,_,hard,hard_ids,inputs = read_inputs(out); inputs += p3_rows
        code_paths = {Path(__file__).resolve(), Path(__file__).with_name('g6_p4_decode.py').resolve(),
                      Path(__file__).with_name('test_g6_p4.py').resolve()}
        for module in tuple(sys.modules.values()):
            value = getattr(module,'__file__',None)
            if isinstance(value,str) and Path(value).is_absolute():
                path = Path(value).resolve()
                if path.is_relative_to(ROOT/'统一模型代码') and path.suffix == '.py':
                    code_paths.add(path)
        codes = [identity(p) for p in sorted(code_paths)]
        runtime.preflight(); runtime.audit_raw()
        select = runtime.features('val_select'); compare = runtime.features('val_compare',select[-1])
        compare_ids = [i for i,r in enumerate(compare[2]) if r['raw_index'] in hard_ids]
        assert len(select[1].counts) == len(compare_ids) == 512
        assert previous['plan.json']['subsets'] == {'val_select':list(range(512)), 'hard_matched':compare_ids}
        plan = dict(gate='G6-P4',seeds=list(SEEDS),arms=['b','c'],modes=list(MODES),
            inputs=inputs,code=codes,torch=torch.__version__,numpy=np.__version__,budget_seconds=3600,
            subsets=previous['plan.json']['subsets'],top_k=8,local_max=7,exclusion_m=30,
            pool='all 3 slots; stable donor/rank order; donor grid and offset preserved; no near merge',
            original_score='sum(log(max(final_sigmoid_recipient_at_donor_grid,1e-20)))',
            factorized_score='log(max_all3(final_sigmoid))+residual_active-logsumexp_active(residual)',
            k01='unchanged native',fallback='native',no_training=True,no_test=True,no_parameter_search=True)
        write(out/'plan.json',plan)
        safe_print('阶段：P4输入校验完成，32条平衡K短测。')
        context,head,_ = load_model(runtime,SEEDS[0],'b',training)
        pilot_ids = []
        for k in range(4):
            selected = [i for i,v in enumerate(select[1].counts.tolist()) if int(v)==k][:8]
            assert all(selected[i]%4 == 0 for i in (0,4))
            pilot_ids.extend(selected)
        t = time.perf_counter()
        pilot = run_samples(runtime,select,'val_select',pilot_ids,context,head,'b',previous[f'{SEEDS[0]}_b_val_select_samples.json'])
        elapsed = time.perf_counter()-t
        estimate = elapsed/32*4096*1.5+120
        write(out/'pilot.json',dict(status='PASS',samples=len(pilot),seconds=elapsed,estimated_remaining_seconds=estimate))
        if time.time()+estimate > runtime.deadline:
            raise RuntimeError('Pilot exceeds approved one-hour budget')
        del context,head,pilot; gc.collect(); torch.cuda.empty_cache()
        for seed in SEEDS:
            for arm in ('b','c'):
                key = f'{seed}_{arm}'; context,head,best = load_model(runtime,seed,arm,training)
                report['tracks'][key] = {'epoch':best['epoch']}
                for label,split,bundle,ids in [('val_select','val_select',select,list(range(512))),
                                              ('hard_matched','val_compare',compare,compare_ids)]:
                    safe_print(f'阶段：{key} / {label}，512条三组对照。')
                    t = time.perf_counter()
                    rows = run_samples(runtime,bundle,split,ids,context,head,arm,previous[f'{key}_{label}_samples.json'])
                    result = summarize_run(rows)
                    for name,value in result['actual']['native'].items():
                        np.testing.assert_equal(value,previous['report.json']['tracks'][key][label]['actual']['final'][name])
                    result.update(seconds=time.perf_counter()-t,replay_matches=True)
                    report['tracks'][key][label] = result
                    write(out/f'{key}_{label}_samples.json',rows); write(out/'report.json',report)
                    runtime.postcheck(f'{key}_{label}')
                    del rows
                del context,head; gc.collect(); torch.cuda.empty_cache()
        runtime.audit_raw()
        for row in inputs+codes:
            verified_read(row,out/'anomalies')
        report.update(status='PASS',seconds=time.perf_counter()-start,
            peak_ram_percent=runtime.peak_ram,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            hard_reference={name:value['overall'] for name,value in hard.items()},
            scope='two development sets; P3 replay; no selection or retraining of checkpoints')
        write(out/'report.json',report)
        (out/'运行摘要.md').write_text(markdown(report),encoding='utf-8')
        outputs = [identity(p) for p in sorted(out.rglob('*')) if p.is_file()]
        for row in outputs:
            verified_read(row,out/'anomalies')
        write(out/'final_audit.json',dict(status='PASS',inputs_unchanged=True,outputs_verified=True,
            files=outputs,training_executed=False,test_executed=False))
        safe_print(f'G6-P4完成：{out}')
    except BaseException:
        write(out/'failure.json',dict(status='FAILED',seconds=time.perf_counter()-start,traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
