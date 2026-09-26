"""Manual entry orchestrator. All expensive stages run only after engineering/resource checks."""
from __future__ import annotations

import argparse
import gc
import shutil
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from 统一模型代码.gates.g7.formal_runtime import Runtime, RUN, BASE, CONFIG, read,write,load,identity,verified_json
from 统一模型代码.gates.g7.formal_train import train,evaluate,needs_extension
from 统一模型代码.gates.g7.compact_model import windows
from 统一模型代码.gates.g7.compact_run import RunLock
from 统一模型代码.gates.g7 import compact_run
from 统一模型代码.gates.g6.g6_p2_runtime import setup_environment
from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.r1.g5_r1_report import paired_many
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import auxiliary_pairs, summarize


def group_train(rt,phases,seeds):
    results={}
    for seed in seeds:
        for phase in phases:
            cfg=rt.config['phases'][phase]
            results[f'{seed}_{phase}']=train(rt,seed,phase,cfg['base_epochs'])
    if any(needs_extension(r,phase,rt.config) for seed in seeds for phase in phases
           for r in [results[f'{seed}_{phase}']]):
        for seed in seeds:
            for phase in phases:
                results[f'{seed}_{phase}']=train(rt,seed,phase,rt.config['phases'][phase]['max_epochs'])
    return results


def coverage(rt,seed):
    rows=[]
    index=rt.local_index(seed,'val_select')
    for i in rt.ids('candidate','val_select'):
        r=rt.data('candidate',seed,'val_select',[i])[0]
        stat=rt.get(index['samples'][str(i)])
        logits=stat['candidate_band_logits']; k=r['count']
        cost=np.zeros((3,k))
        for q in range(3):
            for t in range(k):
                mask=r['ignore'][t]<.5
                cost[q,t]=float(torch.nn.functional.binary_cross_entropy_with_logits(logits[q,mask],r['band'][t,mask]))
        a,b=linear_sum_assignment(cost)
        row=dict(index=i,count=k)
        for top in (5,8):
            centers,valid,points,_=windows(stat['proposals'],top)
            hits=[bool(((centers[q]-r['positions'][t]).abs().amax(-1)<=200).logical_and(
                       valid[q].flatten(1).any(-1)).any()) for q,t in zip(a,b)]
            row[str(top)]=dict(covered=sum(hits),all_covered=all(hits),unique_points=len(points))
        rows.append(row)
    report=dict(seed=seed,scenes=len(rows),sources=sum(r['count'] for r in rows),samples=rows,
        covered={str(t):sum(r[str(t)]['covered'] for r in rows) for t in (5,8)},
        scope='Frequency-assigned per-slot window coverage, not localization recall',
        top5_configuration_changed=False)
    write(rt.out/f'evaluation/{seed}_top5_report.json',report)
    return report


def final_comparison(rt,training):
    from 统一模型代码.gates.g7.formal_hard import evaluate_hard,summarize_hard
    tracks,rows={},{}
    for seed in rt.config['seeds']:
        rt.prepare_local(seed,'val_compare')
        for phase in ('candidate','f','s','e'):
            key=f'{seed}_{phase}'
            path=rt.out/f'evaluation/{key}_val_compare.json'
            pointer=path.with_suffix('.identity.json')
            if pointer.exists():
                value=verified_json(read(pointer),rt.out)
                if value['checkpoint']!=rt.best(seed,phase):
                    raise RuntimeError('Frozen evaluation checkpoint changed')
            else:
                model=rt.model(seed,phase)
                model.restore(load(rt.best(seed,phase),rt.out)['state'])
                metrics,samples=evaluate(rt,model,seed,phase,'val_compare')
                value=dict(checkpoint=rt.best(seed,phase),metrics=metrics,samples=samples)
                write(path,value);write(pointer,identity(path))
                del model;gc.collect();torch.cuda.empty_cache()
            tracks[key],rows[seed,phase]=value['metrics'],value['samples']
        path=rt.out/f'evaluation/{seed}_hard_val_compare.json'
        pointer=path.with_suffix('.identity.json')
        if pointer.exists():
            value=verified_json(read(pointer),rt.out)
            if value['checkpoints']!={k:rt.best(seed,k) for k in ('ch3','d8')}:
                raise RuntimeError('Hard-cascade frozen checkpoints changed')
        else:
            summary,samples=evaluate_hard(rt,seed,'val_compare')
            value=dict(metrics=summary,samples=samples,
                       checkpoints={k:rt.best(seed,k) for k in ('ch3','d8')})
            write(path,value);write(pointer,identity(path))
        tracks[f'{seed}_hard'],rows[seed,'hard']=value['metrics'],value['samples']
    from 统一模型代码.gates.g7.formal_latency import profile_online
    timing_ids=sorted(i for k in range(4) for i in [j for j,r in enumerate(
        rt.manifest['subsets']['val_compare']) if r['true_k']==k][:8])
    if len(timing_ids)!=32:
        raise RuntimeError('Online timing requires eight scenes per K')
    online={}
    for seed in rt.config['seeds']:
        for phase in ('candidate','f','s','e'):
            path=rt.out/f'evaluation/{seed}_{phase}_online.json'
            pointer=path.with_suffix('.identity.json')
            if pointer.exists():
                timed=verified_json(read(pointer),rt.out)
                if timed['checkpoint']!=rt.best(seed,phase) or timed['indices']!=timing_ids:
                    raise RuntimeError('Online timing inputs changed')
            else:
                timed=profile_online(rt,seed,phase,timing_ids)
                timed['checkpoint']=rt.best(seed,phase)
                write(path,timed);write(pointer,identity(path))
            online[f'{seed}_{phase}']=timed
        hard_times=[r['inference_seconds'] for r in rows[seed,'hard'] if r['index'] in timing_ids]
        online[f'{seed}_hard']=dict(indices=timing_ids,samples=len(hard_times),
            mean_seconds=float(np.mean(hard_times)),p95_seconds=float(np.quantile(hard_times,.95)),
            scope=tracks[f'{seed}_hard']['inference_timing_scope'],
            note='Same scenes, times from full hard evaluation; not a separate repeated benchmark')
    paired={}
    for label,a,b in [('selected_input_s_minus_f','f','s'),('feedback_e_minus_s','s','e')]:
        pairs=[(rows[s,a],rows[s,b]) for s in rt.config['seeds']]
        paired[label]=paired_many(pairs,repeats=rt.config['bootstrap_repeats'])
        paired[label].update(auxiliary_pairs(pairs,20260920,rt.config['bootstrap_repeats']))
    # The native hard cascade predicts an unbound spatial set: joint identity metrics are undefined.
    hard_spatial={str(s):{metric:(tracks[f'{s}_e'][metric]-tracks[f'{s}_hard'][metric]
                         if tracks[f'{s}_e'][metric] is not None and tracks[f'{s}_hard'][metric] is not None else None)
                      for metric in ('gospa_m','matched_rmse_m','recall100','count_accuracy')}
                  for s in rt.config['seeds']}
    by_k={f'{seed}_{phase}':{f'K{k}':(summarize_hard if phase=='hard' else summarize)(
        [r for r in samples if r['true_count']==k]) for k in range(4)}
        for (seed,phase),samples in rows.items()}
    report=dict(status='COMPLETED_FORMAL_DEVELOPMENT',tracks=tracks,paired=paired,online_timing=online,
        hard_spatial_e_minus_hard=hard_spatial,by_k=by_k,training=training,test_executed=False,
        scope='Development comparison only; no test or publication-success claim',
        hard_binding_note='Native union D8 has no slot-to-position binding; do not fabricate joint metrics')
    write(rt.out/'evaluation/comparison_report.json',report)
    lines=['# G7正式小区域训练与开发集比较完成','','CH3/D8从随机初始化训练；未读取test。','',
           '| 比较 | 联合Recall差/pp | GOSPA差/m | RMSE差/m |','|---|---:|---:|---:|']
    for label,r in paired.items():
        def number(value,scale=1):
            return 'NA' if value is None else f'{scale*value:.3f}'
        lines.append(f'| {label} | {number(r["joint_recall100_f1_08"]["mean"],100)} | '
                     f'{number(r["gospa_m"]["mean"])} | {number(r["matched_rmse_m"]["mean"])} |')
    lines+=['','回读comparison_report.json及final_audit_report.json。硬级联比较空间与独立频带指标；'
            '其原生输出没有逐源频带—位置绑定，不伪造联合指标。']
    (rt.out/'evaluation/运行摘要.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return report


def run(rt):
    training={}
    print('阶段1/4：仅用正式train随机初始化训练CH3、D8',flush=True)
    for phase in ('ch3','d8'):
        training.update(group_train(rt,[phase],rt.config['seeds']))
    print('阶段2/4：训练全新共同候选生成器',flush=True)
    training.update(group_train(rt,['candidate'],rt.config['seeds']))
    write(rt.out/'training_report.json',dict(status='FOUNDATIONS_AND_CANDIDATE_COMPLETE',training=training))
    print('阶段3/4：固定新候选，生成局部物理统计，运行F/S/E',flush=True)
    for seed in rt.config['seeds']:
        for split in ('train','val_select'):
            rt.prepare_local(seed,split)
        coverage(rt,seed)
        training.update(group_train(rt,['f','s','e'],[seed]))
        write(rt.out/'training_report.json',dict(status='LOCAL_TRAINING',training=training))
    print('阶段4/4：冻结最佳模型，统一val_compare评价；test保持封存',flush=True)
    final_comparison(rt,training)
    rt.inputs.audit_raw()
    for row in rt.registration['files']:
        verified_read(row,rt.out/'anomalies')
    for name in ('data','oracle'):
        registry=rt.dataset if name=='data' else rt.oracle
        for group in registry.values():
            for row in group.values():
                rt.guard();verified_read(row,rt.out/'anomalies')
    for path in sorted((rt.out/'local').glob('*/*/index.identity.json')):
        index=verified_json(read(path),rt.out)
        for row in index['samples'].values():
            rt.guard();verified_read(row,rt.out/'anomalies')
    write(rt.out/'evaluation/final_audit_report.json',dict(status='PASS',test_executed=False,
        scope='scratch_training_and_development_comparison',contract=identity(rt.out/'contract.json'),
        peak_ram_percent=rt.peak_ram_percent,gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        outputs=[identity(p) for p in sorted((rt.out/'evaluation').iterdir()) if p.is_file()
                 and p.name!='final_audit_report.json']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    setup_environment();compact_run.RUN=RUN
    with RunLock():
        rt=Runtime()
        try:
            rt.contract()
            prep=verified_json(read(BASE/'formal_preparation_report.json')['report'],rt.out)
            if prep['status']!='ENGINEERING_PASS' or verified_json(prep['contract'],rt.out)!=rt.registration:
                raise RuntimeError('正式版短验证/源码合同不匹配')
            final=rt.out/'evaluation/final_audit_report.json'
            if final.exists():
                if read(final)['status']!='PASS':
                    raise RuntimeError('Final audit is not PASS')
                for row in read(final)['outputs']:
                    verified_read(row,rt.out/'anomalies')
                print(f'正式开发集阶段已完成，不重复运行：{final.parent}',flush=True)
                return
            remaining=CONFIG['wall_seconds']-(rt.budget['active_seconds'] if rt.budget_path.exists()
                      else prep['charged_seconds_before_training'])
            if not (rt.out/'training').exists() and prep['forecast_remaining_seconds']>remaining:
                raise RuntimeError(f'正式全流程预测{prep["forecast_remaining_seconds"]/3600:.1f}h，'
                     f'超过剩余预算{remaining/3600:.1f}h；未启动，不自动缩减配置。')
            if not (rt.out/'training').exists() and shutil.disk_usage(rt.out).free/2**30 < prep['required_free_gib']:
                raise RuntimeError('局部统计与checkpoint预测空间不足；未启动')
            if args.check:
                print(f'正式入口检查通过；累计上限={CONFIG["wall_seconds"]/3600:g}h，'
                      f'剩余={remaining/3600:.2f}h，预测={prep["forecast_remaining_seconds"]/3600:.2f}h；'
                      '时间及磁盘检查通过，未训练。',flush=True)
                return
            rt.begin(prep['charged_seconds_before_training'])
            try:
                write(rt.out/'execution_status.json',dict(status='RUNNING',started_at=time.time()))
                # Recheck registered scene fingerprint report, no hidden test/data changes.
                verified_read(prep['scene_audit'],rt.out/'anomalies')
                rt.bind_scene_audit(prep['scene_audit'])
                rt.inputs.audit_raw();run(rt)
                write(rt.out/'execution_status.json',dict(status='COMPLETED',finished_at=time.time()))
            except BaseException as exc:
                write(rt.out/'execution_status.json',dict(status='STOPPED',error=repr(exc),time=time.time()))
                raise
            finally:
                rt.finish()
        finally:
            rt.close()


if __name__=='__main__':
    main()
