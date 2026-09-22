"""R2统一联合解码评价，沿用R1指标和场景配对统计。"""
from __future__ import annotations

import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.gates.g5.e2e_g5_model import band_values, g4
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run
from 统一模型代码.gates.g5.r1.g5_r1_decode import candidate_ceiling
from 统一模型代码.gates.g5.r1.g5_r1_report import summarize as r1_summary, strata, paired_many
from 统一模型代码.gates.g5.r2.g5_r2_model import forward_r2, decode_r2
from 统一模型代码.gates.g5.r2.g5_r2_runtime import CONFIG, Progress, write


def summarize(rows):
    result = r1_summary(rows)
    for key in ('band_only_f1','band_only_iou'):
        values = [v for row in rows for v in row[key]]
        result[key] = float(np.mean(values)) if values else None
    return result


@torch.no_grad()
def evaluate(runtime, context, head, arm, bundle, indices=None, final=False, label='验证'):
    features, targets, metadata, index, _ = bundle
    g4.set_mode(context, training=False)
    if head is not None:
        head.eval()
    indices = list(range(len(targets.counts))) if indices is None else list(indices)
    ids_list = [torch.tensor(indices[i:i+4]) for i in range(0,len(indices),4)]
    iterator = batches(features, ids_list, prefetch=False)
    rows = []
    progress = Progress(label, len(indices), runtime.out/'progress.json')
    try:
        for ids, batch in iterator:
            runtime.guard()
            runtime.consumed(index, ids)
            outputs, candidates, scores = forward_r2(context, head, batch, ids, torch.device('cuda:0'), arm)
            logits = outputs[1].detach().cpu()
            for local, i in enumerate(ids.tolist()):
                count = int(targets.counts[i])
                truth = targets.positions[i,:count].numpy()
                bands, ignore = targets.band[i].numpy(), targets.ignore[i].numpy()
                logit = logits[local].numpy()
                decoded = decode_r2(logit, candidates[local], scores[local] if head is not None else None)
                row = R1Run.metric(None, truth, decoded['joint'], logit, decoded['active'], bands, ignore, metadata[i])
                mapping = {}
                if count:
                    cost = np.empty((3,count))
                    for q in range(3):
                        for t in range(count):
                            valid = targets.ignore[i,t] < .5
                            cost[q,t] = float(torch.nn.functional.binary_cross_entropy_with_logits(
                                logits[local,q,valid], targets.band[i,t,valid]))
                    a,b = linear_sum_assignment(cost)
                    mapping = dict(zip(a.tolist(),b.tolist()))
                row['band_only_f1'],row['band_only_iou'] = band_values(logits[local], targets,i,mapping)
                row.update(index=i, truth=truth.tolist(), band_logits=logit.tolist())
                if final:
                    row.update(decode=decoded, true_band=bands.tolist(), ignore=ignore.tolist(),
                               ceiling=candidate_ceiling(truth,decoded['candidates'],logit,bands,ignore))
                rows.append(row)
            progress.update(len(rows))
    finally:
        iterator.close()
        g4.set_mode(context, training=True)
        if head is not None:
            head.train()
    return summarize(rows), rows


def auxiliary_pairs(pairs, seed, repeats=2000):
    """Same scene resampling shared across seeds; ratios computed from pooled source counts."""
    reference = pairs[0][0]
    keys = [r['raw_index'] for r in reference]
    for a,b in pairs:
        if [r['raw_index'] for r in a] != keys or [r['raw_index'] for r in b] != keys:
            raise ValueError('Auxiliary pair mismatch')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero([r['true_count']==k for r in reference]) for k in range(4)]
    ids = np.concatenate([rng.choice(g,(repeats,len(g))) for g in groups if len(g)],axis=1)
    ids = np.concatenate([np.arange(len(reference))[None], ids])
    def values(rows):
        counts = np.asarray([r['true_count'] for r in rows])[ids].sum(1)
        result = {'count_accuracy':np.asarray([r['true_count']==r['predicted_count'] for r in rows])[ids].mean(1)}
        for key in ('band_only_f1','band_only_iou'):
            result[key] = np.asarray([sum(r[key]) for r in rows])[ids].sum(1)/np.maximum(counts,1)
        return result
    results = []
    for a,b in pairs:
        va,vb = values(a),values(b)
        results.append({k:vb[k]-va[k] for k in va})
    output = {}
    for k in results[0]:
        array = np.stack([r[k] for r in results])
        mean = array.mean(0)
        output[k] = {'mean':float(mean[0]),'ci95':np.quantile(mean[1:],[.025,.975]).tolist(),
                     'seed_deltas':array[:,0].tolist()}
    return output


def verdict(comparison):
    joint = comparison['joint_recall100_f1_08']
    gospa, recall = comparison['gospa_m'],comparison['recall100']
    benefit = all(x > 0 for x in joint['seed_deltas']) and joint['ci95'][0] > 0
    safe = gospa['ci95'][1] <= 1 and recall['ci95'][0] >= -.01
    if benefit and safe:
        return 'SUPPORTED_CANDIDATE'
    if benefit:
        return 'LOCATION_ASSOCIATION_TRADEOFF_OR_UNCERTAIN'
    if joint['ci95'][1] <= 0:
        return 'NO_SUPPORTED_IMPROVEMENT'
    return 'INCONCLUSIVE'


def build_report(results, out, training):
    seeds = sorted({s for s,a in results})
    report = {'status':'COMPLETE_FOR_REVIEW', 'test_executed':False,
              'scope':'Reused development scenes, conditional on two fixed training seeds',
              'tracks':{},'paired':{},'training':training}
    for (seed,arm), rows in results.items():
        report['tracks'][f'{seed}_{arm}'] = {'overall':summarize(rows),'strata':strata(rows),
            'candidate_coverage100':sum(r['ceiling']['truth_with_candidate100'] for r in rows),
            'candidate_location_ceiling100':sum(r['ceiling']['feasible_location_tp100'] for r in rows),
            'candidate_joint_ceiling100':sum(r['ceiling']['feasible_joint_tp100'] for r in rows)}
    for label,a,b in (('association_c1_minus_c0','c0','c1'),('feedback_c2_minus_c1','c1','c2')):
        pairs = [(results[s,a],results[s,b]) for s in seeds]
        combined = paired_many(pairs,seed=20260921,repeats=CONFIG['bootstrap_repeats'])
        combined.update(auxiliary_pairs(pairs,20260921,CONFIG['bootstrap_repeats']))
        report['paired'][label] = {'combined':combined,'decision':verdict(combined),
            'per_seed':{str(s):{**paired_many([p],seed=s),**auxiliary_pairs([p],s)}
                        for s,p in zip(seeds,pairs)}}
    write(out/'comparison_report.json',report)
    lines = ['# G5-R2运行摘要','','六轨训练及开发集比较完成；test未读取。','',
             '| 比较 | 联合Recall差/pp | GOSPA差/m | RMSE差/m | 判定 |','|---|---:|---:|---:|---|']
    for name,row in report['paired'].items():
        c = row['combined']
        lines.append(f"| {name} | {100*c['joint_recall100_f1_08']['mean']:.3f} | "
                     f"{c['gospa_m']['mean']:.3f} | {c['matched_rmse_m']['mean']:.3f} | {row['decision']} |")
    lines += ['', '置信区间、逐seed与分层指标见comparison_report.json；RMSE需结合覆盖率解读。',
              '辅助频带/计数下降仅作监控；触及24轮终点的轨道在训练报告中保留预算未解析标记。',
              '完整性结论以final_audit_report.json为准，工程完成不等同于科研成功。']
    (out/'运行摘要.md').write_text('\n'.join(lines),encoding='utf-8')
    return report


def save_rows(path, rows):
    with path.open('x',encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
