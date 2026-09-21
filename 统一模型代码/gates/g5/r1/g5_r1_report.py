"""G5-R1开发集配对统计及可追溯案例图；不挑参数、不选checkpoint。"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def summarize(rows):
    errors = [e for r in rows for e in r['matched_errors_m']]
    true = sum(r['true_count'] for r in rows)
    pred = sum(r['predicted_count'] for r in rows)
    f1 = [x for r in rows for x in r['spatial_band_f1']]
    result = {'scenes': len(rows), 'true_sources': true, 'predicted_sources': pred,
              'gospa_m': float(np.mean([r['gospa_m'] for r in rows])) if rows else None,
              'matched_rmse_m': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
              'matched_coverage': len(errors)/true if true else None,
              'matched_median_p90_p95_m': np.quantile(errors, [.5, .9, .95]).tolist() if errors else None,
              'joint_recall100_f1_08': sum(r['joint_tp'] for r in rows)/true if true else None,
              'joint_precision100_f1_08': sum(r['joint_tp'] for r in rows)/pred if pred else None,
              'spatial_matched_band_f1': float(np.mean(f1)) if f1 else None,
              'spatial_band_f1_per_true_source': sum(f1)/true if true else None,
              'count_accuracy': float(np.mean([r['true_count']==r['predicted_count'] for r in rows])) if rows else None,
              'duplicate30_scenes': sum(r['duplicate30'] for r in rows),
              'tail500_scenes': sum(max(r['matched_errors_m'], default=0)>500 for r in rows)}
    for t in (10, 30, 50, 100):
        tp = sum(r[f'tp_at_{t}m'] for r in rows)
        result[f'recall{t}'] = tp/true if true else None
        result[f'precision{t}'] = tp/pred if pred else None
    for component in ('localization', 'missed', 'false'):
        result[f'gospa_{component}_mean_p_sum'] = float(np.mean([r[f'gospa_{component}_p_sum'] for r in rows])) if rows else None
    return result


def strata(rows):
    groups = {'overall': rows}
    groups.update({f'K{k}': [r for r in rows if r['true_count']==k] for k in range(4)})
    for name, lo, hi in [('near_lt30', 0, 30), ('near_30_100', 30, 100), ('near_ge100', 100, float('inf'))]:
        groups[name] = [r for r in rows if r['min_source_distance_m'] is not None and lo <= r['min_source_distance_m'] < hi]
    groups['snr_lt_minus10'] = [r for r in rows if r['snr_db'] < -10]
    groups['snr_minus10_5'] = [r for r in rows if -10 <= r['snr_db'] < 5]
    groups['snr_ge5'] = [r for r in rows if r['snr_db'] >= 5]
    return {name: summarize(values) for name, values in groups.items()}


def paired_many(pairs, seed=20260920, repeats=2000):
    """同一场景重采样索引应用于所有seed；汇总量为各seed指标的算术平均。"""
    reference = pairs[0][0]
    keys = [(r['raw_index'], r['true_count']) for r in reference]
    for a, b in pairs:
        if keys != [(r['raw_index'], r['true_count']) for r in a] or keys != [(r['raw_index'], r['true_count']) for r in b]:
            raise ValueError('Paired sample identity mismatch')
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero([r['true_count']==k for r in reference]) for k in range(4)]
    indices = np.concatenate([rng.choice(g, (repeats, len(g)), replace=True) for g in groups if len(g)], axis=1)
    indices = np.concatenate([np.arange(len(reference))[None], indices])

    def values(rows):
        def take(name):
            return np.asarray([r[name] for r in rows], dtype=float)[indices]
        true = take('true_count').sum(1)
        pred = take('predicted_count').sum(1)
        counts = np.asarray([len(r['matched_errors_m']) for r in rows])[indices].sum(1)
        squares = np.asarray([sum(e*e for e in r['matched_errors_m']) for r in rows])[indices].sum(1)
        return {'gospa_m': take('gospa_m').mean(1),
                'recall100': take('tp_at_100m').sum(1)/np.maximum(true, 1),
                'joint_recall100_f1_08': take('joint_tp').sum(1)/np.maximum(true, 1),
                'joint_precision100_f1_08': take('joint_tp').sum(1)/np.maximum(pred, 1),
                'matched_rmse_m': np.sqrt(np.divide(squares, counts, out=np.full_like(squares, np.nan), where=counts>0))}
    deltas = []
    for a, b in pairs:
        va, vb = values(a), values(b)
        deltas.append({k: vb[k]-va[k] for k in va})
    result = {'interval_scope': 'scene bootstrap conditional on fixed training seeds; candidate minus baseline'}
    for key in deltas[0]:
        array = np.stack([d[key] for d in deltas])
        mean = array.mean(0)
        result[key] = {'mean': float(mean[0]) if np.isfinite(mean[0]) else None,
                       'ci95': np.quantile(mean[1:], [.025, .975]).tolist() if np.isfinite(mean).all() else None,
                       'seed_deltas': [float(v) if np.isfinite(v) else None for v in array[:, 0]]}
    return result


def build_report(results, out):
    seeds = sorted({key[0] for key in results})
    report = {'status': 'COMPLETE_FOR_REVIEW', 'test_executed': False, 'training_executed': False,
              'tracks': {}, 'paired': {}, 'limits': 'Previously inspected development data; no new test evidence; G5 training budget remains unresolved.'}
    for (seed, track), records in results.items():
        report['tracks'][f'{seed}_{track}'] = {mode: strata([r[mode] for r in records]) for mode in ('original', 'joint')}
        report['tracks'][f'{seed}_{track}']['transitions'] = {
            'duplicate_repaired': sum(r['original']['duplicate30'] and not r['joint']['duplicate30'] for r in records),
            'location_improved_binding_worse': sum(r['joint']['tp_at_100m']>r['original']['tp_at_100m'] and r['joint']['joint_tp']<r['original']['joint_tp'] for r in records),
            'recall_improved': sum(r['joint']['tp_at_100m']>r['original']['tp_at_100m'] for r in records),
            'recall_worsened': sum(r['joint']['tp_at_100m']<r['original']['tp_at_100m'] for r in records),
            'fallback': sum(r['decode']['fallback'] for r in records),
            'candidate_truth_coverage100': sum(r['ceiling']['truth_with_candidate100'] for r in records),
            'candidate_feasible_location_tp100': sum(r['ceiling']['feasible_location_tp100'] for r in records),
            'candidate_feasible_joint_tp100': sum(r['ceiling']['feasible_joint_tp100'] for r in records)}
    comparisons = [('sg_decode', ('sg','original'), ('sg','joint')),
                   ('e2e_decode', ('e2e','original'), ('e2e','joint')),
                   ('feedback_original', ('sg','original'), ('e2e','original')),
                   ('feedback_joint', ('sg','joint'), ('e2e','joint'))]
    for name, (a, am), (b, bm) in comparisons:
        pairs = [([r[am] for r in results[s,a]], [r[bm] for r in results[s,b]]) for s in seeds]
        report['paired'][name] = {'combined': paired_many(pairs),
                                  'per_seed': {str(s): paired_many([pair], seed=s) for s,pair in zip(seeds,pairs)}}
    write(out/'comparison_report.json', report)
    lines = ['# G5-R1运行摘要', '', '完整开发集评价已结束，待研究判读；未训练、未访问test。', '',
             '| 比较（后者减前者） | GOSPA差值/m及95%区间 | 联合Recall差值及95%区间 |', '|---|---|---|']
    for name, data in report['paired'].items():
        a, b = data['combined']['gospa_m'], data['combined']['joint_recall100_f1_08']
        lines.append(f"| {name} | {a['mean']:.4f} {a['ci95']} | {b['mean']:.4f} {b['ci95']} |")
    lines += ['', 'GOSPA下降为改善，联合Recall上升为改善。详细seed、近源、RMSE及失败分类见comparison_report.json和每轨samples.jsonl。',
              '完整性是否通过以final_audit_report.json为准；COMPLETE不等于科研PASS。']
    (out/'运行摘要.md').write_text('\n'.join(lines), encoding='utf-8')
    return report


def plot_cases(results, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pool = [(s, r) for (s,t), rows in results.items() if t=='e2e' for r in rows]
    rules = {
        'repair': lambda r: r['joint']['tp_at_100m']>r['original']['tp_at_100m'],
        'binding_failure': lambda r: r['joint']['tp_at_100m']>r['joint']['joint_tp'],
        'near_harm': lambda r: r['original']['min_source_distance_m'] is not None and r['original']['min_source_distance_m']<100 and r['joint']['gospa_m']>r['original']['gospa_m'],
        'missing_candidate': lambda r: r['ceiling']['truth_with_candidate100']<r['original']['true_count']}
    manifest = []
    (out/'figures').mkdir()
    for name, predicate in rules.items():
        chosen = next(((s,r) for s,r in pool if predicate(r)), None)
        if chosen is None:
            manifest.append({'category': name, 'status': 'NO_CASE'})
            continue
        seed, row = chosen
        fig, axes = plt.subplots(1, 2, figsize=(10,5), constrained_layout=True)
        truth = np.asarray(row['truth'])
        for ax, mode in zip(axes, ('original','joint')):
            ax.scatter(truth[:,0],truth[:,1], marker='*', s=140, color='black', label='Truth')
            for j, point in enumerate(row['decode'][mode]):
                ax.scatter(*point, marker='x', s=65, color=f'C{j}', label=f"Q{row['decode']['active'][j]+1}")
            ax.set(xlabel='x (m)', ylabel='y (m)', title=f"{mode}: TP100={row[mode]['tp_at_100m']}, joint={row[mode]['joint_tp']}")
            ax.set_aspect('equal', adjustable='datalim')
            ax.legend(fontsize=8)
            ax.grid(alpha=.2)
        fig.suptitle(f"{name}: seed={seed}, raw={row['original']['raw_index']}")
        path = out/'figures'/f'{name}.png'
        fig.savefig(path, dpi=300)
        plt.close(fig)
        manifest.append({'category': name, 'path': str(path), 'seed': seed,
                         'raw_index': row['original']['raw_index'], 'selection': 'first qualifying seed/index in fixed order',
                         'source': str(out/f'{seed}_e2e_samples.jsonl')})
    write(out/'figure_manifest.json', manifest)
