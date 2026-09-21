"""G5一次开发集比较、配对区间与冻结Hard参考；不选模型、不访问test。"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_model import build_context,evaluate,g4,load_split,summarize  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_train import load_checkpoint,resource_guard  # noqa: E402


def bootstrap_indices(rows,repetitions,seed):
    rng=np.random.default_rng(seed)
    groups=[np.asarray([i for i,r in enumerate(rows) if r['true_count']==k]) for k in range(4)]
    return np.concatenate([rng.choice(ids,(repetitions,len(ids)),replace=True) for ids in groups if len(ids)],axis=1)


def metrics(rows,indices):
    count=np.asarray([r['true_count'] for r in rows],dtype=np.float64)
    error_count=np.asarray([len(r['matched_errors_m']) for r in rows],dtype=np.float64)
    error_sum=np.asarray([sum(x*x for x in r['matched_errors_m']) for r in rows],dtype=np.float64)
    true_sum=count[indices].sum(axis=-1)
    pairs=error_count[indices].sum(axis=-1)
    result={
        'gospa':np.asarray([r['gospa_m'] for r in rows])[indices].mean(axis=-1),
        'recall100':np.asarray([r['tp_at_100m'] for r in rows])[indices].sum(axis=-1)/np.maximum(true_sum,1),
        'coverage':pairs/np.maximum(true_sum,1),
        'count':np.asarray([r['predicted_count']==r['true_count'] for r in rows])[indices].mean(axis=-1),
        'rmse':np.sqrt(np.divide(error_sum[indices].sum(axis=-1),pairs,
                                out=np.full_like(pairs,np.nan),where=pairs>0)),
    }
    for name in ('band_f1','band_iou','band_only_f1','band_only_iou'):
        values=np.asarray([sum(r[name]) for r in rows])
        denom=np.asarray([len(r[name]) for r in rows])[indices].sum(axis=-1)
        result[name]=values[indices].sum(axis=-1)/np.maximum(denom,1)
    return result


def paired(base,candidate,repetitions,seed):
    assert [(r['raw_index'],r['true_count']) for r in base]==[(r['raw_index'],r['true_count']) for r in candidate]
    indices=bootstrap_indices(base,repetitions,seed)
    sampled_a,sampled_b=metrics(base,indices),metrics(candidate,indices)
    point_a,point_b=metrics(base,np.arange(len(base))),metrics(candidate,np.arange(len(base)))
    out={}
    for key in sampled_a:
        if key in ('rmse','band_f1','band_iou','band_only_f1','band_only_iou') and not any(r['true_count'] for r in base):
            continue
        delta=sampled_b[key]-sampled_a[key]
        if not np.isfinite(delta).all() or not np.isfinite(point_b[key]-point_a[key]):
            # 无有效匹配不能记为零误差，也不删除不利重采样后给出偏乐观区间。
            continue
        out[key]={'mean':float(point_b[key]-point_a[key]),'ci95':np.quantile(delta,[.025,.975]).tolist()}
    if point_a['rmse']>0 and np.isfinite(point_b['rmse']) and np.all(sampled_a['rmse']>0) and np.isfinite(sampled_b['rmse']).all():
        valid=np.ones_like(sampled_a['rmse'],dtype=bool)
        out['rmse_ratio']={'mean':float(point_b['rmse']/point_a['rmse']),
                           'ci95':np.quantile(sampled_b['rmse'][valid]/sampled_a['rmse'][valid],[.025,.975]).tolist()}
    return out


def compare(run):
    from 统一模型代码.gates.g5.e2e_g5_contract import verify_code
    verify_code(run)
    m=g4.read_json(run/'manifest.json')
    fm=g4.read_json(run/'feature_manifest.json')
    training=g4.read_json(run/'training_report.json')
    if training['status']!='PASS':
        raise RuntimeError('All six tracks must finish before comparison')
    destination=run/'comparison'
    destination.mkdir(exist_ok=True)
    device=torch.device('cuda:0')
    results={}
    for seed in m['config']['training_seeds']:
        pair={}
        for track in ('sg','e2e'):
            resource_guard(run,m)
            path=destination/f'{seed}_{track}.json'
            selected=run/'training'/str(seed)/track/'best.pt'
            if path.exists():
                payload=g4.read_json(path)
                if payload['checkpoint']!=g4.read_json(selected.with_suffix('.identity.json')):
                    raise RuntimeError('Existing comparison checkpoint changed')
                pair[track]=payload['evaluation']
                continue
            context=build_context(run,m,seed,device)
            checkpoint=load_checkpoint(selected)
            g4.load_state(context,checkpoint['state'])
            features,targets,cache=load_split(run,m,fm,'val_compare')
            evaluation=evaluate(context,features,targets,device,fm['files']['val_compare']['metadata'])
            payload={'status':'PASS','seed':seed,'track':track,'checkpoint':g4.read_json(selected.with_suffix('.identity.json')),
                     'selected_epoch':checkpoint['epoch'],'evaluation':evaluation,'test_executed':False}
            g4.write_json(path,payload)
            pair[track]=evaluation
            del context,checkpoint,features,targets,cache
            gc.collect()
            torch.cuda.empty_cache()
            print(f'G5 compare {seed}/{track} saved',flush=True)
        comparisons={}
        for scope in ('overall','0','1','2','3'):
            a=pair['sg']['samples']
            b=pair['e2e']['samples']
            if scope!='overall':
                a=[r for r in a if r['true_count']==int(scope)]
                b=[r for r in b if r['true_count']==int(scope)]
            comparisons[scope]=paired(a,b,m['config']['bootstrap_repetitions'],seed+100)
        results[str(seed)]={'tracks':{name:g4.compact(value) for name,value in pair.items()},'paired':comparisons}
    sample_base=g4.read_json(destination/f"{m['config']['training_seeds'][0]}_sg.json")['evaluation']['samples']
    delta_by_seed=[]
    for seed in m['config']['training_seeds']:
        a=g4.read_json(destination/f'{seed}_sg.json')['evaluation']['samples']
        b=g4.read_json(destination/f'{seed}_e2e.json')['evaluation']['samples']
        delta_by_seed.append([y['gospa_m']-x['gospa_m'] for x,y in zip(a,b)])
    averaged=np.mean(delta_by_seed,axis=0)
    sampled=averaged[bootstrap_indices(sample_base,2000,20260920)].mean(axis=-1)
    combined={'mean':float(averaged.mean()),'ci95':np.quantile(sampled,[.025,.975]).tolist(),
              'seed_means':np.mean(delta_by_seed,axis=1).tolist(),'seed_sd':float(np.std(np.mean(delta_by_seed,axis=1),ddof=1)),
              'interval_scope':'sample uncertainty conditional on these three training seeds'}
    harm=[]
    cfg=m['config']
    boundaries={'recall100':cfg['recall_100m_noninferiority'],'coverage':cfg['coverage_noninferiority'],
                'count':-cfg['auxiliary_material_drop'],'band_f1':-cfg['auxiliary_material_drop'],'band_iou':-cfg['auxiliary_material_drop']}
    for scope in ('overall','0','1','2','3'):
        for metric,bound in boundaries.items():
            hit=[seed for seed,data in results.items() if metric in data['paired'][scope] and data['paired'][scope][metric]['ci95'][1]<bound]
            if len(hit)>=2:
                harm.append({'scope':scope,'metric':metric,'seeds':hit,'boundary':bound})
        hit=[seed for seed,data in results.items() if 'rmse_ratio' in data['paired'][scope] and data['paired'][scope]['rmse_ratio']['ci95'][0]>cfg['rmse_ratio_maximum']]
        if len(hit)>=2:
            harm.append({'scope':scope,'metric':'rmse_ratio','seeds':hit,'boundary':cfg['rmse_ratio_maximum']})
    unresolved=any(row['completed_epoch']==64 and row['budget_candidate'] for pair in training['seeds'].values() for row in pair.values())
    undefined_rmse=[{'seed':seed,'scope':scope} for seed,data in results.items()
                    for scope in ('overall','1','2','3') if 'rmse_ratio' not in data['paired'][scope]]
    positive=sum(x<0 for x in combined['seed_means'])
    negative=sum(x>0 for x in combined['seed_means'])
    if harm or (negative>=2 and combined['ci95'][0]>0):
        status='TRADEOFF_OR_REGRESSION'
    elif positive>=2 and combined['ci95'][1]<0 and not unresolved and not undefined_rmse:
        status='FEEDBACK_SUPPORTED_ON_DEV'
    else:
        status='DIRECTIONAL_BUT_INCONCLUSIVE'
    result={'status':status,'results':results,'combined_gospa':combined,'stable_material_harm':harm,
            'training_budget_unresolved':unresolved,'undefined_rmse_comparisons':undefined_rmse,'test_executed':False}
    g4.write_json(run/'comparison_report.json',result)
    return result


def hard_reference(run):
    """复用原R5-A与未增强硬级联的冻结预测，只比较相同K=2/3样本。"""
    import json
    from 统一模型代码.common.g5_verified_io import verified_read
    from 统一模型代码.gates.g5.e2e_g5_contract import verify_code
    verify_code(run)
    m=g4.read_json(run/'manifest.json')
    fm=g4.read_json(run/'feature_manifest.json')
    origin=ROOT.parent/'SourceCount_DPD/outputs/s2g5r5_candidate_k/20260828_222900'
    final=g4.read_json(origin/'final_report.json')
    preflight=g4.read_json(origin/'00_preflight/preflight.json')
    analysis=g4.read_json(origin/'05_selector_analysis/r5a_analysis.json')
    for name in ('preflight','analysis','reload','peaks'):
        row=final['reports'][name]
        verified_read(row,run/'anomalies')
    assert final['status']=='PASS' and not final['test_executed']
    assert preflight['inputs']['raw_validation']['sha256']==m['inputs']['files'][3]['sha256']
    assert preflight['inputs']['val_compare']['sha256']==m['inputs']['files'][2]['sha256']
    for name,old_name in (('ch3_seed42','ch3'),('d8_seed42','d8')):
        row=next(x for x in m['inputs']['artifacts'] if x['name']==name)
        assert row['sha256']==final['frozen_inputs'][old_name]['sha256']
    peaks_report=json.loads(verified_read(final['reports']['peaks'],run/'anomalies'))
    peak_payload=verified_read(peaks_report['outputs']['val_compare'],run/'anomalies')
    peak_rows=[json.loads(line) for line in peak_payload.decode('utf-8').splitlines()]
    peaks={row['raw_index']:row for row in peak_rows}
    if len(peaks)!=len(peak_rows):
        raise RuntimeError('Duplicate historical reference sample')
    root=run/('hard_reference_recovery_v4' if (run/'engineering_v4/contract.json').exists() else 'hard_reference')
    root.mkdir(exist_ok=False)
    with (root/'historical_truth_top4.jsonl').open('xb') as handle:
        handle.write(peak_payload)
    _,targets,cache=load_split(run,m,fm,'val_compare')
    indices={row['raw_index']:i for i,row in enumerate(m['subsets']['val_compare']) if row['true_k'] in (2,3)}
    result={}
    for name,filename,metric_name in (('hard_base','pb_base_samples.jsonl','PB-BASE'),
                                      ('r5a_selected','pb_selected_samples.jsonl','PB-SELECTED')):
        path=origin/'05_selector_analysis'/filename
        registration=g4.identity(path)
        payload=verified_read(registration,run/'anomalies')
        if verified_read(registration,run/'anomalies')!=payload:
            raise RuntimeError('Reference reads inconsistent')
        # 保留当前读取的原字节；selected文件缺少历史逐文件SHA，明确不冒充历史登记。
        with (root/filename).open('xb') as handle:
            handle.write(payload)
        existing=[json.loads(line) for line in payload.decode('utf-8').splitlines()]
        if {row['raw_index'] for row in existing}!=set(indices) or len(existing)!=len(indices):
            raise RuntimeError('Historical reference does not cover the same K2/3 samples')
        rows=[]; historical_rows=[]; max_coordinate_difference=0.0; max_gospa_difference=0.0
        for old in existing:
            index=indices[old['raw_index']]
            meta=fm['files']['val_compare']['metadata'][index]
            assert old['true_count']==meta['true_k']
            truth=targets.positions[index,:old['true_count']].numpy()
            predicted=np.asarray(old['predicted_positions_m'],dtype=np.float32).reshape(-1,2)
            from scipy.optimize import linear_sum_assignment
            historical_truth=np.asarray(peaks[old['raw_index']]['true_positions_m'],dtype=np.float32)
            if historical_truth.shape!=truth.shape or not np.isfinite(historical_truth).all():
                raise RuntimeError('Historical truth shape/nonfinite mismatch')
            a,b=linear_sum_assignment(np.linalg.norm(truth[:,None]-historical_truth[None,:],axis=-1))
            difference=np.abs(truth[a]-historical_truth[b])
            # Bound only float32 coordinate round-off; never relax GOSPA validation.
            bound=4*np.finfo(np.float32).eps*np.maximum(1,np.abs(truth[a]))
            if np.any(difference>bound):
                raise RuntimeError(f'Reference coordinates differ beyond FP32 round-off: {old["raw_index"]}')
            max_coordinate_difference=max(max_coordinate_difference,float(difference.max()))
            historical_gospa=g4.g1.gospa_sample(historical_truth,predicted)
            if historical_gospa['value_m']!=old['gospa_m']:
                raise RuntimeError('Historical GOSPA not exactly reproduced with historical truth')
            gospa=g4.g1.gospa_sample(truth,predicted)
            max_gospa_difference=max(max_gospa_difference,abs(old['gospa_m']-gospa['value_m']))
            historical_rows.append({**old,'band_f1':[],'band_iou':[],'band_only_f1':[],'band_only_iou':[]})
            row={**old,**meta,'index':index,'true_positions_m':truth.tolist(),
                 'historical_gospa_m':old['gospa_m'],'gospa_m':float(gospa['value_m']),
                 'matched_errors_m':g4.distance_errors(truth,predicted),
                 'band_f1':[],'band_iou':[],'band_only_f1':[],'band_only_iou':[]}
            for component in ('localization','missed','false'):
                row[f'gospa_{component}_p_sum']=float(gospa[f'{component}_p_sum'])
            for threshold in (10,30,50,100):
                calculated=g4.g1.maximum_matches_within(truth,predicted,threshold)
                assert calculated==old[f'tp_at_{threshold}m']
            rows.append(row)
        rows.sort(key=lambda row:row['index'])
        summary=summarize(rows)
        old_metrics=analysis['track_point_metrics'][metric_name]
        historical_summary=summarize(historical_rows)
        assert np.isclose(historical_summary['gospa_mean_m'],old_metrics['mean_gospa_m'])
        assert np.isclose(historical_summary['recall_at_100m'],old_metrics['recall_100m'])
        result[name]={'overall':summary,'by_k':{str(k):summarize([r for r in rows if r['true_count']==k]) for k in (2,3)},
                      'samples':rows,'source_current_identity':registration,
                      'historical_per_file_sha_available':False,
                      'max_truth_coordinate_abs_difference_m':max_coordinate_difference,
                      'max_gospa_roundoff_difference_m':max_gospa_difference,
                      'comparison_truth':'G5 canonical metre coordinates; predictions unchanged',
                      'historical_aggregate_and_per_sample_metrics_reproduced':True}
    report={'status':'PASS','scope':'same 512 K=2/3 development samples only',
            'k0_k1_reference_not_implemented':True,'no_new_baseline_training_or_oracle_K':True,
            'references':result,'test_executed':False,'not_equal_pretraining_budget_causal_control':True}
    g4.write_json(run/'hard_reference_report.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--stage',choices=('compare','hard'),required=True)
    args=p.parse_args()
    result=(compare if args.stage=='compare' else hard_reference)(args.run.resolve(strict=True))
    print(result['status'],flush=True)
