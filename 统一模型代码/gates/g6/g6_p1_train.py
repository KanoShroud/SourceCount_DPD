"""Paired continuation, exact epoch commits, and common R1 joint decoding."""
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from 统一模型代码.gates.g6.g6_p1_speed import physical_batches
from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.e2e_g5_model import band_values
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng, save_checkpoint, load_checkpoint
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run
from 统一模型代码.gates.g5.r1.g5_r1_report import paired_many
from 统一模型代码.gates.g5.r2.g5_r2_model import candidate_batch, decode_r2
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import summarize as base_summary, verdict, auxiliary_pairs
from 统一模型代码.gates.g5.r2.g5_r2_train import selection_key, must_extend
from 统一模型代码.gates.g6.g6_p0 import overlap_category
from 统一模型代码.gates.g6.g6_p1_model import forward, identity_hits
from 统一模型代码.gates.g6.g6_p1_runtime import ARMS, SEEDS, Progress, read, write, safe_print, g4


def payload(context, head):
    return {'base':g4.state_payload(context), 'physical':head.state_dict() if head is not None else None}


def restore(context, head, state):
    g4.load_state(context,state['base'])
    if head is not None:
        head.load_state_dict(state['physical'],strict=True)
    elif state['physical'] is not None:
        raise ValueError('Unexpected physical head in C0')


def loss_for(runtime, output, targets, ids):
    return g4.r1.compute_losses(output[1],output[3],output[4],g4.as_cached_targets(targets),
                               ids,runtime.manifest['config'])


def train_step(runtime, context, head, optimizer, params, arm, batch, targets, ids, stats=None):
    runtime.guard()
    if stats is None:
        stats = runtime.stats('train',ids) if head is not None else []
    output,_ = forward(context,head,batch,ids,arm,runtime.physics,stats)
    loss,parts,_ = loss_for(runtime,output,targets,ids)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params,10)
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise RuntimeError('Nonfinite loss/gradient')
    optimizer.step()
    return float(loss.detach()), {k:float(v.detach()) for k,v in parts.items()}


def summarize(rows):
    result = base_summary(rows)
    den = sum(r['spatial_identity_denominator'] for r in rows)
    num = sum(r['spatial_identity_numerator'] for r in rows)
    result.update(spatial_identity_denominator=den,spatial_identity_numerator=num,
                  spatial_identity_rate=num/den if den else None,
                  residual_rms=float(np.sqrt(np.mean([r['residual_mean_square'] for r in rows]))),
                  residual_spatial_std_mean=float(np.mean([r['residual_spatial_std_mean'] for r in rows])))
    return result


@torch.no_grad()
def evaluate(runtime, context, head, arm, bundle, split='val_select', indices=None, label='验证'):
    features,targets,metadata,index,_ = bundle
    indices = list(range(len(targets.counts))) if indices is None else list(indices)
    g4.set_mode(context,training=False)
    if head is not None:
        head.eval()
    iterator = physical_batches(runtime,features,
        [torch.tensor(indices[i:i+4]) for i in range(0,len(indices),4)],split,head is not None)
    progress = Progress(label,len(indices),runtime.out/'progress.json')
    rows = []
    try:
        for ids,batch,stats in iterator:
            runtime.guard()
            runtime.consumed(index,ids)
            output,residual = forward(context,head,batch,ids,arm,runtime.physics,stats)
            candidates = candidate_batch(output[3],output[4])
            logits = output[1].cpu()
            res = residual.cpu()
            for local,i in enumerate(ids.tolist()):
                k = int(targets.counts[i])
                truth = targets.positions[i,:k].numpy()
                bands,ignore = targets.band[i].numpy(),targets.ignore[i].numpy()
                logit = logits[local].numpy()
                decoded = decode_r2(logit,candidates[local],None)
                row = R1Run.metric(None,truth,decoded['joint'],logit,decoded['active'],bands,ignore,metadata[i])
                mapping = {}
                if k:
                    cost = np.empty((3,k))
                    for q in range(3):
                        for t in range(k):
                            valid = targets.ignore[i,t]<.5
                            cost[q,t] = float(torch.nn.functional.binary_cross_entropy_with_logits(
                                logits[local,q,valid],targets.band[i,t,valid]))
                    a,b = linear_sum_assignment(cost)
                    mapping = dict(zip(a.tolist(),b.tolist()))
                row['band_only_f1'],row['band_only_iou'] = band_values(logits[local],targets,i,mapping)
                row.update(identity_hits(truth,decoded['joint'],logit,decoded['active'],bands,ignore))
                row.update(index=i,truth=truth.tolist(),band_logits=logit.tolist(),decode=decoded,
                    overlap_category=overlap_category(bands[:k]),
                    residual_mean_square=float(res[local].square().mean()),
                    residual_spatial_std_mean=float(res[local].flatten(1).std(dim=1,unbiased=False).mean()))
                rows.append(row)
            progress.update(len(rows))
    finally:
        iterator.close()
        g4.set_mode(context,training=True)
        if head is not None:
            head.train()
    return summarize(rows),rows


def initial_signature(rows):
    # Residual statistics are zero at initialization; include all predicted outputs.
    return hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()


def train_track(runtime, seed, arm, end):
    root = runtime.out/f'training/{seed}/{arm}'
    root.mkdir(parents=True,exist_ok=True)
    bundle = runtime.features('train')
    validation = runtime.features('val_select',bundle[-1])
    context,head,optimizer,params = runtime.context(seed,arm)
    generator = torch.Generator().manual_seed(seed)
    def commit(history,best,steps):
        path = root/f'checkpoints/epoch{history[-1]["epoch"]:03d}_{time.time_ns()}/state.pt'
        path.parent.mkdir(parents=True)
        row = save_checkpoint(path,dict(state=payload(context,head),optimizer=optimizer.state_dict(),
            generator=generator.get_state(),rng=rng_state(),history=history,best=best,steps=steps))
        write(root/'completed.json',dict(checkpoint=row,epoch=history[-1]['epoch'],steps=steps))
    try:
        if (root/'completed.json').exists():
            marker = read(root/'completed.json')
            verified_read(marker['checkpoint'],runtime.out/'anomalies')
            cp = load_checkpoint(Path(marker['checkpoint']['path']))
            restore(context,head,cp['state'])
            optimizer.load_state_dict(cp['optimizer'])
            generator.set_state(cp['generator']); restore_rng(cp['rng'])
            history,best,steps = cp['history'],cp['best'],cp['steps']
            del cp
        else:
            metrics,rows = evaluate(runtime,context,head,arm,validation,label=f'{seed}/{arm} epoch0')
            marker = root.parent/'initial_predictions.json'
            signature = initial_signature(rows)
            if marker.exists() and read(marker)['sha256'] != signature:
                raise RuntimeError('Three-track initial predictions disagree')
            if not marker.exists():
                write(marker,{'sha256':signature})
            row = save_checkpoint(root/'initial.pt',{'state':payload(context,head),'epoch':0,'metrics':metrics})
            best = {'epoch':0,'metrics':metrics,'checkpoint':row}
            history,steps = [{'epoch':0,'validation':metrics}],0
            commit(history,best,steps)
        for epoch in range(history[-1]['epoch']+1,end+1):
            g4.set_mode(context,training=True)
            if head is not None:
                head.train()
            order = torch.randperm(len(bundle[1].counts),generator=generator).split(4)
            iterator = physical_batches(runtime,bundle[0],order,'train',head is not None)
            progress = Progress(f'{seed}/{arm} epoch {epoch}/{end}',len(order),runtime.out/'progress.json')
            start = time.perf_counter()
            losses,parts_sum = [],{}
            try:
                for n,(ids,batch,stats) in enumerate(iterator,1):
                    runtime.consumed(bundle[3],ids)
                    loss,parts = train_step(runtime,context,head,optimizer,params,arm,batch,bundle[1],ids,stats)
                    losses.append(loss); steps += 1
                    for key,v in parts.items():
                        parts_sum[key] = parts_sum.get(key,0)+v
                    progress.update(n)
            finally:
                iterator.close()
            elapsed = time.perf_counter()-start
            metrics,val_seconds = None,0
            if epoch%2 == 0:
                start = time.perf_counter()
                metrics,_ = evaluate(runtime,context,head,arm,validation,label=f'{seed}/{arm} 验证{epoch}')
                val_seconds = time.perf_counter()-start
                if selection_key(metrics,epoch)<selection_key(best['metrics'],best['epoch']):
                    path = root/f'selections/epoch{epoch:03d}_{time.time_ns()}/state.pt'
                    path.parent.mkdir(parents=True)
                    row = save_checkpoint(path,{'state':payload(context,head),'epoch':epoch,'metrics':metrics})
                    best = {'epoch':epoch,'metrics':metrics,'checkpoint':row}
            history.append(dict(epoch=epoch,validation=metrics,training_seconds=elapsed,
                validation_seconds=val_seconds,loss=float(np.mean(losses)),
                loss_components={k:v/len(losses) for k,v in parts_sum.items()}))
            commit(history,best,steps)
            write(root/'history.json',history)
            safe_print(f'{seed}/{arm} 完成{epoch}轮，训练{elapsed:.1f}s，验证{val_seconds:.1f}s；best={best["epoch"]}')
        runtime.postcheck(f'{seed}_{arm}_{end}')
        result = dict(seed=seed,arm=arm,completed_epoch=history[-1]['epoch'],optimizer_steps=steps,best=best,
            extend=must_extend(history,end=20),budget_unresolved=end==24 and best['epoch'] in (22,24),
            total_training_seconds=sum(r.get('training_seconds',0) for r in history))
        write(root/f'report_epoch{end}.json',result)
        return result
    finally:
        context = head = optimizer = params = bundle = validation = None
        gc.collect(); torch.cuda.empty_cache()


def train_all(runtime):
    results = {}
    for seed in SEEDS:
        current = {}
        for arm in ARMS:
            path = runtime.out/f'training/{seed}/{arm}/report_epoch20.json'
            current[arm] = read(path) if path.exists() else train_track(runtime,seed,arm,20)
        if any(r['extend'] for r in current.values()):
            for arm in ARMS:
                path = runtime.out/f'training/{seed}/{arm}/report_epoch24.json'
                current[arm] = read(path) if path.exists() else train_track(runtime,seed,arm,24)
        results[str(seed)] = current
        write(runtime.out/'training_report.json',{'status':'RUNNING','seeds':results})
    report = {'status':'PASS','seeds':results,'test_executed':False}
    write(runtime.out/'training_report.json',report)
    return report


def build_report(results, out, training):
    report = {'status':'COMPLETE_FOR_REVIEW','training':training,'tracks':{},'paired':{},
              'scope':'Previously used development data; conditional on two training seeds','test_executed':False}
    for (seed,arm),rows in results.items():
        groups = {f'K{k}':[r for r in rows if r['true_count']==k] for k in range(4)}
        groups.update({f'overlap_{g}':[r for r in rows if r['overlap_category']==g]
                       for g in ('distinct','partial','identical')})
        report['tracks'][f'{seed}_{arm}'] = {'overall':summarize(rows),
                  'strata':{k:summarize(v) for k,v in groups.items() if v}}
    lines = ['# G6-P1运行摘要','','六轨训练及开发集比较完成；test未读取。','',
             '| 比较 | 联合Recall差/pp | GOSPA差/m | RMSE差/m | 判定 |','|---|---:|---:|---:|---|']
    for name,a,b in [('physical_c1_minus_c0','c0','c1'),('feedback_c2_minus_c1','c1','c2')]:
        pairs = [(results[s,a],results[s,b]) for s in SEEDS]
        combined = paired_many(pairs,seed=20260921,repeats=2000)
        combined.update(auxiliary_pairs(pairs,20260921,2000))
        report['paired'][name] = {'combined':combined,'decision':verdict(combined),
            'per_seed':{str(s):paired_many([p],seed=s,repeats=2000) for s,p in zip(SEEDS,pairs)}}
        lines.append(f'| {name} | {100*combined["joint_recall100_f1_08"]["mean"]:.3f} | '
                     f'{combined["gospa_m"]["mean"]:.3f} | {combined["matched_rmse_m"]["mean"]:.3f} | {verdict(combined)} |')
    write(out/'comparison_report.json',report)
    (out/'运行摘要.md').write_text('\n'.join(lines)+'\n\nRMSE结合覆盖率解读；初始/末轮及触边状态见training_report.json。\n',encoding='utf-8')
    return report
