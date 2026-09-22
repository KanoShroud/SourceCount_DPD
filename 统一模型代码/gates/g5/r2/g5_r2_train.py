"""R2三组配对训练；完整epoch提交恢复，未提交尝试保留。"""
from __future__ import annotations

import gc
from pathlib import Path
import time
import numpy as np
import torch

from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng, save_checkpoint, load_checkpoint
from 统一模型代码.gates.g5.r2.g5_r2_runtime import CONFIG, ARMS, SEEDS, Progress, read, write, safe_print, digest_state, g4
from 统一模型代码.gates.g5.r2.g5_r2_model import forward_r2, compute_loss
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import evaluate


def payload(context, head):
    return {'base':g4.state_payload(context), 'association':head.state_dict() if head is not None else None}


def restore(context, head, state):
    g4.load_state(context,state['base'])
    if head is not None:
        head.load_state_dict(state['association'],strict=True)
    elif state['association'] is not None:
        raise ValueError('C0 cannot load association weights')


def selection_key(metrics, epoch):
    return (-metrics['joint_recall100_f1_08'], metrics['gospa_m'], epoch)


def must_extend(history, end=None):
    end = CONFIG['base_epochs'] if end is None else end
    rows = [r for r in history if r.get('validation') is not None and r['epoch'] <= end]
    best = min(rows,key=lambda r:selection_key(r['validation'],r['epoch']))
    values = [next(r['validation']['joint_recall100_f1_08'] for r in rows if r['epoch']==e)
              for e in (end-4,end-2,end)]
    return best['epoch'] in (end-2,end) and values[-1]>values[0] and any(b>a for a,b in zip(values,values[1:]))


def train_step(runtime, context, head, optimizer, parameters, arm, batch, targets, ids):
    runtime.guard()
    outputs,candidates,scores = forward_r2(context,head,batch,ids,torch.device('cuda:0'),arm)
    loss,components,stats = compute_loss(outputs,candidates,scores,targets,ids,runtime.manifest['config'],arm)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameters,runtime.manifest['config']['gradient_clip'])
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise RuntimeError('Nonfinite loss/gradient')
    optimizer.step()
    return float(loss.detach()), {k:float(v.detach()) for k,v in components.items()},stats,float(norm)


def commit(root, context, head, optimizer, generator, history, best, steps):
    epoch = history[-1]['epoch']
    folder = root/'checkpoints'/f'epoch{epoch:03d}_{time.time_ns()}'
    folder.mkdir(parents=True)
    cp = save_checkpoint(folder/'state.pt', {'epoch':epoch,'state':payload(context,head),
        'optimizer':optimizer.state_dict(),'generator':generator.get_state(),'rng':rng_state(),
        'history':history,'best':best,'steps':steps})
    # Commit marker is published only after checkpoint+identity round-trip succeeds.
    write(root/'completed.json',{'epoch':epoch,'checkpoint':cp,'best':best,'steps':steps})
    return cp


def train_track(runtime, seed, arm, end):
    root = runtime.out/'training'/str(seed)/arm
    root.mkdir(parents=True,exist_ok=True)
    bundle = runtime.features('train')
    validation = runtime.features('val_select', bundle[-1])
    context,head,optimizer,parameters = runtime.context(seed,arm)
    generator = torch.Generator().manual_seed(seed)
    pointer = root/'completed.json'
    if pointer.exists():
        previous = read(pointer)
        if read(Path(previous['checkpoint']['path']).with_suffix('.identity.json')) != previous['checkpoint']:
            raise RuntimeError('Committed checkpoint identity changed')
        cp = load_checkpoint(Path(previous['checkpoint']['path']))
        if cp['epoch'] != previous['epoch'] or cp['steps'] != previous['steps']:
            raise RuntimeError('Checkpoint commit mismatch')
        restore(context,head,cp['state'])
        optimizer.load_state_dict(cp['optimizer'])
        generator.set_state(cp['generator'])
        restore_rng(cp['rng'])
        history,best,steps = cp['history'],cp['best'],cp['steps']
        del cp
    else:
        metrics,rows = evaluate(runtime,context,head,arm,validation,label=f'{seed}/{arm} 初始验证')
        # Compare all three initial predictions at each seed, not just aggregate metrics.
        import hashlib
        import json
        signature = hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()
        marker = root.parent/'initial_predictions.json'
        if marker.exists() and read(marker)['sha256'] != signature:
            raise RuntimeError('C0/C1/C2 initial predictions disagree')
        if not marker.exists():
            write(marker,{'sha256':signature})
        initial = root/'initial.pt'
        identity = save_checkpoint(initial,{'state':payload(context,head),'metrics':metrics,'epoch':0})
        best = {'epoch':0,'metrics':metrics,'checkpoint':identity}
        history,steps = [{'epoch':0,'validation':metrics}],0
        write(root/'initial.json',{'base_digest':digest_state(g4.state_payload(context)),
                                  'prediction_digest':signature})
        commit(root,context,head,optimizer,generator,history,best,steps)
    try:
        for epoch in range(history[-1]['epoch']+1,end+1):
            runtime.guard()
            g4.set_mode(context,training=True)
            if head is not None:
                head.train()
            order = torch.randperm(len(bundle[1].counts),generator=generator)
            batch_ids = order.split(CONFIG['batch_size'])
            progress = Progress(f'{seed}/{arm} epoch {epoch}/{end}',len(batch_ids),runtime.out/'progress.json')
            iterator = batches(bundle[0],batch_ids,prefetch=True)
            start = time.perf_counter()
            losses,stats,component_sums = [],{},{}
            max_norm = 0.0
            try:
                for n,(ids,batch) in enumerate(iterator,1):
                    runtime.consumed(bundle[3],ids)
                    loss,components,counts,norm = train_step(runtime,context,head,optimizer,parameters,arm,batch,bundle[1],ids)
                    losses.append(loss)
                    max_norm = max(max_norm,norm)
                    for k,v in components.items():
                        component_sums[k] = component_sums.get(k,0)+v
                    for k,v in counts.items():
                        stats[k] = stats.get(k,0)+v
                    steps += 1
                    progress.update(n)
            finally:
                iterator.close()
            seconds = time.perf_counter()-start
            validation_seconds,metrics = 0,None
            if epoch%2==0:
                begin = time.perf_counter()
                metrics,_ = evaluate(runtime,context,head,arm,validation,label=f'{seed}/{arm} epoch {epoch} 验证')
                validation_seconds = time.perf_counter()-begin
                if selection_key(metrics,epoch)<selection_key(best['metrics'],best['epoch']):
                    choice = root/'selections'/f'epoch{epoch:03d}_{time.time_ns()}'
                    choice.mkdir(parents=True)
                    ident = save_checkpoint(choice/'state.pt',{'state':payload(context,head),'metrics':metrics,'epoch':epoch})
                    best = {'epoch':epoch,'metrics':metrics,'checkpoint':ident}
            history.append({'epoch':epoch,'validation':metrics,'training_seconds':seconds,
                            'validation_seconds':validation_seconds,'loss':float(np.mean(losses)),
                            'association_counts':stats,'gradient_norm_max':max_norm,
                            'loss_components':{k:v/len(losses) for k,v in component_sums.items()}})
            commit(root,context,head,optimizer,generator,history,best,steps)
            write(root/'history.json',history)
            safe_print(f'完成 {seed}/{arm} 第{epoch}轮：训练{seconds:.1f}s，验证{validation_seconds:.1f}s；'
                       f'best={best["epoch"]}，joint Recall={best["metrics"]["joint_recall100_f1_08"]:.4f}')
        result = {'status':'PASS','seed':seed,'arm':arm,'completed_epoch':history[-1]['epoch'],
                  'optimizer_steps':steps,'best':best,'extend':must_extend(history),
                  'budget_unresolved':end==CONFIG['extension_epochs'] and best['epoch'] in (end-2,end),
                  'total_training_seconds':sum(r.get('training_seconds',0) for r in history)}
        runtime.postcheck(f'{seed}_{arm}_epoch{end}')
        write(root/f'report_epoch{end}.json',result)
        return result
    finally:
        del context,head,optimizer,parameters,bundle,validation
        gc.collect()
        torch.cuda.empty_cache()


def train_all(runtime):
    results = {}
    base,extended = CONFIG['base_epochs'],CONFIG['extension_epochs']
    for seed in SEEDS:
        current = {}
        for arm in ARMS:
            path = runtime.out/f'training/{seed}/{arm}/report_epoch{base}.json'
            current[arm] = read(path) if path.exists() else train_track(runtime,seed,arm,base)
        if any(row['extend'] for row in current.values()):
            for arm in ARMS:
                path = runtime.out/f'training/{seed}/{arm}/report_epoch{extended}.json'
                current[arm] = read(path) if path.exists() else train_track(runtime,seed,arm,extended)
        results[str(seed)] = current
        write(runtime.out/'training_report.json',{'status':'RUNNING','seeds':results})
    report = {'status':'PASS','seeds':results,'test_executed':False}
    write(runtime.out/'training_report.json',report)
    return report
