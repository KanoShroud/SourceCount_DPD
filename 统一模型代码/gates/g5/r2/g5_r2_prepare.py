"""短时真实CUDA工程验证，不产生正式训练或val_compare性能结论。"""
from __future__ import annotations

import gc
import time
import torch

from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.gates.g5.e2e_g5_train import save_checkpoint, load_checkpoint, rng_state, restore_rng
from 统一模型代码.gates.g5.r2.g5_r2_runtime import ARMS, SEEDS, CONFIG, BASE, Runtime, write, digest_state, safe_print
from 统一模型代码.gates.g5.r2.g5_r2_model import forward_r2, compute_loss
from 统一模型代码.gates.g5.r2.g5_r2_train import train_step, payload, restore
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import evaluate


def same(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):
        return a.keys()==b.keys() and all(same(v,b[k]) for k,v in a.items())
    if isinstance(a,(list,tuple)):
        return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def gradient_probe(runtime, context, head, arm, batch, targets, ids):
    output,candidates,scores = forward_r2(context,head,batch,ids,torch.device('cuda:0'),arm)
    _,parts,stats = compute_loss(output,candidates,scores,targets,ids,runtime.manifest['config'],arm)
    loc = parts['heatmap']+parts['offset']+.2*parts['association']
    grad = torch.autograd.grad(loc,(output[0],output[1]),allow_unused=True,retain_graph=True)
    norms = [0. if x is None else float(x.norm()) for x in grad]
    if arm != 'c2' and any(x != 0 for x in norms):
        raise RuntimeError('SG feedback was not blocked')
    if arm == 'c2' and not all(x>0 for x in norms):
        raise RuntimeError('E2E localization feedback missing')
    assoc_norms = None
    if arm != 'c0' and stats.get('positive_candidates',0) and stats.get('negative_candidates',0):
        grads = torch.autograd.grad(parts['association'],(output[0],output[1]),allow_unused=True)
        assoc_norms = [0. if x is None else float(x.norm()) for x in grads]
        if arm == 'c2' and not all(x>0 for x in assoc_norms):
            raise RuntimeError('Association feedback missing after head warmup')
        if arm == 'c1' and any(x!=0 for x in assoc_norms):
            raise RuntimeError('SG association feedback not blocked')
    return {'localization_query_band_norms':norms,'association_query_band_norms':assoc_norms,'counts':stats}


def prepare():
    started = time.time()
    out = BASE/f'engineering/{time.strftime("%Y%m%d_%H%M%S")}_{time.time_ns()}'
    runtime = Runtime(out,deadline=started+1200)
    runtime.preflight()
    bundle = runtime.features('train')
    validation = runtime.features('val_select',bundle[-1])
    # Interleave K for the first two steps; all classes and multisource gradients covered.
    groups = [torch.nonzero(bundle[1].counts==k).flatten().tolist() for k in range(4)]
    order = torch.tensor([groups[k][j] for j in range(32) for k in range(4)])
    val_ids = [torch.nonzero(validation[1].counts==k).flatten().tolist()[j]
               for j in range(8) for k in range(4)]
    results, initial = {},None
    try:
        for arm in ARMS:
            safe_print(f'工程测试 {arm}：相同起点、梯度、恢复与32 batch测速')
            context,head,optimizer,parameters = runtime.context(SEEDS[0],arm)
            metrics,rows = evaluate(runtime,context,head,arm,validation,indices=val_ids,label=f'{arm} 初始短验证')
            if initial is None:
                initial = rows
            elif rows != initial:
                raise RuntimeError('Initial C0/C1/C2 predictions differ')
            iterator = batches(bundle[0],list(order[:8].split(4)),prefetch=True)
            try:
                ids1,batch1 = next(iterator)
                runtime.consumed(bundle[3],ids1)
                train_step(runtime,context,head,optimizer,parameters,arm,batch1,bundle[1],ids1)
                ids2,batch2 = next(iterator)
                runtime.consumed(bundle[3],ids2)
                probe = gradient_probe(runtime,context,head,arm,batch2,bundle[1],ids2)
                path = out/arm/'resume.pt'
                path.parent.mkdir()
                save_checkpoint(path,{'state':payload(context,head),'optimizer':optimizer.state_dict(),'rng':rng_state()})
                expected_loss = train_step(runtime,context,head,optimizer,parameters,arm,batch2,bundle[1],ids2)[0]
                expected = digest_state(payload(context,head))
                # Snapshot is on CPU so the second update cannot mutate the reference.
                expected_optimizer = {k:v for k,v in optimizer.state_dict().items()}
                import copy
                expected_optimizer = copy.deepcopy(expected_optimizer)
                saved = load_checkpoint(path)
                restore(context,head,saved['state'])
                optimizer.load_state_dict(saved['optimizer'])
                restore_rng(saved['rng'])
                actual_loss = train_step(runtime,context,head,optimizer,parameters,arm,batch2,bundle[1],ids2)[0]
                if actual_loss!=expected_loss or digest_state(payload(context,head))!=expected or not same(expected_optimizer,optimizer.state_dict()):
                    raise RuntimeError('Checkpoint resume mismatch')
                del saved,expected_optimizer
            finally:
                iterator.close()
            del context,head,optimizer,parameters,batch1,batch2,iterator
            gc.collect()
            torch.cuda.empty_cache()
            # Fresh start: discard engineering updates; identical benchmark order for all arms.
            context,head,optimizer,parameters = runtime.context(SEEDS[0],arm)
            benchmark_order = torch.randperm(len(bundle[1].counts),generator=torch.Generator().manual_seed(SEEDS[0]))[:128]
            iterator = batches(bundle[0],list(benchmark_order.split(4)),prefetch=True)
            start = time.perf_counter()
            try:
                for ids,batch in iterator:
                    runtime.consumed(bundle[3],ids)
                    train_step(runtime,context,head,optimizer,parameters,arm,batch,bundle[1],ids)
            finally:
                iterator.close()
            seconds = time.perf_counter()-start
            begin = time.perf_counter()
            evaluate(runtime,context,head,arm,validation,indices=val_ids,label=f'{arm} 短验证测速')
            val_seconds = time.perf_counter()-begin
            results[arm] = {'train_32_batches_seconds':seconds,'validation_32_seconds':val_seconds,
                            'gradient_probe':probe,'resume_exact':True,'initial_equal':True}
            del context,head,optimizer,parameters,iterator,batch
            gc.collect()
            torch.cuda.empty_cache()
        runtime.postcheck('engineering')
        # Include epoch0, final compare, 25% throughput margin and 20min closeout.
        def estimate(epochs):
            train = sum(r['train_32_batches_seconds']/32*1024*epochs*len(SEEDS) for r in results.values())
            passes = epochs//CONFIG['evaluate_every']+1
            val = sum(r['validation_32_seconds']/32*(512*passes+1024)*len(SEEDS) for r in results.values())
            return (train+val)*1.25+1200
        forecast = estimate(CONFIG['extension_epochs'])
        forecast_base = estimate(CONFIG['base_epochs'])
        report = {'status':'ENGINEERING_PASS','results':results,'formal_training_executed':False,
                  'val_compare_executed':False,'test_executed':False,'seconds':time.time()-started,
                  'forecast_base_seconds':forecast_base,
                  'forecast_max_seconds':forecast,
                  'forecast_within_budget':forecast+time.time()-started<CONFIG['wall_seconds'],
                  'contract_sha256':__import__('hashlib').sha256((BASE/'contract.json').read_bytes()).hexdigest(),
                  'peak_ram_percent':runtime.peak_ram,'output':str(out)}
        write(out/'preparation_report.json',report)
        write(BASE/'preparation_report.json',report)
        safe_print(f'短时工程验证通过。24轮最坏情形预计{forecast/3600:.2f}小时（含规划余量）。')
        return report
    except BaseException as exc:
        import traceback
        write(out/'failure_report.json',{'status':'ENGINEERING_FAILED','error':repr(exc),'traceback':traceback.format_exc()})
        raise
