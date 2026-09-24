"""Short real-model checks only; no formal six-track training."""
import copy
import gc
import shutil
import time

import numpy as np
import torch

from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng, save_checkpoint, load_checkpoint
from 统一模型代码.gates.g5.r2.g5_r2_prepare import same
from 统一模型代码.gates.g6.g6_p1_runtime import (
    BASE, CONFIG, ARMS, SEEDS, Runtime, read, write, safe_print, digest_state, identity)
from 统一模型代码.gates.g6.g6_p1_model import forward
from 统一模型代码.gates.g6.g6_p1_train import payload, restore, loss_for, train_step, evaluate
from 统一模型代码.gates.g6.g6_p1_speed import physical_batches


def gradient_probe(runtime, context, head, arm, batch, targets, ids):
    stats = runtime.stats('train',ids) if head is not None else []
    output,_ = forward(context,head,batch,ids,arm,runtime.physics,stats)
    _,parts,_ = loss_for(runtime,output,targets,ids)
    loc = parts['heatmap']+parts['offset']
    groups = {'band_heads':list(context.ch3.band_heads[:3].parameters()),
              'query_residual':list(context.query_builder.parameters()),
              'cross_attn':list(context.ch3.cross_attn.parameters())}
    params = [p for group in groups.values() for p in group]
    grads = torch.autograd.grad(loc,params+[output[1]],allow_unused=True,retain_graph=True)
    offset,norms = 0,{}
    for name,group in groups.items():
        values = grads[offset:offset+len(group)]; offset += len(group)
        norms[name] = sum(float(g.square().sum()) for g in values if g is not None)**.5
    norms['band_logits'] = 0. if grads[-1] is None else float(grads[-1].norm())
    # Separate derivative of base heatmap tests the old path, even in C2.
    if head is not None:
        from 统一模型代码.gates.g5.e2e_g5_model import forward as base_forward
        base = base_forward(context,batch,ids,torch.device('cuda:0'),stop_gradient=True)
        old = torch.autograd.grad(base[3].sum()+base[4].sum(),params,allow_unused=True)
        norms['old_path'] = sum(float(g.square().sum()) for g in old if g is not None)**.5
        if norms['old_path'] != 0:
            raise RuntimeError('Old latent gradient leakage')
    if arm != 'c2' and any(v != 0 for v in norms.values()):
        raise RuntimeError('SG localization feedback leakage')
    if not all(np.isfinite(v) for v in norms.values()):
        raise RuntimeError('Nonfinite gradient probe')
    return norms


def prepare():
    start = time.perf_counter()
    out = BASE/f'engineering/{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True,exist_ok=False)
    runtime = Runtime(out,deadline=time.time()+3600)
    runtime.preflight()
    runtime.audit_raw()
    bundle = runtime.features('train')
    validation = runtime.features('val_select',bundle[-1])
    order = torch.randperm(4096,generator=torch.Generator().manual_seed(SEEDS[0]))[:128]
    groups = [torch.nonzero(bundle[1].counts==k).flatten().tolist() for k in range(4)]
    probe_ids = torch.tensor([groups[k][0] for k in range(4)])
    val_ids = [torch.nonzero(validation[1].counts==k).flatten().tolist()[j] for j in range(8) for k in range(4)]
    safe_print('阶段1：生成短测所需物理缓存并核对IQ/标签对应。')
    cache_reports = [runtime.ensure_cache('train',sorted(set(order.tolist()+probe_ids.tolist())),bundle[1]),
                     runtime.ensure_cache('val_select',val_ids,validation[1])]
    results,initial,raw_initial = {},None,None
    try:
        for arm in ARMS:
            safe_print(f'阶段2：{arm} 初始输出、0/1/3步梯度及恢复检查。')
            context,head,optimizer,params = runtime.context(SEEDS[0],arm)
            metrics,rows = evaluate(runtime,context,head,arm,validation,indices=val_ids,label=f'{arm} 初始检查')
            if initial is None:
                initial = rows
            elif initial != rows:
                raise RuntimeError('Initial decoded outputs differ')
            iterator = batches(bundle[0],[probe_ids],prefetch=False)
            ids,batch = next(iterator); iterator.close()
            runtime.consumed(bundle[3],ids)
            with torch.no_grad():
                output,_ = forward(context,head,batch,ids,arm,runtime.physics,
                                 runtime.stats('train',ids) if head is not None else [])
                saved_output = [x.cpu().clone() for x in output]
                if raw_initial is None:
                    raw_initial = saved_output
                elif not same(raw_initial,saved_output):
                    raise RuntimeError('Raw initial tensors differ')
                del output,saved_output
            probes = {0:gradient_probe(runtime,context,head,arm,batch,bundle[1],ids)}
            for step in range(1,4):
                train_step(runtime,context,head,optimizer,params,arm,batch,bundle[1],ids)
                if step in (1,3):
                    probes[step] = gradient_probe(runtime,context,head,arm,batch,bundle[1],ids)
            if arm == 'c2' and not all(probes[3][k]>0 for k in ('band_heads','query_residual','cross_attn','band_logits')):
                raise RuntimeError('New physical feedback missing after 3 updates')
            path = out/arm/'resume.pt'
            path.parent.mkdir()
            gen = torch.Generator().manual_seed(SEEDS[0])
            _ = torch.randperm(4096,generator=gen)
            saved = dict(state=payload(context,head),optimizer=optimizer.state_dict(),rng=rng_state(),generator=gen.get_state())
            save_checkpoint(path,saved)
            expected_order = torch.randperm(4096,generator=gen)
            expected_loss = train_step(runtime,context,head,optimizer,params,arm,batch,bundle[1],ids)
            expected = digest_state(payload(context,head))
            expected_optimizer = copy.deepcopy(optimizer.state_dict())
            loaded = load_checkpoint(path)
            restore(context,head,loaded['state']); optimizer.load_state_dict(loaded['optimizer']); restore_rng(loaded['rng'])
            gen.set_state(loaded['generator'])
            assert torch.equal(torch.randperm(4096,generator=gen),expected_order)
            iterator = physical_batches(runtime,bundle[0],[probe_ids],'train',head is not None)
            pref_ids,pref_batch,pref_stats = next(iterator); iterator.close()
            if head is not None:
                direct_stats = runtime.stats('train',pref_ids)
                for key,value in pref_stats.items():
                    assert value.is_pinned()
                    assert torch.equal(value,torch.stack([r[key] for r in direct_stats]))
                del direct_stats
            actual_loss = train_step(runtime,context,head,optimizer,params,arm,pref_batch,bundle[1],pref_ids,pref_stats)
            if expected_loss != actual_loss or expected != digest_state(payload(context,head)) or not same(expected_optimizer,optimizer.state_dict()):
                raise RuntimeError('Checkpoint replay mismatch')
            del context,head,optimizer,params,batch,saved,loaded,expected_optimizer,pref_batch,pref_stats
            gc.collect(); torch.cuda.empty_cache()
            safe_print(f'阶段3：{arm} 全模型32 batch及32条验证计时。')
            context,head,optimizer,params = runtime.context(SEEDS[0],arm)
            # Same balanced warmup, then the same fresh random sample order for all tracks.
            iterator = batches(bundle[0],[probe_ids],prefetch=False)
            ids,batch = next(iterator); iterator.close()
            train_step(runtime,context,head,optimizer,params,arm,batch,bundle[1],ids)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            iterator = physical_batches(runtime,bundle[0],list(order.split(4)),'train',head is not None)
            try:
                for ids,batch,stats in iterator:
                    runtime.consumed(bundle[3],ids)
                    train_step(runtime,context,head,optimizer,params,arm,batch,bundle[1],ids,stats)
            finally:
                iterator.close()
            torch.cuda.synchronize()
            train_seconds = time.perf_counter()-begin
            begin = time.perf_counter()
            evaluate(runtime,context,head,arm,validation,indices=val_ids,label=f'{arm} 短验证')
            val_seconds = time.perf_counter()-begin
            results[arm] = dict(train_32_batches_seconds=train_seconds,validation_32_seconds=val_seconds,
                gradient_probes=probes,initial_equal=True,resume_exact=True,prefetch_replay_exact=True)
            del context,head,optimizer,params,batch
            gc.collect(); torch.cuda.empty_cache()
        runtime.postcheck('engineering')
        runtime.audit_raw()
        # UTF-8 reporting is exercised in-process, without subprocess pipes.
        text = 'G6中文路径与进度日志验证通过'
        (out/'中文编码检查.txt').write_text(text,encoding='utf-8')
        assert (out/'中文编码检查.txt').read_text(encoding='utf-8') == text
        # Train and validation IQ have different measured loading costs; extrapolate separately.
        generation = {r['split']:r for r in cache_reports if r['count']}
        for old_path in sorted((BASE/'engineering').glob('*/report.json')):
            old = read(old_path)
            for split,r in zip(('train','val_select'),old['cache_reports']):
                if r['count'] and split not in generation:
                    generation[split] = {**r,'split':split}
        if set(generation) != {'train','val_select'}:
            raise RuntimeError('Missing per-split cache generation timing')
        per_cache = {s:r['seconds']/r['count'] for s,r in generation.items()}
        bytes_per = max(r['bytes']/r['count'] for r in generation.values())
        remaining = {s:n-len(runtime.cache_rows(s)) for s,n in [('train',4096),('val_select',512),('val_compare',1024)]}
        cache_seconds = remaining['train']*per_cache['train']+(remaining['val_select']+remaining['val_compare'])*per_cache['val_select']
        elapsed = time.perf_counter()-start
        previous_seconds = sum(read(p)['seconds'] for p in (BASE/'engineering').glob('*/report.json'))
        speed_reports = sorted((BASE/'speed_check').glob('*/report.json'))
        optimization_seconds = sum(read(p)['seconds'] for p in speed_reports)
        previous_seconds += optimization_seconds
        def estimate(epochs):
            train = sum(r['train_32_batches_seconds']/32*1024*epochs*2 for r in results.values())
            val = sum(r['validation_32_seconds']/32*(512*(epochs//2+1)+1024)*2 for r in results.values())
            return elapsed+previous_seconds+1.25*(train+val+cache_seconds)+1200
        forecast = estimate(24)
        required = sum(remaining.values())*bytes_per+5*1024**3+50*1024**3
        report = dict(status='ENGINEERING_PASS',results=results,seconds=elapsed,
            preparation_consumed_seconds=elapsed+previous_seconds, cache_seconds_per_sample=per_cache,
            optimization_seconds=optimization_seconds,optimization_reports=[identity(p) for p in speed_reports],
            forecast_base_seconds=estimate(20),forecast_max_seconds=forecast,
            forecast_within_budget=forecast<=CONFIG['wall_seconds'],disk_required_free_bytes=required,
            disk_ready=shutil.disk_usage(BASE).free>=required,cache_reports=cache_reports,
            cache_indexes={s:identity(BASE/f'physical_cache/{s}/index.json') for s in ('train','val_select')},
            pilot_cache_rows={s:dict(runtime.cache_rows(s)) for s in ('train','val_select')},
            cache_bytes_per_sample=bytes_per,
            peak_ram_percent=runtime.peak_ram,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3,
            formal_training_executed=False,val_compare_executed=False,test_executed=False,
            contract_sha256=identity(BASE/'contract.json')['sha256'])
        write(out/'report.json',report)
        write(BASE/'preparation_report.json',{**report,'report':identity(out/'report.json')})
        safe_print(f'工程检查通过。20轮预计{estimate(20)/3600:.2f}h，最长24轮预计{forecast/3600:.2f}h；12h内={report["forecast_within_budget"]}')
        return report
    except BaseException:
        import traceback
        write(out/'failure.json',{'status':'FAILED','traceback':traceback.format_exc(),'seconds':time.perf_counter()-start})
        raise
