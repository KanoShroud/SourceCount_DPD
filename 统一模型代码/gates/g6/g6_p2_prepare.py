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
from 统一模型代码.gates.g6.g6_p2_runtime import (
    BASE, CONFIG, ARMS, SEEDS, Runtime, read, write, safe_print, digest_state, identity)
from 统一模型代码.gates.g6.g6_p2_model import forward
from 统一模型代码.gates.g6.g6_p2_train import payload, restore, loss_for, train_step, evaluate
from 统一模型代码.gates.g6.g6_p1_speed import physical_batches


def gradient_probe(runtime, context, head, arm, batch, targets, ids):
    output,_ = forward(context,head,batch,ids,arm,runtime.physics,runtime.stats('train',ids))
    _,parts,_ = loss_for(runtime,output,targets,ids)
    groups = {
        'semantic_final': [p for h in context.ch3.band_heads[:3] for p in h[2].parameters()]
                          + list(context.query_builder.band_residual.parameters()),
        'head_hidden': [p for h in context.ch3.band_heads[:3] for p in h[0].parameters()],
        'query': list(context.query_builder.anchor.parameters())+list(context.query_builder.cross_attention.parameters()),
        'cross_attn': list(context.ch3.cross_attn.parameters()),
        'selector': list(head.selector.parameters()) if head.selector is not None else []}
    params = [p for group in groups.values() for p in group]
    grads = torch.autograd.grad(parts['heatmap']+parts['offset'],params+[output[1]],allow_unused=True)
    offset,norms = 0,{}
    for name,group in groups.items():
        values = grads[offset:offset+len(group)]; offset += len(group)
        norms[name] = sum(float(g.square().sum()) for g in values if g is not None)**.5
    norms['band_logits'] = 0. if grads[-1] is None else float(grads[-1].norm())
    from 统一模型代码.gates.g5.e2e_g5_model import forward as base_forward
    base = base_forward(context,batch,ids,torch.device('cuda:0'),stop_gradient=True)
    old = torch.autograd.grad(base[3].sum()+base[4].sum(),params,allow_unused=True)
    norms['old_path'] = sum(float(g.square().sum()) for g in old if g is not None)**.5
    if any(norms[k] != 0 for k in ('semantic_final','band_logits','old_path')):
        raise RuntimeError('Semantic or old-path gradient leakage')
    if arm in ('a','b') and any(norms[k] != 0 for k in ('head_hidden','query','cross_attn')):
        raise RuntimeError('Stop-gradient leakage')
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
    safe_print('阶段1：只读复用P1物理缓存，核对两seed初始输出。')
    # Verify both seeds and frozen P1 forward before training any engineering step.
    from 统一模型代码.gates.g6.g6_p1_model import forward as p1_forward
    from 统一模型代码.gates.g6.g6_p2_runtime import g4
    initial_checks = {}
    iterator = batches(bundle[0],[probe_ids],prefetch=False)
    ids,batch = next(iterator); iterator.close()
    runtime.consumed(bundle[3],ids)
    stats = runtime.stats('train',ids)
    for seed in SEEDS:
        first = None
        for arm in ARMS:
            context,head,optimizer,params = runtime.context(seed,arm)
            g4.set_mode(context,training=False); head.eval()
            with torch.no_grad():
                output,_ = forward(context,head,batch,ids,arm,runtime.physics,stats)
                prior,_ = p1_forward(context,head.physical,batch,ids,'c1',runtime.physics,stats)
                assert same(output,prior), 'P2 zero initialization differs from P1 C1'
                values = [v.cpu().clone() for v in output]
                if first is not None:
                    assert same(first,values), 'Three arms not initially equal'
                first = values
            del context,head,optimizer,params,output,prior
            gc.collect(); torch.cuda.empty_cache()
        initial_checks[str(seed)] = {'three_arms_equal':True,'frozen_c1_equal':True}
    del batch,stats
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
            if arm == 'c' and not all(probes[3][k]>0 for k in ('head_hidden','query','cross_attn')):
                raise RuntimeError('New physical feedback missing after 3 updates')
            if arm in ('b','c') and probes[0]['selector'] <= 0:
                raise RuntimeError('Selector lacks initial localization gradient')
            if arm == 'c' and any(probes[0][k] != 0 for k in ('head_hidden','query','cross_attn')):
                raise RuntimeError('Zero selector unexpectedly transmits initial feedback')
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
        elapsed = time.perf_counter()-start
        previous_seconds = sum(read(p)['seconds'] for p in (BASE/'engineering').glob('*/report.json'))
        previous_seconds += sum(read(p)['seconds'] for p in (BASE/'engineering').glob('*/failure.json'))
        def estimate(epochs):
            train = sum(r['train_32_batches_seconds']/32*1024*epochs*2 for r in results.values())
            val = sum(r['validation_32_seconds']/32*(512*(epochs//2+1)+1024)*2 for r in results.values())
            return elapsed+previous_seconds+1.25*(train+val)+1200
        forecast = estimate(24)
        required = 55*1024**3
        report = dict(status='ENGINEERING_PASS',results=results,seconds=elapsed,
            initial_checks=initial_checks,
            preparation_consumed_seconds=elapsed+previous_seconds,
            forecast_base_seconds=estimate(20),forecast_max_seconds=forecast,
            forecast_within_budget=forecast<=CONFIG['wall_seconds'],disk_required_free_bytes=required,
            disk_ready=shutil.disk_usage(BASE).free>=required,
            cache_policy='Frozen P1 cache reused read-only; no generation',
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
