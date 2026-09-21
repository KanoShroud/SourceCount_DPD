"""4096样本manifest上的128步成本试跑与可恢复性检查，不计入正式模型结果。"""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_model import build_context,evaluate,forward,g4,load_split  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_train import load_checkpoint,save_checkpoint,rng_state,restore_rng,resource_guard  # noqa: E402
from 统一模型代码.common.g5_verified_io import RetryVerifiedCache  # noqa: E402


def step(context,optimizer,features,targets,indices,config,device,sg):
    _,logits,_,heatmap,offset=forward(context,features,indices,device,stop_gradient=sg)
    loss,_,_=g4.r1.compute_losses(logits,heatmap,offset,targets,indices,config)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(context.parameters,config['gradient_clip'])
    if not torch.isfinite(loss) or not torch.isfinite(norm):
        raise RuntimeError('Nonfinite pilot loss/gradient')
    optimizer.step()
    return float(loss.detach()),float(norm)


def main(run):
    from 统一模型代码.gates.g5.e2e_g5_contract import verify_code
    run=run.resolve(strict=True)
    verify_code(run)
    manifest=g4.read_json(run/'manifest.json')
    fm=g4.read_json(run/'feature_manifest.json')
    if fm['status']!='PASS' or g4.read_json(run/'provenance_audit.json')['status']!='PASS':
        raise RuntimeError('Data preparation incomplete')
    config=manifest['config']
    root=run/'pilot'
    root.mkdir(exist_ok=False)
    seed=config['training_seeds'][0]
    batches=torch.randperm(4096,generator=torch.Generator().manual_seed(seed)).split(4)
    device=torch.device('cuda:0')
    results={}
    for track in ('sg','e2e'):
        cache=RetryVerifiedCache(config['feature_cache_bytes'],run/'anomalies')
        features,targets,_=load_split(run,manifest,fm,'train',cache)
        selected,select_targets,_=load_split(run,manifest,fm,'val_select',cache)
        cached_targets=g4.as_cached_targets(targets)
        context=build_context(run,manifest,seed,device)
        initial=g4.state_digest(context)
        optimizer=torch.optim.AdamW(context.parameter_groups,weight_decay=config['weight_decay'])
        torch.cuda.reset_peak_memory_stats()
        start=time.perf_counter()
        for index,indices in enumerate(batches[:128],1):
            resource_guard(run,manifest)
            step(context,optimizer,features,cached_targets,indices,config,device,track=='sg')
            if index%32==0:
                print(f'G5 pilot {track}: {index}/128',flush=True)
        train_seconds=time.perf_counter()-start
        checkpoint=root/f'{track}_resume.pt'
        save_checkpoint(checkpoint,{'epoch':0,'state':g4.state_payload(context),'optimizer':optimizer.state_dict(),'rng':rng_state()})
        result1=step(context,optimizer,features,cached_targets,batches[128],config,device,track=='sg')
        digest1=g4.state_digest(context)
        del context,optimizer
        gc.collect()
        torch.cuda.empty_cache()
        context=build_context(run,manifest,seed,device)
        optimizer=torch.optim.AdamW(context.parameter_groups,weight_decay=config['weight_decay'])
        cp=load_checkpoint(checkpoint)
        g4.load_state(context,cp['state'])
        optimizer.load_state_dict(cp['optimizer'])
        restore_rng(cp['rng'])
        result2=step(context,optimizer,features,cached_targets,batches[128],config,device,track=='sg')
        if result1!=result2 or digest1!=g4.state_digest(context):
            raise RuntimeError('Checkpoint resume changes parameter updates')
        start=time.perf_counter()
        evaluation=evaluate(context,selected,select_targets,device,fm['files']['val_select']['metadata'])
        val_seconds=time.perf_counter()-start
        results[track]={'initial_digest':initial,'train_128_seconds':train_seconds,'validation_512_seconds':val_seconds,
                        'checkpoint_resume_exact':True,'cuda_peak_bytes':torch.cuda.max_memory_allocated(),
                        'cache':dict(cache.stats),'metrics_not_scientific_evidence':g4.compact(evaluation)}
        g4.write_json(root/f'{track}_report.json',results[track])
        del context,optimizer,features,selected,cache,cp,evaluation
        gc.collect()
        torch.cuda.empty_cache()
    assert results['sg']['initial_digest']==results['e2e']['initial_digest']
    elapsed=time.time()-manifest['created_at']
    steps=64*(4096//4)
    projection=elapsed+3*sum(v['train_128_seconds']/128*steps+v['validation_512_seconds']*33 for v in results.values())
    # 为checkpoint、整体验证和最终基线预留额外15%，而非把短测当上限保证。
    projection_with_reserve=elapsed+(projection-elapsed)*1.15
    disk=sum(path.stat().st_size for path in run.rglob('*') if path.is_file())
    report={'status':'PASS' if projection_with_reserve<=config['wall_limit_seconds'] and disk<config['new_disk_limit_gib']*2**30 else 'STOP_RESOURCE',
            'tracks':results,'elapsed_preparation_seconds':elapsed,'projected_total_seconds':projection,
            'projected_with_reserve_seconds':projection_with_reserve,'current_disk_bytes':disk,
            'test_executed':False,'pilot_checkpoints_must_not_initialize_formal_training':True}
    g4.write_json(run/'pilot_report.json',report)
    print({k:report[k] for k in ('status','projected_total_seconds','projected_with_reserve_seconds','current_disk_bytes')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    main(p.parse_args().run)
