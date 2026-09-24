"""G6-P1 frozen-checkpoint diagnosis on val_select only; no optimizer updates."""
import gc
import io
import os
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.e2e_g5_model import forward as base_forward, band_values
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run
from 统一模型代码.gates.g5.r2.g5_r2_model import candidate_batch, decode_r2
from 统一模型代码.gates.g6.coherent_dpd import sample_map
from 统一模型代码.gates.g6.g6_p1_model import identity_hits
from 统一模型代码.gates.g6.g6_p1_runtime import BASE, SEEDS, Runtime, RunLock, read, write, identity, safe_print, g4
from 统一模型代码.gates.g6.g6_p1_speed import physical_maps, physical_batches
from 统一模型代码.gates.g6.g6_p1_train import restore, loss_for, summarize


def band_match(logits, bands, ignore, k):
    cost = np.empty((3,k))
    for q in range(3):
        for t in range(k):
            valid = ignore[t] < .5
            cost[q,t] = float(F.binary_cross_entropy_with_logits(logits[q,valid],bands[t,valid]))
    a,b = linear_sum_assignment(cost)
    return dict(zip(a.tolist(),b.tolist()))


def align_logits(early, late):
    """Match predicted slots using probability L1 only, without labels or positions."""
    aligned = torch.empty_like(early)
    permutations=[]
    for i,(a,b) in enumerate(zip(early,late)):
        cost = (b.sigmoid()[:,None]-a.sigmoid()[None]).abs().mean(-1).cpu().numpy()
        rows,cols=linear_sum_assignment(cost)
        aligned[i,rows]=a[cols]
        permutations.append(cols.tolist())
    return aligned,permutations


def metric_rows(output, residual, maps, targets, ids, metadata, matched):
    logits,heat,offset=output[1],output[3],output[4]
    candidates=candidate_batch(heat,offset)
    logits=logits.detach().cpu()
    rows=[]
    for j,i in enumerate(ids.tolist()):
        k=int(targets.counts[i])
        truth=targets.positions[i,:k].numpy()
        bands,ignore=targets.band[i],targets.ignore[i]
        mapping=band_match(logits[j],bands,ignore,k)
        decoded=decode_r2(logits[j].numpy(),candidates[j],None)
        row=R1Run.metric(None,truth,decoded['joint'],logits[j].numpy(),decoded['active'],
                        bands.numpy(),ignore.numpy(),metadata[i])
        row['band_only_f1'],row['band_only_iou']=band_values(logits[j],targets,i,mapping)
        row.update(identity_hits(truth,decoded['joint'],logits[j].numpy(),decoded['active'],
                                 bands.numpy(),ignore.numpy()))
        row.update(index=i,logits=logits[j].tolist(),mapping=mapping,
                   loss_mapping=matched[j],residual_mean_square=float(residual[j].square().mean()),
                   residual_spatial_std_mean=float(residual[j].flatten(1).std(1,unbiased=False).mean()))
        semantics=[]
        responses=sample_map(maps[j],truth).cpu().numpy() if k else None
        for q,t in mapping.items():
            valid=ignore[t]<.5
            y=bands[t]> .5
            p=logits[j,q].sigmoid()
            pos,neg=valid & y,valid & ~y
            v=p.clamp(1e-7,1-1e-7)
            semantics.append(dict(slot=q,source=t,positive_sum=float(p[pos].sum()),positive_n=int(pos.sum()),
                negative_sum=float(p[neg].sum()),negative_n=int(neg.sum()),
                false_positive=int(((p>=.5)&neg).sum()),false_negative=int(((p<.5)&pos).sum()),
                entropy=float((-(v[valid]*v[valid].log()+(1-v[valid])*(1-v[valid]).log())).mean()),
                soft_mass=float(p[valid].sum()),true_mass=int(pos.sum()),
                map_own=float(responses[q,t]),
                map_margin=float(responses[q,t]-np.max(np.delete(responses[q],t))) if k>1 else None,
                heat_own=float(heat[j,q,int(round((truth[t,1]+2000)/10)),int(round((truth[t,0]+2000)/10))].sigmoid())))
        row['semantics']=semantics
        rows.append(row)
    return rows


def summarize_extra(rows,losses):
    r=summarize(rows)
    src=[s for x in rows for s in x['semantics']]
    pos=sum(s['positive_n'] for s in src); neg=sum(s['negative_n'] for s in src)
    margins=[s['map_margin'] for s in src if s['map_margin'] is not None]
    r.update(p_in=sum(s['positive_sum'] for s in src)/pos,
             p_out=sum(s['negative_sum'] for s in src)/neg,
             false_positive_rate=sum(s['false_positive'] for s in src)/neg,
             false_negative_rate=sum(s['false_negative'] for s in src)/pos,
             entropy=float(np.mean([s['entropy'] for s in src])),
             soft_mass=float(np.mean([s['soft_mass'] for s in src])),
             map_identity_margin=float(np.mean(margins)),
             map_own_gt_other_rate=float(np.mean(np.array(margins)>0)),
             map_own=float(np.mean([s['map_own'] for s in src])),
             heat_own=float(np.mean([s['heat_own'] for s in src])),
             matching_disagreement_rate=sum(x['mapping']!=x['loss_mapping'] for x in rows if x['true_count'])/
                                       sum(x['true_count']>0 for x in rows),
             loss={k:float(np.mean([x[k] for x in losses])) for k in losses[0]})
    return r


def load_frozen(runtime, seed, arm, epoch, inputs):
    paths=list((BASE/f'run/training/{seed}/{arm}/checkpoints').glob(f'epoch{epoch:03d}_*/state.pt'))
    if len(paths)!=1:
        raise RuntimeError(f'Ambiguous checkpoint: {seed}/{arm}/{epoch}')
    path=paths[0].resolve(strict=True)
    row=read(path.with_suffix('.identity.json'))
    if path != __import__('pathlib').Path(row['path']).resolve(strict=True):
        raise RuntimeError('Checkpoint sidecar path mismatch')
    saved=torch.load(io.BytesIO(verified_read(row,runtime.out/'anomalies')),map_location='cpu',weights_only=False)
    if saved['history'][-1]['epoch'] != epoch:
        raise RuntimeError('Wrong epoch')
    context,head,optimizer,params=runtime.context(seed,arm)
    del optimizer,params
    restore(context,head,saved['state'])
    g4.set_mode(context,training=False); head.eval()
    inputs.append(row)
    return context,head


def gradients(runtime,context,head,bundle):
    targets=bundle[1]
    # Fixed first eight samples per nonzero K, not selected by outcomes.
    ids=torch.tensor([int(x) for k in (1,2,3) for x in torch.nonzero(targets.counts==k).flatten()[:8]])
    iterator=physical_batches(runtime,bundle[0],ids.split(4),'val_select',True)
    records=[]
    try:
        for idx,batch,stats in iterator:
            runtime.guard(); runtime.consumed(bundle[3],idx)
            with torch.no_grad():
                output=base_forward(context,batch,idx,torch.device('cuda:0'),stop_gradient=True)
            logits=output[1].detach().requires_grad_(True)
            maps=physical_maps(runtime.physics,logits.sigmoid(),stats)
            residual=head(maps)
            total,parts,mapping=loss_for(runtime,(output[0],logits,output[2],output[3]+residual,output[4]),targets,idx)
            gl=torch.autograd.grad(parts['heatmap'],logits,retain_graph=True)[0]
            ga=torch.autograd.grad(parts['band']+parts['exist'],logits)[0]
            n1,n2=float(gl.norm()),float(ga.norm())
            dot=float((gl*ga).sum())
            records.append(dict(ids=idx.tolist(),gradient_domain='band_logits; eval mode; no update',
                physical_heat_norm=n1,aux_norm=n2,ratio=n1/max(n2,1e-30),
                cosine=dot/max(n1*n2,1e-30),aux_change_under_negative_physical_gradient=-dot))
    finally:
        iterator.close()
    return records


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    with RunLock():
        out=BASE/f'diagnosis/{time.strftime("%Y%m%d_%H%M%S")}'
        out.mkdir(parents=True,exist_ok=False)
        runtime=Runtime(out,deadline=time.time()+1800)
        start=time.perf_counter()
        inputs=[]
        report=dict(status='RUNNING',split='val_select',samples=512,epochs=[4,20],seeds=list(SEEDS),
                    training_executed=False,test_executed=False,val_compare_executed=False,
                    tracks={},interventions={},gradient_probes={},checkpoint_inputs=inputs)
        write(out/'plan.json',dict(report,intervention='C2 epoch20 fixed body/head; swap only physical probabilities, only decoded logits, or both, from epoch4; align slots by predicted probability L1',
            scope='Descriptive diagnosis, not checkpoint selection or a new model performance claim',
            code=identity(__file__),contract=identity(BASE/'contract.json'),torch=torch.__version__))
        try:
            audit=read(BASE/'run/evaluation/final_audit_report.json')
            assert audit['status']=='PASS' and audit['six_tracks_complete']
            verified_read(audit['contract'],out/'anomalies')
            runtime.preflight()
            bundle=runtime.features('val_select')
            assert len(bundle[1].counts)==512
            for seed in SEEDS:
                early=None
                for arm,epoch in [('c1',4),('c1',20),('c2',4),('c2',20)]:
                    key=f'{seed}_{arm}_e{epoch}'
                    safe_print(f'诊断：{key}，固定512条val_select。')
                    context,head=load_frozen(runtime,seed,arm,epoch,inputs)
                    native_rows=[]; losses=[]; swaps={k:[] for k in ('physical_early','decode_early','both_early')}
                    swap_losses={k:[] for k in swaps}
                    is_late=arm=='c2' and epoch==20
                    order=list(torch.arange(512).split(4))
                    iterator=physical_batches(runtime,bundle[0],order,'val_select',True)
                    logit_cache=[]; changed=0
                    try:
                        with torch.no_grad():
                            for ids,batch,stats in iterator:
                                runtime.guard(); runtime.consumed(bundle[3],ids)
                                base=base_forward(context,batch,ids,torch.device('cuda:0'),stop_gradient=True)
                                maps=physical_maps(runtime.physics,base[1].sigmoid(),stats)
                                residual=head(maps)
                                output=(*base[:3],base[3]+residual,base[4])
                                _,parts,matched=loss_for(runtime,output,bundle[1],ids)
                                native_rows.extend(metric_rows(output,residual,maps,bundle[1],ids,bundle[2],matched))
                                losses.append({k:float(v) for k,v in parts.items()})
                                logit_cache.append(base[1].cpu())
                                if is_late:
                                    donor,permutations=align_logits(early[ids].cuda(),base[1])
                                    changed+=sum(p != [0,1,2] for p in permutations)
                                    old_maps=physical_maps(runtime.physics,donor.sigmoid(),stats)
                                    old_residual=head(old_maps)
                                    for name in swaps:
                                        sm=old_maps if name!='decode_early' else maps
                                        sr=old_residual if name!='decode_early' else residual
                                        sl=donor if name!='physical_early' else base[1]
                                        so=(base[0],sl,base[2],base[3]+sr,base[4])
                                        _,sp,match=loss_for(runtime,so,bundle[1],ids)
                                        swaps[name].extend(metric_rows(so,sr,sm,bundle[1],ids,bundle[2],match))
                                        swap_losses[name].append({k:float(v) for k,v in sp.items()})
                    finally:
                        iterator.close()
                    if arm=='c2' and epoch==4:
                        early=torch.cat(logit_cache)
                    result=summarize_extra(native_rows,losses)
                    history=read(BASE/f'run/training/{seed}/{arm}/history.json')
                    recorded=next(x['validation'] for x in history if x['epoch']==epoch)
                    for metric in ('joint_recall100_f1_08','gospa_m','band_only_f1','recall100'):
                        np.testing.assert_allclose(result[metric],recorded[metric],rtol=1e-7,atol=1e-7)
                    result['history_replay_pass']=True
                    report['tracks'][key]=result
                    write(out/f'{key}_samples.json',native_rows)
                    if is_late:
                        report['interventions'][str(seed)]=dict(
                            slot_permutation_changed_scenes=changed,
                            **{k:summarize_extra(v,swap_losses[k]) for k,v in swaps.items()})
                        for name,rows in swaps.items():
                            write(out/f'{seed}_{name}_samples.json',rows)
                    if arm=='c2':
                        report['gradient_probes'][key]=gradients(runtime,context,head,bundle)
                    del context,head,base,output,residual,maps,batch,stats,logit_cache
                    gc.collect(); torch.cuda.empty_cache()
                    write(out/'partial.json',report)
            runtime.postcheck('diagnosis')
            for row in inputs:
                verified_read(row,out/'anomalies')
            report.update(status='PASS',seconds=time.perf_counter()-start,peak_ram_percent=runtime.peak_ram,
                          peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
            write(out/'report.json',report)
            write(out/'final_audit.json',dict(status='PASS',report=identity(out/'report.json'),
                input_audit=identity(out/'diagnosis_input_audit.json'),code=identity(__file__),
                original_contract=identity(BASE/'contract.json'),checkpoint_count=len(inputs),
                training_executed=False,test_executed=False))
            safe_print(f'诊断完成：{out}')
        except BaseException:
            write(out/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
            raise


if __name__=='__main__':
    main()
