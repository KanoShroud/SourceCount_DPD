"""Approved P2-D1: fixed checkpoints, three residual modes, val_select only."""
import gc
import io
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.e2e_g5_model import forward as base_forward, band_values
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run
from 统一模型代码.gates.g5.r2.g5_r2_model import candidate_batch, decode_r2
from 统一模型代码.gates.g6.g6_p1_diagnose import band_match
from 统一模型代码.gates.g6.g6_p1_model import identity_hits
from 统一模型代码.gates.g6.g6_p1_speed import physical_batches, physical_maps
from 统一模型代码.gates.g6.g6_p2_model import subband_counts
from 统一模型代码.gates.g6.g6_p2_runtime import (
    BASE, SEEDS, Runtime, RunLock, setup_environment, read, write, identity, safe_print, g4)
from 统一模型代码.gates.g6.g6_p2_train import restore, summarize

MODES = ('full','off','constant')


def coarse_constant(small):
    # Exact physical centers: fine i -> nearest coarse i/5, endpoints unchanged.
    index = (torch.arange(401,device=small.device)+2)//5
    return small[...,index[:,None],index[None,:]]


@torch.no_grad()
def representations(context,head,batch,ids,arm,physics,stats):
    base = base_forward(context,batch,ids,torch.device('cuda:0'),stop_gradient=True)
    weights = head.weights(base[0],base[1],arm)
    maps = physical_maps(physics,weights,stats)
    small = head.physical.net(maps.float().reshape(-1,1,81,81))
    residual = F.interpolate(small,size=(401,401),mode='bilinear',align_corners=True).reshape(-1,3,401,401)
    const = coarse_constant(small.reshape(-1,3,81,81))
    return base,weights,{'full':residual,'off':torch.zeros_like(residual),'constant':const}


def selected_points(decoded,heat,offset):
    result = {}
    for j,q in enumerate(decoded['active']):
        if len(decoded['active'])>=2 and not decoded['fallback']:
            flat = decoded['candidates'][j]['flat_indices'][decoded['selected_ranks'][j]-1]
        else:
            flat = int(np.argmax(heat[q]))
        iy,ix = divmod(flat,401)
        grid = np.array([ix*10-2000,iy*10-2000],dtype=float)
        delta = np.clip(offset[q,:,iy,ix],-1,1)*10
        position = np.asarray(decoded['joint'][j])
        np.testing.assert_allclose(position,grid+delta,atol=2e-4,rtol=0)
        # Adjacent pixels straddling a nearest-center step: residues 2 and 3.
        edge = any(0<v<400 and v%5 in (2,3) for v in (ix,iy))
        result[str(q)] = dict(slot=q,flat=int(flat),grid=grid.tolist(),offset=delta.tolist(),
                             position=position.tolist(),coarse_boundary=edge)
    return result


def rows_for(base,weights,residual,targets,ids,metadata):
    heat = base[3]+residual
    candidates = candidate_batch(heat,base[4])
    logits,weight = base[1].cpu(),weights.cpu()
    # Match candidate_batch's sigmoid ordering including floating-point ties.
    probability,offset = heat.sigmoid().cpu().numpy(),base[4].cpu().numpy()
    res = residual.cpu()
    result = []
    for j,i in enumerate(ids.tolist()):
        k = int(targets.counts[i]); truth=targets.positions[i,:k].numpy()
        band,ignore = targets.band[i],targets.ignore[i]
        z=logits[j].numpy(); dec=decode_r2(z,candidates[j],None)
        row=R1Run.metric(None,truth,dec['joint'],z,dec['active'],band.numpy(),ignore.numpy(),metadata[i])
        mapping=band_match(logits[j],band,ignore,k)
        row['band_only_f1'],row['band_only_iou']=band_values(logits[j],targets,i,mapping)
        row.update(identity_hits(truth,dec['joint'],z,dec['active'],band.numpy(),ignore.numpy()))
        row.update(subband_counts(logits[j],weight[j],band,ignore,mapping))
        selected=selected_points(dec,probability[j],offset[j])
        sources={}
        for t,p in row['spatial_pairs']:
            point=selected[str(dec['active'][p])]
            sources[str(t)]={**point,'error':float(np.linalg.norm(truth[t]-point['position'])),
                            'grid_error':float(np.linalg.norm(truth[t]-point['grid']))}
        row.update(index=i,truth=truth.tolist(),selected=selected,sources=sources,
            active=dec['active'],fallback=dec['fallback'],
            residual_mean_square=float(res[j].square().mean()),
            residual_spatial_std_mean=float(res[j].flatten(1).std(1,unbiased=False).mean()))
        result.append(row)
    return result


def local_changes(native,other,base_heat,residual):
    changes=[]
    for t,p in native['sources'].items():
        q=p['slot']; alt=other['selected'][str(q)]
        truth=np.asarray(native['truth'][int(t)])
        distance=float(np.linalg.norm(np.asarray(p['grid'])-alt['grid']))
        f0,f1=alt['flat'],p['flat']
        dh=float(base_heat[q].ravel()[f1]-base_heat[q].ravel()[f0])
        dr=float(residual[q].ravel()[f1]-residual[q].ravel()[f0])
        changes.append(dict(source=int(t),slot=q,grid_move=distance,
            native_error=p['error'],other_error=float(np.linalg.norm(truth-alt['position'])),
            base_at_native_minus_other=dh,residual_at_native_minus_other=dr,
            local_order_reversed=bool(0<distance<=30 and dh<0 and dh+dr>0),
            native_boundary=p['coarse_boundary'],
            other_boundary=alt['coarse_boundary']))
    return changes


def load_model(runtime,seed,arm,training):
    best=training['seeds'][str(seed)][arm]['best']
    row=best['checkpoint']; path=Path(row['path']).resolve(strict=True)
    if not path.is_relative_to((BASE/f'run/training/{seed}/{arm}').resolve()):
        raise RuntimeError('Checkpoint outside frozen track')
    cp=torch.load(io.BytesIO(verified_read(row,runtime.out/'anomalies')),map_location='cpu',weights_only=False)
    context,head,optimizer,params=runtime.context(seed,arm)
    del optimizer,params
    restore(context,head,cp['state'])
    g4.set_mode(context,training=False); head.eval()
    return context,head,best


def main():
    setup_environment(); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    from 统一模型代码.gates.g6.g6_p2_diag_report import analyze, make_figures
    with RunLock():
        out=BASE/f'diagnosis/{time.strftime("%Y%m%d_%H%M%S")}'
        out.mkdir(parents=True,exist_ok=False)
        runtime=Runtime(out,deadline=time.time()+3600); start=time.perf_counter()
        report=dict(status='RUNNING',split='val_select',samples=512,tracks={},
            training_executed=False,test_executed=False,val_compare_executed=False)
        inputs=[]
        try:
            audit=read(BASE/'run/evaluation/final_audit_report.json')
            assert audit['status']=='PASS' and audit['six_tracks_complete']
            comp=next(r for r in audit['outputs'] if Path(r['path']).name=='comparison_report.json')
            training=json.loads(verified_read(comp,out/'anomalies'))['training']
            verified_read(audit['contract'],out/'anomalies')
            code=[identity(Path(__file__)),identity(Path(__file__).with_name('g6_p2_diag_report.py')),
                  identity(Path(__file__).with_name('test_g6_p2_diag.py'))]
            write(out/'plan.json',dict(report,modes=MODES,seeds=SEEDS,arms=['b','c'],
                code=code,contract=audit['contract'],torch=torch.__version__,
                coordinate_rule='nearest coarse center floor((fine_index+2)/5)',
                case_rule='maximum error change per preregistered category; seed/index/source tie-break',
                attribution='B/C by GT spatial match; intervention by native matched slot; no causal query claim'))
            runtime.preflight(); runtime.audit_raw()
            bundle=runtime.features('val_select'); all_rows={}
            assert len(bundle[1].counts)==512
            for seed in SEEDS:
                for arm in ('b','c'):
                    key=f'{seed}_{arm}'; safe_print(f'阶段：{key} 固定512条，三种物理修正对照。')
                    context,head,best=load_model(runtime,seed,arm,training)
                    inputs.append(best['checkpoint']); rows={mode:[] for mode in MODES}
                    iterator=physical_batches(runtime,bundle[0],torch.arange(512).split(4),'val_select',True)
                    try:
                        for ids,batch,stats in iterator:
                            runtime.guard(); runtime.consumed(bundle[3],ids)
                            base,weights,residuals=representations(context,head,batch,ids,arm,runtime.physics,stats)
                            current={m:rows_for(base,weights,r,bundle[1],ids,bundle[2]) for m,r in residuals.items()}
                            bh,rr=base[3].cpu().numpy(),residuals['full'].cpu().numpy()
                            for j in range(len(ids)):
                                native=current['full'][j]
                                for mode in ('off','constant'):
                                    assert native['active']==current[mode][j]['active']
                                    native[f'vs_{mode}']=local_changes(native,current[mode][j],bh[j],rr[j])
                            for mode in MODES:
                                rows[mode].extend(current[mode])
                    finally:
                        iterator.close()
                    metrics={m:summarize(v) for m,v in rows.items()}
                    for k,v in best['metrics'].items():
                        if k in metrics['full']:
                            np.testing.assert_allclose(metrics['full'][k],v,rtol=0,atol=1e-12,err_msg=f'{key}/{k}')
                    report['tracks'][key]=dict(epoch=best['epoch'],metrics=metrics,replay_matches=True)
                    write(out/f'{key}_samples.json',rows); all_rows[key]=rows
                    runtime.postcheck(key)
                    del context,head,base,weights,residuals,batch,stats,bh,rr
                    gc.collect(); torch.cuda.empty_cache()
            report['analysis']=analyze(all_rows)
            write(out/'report.json',report)
            safe_print('阶段：全量对照完成，按固定规则导出典型样本图。')
            report['figures']=make_figures(runtime,bundle,training,all_rows,report['analysis']['cases'])
            runtime.postcheck('figures'); runtime.audit_raw()
            for row in inputs+code+[audit['contract']]:
                verified_read(row,out/'anomalies')
            report.update(status='PASS',seconds=time.perf_counter()-start,checkpoint_inputs=inputs,
                          peak_ram_percent=runtime.peak_ram,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
            write(out/'report.json',report)
            files=[identity(p) for p in sorted(out.rglob('*')) if p.is_file()]
            write(out/'final_audit.json',dict(status='PASS',files=files,contract=audit['contract'],
                training_executed=False,test_executed=False,val_compare_executed=False))
            safe_print(f'P2-D1完成：{out}')
        except BaseException:
            write(out/'failure.json',dict(status='FAILED',traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
            raise


if __name__=='__main__':
    main()
