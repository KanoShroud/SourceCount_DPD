"""G8-R6: selected failure diagnostics, not population performance evaluation."""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .sequential_projection import ProjectedScore, allowed, refine, window
from .peak_diagnosis import local_ids
from ..physics import grid, peaks
from ..storage import BASE, Guard, read, write, identity, checked_bytes, load_scene, save_scene

GROUPS=[16,17,18,19,21,23,26,34]


def diagnose(iq, sources, truth_weak, old_position, guard):
    start=time.perf_counter()
    calc=ProjectedScore(iq,guard)
    calc.set_sources(sources)
    pts,shape=grid(10)
    z=calc.evaluate(pts).reshape(shape)
    ids=local_ids(z)
    ids=ids[allowed(pts[ids],sources)]
    chosen=ids[peaks(pts[ids],z.ravel()[ids],5)]
    refined=np.asarray([refine(calc,pts[i],sources) for i in chosen])
    scores=calc.evaluate(refined)
    best=refined[np.argmax(scores)]
    best_score=float(scores.max())
    # Only diagnostic scoring; neither truth nor its window enters the search above.
    true_points=window(truth_weak,20,1)
    true_points=true_points[allowed(true_points,sources)]
    true_scores=calc.evaluate(true_points)
    truth_score,old_score=calc.evaluate([truth_weak,old_position])
    local=int(np.argmax(true_scores))
    error=float(np.linalg.norm(best-truth_weak))
    if error<=100:
        status='dense_recovers'
    elif float(true_scores[local])>best_score*(1+1e-8):
        status='better_correct_region_still_missed'
    else:
        status='wrong_region_scores_higher_or_tied'
    result=dict(position=best.tolist(),error_m=error,recall100=error<=100,status=status,
        true_score=float(truth_score),old_position_score=float(old_score),selected_score=best_score,
        true_window_best_score=float(true_scores[local]),true_window_best_position=true_points[local].tolist(),
        relative_correct_minus_selected=float((true_scores[local]-best_score)/max(abs(best_score),1e-30)),
        old_position=list(old_position),fine_candidates=refined.tolist(),seconds=time.perf_counter()-start)
    return result,z


def plots(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures'
    folder.mkdir()
    chosen=[]
    for status in ('dense_recovers','better_correct_region_still_missed','wrong_region_scores_higher_or_tied'):
        candidates=[r for r in rows if r['A']['status']==status and not r['r5_success']]
        if candidates:
            chosen.append(candidates[0])
    controls=[r for r in rows if r['group']==18]
    if controls:
        chosen.append(controls[0])
    for r in chosen:
        labels=[v for v in ('A','B','C','D') if v in r]
        fig,axes=plt.subplots(1,len(labels),figsize=(5*len(labels),4.5),squeeze=False,constrained_layout=True)
        for ax,label in zip(axes[0],labels):
            z=load_scene(r[label]['map_identity'])['score']
            im=ax.imshow(z/z.max(),origin='lower',extent=[-1000,1000,-1000,1000],cmap='viridis')
            truth=np.asarray(r['truth'])
            ax.scatter(*truth[0],marker='*',c='red',s=70,label='Strong truth')
            ax.scatter(*truth[1],facecolors='none',edgecolors='red',s=70,label='Weak truth')
            ax.scatter(*r[label]['position'],marker='+',c='white',s=70,label='Dense search')
            ax.scatter(*r['old_second'],marker='x',c='orange',s=50,label='R5 second')
            ax.set(title=f"{label}: error={r[label]['error_m']:.1f} m",xlabel='x (m)',ylabel='y (m)')
            ax.legend(fontsize=7)
            fig.colorbar(im,ax=ax,label='Score / panel maximum')
        descriptions=dict(A='mixed / estimated anchor',B='mixed / true anchor',C='weak+noise / projected',D='weak+noise / unprojected')
        fig.suptitle(f"Layout {r['group']}, overlap={r['overlap']}\n"+'; '.join(f'{v}: {descriptions[v]}' for v in labels),fontsize=10)
        stem=folder/f"layout_{r['group']}_{r['overlap']}"
        fig.savefig(str(stem)+'.png',dpi=160)
        fig.savefig(str(stem)+'.svg')
        plt.close(fig)
    write(folder/'selection.json',[dict(group=r['group'],overlap=r['overlap'],status=r['A']['status']) for r in chosen])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--short',action='store_true')
    args=parser.parse_args()
    out=(BASE/args.run_id).resolve()
    if not out.is_relative_to(BASE.resolve()) or out==BASE.resolve():
        raise ValueError('Output root')
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    torch.set_num_threads(4)
    found=[(p.parent,read(p)) for p in BASE.glob('*/final_audit_report.json') if read(p).get('scope')=='G8_R5_matched_model_knownK' and read(p).get('status')=='PASS']
    if len(found)!=1:
        raise RuntimeError('Unique audited R5 required')
    prior,audit=found[0]
    inputs=[identity(prior/'final_audit_report.json')]+audit['outputs']+audit['inputs']+audit['sources']
    for item in inputs:
        checked_bytes(item)
    index_paths={r['path'] for r in audit['inputs'] if Path(r['path']).name=='data_index.json'}
    if len(index_paths)!=1:
        raise RuntimeError('Unique R2 index required')
    index=read(next(iter(index_paths)))
    old=read(prior/'rows.json')
    used=read(prior/'report.json')['cumulative_G8_seconds']
    previous=sum(read(p)['seconds'] for p in BASE.glob('*/report.json') if read(p).get('gate')=='G8-R6')
    guard=Guard(out,min(3600-previous,43200-used-previous))
    sources=audit['sources']+[identity(Path(__file__))]
    cases=[(17,.5),(18,.5)] if args.short else [(g,o) for g in GROUPS for o in (.5,1.)]
    for g,o in cases:
        checked_bytes(index[f'{g}_{o}'])
        inputs.append(index[f'{g}_{o}'])
    guard()
    write(out/'manifest.json',dict(gate='G8-R6',short=args.short,cases=cases,
        group_roles=dict(both_failure=[17,19,21,23],both_success=[18,26],discordant=[16,34]),
        fine_grid_m=10,fine_candidates=5,local_refine=[10,1],truth_window=[20,1],
        abcd_trigger='A dense weak error >100m OR group18 controls',no_training=True,test_read=False,
        prior_G8_seconds=used,prior_R6_seconds=previous,budget_seconds=guard.seconds,inputs=inputs,sources=sources))
    rows=[]
    try:
        print('阶段1：固定8布局的评分/搜索检查；按失败情况触发B/C/D',flush=True)
        for number,(g,o) in enumerate(cases):
            scene=load_scene(index[f'{g}_{o}'])
            baseline=next(r for r in old if r['group']==g and r['overlap']==o)
            truth=np.asarray(scene['metadata']['positions'])
            anchor=baseline['initial']
            second=baseline['methods']['projection']['positions'][1]
            mixed=scene['components'].sum(0)+scene['noise']
            weak=scene['components'][1]+scene['noise']
            row=dict(group=g,overlap=o,truth=truth.tolist(),anchor=anchor,old_second=second,
                r5_success=baseline['methods']['projection_joint']['both_recovered100'],
                r5_second_error=float(np.linalg.norm(np.asarray(second)-truth[1])))
            variants=[('A',mixed,[anchor])]
            for label,iq,anchors in variants:
                result,z=diagnose(iq,anchors,truth[1],second,guard)
                result['map_identity']=save_scene(out/f'maps/{g}_{o}_{label}.npz',dict(score=z,metadata=dict(group=g,overlap=o,variant=label)))
                row[label]=result
                if label=='A' and (not result['recall100'] or g==18):
                    variants.extend([('B',mixed,[truth[0]]),('C',weak,[truth[0]]),('D',weak,[])])
            rows.append(row)
            write(out/'rows.json',rows)
            print(f'诊断 {number+1}/{len(cases)}：布局{g} overlap={o}；累计{(time.perf_counter()-start)/60:.1f}分钟',flush=True)
        print('阶段2：汇总、图片和身份审计',flush=True)
        plots(out,rows)
        for item in inputs+sources+[r[v]['map_identity'] for r in rows for v in ('A','B','C','D') if v in r]:
            checked_bytes(item)
        summary={}
        for v in ('A','B','C','D'):
            data=[r[v] for r in rows if v in r]
            if data:
                summary[v]=dict(cases=len(data),weak_recovered=sum(r['recall100'] for r in data),
                    rmse_m=float(np.sqrt(np.mean([r['error_m']**2 for r in data]))),
                    statuses={s:sum(r['status']==s for r in data) for s in sorted({r['status'] for r in data})})
        seconds=time.perf_counter()-start
        write(out/'report.json',dict(gate='G8-R6',status='SHORT_COMPLETED' if args.short else 'COMPLETED',
            seconds=seconds,cumulative_G8_seconds=used+previous+seconds,summary=summary,
            r5_success=sum(r['r5_success'] for r in rows),no_training=True,test_read=False,
            peak_ram_percent=guard.peak_ram_percent,peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
            scope='selected diagnostic cases, not population success rate'))
        outputs=[identity(p) for p in out.rglob('*') if p.is_file()]
        write(out/'final_audit_report.json',dict(status='PASS',scope='G8_R6_short' if args.short else 'G8_R6_failure_diagnostic',inputs=inputs,sources=sources,outputs=outputs))
        print(f'完成：{out}；{seconds/60:.2f}分钟',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':
    main()
