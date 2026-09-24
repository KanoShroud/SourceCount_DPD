"""Source-aligned descriptive decomposition and evidence-linked P2-D1 figures."""
import gc

import numpy as np
import torch

from 统一模型代码.gates.g6.g6_p2_runtime import SEEDS, write


def analyze(all_rows):
    result={'transitions':{},'interventions':{},'cases':[]}
    candidates=[]
    for seed in SEEDS:
        b=all_rows[f'{seed}_b']['full']; c=all_rows[f'{seed}_c']['full']
        counts={k:0 for k in ('common','b_missing','c_missing','both_missing','gain100','loss100',
            'gain10','loss10','loss10_to30','tail500_repaired100','tail500_created',
            'loss10_same_grid','loss10_changed_grid','loss10_c_grid_within10','slot_number_changed')}
        records=[]
        for rb,rc in zip(b,c):
            assert rb['raw_index']==rc['raw_index'] and rb['truth']==rc['truth']
            for t in range(rc['true_count']):
                pb,pc=rb['sources'].get(str(t)),rc['sources'].get(str(t))
                if pb is None or pc is None:
                    name='both_missing' if pb is None and pc is None else ('b_missing' if pb is None else 'c_missing')
                    counts[name]+=1; continue
                eb,ec=pb['error'],pc['error']; same=pb['flat']==pc['flat']
                counts['common']+=1; counts['slot_number_changed']+=int(pb['slot']!=pc['slot'])
                for name,flag in [('gain100',eb>100 and ec<=100),('loss100',eb<=100 and ec>100),
                    ('gain10',eb>10 and ec<=10),('loss10',eb<=10 and ec>10),
                    ('loss10_to30',eb<=10<ec<=30),('tail500_repaired100',eb>500 and ec<=100),
                    ('tail500_created',eb<=100 and ec>500)]:
                    counts[name]+=int(flag)
                if eb<=10<ec:
                    counts['loss10_same_grid' if same else 'loss10_changed_grid']+=1
                    counts['loss10_c_grid_within10']+=int(pc['grid_error']<=10)
                off=next(x for x in rc['vs_off'] if x['source']==t)
                constant=next(x for x in rc['vs_constant'] if x['source']==t)
                record=dict(seed=seed,index=rc['index'],raw_index=rc['raw_index'],source=t,
                    error_b=eb,error_c=ec,grid_b=pb['grid'],grid_c=pc['grid'],
                    grid_error_b=pb['grid_error'],grid_error_c=pc['grid_error'],
                    offset_b=pb['offset'],offset_c=pc['offset'],slot_b=pb['slot'],slot_c=pc['slot'],
                    c_off_same_slot=off,c_constant_same_slot=constant)
                records.append(record)
                categories=[]
                if eb<=10<ec<=30:
                    categories.append(('fine_same_grid' if same else 'fine_changed_grid',ec-eb))
                if eb>500 and ec<=100:
                    categories.append(('tail_repair',eb-ec))
                if ec<=10 and constant['other_error']>30:
                    categories.append(('constant_new_error',constant['other_error']-ec))
                for name,score in categories:
                    candidates.append(dict(record,category=name,selection_score=score))
        result['transitions'][str(seed)]={'counts':counts,'records':records}
    for key,modes in all_rows.items():
        result['interventions'][key]={}
        for mode in ('off','constant'):
            rows=[x for r in modes['full'] for x in r[f'vs_{mode}']]
            losses=[x for x in rows if x['other_error']<=10<x['native_error']]
            changed=[x for x in rows if x['grid_move']>0]
            result['interventions'][key][mode]=dict(native_matched=len(rows),
                native_loses10_vs_other=len(losses),native_gains10_vs_other=sum(x['native_error']<=10<x['other_error'] for x in rows),
                losses_with_local_order_reversal=sum(x['local_order_reversed'] for x in losses),
                losses_with_neighbor_grid_move=sum(0<x['grid_move']<=30 for x in losses),
                changed_grids=len(changed),changed_other_boundary=sum(x['other_boundary'] for x in changed),
                changed_native_boundary=sum(x['native_boundary'] for x in changed),
                other_boundary_total=sum(x['other_boundary'] for x in rows))
    for name in ('fine_same_grid','fine_changed_grid','tail_repair','constant_new_error'):
        available=[x for x in candidates if x['category']==name]
        if available:
            result['cases'].append(sorted(available,key=lambda x:(-x['selection_score'],x['seed'],x['index'],x['source']))[0])
        else:
            result['cases'].append({'category':name,'status':'NO_CASE'})
    return result


def make_figures(runtime,bundle,training,all_rows,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from 统一模型代码.gates.g6.g6_p2_diagnose import load_model, representations
    from 统一模型代码.gates.g6.g6_p1_speed import physical_batches
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
                         'axes.unicode_minus':False,'font.size':10})
    folder=runtime.out/'figures'; folder.mkdir()
    manifest=[]
    names={'fine_same_grid':'同一网格点的细精度损失','fine_changed_grid':'邻近网格跳动的细精度损失',
           'tail_repair':'大错误修复','constant_new_error':'分块常数引入的新错误'}
    for number,case in enumerate(cases,1):
        if case.get('status')=='NO_CASE':
            manifest.append(case); continue
        seed,i,t=case['seed'],case['index'],case['source']; cached={}
        # Recover exactly the original batch context, rather than switch batch size.
        ids=torch.arange(i//4*4,i//4*4+4); local=i%4
        iterator=physical_batches(runtime,bundle[0],[ids],'val_select',True)
        _,batch,stats=next(iterator); iterator.close(); runtime.consumed(bundle[3],ids)
        for arm in ('b','c'):
            context,head,_=load_model(runtime,seed,arm,training)
            base,_,res=representations(context,head,batch,ids,arm,runtime.physics,stats)
            cached[arm]={'base':base[3][local].cpu().numpy(),'offset':base[4][local].cpu().numpy(),
                         **{k:v[local].cpu().numpy() for k,v in res.items()}}
            del context,head,base,res
            gc.collect(); torch.cuda.empty_cache()
        truth=np.asarray(all_rows[f'{seed}_c']['full'][i]['truth']); target=truth[t]
        ix,iy=np.round((target+2000)/10).astype(int)
        x0,x1=max(0,ix-10),min(400,ix+10); y0,y1=max(0,iy-10),min(400,iy+10)
        extent=[x0*10-2000-5,x1*10-2000+5,y0*10-2000-5,y1*10-2000+5]
        fig,axes=plt.subplots(2,4,figsize=(18,9),constrained_layout=True)
        data={}
        for row,arm in enumerate(('b','c')):
            native=all_rows[f'{seed}_{arm}']['full'][i]
            q=native['sources'][str(t)]['slot']; v=cached[arm]
            base=v['base'][q]; full=base+v['full'][q]
            data[arm]={'slot':q,'base':base,'residual':v['full'][q],
                       'constant':v['constant'][q],'offset':v['offset'][q]}
            arrays=[full,base,v['full'][q],full]
            titles=['全场景：原样最终热图','局部：基础热图','局部：双线性物理修正','局部：最终热图与三种位置']
            local_base=base[y0:y1+1,x0:x1+1]; local_full=full[y0:y1+1,x0:x1+1]
            lo=min(local_base.min(),local_full.min()); hi=max(local_base.max(),local_full.max())
            for col,(array,title) in enumerate(zip(arrays,titles)):
                ax=axes[row,col]
                values=array if col==0 else array[y0:y1+1,x0:x1+1]
                kwargs={'cmap':'viridis'}
                if col in (1,3): kwargs.update(vmin=lo,vmax=hi)
                if col==2:
                    vmax=max(float(np.abs(values).max()),1e-8)
                    kwargs={'cmap':'coolwarm','vmin':-vmax,'vmax':vmax}
                im=ax.imshow(values,origin='lower',extent=[-2005,2005,-2005,2005] if col==0 else extent,**kwargs)
                fig.colorbar(im,ax=ax,shrink=.7,label='logit')
                ax.scatter(truth[:,0],truth[:,1],marker='*',c='white',edgecolors='black',s=100,zorder=6,label='真实源')
                ax.set_title(f'{arm.upper()} — {title}'); ax.set_xlabel('x / m'); ax.set_ylabel('y / m')
                ax.set_aspect('equal')
                if col==0:
                    pred=np.asarray(native['predicted_positions_m']).reshape(-1,2)
                    ax.scatter(pred[:,0],pred[:,1],marker='x',c='#ff9900',s=60,label='原样预测')
                else:
                    ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:])
                if col==3:
                    text=[]
                    for mode,color,label in [('full','#ff9900','原样'),('off','#00ccdd','关闭'),('constant','#ff44cc','常数')]:
                        p=all_rows[f'{seed}_{arm}'][mode][i]['selected'][str(q)]
                        grid=np.asarray(p['grid']); pos=np.asarray(p['position'])
                        ax.scatter(*grid,marker='s',s=35,facecolors='none',edgecolors=color,zorder=8)
                        ax.scatter(*pos,marker='x',s=55,c=color,zorder=9,label=label)
                        ax.annotate('',xy=pos,xytext=grid,arrowprops=dict(arrowstyle='->',color=color,lw=1.5))
                        text.append(f'{label}: {np.linalg.norm(pos-target):.1f}m')
                    ax.set_title(f'{arm.upper()} — '+', '.join(text),fontsize=9)
                    ax.legend(fontsize=8,loc='upper right')
        fig.suptitle(f'{names[case["category"]]} | seed={seed}, val_select={i}, raw={case["raw_index"]}, source={t}\n'
                     '白星=真实位置；空方框=10m网格点；箭头=offset；叉号=最终位置。同模型各模式追踪同一槽位。',fontsize=12)
        stem=f'{number:02d}_{case["category"]}_{seed}_{i}'
        fig.savefig(folder/f'{stem}.png',dpi=160)
        fig.savefig(folder/f'{stem}.svg')
        plt.close(fig)
        # Array archive plus selection metadata makes the plotted maps reproducible.
        np.savez_compressed(folder/f'{stem}.npz',truth=truth,
            **{f'{arm}_{key}':value for arm,d in data.items() for key,value in d.items()})
        write(folder/f'{stem}.json',case)
        manifest.append(dict(case,png=str(folder/f'{stem}.png'),svg=str(folder/f'{stem}.svg')))
        del batch,stats,cached
        gc.collect(); torch.cuda.empty_cache()
    return manifest
