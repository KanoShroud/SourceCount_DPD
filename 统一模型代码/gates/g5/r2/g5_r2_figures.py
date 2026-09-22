"""正式运行后从保存的逐样本结果选择可追溯案例；无额外模型推理。"""
import numpy as np
from 统一模型代码.gates.g5.r2.g5_r2_runtime import write


def plot_cases(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out/'figures'
    folder.mkdir(exist_ok=True)
    manifest = []
    for name,predicate in (
            ('association_repair',lambda a,b,c:b['joint_tp']>a['joint_tp']),
            ('feedback_repair',lambda a,b,c:c['joint_tp']>b['joint_tp']),
            ('remaining_binding_error',lambda a,b,c:c['tp_at_100m']>c['joint_tp'])):
        selected = None
        for seed in sorted({s for s,a in results}):
            for a,b,c in zip(results[seed,'c0'],results[seed,'c1'],results[seed,'c2']):
                if predicate(a,b,c):
                    selected = seed,(a,b,c)
                    break
            if selected:
                break
        if selected is None:
            manifest.append({'category':name,'status':'NO_CASE'})
            continue
        seed,rows = selected
        fig,axes = plt.subplots(1,3,figsize=(13,4.5),constrained_layout=True)
        for ax,label,row in zip(axes,('C0: SG','C1: SG + association','C2: E2E + association'),rows):
            truth = np.asarray(row['truth']).reshape(-1,2)
            if len(truth):
                ax.scatter(truth[:,0],truth[:,1],marker='*',s=120,c='black',label='Truth')
            for q,point in zip(row['decode']['active'],row['predicted_positions_m']):
                ax.scatter(*point,marker='x',s=75,color=f'C{q}',label=f'Slot {q+1}')
            ax.set(xlabel='x (m)',ylabel='y (m)',title=f'{label}\nTP100={row["tp_at_100m"]}; joint={row["joint_tp"]}')
            ax.set_aspect('equal',adjustable='datalim')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        fig.suptitle(f'{name}: seed {seed}, raw_index {rows[0]["raw_index"]}')
        path = folder/f'{name}.png'
        fig.savefig(path,dpi=200)
        plt.close(fig)
        manifest.append({'category':name,'seed':seed,'raw_index':rows[0]['raw_index'],'path':str(path),
                         'selection':'first qualifying scene in fixed seed/index order',
                         'source_receipts':[str(out/f'{seed}_{a}/complete.json') for a in ('c0','c1','c2')]})
    write(out/'figure_manifest.json',manifest)
