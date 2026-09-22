"""Fixed positions; only the band/slot-to-position permutation is learned."""
from __future__ import annotations

import gc
import hashlib
import io
import itertools
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run, identity
from 统一模型代码.gates.g5.r1.g5_r1_report import summarize
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import save_rows
from 统一模型代码.gates.g5.r2.g5_r2_model import AssociationHead, decode_r2
from 统一模型代码.gates.g5.r2.g5_r2_runtime import ROOT, SEEDS, Runtime, read, write, safe_print, setup_environment, g4
from 统一模型代码.gates.g5.r3.g5_r3 import Cache, cached_scores

SOURCE = ROOT/'outputs_e2e/unified/e2e_g5_r3/20260922_115824'
BASE = ROOT/'outputs_e2e/unified/e2e_g5_r4'
CONFIG = {'gate':'G5-R4','seeds':list(SEEDS),'epochs':20,'batch':4,'eval_every':2,
          'lr':1e-4,'weight_decay':1e-4,'clip':10.,'embedding_dim':64,'gamma_init':0.,
          'position_selection':'R1_heatmap_only_joint','same_score_tie':'original_permutation_first',
          'loss':'multi_positive_permutation','test_executed':False,'val_compare_executed':False}


class ExplicitMatcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Linear(128,16)
        self.slot=nn.Linear(147,64)
        self.position=nn.Linear(304,64)
        self.gamma=nn.Parameter(torch.zeros(()))


def scores_for(head,sample,device):
    active=sample['active']
    k=len(active)
    local=head.project(sample['local'].to(device)).flatten(1)
    query=sample['query'][active].to(device)
    bands=sample['logits'][active].to(device).sigmoid()
    if isinstance(head,ExplicitMatcher):
        a=F.normalize(head.slot(torch.cat((query,bands),-1)),dim=-1)
        b=F.normalize(head.position(local),dim=-1)
        return head.gamma*(a@b.T)
    joined=torch.cat((local[None].expand(k,-1,-1),query[:,None].expand(-1,k,-1),
                      bands[:,None].expand(-1,k,-1)),dim=-1)
    return head.mlp(joined).squeeze(-1)


def permutation_scores(scores,perms):
    return scores[torch.arange(len(scores),device=scores.device)[None],perms].sum(-1)


def permutation_loss(scores,perms,positive):
    total=permutation_scores(scores,perms)
    return torch.logsumexp(total,0)-torch.logsumexp(total[positive],0)


def fixed_sample(s):
    logits=s['logits'].numpy()
    d=decode_r2(logits,s['record'])
    active=d['active']
    points=np.asarray(d['joint'],dtype=np.float32).reshape(-1,2)
    local=[]
    for q,p in zip(active,points):
        candidates=np.asarray(s['record']['candidates'][q]['positions'],dtype=np.float32)
        hits=np.flatnonzero(np.all(candidates==p,axis=1))
        if not len(hits):
            raise RuntimeError('Fixed position not present in registered descriptor cache')
        local.append(s['local'][q,int(hits[0])].clone())
    k=len(active)
    perms=list(itertools.permutations(range(k)))
    metrics=[]
    for perm in perms:
        row=R1Run.metric(None,s['truth'],points[list(perm)],logits,active,s['bands'],s['ignore'],s['metadata'])
        row['permutation']=list(perm)
        metrics.append(row)
    baseline=metrics[0]
    for row in metrics:
        for key in ('gospa_m','tp_at_100m','true_count','predicted_count'):
            if not np.isclose(row[key],baseline[key],rtol=0,atol=1e-6):
                raise AssertionError(f'Permutation changed location metric {key}')
        np.testing.assert_allclose(sorted(row['matched_errors_m']),sorted(baseline['matched_errors_m']),rtol=0,atol=1e-5)
    quality=[r['joint_tp'] for r in metrics]
    return {'local':torch.stack(local) if local else torch.empty((0,19,128)),
            'query':s['query'].clone(),'logits':s['logits'].clone(),'active':active,
            'perms':torch.tensor(perms,dtype=torch.long).reshape(len(perms),k),
            'positive':torch.tensor([x==max(quality) for x in quality]),'metrics':metrics,
            'points':points,'index':s['metadata']['raw_index']}


class R4Runtime:
    guard=Runtime.guard

    def __init__(self,out):
        self.out=Path(out).resolve(strict=True)
        if not self.out.is_relative_to(BASE.resolve()) or self.out.is_relative_to(SOURCE.resolve()):
            raise ValueError('R4 output isolation')
        self.deadline=None
        self.peak_ram=0
        self.inputs=[]

    def get(self,row):
        path=Path(row['path']).resolve(strict=True)
        if not path.is_relative_to(ROOT):
            raise ValueError('Input escaped workspace')
        self.guard()
        self.inputs.append(row)
        return verified_read(row,self.out/'anomalies')


@torch.no_grad()
def r3_audit(runtime,seed,cache):
    report=read(SOURCE/'report.json')['seeds'][str(seed)]
    outputs={}
    for arm,epoch in (('joint',0),('old',20),('joint',20)):
        row=report[arm]['best']['checkpoint'] if epoch==0 else report[arm]['last']
        cp=torch.load(io.BytesIO(runtime.get(row)),map_location='cpu',weights_only=False)
        head=AssociationHead().cuda()
        head.load_state_dict(cp['state'] if epoch==0 else cp['head'])
        del cp
        margins=[]
        for i in range(len(cache.rows)):
            if i%128==0:
                runtime.guard()
            s=cache.get(i)
            if not len(s['ranks']) or s['positive'].all():
                continue
            scores=cached_scores(head,s,'cuda')
            values=s['heat_sum'].cuda()
            ranks=s['ranks'].cuda()
            for j,q in enumerate(s['active']):
                values=values+scores[q].log_softmax(0)[ranks[:,j]]
            positive=s['positive'].cuda()
            margins.append(float(values[positive].max()-values[~positive].max()))
        outputs[f'{arm}_epoch{epoch}']={'samples':len(margins),'rank_correct_fraction':float(np.mean(np.asarray(margins)>0)),
                                      'rank_margin_mean':float(np.mean(margins))}
        del head
    return outputs


def prepare(runtime,seed,split,audit=False):
    index_path=SOURCE/f'cache/{seed}/{split}/index.json'
    index=__import__('json').loads(runtime.get(identity(index_path)))
    for row in index:
        if not Path(row['path']).resolve(strict=True).is_relative_to(SOURCE.resolve()):
            raise ValueError('Cache escaped registered R3 source')
    cache=Cache(index,runtime.out)
    old_audit=r3_audit(runtime,seed,cache) if audit else None
    folder=runtime.out/f'cache/{seed}/{split}'
    folder.mkdir(parents=True,exist_ok=False)
    pack=folder/'samples.bin'
    records=[]
    base=[]
    ceiling=0
    informative=0
    with pack.open('xb') as handle:
        for i in range(len(index)):
            if i%128==0:
                runtime.guard()
            s=fixed_sample(cache.get(i))
            base.append(s['metrics'][0])
            ceiling+=max(r['joint_tp'] for r in s['metrics'])
            informative+=int(not bool(s['positive'].all()))
            stream=io.BytesIO()
            torch.save(s,stream)
            data=stream.getvalue()
            offset=handle.tell()
            handle.write(data)
            records.append({'path':str(pack.resolve()),'offset':offset,'size_bytes':len(data),
                            'sha256':hashlib.sha256(data).hexdigest()})
    cache.clear()
    for row in index:
        verified_read(row,runtime.out/'anomalies',offset=row['offset'],length=row['size_bytes'])
    write(folder/'index.json',records)
    save_rows(folder/'baseline_samples.jsonl',base)
    summary=summarize(base)
    summary.update(oracle_joint_recall=ceiling/sum(r['true_count'] for r in base),informative_samples=informative)
    write(folder/'preparation.json',{'baseline':summary,'r3_train_audit':old_audit,'source_ranges_verified':len(index)})
    return records,summary,old_audit


@torch.no_grad()
def evaluate(runtime,head,cache,base):
    head.eval()
    rows=[]
    margins=[]
    hits=[]
    for i in range(len(cache.rows)):
        if i%128==0:
            runtime.guard()
        s=cache.get(i)
        if len(s['active'])<2:
            chosen=0
        else:
            values=permutation_scores(scores_for(head,s,'cuda'),s['perms'].cuda())
            chosen=int(values.argmax())
            positive=s['positive'].cuda()
            if not positive.all():
                margins.append(float(values[positive].max()-values[~positive].max()))
                hits.append(bool(positive[chosen]))
        rows.append(s['metrics'][chosen])
    result=summarize(rows)
    result.update(permutation_accuracy=float(np.mean(hits)),rank_margin_mean=float(np.mean(margins)),
                  rank_strict_fraction=float(np.mean(np.asarray(margins)>0)),informative_samples=len(hits))
    gap=base['oracle_joint_recall']-base['joint_recall100_f1_08']
    result['oracle_gap_closure']=(result['joint_recall100_f1_08']-base['joint_recall100_f1_08'])/gap if gap>0 else None
    for name in ('gospa_m','matched_rmse_m','matched_coverage','count_accuracy','recall100'):
        if not np.isclose(result[name],base[name],rtol=0,atol=1e-5):
            raise AssertionError(f'Fixed-location invariant failed: {name}')
    return result,rows


def train(runtime,seed,arm,train_cache,val_cache,baselines,shared_project):
    g4.set_deterministic(seed+1000)
    head=(AssociationHead() if arm=='concat' else ExplicitMatcher()).cuda()
    head.project.load_state_dict(shared_project)
    optimizer=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=1e-4)
    generator=torch.Generator().manual_seed(seed)
    folder=runtime.out/f'training/{seed}/{arm}'
    folder.mkdir(parents=True,exist_ok=False)
    best=None
    history=[]
    for epoch in range(21):
        began=time.time()
        losses=[]
        steps=0
        if epoch:
            head.train()
            for ids in torch.randperm(len(train_cache.rows),generator=generator).split(4):
                runtime.guard()
                terms=[]
                for i in ids.tolist():
                    s=train_cache.get(i)
                    if not s['positive'].all():
                        terms.append(permutation_loss(scores_for(head,s,'cuda'),s['perms'].cuda(),s['positive'].cuda()))
                if terms:
                    loss=torch.stack(terms).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm=nn.utils.clip_grad_norm_(head.parameters(),10.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise RuntimeError('Nonfinite matching training')
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    steps+=1
        tr=va=None
        if epoch%2==0:
            tr,_=evaluate(runtime,head,train_cache,baselines['train'])
            va,rows=evaluate(runtime,head,val_cache,baselines['val_select'])
            cp=folder/f'epoch{epoch:02d}.pt'
            torch.save({'epoch':epoch,'state':head.state_dict(),'optimizer':optimizer.state_dict(),
                        'generator':generator.get_state(),'train':tr,'val':va},cp)
            write(cp.with_suffix('.identity.json'),identity(cp))
            key=(-va['joint_recall100_f1_08'],epoch)
            if best is None or key<tuple(best['key']):
                best={'epoch':epoch,'key':list(key),'train':tr,'val':va,'checkpoint':identity(cp)}
                save_rows(folder/f'best_epoch{epoch:02d}_samples.jsonl',rows)
        history.append({'epoch':epoch,'seconds':time.time()-began,'loss':float(np.mean(losses)) if losses else None,
                        'steps':steps,'train':tr,'val':va,'gamma':float(head.gamma.detach()) if arm=='explicit' else None})
        write(folder/'history.json',history)
        write(runtime.out/'progress.json',{'seed':seed,'arm':arm,'epoch':epoch,'best_epoch':best['epoch'],
                                         'best_joint_recall':best['val']['joint_recall100_f1_08']})
    result={'status':'COMPLETED','best':best,'history':history}
    write(folder/'report.json',result)
    safe_print(f'阶段完成 {seed}/{arm}：20轮，best={best["epoch"]}，joint={best["val"]["joint_recall100_f1_08"]:.4f}')
    return result


def run():
    setup_environment()
    began=time.time()
    out=BASE/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    runtime=R4Runtime(out)
    safe_print(f'G5-R4 开始：{out}')
    prior=read(SOURCE/'final_audit.json')
    if prior['status']!='PASS':
        raise RuntimeError('R3 source is not audited')
    for row in prior['files']+read(SOURCE/'contract.json')['files']:
        runtime.get(row)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    source_files={Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if isinstance(name,str) and Path(name).is_absolute():
            p=Path(name).resolve()
            if p.suffix=='.py' and p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'outputs_e2e'):
                source_files.add(p)
    contract={'config':CONFIG,'files':[identity(p) for p in sorted(source_files)],
              'source_final_audit':identity(SOURCE/'final_audit.json')}
    write(out/'contract.json',contract)
    result={}
    for seed in SEEDS:
        tr,tbase,old=prepare(runtime,seed,'train',audit=True)
        va,vbase,_=prepare(runtime,seed,'val_select')
        safe_print(f'准备完成 {seed}：固定位置/缓存/输入校验通过；train可区分{tbase["informative_samples"]}，val可区分{vbase["informative_samples"]}')
        train_cache,val_cache=Cache(tr,out),Cache(va,out)
        torch.manual_seed(seed+1000)
        initial=AssociationHead().project.state_dict()
        result[str(seed)]={'baseline':{'train':tbase,'val_select':vbase},'r3_train_audit':old}
        for arm in ('concat','explicit'):
            result[str(seed)][arm]=train(runtime,seed,arm,train_cache,val_cache,result[str(seed)]['baseline'],initial)
            write(out/'report.json',{'status':'RUNNING','seeds':result})
        train_cache.clear()
        val_cache.clear()
        for row in tr+va:
            verified_read(row,out/'anomalies',offset=row['offset'],length=row['size_bytes'])
        del train_cache,val_cache
        gc.collect()
        torch.cuda.empty_cache()
    for row in runtime.inputs+contract['files']+[contract['source_final_audit']]:
        verified_read(row,out/'anomalies')
    write(out/'report.json',{'status':'COMPLETED','seeds':result,'wall_seconds':time.time()-began,
                           'peak_ram_percent':runtime.peak_ram,'test_executed':False,'val_compare_executed':False})
    write(out/'final_audit.json',{'status':'PASS','inputs_pre_post_verified':True,'fixed_location_invariants':True,
          'files':[identity(p) for p in sorted(out.rglob('*.json')) if p.name!='final_audit.json']})
    safe_print(f'G5-R4 完成：{out / "report.json"}')


if __name__=='__main__':
    run()
