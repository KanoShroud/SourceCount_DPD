"""G5-R3: same frozen inputs, old slot supervision versus joint ranking."""
from __future__ import annotations

from collections import OrderedDict
import gc
import hashlib
import io
import itertools
from pathlib import Path
import sys
import time

import numpy as np
import torch

from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run, identity
from 统一模型代码.gates.g5.r1.g5_r1_decode import maximum_valid_pairs
from 统一模型代码.gates.g5.r1.g5_r1_report import summarize
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import save_rows
from 统一模型代码.gates.g5.r2.g5_r2_model import (
    AssociationHead, forward_r2, sample_centers, positive_mask, decode_r2,
)
from 统一模型代码.gates.g5.r2.g5_r2_runtime import (
    ROOT, SOURCE, SEEDS, Runtime, Progress, read, write, safe_print, setup_environment, g4,
)

BASE = ROOT/'outputs_e2e/unified/e2e_g5_r3'
CONFIG = {'epochs':20, 'eval_every':2, 'batch':4, 'lr':1e-4, 'weight_decay':1e-4,
          'gradient_clip':10., 'seeds':list(SEEDS), 'cache_bytes':2*1024**3,
          'positive':'max_joint_then_max_location', 'precision':'FP32',
          'val_compare_executed':False, 'test_executed':False}


class R3Runtime(Runtime):
    def __init__(self,out):
        self.out = Path(out).resolve(strict=True)
        if not self.out.is_relative_to(BASE.resolve()):
            raise ValueError('R3 output isolation')
        self.deadline = None
        self.inputs,self.ranges = {},{}
        self.peak_ram = 0
        self.manifest = self.fm = None

    def features(self,split,cache=None):
        if split not in ('train','val_select'):
            raise ValueError('Only approved R3 splits')
        return super().features(split,cache)


def combinations(record,logits,truth,bands,ignore):
    active = np.flatnonzero(logits.max(-1)>=0).tolist()
    if len(active)<2:
        return np.empty((0,len(active)),dtype=np.int64),np.empty(0),np.empty(0,dtype=bool)
    cs = [record['candidates'][q] for q in active]
    ranks,heats,quality = [],[],[]
    band_ok = np.zeros((len(truth),len(active)),dtype=bool)
    for t in range(len(truth)):
        mask = ignore[t]<.5
        for j,q in enumerate(active):
            a,b = bands[t,mask]>.5,logits[q,mask]>=0
            band_ok[t,j] = 2*np.sum(a&b)/max(int(a.sum()+b.sum()),1)>=.8
    for rank in itertools.product(*(range(len(c['scores'])) for c in cs)):
        points = np.asarray([c['positions'][r] for c,r in zip(cs,rank)])
        if any(np.linalg.norm(a-b)<30 for a,b in itertools.combinations(points,2)):
            continue
        # Decoder exports float32 positions before official metrics.
        distance = np.linalg.norm(truth[:,None,:]-points.astype(np.float32)[None,:,:],axis=-1)
        valid = distance<=100
        quality.append((maximum_valid_pairs(valid & band_ok),maximum_valid_pairs(valid)))
        ranks.append(rank)
        heats.append(sum(np.log(max(c['scores'][r],1e-20)) for c,r in zip(cs,rank)))
    best = max(quality) if quality else None
    return (np.asarray(ranks,dtype=np.int64).reshape(-1,len(active)),
            np.asarray(heats,dtype=np.float64),np.asarray([q==best for q in quality],dtype=bool))


def cached_scores(head,sample,device):
    local = sample['local'].to(device)
    projected = head.project(local).flatten(2)
    query = sample['query'].to(device)[:,None].expand(-1,local.shape[1],-1)
    logits = sample['logits'].to(device).sigmoid()[:,None].expand(-1,local.shape[1],-1)
    return head.mlp(torch.cat((projected,query,logits),dim=-1)).squeeze(-1)


def loss_terms(scores,sample,arm):
    if arm=='old':
        terms=[]
        for q in range(3):
            pos,valid = sample['old_positive'][q].to(scores.device),sample['old_valid'][q].to(scores.device)
            if pos.any():
                terms.append(torch.logsumexp(scores[q,valid],0)-torch.logsumexp(scores[q,pos],0))
        return terms
    ranks = sample['ranks'].to(scores.device)
    positive = sample['positive'].to(scores.device)
    if not len(ranks) or positive.all():
        return []
    active = sample['active']
    total = sample['heat_sum'].to(scores.device)
    for j,q in enumerate(active):
        total = total + scores[q].log_softmax(0)[ranks[:,j]]
    return [torch.logsumexp(total,0)-torch.logsumexp(total[positive],0)]


class Cache:
    def __init__(self,rows,out):
        self.rows,self.out = rows,out
        self.items = OrderedDict()
        self.size=0

    def get(self,i):
        if i in self.items:
            self.items.move_to_end(i)
            return self.items[i]
        row=self.rows[i]
        data=verified_read(row,self.out/'anomalies',offset=row['offset'],length=row['size_bytes'])
        sample=torch.load(io.BytesIO(data),map_location='cpu',weights_only=False)
        while self.items and self.size+row['size_bytes']>CONFIG['cache_bytes']:
            old,_=self.items.popitem(last=False)
            self.size-=self.rows[old]['size_bytes']
        self.items[i]=sample
        self.size+=row['size_bytes']
        return sample

    def clear(self):
        self.items.clear()
        self.size=0


@torch.no_grad()
def prepare(runtime,seed,split,context,head):
    folder=runtime.out/f'cache/{seed}/{split}'
    folder.mkdir(parents=True,exist_ok=False)
    bundle=runtime.features(split)
    features,targets,metadata,index,_=bundle
    n=len(targets.counts)
    progress=Progress(f'{seed} 提取 {split}',n,runtime.out/'progress.json')
    g4.set_mode(context,training=False)
    head.eval()
    iterator=batches(features,list(torch.arange(n).split(4)),prefetch=True)
    rows=[]
    baseline=[]
    pack=folder/'samples.bin'
    try:
        with pack.open('xb') as handle:
            for ids,batch in iterator:
                runtime.guard()
                runtime.consumed(index,ids)
                outputs,records,original_scores=forward_r2(context,head,batch,ids,torch.device('cuda:0'),'c1')
                query,logits,_,heat,_=outputs
                spatial=g4.numpy_batch(batch.spatial,ids,torch.device('cuda:0'))
                mappings=g4.r1.g2.assignments(logits,heat,targets.band[ids].cuda(),targets.ignore[ids].cuda(),
                                              targets.positions[ids].cuda(),targets.counts[ids].cuda())
                for j,i in enumerate(ids.tolist()):
                    record=records[j]
                    if any(len(c['scores'])!=8 for c in record['candidates']):
                        raise RuntimeError('Expected registered top8 nonempty candidates')
                    local=torch.stack([sample_centers(spatial[j],torch.tensor(c['positions'],device='cuda',dtype=spatial.dtype))
                                       for c in record['candidates']]).cpu().clone()
                    truth=targets.positions[i,:int(targets.counts[i])].numpy()
                    bands,ignore=targets.band[i].numpy(),targets.ignore[i].numpy()
                    logit=logits[j].cpu().numpy()
                    ranks,h,pos=combinations(record,logit,truth,bands,ignore)
                    old_pos=torch.zeros((3,8),dtype=torch.bool)
                    old_valid=torch.zeros_like(old_pos)
                    for q,t in mappings[j].items():
                        old_pos[q],old_valid[q]=positive_mask(torch.tensor(record['candidates'][q]['positions']),
                                                            torch.tensor(truth),t)
                    sample={'local':local,'query':query[j].cpu().clone(),'logits':logits[j].cpu().clone(),
                            'record':record,'truth':truth,'bands':bands,'ignore':ignore,'metadata':metadata[i],
                            'ranks':torch.from_numpy(ranks),'heat_sum':torch.from_numpy(h),'positive':torch.from_numpy(pos),
                            'active':np.flatnonzero(logit.max(-1)>=0).tolist(),
                            'old_positive':old_pos,'old_valid':old_valid}
                    zero=cached_scores(head,sample,'cuda')
                    for q in range(3):
                        torch.testing.assert_close(zero[q],original_scores[j][q],rtol=0,atol=0)
                    d=decode_r2(logit,record)
                    assert d['joint']==decode_r2(logit,record,list(zero))['joint']
                    baseline.append(R1Run.metric(None,truth,d['joint'],logit,d['active'],bands,ignore,metadata[i]))
                    stream=io.BytesIO()
                    torch.save(sample,stream)
                    data=stream.getvalue()
                    offset=handle.tell()
                    handle.write(data)
                    rows.append({'path':str(pack.resolve()),'offset':offset,'size_bytes':len(data),
                                 'sha256':hashlib.sha256(data).hexdigest()})
                progress.update(len(rows))
    finally:
        iterator.close()
    write(folder/'index.json',rows)
    save_rows(folder/'baseline_samples.jsonl',baseline)
    write(folder/'baseline.json',summarize(baseline))
    runtime.postcheck(f'{seed}_{split}_extraction')
    del bundle,features
    gc.collect()
    return rows,baseline


@torch.no_grad()
def evaluate(runtime,head,cache):
    head.eval()
    rows=[]
    margins=[]
    for i in range(len(cache.rows)):
        if i%64==0:
            runtime.guard()
        sample=cache.get(i)
        scores=cached_scores(head,sample,'cuda')
        logits=sample['logits'].numpy()
        d=decode_r2(logits,sample['record'],list(scores))
        row=R1Run.metric(None,sample['truth'],d['joint'],logits,d['active'],sample['bands'],sample['ignore'],sample['metadata'])
        row.update(decode=d)
        rows.append(row)
        ranks=sample['ranks'].cuda()
        pos=sample['positive'].cuda()
        if len(ranks) and not pos.all():
            s=sample['heat_sum'].cuda()
            for j,q in enumerate(sample['active']):
                s=s+scores[q].log_softmax(0)[ranks[:,j]]
            margins.append(float(s[pos].max()-s[~pos].max()))
    return {**summarize(rows),'rank_margin_mean':float(np.mean(margins)) if margins else None,
            'rank_correct_fraction':float(np.mean(np.asarray(margins)>0)) if margins else None,
            'rank_informative_samples':len(margins)},rows


def train(runtime,seed,arm,train_cache,val_cache,initial):
    folder=runtime.out/f'training/{seed}/{arm}'
    folder.mkdir(parents=True,exist_ok=False)
    head=AssociationHead().cuda()
    head.load_state_dict(initial)
    optimizer=torch.optim.AdamW(head.parameters(),lr=CONFIG['lr'],weight_decay=CONFIG['weight_decay'])
    generator=torch.Generator().manual_seed(seed)
    history=[]
    best=None
    for epoch in range(21):
        began=time.time()
        losses=[]
        steps=0
        if epoch:
            head.train()
            order=torch.randperm(len(train_cache.rows),generator=generator).split(4)
            progress=Progress(f'{seed}/{arm} {epoch}/20',len(order),runtime.out/'progress.json')
            for step,ids in enumerate(order,1):
                runtime.guard()
                terms=[]
                for i in ids.tolist():
                    sample=train_cache.get(i)
                    terms+=loss_terms(cached_scores(head,sample,'cuda'),sample,arm)
                if terms:
                    loss=torch.stack(terms).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(head.parameters(),CONFIG['gradient_clip'])
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise RuntimeError('Nonfinite scorer training')
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    steps+=1
                progress.update(step)
        metrics=None
        if epoch%2==0:
            metrics,rows=evaluate(runtime,head,val_cache)
            key=(-metrics['joint_recall100_f1_08'],metrics['gospa_m'],epoch)
            if best is None or key<tuple(best['selection_key']):
                path=folder/f'best_epoch{epoch:02d}.pt'
                torch.save({'epoch':epoch,'state':head.state_dict(),'metrics':metrics},path)
                save_rows(folder/f'best_epoch{epoch:02d}_samples.jsonl',rows)
                best={'epoch':epoch,'metrics':metrics,'selection_key':list(key),'checkpoint':identity(path)}
        history.append({'epoch':epoch,'loss':float(np.mean(losses)) if losses else None,
                        'optimizer_steps':steps,'seconds':time.time()-began,'validation':metrics})
        torch.save({'epoch':epoch,'head':head.state_dict(),'optimizer':optimizer.state_dict(),
                    'generator':generator.get_state(),'history':history,'best':best},folder/'last.pt')
        write(folder/'history.json',history)
        safe_print(f'{seed}/{arm} epoch={epoch} 耗时{time.time()-began:.1f}s best={best["epoch"]} '
                   f'joint={best["metrics"]["joint_recall100_f1_08"]:.4f}')
    report={'status':'COMPLETED','best':best,'history':history,'last':identity(folder/'last.pt')}
    write(folder/'report.json',report)
    return report


def run():
    setup_environment()
    began=time.time()
    out=BASE/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    runtime=R3Runtime(out)
    safe_print(f'G5-R3 开始：{out}')
    runtime.preflight()
    files={Path(__file__).resolve()}
    for module in tuple(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if isinstance(value,str) and Path(value).is_absolute():
            path=Path(value).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'outputs_e2e'):
                files.add(path)
    contract={'config':CONFIG,'files':[identity(p) for p in sorted(files)],
              'base_checkpoints':[read(SOURCE/f'training/{s}/sg/best.identity.json') for s in SEEDS]}
    write(out/'contract.json',contract)
    reports={}
    for seed in SEEDS:
        context,head,optimizer,parameters=runtime.context(seed,'c1')
        for p in parameters:
            p.requires_grad_(False)
        del optimizer,parameters
        initial={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        tr,_=prepare(runtime,seed,'train',context,head)
        va,baseline=prepare(runtime,seed,'val_select',context,head)
        del context,head
        gc.collect()
        torch.cuda.empty_cache()
        train_cache,val_cache=Cache(tr,out),Cache(va,out)
        reports[str(seed)]={'baseline':summarize(baseline)}
        for arm in ('old','joint'):
            reports[str(seed)][arm]=train(runtime,seed,arm,train_cache,val_cache,initial)
            write(out/'report.json',{'status':'RUNNING','seeds':reports})
        train_cache.clear()
        val_cache.clear()
        for row in tr+va:
            verified_read(row,out/'anomalies',offset=row['offset'],length=row['size_bytes'])
        del train_cache,val_cache
        gc.collect()
    for row in contract['files']+contract['base_checkpoints']:
        verified_read(row,out/'anomalies')
    runtime.postcheck('final')
    write(out/'report.json',{'status':'COMPLETED','seeds':reports,'wall_seconds':time.time()-began,
          'peak_ram_percent':runtime.peak_ram,'test_executed':False,'val_compare_executed':False})
    write(out/'final_audit.json',{'status':'PASS','source_and_checkpoint_identities':True,
          'cache_ranges_rechecked':True,'files':[identity(p) for p in sorted(out.rglob('report.json'))]})
    safe_print(f'G5-R3 完成：{out / "report.json"}')


if __name__=='__main__':
    run()
