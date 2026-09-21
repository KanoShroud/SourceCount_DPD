"""一次性工程v2登记与恢复验证；不运行正式六轨训练。"""

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[4]))
import argparse
import gc
import hashlib
from pathlib import Path
import time

import torch

from 统一模型代码.gates.g5.diagnostics.G5样本读取验证 import RUN, ROOT, NAMES, digest
from 统一模型代码.common.g5_sample_range import register_shard
from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.gates.g5.e2e_g5_contract import source_paths
from 统一模型代码.gates.g5.e2e_g5_model import build_context, g4, load_split
from 统一模型代码.gates.g5.e2e_g5_train import load_checkpoint, save_checkpoint, rng_state, restore_rng
from 统一模型代码.gates.g5.audits.e2e_g5_pilot import step
from 统一模型代码.common.g5_verified_io import RetryVerifiedCache, verified_read
from 统一模型代码.common.verified_feature_loader import VerifiedShardedArray

DEST = RUN/'engineering_v2'
CHANGED = {'e2e_g5_contract.py','e2e_g5_model.py','e2e_g5_train.py'}


def prepare():
    if DEST.exists():
        raise FileExistsError(DEST)
    old = g4.read_json(RUN/'training_code_contract.json')
    for row in old['files']:
        path = Path(row['path'])
        if path.name in CHANGED:
            archived = RUN/'training_source_snapshot_cuda_init'/path.relative_to(ROOT)
            verified_read({**row,'path':str(archived)}, RUN/'anomalies')
        else:
            verified_read(row,RUN/'anomalies')
    DEST.mkdir()
    fm = g4.read_json(RUN/'feature_manifest.json')
    registry = dict(status='PASS',cache_bytes=2*2**30,prefetch_batches=1,batch_size=4,
                    ram_warning_percent=85,ram_action='stop_before_further_allocation',indexes={})
    for split in ('train','val_select','val_compare'):
        index = {}
        for name in NAMES:
            index[name] = []
            for row in fm['files'][split][name]:
                if not Path(row['path']).resolve(strict=True).is_relative_to((RUN/'features'/split).resolve()):
                    raise RuntimeError('Unexpected feature root')
                index[name].extend(register_shard(row,DEST/'anomalies'))
        p = DEST/f'{split}_sample_index.json'
        g4.write_json(p,index)
        registry['indexes'][split] = g4.identity(p)
        print(f'逐样本索引登记完成：{split}',flush=True)
    g4.write_json(DEST/'index_registry.json',registry)


def validate():
    out = DEST/('recovery_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    m = g4.read_json(RUN/'manifest.json'); fm = g4.read_json(RUN/'feature_manifest.json')
    seed = m['config']['training_seeds'][0]; device = torch.device('cuda:0')
    results = {}
    for track in ('sg','e2e'):
        arms = {}
        for kind in ('legacy','prefetch'):
            context = build_context(RUN,m,seed,device)
            optimizer = torch.optim.AdamW(context.parameter_groups,weight_decay=m['config']['weight_decay'])
            generator = torch.Generator().manual_seed(seed)
            if track == 'sg':
                cp = load_checkpoint(RUN/'training/20260921/sg/last.pt')
                assert cp['epoch']==48 and cp['optimizer_steps']==49152
                g4.load_state(context,cp['state']); optimizer.load_state_dict(cp['optimizer'])
                generator.set_state(cp['generator']); restore_rng(cp['rng']); del cp
            ids = list(torch.randperm(4096,generator=generator).split(4))[:8]
            features,targets,cache = load_split(RUN,m,fm,'train')
            if kind == 'legacy':
                cache = RetryVerifiedCache(2*2**30,out/'anomalies')
                features = g4.FeatureStore(*(VerifiedShardedArray(fm['files']['train'][n],cache) for n in NAMES))
            target_cache = g4.as_cached_targets(targets)
            trace=[]
            iterator = batches(features,ids) if kind=='prefetch' else ((i,features) for i in ids)
            for i,(indices,data) in enumerate(iterator):
                trace.append(step(context,optimizer,data,target_cache,indices,m['config'],device,track=='sg'))
                if i==3:
                    path=out/f'{track}_{kind}.pt'
                    save_checkpoint(path,dict(state=g4.state_payload(context),optimizer=optimizer.state_dict(),
                                              rng=rng_state(),generator=generator.get_state()))
            iterator.close()
            final=(g4.state_digest(context),digest(optimizer.state_dict()))
            cp=load_checkpoint(path)
            g4.load_state(context,cp['state']); optimizer.load_state_dict(cp['optimizer']); restore_rng(cp['rng'])
            restored_generator=torch.Generator(); restored_generator.set_state(cp['generator'])
            assert torch.equal(torch.randperm(4096,generator=generator),torch.randperm(4096,generator=restored_generator))
            # Fresh cache/iterator: no queue or cached feature is serialized as training state.
            if kind=='prefetch':
                features,_,cache=load_split(RUN,m,fm,'train')
            replay=batches(features,ids[4:]) if kind=='prefetch' else ((i,features) for i in ids[4:])
            tail=[step(context,optimizer,data,target_cache,i,m['config'],device,track=='sg') for i,data in replay]
            replay.close()
            assert tail==trace[4:] and final==(g4.state_digest(context),digest(optimizer.state_dict()))
            arms[kind]=dict(trace=trace,final=final,checkpoint_resume_exact=True,next_epoch_order_exact=True)
            del context,optimizer,features,cache,cp,targets,target_cache,data,iterator,replay
            gc.collect();torch.cuda.empty_cache()
        assert arms['legacy']==arms['prefetch']
        results[track]=arms
        print(f'{track}：旧加载/新预取、保存恢复、下一epoch顺序均一致',flush=True)
    # Close a live queue early, then re-read exactly the same next indices.
    features,_,_=load_split(RUN,m,fm,'train')
    stream=batches(features,ids); next(stream); stream.close()
    a=list(batches(features,ids[1:2]))[0][1]
    assert all((new[ids[1].tolist()]==old[ids[1].tolist()]).all()
               for new,old in zip((a.spatial,a.e1,a.d2),(features.spatial,features.e1,features.d2)))
    from 统一模型代码.common.g5_runtime_v2 import load_features
    for split in ('val_select','val_compare'):
        new,cache=load_features(RUN,split,2*2**30)
        oldcache=RetryVerifiedCache(2*2**30,out/'anomalies')
        checks=[0,m['config']['counts'][split]-1,0]
        for name,array in zip(NAMES,(new.spatial,new.e1,new.d2)):
            old=VerifiedShardedArray(fm['files'][split][name],oldcache)
            left,right=array[checks],old[checks]
            assert left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()
        del new,cache,oldcache,old,left,right
    post=0
    for name in NAMES:
        for row in fm['files']['train'][name]:
            verified_read(row,out/'anomalies');post+=1
    verified_read(fm['files']['train']['targets'],out/'anomalies')
    report=dict(status='PASS',tracks=results,queue_close_safe=True,train_shards_post_verified=post,
                evaluation_feature_samples_exact=True,evaluation_metrics_computed=False,
                formal_training_started=False,test_executed=False,
                scope='8-step numerical continuation windows, not a full epoch or full pipeline')
    g4.write_json(out/'report.json',report)
    g4.write_json(DEST/'recovery_validation.identity.json',g4.identity(out/'report.json'))


def freeze():
    contract_path=DEST/'contract.json'
    if contract_path.exists(): raise FileExistsError(contract_path)
    for row in g4.read_json(RUN/'training_code_contract.json')['files']:
        if Path(row['path']).name not in CHANGED:
            verified_read(row,DEST/'anomalies')
    ready=g4.read_json(DEST/'recovery_validation.identity.json')
    verified_read(ready,DEST/'anomalies')
    paths=set(source_paths())
    paths.update(ROOT/n for n in ('统一模型代码/common/g5_runtime_v2.py','统一模型代码/common/g5_sample_range.py','运行入口/E2E/G5/G5一键继续.py','统一模型代码/gates/g5/diagnostics/G5工程接入验证.py'))
    snapshot=DEST/'source_snapshot';snapshot.mkdir(exist_ok=False)
    rows=[]
    for path in sorted(paths):
        payload=path.read_bytes();target=snapshot/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as h:h.write(payload)
        rows.append(dict(path=str(path),size_bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()))
    for name in ('training_code_contract.json','manifest.json','input_manifest.json','feature_manifest.json',
                 'provenance_audit.json','user_pause_audit.json','engineering_v2/index_registry.json',
                 'engineering_v2/recovery_validation.identity.json'):
        rows.append(g4.identity(RUN/name))
    registry=g4.read_json(DEST/'index_registry.json')
    rows.extend(registry['indexes'].values());rows.append(ready)
    g4.write_json(contract_path,dict(status='FROZEN_ENGINEERING_V2',created_at=time.time(),files=rows,
                                   scientific_configuration='unchanged; original manifest immutable',
                                   replaces_code_only_after_sg48=True,ram_warning_percent=85,
                                   prefetch_batches=1,legacy_contract=g4.identity(RUN/'training_code_contract.json')))
    from 统一模型代码.gates.g5.e2e_g5_contract import verify_code
    print(verify_code(RUN),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','validate','freeze'))
    globals()[p.parse_args().action]()
