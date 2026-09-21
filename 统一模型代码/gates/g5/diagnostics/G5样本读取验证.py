"""一键执行按样本读取候选的256步SG/E2E配对工程验证，不恢复正式G5。"""

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[4]))
import gc
import hashlib
import json
from pathlib import Path
import time

import psutil
import torch

from 统一模型代码.common.g5_sample_range import register_shard, SampleRangeArray, SampleRangeCache
from 统一模型代码.gates.g5.e2e_g5_contract import verify_code
from 统一模型代码.gates.g5.e2e_g5_model import build_context, g4, load_split
from 统一模型代码.gates.g5.e2e_g5_train import load_checkpoint, save_checkpoint, rng_state, restore_rng
from 统一模型代码.gates.g5.audits.e2e_g5_pilot import step
from 统一模型代码.common.g5_verified_io import RetryVerifiedCache, verified_read

ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / 'outputs_e2e/unified/e2e_g5/20260919_approved'
NAMES = ('ch3_spatial', 'd8_e1', 'd8_d2')


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            a = v.detach().cpu().contiguous().numpy()
            h.update(str((a.dtype.str, a.shape)).encode()); h.update(a.tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def main():
    for p in psutil.process_iter(['cmdline']):
        if any(Path(arg).name == 'e2e_g5_train.py' for arg in (p.info['cmdline'] or [])):
            raise RuntimeError('Formal training is still running')
    verify_code(RUN)
    m = g4.read_json(RUN / 'manifest.json')
    fm = g4.read_json(RUN / 'feature_manifest.json')
    assert g4.read_json(RUN / 'user_pause_audit.json')['status'] == 'SG48_COMPLETE_STOPPED'
    out = ROOT / 'outputs_e2e/unified/g5_sample_range' / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    begun = time.perf_counter()
    def guard():
        if time.perf_counter() - begun > 1800 or psutil.virtual_memory().available < 8*2**30:
            raise RuntimeError('Engineering validation resource limit')
        if psutil.disk_usage(str(out)).free < 50*2**30:
            raise RuntimeError('Insufficient disk space')
    guard()
    sources = [ROOT / '统一模型代码/common/g5_sample_range.py', ROOT / '统一模型代码/gates/g5/diagnostics/G5样本读取验证.py']
    identities = [g4.identity(path) for path in sources]
    g4.write_json(out / 'protocol.json', {'steps':256, 'batch':4, 'seed':20260921,
        'tracks':['sg','e2e'], 'source_identities':identities, 'source_run':str(RUN),
        'cache_bytes':m['config']['feature_cache_bytes'], 'prefetch':False, 'test_executed':False})
    try:
        index = {}
        for name in NAMES:
            index[name] = []
            for row in fm['files']['train'][name]:
                guard()
                if not Path(row['path']).resolve(strict=True).is_relative_to(RUN / 'features/train'):
                    raise RuntimeError('Unexpected source root')
                index[name].extend(register_shard(row, out / 'anomalies'))
            print(f'范围身份登记完成：{name}', flush=True)
        index_path = out / 'sample_index.json'
        g4.write_json(index_path, index)
        identity = g4.identity(index_path)
        g4.write_json(out / 'sample_index.identity.json', identity)
        index = json.loads(verified_read(identity, out / 'anomalies'))
        cfg = m['config']; seed = cfg['training_seeds'][0]
        batches = list(torch.randperm(4096, generator=torch.Generator().manual_seed(seed)).split(4))[:257]
        device = torch.device('cuda:0'); torch.cuda.init()
        results = {}
        for track in ('sg', 'e2e'):
            results[track] = {}
            for kind in (('shard','range') if track == 'sg' else ('range','shard')):
                guard()
                cache = (RetryVerifiedCache if kind == 'shard' else SampleRangeCache)(cfg['feature_cache_bytes'], out / 'anomalies')
                # Targets retain the existing verified read path; no validation/test split opened.
                features, targets, _ = load_split(RUN, m, fm, 'train', cache)
                if kind == 'range':
                    features = g4.FeatureStore(*(SampleRangeArray(index[n],cache) for n in NAMES))
                expected_data = hashlib.sha256()
                # Exact data comparison is outside the timed training loop.
                if kind == 'range':
                    reference_cache = RetryVerifiedCache(cfg['feature_cache_bytes'],out / 'anomalies')
                    reference, _, _ = load_split(RUN,m,fm,'train',reference_cache)
                    for indices in batches:
                        guard()
                        for new, old in zip((features.spatial,features.e1,features.d2),
                                            (reference.spatial,reference.e1,reference.d2)):
                            left, right = new[indices.tolist()], old[indices.tolist()]
                            if left.dtype != right.dtype or left.shape != right.shape or left.tobytes() != right.tobytes():
                                raise RuntimeError('Consumed data mismatch')
                            expected_data.update(left.tobytes())
                    del reference, reference_cache, left, right
                    cache = SampleRangeCache(cfg['feature_cache_bytes'],out / 'anomalies')
                    features = g4.FeatureStore(*(SampleRangeArray(index[n],cache) for n in NAMES))
                context = build_context(RUN,m,seed,device)
                initial = g4.state_digest(context)
                optimizer = torch.optim.AdamW(context.parameter_groups,weight_decay=cfg['weight_decay'])
                target_cache = g4.as_cached_targets(targets)
                trace = []; torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(); started = time.perf_counter()
                for i,indices in enumerate(batches[:256],1):
                    guard()
                    loss,norm = step(context,optimizer,features,target_cache,indices,cfg,device,track=='sg')
                    trace.append((loss,norm,digest([p.grad for p in context.parameters])))
                    if i % 64 == 0:
                        print(f'{track}/{kind}: {i}/256',flush=True)
                torch.cuda.synchronize(); elapsed = time.perf_counter()-started
                final_digest = g4.state_digest(context); opt_digest = digest(optimizer.state_dict())
                path = out / f'{track}_{kind}.pt'
                save_checkpoint(path,{'state':g4.state_payload(context),'optimizer':optimizer.state_dict(),'rng':rng_state()})
                next_value = step(context,optimizer,features,target_cache,batches[256],cfg,device,track=='sg')
                next_state = (g4.state_digest(context),digest(optimizer.state_dict()))
                cp = load_checkpoint(path)
                g4.load_state(context,cp['state']); optimizer.load_state_dict(cp['optimizer']); restore_rng(cp['rng'])
                restored = step(context,optimizer,features,target_cache,batches[256],cfg,device,track=='sg')
                assert next_value == restored and next_state == (g4.state_digest(context),digest(optimizer.state_dict()))
                results[track][kind] = {'seconds':elapsed,'initial':initial,'trace':trace,'final':final_digest,
                    'optimizer':opt_digest,'resume_exact':True,'cache':dict(cache.stats),'rss':psutil.Process().memory_info().rss,
                    'cuda_peak':torch.cuda.max_memory_allocated(),'data_digest':expected_data.hexdigest() if kind=='range' else None}
                g4.write_json(out / f'{track}_{kind}.json',results[track][kind])
                del context,optimizer,features,cache,cp,targets,target_cache
                gc.collect(); torch.cuda.empty_cache()
            a,b = results[track]['shard'],results[track]['range']
            assert all(a[k]==b[k] for k in ('initial','trace','final','optimizer','resume_exact'))
        verify_code(RUN)
        for row in identities:
            verified_read(row,out / 'anomalies')
        g4.write_json(out/'report.json',{'status':'EQUIVALENT','tracks':results,
            'speed_ratio_range_over_shard':{t:results[t]['range']['seconds']/results[t]['shard']['seconds'] for t in results},
            'test_executed':False,'formal_training_resumed':False,'prefetch':False})
        print(f'工程验证完成：{out}',flush=True)
    except BaseException as exc:
        g4.write_json(out/'failure.json',{'status':'STOP','error':repr(exc),'test_executed':False})
        raise


if __name__ == '__main__':
    main()
