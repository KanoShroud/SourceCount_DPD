"""有界单生产者预取及batch吞吐筛选；不修改正式G5合同。"""

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[4]))
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import time

import psutil
import torch

from 统一模型代码.gates.g5.diagnostics.G5样本读取验证 import ROOT, RUN, NAMES, digest
from 统一模型代码.common.g5_sample_range import SampleRangeArray, SampleRangeCache
from 统一模型代码.gates.g5.e2e_g5_contract import verify_code
from 统一模型代码.gates.g5.e2e_g5_model import build_context, g4, load_split
from 统一模型代码.gates.g5.audits.e2e_g5_pilot import step
from 统一模型代码.common.g5_verified_io import verified_read


class BatchArray:
    def __init__(self, indices, array):
        self.indices = indices.tolist()
        self.array = array

    def __getitem__(self, indices):
        if list(indices) != self.indices:
            raise RuntimeError('Prefetch sample order mismatch')
        return self.array


def prepared(features, batches, prefetch, consumed):
    def read(indices):
        arrays = [a[indices.tolist()] for a in (features.spatial, features.e1, features.d2)]
        for array in arrays:
            consumed.update(array.tobytes())
        return g4.FeatureStore(*(BatchArray(indices, a) for a in arrays))
    if not prefetch:
        for indices in batches:
            yield indices, read(indices)
        return
    # Only the worker touches the LRU. At most one future batch beside the current batch.
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(read, batches[0])
        for i, indices in enumerate(batches):
            data = future.result()  # Worker exceptions propagate; verified_read retains evidence.
            future = pool.submit(read, batches[i+1]) if i+1 < len(batches) else None
            yield indices, data


def main():
    for proc in psutil.process_iter(['cmdline']):
        if any(Path(a).name == 'e2e_g5_train.py' for a in (proc.info['cmdline'] or [])):
            raise RuntimeError('Formal training is active')
    verify_code(RUN)
    m = g4.read_json(RUN/'manifest.json')
    fm = g4.read_json(RUN/'feature_manifest.json')
    prior = ROOT/'outputs_e2e/unified/g5_sample_range/20260919_223427'
    index = json.loads(verified_read(g4.read_json(prior/'sample_index.identity.json'), prior/'anomalies'))
    for name in NAMES:
        for row in index[name]:
            if not Path(row['path']).resolve(strict=True).is_relative_to((RUN/'features/train').resolve()):
                raise RuntimeError('Unexpected feature root')
    out = ROOT/'outputs_e2e/unified/g5_prefetch_probe'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    sources = [g4.identity(ROOT/n) for n in ('统一模型代码/gates/g5/diagnostics/g5_prefetch_probe.py','统一模型代码/common/g5_sample_range.py','统一模型代码/gates/g5/diagnostics/G5样本读取验证.py')]
    start = time.perf_counter()
    def guard():
        vm = psutil.virtual_memory()
        if vm.percent >= 85:
            raise RuntimeError('RAM_WARNING_85_PERCENT: probe paused by stopping, no further allocation')
        if time.perf_counter()-start > 1800 or psutil.disk_usage(str(out)).free < 50*2**30:
            raise RuntimeError('Probe time/disk limit')
        return vm.percent
    protocol = dict(batch_sizes=[4,8,16,32],warmup_samples=128,timed_samples=512,
                    repeats=2,cache_bytes=2*2**30,ram_warning_percent=85,
                    ram_policy='stop probe at warning; formal historical contract unchanged',
                    source_identities=sources,test_executed=False,formal_training_resumed=False,
                    precision='unchanged FP32',learning_rates='unchanged',seed=20260921)
    g4.write_json(out/'protocol.json',protocol)
    rows = []
    rejected = []
    unsafe_from = {}
    try:
        # Pre/post full shard checks supplement per-consumption sample hashes.
        def identity_check():
            for name in NAMES:
                for row in fm['files']['train'][name]:
                    guard(); verified_read(row,out/'anomalies')
            verified_read(fm['files']['train']['targets'],out/'anomalies')
        identity_check()
        device = torch.device('cuda:0'); torch.cuda.init()
        order = torch.randperm(4096,generator=torch.Generator().manual_seed(20260921))[:640]
        for repeat in range(2):
            for track in ('sg','e2e'):
                sizes = [4,8,16,32] if repeat == 0 else [32,16,8,4]
                for batch in sizes:
                    if batch >= unsafe_from.get(track, 1000):
                        rejected.append(dict(repeat=repeat,track=track,batch=batch,
                                             reason='larger_or_equal_to_resource_rejected_batch',tested=False))
                        continue
                    pair = []
                    for prefetch in ([False,True] if repeat == 0 else [True,False]):
                        guard()
                        cache = SampleRangeCache(2*2**30,out/'anomalies')
                        _,targets,_ = load_split(RUN,m,fm,'train',cache)
                        features = g4.FeatureStore(*(SampleRangeArray(index[n],cache) for n in NAMES))
                        context = build_context(RUN,m,20260921,device)
                        optimizer = torch.optim.AdamW(context.parameter_groups,weight_decay=m['config']['weight_decay'])
                        target_cache = g4.as_cached_targets(targets)
                        trace = []; consumed = hashlib.sha256(); ram_peak = guard(); rss_peak = 0
                        torch.cuda.reset_peak_memory_stats()
                        timed = None
                        iterator = prepared(features,list(order.split(batch)),prefetch,consumed)
                        try:
                            for i,(indices,data) in enumerate(iterator):
                                ram_peak = max(ram_peak,guard())
                                rss_peak = max(rss_peak,psutil.Process().memory_info().rss)
                                trace.append(step(context,optimizer,data,target_cache,indices,m['config'],device,track=='sg'))
                                if i+1 == 128//batch:
                                    torch.cuda.synchronize(); timed = time.perf_counter()
                            torch.cuda.synchronize()
                            seconds = time.perf_counter()-timed
                        finally:
                            iterator.close()
                        row = dict(repeat=repeat,track=track,batch=batch,prefetch=prefetch,
                                   seconds=seconds,samples_per_second=512/seconds,trace=trace,
                                   state=digest(g4.state_payload(context)),optimizer=digest(optimizer.state_dict()),
                                   consumed_sha256=consumed.hexdigest(),ram_percent_peak=ram_peak,rss_peak=rss_peak,
                                   cuda_allocated_peak=torch.cuda.max_memory_allocated(),
                                   cuda_reserved_peak=torch.cuda.max_memory_reserved(),cache=dict(cache.stats))
                        if row['cuda_reserved_peak'] > 14*2**30:
                            row['status'] = 'REJECTED_GPU_RESERVE'
                            rejected.append(row)
                            unsafe_from[track] = batch
                            g4.write_json(out/f'{repeat}_{track}_b{batch}_rejected.json',row)
                            print(f'{track}/batch={batch}: exceeds 14 GiB reserved; reject candidate',flush=True)
                            del context,optimizer,features,cache,targets,target_cache,data,iterator
                            gc.collect(); torch.cuda.empty_cache()
                            break
                        rows.append(row); pair.append(row)
                        g4.write_json(out/f'{repeat}_{track}_b{batch}_p{int(prefetch)}.json',row)
                        print(json.dumps({k:row[k] for k in ('repeat','track','batch','prefetch','samples_per_second','cuda_allocated_peak')}) ,flush=True)
                        del context,optimizer,features,cache,targets,target_cache,data,iterator
                        gc.collect(); torch.cuda.empty_cache()
                    if len(pair) == 2 and any(pair[0][k] != pair[1][k] for k in ('trace','state','optimizer','consumed_sha256')):
                        raise RuntimeError('Prefetch equivalence failed')
        identity_check(); verify_code(RUN)
        for source in sources:
            verified_read(source,out/'anomalies')
        g4.write_json(out/'report.json',dict(status='PASS',rows=rows,rejected=rejected,protocol=protocol,
                                          prefetch_exact=True,input_identity_post='PASS',
                                          scientific_convergence_tested=False))
        print(f'完成：{out}',flush=True)
    except BaseException as exc:
        g4.write_json(out/'failure.json',dict(status='STOP',error=repr(exc),completed_arms=len(rows)))
        raise


if __name__ == '__main__':
    main()
