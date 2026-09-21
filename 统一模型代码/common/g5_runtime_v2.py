"""G5工程v2：逐样本验证、单生产者预取；不改变科学配置。"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import json
from pathlib import Path

import psutil

from 统一模型代码.common.g5_sample_range import SampleRangeArray, SampleRangeCache
from 统一模型代码.common.g5_verified_io import verified_read

NAMES = ('ch3_spatial', 'd8_e1', 'd8_d2')


def enabled(run):
    return (Path(run)/'engineering_v2/index_registry.json').is_file()


def ram_guard():
    percent = psutil.virtual_memory().percent
    if percent >= 85:
        raise RuntimeError(f'RAM_WARNING_85_PERCENT: system used {percent:.1f}%; stop before further allocation')


@lru_cache(maxsize=6)
def sample_index(run, split):
    run = Path(run).resolve(strict=True)
    registry = json.loads((run/'engineering_v2/index_registry.json').read_text(encoding='utf-8'))
    index = json.loads(verified_read(registry['indexes'][split],run/'anomalies'))
    for name in NAMES:
        for row in index[name]:
            if not Path(row['path']).resolve(strict=True).is_relative_to(run/'features'/split):
                raise RuntimeError('Range index path escaped frozen feature root')
    return index


def load_features(run, split, capacity, cache=None):
    from 统一模型代码.gates.g5.e2e_g5_model import g4
    if cache is not None and not isinstance(cache, SampleRangeCache):
        raise TypeError('Engineering v2 requires shared SampleRangeCache')
    cache = cache if cache is not None else SampleRangeCache(capacity,Path(run)/'anomalies')
    index = sample_index(str(Path(run).resolve()),split)
    return g4.FeatureStore(*(SampleRangeArray(index[n],cache) for n in NAMES)), cache


class BatchArray:
    def __init__(self, indices, array):
        self.indices, self.array = indices.tolist(), array

    def __getitem__(self, indices):
        if list(indices) != self.indices:
            raise RuntimeError('Prefetched sample order mismatch')
        return self.array


def batches(features, indices, prefetch=True):
    """Cache is worker-owned until iterator.close(); pending data never enters checkpoint state."""
    from 统一模型代码.gates.g5.e2e_g5_model import g4
    def read(ids):
        ram_guard()
        arrays = [a[ids.tolist()] for a in (features.spatial,features.e1,features.d2)]
        return g4.FeatureStore(*(BatchArray(ids,a) for a in arrays))
    if not prefetch:
        for ids in indices:
            yield ids, read(ids)
        return
    if not len(indices):
        return
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(read,indices[0])
    try:
        for i,ids in enumerate(indices):
            data = future.result()
            future = pool.submit(read,indices[i+1]) if i+1 < len(indices) else None
            yield ids,data
    finally:
        # Drain instead of suppressing a pending worker error during cancellation/checkpoint exit.
        try:
            if future is not None:
                future.result()
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
