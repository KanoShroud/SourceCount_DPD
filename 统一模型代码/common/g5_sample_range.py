"""独立候选：从已验证C序NPY分片登记逐样本范围，消费同一已校验字节。"""
import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.common.verified_feature_loader import VerifiedCache


def register_shard(row, anomaly_dir):
    payload = verified_read(row, anomaly_dir)
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    readers = {(1, 0): np.lib.format.read_array_header_1_0,
               (2, 0): np.lib.format.read_array_header_2_0}
    if version not in readers:
        raise ValueError('Unsupported NPY version')
    shape, fortran, dtype = readers[version](stream)
    if fortran or dtype.hasobject or dtype.fields or len(shape) < 2:
        raise ValueError('Only plain C-order numeric sample arrays are supported')
    start = stream.tell()
    size = math.prod(shape[1:]) * dtype.itemsize
    if shape[0] != row['stop'] - row['start'] or start + size * shape[0] != len(payload):
        raise ValueError('NPY shape/length/registration mismatch')
    return [dict(path=str(Path(row['path']).resolve(strict=True)), index=row['start'] + local,
                 offset=start + local * size, size_bytes=size, shape=list(shape[1:]), dtype=dtype.str,
                 sha256=hashlib.sha256(memoryview(payload)[start+local*size:start+(local+1)*size]).hexdigest(),
                 parent_sha256=row['sha256']) for local in range(shape[0])]


class SampleRangeCache(VerifiedCache):
    def get(self, row):
        key = (row['path'], row['offset'], row['size_bytes'], row['sha256'],
               tuple(row['shape']), row['dtype'])
        if key in self.items:
            self.stats['hits'] += 1
            self.items.move_to_end(key)
            return self.items[key][0]
        size = row['size_bytes']
        if size <= 0 or size > self.capacity:
            raise ValueError('Invalid sample size/cache capacity')
        while self.items and self.bytes + size > self.capacity:
            _, (_, removed) = self.items.popitem(last=False)
            self.bytes -= removed
        begin = time.perf_counter()
        payload = verified_read(row, self.anomaly_dir, offset=row['offset'], length=size)
        self.stats['read_seconds'] += time.perf_counter() - begin
        self.stats['misses'] += 1
        self.stats['read_bytes'] += size
        array = np.frombuffer(payload, dtype=np.dtype(row['dtype'])).reshape(row['shape'])
        if array.flags.writeable:
            raise RuntimeError('Range buffer must be immutable')
        self.items[key] = (array, size)
        self.bytes += size
        self.stats['peak_bytes'] = max(self.bytes, self.stats['peak_bytes'])
        return array


class SampleRangeArray:
    def __init__(self, rows, cache):
        self.rows = {row['index']: row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError('Duplicate sample index')
        self.cache = cache

    def __getitem__(self, indices):
        indices = [int(i) for i in indices]
        if not indices:
            raise ValueError('Empty batch')
        first = self.cache.get(self.rows[indices[0]])
        result = np.empty((len(indices), *first.shape), dtype=first.dtype)
        result[0] = first
        del first
        for i, index in enumerate(indices[1:], 1):
            result[i] = self.cache.get(self.rows[index])
        return result
