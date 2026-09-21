"""按实际消费字节校验的有界只读NPY特征加载器，不改变batch顺序。"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
from pathlib import Path
import time

import numpy as np


def decode_npy(payload: bytes) -> np.ndarray:
    """仅支持本项目无对象dtype的NPY v1/v2；数组直接引用已校验的bytes。"""
    stream = io.BytesIO(payload)
    version = np.lib.format.read_magic(stream)
    reader = {(1, 0): np.lib.format.read_array_header_1_0,
              (2, 0): np.lib.format.read_array_header_2_0}.get(version)
    if reader is None:
        raise ValueError(f"Unsupported NPY version: {version}")
    shape, fortran, dtype = reader(stream)
    if dtype.hasobject:
        raise ValueError("Object arrays are forbidden")
    offset = stream.tell()
    count = int(np.prod(shape, dtype=np.int64))
    if len(payload) != offset + count * dtype.itemsize:
        raise ValueError("NPY payload length mismatch")
    array = np.frombuffer(payload, dtype=dtype, count=count, offset=offset)
    return array.reshape(shape, order="F" if fortran else "C")


class VerifiedCache:
    def __init__(self, capacity_bytes: int, anomaly_dir: Path):
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity = capacity_bytes
        self.anomaly_dir = Path(anomaly_dir)
        self.items = OrderedDict()
        self.bytes = 0
        self.stats = dict(hits=0, misses=0, read_bytes=0, peak_bytes=0,
                          read_seconds=0.0, sha_seconds=0.0, parse_seconds=0.0)

    def get(self, row: dict) -> np.ndarray:
        key = (str(Path(row["path"]).resolve()), row["sha256"], int(row["size_bytes"]))
        if key in self.items:
            self.stats["hits"] += 1
            self.items.move_to_end(key)
            return self.items[key][0]
        size = key[2]
        if size > self.capacity:
            raise ValueError("Single shard exceeds configured cache capacity")
        while self.items and self.bytes + size > self.capacity:
            _, (_, removed_size) = self.items.popitem(last=False)
            self.bytes -= removed_size
        self.stats["misses"] += 1
        start = time.perf_counter()
        payload = Path(key[0]).read_bytes()
        self.stats["read_seconds"] += time.perf_counter() - start
        self.stats["read_bytes"] += len(payload)
        start = time.perf_counter()
        actual = hashlib.sha256(payload).hexdigest()
        self.stats["sha_seconds"] += time.perf_counter() - start
        if len(payload) != size or actual != key[1]:
            self.anomaly_dir.mkdir(parents=True, exist_ok=True)
            path = self.anomaly_dir / f"read_{time.time_ns()}_{actual}.bin"
            with path.open("xb") as handle:
                handle.write(payload)
            raise RuntimeError(f"Input identity mismatch: {key[0]}; captured at {path}")
        start = time.perf_counter()
        array = decode_npy(payload)
        self.stats["parse_seconds"] += time.perf_counter() - start
        if array.flags.writeable:
            raise RuntimeError("Decoded cache must be immutable")
        self.items[key] = (array, size)
        self.bytes += size
        self.stats["peak_bytes"] = max(self.stats["peak_bytes"], self.bytes)
        return array


class VerifiedShardedArray:
    def __init__(self, rows: list[dict], cache: VerifiedCache):
        self.rows = rows
        self.cache = cache
        self.lookup = {}
        self.copy_seconds = 0.0
        for row_index, row in enumerate(rows):
            for index in range(int(row["start"]), int(row["stop"])):
                if index in self.lookup:
                    raise ValueError("Overlapping sample indices")
                self.lookup[index] = row_index

    def take(self, indices: list[int]) -> np.ndarray:
        if not indices:
            raise ValueError("Empty batch")
        grouped = {}
        for output_index, index in enumerate(indices):
            row_index = self.lookup[int(index)]
            grouped.setdefault(row_index, []).append((output_index, int(index)))
        result = None
        for row_index, entries in grouped.items():
            row = self.rows[row_index]
            array = self.cache.get(row)
            start = time.perf_counter()
            if result is None:
                result = np.empty((len(indices), *array.shape[1:]), dtype=array.dtype)
            for output_index, index in entries:
                result[output_index] = array[index - row["start"]]
            self.copy_seconds += time.perf_counter() - start
            del array
        return result

    def __getitem__(self, indices):
        return self.take(list(indices))
