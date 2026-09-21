"""G5输入：最多重读一次，保留异常字节；校验与消费同一缓冲。"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time

from 统一模型代码.common.verified_feature_loader import VerifiedCache, decode_npy


def event(directory: Path, payload: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time_ns": time.time_ns(), **payload}, ensure_ascii=False) + "\n")
        handle.flush()


def verified_read(row: dict, directory: Path, *, offset: int = 0,
                  length: int | None = None, attempts: int = 2) -> bytes:
    """登记身份不匹配或IO异常才重读；成功内容就是返回给解码器的内容。"""
    if attempts not in (1, 2):
        raise ValueError("Only zero or one retry is permitted")
    path = Path(row["path"]).resolve()
    for attempt in range(1, attempts + 1):
        payload = b""
        try:
            with path.open("rb") as handle:
                handle.seek(offset)
                payload = handle.read() if length is None else handle.read(length)
            actual = hashlib.sha256(payload).hexdigest()
            if len(payload) != row["size_bytes"] or actual != row["sha256"]:
                raise ValueError(f"Identity mismatch: {actual}, {len(payload)} bytes")
        except (OSError, ValueError) as exc:
            directory.mkdir(parents=True, exist_ok=True)
            capture = directory / f"abnormal_{time.time_ns()}_{attempt}.bin"
            with capture.open("xb") as handle:
                handle.write(payload)
            try:
                stat = path.stat()
                file_state = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "file_id": stat.st_ino}
            except OSError as stat_error:
                file_state = {"error": repr(stat_error)}
            event(directory, {"status": "READ_FAILED", "path": str(path), "file_state": file_state,
                              "offset": offset, "attempt": attempt,
                              "expected": row, "error": repr(exc),
                              "captured_sha256": hashlib.sha256(payload).hexdigest(),
                              "captured": str(capture), "captured_bytes": len(payload),
                              "note": "IO exception may expose no bytes"})
            if attempt == attempts:
                raise RuntimeError(f"Input failed after {attempts} attempts: {path}") from exc
        else:
            if attempt > 1:
                event(directory, {"status": "RECOVERED_ONCE", "path": str(path),
                                  "offset": offset, "sha256": actual, "size_bytes": len(payload)})
            return payload
    raise AssertionError("unreachable")


class RetryVerifiedCache(VerifiedCache):
    """G5专用策略；不改变旧加载器与历史实验合同。"""

    def get(self, row: dict):
        key = (str(Path(row["path"]).resolve()), row["sha256"], int(row["size_bytes"]))
        if key in self.items:
            self.stats["hits"] += 1
            self.items.move_to_end(key)
            return self.items[key][0]
        size = key[2]
        if size > self.capacity:
            raise ValueError("Single shard exceeds cache capacity")
        while self.items and self.bytes + size > self.capacity:
            _, (_, old_size) = self.items.popitem(last=False)
            self.bytes -= old_size
        self.stats["misses"] += 1
        start = time.perf_counter()
        payload = verified_read(row, self.anomaly_dir)
        self.stats["read_seconds"] += time.perf_counter() - start
        self.stats["read_bytes"] += len(payload)
        array = decode_npy(payload)
        if array.flags.writeable:
            raise RuntimeError("Feature array must be read-only")
        self.items[key] = (array, size)
        self.bytes += size
        self.stats["peak_bytes"] = max(self.bytes, self.stats["peak_bytes"])
        return array


class VerifiedFile(io.RawIOBase):
    """供h5py fileobj驱动使用：按已登记块SHA懒读取，避免整个MAT常驻RAM。"""

    def __init__(self, manifest: dict, anomaly_dir: Path):
        super().__init__()
        self.manifest = manifest
        self.anomaly_dir = anomaly_dir
        self.position = 0
        self.block_index = None
        self.block = b""

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        base = {0: 0, 1: self.position, 2: self.manifest["size_bytes"]}[whence]
        position = base + offset
        if position < 0:
            raise ValueError("Negative seek")
        self.position = position
        return position

    def readinto(self, buffer):
        total = min(len(buffer), max(0, self.manifest["size_bytes"] - self.position))
        copied = 0
        while copied < total:
            index, local = divmod(self.position, self.manifest["block_size"])
            if index != self.block_index:
                row = self.manifest["blocks"][index]
                self.block = verified_read({"path": self.manifest["path"], **row},
                                           self.anomaly_dir,
                                           offset=index * self.manifest["block_size"],
                                           length=row["size_bytes"])
                self.block_index = index
            size = min(total - copied, len(self.block) - local)
            buffer[copied:copied + size] = self.block[local:local + size]
            copied += size
            self.position += size
        return copied
