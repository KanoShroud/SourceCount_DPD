"""G5一次重读策略与HDF5按块读取的轻量契约测试。"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.common.g5_verified_io import RetryVerifiedCache, VerifiedFile, verified_read  # noqa: E402


def main():
    out = Path(tempfile.mkdtemp(prefix="g5_io_smoke_", dir=ROOT / "outputs_e2e"))
    source = out / "feature.npy"
    np.save(source, np.arange(24, dtype=np.float32).reshape(4, 6))
    good = source.read_bytes()
    row = {"path": str(source), "size_bytes": len(good), "sha256": hashlib.sha256(good).hexdigest()}
    real_open = Path.open
    counter = [0]

    def first_bad(path, *args, **kwargs):
        if path == source and args and args[0] == "rb":
            counter[0] += 1
            return io.BytesIO(b"wrong" if counter[0] == 1 else good)
        return real_open(path, *args, **kwargs)

    with patch.object(Path, "open", first_bad):
        actual = verified_read(row, out / "recover")
    assert actual == good and counter[0] == 2
    events = [json.loads(x) for x in (out / "recover/events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["status"] for x in events] == ["READ_FAILED", "RECOVERED_ONCE"]
    assert Path(events[0]["captured"]).read_bytes() == b"wrong"
    counter[0] = 0

    def always_bad(path, *args, **kwargs):
        if path == source and args and args[0] == "rb":
            counter[0] += 1
            return io.BytesIO(b"bad")
        return real_open(path, *args, **kwargs)

    with patch.object(Path, "open", always_bad):
        try:
            verified_read(row, out / "stop")
        except RuntimeError:
            pass
        else:
            raise AssertionError("Must stop after two bad reads")
    assert counter[0] == 2
    cache = RetryVerifiedCache(1024, out / "cache")
    arr = cache.get(row)
    assert not arr.flags.writeable and np.array_equal(arr, np.arange(24).reshape(4, 6))
    assert cache.get(row) is arr and cache.stats["hits"] == 1
    hdf = out / "sample.h5"
    values = np.arange(1000, dtype=np.float32).reshape(100, 10)
    with h5py.File(hdf, "w") as handle:
        handle.create_dataset("value", data=values, compression="gzip", chunks=(10, 10))
    payload = hdf.read_bytes()
    block_size = 1024
    blocks = [{"size_bytes": len(payload[i:i+block_size]),
               "sha256": hashlib.sha256(payload[i:i+block_size]).hexdigest()}
              for i in range(0, len(payload), block_size)]
    manifest = {"path": str(hdf), "size_bytes": len(payload), "block_size": block_size, "blocks": blocks}
    with VerifiedFile(manifest, out / "hdf_anomaly") as stream:
        with h5py.File(stream, "r") as handle:
            assert np.array_equal(handle["value"][::3], values[::3])
    report = {"status": "PASS", "retry_exactly_once": True, "bad_bytes_preserved": True,
              "immutable_cache": True, "hdf_lazily_verified": True, "中文": "日志编码通过"}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
