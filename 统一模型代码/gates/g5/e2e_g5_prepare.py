"""G5只读源接管及分块身份登记；不读取test。"""
from __future__ import annotations

import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import h5py
import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g4 import e2e_g4 as g4  # noqa: E402
from 统一模型代码.common.diagnostics import cache_read_path_probe as win  # noqa: E402
from 统一模型代码.common.g5_verified_io import VerifiedFile, event  # noqa: E402

SOURCE = ROOT / "outputs_e2e/unified/e2e_g4/20260905_164500"
CONFIG = ROOT / "统一模型代码/configs/e2e_g5.json"
BLOCK = 8 * 1024 * 1024


def direct_blocks(path):
    handle = win.open_read(path, True)
    address = None
    try:
        values = [w.DWORD() for _ in range(4)]
        if not win.diskspace(path.anchor, *[c.byref(v) for v in values]):
            raise c.WinError(c.get_last_error())
        sector = values[1].value
        address = win.alloc(None, BLOCK + 65536, 0x3000, 4)
        if not address:
            raise c.WinError(c.get_last_error())
        aligned = (address + 65535) // 65536 * 65536
        if aligned % sector or BLOCK % sector:
            raise RuntimeError("Unsupported alignment")
        remaining = path.stat().st_size
        while remaining:
            wanted = min(BLOCK, ((remaining + sector - 1) // sector) * sector)
            got = w.DWORD()
            if not win.read(handle, aligned, wanted, c.byref(got), None):
                raise c.WinError(c.get_last_error())
            used = min(got.value, remaining)
            if not used:
                raise RuntimeError("Short direct read")
            yield c.string_at(aligned, used)
            remaining -= used
    finally:
        if address:
            win.free(address, 0, 0x8000)
        win.close(handle)


def guard(run):
    if psutil.virtual_memory().available < 8 * 2**30:
        raise RuntimeError("RAM available below 8 GiB")
    if shutil.disk_usage(run).free < 50 * 2**30:
        raise RuntimeError("Disk free below 50 GiB")


def snapshot(row, target, run):
    path = Path(row["path"]).resolve(strict=True)
    if not path.is_relative_to(ROOT / "outputs_e2e"):
        raise ValueError("Use previously registered local snapshot only")
    for attempt in (1, 2):
        temporary = target.with_suffix(f".attempt{attempt}.pending")
        blocks, digest, size = [], hashlib.sha256(), 0
        try:
            with temporary.open("xb") as output:
                for payload in direct_blocks(path):
                    guard(run)
                    output.write(payload)
                    digest.update(payload)
                    size += len(payload)
                    blocks.append({"size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
                output.flush()
                os.fsync(output.fileno())
            if size != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("Source full-file identity mismatch")
        except (OSError, ValueError, RuntimeError) as exc:
            event(run / "anomalies", {"status": "SNAPSHOT_READ_FAILED", "attempt": attempt,
                                      "path": str(path), "captured": str(temporary),
                                      "size_bytes": size, "actual_sha256": digest.hexdigest(),
                                      "expected": row, "error": repr(exc)})
            if attempt == 2 or str(exc).startswith(("RAM", "Disk")):
                raise
        else:
            temporary.rename(target)
            if attempt == 2:
                event(run / "anomalies", {"status": "RECOVERED_ONCE", "path": str(path), "sha256": digest.hexdigest()})
            return {"path": str(target), "size_bytes": size, "sha256": digest.hexdigest(),
                    "block_size": BLOCK, "blocks": blocks, "source": row,
                    "read_mode": "FILE_FLAG_NO_BUFFERING"}
    raise AssertionError("unreachable")


def configure(run, inputs):
    g4.configure_recorded_snapshot({"inputs": inputs})
    registered = {str(Path(row["path"]).resolve()): row for row in inputs["files"]}
    original = g4.g1.SampleStore

    class Store(original):
        def __init__(self, split):
            self.split = split
            self.streams = []

            def opened(path):
                row = registered[str(path.resolve())]
                stream = VerifiedFile(row, run / "anomalies")
                self.streams.append(stream)
                return h5py.File(stream, "r")

            if split == "train":
                self.coarse_path = g4.g1.COARSE_TRAIN
                parts = g4.g1.RAW_TRAIN_PARTS
            elif split in ("val_select", "val_compare"):
                self.coarse_path = g4.g1.COARSE_VAL_SELECT if split == "val_select" else g4.g1.COARSE_VAL_COMPARE
                parts = [(0, 2048, g4.g1.RAW_VALIDATION)]
            else:
                raise ValueError("test and unknown splits forbidden")
            self.raw_parts = [(lo, hi, opened(path)) for lo, hi, path in parts]
            self.coarse = opened(self.coarse_path)

        def close(self):
            super().close()
            for stream in self.streams:
                stream.close()

    g4.g1.SampleStore = Store


def prepare(run):
    run = run.resolve()
    if not run.is_relative_to(ROOT / "outputs_e2e/unified/e2e_g5"):
        raise ValueError("G5 output isolation violation")
    run.mkdir(parents=True, exist_ok=False)
    started = time.time()
    report = {"status": "PREPARING", "created_at": started, "test_executed": False,
              "config": g4.read_json(CONFIG), "source_g4": g4.identity(SOURCE / "manifest.json")}
    g4.write_json(run / "preparation_status.json", report)
    try:
        source = g4.read_json(SOURCE / "manifest.json")
        snapshots = run / "input_snapshot"
        snapshots.mkdir()
        inputs = {"files": [], "artifacts": []}
        for kind in inputs:
            for index, row in enumerate(source["inputs"][kind]):
                target = snapshots / f"{kind}_{index:02d}_{Path(row['path']).name}"
                result = snapshot(row, target, run)
                if "name" in row:
                    result["name"] = row["name"]
                inputs[kind].append(result)
                g4.write_json(run / "input_snapshot_progress.json", inputs)
                print(f"verified snapshot {kind}/{index}: {result['size_bytes']} bytes", flush=True)
        g4.write_json(run / "input_manifest.json", inputs)
        configure(run, inputs)
        subsets, pools = {}, {}
        for split, count in report["config"]["counts"].items():
            pool = g4.all_records(split)
            pools[split] = len(pool)
            old = source["subsets"][split] if split != "val_compare" else []
            old_ids = {int(row["local_index"]) for row in old}
            candidates = [row for row in pool if row["local_index"] not in old_ids]
            per_k = (count - len(old)) // 4
            selected = old + g4.choose_balanced(candidates, per_k, report["config"]["data_seed"])
            selected.sort(key=lambda row: (row["true_k"], row["local_index"]))
            subsets[split] = g4.annotate(selected, split)
            assert len(selected) == count
            assert all(sum(x["true_k"] == k for x in selected) == count // 4 for k in range(4))
            print(f"selected {split}: {len(selected)} / {len(pool)}", flush=True)
        assert not ({row["raw_index"] for row in subsets["val_select"]} &
                    {row["raw_index"] for row in subsets["val_compare"]})
        # train与validation的raw_index分别属于不同母文件，不将局部索引误当全局ID。
        raw_train = {Path(part[2]).resolve() for part in g4.g1.RAW_TRAIN_PARTS}
        assert g4.g1.RAW_VALIDATION.resolve() not in raw_train
        report.update(status="INPUTS_PREPARED", inputs=inputs, subsets=subsets, pool_sizes=pools,
                      pretraining_overlap_audit="PENDING_PROVENANCE_AUDIT",
                      seconds=time.time() - started,
                      code=[g4.identity(Path(__file__)), g4.identity(CONFIG)])
        g4.write_json(run / "manifest.json", report)
    except Exception as exc:
        report.update(status="STOP", error=repr(exc), seconds=time.time() - started)
        raise
    finally:
        g4.write_json(run / "preparation_status.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    report = prepare(args.run)
    print(json.dumps({k: report[k] for k in ("status", "seconds", "pool_sizes")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
