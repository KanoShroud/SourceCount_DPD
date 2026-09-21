"""小范围只读比较文件读取路径；异常字节仅保存到全新审计目录。"""
from __future__ import annotations

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[3]))

import argparse
import ctypes as c
from ctypes import wintypes as w
from datetime import datetime
import hashlib
import io
import json
import mmap
from pathlib import Path
import time

import numpy as np
import psutil

K = c.WinDLL("kernel32", use_last_error=True)


def api(name, result, arguments):
    fn = getattr(K, name)
    fn.restype, fn.argtypes = result, arguments
    return fn


create = api("CreateFileW", w.HANDLE, [w.LPCWSTR, w.DWORD, w.DWORD, c.c_void_p,
                                    w.DWORD, w.DWORD, w.HANDLE])
close = api("CloseHandle", w.BOOL, [w.HANDLE])
read = api("ReadFile", w.BOOL, [w.HANDLE, c.c_void_p, w.DWORD,
                              c.POINTER(w.DWORD), c.c_void_p])
alloc = api("VirtualAlloc", c.c_void_p, [c.c_void_p, c.c_size_t, w.DWORD, w.DWORD])
free = api("VirtualFree", w.BOOL, [c.c_void_p, c.c_size_t, w.DWORD])
diskspace = api("GetDiskFreeSpaceW", w.BOOL, [w.LPCWSTR] + [c.POINTER(w.DWORD)] * 4)


def open_read(path, direct=False):
    # GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING. Deny concurrent write/delete.
    handle = create(str(path), 0x80000000, 1, None, 3,
                    0x20000000 if direct else 0x80, None)
    if handle == c.c_void_p(-1).value:
        raise c.WinError(c.get_last_error())
    return handle


def direct_bytes(path):
    handle = open_read(path, True)
    address = None
    try:
        values = [w.DWORD() for _ in range(4)]
        if not diskspace(path.anchor, *[c.byref(v) for v in values]):
            raise c.WinError(c.get_last_error())
        sector = values[1].value
        block = 8 * 1024 * 1024
        address = alloc(None, block + 65536, 0x3000, 4)
        if not address:
            raise c.WinError(c.get_last_error())
        aligned = (address + 65535) // 65536 * 65536
        if aligned % sector or block % sector:
            raise RuntimeError("Unsupported sector alignment")
        remaining = path.stat().st_size
        parts = []
        while remaining:
            wanted = min(block, (remaining + sector - 1) // sector * sector)
            got = w.DWORD()
            if not read(handle, aligned, wanted, c.byref(got), None):
                raise c.WinError(c.get_last_error())
            used = min(got.value, remaining)
            if not used:
                raise RuntimeError("Unexpected short read")
            parts.append(c.string_at(aligned, used))
            remaining -= used
        return b"".join(parts), {"logical_sector_bytes": sector,
                                 "buffer_alignment_bytes": 65536,
                                 "flag": "FILE_FLAG_NO_BUFFERING"}
    finally:
        if address:
            free(address, 0, 0x8000)
        close(handle)


def stat(path):
    s = path.stat()
    return {"path": str(path.resolve()), "file_id": s.st_ino, "device": s.st_dev,
            "size": s.st_size, "mtime_ns": s.st_mtime_ns, "nlink": s.st_nlink}


def processes(paths):
    matched, unavailable = [], 0
    targets = {str(p).casefold() for p in paths}
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            files = [f.path for f in proc.open_files() if f.path.casefold() in targets]
            if files:
                matched.append({**proc.info, "files": files})
        except (psutil.Error, OSError):
            unavailable += 1
    return {"matched_open_files": matched, "unavailable_processes": unavailable,
            "limitation": "Snapshot only; Windows open_files may omit handles; not write-event tracing or proof of write access."}


def summary(payload):
    result = {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
    try:
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        result.update(shape=list(array.shape), dtype=str(array.dtype),
                      finite=bool(np.isfinite(array).all()),
                      array_sha256=hashlib.sha256(array.tobytes(order="C")).hexdigest())
    except Exception as exc:
        result["decode_error"] = repr(exc)
    return result


def write_json(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def differences(left, right):
    count, first = 0, []
    for start in range(0, min(len(left), len(right)), 1024 * 1024):
        end = min(start + 1024 * 1024, len(left), len(right))
        offsets = np.flatnonzero(np.frombuffer(left[start:end], dtype=np.uint8) !=
                                 np.frombuffer(right[start:end], dtype=np.uint8))
        count += len(offsets)
        first.extend((offsets[:max(0, 32 - len(first))] + start).tolist())
    return {"different_bytes": count + abs(len(left) - len(right)),
            "first_offsets": first, "left_size": len(left), "right_size": len(right)}


def probe(row, destination):
    path = Path(row["path"]).resolve()
    result = {"expected": row, "before": stat(path), "reads": []}
    initial = path.read_bytes()
    baseline = summary(initial)
    result["initial_unlocked"] = baseline
    retained = {}

    def capture(payload, label):
        digest = hashlib.sha256(payload).hexdigest()
        if digest not in retained:
            artifact = destination / f"{label}_{digest[:12]}.bin"
            with artifact.open("xb") as f:
                f.write(payload)
            retained[digest] = str(artifact)

    if baseline["sha256"] != row["sha256"]:
        capture(initial, "initial")
    handle = open_read(path)
    try:
        result["write_delete_exclusion_acquired"] = True
        for cycle in range(3):
            methods = ["ordinary", "mmap", "direct"]
            if cycle == 1:
                methods.reverse()
            for method in methods:
                started = time.perf_counter()
                extra = {}
                if method == "direct":
                    payload, extra = direct_bytes(path)
                elif method == "mmap":
                    with path.open("rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                        payload = mm[:]
                else:
                    payload = path.read_bytes()
                record = {"cycle": cycle, "method": method, **summary(payload), **extra,
                          "seconds": time.perf_counter() - started}
                record["matches_registered"] = record["sha256"] == row["sha256"]
                if not record["matches_registered"] or record["sha256"] != baseline["sha256"]:
                    capture(initial, "initial")
                    capture(payload, f"{cycle}_{method}")
                    record["difference_from_initial"] = differences(initial, payload)
                    record["process_snapshot"] = processes([path])
                result["reads"].append(record)
                write_json(destination / f"read_{cycle}_{method}.json", record)
    finally:
        close(handle)
    result["after"] = stat(path)
    result["identity_unchanged"] = result["before"] == result["after"]
    result["captured_payloads"] = retained
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    workspace = Path(__file__).resolve().parents[3]
    if not output.is_relative_to(workspace / "outputs_e2e"):
        raise ValueError("Output must be inside outputs_e2e")
    output.mkdir(parents=True, exist_ok=False)
    old = json.loads((source / "cache_postcheck_failed_v2.json").read_text(encoding="utf-8"))
    later = json.loads((source / "cache_postcheck_report.json").read_text(encoding="utf-8"))
    selected = [old["mismatches"][0]["expected"]] + [r["expected"] for r in later["mismatches"][:2]]
    manifest = json.loads((source / "feature_cache_v2_manifest.json").read_text(encoding="utf-8"))
    selected.append(manifest["files"]["train"]["ch3_spatial"][0])
    for row in selected:
        if not Path(row["path"]).resolve().is_relative_to(source / "feature_cache_v2"):
            raise ValueError("Input outside selected cache")
    started = time.perf_counter()
    report = {"schema": "cache-read-path-probe-v1", "started": datetime.now().isoformat(),
              "test_executed": False, "training_executed": False,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "processes_before": processes([Path(r["path"]) for r in selected]), "files": []}
    for index, row in enumerate(selected):
        folder = output / f"file_{index}"
        folder.mkdir()
        try:
            result = probe(row, folder)
        except Exception as exc:
            result = {"expected": row, "error": repr(exc), "winerror": getattr(exc, "winerror", None)}
        report["files"].append(result)
        write_json(folder / "report.json", result)
        print(f"读取路径检查 {index + 1}/4: {result.get('error', 'completed')}", flush=True)
    report["processes_after"] = processes([Path(r["path"]) for r in selected])
    report["seconds"] = time.perf_counter() - started
    report["status"] = "NOT_REPRODUCED" if all(
        "error" not in r and r["identity_unchanged"] and
        r["initial_unlocked"]["sha256"] == r["expected"]["sha256"] and
        all(v["matches_registered"] for v in r["reads"])
        for r in report["files"]) else "ANOMALY_OR_PROBE_ERROR"
    write_json(output / "report.json", report)
    print(json.dumps({"status": report["status"], "seconds": report["seconds"], "output": str(output)}))


if __name__ == "__main__":
    main()
