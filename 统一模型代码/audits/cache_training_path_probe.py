"""真实G4 A2训练操作下比较两种懒加载；仅产生隔离的诊断权重。"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import ctypes as c
from ctypes import wintypes as w
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from 统一模型代码 import e2e_g4 as g4  # noqa: E402
from 统一模型代码.audits import cache_read_path_probe as probe  # noqa: E402


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write(path, value):
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


class Watcher(threading.Thread):
    def __init__(self, path, destination):
        super().__init__(daemon=True)
        self.path, self.destination = path, destination
        self.stopping = threading.Event()
        self.events, self.errors = [], []
        self.handle = probe.create(str(path), 1, 7, None, 3, 0x02000000, None)
        if self.handle == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())

    def run(self):
        fn = probe.api("ReadDirectoryChangesW", w.BOOL,
                       [w.HANDLE, c.c_void_p, w.DWORD, w.BOOL, w.DWORD,
                        c.POINTER(w.DWORD), c.c_void_p, c.c_void_p])
        buffer = c.create_string_buffer(65536)
        try:
            with self.destination.open("x", encoding="utf-8") as log:
                while not self.stopping.is_set():
                    got = w.DWORD()
                    ok = fn(self.handle, buffer, len(buffer), True, 0x19,
                            c.byref(got), None, None)
                    if not ok:
                        error = c.get_last_error()
                        if error != 995 or not self.stopping.is_set():
                            self.errors.append({"winerror": error})
                        break
                    if not got.value:
                        self.errors.append({"error": "notification_buffer_overflow"})
                        continue
                    offset = 0
                    while offset < got.value:
                        following, action, length = struct.unpack_from("III", buffer.raw, offset)
                        name = buffer.raw[offset + 12:offset + 12 + length].decode("utf-16-le")
                        event = {"time_ns": time.time_ns(), "action": action,
                                 "path": str(self.path / name)}
                        self.events.append(event)
                        log.write(json.dumps(event, ensure_ascii=False) + "\n")
                        log.flush()
                        if not following:
                            break
                        offset += following
        except Exception as exc:
            self.errors.append({"error": repr(exc)})

    def stop(self):
        self.stopping.set()
        cancel = probe.api("CancelIoEx", w.BOOL, [w.HANDLE, c.c_void_p])
        cancel(self.handle, None)
        self.join(3)
        if self.is_alive():
            self.errors.append({"error": "watcher_stop_timeout"})
        else:
            probe.close(self.handle)
        return {"events": len(self.events), "errors": self.errors,
                "limitation": "Directory notifications, not PID-attributed WriteFile tracing."}


class Cache:
    def __init__(self, limit):
        self.limit, self.bytes, self.peak = limit, 0, 0
        self.items = OrderedDict()
        self.hits, self.misses, self.read_bytes = 0, 0, 0

    def get(self, row, output):
        path = row["path"]
        if path in self.items:
            self.hits += 1
            self.items.move_to_end(path)
            return self.items[path]
        self.misses += 1
        # Evict before loading, while allowing a single new shard temporarily in memory.
        while self.items and self.bytes + row["size_bytes"] > self.limit:
            _, old = self.items.popitem(last=False)
            self.bytes -= old.nbytes
        payload = Path(path).read_bytes()
        self.read_bytes += len(payload)
        if sha(payload) != row["sha256"]:
            artifact = output / f"bad_shard_{time.time_ns()}.bin"
            artifact.write_bytes(payload)
            raise RuntimeError(f"Loaded bytes SHA mismatch: {path}; saved {artifact}")
        array = np.load(io.BytesIO(payload), allow_pickle=False)
        array.setflags(write=False)
        self.items[path] = array
        self.bytes += array.nbytes
        self.peak = max(self.peak, self.bytes)
        if self.bytes > self.limit:
            raise RuntimeError("Shard exceeds cache limit")
        return array


class AuditedArray(g4.ShardedArray):
    def __init__(self, rows, expected, backend, cache, output):
        self.rows, self.expected, self.backend = rows, expected, backend
        self.cache, self.output = cache, output
        self.arrays = ([np.load(r["path"], mmap_mode="r", allow_pickle=False) for r in rows]
                       if backend == "mmap" else [])
        self.starts, self.stops = [r["start"] for r in rows], [r["stop"] for r in rows]
        self.calls = []

    def take(self, indices):
        if self.backend == "mmap":
            value = super().take(indices)
        else:
            selected = []
            for index in indices:
                row = next(r for r in self.rows if r["start"] <= index < r["stop"])
                selected.append(self.cache.get(row, self.output)[index - row["start"]].copy())
            value = np.stack(selected)
        for index, sample in zip(indices, value):
            actual = sha(sample.tobytes())
            if actual != self.expected[index]:
                np.save(self.output / f"bad_batch_{time.time_ns()}.npy", value)
                raise RuntimeError(f"Actual consumed sample mismatch: {index}")
        self.calls.append(sha(value.tobytes()))
        return value


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=30)
    return {"args": args, "returncode": result.returncode,
            "stdout": result.stdout or "", "stderr": result.stderr or ""}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs_e2e"):
        raise ValueError("Output outside outputs_e2e")
    output.mkdir(parents=True, exist_ok=False)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    manifest = g4.read_json(source / "manifest.json")
    features = g4.read_json(source / "feature_cache_v2_manifest.json")["files"]["train"]
    starts = [0, 240, 352, 416]
    indices = torch.tensor([i for start in starts for i in range(start, start + 16)])
    selected = {name: [r for r in features[name] if r["start"] in starts]
                for name in ("ch3_spatial", "d8_e1", "d8_d2")}
    before, expected = {}, {}
    for name, rows in selected.items():
        expected[name] = {}
        for row in rows:
            path = Path(row["path"]).resolve()
            if not path.is_relative_to(source / "feature_cache_v2" / "train"):
                raise ValueError("Unexpected input")
            payload = path.read_bytes()
            if len(payload) != row["size_bytes"] or sha(payload) != row["sha256"]:
                (output / f"preflight_bad_{path.name}.bin").write_bytes(payload)
                raise RuntimeError(f"Preflight mismatch: {path}")
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            for index, sample in enumerate(array, row["start"]):
                expected[name][index] = sha(sample.tobytes())
            before[str(path)] = probe.stat(path)
    del payload, array
    seed, config = 20260907, manifest["config"]
    initial = g4.G3_INITIAL[seed].resolve()
    if g4.identity(initial) != manifest["initial_checkpoints"][str(seed)]:
        raise RuntimeError("Initialization identity mismatch")
    g4.configure_recorded_snapshot(manifest)
    target_row = features["targets"]
    target_bytes = Path(target_row["path"]).read_bytes()
    if sha(target_bytes) != target_row["sha256"]:
        raise RuntimeError("Targets identity mismatch")
    targets = g4.Targets(**torch.load(io.BytesIO(target_bytes), map_location="cpu", weights_only=False))
    orders = []
    generator = torch.Generator().manual_seed(seed)
    for _ in range(8):
        order = indices[torch.randperm(len(indices), generator=generator)]
        orders.extend(order.split(4))
    report = {"schema": "cache-training-path-probe-v1", "source": str(source),
              "seed": seed, "config": config, "samples": indices.tolist(),
              "steps_per_arm": len(orders), "cache_cap_mib": 256,
              "test_executed": False, "performance_evidence": False,
              "initial_checkpoint": manifest["initial_checkpoints"][str(seed)],
              "script_sha256": sha(Path(__file__).read_bytes()), "arms": {}}
    write(output / "contract.json", report)
    watchers = [Watcher(source / "feature_cache_v2" / "train", output / "source_events.jsonl"),
                Watcher(checkpoints, output / "checkpoint_events.jsonl")]
    for watcher in watchers:
        watcher.start()
    started_trace = False
    started = time.perf_counter()
    try:
        status = command(["wpr", "-status"])
        report["wpr_status"] = status
        if "WPR is not recording" in status["stdout"]:
            temporary = output / "etw_temp"
            temporary.mkdir()
            result = command(["wpr", "-start", "FileIO", "-filemode", "-recordtempto", str(temporary)])
            report["wpr_start"] = result
            started_trace = result["returncode"] == 0
        time.sleep(0.1)
        write(checkpoints / "notification_probe.json", {"purpose": "watcher self-check"})
        for backend in ("mmap", "verified_lru"):
            cache = Cache(256 * 1024 * 1024)
            arrays = [AuditedArray(selected[n], expected[n], backend, cache, output)
                      for n in selected]
            store = g4.FeatureStore(*arrays)
            device = torch.device("cuda")
            context = g4.build_context("a2_joint_tail", seed, device, config)
            initial_digest = g4.state_digest(context)
            optimizer = torch.optim.AdamW(context.parameter_groups, weight_decay=float(config["weight_decay"]))
            cached_targets = g4.as_cached_targets(targets)
            logs = []
            arm_started = time.perf_counter()
            peak_rss = 0
            torch.cuda.reset_peak_memory_stats()
            with (output / f"{backend}_steps.jsonl").open("x", encoding="utf-8") as journal:
                for step, batch in enumerate(orders, 1):
                    if time.perf_counter() - started > 900:
                        raise RuntimeError("Diagnostic exceeded 15-minute bound")
                    _, logits, _, heatmap, offset = g4.forward_indices(context, store, batch, device)
                    loss, _, _ = g4.r1.compute_losses(logits, heatmap, offset, cached_targets, batch, config)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(context.parameters, float(config["gradient_clip"]))
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise RuntimeError("Nonfinite loss/gradient")
                    optimizer.step()
                    record = {"step": step, "indices": batch.tolist(), "loss": float(loss.detach()),
                              "gradient_norm": float(norm), "input_sha": [a.calls[-1] for a in arrays]}
                    logs.append(record)
                    journal.write(json.dumps(record) + "\n")
                    journal.flush()
                    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                    if step % 16 == 0:
                        path = checkpoints / f"{backend}_{step:03d}.pt"
                        torch.save({"state": g4.state_payload(context), "optimizer": optimizer.state_dict(),
                                    "step": step, "diagnostic_only": True}, path)
                        loaded = torch.load(path, map_location="cpu", weights_only=False)
                        for group, state in g4.state_payload(context).items():
                            for name, tensor in state.items():
                                if not torch.equal(tensor.detach().cpu(), loaded["state"][group][name]):
                                    raise RuntimeError("Checkpoint round-trip mismatch")
                        del loaded
                        print(f"{backend}: {step}/{len(orders)} loss={record['loss']:.6f}", flush=True)
            report["arms"][backend] = {"initial_digest": initial_digest, "final_digest": g4.state_digest(context),
                "seconds": time.perf_counter() - arm_started, "peak_rss_bytes": peak_rss,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cache_peak_bytes": cache.peak, "cache_hits": cache.hits, "cache_misses": cache.misses,
                "cache_read_bytes": cache.read_bytes, "steps": logs}
            write(output / f"{backend}_report.json", report["arms"][backend])
            del context, optimizer, cached_targets, store, arrays, cache, loss, logits, heatmap, offset
            gc.collect()
            torch.cuda.empty_cache()
        a, b = report["arms"].values()
        report["same_initialization"] = a["initial_digest"] == b["initial_digest"]
        report["same_input_loss_gradient_sequence"] = a["steps"] == b["steps"]
        report["same_final_state"] = a["final_digest"] == b["final_digest"]
        report["postcheck"] = []
        for rows in selected.values():
            for row in rows:
                path = Path(row["path"])
                report["postcheck"].append({"path": str(path), "sha_match": sha(path.read_bytes()) == row["sha256"],
                                            "identity_same": probe.stat(path) == before[str(path)]})
        report["status"] = "NOT_REPRODUCED_LOADERS_EQUIVALENT" if (
            report["same_initialization"] and report["same_input_loss_gradient_sequence"] and
            report["same_final_state"] and all(r["sha_match"] and r["identity_same"] for r in report["postcheck"])
        ) else "DIFFERENCE_DETECTED"
    except Exception as exc:
        report["status"], report["error"] = "STOP", repr(exc)
        report["process_snapshot"] = probe.processes([Path(p) for p in before])
    finally:
        if started_trace:
            report["wpr_stop"] = command(["wpr", "-stop", str(output / "fileio.etl")])
        report["watchers"] = [watcher.stop() for watcher in watchers]
        report["watcher_positive_control"] = any("notification_probe.json" in e["path"] for e in watchers[1].events)
        report["seconds"] = time.perf_counter() - started
        if any(w["errors"] for w in report["watchers"]) or not report["watcher_positive_control"]:
            report["observation_warning"] = "Directory observation incomplete"
        write(output / "report.json", report)
    print(json.dumps({"status": report["status"], "error": report.get("error"), "seconds": report["seconds"]}), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
    main()
