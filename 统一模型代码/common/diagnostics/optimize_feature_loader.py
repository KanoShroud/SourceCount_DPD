"""512样本有界加载优化与256步配对验证；不执行G5或test。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g4 import e2e_g4 as g4  # noqa: E402
from 统一模型代码.common.diagnostics import cache_training_path_probe as old  # noqa: E402
from 统一模型代码.common.verified_feature_loader import VerifiedCache, VerifiedShardedArray, decode_npy  # noqa: E402

NAMES = ("ch3_spatial", "d8_e1", "d8_d2")
STARTED = time.perf_counter()


def guard():
    if time.perf_counter() - STARTED > 3600:
        raise RuntimeError("60 minute budget exceeded")
    if psutil.virtual_memory().available < 8 * 2**30:
        raise RuntimeError("Available RAM below 8 GiB")


def digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array))).hexdigest()


def identity_payload(row, output):
    path = Path(row["path"]).resolve(strict=True)
    data = path.read_bytes()
    if len(data) != row["size_bytes"] or old.sha(data) != row["sha256"]:
        artifact = output / f"bad_{time.time_ns()}.bin"
        artifact.write_bytes(data)
        raise RuntimeError(f"Input identity mismatch: {path}; captured {artifact}")
    return data


def prepare(source, output):
    manifest = g4.read_json(source / "feature_cache_v2_manifest.json")["files"]["train"]
    rows16 = {n: [r for r in manifest[n] if r["start"] < 512] for n in NAMES}
    rows4 = {n: [] for n in NAMES}
    expected = {n: {} for n in NAMES}
    initial = {}
    destination = output / "shards4"
    destination.mkdir()
    for name, rows in rows16.items():
        for row in rows:
            guard()
            path = Path(row["path"]).resolve(strict=True)
            if not path.is_relative_to(source / "feature_cache_v2" / "train"):
                raise ValueError("Unexpected source path")
            initial[str(path)] = old.probe.stat(path)
            payload = identity_payload(row, output)
            array = decode_npy(payload)
            for index, sample in enumerate(array, row["start"]):
                expected[name][index] = digest(sample)
            for local in range(0, len(array), 4):
                start = row["start"] + local
                target = destination / f"{name}_{start:04d}.npy"
                temporary = target.with_suffix(".tmp")
                with temporary.open("xb") as handle:
                    np.save(handle, array[local:local + 4], allow_pickle=False)
                    handle.flush()
                    os.fsync(handle.fileno())
                readback = temporary.read_bytes()
                restored = decode_npy(readback)
                if not np.array_equal(restored, array[local:local + 4]):
                    raise RuntimeError("Reshard round-trip mismatch")
                temporary.rename(target)
                rows4[name].append(dict(path=str(target), start=start, stop=start + len(restored),
                                        size_bytes=len(readback), sha256=old.sha(readback)))
            print(f"prepare {name} {row['stop']}/512", flush=True) if row["stop"] % 128 == 0 else None
    old.write(output / "shards_manifest.json", {"source16": rows16, "new4": rows4,
                                                "sample_sha256": expected, "initial_stat": initial})
    return rows16, rows4, expected, initial, manifest["targets"]


class LegacyArray:
    def __init__(self, rows, cache, output):
        self.rows, self.cache, self.output = rows, cache, output

    def take(self, indices):
        arrays = []
        for index in indices:
            row = next(r for r in self.rows if r["start"] <= index < r["stop"])
            arrays.append(self.cache.get(row, self.output)[index - row["start"]].copy())
        return np.stack(arrays)


def make_arrays(spec, rows16, rows4, output):
    kind, shard, cap = spec
    rows = rows16 if shard == 16 else rows4
    cache = None
    if kind == "mmap":
        arrays = [g4.ShardedArray(rows[n]) for n in NAMES]
    elif kind == "legacy":
        cache = old.Cache(cap * 2**20)
        arrays = [LegacyArray(rows[n], cache, output) for n in NAMES]
    else:
        cache = VerifiedCache(cap * 2**20, output / "anomalies")
        arrays = [VerifiedShardedArray(rows[n], cache) for n in NAMES]
    return arrays, cache


def cache_stats(cache):
    if cache is None:
        return {}
    if isinstance(cache, VerifiedCache):
        return dict(cache.stats)
    return dict(hits=cache.hits, misses=cache.misses, read_bytes=cache.read_bytes,
                peak_bytes=cache.peak)


def replay(spec, rows16, rows4, expected, batches, output):
    arrays, cache = make_arrays(spec, rows16, rows4, output)
    load_seconds = 0.0
    peak = 0
    start = time.perf_counter()
    for batch in batches:
        guard()
        for name, array in zip(NAMES, arrays):
            tick = time.perf_counter()
            value = array.take(batch.tolist())
            load_seconds += time.perf_counter() - tick
            for index, sample in zip(batch.tolist(), value):
                if digest(sample) != expected[name][index]:
                    np.save(output / f"bad_replay_{time.time_ns()}.npy", value)
                    raise RuntimeError("Replay consumed sample mismatch")
            del value
        peak = max(peak, psutil.Process().memory_info().rss)
    result = dict(load_seconds=load_seconds, total_seconds=time.perf_counter() - start,
                  peak_sampled_rss=peak, cache=cache_stats(cache))
    del arrays, cache
    gc.collect()
    return result


class AuditArray:
    def __init__(self, array, expected):
        self.array, self.expected = array, expected
        self.last = None

    def __getitem__(self, indices):
        value = self.array.take(indices)
        for index, sample in zip(indices, value):
            if digest(sample) != self.expected[index]:
                raise RuntimeError("Training consumed input mismatch")
        self.last = digest(value)
        return value


def equal(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(v, right[k]) for k, v in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right))
    return left == right


def train(name, spec, rows16, rows4, expected, batches, source, output, target_row):
    arrays, cache = make_arrays(spec, rows16, rows4, output)
    audited = [AuditArray(array, expected[n]) for n, array in zip(NAMES, arrays)]
    features = g4.FeatureStore(*audited)
    manifest = g4.read_json(source / "manifest.json")
    config = manifest["config"]
    g4.configure_recorded_snapshot(manifest)
    initial = g4.G3_INITIAL[20260907].resolve(strict=True)
    if g4.identity(initial) != manifest["initial_checkpoints"]["20260907"]:
        raise RuntimeError("Checkpoint identity mismatch")
    target_data = identity_payload(target_row, output)
    targets = g4.Targets(**torch.load(io.BytesIO(target_data), map_location="cpu", weights_only=False))
    cached_targets = g4.as_cached_targets(targets)
    context = g4.build_context("a2_joint_tail", 20260907, torch.device("cuda"), config)
    optimizer = torch.optim.AdamW(context.parameter_groups, weight_decay=config["weight_decay"])
    records, checkpoint_rows = [], []
    arm = output / name
    arm.mkdir()
    initial_digest = g4.state_digest(context)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    peak, minimum_available = 0, psutil.virtual_memory().available
    with (arm / "steps.jsonl").open("x", encoding="utf-8") as journal:
        for step, batch in enumerate(batches, 1):
            guard()
            _, logits, _, heatmap, offset = g4.forward_indices(context, features, batch, torch.device("cuda"))
            loss, _, _ = g4.r1.compute_losses(logits, heatmap, offset, cached_targets, batch, config)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(context.parameters, config["gradient_clip"])
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise RuntimeError("Nonfinite loss/gradient")
            optimizer.step()
            row = dict(step=step, indices=batch.tolist(), input_sha=[a.last for a in audited],
                       loss=float(loss.detach()), gradient_norm=float(norm))
            records.append(row)
            journal.write(json.dumps(row) + "\n")
            journal.flush()
            peak = max(peak, psutil.Process().memory_info().rss)
            minimum_available = min(minimum_available, psutil.virtual_memory().available)
            if step % 64 == 0:
                checkpoint = arm / f"step_{step:03d}.pt"
                state = dict(state=g4.state_payload(context), optimizer=optimizer.state_dict(), step=step)
                torch.save(state, checkpoint)
                restored = torch.load(checkpoint, map_location="cpu", weights_only=False)
                cpu_state = {group: {key: value.detach().cpu() for key, value in values.items()}
                             for group, values in state["state"].items()}
                if not equal(cpu_state, restored["state"]):
                    raise RuntimeError("Checkpoint round-trip mismatch")
                checkpoint_rows.append(g4.identity(checkpoint))
                del restored, cpu_state, state
                print(f"train {name} {step}/256 loss={row['loss']:.6f}", flush=True)
    result = dict(initial_digest=initial_digest, final_digest=g4.state_digest(context),
                  seconds=time.perf_counter() - start, peak_sampled_rss=peak,
                  minimum_available_bytes=minimum_available, cuda_peak=torch.cuda.max_memory_allocated(),
                  cache=cache_stats(cache), steps=records, checkpoints=checkpoint_rows)
    old.write(arm / "report.json", result)
    del context, optimizer, features, arrays, audited, cache, targets, cached_targets, loss, logits, heatmap, offset
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    if not output.is_relative_to(ROOT / "outputs_e2e") or output.is_relative_to(source):
        raise ValueError("Output isolation violation")
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="RUNNING", test_executed=False, g5_executed=False,
                  source=str(source), samples=512, cache_floor_available_gib=8, budget_seconds=3600,
                  code=[g4.identity(Path(__file__)), g4.identity(ROOT / '统一模型代码/common/verified_feature_loader.py')])
    old.write(output / "contract.json", report)
    try:
        guard()
        rows16, rows4, expected, before, targets = prepare(source, output)
        generator = torch.Generator().manual_seed(20260907)
        batches = []
        for _ in range(2):
            batches.extend(torch.randperm(512, generator=generator).split(4))
        specs = {"mmap": ("mmap", 16, 0), "legacy256": ("legacy", 16, 256),
                 "group16_1g": ("verified", 16, 1024), "group4_1g": ("verified", 4, 1024)}
        results = {k: [] for k in specs}
        for repeat in range(2):
            keys = list(specs) if repeat == 0 else list(reversed(specs))
            for name in keys:
                result = replay(specs[name], rows16, rows4, expected, batches[:128], output)
                results[name].append(result)
                old.write(output / f"replay_{name}_{repeat}.json", result)
                print(f"replay {name}/{repeat}: load={result['load_seconds']:.2f}s", flush=True)
        winner = min(("group16_1g", "group4_1g"), key=lambda k: statistics.median(r["load_seconds"] for r in results[k]))
        best = results["group4_1g"]
        misses = sum(v["cache"]["misses"] for v in best)
        hits = sum(v["cache"]["hits"] for v in best)
        base = statistics.median(v["load_seconds"] for v in results["mmap"])
        if winner == "group4_1g" and misses / max(1, hits + misses) > .5 and statistics.median(v["load_seconds"] for v in best) > 1.3 * base:
            specs["group4_2g"] = ("verified", 4, 2048)
            results["group4_2g"] = []
            for repeat in range(2):
                result = replay(specs["group4_2g"], rows16, rows4, expected, batches[:128], output)
                results["group4_2g"].append(result)
                old.write(output / f"replay_group4_2g_{repeat}.json", result)
            if statistics.median(v["load_seconds"] for v in results["group4_2g"]) < statistics.median(v["load_seconds"] for v in best):
                winner = "group4_2g"
        report.update(replay=results, candidate=winner, candidate_spec=specs[winner])
        old.write(output / "selection.json", report)
        report["baseline_train"] = train("baseline_train", specs["mmap"], rows16, rows4, expected, batches, source, output, targets)
        report["candidate_train"] = train("candidate_train", specs[winner], rows16, rows4, expected, batches, source, output, targets)
        a, b = report["baseline_train"], report["candidate_train"]
        report["training_equivalent"] = a["steps"] == b["steps"] and a["initial_digest"] == b["initial_digest"] and a["final_digest"] == b["final_digest"]
        report["checkpoints_equivalent"] = all(equal(
            torch.load(output / "baseline_train" / f"step_{s:03d}.pt", map_location="cpu", weights_only=False),
            torch.load(output / "candidate_train" / f"step_{s:03d}.pt", map_location="cpu", weights_only=False)) for s in (64, 128, 192, 256))
        report["time_ratio"] = b["seconds"] / a["seconds"]
        report["source_unchanged"] = True
        for rows in rows16.values():
            for row in rows:
                guard()
                identity_payload(row, output)
                report["source_unchanged"] &= old.probe.stat(Path(row["path"])) == before[row["path"]]
        for rows in rows4.values():
            for row in rows:
                guard()
                identity_payload(row, output)
        valid = report["training_equivalent"] and report["checkpoints_equivalent"] and report["source_unchanged"]
        report["status"] = ("PASS" if report["time_ratio"] <= 1.3 else "CORRECT_BUT_SPEED_TARGET_UNMET") if valid else "FAIL"
    except Exception as exc:
        report.update(status="STOP", error=repr(exc))
        raise
    finally:
        report["seconds"] = time.perf_counter() - STARTED
        old.write(output / "final_report.json", report)
        print(json.dumps({k: report.get(k) for k in ("status", "candidate", "time_ratio", "seconds", "error")}), flush=True)


if __name__ == "__main__":
    main()
