"""Gate 3A：第四章中等规模数据审计、确定性比较与训练预算汇总。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_root in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import gate2_smoke_ch4 as gate2  # noqa: E402
from yolo_config import EDGE, GRID_SIZE, MAX_SRC  # noqa: E402


SPLITS = ("train", "val", "test")
EXPECTED_RCV = 4
EXPECTED_LEN = 4096
EXPECTED_N_SUB = 19
FS = 100e6


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def expected_dict(values: list[int]) -> dict[str, int]:
    require(len(values) == 3, "expected_samples 必须恰有 train/val/test 三项")
    require(all(value > 0 for value in values), "expected_samples 必须为正整数")
    return dict(zip(SPLITS, values, strict=True))


def expected_counts(n_samples: int) -> dict[int, int]:
    require(n_samples % 2 == 0, "Gate 3A 要求每个 split 为偶数，以保证 N=2/3 平衡")
    return {2: n_samples // 2, 3: n_samples // 2}


def sample_digest(data: dict[str, Any], index: int) -> str:
    digest = hashlib.sha256()
    for name in ("sig_real", "sig_imag", "src_count", "src_pos", "fc_offset", "bandwidth"):
        value = np.ascontiguousarray(data[name][index])
        digest.update(name.encode("ascii"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.view(np.uint8).tobytes())
    return digest.hexdigest()


def audit_mat(data_dir: Path, expected: dict[str, int], seed: int) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report: dict[str, Any] = {"status": "PASS", "seed": seed, "splits": {}}
    mat_data: dict[str, dict[str, Any]] = {}
    all_digests: dict[str, str] = {}

    for split in SPLITS:
        expected_n = expected[split]
        path = data_dir / f"{split}_data.mat"
        require(path.is_file(), f"缺少 {path}")
        data = gate2.read_mat(path)
        mat_data[split] = data

        with h5py.File(path, "r") as handle:
            require("trials_list_val" in handle, f"{split} 缺少 trials_list_val")
            trials_list = np.asarray(handle["trials_list_val"]).reshape(-1).astype(int).tolist()
        require(trials_list == [expected[name] for name in SPLITS],
                f"{split} trials_list_val={trials_list}")
        require(data["runtime_mode"] == "smoke", f"{split} runtime_mode={data['runtime_mode']!r}")
        require(data["random_seed"] == seed, f"{split} seed={data['random_seed']}")
        require(data["src_count"].shape == (expected_n,), f"{split} src_count shape")
        require(data["sig_real"].shape == (expected_n, EXPECTED_RCV, EXPECTED_LEN), f"{split} IQ shape")
        require(data["sig_imag"].shape == data["sig_real"].shape, f"{split} IQ complex shape")
        require(data["src_pos"].shape == (expected_n, MAX_SRC, 2), f"{split} src_pos shape")
        require(data["fc_offset"].shape == (expected_n, MAX_SRC), f"{split} fc_offset shape")
        require(data["bandwidth"].shape == (expected_n, MAX_SRC), f"{split} BW shape")
        require(data["band_mask"].shape == (expected_n, MAX_SRC, EXPECTED_N_SUB), f"{split} band shape")
        require(data["ignore_mask"].shape == data["band_mask"].shape, f"{split} ignore shape")

        for name in (
            "sig_real", "sig_imag", "src_pos", "fc_offset", "pt_w",
            "symbol_rate", "bandwidth", "band_mask", "ignore_mask",
        ):
            require(data[name].dtype == np.float32, f"{split} {name} dtype={data[name].dtype}")
            require(np.isfinite(data[name]).all(), f"{split} {name} 含 NaN/Inf")
        require(not np.logical_and(data["band_mask"] > 0.5, data["ignore_mask"] > 0.5).any(),
                f"{split} band/ignore 重叠")

        counts = Counter(int(value) for value in data["src_count"])
        require(dict(counts) == expected_counts(expected_n), f"{split} 源数分布={dict(counts)}")
        band_stats: dict[int, list[int]] = defaultdict(list)
        split_digests: set[str] = set()
        for sample_idx, n_value in enumerate(data["src_count"]):
            n_src = int(n_value)
            active = slice(0, n_src)
            inactive = slice(n_src, MAX_SRC)
            for name in ("fc_offset", "pt_w", "symbol_rate", "bandwidth"):
                require(np.count_nonzero(data[name][sample_idx, inactive]) == 0,
                        f"{split}[{sample_idx}] {name} 空槽非零")
            require(np.count_nonzero(data["src_pos"][sample_idx, inactive]) == 0,
                    f"{split}[{sample_idx}] src_pos 空槽非零")
            positions = data["src_pos"][sample_idx, active]
            require((np.abs(positions) <= EDGE).all(), f"{split}[{sample_idx}] 位置越界")
            centers = data["fc_offset"][sample_idx, active].astype(np.float64)
            bandwidths = data["bandwidth"][sample_idx, active].astype(np.float64)
            require((bandwidths > 0).all(), f"{split}[{sample_idx}] 带宽非正")
            lows = centers - bandwidths / 2
            highs = centers + bandwidths / 2
            require((lows >= -FS / 2).all() and (highs <= FS / 2).all(),
                    f"{split}[{sample_idx}] 频率越过 Nyquist")
            groups = gate2.independent_frequency_groups(centers, bandwidths)
            require(len(groups) == 1 and len(groups[0]) == n_src,
                    f"{split}[{sample_idx}] 独立分组={groups}")
            f_axis = np.arange(-EXPECTED_LEN // 2, EXPECTED_LEN // 2) * (FS / EXPECTED_LEN)
            n_band = int(np.count_nonzero((f_axis >= lows.min()) & (f_axis < highs.max())))
            require(n_band > 0, f"{split}[{sample_idx}] 带内频点为零")
            band_stats[n_src].append(n_band)

            digest = sample_digest(data, sample_idx)
            require(digest not in split_digests, f"{split} 内出现完全重复样本: {sample_idx}")
            require(digest not in all_digests,
                    f"split 泄漏: {split}[{sample_idx}] 与 {all_digests.get(digest)} 相同")
            split_digests.add(digest)
            all_digests[digest] = f"{split}[{sample_idx}]"

        report["splits"][split] = {
            "path": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "sample_count": expected_n,
            "source_count_distribution": {str(k): int(v) for k, v in sorted(counts.items())},
            "iq_shape": list(data["sig_real"].shape),
            "iq_dtype": str(data["sig_real"].dtype),
            "unique_sample_digests": len(split_digests),
            "n_band_by_source_count": {
                str(key): {"min": min(values), "max": max(values)}
                for key, values in sorted(band_stats.items())
            },
        }
    report["cross_split_unique_sample_digests"] = len(all_digests)
    return report, mat_data


def tensor_tail_shapes() -> dict[str, tuple[int, ...]]:
    return {
        "fine_dpd": (1, GRID_SIZE, GRID_SIZE),
        "hyp_mask": (MAX_SRC, GRID_SIZE, GRID_SIZE),
        "gauss_label": (1, GRID_SIZE, GRID_SIZE),
        "gauss_multi": (MAX_SRC, GRID_SIZE, GRID_SIZE),
        "pos_label": (MAX_SRC, 2),
        "n_src": (),
        "sample_idx": (),
        "group_idx": (),
    }


def load_prefix(root: Path, split: str, count: int) -> dict[str, torch.Tensor]:
    index_path = root / split / f"loc_{split}_index.pt"
    index = torch.load(index_path, map_location="cpu", weights_only=False)
    pieces: dict[str, list[torch.Tensor]] = {key: [] for key in tensor_tail_shapes()}
    remaining = count
    for shard_name in index["shard_files"]:
        shard = torch.load(root / split / shard_name, map_location="cpu", weights_only=False)
        take = min(remaining, len(shard["n_src"]))
        for key in pieces:
            pieces[key].append(shard[key][:take])
        remaining -= take
        if remaining == 0:
            break
    require(remaining == 0, f"{root}/{split} 不足 {count} 个任务")
    return {key: torch.cat(values, dim=0) for key, values in pieces.items()}


def compare_prefixes(full_root: Path, reference_root: Path, reference: dict[str, int]) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "PASS", "splits": {}}
    for split in SPLITS:
        count = reference[split]
        full = load_prefix(full_root, split, count)
        pilot = load_prefix(reference_root, split, count)
        diffs: dict[str, float] = {}
        for key in tensor_tail_shapes():
            require(full[key].shape == pilot[key].shape, f"{split} prefix {key} shape")
            if full[key].dtype.is_floating_point:
                diff = float(torch.max(torch.abs(full[key].float() - pilot[key].float())).item())
                diffs[f"{key}_max_abs_diff"] = diff
                require(diff == 0.0, f"{split} pilot/full {key} 差={diff}")
            else:
                require(torch.equal(full[key], pilot[key]), f"{split} pilot/full {key} 不同")
                diffs[f"{key}_max_abs_diff"] = 0.0
        report["splits"][split] = {"task_count": count, **diffs}
    return report


def audit_loc(
    data_dir: Path,
    loc_root: Path,
    expected: dict[str, int],
    mat_expected: dict[str, int],
    seed: int,
    shard_size: int,
    reference_root: Path | None,
    reference: dict[str, int] | None,
) -> dict[str, Any]:
    mat_report, mat_data = audit_mat(data_dir, mat_expected, seed)
    report: dict[str, Any] = {
        "status": "PASS",
        "mat_audit": mat_report,
        "loc_data_dir": str(loc_root.resolve()),
        "splits": {},
    }
    coordinate_samples: dict[str, dict[int, dict[str, torch.Tensor]]] = {
        split: {} for split in SPLITS
    }
    required = set(tensor_tail_shapes())

    for split in SPLITS:
        expected_n = expected[split]
        index_path = loc_root / split / f"loc_{split}_index.pt"
        require(index_path.is_file(), f"缺少 {index_path}")
        index = torch.load(index_path, map_location="cpu", weights_only=False)
        require(index["n_total_tasks"] == expected_n, f"{split} index task 数")
        expected_shards = math.ceil(expected_n / shard_size)
        require(index["n_shards"] == expected_shards,
                f"{split} shard 数={index['n_shards']}, expected={expected_shards}")
        require(len(index["shard_files"]) == expected_shards, f"{split} shard_files 长度")
        require(len(set(index["shard_files"])) == expected_shards, f"{split} shard 文件名重复")

        counts: Counter[int] = Counter()
        seen_indices: set[int] = set()
        max_gauss_diff = 0.0
        max_multi_diff = 0.0
        total_bytes = index_path.stat().st_size
        task_cursor = 0

        for shard_number, shard_name in enumerate(index["shard_files"]):
            shard_path = loc_root / split / shard_name
            require(shard_path.is_file(), f"缺少 {shard_path}")
            total_bytes += shard_path.stat().st_size
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            missing = sorted(required.difference(shard))
            require(not missing, f"{shard_name} 缺字段 {missing}")
            chunk = len(shard["n_src"])
            expected_chunk = min(shard_size, expected_n - shard_number * shard_size)
            require(chunk == expected_chunk, f"{split}/{shard_name} task 数={chunk}")
            for key, tail in tensor_tail_shapes().items():
                require(tuple(shard[key].shape) == (chunk, *tail),
                        f"{split}/{shard_name} {key} shape={tuple(shard[key].shape)}")
            require(shard["fine_dpd"].dtype == torch.float16, f"{split} fine dtype")
            require(shard["hyp_mask"].dtype == torch.float16, f"{split} hyp dtype")
            require(shard["gauss_label"].dtype == torch.float16, f"{split} gauss dtype")
            require(shard["gauss_multi"].dtype == torch.float16, f"{split} multi dtype")
            require(shard["pos_label"].dtype == torch.float32, f"{split} pos dtype")
            for key in ("fine_dpd", "hyp_mask", "gauss_label", "gauss_multi", "pos_label"):
                require(bool(torch.isfinite(shard[key]).all()), f"{split}/{shard_name} {key} 含 NaN/Inf")
            require(bool((shard["fine_dpd"] >= 0).all()), f"{split} fine_dpd 有负数")
            require(bool((shard["hyp_mask"] >= 0).all() and (shard["hyp_mask"] <= 1).all()), f"{split} hyp 范围")
            require(bool((shard["gauss_label"] >= 0).all() and (shard["gauss_label"] <= 1).all()), f"{split} gauss 范围")
            require(bool((shard["gauss_multi"] >= 0).all() and (shard["gauss_multi"] <= 1).all()), f"{split} multi 范围")
            require(bool(shard["fine_dpd"].flatten(1).std(dim=1).gt(0).all()), f"{split} fine_dpd 恒定")

            for local_idx in range(chunk):
                task_idx = task_cursor + local_idx
                sample_idx = int(shard["sample_idx"][local_idx].item())
                group_idx = int(shard["group_idx"][local_idx].item())
                n_src = int(shard["n_src"][local_idx].item())
                require(sample_idx == task_idx and group_idx == 0,
                        f"{split}[{task_idx}] index 映射")
                require(sample_idx not in seen_indices, f"{split} sample_idx 重复: {sample_idx}")
                seen_indices.add(sample_idx)
                counts[n_src] += 1
                require(n_src == int(mat_data[split]["src_count"][sample_idx]), f"{split}[{task_idx}] n_src")

                positions = mat_data[split]["src_pos"][sample_idx, :n_src]
                positions = positions[np.argsort(np.linalg.norm(positions, axis=1))]
                expected_pos = np.zeros((MAX_SRC, 2), dtype=np.float32)
                expected_pos[:n_src] = positions / EDGE
                pos_diff = float(np.max(np.abs(shard["pos_label"][local_idx].numpy() - expected_pos)))
                require(pos_diff <= 1e-6, f"{split}[{task_idx}] pos 最大差={pos_diff}")
                require(bool(shard["hyp_mask"][local_idx, n_src:].eq(0).all()), f"{split}[{task_idx}] hyp 空通道")
                require(bool(shard["gauss_multi"][local_idx, n_src:].eq(0).all()), f"{split}[{task_idx}] multi 空通道")

                expected_gauss, expected_multi = gate2.expected_gaussians(positions)
                stored_gauss = shard["gauss_label"][local_idx, 0].numpy().astype(np.float32)
                stored_multi = shard["gauss_multi"][local_idx].numpy().astype(np.float32)
                gauss_diff = float(np.max(np.abs(stored_gauss - expected_gauss.astype(np.float16).astype(np.float32))))
                multi_diff = float(np.max(np.abs(stored_multi - expected_multi.astype(np.float16).astype(np.float32))))
                max_gauss_diff = max(max_gauss_diff, gauss_diff)
                max_multi_diff = max(max_multi_diff, multi_diff)
                require(gauss_diff <= 1e-3, f"{split}[{task_idx}] gauss 差={gauss_diff}")
                require(multi_diff <= 1e-3, f"{split}[{task_idx}] multi 差={multi_diff}")

                if n_src not in coordinate_samples[split]:
                    coordinate_samples[split][n_src] = {
                        "pos_label": shard["pos_label"][local_idx:local_idx + 1].clone(),
                        "n_src": shard["n_src"][local_idx:local_idx + 1].clone(),
                        "gauss_label": shard["gauss_label"][local_idx:local_idx + 1].clone(),
                    }
            task_cursor += chunk
            del shard

        require(task_cursor == expected_n, f"{split} 总任务数={task_cursor}")
        require(seen_indices == set(range(expected_n)), f"{split} sample_idx 不连续")
        expected_prefix_counts = Counter(
            int(value) for value in mat_data[split]["src_count"][:expected_n]
        )
        require(counts == expected_prefix_counts,
                f"{split} shard 源数={dict(counts)}, MAT 前缀={dict(expected_prefix_counts)}")
        require(set(coordinate_samples[split]) == {2, 3}, f"{split} 坐标抽样未覆盖 N=2/3")
        report["splits"][split] = {
            "n_total_tasks": task_cursor,
            "n_shards": int(index["n_shards"]),
            "source_count_distribution": {str(k): int(v) for k, v in sorted(counts.items())},
            "max_gauss_abs_diff": max_gauss_diff,
            "max_gauss_multi_abs_diff": max_multi_diff,
            "total_bytes": total_bytes,
        }

    coordinate_inputs: dict[str, dict[str, torch.Tensor]] = {}
    for split in SPLITS:
        coordinate_inputs[split] = {
            key: torch.cat([coordinate_samples[split][n][key] for n in (2, 3)], dim=0)
            for key in ("pos_label", "n_src", "gauss_label")
        }
    report["coordinate_contract"] = gate2.audit_coordinate_contract(coordinate_inputs)

    if reference_root is not None:
        require(reference is not None, "reference_samples 缺失")
        report["pilot_vs_full_prefix"] = compare_prefixes(loc_root, reference_root, reference)
    return report


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def find_last_checkpoint(run_dir: Path) -> Path:
    matches = list(run_dir.glob("last_yolo_*.pth"))
    require(len(matches) == 1, f"{run_dir} last checkpoint 数={len(matches)}")
    return matches[0]


def compare_nested(left: Any, right: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if not torch.equal(left, right):
            max_diff = float(torch.max(torch.abs(left.float() - right.float())).item()) if left.numel() else 0.0
            mismatches.append(f"{path}: tensor mismatch max_abs_diff={max_diff}")
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if not np.array_equal(left, right):
            mismatches.append(f"{path}: ndarray mismatch")
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            mismatches.append(f"{path}: dict keys mismatch")
            return
        for key in left:
            compare_nested(left[key], right[key], f"{path}.{key}", mismatches)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            mismatches.append(f"{path}: length mismatch")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            compare_nested(left_item, right_item, f"{path}[{index}]", mismatches)
        return
    if left != right:
        mismatches.append(f"{path}: {left!r} != {right!r}")


def compare_training(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left_history = load_json(left_dir / "epoch_history.json")
    right_history = load_json(right_dir / "epoch_history.json")
    require(len(left_history) == len(right_history), "两次训练 epoch 数不同")
    excluded = {"epoch_seconds", "process_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"}
    history_mismatches: list[str] = []
    for index, (left, right) in enumerate(zip(left_history, right_history, strict=True)):
        keys = set(left).union(right).difference(excluded)
        for key in sorted(keys):
            if left.get(key) != right.get(key):
                history_mismatches.append(
                    f"epoch={index + 1} {key}: {left.get(key)!r} != {right.get(key)!r}"
                )

    left_checkpoint = torch.load(find_last_checkpoint(left_dir), map_location="cpu", weights_only=False)
    right_checkpoint = torch.load(find_last_checkpoint(right_dir), map_location="cpu", weights_only=False)
    checkpoint_mismatches: list[str] = []
    for key in ("model", "optimizer", "scheduler", "scaler", "rng_state"):
        compare_nested(left_checkpoint[key], right_checkpoint[key], key, checkpoint_mismatches)

    left_final = load_json(left_dir / "final_validation.json")
    right_final = load_json(right_dir / "final_validation.json")
    final_mismatches: list[str] = []
    compare_nested(left_final, right_final, "final_validation", final_mismatches)

    if history_mismatches:
        raise AssertionError("逐 epoch 指标不一致: " + history_mismatches[0])
    if checkpoint_mismatches:
        raise AssertionError("checkpoint 状态不一致: " + checkpoint_mismatches[0])
    if final_mismatches:
        raise AssertionError("最终验证不一致: " + final_mismatches[0])
    return {
        "status": "PASS",
        "determinism_class": "same seed, data, code, environment",
        "numeric_requirement": "exact match",
        "epochs_compared": len(left_history),
        "history_mismatches": history_mismatches,
        "checkpoint_mismatches": checkpoint_mismatches,
        "final_validation_mismatches": final_mismatches,
        "left_checkpoint": str(find_last_checkpoint(left_dir).resolve()),
        "right_checkpoint": str(find_last_checkpoint(right_dir).resolve()),
    }


def summarize_calibration(
    run_dirs: list[Path],
    monitor_paths: list[Path],
    projected_epochs: int,
    scale_factor: float,
) -> dict[str, Any]:
    require(len(run_dirs) == len(monitor_paths) == 2, "Gate 3A 固定比较两次 pilot")
    run_summaries: list[dict[str, Any]] = []
    medians: list[float] = []
    warnings: set[str] = set()
    red_flags: set[str] = set()
    for run_dir, monitor_path in zip(run_dirs, monitor_paths, strict=True):
        history = load_json(run_dir / "epoch_history.json")
        require(len(history) == 5, f"{run_dir} 预期 5 epoch")
        stable = [float(item["epoch_seconds"]) for item in history[1:]]
        median_seconds = float(np.median(stable))
        medians.append(median_seconds)
        monitor = load_json(monitor_path)
        require(monitor["status"] == "PASS", f"{monitor_path} status={monitor['status']}")
        warnings.update(monitor.get("warnings", []))
        red_flags.update(monitor.get("red_flags", []))
        run_summaries.append({
            "run_dir": str(run_dir.resolve()),
            "median_pilot_epoch_seconds_excluding_first": median_seconds,
            "epoch_seconds": [float(item["epoch_seconds"]) for item in history],
            "monitor": monitor,
        })

    projected_seconds = max(medians) * scale_factor * projected_epochs
    if red_flags or projected_seconds > 180 * 60:
        status = "FAIL"
    elif warnings or projected_seconds > 150 * 60:
        status = "REVIEW_REQUIRED"
    else:
        status = "PASS"
    return {
        "status": status,
        "runs": run_summaries,
        "pilot_to_full_scale_factor": scale_factor,
        "projected_epochs": projected_epochs,
        "projected_full_run_seconds": projected_seconds,
        "projected_full_run_minutes": projected_seconds / 60,
        "warning_threshold_minutes": 150,
        "hard_threshold_minutes": 180,
        "warnings": sorted(warnings),
        "red_flags": sorted(red_flags),
        "performance_interpretation_allowed": False,
    }


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def finalize_gate3a(
    run_root: Path,
    full_loc_root: Path,
    formal_data_dir: Path,
    formal_output_dir: Path,
) -> dict[str, Any]:
    required_reports = {
        "medium_mat_audit": run_root / "02_medium_iq" / "mat_audit.json",
        "pilot_loc_audit": run_root / "03_pilot_loc" / "audit.json",
        "pilot_train_a": run_root / "04_pilot_train" / "run_a" / "training_summary.json",
        "pilot_train_b": run_root / "04_pilot_train" / "run_b" / "training_summary.json",
        "determinism": (
            run_root / "04_pilot_train" / "determinism_comparison_after_fix_v3.json"
        ),
        "calibration": (
            run_root / "04_pilot_train" / "calibration_summary_after_full_loc.json"
        ),
        "full_loc_audit": full_loc_root / "audit.json",
    }
    reports: dict[str, Any] = {}
    for name, path in required_reports.items():
        report = load_json(path)
        require(report.get("status") == "PASS", f"{path} status={report.get('status')}")
        reports[name] = {"path": str(path.resolve()), "status": report["status"]}

    gate2_monitors = sorted(
        (run_root / "01_gate2_regression" / "stages").glob("*/stage_monitor_report.json")
    )
    require(len(gate2_monitors) == 10, f"Gate 2 回归监控报告应为 10 个，实际 {len(gate2_monitors)}")
    monitor_paths = [
        *gate2_monitors,
        run_root / "02_medium_iq" / "stage" / "stage_monitor_report.json",
        run_root / "02_medium_iq" / "audit_stage" / "stage_monitor_report.json",
        *(run_root / "03_pilot_loc" / "stages" / split / "stage_monitor_report.json"
          for split in SPLITS),
        run_root / "03_pilot_loc" / "audit_stage" / "stage_monitor_report.json",
        run_root / "04_pilot_train" / "monitor_a" / "stage_monitor_report.json",
        run_root / "04_pilot_train" / "monitor_b" / "stage_monitor_report.json",
        (run_root / "04_pilot_train" / "compare_stage_after_fix_v3"
         / "stage_monitor_report.json"),
        (run_root / "04_pilot_train" / "calibration_stage_after_full_loc"
         / "stage_monitor_report.json"),
        *(full_loc_root / "stages" / split / "stage_monitor_report.json"
          for split in SPLITS),
        full_loc_root / "audit_stage" / "stage_monitor_report.json",
        run_root / "00_encoding_selftest_after_fix_v3" / "stage_monitor_report.json",
    ]
    monitors: list[dict[str, Any]] = []
    for path in monitor_paths:
        monitor = load_json(path)
        require(monitor.get("status") == "PASS", f"{path} status={monitor.get('status')}")
        require(not monitor.get("warnings"), f"{path} 存在 warning: {monitor.get('warnings')}")
        require(not monitor.get("red_flags"), f"{path} 存在 red flag: {monitor.get('red_flags')}")
        monitors.append({
            "path": str(path.resolve()),
            "stage": monitor["stage"],
            "status": monitor["status"],
            "duration_seconds": float(monitor.get("duration_seconds", 0.0)),
        })

    determinism = load_json(required_reports["determinism"])
    require(determinism["epochs_compared"] == 5, "确定性比较应覆盖 5 epoch")
    for key in (
        "history_mismatches",
        "checkpoint_mismatches",
        "final_validation_mismatches",
    ):
        require(not determinism[key], f"确定性比较存在 {key}: {determinism[key]}")

    full_audit = load_json(required_reports["full_loc_audit"])
    require(full_audit["mat_audit"]["cross_split_unique_sample_digests"] == 1536,
            "完整 MAT 跨 split 唯一样本数应为 1536")
    require(full_audit["pilot_vs_full_prefix"]["status"] == "PASS",
            "pilot 与完整数据前缀比较未通过")
    require(sum(item["n_total_tasks"] for item in full_audit["splits"].values()) == 1536,
            "完整定位任务总数应为 1536")

    calibration = load_json(required_reports["calibration"])
    require(calibration["projected_epochs"] == 60, "训练预算应按 60 epoch 汇总")
    require(calibration["projected_full_run_minutes"] <= 180,
            "60 epoch 训练预算超过 180 分钟硬门槛")

    full_monitors = {
        split: load_json(full_loc_root / "stages" / split / "stage_monitor_report.json")
        for split in SPLITS
    }
    full_dpd_seconds = sum(item["duration_seconds"] for item in full_monitors.values())
    require(full_dpd_seconds <= 45 * 60, "完整 DPD 三 split 总墙钟超过 45 分钟")

    output_bytes = directory_size(run_root)
    require(output_bytes <= 8 * 1024**3, "Gate 3A 总输出超过 8 GiB 红线")
    return {
        "status": "PASS",
        "gate": "Gate 3A",
        "scope": "第四章中等规模数据、D8 no-AMP pilot 可重复性与资源预算",
        "run_root": str(run_root.resolve()),
        "reports": reports,
        "monitor_summary": {
            "pass_count": len(monitors),
            "warnings": [],
            "red_flags": [],
            "minimum_system_available_bytes": min(
                load_json(path)["minimum_system_available_bytes"]
                for path in monitor_paths
                if "minimum_system_available_bytes" in load_json(path)
            ),
            "maximum_process_tree_rss_bytes": max(
                load_json(path)["maximum_process_tree_rss_bytes"]
                for path in monitor_paths
                if "maximum_process_tree_rss_bytes" in load_json(path)
            ),
            "maximum_gpu_used_mib": max(
                load_json(path).get("maximum_gpu_used_mib") or 0
                for path in monitor_paths
            ),
        },
        "data": {
            "mat_samples": {"train": 1024, "val": 256, "test": 256},
            "loc_tasks": {"train": 1024, "val": 256, "test": 256},
            "loc_shards": {"train": 8, "val": 2, "test": 2},
            "pilot_vs_full_prefix": "exact_match",
            "full_loc_bytes": sum(
                item["total_bytes"] for item in full_audit["splits"].values()
            ),
        },
        "reproducibility": {
            "seed": 42,
            "model": "D8 dualhead",
            "amp": False,
            "batch_size": 8,
            "epochs_compared": 5,
            "numeric_requirement": "exact match",
            "result": "PASS",
        },
        "budgets": {
            "full_dpd_seconds": full_dpd_seconds,
            "full_dpd_limit_seconds": 45 * 60,
            "projected_60_epoch_minutes": calibration["projected_full_run_minutes"],
            "training_warning_minutes": calibration["warning_threshold_minutes"],
            "training_hard_limit_minutes": calibration["hard_threshold_minutes"],
            "gate3_output_bytes_before_final_report": output_bytes,
            "gate3_output_red_limit_bytes": 8 * 1024**3,
        },
        "isolation": {
            "formal_data_dir": str(formal_data_dir.resolve()),
            "formal_data_dir_exists": formal_data_dir.exists(),
            "formal_output_dir": str(formal_output_dir.resolve()),
            "formal_output_dir_exists": formal_output_dir.exists(),
        },
        "preserved_failed_evidence": [
            {
                "path": str((run_root / "04_pilot_train" / "compare_stage"
                             / "stage_monitor_report.json").resolve()),
                "status": "CRASHED",
                "resolution": "比较脚本空列表提前求值修复后，在全新 v3 目录 PASS",
            },
            {
                "path": str((run_root / "05_full_loc" / "stages" / "train"
                             / "stage_monitor_report.json").resolve()),
                "status": "TIMEOUT",
                "resolution": "日志编码与转发修复后，在全新目录完成三 split 并 PASS",
            },
        ],
        "performance_interpretation_allowed": False,
        "next_gate": "Gate 3B 需另行审批；Gate 3A 不构成论文性能复现",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 3A 第四章审计与可重复性工具")
    sub = parser.add_subparsers(dest="command", required=True)

    mat = sub.add_parser("audit-mat")
    mat.add_argument("--data_dir", type=Path, required=True)
    mat.add_argument("--expected_samples", type=int, nargs=3, required=True)
    mat.add_argument("--seed", type=int, required=True)
    mat.add_argument("--output", type=Path, required=True)

    loc = sub.add_parser("audit-loc")
    loc.add_argument("--data_dir", type=Path, required=True)
    loc.add_argument("--loc_data_dir", type=Path, required=True)
    loc.add_argument("--expected_samples", type=int, nargs=3, required=True)
    loc.add_argument("--mat_samples", type=int, nargs=3,
                     help="MAT 文件完整样本数；未指定时等于 expected_samples")
    loc.add_argument("--seed", type=int, required=True)
    loc.add_argument("--shard_size", type=int, required=True)
    loc.add_argument("--reference_loc_data_dir", type=Path)
    loc.add_argument("--reference_samples", type=int, nargs=3)
    loc.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare-training")
    compare.add_argument("--left_dir", type=Path, required=True)
    compare.add_argument("--right_dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    summary = sub.add_parser("summarize-calibration")
    summary.add_argument("--run_dirs", type=Path, nargs=2, required=True)
    summary.add_argument("--monitor_paths", type=Path, nargs=2, required=True)
    summary.add_argument("--projected_epochs", type=int, default=60)
    summary.add_argument("--scale_factor", type=float, default=8.0)
    summary.add_argument("--output", type=Path, required=True)

    final = sub.add_parser("finalize-gate3a")
    final.add_argument("--run_root", type=Path, required=True)
    final.add_argument("--full_loc_root", type=Path, required=True)
    final.add_argument("--formal_data_dir", type=Path, required=True)
    final.add_argument("--formal_output_dir", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    try:
        if args.command == "audit-mat":
            result, _ = audit_mat(args.data_dir.resolve(), expected_dict(args.expected_samples), args.seed)
        elif args.command == "audit-loc":
            if (args.reference_loc_data_dir is None) != (args.reference_samples is None):
                raise ValueError("reference_loc_data_dir 与 reference_samples 必须同时提供")
            result = audit_loc(
                args.data_dir.resolve(),
                args.loc_data_dir.resolve(),
                expected_dict(args.expected_samples),
                expected_dict(args.mat_samples or args.expected_samples),
                args.seed,
                args.shard_size,
                args.reference_loc_data_dir.resolve() if args.reference_loc_data_dir else None,
                expected_dict(args.reference_samples) if args.reference_samples else None,
            )
        elif args.command == "compare-training":
            result = compare_training(args.left_dir.resolve(), args.right_dir.resolve())
        elif args.command == "summarize-calibration":
            result = summarize_calibration(
                [path.resolve() for path in args.run_dirs],
                [path.resolve() for path in args.monitor_paths],
                args.projected_epochs,
                args.scale_factor,
            )
        else:
            result = finalize_gate3a(
                args.run_root.resolve(),
                args.full_loc_root.resolve(),
                args.formal_data_dir.resolve(),
                args.formal_output_dir.resolve(),
            )
        write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result.get("status") == "PASS" else 2
    except Exception as exc:  # noqa: BLE001 - 门禁必须保留失败证据
        failure = {
            "status": "FAIL",
            "command": args.command,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
