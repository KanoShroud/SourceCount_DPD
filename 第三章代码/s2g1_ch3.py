"""S2-G1 第三章数据、训练重复性和总门禁审计工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psutil
import torch

from gate1_smoke_ch3 import recompute_labels


EXPECTED_SAMPLES = {"train": 2048, "val": 512, "test": 512}
EXPECTED_CLASS_COUNTS = {
    "train": [410, 410, 614, 614],
    "val": [102, 102, 154, 154],
    "test": [102, 102, 154, 154],
}
EXPECTED_SEED = 20260823
EXPECTED_N_SUB = 19
EXPECTED_MAX_SRC = 3
EXPECTED_GRID = 81
GIB = 1024**3


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法 JSON 序列化 {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_default,
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scalar(handle: h5py.File, name: str) -> float:
    require(name in handle, f"缺少元数据 {name}")
    return float(np.asarray(handle[name]).reshape(-1)[0])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint(project_root: Path) -> list[dict[str, Any]]:
    files = [
        project_root / "第三章代码" / "main30.m",
        project_root / "第三章代码" / "train_v26.py",
        project_root / "第三章代码" / "s2g1_train_ch3.py",
        project_root / "第三章代码" / "s2g1_ch3.py",
        project_root / "第四章代码" / "gate3_stage_runner.py",
    ]
    return [
        {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def gpu_used_mib() -> float | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def residual_processes() -> list[dict[str, Any]]:
    markers = ("main30.m", "s2g1_train_ch3.py", "s2g1_ch3.py")
    residual = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info.get("cmdline") or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        process_name = str(process.info.get("name") or "").lower()
        relevant_process = "python" in process_name or "matlab" in process_name
        if (
            relevant_process
            and process.pid != psutil.Process().pid
            and any(marker in command for marker in markers)
        ):
            residual.append(
                {"pid": process.pid, "name": process.info.get("name"), "command": command}
            )
    return residual


def run_preflight(duration_seconds: float, interval_seconds: float) -> dict[str, Any]:
    require(duration_seconds >= 10, "preflight采样时间至少10秒")
    require(interval_seconds >= 1, "preflight采样间隔至少1秒")
    project_root = Path(__file__).resolve().parents[1]
    samples = []
    started = time.perf_counter()
    while True:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(project_root.anchor)
        samples.append(
            {
                "elapsed_seconds": time.perf_counter() - started,
                "system_available_bytes": int(memory.available),
                "system_used_bytes": int(memory.used),
                "gpu_used_mib": gpu_used_mib(),
                "disk_free_bytes": int(disk.free),
            }
        )
        if time.perf_counter() - started >= duration_seconds:
            break
        time.sleep(interval_seconds)
    available = [sample["system_available_bytes"] for sample in samples]
    gpu_values = [sample["gpu_used_mib"] for sample in samples if sample["gpu_used_mib"] is not None]
    disk_values = [sample["disk_free_bytes"] for sample in samples]
    median_available = statistics.median(available)
    maximum_gpu = max(gpu_values) if gpu_values else None
    minimum_disk = min(disk_values)
    residual = residual_processes()
    require(median_available >= 14 * GIB, "可用RAM中位数低于14 GiB")
    require(maximum_gpu is not None, "无法读取cuda:0显存")
    require(maximum_gpu < 12 * 1024, "启动前GPU总占用达到12 GiB预警线")
    require(minimum_disk >= 120 * GIB, "磁盘剩余低于120 GiB")
    require(not residual, f"发现遗留实验进程: {residual}")
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "status": "PASS",
        "duration_seconds": time.perf_counter() - started,
        "sample_count": len(samples),
        "median_system_available_bytes": int(median_available),
        "minimum_system_available_bytes": min(available),
        "maximum_gpu_used_mib": maximum_gpu,
        "minimum_disk_free_bytes": minimum_disk,
        "warnings": ["available_ram_below_16_gib"] if median_available < 16 * GIB else [],
        "residual_processes": residual,
        "git_head": git_head,
        "git_status": git_status,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_0": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "samples": samples,
    }
def build_manifest(data_dir: Path) -> dict[str, Any]:
    files = []
    for split in ("train", "val", "test"):
        path = data_dir / f"{split}_data.mat"
        require(path.is_file(), f"缺少数据文件: {path}")
        files.append(
            {
                "split": split,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "status": "PASS",
        "data_dir": str(data_dir.resolve()),
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "files": files,
    }


def audit_split(path: Path, split: str, seen_hashes: set[str]) -> dict[str, Any]:
    require(path.is_file(), f"缺少 {split} 数据: {path}")
    expected_n = EXPECTED_SAMPLES[split]
    with h5py.File(path, "r") as handle:
        required = {
            "mtr_sub_all",
            "src_count_all",
            "band_mask_all",
            "ignore_mask_all",
            "fc_offset_all",
            "src_pos_all",
            "position_retry_count_all",
            "position_constraint_failure_count_val",
            "position_min_separation_val",
            "position_max_retry_val",
            "sub_energy_all",
            "cov_mat_real_all",
            "cov_mat_imag_all",
            "N_sub_val",
            "max_src_val",
            "num_grid",
            "B_win_val",
            "B_step_val",
            "fs_val",
            "symbolRate_val",
            "BW_actual_val",
            "arfa_val",
            "sub_f_lo_val",
            "sub_f_hi_val",
            "thresh_val",
            "smoke_seed_val",
            "trials_list_val",
        }
        missing = sorted(required.difference(handle.keys()))
        require(not missing, f"{split} 缺少字段: {missing}")
        spectra_dataset = handle["mtr_sub_all"]
        require(
            spectra_dataset.shape == (EXPECTED_GRID, EXPECTED_GRID, EXPECTED_N_SUB, expected_n),
            f"{split} spectra原始shape={spectra_dataset.shape}",
        )
        require(spectra_dataset.dtype == np.float32, f"{split} spectra dtype={spectra_dataset.dtype}")
        duplicate_count = 0
        split_hashes: set[str] = set()
        spectra_min = float("inf")
        spectra_max = float("-inf")
        for sample_index in range(expected_n):
            sample = np.asarray(spectra_dataset[..., sample_index], dtype=np.float32)
            require(np.isfinite(sample).all(), f"{split}[{sample_index}] spectra含NaN/Inf")
            require((sample >= 0).all(), f"{split}[{sample_index}] spectra含负值")
            spectra_min = min(spectra_min, float(sample.min()))
            spectra_max = max(spectra_max, float(sample.max()))
            sample_hash = hashlib.sha256(sample.tobytes(order="C")).hexdigest()
            if sample_hash in split_hashes or sample_hash in seen_hashes:
                duplicate_count += 1
            split_hashes.add(sample_hash)
        require(duplicate_count == 0, f"{split} 发现重复或跨split泄漏样本 {duplicate_count} 条")
        seen_hashes.update(split_hashes)

        src_count = np.asarray(handle["src_count_all"]).reshape(-1).astype(np.int64)
        band_mask = np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0)
        ignore_mask = np.asarray(handle["ignore_mask_all"], dtype=np.float32).transpose(2, 1, 0)
        fc_offset = np.asarray(handle["fc_offset_all"], dtype=np.float32).T
        source_positions = np.asarray(handle["src_pos_all"], dtype=np.float32).transpose(2, 1, 0)
        position_retry_count = (
            np.asarray(handle["position_retry_count_all"]).reshape(-1).astype(np.int64)
        )
        position_constraint_failure_count = int(
            scalar(handle, "position_constraint_failure_count_val")
        )
        position_min_separation = scalar(handle, "position_min_separation_val")
        position_max_retry = int(scalar(handle, "position_max_retry_val"))
        sub_energy = np.asarray(handle["sub_energy_all"], dtype=np.float32).T
        cov_real = np.asarray(handle["cov_mat_real_all"])
        cov_imag = np.asarray(handle["cov_mat_imag_all"])
        sub_f_lo = np.asarray(handle["sub_f_lo_val"]).reshape(-1).astype(np.float64)
        sub_f_hi = np.asarray(handle["sub_f_hi_val"]).reshape(-1).astype(np.float64)
        metadata = {
            "n_sub": int(scalar(handle, "N_sub_val")),
            "max_src": int(scalar(handle, "max_src_val")),
            "num_grid": int(scalar(handle, "num_grid")),
            "band_window": scalar(handle, "B_win_val"),
            "band_step": scalar(handle, "B_step_val"),
            "sample_rate": scalar(handle, "fs_val"),
            "symbol_rate": scalar(handle, "symbolRate_val"),
            "actual_bandwidth": scalar(handle, "BW_actual_val"),
            "rolloff": scalar(handle, "arfa_val"),
            "threshold": scalar(handle, "thresh_val"),
            "generation_seed": int(scalar(handle, "smoke_seed_val")),
            "trials_list": np.asarray(handle["trials_list_val"]).reshape(-1).astype(int).tolist(),
        }

    require(src_count.shape == (expected_n,), f"{split} src_count shape={src_count.shape}")
    require(
        band_mask.shape == (expected_n, EXPECTED_MAX_SRC, EXPECTED_N_SUB),
        f"{split} band_mask shape={band_mask.shape}",
    )
    require(ignore_mask.shape == band_mask.shape, f"{split} ignore_mask shape={ignore_mask.shape}")
    require(fc_offset.shape == (expected_n, EXPECTED_MAX_SRC), f"{split} fc_offset shape={fc_offset.shape}")
    require(
        source_positions.shape == (expected_n, EXPECTED_MAX_SRC, 2),
        f"{split} src_pos shape={source_positions.shape}",
    )
    require(position_retry_count.shape == (expected_n,), f"{split} 位置重试shape错误")
    require((position_retry_count >= 0).all(), f"{split} 位置重试次数出现负值")
    require(position_min_separation == 300.0, f"{split} 最小源间距元数据错误")
    require(position_max_retry == 100, f"{split} 位置最大重试元数据错误")
    require(
        int(position_retry_count.sum()) == position_constraint_failure_count,
        f"{split} 位置重试总数与拒绝计数不一致",
    )
    require(
        int(position_retry_count.max(initial=0)) < position_max_retry,
        f"{split} 存在达到最大次数但未失败的位置重试",
    )
    require(np.isfinite(band_mask).all(), f"{split} band_mask含NaN/Inf")
    require(np.isfinite(ignore_mask).all(), f"{split} ignore_mask含NaN/Inf")
    require(np.isfinite(fc_offset).all(), f"{split} fc_offset含NaN/Inf")
    require(np.isfinite(source_positions).all(), f"{split} src_pos含NaN/Inf")
    require(np.isfinite(sub_energy).all() and (sub_energy >= 0).all(), f"{split} sub_energy异常")
    require(np.isfinite(cov_real).all() and np.isfinite(cov_imag).all(), f"{split} covariance异常")
    require(np.isin(src_count, [0, 1, 2, 3]).all(), f"{split} 源数越界")
    require(np.isin(band_mask, [0.0, 1.0]).all(), f"{split} band_mask非二值")
    require(np.isin(ignore_mask, [0.0, 1.0]).all(), f"{split} ignore_mask非二值")
    require(not np.logical_and(band_mask == 1, ignore_mask == 1).any(), f"{split} 标签互斥失败")
    class_counts = np.bincount(src_count, minlength=4).tolist()
    require(class_counts == EXPECTED_CLASS_COUNTS[split], f"{split} 类别分布={class_counts}")
    separation_violations = []
    minimum_active_separation = float("inf")
    for sample_index, count_value in enumerate(src_count):
        count = int(count_value)
        require(not band_mask[sample_index, count:, :].any(), f"{split}[{sample_index}] 空槽band非零")
        require(not ignore_mask[sample_index, count:, :].any(), f"{split}[{sample_index}] 空槽ignore非零")
        require(np.all(fc_offset[sample_index, count:] == 0), f"{split}[{sample_index}] 空槽频偏非零")
        active_fc = fc_offset[sample_index, :count]
        require(count < 2 or np.all(np.diff(active_fc) >= 0), f"{split}[{sample_index}] 频率未排序")
        require(
            np.all(source_positions[sample_index, count:, :] == 0),
            f"{split}[{sample_index}] 空槽位置非零",
        )
        if count >= 2:
            active_positions = source_positions[sample_index, :count, :]
            distances = np.sqrt(
                np.sum(
                    (active_positions[:, None, :] - active_positions[None, :, :]) ** 2,
                    axis=-1,
                )
            )
            pair_distances = distances[np.triu_indices(count, 1)]
            current_minimum = float(pair_distances.min())
            minimum_active_separation = min(minimum_active_separation, current_minimum)
            if current_minimum < 300.0:
                separation_violations.append(
                    {
                        "sample_index": sample_index,
                        "source_count": count,
                        "minimum_separation_m": current_minimum,
                    }
                )
    require(
        not separation_violations,
        f"{split} 有 {len(separation_violations)} 条样本违反300m最小源间距，"
        f"前5条={separation_violations[:5]}",
    )
    require(metadata["n_sub"] == EXPECTED_N_SUB, f"{split} N_sub元数据错误")
    require(metadata["max_src"] == EXPECTED_MAX_SRC, f"{split} max_src元数据错误")
    require(metadata["num_grid"] == EXPECTED_GRID, f"{split} 网格元数据错误")
    require(metadata["generation_seed"] == EXPECTED_SEED, f"{split} seed元数据错误")
    require(metadata["trials_list"] == [2048, 512, 512], f"{split} trials元数据错误")
    require(sub_f_lo.shape == (EXPECTED_N_SUB,), f"{split} 子带下界shape错误")
    require(sub_f_hi.shape == (EXPECTED_N_SUB,), f"{split} 子带上界shape错误")
    recomputed_band, recomputed_ignore = recompute_labels(
        src_count,
        fc_offset,
        sub_f_lo,
        sub_f_hi,
        metadata["symbol_rate"],
        metadata["actual_bandwidth"],
        metadata["band_window"],
        metadata["threshold"],
    )
    band_mismatches = int(np.count_nonzero(recomputed_band != band_mask))
    ignore_mismatches = int(np.count_nonzero(recomputed_ignore != ignore_mask))
    require(band_mismatches == 0, f"{split} band标签复算不一致 {band_mismatches} 项")
    require(ignore_mismatches == 0, f"{split} ignore标签复算不一致 {ignore_mismatches} 项")
    return {
        "status": "PASS",
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "file_sha256": sha256_file(path),
        "logical_spectra_shape": [expected_n, EXPECTED_N_SUB, EXPECTED_GRID, EXPECTED_GRID],
        "class_counts": dict(sorted(Counter(src_count.tolist()).items())),
        "sample_hash_count": len(split_hashes),
        "duplicate_or_cross_split_count": duplicate_count,
        "minimum_active_source_separation_m": minimum_active_separation,
        "source_separation_violation_count": len(separation_violations),
        "position_retry_sample_count": int(np.count_nonzero(position_retry_count)),
        "position_retry_total": int(position_retry_count.sum()),
        "position_constraint_failure_count": position_constraint_failure_count,
        "position_max_retry": position_max_retry,
        "spectra_min": spectra_min,
        "spectra_max": spectra_max,
        "band_positive_count": int(np.count_nonzero(band_mask)),
        "ignore_count": int(np.count_nonzero(ignore_mask)),
        "label_recompute_band_mismatches": band_mismatches,
        "label_recompute_ignore_mismatches": ignore_mismatches,
        "metadata": metadata,
    }


def audit_all(data_dir: Path) -> dict[str, Any]:
    seen_hashes: set[str] = set()
    splits = {
        split: audit_split(data_dir / f"{split}_data.mat", split, seen_hashes)
        for split in ("train", "val", "test")
    }
    reference = splits["train"]["metadata"]
    for split in ("val", "test"):
        require(splits[split]["metadata"] == reference, f"{split} 元数据与train不一致")
    return {
        "status": "PASS",
        "data_dir": str(data_dir.resolve()),
        "total_unique_sample_hashes": len(seen_hashes),
        "splits": splits,
    }


def compare_nested(left: Any, right: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape or not torch.equal(left, right):
            mismatches.append(path)
        return
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            mismatches.append(path)
        return
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            mismatches.append(f"{path}.keys")
            return
        for key in left:
            compare_nested(left[key], right[key], f"{path}.{key}", mismatches)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            mismatches.append(f"{path}.length")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            compare_nested(left_item, right_item, f"{path}[{index}]", mismatches)
        return
    if type(left) is not type(right) or left != right:
        mismatches.append(path)


def normalized_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ignored = {
        "epoch_seconds",
        "process_rss_bytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    }
    return [{key: value for key, value in row.items() if key not in ignored} for row in history]


def normalized_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in {"output_dir"}
    }


def compare_training(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left_history = normalized_history(load_json(left_dir / "epoch_history.json"))
    right_history = normalized_history(load_json(right_dir / "epoch_history.json"))
    left_initial = load_json(left_dir / "initial_validation.json")
    right_initial = load_json(right_dir / "initial_validation.json")
    left_final = load_json(left_dir / "final_validation.json")
    right_final = load_json(right_dir / "final_validation.json")
    left_config = normalized_config(load_json(left_dir / "run_config.json"))
    right_config = normalized_config(load_json(right_dir / "run_config.json"))
    left_summary = load_json(left_dir / "training_summary.json")
    right_summary = load_json(right_dir / "training_summary.json")
    mismatches: list[str] = []
    compare_nested(left_history, right_history, "history", mismatches)
    compare_nested(left_initial, right_initial, "initial", mismatches)
    compare_nested(left_final, right_final, "final", mismatches)
    compare_nested(left_config, right_config, "config", mismatches)
    for key in (
        "epochs_completed",
        "stopped_early",
        "best_epoch",
        "best_validation",
        "initial_validation",
        "learning_gate",
        "convergence_gate",
    ):
        compare_nested(left_summary[key], right_summary[key], f"summary.{key}", mismatches)

    checkpoint_results = {}
    for kind in ("best", "last"):
        left_path = next(left_dir.glob(f"{kind}_model_v26_*.pth"))
        right_path = next(right_dir.glob(f"{kind}_model_v26_*.pth"))
        left_checkpoint = torch.load(left_path, map_location="cpu", weights_only=False)
        right_checkpoint = torch.load(right_path, map_location="cpu", weights_only=False)
        state_mismatches: list[str] = []
        for key in ("epoch", "model", "optimizer", "scheduler", "validation", "rng_state"):
            compare_nested(
                left_checkpoint[key],
                right_checkpoint[key],
                f"{kind}.{key}",
                state_mismatches,
            )
        mismatches.extend(state_mismatches)
        checkpoint_results[kind] = {
            "left": str(left_path.resolve()),
            "right": str(right_path.resolve()),
            "left_sha256": sha256_file(left_path),
            "right_sha256": sha256_file(right_path),
            "state_mismatch_count": len(state_mismatches),
        }
    require(not mismatches, f"两次训练不一致，前20项: {mismatches[:20]}")
    return {
        "status": "PASS",
        "determinism_class": "same data, configuration, seed, code and environment",
        "scientific_mismatch_count": 0,
        "mismatches": [],
        "epochs_compared": len(left_history),
        "best_epoch": left_summary["best_epoch"],
        "checkpoints": checkpoint_results,
    }


def freeze_checkpoint(run_dir: Path, comparison_path: Path) -> dict[str, Any]:
    comparison = load_json(comparison_path)
    summary = load_json(run_dir / "training_summary.json")
    require(comparison["status"] == "PASS", "训练重复性未通过")
    require(summary["learning_gate"]["pass"], "有效学习门槛未通过")
    require(summary["convergence_gate"]["pass"], "收敛门槛未通过")
    checkpoint = next(run_dir.glob("best_model_v26_*.pth"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(int(payload["epoch"]) == int(summary["best_epoch"]), "best epoch不一致")
    return {
        "status": "PASS",
        "selection_rule": "minimum validation loss after warmup; test not accessed",
        "source_run": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(payload["epoch"]),
        "validation": payload["validation"],
        "learning_gate": summary["learning_gate"],
        "convergence_gate": summary["convergence_gate"],
    }


def compare_evaluations(left: Path, right: Path) -> dict[str, Any]:
    left_report = load_json(left)
    right_report = load_json(right)
    mismatches: list[str] = []
    for key in (
        "status",
        "evaluation_mode",
        "split",
        "checkpoint",
        "checkpoint_epoch",
        "device",
        "batch_size",
        "seed",
        "deterministic",
        "metrics",
    ):
        compare_nested(left_report[key], right_report[key], key, mismatches)
    require(not mismatches, f"独立评估不一致: {mismatches[:20]}")
    return {
        "status": "PASS",
        "split": left_report["split"],
        "checkpoint": left_report["checkpoint"],
        "mismatch_count": 0,
        "metrics": left_report["metrics"],
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    before = load_json(args.manifest_before)
    after = load_json(args.manifest_after)
    audit = load_json(args.data_audit)
    capacity = load_json(args.capacity)
    comparison = load_json(args.training_comparison)
    frozen = load_json(args.frozen)
    validation = load_json(args.validation_comparison)
    test = load_json(args.test_report)
    require(before["files"] == after["files"], "训练前后数据manifest不一致")
    require(audit["status"] == "PASS", "数据审计未通过")
    require(capacity["status"] == "PASS", "容量检查未通过")
    require(comparison["status"] == "PASS", "训练重复性未通过")
    require(frozen["status"] == "PASS", "checkpoint冻结未通过")
    require(validation["status"] == "PASS", "validation重复评估未通过")
    require(test["status"] == "PASS" and test["split"] == "test", "test评估未通过")
    require(test["checkpoint"] == frozen["checkpoint"], "test未使用冻结checkpoint")
    monitor_paths = sorted(args.run_root.rglob("stage_monitor_report.json"))
    require(len(monitor_paths) >= 8, f"资源监控报告不足: {len(monitor_paths)}")
    monitors = [load_json(path) for path in monitor_paths]
    for path, monitor in zip(monitor_paths, monitors, strict=True):
        require(monitor["status"] == "PASS", f"{path} status={monitor['status']}")
        require(not monitor.get("red_flags"), f"{path}存在资源红线")
    project_root = Path(__file__).resolve().parents[1]
    return {
        "gate": "S2-G1",
        "experiment_id": f"CH3-S2G1-{args.run_root.name[:8]}",
        "status": "PASS",
        "run_root": str(args.run_root.resolve()),
        "data": {
            "train_val_test": [2048, 512, 512],
            "total_bytes": before["total_bytes"],
            "sha256_before_after": "exact_match",
            "audit": str(args.data_audit.resolve()),
        },
        "training": {
            "configuration": "Transformer M=10, FP32, batch 64, 100 epoch max",
            "strict_repeatability": "exact_match",
            "best_epoch": frozen["checkpoint_epoch"],
            "learning_gate": frozen["learning_gate"],
            "convergence_gate": frozen["convergence_gate"],
        },
        "frozen_checkpoint": frozen,
        "validation": validation,
        "test": test,
        "monitor_summary": {
            "report_count": len(monitors),
            "warnings": sorted({item for monitor in monitors for item in monitor.get("warnings", [])}),
            "minimum_system_available_bytes": min(
                monitor["minimum_system_available_bytes"] for monitor in monitors
            ),
            "maximum_process_tree_rss_bytes": max(
                monitor["maximum_process_tree_rss_bytes"] for monitor in monitors
            ),
            "maximum_gpu_used_mib": max(
                monitor.get("maximum_gpu_used_mib") or 0 for monitor in monitors
            ),
        },
        "code_fingerprint": code_fingerprint(project_root),
        "scope": "缩减数据收敛与重复性证据；不是论文正式指标复现",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2-G1 第三章审计工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest-data")
    manifest.add_argument("--data_dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--duration_seconds", type=float, default=60.0)
    preflight.add_argument("--interval_seconds", type=float, default=2.0)
    preflight.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit-data")
    audit.add_argument("--data_dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare-training")
    compare.add_argument("--left_dir", type=Path, required=True)
    compare.add_argument("--right_dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    freeze = subparsers.add_parser("freeze-checkpoint")
    freeze.add_argument("--run_dir", type=Path, required=True)
    freeze.add_argument("--comparison", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    eval_compare = subparsers.add_parser("compare-evaluations")
    eval_compare.add_argument("--left", type=Path, required=True)
    eval_compare.add_argument("--right", type=Path, required=True)
    eval_compare.add_argument("--output", type=Path, required=True)

    final = subparsers.add_parser("finalize")
    final.add_argument("--run_root", type=Path, required=True)
    final.add_argument("--manifest_before", type=Path, required=True)
    final.add_argument("--manifest_after", type=Path, required=True)
    final.add_argument("--data_audit", type=Path, required=True)
    final.add_argument("--capacity", type=Path, required=True)
    final.add_argument("--training_comparison", type=Path, required=True)
    final.add_argument("--frozen", type=Path, required=True)
    final.add_argument("--validation_comparison", type=Path, required=True)
    final.add_argument("--test_report", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"拒绝覆盖报告: {args.output}")
    try:
        if args.command == "preflight":
            result = run_preflight(args.duration_seconds, args.interval_seconds)
        elif args.command == "manifest-data":
            result = build_manifest(args.data_dir.resolve())
        elif args.command == "audit-data":
            result = audit_all(args.data_dir.resolve())
        elif args.command == "compare-training":
            result = compare_training(args.left_dir.resolve(), args.right_dir.resolve())
        elif args.command == "freeze-checkpoint":
            result = freeze_checkpoint(args.run_dir.resolve(), args.comparison.resolve())
        elif args.command == "compare-evaluations":
            result = compare_evaluations(args.left.resolve(), args.right.resolve())
        elif args.command == "finalize":
            result = finalize(args)
        else:
            raise AssertionError(args.command)
    except Exception as error:
        result = {
            "status": "FAIL",
            "command": args.command,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(args.output.resolve(), result)
        print(json.dumps(result, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    write_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
