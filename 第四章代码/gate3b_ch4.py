"""Gate 3B：D8 60 epoch 双重复训练、冻结评估与最终门禁。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch


SPLITS = ("train", "val", "test")
EXPECTED_TASKS = {"train": 1024, "val": 256, "test": 256}
EXPECTED_SHARDS = {"train": 8, "val": 2, "test": 2}
GIB = 1024**3
SCRIPT_DIR = Path(__file__).resolve().parent


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    raise TypeError(f"无法 JSON 序列化 {type(value).__name__}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload, handle, ensure_ascii=False, indent=2, allow_nan=False,
            default=json_default,
        )


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def residual_experiment_processes(run_root: Path) -> list[dict[str, Any]]:
    markers = ("train_yolo.py", "eval_ch4_checkpoint.py", "gate3_stage_runner.py")
    root_text = str(run_root.resolve()).lower()
    residual: list[dict[str, Any]] = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            if process.info["pid"] == os.getpid():
                continue
            command = " ".join(process.info.get("cmdline") or [])
            lowered = command.lower()
            if root_text in lowered and any(marker in lowered for marker in markers):
                residual.append({"pid": process.info["pid"], "command": command})
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return residual


def require_finite_nested(value: Any, path: str = "root") -> None:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            require(bool(torch.isfinite(value).all()), f"{path} 含 NaN/Inf")
        return
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            require(bool(np.isfinite(value).all()), f"{path} 含 NaN/Inf")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite_nested(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            require_finite_nested(item, f"{path}[{index}]")
        return
    if isinstance(value, float):
        require(math.isfinite(value), f"{path} 非有限: {value}")


def compare_nested(left: Any, right: Any, path: str, mismatches: list[str]) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if not torch.equal(left, right):
            if left.shape == right.shape and left.numel() and left.is_floating_point():
                max_diff = float(torch.max(torch.abs(left - right)).item())
                mismatches.append(f"{path}: tensor mismatch max_abs_diff={max_diff}")
            else:
                mismatches.append(f"{path}: tensor mismatch")
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


def single_checkpoint(run_dir: Path, kind: str) -> Path:
    matches = list(run_dir.glob(f"{kind}_yolo_*.pth"))
    require(len(matches) == 1, f"{run_dir} {kind} checkpoint 数={len(matches)}")
    return matches[0]


def build_data_manifest(data_dir: Path) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}
    for split in SPLITS:
        split_dir = data_dir / split
        index_path = split_dir / f"loc_{split}_index.pt"
        require(index_path.is_file(), f"缺少 {index_path}")
        index = torch.load(index_path, map_location="cpu", weights_only=False)
        require(index.get("n_total_tasks") == EXPECTED_TASKS[split], f"{split} task 数不符")
        require(index.get("n_shards") == EXPECTED_SHARDS[split], f"{split} shard 数不符")
        shard_names = index.get("shard_files")
        require(isinstance(shard_names, list), f"{split} shard_files 非列表")
        require(len(shard_names) == EXPECTED_SHARDS[split], f"{split} shard_files 数不符")
        require(len(set(shard_names)) == len(shard_names), f"{split} shard 文件名重复")
        paths = [index_path, *(split_dir / name for name in shard_names)]
        split_bytes = 0
        for path in paths:
            require(path.is_file(), f"缺少 {path}")
            size = path.stat().st_size
            split_bytes += size
            entries.append({
                "relative_path": path.relative_to(data_dir).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            })
        split_summary[split] = {
            "task_count": EXPECTED_TASKS[split],
            "shard_count": EXPECTED_SHARDS[split],
            "file_count": len(paths),
            "total_bytes": split_bytes,
        }
    entries.sort(key=lambda item: item["relative_path"])
    require(len(entries) == 15, f"数据清单文件数应为 15，实际 {len(entries)}")
    return {
        "status": "PASS",
        "manifest_version": 1,
        "data_role": "Gate 3A legacy root reused read-only by Gate 3B",
        "data_dir": str(data_dir.resolve()),
        "splits": split_summary,
        "file_count": len(entries),
        "total_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    }


def expected_training_args() -> dict[str, Any]:
    return {
        "method": "dualhead",
        "amp": False,
        "dice_weight": 0.0,
        "grad_alpha": 1.0,
        "offset_weight": 1.0,
        "conf_weight_offset": False,
        "soft_conf": False,
        "batch_size": 8,
        "val_batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 5e-3,
        "dropout": 0.4,
        "eval_every": 1,
        "epochs": 60,
        "patience": 60,
        "seed": 42,
        "deterministic": True,
        "fail_on_nonfinite": True,
        "save_last_every_epoch": True,
        "require_empty_output": True,
        "gate3_d8": False,
        "gate3b_d8": True,
    }


def audit_training(run_dir: Path, monitor_path: Path) -> dict[str, Any]:
    config = load_json(run_dir / "run_config.json")
    summary = load_json(run_dir / "training_summary.json")
    history = load_json(run_dir / "epoch_history.json")
    initial = load_json(run_dir / "initial_validation.json")
    final = load_json(run_dir / "final_validation.json")
    monitor = load_json(monitor_path)

    require(config.get("status") == "CONFIGURED", "run_config 状态不符")
    require(config.get("model_label") == "D8", "模型标签不是 D8")
    saved_args = config.get("args")
    require(isinstance(saved_args, dict), "run_config 缺少 args")
    mismatches = [
        f"{key}={saved_args.get(key)!r}, expected={value!r}"
        for key, value in expected_training_args().items()
        if saved_args.get(key) != value
    ]
    require(not mismatches, "训练冻结配置不一致: " + "; ".join(mismatches))
    require(summary.get("status") == "PASS", "training_summary 非 PASS")
    require(summary.get("epochs_completed") == 60, "未完成 60 epoch")
    require(summary.get("stopped_early") is False, "训练发生 early stopping")
    require(len(history) == 60, f"epoch_history 长度={len(history)}")
    require([item.get("epoch") for item in history] == list(range(1, 61)), "epoch 序列不连续")
    require(all(item.get("train_batches") == 128 for item in history), "train batch 数不是每 epoch 128")
    require(math.isclose(float(history[-1]["learning_rate"]), 1e-6, rel_tol=0.0, abs_tol=1e-12),
            f"最终学习率={history[-1]['learning_rate']}")
    require(math.isclose(float(summary["final_learning_rate"]), 1e-6,
                         rel_tol=0.0, abs_tol=1e-12), "summary 最终学习率不符")
    require_finite_nested(history, "history")
    require_finite_nested(initial, "initial_validation")
    require_finite_nested(final, "final_validation")

    best_path = single_checkpoint(run_dir, "best")
    last_path = single_checkpoint(run_dir, "last")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    require_finite_nested(best["model"], "best.model")
    require_finite_nested(last["model"], "last.model")
    best_epoch = summary.get("best_epoch")
    require(isinstance(best_epoch, int) and 6 <= best_epoch <= 60,
            f"best_epoch 非法: {best_epoch!r}")
    require(best.get("epoch") == best_epoch, "best checkpoint epoch 与 summary 不同")
    require(last.get("epoch") == 60, "last checkpoint 不是第 60 epoch")
    require(best.get("best_rmse") == summary.get("best_rmse"), "best_rmse 不一致")
    require(final.get("checkpoint_selection") == "best_rmse", "最终验证未使用 best")

    initial_metrics = initial.get("metrics")
    final_metrics = final.get("metrics")
    require(isinstance(initial_metrics, dict) and isinstance(final_metrics, dict), "验证指标缺失")
    require(final_metrics.get("rmse") == summary.get("best_rmse"),
            "最终 validation RMSE 与 best_rmse 不一致")
    rmse_improved = float(final_metrics["rmse"]) < float(initial_metrics["rmse"])
    loss_improved = float(final_metrics["val_loss"]) < float(initial_metrics["val_loss"])
    require(rmse_improved and loss_improved,
            "训练未同时改善 validation RMSE 与 validation loss")
    require(initial.get("rng_state_restored") is True, "初始验证后未声明恢复 RNG")

    require(monitor.get("status") == "PASS", f"训练监控状态={monitor.get('status')}")
    warnings = monitor.get("warnings", [])
    red_flags = monitor.get("red_flags", [])
    require(not red_flags, f"训练存在 red flag: {red_flags}")
    status = "REVIEW_REQUIRED" if warnings else "PASS"
    return {
        "status": status,
        "run_dir": str(run_dir.resolve()),
        "epochs_completed": 60,
        "train_batches_per_epoch": 128,
        "best_epoch": best_epoch,
        "best_rmse": float(summary["best_rmse"]),
        "initial_validation": initial_metrics,
        "final_validation": final_metrics,
        "learning": {
            "status": "LEARNING_CONFIRMED",
            "rmse_improved": rmse_improved,
            "val_loss_improved": loss_improved,
        },
        "trend": (
            "STILL_IMPROVING_AT_LIMIT" if best_epoch >= 56
            else "NO_BEST_IN_FINAL_FIVE_EPOCHS"
        ),
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "monitor": {
            "path": str(monitor_path.resolve()),
            "duration_seconds": monitor.get("duration_seconds"),
            "warnings": warnings,
            "red_flags": red_flags,
            "minimum_system_available_bytes": monitor.get("minimum_system_available_bytes"),
            "maximum_process_tree_rss_bytes": monitor.get("maximum_process_tree_rss_bytes"),
            "maximum_gpu_used_mib": monitor.get("maximum_gpu_used_mib"),
        },
        "performance_interpretation_allowed": False,
    }


def compare_training(left_dir: Path, right_dir: Path) -> dict[str, Any]:
    left_history = load_json(left_dir / "epoch_history.json")
    right_history = load_json(right_dir / "epoch_history.json")
    require(len(left_history) == len(right_history) == 60, "两次训练应均为 60 epoch")
    excluded = {
        "epoch_seconds", "process_rss_bytes", "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
    }
    history_mismatches: list[str] = []
    for index, (left, right) in enumerate(zip(left_history, right_history, strict=True)):
        keys = set(left).union(right).difference(excluded)
        for key in sorted(keys):
            if left.get(key) != right.get(key):
                history_mismatches.append(
                    f"epoch={index + 1} {key}: {left.get(key)!r} != {right.get(key)!r}"
                )

    artifact_mismatches: dict[str, list[str]] = {}
    for filename in ("initial_validation.json", "final_validation.json"):
        mismatches: list[str] = []
        compare_nested(load_json(left_dir / filename), load_json(right_dir / filename),
                       filename, mismatches)
        artifact_mismatches[filename] = mismatches

    summary_keys = (
        "model_label", "epochs_completed", "stopped_early", "best_epoch",
        "best_rmse", "final_learning_rate", "initial_validation_metrics",
    )
    left_summary = load_json(left_dir / "training_summary.json")
    right_summary = load_json(right_dir / "training_summary.json")
    summary_mismatches = [
        f"{key}: {left_summary.get(key)!r} != {right_summary.get(key)!r}"
        for key in summary_keys if left_summary.get(key) != right_summary.get(key)
    ]

    checkpoint_mismatches: dict[str, list[str]] = {}
    checkpoint_paths: dict[str, dict[str, str]] = {}
    for kind in ("best", "last"):
        left_path = single_checkpoint(left_dir, kind)
        right_path = single_checkpoint(right_dir, kind)
        left_checkpoint = torch.load(left_path, map_location="cpu", weights_only=False)
        right_checkpoint = torch.load(right_path, map_location="cpu", weights_only=False)
        mismatches = []
        for key in ("model", "optimizer", "scheduler", "scaler", "rng_state"):
            compare_nested(left_checkpoint[key], right_checkpoint[key],
                           f"{kind}.{key}", mismatches)
        for key in ("epoch", "best_rmse", "method", "save_tag"):
            compare_nested(left_checkpoint.get(key), right_checkpoint.get(key),
                           f"{kind}.{key}", mismatches)
        checkpoint_mismatches[kind] = mismatches
        checkpoint_paths[kind] = {
            "left": str(left_path.resolve()),
            "right": str(right_path.resolve()),
        }

    require(not history_mismatches,
            "逐 epoch 指标不一致: " + (history_mismatches[0] if history_mismatches else ""))
    for filename, mismatches in artifact_mismatches.items():
        require(not mismatches, f"{filename} 不一致: {mismatches[0] if mismatches else ''}")
    require(not summary_mismatches,
            "training_summary 不一致: " + (summary_mismatches[0] if summary_mismatches else ""))
    for kind, mismatches in checkpoint_mismatches.items():
        require(not mismatches, f"{kind} checkpoint 不一致: {mismatches[0] if mismatches else ''}")
    return {
        "status": "PASS",
        "determinism_class": "same seed, data, code, environment",
        "numeric_requirement": "exact match excluding timing and resource fields",
        "epochs_compared": 60,
        "history_mismatches": history_mismatches,
        "artifact_mismatches": artifact_mismatches,
        "summary_mismatches": summary_mismatches,
        "checkpoint_mismatches": checkpoint_mismatches,
        "checkpoint_paths": checkpoint_paths,
        "performance_interpretation_allowed": False,
    }


def freeze_checkpoint(run_dir: Path, comparison_path: Path) -> dict[str, Any]:
    comparison = load_json(comparison_path)
    require(comparison.get("status") == "PASS", "训练重复性比较未通过")
    checkpoint_path = single_checkpoint(run_dir, "best")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint.get("method") == "dualhead", "冻结 checkpoint method 不符")
    require(checkpoint.get("save_tag") == "dualhead_std", "冻结 checkpoint save_tag 不符")
    require(checkpoint.get("args", {}).get("gate3b_d8") is True, "不是 Gate 3B checkpoint")
    require(isinstance(checkpoint.get("epoch"), int) and 6 <= checkpoint["epoch"] <= 60,
            "冻结 checkpoint epoch 非法")
    require_finite_nested(checkpoint["model"], "frozen.model")
    return {
        "status": "PASS",
        "selection_split": "validation",
        "selection_metric": "minimum_rmse_from_epoch_6_to_60",
        "test_used_for_selection": False,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_best_rmse": checkpoint["best_rmse"],
        "model": "D8 dualhead_std",
        "amp": False,
        "oracle_k_only": True,
        "performance_interpretation_allowed": False,
    }


def compare_evaluations(
    left_path: Path, right_path: Path, training_final_path: Path | None,
) -> dict[str, Any]:
    left = load_json(left_path)
    right = load_json(right_path)
    keys = (
        "status", "split", "checkpoint", "checkpoint_sha256", "checkpoint_epoch",
        "checkpoint_best_rmse", "model", "amp", "batch_size", "seed",
        "deterministic", "evaluation_mode", "task_count",
        "source_count_distribution", "metrics", "performance_interpretation_allowed",
    )
    mismatches = [
        f"{key}: {left.get(key)!r} != {right.get(key)!r}"
        for key in keys if left.get(key) != right.get(key)
    ]
    require(not mismatches, "独立评估不一致: " + (mismatches[0] if mismatches else ""))
    require(left.get("status") == "PASS", "评估状态非 PASS")
    require(left.get("evaluation_mode") == "oracle-K_ground_truth_source_count",
            "评估未标记 oracle-K")
    require(left.get("task_count") == 256, "评估任务数不是 256")
    require(left.get("source_count_distribution") == {"2": 128, "3": 128},
            f"源数分布不符: {left.get('source_count_distribution')}")
    metrics = left.get("metrics")
    require(isinstance(metrics, dict), "评估 metrics 缺失")
    require(metrics.get("count_N2") == 256 and metrics.get("count_N3") == 384,
            "N2/N3 误差样本数不符")
    require_finite_nested(metrics, "evaluation.metrics")
    training_match = None
    if training_final_path is not None:
        require(left.get("split") == "val", "训练 final 只允许对照 val")
        training_final = load_json(training_final_path)
        training_metrics = training_final.get("metrics")
        training_mismatches: list[str] = []
        compare_nested(metrics, training_metrics, "validation.metrics", training_mismatches)
        require(not training_mismatches,
                "独立 validation 与训练 best validation 不一致: "
                + (training_mismatches[0] if training_mismatches else ""))
        training_match = "exact"
    return {
        "status": "PASS",
        "split": left["split"],
        "numeric_requirement": "exact match excluding elapsed time and environment",
        "mismatches": mismatches,
        "training_best_validation_match": training_match,
        "checkpoint_sha256": left["checkpoint_sha256"],
        "task_count": left["task_count"],
        "source_count_distribution": left["source_count_distribution"],
        "metrics": metrics,
        "evaluation_mode": left["evaluation_mode"],
        "performance_interpretation_allowed": False,
    }


def manifest_files(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return manifest.get("files", [])


def finalize_gate3b(
    run_root: Path,
    manifest_before_path: Path,
    manifest_after_path: Path,
    audit_a_path: Path,
    audit_b_path: Path,
    training_comparison_path: Path,
    frozen_path: Path,
    val_comparison_path: Path,
    test_comparison_path: Path,
    formal_data_dir: Path,
    formal_output_dir: Path,
) -> dict[str, Any]:
    paths = {
        "manifest_before": manifest_before_path,
        "manifest_after": manifest_after_path,
        "audit_a": audit_a_path,
        "audit_b": audit_b_path,
        "training_comparison": training_comparison_path,
        "frozen_checkpoint": frozen_path,
        "validation_comparison": val_comparison_path,
        "test_comparison": test_comparison_path,
    }
    reports = {name: load_json(path) for name, path in paths.items()}
    for name, report in reports.items():
        require(report.get("status") == "PASS", f"{name} status={report.get('status')}")
    require(manifest_files(reports["manifest_before"]) == manifest_files(reports["manifest_after"]),
            "训练前后数据清单或 SHA256 发生变化")
    require(reports["manifest_before"].get("total_bytes") == 3_951_967_293,
            f"输入数据字节数变化: {reports['manifest_before'].get('total_bytes')}")
    require(reports["training_comparison"].get("epochs_compared") == 60,
            "训练重复性比较未覆盖 60 epoch")
    frozen_hash = reports["frozen_checkpoint"]["checkpoint_sha256"]
    require(reports["validation_comparison"].get("checkpoint_sha256") == frozen_hash,
            "validation 未使用冻结 checkpoint")
    require(reports["test_comparison"].get("checkpoint_sha256") == frozen_hash,
            "test 未使用冻结 checkpoint")
    require(reports["audit_a"]["learning"]["status"] == "LEARNING_CONFIRMED",
            "训练 A 未确认学习")
    require(reports["audit_b"]["learning"]["status"] == "LEARNING_CONFIRMED",
            "训练 B 未确认学习")
    require(not formal_data_dir.exists(), f"formal data 目录不应存在: {formal_data_dir}")
    require(not formal_output_dir.exists(), f"formal output 目录不应存在: {formal_output_dir}")
    residual_processes = residual_experiment_processes(run_root)
    require(not residual_processes, f"存在残留 Gate 3B 进程: {residual_processes}")

    monitor_paths = sorted(run_root.rglob("stage_monitor_report.json"))
    require(len(monitor_paths) >= 8, f"监控报告数量不足: {len(monitor_paths)}")
    monitors = [load_json(path) for path in monitor_paths]
    for path, monitor in zip(monitor_paths, monitors, strict=True):
        require(monitor.get("status") == "PASS", f"{path} status={monitor.get('status')}")
        require(not monitor.get("red_flags"), f"{path} red_flags={monitor.get('red_flags')}")
    warnings = sorted({item for monitor in monitors for item in monitor.get("warnings", [])})
    output_bytes = directory_size(run_root)
    require(output_bytes <= 4 * GIB, f"Gate 3B 输出超过 4 GiB: {output_bytes}")
    if output_bytes > 2 * GIB:
        warnings.append("gate3b_output_above_2_gib")

    best_epoch = int(reports["audit_a"]["best_epoch"])
    if warnings:
        status = "REVIEW_REQUIRED"
    elif best_epoch >= 56:
        status = "PASS_WITH_MORE_EPOCHS_RECOMMENDED"
    else:
        status = "PASS"
    test_metrics = dict(reports["test_comparison"]["metrics"])
    test_metrics["test_loss"] = test_metrics.pop("val_loss")
    code_artifacts = {}
    for name in (
        "train_yolo.py", "eval_ch4_checkpoint.py", "gate3_stage_runner.py",
        "gate3b_ch4.py",
    ):
        path = SCRIPT_DIR / name
        code_artifacts[name] = {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return {
        "status": status,
        "gate": "Gate 3B",
        "scope": "D8 no-AMP 小规模 60 epoch 双重复训练与 oracle-K 独立评估",
        "run_root": str(run_root.resolve()),
        "legacy_gate3a_root_renamed": False,
        "legacy_gate3a_root_policy": (
            "outputs/gate3_ch4 is preserved as immutable historical evidence; "
            "Gate 3B uses outputs/gate3b_ch4"
        ),
        "reports": {
            name: {"path": str(paths[name].resolve()), "status": report["status"]}
            for name, report in reports.items()
        },
        "data_integrity": {
            "manifest_file_count": reports["manifest_before"]["file_count"],
            "total_bytes": reports["manifest_before"]["total_bytes"],
            "before_after_sha256": "exact_match",
        },
        "reproducibility": {
            "model": "D8 dualhead_std",
            "amp": False,
            "batch_size": 8,
            "epochs": 60,
            "seed": 42,
            "numeric_requirement": "exact match",
            "result": "REPRODUCIBLE",
        },
        "learning": reports["audit_a"]["learning"],
        "trend": reports["audit_a"]["trend"],
        "best_epoch": best_epoch,
        "frozen_checkpoint": reports["frozen_checkpoint"],
        "validation_oracle_k": reports["validation_comparison"]["metrics"],
        "test_oracle_k": test_metrics,
        "raw_test_loss_field_semantics": (
            "eval_ch4_checkpoint raw metrics.val_loss is the loss on the requested split; "
            "for split=test it is reported here as test_loss"
        ),
        "code_artifacts": code_artifacts,
        "monitor_summary": {
            "report_count": len(monitors),
            "warnings": warnings,
            "red_flags": [],
            "minimum_system_available_bytes": min(
                item["minimum_system_available_bytes"] for item in monitors
            ),
            "maximum_process_tree_rss_bytes": max(
                item["maximum_process_tree_rss_bytes"] for item in monitors
            ),
            "maximum_gpu_used_mib": max(
                item.get("maximum_gpu_used_mib") or 0 for item in monitors
            ),
        },
        "budgets": {
            "gate3b_output_bytes_before_final_report": output_bytes,
            "output_warning_bytes": 2 * GIB,
            "output_red_bytes": 4 * GIB,
            "training_warning_minutes_each": 30,
            "training_hard_limit_minutes_each": 180,
        },
        "isolation": {
            "formal_data_dir": str(formal_data_dir.resolve()),
            "formal_data_dir_exists": formal_data_dir.exists(),
            "formal_output_dir": str(formal_output_dir.resolve()),
            "formal_output_dir_exists": formal_output_dir.exists(),
            "residual_experiment_processes": residual_processes,
        },
        "evaluation_mode": "oracle-K_ground_truth_source_count",
        "performance_interpretation_allowed": False,
        "paper_metric_reproduced": False,
        "next_gate": "第三章小规模收敛与第四章 predicted-K/论文结果追溯需另行审批",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate 3B 第四章训练与评估审计工具")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest-data")
    manifest.add_argument("--data_dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    audit = sub.add_parser("audit-training")
    audit.add_argument("--run_dir", type=Path, required=True)
    audit.add_argument("--monitor", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare-training")
    compare.add_argument("--left_dir", type=Path, required=True)
    compare.add_argument("--right_dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    freeze = sub.add_parser("freeze-checkpoint")
    freeze.add_argument("--run_dir", type=Path, required=True)
    freeze.add_argument("--comparison", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    eval_compare = sub.add_parser("compare-evaluations")
    eval_compare.add_argument("--left", type=Path, required=True)
    eval_compare.add_argument("--right", type=Path, required=True)
    eval_compare.add_argument("--training_final_validation", type=Path)
    eval_compare.add_argument("--output", type=Path, required=True)

    final = sub.add_parser("finalize-gate3b")
    final.add_argument("--run_root", type=Path, required=True)
    final.add_argument("--manifest_before", type=Path, required=True)
    final.add_argument("--manifest_after", type=Path, required=True)
    final.add_argument("--audit_a", type=Path, required=True)
    final.add_argument("--audit_b", type=Path, required=True)
    final.add_argument("--training_comparison", type=Path, required=True)
    final.add_argument("--frozen", type=Path, required=True)
    final.add_argument("--val_comparison", type=Path, required=True)
    final.add_argument("--test_comparison", type=Path, required=True)
    final.add_argument("--formal_data_dir", type=Path, required=True)
    final.add_argument("--formal_output_dir", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    try:
        if args.command == "manifest-data":
            result = build_data_manifest(args.data_dir.resolve())
        elif args.command == "audit-training":
            result = audit_training(args.run_dir.resolve(), args.monitor.resolve())
        elif args.command == "compare-training":
            result = compare_training(args.left_dir.resolve(), args.right_dir.resolve())
        elif args.command == "freeze-checkpoint":
            result = freeze_checkpoint(args.run_dir.resolve(), args.comparison.resolve())
        elif args.command == "compare-evaluations":
            result = compare_evaluations(
                args.left.resolve(), args.right.resolve(),
                args.training_final_validation.resolve()
                if args.training_final_validation else None,
            )
        else:
            result = finalize_gate3b(
                args.run_root.resolve(), args.manifest_before.resolve(),
                args.manifest_after.resolve(), args.audit_a.resolve(),
                args.audit_b.resolve(), args.training_comparison.resolve(),
                args.frozen.resolve(), args.val_comparison.resolve(),
                args.test_comparison.resolve(), args.formal_data_dir.resolve(),
                args.formal_output_dir.resolve(),
            )
        write_json(output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=json_default))
        return 0 if result.get("status") in {
            "PASS", "PASS_WITH_MORE_EPOCHS_RECOMMENDED",
        } else 2
    except Exception as exc:  # noqa: BLE001 - 门禁必须持久化失败证据
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
