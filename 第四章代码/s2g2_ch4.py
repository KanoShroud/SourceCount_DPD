"""S2-G2：第四章缩减数据训练充分性、长尾误差和接口 checkpoint 审计。"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch

from gate3b_ch4 import (
    build_data_manifest,
    compare_nested,
    directory_size,
    load_json,
    require,
    require_finite_nested,
    sha256_file,
    single_checkpoint,
    write_json,
)


GIB = 1024**3
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
EXPECTED_DATA_BYTES = 3_951_967_293
EXPECTED_GATE3B_SHA256 = "8cf7f9541b3269539b7b16580d7a13dd2b9a9770afa1b0032b3ec491844f0153"
EXPECTED_GATE3B_RMSE = 150.7919210688925


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
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        return float(result.stdout.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def residual_processes() -> list[dict[str, Any]]:
    markers = (
        "train_yolo.py",
        "eval_ch4_checkpoint.py",
        "s2g2_ch4.py",
        "gate3_stage_runner.py",
    )
    residual = []
    current_pid = os.getpid()
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if process.info["pid"] == current_pid:
                continue
            name = str(process.info.get("name") or "").lower()
            if "python" not in name:
                continue
            command = " ".join(process.info.get("cmdline") or [])
            if any(marker in command for marker in markers):
                residual.append({
                    "pid": process.info["pid"],
                    "name": process.info.get("name"),
                    "command": command,
                })
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
    return residual


def code_fingerprint() -> dict[str, dict[str, Any]]:
    names = (
        "train_yolo.py",
        "eval_ch4_checkpoint.py",
        "gate3_stage_runner.py",
        "gate3b_ch4.py",
        "s2g2_ch4.py",
    )
    return {
        name: {
            "path": str((SCRIPT_DIR / name).resolve()),
            "size_bytes": (SCRIPT_DIR / name).stat().st_size,
            "sha256": sha256_file(SCRIPT_DIR / name),
        }
        for name in names
    }


def run_preflight(duration_seconds: float, interval_seconds: float) -> dict[str, Any]:
    require(duration_seconds >= 10, "preflight采样时间至少10秒")
    require(interval_seconds >= 1, "preflight采样间隔至少1秒")
    samples = []
    started = time.perf_counter()
    while True:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(PROJECT_ROOT.anchor)
        samples.append({
            "elapsed_seconds": time.perf_counter() - started,
            "system_available_bytes": int(memory.available),
            "system_used_bytes": int(memory.used),
            "gpu_used_mib": gpu_used_mib(),
            "disk_free_bytes": int(disk.free),
        })
        if time.perf_counter() - started >= duration_seconds:
            break
        time.sleep(interval_seconds)

    available = [sample["system_available_bytes"] for sample in samples]
    gpu_values = [
        sample["gpu_used_mib"] for sample in samples
        if sample["gpu_used_mib"] is not None
    ]
    disk_values = [sample["disk_free_bytes"] for sample in samples]
    median_available = statistics.median(available)
    maximum_gpu = max(gpu_values) if gpu_values else None
    minimum_disk = min(disk_values)
    residual = residual_processes()
    require(torch.cuda.is_available(), "PyTorch无法访问CUDA")
    require(torch.cuda.device_count() >= 1, "未发现cuda:0")
    require(median_available >= 14 * GIB, "可用RAM中位数低于14 GiB")
    require(maximum_gpu is not None, "无法读取cuda:0显存")
    require(maximum_gpu < 12 * 1024, "启动前GPU总占用达到12 GiB预警线")
    require(minimum_disk >= 120 * GIB, "磁盘剩余低于120 GiB")
    require(not residual, f"发现遗留实验进程: {residual}")

    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "status", "--short"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8", errors="replace",
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
        "code_fingerprint": code_fingerprint(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_0": torch.cuda.get_device_name(0),
        },
        "samples": samples,
    }


def manifest_data(data_dir: Path) -> dict[str, Any]:
    manifest = build_data_manifest(data_dir)
    require(manifest["total_bytes"] == EXPECTED_DATA_BYTES, "Gate 3A数据总字节数变化")
    manifest["data_role"] = "Gate 3A缩减定位数据，由S2-G2只读复用"
    return manifest


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
        "epochs": 200,
        "patience": 200,
        "seed": 42,
        "device": "cuda:0",
        "peak_size": 9,
        "box_size": 9,
        "deterministic": True,
        "fail_on_nonfinite": True,
        "save_last_every_epoch": True,
        "require_empty_output": True,
        "gate3_d8": False,
        "gate3b_d8": False,
        "s2g2_d8": True,
    }


def audit_training(run_dir: Path, monitor_path: Path) -> dict[str, Any]:
    config = load_json(run_dir / "run_config.json")
    summary = load_json(run_dir / "training_summary.json")
    history = load_json(run_dir / "epoch_history.json")
    initial = load_json(run_dir / "initial_validation.json")
    final = load_json(run_dir / "final_validation.json")
    monitor = load_json(monitor_path)

    require(config.get("status") == "CONFIGURED", "run_config状态不符")
    require(config.get("model_label") == "D8", "模型标签不是D8")
    saved_args = config.get("args")
    require(isinstance(saved_args, dict), "run_config缺少args")
    mismatches = [
        f"{key}={saved_args.get(key)!r}, expected={value!r}"
        for key, value in expected_training_args().items()
        if saved_args.get(key) != value
    ]
    require(not mismatches, "S2-G2冻结配置不一致: " + "; ".join(mismatches))
    require(summary.get("status") == "PASS", "training_summary非PASS")
    require(summary.get("epochs_completed") == 200, "未完整执行200 epoch")
    require(summary.get("stopped_early") is False, "训练发生early stopping")
    require(len(history) == 200, f"epoch_history长度={len(history)}")
    require([item.get("epoch") for item in history] == list(range(1, 201)), "epoch序列不连续")
    require(all(item.get("train_batches") == 128 for item in history), "每epoch训练batch数不是128")
    require(math.isclose(float(history[-1]["learning_rate"]), 1e-6, rel_tol=0.0, abs_tol=1e-12),
            f"最终学习率={history[-1]['learning_rate']}")
    require(math.isclose(float(summary["final_learning_rate"]), 1e-6,
                         rel_tol=0.0, abs_tol=1e-12), "summary最终学习率不符")
    require_finite_nested(history, "history")
    require_finite_nested(initial, "initial_validation")
    require_finite_nested(final, "final_validation")

    best_path = single_checkpoint(run_dir, "best")
    last_path = single_checkpoint(run_dir, "last")
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    require_finite_nested(best["model"], "best.model")
    require_finite_nested(last["model"], "last.model")
    eligible = [record for record in history if int(record["epoch"]) >= 6]
    recomputed_best = min(eligible, key=lambda record: float(record["rmse"]))
    best_epoch = int(recomputed_best["epoch"])
    best_rmse = float(recomputed_best["rmse"])
    require(summary.get("best_epoch") == best_epoch, "summary best_epoch与历史复算不一致")
    require(float(summary["best_rmse"]) == best_rmse, "summary best_rmse与历史复算不一致")
    require(best.get("epoch") == best_epoch, "best checkpoint epoch不一致")
    require(float(best["best_rmse"]) == best_rmse, "best checkpoint RMSE不一致")
    require(last.get("epoch") == 200, "last checkpoint不是epoch 200")

    initial_metrics = initial.get("metrics")
    final_metrics = final.get("metrics")
    require(isinstance(initial_metrics, dict) and isinstance(final_metrics, dict), "验证指标缺失")
    require(float(final_metrics["rmse"]) == best_rmse, "最终validation未对应best checkpoint")
    rmse_improved = float(final_metrics["rmse"]) < float(initial_metrics["rmse"])
    loss_improved = float(final_metrics["val_loss"]) < float(initial_metrics["val_loss"])
    require(rmse_improved and loss_improved, "训练未同时改善validation RMSE和loss")
    require(initial.get("rng_state_restored") is True, "初始validation后未声明恢复RNG")

    pre_180 = min(float(record["rmse"]) for record in eligible if int(record["epoch"]) <= 180)
    gain_last_20 = (pre_180 - best_rmse) / pre_180
    trend = (
        "TREND_CLOSED"
        if best_epoch <= 180 or gain_last_20 < 0.01
        else "TREND_OPEN_AT_200"
    )
    best_record = next(record for record in history if int(record["epoch"]) == best_epoch)
    last_record = history[-1]
    possible_overfit = bool(
        float(last_record["train_loss"]) < float(best_record["train_loss"])
        and float(last_record["val_loss"]) > float(best_record["val_loss"])
        and float(last_record["rmse"]) > best_rmse
    )

    require(monitor.get("status") == "PASS", f"训练监控状态={monitor.get('status')}")
    warnings = monitor.get("warnings", [])
    red_flags = monitor.get("red_flags", [])
    require(not red_flags, f"训练存在red flag: {red_flags}")
    return {
        "status": "REVIEW_REQUIRED" if warnings else "PASS",
        "run_dir": str(run_dir.resolve()),
        "epochs_completed": 200,
        "train_batches_per_epoch": 128,
        "best_epoch": best_epoch,
        "best_rmse": best_rmse,
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path.resolve()),
        "last_checkpoint_sha256": sha256_file(last_path),
        "initial_validation": initial_metrics,
        "final_validation": final_metrics,
        "learning": {
            "status": "LEARNING_CONFIRMED",
            "rmse_improved": rmse_improved,
            "val_loss_improved": loss_improved,
        },
        "training_trend": {
            "status": trend,
            "best_epoch": best_epoch,
            "best_rmse_m": best_rmse,
            "best_rmse_through_epoch_180_m": pre_180,
            "last_20_relative_gain": gain_last_20,
        },
        "overfit_diagnostic": {
            "possible_overfit_pattern": possible_overfit,
            "best_epoch_train_loss": float(best_record["train_loss"]),
            "best_epoch_val_loss": float(best_record["val_loss"]),
            "final_train_loss": float(last_record["train_loss"]),
            "final_val_loss": float(last_record["val_loss"]),
            "final_rmse_m": float(last_record["rmse"]),
            "interpretation": "descriptive_only_not_a_checkpoint_selection_rule",
        },
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


def evaluation_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    details = report.get("error_details")
    require(isinstance(details, dict), "评估报告缺少error_details")
    records = details.get("per_source_errors")
    require(isinstance(records, list) and len(records) == 640,
            f"逐源误差数量应为640，实际={len(records) if isinstance(records, list) else None}")
    require_finite_nested(details, "error_details")
    return records


def compare_checkpoints(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = load_json(baseline_path)
    candidate = load_json(candidate_path)
    for name, report in (("baseline", baseline), ("candidate", candidate)):
        require(report.get("status") == "PASS", f"{name}评估非PASS")
        require(report.get("split") == "val", f"{name}不是validation评估")
        require(report.get("task_count") == 256, f"{name}任务数不是256")
        require(report.get("evaluation_mode") == "oracle-K_ground_truth_source_count",
                f"{name}评估模式不符")
        require(report.get("source_count_distribution") == {"2": 128, "3": 128},
                f"{name}源数分布不符")
    require(baseline.get("checkpoint_sha256") == EXPECTED_GATE3B_SHA256,
            "Gate 3B checkpoint SHA256不符")
    baseline_rmse = float(baseline["metrics"]["rmse"])
    require(math.isclose(baseline_rmse, EXPECTED_GATE3B_RMSE, rel_tol=0.0, abs_tol=1e-12),
            f"Gate 3B RMSE未被当前评估器精确复现: {baseline_rmse}")

    baseline_records = evaluation_records(baseline)
    candidate_records = evaluation_records(candidate)
    baseline_map = {
        (record["sample_index"], record["true_source_index"]): record
        for record in baseline_records
    }
    candidate_map = {
        (record["sample_index"], record["true_source_index"]): record
        for record in candidate_records
    }
    require(set(baseline_map) == set(candidate_map), "两次评估逐源键不一致")
    paired = []
    for key in sorted(baseline_map):
        left = baseline_map[key]
        right = candidate_map[key]
        require(left["source_count"] == right["source_count"], f"{key}源数不一致")
        require(left["true_x_m"] == right["true_x_m"] and left["true_y_m"] == right["true_y_m"],
                f"{key}真实坐标不一致")
        delta = float(left["error_m"]) - float(right["error_m"])
        paired.append({
            "sample_index": key[0],
            "true_source_index": key[1],
            "source_count": left["source_count"],
            "baseline_error_m": float(left["error_m"]),
            "candidate_error_m": float(right["error_m"]),
            "candidate_improvement_m": delta,
        })
    deltas = np.asarray([record["candidate_improvement_m"] for record in paired], dtype=np.float64)
    candidate_rmse = float(candidate["metrics"]["rmse"])
    selected = candidate if candidate_rmse < baseline_rmse else baseline
    checkpoint_decision = (
        "NEW_200_SELECTED" if selected is candidate else "GATE3B_60_RETAINED"
    )
    return {
        "status": "PASS",
        "selection_split": "validation",
        "selection_metric": "oracle-K RMSE; lower wins; exact tie retains Gate 3B",
        "test_used_for_selection": False,
        "baseline": {
            "evaluation": str(baseline_path.resolve()),
            "checkpoint": baseline["checkpoint"],
            "checkpoint_sha256": baseline["checkpoint_sha256"],
            "checkpoint_epoch": baseline["checkpoint_epoch"],
            "metrics": baseline["metrics"],
            "error_statistics": baseline["error_details"]["statistics"],
        },
        "candidate": {
            "evaluation": str(candidate_path.resolve()),
            "checkpoint": candidate["checkpoint"],
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "checkpoint_epoch": candidate["checkpoint_epoch"],
            "metrics": candidate["metrics"],
            "error_statistics": candidate["error_details"]["statistics"],
        },
        "checkpoint_decision": checkpoint_decision,
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_checkpoint_epoch": selected["checkpoint_epoch"],
        "paired_error_analysis": {
            "source_count": len(paired),
            "candidate_improved_count": int(np.sum(deltas > 0)),
            "candidate_worsened_count": int(np.sum(deltas < 0)),
            "exactly_equal_count": int(np.sum(deltas == 0)),
            "mean_candidate_improvement_m": float(np.mean(deltas)),
            "median_candidate_improvement_m": float(np.median(deltas)),
            "largest_improvements": sorted(
                paired, key=lambda record: record["candidate_improvement_m"], reverse=True
            )[:10],
            "largest_deteriorations": sorted(
                paired, key=lambda record: record["candidate_improvement_m"]
            )[:10],
        },
        "paper_metric_reproduced": False,
        "performance_interpretation_allowed": False,
    }


def freeze_checkpoint(comparison_path: Path) -> dict[str, Any]:
    comparison = load_json(comparison_path)
    require(comparison.get("status") == "PASS", "checkpoint比较报告非PASS")
    checkpoint_path = Path(comparison["selected_checkpoint"]).resolve()
    expected_sha256 = comparison["selected_checkpoint_sha256"]
    require(checkpoint_path.is_file(), f"所选checkpoint不存在: {checkpoint_path}")
    require(sha256_file(checkpoint_path) == expected_sha256, "所选checkpoint SHA256变化")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    require(checkpoint.get("method") == "dualhead", "所选checkpoint method不符")
    require(checkpoint.get("save_tag") == "dualhead_std", "所选checkpoint save_tag不符")
    require_finite_nested(checkpoint["model"], "selected.model")
    return {
        "status": "PASS",
        "checkpoint_decision": comparison["checkpoint_decision"],
        "selection_split": "validation",
        "selection_metric": comparison["selection_metric"],
        "test_used_for_selection": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": expected_sha256,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_rmse": checkpoint.get("best_rmse"),
        "model": "D8 dualhead_std",
        "amp": False,
        "evaluation_mode": "oracle-K_ground_truth_source_count",
        "training_reproducibility": "NOT_RETESTED_AT_200",
        "performance_interpretation_allowed": False,
    }


def compare_evaluations(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = load_json(left_path)
    right = load_json(right_path)
    for name, report in (("left", left), ("right", right)):
        require(report.get("status") == "PASS", f"{name}评估非PASS")
        require(report.get("split") == "val", f"{name}不是validation评估")
    require(left.get("checkpoint_sha256") == right.get("checkpoint_sha256"),
            "两次评估checkpoint SHA256不一致")
    mismatches: list[str] = []
    for key in (
        "checkpoint_epoch",
        "checkpoint_best_rmse",
        "model",
        "amp",
        "batch_size",
        "seed",
        "deterministic",
        "evaluation_mode",
        "task_count",
        "source_count_distribution",
        "metrics",
        "error_details",
    ):
        compare_nested(left.get(key), right.get(key), key, mismatches)
    require(not mismatches, "两次独立validation不一致: " + (mismatches[0] if mismatches else ""))
    return {
        "status": "PASS",
        "numeric_requirement": "exact match excluding elapsed time and environment",
        "checkpoint_sha256": left["checkpoint_sha256"],
        "metrics": left["metrics"],
        "error_statistics": left["error_details"]["statistics"],
        "per_source_error_count": len(left["error_details"]["per_source_errors"]),
        "checkpoint_reload": "EXACT",
        "mismatches": mismatches,
        "performance_interpretation_allowed": False,
    }


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    reports = {
        "preflight": load_json(args.preflight.resolve()),
        "manifest_before": load_json(args.manifest_before.resolve()),
        "training_audit": load_json(args.training_audit.resolve()),
        "checkpoint_comparison": load_json(args.checkpoint_comparison.resolve()),
        "frozen_checkpoint": load_json(args.frozen.resolve()),
        "validation_reload": load_json(args.validation_reload.resolve()),
        "manifest_after": load_json(args.manifest_after.resolve()),
    }
    for name, report in reports.items():
        require(report.get("status") == "PASS", f"{name} status={report.get('status')}")
    require(reports["manifest_before"]["files"] == reports["manifest_after"]["files"],
            "训练前后数据SHA256发生变化")
    require(reports["manifest_before"]["total_bytes"] == EXPECTED_DATA_BYTES,
            "数据总字节数变化")
    frozen_sha = reports["frozen_checkpoint"]["checkpoint_sha256"]
    require(reports["validation_reload"]["checkpoint_sha256"] == frozen_sha,
            "独立validation未使用冻结checkpoint")

    formal_data_dir = PROJECT_ROOT / "data" / "chapter4"
    formal_output_dir = PROJECT_ROOT / "outputs" / "formal" / "chapter4"
    require(not formal_data_dir.exists(), f"formal data目录不应存在: {formal_data_dir}")
    require(not formal_output_dir.exists(), f"formal output目录不应存在: {formal_output_dir}")
    residual = residual_processes()
    require(not residual, f"存在遗留S2-G2进程: {residual}")

    run_root = args.run_root.resolve()
    monitor_paths = sorted(run_root.rglob("stage_monitor_report.json"))
    monitor_entries = [
        {"path": str(path.resolve()), "report": load_json(path)}
        for path in monitor_paths
    ]
    passed_monitors = [
        item for item in monitor_entries if item["report"].get("status") == "PASS"
    ]
    failed_attempts = [
        {
            "path": item["path"],
            "stage": item["report"].get("stage"),
            "status": item["report"].get("status"),
            "exit_code": item["report"].get("exit_code"),
        }
        for item in monitor_entries
        if item["report"].get("status") != "PASS"
    ]
    require(len(passed_monitors) >= 4,
            f"PASS监控报告数量不足: {len(passed_monitors)}")
    monitors = [item["report"] for item in passed_monitors]
    for monitor in monitors:
        require(not monitor.get("red_flags"), f"监控存在red flag: {monitor.get('red_flags')}")
    warnings = sorted({warning for monitor in monitors for warning in monitor.get("warnings", [])})
    output_bytes = directory_size(run_root)
    require(output_bytes <= 4 * GIB, f"S2-G2输出超过4 GiB: {output_bytes}")
    if output_bytes > 2 * GIB:
        warnings.append("s2g2_output_above_2_gib")
    require(not warnings, f"S2-G2存在待审资源warning: {warnings}")

    trend = reports["training_audit"]["training_trend"]["status"]
    status = "PASS" if trend == "TREND_CLOSED" else "PASS_WITH_OPEN_TREND"
    return {
        "status": status,
        "gate": "S2-G2",
        "scope": "第四章缩减数据训练充分性诊断与S2-G3接口checkpoint建立",
        "run_root": str(run_root),
        "data_integrity": {
            "file_count": reports["manifest_before"]["file_count"],
            "total_bytes": reports["manifest_before"]["total_bytes"],
            "before_after_sha256": "exact_match",
        },
        "training": {
            "model": "D8 dualhead_std",
            "epochs": 200,
            "patience": 200,
            "amp": False,
            "batch_size": 8,
            "seed": 42,
            "best_epoch": reports["training_audit"]["best_epoch"],
            "best_rmse_m": reports["training_audit"]["best_rmse"],
            "learning": reports["training_audit"]["learning"],
            "training_trend": reports["training_audit"]["training_trend"],
            "overfit_diagnostic": reports["training_audit"]["overfit_diagnostic"],
            "training_reproducibility": "NOT_RETESTED_AT_200",
        },
        "checkpoint_decision": reports["checkpoint_comparison"]["checkpoint_decision"],
        "checkpoint_comparison": reports["checkpoint_comparison"],
        "frozen_checkpoint": reports["frozen_checkpoint"],
        "validation_reload": reports["validation_reload"],
        "monitor_summary": {
            "pass_report_count": len(monitors),
            "preserved_failed_attempts": failed_attempts,
            "warnings": warnings,
            "red_flags": [],
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
        "output_bytes_before_final_report": output_bytes,
        "code_fingerprint": code_fingerprint(),
        "isolation": {
            "formal_data_dir_exists": formal_data_dir.exists(),
            "formal_output_dir_exists": formal_output_dir.exists(),
            "residual_processes": residual,
        },
        "evaluation_mode": "oracle-K_ground_truth_source_count",
        "test_run_in_s2g2": False,
        "paper_metric_reproduced": False,
        "performance_interpretation_allowed": False,
        "next_gate": "S2-G3 oracle/predicted band/K接口闭环，需另行审批",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2-G2第四章训练与checkpoint审计工具")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--duration_seconds", type=float, default=60.0)
    preflight.add_argument("--interval_seconds", type=float, default=2.0)
    preflight.add_argument("--output", type=Path, required=True)

    manifest = sub.add_parser("manifest-data")
    manifest.add_argument("--data_dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    audit = sub.add_parser("audit-training")
    audit.add_argument("--run_dir", type=Path, required=True)
    audit.add_argument("--monitor", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    compare = sub.add_parser("compare-checkpoints")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    freeze = sub.add_parser("freeze-checkpoint")
    freeze.add_argument("--comparison", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    eval_compare = sub.add_parser("compare-evaluations")
    eval_compare.add_argument("--left", type=Path, required=True)
    eval_compare.add_argument("--right", type=Path, required=True)
    eval_compare.add_argument("--output", type=Path, required=True)

    final = sub.add_parser("finalize")
    final.add_argument("--run_root", type=Path, required=True)
    final.add_argument("--preflight", type=Path, required=True)
    final.add_argument("--manifest_before", type=Path, required=True)
    final.add_argument("--training_audit", type=Path, required=True)
    final.add_argument("--checkpoint_comparison", type=Path, required=True)
    final.add_argument("--frozen", type=Path, required=True)
    final.add_argument("--validation_reload", type=Path, required=True)
    final.add_argument("--manifest_after", type=Path, required=True)
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
            result = manifest_data(args.data_dir.resolve())
        elif args.command == "audit-training":
            result = audit_training(args.run_dir.resolve(), args.monitor.resolve())
        elif args.command == "compare-checkpoints":
            result = compare_checkpoints(args.baseline.resolve(), args.candidate.resolve())
        elif args.command == "freeze-checkpoint":
            result = freeze_checkpoint(args.comparison.resolve())
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
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
