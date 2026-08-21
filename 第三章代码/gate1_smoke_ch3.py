"""Gate 1：第三章极小数据审计与一次训练闭环。

该入口只读取本次 smoke 目录，执行数据/标签审计以及一个 batch 的
forward、loss、backward、optimizer step、validation、save/reload。
它不会调用 ``train_v26.train``，也不会执行正式训练。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_root in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from train_v26 import (  # noqa: E402
    SourceDetectionDataset,
    SourceDetectionNet,
    compute_loss,
)


EXPECTED_SAMPLES = {"train": 8, "val": 4, "test": 4}
EXPECTED_CLASS_COUNTS = {
    "train": [2, 2, 2, 2],
    "val": [1, 1, 1, 1],
    "test": [1, 1, 1, 1],
}
EXPECTED_N_SUB = 19
EXPECTED_GRID = 81
EXPECTED_MAX_SRC = 3
MODEL_MAX_SRC = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第三章 Gate 1：MAT 审计与单 batch 训练闭环"
    )
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def scalar(handle: h5py.File, name: str) -> float:
    require(name in handle, f"缺少元数据 {name}")
    return float(np.asarray(handle[name]).reshape(-1)[0])


def logical_arrays(handle: h5py.File) -> dict[str, np.ndarray]:
    required = {
        "mtr_sub_all",
        "src_count_all",
        "band_mask_all",
        "ignore_mask_all",
        "fc_offset_all",
        "sub_energy_all",
        "cov_mat_real_all",
        "cov_mat_imag_all",
    }
    missing = sorted(required.difference(handle.keys()))
    require(not missing, f"MAT 文件缺少数据集: {missing}")

    return {
        "spectra": np.asarray(handle["mtr_sub_all"]).transpose(3, 2, 1, 0),
        "src_count": np.asarray(handle["src_count_all"]).reshape(-1),
        "band_mask": np.asarray(handle["band_mask_all"]).transpose(2, 1, 0),
        "ignore_mask": np.asarray(handle["ignore_mask_all"]).transpose(2, 1, 0),
        "fc_offset": np.asarray(handle["fc_offset_all"]).T,
        "sub_energy": np.asarray(handle["sub_energy_all"]).T,
        "cov_real": np.asarray(handle["cov_mat_real_all"]).transpose(4, 3, 2, 1, 0)
        if handle["cov_mat_real_all"].ndim == 5
        else np.asarray(handle["cov_mat_real_all"]).transpose(3, 2, 1, 0),
        "cov_imag": np.asarray(handle["cov_mat_imag_all"]).transpose(4, 3, 2, 1, 0)
        if handle["cov_mat_imag_all"].ndim == 5
        else np.asarray(handle["cov_mat_imag_all"]).transpose(3, 2, 1, 0),
    }


def recompute_labels(
    src_count: np.ndarray,
    fc_offset: np.ndarray,
    sub_f_lo: np.ndarray,
    sub_f_hi: np.ndarray,
    symbol_rate: float,
    actual_bandwidth: float,
    band_window: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = len(src_count)
    expected_band = np.zeros(
        (n_samples, EXPECTED_MAX_SRC, EXPECTED_N_SUB), dtype=np.float32
    )
    expected_ignore = np.zeros_like(expected_band)

    for sample_idx, count_value in enumerate(src_count.astype(int)):
        for source_idx in range(count_value):
            center = float(fc_offset[sample_idx, source_idx])
            main_lo = center - symbol_rate / 2.0
            main_hi = center + symbol_rate / 2.0
            rolloff_lo = center - actual_bandwidth / 2.0
            rolloff_hi = center + actual_bandwidth / 2.0
            for band_idx in range(EXPECTED_N_SUB):
                overlap_main = max(
                    0.0,
                    min(main_hi, float(sub_f_hi[band_idx]))
                    - max(main_lo, float(sub_f_lo[band_idx])),
                )
                overlap_all = max(
                    0.0,
                    min(rolloff_hi, float(sub_f_hi[band_idx]))
                    - max(rolloff_lo, float(sub_f_lo[band_idx])),
                )
                overlap_rolloff = overlap_all - overlap_main
                coverage = overlap_main / band_window
                if coverage >= threshold:
                    expected_band[sample_idx, source_idx, band_idx] = 1.0
                elif coverage > 0.0 or overlap_rolloff > 0.0:
                    expected_ignore[sample_idx, source_idx, band_idx] = 1.0

    return expected_band, expected_ignore


def audit_mat_file(path: Path, split: str) -> tuple[dict[str, Any], dict[str, float]]:
    require(path.is_file(), f"缺少 {split} 数据文件: {path}")
    with h5py.File(path, "r") as handle:
        raw_shapes = {name: list(handle[name].shape) for name in handle.keys()}
        raw_dtypes = {name: str(handle[name].dtype) for name in handle.keys()}
        arrays = logical_arrays(handle)
        metadata = {
            "n_sub": scalar(handle, "N_sub_val"),
            "max_src": scalar(handle, "max_src_val"),
            "num_grid": scalar(handle, "num_grid"),
            "band_window": scalar(handle, "B_win_val"),
            "band_step": scalar(handle, "B_step_val"),
            "sample_rate": scalar(handle, "fs_val"),
            "symbol_rate": scalar(handle, "symbolRate_val"),
            "actual_bandwidth": scalar(handle, "BW_actual_val"),
            "rolloff": scalar(handle, "arfa_val"),
            "threshold": scalar(handle, "thresh_val"),
            "smoke_seed": scalar(handle, "smoke_seed_val"),
        }
        sub_f_lo = np.asarray(handle["sub_f_lo_val"]).reshape(-1).astype(np.float64)
        sub_f_hi = np.asarray(handle["sub_f_hi_val"]).reshape(-1).astype(np.float64)

    expected_n = EXPECTED_SAMPLES[split]
    spectra = arrays["spectra"]
    src_count = arrays["src_count"].astype(np.int64)
    band_mask = arrays["band_mask"]
    ignore_mask = arrays["ignore_mask"]
    fc_offset = arrays["fc_offset"]

    require(
        spectra.shape == (expected_n, EXPECTED_N_SUB, EXPECTED_GRID, EXPECTED_GRID),
        f"{split} spectra 逻辑 shape 错误: {spectra.shape}",
    )
    require(spectra.dtype == np.float32, f"{split} spectra dtype={spectra.dtype}")
    require(np.isfinite(spectra).all(), f"{split} spectra 包含 NaN/Inf")
    require((spectra >= 0).all(), f"{split} spectra 包含负值")
    require(src_count.shape == (expected_n,), f"{split} src_count shape={src_count.shape}")
    require(
        band_mask.shape == (expected_n, EXPECTED_MAX_SRC, EXPECTED_N_SUB),
        f"{split} band_mask shape={band_mask.shape}",
    )
    require(ignore_mask.shape == band_mask.shape, f"{split} ignore_mask shape错误")
    require(fc_offset.shape == (expected_n, EXPECTED_MAX_SRC), f"{split} fc_offset shape错误")

    for name, array in arrays.items():
        require(np.isfinite(array).all(), f"{split} {name} 包含 NaN/Inf")
    require((arrays["sub_energy"] >= 0).all(), f"{split} sub_energy 包含负值")
    require(np.isin(band_mask, [0.0, 1.0]).all(), f"{split} band_mask 非二值")
    require(np.isin(ignore_mask, [0.0, 1.0]).all(), f"{split} ignore_mask 非二值")
    require(not np.logical_and(band_mask == 1, ignore_mask == 1).any(), f"{split} 标签互斥失败")
    require(np.isin(src_count, [0, 1, 2, 3]).all(), f"{split} 信源数越界")

    class_counts = np.bincount(src_count, minlength=4).tolist()
    require(class_counts == EXPECTED_CLASS_COUNTS[split], f"{split} 类别分布={class_counts}")

    for sample_idx, count_value in enumerate(src_count):
        count = int(count_value)
        require(
            not band_mask[sample_idx, count:, :].any(),
            f"{split}[{sample_idx}] 空槽位 band_mask 非零",
        )
        require(
            not ignore_mask[sample_idx, count:, :].any(),
            f"{split}[{sample_idx}] 空槽位 ignore_mask 非零",
        )
        active_fc = fc_offset[sample_idx, :count]
        require(
            count < 2 or np.all(np.diff(active_fc) >= 0),
            f"{split}[{sample_idx}] 活跃源频率未排序",
        )
        require(
            np.all(fc_offset[sample_idx, count:] == 0),
            f"{split}[{sample_idx}] 空槽位频偏非零",
        )

    require(int(metadata["n_sub"]) == EXPECTED_N_SUB, f"{split} N_sub 元数据错误")
    require(int(metadata["max_src"]) == EXPECTED_MAX_SRC, f"{split} max_src 元数据错误")
    require(int(metadata["num_grid"]) == EXPECTED_GRID, f"{split} num_grid 元数据错误")
    require(int(metadata["smoke_seed"]) == 20260820, f"{split} smoke seed 元数据错误")
    require(sub_f_lo.shape == (EXPECTED_N_SUB,), f"{split} sub_f_lo shape错误")
    require(sub_f_hi.shape == (EXPECTED_N_SUB,), f"{split} sub_f_hi shape错误")
    require(np.all(np.diff(sub_f_lo) > 0), f"{split} 子带下边界未递增")
    require(np.all(sub_f_hi > sub_f_lo), f"{split} 子带上下边界错误")

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
    require(band_mismatches == 0, f"{split} band 标签复算不一致 {band_mismatches} 项")
    require(ignore_mismatches == 0, f"{split} ignore 标签复算不一致 {ignore_mismatches} 项")

    report = {
        "path": str(path.resolve()),
        "file_size_bytes": path.stat().st_size,
        "raw_shapes": raw_shapes,
        "raw_dtypes": raw_dtypes,
        "logical_shapes": {name: list(array.shape) for name, array in arrays.items()},
        "class_counts": dict(sorted(Counter(src_count.tolist()).items())),
        "band_positive_count": int(np.count_nonzero(band_mask)),
        "ignore_count": int(np.count_nonzero(ignore_mask)),
        "label_recompute_band_mismatches": band_mismatches,
        "label_recompute_ignore_mismatches": ignore_mismatches,
        "spectra_min": float(spectra.min()),
        "spectra_max": float(spectra.max()),
        "metadata": metadata,
        "status": "PASS",
    }
    return report, metadata


def audit_all(data_dir: Path) -> dict[str, Any]:
    split_reports: dict[str, Any] = {}
    metadata_by_split: dict[str, dict[str, float]] = {}
    for split in ("train", "val", "test"):
        report, metadata = audit_mat_file(data_dir / f"{split}_data.mat", split)
        split_reports[split] = report
        metadata_by_split[split] = metadata

    reference = metadata_by_split["train"]
    for split in ("val", "test"):
        for key, value in reference.items():
            require(
                np.isclose(value, metadata_by_split[split][key], rtol=0.0, atol=0.0),
                f"{split} 元数据 {key} 与 train 不一致",
            )
    return {"status": "PASS", "splits": split_reports}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def process_rss_bytes() -> int:
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return int(rss)


def run_model_smoke(args: argparse.Namespace) -> dict[str, Any]:
    require(args.batch_size == 2, "Gate 1 固定 batch_size=2")
    require(args.device == "cuda:0", "Gate 1 固定使用 cuda:0")
    require(torch.cuda.is_available(), "PyTorch 未检测到 CUDA")
    require(torch.cuda.device_count() == 1, f"预期 1 张 GPU，实际 {torch.cuda.device_count()}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    rss_samples = [process_rss_bytes()]
    train_set = SourceDetectionDataset(
        args.data_dir / "train_data.mat",
        augment=True,
        normalize="sample_zscore",
        max_src_override=MODEL_MAX_SRC,
    )
    val_set = SourceDetectionDataset(
        args.data_dir / "val_data.mat",
        augment=False,
        normalize="sample_zscore",
        max_src_override=MODEL_MAX_SRC,
    )
    rss_samples.append(process_rss_bytes())

    ignore_candidates = np.flatnonzero(train_set.ignore_mask.reshape(len(train_set), -1).any(axis=1))
    require(len(ignore_candidates) > 0, "train 中没有 ignore 元素，无法验证 ignore-loss 不变性")
    first_idx = int(ignore_candidates[0])
    second_idx = 0 if first_idx != 0 else 1
    selected_indices = [first_idx, second_idx]
    train_loader = DataLoader(
        Subset(train_set, selected_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    x, src_count, band_mask, ignore_mask = next(iter(train_loader))
    require(tuple(x.shape) == (2, EXPECTED_N_SUB, EXPECTED_GRID, EXPECTED_GRID), f"train x shape={tuple(x.shape)}")
    require(tuple(band_mask.shape) == (2, MODEL_MAX_SRC, EXPECTED_N_SUB), f"train label shape={tuple(band_mask.shape)}")
    require(bool(ignore_mask.any()), "选定 batch 不含 ignore 元素")

    x = x.to(device, non_blocking=True)
    band_mask = band_mask.to(device, non_blocking=True)
    ignore_mask = ignore_mask.to(device, non_blocking=True)
    model = SourceDetectionNet(
        n_sub=EXPECTED_N_SUB,
        max_src=MODEL_MAX_SRC,
        mode="transformer",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    rss_samples.append(process_rss_bytes())

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    require(tuple(logits.shape) == (2, MODEL_MAX_SRC, EXPECTED_N_SUB), f"logits shape={tuple(logits.shape)}")
    require(bool(torch.isfinite(logits).all()), "forward 输出包含 NaN/Inf")
    loss = compute_loss(logits, band_mask, ignore_mask, gamma=2.0)
    require(bool(torch.isfinite(loss)), "train loss 非有限")

    altered_logits = logits.detach().clone()
    altered_logits[ignore_mask.bool()] += 37.0
    altered_loss = compute_loss(altered_logits, band_mask, ignore_mask, gamma=2.0)
    ignore_loss_diff = float(torch.abs(loss.detach() - altered_loss).item())
    require(ignore_loss_diff <= 1e-6, f"ignore-loss 差值={ignore_loss_diff}")

    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    require(gradients, "没有生成任何梯度")
    require(all(bool(torch.isfinite(gradient).all()) for gradient in gradients), "梯度包含 NaN/Inf")
    gradient_norm = float(
        torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)).item()
    )
    require(gradient_norm > 0.0, "梯度范数为零")

    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer.step()
    parameter_max_change = max(
        float(torch.max(torch.abs(parameter.detach() - parameters_before[name])).item())
        for name, parameter in model.named_parameters()
    )
    require(np.isfinite(parameter_max_change) and parameter_max_change > 0.0, "optimizer step 未更新参数")
    require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()), "更新后参数包含 NaN/Inf")
    rss_samples.append(process_rss_bytes())

    val_x, _, val_band, val_ignore = next(iter(val_loader))
    val_x = val_x.to(device, non_blocking=True)
    val_band = val_band.to(device, non_blocking=True)
    val_ignore = val_ignore.to(device, non_blocking=True)
    model.eval()
    with torch.no_grad():
        val_logits = model(val_x)
        val_loss = compute_loss(val_logits, val_band, val_ignore, gamma=2.0)
    require(tuple(val_logits.shape) == (2, MODEL_MAX_SRC, EXPECTED_N_SUB), f"val logits shape={tuple(val_logits.shape)}")
    require(bool(torch.isfinite(val_logits).all()) and bool(torch.isfinite(val_loss)), "验证输出或 loss 非有限")

    checkpoint_path = args.output_dir / "gate1_smoke_checkpoint.pth"
    checkpoint = {
        "model_state": model.state_dict(),
        "n_sub": EXPECTED_N_SUB,
        "max_src": MODEL_MAX_SRC,
        "mode": "transformer",
        "seed": args.seed,
    }
    torch.save(checkpoint, checkpoint_path)
    require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0, "checkpoint 未保存")

    reloaded = SourceDetectionNet(
        n_sub=EXPECTED_N_SUB,
        max_src=MODEL_MAX_SRC,
        mode="transformer",
    ).to(device)
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded.load_state_dict(loaded_checkpoint["model_state"], strict=True)
    reloaded.eval()
    with torch.no_grad():
        reference_logits = model(val_x)
        reloaded_logits = reloaded(val_x)
    reload_max_abs_diff = float(torch.max(torch.abs(reference_logits - reloaded_logits)).item())
    require(reload_max_abs_diff <= 1e-6, f"checkpoint 重载差值={reload_max_abs_diff}")
    rss_samples.append(process_rss_bytes())

    torch.cuda.synchronize(device)
    return {
        "status": "PASS",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "model_mode": "transformer",
        "model_max_src": MODEL_MAX_SRC,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": args.batch_size,
        "selected_train_indices": selected_indices,
        "train_source_counts": src_count.tolist(),
        "input_shape": list(x.shape),
        "label_shape": list(band_mask.shape),
        "output_shape": list(logits.shape),
        "train_loss": float(loss.detach().item()),
        "ignore_loss_abs_diff": ignore_loss_diff,
        "gradient_norm": gradient_norm,
        "parameter_max_change": parameter_max_change,
        "val_loss": float(val_loss.item()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "reload_max_abs_diff": reload_max_abs_diff,
        "rss_start_bytes": rss_samples[0],
        "rss_peak_observed_bytes": max(rss_samples),
        "rss_end_bytes": rss_samples[-1],
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "gate1_report.json"
    audit_path = args.output_dir / "data_audit.json"
    started = time.perf_counter()
    set_seed(args.seed)

    base_report: dict[str, Any] = {
        "gate": "Gate 1 / chapter 3 smoke",
        "status": "RUNNING",
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
    }

    try:
        audit_started = time.perf_counter()
        audit_report = audit_all(args.data_dir)
        audit_report["elapsed_seconds"] = time.perf_counter() - audit_started
        write_json(audit_path, audit_report)

        model_started = time.perf_counter()
        model_report = run_model_smoke(args)
        model_report["elapsed_seconds"] = time.perf_counter() - model_started

        base_report.update(
            {
                "status": "PASS",
                "elapsed_seconds": time.perf_counter() - started,
                "data_audit_path": str(audit_path),
                "model_smoke": model_report,
            }
        )
        write_json(report_path, base_report)
        print(json.dumps(base_report, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        if not audit_path.exists():
            write_json(
                audit_path,
                {
                    "status": "FAIL",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
        base_report.update(
            {
                "status": "FAIL",
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(report_path, base_report)
        print(json.dumps(base_report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
