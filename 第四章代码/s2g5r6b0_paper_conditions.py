"""S2-G5-R6-B0论文控制条件下的D8双轨诊断。

该入口不训练模型、不读取冻结test，只使用新生成的4A/4B控制诊断IQ，
比较Exact-Oracle与Hard-19-Actual-Oracle细DPD上的三个冻结D8 seed。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CH4_DIR = PROJECT_ROOT / "第四章代码"
if str(CH4_DIR) not in sys.path:
    sys.path.insert(0, str(CH4_DIR))

from dpd_calculator_torch import DPDGeometry, compute_fine_dpd  # noqa: E402
from s2g3_composability import (  # noqa: E402
    decode_d8_sample,
    gospa_sample,
    matched_distances,
    subband_union_to_fft_mask,
)
from s2g4_coarse_d8 import build_model  # noqa: E402
from train_yolo import configure_reproducibility  # noqa: E402


R6A_ROOT = PROJECT_ROOT / "outputs" / "s2g5r6" / "20260829_170441"
R6A_REPORT = R6A_ROOT / "validation" / "final_report.json"
D8_TRAIN_MAT = (
    PROJECT_ROOT / "outputs" / "s2g4r4_scale" / "20260826_132829"
    / "04_matlab_runtime" / "smoke" / "chapter4" / "data" / "train_data.mat"
)
SEEDS = (42, 1042, 2042)
MODES = ("exact", "hard_actual")
EDGE = 2000.0
GRID_STEP = 10.0
FS = 100e6
LEN = 4096
HARD_THRESHOLD = 0.2
BOOTSTRAP_SEED = 20260830
BOOTSTRAP_REPETITIONS = 2000
EXTENSION_CONDITIONS = {
    "4B_K2_DIST_1000",
    "4B_K3_DIST_800",
    "4B_K3_DIST_1000",
}

PAPER_RMSE_M = {
    "4A_K2_SNR_-10": 97.4,
    "4A_K2_SNR_-6": 17.1,
    "4A_K2_SNR_+0": 12.5,
    "4A_K2_SNR_+6": 11.3,
    "4A_K3_SNR_-10": 166.0,
    "4A_K3_SNR_-6": 18.1,
    "4A_K3_SNR_+0": 13.7,
    "4A_K3_SNR_+6": 12.8,
    "4B_K2_DIST_800": 15.4,
    "4B_K2_DIST_1000": 18.9,
    "4B_K3_DIST_800": 19.9,
    "4B_K3_DIST_1000": 40.3,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, include_hash: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"文件不存在: {resolved}")
    result = {"path": str(resolved), "size_bytes": resolved.stat().st_size}
    if include_hash:
        result["sha256"] = sha256_file(resolved)
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"拒绝覆盖已有JSON: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), f"拒绝覆盖已有JSONL: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def checkpoint_identities() -> dict[int, dict[str, Any]]:
    report = load_json(R6A_REPORT)
    require(report.get("status") == "PASS", "R6-A最终报告非PASS")
    result: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        pair = report["diagonal"][f"{seed}/{seed}"]
        recorded = pair["checkpoints"]["d8"]
        path = Path(recorded["path"])
        current = file_identity(path)
        require(current["sha256"] == recorded["sha256"], f"D8 seed {seed} SHA变化")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        require(checkpoint.get("method") == "dualhead", f"D8 seed {seed} method错误")
        require(checkpoint.get("save_tag") == "dualhead_std", f"D8 seed {seed}不是D8")
        args = checkpoint.get("args", {})
        require(int(args.get("seed", -1)) == seed, f"D8 seed {seed}配置中的seed错误")
        current["epoch"] = int(checkpoint.get("epoch", -1))
        current["seed"] = seed
        result[seed] = current
        del checkpoint
    return result


def _mat_samples_slots(handle: h5py.File, name: str) -> np.ndarray:
    array = np.asarray(handle[name])
    if array.ndim == 2 and array.shape[0] <= 10:
        array = array.T
    return array


def _mat_positions(handle: h5py.File) -> np.ndarray:
    array = np.asarray(handle["src_pos_all"], dtype=np.float32)
    require(array.ndim == 3, "src_pos_all维数错误")
    if array.shape[0] == 2:
        array = np.transpose(array, (2, 1, 0))
    return array


def training_support_audit() -> dict[str, Any]:
    with h5py.File(D8_TRAIN_MAT, "r") as handle:
        counts = np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1)
        symbol_rate = _mat_samples_slots(handle, "symbolRate_all").astype(np.float64)
        bandwidth = _mat_samples_slots(handle, "BW_actual_all").astype(np.float64)
        center = _mat_samples_slots(handle, "fc_offset_all").astype(np.float64)
        snr = np.asarray(handle["avg_snr_all"], dtype=np.float64).reshape(-1)
        positions = _mat_positions(handle).astype(np.float64)
    require(counts.size == 8192, "冻结D8训练MAT不是8192条")
    valid = np.arange(symbol_rate.shape[1])[None, :] < counts[:, None]
    radii = np.linalg.norm(positions, axis=2)
    spreads = np.array([
        np.ptp(center[index, :count]) for index, count in enumerate(counts)
    ])
    joint = 0
    for index, count in enumerate(counts):
        if (
            np.all(symbol_rate[index, :count] == 10e6)
            and np.all(center[index, :count] == 0)
            and np.all(np.abs(radii[index, :count] - 800) <= 5)
            and abs(snr[index]) <= 0.25
        ):
            joint += 1
    return {
        "source_file": file_identity(D8_TRAIN_MAT),
        "sample_count": int(counts.size),
        "source_count_histogram": {
            str(value): int(np.sum(counts == value)) for value in sorted(np.unique(counts))
        },
        "active_source_count": int(valid.sum()),
        "symbol_rate_mhz": {
            "min": float(symbol_rate[valid].min() / 1e6),
            "max": float(symbol_rate[valid].max() / 1e6),
            "exact_10mhz_sources": int(np.sum(symbol_rate[valid] == 10e6)),
        },
        "bw_actual_mhz": {
            "min": float(bandwidth[valid].min() / 1e6),
            "max": float(bandwidth[valid].max() / 1e6),
            "exact_13mhz_sources": int(np.sum(bandwidth[valid] == 13e6)),
        },
        "recorded_avg_snr_db": {
            "min": float(snr.min()), "max": float(snr.max()),
            "within_0p25_db_of_zero_samples": int(np.sum(np.abs(snr) <= 0.25)),
            "exact_zero_samples": int(np.sum(snr == 0)),
        },
        "active_source_radius_m": {
            "min": float(radii[valid].min()), "max": float(radii[valid].max()),
            "within_5m_of_800_sources": int(np.sum(np.abs(radii[valid] - 800) <= 5)),
            "exact_800m_sources": int(np.sum(radii[valid] == 800)),
        },
        "cochannel_support": {
            "minimum_center_spread_hz": float(spreads.min()),
            "spread_le_50khz_samples": int(np.sum(spreads <= 50e3)),
            "all_centers_exact_zero_samples": int(np.sum(np.all(center == 0, axis=1))),
        },
        "joint_paper_4a_zero_db_near_match_samples": joint,
        "interpretation": (
            "论文控制参数的边际范围被训练分布覆盖，但10MHz/13MHz、严格同频、"
            "固定800m和0dB的联合组合没有原样出现。"
        ),
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    require(run_root.is_dir(), f"运行根不存在: {run_root}")
    require(not any(run_root.iterdir()), f"R6-B0运行根必须为空: {run_root}")
    checkpoints = checkpoint_identities()
    references = {
        "rmse_m": PAPER_RMSE_M,
        "source": "论文图4A/4B与原绘图脚本中的D8硬编码数组",
        "plot_4a": file_identity(CH4_DIR / "plot_4A.py"),
        "plot_4b": file_identity(CH4_DIR / "plot_4B.py"),
        "uncertainty_available": False,
    }
    report = {
        "status": "PASS",
        "gate": "S2-G5-R6-B0",
        "scope": "paper_condition_control_diagnostic_not_frozen_test",
        "run_root": str(run_root),
        "checkpoints": {str(seed): value for seed, value in checkpoints.items()},
        "paper_references": references,
        "training_support": training_support_audit(),
        "fixed_rules": {
            "training_executed": False,
            "test_executed": False,
            "ch3_executed": False,
            "oracle_k": True,
            "modes": list(MODES),
            "primary_samples_per_condition": 512,
            "maximum_generated_iq_per_condition": 1000,
            "hard19_threshold": HARD_THRESHOLD,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "code": file_identity(Path(__file__)),
    }
    write_json(run_root / "preflight.json", report)
    return report


def _scalar(handle: h5py.File, name: str) -> float:
    return float(np.asarray(handle[name]).reshape(-1)[0])


def audit_one_iq(path: Path, entry: dict[str, Any], expected: int) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        required = {
            "sig_rcv_real_all", "sig_rcv_imag_all", "src_count_all", "src_pos_all",
            "fc_offset_all", "symbolRate_all", "BW_actual_all", "snr_param_all",
            "dist_param_all", "sample_id_all", "rcv_pos_val", "sub_f_lo_val",
            "sub_f_hi_val", "B_win_val", "B_step_val", "fs_val", "lamda_val",
        }
        require(required.issubset(handle.keys()), f"{path.name}缺少字段")
        counts = np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1)
        require(counts.size == expected and np.all(counts == int(entry["n_src"])),
                f"{path.name}样本数或K错误")
        sample_ids = np.asarray(handle["sample_id_all"], dtype=np.int64).reshape(-1)
        require(np.array_equal(sample_ids, np.arange(expected)), f"{path.name}样本ID错误")
        centers = _mat_samples_slots(handle, "fc_offset_all")
        rates = _mat_samples_slots(handle, "symbolRate_all")
        widths = _mat_samples_slots(handle, "BW_actual_all")
        count = int(entry["n_src"])
        require(np.all(centers[:, :count] == 0), f"{path.name}不是严格同频")
        require(np.all(rates[:, :count] == 10e6), f"{path.name}符号率不是10MHz")
        require(np.all(widths[:, :count] == 13e6), f"{path.name}实际带宽不是13MHz")
        positions = _mat_positions(handle)
        radii = np.linalg.norm(positions[:, :count], axis=2)
        expected_distance = 800.0 if entry["experiment"] == "4A" else float(entry["parameter_value"])
        require(np.allclose(radii, expected_distance, rtol=0.0, atol=2e-3),
                f"{path.name}距离不符合控制条件")
        snr_target = np.asarray(handle["snr_param_all"], dtype=np.float32).reshape(-1)
        expected_snr = float(entry["parameter_value"]) if entry["experiment"] == "4A" else 0.0
        require(np.all(snr_target == expected_snr), f"{path.name}目标SNR错误")
        require(_scalar(handle, "fs_val") == FS and _scalar(handle, "lamda_val") == GRID_STEP,
                f"{path.name}采样率或网格步长错误")
        real = handle["sig_rcv_real_all"]
        imag = handle["sig_rcv_imag_all"]
        require(real.shape == (LEN, 4, expected) and imag.shape == real.shape,
                f"{path.name}IQ shape错误: {real.shape}")
        finite = True
        for start in range(0, expected, 64):
            stop = min(start + 64, expected)
            finite &= bool(np.isfinite(real[:, :, start:stop]).all())
            finite &= bool(np.isfinite(imag[:, :, start:stop]).all())
        require(finite, f"{path.name}IQ含NaN/Inf")
        result = {
            "condition_id": entry["condition_id"],
            "experiment": entry["experiment"],
            "n_src": count,
            "parameter_name": entry["parameter_name"],
            "parameter_value": float(entry["parameter_value"]),
            "seed": int(entry["seed"]),
            "sample_count": expected,
            "target_snr_db": expected_snr,
            "distance_m": expected_distance,
            "actual_avg_snr_db": {
                "min": float(np.asarray(handle["avg_snr_all"]).min()),
                "max": float(np.asarray(handle["avg_snr_all"]).max()),
            },
            "identity": file_identity(path),
        }
    return result


def run_audit_iq(args: argparse.Namespace) -> dict[str, Any]:
    generation = load_json(args.generation_report.resolve())
    require(generation.get("status") == "PASS", "MATLAB生成报告非PASS")
    expected = int(generation["samples_per_condition"])
    require(expected == args.expected_samples, "MATLAB样本数与审计参数不一致")
    require(generation["condition_count"] == 12, "控制条件不是12个")
    entries = []
    for source in generation["conditions"]:
        path = Path(source["path"])
        entries.append(audit_one_iq(path, source, expected))
    require({entry["condition_id"] for entry in entries} == set(PAPER_RMSE_M),
            "IQ控制条件集合与论文锚点不一致")
    report = {
        "status": "PASS", "gate": "S2-G5-R6-B0", "stage": "iq_audit",
        "sample_count": sum(entry["sample_count"] for entry in entries),
        "samples_per_condition": expected,
        "conditions": entries,
        "generation_report": file_identity(args.generation_report.resolve()),
        "test_executed": False, "training_executed": False,
    }
    write_json(args.output.resolve(), report)
    return report


def _read_iq_sample(handle: h5py.File, index: int) -> np.ndarray:
    real = np.asarray(handle["sig_rcv_real_all"][:, :, index], dtype=np.float32).T
    imag = np.asarray(handle["sig_rcv_imag_all"][:, :, index], dtype=np.float32).T
    return real + 1j * imag


def _read_position_sample(handle: h5py.File, index: int, count: int) -> np.ndarray:
    position = np.asarray(handle["src_pos_all"][:, :count, index], dtype=np.float32).T
    require(position.shape == (count, 2), "位置shape错误")
    return position


def _hard_mask(
    centers: np.ndarray, widths: np.ndarray, count: int,
    sub_lo: np.ndarray, sub_hi: np.ndarray, b_win: float,
) -> tuple[np.ndarray, np.ndarray]:
    slots = np.zeros((count, len(sub_lo)), dtype=bool)
    for source in range(count):
        band_lo = float(centers[source] - widths[source] / 2)
        band_hi = float(centers[source] + widths[source] / 2)
        overlap = np.maximum(0.0, np.minimum(band_hi, sub_hi) - np.maximum(band_lo, sub_lo))
        slots[source] = overlap / b_win >= HARD_THRESHOLD
    return subband_union_to_fft_mask(slots, sub_lo, sub_hi), slots


def _exact_mask(centers: np.ndarray, widths: np.ndarray, count: int) -> np.ndarray:
    f_axis = np.arange(-LEN // 2, LEN // 2, dtype=np.float64) * (FS / LEN)
    mask = np.zeros(LEN, dtype=bool)
    for source in range(count):
        lo = float(centers[source] - widths[source] / 2)
        hi = float(centers[source] + widths[source] / 2)
        mask |= (f_axis >= lo) & (f_axis < hi)
    return mask


def run_build_dpd(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.iq_manifest.resolve())
    require(manifest.get("status") == "PASS", "IQ manifest非PASS")
    start_index = int(args.start_index)
    stop_index = int(args.stop_index)
    require(0 <= start_index < stop_index <= manifest["samples_per_condition"], "DPD索引范围错误")
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"拒绝覆盖DPD目录: {output_dir}")
    output_dir.mkdir(parents=True)
    selected_ids = set(args.condition_id or PAPER_RMSE_M)
    require(selected_ids.issubset(PAPER_RMSE_M), "请求了未知控制条件")
    modes = tuple(args.mode or MODES)
    require(set(modes).issubset(MODES), "DPD模式错误")
    device = torch.device(args.device)
    require(device.type == "cuda" and (device.index or 0) == 0, "R6-B0固定cuda:0")
    torch.cuda.set_device(0)
    started_all = time.perf_counter()
    condition_reports = []
    total_tasks = 0
    for entry in manifest["conditions"]:
        condition_id = entry["condition_id"]
        if condition_id not in selected_ids:
            continue
        source_path = Path(entry["identity"]["path"])
        require(sha256_file(source_path) == entry["identity"]["sha256"], f"{condition_id} IQ SHA变化")
        condition_started = time.perf_counter()
        with h5py.File(source_path, "r") as handle:
            rcv_pos = np.asarray(handle["rcv_pos_val"], dtype=np.float32).T
            geometry = DPDGeometry(rcv_pos, np.array([0, 0], dtype=np.float32), EDGE,
                                   GRID_STEP, FS, LEN, device)
            centers_all = _mat_samples_slots(handle, "fc_offset_all").astype(np.float64)
            widths_all = _mat_samples_slots(handle, "BW_actual_all").astype(np.float64)
            sub_lo = np.asarray(handle["sub_f_lo_val"], dtype=np.float64).reshape(-1)
            sub_hi = np.asarray(handle["sub_f_hi_val"], dtype=np.float64).reshape(-1)
            b_win = _scalar(handle, "B_win_val")
            count = int(entry["n_src"])
            mode_buffers: dict[str, list[torch.Tensor]] = {mode: [] for mode in modes}
            shared_position: list[torch.Tensor] = []
            shared_indices: list[int] = []
            shard_files: dict[str, list[str]] = {mode: [] for mode in modes}
            mode_seconds: dict[str, list[float]] = {mode: [] for mode in modes}
            mask_bins: dict[str, list[int]] = {mode: [] for mode in modes}
            exact_hard_different = 0

            def flush() -> None:
                if not shared_indices:
                    return
                positions = torch.stack(shared_position)
                indices = torch.tensor(shared_indices, dtype=torch.int64)
                counts = torch.full((len(shared_indices),), count, dtype=torch.int64)
                for mode in modes:
                    shard_index = len(shard_files[mode])
                    mode_dir = output_dir / mode / condition_id
                    mode_dir.mkdir(parents=True, exist_ok=True)
                    path = mode_dir / f"shard_{shard_index:04d}.pt"
                    payload = {
                        "fine_dpd": torch.stack(mode_buffers[mode]),
                        "pos_label": positions,
                        "n_src": counts,
                        "sample_idx": indices,
                        "condition_id": condition_id,
                        "mode": mode,
                    }
                    torch.save(payload, path)
                    shard_files[mode].append(str(path.resolve()))
                    mode_buffers[mode].clear()
                shared_position.clear()
                shared_indices.clear()

            for sample_index in range(start_index, stop_index):
                signal = _read_iq_sample(handle, sample_index)
                positions = _read_position_sample(handle, sample_index, count)
                centers = centers_all[sample_index, :count]
                widths = widths_all[sample_index, :count]
                masks = {}
                if "exact" in modes:
                    masks["exact"] = _exact_mask(centers, widths, count)
                if "hard_actual" in modes:
                    masks["hard_actual"], _ = _hard_mask(
                        centers, widths, count, sub_lo, sub_hi, b_win,
                    )
                if "exact" in masks and "hard_actual" in masks:
                    exact_hard_different += int(not np.array_equal(masks["exact"], masks["hard_actual"]))
                for mode, mask in masks.items():
                    require(bool(mask.any()), f"{condition_id}/{sample_index}/{mode}频带为空")
                    task_started = time.perf_counter()
                    spectrum = compute_fine_dpd(
                        signal, geometry, freq_mask=mask, chunk_size=args.chunk_size,
                    )
                    mode_seconds[mode].append(time.perf_counter() - task_started)
                    require(bool(torch.isfinite(spectrum).all()) and bool(torch.all(spectrum >= 0)),
                            f"{condition_id}/{sample_index}/{mode} DPD非法")
                    mode_buffers[mode].append(torch.log1p(spectrum).unsqueeze(0).half().cpu())
                    mask_bins[mode].append(int(mask.sum()))
                    total_tasks += 1
                position_label = np.zeros((3, 2), dtype=np.float32)
                position_label[:count] = positions / EDGE
                shared_position.append(torch.from_numpy(position_label))
                shared_indices.append(sample_index)
                if len(shared_indices) == args.shard_size:
                    flush()
                if (sample_index - start_index + 1) % 32 == 0 or sample_index + 1 == stop_index:
                    print(
                        f"[R6-B0 DPD] {condition_id} {sample_index + 1 - start_index}/"
                        f"{stop_index - start_index}", flush=True,
                    )
            flush()
            del geometry
            torch.cuda.empty_cache()
        if set(modes) == set(MODES):
            require(exact_hard_different == stop_index - start_index,
                    f"{condition_id}存在Exact/Hard意外相同样本")
        condition_reports.append({
            "condition_id": condition_id,
            "n_src": count,
            "parameter_name": entry["parameter_name"],
            "parameter_value": entry["parameter_value"],
            "sample_start": start_index,
            "sample_stop_exclusive": stop_index,
            "sample_count": stop_index - start_index,
            "source_iq": entry["identity"],
            "shards": shard_files,
            "mask_bins": {
                mode: {"min": min(values), "max": max(values)} for mode, values in mask_bins.items()
            },
            "task_seconds": {
                mode: {
                    "count": len(values), "mean": float(np.mean(values)),
                    "median": float(np.median(values)), "max": float(np.max(values)),
                } for mode, values in mode_seconds.items()
            },
            "exact_hard_different_samples": exact_hard_different,
            "duration_seconds": time.perf_counter() - condition_started,
        })
    require({entry["condition_id"] for entry in condition_reports} == selected_ids,
            "DPD未覆盖全部请求条件")
    report = {
        "status": "PASS", "gate": "S2-G5-R6-B0", "stage": "build_dpd",
        "modes": list(modes), "device": str(device),
        "sample_start": start_index, "sample_stop_exclusive": stop_index,
        "samples_per_condition": stop_index - start_index,
        "condition_count": len(condition_reports), "total_dpd_tasks": total_tasks,
        "conditions": condition_reports,
        "duration_seconds": time.perf_counter() - started_all,
        "test_executed": False, "training_executed": False,
        "code": file_identity(Path(__file__)),
    }
    write_json(output_dir / "index.json", report)
    return report


def _normalize_batch(batch: torch.Tensor) -> torch.Tensor:
    batch = batch.float()
    dims = (1, 2, 3)
    return (batch - batch.mean(dim=dims, keepdim=True)) / (
        batch.std(dim=dims, keepdim=True) + 1e-6
    )


def numeric_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = np.asarray([
        value for row in rows for value in row["matched_errors_m"]
    ], dtype=np.float64)
    gospa = np.asarray([row["gospa_m"] for row in rows], dtype=np.float64)
    require(errors.size > 0 and np.isfinite(errors).all(), "匹配误差为空或非法")
    thresholds = (5, 10, 20, 30, 50, 100, 200, 500)
    return {
        "sample_count": len(rows),
        "source_count": int(errors.size),
        "rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "within": {f"{value}m": float(np.mean(errors <= value)) for value in thresholds},
        "above_100m": int(np.sum(errors > 100)),
        "above_500m": int(np.sum(errors > 500)),
        "mean_gospa_m": float(np.mean(gospa)),
    }


def evaluate_condition(
    model: torch.nn.Module, device: torch.device, shard_paths: list[str], batch_size: int,
    seed: int, mode: str, condition_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for shard_name in shard_paths:
            shard = torch.load(Path(shard_name), map_location="cpu", weights_only=False)
            require(shard["condition_id"] == condition_id and shard["mode"] == mode,
                    "DPD分片身份错误")
            for start in range(0, len(shard["n_src"]), batch_size):
                stop = min(start + batch_size, len(shard["n_src"]))
                batch = _normalize_batch(shard["fine_dpd"][start:stop]).to(device)
                heatmap, offset = model(batch)
                require(bool(torch.isfinite(heatmap).all() and torch.isfinite(offset).all()),
                        "D8输出NaN/Inf")
                for local in range(stop - start):
                    source_index = start + local
                    count = int(shard["n_src"][source_index])
                    true_positions = shard["pos_label"][source_index, :count].numpy() * EDGE
                    predicted, scores = decode_d8_sample(heatmap[local], offset[local], count)
                    matches = matched_distances(true_positions, predicted)
                    distances = [float(value) for _, _, value in matches]
                    require(len(distances) == count, "oracle-K下匹配数不等于真实源数")
                    gospa = gospa_sample(true_positions, predicted)
                    rows.append({
                        "seed": seed, "mode": mode, "condition_id": condition_id,
                        "sample_index": int(shard["sample_idx"][source_index]),
                        "true_count": count, "predicted_count": int(len(predicted)),
                        "matched_errors_m": distances,
                        "predicted_positions_m": predicted.tolist(),
                        "peak_scores": scores.tolist(),
                        "gospa_m": float(gospa["value_m"]),
                    })
                del batch, heatmap, offset
    rows.sort(key=lambda row: row["sample_index"])
    require(len({row["sample_index"] for row in rows}) == len(rows), "评价样本ID重复")
    return rows


def sample_bootstrap_rmse(
    rows_by_seed: dict[int, list[dict[str, Any]]], paper: float, repetitions: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    ordered = [rows_by_seed[seed] for seed in SEEDS]
    sample_ids = [[row["sample_index"] for row in rows] for rows in ordered]
    require(all(ids == sample_ids[0] for ids in sample_ids[1:]), "三个seed样本顺序不一致")
    n = len(ordered[0])
    squared = []
    for rows in ordered:
        squared.append(np.asarray([
            np.mean(np.square(row["matched_errors_m"])) for row in rows
        ], dtype=np.float64))
    values = np.empty(repetitions, dtype=np.float64)
    for repeat in range(repetitions):
        indices = rng.integers(0, n, size=n)
        seed_rmse = [math.sqrt(float(np.mean(array[indices]))) for array in squared]
        values[repeat] = float(np.median(seed_rmse))
    boundary = paper + max(5.0, 0.25 * paper)
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "unit": "sample",
        "repetitions": repetitions,
        "seed_median_rmse_m_ci95": [float(low), float(high)],
        "paper_compatibility_boundary_m": boundary,
        "boundary_crossed": bool(low <= boundary <= high),
        "does_not_measure_training_seed_uncertainty": True,
    }


def fallacy_scan() -> list[dict[str, str]]:
    return [
        {"fallacy": "selection_bias", "status": "CAUTION", "note": "只选4A/4B的12个锚点，不代表完整4A-4D。"},
        {"fallacy": "aggregation_bias", "status": "CONTROLLED", "note": "按K、实验和参数条件分层，不只报告总体RMSE。"},
        {"fallacy": "multiple_comparisons", "status": "CAUTION", "note": "12锚点用于诊断，不做显著性筛选或最优条件挑选。"},
        {"fallacy": "p_value_misuse", "status": "NOT_APPLICABLE", "note": "不计算或解释p值。"},
        {"fallacy": "effect_size_omission", "status": "CONTROLLED", "note": "同时报告绝对RMSE、相对论文值和阈值内比例。"},
        {"fallacy": "confidence_interval_misuse", "status": "CAUTION", "note": "bootstrap只衡量固定样本波动，论文参考值没有原始CI。"},
        {"fallacy": "pseudoreplication", "status": "CONTROLLED", "note": "按样本重采样，同一样本内多个信源不作为独立bootstrap单位。"},
        {"fallacy": "seed_independence", "status": "CAUTION", "note": "三个训练seed描述性汇总，不强行进行显著性检验。"},
        {"fallacy": "test_leakage", "status": "CONTROLLED", "note": "这是新控制诊断集，不读取R6-D/E冻结test，也不据此选择checkpoint。"},
        {"fallacy": "causal_overclaim", "status": "CAUTION", "note": "Exact/Hard双轨只能帮助定位表示影响，不能单独证明唯一因果机制。"},
        {"fallacy": "overgeneralization", "status": "CAUTION", "note": "样本数低于论文且只评价当前8k Hard-D8，不能称为完整论文复现。"},
    ]


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    index = load_json(args.dpd_index.resolve())
    require(index.get("status") == "PASS" and set(index["modes"]) == set(MODES), "DPD index错误")
    require(index["samples_per_condition"] == args.expected_samples, "DPD样本数错误")
    checkpoints = checkpoint_identities()
    device = torch.device(args.device)
    require(device.type == "cuda" and (device.index or 0) == 0, "R6-B0固定cuda:0")
    torch.cuda.set_device(0)
    output_dir = args.output_dir.resolve()
    require(not output_dir.exists(), f"拒绝覆盖评价目录: {output_dir}")
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    conditions = {entry["condition_id"]: entry for entry in index["conditions"]}
    require(set(conditions) == set(PAPER_RMSE_M), "DPD index未覆盖12锚点")
    sample_count_by_condition = {
        condition_id: int(entry["sample_count"])
        for condition_id, entry in conditions.items()
    }
    extension_identity = None
    if args.extension_dpd_index is not None:
        extension_path = args.extension_dpd_index.resolve()
        extension = load_json(extension_path)
        require(extension.get("status") == "PASS" and set(extension["modes"]) == set(MODES),
                "扩展DPD index错误")
        extension_conditions = {
            entry["condition_id"]: entry for entry in extension["conditions"]
        }
        require(set(extension_conditions) == EXTENSION_CONDITIONS,
                "扩展DPD必须只覆盖三个预注册歧义条件")
        for condition_id, extra in extension_conditions.items():
            base = conditions[condition_id]
            require(int(base["sample_start"]) == 0, f"{condition_id}基础DPD不是从0开始")
            require(int(extra["sample_start"]) == int(base["sample_stop_exclusive"]),
                    f"{condition_id}扩展DPD与基础范围不连续")
            require(base["source_iq"]["sha256"] == extra["source_iq"]["sha256"],
                    f"{condition_id}扩展DPD的IQ身份变化")
            for mode in MODES:
                base["shards"][mode].extend(extra["shards"][mode])
            sample_count_by_condition[condition_id] += int(extra["sample_count"])
        extension_identity = file_identity(extension_path)
    per_seed: dict[str, Any] = {}
    row_cache: dict[int, dict[str, dict[str, list[dict[str, Any]]]]] = {}
    for seed in SEEDS:
        configure_reproducibility(seed, True)
        model, _ = build_model(Path(checkpoints[seed]["path"]), device)
        row_cache[seed] = {mode: {} for mode in MODES}
        per_seed[str(seed)] = {mode: {} for mode in MODES}
        for mode in MODES:
            all_mode_rows = []
            for condition_id in PAPER_RMSE_M:
                rows = evaluate_condition(
                    model, device, conditions[condition_id]["shards"][mode],
                    args.batch_size, seed, mode, condition_id,
                )
                require(len(rows) == sample_count_by_condition[condition_id],
                        f"{seed}/{mode}/{condition_id}样本数错误")
                row_cache[seed][mode][condition_id] = rows
                per_seed[str(seed)][mode][condition_id] = numeric_metrics(rows)
                all_mode_rows.extend(rows)
                print(f"[R6-B0 eval] seed={seed} mode={mode} {condition_id}", flush=True)
            write_jsonl(output_dir / "samples" / f"seed_{seed}_{mode}.jsonl", all_mode_rows)
        del model
        torch.cuda.empty_cache()

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    anchor_analysis: dict[str, Any] = {mode: {} for mode in MODES}
    for mode in MODES:
        for condition_id, paper in PAPER_RMSE_M.items():
            seed_rmse = {
                seed: per_seed[str(seed)][mode][condition_id]["rmse_m"] for seed in SEEDS
            }
            median_rmse = float(np.median(list(seed_rmse.values())))
            tolerance = max(5.0, 0.25 * paper)
            compatible_by_seed = {seed: value <= paper + tolerance for seed, value in seed_rmse.items()}
            bootstrap = sample_bootstrap_rmse(
                {seed: row_cache[seed][mode][condition_id] for seed in SEEDS},
                paper, BOOTSTRAP_REPETITIONS, rng,
            )
            anchor_analysis[mode][condition_id] = {
                "paper_rmse_m": paper,
                "seed_rmse_m": {str(seed): value for seed, value in seed_rmse.items()},
                "seed_median_rmse_m": median_rmse,
                "difference_from_paper_m": median_rmse - paper,
                "ratio_to_paper": median_rmse / paper,
                "tolerance_m": tolerance,
                "compatible_by_seed": {str(seed): value for seed, value in compatible_by_seed.items()},
                "compatible_seed_count": int(sum(compatible_by_seed.values())),
                "anchor_compatible": bool(sum(compatible_by_seed.values()) >= 2 and median_rmse <= paper + tolerance),
                "grossly_far": bool(median_rmse > 2 * paper),
                "bootstrap": bootstrap,
            }

    exact_rows = anchor_analysis["exact"].values()
    close_count = sum(row["anchor_compatible"] for row in exact_rows)
    gross_count = sum(row["grossly_far"] for row in exact_rows)
    if close_count >= 9 and gross_count == 0:
        decision = "PAPER_CONDITION_CLOSE"
    elif close_count <= 6 or gross_count >= 3:
        decision = "PAPER_CONDITION_FAR"
    else:
        decision = "PAPER_CONDITION_MIXED"
    ambiguous = [
        condition_id for condition_id, row in anchor_analysis["exact"].items()
        if row["bootstrap"]["boundary_crossed"]
    ]
    representation = {}
    for condition_id in PAPER_RMSE_M:
        exact = anchor_analysis["exact"][condition_id]["seed_median_rmse_m"]
        hard = anchor_analysis["hard_actual"][condition_id]["seed_median_rmse_m"]
        representation[condition_id] = {
            "exact_seed_median_rmse_m": exact,
            "hard_seed_median_rmse_m": hard,
            "hard_minus_exact_m": hard - exact,
            "hard_over_exact": hard / exact,
        }
    report = {
        "status": "PASS", "gate": "S2-G5-R6-B0",
        "experiment_id": "CH4-S2G5-R6B0-20260830",
        "scope": "paper_condition_control_diagnostic_not_frozen_test",
        "scientific_decision": decision,
        "exact_close_anchor_count": close_count,
        "exact_grossly_far_anchor_count": gross_count,
        "ambiguous_exact_anchors": ambiguous,
        "extension_recommended": bool(ambiguous) and args.extension_dpd_index is None,
        "extension_completed": args.extension_dpd_index is not None,
        "uncertainty_remaining_after_extension": (
            bool(ambiguous) and args.extension_dpd_index is not None
        ),
        "checkpoints": {str(seed): value for seed, value in checkpoints.items()},
        "paper_reference": PAPER_RMSE_M,
        "per_seed": per_seed,
        "anchor_analysis": anchor_analysis,
        "representation_comparison": representation,
        "fallacy_scan": {
            "coverage": "11/11", "items": fallacy_scan(),
        },
        "duration_seconds": time.perf_counter() - started,
        "training_executed": False, "test_executed": False, "ch3_executed": False,
        "interpretation_boundaries": {
            "not_full_paper_reproduction": True,
            "paper_raw_errors_or_ci_available": False,
            "checkpoint_selection_performed": False,
            "scale_extension_authorized": args.extension_dpd_index is not None,
            "frozen_test_read": False,
        },
        "inputs": {
            "dpd_index": file_identity(args.dpd_index.resolve()),
            "extension_dpd_index": extension_identity,
        },
        "sample_count_by_condition": sample_count_by_condition,
        "code": file_identity(Path(__file__)),
    }
    write_json(output_dir / "final_report.json", report)
    summary = {
        key: report[key] for key in (
            "status", "gate", "experiment_id", "scientific_decision",
            "exact_close_anchor_count", "exact_grossly_far_anchor_count",
            "ambiguous_exact_anchors", "extension_recommended", "extension_completed",
            "uncertainty_remaining_after_extension",
            "representation_comparison", "training_executed", "test_executed",
        )
    }
    summary["anchor_analysis"] = anchor_analysis
    write_json(output_dir / "paper_condition_summary.json", summary)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--run_root", type=Path, required=True)

    audit = sub.add_parser("audit_iq")
    audit.add_argument("--generation_report", type=Path, required=True)
    audit.add_argument("--expected_samples", type=int, default=1000)
    audit.add_argument("--output", type=Path, required=True)

    build = sub.add_parser("build_dpd")
    build.add_argument("--iq_manifest", type=Path, required=True)
    build.add_argument("--output_dir", type=Path, required=True)
    build.add_argument("--start_index", type=int, default=0)
    build.add_argument("--stop_index", type=int, default=512)
    build.add_argument("--condition_id", action="append")
    build.add_argument("--mode", action="append", choices=MODES)
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--chunk_size", type=int, default=40000)
    build.add_argument("--shard_size", type=int, default=64)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--dpd_index", type=Path, required=True)
    evaluate.add_argument("--extension_dpd_index", type=Path)
    evaluate.add_argument("--output_dir", type=Path, required=True)
    evaluate.add_argument("--expected_samples", type=int, default=512)
    evaluate.add_argument("--batch_size", type=int, default=8)
    evaluate.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "audit_iq":
        result = run_audit_iq(args)
    elif args.command == "build_dpd":
        result = run_build_dpd(args)
    elif args.command == "evaluate":
        result = run_evaluate(args)
    else:
        raise RuntimeError(f"未知命令: {args.command}")
    print(json.dumps({
        "status": result["status"], "gate": result["gate"], "command": args.command,
    }, ensure_ascii=False))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8:replace")
    main()
