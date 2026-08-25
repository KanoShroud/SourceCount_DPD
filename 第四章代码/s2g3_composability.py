"""S2-G3跨章可组合性审查与条件式接口评估。

该入口只读复用既有IQ与checkpoint，所有输出由命令行显式指定。它不修改
第三/第四章原始训练、生成或评估入口，也不自动调阈值或改变科研定义。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CH3_DIR = PROJECT_ROOT / "第三章代码"
for import_root in (PROJECT_ROOT, CH3_DIR):
    if str(import_root) not in sys.path:
        sys.path.append(str(import_root))
if str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR))

from dpd_calculator_torch import DPDGeometry, compute_fine_dpd  # noqa: E402
from eval_ch4_checkpoint import require_s2g2_checkpoint  # noqa: E402
from train_v26 import SourceDetectionNet  # noqa: E402
from train_yolo import configure_reproducibility  # noqa: E402
from yolo_config import EDGE, PEAK_SIZE  # noqa: E402
from yolo_model import YOLOv8Loc, nms_heatmap, pixel_to_phys  # noqa: E402


FS = 100e6
LEN = 4096
N_SUB = 19
MAX_TRUE_SRC = 3
CH3_MAX_SRC = 10
COARSE_EDGE = 2000
COARSE_LAMDA = 50
FINE_EDGE = 2000
FINE_LAMDA = 10
GOSPA_P = 2.0
GOSPA_C = 100.0
GOSPA_ALPHA = 2.0
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_REPLICATES = 10_000


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"无法JSON序列化{type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有JSON: {path}")
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
    with path.resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"输入文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def configure_device(device_text: str, seed: int = 42) -> torch.device:
    configure_reproducibility(seed, True)
    device = torch.device(device_text)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "请求CUDA但torch.cuda不可用")
        torch.cuda.set_device(device.index or 0)
    return device


def mat_scalar(handle: h5py.File, name: str) -> float:
    require(name in handle, f"MAT缺少标量字段{name}")
    value = np.asarray(handle[name]).reshape(-1)
    require(value.size == 1, f"MAT字段{name}不是标量")
    return float(value[0])


def load_ch4_mat(path: Path, include_iq: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    with h5py.File(resolved, "r") as handle:
        required = {
            "src_count_all", "band_mask_all", "ignore_mask_all", "avg_snr_all",
            "fc_offset_all", "Pt_W_all", "src_pos_all", "symbolRate_all", "BW_actual_all",
            "N_sub_val", "max_src_val", "B_win_val", "B_step_val", "fs_val",
            "sub_f_lo_val", "sub_f_hi_val", "thresh_val",
        }
        if include_iq:
            required.update({"sig_rcv_real_all", "sig_rcv_imag_all"})
        missing = sorted(required.difference(handle.keys()))
        require(not missing, f"{resolved}缺少字段: {missing}")
        payload: dict[str, Any] = {
            "src_count": np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1),
            "band_mask": np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "ignore_mask": np.asarray(handle["ignore_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "avg_snr": np.asarray(handle["avg_snr_all"], dtype=np.float32).reshape(-1),
            "fc_offset": np.asarray(handle["fc_offset_all"], dtype=np.float32).T,
            "pt_w": np.asarray(handle["Pt_W_all"], dtype=np.float32).T,
            "src_pos": np.asarray(handle["src_pos_all"], dtype=np.float32).transpose(2, 1, 0),
            "symbol_rate": np.asarray(handle["symbolRate_all"], dtype=np.float32).T,
            "bw_actual": np.asarray(handle["BW_actual_all"], dtype=np.float32).T,
            "n_sub": int(mat_scalar(handle, "N_sub_val")),
            "max_src": int(mat_scalar(handle, "max_src_val")),
            "b_win": mat_scalar(handle, "B_win_val"),
            "b_step": mat_scalar(handle, "B_step_val"),
            "fs": mat_scalar(handle, "fs_val"),
            "sub_f_lo": np.asarray(handle["sub_f_lo_val"], dtype=np.float64).reshape(-1),
            "sub_f_hi": np.asarray(handle["sub_f_hi_val"], dtype=np.float64).reshape(-1),
            "threshold": mat_scalar(handle, "thresh_val"),
        }
        if include_iq:
            payload["sig_real"] = np.asarray(
                handle["sig_rcv_real_all"], dtype=np.float32
            ).transpose(2, 1, 0)
            payload["sig_imag"] = np.asarray(
                handle["sig_rcv_imag_all"], dtype=np.float32
            ).transpose(2, 1, 0)
    n = payload["src_count"].size
    require(payload["band_mask"].shape == (n, MAX_TRUE_SRC, N_SUB), "第四章band shape错误")
    require(payload["ignore_mask"].shape == (n, MAX_TRUE_SRC, N_SUB), "第四章ignore shape错误")
    require(payload["fc_offset"].shape == (n, MAX_TRUE_SRC), "第四章fc shape错误")
    require(payload["pt_w"].shape == (n, MAX_TRUE_SRC), "第四章功率shape错误")
    require(payload["src_pos"].shape == (n, MAX_TRUE_SRC, 2), "第四章位置shape错误")
    require(payload["symbol_rate"].shape == (n, MAX_TRUE_SRC), "第四章symbol rate shape错误")
    require(payload["bw_actual"].shape == (n, MAX_TRUE_SRC), "第四章BW shape错误")
    require(payload["n_sub"] == N_SUB and payload["max_src"] == MAX_TRUE_SRC, "第四章维度契约错误")
    require(payload["fs"] == FS, "第四章采样率契约错误")
    if include_iq:
        require(payload["sig_real"].shape == (n, 4, LEN), "第四章IQ实部shape错误")
        require(payload["sig_imag"].shape == (n, 4, LEN), "第四章IQ虚部shape错误")
        require(np.isfinite(payload["sig_real"]).all(), "第四章IQ实部含NaN/Inf")
        require(np.isfinite(payload["sig_imag"]).all(), "第四章IQ虚部含NaN/Inf")
    return payload


def load_ch3_metadata(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    with h5py.File(resolved, "r") as handle:
        required = {
            "src_count_all", "avg_snr_all", "fc_offset_all", "Pt_W_all", "src_pos_all",
            "symbolRate_val", "BW_actual_val", "fs_val", "B_win_val", "B_step_val",
            "N_sub_val", "max_src_val", "band_mask_all", "ignore_mask_all",
        }
        missing = sorted(required.difference(handle.keys()))
        require(not missing, f"{resolved}缺少字段: {missing}")
        return {
            "src_count": np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1),
            "avg_snr": np.asarray(handle["avg_snr_all"], dtype=np.float32).reshape(-1),
            "fc_offset": np.asarray(handle["fc_offset_all"], dtype=np.float32).T,
            "pt_w": np.asarray(handle["Pt_W_all"], dtype=np.float32).T,
            "src_pos": np.asarray(handle["src_pos_all"], dtype=np.float32).transpose(2, 1, 0),
            "symbol_rate_scalar": mat_scalar(handle, "symbolRate_val"),
            "bw_actual_scalar": mat_scalar(handle, "BW_actual_val"),
            "fs": mat_scalar(handle, "fs_val"),
            "b_win": mat_scalar(handle, "B_win_val"),
            "b_step": mat_scalar(handle, "B_step_val"),
            "n_sub": int(mat_scalar(handle, "N_sub_val")),
            "max_src": int(mat_scalar(handle, "max_src_val")),
            "band_mask": np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "ignore_mask": np.asarray(handle["ignore_mask_all"], dtype=np.float32).transpose(2, 1, 0),
        }


def active_values(values: np.ndarray, src_count: np.ndarray) -> np.ndarray:
    selected = [values[index, : int(count)] for index, count in enumerate(src_count) if count > 0]
    if not selected:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(selected).astype(np.float64)


def active_positions(src_pos: np.ndarray, src_count: np.ndarray) -> np.ndarray:
    selected = [src_pos[index, : int(count)] for index, count in enumerate(src_count) if count > 0]
    if not selected:
        return np.zeros((0, 2), dtype=np.float64)
    return np.concatenate(selected, axis=0).astype(np.float64)


def nearest_receiver_distances(src_pos: np.ndarray, src_count: np.ndarray) -> np.ndarray:
    positions = active_positions(src_pos, src_count)
    if not len(positions):
        return np.zeros(0, dtype=np.float64)
    angles = np.arange(4) * 2 * np.pi / 4
    receivers = np.stack([500.0 * np.cos(angles), 500.0 * np.sin(angles)], axis=1)
    return np.linalg.norm(positions[:, None, :] - receivers[None, :, :], axis=2).min(axis=1)


def pair_separations(src_pos: np.ndarray, src_count: np.ndarray) -> np.ndarray:
    result: list[float] = []
    for index, count_value in enumerate(src_count):
        count = int(count_value)
        for first in range(count):
            for second in range(first + 1, count):
                result.append(float(np.linalg.norm(src_pos[index, first] - src_pos[index, second])))
    return np.asarray(result, dtype=np.float64)


def connected_component_count(fc: np.ndarray, bw: np.ndarray, count: int) -> int:
    if count <= 0:
        return 0
    parent = list(range(count))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for first in range(count):
        for second in range(first + 1, count):
            lo1, hi1 = fc[first] - bw[first] / 2, fc[first] + bw[first] / 2
            lo2, hi2 = fc[second] - bw[second] / 2, fc[second] + bw[second] / 2
            if lo1 < hi2 and lo2 < hi1:
                root1, root2 = find(first), find(second)
                if root1 != root2:
                    parent[root1] = root2
    return len({find(index) for index in range(count)})


def numeric_summary(values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(finite.min()),
        "q05": float(np.percentile(finite, 5)),
        "median": float(np.median(finite)),
        "q95": float(np.percentile(finite, 95)),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
    }


def dataset_distribution_ch3(payload: dict[str, Any]) -> dict[str, Any]:
    counts = payload["src_count"]
    positions = active_positions(payload["src_pos"], counts)
    radii = np.linalg.norm(positions, axis=1)
    bw = np.full_like(payload["fc_offset"], payload["bw_actual_scalar"], dtype=np.float64)
    components = [
        connected_component_count(payload["fc_offset"][idx], bw[idx], int(count))
        for idx, count in enumerate(counts)
    ]
    valid_snr = payload["avg_snr"][payload["avg_snr"] > -900]
    return {
        "sample_count": int(counts.size),
        "source_count": {str(k): int(v) for k, v in sorted(Counter(counts.tolist()).items())},
        "symbol_rate_hz": numeric_summary(np.asarray([payload["symbol_rate_scalar"]])),
        "actual_bandwidth_hz": numeric_summary(np.asarray([payload["bw_actual_scalar"]])),
        "active_source_radius_m": numeric_summary(radii),
        "nearest_source_to_receiver_distance_m": numeric_summary(
            nearest_receiver_distances(payload["src_pos"], counts)
        ),
        "source_pair_separation_m": numeric_summary(pair_separations(payload["src_pos"], counts)),
        "transmit_power_w": numeric_summary(active_values(payload["pt_w"], counts)),
        "weakest_average_snr_db_code_definition": numeric_summary(valid_snr),
        "frequency_component_count": {str(k): int(v) for k, v in sorted(Counter(components).items())},
        "shared_acquisition": {
            "fs_hz": payload["fs"], "observation_length": LEN,
            "n_sub": payload["n_sub"], "subband_window_hz": payload["b_win"],
            "subband_step_hz": payload["b_step"], "receiver_count": 4,
            "receiver_radius_m": 500.0, "carrier_hz": 5.8e9,
        },
    }


def dataset_distribution_ch4(payload: dict[str, Any]) -> dict[str, Any]:
    counts = payload["src_count"]
    positions = active_positions(payload["src_pos"], counts)
    radii = np.linalg.norm(positions, axis=1)
    components = [
        connected_component_count(
            payload["fc_offset"][idx], payload["bw_actual"][idx], int(count)
        ) for idx, count in enumerate(counts)
    ]
    valid_snr = payload["avg_snr"][payload["avg_snr"] > -900]
    return {
        "sample_count": int(counts.size),
        "source_count": {str(k): int(v) for k, v in sorted(Counter(counts.tolist()).items())},
        "symbol_rate_hz": numeric_summary(active_values(payload["symbol_rate"], counts)),
        "actual_bandwidth_hz": numeric_summary(active_values(payload["bw_actual"], counts)),
        "active_source_radius_m": numeric_summary(radii),
        "nearest_source_to_receiver_distance_m": numeric_summary(
            nearest_receiver_distances(payload["src_pos"], counts)
        ),
        "source_pair_separation_m": numeric_summary(pair_separations(payload["src_pos"], counts)),
        "transmit_power_w": numeric_summary(active_values(payload["pt_w"], counts)),
        "weakest_average_snr_db_code_definition": numeric_summary(valid_snr),
        "frequency_component_count": {str(k): int(v) for k, v in sorted(Counter(components).items())},
        "shared_acquisition": {
            "fs_hz": payload["fs"], "observation_length": LEN,
            "n_sub": payload["n_sub"], "subband_window_hz": payload["b_win"],
            "subband_step_hz": payload["b_step"], "receiver_count": 4,
            "receiver_radius_m": 500.0, "carrier_hz": 5.8e9,
        },
    }


def load_exact_tasks(data_dir: Path, split: str = "val") -> dict[str, torch.Tensor]:
    split_dir = data_dir.resolve() / split
    index_path = split_dir / f"loc_{split}_index.pt"
    require(index_path.is_file(), f"第四章定位索引不存在: {index_path}")
    index = torch.load(index_path, map_location="cpu", weights_only=False)
    fields = {name: [] for name in ("fine_dpd", "pos_label", "n_src", "sample_idx", "group_idx")}
    for shard_name in index["shard_files"]:
        shard_path = split_dir / shard_name
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        for name in fields:
            require(name in shard, f"{shard_path}缺少{name}")
            fields[name].append(shard[name])
    result = {name: torch.cat(values, dim=0) for name, values in fields.items()}
    require(len(result["n_src"]) == int(index["n_total_tasks"]), "定位任务数与索引不一致")
    return result


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    ch3_splits: dict[str, Any] = {}
    ch4_splits: dict[str, Any] = {}
    identities: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        ch3_path = args.ch3_data_dir.resolve() / f"{split}_data.mat"
        ch4_path = args.ch4_data_dir.resolve() / f"{split}_data.mat"
        ch3_payload = load_ch3_metadata(ch3_path)
        ch4_payload = load_ch4_mat(ch4_path, include_iq=False)
        ch3_splits[split] = dataset_distribution_ch3(ch3_payload)
        ch4_splits[split] = dataset_distribution_ch4(ch4_payload)
        identities[f"ch3_{split}"] = file_identity(ch3_path)
        identities[f"ch4_{split}"] = file_identity(ch4_path)

    val_raw = load_ch4_mat(args.ch4_data_dir.resolve() / "val_data.mat", include_iq=False)
    exact = load_exact_tasks(args.exact_loc_data_dir, "val")
    expected_indices = torch.arange(len(val_raw["src_count"]), dtype=torch.long)
    require(torch.equal(exact["sample_idx"], expected_indices), "exact定位任务与原IQ不是0..N-1一一对应")
    require(bool(torch.all(exact["group_idx"] == 0)), "validation存在非零group_idx")
    require(
        np.array_equal(exact["n_src"].numpy(), val_raw["src_count"]),
        "exact定位任务源数与原IQ不一致",
    )
    require(set(val_raw["src_count"].tolist()) == {2, 3}, "S2-G3 validation不是仅2/3源")
    require(len(val_raw["src_count"]) == 256, "S2-G3 validation样本数不是256")

    exact_index = args.exact_loc_data_dir.resolve() / "val" / "loc_val_index.pt"
    identities["exact_val_index"] = file_identity(exact_index)
    identities["ch3_checkpoint"] = file_identity(args.ch3_checkpoint)
    identities["ch4_checkpoint"] = file_identity(args.ch4_checkpoint)

    contract_rows = [
        {"item": "array_and_sampling", "status": "MATCH", "detail": "4站、5.8GHz、100MHz、4096点一致"},
        {"item": "coarse_subbands", "status": "MATCH", "detail": "19个10MHz窗、5MHz步长一致"},
        {"item": "signal_family", "status": "MATCH", "detail": "BPSK与RRC rolloff=0.25一致"},
        {"item": "source_count", "status": "DISTRIBUTION_SHIFT", "detail": "第三章0/1/2/3；第四章仅2/3"},
        {"item": "symbol_rate", "status": "SUPPORT_AND_DISTRIBUTION_SHIFT", "detail": "第三章固定10MHz；第四章逐源2–20MHz"},
        {"item": "frequency_topology", "status": "DISTRIBUTION_SHIFT", "detail": "第三章混合拓扑；第四章全部属于单一重叠组"},
        {"item": "source_geometry", "status": "SUPPORT_AND_DISTRIBUTION_SHIFT", "detail": "半径范围与最小源间距不同"},
        {"item": "power_and_snr", "status": "DEFINITION_MISMATCH", "detail": "随机发射功率/全噪声功率与目标带内SNR反推功率不可直接数值等同比较"},
        {"item": "band_label", "status": "FORMULA_MATCH_VARIABLE_INPUT", "detail": "覆盖阈值与ignore语义一致，但第四章使用逐源symbol rate/BW"},
    ]
    payload = {
        "status": "PASS",
        "gate": "S2-G3",
        "stage": "distribution_contract_audit",
        "ch3": ch3_splits,
        "ch4": ch4_splits,
        "contract_rows": contract_rows,
        "shared_validation_identity": {
            "raw_sample_count": 256,
            "exact_task_count": int(len(exact["n_src"])),
            "sample_indices_exact_0_to_255": True,
            "all_group_idx_zero": True,
            "source_count_distribution": {
                str(k): int(v) for k, v in sorted(Counter(val_raw["src_count"].tolist()).items())
            },
        },
        "input_identities": identities,
        "duration_seconds": time.perf_counter() - started,
        "interpretation": "两章共享物理采集和粗DPD契约，但存在足以阻止直接性能归因的数据分布差异。",
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    markdown_path = args.markdown.resolve()
    if markdown_path.exists():
        raise FileExistsError(f"拒绝覆盖已有Markdown: {markdown_path}")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "# S2-G3 数据分布契约审计",
        "",
        "| 项目 | 状态 | 说明 |",
        "|---|---|---|",
    ]
    rows.extend(f"| {row['item']} | {row['status']} | {row['detail']} |" for row in contract_rows)
    rows.extend([
        "",
        "结论：两章并非无关系统，但现有冻结模型的直接组合必须被视为待检验迁移假设，不能预设成立。",
    ])
    markdown_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return payload


def load_coarse_dpd(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with h5py.File(path.resolve(), "r") as handle:
        required = {"mtr_sub_all", "src_count_all", "band_mask_all", "ignore_mask_all", "N_sub_val", "max_src_val"}
        missing = sorted(required.difference(handle.keys()))
        require(not missing, f"粗DPD MAT缺少字段: {missing}")
        spectra = np.asarray(handle["mtr_sub_all"], dtype=np.float32).transpose(3, 2, 1, 0)
        metadata = {
            "src_count": np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1),
            "band_mask": np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "ignore_mask": np.asarray(handle["ignore_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "n_sub": int(mat_scalar(handle, "N_sub_val")),
            "data_max_src": int(mat_scalar(handle, "max_src_val")),
        }
    require(spectra.ndim == 4 and spectra.shape[1:] == (N_SUB, 81, 81), f"粗DPD shape错误: {spectra.shape}")
    require(np.isfinite(spectra).all() and np.all(spectra >= 0), "粗DPD含NaN/Inf或负值")
    return spectra, metadata


def subband_union_to_fft_mask(slot_mask: np.ndarray, sub_lo: np.ndarray, sub_hi: np.ndarray) -> np.ndarray:
    f_axis = np.arange(-LEN // 2, LEN // 2, dtype=np.float64) * (FS / LEN)
    union = np.asarray(slot_mask, dtype=bool).any(axis=0)
    result = np.zeros(LEN, dtype=bool)
    for sub_index in np.flatnonzero(union):
        result |= (f_axis >= sub_lo[sub_index]) & (f_axis < sub_hi[sub_index])
    return result


def actual_union_mask(fc: np.ndarray, bw: np.ndarray, count: int) -> np.ndarray:
    f_axis = np.arange(-LEN // 2, LEN // 2, dtype=np.float64) * (FS / LEN)
    result = np.zeros(LEN, dtype=bool)
    for source in range(count):
        result |= (
            (f_axis >= float(fc[source] - bw[source] / 2))
            & (f_axis < float(fc[source] + bw[source] / 2))
        )
    return result


def bootstrap_mean_difference_lower(
    observed: np.ndarray, baseline: np.ndarray, seed: int, replicates: int,
) -> tuple[float, float, float]:
    observed = np.asarray(observed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    require(observed.shape == baseline.shape and observed.ndim == 1, "bootstrap输入shape错误")
    delta = observed - baseline
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    n = len(delta)
    for start in range(0, replicates, 500):
        block = min(500, replicates - start)
        indices = rng.integers(0, n, size=(block, n))
        estimates[start : start + block] = delta[indices].mean(axis=1)
    return float(delta.mean()), float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    require(0 <= successes <= total and total > 0, "Wilson输入非法")
    phat = successes / total
    denominator = 1 + z * z / total
    center = phat + z * z / (2 * total)
    margin = z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))
    return (center - margin) / denominator


def build_ch3_model(checkpoint_path: Path, device: torch.device) -> tuple[SourceDetectionNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location=device, weights_only=False)
    config = checkpoint.get("config")
    require(isinstance(config, dict), "第三章checkpoint缺少config")
    require(config.get("mode") == "transformer", "第三章checkpoint不是Transformer")
    require(int(config.get("max_src", -1)) == CH3_MAX_SRC, "第三章checkpoint不是M=10")
    require(int(config.get("n_sub", -1)) == N_SUB, "第三章checkpoint N_sub不为19")
    require(float(config.get("threshold", -1)) == 0.5, "第三章checkpoint阈值不是0.5")
    require(int(checkpoint.get("epoch", -1)) == 96, "第三章checkpoint epoch不是96")
    model = SourceDetectionNet(n_sub=N_SUB, max_src=CH3_MAX_SRC, mode="transformer").to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    require(all(bool(torch.isfinite(value).all()) for value in model.state_dict().values()), "第三章权重含NaN/Inf")
    return model, checkpoint


def run_infer(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = configure_device(args.device, args.seed)
    spectra, metadata = load_coarse_dpd(args.coarse_mat)
    raw = load_ch4_mat(args.raw_mat, include_iq=False)
    n = len(spectra)
    require(n == 256 and len(raw["src_count"]) == n, "跨分布推理样本数不为256")
    require(np.array_equal(metadata["src_count"], raw["src_count"]), "粗DPD源数与原IQ不一致")
    require(np.array_equal(metadata["band_mask"], raw["band_mask"]), "粗DPD band标签与原IQ不一致")
    require(np.array_equal(metadata["ignore_mask"], raw["ignore_mask"]), "粗DPD ignore标签与原IQ不一致")

    spectra = np.log(spectra + 1.0)
    means = spectra.mean(axis=(1, 2, 3), keepdims=True)
    stds = spectra.std(axis=(1, 2, 3), keepdims=True) + 1e-6
    spectra = (spectra - means) / stds
    require(np.isfinite(spectra).all(), "第三章归一化输入含NaN/Inf")
    model, checkpoint = build_ch3_model(args.checkpoint, device)
    logits_blocks: list[np.ndarray] = []
    with torch.no_grad():
        for start_index in range(0, n, args.batch_size):
            tensor = torch.from_numpy(spectra[start_index : start_index + args.batch_size]).to(device)
            logits = model(tensor)
            require(bool(torch.isfinite(logits).all()), "第三章推理logits含NaN/Inf")
            logits_blocks.append(logits.cpu().numpy().astype(np.float32))
    logits = np.concatenate(logits_blocks, axis=0)
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    pred = probs > 0.5
    k_pred = pred.any(axis=-1).sum(axis=-1).astype(np.int64)
    true_count = raw["src_count"].astype(np.int64)

    confusion = np.zeros((4, CH3_MAX_SRC + 1), dtype=np.int64)
    for truth, prediction in zip(true_count, k_pred, strict=True):
        confusion[truth, prediction] += 1
    exact = int(np.sum(true_count == k_pred))
    class_acc = {
        str(count): float(np.mean(k_pred[true_count == count] == count)) for count in (2, 3)
    }

    slot_totals = np.zeros(4, dtype=np.int64)  # tp fp fn tn on active slots/non-ignore bins
    for sample_index, count_value in enumerate(true_count):
        count = int(count_value)
        valid = raw["ignore_mask"][sample_index, :count] == 0
        truth = raw["band_mask"][sample_index, :count] > 0.5
        prediction = pred[sample_index, :count]
        slot_totals += np.asarray([
            np.sum(prediction & truth & valid),
            np.sum(prediction & ~truth & valid),
            np.sum(~prediction & truth & valid),
            np.sum(~prediction & ~truth & valid),
        ], dtype=np.int64)
    tp, fp, fn, tn = [int(value) for value in slot_totals]
    slot_precision = tp / max(tp + fp, 1)
    slot_recall = tp / max(tp + fn, 1)
    slot_f1 = 2 * slot_precision * slot_recall / max(slot_precision + slot_recall, 1e-12)

    union_iou = np.zeros(n, dtype=np.float64)
    all_band_iou = np.zeros(n, dtype=np.float64)
    physical_coverage = np.zeros(n, dtype=np.float64)
    physical_purity = np.zeros(n, dtype=np.float64)
    bandwidth_expansion = np.zeros(n, dtype=np.float64)
    empty_band = np.zeros(n, dtype=bool)
    fft_masks = np.zeros((n, LEN), dtype=np.uint8)
    for sample_index, count_value in enumerate(true_count):
        count = int(count_value)
        predicted_mask = subband_union_to_fft_mask(
            pred[sample_index], raw["sub_f_lo"], raw["sub_f_hi"]
        )
        truth_mask = actual_union_mask(
            raw["fc_offset"][sample_index], raw["bw_actual"][sample_index], count
        )
        fft_masks[sample_index] = predicted_mask.astype(np.uint8)
        intersection = int(np.sum(predicted_mask & truth_mask))
        union = int(np.sum(predicted_mask | truth_mask))
        pred_width = int(np.sum(predicted_mask))
        true_width = int(np.sum(truth_mask))
        union_iou[sample_index] = intersection / max(union, 1)
        all_band_iou[sample_index] = true_width / LEN
        physical_coverage[sample_index] = intersection / max(true_width, 1)
        physical_purity[sample_index] = intersection / max(pred_width, 1)
        bandwidth_expansion[sample_index] = pred_width / max(true_width, 1)
        empty_band[sample_index] = pred_width == 0

    iou_delta_mean, iou_delta_lower, iou_delta_upper = bootstrap_mean_difference_lower(
        union_iou, all_band_iou, BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES
    )
    count_wilson = wilson_lower(exact, n)
    empty_rate = float(empty_band.mean())
    above_capacity_rate = float(np.mean(k_pred > MAX_TRUE_SRC))
    gates = {
        "count_accuracy_wilson_lower_above_0_5": count_wilson > 0.5,
        "physical_iou_bootstrap_lower_above_all_band": iou_delta_lower > 0.0,
        "median_true_band_coverage_at_least_0_8": float(np.median(physical_coverage)) >= 0.8,
        "empty_predicted_band_rate_at_most_0_05": empty_rate <= 0.05,
        "k_above_3_rate_at_most_0_05": above_capacity_rate <= 0.05,
    }
    model_gate_pass = all(gates.values())

    npz_path = args.output_npz.resolve()
    if npz_path.exists():
        raise FileExistsError(f"拒绝覆盖已有推理NPZ: {npz_path}")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        logits=logits,
        probabilities=probs.astype(np.float32),
        band_prediction=pred.astype(np.uint8),
        k_prediction=k_pred,
        true_count=true_count,
        predicted_fft_mask=fft_masks,
        sample_index=np.arange(n, dtype=np.int64),
    )
    report = {
        "status": "PASS",
        "stage": "ch3_cross_distribution_inference",
        "checkpoint": file_identity(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "threshold": 0.5,
        "sample_count": n,
        "count": {
            "accuracy": exact / n,
            "balanced_accuracy": float(np.mean(list(class_acc.values()))),
            "class_accuracy": class_acc,
            "mae": float(np.mean(np.abs(k_pred - true_count))),
            "under_rate": float(np.mean(k_pred < true_count)),
            "over_rate": float(np.mean(k_pred > true_count)),
            "k_zero_rate": float(np.mean(k_pred == 0)),
            "k_above_3_rate": above_capacity_rate,
            "wilson_95_lower": count_wilson,
            "confusion_true_0_3_pred_0_10": confusion.tolist(),
            "predicted_distribution": {str(k): int(v) for k, v in sorted(Counter(k_pred.tolist()).items())},
        },
        "band": {
            "active_slot_precision": slot_precision,
            "active_slot_recall": slot_recall,
            "active_slot_f1": slot_f1,
            "physical_union_iou": numeric_summary(union_iou),
            "all_band_trivial_iou": numeric_summary(all_band_iou),
            "iou_minus_all_band_bootstrap": {
                "mean": iou_delta_mean, "ci95_lower": iou_delta_lower, "ci95_upper": iou_delta_upper,
                "replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
            },
            "true_band_coverage": numeric_summary(physical_coverage),
            "predicted_band_purity": numeric_summary(physical_purity),
            "bandwidth_expansion_ratio": numeric_summary(bandwidth_expansion),
            "empty_band_rate": empty_rate,
        },
        "model_gate_checks_without_repeatability": gates,
        "model_gate_pass_without_repeatability": model_gate_pass,
        "npz": str(npz_path),
        "duration_seconds": time.perf_counter() - started,
        "performance_interpretation_allowed": False,
    }
    write_json(args.output_json, report)
    return report


def run_compare_inference(args: argparse.Namespace) -> dict[str, Any]:
    first = np.load(args.first_npz.resolve())
    second = np.load(args.second_npz.resolve())
    required = ("logits", "probabilities", "band_prediction", "k_prediction", "predicted_fft_mask")
    comparisons = {}
    for name in required:
        require(name in first.files and name in second.files, f"推理NPZ缺少{name}")
        comparisons[name] = bool(np.array_equal(first[name], second[name]))
    exact_repeat = all(comparisons.values())
    first_report = load_json(args.first_json)
    second_report = load_json(args.second_json)
    require(first_report["checkpoint"]["sha256"] == second_report["checkpoint"]["sha256"], "两次推理checkpoint不同")
    require(first_report["sample_count"] == second_report["sample_count"] == 256, "两次推理样本数错误")
    scientific_equal = first_report["count"] == second_report["count"] and first_report["band"] == second_report["band"]
    require(exact_repeat and scientific_equal, "第三章两次独立推理不完全一致")
    gates = dict(first_report["model_gate_checks_without_repeatability"])
    gates["independent_inference_exact_repeat"] = exact_repeat
    payload = {
        "status": "PASS",
        "stage": "compare_ch3_cross_distribution_inference",
        "array_exact_comparison": comparisons,
        "scientific_json_equal": scientific_equal,
        "model_gate_checks": gates,
        "model_gate_pass": all(gates.values()),
        "first_report": str(args.first_json.resolve()),
        "second_report": str(args.second_json.resolve()),
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def load_fine_custom(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    required = {"fine_dpd", "pos_label", "n_src", "sample_idx", "group_idx", "empty_band"}
    missing = sorted(required.difference(payload.keys()))
    require(not missing, f"细DPD文件缺少字段: {missing}")
    return payload


def run_generate_fine(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有细DPD: {output}")
    device = configure_device(args.device, args.seed)
    raw = load_ch4_mat(args.raw_mat, include_iq=True)
    n = len(raw["src_count"])
    require(n == 256, "细DPD输入validation不是256条")
    predictions = None
    if args.band_mode == "predicted":
        require(args.prediction_npz is not None, "predicted模式必须提供prediction_npz")
        predictions = np.load(args.prediction_npz.resolve())
        require(predictions["band_prediction"].shape == (n, CH3_MAX_SRC, N_SUB), "预测band shape错误")

    angles = np.arange(4) * 2 * np.pi / 4
    receiver_positions = np.stack([500.0 * np.cos(angles), 500.0 * np.sin(angles)], axis=1)
    geometry = DPDGeometry(
        receiver_positions, [0.0, 0.0], FINE_EDGE, FINE_LAMDA, FS, LEN, device
    )
    fine_dpd: list[torch.Tensor] = []
    pos_labels: list[torch.Tensor] = []
    empty_flags: list[bool] = []
    n_band_values: list[int] = []
    task_seconds: list[float] = []
    for sample_index, count_value in enumerate(raw["src_count"]):
        count = int(count_value)
        if args.band_mode == "coarse_oracle":
            slot_mask = raw["band_mask"][sample_index, :count] > 0.5
        else:
            assert predictions is not None
            slot_mask = predictions["band_prediction"][sample_index] > 0
        freq_mask = subband_union_to_fft_mask(slot_mask, raw["sub_f_lo"], raw["sub_f_hi"])
        empty = not bool(freq_mask.any())
        empty_flags.append(empty)
        n_band_values.append(int(freq_mask.sum()))
        sample_started = time.perf_counter()
        if empty:
            mtr_log = torch.zeros((401, 401), dtype=torch.float32)
        else:
            signal = raw["sig_real"][sample_index] + 1j * raw["sig_imag"][sample_index]
            mtr = compute_fine_dpd(signal, geometry, freq_mask=freq_mask, chunk_size=args.chunk_size)
            require(bool(torch.isfinite(mtr).all()) and bool(torch.all(mtr >= 0)), "细DPD含NaN/Inf或负值")
            mtr_log = torch.log(mtr + 1.0)
        task_seconds.append(time.perf_counter() - sample_started)
        fine_dpd.append(mtr_log.unsqueeze(0).half())
        positions = raw["src_pos"][sample_index, :count]
        positions = positions[np.argsort(np.linalg.norm(positions, axis=1))]
        pos_label = np.zeros((MAX_TRUE_SRC, 2), dtype=np.float32)
        pos_label[:count] = positions / EDGE
        pos_labels.append(torch.from_numpy(pos_label))
        if (sample_index + 1) % 16 == 0 or sample_index + 1 == n:
            print(
                f"[{args.band_mode}] {sample_index + 1}/{n} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    payload = {
        "fine_dpd": torch.stack(fine_dpd),
        "pos_label": torch.stack(pos_labels),
        "n_src": torch.from_numpy(raw["src_count"].astype(np.int64)),
        "sample_idx": torch.arange(n, dtype=torch.long),
        "group_idx": torch.zeros(n, dtype=torch.long),
        "empty_band": torch.tensor(empty_flags, dtype=torch.bool),
        "band_mode": args.band_mode,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report = {
        "status": "PASS",
        "stage": "generate_fine_dpd",
        "band_mode": args.band_mode,
        "sample_count": n,
        "empty_band_count": int(sum(empty_flags)),
        "n_band_bins": numeric_summary(np.asarray(n_band_values)),
        "task_seconds": numeric_summary(np.asarray(task_seconds)),
        "output": file_identity(output),
        "duration_seconds": time.perf_counter() - started,
        "performance_interpretation_allowed": False,
    }
    if args.band_mode == "coarse_oracle":
        require(not any(empty_flags), "Coarse-Oracle出现空频带")
    write_json(args.report, report)
    return report


def build_d8(checkpoint_path: Path, device: torch.device) -> tuple[YOLOv8Loc, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location=device, weights_only=False)
    require_s2g2_checkpoint(checkpoint)
    require(int(checkpoint.get("epoch", -1)) == 93, "S2-G2冻结checkpoint epoch不是93")
    state = checkpoint.get("model")
    require(isinstance(state, dict), "第四章checkpoint缺少model state")
    model = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def decode_d8_sample(
    heatmap: torch.Tensor, offset: torch.Tensor, count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros(0, dtype=np.float32)
    hm_nms = nms_heatmap(torch.sigmoid(heatmap.unsqueeze(0)), PEAK_SIZE)[0, 0]
    k = min(count, int(hm_nms.numel()))
    scores, indices = hm_nms.reshape(-1).topk(k)
    width = hm_nms.shape[1]
    x = (indices % width).float()
    y = (indices // width).float()
    for rank in range(k):
        ix = int(x[rank].item())
        iy = int(y[rank].item())
        dx = offset[0, iy, ix].float()
        dy = offset[1, iy, ix].float()
        if torch.isfinite(dx) and torch.isfinite(dy):
            x[rank] += dx.clamp(-1, 1)
            y[rank] += dy.clamp(-1, 1)
    positions = pixel_to_phys(torch.stack([x, y], dim=1)).detach().cpu().numpy()
    # Preserve the original fourth-chapter float32 distance arithmetic.  The
    # summary stage promotes the collected scalars to float64 afterwards.
    return positions.astype(np.float32, copy=False), scores.detach().cpu().numpy().astype(np.float32, copy=False)


def distance_matrix(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    return np.linalg.norm(true[:, None, :] - predicted[None, :, :], axis=2)


def matched_distances(true: np.ndarray, predicted: np.ndarray) -> list[tuple[int, int, float]]:
    if len(true) == 0 or len(predicted) == 0:
        return []
    matrix = distance_matrix(true, predicted)
    truth_idx, pred_idx = linear_sum_assignment(matrix)
    return [
        (int(t), int(p), float(matrix[t, p]))
        for t, p in zip(truth_idx, pred_idx, strict=True)
    ]


def maximum_matches_within(true: np.ndarray, predicted: np.ndarray, threshold: float) -> int:
    if len(true) == 0 or len(predicted) == 0:
        return 0
    valid = distance_matrix(true, predicted) <= threshold
    truth_idx, pred_idx = linear_sum_assignment((~valid).astype(np.int8))
    return int(np.sum(valid[truth_idx, pred_idx]))


def gospa_sample(true: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    n_true, n_pred = len(true), len(predicted)
    matched = min(n_true, n_pred)
    localization_sum = 0.0
    if matched:
        matrix = np.minimum(distance_matrix(true, predicted), GOSPA_C) ** GOSPA_P
        truth_idx, pred_idx = linear_sum_assignment(matrix)
        localization_sum = float(matrix[truth_idx, pred_idx].sum())
    missed = max(n_true - n_pred, 0)
    false = max(n_pred - n_true, 0)
    unit = GOSPA_C**GOSPA_P / GOSPA_ALPHA
    missed_sum = missed * unit
    false_sum = false * unit
    total_sum = localization_sum + missed_sum + false_sum
    return {
        "value_m": float(total_sum ** (1.0 / GOSPA_P)),
        "localization_p_sum": localization_sum,
        "missed_p_sum": missed_sum,
        "false_p_sum": false_sum,
        "missed_count": missed,
        "false_count": false,
    }


def summarize_track(samples: list[dict[str, Any]], matched_errors: list[float]) -> dict[str, Any]:
    gospa_values = np.asarray([sample["gospa_m"] for sample in samples], dtype=np.float64)
    exact_count = np.asarray([sample["true_count"] == sample["predicted_count"] for sample in samples])
    output_counts = np.asarray([sample["predicted_count"] for sample in samples], dtype=np.int64)
    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "true_source_count": int(sum(sample["true_count"] for sample in samples)),
        "predicted_source_count": int(output_counts.sum()),
        "exact_count_rate": float(exact_count.mean()),
        "empty_output_rate": float(np.mean(output_counts == 0)),
        "output_count": numeric_summary(output_counts),
        "gospa": numeric_summary(gospa_values),
        "gospa_components_mean_p_sum": {
            "localization": float(np.mean([sample["gospa_localization_p_sum"] for sample in samples])),
            "missed": float(np.mean([sample["gospa_missed_p_sum"] for sample in samples])),
            "false": float(np.mean([sample["gospa_false_p_sum"] for sample in samples])),
        },
        "matched_pair_count": len(matched_errors),
        "matched_pair_coverage_of_true": len(matched_errors) / max(sum(sample["true_count"] for sample in samples), 1),
    }
    if matched_errors:
        errors = np.asarray(matched_errors, dtype=np.float64)
        summary["matched_errors_m"] = {
            "count": int(errors.size),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mean": float(errors.mean()),
            "median": float(np.median(errors)),
            "p90": float(np.percentile(errors, 90)),
            "p95": float(np.percentile(errors, 95)),
            "max": float(errors.max()),
        }
    threshold_metrics = {}
    for threshold in (10.0, 30.0, 50.0, 100.0):
        tp = sum(sample[f"tp_at_{int(threshold)}m"] for sample in samples)
        true_total = sum(sample["true_count"] for sample in samples)
        pred_total = sum(sample["predicted_count"] for sample in samples)
        fp = pred_total - tp
        fn = true_total - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        threshold_metrics[f"{int(threshold)}m"] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    summary["set_detection"] = threshold_metrics
    return summary


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    device = configure_device(args.device, args.seed)
    if args.fine_source == "exact":
        tasks = load_exact_tasks(args.fine_path, "val")
        empty_band = torch.zeros(len(tasks["n_src"]), dtype=torch.bool)
    else:
        tasks = load_fine_custom(args.fine_path)
        empty_band = tasks["empty_band"].bool()
    n = len(tasks["n_src"])
    require(n == 256, "D8评估任务数不是256")
    require(torch.equal(tasks["sample_idx"], torch.arange(n)), "D8任务sample_idx不是0..255")
    require(bool(torch.all(tasks["group_idx"] == 0)), "D8任务存在非零group_idx")
    if args.k_mode == "oracle":
        requested_count = tasks["n_src"].numpy().astype(np.int64)
    else:
        require(args.prediction_npz is not None, "predicted K必须提供prediction_npz")
        prediction = np.load(args.prediction_npz.resolve())
        requested_count = prediction["k_prediction"].astype(np.int64)
        require(requested_count.shape == (n,), "predicted K shape错误")
        require(np.all((requested_count >= 0) & (requested_count <= CH3_MAX_SRC)), "predicted K越界")

    model, checkpoint = build_d8(args.checkpoint, device)
    samples: list[dict[str, Any]] = []
    matched_error_values: list[float] = []
    batch_size = args.batch_size
    with torch.no_grad():
        for start_index in range(0, n, batch_size):
            stop_index = min(start_index + batch_size, n)
            dpd = tasks["fine_dpd"][start_index:stop_index].float()
            # Keep the reduction order identical to LocDataset.__getitem__.
            # A vectorized per-sample reduction is numerically equivalent, but
            # changes the last few float32 bits and prevents exact S2-G2 replay.
            dpd = torch.stack([
                (sample - sample.mean()) / (sample.std() + 1e-6)
                for sample in dpd
            ]).to(device)
            heatmap, offset = model(dpd)
            require(bool(torch.isfinite(heatmap).all() and torch.isfinite(offset).all()), "D8输出含NaN/Inf")
            for local_index in range(stop_index - start_index):
                sample_index = start_index + local_index
                true_count = int(tasks["n_src"][sample_index].item())
                true_positions = tasks["pos_label"][sample_index, :true_count].numpy() * EDGE
                output_count = int(requested_count[sample_index])
                if bool(empty_band[sample_index].item()):
                    predicted_positions = np.zeros((0, 2), dtype=np.float32)
                    scores = np.zeros(0, dtype=np.float32)
                else:
                    predicted_positions, scores = decode_d8_sample(
                        heatmap[local_index], offset[local_index], output_count
                    )
                matches = matched_distances(true_positions, predicted_positions)
                matched_error_values.extend(distance for _, _, distance in matches)
                gospa = gospa_sample(true_positions, predicted_positions)
                record: dict[str, Any] = {
                    "sample_index": sample_index,
                    "true_count": true_count,
                    "requested_count": output_count,
                    "predicted_count": int(len(predicted_positions)),
                    "empty_band": bool(empty_band[sample_index].item()),
                    "predicted_positions_m": predicted_positions.tolist(),
                    "peak_scores": scores.tolist(),
                    "matched_errors_m": [distance for _, _, distance in matches],
                    "gospa_m": gospa["value_m"],
                    "gospa_localization_p_sum": gospa["localization_p_sum"],
                    "gospa_missed_p_sum": gospa["missed_p_sum"],
                    "gospa_false_p_sum": gospa["false_p_sum"],
                }
                for threshold in (10, 30, 50, 100):
                    record[f"tp_at_{threshold}m"] = maximum_matches_within(
                        true_positions, predicted_positions, float(threshold)
                    )
                samples.append(record)

    overall = summarize_track(samples, matched_error_values)
    stratified = {}
    for count in (2, 3):
        selected = [sample for sample in samples if sample["true_count"] == count]
        selected_errors = [error for sample in selected for error in sample["matched_errors_m"]]
        stratified[f"N{count}"] = summarize_track(selected, selected_errors)
    payload = {
        "status": "PASS",
        "stage": "evaluate_interface_track",
        "track": args.track,
        "fine_source": args.fine_source,
        "k_mode": args.k_mode,
        "checkpoint": file_identity(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "model": "D8 dualhead_std",
        "amp": False,
        "sample_count": n,
        "gospa_config": {"p": GOSPA_P, "c_m": GOSPA_C, "alpha": GOSPA_ALPHA},
        "metrics": overall,
        "stratified": stratified,
        "worst_samples": sorted(samples, key=lambda item: item["gospa_m"], reverse=True)[:10],
        "duration_seconds": time.perf_counter() - started,
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    jsonl_path = args.samples_jsonl.resolve()
    if jsonl_path.exists():
        raise FileExistsError(f"拒绝覆盖已有JSONL: {jsonl_path}")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False, default=json_default) + "\n")
    return payload


def run_representation(args: argparse.Namespace) -> dict[str, Any]:
    exact = load_json(args.exact_eval)
    coarse = load_json(args.coarse_eval)
    coarse_generation = load_json(args.coarse_generate_report)
    s2g2 = load_json(args.s2g2_report)
    expected_rmse = float(s2g2["validation_reload"]["error_statistics"]["overall"]["rmse_m"])
    expected_count = int(s2g2["validation_reload"]["per_source_error_count"])
    exact_rmse = float(exact["metrics"]["matched_errors_m"]["rmse"])
    exact_count = int(exact["metrics"]["matched_pair_count"])
    regression_diff = abs(exact_rmse - expected_rmse)
    regression_pass = regression_diff <= 1e-9 and exact_count == expected_count == 640
    require(regression_pass, f"Exact-Oracle未精确回归S2-G2: diff={regression_diff}")

    coarse_rmse = float(coarse["metrics"]["matched_errors_m"]["rmse"])
    exact_recall = float(exact["metrics"]["set_detection"]["100m"]["recall"])
    coarse_recall = float(coarse["metrics"]["set_detection"]["100m"]["recall"])
    rmse_ratio = coarse_rmse / expected_rmse
    recall_drop = exact_recall - coarse_recall
    checks = {
        "exact_oracle_regression": regression_pass,
        "coarse_oracle_rmse_increase_at_most_10_percent": rmse_ratio <= 1.10,
        "coarse_oracle_recall_100m_drop_at_most_0_05": recall_drop <= 0.05,
        "coarse_oracle_all_samples_present": int(coarse["sample_count"]) == 256,
        "coarse_oracle_no_empty_band": int(coarse_generation["empty_band_count"]) == 0,
    }
    payload = {
        "status": "PASS",
        "stage": "coarse_band_representation_gate",
        "exact_oracle_expected_rmse_m": expected_rmse,
        "exact_oracle_observed_rmse_m": exact_rmse,
        "exact_oracle_absolute_difference_m": regression_diff,
        "coarse_oracle_rmse_m": coarse_rmse,
        "coarse_to_exact_rmse_ratio": rmse_ratio,
        "exact_recall_100m": exact_recall,
        "coarse_recall_100m": coarse_recall,
        "recall_100m_drop": recall_drop,
        "checks": checks,
        "representation_gate_pass": all(checks.values()),
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    audit = load_json(args.audit)
    inference = load_json(args.inference_compare)
    representation = load_json(args.representation)
    mandatory_status = {
        "distribution_contract": audit.get("status") == "PASS",
        "inference_repeatability": inference.get("status") == "PASS",
        "representation_execution": representation.get("status") == "PASS",
    }
    require(all(mandatory_status.values()), f"S2-G3必需阶段未完成: {mandatory_status}")
    model_pass = bool(inference["model_gate_pass"])
    representation_pass = bool(representation["representation_gate_pass"])
    if model_pass and representation_pass:
        scientific_status = "DIRECT_REUSE_CANDIDATE"
        require(args.conditional_tracks, "前置门槛通过时必须提供条件式四轨结果")
        conditional = [load_json(path) for path in args.conditional_tracks]
        required_tracks = {"OB-OK", "OB-PK", "PB-OK", "PB-PK"}
        observed_tracks = {item["track"] for item in conditional}
        require(observed_tracks == required_tracks, f"条件式四轨不完整: {observed_tracks}")
        require(all(item.get("status") == "PASS" for item in conditional), "条件式四轨存在失败")
    elif model_pass and not representation_pass:
        scientific_status = "REQUIRES_INTERFACE_REDESIGN"
        conditional = []
    elif representation_pass and not model_pass:
        scientific_status = "REQUIRES_CH3_ADAPTATION"
        conditional = []
    else:
        scientific_status = "REQUIRES_BOTH_ADAPTATIONS"
        conditional = []

    payload = {
        "status": "PASS",
        "gate": "S2-G3",
        "engineering_status": "PASS",
        "scientific_status": scientific_status,
        "mandatory_status": mandatory_status,
        "representation_gate_pass": representation_pass,
        "model_gate_pass": model_pass,
        "conditional_four_tracks_executed": bool(conditional),
        "conditional_track_summaries": {
            item["track"]: item["metrics"] for item in conditional
        },
        "test_executed": False,
        "new_iq_generated": False,
        "training_or_finetuning_executed": False,
        "threshold_tuned": False,
        "paper_endpoint_performance_claim_allowed": False,
        "decision": {
            "DIRECT_REUSE_CANDIDATE": "后续另行审批统一分布的新validation/test和正式联合实验。",
            "REQUIRES_CH3_ADAPTATION": "离散接口可用，但第三章冻结模型不能直接跨分布；下一步设计共享IQ及第三章适配。",
            "REQUIRES_INTERFACE_REDESIGN": (
                "当前19子带映射与冻结D8不满足直接复用门；下一步需区分接口信息损失与"
                "下游分布失配，并评估连续边界、概率频带、更细表示或接口适配。"
            ),
            "REQUIRES_BOTH_ADAPTATIONS": "离散接口与第三章跨分布能力均不足，不能直接连接现有模型。",
        }[scientific_status],
        "representation_failure_causal_boundary": (
            "当前冻结D8对Coarse-Oracle的退化不能单独区分频带表示的信息损失与下游模型的输入分布失配。"
            if not representation_pass else
            "表示门通过，但仍不构成信息充分性的理论证明。"
        ),
        "inputs": {
            "audit": str(args.audit.resolve()),
            "inference_compare": str(args.inference_compare.resolve()),
            "representation": str(args.representation.resolve()),
        },
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S2-G3跨章可组合性审查")
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="数据分布契约和共享样本身份审计")
    audit.add_argument("--ch3_data_dir", type=Path, required=True)
    audit.add_argument("--ch4_data_dir", type=Path, required=True)
    audit.add_argument("--exact_loc_data_dir", type=Path, required=True)
    audit.add_argument("--ch3_checkpoint", type=Path, required=True)
    audit.add_argument("--ch4_checkpoint", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--markdown", type=Path, required=True)

    inference = sub.add_parser("infer", help="第三章冻结模型跨分布推理")
    inference.add_argument("--coarse_mat", type=Path, required=True)
    inference.add_argument("--raw_mat", type=Path, required=True)
    inference.add_argument("--checkpoint", type=Path, required=True)
    inference.add_argument("--output_npz", type=Path, required=True)
    inference.add_argument("--output_json", type=Path, required=True)
    inference.add_argument("--device", default="cuda:0")
    inference.add_argument("--batch_size", type=int, default=64)
    inference.add_argument("--seed", type=int, default=42)

    compare = sub.add_parser("compare-inference", help="两次独立第三章推理精确比较")
    compare.add_argument("--first_npz", type=Path, required=True)
    compare.add_argument("--second_npz", type=Path, required=True)
    compare.add_argument("--first_json", type=Path, required=True)
    compare.add_argument("--second_json", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    fine = sub.add_parser("generate-fine", help="生成Coarse-Oracle或Predicted-Band细DPD")
    fine.add_argument("--raw_mat", type=Path, required=True)
    fine.add_argument("--band_mode", choices=["coarse_oracle", "predicted"], required=True)
    fine.add_argument("--prediction_npz", type=Path)
    fine.add_argument("--output", type=Path, required=True)
    fine.add_argument("--report", type=Path, required=True)
    fine.add_argument("--device", default="cuda:0")
    fine.add_argument("--chunk_size", type=int, default=40000)
    fine.add_argument("--seed", type=int, default=42)

    evaluate = sub.add_parser("evaluate", help="D8集合定位评估")
    evaluate.add_argument("--track", choices=["OB-OK", "OB-PK", "CO-OK", "PB-OK", "PB-PK"], required=True)
    evaluate.add_argument("--fine_source", choices=["exact", "custom"], required=True)
    evaluate.add_argument("--fine_path", type=Path, required=True)
    evaluate.add_argument("--k_mode", choices=["oracle", "predicted"], required=True)
    evaluate.add_argument("--prediction_npz", type=Path)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--samples_jsonl", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch_size", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=42)

    representation = sub.add_parser("representation", help="Exact/Coarse-Oracle表示门禁")
    representation.add_argument("--exact_eval", type=Path, required=True)
    representation.add_argument("--coarse_eval", type=Path, required=True)
    representation.add_argument("--coarse_generate_report", type=Path, required=True)
    representation.add_argument("--s2g2_report", type=Path, required=True)
    representation.add_argument("--output", type=Path, required=True)

    final = sub.add_parser("finalize", help="机械汇总工程和科研状态")
    final.add_argument("--audit", type=Path, required=True)
    final.add_argument("--inference_compare", type=Path, required=True)
    final.add_argument("--representation", type=Path, required=True)
    final.add_argument("--conditional_tracks", type=Path, nargs="*")
    final.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if hasattr(args, "batch_size"):
        require(args.batch_size > 0, "batch_size必须为正")
    if hasattr(args, "chunk_size"):
        require(args.chunk_size > 0, "chunk_size必须为正")
    if args.command == "audit":
        result = run_audit(args)
    elif args.command == "infer":
        result = run_infer(args)
    elif args.command == "compare-inference":
        result = run_compare_inference(args)
    elif args.command == "generate-fine":
        result = run_generate_fine(args)
    elif args.command == "evaluate":
        result = run_evaluate(args)
    elif args.command == "representation":
        result = run_representation(args)
    elif args.command == "finalize":
        result = run_finalize(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
