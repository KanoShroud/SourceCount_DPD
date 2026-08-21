"""Gate 2：第四章 D8 极小数据审计与单步工程闭环。

本入口只读取本次 Gate 2 smoke 目录。``audit`` 模式独立检查 MATLAB
数据、定位分片、坐标/offset/Top-K 契约；``model`` 模式使用已确认的
D8 配置完成 no-AMP 单步，或对 batch 2/4/8 做独立进程容量探测。
它不会调用 ``train_yolo.main``，也不会执行正式训练。
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
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Subset


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_root in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from train_yolo import (  # noqa: E402
    LocDataset,
    collate_fn_hm,
    compute_offset_loss,
)
from yolo_config import EDGE, GRID_SIZE, LAMDA, MAX_SRC, PEAK_SIZE  # noqa: E402
from yolo_model import YOLOv8Loc, focal_loss_hm, nms_heatmap, pixel_to_phys  # noqa: E402


EXPECTED_SAMPLES = {"train": 4, "val": 2, "test": 2}
EXPECTED_COUNTS = {
    "train": {2: 2, 3: 2},
    "val": {2: 1, 3: 1},
    "test": {2: 1, 3: 1},
}
EXPECTED_SEED = 20260821
EXPECTED_LEN = 4096
EXPECTED_RCV = 4
EXPECTED_N_SUB = 19
FS = 100e6
GAUSS_SIGMA = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="第四章 Gate 2：D8 数据契约审计与 no-AMP 单步闭环"
    )
    parser.add_argument("--mode", choices=["mat-audit", "audit", "model"], required=True)
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--loc_data_dir", type=Path, required=True)
    parser.add_argument("--pilot_data_dir", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=EXPECTED_SEED)
    parser.add_argument("--memory_probe_only", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.val_batch_size <= 0:
        parser.error("batch_size 与 val_batch_size 必须为正整数")
    if args.mode == "audit" and args.pilot_data_dir is None:
        parser.error("audit 模式必须提供 --pilot_data_dir")
    return args


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


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


def matlab_string(dataset: h5py.Dataset) -> str:
    values = np.asarray(dataset).reshape(-1)
    if values.dtype.kind in "ui":
        return "".join(chr(int(value)) for value in values if int(value) != 0)
    return "".join(str(value) for value in values).strip()


def scalar(handle: h5py.File, name: str) -> float:
    require(name in handle, f"缺少元数据 {name}")
    return float(np.asarray(handle[name]).reshape(-1)[0])


def independent_frequency_groups(
    centers: np.ndarray, bandwidths: np.ndarray
) -> list[list[int]]:
    """不调用生成器实现，独立按频率区间连通关系分组。"""
    n_src = len(centers)
    lows = centers - bandwidths / 2
    highs = centers + bandwidths / 2
    adjacency = [set() for _ in range(n_src)]
    for left in range(n_src):
        for right in range(left + 1, n_src):
            if lows[left] < highs[right] and lows[right] < highs[left]:
                adjacency[left].add(right)
                adjacency[right].add(left)

    groups: list[list[int]] = []
    unseen = set(range(n_src))
    while unseen:
        root = min(unseen)
        stack = [root]
        component: list[int] = []
        unseen.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in sorted(adjacency[node]):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        groups.append(sorted(component))
    return groups


def read_mat(path: Path) -> dict[str, Any]:
    required = {
        "src_count_all",
        "band_mask_all",
        "ignore_mask_all",
        "fc_offset_all",
        "Pt_W_all",
        "src_pos_all",
        "sig_rcv_real_all",
        "sig_rcv_imag_all",
        "symbolRate_all",
        "BW_actual_all",
        "runtime_mode_val",
        "random_seed_val",
    }
    with h5py.File(path, "r") as handle:
        missing = sorted(required.difference(handle.keys()))
        require(not missing, f"{path.name} 缺少数据集: {missing}")
        arrays = {
            "src_count": np.asarray(handle["src_count_all"]).reshape(-1),
            "band_mask": np.asarray(handle["band_mask_all"]).transpose(2, 1, 0),
            "ignore_mask": np.asarray(handle["ignore_mask_all"]).transpose(2, 1, 0),
            "fc_offset": np.asarray(handle["fc_offset_all"]).T,
            "pt_w": np.asarray(handle["Pt_W_all"]).T,
            "src_pos": np.asarray(handle["src_pos_all"]).transpose(2, 1, 0),
            "sig_real": np.asarray(handle["sig_rcv_real_all"]).transpose(2, 1, 0),
            "sig_imag": np.asarray(handle["sig_rcv_imag_all"]).transpose(2, 1, 0),
            "symbol_rate": np.asarray(handle["symbolRate_all"]).T,
            "bandwidth": np.asarray(handle["BW_actual_all"]).T,
            "runtime_mode": matlab_string(handle["runtime_mode_val"]),
            "random_seed": int(scalar(handle, "random_seed_val")),
        }
    return arrays


def audit_mat_files(data_dir: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report: dict[str, Any] = {"status": "PASS", "splits": {}}
    logical_data: dict[str, dict[str, Any]] = {}
    for split, expected_n in EXPECTED_SAMPLES.items():
        path = data_dir / f"{split}_data.mat"
        require(path.is_file(), f"缺少 {path}")
        data = read_mat(path)
        logical_data[split] = data

        require(data["runtime_mode"] == "smoke", f"{split} runtime_mode={data['runtime_mode']!r}")
        require(data["random_seed"] == EXPECTED_SEED, f"{split} seed={data['random_seed']}")
        require(data["src_count"].shape == (expected_n,), f"{split} src_count shape")
        require(data["sig_real"].shape == (expected_n, EXPECTED_RCV, EXPECTED_LEN), f"{split} IQ shape")
        require(data["sig_imag"].shape == data["sig_real"].shape, f"{split} IQ complex shape")
        require(data["src_pos"].shape == (expected_n, MAX_SRC, 2), f"{split} src_pos shape")
        require(data["fc_offset"].shape == (expected_n, MAX_SRC), f"{split} fc_offset shape")
        require(data["bandwidth"].shape == (expected_n, MAX_SRC), f"{split} BW shape")
        require(data["band_mask"].shape == (expected_n, MAX_SRC, EXPECTED_N_SUB), f"{split} band shape")
        require(data["ignore_mask"].shape == data["band_mask"].shape, f"{split} ignore shape")

        float32_names = (
            "sig_real", "sig_imag", "src_pos", "fc_offset", "pt_w",
            "symbol_rate", "bandwidth", "band_mask", "ignore_mask",
        )
        for name in float32_names:
            require(data[name].dtype == np.float32, f"{split} {name} dtype={data[name].dtype}")
            require(np.isfinite(data[name]).all(), f"{split} {name} 含 NaN/Inf")
        require(np.isfinite(data["src_count"]).all(), f"{split} src_count 含 NaN/Inf")
        require(not np.logical_and(data["band_mask"] > 0.5, data["ignore_mask"] > 0.5).any(),
                f"{split} band/ignore 重叠")

        counts = Counter(int(value) for value in data["src_count"])
        require(dict(counts) == EXPECTED_COUNTS[split], f"{split} 源数分布={dict(counts)}")
        tasks: list[dict[str, Any]] = []
        for sample_idx, n_value in enumerate(data["src_count"]):
            n_src = int(n_value)
            active = slice(0, n_src)
            inactive = slice(n_src, MAX_SRC)
            for name in ("fc_offset", "pt_w", "symbol_rate", "bandwidth"):
                require(np.isfinite(data[name][sample_idx, active]).all(), f"{split}[{sample_idx}] {name}")
                require(np.count_nonzero(data[name][sample_idx, inactive]) == 0,
                        f"{split}[{sample_idx}] {name} 空槽非零")
            require(np.count_nonzero(data["src_pos"][sample_idx, inactive]) == 0,
                    f"{split}[{sample_idx}] src_pos 空槽非零")
            positions = data["src_pos"][sample_idx, active]
            require(np.isfinite(positions).all(), f"{split}[{sample_idx}] 位置非有限")
            require((np.abs(positions) <= EDGE).all(), f"{split}[{sample_idx}] 位置越界")
            centers = data["fc_offset"][sample_idx, active].astype(np.float64)
            bandwidths = data["bandwidth"][sample_idx, active].astype(np.float64)
            require((bandwidths > 0).all(), f"{split}[{sample_idx}] 带宽非正")
            lows = centers - bandwidths / 2
            highs = centers + bandwidths / 2
            require((lows >= -FS / 2).all() and (highs <= FS / 2).all(),
                    f"{split}[{sample_idx}] 频率越过 Nyquist")
            groups = independent_frequency_groups(centers, bandwidths)
            require(len(groups) == 1 and len(groups[0]) == n_src,
                    f"{split}[{sample_idx}] 独立分组={groups}")
            freq_lo = float(lows.min())
            freq_hi = float(highs.max())
            f_axis = np.arange(-EXPECTED_LEN // 2, EXPECTED_LEN // 2) * (FS / EXPECTED_LEN)
            n_band = int(np.count_nonzero((f_axis >= freq_lo) & (f_axis < freq_hi)))
            require(n_band > 0, f"{split}[{sample_idx}] 带内频点为零")
            tasks.append({
                "sample_idx": sample_idx,
                "group_idx": 0,
                "n_src": n_src,
                "freq_lo_hz": freq_lo,
                "freq_hi_hz": freq_hi,
                "n_band": n_band,
            })

        report["splits"][split] = {
            "path": str(path.resolve()),
            "file_size_bytes": path.stat().st_size,
            "sample_count": expected_n,
            "source_count_distribution": {str(k): int(v) for k, v in sorted(counts.items())},
            "iq_shape": list(data["sig_real"].shape),
            "iq_dtype": str(data["sig_real"].dtype),
            "runtime_mode": data["runtime_mode"],
            "random_seed": data["random_seed"],
            "tasks": tasks,
        }
    return report, logical_data


def load_split_shards(
    root: Path, split: str, expected_tasks: int
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    split_dir = root / split
    index_path = split_dir / f"loc_{split}_index.pt"
    require(index_path.is_file(), f"缺少 {index_path}")
    index = torch.load(index_path, map_location="cpu", weights_only=False)
    require(index["n_total_tasks"] == expected_tasks, f"{split} index task 数")
    require(index["n_shards"] == 1, f"{split} 预期 1 个 shard")
    required = {
        "fine_dpd", "hyp_mask", "gauss_label", "gauss_multi", "pos_label",
        "n_src", "sample_idx", "group_idx",
    }
    chunks: dict[str, list[torch.Tensor]] = {name: [] for name in required}
    for shard_name in index["shard_files"]:
        shard_path = split_dir / shard_name
        require(shard_path.is_file(), f"缺少 {shard_path}")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
        missing = sorted(required.difference(shard.keys()))
        require(not missing, f"{shard_path.name} 缺字段 {missing}")
        for name in required:
            chunks[name].append(shard[name])
    merged = {name: torch.cat(values, dim=0) for name, values in chunks.items()}
    return merged, index


def expected_gaussians(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(GRID_SIZE, dtype=np.float32)
    rows, cols = np.meshgrid(axis, axis, indexing="ij")
    combined = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    multi = np.zeros((MAX_SRC, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for source_idx, position in enumerate(positions):
        px = (float(position[0]) + EDGE) / LAMDA
        py = (float(position[1]) + EDGE) / LAMDA
        gaussian = np.exp(-((rows - py) ** 2 + (cols - px) ** 2) / (2 * GAUSS_SIGMA**2))
        gaussian = gaussian.astype(np.float32)
        combined = np.maximum(combined, gaussian)
        multi[source_idx] = gaussian
    maximum = float(combined.max())
    if maximum > 0:
        combined /= maximum
    return combined, multi


def audit_shards(
    loc_data_dir: Path,
    pilot_data_dir: Path,
    mat_data: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, torch.Tensor]]]:
    report: dict[str, Any] = {"status": "PASS", "splits": {}}
    split_tensors: dict[str, dict[str, torch.Tensor]] = {}
    for split, expected_n in EXPECTED_SAMPLES.items():
        tensors, index = load_split_shards(loc_data_dir, split, expected_n)
        split_tensors[split] = tensors
        require(tuple(tensors["fine_dpd"].shape) == (expected_n, 1, GRID_SIZE, GRID_SIZE), f"{split} fine shape")
        require(tuple(tensors["hyp_mask"].shape) == (expected_n, MAX_SRC, GRID_SIZE, GRID_SIZE), f"{split} hyp shape")
        require(tuple(tensors["gauss_label"].shape) == (expected_n, 1, GRID_SIZE, GRID_SIZE), f"{split} gauss shape")
        require(tuple(tensors["gauss_multi"].shape) == (expected_n, MAX_SRC, GRID_SIZE, GRID_SIZE), f"{split} multi shape")
        require(tuple(tensors["pos_label"].shape) == (expected_n, MAX_SRC, 2), f"{split} pos shape")
        require(tensors["fine_dpd"].dtype == torch.float16, f"{split} fine dtype")
        require(tensors["hyp_mask"].dtype == torch.float16, f"{split} hyp dtype")
        require(tensors["gauss_label"].dtype == torch.float16, f"{split} gauss dtype")
        require(tensors["gauss_multi"].dtype == torch.float16, f"{split} multi dtype")
        require(tensors["pos_label"].dtype == torch.float32, f"{split} pos dtype")
        for name in ("fine_dpd", "hyp_mask", "gauss_label", "gauss_multi", "pos_label"):
            require(bool(torch.isfinite(tensors[name]).all()), f"{split} {name} 含 NaN/Inf")
        require(bool((tensors["fine_dpd"] >= 0).all()), f"{split} fine_dpd 有负数")
        require(bool((tensors["hyp_mask"] >= 0).all() and (tensors["hyp_mask"] <= 1).all()), f"{split} hyp 范围")
        require(bool((tensors["gauss_label"] >= 0).all() and (tensors["gauss_label"] <= 1).all()), f"{split} gauss 范围")
        require(bool((tensors["gauss_multi"] >= 0).all() and (tensors["gauss_multi"] <= 1).all()), f"{split} multi 范围")
        require(bool(tensors["fine_dpd"].flatten(1).std(dim=1).gt(0).all()), f"{split} fine_dpd 恒定")
        require(Counter(tensors["n_src"].tolist()) == Counter(EXPECTED_COUNTS[split]), f"{split} shard 源数")

        max_gauss_diff = 0.0
        max_multi_diff = 0.0
        for task_idx in range(expected_n):
            sample_idx = int(tensors["sample_idx"][task_idx].item())
            group_idx = int(tensors["group_idx"][task_idx].item())
            n_src = int(tensors["n_src"][task_idx].item())
            require(sample_idx == task_idx and group_idx == 0, f"{split}[{task_idx}] index 映射")
            require(n_src == int(mat_data[split]["src_count"][sample_idx]), f"{split}[{task_idx}] n_src")
            positions = mat_data[split]["src_pos"][sample_idx, :n_src]
            positions = positions[np.argsort(np.linalg.norm(positions, axis=1))]
            expected_pos = np.zeros((MAX_SRC, 2), dtype=np.float32)
            expected_pos[:n_src] = positions / EDGE
            pos_diff = float(np.max(np.abs(tensors["pos_label"][task_idx].numpy() - expected_pos)))
            require(pos_diff <= 1e-6, f"{split}[{task_idx}] pos 最大差={pos_diff}")
            require(np.count_nonzero(tensors["pos_label"][task_idx, n_src:].numpy()) == 0,
                    f"{split}[{task_idx}] pos 空槽")
            require(bool(tensors["hyp_mask"][task_idx, :n_src].flatten(1).max(dim=1).values.gt(0.9).all()),
                    f"{split}[{task_idx}] hyp 活跃通道")
            require(bool(tensors["hyp_mask"][task_idx, n_src:].eq(0).all()), f"{split}[{task_idx}] hyp 空通道")
            require(bool(tensors["gauss_multi"][task_idx, n_src:].eq(0).all()), f"{split}[{task_idx}] multi 空通道")

            expected_gauss, expected_multi = expected_gaussians(positions)
            stored_gauss = tensors["gauss_label"][task_idx, 0].numpy().astype(np.float32)
            stored_multi = tensors["gauss_multi"][task_idx].numpy().astype(np.float32)
            gauss_diff = float(np.max(np.abs(stored_gauss - expected_gauss.astype(np.float16).astype(np.float32))))
            multi_diff = float(np.max(np.abs(stored_multi - expected_multi.astype(np.float16).astype(np.float32))))
            max_gauss_diff = max(max_gauss_diff, gauss_diff)
            max_multi_diff = max(max_multi_diff, multi_diff)
            require(gauss_diff <= 1e-3, f"{split}[{task_idx}] gauss 差={gauss_diff}")
            require(multi_diff <= 1e-3, f"{split}[{task_idx}] multi 差={multi_diff}")

        report["splits"][split] = {
            "n_total_tasks": int(index["n_total_tasks"]),
            "n_shards": int(index["n_shards"]),
            "fine_dpd_shape": list(tensors["fine_dpd"].shape),
            "source_count_distribution": dict(Counter(str(v) for v in tensors["n_src"].tolist())),
            "max_gauss_abs_diff": max_gauss_diff,
            "max_gauss_multi_abs_diff": max_multi_diff,
        }

    pilot_tensors, pilot_index = load_split_shards(pilot_data_dir, "train", 1)
    require(pilot_index["n_total_tasks"] == 1, "pilot 必须恰有 1 个任务")
    full_first = split_tensors["train"]
    pilot_comparison: dict[str, Any] = {}
    for name in ("fine_dpd", "hyp_mask", "gauss_label", "gauss_multi", "pos_label"):
        difference = float(torch.max(torch.abs(pilot_tensors[name][0].float() - full_first[name][0].float())).item())
        pilot_comparison[f"{name}_max_abs_diff"] = difference
        require(difference <= 1e-3, f"pilot/full {name} 差={difference}")
    for name in ("n_src", "sample_idx", "group_idx"):
        require(int(pilot_tensors[name][0].item()) == int(full_first[name][0].item()), f"pilot/full {name}")
    report["pilot_vs_full_train_first"] = pilot_comparison
    return report, split_tensors


def match_errors(predicted: np.ndarray, truth: np.ndarray) -> np.ndarray:
    cost = np.linalg.norm(truth[:, None, :] - predicted[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cost[rows, cols]


def audit_coordinate_contract(split_tensors: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    all_pos = torch.cat([split_tensors[split]["pos_label"] for split in EXPECTED_SAMPLES], dim=0)
    all_n = torch.cat([split_tensors[split]["n_src"] for split in EXPECTED_SAMPLES], dim=0)
    batch = len(all_n)
    oracle_offset = torch.zeros(batch, 2, GRID_SIZE, GRID_SIZE, dtype=torch.float32)
    asymmetric: tuple[int, int, int, int] | None = None
    for batch_idx in range(batch):
        for source_idx in range(int(all_n[batch_idx].item())):
            px = (all_pos[batch_idx, source_idx, 0] * EDGE + EDGE) / LAMDA
            py = (all_pos[batch_idx, source_idx, 1] * EDGE + EDGE) / LAMDA
            ix = int(px.round().item())
            iy = int(py.round().item())
            dx = px - ix
            dy = py - iy
            oracle_offset[batch_idx, 0, iy, ix] = dx
            oracle_offset[batch_idx, 1, iy, ix] = dy
            if asymmetric is None and abs(float((dx - dy).item())) > 1e-3:
                asymmetric = (batch_idx, source_idx, iy, ix)

    zero_loss = compute_offset_loss(oracle_offset, all_pos, all_n, torch.device("cpu"))
    require(float(zero_loss.item()) <= 1e-6, f"oracle offset loss={zero_loss.item()}")
    swapped = oracle_offset.clone()
    if asymmetric is not None:
        batch_idx, _, iy, ix = asymmetric
        dx = swapped[batch_idx, 0, iy, ix].clone()
        swapped[batch_idx, 0, iy, ix] = swapped[batch_idx, 1, iy, ix]
        swapped[batch_idx, 1, iy, ix] = dx
        swapped_loss = compute_offset_loss(swapped, all_pos, all_n, torch.device("cpu"))
        swapped_mode = "real_asymmetric_source"
    else:
        synthetic_pos = torch.tensor([[[0.00075, 0.00175], [0.0, 0.0], [0.0, 0.0]]])
        synthetic_n = torch.tensor([1])
        synthetic = torch.zeros(1, 2, GRID_SIZE, GRID_SIZE)
        px = float((synthetic_pos[0, 0, 0] * EDGE + EDGE) / LAMDA)
        py = float((synthetic_pos[0, 0, 1] * EDGE + EDGE) / LAMDA)
        ix, iy = round(px), round(py)
        synthetic[0, 0, iy, ix] = py - iy
        synthetic[0, 1, iy, ix] = px - ix
        swapped_loss = compute_offset_loss(synthetic, synthetic_pos, synthetic_n, torch.device("cpu"))
        swapped_mode = "synthetic_asymmetric_source"
    require(float(swapped_loss.item()) > 1e-6, f"交换 dx/dy 后 loss={swapped_loss.item()}")

    task_reports: list[dict[str, Any]] = []
    cursor = 0
    observed_k: set[int] = set()
    for split in EXPECTED_SAMPLES:
        tensors = split_tensors[split]
        for task_idx in range(len(tensors["n_src"])):
            n_src = int(tensors["n_src"][task_idx].item())
            observed_k.add(n_src)
            heatmap = tensors["gauss_label"][task_idx:task_idx + 1].float()
            hm_nms = nms_heatmap(heatmap, PEAK_SIZE)
            values, indices = hm_nms[0, 0].reshape(-1).topk(n_src)
            x = (indices % GRID_SIZE).float()
            y = (indices // GRID_SIZE).float()
            for peak_idx in range(n_src):
                ix = int(x[peak_idx].item())
                iy = int(y[peak_idx].item())
                x[peak_idx] += oracle_offset[cursor, 0, iy, ix]
                y[peak_idx] += oracle_offset[cursor, 1, iy, ix]
            predicted = pixel_to_phys(torch.stack([x, y], dim=1)).numpy()
            truth = tensors["pos_label"][task_idx, :n_src].numpy() * EDGE
            errors = match_errors(predicted, truth)
            require(len(predicted) == n_src, f"{split}[{task_idx}] Top-K 数量")
            require(np.isfinite(predicted).all(), f"{split}[{task_idx}] Top-K 非有限")
            require((np.abs(predicted) <= EDGE + 1e-3).all(), f"{split}[{task_idx}] Top-K 越界")
            require(float(errors.max()) <= 1e-2, f"{split}[{task_idx}] oracle 误差={errors.max()}")
            task_reports.append({
                "split": split,
                "task_idx": task_idx,
                "n_src": n_src,
                "topk_scores": values.tolist(),
                "max_match_error_m": float(errors.max()),
            })
            cursor += 1
    require(observed_k == {2, 3}, f"未同时覆盖 K=2/3: {observed_k}")
    return {
        "status": "PASS",
        "oracle_offset_loss": float(zero_loss.item()),
        "swapped_offset_loss": float(swapped_loss.item()),
        "swapped_test_mode": swapped_mode,
        "observed_k": sorted(observed_k),
        "tasks": task_reports,
    }


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    mat_report, mat_data = audit_mat_files(args.data_dir)
    write_json(args.output_dir / "mat_audit.json", mat_report)
    shard_report, split_tensors = audit_shards(
        args.loc_data_dir, args.pilot_data_dir, mat_data
    )
    write_json(args.output_dir / "shard_audit.json", shard_report)
    coordinate_report = audit_coordinate_contract(split_tensors)
    write_json(args.output_dir / "coordinate_contract.json", coordinate_report)
    return {
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "mat_audit": str((args.output_dir / "mat_audit.json").resolve()),
        "shard_audit": str((args.output_dir / "shard_audit.json").resolve()),
        "coordinate_contract": str((args.output_dir / "coordinate_contract.json").resolve()),
    }


def run_mat_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    mat_report, _ = audit_mat_files(args.data_dir)
    report_path = args.output_dir / "mat_audit.json"
    write_json(report_path, mat_report)
    return {
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "mat_audit": str(report_path.resolve()),
    }


def decode_random_diagnostic(
    heatmap_logits: torch.Tensor,
    offsets: torch.Tensor,
    positions: torch.Tensor,
    n_src: torch.Tensor,
) -> dict[str, Any]:
    n_value = int(n_src[0].item())
    hm_nms = nms_heatmap(torch.sigmoid(heatmap_logits[:1]), PEAK_SIZE)
    values, indices = hm_nms[0, 0].reshape(-1).topk(n_value)
    x = (indices % GRID_SIZE).float()
    y = (indices // GRID_SIZE).float()
    for peak_idx in range(n_value):
        ix = int(x[peak_idx].item())
        iy = int(y[peak_idx].item())
        x[peak_idx] += offsets[0, 0, iy, ix].float().clamp(-1, 1)
        y[peak_idx] += offsets[0, 1, iy, ix].float().clamp(-1, 1)
    predicted = pixel_to_phys(torch.stack([x, y], dim=1)).detach().cpu().numpy()
    truth = positions[0, :n_value].numpy() * EDGE
    errors = match_errors(predicted, truth)
    return {
        "n_src": n_value,
        "coordinates_m": predicted.tolist(),
        "scores": values.detach().cpu().tolist(),
        "matching_errors_m": errors.tolist(),
        "performance_interpretation_allowed": False,
    }


def run_model(args: argparse.Namespace) -> dict[str, Any]:
    require(args.device == "cuda:0", "Gate 2 固定使用 cuda:0")
    require(args.seed == EXPECTED_SEED, f"Gate 2 固定 seed={EXPECTED_SEED}")
    require(args.val_batch_size == 1, "Gate 2 固定 val_batch_size=1")
    require(args.batch_size in (1, 2, 4, 8), "Gate 2 batch 仅允许 1/2/4/8")
    require(args.memory_probe_only == (args.batch_size > 1),
            "batch 1 是完整闭环；batch 2/4/8 必须标记 memory_probe_only")
    require(torch.cuda.is_available(), "PyTorch 未检测到 CUDA")
    require(torch.cuda.device_count() == 1, f"预期 1 张 GPU，实际 {torch.cuda.device_count()}")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    rss_samples = [process_rss_bytes()]

    train_set = LocDataset(
        str(args.loc_data_dir), "train", method="dualhead", augment=False
    )
    repeated_indices = [index % len(train_set) for index in range(args.batch_size)]
    train_loader = DataLoader(
        Subset(train_set, repeated_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_hm,
    )
    rss_samples.append(process_rss_bytes())
    dpd, target, positions, n_src = next(iter(train_loader))
    require(tuple(dpd.shape) == (args.batch_size, 1, GRID_SIZE, GRID_SIZE), f"input shape={tuple(dpd.shape)}")
    require(tuple(target.shape) == (args.batch_size, 1, GRID_SIZE, GRID_SIZE), f"target shape={tuple(target.shape)}")

    dpd = dpd.to(device, non_blocking=True)
    target_device = target.to(device, non_blocking=True)
    model = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-3)
    scaler = torch.amp.GradScaler(enabled=False)
    require(not scaler.is_enabled(), "D8 Gate 2 禁止 AMP GradScaler")
    rss_samples.append(process_rss_bytes())

    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    step_started = time.perf_counter()
    with torch.amp.autocast(device_type="cuda", enabled=False):
        pred_heatmap, pred_offset = model(dpd)
    require(tuple(pred_heatmap.shape) == (args.batch_size, 1, GRID_SIZE, GRID_SIZE), f"heatmap shape={tuple(pred_heatmap.shape)}")
    require(tuple(pred_offset.shape) == (args.batch_size, 2, GRID_SIZE, GRID_SIZE), f"offset shape={tuple(pred_offset.shape)}")
    require(bool(torch.isfinite(pred_heatmap).all()), "heatmap 含 NaN/Inf")
    require(bool(torch.isfinite(pred_offset).all()), "offset 含 NaN/Inf")
    focal = focal_loss_hm(pred_heatmap.float(), target_device)
    offset = compute_offset_loss(pred_offset.float(), positions, n_src, device)
    loss = focal + offset
    require(bool(torch.isfinite(focal) and torch.isfinite(offset) and torch.isfinite(loss)), "D8 loss 非有限")
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    require(gradients, "没有生成梯度")
    require(all(bool(torch.isfinite(gradient).all()) for gradient in gradients), "梯度含 NaN/Inf")
    gradient_norm = float(
        torch.sqrt(sum(torch.sum(gradient.detach() ** 2) for gradient in gradients)).item()
    )
    require(np.isfinite(gradient_norm) and gradient_norm > 0, "梯度范数无效")
    changed_parameter = next(
        parameter for parameter in model.parameters()
        if parameter.grad is not None and bool(parameter.grad.detach().abs().max() > 0)
    )
    parameter_before = changed_parameter.detach().clone()
    optimizer.step()
    parameter_max_change = float(
        torch.max(torch.abs(changed_parameter.detach() - parameter_before)).item()
    )
    require(np.isfinite(parameter_max_change) and parameter_max_change > 0, "optimizer 未更新参数")
    require(all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()), "更新后参数非有限")
    torch.cuda.synchronize(device)
    step_elapsed = time.perf_counter() - step_started
    rss_samples.append(process_rss_bytes())

    report: dict[str, Any] = {
        "status": "PASS",
        "model": "D8",
        "method": "dualhead",
        "dice_weight": 0.0,
        "grad_alpha": 1.0,
        "conf_weight_offset": False,
        "soft_conf": False,
        "amp": False,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "memory_probe_only": args.memory_probe_only,
        "repeated_train_indices": repeated_indices,
        "input_shape": list(dpd.shape),
        "heatmap_shape": list(pred_heatmap.shape),
        "offset_shape": list(pred_offset.shape),
        "focal_loss": float(focal.detach().item()),
        "offset_loss": float(offset.detach().item()),
        "total_loss": float(loss.detach().item()),
        "gradient_norm": gradient_norm,
        "parameter_max_change": parameter_max_change,
        "step_elapsed_seconds": step_elapsed,
        "optimizer_state_entries": len(optimizer.state),
    }

    if args.batch_size == 1:
        val_set = LocDataset(
            str(args.loc_data_dir), "val", method="dualhead", augment=False
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn_hm,
        )
        val_dpd, val_target, val_positions, val_n = next(iter(val_loader))
        val_dpd_device = val_dpd.to(device, non_blocking=True)
        val_target_device = val_target.to(device, non_blocking=True)
        model.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
            val_heatmap, val_offset = model(val_dpd_device)
            val_focal = focal_loss_hm(val_heatmap.float(), val_target_device)
            val_offset_loss = compute_offset_loss(
                val_offset.float(), val_positions, val_n, device
            )
            val_loss = val_focal + val_offset_loss
        require(bool(torch.isfinite(val_heatmap).all() and torch.isfinite(val_offset).all()), "验证输出非有限")
        require(bool(torch.isfinite(val_loss)), "验证 loss 非有限")
        diagnostic = decode_random_diagnostic(
            val_heatmap, val_offset, val_positions, val_n
        )

        checkpoint_path = args.output_dir / "gate2_d8_checkpoint.pth"
        checkpoint = {
            "model_state": model.state_dict(),
            "model": "D8",
            "method": "dualhead",
            "dice_weight": 0.0,
            "grad_alpha": 1.0,
            "amp": False,
            "seed": args.seed,
        }
        torch.save(checkpoint, checkpoint_path)
        require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 0, "checkpoint 未保存")
        reloaded = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0).to(device)
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
        reloaded.load_state_dict(loaded["model_state"], strict=True)
        reloaded.eval()
        with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=False):
            reload_heatmap, reload_offset = reloaded(val_dpd_device)
        heatmap_diff = float(torch.max(torch.abs(val_heatmap - reload_heatmap)).item())
        offset_diff = float(torch.max(torch.abs(val_offset - reload_offset)).item())
        require(heatmap_diff <= 1e-6 and offset_diff <= 1e-6,
                f"checkpoint 重载差 heatmap={heatmap_diff}, offset={offset_diff}")
        report.update({
            "val_loss": float(val_loss.item()),
            "val_focal_loss": float(val_focal.item()),
            "val_offset_loss": float(val_offset_loss.item()),
            "random_model_decode_diagnostic": diagnostic,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "reload_heatmap_max_abs_diff": heatmap_diff,
            "reload_offset_max_abs_diff": offset_diff,
        })

    torch.cuda.synchronize(device)
    rss_samples.append(process_rss_bytes())
    report.update({
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "rss_start_bytes": rss_samples[0],
        "rss_peak_observed_bytes": max(rss_samples),
        "rss_end_bytes": rss_samples[-1],
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "gpu_name": torch.cuda.get_device_name(device),
    })
    return report


def main() -> int:
    args = parse_args()
    args.data_dir = args.data_dir.resolve()
    args.loc_data_dir = args.loc_data_dir.resolve()
    if args.pilot_data_dir is not None:
        args.pilot_data_dir = args.pilot_data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    started = time.perf_counter()
    if args.mode == "mat-audit":
        report_path = args.output_dir / "mat_audit_stage.json"
    elif args.mode == "audit":
        report_path = args.output_dir / "gate2_audit_report.json"
    elif args.batch_size == 1:
        report_path = args.output_dir / "batch1_d8_report.json"
    else:
        report_path = args.output_dir / f"batch_{args.batch_size}.json"

    base_report: dict[str, Any] = {
        "gate": "Gate 2 / chapter 4 D8 smoke",
        "mode": args.mode,
        "status": "RUNNING",
        "data_dir": str(args.data_dir),
        "loc_data_dir": str(args.loc_data_dir),
        "pilot_data_dir": str(args.pilot_data_dir) if args.pilot_data_dir else None,
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
        if args.mode == "mat-audit":
            result = run_mat_audit(args)
        elif args.mode == "audit":
            result = run_audit(args)
        else:
            result = run_model(args)
        base_report.update(result)
        base_report["elapsed_seconds"] = time.perf_counter() - started
        write_json(report_path, base_report)
        print(json.dumps(base_report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - 门禁必须保存完整失败证据
        base_report.update({
            "status": "FAIL",
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        write_json(report_path, base_report)
        print(json.dumps(base_report, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
