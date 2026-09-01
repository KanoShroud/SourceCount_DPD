"""执行 E2E-G0 连续桥、梯度、冻结前向、解码与资源门禁。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import psutil
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
CH3_DIR = PROJECT_ROOT / "第三章代码"
CH4_DIR = PROJECT_ROOT / "第四章代码"
for import_root in (CH4_DIR, CH3_DIR, PROJECT_ROOT):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
for import_root in (PROJECT_ROOT, CH3_DIR, CH4_DIR):
    sys.path.insert(0, str(import_root))

from dpd_calculator_torch import DPDGeometry, compute_fine_dpd  # noqa: E402
from s2g1_train_ch3 import SourceDetectionNet  # noqa: E402
from s2g3_composability import gospa_sample  # noqa: E402
from s2g4_coarse_d8 import build_model as build_d8_model  # noqa: E402
from train_yolo import compute_offset_loss  # noqa: E402
from yolo_model import focal_loss_hm  # noqa: E402
from 统一模型代码.audits.gradient_audit import (  # noqa: E402
    compare_directional_derivatives,
    tensor_summary,
)
from 统一模型代码.models.final_decoder import decode_final_output  # noqa: E402
from 统一模型代码.physics.band_bridge import (  # noqa: E402
    build_subband_fft_matrix,
    continuous_band_bridge,
)
from 统一模型代码.physics.fine_dpd_autograd import (  # noqa: E402
    compute_fine_dpd_autograd,
)
from 统一模型代码.reference_artifacts import verify_artifacts  # noqa: E402
from 统一模型代码.runtime_paths import (  # noqa: E402
    OUTPUT_ROOT,
    REFERENCE_OUTPUT_ROOT,
    reference_output_path,
    validate_output_path,
    validate_split_roots,
)


FS = 100e6
N_FFT = 4096
N_SUB = 19
MAX_TRUE_SRC = 3
CH3_MAX_SRC = 10
FINE_EDGE = 2000.0
FINE_STEP = 10.0
GRID_SIZE = 401
PEAK_SIZE = 9
GAUSS_SIGMA = 2.0
SELECTION_SEED = 20260831
DIRECTION_SEED = 2026083101

COARSE_VALIDATION = reference_output_path(
    "s2g5r4_ch3_scale/20260828_151735/training_views/data_16k/val_data.mat"
)
RAW_VALIDATION = reference_output_path(
    "s2g5r2_ch3/20260827_191207/smoke/chapter4/data/val_data.mat"
)


class ResourceLimitError(RuntimeError):
    """完整图超过预注册资源停止线。"""

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"输入文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def write_json(path: Path, payload: Any) -> None:
    resolved = validate_output_path(path)
    if resolved.exists():
        raise FileExistsError(f"拒绝覆盖已有报告: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> Any:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def code_identity() -> dict[str, Any]:
    tracked = [
        PACKAGE_ROOT / "physics" / "band_bridge.py",
        PACKAGE_ROOT / "physics" / "fine_dpd_autograd.py",
        PACKAGE_ROOT / "models" / "final_decoder.py",
        PACKAGE_ROOT / "audits" / "gradient_audit.py",
        PACKAGE_ROOT / "audits" / "precision_gradient_audit.py",
        Path(__file__).resolve(),
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8:replace"
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
        env=environment,
    )
    return {
        "git_head": completed.stdout.strip(),
        "files": [file_identity(path) for path in tracked],
    }


def _h5_vector(handle: h5py.File, key: str, dtype: Any) -> np.ndarray:
    return np.asarray(handle[key], dtype=dtype).reshape(-1)


def load_selection_metadata() -> dict[str, np.ndarray]:
    with h5py.File(COARSE_VALIDATION.resolve(), "r") as coarse:
        payload = {
            "raw_index": _h5_vector(coarse, "sample_idx_all", np.int64),
            "source_count": _h5_vector(coarse, "src_count_all", np.int64),
            "snr_db": _h5_vector(coarse, "avg_snr_all", np.float64),
            "band_mask": np.asarray(coarse["band_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "fc_hz": np.asarray(coarse["fc_offset_all"], dtype=np.float64).T,
            "bw_hz": np.asarray(coarse["BW_actual_all"], dtype=np.float64).T,
            "symbol_rate_hz": np.asarray(coarse["symbolRate_all"], dtype=np.float64).T,
            "positions_m": np.asarray(coarse["src_pos_all"], dtype=np.float64).transpose(2, 1, 0),
            "sub_lo_hz": _h5_vector(coarse, "sub_f_lo_val", np.float64),
            "sub_hi_hz": _h5_vector(coarse, "sub_f_hi_val", np.float64),
        }
    with h5py.File(RAW_VALIDATION.resolve(), "r") as raw:
        raw_count = _h5_vector(raw, "src_count_all", np.int64)
        raw_snr = _h5_vector(raw, "avg_snr_all", np.float64)
        raw_fc = np.asarray(raw["fc_offset_all"], dtype=np.float64).T
        raw_bw = np.asarray(raw["BW_actual_all"], dtype=np.float64).T
        raw_positions = np.asarray(raw["src_pos_all"], dtype=np.float64).transpose(2, 1, 0)
    indices = payload["raw_index"]
    require(np.array_equal(payload["source_count"], raw_count[indices]), "coarse/raw K不一致")
    require(np.array_equal(payload["snr_db"], raw_snr[indices]), "coarse/raw SNR不一致")
    require(np.array_equal(payload["fc_hz"], raw_fc[indices]), "coarse/raw fc不一致")
    require(np.array_equal(payload["bw_hz"], raw_bw[indices]), "coarse/raw BW不一致")
    require(np.array_equal(payload["positions_m"], raw_positions[indices]), "coarse/raw位置不一致")
    return payload


def _interval_overlap(fc: np.ndarray, bw: np.ndarray, count: int) -> float:
    overlap = 0.0
    for left in range(count):
        for right in range(left + 1, count):
            lo = max(fc[left] - bw[left] / 2, fc[right] - bw[right] / 2)
            hi = min(fc[left] + bw[left] / 2, fc[right] + bw[right] / 2)
            overlap += max(0.0, float(hi - lo))
    return overlap


def _minimum_source_distance(positions: np.ndarray, count: int) -> float | None:
    if count < 2:
        return None
    values = []
    for left in range(count):
        for right in range(left + 1, count):
            values.append(float(np.linalg.norm(positions[left] - positions[right])))
    return min(values)


def selection_rows(metadata: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = metadata["source_count"]
    for local_index, count_value in enumerate(counts):
        count = int(count_value)
        fc = metadata["fc_hz"][local_index]
        bw = metadata["bw_hz"][local_index]
        positions = metadata["positions_m"][local_index]
        active_fc = fc[:count]
        active_bw = bw[:count]
        if count:
            edge_margin = float(
                np.min(FS / 2 - np.abs(active_fc) - active_bw / 2)
            )
            total_bw = float(active_bw.sum())
        else:
            edge_margin = FS / 2
            total_bw = 0.0
        union_bins = int(metadata["band_mask"][local_index, :count].any(axis=0).sum())
        rows.append(
            {
                "local_index": int(local_index),
                "raw_index": int(metadata["raw_index"][local_index]),
                "true_k": count,
                "snr_db": float(metadata["snr_db"][local_index]),
                "total_bw_hz": total_bw,
                "edge_margin_hz": edge_margin,
                "overlap_hz": _interval_overlap(fc, bw, count),
                "min_source_distance_m": _minimum_source_distance(positions, count),
                "hard_subband_count": union_bins,
            }
        )
    return rows


def choose_pilot(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for count in range(4):
        candidates = [row for row in rows if row["true_k"] == count]
        require(len(candidates) >= 4, f"K={count}不足4条validation样本")
        if count == 0:
            specs = [
                ("lowest_snr", "snr_db", False),
                ("highest_snr", "snr_db", True),
                ("lower_mid_snr", "snr_db", False),
                ("upper_mid_snr", "snr_db", True),
            ]
        elif count == 1:
            specs = [
                ("lowest_snr", "snr_db", False),
                ("narrowest_total_bw", "total_bw_hz", False),
                ("closest_band_edge", "edge_margin_hz", False),
                ("highest_snr", "snr_db", True),
            ]
        else:
            specs = [
                ("lowest_snr", "snr_db", False),
                ("largest_frequency_overlap", "overlap_hz", True),
                ("closest_sources", "min_source_distance_m", False),
                ("widest_hard_support", "hard_subband_count", True),
            ]
        used: set[int] = set()
        for reason, key, reverse in specs:
            ordered = sorted(
                candidates,
                key=lambda row: (row[key], -row["local_index"])
                if reverse
                else (row[key], row["local_index"]),
                reverse=reverse,
            )
            selected = next(
                (row for row in ordered if row["local_index"] not in used), None
            )
            if selected is None:
                selected = next(
                    row for row in candidates if row["local_index"] not in used
                )
            used.add(selected["local_index"])
            chosen.append({**selected, "selection_reason": reason})
    require(len(chosen) == 16 and len({row["local_index"] for row in chosen}) == 16, "pilot选择不唯一")
    return chosen


def environment_report() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available()
        else None,
        "system_ram_bytes": int(memory.total),
    }


def prepare_run(run_id: str, precision_contract: str = "legacy_fp32") -> Path:
    validate_split_roots(require_reference=True)
    require(
        precision_contract in {"legacy_fp32", "mixed_fp64_physics"},
        f"未知精度合同: {precision_contract}",
    )
    output_family = "e2e_g0r1" if precision_contract == "mixed_fp64_physics" else "e2e_g0"
    gate_name = "E2E-G0-R1" if precision_contract == "mixed_fp64_physics" else "E2E-G0"
    run_root = validate_output_path(
        OUTPUT_ROOT / "smoke" / "unified" / output_family / Path(run_id).name
    )
    if run_root.exists():
        raise FileExistsError(f"拒绝复用G0目录: {run_root}")
    run_root.mkdir(parents=True)
    metadata = load_selection_metadata()
    pilot = choose_pilot(selection_rows(metadata))
    k3 = [row for row in pilot if row["true_k"] == 3]
    resource_sample = max(
        k3,
        key=lambda row: (
            row["hard_subband_count"], row["total_bw_hz"], -row["local_index"]
        ),
    )
    artifact_records = verify_artifacts(["ch3_seed42", "d8_seed42"])
    manifest = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "source_type": "local_frozen_validation_and_checkpoints",
            "read_status": "IDENTITY_VERIFIED",
            "verification_status": "UNVERIFIED",
        },
        "status": "PREPARED",
        "gate": gate_name,
        "run_id": Path(run_id).name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_output_root": str(REFERENCE_OUTPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "reference_read_only": True,
        "test_executed": False,
        "training_executed": False,
        "environment": environment_report(),
        "code": code_identity(),
        "inputs": {
            "coarse_validation": file_identity(COARSE_VALIDATION),
            "raw_validation": file_identity(RAW_VALIDATION),
            "artifacts": artifact_records,
        },
        "selection_seed": SELECTION_SEED,
        "direction_seed": DIRECTION_SEED,
        "precision_contract": {
            "name": precision_contract,
            "ch3": "float32",
            "bridge_probabilities": "float32",
            "frequency_mapping_and_physics": (
                "float64/complex128"
                if precision_contract == "mixed_fp64_physics"
                else "float32/complex64"
            ),
            "log1p_and_zscore": (
                "float64" if precision_contract == "mixed_fp64_physics" else "float32"
            ),
            "d8_input_and_network": "float32",
            "loss": "float32",
            "gradient_audit": (
                "layered_fp64_and_full_double_spotcheck"
                if precision_contract == "mixed_fp64_physics"
                else "legacy_full_fp32_finite_difference"
            ),
        },
        "pilot_samples": pilot,
        "backward_samples": {
            str(count): next(
                row for row in pilot if row["true_k"] == count
            )["local_index"]
            for count in range(4)
        },
        "resource_sample_local_index": resource_sample["local_index"],
        "thresholds": {
            "equivalence_atol_floor": 1e-6,
            "equivalence_rtol_floor": 1e-5,
            "gradient_relative_tolerance": 0.05,
            "gpu_limit_bytes": int(14.0 * 1024**3),
            "ram_limit_bytes": int(28.0 * 1024**3),
            "sample_wall_limit_seconds": 300.0,
        },
    }
    write_json(run_root / "manifest.json", manifest)
    print(json.dumps({"status": "PREPARED", "run_root": str(run_root)}, ensure_ascii=False))
    return run_root


def verify_manifest(run_root: Path) -> dict[str, Any]:
    manifest = load_json(run_root / "manifest.json")
    require(manifest.get("status") == "PREPARED", "manifest状态不是PREPARED")
    require(file_identity(COARSE_VALIDATION) == manifest["inputs"]["coarse_validation"], "coarse validation身份变化")
    require(file_identity(RAW_VALIDATION) == manifest["inputs"]["raw_validation"], "raw validation身份变化")
    current_artifacts = verify_artifacts(["ch3_seed42", "d8_seed42"])
    require(current_artifacts == manifest["inputs"]["artifacts"], "冻结checkpoint身份变化")
    current_code = code_identity()
    require(current_code == manifest["code"], "G0代码在prepare后发生变化")
    return manifest


def load_selected_arrays(
    manifest: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    selected = sorted(
        manifest["pilot_samples"], key=lambda row: row["local_index"]
    )
    result: dict[int, dict[str, Any]] = {}
    with h5py.File(COARSE_VALIDATION.resolve(), "r") as coarse, h5py.File(
        RAW_VALIDATION.resolve(), "r"
    ) as raw:
        for row in selected:
            local_index = int(row["local_index"])
            raw_index = int(row["raw_index"])
            spectrum = np.asarray(
                coarse["mtr_sub_all"][:, :, :, local_index], dtype=np.float32
            ).transpose(2, 1, 0)
            spectrum = np.log(spectrum + 1.0)
            spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-6)
            real = np.asarray(
                raw["sig_rcv_real_all"][:, :, raw_index], dtype=np.float32
            ).T
            imag = np.asarray(
                raw["sig_rcv_imag_all"][:, :, raw_index], dtype=np.float32
            ).T
            count = int(np.asarray(coarse["src_count_all"][:, local_index]).item())
            result[local_index] = {
                "local_index": local_index,
                "raw_index": raw_index,
                "true_k": count,
                "coarse_dpd": torch.from_numpy(spectrum.copy()),
                "signal": real + 1j * imag,
                "band_truth": np.asarray(
                    coarse["band_mask_all"][:, :, local_index], dtype=np.float32
                ).T,
                "positions_m": np.asarray(
                    coarse["src_pos_all"][:, :, local_index], dtype=np.float32
                ).T,
                "snr_db": float(np.asarray(coarse["avg_snr_all"][:, local_index]).item()),
            }
        sub_lo = _h5_vector(coarse, "sub_f_lo_val", np.float64)
        sub_hi = _h5_vector(coarse, "sub_f_hi_val", np.float64)
    result[-1] = {"sub_lo_hz": sub_lo, "sub_hi_hz": sub_hi}
    return result


def build_models(device: torch.device) -> tuple[SourceDetectionNet, torch.nn.Module, dict[str, Any]]:
    artifact_records = verify_artifacts(["ch3_seed42", "d8_seed42"])
    paths = {row["name"]: Path(row["path"]) for row in artifact_records}
    checkpoint = torch.load(
        paths["ch3_seed42"], map_location=device, weights_only=False
    )
    config = checkpoint.get("config")
    require(isinstance(config, dict), "CH3 checkpoint缺少config")
    require(int(config.get("n_sub", -1)) == N_SUB, "CH3 n_sub错误")
    require(int(config.get("max_src", -1)) == CH3_MAX_SRC, "CH3 max_src错误")
    ch3 = SourceDetectionNet(
        n_sub=N_SUB, max_src=CH3_MAX_SRC, mode=str(config.get("mode"))
    ).to(device)
    ch3.load_state_dict(checkpoint["model"], strict=True)
    ch3.eval()
    for parameter in ch3.parameters():
        parameter.requires_grad_(False)
    d8, d8_checkpoint = build_d8_model(paths["d8_seed42"], device)
    for parameter in d8.parameters():
        parameter.requires_grad_(False)
    return ch3, d8, {
        "ch3_epoch": int(checkpoint.get("epoch", -1)),
        "ch3_threshold": float(config.get("threshold", 0.5)),
        "d8_epoch": int(d8_checkpoint.get("epoch", -1)),
        "d8_method": d8_checkpoint.get("method"),
        "d8_save_tag": d8_checkpoint.get("save_tag"),
    }


def infer_ch3(
    ch3: SourceDetectionNet,
    samples: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[int, torch.Tensor]:
    local_indices = [int(row["local_index"]) for row in manifest["pilot_samples"]]
    batch = torch.stack([samples[index]["coarse_dpd"] for index in local_indices]).to(device)
    with torch.no_grad():
        logits = ch3(batch)
    require(bool(torch.isfinite(logits).all()), "CH3 logits含NaN/Inf")
    return {
        local_index: logits[row].detach().clone()
        for row, local_index in enumerate(local_indices)
    }


def receiver_geometry(device: torch.device, *, edge: float, step: float) -> DPDGeometry:
    angles = np.arange(4) * 2 * np.pi / 4
    receivers = np.stack(
        [500.0 * np.cos(angles), 500.0 * np.sin(angles)], axis=1
    )
    return DPDGeometry(receivers, [0.0, 0.0], edge, step, FS, N_FFT, device)


def d8_input(dpd: torch.Tensor) -> torch.Tensor:
    transformed = torch.log1p(dpd)
    return (transformed - transformed.mean()) / (transformed.std() + 1e-6)


def _uses_mixed_fp64_physics(args: argparse.Namespace) -> bool:
    return getattr(args, "precision_contract", "legacy_fp32") == "mixed_fp64_physics"


def d8_model_input(dpd: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """在FP64归一化完成后，按合同转回冻结D8使用的FP32。"""
    normalized = d8_input(dpd)
    return normalized.to(torch.float32) if _uses_mixed_fp64_physics(args) else normalized


def gaussian_target(
    positions_m: np.ndarray, count: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.arange(GRID_SIZE, dtype=torch.float32, device=device)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32, device=device)
    sorted_positions = positions_m[:count]
    if count:
        order = np.argsort(np.linalg.norm(sorted_positions, axis=1))
        sorted_positions = sorted_positions[order]
    for position in sorted_positions:
        px = (float(position[0]) + FINE_EDGE) / FINE_STEP
        py = (float(position[1]) + FINE_EDGE) / FINE_STEP
        current = torch.exp(
            -((rows - py).square() + (columns - px).square())
            / (2.0 * GAUSS_SIGMA**2)
        )
        target = torch.maximum(target, current)
    if bool(target.max() > 0):
        target = target / target.max()
    pos_label = torch.zeros((1, MAX_TRUE_SRC, 2), dtype=torch.float32)
    if count:
        pos_label[0, :count] = torch.from_numpy(
            np.asarray(sorted_positions, dtype=np.float32) / FINE_EDGE
        )
    n_src = torch.tensor([count], dtype=torch.long)
    return target[None, None], pos_label, n_src


@dataclass
class ResourceGuard:
    device: torch.device
    gpu_limit_bytes: int
    ram_limit_bytes: int
    wall_limit_seconds: float

    def __post_init__(self) -> None:
        self.started = time.perf_counter()
        self.process = psutil.Process()
        self.peak_rss = 0
        self.peak_gpu_allocated = 0
        self.peak_gpu_reserved = 0
        self.calls = 0

    def __call__(self, completed: int, total: int) -> None:
        del completed, total
        self.calls += 1
        memory = self.process.memory_info()
        peak_rss = int(getattr(memory, "peak_wset", memory.rss))
        self.peak_rss = max(self.peak_rss, peak_rss, int(memory.rss))
        if self.device.type == "cuda":
            self.peak_gpu_allocated = max(
                self.peak_gpu_allocated,
                int(torch.cuda.max_memory_allocated(self.device)),
            )
            self.peak_gpu_reserved = max(
                self.peak_gpu_reserved,
                int(torch.cuda.max_memory_reserved(self.device)),
            )
        elapsed = time.perf_counter() - self.started
        if self.peak_rss > self.ram_limit_bytes:
            raise ResourceLimitError(
                f"RAM峰值{self.peak_rss}超过{self.ram_limit_bytes}",
                self.snapshot(),
            )
        if self.peak_gpu_reserved > self.gpu_limit_bytes:
            raise ResourceLimitError(
                f"VRAM reserved峰值{self.peak_gpu_reserved}超过{self.gpu_limit_bytes}",
                self.snapshot(),
            )
        if elapsed > self.wall_limit_seconds:
            raise ResourceLimitError(
                f"单样本forward+backward已超过{self.wall_limit_seconds}s",
                self.snapshot(),
            )

    def snapshot(self) -> dict[str, Any]:
        return {
            "duration_seconds": time.perf_counter() - self.started,
            "peak_process_rss_bytes": self.peak_rss,
            "peak_gpu_allocated_bytes": self.peak_gpu_allocated,
            "peak_gpu_reserved_bytes": self.peak_gpu_reserved,
            "progress_callback_calls": self.calls,
        }

    def report(self) -> dict[str, Any]:
        self(0, 0)
        return self.snapshot()


def bridge_for_logits(
    logits: torch.Tensor,
    matrix: torch.Tensor,
    *,
    compute_dtype: torch.dtype | None = None,
) -> Any:
    if compute_dtype is not None:
        probabilities = torch.sigmoid(logits)
        return continuous_band_bridge(
            probabilities.to(compute_dtype),
            matrix.to(compute_dtype),
            values_are_logits=False,
            max_count=MAX_TRUE_SRC,
        )
    return continuous_band_bridge(
        logits, matrix, values_are_logits=True, max_count=MAX_TRUE_SRC
    )


def bridge_for_contract(
    logits: torch.Tensor, matrix: torch.Tensor, args: argparse.Namespace
) -> Any:
    compute_dtype = torch.float64 if _uses_mixed_fp64_physics(args) else None
    return bridge_for_logits(logits, matrix, compute_dtype=compute_dtype)


def run_new_dpd(
    signal: np.ndarray,
    geometry: DPDGeometry,
    weights: torch.Tensor,
    fixed_support: torch.Tensor,
    args: argparse.Namespace,
    *,
    use_checkpoint: bool,
    guard: ResourceGuard | None = None,
) -> torch.Tensor:
    real_dtype = torch.float64 if _uses_mixed_fp64_physics(args) else torch.float32
    return compute_fine_dpd_autograd(
        signal,
        geometry,
        weights,
        fixed_support=fixed_support,
        grid_chunk_size=args.grid_chunk,
        frequency_chunk_size=args.frequency_chunk,
        eig_device=args.eig_device,
        use_checkpoint=use_checkpoint,
        progress_callback=guard,
        real_dtype=real_dtype,
    )


def equivalence_report(
    samples: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    matrix: torch.Tensor,
    fixed_support: torch.Tensor,
    d8: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    geometry = receiver_geometry(device, edge=100.0, step=10.0)
    sample_index = int(manifest["backward_samples"]["2"])
    signal = samples[sample_index]["signal"]
    binary_cases = {
        "single_band": torch.nn.functional.one_hot(
            torch.tensor([N_SUB // 2], device=device), N_SUB
        ).to(torch.float32),
        "adjacent_bands": torch.nn.functional.one_hot(
            torch.tensor([N_SUB // 2 - 1, N_SUB // 2], device=device), N_SUB
        ).to(torch.float32),
        "oracle_slots": torch.from_numpy(
            samples[sample_index]["band_truth"]
        ).to(device),
        "all_bands": torch.ones((1, N_SUB), dtype=torch.float32, device=device),
    }
    records = []
    repeat_abs = 0.0
    repeat_rel = 0.0
    for name, probabilities in binary_cases.items():
        padded = torch.zeros(
            (CH3_MAX_SRC, N_SUB), dtype=torch.float32, device=device
        )
        padded[: probabilities.shape[0]] = probabilities
        bridge = continuous_band_bridge(
            padded, matrix, values_are_logits=False, max_count=MAX_TRUE_SRC
        )
        weights = bridge.frequency_weights.detach()
        mask = weights.bool().detach().cpu().numpy()
        legacy_mask_a = compute_fine_dpd(
            signal, geometry, freq_mask=mask, chunk_size=args.legacy_chunk
        )
        legacy_mask_b = compute_fine_dpd(
            signal, geometry, freq_mask=mask, chunk_size=args.legacy_chunk
        )
        legacy_weight = compute_fine_dpd(
            signal,
            geometry,
            freq_weights=weights.detach().cpu().numpy(),
            chunk_size=args.legacy_chunk,
        )
        new_value = run_new_dpd(
            signal,
            geometry,
            weights,
            fixed_support & weights.gt(0),
            args,
            use_checkpoint=False,
        ).detach().cpu()
        legacy_input = d8_model_input(legacy_mask_a.to(device), args)[None, None]
        new_input = d8_model_input(new_value.to(device), args)[None, None]
        with torch.no_grad():
            legacy_heatmap, legacy_offset = d8(legacy_input)
            new_heatmap, new_offset = d8(new_input)
            legacy_decoded = decode_final_output(
                band_probabilities=bridge.probabilities,
                cardinality_distribution=bridge.cardinality_distribution,
                heatmap_logits=legacy_heatmap[0],
                offset=legacy_offset[0],
            )
            new_decoded = decode_final_output(
                band_probabilities=bridge.probabilities,
                cardinality_distribution=bridge.cardinality_distribution,
                heatmap_logits=new_heatmap[0],
                offset=new_offset[0],
            )
        peak_indices_equal = bool(
            torch.equal(legacy_decoded.peak_indices, new_decoded.peak_indices)
        )
        if legacy_decoded.position_set_m.numel() == 0:
            position_max_abs_m = 0.0
        else:
            position_max_abs_m = float(
                (
                    legacy_decoded.position_set_m - new_decoded.position_set_m
                ).abs().max().item()
            )
        current_repeat_abs = float((legacy_mask_a - legacy_mask_b).abs().max().item())
        denominator = legacy_mask_a.abs().clamp_min(1e-12)
        current_repeat_rel = float(
            ((legacy_mask_a - legacy_mask_b).abs() / denominator).max().item()
        )
        repeat_abs = max(repeat_abs, current_repeat_abs)
        repeat_rel = max(repeat_rel, current_repeat_rel)
        records.append(
            {
                "case": name,
                "support_bins": int(mask.sum()),
                "legacy_mask_vs_binary_weight_max_abs": float(
                    (legacy_mask_a - legacy_weight).abs().max().item()
                ),
                "legacy_vs_new_max_abs": float(
                    (legacy_mask_a - new_value).abs().max().item()
                ),
                "legacy_vs_new_max_rel": float(
                    (
                        (legacy_mask_a - new_value).abs()
                        / legacy_mask_a.abs().clamp_min(1e-12)
                    ).max().item()
                ),
                "normalized_input_max_abs": float(
                    (d8_input(legacy_mask_a) - d8_input(new_value)).abs().max().item()
                ),
                "d8_heatmap_max_abs": float(
                    (legacy_heatmap - new_heatmap).abs().max().item()
                ),
                "d8_offset_max_abs": float(
                    (legacy_offset - new_offset).abs().max().item()
                ),
                "decoded_k": legacy_decoded.predicted_k,
                "peak_indices_equal": peak_indices_equal,
                "position_max_abs_m": position_max_abs_m,
            }
        )
    atol = max(1e-6, 5.0 * repeat_abs)
    rtol = max(1e-5, 5.0 * repeat_rel)
    for record in records:
        record["pass"] = bool(
            record["legacy_mask_vs_binary_weight_max_abs"] <= atol
            and (
                record["legacy_vs_new_max_abs"] <= atol
                or record["legacy_vs_new_max_rel"] <= rtol
            )
            and record["peak_indices_equal"]
            and record["position_max_abs_m"] <= 0.01
        )
    return {
        "status": "PASS" if all(row["pass"] for row in records) else "FAIL",
        "grid": [21, 21],
        "sample_local_index": sample_index,
        "legacy_repeat_abs": repeat_abs,
        "legacy_repeat_rel": repeat_rel,
        "support_policy": "binary_nonzero_support_for_formula_equivalence",
        "downstream_policy": "same_hard_peak_indices_and_position_delta_le_0.01m",
        "atol": atol,
        "rtol": rtol,
        "cases": records,
    }


def stability_report(
    samples: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    matrix: torch.Tensor,
    fixed_support: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    geometry = receiver_geometry(device, edge=100.0, step=10.0)
    sample_index = int(manifest["backward_samples"]["0"])
    signal = samples[sample_index]["signal"]
    generator = torch.Generator(device=device).manual_seed(DIRECTION_SEED)
    cases = {
        "all_zero": torch.zeros((CH3_MAX_SRC, N_SUB), device=device),
        "near_zero": torch.full((CH3_MAX_SRC, N_SUB), 1e-8, device=device),
        "single_band": torch.nn.functional.pad(
            torch.ones((1, 1), device=device), (N_SUB // 2, N_SUB - N_SUB // 2 - 1, 0, CH3_MAX_SRC - 1)
        ),
        "adjacent_overlap": torch.nn.functional.pad(
            torch.ones((1, 2), device=device), (N_SUB // 2 - 1, N_SUB - N_SUB // 2 - 1, 0, CH3_MAX_SRC - 1)
        ),
        "all_one": torch.ones((CH3_MAX_SRC, N_SUB), device=device),
        "ordinary_continuous": torch.rand(
            (CH3_MAX_SRC, N_SUB), generator=generator, device=device
        ),
    }
    records = []
    for name, probabilities in cases.items():
        bridge = continuous_band_bridge(
            probabilities, matrix, values_are_logits=False, max_count=MAX_TRUE_SRC
        )
        dpd = run_new_dpd(
            signal,
            geometry,
            bridge.frequency_weights,
            fixed_support,
            args,
            use_checkpoint=False,
        )
        normalized = d8_input(dpd)
        record = {
            "case": name,
            "subband_union": tensor_summary(bridge.subband_union),
            "frequency_weights": tensor_summary(bridge.frequency_weights),
            "dpd": tensor_summary(dpd),
            "normalized_dpd": tensor_summary(normalized),
            "pass": bool(
                torch.isfinite(dpd).all()
                and torch.isfinite(normalized).all()
                and torch.all(dpd >= 0)
            ),
        }
        if name == "all_zero":
            record["all_zero_normalized_exact"] = bool(torch.count_nonzero(normalized) == 0)
            record["pass"] = bool(record["pass"] and record["all_zero_normalized_exact"])
        records.append(record)
    return {
        "status": "PASS" if all(row["pass"] for row in records) else "FAIL",
        "cases": records,
    }


def operator_gradient_report(
    samples: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    logits_by_sample: dict[int, torch.Tensor],
    matrix: torch.Tensor,
    fixed_support: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    geometry = receiver_geometry(device, edge=100.0, step=10.0)
    sample_index = int(manifest["backward_samples"]["2"])
    signal = samples[sample_index]["signal"]
    point = logits_by_sample[sample_index].detach().clone().requires_grad_(True)
    generator = torch.Generator(device=device).manual_seed(DIRECTION_SEED)
    direction = torch.randn(point.shape, generator=generator, device=device)
    direction = direction / torch.linalg.vector_norm(direction)
    bridge = bridge_for_contract(point, matrix, args)
    dpd = run_new_dpd(
        signal,
        geometry,
        bridge.frequency_weights,
        fixed_support,
        args,
        use_checkpoint=False,
    )
    probe = torch.randn(dpd.shape, generator=generator, device=device)
    probe = probe / torch.linalg.vector_norm(probe)
    scalar = torch.sum(dpd * probe)
    gradient = torch.autograd.grad(scalar, point)[0]
    autograd_value = float(torch.sum(gradient * direction).item())

    finite = []
    with torch.no_grad():
        for step in (1e-2, 3e-3, 1e-3):
            values = []
            for sign in (1.0, -1.0):
                candidate = point.detach() + sign * step * direction
                current_bridge = bridge_for_contract(candidate, matrix, args)
                current_dpd = run_new_dpd(
                    signal,
                    geometry,
                    current_bridge.frequency_weights,
                    fixed_support,
                    args,
                    use_checkpoint=False,
                )
                values.append(float(torch.sum(current_dpd * probe).item()))
            finite.append(
                {
                    "step": step,
                    "plus": values[0],
                    "minus": values[1],
                    "directional_derivative": (values[0] - values[1]) / (2 * step),
                }
            )
    comparison = compare_directional_derivatives(
        autograd_value, finite, relative_tolerance=0.05
    )
    return {
        "status": comparison["status"],
        "sample_local_index": sample_index,
        "scalar": float(scalar.detach().item()),
        "gradient": tensor_summary(gradient),
        "directional_comparison": comparison,
    }


def decoder_report(device: torch.device) -> dict[str, Any]:
    records = []
    heatmap = torch.zeros((1, GRID_SIZE, GRID_SIZE), device=device)
    heatmap[0, 20, 10] = 2.0
    heatmap[0, 20, 20] = 2.0
    heatmap[0, 100, 200] = 1.0
    offset = torch.zeros((2, GRID_SIZE, GRID_SIZE), device=device)
    probabilities = torch.zeros((CH3_MAX_SRC, N_SUB), device=device)
    probabilities[0, N_SUB // 2] = 0.9
    for count in range(4):
        cardinality = torch.zeros(4, device=device)
        cardinality[count] = 1.0
        first = decode_final_output(
            band_probabilities=probabilities,
            cardinality_distribution=cardinality,
            heatmap_logits=heatmap,
            offset=offset,
        )
        second = decode_final_output(
            band_probabilities=probabilities,
            cardinality_distribution=cardinality,
            heatmap_logits=heatmap,
            offset=offset,
        )
        deterministic = bool(
            torch.equal(first.band_mask_hard, second.band_mask_hard)
            and torch.equal(first.position_set_m, second.position_set_m)
            and torch.equal(first.scores, second.scores)
        )
        records.append(
            {
                "requested_k": count,
                "predicted_k": first.predicted_k,
                "position_count": len(first.position_set_m),
                "positions_m": first.position_set_m.detach().cpu().tolist(),
                "hard_count": first.hard_count,
                "hard_count_mismatch": first.hard_count_mismatch,
                "deterministic": deterministic,
                "pass": deterministic
                and first.predicted_k == count
                and len(first.position_set_m) == count,
            }
        )
    return {
        "status": "PASS" if all(row["pass"] for row in records) else "FAIL",
        "tie_break": "stable_descending_score_then_flat_index",
        "cases": records,
    }


def system_forward_backward(
    *,
    sample: dict[str, Any],
    logits: torch.Tensor,
    matrix: torch.Tensor,
    fixed_support: torch.Tensor,
    geometry: DPDGeometry,
    d8: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    guard: ResourceGuard,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    variable = logits.detach().clone().requires_grad_(True)
    bridge = bridge_for_contract(variable, matrix, args)
    dpd = run_new_dpd(
        sample["signal"],
        geometry,
        bridge.frequency_weights,
        fixed_support,
        args,
        use_checkpoint=True,
        guard=guard,
    )
    normalized = d8_model_input(dpd, args)[None, None]
    heatmap, offset = d8(normalized)
    target, positions, counts = gaussian_target(
        sample["positions_m"], sample["true_k"], device
    )
    focal = focal_loss_hm(heatmap.float(), target)
    offset_loss = compute_offset_loss(
        offset.float(), positions, counts, device
    )
    loss = focal + offset_loss
    loss.backward()
    guard(geometry.num_grid, geometry.num_grid)
    gradient = variable.grad
    require(gradient is not None, "定位loss未生成band logits梯度")
    return {
        "loss": float(loss.detach().item()),
        "focal_loss": float(focal.detach().item()),
        "offset_loss": float(offset_loss.detach().item()),
        "gradient_tensor": gradient.detach().cpu(),
        "gradient": tensor_summary(gradient),
        "frequency_weights": tensor_summary(bridge.frequency_weights),
        "dpd": tensor_summary(dpd),
        "normalized_dpd": tensor_summary(normalized),
        "heatmap_tensor": heatmap.detach().cpu(),
        "offset_tensor": offset.detach().cpu(),
        "resource": guard.report(),
    }


@torch.no_grad()
def system_loss_no_grad(
    *,
    sample: dict[str, Any],
    logits: torch.Tensor,
    matrix: torch.Tensor,
    fixed_support: torch.Tensor,
    geometry: DPDGeometry,
    d8: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    guard: ResourceGuard,
) -> tuple[float, dict[str, Any]]:
    bridge = bridge_for_contract(logits, matrix, args)
    dpd = run_new_dpd(
        sample["signal"],
        geometry,
        bridge.frequency_weights,
        fixed_support,
        args,
        use_checkpoint=False,
        guard=guard,
    )
    heatmap, offset = d8(d8_model_input(dpd, args)[None, None])
    target, positions, counts = gaussian_target(
        sample["positions_m"], sample["true_k"], device
    )
    loss = focal_loss_hm(heatmap.float(), target) + compute_offset_loss(
        offset.float(), positions, counts, device
    )
    guard(geometry.num_grid, geometry.num_grid)
    return float(loss.item()), guard.report()


def resource_and_system_gradient_report(
    samples: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    logits_by_sample: dict[int, torch.Tensor],
    matrix: torch.Tensor,
    fixed_support: torch.Tensor,
    d8: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = manifest["thresholds"]
    geometry = receiver_geometry(device, edge=FINE_EDGE, step=FINE_STEP)
    resource_index = int(manifest["resource_sample_local_index"])
    resource_runs = []
    first_payload: dict[str, Any] | None = None
    for repeat in range(2):
        guard = ResourceGuard(
            device=device,
            gpu_limit_bytes=int(thresholds["gpu_limit_bytes"]),
            ram_limit_bytes=int(thresholds["ram_limit_bytes"]),
            wall_limit_seconds=float(thresholds["sample_wall_limit_seconds"]),
        )
        payload = system_forward_backward(
            sample=samples[resource_index],
            logits=logits_by_sample[resource_index],
            matrix=matrix,
            fixed_support=fixed_support,
            geometry=geometry,
            d8=d8,
            device=device,
            args=args,
            guard=guard,
        )
        if first_payload is None:
            first_payload = payload
            differences = None
        else:
            differences = {
                "loss_abs": abs(payload["loss"] - first_payload["loss"]),
                "gradient_max_abs": float(
                    (payload["gradient_tensor"] - first_payload["gradient_tensor"])
                    .abs()
                    .max()
                    .item()
                ),
                "heatmap_max_abs": float(
                    (payload["heatmap_tensor"] - first_payload["heatmap_tensor"])
                    .abs()
                    .max()
                    .item()
                ),
                "offset_max_abs": float(
                    (payload["offset_tensor"] - first_payload["offset_tensor"])
                    .abs()
                    .max()
                    .item()
                ),
            }
        resource_runs.append(
            {
                key: value
                for key, value in payload.items()
                if not key.endswith("_tensor")
            }
            | {"repeat": repeat + 1, "differences_from_repeat1": differences}
        )
        del payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    require(first_payload is not None, "资源重复未执行")
    deterministic = resource_runs[1]["differences_from_repeat1"]
    deterministic_pass = all(float(value) <= 1e-6 for value in deterministic.values())
    resource_report = {
        "status": "PASS" if deterministic_pass else "FAIL",
        "sample_local_index": resource_index,
        "grid": [GRID_SIZE, GRID_SIZE],
        "batch_size": 1,
        "frequency_support_bins": int(fixed_support.sum().item()),
        "runs": resource_runs,
        "deterministic_pass": deterministic_pass,
        "thresholds": thresholds,
    }

    gradient_records = {
        "3": {
            "sample_local_index": resource_index,
            "gradient": first_payload["gradient"],
            "loss": first_payload["loss"],
        }
    }
    k2_gradient: torch.Tensor | None = None
    for count in (0, 1, 2):
        sample_index = int(manifest["backward_samples"][str(count)])
        guard = ResourceGuard(
            device=device,
            gpu_limit_bytes=int(thresholds["gpu_limit_bytes"]),
            ram_limit_bytes=int(thresholds["ram_limit_bytes"]),
            wall_limit_seconds=float(thresholds["sample_wall_limit_seconds"]),
        )
        payload = system_forward_backward(
            sample=samples[sample_index],
            logits=logits_by_sample[sample_index],
            matrix=matrix,
            fixed_support=fixed_support,
            geometry=geometry,
            d8=d8,
            device=device,
            args=args,
            guard=guard,
        )
        gradient_records[str(count)] = {
            "sample_local_index": sample_index,
            "gradient": payload["gradient"],
            "loss": payload["loss"],
            "resource": payload["resource"],
        }
        if count == 2:
            k2_gradient = payload["gradient_tensor"].to(device)
        del payload
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    active_nonzero = all(
        int(gradient_records[str(count)]["gradient"]["nonzero_count"]) > 0
        for count in (1, 2, 3)
    )
    all_finite = all(
        int(gradient_records[str(count)]["gradient"]["nonfinite_count"]) == 0
        for count in range(4)
    )
    require(k2_gradient is not None, "K=2系统梯度缺失")
    layered_audit = None
    if _uses_mixed_fp64_physics(args):
        from 统一模型代码.audits.precision_gradient_audit import (
            run_layered_gradient_audit,
        )

        layered_audit = run_layered_gradient_audit(
            samples=samples,
            manifest=manifest,
            logits_by_sample=logits_by_sample,
            matrix=matrix,
            fixed_support=fixed_support,
            d8=d8,
            device=device,
        )
        system_directional = {
            "status": "REPLACED_BY_LAYERED_AUDIT",
            "reason": "FP32完整D8 loss有限差分在G0D中已证实分辨率不足",
        }
        gradient_pass = active_nonzero and all_finite and layered_audit["status"] == "PASS"
    else:
        k2_index = int(manifest["backward_samples"]["2"])
        generator = torch.Generator(device=device).manual_seed(DIRECTION_SEED + 2)
        direction = torch.randn(
            logits_by_sample[k2_index].shape, generator=generator, device=device
        )
        direction = direction / torch.linalg.vector_norm(direction)
        autograd_direction = float(torch.sum(k2_gradient * direction).item())
        finite_records = []
        finite_resources = []
        for step in (1e-2, 3e-3):
            values = []
            for sign in (1.0, -1.0):
                guard = ResourceGuard(
                    device=device,
                    gpu_limit_bytes=int(thresholds["gpu_limit_bytes"]),
                    ram_limit_bytes=int(thresholds["ram_limit_bytes"]),
                    wall_limit_seconds=float(thresholds["sample_wall_limit_seconds"]),
                )
                value, current_resource = system_loss_no_grad(
                    sample=samples[k2_index],
                    logits=logits_by_sample[k2_index] + sign * step * direction,
                    matrix=matrix,
                    fixed_support=fixed_support,
                    geometry=geometry,
                    d8=d8,
                    device=device,
                    args=args,
                    guard=guard,
                )
                values.append(value)
                finite_resources.append(
                    {"step": step, "sign": sign, **current_resource}
                )
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            finite_records.append(
                {
                    "step": step,
                    "plus": values[0],
                    "minus": values[1],
                    "directional_derivative": (values[0] - values[1]) / (2 * step),
                }
            )
        system_directional = compare_directional_derivatives(
            autograd_direction, finite_records, relative_tolerance=0.05
        )
        system_directional["resource_runs"] = finite_resources
        gradient_pass = (
            active_nonzero and all_finite and system_directional["status"] == "PASS"
        )
    gradient_report = {
        "status": "PASS" if gradient_pass else "FAIL",
        "full_resolution_samples": gradient_records,
        "active_k_all_nonzero": active_nonzero,
        "all_k_finite": all_finite,
        "system_directional_comparison": system_directional,
        "layered_gradient_audit": layered_audit,
    }
    return resource_report, gradient_report


def frozen_forward_report(
    samples: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    logits_by_sample: dict[int, torch.Tensor],
    matrix: torch.Tensor,
    fixed_support: torch.Tensor,
    d8: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    geometry = receiver_geometry(device, edge=FINE_EDGE, step=FINE_STEP)
    thresholds = manifest["thresholds"]
    records = []
    for selected in manifest["pilot_samples"]:
        local_index = int(selected["local_index"])
        sample = samples[local_index]
        logits = logits_by_sample[local_index]
        bridge = bridge_for_contract(logits, matrix, args)
        probabilities = bridge.probabilities
        hard_subband_mask = (probabilities >= 0.5).any(dim=0)
        hard_mask = (
            matrix.bool() & hard_subband_mask.to(matrix.device).unsqueeze(0)
        ).any(dim=1)
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        guard = ResourceGuard(
            device=device,
            gpu_limit_bytes=int(thresholds["gpu_limit_bytes"]),
            ram_limit_bytes=int(thresholds["ram_limit_bytes"]),
            wall_limit_seconds=float(thresholds["sample_wall_limit_seconds"]),
        )
        if bool(hard_mask.any()):
            legacy = compute_fine_dpd(
                sample["signal"],
                geometry,
                freq_mask=hard_mask.detach().cpu().numpy(),
                chunk_size=args.legacy_chunk,
            ).to(device)
        else:
            legacy = torch.full(
                (GRID_SIZE, GRID_SIZE), 1e-6, dtype=torch.float32, device=device
            )
        soft = run_new_dpd(
            sample["signal"],
            geometry,
            bridge.frequency_weights,
            fixed_support,
            args,
            use_checkpoint=False,
            guard=guard,
        )
        hard_input = d8_model_input(legacy, args)[None, None]
        soft_input = d8_model_input(soft, args)[None, None]
        with torch.no_grad():
            hard_heatmap, hard_offset = d8(hard_input)
            soft_heatmap, soft_offset = d8(soft_input)
        guard(geometry.num_grid, geometry.num_grid)
        hard_count = min(int((probabilities >= 0.5).any(dim=-1).sum().item()), 3)
        hard_cardinality = torch.zeros(4, device=device)
        hard_cardinality[hard_count] = 1.0
        hard_decoded = decode_final_output(
            band_probabilities=probabilities,
            cardinality_distribution=hard_cardinality,
            heatmap_logits=hard_heatmap[0],
            offset=hard_offset[0],
        )
        soft_decoded = decode_final_output(
            band_probabilities=probabilities,
            cardinality_distribution=bridge.cardinality_distribution,
            heatmap_logits=soft_heatmap[0],
            offset=soft_offset[0],
        )
        true_positions = sample["positions_m"][: sample["true_k"]]
        hard_gospa = gospa_sample(
            true_positions, hard_decoded.position_set_m.detach().cpu().numpy()
        )
        soft_gospa = gospa_sample(
            true_positions, soft_decoded.position_set_m.detach().cpu().numpy()
        )
        hard_flat = hard_input.reshape(-1)
        soft_flat = soft_input.reshape(-1)
        if float(hard_flat.std().item()) > 0 and float(soft_flat.std().item()) > 0:
            correlation = float(torch.corrcoef(torch.stack([hard_flat, soft_flat]))[0, 1].item())
        else:
            correlation = None
        soft_collapse = bool(sample["true_k"] > 0 and soft_input.std() <= 1e-8)
        finite = bool(
            torch.isfinite(soft).all()
            and torch.isfinite(soft_heatmap).all()
            and torch.isfinite(soft_offset).all()
        )
        records.append(
            {
                "local_index": local_index,
                "raw_index": sample["raw_index"],
                "true_k": sample["true_k"],
                "snr_db": sample["snr_db"],
                "hard_count": hard_count,
                "soft_predicted_k": soft_decoded.predicted_k,
                "hard_count_mismatch": soft_decoded.hard_count_mismatch,
                "frequency_weights": tensor_summary(bridge.frequency_weights),
                "hard_dpd": tensor_summary(legacy),
                "soft_dpd": tensor_summary(soft),
                "normalized_correlation": correlation,
                "hard_gospa_m": float(hard_gospa["value_m"]),
                "soft_gospa_m": float(soft_gospa["value_m"]),
                "finite": finite,
                "soft_collapse": soft_collapse,
                "resource": guard.report(),
                "pass": bool(finite and not soft_collapse),
            }
        )
        del legacy, soft, hard_input, soft_input, hard_heatmap, hard_offset, soft_heatmap, soft_offset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {
        "status": "PASS" if all(row["pass"] for row in records) else "FAIL",
        "performance_is_diagnostic_only": True,
        "samples": records,
    }


def execute_run(run_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    run_root = validate_output_path(run_root)
    manifest = verify_manifest(run_root)
    require(
        manifest["precision_contract"]["name"] == args.precision_contract,
        "execute精度合同与manifest不一致",
    )
    require(torch.cuda.is_available(), "G0完整门需要CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.manual_seed(SELECTION_SEED)
    np.random.seed(SELECTION_SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    samples = load_selected_arrays(manifest)
    ch3, d8, model_report = build_models(device)
    logits_by_sample = infer_ch3(ch3, samples, manifest, device)
    sub_lo = torch.from_numpy(samples[-1]["sub_lo_hz"]).to(device)
    sub_hi = torch.from_numpy(samples[-1]["sub_hi_hz"]).to(device)
    matrix = build_subband_fft_matrix(
        sub_lo,
        sub_hi,
        sample_rate_hz=FS,
        n_fft=N_FFT,
        dtype=torch.float32,
        device=device,
    )
    fixed_support = matrix.bool().any(dim=1)
    require(int(fixed_support.sum().item()) == N_FFT, "19子带固定支持未覆盖完整FFT轴")

    equivalence = equivalence_report(
        samples, manifest, matrix, fixed_support, d8, device, args
    )
    write_json(run_root / "operator_equivalence.json", equivalence)
    require(equivalence["status"] == "PASS", "二值等价门失败")

    stability = stability_report(
        samples, manifest, matrix, fixed_support, device, args
    )
    write_json(run_root / "stability_cases.json", stability)
    require(stability["status"] == "PASS", "连续边界稳定性门失败")

    operator_gradient = operator_gradient_report(
        samples,
        manifest,
        logits_by_sample,
        matrix,
        fixed_support,
        device,
        args,
    )
    write_json(run_root / "operator_gradient.json", operator_gradient)
    require(operator_gradient["status"] == "PASS", "算子方向梯度门失败")

    decoder = decoder_report(device)
    write_json(run_root / "decoder_audit.json", decoder)
    require(decoder["status"] == "PASS", "Hard最终解码门失败")

    resource, system_gradient = resource_and_system_gradient_report(
        samples,
        manifest,
        logits_by_sample,
        matrix,
        fixed_support,
        d8,
        device,
        args,
    )
    write_json(run_root / "resource_report.json", resource)
    write_json(run_root / "gradient_audit.json", system_gradient)
    require(resource["status"] == "PASS", "完整资源或确定性门失败")
    require(system_gradient["status"] == "PASS", "完整系统梯度门失败")

    frozen = frozen_forward_report(
        samples,
        manifest,
        logits_by_sample,
        matrix,
        fixed_support,
        d8,
        device,
        args,
    )
    write_json(run_root / "frozen_forward.json", frozen)
    require(frozen["status"] == "PASS", "冻结Soft前向出现非有限或塌缩")

    final = {
        "material_passport": {
            **manifest["material_passport"],
            "verification_status": "VERIFIED",
        },
        "status": "G0_PASS",
        "gate": manifest["gate"],
        "run_root": str(run_root),
        "models": model_report,
        "stages": {
            "operator_equivalence": equivalence["status"],
            "stability": stability["status"],
            "operator_gradient": operator_gradient["status"],
            "decoder": decoder["status"],
            "resource": resource["status"],
            "system_gradient": system_gradient["status"],
            "frozen_forward": frozen["status"],
        },
        "g1_unlocked": True,
        "test_executed": False,
        "training_executed": False,
        "performance_interpretation_allowed": False,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(run_root / "final_report.json", final)
    return final


def failure_status(error: BaseException) -> str:
    if isinstance(error, ResourceLimitError):
        return "STOP_RESOURCE"
    if isinstance(error, torch._C._LinAlgError):
        return "STOP_GRADIENT"
    message = str(error)
    if "等价" in message:
        return "STOP_NUMERIC"
    if "梯度" in message:
        return "STOP_GRADIENT"
    if "K=0" in message or "稳定" in message:
        return "REVISE_K0_CONTRACT"
    return "EXECUTION_ERROR"


def write_failure(run_root: Path, manifest: dict[str, Any] | None, error: BaseException) -> dict[str, Any]:
    payload = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "source_type": "local_frozen_validation_and_checkpoints",
            "read_status": "PARTIAL_EXECUTION",
            "verification_status": "ANALYZED",
        },
        "status": failure_status(error),
        "gate": manifest.get("gate", "E2E-G0") if manifest else "E2E-G0",
        "run_root": str(run_root),
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "g1_unlocked": False,
        "test_executed": False,
        "training_executed": False,
        "performance_interpretation_allowed": False,
        "manifest_loaded": manifest is not None,
        "resource_details": error.details
        if isinstance(error, ResourceLimitError)
        else None,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if isinstance(error, ResourceLimitError):
        resource_target = run_root / "resource_report.json"
        if not resource_target.exists():
            write_json(
                resource_target,
                {
                    "status": "STOP_RESOURCE",
                    "error": str(error),
                    "partial": error.details,
                },
            )
    target = run_root / "final_report.json"
    if not target.exists():
        write_json(target, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="冻结样本和身份manifest")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument(
        "--precision-contract",
        choices=("legacy_fp32", "mixed_fp64_physics"),
        default="legacy_fp32",
    )
    execute = subparsers.add_parser("execute", help="执行全部G0门禁")
    execute.add_argument("--run-root", type=Path, required=True)
    execute.add_argument("--grid-chunk", type=int, default=1024)
    execute.add_argument("--frequency-chunk", type=int, default=128)
    execute.add_argument("--legacy-chunk", type=int, default=40000)
    execute.add_argument("--eig-device", choices=("cpu", "cuda"), default="cpu")
    execute.add_argument(
        "--precision-contract",
        choices=("legacy_fp32", "mixed_fp64_physics"),
        default="legacy_fp32",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        prepare_run(args.run_id, args.precision_contract)
        return 0
    run_root = validate_output_path(args.run_root)
    manifest = None
    try:
        manifest = load_json(run_root / "manifest.json")
        result = execute_run(run_root, args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except BaseException as error:
        result = write_failure(run_root, manifest, error)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
