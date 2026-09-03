"""E2E-G1 Soft-SG/Soft-E2E 配对因果训练与冻结评价。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import numpy as np
import psutil
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CH3_DIR = PROJECT_ROOT / "第三章代码"
CH4_DIR = PROJECT_ROOT / "第四章代码"
for import_root in (PROJECT_ROOT, CH3_DIR, CH4_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from s2g1_train_ch3 import SourceDetectionNet, compute_loss as ch3_band_loss  # noqa: E402
from s2g3_composability import (  # noqa: E402
    gospa_sample,
    matched_distances,
    maximum_matches_within,
    summarize_track,
)
from s2g4_coarse_d8 import build_model as build_d8_model  # noqa: E402
from train_yolo import compute_offset_loss  # noqa: E402
from yolo_model import focal_loss_hm  # noqa: E402

from 统一模型代码.models.final_decoder import decode_final_output  # noqa: E402
from 统一模型代码.physics.band_bridge import (  # noqa: E402
    build_subband_fft_matrix,
    continuous_band_bridge,
)
from 统一模型代码.physics.fine_dpd_autograd import compute_fine_dpd_autograd  # noqa: E402
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
CH3_MAX_SRC = 10
MAX_TRUE_SRC = 3
FINE_EDGE = 2000.0
FINE_STEP = 10.0
GRID_SIZE = 401
PEAK_SIZE = 9
GAUSS_SIGMA = 2.0

CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g1.json"
GATE_NAME = "E2E-G1"
OUTPUT_SUBDIR = "e2e_g1"
EXTRA_CODE_PATHS: list[Path] = []
SOURCE_G1_MANIFEST: Path | None = None
USING_LOCAL_SNAPSHOT = False
COARSE_TRAIN = reference_output_path(
    "s2g5r4_ch3_scale/20260828_151735/coarse_pool/16k/train_coarse_16k.mat"
)
COARSE_VAL_SELECT = reference_output_path(
    "s2g5r2_ch3/20260827_191207/coarse_subsets/val_select.mat"
)
COARSE_VAL_COMPARE = reference_output_path(
    "s2g5r2_ch3/20260827_191207/coarse_subsets/val_compare.mat"
)
RAW_TRAIN_PARTS = (
    (
        0,
        4096,
        reference_output_path(
            "s2g5r2_ch3/20260827_191207/smoke/chapter4/data/train_data.mat"
        ),
    ),
    (
        4096,
        8192,
        reference_output_path(
            "s2g5r2_ch3/20260827_191207/8k_add/smoke/chapter4/data/train_data.mat"
        ),
    ),
    (
        8192,
        16384,
        reference_output_path(
            "s2g5r4_ch3_scale/20260828_151735/16k_add/smoke/chapter4/data/train_data.mat"
        ),
    ),
)
RAW_VALIDATION = reference_output_path(
    "s2g5r2_ch3/20260827_191207/smoke/chapter4/data/val_data.mat"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    resolved = validate_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(resolved)


def append_jsonl(path: Path, payload: Any) -> None:
    resolved = validate_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


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


def state_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def finite(tensor: torch.Tensor) -> bool:
    return bool(torch.isnan(tensor).sum().item() == 0 and torch.isinf(tensor).sum().item() == 0)


def environment_report() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": int(torch.cuda.get_device_properties(0).total_memory)
        if torch.cuda.is_available()
        else None,
        "system_ram_bytes": int(memory.total),
    }


def code_identity() -> dict[str, Any]:
    paths = [
        Path(__file__).resolve(),
        CONFIG_PATH.resolve(),
        PACKAGE_ROOT / "physics" / "band_bridge.py",
        PACKAGE_ROOT / "physics" / "fine_dpd_autograd.py",
        PACKAGE_ROOT / "models" / "final_decoder.py",
        CH3_DIR / "train_v26.py",
        CH3_DIR / "s2g1_train_ch3.py",
        CH4_DIR / "yolo_model.py",
        CH4_DIR / "train_yolo.py",
    ]
    paths.extend(EXTRA_CODE_PATHS)
    return {"files": [file_identity(path) for path in paths]}


def vector(handle: h5py.File, key: str, dtype: Any) -> np.ndarray:
    return np.asarray(handle[key], dtype=dtype).reshape(-1)


def choose_balanced(
    path: Path,
    per_k: int,
    seed: int,
    *,
    split: str,
    balance_lineage: bool = False,
    exclude_local_indices: set[int] | None = None,
) -> list[dict[str, int | str]]:
    rng = np.random.default_rng(seed)
    with h5py.File(path.resolve(), "r") as handle:
        counts = vector(handle, "src_count_all", np.int64)
        raw_indices = vector(handle, "sample_idx_all", np.int64)
    selected: list[dict[str, int | str]] = []
    excluded = exclude_local_indices or set()
    for count in range(4):
        candidates = np.flatnonzero(counts == count)
        if excluded:
            candidates = candidates[~np.isin(candidates, np.fromiter(excluded, dtype=np.int64))]
        if balance_lineage:
            old = candidates[candidates < 8192]
            new = candidates[candidates >= 8192]
            require(per_k % 2 == 0, "lineage均衡要求每K样本数为偶数")
            half = per_k // 2
            require(len(old) >= half and len(new) >= half, f"K={count} lineage样本不足")
            chosen = np.concatenate(
                [rng.choice(old, half, replace=False), rng.choice(new, half, replace=False)]
            )
            rng.shuffle(chosen)
        else:
            require(len(candidates) >= per_k, f"{split} K={count}样本不足")
            chosen = rng.choice(candidates, per_k, replace=False)
        for local_index in chosen.tolist():
            selected.append(
                {
                    "split": split,
                    "local_index": int(local_index),
                    "raw_index": int(raw_indices[local_index]),
                    "true_k": count,
                }
            )
    rng.shuffle(selected)
    return selected


def prepare_run(run_id: str) -> Path:
    if not USING_LOCAL_SNAPSHOT:
        validate_split_roots(require_reference=True)
    config = load_json(CONFIG_PATH)
    run_root = validate_output_path(OUTPUT_ROOT / "unified" / OUTPUT_SUBDIR / Path(run_id).name)
    if run_root.exists():
        raise FileExistsError(f"拒绝复用G1目录: {run_root}")
    run_root.mkdir(parents=True)
    seed = int(config["seed"])
    source_manifest = load_json(SOURCE_G1_MANIFEST) if SOURCE_G1_MANIFEST else None
    if source_manifest is None:
        train = choose_balanced(
            COARSE_TRAIN,
            int(config["train_samples_per_k"]),
            seed,
            split="train",
            balance_lineage=True,
        )
        excluded_select: set[int] = set()
        excluded_compare: set[int] = set()
    else:
        train = source_manifest["subsets"]["train"]
        require(
            len(train) == 4 * int(config["train_samples_per_k"]),
            "源G1训练清单规模与G1-R1配置不一致",
        )
        excluded_select = {
            int(row["local_index"]) for row in source_manifest["subsets"]["val_select"]
        }
        excluded_compare = {
            int(row["local_index"]) for row in source_manifest["subsets"]["val_compare"]
        }
    val_select = choose_balanced(
        COARSE_VAL_SELECT,
        int(config["val_select_samples_per_k"]),
        seed + 1,
        split="val_select",
        exclude_local_indices=excluded_select,
    )
    val_compare = choose_balanced(
        COARSE_VAL_COMPARE,
        int(config["val_compare_samples_per_k"]),
        seed + 2,
        split="val_compare",
        exclude_local_indices=excluded_compare,
    )
    artifacts = verify_artifacts(["ch3_seed42", "d8_seed42"])
    input_paths = [COARSE_TRAIN, COARSE_VAL_SELECT, COARSE_VAL_COMPARE, RAW_VALIDATION]
    input_paths.extend(part[2] for part in RAW_TRAIN_PARTS)
    manifest = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "UNVERIFIED",
        },
        "status": "PREPARED",
        "gate": GATE_NAME,
        "run_id": Path(run_id).name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_output_root": str(REFERENCE_OUTPUT_ROOT),
        "output_root": str(OUTPUT_ROOT),
        "reference_read_only": True,
        "test_executed": False,
        "environment": environment_report(),
        "config": config,
        "code": code_identity(),
        "inputs": {
            "files": [file_identity(path) for path in input_paths],
            "artifacts": artifacts,
        },
        "subsets": {
            "train": train,
            "val_select": val_select,
            "val_compare": val_compare,
        },
        "source_g1_manifest": file_identity(SOURCE_G1_MANIFEST)
        if SOURCE_G1_MANIFEST
        else None,
    }
    write_json(run_root / "manifest.json", manifest)
    print(str(run_root), flush=True)
    return run_root


def verify_manifest(run_root: Path) -> dict[str, Any]:
    manifest = load_json(run_root / "manifest.json")
    require(manifest["gate"] == GATE_NAME, "manifest Gate错误")
    require(manifest["reference_read_only"] is True, "参考根未标记只读")
    require(manifest["test_executed"] is False, "G1禁止读取test")
    require(code_identity() == manifest["code"], "G1代码或配置在prepare后发生变化")
    require(verify_artifacts(["ch3_seed42", "d8_seed42"]) == manifest["inputs"]["artifacts"], "checkpoint身份变化")
    for expected in manifest["inputs"]["files"]:
        require(file_identity(Path(expected["path"])) == expected, f"输入身份变化: {expected['path']}")
    return manifest


class SampleStore:
    def __init__(self, split: str):
        self.split = split
        if split == "train":
            self.coarse_path = COARSE_TRAIN
            self.raw_parts = [(lo, hi, h5py.File(path.resolve(), "r")) for lo, hi, path in RAW_TRAIN_PARTS]
        elif split == "val_select":
            self.coarse_path = COARSE_VAL_SELECT
            self.raw_parts = [(0, 2048, h5py.File(RAW_VALIDATION.resolve(), "r"))]
        elif split == "val_compare":
            self.coarse_path = COARSE_VAL_COMPARE
            self.raw_parts = [(0, 2048, h5py.File(RAW_VALIDATION.resolve(), "r"))]
        else:
            raise ValueError(f"未知split: {split}")
        self.coarse = h5py.File(self.coarse_path.resolve(), "r")

    def close(self) -> None:
        self.coarse.close()
        for _, _, handle in self.raw_parts:
            handle.close()

    def __enter__(self) -> "SampleStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _raw(self, raw_index: int) -> tuple[h5py.File, int]:
        if self.split != "train":
            return self.raw_parts[0][2], raw_index
        for lo, hi, handle in self.raw_parts:
            if lo <= raw_index < hi:
                return handle, raw_index - lo
        raise IndexError(f"训练raw_index越界: {raw_index}")

    def sample(self, record: dict[str, Any]) -> dict[str, Any]:
        local_index = int(record["local_index"])
        raw_index = int(record["raw_index"])
        raw, raw_local = self._raw(raw_index)
        spectrum = np.asarray(
            self.coarse["mtr_sub_all"][:, :, :, local_index], dtype=np.float32
        ).transpose(2, 1, 0)
        spectrum = np.log(spectrum + 1.0)
        spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-6)
        real = np.asarray(raw["sig_rcv_real_all"][:, :, raw_local], dtype=np.float32).T
        imag = np.asarray(raw["sig_rcv_imag_all"][:, :, raw_local], dtype=np.float32).T
        true_k = int(np.asarray(self.coarse["src_count_all"][:, local_index]).item())
        require(true_k == int(record["true_k"]), "manifest与数据K不一致")
        band = np.asarray(
            self.coarse["band_mask_all"][:, :, local_index], dtype=np.float32
        ).T
        ignore = np.asarray(
            self.coarse["ignore_mask_all"][:, :, local_index], dtype=np.float32
        ).T
        band_pad = np.zeros((CH3_MAX_SRC, N_SUB), dtype=np.float32)
        ignore_pad = np.zeros((CH3_MAX_SRC, N_SUB), dtype=np.float32)
        band_pad[:MAX_TRUE_SRC] = band
        ignore_pad[:MAX_TRUE_SRC] = ignore
        positions = np.asarray(
            self.coarse["src_pos_all"][:, :, local_index], dtype=np.float32
        ).T
        return {
            **record,
            "coarse_dpd": torch.from_numpy(spectrum.copy()),
            "signal": real + 1j * imag,
            "band_truth": torch.from_numpy(band_pad),
            "ignore_truth": torch.from_numpy(ignore_pad),
            "positions_m": positions,
        }

    def subband_edges(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            vector(self.coarse, "sub_f_lo_val", np.float64),
            vector(self.coarse, "sub_f_hi_val", np.float64),
        )


def receiver_geometry(device: torch.device) -> Any:
    from dpd_calculator_torch import DPDGeometry

    angles = np.arange(4) * 2 * np.pi / 4
    receivers = np.stack([500.0 * np.cos(angles), 500.0 * np.sin(angles)], axis=1)
    return DPDGeometry(receivers, [0.0, 0.0], FINE_EDGE, FINE_STEP, FS, N_FFT, device)


def build_models(device: torch.device) -> tuple[SourceDetectionNet, torch.nn.Module, dict[str, Any]]:
    artifacts = verify_artifacts(["ch3_seed42", "d8_seed42"])
    paths = {row["name"]: Path(row["path"]) for row in artifacts}
    checkpoint = torch.load(paths["ch3_seed42"], map_location=device, weights_only=False)
    config = checkpoint["config"]
    ch3 = SourceDetectionNet(
        n_sub=int(config["n_sub"]),
        max_src=int(config["max_src"]),
        mode=str(config["mode"]),
    ).to(device)
    ch3.load_state_dict(checkpoint["model"], strict=True)
    d8, d8_checkpoint = build_d8_model(paths["d8_seed42"], device)
    return ch3, d8, {
        "ch3_epoch": int(checkpoint["epoch"]),
        "d8_epoch": int(d8_checkpoint["epoch"]),
        "ch3_sha256": artifacts[0]["sha256"],
        "d8_sha256": artifacts[1]["sha256"],
    }


def configure_trainable(ch3: SourceDetectionNet, d8: torch.nn.Module, training: bool) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    for parameter in ch3.parameters():
        parameter.requires_grad_(False)
    for parameter in d8.parameters():
        parameter.requires_grad_(False)
    ch3_parameters = list(ch3.band_heads.parameters())
    d8_parameters = list(d8.decoder.parameters())
    for parameter in (*ch3_parameters, *d8_parameters):
        parameter.requires_grad_(True)
    ch3.eval()
    d8.eval()
    if training:
        ch3.band_heads.train()
        d8.decoder.train()
    return ch3_parameters, d8_parameters


def trainable_state(ch3: SourceDetectionNet, d8: torch.nn.Module) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "ch3_band_heads": {key: value.detach().cpu() for key, value in ch3.band_heads.state_dict().items()},
        "d8_decoder": {key: value.detach().cpu() for key, value in d8.decoder.state_dict().items()},
    }


def load_trainable_state(ch3: SourceDetectionNet, d8: torch.nn.Module, state: dict[str, Any]) -> None:
    ch3.band_heads.load_state_dict(state["ch3_band_heads"], strict=True)
    d8.decoder.load_state_dict(state["d8_decoder"], strict=True)


def bridge(logits: torch.Tensor, matrix: torch.Tensor, config: dict[str, Any]) -> Any:
    return continuous_band_bridge(
        logits.to(torch.float64),
        matrix.to(torch.float64),
        values_are_logits=True,
        max_count=MAX_TRUE_SRC,
        slot_existence_mode=str(config.get("slot_existence_mode", "noisy_or")),
        slot_temperature=float(config.get("slot_temperature", 1.0)),
    )


def slot_existence_loss(
    logits: torch.Tensor,
    band_truth: torch.Tensor,
    true_k: int,
    config: dict[str, Any],
) -> torch.Tensor:
    """直接监督频率有序槽位是否存在，避免相关子带被当作独立事件。"""
    slot_targets = (band_truth > 0.5).any(dim=-1).to(logits.dtype)
    require(int(slot_targets.sum().item()) == true_k, "槽位存在标签数与真实K不一致")
    slot_logits = logits.amax(dim=-1) / float(config.get("slot_temperature", 1.0))
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        slot_logits, slot_targets, reduction="none"
    )
    positive = slot_targets > 0.5
    negative = ~positive
    if bool(positive.any()) and bool(negative.any()):
        return 0.5 * elementwise[positive].mean() + 0.5 * elementwise[negative].mean()
    return elementwise.mean()


def cardinality_for_decode(
    current_bridge: Any,
    config: dict[str, Any],
) -> torch.Tensor:
    if str(config.get("final_count_decode", "poisson_binomial")) != "hard_slot":
        return current_bridge.cardinality_distribution
    threshold = float(config["band_threshold"])
    hard_count = min(
        int((current_bridge.probabilities >= threshold).any(dim=-1).sum().item()),
        MAX_TRUE_SRC,
    )
    cardinality = torch.zeros(
        MAX_TRUE_SRC + 1,
        dtype=current_bridge.cardinality_distribution.dtype,
        device=current_bridge.cardinality_distribution.device,
    )
    cardinality[hard_count] = 1.0
    return cardinality


def d8_input(dpd: torch.Tensor) -> torch.Tensor:
    transformed = torch.log1p(dpd)
    normalized = (transformed - transformed.mean()) / (transformed.std() + 1e-6)
    return normalized.to(torch.float32)[None, None]


def online_dpd(
    signal: np.ndarray,
    weights: torch.Tensor,
    matrix: torch.Tensor,
    geometry: Any,
    config: dict[str, Any],
    *,
    backward: bool,
) -> torch.Tensor:
    return compute_fine_dpd_autograd(
        signal,
        geometry,
        weights,
        fixed_support=matrix.any(dim=1),
        grid_chunk_size=int(config["grid_chunk"]),
        frequency_chunk_size=int(config["frequency_chunk"]),
        eig_device=str(config["eig_device"]),
        use_checkpoint=backward,
        real_dtype=torch.float64,
    )


def gaussian_target(positions_m: np.ndarray, count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    axis = torch.arange(GRID_SIZE, dtype=torch.float32, device=device)
    rows, columns = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32, device=device)
    active = np.asarray(positions_m[:count], dtype=np.float32)
    if count:
        active = active[np.argsort(np.linalg.norm(active, axis=1))]
    for position in active:
        px = (float(position[0]) + FINE_EDGE) / FINE_STEP
        py = (float(position[1]) + FINE_EDGE) / FINE_STEP
        current = torch.exp(-((rows - py).square() + (columns - px).square()) / (2.0 * GAUSS_SIGMA**2))
        target = torch.maximum(target, current)
    if bool(target.max() > 0):
        target = target / target.max()
    pos_label = torch.zeros((1, MAX_TRUE_SRC, 2), dtype=torch.float32)
    if count:
        pos_label[0, :count] = torch.from_numpy(active / FINE_EDGE)
    return target[None, None], pos_label, torch.tensor([count], dtype=torch.long)


def losses(
    logits: torch.Tensor,
    sample: dict[str, Any],
    matrix: torch.Tensor,
    geometry: Any,
    d8: torch.nn.Module,
    device: torch.device,
    config: dict[str, Any],
    track: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Any, torch.Tensor, torch.Tensor]:
    band_truth = sample["band_truth"].to(device)[None]
    ignore_truth = sample["ignore_truth"].to(device)[None]
    loss_band = ch3_band_loss(logits, band_truth, ignore_truth)
    true_k = int(sample["true_k"])
    own_bridge = bridge(logits[0], matrix, config)
    if str(config.get("existence_loss", "poisson_binomial_nll")) == "slot_bce":
        loss_exist = slot_existence_loss(logits[0], band_truth[0], true_k, config)
    else:
        loss_exist = -torch.log(
            own_bridge.cardinality_distribution[true_k].clamp_min(1e-12)
        ).to(torch.float32)
    bridge_logits = logits[0].detach() if track == "soft_sg" else logits[0]
    loc_bridge = bridge(bridge_logits, matrix, config)
    dpd = online_dpd(
        sample["signal"],
        loc_bridge.frequency_weights,
        matrix,
        geometry,
        config,
        backward=track == "soft_e2e",
    )
    heatmap, offset = d8(d8_input(dpd))
    target, positions, counts = gaussian_target(sample["positions_m"], true_k, device)
    loss_heatmap = focal_loss_hm(heatmap.float(), target)
    loss_offset = compute_offset_loss(offset.float(), positions, counts, device)
    components = {
        "band": loss_band,
        "exist": loss_exist,
        "heatmap": loss_heatmap,
        "offset": loss_offset,
    }
    weights = config["loss_weights"]
    total = sum(float(weights[name]) * value for name, value in components.items())
    return total, components, own_bridge, heatmap, offset


def gradient_vector(parameters: Iterable[torch.nn.Parameter], replacements: list[torch.Tensor] | None = None) -> torch.Tensor:
    pieces = []
    for index, parameter in enumerate(parameters):
        value = replacements[index] if replacements is not None else parameter.grad
        pieces.append((torch.zeros_like(parameter) if value is None else value).reshape(-1))
    return torch.cat(pieces)


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) == 0.0:
        return None
    return float(torch.dot(left, right).div(denominator).item())


def sample_order(records: list[dict[str, Any]], seed: int, epoch: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed + epoch * 1009)
    order = rng.permutation(len(records))
    return [records[int(index)] for index in order]


def ch3_metric_arrays(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from s2g5_r2_ch3 import metrics

    prediction = np.asarray([row["band_prediction"] for row in rows], dtype=bool)
    truth = np.asarray([row["band_truth"] for row in rows], dtype=bool)
    ignore = np.asarray([row["band_ignore"] for row in rows], dtype=bool)
    counts = np.asarray([row["true_count"] for row in rows], dtype=np.int64)
    result = metrics(
        {"prediction": prediction, "truth": truth, "ignore": ignore, "source_count": counts}
    )
    pb_pred = np.asarray([row["predicted_count"] for row in rows], dtype=np.int64)
    result["poisson_binomial_count_accuracy"] = float(np.mean(pb_pred == counts))
    result["poisson_binomial_balanced_count_accuracy"] = float(
        np.mean([np.mean(pb_pred[counts == count] == count) for count in range(4)])
    )
    result["poisson_binomial_confusion_true_0_3_pred_0_3"] = [
        [int(np.sum((counts == truth_k) & (pb_pred == pred_k))) for pred_k in range(4)]
        for truth_k in range(4)
    ]
    return result


@torch.no_grad()
def evaluate_records(
    ch3: SourceDetectionNet,
    d8: torch.nn.Module,
    records: list[dict[str, Any]],
    split: str,
    device: torch.device,
    config: dict[str, Any],
    *,
    mode: str = "soft",
    progress_path: Path | None = None,
) -> dict[str, Any]:
    configure_trainable(ch3, d8, training=False)
    geometry = receiver_geometry(device)
    samples: list[dict[str, Any]] = []
    matched_errors: list[float] = []
    started = time.perf_counter()
    with SampleStore(split) as store:
        lo, hi = store.subband_edges()
        matrix = build_subband_fft_matrix(
            torch.from_numpy(lo), torch.from_numpy(hi), sample_rate_hz=FS, n_fft=N_FFT,
            dtype=torch.float64, device=device,
        )
        for ordinal, record in enumerate(records, start=1):
            sample = store.sample(record)
            logits = ch3(sample["coarse_dpd"].to(device)[None])
            current_bridge = bridge(logits[0], matrix, config)
            if mode == "hard":
                hard_probability = (current_bridge.probabilities >= float(config["band_threshold"])).to(torch.float64)
                current_bridge = continuous_band_bridge(
                    hard_probability, matrix, values_are_logits=False, max_count=MAX_TRUE_SRC
                )
            elif mode == "oracle":
                oracle_probability = sample["band_truth"].to(device, dtype=torch.float64)
                current_bridge = continuous_band_bridge(
                    oracle_probability, matrix, values_are_logits=False, max_count=MAX_TRUE_SRC
                )
            dpd = online_dpd(
                sample["signal"], current_bridge.frequency_weights, matrix, geometry,
                config, backward=False,
            )
            heatmap, offset = d8(d8_input(dpd))
            decoded = decode_final_output(
                band_probabilities=current_bridge.probabilities.to(torch.float32),
                cardinality_distribution=cardinality_for_decode(
                    current_bridge, config
                ).to(torch.float32),
                heatmap_logits=heatmap[0], offset=offset[0],
                band_threshold=float(config["band_threshold"]), max_count=MAX_TRUE_SRC,
                peak_size=PEAK_SIZE, edge_m=FINE_EDGE, grid_step_m=FINE_STEP,
            )
            true_count = int(sample["true_k"])
            true_positions = np.asarray(sample["positions_m"][:true_count], dtype=np.float32)
            predicted_positions = decoded.position_set_m.detach().cpu().numpy().astype(np.float32)
            gospa = gospa_sample(true_positions, predicted_positions)
            matches = matched_distances(true_positions, predicted_positions)
            matched_errors.extend(item[2] for item in matches)
            row = {
                "split": split,
                "local_index": int(record["local_index"]),
                "raw_index": int(record["raw_index"]),
                "true_count": true_count,
                "predicted_count": int(decoded.predicted_k),
                "hard_count": int(decoded.hard_count),
                "hard_count_mismatch": bool(decoded.hard_count_mismatch),
                "band_prediction": decoded.band_mask_hard.detach().cpu().numpy().tolist(),
                "band_truth": (sample["band_truth"].numpy() > 0.5).tolist(),
                "band_ignore": (sample["ignore_truth"].numpy() > 0.5).tolist(),
                "predicted_positions_m": predicted_positions.tolist(),
                "gospa_m": gospa["value_m"],
                "gospa_localization_p_sum": gospa["localization_p_sum"],
                "gospa_missed_p_sum": gospa["missed_p_sum"],
                "gospa_false_p_sum": gospa["false_p_sum"],
            }
            for threshold in (10, 30, 50, 100):
                row[f"tp_at_{threshold}m"] = maximum_matches_within(
                    true_positions, predicted_positions, float(threshold)
                )
            samples.append(row)
            if progress_path is not None:
                append_jsonl(progress_path, {"sample": ordinal, "total": len(records), "gospa_m": row["gospa_m"]})
            if ordinal % 8 == 0 or ordinal == len(records):
                print(f"[{mode}:{split}] {ordinal}/{len(records)}", flush=True)
    return {
        "mode": mode,
        "split": split,
        "sample_count": len(samples),
        "system": summarize_track(samples, matched_errors),
        "ch3": ch3_metric_arrays(samples),
        "samples": samples,
        "duration_seconds": time.perf_counter() - started,
    }


def save_checkpoint(
    path: Path,
    ch3: SourceDetectionNet,
    d8: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    track: str,
    config: dict[str, Any],
) -> None:
    resolved = validate_output_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "gate": GATE_NAME,
            "track": track,
            "epoch": epoch,
            "seed": int(config["seed"]),
            "trainable_state": trainable_state(ch3, d8),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        resolved,
    )


def load_checkpoint(path: Path, ch3: SourceDetectionNet, d8: torch.nn.Module) -> dict[str, Any]:
    payload = torch.load(path.resolve(), map_location="cpu", weights_only=False)
    require(payload["gate"] == GATE_NAME, "checkpoint Gate错误")
    load_trainable_state(ch3, d8, payload["trainable_state"])
    return payload


def semantic_contract(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """G1-R1槽位存在表示的无训练语义合同。"""
    probability = torch.full((CH3_MAX_SRC, N_SUB), 0.01, dtype=torch.float64, device=device)
    logits = torch.logit(probability).requires_grad_(True)
    matrix = torch.ones((8, N_SUB), dtype=torch.float64, device=device)
    corrected = continuous_band_bridge(
        logits,
        matrix,
        values_are_logits=True,
        max_count=MAX_TRUE_SRC,
        slot_existence_mode="max_logit",
        slot_temperature=float(config.get("slot_temperature", 1.0)),
    )
    legacy = continuous_band_bridge(
        logits,
        matrix,
        values_are_logits=True,
        max_count=MAX_TRUE_SRC,
        slot_existence_mode="noisy_or",
    )
    corrected.slot_existence.sum().backward()
    gradient = logits.grad.detach()

    boundary_logits = torch.full((2, N_SUB), -2.0, dtype=torch.float64, device=device)
    boundary_logits[1, 7] = 0.25
    boundary = continuous_band_bridge(
        boundary_logits,
        matrix,
        values_are_logits=True,
        max_count=MAX_TRUE_SRC,
        slot_existence_mode="max_logit",
        slot_temperature=float(config.get("slot_temperature", 1.0)),
    )
    hard_from_probability = (boundary.probabilities >= 0.5).any(dim=-1)
    hard_from_logit = boundary_logits.amax(dim=-1) >= 0.0
    permuted = continuous_band_bridge(
        boundary_logits[:, torch.arange(N_SUB - 1, -1, -1, device=device)],
        matrix,
        values_are_logits=True,
        max_count=MAX_TRUE_SRC,
        slot_existence_mode="max_logit",
        slot_temperature=float(config.get("slot_temperature", 1.0)),
    )
    checks = {
        "tail_probability_not_accumulated": bool(
            torch.allclose(corrected.slot_existence, probability[:, 0], atol=1e-12, rtol=0.0)
        ),
        "legacy_tail_is_larger": bool(torch.all(legacy.slot_existence > corrected.slot_existence)),
        "hard_boundary_exact": bool(torch.equal(hard_from_probability, hard_from_logit)),
        "permutation_invariant": bool(
            torch.allclose(boundary.slot_existence, permuted.slot_existence, atol=0.0, rtol=0.0)
        ),
        "frequency_bridge_unchanged": bool(
            torch.allclose(corrected.frequency_weights, legacy.frequency_weights, atol=0.0, rtol=0.0)
        ),
        "gradient_finite_nonzero": finite(gradient) and bool(torch.count_nonzero(gradient).item()),
        "cardinality_normalized": bool(
            torch.allclose(
                corrected.cardinality_distribution.sum(),
                torch.tensor(1.0, dtype=torch.float64, device=device),
                atol=1e-12,
                rtol=0.0,
            )
        ),
    }
    return {
        "checks": checks,
        "tail_slot_existence_corrected": corrected.slot_existence.detach().cpu().tolist(),
        "tail_slot_existence_legacy": legacy.slot_existence.detach().cpu().tolist(),
        "gradient_nonzero_elements": int(torch.count_nonzero(gradient).item()),
    }


def run_preflight(run_root: Path) -> dict[str, Any]:
    manifest = verify_manifest(run_root)
    config = manifest["config"]
    seed = int(config["seed"])
    set_deterministic(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    require(device.type == "cuda", "G1要求CUDA")
    ch3_a, d8_a, model_info = build_models(device)
    ch3_b, d8_b, _ = build_models(device)
    initial_a = trainable_state(ch3_a, d8_a)
    initial_b = trainable_state(ch3_b, d8_b)
    hashes_a = {name: state_hash(value) for name, value in initial_a.items()}
    hashes_b = {name: state_hash(value) for name, value in initial_b.items()}
    require(hashes_a == hashes_b, "两轨初始化不一致")
    configure_trainable(ch3_a, d8_a, training=True)
    configure_trainable(ch3_b, d8_b, training=True)
    semantic = semantic_contract(config, device)
    require(all(semantic["checks"].values()), f"槽位存在语义合同未通过: {semantic['checks']}")
    records = []
    for count in range(4):
        records.extend(
            [
                row
                for row in manifest["subsets"]["train"]
                if int(row["true_k"]) == count
            ][:4]
        )
    require(len(records) == 16, "P1要求K=0/1/2/3各4条样本")
    diagnostics = []
    with SampleStore("train") as store:
        lo, hi = store.subband_edges()
        matrix = build_subband_fft_matrix(
            torch.from_numpy(lo), torch.from_numpy(hi), sample_rate_hz=FS, n_fft=N_FFT,
            dtype=torch.float64, device=device,
        )
        geometry = receiver_geometry(device)
        for record in records:
            sample = store.sample(record)
            row: dict[str, Any] = {"true_k": int(record["true_k"]), "tracks": {}}
            pair_seed = seed + int(record["true_k"])
            for track, current_ch3, current_d8 in (
                ("soft_sg", ch3_a, d8_a), ("soft_e2e", ch3_b, d8_b)
            ):
                set_deterministic(pair_seed)
                current_ch3.zero_grad(set_to_none=True)
                current_d8.zero_grad(set_to_none=True)
                started = time.perf_counter()
                logits = current_ch3(sample["coarse_dpd"].to(device)[None])
                total, components, _, heatmap, offset = losses(
                    logits, sample, matrix, geometry, current_d8, device, config, track
                )
                total.backward()
                ch3_loc_present = any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in current_ch3.band_heads.parameters())
                d8_grad_present = any(parameter.grad is not None and bool(parameter.grad.abs().sum() > 0) for parameter in current_d8.decoder.parameters())
                row["tracks"][track] = {
                    "losses": {name: float(value.detach().item()) for name, value in components.items()},
                    "total_loss": float(total.detach().item()),
                    "heatmap_finite": finite(heatmap),
                    "offset_finite": finite(offset),
                    "ch3_gradient_present": ch3_loc_present,
                    "d8_gradient_present": d8_grad_present,
                    "duration_seconds": time.perf_counter() - started,
                }
                del total, components, heatmap, offset, logits
                gc.collect()
                torch.cuda.empty_cache()
            loss_a = row["tracks"]["soft_sg"]["total_loss"]
            loss_b = row["tracks"]["soft_e2e"]["total_loss"]
            row["initial_loss_abs_difference"] = abs(loss_a - loss_b)
            require(row["initial_loss_abs_difference"] <= 1e-6, "两轨初始化loss不一致")
            require(row["tracks"]["soft_sg"]["d8_gradient_present"], "Soft-SG没有D8梯度")
            require(row["tracks"]["soft_e2e"]["d8_gradient_present"], "Soft-E2E没有D8梯度")
            diagnostics.append(row)
    durations = [track["duration_seconds"] for row in diagnostics for track in row["tracks"].values()]
    projected = float(np.mean(durations)) * (
        2 * int(config["epochs"]) * len(manifest["subsets"]["train"])
        + 2 * (int(config["epochs"]) + 1) * len(manifest["subsets"]["val_select"])
        + 3 * len(manifest["subsets"]["val_compare"])
    )
    checks = {
        "semantic_contract_passed": all(semantic["checks"].values()),
        "initial_state_identical": hashes_a == hashes_b,
        "all_losses_identical": all(row["initial_loss_abs_difference"] <= 1e-6 for row in diagnostics),
        "all_outputs_finite": all(
            track["heatmap_finite"] and track["offset_finite"]
            for row in diagnostics for track in row["tracks"].values()
        ),
        "all_d8_gradients_present": all(
            track["d8_gradient_present"] for row in diagnostics for track in row["tracks"].values()
        ),
        "projected_wall_within_limit": projected <= float(config["projected_total_wall_limit_seconds"]),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "STOP_ENGINEERING",
        "gate": f"{GATE_NAME}-P0P1",
        "models": model_info,
        "semantic_contract": semantic,
        "initial_trainable_hashes": hashes_a,
        "diagnostics": diagnostics,
        "projected_total_wall_seconds": projected,
        "checks": checks,
        "test_executed": False,
    }
    write_json(run_root / "preflight_report.json", report)
    require(all(checks.values()), f"G1-P0未通过: {checks}")
    print(json.dumps({"status": report["status"], "projected_seconds": projected}, ensure_ascii=False), flush=True)
    return report


def run_training(run_root: Path, track: str) -> dict[str, Any]:
    require(track in {"soft_sg", "soft_e2e"}, "训练轨错误")
    manifest = verify_manifest(run_root)
    require(load_json(run_root / "preflight_report.json")["status"] == "PASS", "G1-P0未通过")
    config = manifest["config"]
    seed = int(config["seed"])
    set_deterministic(seed)
    device = torch.device("cuda:0")
    ch3, d8, model_info = build_models(device)
    ch3_parameters, d8_parameters = configure_trainable(ch3, d8, training=True)
    optimizer = torch.optim.AdamW(
        [
            {"params": ch3_parameters, "lr": float(config["learning_rate"]), "weight_decay": float(config["ch3_weight_decay"])},
            {"params": d8_parameters, "lr": float(config["learning_rate"]), "weight_decay": float(config["d8_weight_decay"])},
        ]
    )
    track_root = validate_output_path(run_root / "training" / track)
    if track_root.exists():
        raise FileExistsError(f"拒绝复用训练目录: {track_root}")
    track_root.mkdir(parents=True)
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    save_checkpoint(track_root / "epoch_000.pth", ch3, d8, optimizer, 0, track, config)
    initial_eval = evaluate_records(
        ch3, d8, manifest["subsets"]["val_select"], "val_select", device, config,
        progress_path=track_root / "validation_progress.jsonl",
    )
    initial_eval.pop("samples")
    history.append({"epoch": 0, "validation": initial_eval, "training": None})
    write_json(track_root / "history.json", history)
    accumulation = int(config["gradient_accumulation"])
    require(len(manifest["subsets"]["train"]) % accumulation == 0, "训练样本数不能整除梯度累积")
    with SampleStore("train") as store:
        lo, hi = store.subband_edges()
        matrix = build_subband_fft_matrix(
            torch.from_numpy(lo), torch.from_numpy(hi), sample_rate_hz=FS, n_fft=N_FFT,
            dtype=torch.float64, device=device,
        )
        geometry = receiver_geometry(device)
        for epoch in range(1, int(config["epochs"]) + 1):
            configure_trainable(ch3, d8, training=True)
            optimizer.zero_grad(set_to_none=True)
            component_sums = Counter()
            step_diagnostics = []
            chapter_accum = [torch.zeros_like(parameter) for parameter in ch3_parameters]
            epoch_started = time.perf_counter()
            order = sample_order(manifest["subsets"]["train"], seed, epoch)
            for ordinal, record in enumerate(order, start=1):
                sample = store.sample(record)
                logits = ch3(sample["coarse_dpd"].to(device)[None])
                total, components, _, _, _ = losses(
                    logits, sample, matrix, geometry, d8, device, config, track
                )
                chapter_loss = components["band"] + components["exist"]
                chapter_grad = torch.autograd.grad(
                    chapter_loss / accumulation,
                    ch3_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                for index, value in enumerate(chapter_grad):
                    if value is not None:
                        chapter_accum[index].add_(value.detach())
                (total / accumulation).backward()
                require(finite(total), f"训练loss非有限: epoch={epoch}, sample={ordinal}")
                for name, value in components.items():
                    component_sums[name] += float(value.detach().item())
                if ordinal % accumulation == 0:
                    chapter_vector = gradient_vector(ch3_parameters, chapter_accum)
                    total_vector = gradient_vector(ch3_parameters)
                    localization_vector = total_vector - chapter_vector
                    diagnostics = {
                        "epoch": epoch,
                        "optimizer_step": ordinal // accumulation,
                        "chapter_grad_norm": float(torch.linalg.vector_norm(chapter_vector).item()),
                        "localization_grad_norm": float(torch.linalg.vector_norm(localization_vector).item()),
                        "total_ch3_grad_norm": float(torch.linalg.vector_norm(total_vector).item()),
                        "gradient_cosine": cosine(chapter_vector, localization_vector),
                        "localization_nonzero": bool(torch.count_nonzero(localization_vector).item()),
                    }
                    require(finite(total_vector), "CH3梯度非有限")
                    require(finite(gradient_vector(d8_parameters)), "D8梯度非有限")
                    diagnostics["ch3_clip_pre_norm"] = float(
                        torch.nn.utils.clip_grad_norm_(
                            ch3_parameters, float(config["ch3_grad_clip"]), error_if_nonfinite=True
                        ).item()
                    )
                    diagnostics["d8_clip_pre_norm"] = float(
                        torch.nn.utils.clip_grad_norm_(
                            d8_parameters, float(config["d8_grad_clip"]), error_if_nonfinite=True
                        ).item()
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_diagnostics.append(diagnostics)
                    append_jsonl(track_root / "gradient_steps.jsonl", diagnostics)
                    chapter_accum = [torch.zeros_like(parameter) for parameter in ch3_parameters]
                if ordinal % 8 == 0 or ordinal == len(order):
                    print(f"[{track}:epoch{epoch}] {ordinal}/{len(order)}", flush=True)
                del total, components, logits, chapter_loss, chapter_grad
            save_checkpoint(track_root / f"epoch_{epoch:03d}.pth", ch3, d8, optimizer, epoch, track, config)
            validation = evaluate_records(
                ch3, d8, manifest["subsets"]["val_select"], "val_select", device, config,
                progress_path=track_root / "validation_progress.jsonl",
            )
            validation.pop("samples")
            epoch_record = {
                "epoch": epoch,
                "training": {
                    "mean_losses": {name: value / len(order) for name, value in component_sums.items()},
                    "optimizer_steps": len(step_diagnostics),
                    "localization_nonzero_step_rate": float(np.mean([row["localization_nonzero"] for row in step_diagnostics])),
                    "duration_seconds": time.perf_counter() - epoch_started,
                },
                "validation": validation,
            }
            history.append(epoch_record)
            write_json(track_root / "history.json", history)
            print(
                f"[{track}:epoch{epoch}] val_gospa={validation['system']['gospa']['mean']:.6f} "
                f"recall100={validation['system']['set_detection']['100m']['recall']:.6f}",
                flush=True,
            )
    initial_ch3 = history[0]["validation"]["ch3"]
    eligible = []
    for row in history:
        current = row["validation"]["ch3"]
        ch3_ok = (
            current["active_band_macro_f1"] >= initial_ch3["active_band_macro_f1"] - float(config["ch3_noninferiority_absolute"])
            and current["balanced_count_accuracy"] >= initial_ch3["balanced_count_accuracy"] - float(config["ch3_noninferiority_absolute"])
        )
        if ch3_ok:
            eligible.append(row)
    require(eligible, f"{track}没有CH3非劣候选checkpoint")
    best = min(
        eligible,
        key=lambda row: (
            row["validation"]["system"]["gospa"]["mean"],
            -row["validation"]["system"]["exact_count_rate"],
            row["validation"]["system"]["gospa_components_mean_p_sum"]["missed"]
            + row["validation"]["system"]["gospa_components_mean_p_sum"]["false"],
            row["validation"]["system"].get("matched_errors_m", {}).get("p90", math.inf),
            int(row["epoch"]),
        ),
    )
    selected_path = track_root / f"epoch_{int(best['epoch']):03d}.pth"
    summary = {
        "status": "COMPLETED",
        "gate": f"{GATE_NAME}-P2-TRAIN",
        "track": track,
        "model_info": model_info,
        "epochs_completed": int(config["epochs"]),
        "sample_presentations": int(config["epochs"]) * len(manifest["subsets"]["train"]),
        "optimizer_steps": int(config["epochs"]) * len(manifest["subsets"]["train"]) // accumulation,
        "selected_epoch": int(best["epoch"]),
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_checkpoint_sha256": sha256_file(selected_path),
        "selection_rule": "CH3 noninferiority then lexicographic system GOSPA/count/cardinality-tail/epoch",
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
    }
    write_json(track_root / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def run_evaluation(run_root: Path, track: str) -> dict[str, Any]:
    require(track in {"soft_sg", "soft_e2e", "hard_frozen"}, "评价轨错误")
    manifest = verify_manifest(run_root)
    config = manifest["config"]
    set_deterministic(int(config["seed"]))
    device = torch.device("cuda:0")
    ch3, d8, _ = build_models(device)
    if track != "hard_frozen":
        summary = load_json(run_root / "training" / track / "training_summary.json")
        checkpoint_path = Path(summary["selected_checkpoint"])
        checkpoint = load_checkpoint(checkpoint_path, ch3, d8)
        require(checkpoint["track"] == track, "训练checkpoint轨道错误")
        mode = "soft"
    else:
        checkpoint_path = None
        mode = "hard"
    evaluation_root = validate_output_path(run_root / "evaluation" / track)
    if evaluation_root.exists():
        raise FileExistsError(f"拒绝复用评价目录: {evaluation_root}")
    evaluation_root.mkdir(parents=True)
    result = evaluate_records(
        ch3, d8, manifest["subsets"]["val_compare"], "val_compare", device, config,
        mode=mode, progress_path=evaluation_root / "progress.jsonl",
    )
    samples = result.pop("samples")
    for row in samples:
        append_jsonl(evaluation_root / "samples.jsonl", row)
    oracle_result = None
    if track != "hard_frozen":
        oracle_result = evaluate_records(
            ch3, d8, manifest["subsets"]["val_compare"], "val_compare", device, config,
            mode="oracle", progress_path=evaluation_root / "oracle_progress.jsonl",
        )
        oracle_result.pop("samples")
    payload = {
        "status": "COMPLETED",
        "gate": f"{GATE_NAME}-P2-EVAL",
        "track": track,
        "checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
        "checkpoint_sha256": sha256_file(checkpoint_path) if checkpoint_path else None,
        "result": result,
        "oracle_result": oracle_result,
        "test_executed": False,
    }
    write_json(evaluation_root / "evaluation_report.json", payload)
    print(json.dumps({"track": track, "gospa": result["system"]["gospa"]["mean"]}, ensure_ascii=False), flush=True)
    return payload


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_bootstrap(left: list[dict[str, Any]], right: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    require(len(left) == len(right), "配对评价样本数不一致")
    require(
        [(row["local_index"], row["true_count"]) for row in left]
        == [(row["local_index"], row["true_count"]) for row in right],
        "配对评价样本身份/顺序不一致",
    )
    counts = np.asarray([row["true_count"] for row in left], dtype=np.int64)
    left_gospa = np.asarray([row["gospa_m"] for row in left], dtype=np.float64)
    right_gospa = np.asarray([row["gospa_m"] for row in right], dtype=np.float64)
    threshold = int(config["set_detection_threshold_m"])

    def recall(rows: list[dict[str, Any]], indices: np.ndarray) -> float:
        tp = sum(int(rows[int(index)][f"tp_at_{threshold}m"]) for index in indices)
        truth = sum(int(rows[int(index)]["true_count"]) for index in indices)
        return tp / max(truth, 1)

    rng = np.random.default_rng(int(config["bootstrap_seed"]))
    gospa_delta = []
    recall_delta = []
    exact_delta = []
    all_indices = np.arange(len(left), dtype=np.int64)
    for _ in range(int(config["bootstrap_repetitions"])):
        sampled = np.concatenate(
            [rng.choice(np.flatnonzero(counts == count), int(np.sum(counts == count)), replace=True) for count in range(4)]
        )
        gospa_delta.append(float(np.mean(right_gospa[sampled] - left_gospa[sampled])))
        recall_delta.append(recall(right, sampled) - recall(left, sampled))
        left_exact = np.mean([left[int(index)]["true_count"] == left[int(index)]["predicted_count"] for index in sampled])
        right_exact = np.mean([right[int(index)]["true_count"] == right[int(index)]["predicted_count"] for index in sampled])
        exact_delta.append(float(right_exact - left_exact))

    def summary(values: list[float]) -> dict[str, Any]:
        array = np.asarray(values, dtype=np.float64)
        return {"mean": float(array.mean()), "ci95": [float(value) for value in np.quantile(array, [0.025, 0.975])]}

    return {
        "comparison": "soft_e2e_minus_soft_sg",
        "stratified_by_k": True,
        "repetitions": int(config["bootstrap_repetitions"]),
        "point": {
            "gospa_delta_m": float(np.mean(right_gospa - left_gospa)),
            "recall_100m_delta": recall(right, all_indices) - recall(left, all_indices),
            "exact_count_delta": float(
                np.mean([row["true_count"] == row["predicted_count"] for row in right])
                - np.mean([row["true_count"] == row["predicted_count"] for row in left])
            ),
        },
        "bootstrap": {
            "gospa_delta_m": summary(gospa_delta),
            "recall_100m_delta": summary(recall_delta),
            "exact_count_delta": summary(exact_delta),
        },
    }


def run_finalize(run_root: Path) -> dict[str, Any]:
    manifest = verify_manifest(run_root)
    config = manifest["config"]
    reports = {
        track: load_json(run_root / "evaluation" / track / "evaluation_report.json")
        for track in ("hard_frozen", "soft_sg", "soft_e2e")
    }
    samples = {
        track: read_jsonl(run_root / "evaluation" / track / "samples.jsonl")
        for track in reports
    }
    comparison = paired_bootstrap(samples["soft_sg"], samples["soft_e2e"], config)
    sg = reports["soft_sg"]["result"]
    e2e = reports["soft_e2e"]["result"]
    point = comparison["point"]
    intervals = comparison["bootstrap"]
    hard = reports["hard_frozen"]["result"]

    def k_summary(rows: list[dict[str, Any]], count: int) -> dict[str, float]:
        selected = [row for row in rows if int(row["true_count"]) == count]
        require(bool(selected), f"K={count}评价样本为空")
        return {
            "gospa_mean_m": float(np.mean([row["gospa_m"] for row in selected])),
            "exact_count_rate": float(
                np.mean([int(row["predicted_count"]) == count for row in selected])
            ),
        }

    stratified = {
        track: {str(count): k_summary(samples[track], count) for count in range(4)}
        for track in samples
    }
    checks = {
        "sg_gospa_recovers_hard_scale": sg["system"]["gospa"]["mean"]
        <= hard["system"]["gospa"]["mean"]
        * float(config.get("sg_hard_gospa_relative_limit", 1.05)),
        "sg_exact_count_noninferior_to_hard": sg["system"]["exact_count_rate"]
        >= hard["system"]["exact_count_rate"]
        - float(config["exact_count_noninferiority_absolute"]),
        "gospa_point_improves": point["gospa_delta_m"] < 0.0,
        "gospa_ci_upper_below_zero": intervals["gospa_delta_m"]["ci95"][1] < 0.0,
        "recall_point_improves": point["recall_100m_delta"] > 0.0,
        "recall_ci_lower_noninferior": intervals["recall_100m_delta"]["ci95"][0] >= -float(config["recall_noninferiority_absolute"]),
        "exact_count_noninferior": point["exact_count_delta"] >= -float(config["exact_count_noninferiority_absolute"]),
        "ch3_band_noninferior": e2e["ch3"]["active_band_macro_f1"] >= sg["ch3"]["active_band_macro_f1"] - float(config["ch3_noninferiority_absolute"]),
        "ch3_hard_count_noninferior": e2e["ch3"]["balanced_count_accuracy"] >= sg["ch3"]["balanced_count_accuracy"] - float(config["ch3_noninferiority_absolute"]),
        "d8_oracle_gospa_noninferior": reports["soft_e2e"]["oracle_result"]["system"]["gospa"]["mean"]
        <= reports["soft_sg"]["oracle_result"]["system"]["gospa"]["mean"]
        * float(config["d8_oracle_gospa_relative_limit"]),
        "k3_gospa_noninferior": stratified["soft_e2e"]["3"]["gospa_mean_m"]
        <= stratified["soft_sg"]["3"]["gospa_mean_m"]
        * float(config.get("k3_gospa_relative_limit", 1.05)),
        "k3_exact_count_noninferior": stratified["soft_e2e"]["3"]["exact_count_rate"]
        >= stratified["soft_sg"]["3"]["exact_count_rate"]
        - float(config.get("k3_exact_count_noninferiority_absolute", 0.02)),
    }
    pass_name = str(config.get("pass_decision", "G1_PASS_TO_G2"))
    inconclusive_name = str(config.get("inconclusive_decision", "G1_INCONCLUSIVE"))
    no_go_name = str(config.get("no_go_decision", "G1_NO_GO"))
    if all(checks.values()):
        decision = pass_name
        g2_unlocked = True
    elif point["gospa_delta_m"] < 0.0 and intervals["gospa_delta_m"]["ci95"][1] >= 0.0:
        decision = inconclusive_name
        g2_unlocked = False
    else:
        decision = no_go_name
        g2_unlocked = False
    payload = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run/validate",
            "verification_status": "ANALYZED",
        },
        "status": decision,
        "gate": GATE_NAME,
        "run_root": str(run_root.resolve()),
        "tracks": {track: report["result"] for track, report in reports.items()},
        "paired_comparison": comparison,
        "stratified_by_k": stratified,
        "checks": checks,
        "g2_unlocked": g2_unlocked,
        "test_executed": False,
        "scope": "single training seed; fixed K0-3 pilot validation; no test",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(run_root / "final_report.json", payload)
    print(json.dumps({"status": decision, "checks": checks}, ensure_ascii=False), flush=True)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E-G1配对因果训练")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    for name in ("preflight", "finalize"):
        current = subparsers.add_parser(name)
        current.add_argument("--run-root", type=Path, required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--run-root", type=Path, required=True)
    train.add_argument("--track", choices=("soft_sg", "soft_e2e"), required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--run-root", type=Path, required=True)
    evaluate.add_argument("--track", choices=("soft_sg", "soft_e2e", "hard_frozen"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare_run(args.run_id)
    elif args.command == "preflight":
        run_preflight(args.run_root.resolve())
    elif args.command == "train":
        run_training(args.run_root.resolve(), args.track)
    elif args.command == "evaluate":
        run_evaluation(args.run_root.resolve(), args.track)
    elif args.command == "finalize":
        run_finalize(args.run_root.resolve())
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
