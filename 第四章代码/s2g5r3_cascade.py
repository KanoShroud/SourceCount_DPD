"""S2-G5-R3 Hard-19-Actual oracle/predicted four-track cascade audit.

This gate is evaluation only.  It reuses the frozen S2-G5-R2 CH3 checkpoint,
the frozen S2-G4-R4 Hard-19-Actual D8 checkpoint, and the fixed R2
``val_compare`` subset.  It never trains a model or reads the held-out test set.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
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


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CH3_DIR = PROJECT_ROOT / "第三章代码"
for import_root in (SCRIPT_DIR, CH3_DIR, PROJECT_ROOT):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(CH3_DIR))
sys.path.insert(2, str(PROJECT_ROOT))

from dpd_calculator_torch import DPDGeometry, compute_fine_dpd  # noqa: E402
from s2g1_train_ch3 import SourceDetectionDataset, SourceDetectionNet  # noqa: E402
from s2g3_composability import (  # noqa: E402
    EDGE,
    FS,
    LEN,
    decode_d8_sample,
    gospa_sample,
    load_ch4_mat,
    matched_distances,
    maximum_matches_within,
    numeric_summary,
    subband_union_to_fft_mask,
    summarize_track,
)
from s2g4_coarse_d8 import build_model as build_d8_model  # noqa: E402
from s2g5_r2_ch3 import metrics as ch3_metrics  # noqa: E402
from train_yolo import configure_reproducibility  # noqa: E402


R2_ROOT = PROJECT_ROOT / "outputs" / "s2g5r2_ch3" / "20260827_191207"
R4_ROOT = PROJECT_ROOT / "outputs" / "s2g4r4_scale" / "20260826_132829"
CH3_CHECKPOINT = R2_ROOT / "train_8k" / "best_model_v26_B_M10.pth"
D8_CHECKPOINT = (
    R4_ROOT / "09_training" / "n8192" / "hard_actual" / "best_yolo_dualhead_std.pth"
)
RAW_VALIDATION = R2_ROOT / "smoke" / "chapter4" / "data" / "val_data.mat"
COARSE_COMPARE = R2_ROOT / "coarse_subsets" / "val_compare.mat"
R2_MANIFEST = R2_ROOT / "manifest" / "data_manifest.json"
R2_COMPARISON = R2_ROOT / "analysis" / "8k" / "compare_4k_8k.json"
R2_PREDICTIONS = R2_ROOT / "analysis" / "8k" / "predictions_4k_8k.jsonl"

EXPECTED_SHA256 = {
    "ch3_checkpoint": "291ee9bce04b3a5a603568285d8505fd042d0937cc3ab48f6415ed0f24b80e2c",
    "d8_checkpoint": "4caaf2b96c2f8eb666b417f0cffe4ab90760315f9bd92c4d6ce4afcd425e0e7b",
    "raw_validation": "84c5382ec3a05e16aeb038e86e9a6df4e14784f4005bc75e5963235c83551789",
    "coarse_compare": "a35cb199299e17cc86d3cf9793e63e76a7c92650e35558d6d82106188ea90005",
    "r2_manifest": "ac35d5865c0a84ebc6e19bc34f3682d15278d00c826736de42adf767c4bb90a3",
}
BOOTSTRAP_SEED = 20260831
BOOTSTRAP_REPETITIONS = 2000
LOCALIZATION_COUNTS = (2, 3)
TRACKS = ("OB-OK", "OB-PK", "PB-OK", "PB-PK")
AUTHORITATIVE_FOUR_TRACK_DIR = "05_four_tracks_after_gpu_memory_fix"


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
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有JSON: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有JSONL: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def torch_save_new(path: Path, payload: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有张量: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(map(int, values)).items())}


def load_compare_metadata() -> dict[str, np.ndarray | float | int]:
    with h5py.File(COARSE_COMPARE.resolve(), "r") as handle:
        payload: dict[str, np.ndarray | float | int] = {
            "sample_idx": np.asarray(handle["sample_idx_all"], dtype=np.int64).reshape(-1),
            "source_count": np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1),
            "band_mask": np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "ignore_mask": np.asarray(handle["ignore_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "src_pos": np.asarray(handle["src_pos_all"], dtype=np.float32).transpose(2, 1, 0),
            "fc_offset": np.asarray(handle["fc_offset_all"], dtype=np.float32).T,
            "bw_actual": np.asarray(handle["BW_actual_all"], dtype=np.float32).T,
            "symbol_rate": np.asarray(handle["symbolRate_all"], dtype=np.float32).T,
            "threshold": float(np.asarray(handle["thresh_val"]).reshape(-1)[0]),
            "n_sub": int(np.asarray(handle["N_sub_val"]).reshape(-1)[0]),
            "max_src": int(np.asarray(handle["max_src_val"]).reshape(-1)[0]),
        }
    return payload


def compare_subset_to_raw(
    metadata: dict[str, np.ndarray | float | int], raw: dict[str, Any]
) -> dict[str, bool]:
    indices = np.asarray(metadata["sample_idx"], dtype=np.int64)
    checks = {
        "source_count_exact": np.array_equal(metadata["source_count"], raw["src_count"][indices]),
        "band_mask_exact": np.array_equal(metadata["band_mask"], raw["band_mask"][indices]),
        "src_pos_exact": np.array_equal(metadata["src_pos"], raw["src_pos"][indices]),
        "fc_offset_exact": np.array_equal(metadata["fc_offset"], raw["fc_offset"][indices]),
        "bw_actual_exact": np.array_equal(metadata["bw_actual"], raw["bw_actual"][indices]),
        "symbol_rate_exact": np.array_equal(metadata["symbol_rate"], raw["symbol_rate"][indices]),
    }
    require(all(checks.values()), f"val_compare与原始validation字段不一致: {checks}")
    return checks


def verify_input_hashes() -> dict[str, dict[str, Any]]:
    paths = {
        "ch3_checkpoint": CH3_CHECKPOINT,
        "d8_checkpoint": D8_CHECKPOINT,
        "raw_validation": RAW_VALIDATION,
        "coarse_compare": COARSE_COMPARE,
        "r2_manifest": R2_MANIFEST,
    }
    identities = {name: file_identity(path) for name, path in paths.items()}
    for name, expected in EXPECTED_SHA256.items():
        require(identities[name]["sha256"] == expected, f"{name} SHA256不符")
    return identities


def checkpoint_contracts() -> dict[str, Any]:
    ch3 = torch.load(CH3_CHECKPOINT.resolve(), map_location="cpu", weights_only=False)
    config = ch3.get("config", {})
    require(ch3.get("epoch") == 105, "CH3冻结checkpoint epoch不是105")
    require(config.get("mode") == "transformer", "CH3模型不是Transformer")
    require(config.get("max_src") == 10 and config.get("n_sub") == 19, "CH3输出维度错误")
    require(config.get("threshold") == 0.5, "CH3阈值不是0.5")
    require(config.get("seed") == 42 and config.get("deterministic") is True, "CH3随机性配置错误")
    d8 = torch.load(D8_CHECKPOINT.resolve(), map_location="cpu", weights_only=False)
    args = d8.get("args", {})
    require(d8.get("epoch") == 59, "Hard-D8冻结checkpoint epoch不是59")
    require(d8.get("method") == "dualhead" and d8.get("save_tag") == "dualhead_std", "D8身份错误")
    require(args.get("s2g4r4_scratch") is True, "D8不是S2-G4-R4受控训练")
    require(args.get("amp") is False and args.get("batch_size") == 8, "D8 FP32/batch身份错误")
    require(args.get("run_label") == "n8192_hard_seed42", "D8不是8k Hard-Actual臂")
    return {
        "ch3": {"epoch": 105, "config": config},
        "d8": {"epoch": 59, "args": args},
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    identities = verify_input_hashes()
    available_gib = psutil.virtual_memory().available / 2**30
    require(available_gib >= 10.0, f"启动可用RAM不足10 GiB: {available_gib:.2f}")
    metadata = load_compare_metadata()
    raw = load_ch4_mat(RAW_VALIDATION, include_iq=False)
    require(len(raw["src_count"]) == 2048, "原始validation不是2048条")
    indices = np.asarray(metadata["sample_idx"], dtype=np.int64)
    counts = np.asarray(metadata["source_count"], dtype=np.int64)
    require(indices.shape == (1024,) and len(np.unique(indices)) == 1024, "val_compare索引错误")
    require(np.all((indices >= 0) & (indices < 2048)), "val_compare原始索引越界")
    require(histogram(counts) == {"0": 256, "1": 256, "2": 256, "3": 256}, "val_compare未按K均衡")
    localization = np.flatnonzero(np.isin(counts, LOCALIZATION_COUNTS))
    require(localization.size == 512, "K=2/3定位集合不是512条")
    field_checks = compare_subset_to_raw(metadata, raw)
    require(
        bool(np.isclose(float(metadata["threshold"]), 0.2, rtol=0.0, atol=1e-7)),
        "Hard-19-Actual阈值不是0.2",
    )
    require(metadata["n_sub"] == 19 and metadata["max_src"] == 3, "数据维度契约错误")
    report = {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "stage": "preflight",
        "scope": "fixed val_compare; K0-3 routing and K2/3 localization; no test or training",
        "inputs": identities,
        "checkpoint_contracts": checkpoint_contracts(),
        "sample_contract": {
            "raw_validation_count": 2048,
            "val_compare_count": 1024,
            "val_compare_histogram": histogram(counts),
            "localization_count": int(localization.size),
            "localization_histogram": histogram(counts[localization]),
            "raw_index_min": int(indices.min()),
            "raw_index_max": int(indices.max()),
            "field_checks": field_checks,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "available_ram_gib": available_gib,
        },
        "fixed_rules": {
            "ch3_probability_threshold": 0.5,
            "hard19_actual_coverage_threshold": 0.2,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "test_executed": False,
            "training_executed": False,
        },
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(args.run_root / "00_preflight" / "preflight_report.json", report)
    return report


def build_ch3(device: torch.device) -> tuple[SourceDetectionNet, dict[str, Any]]:
    checkpoint = torch.load(CH3_CHECKPOINT.resolve(), map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = SourceDetectionNet(
        n_sub=int(config["n_sub"]),
        max_src=int(config["max_src"]),
        mode=str(config["mode"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint


@torch.no_grad()
def infer_ch3_arrays(
    data_path: Path, *, indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    configure_reproducibility(42, True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    dataset = SourceDetectionDataset(
        data_path.resolve(), augment=False, normalize="sample_zscore", max_src_override=10,
    )
    if indices is None:
        selected = np.arange(len(dataset), dtype=np.int64)
    else:
        selected = np.asarray(indices, dtype=np.int64)
    model, checkpoint = build_ch3(device)
    spectra = torch.stack([dataset[int(index)][0] for index in selected])
    truths = np.stack([(dataset[int(index)][2] > 0.5).numpy() for index in selected])
    ignores = np.stack([(dataset[int(index)][3] > 0.5).numpy() for index in selected])
    counts = np.asarray([int(dataset[int(index)][1]) for index in selected], dtype=np.int64)
    logits_parts: list[np.ndarray] = []
    for start in range(0, len(selected), 64):
        logits = model(spectra[start:start + 64].to(device))
        require(bool(torch.isfinite(logits).all()), "CH3推理logits含NaN/Inf")
        logits_parts.append(logits.cpu().numpy())
    logits = np.concatenate(logits_parts).astype(np.float32, copy=False)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    threshold = float(checkpoint["config"]["threshold"])
    prediction = probabilities > threshold
    return {
        "selected_local_idx": selected,
        "logits": logits,
        "probabilities": probabilities.astype(np.float32, copy=False),
        "prediction": prediction,
        "truth": truths,
        "ignore": ignores,
        "source_count": counts,
        "k_prediction": prediction.any(axis=2).sum(axis=1).astype(np.int64),
    }


def reference_prediction_arrays() -> tuple[np.ndarray, np.ndarray]:
    rows = load_jsonl(R2_PREDICTIONS)
    require(len(rows) == 1024, "R2保存的预测不是1024条")
    prediction = np.zeros((1024, 10, 19), dtype=bool)
    counts = np.zeros(1024, dtype=np.int64)
    for expected, row in enumerate(rows):
        require(int(row["sample_index"]) == expected, "R2预测样本索引不连续")
        counts[expected] = int(row["predicted_count_8k"])
        bands = row["predicted_bands_8k"]
        require(len(bands) == 10, "R2预测槽位数不是10")
        for slot, bins in enumerate(bands):
            prediction[expected, slot, np.asarray(bins, dtype=np.int64)] = True
    return prediction, counts


def nested_max_abs(left: Any, right: Any) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        require(set(left) == set(right), "指标字段集合不一致")
        return max((nested_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, list) and isinstance(right, list):
        require(len(left) == len(right), "指标列表长度不一致")
        return max((nested_max_abs(a, b) for a, b in zip(left, right, strict=True)), default=0.0)
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right))
    require(left == right, f"非数值指标不一致: {left!r} != {right!r}")
    return 0.0


def run_infer(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    verify_input_hashes()
    arrays = infer_ch3_arrays(COARSE_COMPARE)
    metadata = load_compare_metadata()
    require(np.array_equal(arrays["source_count"], metadata["source_count"]), "CH3数据源数顺序错误")
    reference_prediction, reference_count = reference_prediction_arrays()
    require(np.array_equal(arrays["prediction"], reference_prediction), "CH3二值预测未精确复现R2")
    require(np.array_equal(arrays["k_prediction"], reference_count), "CH3 K预测未精确复现R2")
    require(int(arrays["k_prediction"].max()) <= 3, "CH3产生K>3，违反R3冻结契约")
    metrics = ch3_metrics({
        "prediction": arrays["prediction"],
        "truth": arrays["truth"],
        "ignore": arrays["ignore"],
        "source_count": arrays["source_count"],
    })
    reference_metrics = load_json(R2_COMPARISON)["metrics_8k"]
    metric_max_abs = nested_max_abs(metrics, reference_metrics)
    require(metric_max_abs <= 1e-12, f"CH3聚合指标未在1e-12内复现: {metric_max_abs}")
    output_npz = args.run_root / "01_ch3_inference" / "ch3_predictions.npz"
    if output_npz.exists():
        raise FileExistsError(f"拒绝覆盖: {output_npz}")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        logits=arrays["logits"],
        probabilities=arrays["probabilities"],
        band_prediction=arrays["prediction"].astype(np.uint8),
        band_truth=arrays["truth"].astype(np.uint8),
        ignore_mask=arrays["ignore"].astype(np.uint8),
        source_count=arrays["source_count"],
        k_prediction=arrays["k_prediction"],
        raw_sample_idx=np.asarray(metadata["sample_idx"], dtype=np.int64),
    )
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, predicted in zip(arrays["source_count"], arrays["k_prediction"], strict=True):
        confusion[int(truth), int(predicted)] += 1
    localization = np.flatnonzero(np.isin(arrays["source_count"], LOCALIZATION_COUNTS))
    routing = {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "sample_count": 1024,
        "truth_histogram": histogram(arrays["source_count"]),
        "prediction_histogram": histogram(arrays["k_prediction"]),
        "confusion_true_rows_pred_columns_k0_k3": confusion.tolist(),
        "k0": {
            "sample_count": 256,
            "correct_stop": int(confusion[0, 0]),
            "false_route": int(confusion[0, 1:].sum()),
        },
        "k1": {
            "sample_count": 256,
            "correct_pending_route": int(confusion[1, 1]),
            "false_stop": int(confusion[1, 0]),
            "false_d8_route": int(confusion[1, 2:].sum()),
            "locator_status": "single_source_locator_pending",
        },
        "localization_subset": {
            "sample_count": int(localization.size),
            "truth_histogram": histogram(arrays["source_count"][localization]),
            "predicted_histogram": histogram(arrays["k_prediction"][localization]),
            "local_indices": localization.tolist(),
            "raw_indices": np.asarray(metadata["sample_idx"], dtype=np.int64)[localization].tolist(),
        },
        "k1_diagnostic_rule": "On true K2/3 samples, predicted K1 uses D8 Top-1 only as counterfactual count-error diagnosis.",
        "test_executed": False,
    }
    report = {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "stage": "ch3_inference",
        "checkpoint": file_identity(CH3_CHECKPOINT),
        "data": file_identity(COARSE_COMPARE),
        "prediction_output": file_identity(output_npz),
        "metrics": metrics,
        "r2_binary_predictions_exact": True,
        "r2_k_predictions_exact": True,
        "r2_metrics_max_abs_difference": metric_max_abs,
        "routing_report": str((args.run_root / "02_routing" / "routing_report.json").resolve()),
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
        "training_executed": False,
    }
    write_json(args.run_root / "01_ch3_inference" / "inference_report.json", report)
    write_json(args.run_root / "02_routing" / "routing_report.json", routing)
    return report


def receiver_geometry(device: torch.device) -> DPDGeometry:
    angles = np.arange(4) * 2 * np.pi / 4
    receivers = np.stack([500.0 * np.cos(angles), 500.0 * np.sin(angles)], axis=1)
    return DPDGeometry(receivers, [0.0, 0.0], 2000, 10, FS, LEN, device)


def one_fine_dpd(
    raw: dict[str, Any], raw_index: int, slot_mask: np.ndarray,
    geometry: DPDGeometry, chunk_size: int,
) -> tuple[torch.Tensor, np.ndarray, float]:
    fft_mask = subband_union_to_fft_mask(slot_mask, raw["sub_f_lo"], raw["sub_f_hi"])
    started = time.perf_counter()
    if not bool(fft_mask.any()):
        return torch.zeros((1, 401, 401), dtype=torch.float16), fft_mask, time.perf_counter() - started
    signal = raw["sig_real"][raw_index] + 1j * raw["sig_imag"][raw_index]
    mtr = compute_fine_dpd(signal, geometry, freq_mask=fft_mask, chunk_size=chunk_size)
    require(bool(torch.isfinite(mtr).all()) and bool(torch.all(mtr >= 0)), "细DPD含NaN/Inf或负值")
    return torch.log(mtr + 1.0).unsqueeze(0).half().cpu(), fft_mask, time.perf_counter() - started


def run_pilot(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    verify_input_hashes()
    metadata = load_compare_metadata()
    counts = np.asarray(metadata["source_count"], dtype=np.int64)
    selected = np.concatenate([
        np.flatnonzero(counts == 2)[:2],
        np.flatnonzero(counts == 3)[:2],
    ])
    arrays = infer_ch3_arrays(COARSE_COMPARE, indices=selected)
    raw = load_ch4_mat(RAW_VALIDATION, include_iq=True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    geometry = receiver_geometry(device)
    dpds: list[torch.Tensor] = []
    modes: list[str] = []
    empty_flags: list[bool] = []
    raw_indices = np.asarray(metadata["sample_idx"], dtype=np.int64)[selected]
    truth = np.asarray(metadata["band_mask"], dtype=np.float32)[selected] > 0.5
    for row, (local_index, raw_index) in enumerate(zip(selected, raw_indices, strict=True)):
        count = int(counts[local_index])
        for mode, slot_mask in (
            ("oracle", truth[row, :count]),
            ("predicted", arrays["prediction"][row]),
        ):
            dpd, fft_mask, _ = one_fine_dpd(raw, int(raw_index), slot_mask, geometry, 40000)
            dpds.append(dpd)
            modes.append(mode)
            empty_flags.append(not bool(fft_mask.any()))
    model, _ = build_d8_model(D8_CHECKPOINT, device)
    batch = torch.stack([
        (sample.float() - sample.float().mean()) / (sample.float().std() + 1e-6)
        for sample in dpds
    ]).to(device)
    with torch.no_grad():
        heatmap, offset = model(batch)
    require(bool(torch.isfinite(heatmap).all() and torch.isfinite(offset).all()), "pilot D8输出非法")
    decode_counts = {}
    for requested in range(4):
        positions, _ = decode_d8_sample(heatmap[0], offset[0], requested)
        decode_counts[str(requested)] = int(len(positions))
    require(decode_counts == {"0": 0, "1": 1, "2": 2, "3": 3}, "K=0/1/2/3解码pilot失败")
    report = {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "stage": "four_sample_pilot",
        "selected_local_indices": selected.tolist(),
        "selected_raw_indices": raw_indices.tolist(),
        "truth_histogram": histogram(counts[selected]),
        "dpd_count": len(dpds),
        "dpd_modes": modes,
        "predicted_empty_band_count": int(sum(
            flag for flag, mode in zip(empty_flags, modes, strict=True) if mode == "predicted"
        )),
        "oracle_empty_band_count": int(sum(
            flag for flag, mode in zip(empty_flags, modes, strict=True) if mode == "oracle"
        )),
        "decode_output_count_by_requested_k": decode_counts,
        "synthetic_empty_band_rule_checked": bool(torch.count_nonzero(torch.zeros((1, 401, 401))) == 0),
        "duration_seconds": time.perf_counter() - started,
        "performance_interpretation_allowed": False,
    }
    require(report["oracle_empty_band_count"] == 0, "pilot Oracle频带为空")
    write_json(args.run_root / "pilot" / "pilot_report.json", report)
    return report


def run_build_fine(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    verify_input_hashes()
    require(args.mode in {"oracle", "predicted"}, "未知频带模式")
    output_dir = args.run_root / ("03_fine_oracle" if args.mode == "oracle" else "04_fine_predicted")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空细DPD目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.run_root / "01_ch3_inference" / "ch3_predictions.npz"
    require(predictions_path.is_file(), "CH3预测文件不存在")
    predictions = np.load(predictions_path.resolve())
    metadata = load_compare_metadata()
    counts = np.asarray(metadata["source_count"], dtype=np.int64)
    local_indices = np.flatnonzero(np.isin(counts, LOCALIZATION_COUNTS))
    raw_indices = np.asarray(metadata["sample_idx"], dtype=np.int64)[local_indices]
    predicted_band = predictions["band_prediction"].astype(bool)
    k_prediction = predictions["k_prediction"].astype(np.int64)
    require(predicted_band.shape == (1024, 10, 19), "CH3预测band shape错误")
    require(np.array_equal(predictions["source_count"], counts), "预测文件源数顺序错误")
    raw = load_ch4_mat(RAW_VALIDATION, include_iq=True)
    device = torch.device("cuda:0")
    configure_reproducibility(42, True)
    torch.cuda.set_device(0)
    geometry = receiver_geometry(device)
    shard_entries: list[dict[str, Any]] = []
    all_empty: list[bool] = []
    all_bins: list[int] = []
    all_seconds: list[float] = []
    for shard_number, start in enumerate(range(0, len(local_indices), args.shard_size)):
        stop = min(start + args.shard_size, len(local_indices))
        shard_local = local_indices[start:stop]
        shard_raw = raw_indices[start:stop]
        fine_values: list[torch.Tensor] = []
        pos_values: list[torch.Tensor] = []
        empty_values: list[bool] = []
        bin_values: list[int] = []
        for local_index, raw_index in zip(shard_local, shard_raw, strict=True):
            true_count = int(counts[local_index])
            if args.mode == "oracle":
                slot_mask = np.asarray(metadata["band_mask"])[local_index, :true_count] > 0.5
            else:
                slot_mask = predicted_band[local_index]
            fine, fft_mask, elapsed = one_fine_dpd(
                raw, int(raw_index), slot_mask, geometry, args.chunk_size,
            )
            empty = not bool(fft_mask.any())
            positions = raw["src_pos"][raw_index, :true_count]
            positions = positions[np.argsort(np.linalg.norm(positions, axis=1))]
            pos_label = np.zeros((3, 2), dtype=np.float32)
            pos_label[:true_count] = positions / EDGE
            fine_values.append(fine)
            pos_values.append(torch.from_numpy(pos_label))
            empty_values.append(empty)
            bin_values.append(int(fft_mask.sum()))
            all_seconds.append(elapsed)
        payload = {
            "fine_dpd": torch.stack(fine_values),
            "pos_label": torch.stack(pos_values),
            "n_src": torch.from_numpy(counts[shard_local].astype(np.int64)),
            "predicted_k": torch.from_numpy(k_prediction[shard_local].astype(np.int64)),
            "local_idx": torch.from_numpy(shard_local.astype(np.int64)),
            "raw_idx": torch.from_numpy(shard_raw.astype(np.int64)),
            "empty_band": torch.tensor(empty_values, dtype=torch.bool),
            "frequency_bin_count": torch.tensor(bin_values, dtype=torch.int64),
            "band_mode": args.mode,
        }
        shard_path = output_dir / f"part_{shard_number:03d}.pt"
        torch_save_new(shard_path, payload)
        shard_entries.append({
            **file_identity(shard_path),
            "sample_count": int(stop - start),
            "local_index_first": int(shard_local[0]),
            "local_index_last": int(shard_local[-1]),
        })
        all_empty.extend(empty_values)
        all_bins.extend(bin_values)
        print(
            f"[{args.mode}] {stop}/{len(local_indices)} elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
    empty_count = int(sum(all_empty))
    if args.mode == "oracle":
        require(empty_count == 0, "Oracle Hard-19-Actual出现空频带")
    else:
        expected_empty = int(np.sum(k_prediction[local_indices] == 0))
        require(empty_count == expected_empty, "Predicted空频带数与K_pred=0不一致")
    report = {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "stage": "build_fine_dpd",
        "band_mode": args.mode,
        "sample_count": len(local_indices),
        "source_count_histogram": histogram(counts[local_indices]),
        "predicted_k_histogram": histogram(k_prediction[local_indices]),
        "empty_band_count": empty_count,
        "frequency_bin_count": numeric_summary(np.asarray(all_bins)),
        "per_sample_seconds": numeric_summary(np.asarray(all_seconds)),
        "chunk_size": args.chunk_size,
        "shard_size": args.shard_size,
        "shards": shard_entries,
        "input_predictions": file_identity(predictions_path),
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
        "training_executed": False,
    }
    write_json(output_dir / "index.json", report)
    return report


def load_fine_shards(index_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    index = load_json(index_path)
    require(index.get("status") == "PASS" and index.get("sample_count") == 512, "细DPD索引状态错误")
    shards = []
    for entry in index["shards"]:
        path = Path(entry["path"])
        require(sha256_file(path) == entry["sha256"], f"细DPD shard SHA变化: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        require(bool(torch.isfinite(payload["fine_dpd"]).all()), "细DPD shard含NaN/Inf")
        shards.append(payload)
    return index, shards


def sample_record(
    *, track: str, local_index: int, raw_index: int, true_count: int,
    requested_count: int, true_positions: np.ndarray, heatmap: torch.Tensor,
    offset: torch.Tensor, empty_band: bool,
) -> dict[str, Any]:
    if empty_band:
        predicted_positions = np.zeros((0, 2), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
    else:
        predicted_positions, scores = decode_d8_sample(heatmap, offset, requested_count)
    matches = matched_distances(true_positions, predicted_positions)
    distances = [distance for _, _, distance in matches]
    gospa = gospa_sample(true_positions, predicted_positions)
    record = {
        "track": track,
        "local_index": local_index,
        "raw_index": raw_index,
        "true_count": true_count,
        "requested_count": requested_count,
        "predicted_count": int(len(predicted_positions)),
        "empty_band": empty_band,
        "predicted_positions_m": predicted_positions.tolist(),
        "peak_scores": scores.tolist(),
        "matched_errors_m": distances,
        "gospa_m": gospa["value_m"],
        "gospa_localization_p_sum": gospa["localization_p_sum"],
        "gospa_missed_p_sum": gospa["missed_p_sum"],
        "gospa_false_p_sum": gospa["false_p_sum"],
    }
    for threshold in (10, 30, 50, 100):
        record[f"tp_at_{threshold}m"] = maximum_matches_within(
            true_positions, predicted_positions, float(threshold),
        )
    return record


def evaluate_once() -> dict[str, list[dict[str, Any]]]:
    configure_reproducibility(42, True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model, _ = build_d8_model(D8_CHECKPOINT, device)
    records: dict[str, list[dict[str, Any]]] = {track: [] for track in TRACKS}
    for mode, prefix, index_path in (
        ("oracle", "OB", PROJECT_ROOT / "unused"),
        ("predicted", "PB", PROJECT_ROOT / "unused"),
    ):
        del index_path
        # The caller binds the active run root through EVALUATION_RUN_ROOT.
        require(EVALUATION_RUN_ROOT is not None, "评估run root未绑定")
        fine_index = EVALUATION_RUN_ROOT / (
            "03_fine_oracle/index.json" if mode == "oracle" else "04_fine_predicted/index.json"
        )
        _, shards = load_fine_shards(fine_index)
        for shard in shards:
            fine = shard["fine_dpd"].float()
            normalized = torch.stack([
                (sample - sample.mean()) / (sample.std() + 1e-6) for sample in fine
            ])
            with torch.no_grad():
                heatmap, offset = model(normalized.to(device))
            require(bool(torch.isfinite(heatmap).all() and torch.isfinite(offset).all()), "D8输出含NaN/Inf")
            for row in range(len(fine)):
                true_count = int(shard["n_src"][row])
                predicted_k = int(shard["predicted_k"][row])
                require(true_count in LOCALIZATION_COUNTS and 0 <= predicted_k <= 3, "评估K越界")
                true_positions = shard["pos_label"][row, :true_count].numpy() * EDGE
                common = {
                    "local_index": int(shard["local_idx"][row]),
                    "raw_index": int(shard["raw_idx"][row]),
                    "true_count": true_count,
                    "true_positions": true_positions,
                    "heatmap": heatmap[row],
                    "offset": offset[row],
                    "empty_band": bool(shard["empty_band"][row]),
                }
                records[f"{prefix}-OK"].append(sample_record(
                    track=f"{prefix}-OK", requested_count=true_count, **common,
                ))
                records[f"{prefix}-PK"].append(sample_record(
                    track=f"{prefix}-PK", requested_count=predicted_k, **common,
                ))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


EVALUATION_RUN_ROOT: Path | None = None


def track_report(track: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [distance for row in rows for distance in row["matched_errors_m"]]
    metrics = summarize_track(rows, matched)
    stratified = {}
    for count in LOCALIZATION_COUNTS:
        selected = [row for row in rows if row["true_count"] == count]
        selected_errors = [distance for row in selected for distance in row["matched_errors_m"]]
        stratified[f"K{count}"] = summarize_track(selected, selected_errors)
    return {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "track": track,
        "band_mode": "oracle" if track.startswith("OB") else "predicted",
        "k_mode": "oracle" if track.endswith("OK") else "predicted",
        "sample_count": len(rows),
        "metrics": metrics,
        "stratified": stratified,
        "worst_samples": sorted(rows, key=lambda row: row["gospa_m"], reverse=True)[:10],
        "k1_semantics": "D8 Top-1 is counterfactual diagnosis on true K2/3 samples, not the pending true-K1 locator.",
        "test_executed": False,
        "training_executed": False,
    }


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    global EVALUATION_RUN_ROOT
    started = time.perf_counter()
    verify_input_hashes()
    EVALUATION_RUN_ROOT = args.run_root.resolve()
    primary = evaluate_once()
    replay = evaluate_once()
    replay_max_abs = nested_max_abs(primary, replay)
    require(replay_max_abs <= 1e-12, f"D8独立重载结果不一致: {replay_max_abs}")
    output_dir = args.run_root / AUTHORITATIVE_FOUR_TRACK_DIR
    reports = {}
    for track in TRACKS:
        require(len(primary[track]) == 512, f"{track}样本数不是512")
        local = [row["local_index"] for row in primary[track]]
        raw = [row["raw_index"] for row in primary[track]]
        require(len(set(local)) == 512 and len(set(raw)) == 512, f"{track}样本身份重复")
        report = track_report(track, primary[track])
        samples_path = output_dir / f"{track.lower().replace('-', '_')}_samples.jsonl"
        write_jsonl(samples_path, primary[track])
        report["samples"] = file_identity(samples_path)
        report_path = output_dir / f"{track.lower().replace('-', '_')}.json"
        write_json(report_path, report)
        reports[track] = file_identity(report_path)
    identities = [
        [(row["local_index"], row["raw_index"], row["true_count"]) for row in primary[track]]
        for track in TRACKS
    ]
    require(all(identity == identities[0] for identity in identities[1:]), "四轨样本身份不一致")
    audit = {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "stage": "four_track_evaluation",
        "sample_count_per_track": 512,
        "tracks": reports,
        "checkpoint": file_identity(D8_CHECKPOINT),
        "checkpoint_reload_max_abs_difference": replay_max_abs,
        "sample_identity_exact_across_tracks": True,
        "d8_forward_passes_per_replay": 2,
        "four_decodes_from_two_band_inputs": True,
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
        "training_executed": False,
    }
    write_json(output_dir / "evaluation_audit.json", audit)
    return audit


def ci95(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def point_track_values(rows: list[dict[str, Any]], indices: np.ndarray | None = None) -> tuple[float, float]:
    selected = rows if indices is None else [rows[int(index)] for index in indices]
    gospa = float(np.mean([row["gospa_m"] for row in selected]))
    recall = sum(row["tp_at_100m"] for row in selected) / sum(row["true_count"] for row in selected)
    return gospa, float(recall)


def run_analyze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.run_root / AUTHORITATIVE_FOUR_TRACK_DIR
    tracks = {
        track: load_jsonl(output_dir / f"{track.lower().replace('-', '_')}_samples.jsonl")
        for track in TRACKS
    }
    identities = {
        track: [(row["local_index"], row["raw_index"], row["true_count"]) for row in rows]
        for track, rows in tracks.items()
    }
    require(all(identity == identities["OB-OK"] for identity in identities.values()), "分析阶段四轨身份不一致")
    counts = np.asarray([row["true_count"] for row in tracks["OB-OK"]], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = {
        "band_gospa_delta": [],
        "count_gospa_delta": [],
        "full_gospa_delta": [],
        "interaction_gospa": [],
        "band_minus_count_gospa": [],
        "full_relative_gospa": [],
        "full_recall_delta": [],
    }
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = np.concatenate([
            rng.choice(np.flatnonzero(counts == count), size=int(np.sum(counts == count)), replace=True)
            for count in LOCALIZATION_COUNTS
        ])
        values = {track: point_track_values(rows, sampled) for track, rows in tracks.items()}
        oo_g, oo_r = values["OB-OK"]
        op_g, _ = values["OB-PK"]
        po_g, _ = values["PB-OK"]
        pp_g, pp_r = values["PB-PK"]
        bootstrap["band_gospa_delta"].append(po_g - oo_g)
        bootstrap["count_gospa_delta"].append(op_g - oo_g)
        bootstrap["full_gospa_delta"].append(pp_g - oo_g)
        bootstrap["interaction_gospa"].append(pp_g - po_g - op_g + oo_g)
        bootstrap["band_minus_count_gospa"].append(po_g - op_g)
        bootstrap["full_relative_gospa"].append(pp_g / oo_g - 1.0)
        bootstrap["full_recall_delta"].append(pp_r - oo_r)
    point = {track: point_track_values(rows) for track, rows in tracks.items()}
    oo_g, oo_r = point["OB-OK"]
    op_g, _ = point["OB-PK"]
    po_g, _ = point["PB-OK"]
    pp_g, pp_r = point["PB-PK"]
    effects = {
        "band_gospa_delta_pb_ok_minus_ob_ok": po_g - oo_g,
        "count_gospa_delta_ob_pk_minus_ob_ok": op_g - oo_g,
        "full_gospa_delta_pb_pk_minus_ob_ok": pp_g - oo_g,
        "interaction_gospa": pp_g - po_g - op_g + oo_g,
        "band_minus_count_gospa_pb_ok_minus_ob_pk": po_g - op_g,
        "full_relative_gospa_increase": pp_g / oo_g - 1.0,
        "full_recall_100m_delta": pp_r - oo_r,
    }
    bootstrap_summary = {
        key: {
            "bootstrap_mean": float(np.mean(values)),
            "ci95": ci95(np.asarray(values, dtype=np.float64)),
        }
        for key, values in bootstrap.items()
    }
    direct_supported = (
        bootstrap_summary["full_relative_gospa"]["ci95"][1] <= 0.10
        and bootstrap_summary["full_recall_delta"]["ci95"][0] >= -0.05
    )
    comparison_ci = bootstrap_summary["band_minus_count_gospa"]["ci95"]
    if direct_supported:
        priority = "DIRECT_CASCADE_SUPPORTED"
    elif comparison_ci[0] > 0:
        priority = "BAND_PATH_PRIORITY"
    elif comparison_ci[1] < 0:
        priority = "COUNT_PATH_PRIORITY"
    else:
        priority = "JOINT_OR_UNCERTAIN_PRIORITY"
    interaction_amplification = bootstrap_summary["interaction_gospa"]["ci95"][0] > 0
    per_k = {}
    for count in LOCALIZATION_COUNTS:
        selected = np.flatnonzero(counts == count)
        per_k[f"K{count}"] = {
            track: {"mean_gospa_m": point_track_values(rows, selected)[0], "recall_100m": point_track_values(rows, selected)[1]}
            for track, rows in tracks.items()
        }
    report = {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "stage": "paired_four_track_analysis",
        "scope": "512 fixed K2/3 val_compare samples; single CH3 and D8 training seed; not test or paper performance",
        "track_point_metrics": {
            track: {"mean_gospa_m": values[0], "recall_100m": values[1]}
            for track, values in point.items()
        },
        "effects": effects,
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "repetitions": BOOTSTRAP_REPETITIONS,
            "stratified_by_true_k": True,
            "metrics": bootstrap_summary,
        },
        "per_k": per_k,
        "decision": {
            "engineering_status": "PASS",
            "interface_status": "DIRECT_CASCADE_SUPPORTED" if direct_supported else "DIRECT_CASCADE_NOT_YET_SUPPORTED",
            "r4_priority": priority,
            "interaction_amplification": interaction_amplification,
            "direct_equivalence_checks": {
                "relative_gospa_ci_upper_at_most_0_10": bootstrap_summary["full_relative_gospa"]["ci95"][1] <= 0.10,
                "recall_delta_ci_lower_at_least_minus_0_05": bootstrap_summary["full_recall_delta"]["ci95"][0] >= -0.05,
            },
        },
        "interpretation_boundaries": {
            "shared_logits": "Predicted band and predicted K share CH3 logits; four tracks isolate propagation paths, not statistically independent causes.",
            "k1": "Predicted K1 on true K2/3 uses D8 Top-1 as a counterfactual diagnostic only.",
            "bootstrap": "Sample bootstrap does not measure training-seed variance.",
            "next_gate": "No R4, 16k, seed extension, threshold tuning, or test is authorized automatically.",
        },
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
        "training_executed": False,
    }
    write_json(args.run_root / "06_analysis" / "paired_analysis.json", report)
    return report


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = verify_input_hashes()
    required_reports = {
        "preflight": args.run_root / "00_preflight" / "preflight_report.json",
        "pilot": args.run_root / "pilot" / "pilot_report.json",
        "inference": args.run_root / "01_ch3_inference" / "inference_report.json",
        "routing": args.run_root / "02_routing" / "routing_report.json",
        "oracle_fine": args.run_root / "03_fine_oracle" / "index.json",
        "predicted_fine": args.run_root / "04_fine_predicted" / "index.json",
        "evaluation": args.run_root / AUTHORITATIVE_FOUR_TRACK_DIR / "evaluation_audit.json",
        "analysis": args.run_root / "06_analysis" / "paired_analysis.json",
    }
    reports = {name: load_json(path) for name, path in required_reports.items()}
    require(all(report.get("status") == "PASS" for report in reports.values()), "存在未通过的R3科研阶段")
    monitor_reports = sorted((args.run_root / "monitor").glob("*/stage_monitor_report.json"))
    monitor_payloads = [load_json(path) for path in monitor_reports]
    authoritative_stages = {
        "02_preflight_after_float_tolerance",
        "03_four_sample_pilot",
        "04_ch3_inference",
        "05_build_oracle_fine",
        "06_build_predicted_fine",
        "08_four_track_after_gpu_cleanup",
        "09_paired_analysis",
    }
    pass_payloads = [report for report in monitor_payloads if report.get("status") == "PASS"]
    pass_stages = {str(report.get("stage")) for report in pass_payloads}
    require(authoritative_stages.issubset(pass_stages), "R3权威PASS监控阶段不完整")
    retained_failures = [report for report in monitor_payloads if report.get("status") != "PASS"]
    allowed_failures = {
        ("01_preflight", "CRASHED"),
        ("07_four_track_evaluation", "COMPLETED_WITH_RED_FLAGS"),
    }
    observed_failures = {
        (str(report.get("stage")), str(report.get("status"))) for report in retained_failures
    }
    require(observed_failures == allowed_failures, f"存在未解释的失败监控阶段: {observed_failures}")
    analysis = reports["analysis"]
    total_size = sum(path.stat().st_size for path in args.run_root.rglob("*") if path.is_file())
    final = {
        "status": "PASS",
        "gate": "S2-G5-R3",
        "experiment_id": "SYS-S2G5-R3-20260828",
        "engineering_status": "PASS",
        "scientific_decision": analysis["decision"],
        "track_point_metrics": analysis["track_point_metrics"],
        "effects": analysis["effects"],
        "bootstrap": analysis["bootstrap"],
        "input_identities_unchanged": inputs,
        "stage_reports": {name: file_identity(path) for name, path in required_reports.items()},
        "monitor_summary": {
            "authoritative_pass_count_before_finalize": len(pass_payloads),
            "authoritative_pass_stages": sorted(authoritative_stages),
            "retained_failed_attempts": [
                {
                    "stage": report["stage"],
                    "status": report["status"],
                    "warnings": report.get("warnings", []),
                    "red_flags": report.get("red_flags", []),
                    "log_path": report.get("log_path"),
                }
                for report in retained_failures
            ],
            "warning_union": sorted({warning for report in pass_payloads for warning in report.get("warnings", [])}),
            "red_flag_union": sorted({flag for report in pass_payloads for flag in report.get("red_flags", [])}),
            "minimum_system_available_bytes": min(report["minimum_system_available_bytes"] for report in pass_payloads),
            "maximum_process_tree_rss_bytes": max(report["maximum_process_tree_rss_bytes"] for report in pass_payloads),
            "maximum_gpu_used_mib": max(
                (report["maximum_gpu_used_mib"] for report in pass_payloads if report["maximum_gpu_used_mib"] is not None),
                default=None,
            ),
            "total_authoritative_monitored_duration_seconds": sum(
                report["duration_seconds"] for report in pass_payloads
            ),
        },
        "output_size_bytes_before_finalize": total_size,
        "code": file_identity(Path(__file__)),
        "prohibitions": {
            "test_executed": False,
            "training_executed": False,
            "threshold_tuned": False,
            "oracle_fallback_used": False,
            "k_clipped": False,
            "r4_executed": False,
        },
        "scope": "Fixed validation diagnostic only; not final paper performance and not a deployable K0-3 system while the true-K1 locator is pending.",
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(args.run_root / "final_report.json", final)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S2-G5-R3四轨级联误差隔离")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "pilot", "infer", "evaluate", "analyze", "finalize"):
        current = sub.add_parser(name)
        current.add_argument("--run_root", type=Path, required=True)
    fine = sub.add_parser("build-fine")
    fine.add_argument("--run_root", type=Path, required=True)
    fine.add_argument("--mode", choices=("oracle", "predicted"), required=True)
    fine.add_argument("--chunk_size", type=int, default=40000)
    fine.add_argument("--shard_size", type=int, default=64)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.run_root = args.run_root.resolve()
    require(args.run_root != PROJECT_ROOT and PROJECT_ROOT in args.run_root.parents, "run_root必须位于项目内且不能是项目根")
    handlers = {
        "preflight": run_preflight,
        "pilot": run_pilot,
        "infer": run_infer,
        "build-fine": run_build_fine,
        "evaluate": run_evaluate,
        "analyze": run_analyze,
        "finalize": run_finalize,
    }
    result = handlers[args.command](args)
    print(json.dumps({
        "status": result.get("status"),
        "gate": result.get("gate"),
        "stage": result.get("stage", args.command),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
