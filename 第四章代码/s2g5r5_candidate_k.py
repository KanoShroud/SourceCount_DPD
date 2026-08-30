"""S2-G5-R5冻结CH3/D8的有序候选K与空间证据联合判决。

本入口只读取R4冻结数据与checkpoint，不运行test，不修改原0.5阈值，
也不训练CH3或D8。val_select只用于温度和选择器拟合；val_compare只评价一次。
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import h5py
import joblib
import numpy as np
import torch
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CH3_DIR = PROJECT_ROOT / "第三章代码"
for import_root in (SCRIPT_DIR, CH3_DIR, PROJECT_ROOT):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(1, str(CH3_DIR))
sys.path.insert(2, str(PROJECT_ROOT))

import s2g5r3_cascade as r3  # noqa: E402
from s2g5_r4_lazy_dataset import LazySourceDetectionDataset  # noqa: E402
from s2g1_train_ch3 import SourceDetectionDataset, SourceDetectionNet  # noqa: E402
from s2g4_coarse_d8 import build_model as build_d8_model  # noqa: E402
from train_yolo import configure_reproducibility  # noqa: E402


R4_ROOT = PROJECT_ROOT / "outputs" / "s2g5r4_ch3_scale" / "20260828_151735"
R2_ROOT = PROJECT_ROOT / "outputs" / "s2g5r2_ch3" / "20260827_191207"
CH3_CHECKPOINT = R4_ROOT / "train_16k" / "best_model_v26_B_M10.pth"
D8_CHECKPOINT = (
    PROJECT_ROOT / "outputs" / "s2g4r4_scale" / "20260826_132829"
    / "09_training" / "n8192" / "hard_actual" / "best_yolo_dualhead_std.pth"
)
TRAIN_MAT = R4_ROOT / "training_views" / "data_16k" / "train_data.mat"
TRAIN_CACHE = R4_ROOT / "lazy_cache" / "train_16k_sample_zscore.npy"
TRAIN_CACHE_MANIFEST = R4_ROOT / "lazy_cache" / "train_16k_sample_zscore.json"
VAL_SELECT = R2_ROOT / "coarse_subsets" / "val_select.mat"
VAL_COMPARE = R2_ROOT / "coarse_subsets" / "val_compare.mat"
RAW_VALIDATION = R2_ROOT / "smoke" / "chapter4" / "data" / "val_data.mat"
R4_CASCADE = R4_ROOT / "cascade_16k"
COMPARE_FINE_INDEX = R4_CASCADE / "04_fine_predicted" / "index.json"
R4_OB_OK = R4_CASCADE / "05_four_tracks_16k_batch8" / "ob_ok_samples.jsonl"
R4_PB_PK = R4_CASCADE / "05_four_tracks_16k_batch8" / "pb_pk_samples.jsonl"

EXPECTED_CH3_SHA256 = "f2f7a7c345f1866b871282670f45671d930de34bb06493a9828a9d04a38699c4"
EXPECTED_D8_SHA256 = "4caaf2b96c2f8eb666b417f0cffe4ab90760315f9bd92c4d6ce4afcd425e0e7b"
LOCALIZATION_COUNTS = (2, 3)
BOOTSTRAP_SEED = 20260905
BOOTSTRAP_REPETITIONS = 2000
EPSILON = 1e-7


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(map(int, values)).items())}


def identity(path: Path) -> dict[str, Any]:
    return r3.file_identity(path.resolve())


def load_metadata(path: Path) -> dict[str, Any]:
    with h5py.File(path.resolve(), "r") as handle:
        return {
            "sample_idx": np.asarray(handle["sample_idx_all"], dtype=np.int64).reshape(-1),
            "source_count": np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1),
            "band_mask": np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0),
            "src_pos": np.asarray(handle["src_pos_all"], dtype=np.float32).transpose(2, 1, 0),
            "threshold": float(np.asarray(handle["thresh_val"]).reshape(-1)[0]),
        }


def build_ch3(device: torch.device) -> tuple[SourceDetectionNet, dict[str, Any]]:
    payload = torch.load(CH3_CHECKPOINT, map_location=device, weights_only=False)
    config = payload["config"]
    model = SourceDetectionNet(
        n_sub=int(config["n_sub"]), max_src=int(config["max_src"]), mode=str(config["mode"]),
    ).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model, payload


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    inputs = {
        "ch3_checkpoint": identity(CH3_CHECKPOINT),
        "d8_checkpoint": identity(D8_CHECKPOINT),
        "train_mat": identity(TRAIN_MAT),
        "train_cache": identity(TRAIN_CACHE),
        "train_cache_manifest": identity(TRAIN_CACHE_MANIFEST),
        "val_select": identity(VAL_SELECT),
        "val_compare": identity(VAL_COMPARE),
        "raw_validation": identity(RAW_VALIDATION),
        "compare_fine_index": identity(COMPARE_FINE_INDEX),
        "r4_ob_ok": identity(R4_OB_OK),
        "r4_pb_pk": identity(R4_PB_PK),
    }
    require(inputs["ch3_checkpoint"]["sha256"] == EXPECTED_CH3_SHA256, "CH3 SHA错误")
    require(inputs["d8_checkpoint"]["sha256"] == EXPECTED_D8_SHA256, "D8 SHA错误")
    checkpoint = torch.load(CH3_CHECKPOINT, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    require(checkpoint["epoch"] == 78, "CH3 best epoch不是78")
    require(config["mode"] == "transformer" and config["max_src"] == 10, "CH3结构错误")
    require(config["threshold"] == 0.5 and config["seed"] == 42, "CH3阈值或seed错误")
    select = load_metadata(VAL_SELECT)
    compare = load_metadata(VAL_COMPARE)
    for name, metadata in (("val_select", select), ("val_compare", compare)):
        require(len(metadata["source_count"]) == 1024, f"{name}样本数错误")
        require(histogram(metadata["source_count"]) == {"0": 256, "1": 256, "2": 256, "3": 256}, f"{name}分层错误")
        require(math.isclose(metadata["threshold"], 0.2, abs_tol=1e-7), f"{name}阈值错误")
    require(len(np.intersect1d(select["sample_idx"], compare["sample_idx"])) == 0, "两个validation子集重叠")
    manifest = load_json(TRAIN_CACHE_MANIFEST)
    require(manifest["status"] == "PASS" and manifest["source_meta"]["sample_count"] == 16384, "16k缓存manifest错误")
    require(torch.cuda.is_available(), "cuda不可用")
    report = {
        "status": "PASS",
        "gate": "S2-G5-R5-A",
        "stage": "preflight",
        "inputs": inputs,
        "sample_contract": {
            "train": 16384,
            "val_select": 1024,
            "val_compare": 1024,
            "select_compare_disjoint": True,
            "localization_each": 512,
        },
        "fixed_rules": {
            "probe_features": ["slot_max", "slot_mean", "slot_top3_mean"],
            "candidate": "top1_plus_highest_probability_adjacent_k",
            "selector": "standard_scaler_plus_l2_logistic_regression",
            "d8_batch": 8,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "test_executed": False,
        },
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(args.run_root / "00_preflight" / "preflight.json", report)
    return report


def slot_features(probabilities: np.ndarray) -> np.ndarray:
    require(probabilities.ndim == 3 and probabilities.shape[1:] == (10, 19), "CH3概率shape错误")
    maximum = probabilities.max(axis=2)
    mean = probabilities.mean(axis=2)
    top3 = np.partition(probabilities, -3, axis=2)[:, :, -3:].mean(axis=2)
    return np.concatenate([maximum, mean, top3], axis=1).astype(np.float32)


@torch.no_grad()
def infer_dataset(model: torch.nn.Module, dataset: Any, batch_size: int = 64) -> dict[str, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    device = next(model.parameters()).device
    logits: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    band_truth: list[np.ndarray] = []
    ignore: list[np.ndarray] = []
    for spectra, source_count, bands, ignored in loader:
        current = model(spectra.to(device, non_blocking=True))
        require(bool(torch.isfinite(current).all()), "CH3 logits含NaN/Inf")
        logits.append(current.cpu().numpy())
        counts.append(source_count.numpy())
        band_truth.append((bands > 0.5).numpy())
        ignore.append((ignored > 0.5).numpy())
    joined = np.concatenate(logits).astype(np.float32, copy=False)
    probabilities = (1.0 / (1.0 + np.exp(-joined))).astype(np.float32, copy=False)
    prediction = probabilities > 0.5
    return {
        "logits": joined,
        "probabilities": probabilities,
        "features": slot_features(probabilities),
        "source_count": np.concatenate(counts).astype(np.int64),
        "band_truth": np.concatenate(band_truth),
        "ignore_mask": np.concatenate(ignore),
        "band_prediction": prediction,
        "baseline_k": prediction.any(axis=2).sum(axis=1).astype(np.int64),
    }


def save_arrays(path: Path, arrays: dict[str, np.ndarray], *, raw_indices: np.ndarray | None = None) -> None:
    path = path.resolve()
    require(not path.exists(), f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    if raw_indices is not None:
        payload["raw_sample_idx"] = np.asarray(raw_indices, dtype=np.int64)
    np.savez_compressed(path, **payload)


def run_infer(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    require((args.run_root / "00_preflight" / "preflight.json").is_file(), "preflight未完成")
    configure_reproducibility(42, True)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    model, _ = build_ch3(device)
    train = LazySourceDetectionDataset(
        TRAIN_MAT, cache_path=TRAIN_CACHE, manifest_path=TRAIN_CACHE_MANIFEST,
        augment=False, normalize="sample_zscore", max_src_override=10, verify_source_hash=False,
    )
    datasets = {
        "train": train,
        "val_select": SourceDetectionDataset(VAL_SELECT, augment=False, normalize="sample_zscore", max_src_override=10),
        "val_compare": SourceDetectionDataset(VAL_COMPARE, augment=False, normalize="sample_zscore", max_src_override=10),
    }
    outputs = {}
    for name, dataset in datasets.items():
        arrays = infer_dataset(model, dataset)
        metadata = None if name == "train" else load_metadata(VAL_SELECT if name == "val_select" else VAL_COMPARE)
        if metadata is not None:
            require(np.array_equal(arrays["source_count"], metadata["source_count"]), f"{name}源数身份错误")
        output = args.run_root / "01_frozen_logits" / f"{name}.npz"
        save_arrays(output, arrays, raw_indices=None if metadata is None else metadata["sample_idx"])
        outputs[name] = identity(output)
        print(f"inference {name}: {len(dataset)}", flush=True)
    del model, datasets, train
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "status": "PASS", "gate": "S2-G5-R5-A", "stage": "frozen_logits",
        "outputs": outputs, "duration_seconds": time.perf_counter() - started,
        "training_executed": False, "test_executed": False,
    }
    write_json(args.run_root / "01_frozen_logits" / "inference_report.json", report)
    return report


def fit_conditional_models(features: np.ndarray, counts: np.ndarray) -> list[Pipeline]:
    models = []
    for threshold in (1, 2, 3):
        risk = counts >= threshold - 1
        target = (counts[risk] >= threshold).astype(np.int64)
        require(len(np.unique(target)) == 2, f"K>={threshold}条件标签缺少类别")
        model = Pipeline([
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)),
        ])
        model.fit(features[risk], target)
        models.append(model)
    return models


def conditional_logits(models: list[Pipeline], features: np.ndarray) -> np.ndarray:
    return np.column_stack([model.decision_function(features) for model in models]).astype(np.float64)


def conditional_nll(logits: np.ndarray, counts: np.ndarray, temperature: float) -> float:
    total = 0.0
    observations = 0
    for column, threshold in enumerate((1, 2, 3)):
        risk = counts >= threshold - 1
        target = (counts[risk] >= threshold).astype(np.float64)
        current = logits[risk, column] / temperature
        total += float(np.logaddexp(0.0, current).sum() - np.dot(target, current))
        observations += int(risk.sum())
    return total / observations


def ordinal_probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    conditional = 1.0 / (1.0 + np.exp(-logits / temperature))
    r1 = conditional[:, 0]
    r2 = r1 * conditional[:, 1]
    r3_value = r2 * conditional[:, 2]
    probabilities = np.column_stack([1.0 - r1, r1 - r2, r2 - r3_value, r3_value])
    require(np.all(probabilities >= -1e-12), "有序K概率出现负值")
    require(np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-12), "有序K概率和不为1")
    return np.clip(probabilities, 0.0, 1.0)


def adjacent_candidates(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top1 = probabilities.argmax(axis=1).astype(np.int64)
    second = np.empty_like(top1)
    for index, value in enumerate(top1):
        neighbors = [candidate for candidate in (value - 1, value + 1) if 0 <= candidate <= 3]
        second[index] = max(neighbors, key=lambda candidate: probabilities[index, candidate])
    candidates = np.sort(np.column_stack([top1, second]), axis=1)
    require(np.all(candidates[:, 1] - candidates[:, 0] == 1), "候选K不相邻")
    return top1, candidates


def count_metrics(probabilities: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    prediction = probabilities.argmax(axis=1)
    confusion = np.zeros((4, 4), dtype=np.int64)
    for actual, predicted in zip(truth, prediction, strict=True):
        confusion[int(actual), int(predicted)] += 1
    class_accuracy = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    nll = -np.log(np.clip(probabilities[np.arange(len(truth)), truth], EPSILON, 1.0)).mean()
    one_hot = np.eye(4)[truth]
    brier = np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    confidence = probabilities.max(axis=1)
    correct = prediction == truth
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        selected = (confidence >= lower) & (confidence < lower + 0.1 + 1e-12)
        if np.any(selected):
            ece += float(selected.mean() * abs(correct[selected].mean() - confidence[selected].mean()))
    return {
        "accuracy": float(correct.mean()),
        "balanced_accuracy": float(class_accuracy.mean()),
        "class_accuracy": {str(index): float(value) for index, value in enumerate(class_accuracy)},
        "confusion_true_rows_pred_columns": confusion.tolist(),
        "nll": float(nll), "brier": float(brier), "ece_10_bin": float(ece),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    arrays = {name: np.load(args.run_root / "01_frozen_logits" / f"{name}.npz") for name in ("train", "val_select", "val_compare")}
    models = fit_conditional_models(arrays["train"]["features"], arrays["train"]["source_count"])
    select_logits = conditional_logits(models, arrays["val_select"]["features"])
    optimization = minimize_scalar(
        lambda log_temperature: conditional_nll(select_logits, arrays["val_select"]["source_count"], math.exp(log_temperature)),
        bounds=(math.log(0.05), math.log(20.0)), method="bounded", options={"xatol": 1e-8},
    )
    require(bool(optimization.success), "温度拟合失败")
    temperature = float(math.exp(optimization.x))
    model_path = args.run_root / "02_ordinal_probe" / "ordinal_probe.joblib"
    require(not model_path.exists(), f"拒绝覆盖: {model_path}")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "temperature": temperature}, model_path)
    reports = {}
    for name in ("val_select", "val_compare"):
        current = arrays[name]
        logits = conditional_logits(models, current["features"])
        probabilities = ordinal_probabilities(logits, temperature)
        top1, candidates = adjacent_candidates(probabilities)
        truth = current["source_count"].astype(np.int64)
        candidate_hit = np.any(candidates == truth[:, None], axis=1)
        output = args.run_root / "02_ordinal_probe" / f"{name}_ordinal.npz"
        save_arrays(output, {
            "conditional_logits": logits.astype(np.float32),
            "k_probabilities": probabilities.astype(np.float32),
            "top1_k": top1,
            "candidate_k": candidates,
            "candidate_hit": candidate_hit,
            "source_count": truth,
            "slot_scores": current["probabilities"].max(axis=2).astype(np.float32),
            "band_prediction": current["band_prediction"],
            "raw_sample_idx": current["raw_sample_idx"],
            "baseline_k": current["baseline_k"],
        })
        errors = top1 != truth
        reports[name] = {
            "metrics": count_metrics(probabilities, truth),
            "candidate_coverage_all": float(candidate_hit.mean()),
            "candidate_coverage_among_top1_errors": float(candidate_hit[errors].mean()) if np.any(errors) else 1.0,
            "top1_histogram": histogram(top1),
            "candidate_output": identity(output),
        }
    report = {
        "status": "PASS", "gate": "S2-G5-R5-A", "stage": "ordinal_probe",
        "temperature": temperature,
        "temperature_fit_set": "val_select",
        "probe_fit_set": "train_16k",
        "model": identity(model_path),
        "splits": reports,
        "feature_selection_after_val_compare": False,
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
    }
    write_json(args.run_root / "02_ordinal_probe" / "probe_report.json", report)
    return report


def build_select_fine(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = args.run_root / "03_select_fine_predicted"
    require(not output_dir.exists() or not any(output_dir.iterdir()), f"拒绝覆盖非空目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = load_metadata(VAL_SELECT)
    frozen = np.load(args.run_root / "01_frozen_logits" / "val_select.npz")
    require(np.array_equal(frozen["source_count"], metadata["source_count"]), "val_select身份错误")
    counts = metadata["source_count"].astype(np.int64)
    local_indices = np.flatnonzero(np.isin(counts, LOCALIZATION_COUNTS))
    raw_indices = metadata["sample_idx"][local_indices]
    raw = r3.load_ch4_mat(RAW_VALIDATION, include_iq=True)
    configure_reproducibility(42, True)
    torch.cuda.set_device(0)
    geometry = r3.receiver_geometry(torch.device("cuda:0"))
    shard_entries = []
    all_empty = []
    all_seconds = []
    for shard_number, start in enumerate(range(0, len(local_indices), args.shard_size)):
        shard_local = local_indices[start:start + args.shard_size]
        shard_raw = raw_indices[start:start + args.shard_size]
        fine_values = []
        pos_values = []
        empty_values = []
        for local_index, raw_index in zip(shard_local, shard_raw, strict=True):
            count = int(counts[local_index])
            fine, fft_mask, elapsed = r3.one_fine_dpd(
                raw, int(raw_index), frozen["band_prediction"][local_index].astype(bool), geometry, args.chunk_size,
            )
            positions = raw["src_pos"][raw_index, :count]
            positions = positions[np.argsort(np.linalg.norm(positions, axis=1))]
            label = np.zeros((3, 2), dtype=np.float32)
            label[:count] = positions / r3.EDGE
            fine_values.append(fine)
            pos_values.append(torch.from_numpy(label))
            empty_values.append(not bool(fft_mask.any()))
            all_seconds.append(elapsed)
        payload = {
            "fine_dpd": torch.stack(fine_values),
            "pos_label": torch.stack(pos_values),
            "n_src": torch.from_numpy(counts[shard_local]),
            "local_idx": torch.from_numpy(shard_local),
            "raw_idx": torch.from_numpy(shard_raw),
            "empty_band": torch.tensor(empty_values, dtype=torch.bool),
        }
        shard_path = output_dir / f"part_{shard_number:03d}.pt"
        r3.torch_save_new(shard_path, payload)
        shard_entries.append({**identity(shard_path), "sample_count": len(shard_local)})
        all_empty.extend(empty_values)
        print(f"select fine DPD: {min(start + args.shard_size, len(local_indices))}/{len(local_indices)}", flush=True)
    expected_empty = int(np.sum(frozen["baseline_k"][local_indices] == 0))
    require(int(sum(all_empty)) == expected_empty, "val_select空频带与原K=0不一致")
    report = {
        "status": "PASS", "gate": "S2-G5-R5-A", "stage": "build_val_select_predicted_fine",
        "sample_count": len(local_indices), "source_count_histogram": histogram(counts[local_indices]),
        "empty_band_count": int(sum(all_empty)), "shards": shard_entries,
        "per_sample_seconds": r3.numeric_summary(np.asarray(all_seconds)),
        "duration_seconds": time.perf_counter() - started, "test_executed": False,
    }
    write_json(output_dir / "index.json", report)
    return report


def load_generic_shards(index_path: Path) -> list[dict[str, Any]]:
    index = load_json(index_path)
    require(index["status"] == "PASS" and index["sample_count"] == 512, "细DPD索引错误")
    payloads = []
    for entry in index["shards"]:
        path = Path(entry["path"])
        require(r3.sha256_file(path) == entry["sha256"], f"shard SHA变化: {path}")
        payloads.append(torch.load(path, map_location="cpu", weights_only=False))
    return payloads


def extract_peaks_for_split(index_path: Path) -> list[dict[str, Any]]:
    configure_reproducibility(42, True)
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model, _ = build_d8_model(D8_CHECKPOINT, device)
    rows = []
    for shard in load_generic_shards(index_path):
        fine = shard["fine_dpd"].float()
        normalized = torch.stack([(sample - sample.mean()) / (sample.std() + 1e-6) for sample in fine])
        heat_parts = []
        offset_parts = []
        with torch.no_grad():
            for start in range(0, len(normalized), 8):
                heatmap, offset = model(normalized[start:start + 8].to(device))
                heat_parts.append(heatmap.cpu())
                offset_parts.append(offset.cpu())
        heatmaps = torch.cat(heat_parts)
        offsets = torch.cat(offset_parts)
        require(bool(torch.isfinite(heatmaps).all() and torch.isfinite(offsets).all()), "D8输出非法")
        for index in range(len(fine)):
            empty = bool(shard["empty_band"][index])
            if empty:
                positions = np.zeros((0, 2), dtype=np.float32)
                scores = np.zeros(0, dtype=np.float32)
            else:
                positions, scores = r3.decode_d8_sample(heatmaps[index], offsets[index], 4)
            count = int(shard["n_src"][index])
            rows.append({
                "local_index": int(shard["local_idx"][index]),
                "raw_index": int(shard["raw_idx"][index]),
                "true_count": count,
                "true_positions_m": (shard["pos_label"][index, :count].numpy() * r3.EDGE).tolist(),
                "empty_band": empty,
                "peak_positions_m": positions.tolist(),
                "peak_scores": scores.tolist(),
            })
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return sorted(rows, key=lambda row: row["local_index"])


def run_peaks(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    outputs = {}
    for split, index_path in (
        ("val_select", args.run_root / "03_select_fine_predicted" / "index.json"),
        ("val_compare", COMPARE_FINE_INDEX),
    ):
        rows = extract_peaks_for_split(index_path)
        require(len(rows) == 512, f"{split}峰值样本数错误")
        output = args.run_root / "04_d8_peaks" / f"{split}_top4.jsonl"
        write_jsonl(output, rows)
        outputs[split] = identity(output)
        print(f"D8 peaks {split}: {len(rows)}", flush=True)
    report = {
        "status": "PASS", "gate": "S2-G5-R5-A", "stage": "d8_top4_peaks",
        "d8_batch": 8, "d8_checkpoint": identity(D8_CHECKPOINT), "outputs": outputs,
        "duration_seconds": time.perf_counter() - started, "test_executed": False,
    }
    write_json(args.run_root / "04_d8_peaks" / "peaks_report.json", report)
    return report


def selector_features(
    probabilities: np.ndarray, slot_scores: np.ndarray, candidates: np.ndarray,
    peak_positions: list[list[float]], peak_scores: list[float],
) -> np.ndarray | None:
    lower, upper = map(int, candidates)
    if upper < 2 or len(peak_scores) < upper + 1 or len(peak_positions) < upper:
        return None
    disputed = upper - 1
    score = float(peak_scores[disputed])
    previous = float(peak_scores[disputed - 1]) if disputed > 0 else 0.0
    following = float(peak_scores[disputed + 1])
    disputed_position = np.asarray(peak_positions[disputed], dtype=np.float64)
    accepted = np.asarray(peak_positions[:lower], dtype=np.float64)
    spacing = float(np.min(np.linalg.norm(accepted - disputed_position, axis=1)) / 300.0) if lower else 10.0
    return np.asarray([
        math.log(float(probabilities[upper]) + EPSILON) - math.log(float(probabilities[lower]) + EPSILON),
        float(slot_scores[disputed]), score, score - previous, score - following, spacing,
    ], dtype=np.float64)


def load_peak_map(path: Path) -> dict[int, dict[str, Any]]:
    rows = r3.load_jsonl(path)
    result = {int(row["local_index"]): row for row in rows}
    require(len(result) == 512, "D8峰值样本身份重复")
    return result


def fit_selector(args: argparse.Namespace) -> tuple[Pipeline, dict[str, Any]]:
    ordinal = np.load(args.run_root / "02_ordinal_probe" / "val_select_ordinal.npz")
    peaks = load_peak_map(args.run_root / "04_d8_peaks" / "val_select_top4.jsonl")
    features = []
    targets = []
    total_localization = 0
    covered = 0
    usable = 0
    for local_index, truth in enumerate(ordinal["source_count"]):
        if int(truth) not in LOCALIZATION_COUNTS:
            continue
        total_localization += 1
        candidates = ordinal["candidate_k"][local_index]
        if int(truth) not in candidates:
            continue
        covered += 1
        row = peaks[local_index]
        current = selector_features(
            ordinal["k_probabilities"][local_index], ordinal["slot_scores"][local_index],
            candidates, row["peak_positions_m"], row["peak_scores"],
        )
        if current is None:
            continue
        features.append(current)
        targets.append(int(truth) == int(candidates[1]))
        usable += 1
    matrix = np.stack(features)
    target = np.asarray(targets, dtype=np.int64)
    require(len(np.unique(target)) == 2 and usable >= 100, "选择器可用校准样本不足")
    selector = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)),
    ])
    selector.fit(matrix, target)
    diagnostics = {
        "localization_samples": total_localization,
        "truth_in_candidate_pair": covered,
        "usable_training_samples": usable,
        "target_larger_histogram": histogram(target),
        "training_accuracy_descriptive_only": float(selector.score(matrix, target)),
    }
    return selector, diagnostics


def decoded_record(row: dict[str, Any], requested: int, track: str) -> dict[str, Any]:
    positions = np.asarray(row["peak_positions_m"][:requested], dtype=np.float32)
    scores = np.asarray(row["peak_scores"][:requested], dtype=np.float32)
    true_positions = np.asarray(row["true_positions_m"], dtype=np.float32)
    if row["empty_band"]:
        positions = np.zeros((0, 2), dtype=np.float32)
        scores = np.zeros(0, dtype=np.float32)
    matches = r3.matched_distances(true_positions, positions)
    gospa = r3.gospa_sample(true_positions, positions)
    record = {
        "track": track, "local_index": int(row["local_index"]), "raw_index": int(row["raw_index"]),
        "true_count": int(row["true_count"]), "requested_count": int(requested),
        "predicted_count": int(len(positions)), "empty_band": bool(row["empty_band"]),
        "predicted_positions_m": positions.tolist(), "peak_scores": scores.tolist(),
        "matched_errors_m": [distance for _, _, distance in matches],
        "gospa_m": gospa["value_m"], "gospa_localization_p_sum": gospa["localization_p_sum"],
        "gospa_missed_p_sum": gospa["missed_p_sum"], "gospa_false_p_sum": gospa["false_p_sum"],
    }
    for threshold in (10, 30, 50, 100):
        record[f"tp_at_{threshold}m"] = r3.maximum_matches_within(true_positions, positions, float(threshold))
    return record


def bootstrap_tracks(tracks: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = np.asarray([row["true_count"] for row in tracks["OB-OK"]], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {"selected_relative_to_ob_ok": [], "selected_recall_delta": [], "selected_minus_baseline_gospa": []}
    for _ in range(BOOTSTRAP_REPETITIONS):
        sampled = np.concatenate([
            rng.choice(np.flatnonzero(counts == count), size=int(np.sum(counts == count)), replace=True)
            for count in LOCALIZATION_COUNTS
        ])
        ob_g, ob_r = r3.point_track_values(tracks["OB-OK"], sampled)
        base_g, _ = r3.point_track_values(tracks["PB-BASE"], sampled)
        selected_g, selected_r = r3.point_track_values(tracks["PB-SELECTED"], sampled)
        values["selected_relative_to_ob_ok"].append(selected_g / ob_g - 1.0)
        values["selected_recall_delta"].append(selected_r - ob_r)
        values["selected_minus_baseline_gospa"].append(selected_g - base_g)
    return {
        key: {"mean": float(np.mean(current)), "ci95": r3.ci95(np.asarray(current, dtype=np.float64))}
        for key, current in values.items()
    }


def run_select_analyze(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    selector, selector_fit = fit_selector(args)
    selector_path = args.run_root / "05_selector_analysis" / "d8_selector.joblib"
    selector_path.parent.mkdir(parents=True, exist_ok=True)
    require(not selector_path.exists(), f"拒绝覆盖: {selector_path}")
    joblib.dump(selector, selector_path)
    ordinal = np.load(args.run_root / "02_ordinal_probe" / "val_compare_ordinal.npz")
    peak_rows = load_peak_map(args.run_root / "04_d8_peaks" / "val_compare_top4.jsonl")
    baseline_rows = r3.load_jsonl(R4_PB_PK)
    ob_ok_rows = r3.load_jsonl(R4_OB_OK)
    require(len(baseline_rows) == len(ob_ok_rows) == 512, "R4参考轨样本数错误")
    baseline_map = {int(row["local_index"]): row for row in baseline_rows}
    ob_map = {int(row["local_index"]): row for row in ob_ok_rows}
    tracks = {name: [] for name in ("OB-OK", "PB-BASE", "PB-PROBE", "PB-SELECTED", "PB-CANDIDATE-ORACLE")}
    selector_used = 0
    selector_fallback = 0
    selected_k = np.full(1024, -1, dtype=np.int64)
    for local_index, truth in enumerate(ordinal["source_count"]):
        if int(truth) not in LOCALIZATION_COUNTS:
            continue
        peak = peak_rows[local_index]
        candidates = ordinal["candidate_k"][local_index].astype(np.int64)
        feature = selector_features(
            ordinal["k_probabilities"][local_index], ordinal["slot_scores"][local_index],
            candidates, peak["peak_positions_m"], peak["peak_scores"],
        )
        if feature is None:
            chosen = int(ordinal["top1_k"][local_index])
            selector_fallback += 1
        else:
            choose_larger = int(selector.predict(feature[None, :])[0])
            chosen = int(candidates[choose_larger])
            selector_used += 1
        selected_k[local_index] = chosen
        probe_k = int(ordinal["top1_k"][local_index])
        candidate_records = [decoded_record(peak, int(value), "PB-CANDIDATE") for value in candidates]
        oracle = min(candidate_records, key=lambda row: row["gospa_m"])
        tracks["OB-OK"].append(ob_map[local_index])
        tracks["PB-BASE"].append(baseline_map[local_index])
        tracks["PB-PROBE"].append(decoded_record(peak, probe_k, "PB-PROBE"))
        tracks["PB-SELECTED"].append(decoded_record(peak, chosen, "PB-SELECTED"))
        tracks["PB-CANDIDATE-ORACLE"].append({**oracle, "track": "PB-CANDIDATE-ORACLE"})
    identities = [[(row["local_index"], row["raw_index"], row["true_count"]) for row in rows] for rows in tracks.values()]
    require(all(current == identities[0] for current in identities[1:]), "R5各轨样本身份不一致")
    output_dir = args.run_root / "05_selector_analysis"
    point = {}
    for name, rows in tracks.items():
        output = output_dir / f"{name.lower().replace('-', '_')}_samples.jsonl"
        write_jsonl(output, rows)
        gospa, recall = r3.point_track_values(rows)
        point[name] = {"mean_gospa_m": gospa, "recall_100m": recall}
    bootstrap = bootstrap_tracks(tracks)
    direct = (
        bootstrap["selected_relative_to_ob_ok"]["ci95"][1] <= 0.10
        and bootstrap["selected_recall_delta"]["ci95"][0] >= -0.05
    )
    stable_improvement = bootstrap["selected_minus_baseline_gospa"]["ci95"][1] < 0.0
    if direct:
        decision = "DIRECT_CASCADE_SUPPORTED"
    elif stable_improvement:
        decision = "ENTER_R5B_ORDINAL_HEAD"
    else:
        decision = "ENTER_COUNT_REPRESENTATION_DATA_DIAGNOSIS"
    truth = ordinal["source_count"].astype(np.int64)
    localization = np.isin(truth, LOCALIZATION_COUNTS)
    selected_count_accuracy = float(np.mean(selected_k[localization] == truth[localization]))
    report = {
        "status": "PASS", "gate": "S2-G5-R5-A", "stage": "selector_and_paired_analysis",
        "selector_fit": selector_fit, "selector": identity(selector_path),
        "selector_used": selector_used, "selector_fallback": selector_fallback,
        "localization_selected_count_accuracy": selected_count_accuracy,
        "track_point_metrics": point, "paired_bootstrap": bootstrap,
        "decision": decision,
        "decision_checks": {
            "direct_relative_gospa_ci_upper_at_most_0_10": bootstrap["selected_relative_to_ob_ok"]["ci95"][1] <= 0.10,
            "direct_recall_delta_ci_lower_at_least_minus_0_05": bootstrap["selected_recall_delta"]["ci95"][0] >= -0.05,
            "selected_gospa_stably_better_than_baseline": stable_improvement,
        },
        "scope": "fixed val_compare K2/3; frozen CH3/D8; no test; single training seed",
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
    }
    write_json(output_dir / "r5a_analysis.json", report)
    return report


def run_reload_audit(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    peak_differences = {}
    for split, index_path in (
        ("val_select", args.run_root / "03_select_fine_predicted" / "index.json"),
        ("val_compare", COMPARE_FINE_INDEX),
    ):
        saved = r3.load_jsonl(args.run_root / "04_d8_peaks" / f"{split}_top4.jsonl")
        replay = extract_peaks_for_split(index_path)
        difference = r3.nested_max_abs(saved, replay)
        require(difference <= 1e-12, f"{split} D8独立重载峰值不一致: {difference}")
        peak_differences[split] = difference

    selector = joblib.load(args.run_root / "05_selector_analysis" / "d8_selector.joblib")
    ordinal = np.load(args.run_root / "02_ordinal_probe" / "val_compare_ordinal.npz")
    peaks = load_peak_map(args.run_root / "04_d8_peaks" / "val_compare_top4.jsonl")
    saved_rows = r3.load_jsonl(args.run_root / "05_selector_analysis" / "pb_selected_samples.jsonl")
    saved_k = {int(row["local_index"]): int(row["requested_count"]) for row in saved_rows}
    replay_k = {}
    for local_index, truth in enumerate(ordinal["source_count"]):
        if int(truth) not in LOCALIZATION_COUNTS:
            continue
        candidates = ordinal["candidate_k"][local_index].astype(np.int64)
        peak = peaks[local_index]
        feature = selector_features(
            ordinal["k_probabilities"][local_index], ordinal["slot_scores"][local_index],
            candidates, peak["peak_positions_m"], peak["peak_scores"],
        )
        if feature is None:
            chosen = int(ordinal["top1_k"][local_index])
        else:
            chosen = int(candidates[int(selector.predict(feature[None, :])[0])])
        replay_k[local_index] = chosen
    require(saved_k == replay_k, "选择器独立重载结果不一致")
    report = {
        "status": "PASS", "gate": "S2-G5-R5-A", "stage": "independent_reload_audit",
        "d8_peak_reload_max_abs_difference": peak_differences,
        "selector_reload_prediction_exact": True,
        "selector_sample_count": len(replay_k),
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
    }
    write_json(args.run_root / "06_reload_audit" / "reload_audit.json", report)
    return report


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    required = {
        "preflight": args.run_root / "00_preflight" / "preflight.json",
        "inference": args.run_root / "01_frozen_logits" / "inference_report.json",
        "probe": args.run_root / "02_ordinal_probe" / "probe_report.json",
        "fine": args.run_root / "03_select_fine_predicted" / "index.json",
        "peaks": args.run_root / "04_d8_peaks" / "peaks_report.json",
        "analysis": args.run_root / "05_selector_analysis" / "r5a_analysis.json",
        "reload": args.run_root / "06_reload_audit" / "reload_audit.json",
    }
    payloads = {name: load_json(path) for name, path in required.items()}
    require(all(payload["status"] == "PASS" for payload in payloads.values()), "R5-A阶段状态未全部PASS")
    analysis = payloads["analysis"]
    report = {
        "status": "PASS", "gate": "S2-G5-R5-A", "experiment_id": "SYS-S2G5-R5A-20260828",
        "decision": analysis["decision"],
        "reports": {name: identity(path) for name, path in required.items()},
        "frozen_inputs": {"ch3": identity(CH3_CHECKPOINT), "d8": identity(D8_CHECKPOINT)},
        "test_executed": False, "r5b_training_executed": False,
    }
    write_json(args.run_root / "final_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("infer")
    sub.add_parser("probe")
    fine = sub.add_parser("build-select-fine")
    fine.add_argument("--chunk_size", type=int, default=40000)
    fine.add_argument("--shard_size", type=int, default=64)
    sub.add_parser("peaks")
    sub.add_parser("select-analyze")
    sub.add_parser("reload-audit")
    sub.add_parser("finalize")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.run_root = args.run_root.resolve()
    commands = {
        "preflight": run_preflight,
        "infer": run_infer,
        "probe": run_probe,
        "build-select-fine": build_select_fine,
        "peaks": run_peaks,
        "select-analyze": run_select_analyze,
        "reload-audit": run_reload_audit,
        "finalize": run_finalize,
    }
    result = commands[args.command](args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
