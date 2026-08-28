"""S2-G5-R2嵌套数据、CH3规模比较与8k决策工具。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from s2g1_train_ch3 import SourceDetectionDataset, SourceDetectionNet


EXPECTED_COUNTS = {0: 1024, 1: 1024, 2: 1024, 3: 1024}
VAL_COUNTS = {0: 512, 1: 512, 2: 512, 3: 512}
MANIFEST_SEED = 20260829
BOOTSTRAP_SEED = 20260830
EIGHT_K_COUNTS = {0: 2048, 1: 2048, 2: 2048, 3: 2048}
COARSE_SAMPLE_FIELDS = {
    "mtr_sub_all",
    "src_count_all",
    "band_mask_all",
    "ignore_mask_all",
    "avg_snr_all",
    "fc_offset_all",
    "src_pos_all",
    "symbolRate_all",
    "BW_actual_all",
    "sample_idx_all",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_counts(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        return np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1)


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(map(int, values)).items())}


def stratified_pick(
    counts: np.ndarray,
    per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    groups: dict[int, np.ndarray] = {}
    selected = []
    for source_count in range(4):
        candidates = np.flatnonzero(counts == source_count)
        require(candidates.size >= per_class, f"K={source_count}样本不足")
        shuffled = rng.permutation(candidates)
        groups[source_count] = shuffled
        selected.extend(map(int, shuffled[:per_class]))
    return np.asarray(sorted(selected), dtype=np.int64), groups


def subset_entry(indices: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    return {
        "sample_count": int(indices.size),
        "source_count_histogram": histogram(counts[indices]),
        "indices": indices.tolist(),
    }


def build_manifest(train_path: Path, val_path: Path) -> dict[str, Any]:
    train_counts = source_counts(train_path)
    val_counts = source_counts(val_path)
    require(train_counts.size == 4096, "train pool必须为4096条")
    require(val_counts.size == 2048, "validation pool必须为2048条")
    require(Counter(map(int, train_counts)) == Counter(EXPECTED_COUNTS), "train源数分层错误")
    require(Counter(map(int, val_counts)) == Counter(VAL_COUNTS), "validation源数分层错误")

    rng = np.random.default_rng(MANIFEST_SEED)
    train_1k, _ = stratified_pick(train_counts, 256, rng)
    train_4k = np.arange(train_counts.size, dtype=np.int64)
    val_select_parts = []
    val_compare_parts = []
    for source_count in range(4):
        candidates = rng.permutation(np.flatnonzero(val_counts == source_count))
        val_select_parts.extend(map(int, candidates[:256]))
        val_compare_parts.extend(map(int, candidates[256:512]))
    val_select = np.asarray(sorted(val_select_parts), dtype=np.int64)
    val_compare = np.asarray(sorted(val_compare_parts), dtype=np.int64)

    require(set(train_1k).issubset(set(train_4k)), "train_1k不是train_4k子集")
    require(set(val_select).isdisjoint(set(val_compare)), "两个validation子集重叠")
    return {
        "status": "PASS",
        "gate": "S2-G5-R2",
        "manifest_seed": MANIFEST_SEED,
        "index_base": 0,
        "train_source": str(train_path.resolve()),
        "train_source_sha256": sha256_file(train_path),
        "val_source": str(val_path.resolve()),
        "val_source_sha256": sha256_file(val_path),
        "train_1k": subset_entry(train_1k, train_counts),
        "train_4k": subset_entry(train_4k, train_counts),
        "val_select": subset_entry(val_select, val_counts),
        "val_compare": subset_entry(val_compare, val_counts),
        "invariants": {
            "train_1k_subset_of_train_4k": True,
            "validation_disjoint": True,
            "all_subsets_balanced_k0_k3": True,
        },
    }


def audit_coarse(
    path: Path,
    expected_samples: int,
    expected_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        required = {
            "mtr_sub_all",
            "src_count_all",
            "band_mask_all",
            "ignore_mask_all",
            "sample_idx_all",
            "N_sub_val",
            "max_src_val",
            "thresh_val",
        }
        missing = sorted(required.difference(handle.keys()))
        require(not missing, f"{path.name}缺少字段: {missing}")
        spectra = handle["mtr_sub_all"]
        require(spectra.shape == (81, 81, 19, expected_samples), f"粗DPD shape错误: {spectra.shape}")
        for start in range(0, expected_samples, 64):
            block = np.asarray(spectra[..., start : start + 64], dtype=np.float32)
            require(np.isfinite(block).all(), f"粗DPD在样本{start}附近含NaN/Inf")
        counts = np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1)
        band = np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0)
        ignore = np.asarray(handle["ignore_mask_all"], dtype=np.float32).transpose(2, 1, 0)
        sample_indices = np.asarray(handle["sample_idx_all"], dtype=np.int64).reshape(-1)
        require(counts.size == expected_samples, "源数样本数错误")
        require(band.shape == (expected_samples, 3, 19), "band shape错误")
        require(ignore.shape == band.shape, "ignore shape错误")
        if expected_indices is None:
            expected_indices = np.arange(expected_samples, dtype=np.int64)
        require(np.array_equal(sample_indices, expected_indices), "sample_idx与期望索引不一致")
        require(not np.any((band > 0) & (ignore > 0)), "band与ignore重叠")
        active = np.arange(3)[None, :] < counts[:, None]
        require(np.all(band.sum(axis=2)[active] >= 1), "存在无正子带的有效源")
        require(not np.any(band[~active]) and not np.any(ignore[~active]), "空槽标签非零")
        require(abs(float(np.asarray(handle["thresh_val"]).reshape(-1)[0]) - 0.2) < 1e-6, "阈值错误")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "samples": expected_samples,
        "source_count_histogram": histogram(counts),
        "shape": [expected_samples, 19, 81, 81],
        "status": "PASS",
    }


def audit_subset(path: Path, manifest: dict[str, Any], subset_name: str) -> dict[str, Any]:
    expected = np.asarray(manifest[subset_name]["indices"], dtype=np.int64)
    report = audit_coarse(path, expected.size, expected)
    with h5py.File(path, "r") as handle:
        observed = np.asarray(handle["sample_idx_all"], dtype=np.int64).reshape(-1)
    require(np.array_equal(observed, expected), f"{subset_name}索引与manifest不一致")
    require(report["source_count_histogram"] == manifest[subset_name]["source_count_histogram"], f"{subset_name}分层错误")
    report["subset_name"] = subset_name
    report["indices_match_manifest"] = True
    return report


def _dataset_creation_options(dataset: h5py.Dataset, shape: tuple[int, ...]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if dataset.chunks is not None:
        options["chunks"] = tuple(min(chunk, size) for chunk, size in zip(dataset.chunks, shape, strict=True))
    if dataset.compression is not None:
        options["compression"] = dataset.compression
        options["compression_opts"] = dataset.compression_opts
    if dataset.shuffle:
        options["shuffle"] = True
    if dataset.fletcher32:
        options["fletcher32"] = True
    return options


def _copy_attrs(source: h5py.Dataset, target: h5py.Dataset) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _arrays_equal_chunked(
    left: h5py.Dataset,
    right: h5py.Dataset,
    samples: int,
    block_size: int = 32,
) -> bool:
    for start in range(0, samples, block_size):
        stop = min(start + block_size, samples)
        selection = (slice(None),) * (left.ndim - 1) + (slice(start, stop),)
        if not np.array_equal(left[selection], right[selection]):
            return False
    return True


def extend_coarse_to_8k(
    old_path: Path,
    added_path: Path,
    output_path: Path,
    block_size: int = 32,
) -> dict[str, Any]:
    require(old_path.is_file(), f"旧4k粗DPD不存在: {old_path}")
    require(added_path.is_file(), f"新增4k粗DPD不存在: {added_path}")
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖: {output_path}")
    old_audit = audit_coarse(old_path, 4096)
    added_audit = audit_coarse(added_path, 4096)
    require(old_audit["source_count_histogram"] == {str(key): 1024 for key in range(4)}, "旧4k分层错误")
    require(added_audit["source_count_histogram"] == {str(key): 1024 for key in range(4)}, "新增4k分层错误")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scientific_static_fields = {
        "N_sub_val",
        "max_src_val",
        "thresh_val",
        "B_win_val",
        "B_step_val",
        "fs_val",
        "edge_val",
        "lamda_val",
        "num_count_classes",
        "num_grid_val",
        "sub_f_lo_val",
        "sub_f_hi_val",
        "s2g3_algorithm",
    }
    with h5py.File(old_path, "r") as old, h5py.File(added_path, "r") as added:
        require(COARSE_SAMPLE_FIELDS.issubset(old.keys()), "旧4k缺少样本字段")
        require(COARSE_SAMPLE_FIELDS.issubset(added.keys()), "新增4k缺少样本字段")
        for name in scientific_static_fields:
            require(name in old and name in added, f"静态契约字段缺失: {name}")
            require(np.array_equal(old[name][...], added[name][...]), f"新旧粗DPD契约不一致: {name}")
        with h5py.File(output_path, "w") as output:
            for key, value in old.attrs.items():
                output.attrs[key] = value
            for name, dataset in old.items():
                if not isinstance(dataset, h5py.Dataset):
                    old.copy(name, output)
                    continue
                if name not in COARSE_SAMPLE_FIELDS:
                    old.copy(name, output)
                    continue
                added_dataset = added[name]
                require(dataset.shape[:-1] == added_dataset.shape[:-1], f"字段{name}非样本维不一致")
                require(dataset.dtype == added_dataset.dtype, f"字段{name} dtype不一致")
                combined_shape = dataset.shape[:-1] + (8192,)
                target = output.create_dataset(
                    name,
                    shape=combined_shape,
                    dtype=dataset.dtype,
                    **_dataset_creation_options(dataset, combined_shape),
                )
                _copy_attrs(dataset, target)
                if name == "sample_idx_all":
                    target[...] = np.arange(8192, dtype=dataset.dtype).reshape(dataset.shape[:-1] + (8192,))
                    continue
                for source, offset in ((dataset, 0), (added_dataset, 4096)):
                    for start in range(0, 4096, block_size):
                        stop = min(start + block_size, 4096)
                        source_selection = (slice(None),) * (source.ndim - 1) + (slice(start, stop),)
                        target_selection = (slice(None),) * (target.ndim - 1) + (
                            slice(offset + start, offset + stop),
                        )
                        target[target_selection] = source[source_selection]

    combined_audit = audit_coarse(output_path, 8192)
    require(combined_audit["source_count_histogram"] == {str(key): 2048 for key in range(4)}, "8k分层错误")
    with h5py.File(old_path, "r") as old, h5py.File(output_path, "r") as combined:
        prefix_fields = {}
        for name in sorted(COARSE_SAMPLE_FIELDS - {"sample_idx_all"}):
            prefix_fields[name] = _arrays_equal_chunked(old[name], combined[name], 4096, block_size)
        require(all(prefix_fields.values()), "8k前4096条未严格保持旧4k")
    return {
        "status": "PASS",
        "gate": "S2-G5-R2-8K",
        "old_4k": old_audit,
        "added_4k": added_audit,
        "combined_8k": combined_audit,
        "strict_prefix_match": True,
        "prefix_field_checks": prefix_fields,
        "lineage": {
            "old_4k_indices": [0, 4095],
            "added_4k_indices": [4096, 8191],
            "combined_sample_idx": [0, 8191],
        },
    }


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def initialization_identity(seed: int = 42) -> dict[str, Any]:
    hashes = []
    for _ in range(2):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = SourceDetectionNet(n_sub=19, max_src=10, mode="transformer")
        hashes.append(state_sha256(model))
    require(hashes[0] == hashes[1], "相同seed的模型初始化hash不一致")
    return {"status": "PASS", "seed": seed, "initial_model_sha256": hashes[0]}


@torch.no_grad()
def infer(checkpoint_path: Path, data_path: Path, batch_size: int = 64) -> dict[str, np.ndarray]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset = SourceDetectionDataset(
        data_path,
        augment=False,
        normalize="sample_zscore",
        max_src_override=int(config["max_src"]),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = SourceDetectionNet(
        n_sub=int(config["n_sub"]),
        max_src=int(config["max_src"]),
        mode=str(config["mode"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    predictions = []
    truths = []
    ignores = []
    counts = []
    for spectra, source_count, band_mask, ignore_mask in loader:
        logits = model(spectra.to(device, non_blocking=True))
        require(torch.isfinite(logits).all().item(), "推理logits含NaN/Inf")
        predictions.append((torch.sigmoid(logits) > float(config["threshold"])).cpu().numpy())
        truths.append((band_mask > 0.5).numpy())
        ignores.append((ignore_mask > 0.5).numpy())
        counts.append(source_count.numpy())
    return {
        "prediction": np.concatenate(predictions),
        "truth": np.concatenate(truths),
        "ignore": np.concatenate(ignores),
        "source_count": np.concatenate(counts).astype(np.int64),
    }


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / max(2 * tp + fp + fn, 1)


def metrics(arrays: dict[str, np.ndarray], indices: np.ndarray | None = None) -> dict[str, Any]:
    prediction = arrays["prediction"] if indices is None else arrays["prediction"][indices]
    truth = arrays["truth"] if indices is None else arrays["truth"][indices]
    ignore = arrays["ignore"] if indices is None else arrays["ignore"][indices]
    source_count = arrays["source_count"] if indices is None else arrays["source_count"][indices]
    pred_count = prediction.any(axis=2).sum(axis=1)
    class_accuracy = {}
    for count in range(4):
        mask = source_count == count
        require(mask.any(), f"评估缺少K={count}")
        class_accuracy[str(count)] = float(np.mean(pred_count[mask] == count))
    valid = ~ignore
    slot_f1 = []
    for slot in range(3):
        active = source_count > slot
        current_pred = prediction[active, slot][valid[active, slot]]
        current_truth = truth[active, slot][valid[active, slot]]
        tp = int(np.sum(current_pred & current_truth))
        fp = int(np.sum(current_pred & ~current_truth))
        fn = int(np.sum(~current_pred & current_truth))
        slot_f1.append(f1(tp, fp, fn))
    active_ious = []
    exact_matches = []
    for sample_idx, count in enumerate(source_count):
        for slot in range(int(count)):
            current_valid = valid[sample_idx, slot]
            pred = prediction[sample_idx, slot, current_valid]
            target = truth[sample_idx, slot, current_valid]
            union = np.sum(pred | target)
            active_ious.append(float(np.sum(pred & target) / max(union, 1)))
            exact_matches.append(bool(np.array_equal(pred, target)))
    zero = source_count == 0
    inactive = np.arange(prediction.shape[1])[None, :] >= source_count[:, None]
    return {
        "sample_count": int(source_count.size),
        "balanced_count_accuracy": float(np.mean(list(class_accuracy.values()))),
        "count_class_accuracy": class_accuracy,
        "count_accuracy": float(np.mean(pred_count == source_count)),
        "count_mae": float(np.mean(np.abs(pred_count - source_count))),
        "under_count_rate": float(np.mean(pred_count < source_count)),
        "over_count_rate": float(np.mean(pred_count > source_count)),
        "zero_source_false_alarm_rate": float(np.mean(pred_count[zero] > 0)),
        "active_band_macro_f1": float(np.mean(slot_f1)),
        "active_source_mean_iou": float(np.mean(active_ious)),
        "active_source_exact_match_rate": float(np.mean(exact_matches)),
        "inactive_slot_activation_rate": float(np.mean(prediction.any(axis=2)[inactive])),
    }


def paired_bootstrap(
    one_k: dict[str, np.ndarray],
    four_k: dict[str, np.ndarray],
    repetitions: int = 2000,
) -> dict[str, Any]:
    counts = one_k["source_count"]
    require(np.array_equal(counts, four_k["source_count"]), "两模型评估样本顺序不一致")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas = {"balanced_count_accuracy": [], "active_band_macro_f1": []}
    groups = [np.flatnonzero(counts == count) for count in range(4)]
    for _ in range(repetitions):
        sampled = np.concatenate([rng.choice(group, size=group.size, replace=True) for group in groups])
        metrics_1k = metrics(one_k, sampled)
        metrics_4k = metrics(four_k, sampled)
        for key in deltas:
            deltas[key].append(metrics_4k[key] - metrics_1k[key])
    return {
        key: {
            "mean_delta_4k_minus_1k": float(np.mean(values)),
            "ci95": [float(value) for value in np.percentile(values, [2.5, 97.5])],
        }
        for key, values in deltas.items()
    }


def paired_bootstrap_8k(
    four_k: dict[str, np.ndarray],
    eight_k: dict[str, np.ndarray],
    repetitions: int = 2000,
) -> dict[str, Any]:
    counts = four_k["source_count"]
    require(np.array_equal(counts, eight_k["source_count"]), "4k/8k评估样本顺序不一致")
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    deltas = {"balanced_count_accuracy": [], "active_band_macro_f1": []}
    groups = [np.flatnonzero(counts == count) for count in range(4)]
    for _ in range(repetitions):
        sampled = np.concatenate([rng.choice(group, size=group.size, replace=True) for group in groups])
        baseline = metrics(four_k, sampled)
        candidate = metrics(eight_k, sampled)
        for key in deltas:
            deltas[key].append(candidate[key] - baseline[key])
    return {
        key: {
            "mean_delta_8k_minus_4k": float(np.mean(values)),
            "ci95": [float(value) for value in np.percentile(values, [2.5, 97.5])],
        }
        for key, values in deltas.items()
    }


def decide_8k(
    summary_8k: dict[str, Any],
    metrics_4k: dict[str, Any],
    metrics_8k: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    delta_count = metrics_8k["balanced_count_accuracy"] - metrics_4k["balanced_count_accuracy"]
    delta_band = metrics_8k["active_band_macro_f1"] - metrics_4k["active_band_macro_f1"]
    class_deltas = {
        key: metrics_8k["count_class_accuracy"][key] - metrics_4k["count_class_accuracy"][key]
        for key in metrics_4k["count_class_accuracy"]
    }
    zero_fpr_delta = metrics_8k["zero_source_false_alarm_rate"] - metrics_4k["zero_source_false_alarm_rate"]
    ci_count = bootstrap["balanced_count_accuracy"]["ci95"]
    ci_band = bootstrap["active_band_macro_f1"]["ci95"]
    learning_pass = bool(summary_8k["learning_gate"]["pass"])
    convergence_pass = bool(summary_8k["convergence_gate"]["pass"])
    both_nonnegative = delta_count >= 0 and delta_band >= 0
    material_gain = max(delta_count, delta_band) >= 0.01
    supported_gain = ci_count[0] > 0 or ci_band[0] > 0
    other_ci_noninferior = (ci_count[0] > 0 and ci_band[0] >= -0.01) or (
        ci_band[0] > 0 and ci_count[0] >= -0.01
    )
    no_class_collapse = min(class_deltas.values()) >= -0.05
    no_zero_regression = zero_fpr_delta <= 0.02

    if not learning_pass:
        decision = "STOP_8K_MODEL_OR_TRAINING_DIAGNOSIS"
    elif not convergence_pass:
        decision = "STOP_8K_TRAINING_SCHEDULE_DIAGNOSIS"
    elif (
        both_nonnegative
        and material_gain
        and supported_gain
        and other_ci_noninferior
        and no_class_collapse
        and no_zero_regression
    ):
        decision = "FREEZE_8K_FOR_R3"
    elif not no_class_collapse or not no_zero_regression:
        decision = "KEEP_4K_SCALE_REGRESSION"
    elif delta_count <= 0 and delta_band <= 0:
        decision = "KEEP_4K_FOR_R3"
    else:
        decision = "INCONCLUSIVE_SCALE_USE_4K_FOR_R3"
    return {
        "decision": decision,
        "delta_8k_minus_4k": {
            "balanced_count_accuracy": delta_count,
            "active_band_macro_f1": delta_band,
            "count_class_accuracy": class_deltas,
            "zero_source_false_alarm_rate": zero_fpr_delta,
        },
        "checks": {
            "learning_pass": learning_pass,
            "convergence_pass": convergence_pass,
            "both_primary_nonnegative": both_nonnegative,
            "at_least_one_gain_1pp": material_gain,
            "paired_ci_supports_one_primary": supported_gain,
            "other_primary_ci_noninferior_1pp": other_ci_noninferior,
            "no_class_drop_over_5pp": no_class_collapse,
            "zero_fpr_increase_not_over_2pp": no_zero_regression,
        },
    }


def decide(
    summary_4k: dict[str, Any],
    metrics_1k: dict[str, Any],
    metrics_4k: dict[str, Any],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    delta_count = metrics_4k["balanced_count_accuracy"] - metrics_1k["balanced_count_accuracy"]
    delta_band = metrics_4k["active_band_macro_f1"] - metrics_1k["active_band_macro_f1"]
    class_deltas = {
        key: metrics_4k["count_class_accuracy"][key] - metrics_1k["count_class_accuracy"][key]
        for key in metrics_1k["count_class_accuracy"]
    }
    zero_fpr_delta = metrics_4k["zero_source_false_alarm_rate"] - metrics_1k["zero_source_false_alarm_rate"]
    learning_pass = bool(summary_4k["learning_gate"]["pass"])
    convergence_pass = bool(summary_4k["convergence_gate"]["pass"])
    both_improve = delta_count > 0 and delta_band > 0
    material_point_gain = max(delta_count, delta_band) >= 0.02
    no_class_collapse = min(class_deltas.values()) >= -0.05
    no_zero_regression = zero_fpr_delta <= 0.02
    ci_count = bootstrap["balanced_count_accuracy"]["ci95"]
    ci_band = bootstrap["active_band_macro_f1"]["ci95"]
    one_supported = ci_count[0] > 0 or ci_band[0] > 0
    other_noninferior = (ci_count[0] > 0 and delta_band > -0.01) or (ci_band[0] > 0 and delta_count > -0.01)
    near_plateau = abs(delta_count) < 0.01 and abs(delta_band) < 0.01
    ci_cross_zero = ci_count[0] <= 0 <= ci_count[1] and ci_band[0] <= 0 <= ci_band[1]
    interface_reference = (
        metrics_4k["balanced_count_accuracy"] >= 0.80
        and metrics_4k["active_band_macro_f1"] >= 0.80
    )

    if not learning_pass:
        decision = "STOP_MODEL_OR_TRAINING_DIAGNOSIS"
    elif not convergence_pass:
        decision = "STOP_TRAINING_SCHEDULE_DIAGNOSIS"
    elif no_class_collapse and no_zero_regression and (
        (both_improve and material_point_gain) or (one_supported and other_noninferior)
    ):
        decision = "ENTER_8K"
    elif near_plateau and ci_cross_zero and interface_reference:
        decision = "PROCEED_R3_NO_8K"
    elif near_plateau and ci_cross_zero:
        decision = "STOP_MODEL_LABEL_LOSS_DIAGNOSIS"
    elif delta_count <= -0.02 or delta_band <= -0.02:
        decision = "STOP_SCALE_REGRESSION_DIAGNOSIS"
    else:
        decision = "REVIEW_MIXED_NO_AUTO_8K"
    return {
        "decision": decision,
        "delta_4k_minus_1k": {
            "balanced_count_accuracy": delta_count,
            "active_band_macro_f1": delta_band,
            "count_class_accuracy": class_deltas,
            "zero_source_false_alarm_rate": zero_fpr_delta,
        },
        "checks": {
            "learning_pass": learning_pass,
            "convergence_pass": convergence_pass,
            "both_primary_improve": both_improve,
            "material_point_gain": material_point_gain,
            "no_class_drop_over_5pp": no_class_collapse,
            "zero_fpr_increase_not_over_2pp": no_zero_regression,
            "paired_ci_supports_one_primary": one_supported,
            "other_primary_noninferior_1pp": other_noninferior,
            "near_plateau_both_under_1pp": near_plateau,
            "interface_reference_0p80": interface_reference,
        },
    }


def write_predictions(path: Path, arrays_1k: dict[str, np.ndarray], arrays_4k: dict[str, np.ndarray]) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, count in enumerate(arrays_1k["source_count"]):
            row = {
                "sample_index": index,
                "source_count": int(count),
                "predicted_count_1k": int(arrays_1k["prediction"][index].any(axis=1).sum()),
                "predicted_count_4k": int(arrays_4k["prediction"][index].any(axis=1).sum()),
                "true_bands": [np.flatnonzero(slot).tolist() for slot in arrays_1k["truth"][index, : int(count)]],
                "predicted_bands_1k": [np.flatnonzero(slot).tolist() for slot in arrays_1k["prediction"][index]],
                "predicted_bands_4k": [np.flatnonzero(slot).tolist() for slot in arrays_4k["prediction"][index]],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_predictions_8k(
    path: Path,
    arrays_4k: dict[str, np.ndarray],
    arrays_8k: dict[str, np.ndarray],
) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, count in enumerate(arrays_4k["source_count"]):
            row = {
                "sample_index": index,
                "source_count": int(count),
                "predicted_count_4k": int(arrays_4k["prediction"][index].any(axis=1).sum()),
                "predicted_count_8k": int(arrays_8k["prediction"][index].any(axis=1).sum()),
                "true_bands": [
                    np.flatnonzero(slot).tolist() for slot in arrays_4k["truth"][index, : int(count)]
                ],
                "predicted_bands_4k": [
                    np.flatnonzero(slot).tolist() for slot in arrays_4k["prediction"][index]
                ],
                "predicted_bands_8k": [
                    np.flatnonzero(slot).tolist() for slot in arrays_8k["prediction"][index]
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_finalize(run_root: Path) -> dict[str, Any]:
    manifest = load_json(run_root / "manifest" / "data_manifest.json")
    final_report = load_json(run_root / "analysis" / "final_report.json")
    reload_1k = load_json(run_root / "analysis" / "reload_1k.json")
    reload_4k = load_json(run_root / "analysis" / "reload_4k.json")
    identity = load_json(run_root / "audit" / "initialization_identity.json")
    require(identity["status"] == "PASS", "初始化身份检查未通过")
    require(sha256_file(Path(manifest["train_source"])) == manifest["train_source_sha256"], "train粗DPD训练后hash变化")
    require(sha256_file(Path(manifest["val_source"])) == manifest["val_source_sha256"], "validation粗DPD训练后hash变化")
    for reload, final_metrics, name in (
        (reload_1k, final_report["metrics_1k"], "1k"),
        (reload_4k, final_report["metrics_4k"], "4k"),
    ):
        for key in ("balanced_count_accuracy", "active_band_macro_f1"):
            require(reload["metrics"][key] == final_metrics[key], f"{name}独立重载{key}不一致")

    expected_stages = [
        "01_matlab_iq",
        "02_coarse_dpd",
        "03_materialize",
        "04_capacity",
        "05_train_1k",
        "06_train_4k",
        "07b_reload_compare",
        "08_analysis",
    ]
    monitor_summary = {}
    for stage in expected_stages:
        report = load_json(run_root / "monitor" / stage / "stage_monitor_report.json")
        require(report["status"] == "PASS", f"监控阶段未通过: {stage}")
        require(not report["red_flags"], f"监控阶段存在red flag: {stage}")
        monitor_summary[stage] = {
            "status": report["status"],
            "duration_seconds": report["duration_seconds"],
            "warning_count": len(report["warnings"]),
            "red_flag_count": len(report["red_flags"]),
            "minimum_system_available_bytes": report["minimum_system_available_bytes"],
            "maximum_process_tree_rss_bytes": report["maximum_process_tree_rss_bytes"],
            "maximum_gpu_used_mib": report["maximum_gpu_used_mib"],
        }
    failed_attempt = load_json(run_root / "monitor" / "07_reload_compare" / "stage_monitor_report.json")
    require(failed_attempt["status"] == "CRASHED", "预期保留的参数拼写失败状态错误")

    logical_bytes = 0
    unique_bytes = 0
    identities = set()
    file_count = 0
    for path in run_root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        logical_bytes += stat.st_size
        file_count += 1
        identity_key = (stat.st_dev, stat.st_ino)
        if identity_key not in identities:
            identities.add(identity_key)
            unique_bytes += stat.st_size
    return {
        "status": "PASS",
        "gate": "S2-G5-R2",
        "decision": final_report["decision"]["decision"],
        "data_hash_unchanged_after_training": True,
        "checkpoint_reload_exact_for_primary_metrics": True,
        "initial_model_sha256": identity["initial_model_sha256"],
        "monitor_stages": monitor_summary,
        "retained_failed_attempt": {
            "stage": "07_reload_compare",
            "status": "CRASHED",
            "cause": "outer command used evaluate instead of existing eval subcommand",
            "scientific_outputs_written": False,
        },
        "file_count": file_count,
        "logical_bytes_including_hardlinks": logical_bytes,
        "unique_file_bytes": unique_bytes,
        "environment": {
            "python": os.sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "scope": "single-seed validation scale diagnosis; no test and no automatic 8k run",
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    arrays_1k = infer(args.checkpoint_1k, args.val_compare)
    arrays_4k = infer(args.checkpoint_4k, args.val_compare)
    require(np.array_equal(arrays_1k["truth"], arrays_4k["truth"]), "评估真值不一致")
    metrics_1k = metrics(arrays_1k)
    metrics_4k = metrics(arrays_4k)
    bootstrap = paired_bootstrap(arrays_1k, arrays_4k, args.bootstrap_repetitions)
    summary_1k = load_json(args.summary_1k)
    summary_4k = load_json(args.summary_4k)
    write_predictions(args.predictions, arrays_1k, arrays_4k)
    result = {
        "status": "PASS",
        "gate": "S2-G5-R2",
        "evidence_scope": "single-seed validation scale diagnosis; not test or paper performance",
        "checkpoint_1k": str(args.checkpoint_1k.resolve()),
        "checkpoint_1k_sha256": sha256_file(args.checkpoint_1k),
        "checkpoint_4k": str(args.checkpoint_4k.resolve()),
        "checkpoint_4k_sha256": sha256_file(args.checkpoint_4k),
        "metrics_1k": metrics_1k,
        "metrics_4k": metrics_4k,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "paired_bootstrap": bootstrap,
        "decision": decide(summary_4k, metrics_1k, metrics_4k, bootstrap),
        "training_summary_1k": summary_1k,
        "training_summary_4k": summary_4k,
        "predictions_jsonl": str(args.predictions.resolve()),
    }
    return result


def run_compare_8k(args: argparse.Namespace) -> dict[str, Any]:
    arrays_4k = infer(args.checkpoint_4k, args.val_compare)
    arrays_8k = infer(args.checkpoint_8k, args.val_compare)
    require(np.array_equal(arrays_4k["truth"], arrays_8k["truth"]), "4k/8k评估真值不一致")
    require(np.array_equal(arrays_4k["ignore"], arrays_8k["ignore"]), "4k/8k ignore不一致")
    metrics_4k = metrics(arrays_4k)
    metrics_8k = metrics(arrays_8k)
    r2_report = load_json(args.r2_report)
    for key in ("balanced_count_accuracy", "active_band_macro_f1"):
        require(metrics_4k[key] == r2_report["metrics_4k"][key], f"4k重评指标漂移: {key}")
    bootstrap = paired_bootstrap_8k(arrays_4k, arrays_8k, args.bootstrap_repetitions)
    summary_8k = load_json(args.summary_8k)
    write_predictions_8k(args.predictions, arrays_4k, arrays_8k)
    return {
        "status": "PASS",
        "gate": "S2-G5-R2-8K",
        "evidence_scope": "single-seed fixed-validation 4k-to-8k scale diagnosis; not test or paper performance",
        "checkpoint_4k": str(args.checkpoint_4k.resolve()),
        "checkpoint_4k_sha256": sha256_file(args.checkpoint_4k),
        "checkpoint_8k": str(args.checkpoint_8k.resolve()),
        "checkpoint_8k_sha256": sha256_file(args.checkpoint_8k),
        "r2_4k_metrics_exactly_reproduced": True,
        "metrics_4k": metrics_4k,
        "metrics_8k": metrics_8k,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": BOOTSTRAP_SEED + 1,
        "paired_bootstrap": bootstrap,
        "decision": decide_8k(summary_8k, metrics_4k, metrics_8k, bootstrap),
        "training_summary_8k": summary_8k,
        "predictions_jsonl": str(args.predictions.resolve()),
    }


def run_finalize_8k(run_root: Path) -> dict[str, Any]:
    extension = load_json(run_root / "audit" / "8k" / "build_8k_report.json")
    comparison = load_json(run_root / "analysis" / "8k" / "compare_4k_8k.json")
    reload_8k = load_json(run_root / "analysis" / "8k" / "reload_8k.json")
    identity = load_json(run_root / "audit" / "initialization_identity.json")
    r2_manifest = load_json(run_root / "manifest" / "data_manifest.json")
    subset_audit = load_json(run_root / "audit" / "coarse_subsets.json")
    run_config = load_json(run_root / "train_8k" / "run_config.json")
    require(identity["status"] == "PASS", "R2初始化身份检查未通过")
    require(extension["status"] == "PASS" and extension["strict_prefix_match"], "8k嵌套审计未通过")
    require(comparison["status"] == "PASS", "4k/8k配对比较未通过")
    require(
        sha256_file(Path(comparison["checkpoint_4k"])) == comparison["checkpoint_4k_sha256"],
        "4k checkpoint hash变化",
    )
    require(
        sha256_file(Path(comparison["checkpoint_8k"])) == comparison["checkpoint_8k_sha256"],
        "8k checkpoint hash变化",
    )
    for key in ("balanced_count_accuracy", "active_band_macro_f1"):
        require(
            np.isclose(
                reload_8k["metrics"][key],
                comparison["metrics_8k"][key],
                rtol=0.0,
                atol=1e-12,
            ),
            f"8k独立重载{key}不一致",
        )
    require(
        sha256_file(Path(r2_manifest["train_source"])) == r2_manifest["train_source_sha256"],
        "R2旧4k训练粗DPD hash变化",
    )
    require(
        sha256_file(Path(r2_manifest["val_source"])) == r2_manifest["val_source_sha256"],
        "R2 validation粗DPD hash变化",
    )
    for subset_name in ("val_select", "val_compare"):
        evidence = subset_audit[subset_name]
        require(sha256_file(Path(evidence["path"])) == evidence["sha256"], f"{subset_name} hash变化")
    for entry in ("old_4k", "added_4k", "combined_8k"):
        evidence = extension[entry]
        require(sha256_file(Path(evidence["path"])) == evidence["sha256"], f"{entry} hash变化")

    expected_config = {
        "mode": "transformer",
        "max_src": 10,
        "n_sub": 19,
        "epochs": 150,
        "batch_size": 64,
        "val_batch_size": 64,
        "num_workers": 0,
        "lr": 0.001,
        "min_lr": 0.000001,
        "weight_decay": 0.0005,
        "warmup_epochs": 5,
        "patience": 25,
        "gamma": 2.0,
        "threshold": 0.5,
        "grad_clip": 1.0,
        "seed": 42,
        "deterministic": True,
        "device": "cuda:0",
    }
    for key, expected in expected_config.items():
        require(run_config[key] == expected, f"8k训练配置漂移: {key}")

    expected_stages = [
        "09_iq_8k_add",
        "10_audit_iq_8k_add",
        "11_coarse_8k_add",
        "12_build_8k",
        "13_capacity_8k",
        "14_train_8k",
        "15_reload_8k",
        "16_compare_4k_8k",
    ]
    monitor_summary = {}
    for stage in expected_stages:
        report = load_json(run_root / "monitor" / stage / "stage_monitor_report.json")
        require(report["status"] == "PASS", f"8k监控阶段未通过: {stage}")
        require(not report["red_flags"], f"8k监控阶段存在red flag: {stage}")
        monitor_summary[stage] = {
            "status": report["status"],
            "duration_seconds": report["duration_seconds"],
            "warning_count": len(report["warnings"]),
            "red_flag_count": len(report["red_flags"]),
            "minimum_system_available_bytes": report["minimum_system_available_bytes"],
            "maximum_process_tree_rss_bytes": report["maximum_process_tree_rss_bytes"],
            "maximum_gpu_used_mib": report["maximum_gpu_used_mib"],
        }
    return {
        "status": "PASS",
        "gate": "S2-G5-R2-8K",
        "decision": comparison["decision"]["decision"],
        "strict_train_4k_prefix_of_train_8k": True,
        "r2_data_hash_unchanged": True,
        "fixed_validation_hash_unchanged": True,
        "checkpoint_hash_unchanged": True,
        "training_config_exactly_matches_preregistration": True,
        "checkpoint_reload_consistent_for_primary_metrics": True,
        "checkpoint_reload_absolute_tolerance": 1e-12,
        "r2_4k_metrics_exactly_reproduced": True,
        "initial_model_sha256": identity["initial_model_sha256"],
        "monitor_stages": monitor_summary,
        "scope": "single training seed and fixed validation; no test, R3, D8, K=1 locator, extra seed, or 16k",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--train_coarse", type=Path, required=True)
    manifest.add_argument("--val_coarse", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    audit = subparsers.add_parser("audit-subsets")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--train_1k", type=Path, required=True)
    audit.add_argument("--train_4k", type=Path, required=True)
    audit.add_argument("--val_select", type=Path, required=True)
    audit.add_argument("--val_compare", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    identity = subparsers.add_parser("initialization-identity")
    identity.add_argument("--output", type=Path, required=True)
    identity.add_argument("--seed", type=int, default=42)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--checkpoint_1k", type=Path, required=True)
    compare.add_argument("--checkpoint_4k", type=Path, required=True)
    compare.add_argument("--summary_1k", type=Path, required=True)
    compare.add_argument("--summary_4k", type=Path, required=True)
    compare.add_argument("--val_compare", type=Path, required=True)
    compare.add_argument("--predictions", type=Path, required=True)
    compare.add_argument("--bootstrap_repetitions", type=int, default=2000)
    compare.add_argument("--output", type=Path, required=True)

    extend_8k = subparsers.add_parser("extend-8k")
    extend_8k.add_argument("--old_4k", type=Path, required=True)
    extend_8k.add_argument("--added_4k", type=Path, required=True)
    extend_8k.add_argument("--combined_8k", type=Path, required=True)
    extend_8k.add_argument("--block_size", type=int, default=32)
    extend_8k.add_argument("--output", type=Path, required=True)

    compare_8k = subparsers.add_parser("compare-8k")
    compare_8k.add_argument("--checkpoint_4k", type=Path, required=True)
    compare_8k.add_argument("--checkpoint_8k", type=Path, required=True)
    compare_8k.add_argument("--summary_8k", type=Path, required=True)
    compare_8k.add_argument("--val_compare", type=Path, required=True)
    compare_8k.add_argument("--r2_report", type=Path, required=True)
    compare_8k.add_argument("--predictions", type=Path, required=True)
    compare_8k.add_argument("--bootstrap_repetitions", type=int, default=2000)
    compare_8k.add_argument("--output", type=Path, required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--run_root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)

    finalize_8k = subparsers.add_parser("finalize-8k")
    finalize_8k.add_argument("--run_root", type=Path, required=True)
    finalize_8k.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "manifest":
        result = build_manifest(args.train_coarse, args.val_coarse)
    elif args.command == "audit-subsets":
        manifest = load_json(args.manifest)
        result = {
            "status": "PASS",
            "train_1k": audit_subset(args.train_1k, manifest, "train_1k"),
            "train_4k": audit_subset(args.train_4k, manifest, "train_4k"),
            "val_select": audit_subset(args.val_select, manifest, "val_select"),
            "val_compare": audit_subset(args.val_compare, manifest, "val_compare"),
        }
    elif args.command == "initialization-identity":
        result = initialization_identity(args.seed)
    elif args.command == "compare":
        result = run_compare(args)
    elif args.command == "extend-8k":
        require(args.block_size > 0, "block_size必须为正整数")
        result = extend_coarse_to_8k(args.old_4k, args.added_4k, args.combined_8k, args.block_size)
    elif args.command == "compare-8k":
        result = run_compare_8k(args)
    elif args.command == "finalize":
        result = run_finalize(args.run_root.resolve())
    elif args.command == "finalize-8k":
        result = run_finalize_8k(args.run_root.resolve())
    else:
        raise AssertionError(f"未知命令: {args.command}")
    write_json(args.output, result)
    display_result = result
    if args.command == "manifest":
        display_result = {
            key: value
            for key, value in result.items()
            if key not in {"train_1k", "train_4k", "val_select", "val_compare"}
        }
        display_result["subsets"] = {
            name: {
                "sample_count": result[name]["sample_count"],
                "source_count_histogram": result[name]["source_count_histogram"],
            }
            for name in ("train_1k", "train_4k", "val_select", "val_compare")
        }
    print(json.dumps(display_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
