"""S2-G5-R4 16k CH3规模扩展、容量门禁与训练后诊断入口。

本文件只复用既有CH3/DPD实现；不修改模型、loss、阈值或旧R2/R3证据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import psutil
import torch

from s2g5_r2_ch3 import (
    COARSE_SAMPLE_FIELDS,
    _arrays_equal_chunked,
    _copy_attrs,
    _dataset_creation_options,
    audit_coarse,
    infer,
    metrics,
)
from s2g5_r1_contract import audit_mat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R2_ROOT = PROJECT_ROOT / "outputs" / "s2g5r2_ch3" / "20260827_191207"
R3_ROOT = PROJECT_ROOT / "outputs" / "s2g5r3_cascade" / "20260828_131043"
OLD_8K = R2_ROOT / "coarse_pool" / "8k" / "train_coarse_8k.mat"
VAL_SELECT = R2_ROOT / "coarse_subsets" / "val_select.mat"
VAL_COMPARE = R2_ROOT / "coarse_subsets" / "val_compare.mat"
CHECKPOINT_8K = R2_ROOT / "train_8k" / "best_model_v26_B_M10.pth"
R2_FINAL = R2_ROOT / "analysis" / "8k" / "final_gate_audit_v3.json"
R3_FINAL = R3_ROOT / "final_report.json"
EXPECTED_HASHES = {
    "old_8k": "350dcb1ea5dc411ed8161f409d249c9493dd52862dfd5439ab85dbde5b1e2e67",
    "checkpoint_8k": "291ee9bce04b3a5a603568285d8505fd042d0937cc3ab48f6415ed0f24b80e2c",
    "val_compare": "a35cb199299e17cc86d3cf9793e63e76a7c92650e35558d6d82106188ea90005",
}
BOOTSTRAP_SEED = 20260902
STATIC_FIELDS = {
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(Counter(map(int, values)).items())}


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    inputs = {
        "old_8k": OLD_8K,
        "val_select": VAL_SELECT,
        "val_compare": VAL_COMPARE,
        "checkpoint_8k": CHECKPOINT_8K,
        "r2_final": R2_FINAL,
        "r3_final": R3_FINAL,
    }
    require(all(path.is_file() for path in inputs.values()), "冻结输入存在缺失")
    identities = {
        name: {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in inputs.items()
    }
    for name, expected in EXPECTED_HASHES.items():
        require(identities[name]["sha256"] == expected, f"冻结输入hash变化: {name}")
    require(load_json(R2_FINAL)["decision"] == "FREEZE_8K_FOR_R3", "R2最终判定不匹配")
    require(load_json(R3_FINAL)["scientific_decision"]["r4_priority"] == "COUNT_PATH_PRIORITY", "R3判定不匹配")
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(args.run_root.anchor)
    require(memory.available >= 10 * 1024**3, "启动可用RAM低于10 GiB")
    require(disk.free >= 60 * 1024**3, "启动磁盘剩余低于60 GiB")
    return {
        "status": "PASS",
        "gate": "S2-G5-R4",
        "scope": "pre-training input freeze; no data generation, training, test, or threshold tuning",
        "run_root": str(args.run_root.resolve()),
        "frozen_inputs": identities,
        "system_available_bytes": int(memory.available),
        "disk_free_bytes": int(disk.free),
        "fixed_contract": {
            "combined_samples": 16384,
            "source_count_histogram": {str(key): 4096 for key in range(4)},
            "threshold": 0.5,
            "matlab_seed_add8k": 20260901,
            "training_seed": 42,
            "test_access": False,
        },
    }


def _matlab_scalar(handle: h5py.File, name: str) -> int:
    return int(np.asarray(handle[name]).reshape(-1)[0])


def audit_iq_metadata(path: Path) -> dict[str, Any]:
    contract = audit_mat(path, expected_profile="s2g5r4_16k_add")
    with h5py.File(path, "r") as handle:
        require(_matlab_scalar(handle, "random_seed_val") == 20260901, "新增8k随机种子错误")
        trials = np.asarray(handle["trials_list_val"], dtype=np.int64).reshape(-1)
        require(np.array_equal(trials, [8192, 4, 4]), f"新增8k trials错误: {trials.tolist()}")
        counts = np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1)
        require(counts.size == 8192, "新增IQ训练样本数不是8192")
        require(histogram(counts) == {str(key): 2048 for key in range(4)}, "新增IQ源数未严格等量")
        positions = np.asarray(handle["src_pos_all"], dtype=np.float64).transpose(2, 1, 0)
        receiver_angles = np.arange(4) * 2 * np.pi / 4
        receivers = np.column_stack((500 * np.cos(receiver_angles), 500 * np.sin(receiver_angles)))
        minimum_source_distance = float("inf")
        minimum_receiver_distance = float("inf")
        for sample_positions, count in zip(positions, counts, strict=True):
            active_positions = sample_positions[: int(count)]
            if count >= 2:
                distances = np.linalg.norm(
                    active_positions[:, None, :] - active_positions[None, :, :], axis=2
                )
                np.fill_diagonal(distances, np.inf)
                minimum_source_distance = min(minimum_source_distance, float(np.min(distances)))
            if count >= 1:
                receiver_distances = np.linalg.norm(
                    active_positions[:, None, :] - receivers[None, :, :], axis=2
                )
                minimum_receiver_distance = min(
                    minimum_receiver_distance, float(np.min(receiver_distances))
                )
        require(minimum_source_distance >= 150 - 1e-4, "新增IQ存在源间距小于150 m")
        require(minimum_receiver_distance >= 150 - 1e-4, "新增IQ存在源站距小于150 m")
    return {
        "status": "PASS",
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "samples": 8192,
        "source_count_histogram": histogram(counts),
        "random_seed": 20260901,
        "trials": [8192, 4, 4],
        "minimum_source_distance_m": minimum_source_distance,
        "minimum_receiver_distance_m": minimum_receiver_distance,
        "contract_audit": contract,
    }


def extend_to_16k(old_path: Path, added_path: Path, output_path: Path, block_size: int) -> dict[str, Any]:
    require(old_path.is_file(), f"旧8k不存在: {old_path}")
    require(added_path.is_file(), f"新增8k不存在: {added_path}")
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖: {output_path}")
    old_audit = audit_coarse(old_path, 8192)
    added_audit = audit_coarse(added_path, 8192)
    expected_half = {str(key): 2048 for key in range(4)}
    require(old_audit["source_count_histogram"] == expected_half, "旧8k分层错误")
    require(added_audit["source_count_histogram"] == expected_half, "新增8k分层错误")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(old_path, "r") as old, h5py.File(added_path, "r") as added:
        require(COARSE_SAMPLE_FIELDS.issubset(old.keys()), "旧8k缺少样本字段")
        require(COARSE_SAMPLE_FIELDS.issubset(added.keys()), "新增8k缺少样本字段")
        for name in STATIC_FIELDS:
            require(name in old and name in added, f"静态字段缺失: {name}")
            require(np.array_equal(old[name][...], added[name][...]), f"新旧契约不一致: {name}")
        with h5py.File(output_path, "w") as output:
            for key, value in old.attrs.items():
                output.attrs[key] = value
            for name, dataset in old.items():
                if not isinstance(dataset, h5py.Dataset) or name not in COARSE_SAMPLE_FIELDS:
                    old.copy(name, output)
                    continue
                added_dataset = added[name]
                require(dataset.shape[:-1] == added_dataset.shape[:-1], f"字段{name} shape不一致")
                require(dataset.dtype == added_dataset.dtype, f"字段{name} dtype不一致")
                combined_shape = dataset.shape[:-1] + (16384,)
                target = output.create_dataset(
                    name,
                    shape=combined_shape,
                    dtype=dataset.dtype,
                    **_dataset_creation_options(dataset, combined_shape),
                )
                _copy_attrs(dataset, target)
                if name == "sample_idx_all":
                    target[...] = np.arange(16384, dtype=dataset.dtype).reshape(combined_shape)
                    continue
                for source, offset in ((dataset, 0), (added_dataset, 8192)):
                    for start in range(0, 8192, block_size):
                        stop = min(start + block_size, 8192)
                        src_sel = (slice(None),) * (source.ndim - 1) + (slice(start, stop),)
                        dst_sel = (slice(None),) * (target.ndim - 1) + (slice(offset + start, offset + stop),)
                        target[dst_sel] = source[src_sel]
    combined_audit = audit_coarse(output_path, 16384)
    require(combined_audit["source_count_histogram"] == {str(key): 4096 for key in range(4)}, "16k分层错误")
    with h5py.File(old_path, "r") as old, h5py.File(output_path, "r") as combined:
        prefix = {
            name: _arrays_equal_chunked(old[name], combined[name], 8192, block_size)
            for name in sorted(COARSE_SAMPLE_FIELDS - {"sample_idx_all"})
        }
    require(all(prefix.values()), "16k前8192条不是冻结旧8k的严格前缀")
    return {
        "status": "PASS",
        "gate": "S2-G5-R4",
        "old_8k": old_audit,
        "added_8k": added_audit,
        "combined_16k": combined_audit,
        "strict_old_8k_prefix": True,
        "prefix_field_checks": prefix,
        "lineage": {"old_8k": [0, 8191], "added_8k": [8192, 16383]},
    }


def make_training_view(combined: Path, val_select: Path, data_dir: Path) -> dict[str, Any]:
    require(combined.is_file() and val_select.is_file(), "训练视图输入缺失")
    if data_dir.exists() and any(data_dir.iterdir()):
        raise FileExistsError(f"训练视图目录非空: {data_dir}")
    data_dir.mkdir(parents=True, exist_ok=True)
    train_target = data_dir / "train_data.mat"
    val_target = data_dir / "val_data.mat"
    os.link(combined, train_target)
    os.link(val_select, val_target)
    require(os.path.samefile(combined, train_target), "train hardlink身份错误")
    require(os.path.samefile(val_select, val_target), "validation hardlink身份错误")
    return {
        "status": "PASS",
        "data_dir": str(data_dir.resolve()),
        "train": str(train_target.resolve()),
        "validation": str(val_target.resolve()),
        "train_hardlink_identity": True,
        "validation_hardlink_identity": True,
    }


def paired_bootstrap_8k_16k(
    arrays_8k: dict[str, np.ndarray],
    arrays_16k: dict[str, np.ndarray],
    repetitions: int,
) -> dict[str, Any]:
    counts = arrays_8k["source_count"]
    require(np.array_equal(counts, arrays_16k["source_count"]), "8k/16k评估样本顺序不一致")
    groups = [np.flatnonzero(counts == count) for count in range(4)]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = {"balanced_count_accuracy": [], "active_band_macro_f1": []}
    for _ in range(repetitions):
        selected = np.concatenate([rng.choice(group, group.size, replace=True) for group in groups])
        old_metrics = metrics(arrays_8k, selected)
        new_metrics = metrics(arrays_16k, selected)
        for key in values:
            values[key].append(new_metrics[key] - old_metrics[key])
    return {
        key: {
            "mean_delta_16k_minus_8k": float(np.mean(current)),
            "ci95": [float(value) for value in np.percentile(current, [2.5, 97.5])],
        }
        for key, current in values.items()
    }


def infer_with_scores(checkpoint: Path, data_path: Path) -> dict[str, np.ndarray]:
    arrays = infer(checkpoint, data_path, 64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    from train_v26 import SourceDetectionDataset, SourceDetectionNet

    dataset = SourceDetectionDataset(data_path, augment=False, normalize="sample_zscore", max_src_override=10)
    model = SourceDetectionNet(n_sub=19, max_src=10, mode="transformer").to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(dataset), 64):
            batch = torch.stack([dataset[index][0] for index in range(start, min(start + 64, len(dataset)))])
            logits.append(model(batch.to(device)).cpu().numpy())
    probabilities = 1.0 / (1.0 + np.exp(-np.concatenate(logits)))
    arrays["probabilities"] = probabilities.astype(np.float32)
    return arrays


def confidence_diagnostic(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    probabilities = arrays["probabilities"]
    truth_k = arrays["source_count"]
    slot_scores = probabilities.max(axis=2)
    active = slot_scores > 0.5
    predicted_k = active.sum(axis=1)
    closest_slot = np.argmin(np.abs(slot_scores - 0.5), axis=1)
    alternative_k = predicted_k + np.where(active[np.arange(active.shape[0]), closest_slot], -1, 1)
    candidate_hit = (truth_k == predicted_k) | (truth_k == alternative_k)
    errors = predicted_k != truth_k
    confidence = np.min(np.abs(slot_scores - 0.5), axis=1)
    error_top2 = float(np.mean(candidate_hit[errors])) if np.any(errors) else 1.0
    return {
        "status": "PASS",
        "definition": {
            "slot_score": "max_band_probability_per_slot",
            "confidence": "minimum_absolute_slot_margin_to_0.5",
            "second_candidate": "flip_the_closest_slot",
            "not_a_calibrated_k_probability": True,
        },
        "sample_count": int(truth_k.size),
        "error_count": int(errors.sum()),
        "top2_coverage_all": float(np.mean(candidate_hit)),
        "top2_coverage_among_errors": error_top2,
        "confidence_mean_correct": float(np.mean(confidence[~errors])) if np.any(~errors) else None,
        "confidence_mean_error": float(np.mean(confidence[errors])) if np.any(errors) else None,
        "interface_k_above_3_count": int(np.sum(predicted_k > 3)),
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    arrays_8k = infer_with_scores(CHECKPOINT_8K, VAL_COMPARE)
    arrays_16k = infer_with_scores(args.checkpoint_16k, VAL_COMPARE)
    require(np.array_equal(arrays_8k["truth"], arrays_16k["truth"]), "评估真值不一致")
    result = {
        "status": "PASS",
        "gate": "S2-G5-R4",
        "metrics_8k": metrics(arrays_8k),
        "metrics_16k": metrics(arrays_16k),
        "paired_bootstrap": paired_bootstrap_8k_16k(arrays_8k, arrays_16k, args.repetitions),
        "confidence_8k": confidence_diagnostic(arrays_8k),
        "confidence_16k": confidence_diagnostic(arrays_16k),
        "scope": "fixed validation scale/confidence diagnosis; no test, calibration, or threshold tuning",
    }
    return result


def run_pretrain_finalize(args: argparse.Namespace) -> dict[str, Any]:
    monitor_root = args.run_root / "monitor"
    pass_stages = [
        "01_preflight",
        "02_profile_reject",
        "03_default_smoke",
        "04_iq_add8k",
        "05_audit_iq_add8k",
        "06_coarse_add8k",
        "07_build_16k",
    ]
    monitors: dict[str, Any] = {}
    for stage in pass_stages:
        report_path = monitor_root / stage / "stage_monitor_report.json"
        require(report_path.is_file(), f"缺少监控报告: {stage}")
        report = load_json(report_path)
        require(report["status"] == "PASS" and not report["red_flags"], f"前置阶段未通过: {stage}")
        monitors[stage] = report
    capacity_path = monitor_root / "08_capacity_16k" / "stage_monitor_report.json"
    capacity_monitor = load_json(capacity_path)
    require(capacity_monitor["exit_code"] == 0, "容量命令自身执行失败")
    require(
        capacity_monitor["red_flags"] == ["system_available_ram_below_6_gib"],
        "容量阶段红线状态不是预期的单一RAM红线",
    )
    capacity = load_json(args.run_root / "audit" / "capacity_batch64.json")
    require(capacity["status"] == "PASS", "forward/backward容量计算未完成")
    build = load_json(args.run_root / "audit" / "build_16k_report.json")
    require(build["strict_old_8k_prefix"], "16k严格前缀门未通过")
    combined = Path(build["combined_16k"]["path"])
    require(sha256_file(combined) == build["combined_16k"]["sha256"], "16k文件hash变化")
    training_view = args.run_root / "training_views" / "data_16k"
    require(os.path.samefile(combined, training_view / "train_data.mat"), "训练视图不再指向16k文件")
    require(os.path.samefile(VAL_SELECT, training_view / "val_data.mat"), "训练validation不再指向冻结val_select")
    default_smoke = args.run_root / "compat_default" / "smoke" / "chapter4" / "data" / "train_data.mat"
    with h5py.File(default_smoke, "r") as handle:
        require(_matlab_scalar(handle, "random_seed_val") == 20260821, "默认smoke seed漂移")
        require(np.array_equal(np.asarray(handle["trials_list_val"]).reshape(-1), [4, 2, 2]), "默认smoke规模漂移")
    return {
        "status": "STOPPED_AT_CAPACITY_GATE",
        "gate": "S2-G5-R4",
        "ready_for_long_training": False,
        "reason": "16k eager Dataset capacity completed but system available RAM fell below the 6 GiB red line",
        "completed_pass_stages": pass_stages,
        "capacity_monitor": {
            "status": capacity_monitor["status"],
            "duration_seconds": capacity_monitor["duration_seconds"],
            "minimum_system_available_bytes": capacity_monitor["minimum_system_available_bytes"],
            "maximum_process_tree_rss_bytes": capacity_monitor["maximum_process_tree_rss_bytes"],
            "maximum_gpu_used_mib": capacity_monitor["maximum_gpu_used_mib"],
            "red_flags": capacity_monitor["red_flags"],
        },
        "capacity_computation": capacity,
        "combined_16k": build["combined_16k"],
        "strict_old_8k_prefix": True,
        "default_smoke_compatible": True,
        "training_entry": str((PROJECT_ROOT / "第三章代码" / "s2g5r4_pycharm_train.py").resolve()),
        "training_entry_will_refuse_current_capacity_state": True,
        "prohibitions": {
            "long_training_started": False,
            "test_accessed": False,
            "threshold_tuned": False,
            "model_or_loss_changed": False,
            "automatic_lazy_loading_change": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    preflight.add_argument("--run_root", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    iq = sub.add_parser("audit-iq")
    iq.add_argument("--input", type=Path, required=True)
    iq.add_argument("--output", type=Path, required=True)
    extend = sub.add_parser("extend-16k")
    extend.add_argument("--added_8k", type=Path, required=True)
    extend.add_argument("--combined_16k", type=Path, required=True)
    extend.add_argument("--data_dir", type=Path, required=True)
    extend.add_argument("--block_size", type=int, default=32)
    extend.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare")
    compare.add_argument("--checkpoint_16k", type=Path, required=True)
    compare.add_argument("--repetitions", type=int, default=2000)
    compare.add_argument("--output", type=Path, required=True)
    finalize = sub.add_parser("finalize-pretrain")
    finalize.add_argument("--run_root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "audit-iq":
        result = audit_iq_metadata(args.input)
    elif args.command == "extend-16k":
        result = extend_to_16k(OLD_8K, args.added_8k, args.combined_16k, args.block_size)
        result["training_view"] = make_training_view(args.combined_16k, VAL_SELECT, args.data_dir)
    elif args.command == "compare":
        result = run_compare(args)
    elif args.command == "finalize-pretrain":
        result = run_pretrain_finalize(args)
    else:
        raise AssertionError(f"未知命令: {args.command}")
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
