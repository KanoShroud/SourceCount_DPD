"""S2-G4 Coarse-Oracle D8受控适配与根因诊断。

所有子命令只写显式指定的新输出。原始IQ、Exact定位数据和既有checkpoint
均只读使用；本入口不生成test结果，不修改第三章，也不调整频带定义。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_root in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from dpd_calculator_torch import DPDGeometry, compute_fine_dpd  # noqa: E402
from eval_ch4_checkpoint import require_s2g2_checkpoint  # noqa: E402
from s2g3_composability import (  # noqa: E402
    EDGE,
    FS,
    LEN,
    decode_d8_sample,
    file_identity,
    gospa_sample,
    load_ch4_mat,
    matched_distances,
    maximum_matches_within,
    numeric_summary,
    subband_union_to_fft_mask,
    summarize_track,
)
from train_yolo import (  # noqa: E402
    LocDataset,
    collate_fn_hm,
    configure_reproducibility,
)
from yolo_model import YOLOv8Loc  # noqa: E402


EXACT_CHECKPOINT_SHA256 = "5d5224d8d6478739af5042391ef1e718c08e2f81565882e133372c9fbd116dfd"
S2G3_COARSE_VAL_SHA256 = "a968afe4514bbbbfee20650112a197d7d7e86f73e2ff9e0e47f29308d886802e"
BOOTSTRAP_SEED = 20260824
BOOTSTRAP_REPLICATES = 10_000
S2G4R_T020_THRESHOLD = 0.2


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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
            payload, handle, ensure_ascii=False, indent=2,
            allow_nan=False, default=json_default,
        )


def load_json(path: Path) -> Any:
    with path.resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def torch_save_new(path: Path, payload: Any) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有PyTorch文件: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"临时文件已存在: {temporary}")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_exact_index(data_dir: Path, split: str) -> tuple[dict[str, Any], Path]:
    split_dir = data_dir.resolve() / split
    index_path = split_dir / f"loc_{split}_index.pt"
    require(index_path.is_file(), f"Exact索引不存在: {index_path}")
    index = torch.load(index_path, map_location="cpu", weights_only=False)
    require(isinstance(index.get("shard_files"), list), "Exact索引缺少shard_files")
    return index, split_dir


def validate_raw_positions(raw: dict[str, Any], shard: dict[str, Any]) -> None:
    for local, sample_tensor in enumerate(shard["sample_idx"]):
        sample = int(sample_tensor.item())
        count = int(shard["n_src"][local].item())
        require(count == int(raw["src_count"][sample]), f"样本{sample}源数不一致")
        positions = raw["src_pos"][sample, :count]
        positions = positions[np.argsort(np.linalg.norm(positions, axis=1))] / EDGE
        saved = shard["pos_label"][local, :count].numpy()
        require(np.array_equal(saved, positions.astype(np.float32)), f"样本{sample}位置标签不一致")


def recompute_positive_slots(
    raw: dict[str, Any], sample: int, count: int, threshold: float,
) -> np.ndarray:
    require(threshold == S2G4R_T020_THRESHOLD, "S2-G4-R1覆盖阈值必须严格为0.2")
    result = np.zeros((count, len(raw["sub_f_lo"])), dtype=bool)
    for source in range(count):
        center = float(raw["fc_offset"][sample, source])
        symbol_rate = float(raw["symbol_rate"][sample, source])
        require(math.isfinite(center) and symbol_rate > 0, f"样本{sample}源{source}频率元数据非法")
        main_lo = center - symbol_rate / 2
        main_hi = center + symbol_rate / 2
        overlap = np.maximum(
            0.0,
            np.minimum(main_hi, raw["sub_f_hi"]) - np.maximum(main_lo, raw["sub_f_lo"]),
        )
        coverage = overlap / float(raw["b_win"])
        result[source] = coverage >= threshold
    return result


def coarse_mask(
    raw: dict[str, Any], sample: int, count: int, *, s2g4r_t020: bool = False,
) -> np.ndarray:
    if s2g4r_t020:
        slot_mask = recompute_positive_slots(raw, sample, count, S2G4R_T020_THRESHOLD)
    else:
        slot_mask = raw["band_mask"][sample, :count] > 0.5
    return subband_union_to_fft_mask(slot_mask, raw["sub_f_lo"], raw["sub_f_hi"])


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    resource_baseline = load_json(args.resource_baseline)
    require(resource_baseline.get("status") == "PASS", "轻量资源基线未通过")
    require(
        float(resource_baseline["median_available_bytes"]) >= 14 * 1024**3,
        "轻量资源基线低于14GiB",
    )
    require(torch.cuda.is_available(), "torch.cuda不可用")
    torch.cuda.set_device(0)
    require(torch.cuda.current_device() == 0, "当前GPU不是cuda:0")
    checkpoint = torch.load(args.exact_checkpoint.resolve(), map_location="cpu", weights_only=False)
    require_s2g2_checkpoint(checkpoint)
    require(int(checkpoint.get("epoch", -1)) == 93, "Exact-D8 checkpoint epoch不是93")
    checkpoint_id = file_identity(args.exact_checkpoint)
    require(checkpoint_id["sha256"] == EXACT_CHECKPOINT_SHA256, "Exact-D8 checkpoint SHA256不符")
    coarse_reference_id = file_identity(args.coarse_val_reference)
    require(coarse_reference_id["sha256"] == S2G3_COARSE_VAL_SHA256, "S2-G3 Coarse validation SHA256不符")

    identities: dict[str, Any] = {
        "exact_checkpoint": checkpoint_id,
        "coarse_val_reference": coarse_reference_id,
    }
    counts = {}
    for split, expected in (("train", 1024), ("val", 256)):
        raw_path = args.raw_iq_dir.resolve() / f"{split}_data.mat"
        raw = load_ch4_mat(raw_path, include_iq=False)
        require(len(raw["src_count"]) == expected, f"{split} IQ数量不是{expected}")
        require(set(raw["src_count"].tolist()) == {2, 3}, f"{split}不是仅2/3源")
        index, split_dir = load_exact_index(args.exact_data_dir, split)
        require(int(index["n_total_tasks"]) == expected, f"{split} Exact任务数不是{expected}")
        identities[f"raw_{split}"] = file_identity(raw_path)
        identities[f"exact_{split}_index"] = file_identity(
            split_dir / f"loc_{split}_index.pt"
        )
        counts[split] = expected

    forbidden = [PROJECT_ROOT / "data" / "chapter4", PROJECT_ROOT / "outputs" / "formal" / "chapter4"]
    require(not any(path.exists() for path in forbidden), f"formal隔离路径存在: {forbidden}")
    payload = {
        "status": "PASS",
        "stage": "s2g4_preflight",
        "gpu": torch.cuda.get_device_name(0),
        "cuda_device": 0,
        "sample_counts": counts,
        "resource_baseline": resource_baseline,
        "input_identities": identities,
        "formal_paths_absent": True,
        "test_executed": False,
        "duration_seconds": time.perf_counter() - started,
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    configure_reproducibility(args.seed, True)
    device = torch.device(args.device)
    require(device.type == "cuda" and (device.index or 0) == 0, "S2-G4固定cuda:0")
    require(torch.cuda.is_available(), "torch.cuda不可用")
    torch.cuda.set_device(0)
    if args.s2g4r_t020:
        require(args.reference_coarse_val is None, "0.2数据不得与0.3 S2-G3参考张量比较")

    raw_path = args.raw_iq_dir.resolve() / f"{args.split}_data.mat"
    raw = load_ch4_mat(raw_path, include_iq=True)
    exact_index, exact_split_dir = load_exact_index(args.exact_data_dir, args.split)
    total = int(exact_index["n_total_tasks"])
    expected = 1024 if args.split == "train" else 256
    require(total == len(raw["src_count"]) == expected, f"{args.split}样本数不等于{expected}")
    limit = total if args.max_samples is None else min(args.max_samples, total)
    require(limit > 0, "max_samples必须大于0")

    output_split = args.output_data_dir.resolve() / args.split
    if output_split.exists():
        raise FileExistsError(f"拒绝复用已有split目录: {output_split}")
    output_split.mkdir(parents=True)

    angles = np.arange(4) * 2 * np.pi / 4
    receivers = np.stack([500.0 * np.cos(angles), 500.0 * np.sin(angles)], axis=1)
    geometry = DPDGeometry(receivers, [0.0, 0.0], 2000, 10, FS, LEN, device)
    shard_files = []
    output_dpd_parts = []
    n_band_bins = []
    task_seconds = []
    empty_count = 0
    produced = 0
    input_shards = []

    for source_name in exact_index["shard_files"]:
        if produced >= limit:
            break
        source_path = exact_split_dir / source_name
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        required = {"gauss_label", "pos_label", "n_src", "sample_idx", "group_idx"}
        require(required.issubset(source), f"{source_path}缺少D8监督字段")
        validate_raw_positions(raw, source)
        take = min(len(source["n_src"]), limit - produced)
        coarse_values = []
        for local in range(take):
            sample = int(source["sample_idx"][local].item())
            require(sample == produced + local, f"sample_idx不连续: {sample} != {produced + local}")
            count = int(source["n_src"][local].item())
            mask = coarse_mask(raw, sample, count, s2g4r_t020=args.s2g4r_t020)
            n_band_bins.append(int(mask.sum()))
            if not bool(mask.any()):
                empty_count += 1
                raise AssertionError(f"样本{sample}出现空Coarse-Oracle频带")
            signal = raw["sig_real"][sample] + 1j * raw["sig_imag"][sample]
            task_started = time.perf_counter()
            mtr = compute_fine_dpd(signal, geometry, freq_mask=mask, chunk_size=args.chunk_size)
            task_seconds.append(time.perf_counter() - task_started)
            require(bool(torch.isfinite(mtr).all()) and bool(torch.all(mtr >= 0)), f"样本{sample}细DPD非法")
            coarse_values.append(torch.log(mtr + 1.0).unsqueeze(0).half().cpu())
        payload = {
            "fine_dpd": torch.stack(coarse_values),
            "gauss_label": source["gauss_label"][:take].clone(),
            "pos_label": source["pos_label"][:take].clone(),
            "n_src": source["n_src"][:take].clone(),
            "sample_idx": source["sample_idx"][:take].clone(),
            "group_idx": source["group_idx"][:take].clone(),
            "s2g4_band_mode": (
                "coarse_oracle_19_subband_union_t020"
                if args.s2g4r_t020 else "coarse_oracle_19_subband_union"
            ),
            "coverage_threshold": (
                S2G4R_T020_THRESHOLD if args.s2g4r_t020 else float(raw["threshold"])
            ),
        }
        output_name = f"loc_{args.split}_{len(shard_files):03d}.pt"
        output_path = output_split / output_name
        torch_save_new(output_path, payload)
        shard_files.append(output_name)
        output_dpd_parts.append(payload["fine_dpd"])
        input_shards.append(file_identity(source_path))
        produced += take
        print(f"[S2-G4][{args.split}] {produced}/{limit}", flush=True)

    require(produced == limit, f"实际生成{produced}条，不等于要求{limit}")
    index_payload = dict(exact_index)
    index_payload.update({
        "shard_files": shard_files,
        "n_total_tasks": produced,
        "n_shards": len(shard_files),
        "filter_mode": (
            "coarse_oracle_19_subband_union_t020"
            if args.s2g4r_t020 else "coarse_oracle_19_subband_union"
        ),
        "coverage_threshold": (
            S2G4R_T020_THRESHOLD if args.s2g4r_t020 else float(raw["threshold"])
        ),
        "source_exact_data_dir": str(args.exact_data_dir.resolve()),
        "source_raw_iq": str(raw_path),
        "compact_fields": ["fine_dpd", "gauss_label", "pos_label", "n_src", "sample_idx", "group_idx"],
    })
    index_path = output_split / f"loc_{args.split}_index.pt"
    torch_save_new(index_path, index_payload)

    parity = None
    if args.reference_coarse_val is not None:
        require(args.split == "val" and produced == 256, "参考Coarse validation仅允许完整val")
        reference = torch.load(args.reference_coarse_val.resolve(), map_location="cpu", weights_only=False)
        observed = torch.cat(output_dpd_parts, dim=0)
        require(reference["fine_dpd"].shape == observed.shape, "S2-G3参考Coarse shape不一致")
        max_abs = float((reference["fine_dpd"] - observed).abs().max().item())
        exact_equal = bool(torch.equal(reference["fine_dpd"], observed))
        require(exact_equal, f"S2-G4 val与S2-G3 Coarse逐张量不一致: max_abs={max_abs}")
        parity = {"tensor_exact": exact_equal, "max_abs_difference": max_abs}

    output_identities = [file_identity(output_split / name) for name in shard_files]
    payload = {
        "status": "PASS",
        "stage": "build_coarse_oracle_dataset",
        "split": args.split,
        "sample_count": produced,
        "shard_count": len(shard_files),
        "empty_band_count": empty_count,
        "coverage_threshold": (
            S2G4R_T020_THRESHOLD if args.s2g4r_t020 else float(raw["threshold"])
        ),
        "labels_recomputed_from_physical_metadata": bool(args.s2g4r_t020),
        "n_band_bins": numeric_summary(np.asarray(n_band_bins)),
        "task_seconds": numeric_summary(np.asarray(task_seconds)),
        "raw_iq": file_identity(raw_path),
        "input_exact_shards": input_shards,
        "output_index": file_identity(index_path),
        "output_shards": output_identities,
        "s2g3_validation_parity": parity,
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
        "performance_interpretation_allowed": False,
    }
    write_json(args.report, payload)
    return payload


def run_band_support_audit(args: argparse.Namespace) -> dict[str, Any]:
    splits = {}
    for split in ("train", "val"):
        raw_path = args.raw_iq_dir.resolve() / f"{split}_data.mat"
        raw = load_ch4_mat(raw_path, include_iq=False)
        empty_samples = []
        unsupported_sources = []
        active_source_rates = []
        for sample, count_value in enumerate(raw["src_count"]):
            count = int(count_value)
            positive = (
                recompute_positive_slots(raw, sample, count, S2G4R_T020_THRESHOLD)
                if args.s2g4r_t020 else raw["band_mask"][sample, :count] > 0.5
            )
            rates = raw["symbol_rate"][sample, :count]
            active_source_rates.extend(rates.tolist())
            if not bool(positive.any()):
                empty_samples.append(sample)
            for source in range(count):
                if not bool(positive[source].any()):
                    unsupported_sources.append({
                        "sample_index": sample,
                        "source_index": source,
                        "source_count": count,
                        "symbol_rate_hz": float(raw["symbol_rate"][sample, source]),
                        "actual_bandwidth_hz": float(raw["bw_actual"][sample, source]),
                    })
        rates = np.asarray(active_source_rates, dtype=np.float64)
        minimum_rate = float(raw["b_win"]) * (
            S2G4R_T020_THRESHOLD if args.s2g4r_t020 else float(raw["threshold"])
        )
        rates_below_minimum = int(np.sum(rates < minimum_rate))
        if not args.s2g4r_t020:
            require(
                rates_below_minimum == len(unsupported_sources),
                f"{split}无正子带源与symbolRate<{minimum_rate}Hz集合不完全一致",
            )
        splits[split] = {
            "raw_iq": file_identity(raw_path),
            "sample_count": int(len(raw["src_count"])),
            "active_source_count": int(len(rates)),
            "sample_empty_band_count": len(empty_samples),
            "sample_empty_band_indices": empty_samples,
            "source_without_positive_subband_count": len(unsupported_sources),
            "source_symbol_rate_below_minimum_count": rates_below_minimum,
            **(
                {"source_symbol_rate_below_3mhz_count": rates_below_minimum}
                if not args.s2g4r_t020 else {}
            ),
            "unsupported_sources": unsupported_sources,
            "unsupported_symbol_rate_hz": numeric_summary(np.asarray([
                item["symbol_rate_hz"] for item in unsupported_sources
            ])),
        }
    threshold = S2G4R_T020_THRESHOLD if args.s2g4r_t020 else 0.3
    training_allowed = all(
        split["sample_empty_band_count"] == 0
        and split["source_without_positive_subband_count"] == 0
        for split in splits.values()
    )
    payload = {
        "status": "PASS",
        "stage": "s2g4_coarse_oracle_band_support_audit",
        "label_rule": {
            "subband_width_hz": 10e6,
            "positive_coverage_threshold": threshold,
            "minimum_full_mainlobe_width_for_positive_hz": 10e6 * threshold,
        },
        "splits": splits,
        "structural_finding": (
            "0.2阈值在当前train/validation上逐源支持完整，可进入快速D8适配诊断。"
            if args.s2g4r_t020 and training_allowed else
            "当前正标签规则仍存在无正子带信源或空联合频带。"
        ),
        "labels_recomputed_from_physical_metadata": bool(args.s2g4r_t020),
        "s2g4_training_allowed": training_allowed,
        "test_executed": False,
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def run_finalize_stopped(args: argparse.Namespace) -> dict[str, Any]:
    audit = load_json(args.band_support_audit)
    pilot_train = load_json(args.pilot_train_report)
    pilot_val = load_json(args.pilot_val_report)
    pilot_monitor = load_json(args.pilot_train_monitor)
    failed_monitor = load_json(args.failed_build_monitor)
    require(audit["s2g4_training_allowed"] is False, "频带支持审计未触发停止条件")
    require(audit["splits"]["train"]["sample_empty_band_count"] > 0, "训练集没有空频带样本")
    require(pilot_train["status"] == "PASS" and pilot_val["status"] == "PASS", "数据pilot未通过")
    require(pilot_monitor["status"] == "PASS", "训练pilot未通过")
    require(failed_monitor["status"] == "CRASHED", "完整数据构造未按预期停止")
    payload = {
        "status": "STOPPED_AT_DATA_GATE",
        "gate": "S2-G4",
        "engineering_status": "PARTIAL_PASS",
        "scientific_status": "INCONCLUSIVE_FOR_D8_VS_REPRESENTATION",
        "structural_finding": "REPRESENTATION_SUPPORT_GAP_CONFIRMED",
        "stop_reason": audit["structural_finding"],
        "label_rule": audit["label_rule"],
        "band_support_summary": {
            split: {
                key: audit["splits"][split][key]
                for key in (
                    "sample_count",
                    "active_source_count",
                    "sample_empty_band_count",
                    "sample_empty_band_indices",
                    "source_without_positive_subband_count",
                    "source_symbol_rate_below_3mhz_count",
                    "unsupported_symbol_rate_hz",
                )
            }
            for split in ("train", "val")
        },
        "completed": {
            "identity_and_resource_preflight": True,
            "coarse_data_pilot_8_train_8_val": True,
            "one_batch_training_pilot": True,
            "full_train_val_metadata_support_audit": True,
        },
        "not_executed": {
            "complete_coarse_train_dataset": True,
            "complete_coarse_validation_dataset": True,
            "scratch_d8_200_epochs": True,
            "exact_d8_finetune_60_epochs": True,
            "six_track_validation_evaluation": True,
            "test": True,
            "predicted_band": True,
            "predicted_k": True,
            "ch3_modification_or_training": True,
        },
        "causal_question_answered": False,
        "performance_interpretation_allowed": False,
        "fallback_or_threshold_adjustment_used": False,
        "evidence": {
            "band_support_audit": file_identity(args.band_support_audit),
            "pilot_train_report": file_identity(args.pilot_train_report),
            "pilot_val_report": file_identity(args.pilot_val_report),
            "pilot_train_monitor": file_identity(args.pilot_train_monitor),
            "failed_build_monitor": file_identity(args.failed_build_monitor),
        },
    }
    write_json(args.output, payload)
    return payload


def build_model(checkpoint_path: Path, device: torch.device) -> tuple[YOLOv8Loc, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path.resolve(), map_location=device, weights_only=False)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        state = checkpoint.get("model_state")
    require(isinstance(state, dict), "checkpoint缺少model/model_state")
    require(checkpoint.get("method") == "dualhead", "checkpoint method不是dualhead")
    require(checkpoint.get("save_tag") == "dualhead_std", "checkpoint不是D8 dualhead_std")
    nonfinite = [name for name, value in state.items() if not bool(torch.isfinite(value).all())]
    require(not nonfinite, f"checkpoint权重含NaN/Inf: {nonfinite[:1]}")
    model = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, checkpoint


def run_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    configure_reproducibility(args.seed, True)
    device = torch.device(args.device)
    require(device.type == "cuda" and (device.index or 0) == 0, "S2-G4评估固定cuda:0")
    torch.cuda.set_device(0)
    dataset = LocDataset(str(args.data_dir.resolve()), "val", method="dualhead", augment=False)
    require(len(dataset) == 256, "validation任务数不是256")
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
        pin_memory=True, collate_fn=collate_fn_hm,
    )
    model, checkpoint = build_model(args.checkpoint, device)
    samples = []
    matched_errors: list[float] = []
    sample_offset = 0
    with torch.no_grad():
        for dpd, _, pos, n_src in loader:
            dpd = dpd.to(device)
            heatmap, offset = model(dpd)
            require(bool(torch.isfinite(heatmap).all() and torch.isfinite(offset).all()), "D8输出非法")
            for local in range(len(n_src)):
                sample = sample_offset + local
                count = int(n_src[local].item())
                true_positions = pos[local, :count].numpy() * EDGE
                predicted_positions, scores = decode_d8_sample(heatmap[local], offset[local], count)
                matches = matched_distances(true_positions, predicted_positions)
                distances = [distance for _, _, distance in matches]
                matched_errors.extend(distances)
                gospa = gospa_sample(true_positions, predicted_positions)
                record = {
                    "sample_index": sample,
                    "true_count": count,
                    "predicted_count": int(len(predicted_positions)),
                    "predicted_positions_m": predicted_positions.tolist(),
                    "peak_scores": scores.tolist(),
                    "matched_errors_m": distances,
                    "matched_rmse_m": float(math.sqrt(np.mean(np.square(distances)))),
                    "matched_mean_m": float(np.mean(distances)),
                    "matched_max_m": float(np.max(distances)),
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
            sample_offset += len(n_src)
    require(len(samples) == 256 and len(matched_errors) == 640, "评估样本或逐源误差数量错误")
    metrics = summarize_track(samples, matched_errors)
    errors = np.asarray(matched_errors, dtype=np.float64)
    metrics["extreme_error_counts"] = {
        "above_100m": int(np.sum(errors > 100)),
        "above_500m": int(np.sum(errors > 500)),
        "above_1000m": int(np.sum(errors > 1000)),
    }
    stratified = {}
    for count in (2, 3):
        selected = [sample for sample in samples if sample["true_count"] == count]
        selected_errors = [error for sample in selected for error in sample["matched_errors_m"]]
        stratified[f"N{count}"] = summarize_track(selected, selected_errors)

    samples_path = args.samples_jsonl.resolve()
    if samples_path.exists():
        raise FileExistsError(f"拒绝覆盖已有JSONL: {samples_path}")
    samples_path.parent.mkdir(parents=True, exist_ok=True)
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
    payload = {
        "status": "PASS",
        "stage": "s2g4_cross_evaluation",
        "track": args.track,
        "data_mode": args.data_mode,
        "evaluation_mode": "oracle-K_ground_truth_source_count",
        "checkpoint": file_identity(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "sample_count": len(samples),
        "metrics": metrics,
        "stratified": stratified,
        "worst_samples": sorted(samples, key=lambda item: item["matched_max_m"], reverse=True)[:10],
        "samples_jsonl": str(samples_path),
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def trend_diagnostic(summary: dict[str, Any]) -> dict[str, Any]:
    history = load_json(Path(summary["history_json"]))
    require(history, "训练history为空")
    rmse = np.asarray([row["rmse"] for row in history], dtype=np.float64)
    epochs = np.asarray([row["epoch"] for row in history], dtype=np.int64)
    require(np.isfinite(rmse).all(), "训练history RMSE含NaN/Inf")
    best_index = int(np.argmin(rmse))
    best_epoch = int(epochs[best_index])
    final_window_start = max(len(rmse) - 20, 1)
    prior_best = float(rmse[:final_window_start].min())
    final_best = float(rmse[final_window_start:].min())
    final_improvement = max(0.0, (prior_best - final_best) / prior_best)
    closed = best_epoch <= int(epochs[final_window_start - 1]) or final_improvement < 0.01
    initial_rmse = float(summary["initial_validation_metrics"]["rmse"])
    best_rmse = float(summary["best_rmse"])
    learning = best_rmse < 0.9 * initial_rmse
    return {
        "best_epoch": best_epoch,
        "best_rmse_m": best_rmse,
        "initial_rmse_m": initial_rmse,
        "relative_improvement_from_initial": (initial_rmse - best_rmse) / initial_rmse,
        "last_20_relative_improvement": final_improvement,
        "trend_closed": closed,
        "effective_learning": learning,
    }


def paired_bootstrap(first_rows: list[dict[str, Any]], second_rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    require(len(first_rows) == len(second_rows) == 256, "bootstrap样本数错误")
    require(
        [row["sample_index"] for row in first_rows] == [row["sample_index"] for row in second_rows],
        "bootstrap样本索引不一致",
    )
    first = np.asarray([row[field] for row in first_rows], dtype=np.float64)
    second = np.asarray([row[field] for row in second_rows], dtype=np.float64)
    delta = second - first
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPLICATES, 500):
        block = min(500, BOOTSTRAP_REPLICATES - start)
        indices = rng.integers(0, len(delta), size=(block, len(delta)))
        estimates[start:start + block] = delta[indices].mean(axis=1)
    return {
        "field": field,
        "paired_mean_difference_second_minus_first": float(delta.mean()),
        "bootstrap_95_ci": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))],
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
    }


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    evaluations = {
        "EE": load_json(args.ee), "EC": load_json(args.ec),
        "CE": load_json(args.ce), "CC": load_json(args.cc),
        "FE": load_json(args.fe), "FC": load_json(args.fc),
    }
    require(all(item.get("status") == "PASS" for item in evaluations.values()), "交叉评估存在非PASS")
    require(all(item.get("evaluation_mode") == "oracle-K_ground_truth_source_count" for item in evaluations.values()), "存在非oracle-K评估")
    scratch_summary = load_json(args.scratch_summary)
    finetune_summary = load_json(args.finetune_summary)
    scratch_trend = trend_diagnostic(scratch_summary)
    finetune_trend = trend_diagnostic(finetune_summary)

    metric = lambda key: evaluations[key]["metrics"]
    r_ee = float(metric("EE")["matched_errors_m"]["rmse"])
    r_ec = float(metric("EC")["matched_errors_m"]["rmse"])
    r_cc = float(metric("CC")["matched_errors_m"]["rmse"])
    require(abs(r_ee - 130.72651561958395) <= 1e-9, f"EE未回归S2-G2: {r_ee}")
    require(abs(r_ec - 164.14838236487265) <= 1e-9, f"EC未回归S2-G3: {r_ec}")
    gap = r_ec - r_ee
    require(gap > 0, "EC与EE基线差值不是正数")
    recovery = (r_ec - r_cc) / gap
    rmse_ratio = r_cc / r_ee
    gospa_ratio = float(metric("CC")["gospa"]["mean"]) / float(metric("EE")["gospa"]["mean"])
    recall_drop = (
        float(metric("EE")["set_detection"]["100m"]["recall"])
        - float(metric("CC")["set_detection"]["100m"]["recall"])
    )
    checks = {
        "scratch_trend_closed": bool(scratch_trend["trend_closed"]),
        "scratch_effective_learning": bool(scratch_trend["effective_learning"]),
        "cc_rmse_within_10_percent_of_ee": rmse_ratio <= 1.10,
        "cc_gospa_within_10_percent_of_ee": gospa_ratio <= 1.10,
        "cc_recall_100m_drop_at_most_0_05": recall_drop <= 0.05,
    }
    if not checks["scratch_trend_closed"] or not checks["scratch_effective_learning"]:
        scientific_status = "INCONCLUSIVE"
    elif all((
        checks["cc_rmse_within_10_percent_of_ee"],
        checks["cc_gospa_within_10_percent_of_ee"],
        checks["cc_recall_100m_drop_at_most_0_05"],
    )):
        scientific_status = "D8_DISTRIBUTION_SHIFT_DOMINANT"
    elif recovery >= 0.50:
        scientific_status = "MIXED_CAUSES"
    else:
        scientific_status = "REPRESENTATION_LIMIT_CANDIDATE"

    ee_rows = load_jsonl(Path(evaluations["EE"]["samples_jsonl"]))
    cc_rows = load_jsonl(Path(evaluations["CC"]["samples_jsonl"]))
    payload = {
        "status": "PASS",
        "gate": "S2-G4",
        "engineering_status": "PASS",
        "scientific_status": scientific_status,
        "scope": "Coarse-Oracle D8受控适配诊断；缩减validation、oracle-K、非论文正式性能",
        "baseline_regression": {
            "EE_rmse_m": r_ee,
            "EC_rmse_m": r_ec,
            "s2g2_exact_reproduced": True,
            "s2g3_coarse_reproduced": True,
        },
        "scratch_training": scratch_trend,
        "finetune_training": finetune_trend,
        "causal_metrics": {
            "CC_rmse_m": r_cc,
            "rmse_recovery_fraction": recovery,
            "CC_to_EE_rmse_ratio": rmse_ratio,
            "CC_to_EE_mean_gospa_ratio": gospa_ratio,
            "EE_minus_CC_recall_100m": recall_drop,
        },
        "decision_checks": checks,
        "paired_bootstrap_CC_minus_EE": {
            "sample_rmse": paired_bootstrap(ee_rows, cc_rows, "matched_rmse_m"),
            "gospa": paired_bootstrap(ee_rows, cc_rows, "gospa_m"),
        },
        "evaluation_summaries": {
            key: {
                "track": value["track"],
                "checkpoint": value["checkpoint"],
                "metrics": value["metrics"],
                "stratified": value["stratified"],
                "worst_samples": value["worst_samples"],
            }
            for key, value in evaluations.items()
        },
        "fine_tune_interpretation_boundary": "单一lr/epoch微调只评价低成本方案，不证明微调路线普遍有效或无效。",
        "representation_causal_boundary": "若判为表示不足候选，也只对当前D8结构、训练预算和缩减数据成立，不是信息论证明。",
        "test_executed": False,
        "predicted_band_executed": False,
        "predicted_k_executed": False,
        "ch3_modified_or_trained": False,
        "paper_endpoint_performance_claim_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def run_finalize_r1(args: argparse.Namespace) -> dict[str, Any]:
    evaluations = {
        "EE": load_json(args.ee),
        "EC20": load_json(args.ec20),
        "CC20": load_json(args.cc20),
    }
    require(all(item.get("status") == "PASS" for item in evaluations.values()), "三轨评估存在非PASS")
    require(
        all(item.get("evaluation_mode") == "oracle-K_ground_truth_source_count" for item in evaluations.values()),
        "存在非oracle-K评估",
    )
    scratch_summary = load_json(args.scratch_summary)
    scratch_trend = trend_diagnostic(scratch_summary)
    metric = lambda key: evaluations[key]["metrics"]
    r_ee = float(metric("EE")["matched_errors_m"]["rmse"])
    r_ec20 = float(metric("EC20")["matched_errors_m"]["rmse"])
    r_cc20 = float(metric("CC20")["matched_errors_m"]["rmse"])
    require(abs(r_ee - 130.72651561958395) <= 1e-9, f"EE未回归S2-G2: {r_ee}")
    gap = r_ec20 - r_ee
    recovery = None if gap <= 0 else (r_ec20 - r_cc20) / gap
    rmse_ratio = r_cc20 / r_ee
    gospa_ratio = float(metric("CC20")["gospa"]["mean"]) / float(metric("EE")["gospa"]["mean"])
    recall_drop = (
        float(metric("EE")["set_detection"]["100m"]["recall"])
        - float(metric("CC20")["set_detection"]["100m"]["recall"])
    )
    checks = {
        "scratch_trend_closed": bool(scratch_trend["trend_closed"]),
        "scratch_effective_learning": bool(scratch_trend["effective_learning"]),
        "cc20_rmse_within_10_percent_of_ee": rmse_ratio <= 1.10,
        "cc20_gospa_within_10_percent_of_ee": gospa_ratio <= 1.10,
        "cc20_recall_100m_drop_at_most_0_05": recall_drop <= 0.05,
    }
    if not checks["scratch_trend_closed"] or not checks["scratch_effective_learning"]:
        scientific_status = "INCONCLUSIVE_T020"
    elif all((
        checks["cc20_rmse_within_10_percent_of_ee"],
        checks["cc20_gospa_within_10_percent_of_ee"],
        checks["cc20_recall_100m_drop_at_most_0_05"],
    )):
        scientific_status = "D8_DISTRIBUTION_SHIFT_DOMINANT_T020"
    elif recovery is not None and recovery >= 0.50:
        scientific_status = "MIXED_CAUSES_T020"
    else:
        scientific_status = "REPRESENTATION_LIMIT_CANDIDATE_T020"

    ee_rows = load_jsonl(Path(evaluations["EE"]["samples_jsonl"]))
    cc20_rows = load_jsonl(Path(evaluations["CC20"]["samples_jsonl"]))
    payload = {
        "status": "PASS",
        "gate": "S2-G4-R1",
        "engineering_status": "PASS",
        "scientific_status": scientific_status,
        "scope": "0.2 Coarse-Oracle、缩减validation、oracle-K、非论文正式性能",
        "coverage_threshold": S2G4R_T020_THRESHOLD,
        "baseline_metrics": {"R_EE_m": r_ee, "R_EC20_m": r_ec20},
        "scratch_training": scratch_trend,
        "causal_metrics": {
            "R_CC20_m": r_cc20,
            "rmse_recovery_fraction": recovery,
            "CC20_to_EE_rmse_ratio": rmse_ratio,
            "CC20_to_EE_mean_gospa_ratio": gospa_ratio,
            "EE_minus_CC20_recall_100m": recall_drop,
        },
        "decision_checks": checks,
        "paired_bootstrap_CC20_minus_EE": {
            "sample_rmse": paired_bootstrap(ee_rows, cc20_rows, "matched_rmse_m"),
            "gospa": paired_bootstrap(ee_rows, cc20_rows, "gospa_m"),
        },
        "evaluation_summaries": {
            key: {
                "track": value["track"],
                "checkpoint": value["checkpoint"],
                "metrics": value["metrics"],
                "stratified": value["stratified"],
                "worst_samples": value["worst_samples"],
            }
            for key, value in evaluations.items()
        },
        "ch3_compatibility_boundary": "现有CH3仍按0.3标签训练；本结果不证明其可直接输出0.2接口。",
        "test_executed": False,
        "predicted_band_executed": False,
        "predicted_k_executed": False,
        "ch3_modified_or_trained": False,
        "finetune_executed": False,
        "paper_endpoint_performance_claim_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S2-G4 Coarse-Oracle D8适配诊断")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--raw_iq_dir", type=Path, required=True)
    preflight.add_argument("--exact_data_dir", type=Path, required=True)
    preflight.add_argument("--exact_checkpoint", type=Path, required=True)
    preflight.add_argument("--coarse_val_reference", type=Path, required=True)
    preflight.add_argument("--resource_baseline", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    build = sub.add_parser("build")
    build.add_argument("--split", choices=["train", "val"], required=True)
    build.add_argument("--raw_iq_dir", type=Path, required=True)
    build.add_argument("--exact_data_dir", type=Path, required=True)
    build.add_argument("--output_data_dir", type=Path, required=True)
    build.add_argument("--max_samples", type=int)
    build.add_argument("--reference_coarse_val", type=Path)
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--chunk_size", type=int, default=40000)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--s2g4r_t020", action="store_true", default=False)
    build.add_argument("--report", type=Path, required=True)

    support = sub.add_parser("audit-band-support")
    support.add_argument("--raw_iq_dir", type=Path, required=True)
    support.add_argument("--s2g4r_t020", action="store_true", default=False)
    support.add_argument("--output", type=Path, required=True)

    stopped = sub.add_parser("finalize-stopped")
    stopped.add_argument("--band_support_audit", type=Path, required=True)
    stopped.add_argument("--pilot_train_report", type=Path, required=True)
    stopped.add_argument("--pilot_val_report", type=Path, required=True)
    stopped.add_argument("--pilot_train_monitor", type=Path, required=True)
    stopped.add_argument("--failed_build_monitor", type=Path, required=True)
    stopped.add_argument("--output", type=Path, required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--track", choices=["EE", "EC", "CE", "CC", "FE", "FC", "EC20", "CC20"], required=True
    )
    evaluate.add_argument("--data_mode", choices=["exact", "coarse", "coarse_t020"], required=True)
    evaluate.add_argument("--data_dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--samples_jsonl", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch_size", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=42)

    finalize = sub.add_parser("finalize")
    for name in ("ee", "ec", "ce", "cc", "fe", "fc"):
        finalize.add_argument(f"--{name}", type=Path, required=True)
    finalize.add_argument("--scratch_summary", type=Path, required=True)
    finalize.add_argument("--finetune_summary", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)

    finalize_r1 = sub.add_parser("finalize-r1")
    finalize_r1.add_argument("--ee", type=Path, required=True)
    finalize_r1.add_argument("--ec20", type=Path, required=True)
    finalize_r1.add_argument("--cc20", type=Path, required=True)
    finalize_r1.add_argument("--scratch_summary", type=Path, required=True)
    finalize_r1.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        payload = run_preflight(args)
    elif args.command == "build":
        require(args.max_samples is None or args.max_samples > 0, "max_samples必须为正数")
        require(args.chunk_size > 0, "chunk_size必须为正数")
        payload = run_build(args)
    elif args.command == "evaluate":
        require(args.batch_size == 8 and args.seed == 42, "S2-G4评估固定batch=8、seed=42")
        payload = run_evaluate(args)
    elif args.command == "audit-band-support":
        payload = run_band_support_audit(args)
    elif args.command == "finalize-stopped":
        payload = run_finalize_stopped(args)
    elif args.command == "finalize":
        payload = run_finalize(args)
    elif args.command == "finalize-r1":
        payload = run_finalize_r1(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
