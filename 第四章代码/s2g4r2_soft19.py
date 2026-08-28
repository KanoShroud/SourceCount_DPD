"""S2-G4-R2 Soft-19 oracle适配诊断。

本入口只读复用第四章缩减IQ、Exact定位监督和冻结D8。Hard-19-Actual是
Soft-19-Actual的公平控制；二者都基于逐源BW_actual，不修改第三章或原入口。
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from dpd_calculator_torch import DPDGeometry, compute_fine_dpd
from s2g3_composability import (
    FS,
    LEN,
    actual_union_mask,
    file_identity,
    load_ch4_mat,
    numeric_summary,
    subband_union_to_fft_mask,
)
from s2g4_coarse_d8 import (
    load_exact_index,
    load_json,
    load_jsonl,
    paired_bootstrap,
    require,
    run_evaluate,
    run_preflight,
    torch_save_new,
    trend_diagnostic,
    validate_raw_positions,
    write_json,
)
from train_yolo import configure_reproducibility


HARD_THRESHOLD = 0.2
REPRESENTATION_MODES = ("hard_actual", "soft19_actual")


def actual_slot_coverages(raw: dict[str, Any], sample: int, count: int) -> np.ndarray:
    """返回逐源、逐窗口的BW_actual覆盖率，shape=(count, 19)。"""
    result = np.zeros((count, len(raw["sub_f_lo"])), dtype=np.float32)
    for source in range(count):
        center = float(raw["fc_offset"][sample, source])
        bandwidth = float(raw["bw_actual"][sample, source])
        require(math.isfinite(center) and math.isfinite(bandwidth) and bandwidth > 0,
                f"样本{sample}源{source}实际频带元数据非法")
        band_lo = center - bandwidth / 2
        band_hi = center + bandwidth / 2
        overlap = np.maximum(
            0.0,
            np.minimum(band_hi, raw["sub_f_hi"]) - np.maximum(band_lo, raw["sub_f_lo"]),
        )
        result[source] = (overlap / float(raw["b_win"])).astype(np.float32)
    require(np.isfinite(result).all() and np.all((result >= 0) & (result <= 1)),
            f"样本{sample}覆盖率超出[0,1]")
    return result


def hard_actual_mask(raw: dict[str, Any], sample: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    coverages = actual_slot_coverages(raw, sample, count)
    slots = coverages >= HARD_THRESHOLD
    return subband_union_to_fft_mask(slots, raw["sub_f_lo"], raw["sub_f_hi"]), slots


def soft_actual_weights(raw: dict[str, Any], sample: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    coverages = actual_slot_coverages(raw, sample, count)
    subband_weights = coverages.max(axis=0)
    f_axis = np.arange(-LEN // 2, LEN // 2, dtype=np.float64) * (FS / LEN)
    weights = np.zeros(LEN, dtype=np.float32)
    for subband, value in enumerate(subband_weights):
        if value <= 0:
            continue
        inside = (f_axis >= raw["sub_f_lo"][subband]) & (f_axis < raw["sub_f_hi"][subband])
        weights[inside] = np.maximum(weights[inside], value)
    require(np.isfinite(weights).all() and np.all((weights >= 0) & (weights <= 1)),
            f"样本{sample}Soft-19 FFT权重非法")
    require(bool(np.any(weights > 0)), f"样本{sample}Soft-19权重全零")
    return weights, coverages


def sample_representation(
    raw: dict[str, Any], sample: int, count: int, mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    require(mode in REPRESENTATION_MODES, f"未知表示模式: {mode}")
    if mode == "hard_actual":
        mask, slots = hard_actual_mask(raw, sample, count)
        return mask.astype(np.float32), slots.astype(np.float32)
    return soft_actual_weights(raw, sample, count)


def run_representation_audit(args: argparse.Namespace) -> dict[str, Any]:
    split_reports: dict[str, Any] = {}
    for split, expected in (("train", 1024), ("val", 256)):
        raw_path = args.raw_iq_dir.resolve() / f"{split}_data.mat"
        raw = load_ch4_mat(raw_path, include_iq=False)
        require(len(raw["src_count"]) == expected, f"{split}样本数不是{expected}")
        hard_bins = []
        soft_bins = []
        soft_equivalent_bandwidth = []
        exact_bins = []
        hard_unsupported = []
        soft_empty = []
        soft_positive_values = []
        for sample, count_value in enumerate(raw["src_count"]):
            count = int(count_value)
            coverages = actual_slot_coverages(raw, sample, count)
            unsupported = np.flatnonzero(~(coverages >= HARD_THRESHOLD).any(axis=1))
            for source in unsupported:
                hard_unsupported.append({"sample_index": sample, "source_index": int(source)})
            hard_mask, _ = hard_actual_mask(raw, sample, count)
            weights, _ = soft_actual_weights(raw, sample, count)
            if not bool(np.any(weights > 0)):
                soft_empty.append(sample)
            exact_mask = actual_union_mask(raw["fc_offset"][sample], raw["bw_actual"][sample], count)
            hard_bins.append(int(hard_mask.sum()))
            soft_bins.append(int(np.count_nonzero(weights)))
            exact_bins.append(int(exact_mask.sum()))
            soft_equivalent_bandwidth.append(float(weights.sum() * FS / LEN))
            soft_positive_values.extend(weights[weights > 0].tolist())
        require(not hard_unsupported, f"{split} Hard-19-Actual存在无正子带源")
        require(not soft_empty, f"{split} Soft-19-Actual存在全零样本")
        split_reports[split] = {
            "raw_iq": file_identity(raw_path),
            "sample_count": expected,
            "active_source_count": int(np.sum(raw["src_count"])),
            "hard_source_without_positive_subband_count": len(hard_unsupported),
            "soft_empty_sample_count": len(soft_empty),
            "hard_nonzero_fft_bins": numeric_summary(np.asarray(hard_bins)),
            "soft_nonzero_fft_bins": numeric_summary(np.asarray(soft_bins)),
            "exact_nonzero_fft_bins": numeric_summary(np.asarray(exact_bins)),
            "soft_equivalent_bandwidth_hz": numeric_summary(
                np.asarray(soft_equivalent_bandwidth)
            ),
            "soft_positive_fft_weights": numeric_summary(np.asarray(soft_positive_values)),
        }
    payload = {
        "status": "PASS",
        "stage": "s2g4r2_representation_audit",
        "physical_band_definition": "fc_offset_plus_minus_BW_actual_over_2",
        "subband_width_hz": 10e6,
        "subband_step_hz": 5e6,
        "subband_count": 19,
        "hard_threshold": HARD_THRESHOLD,
        "soft_subband_aggregation": "max_over_sources",
        "soft_fft_aggregation": "max_over_overlapping_subbands",
        "soft_fft_amplitude_rule": "sqrt(weight)",
        "splits": split_reports,
        "training_allowed": True,
        "test_executed": False,
        "performance_interpretation_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def run_weight_selftest(args: argparse.Namespace) -> dict[str, Any]:
    configure_reproducibility(42, True)
    device = torch.device(args.device)
    require(device.type == "cuda" and (device.index or 0) == 0, "权重语义测试固定cuda:0")
    torch.cuda.set_device(0)
    receivers = np.asarray([[5, 0], [0, 5], [-5, 0], [0, -5]], dtype=np.float32)
    geometry = DPDGeometry(receivers, [0.0, 0.0], 20, 10, FS, 256, device)
    generator = torch.Generator(device="cpu").manual_seed(42)
    real = torch.randn((4, 256), generator=generator)
    imag = torch.randn((4, 256), generator=generator)
    signal = torch.complex(real, imag)
    mask = np.zeros(256, dtype=bool)
    mask[48:183] = True
    binary = mask.astype(np.float32)
    graded = np.zeros(256, dtype=np.float32)
    graded[48:96] = 0.2
    graded[96:144] = 0.8
    graded[144:183] = 0.4

    bool_output = compute_fine_dpd(signal, geometry, freq_mask=mask, chunk_size=25)
    binary_output = compute_fine_dpd(
        signal, geometry, freq_weights=binary, chunk_size=25,
    )
    full_output = compute_fine_dpd(signal, geometry, chunk_size=25)
    ones_output = compute_fine_dpd(
        signal, geometry, freq_weights=np.ones(256, dtype=np.float32), chunk_size=25,
    )
    graded_output = compute_fine_dpd(
        signal, geometry, freq_weights=graded, chunk_size=25,
    )
    scaled_output = compute_fine_dpd(
        signal, geometry, freq_weights=graded * 0.25, chunk_size=25,
    )
    binary_max_abs = float((bool_output - binary_output).abs().max().item())
    ones_max_abs = float((full_output - ones_output).abs().max().item())
    scale_max_abs = float((graded_output - scaled_output).abs().max().item())
    require(binary_max_abs <= 1e-6, f"二值权重未回归布尔掩模: {binary_max_abs}")
    require(ones_max_abs <= 1e-6, f"全1权重未回归全频带: {ones_max_abs}")
    require(scale_max_abs <= 1e-4, f"统一缩放未在归一化后保持不变: {scale_max_abs}")
    rejected = {}
    for name, bad in (
        ("negative", np.full(256, -0.1, dtype=np.float32)),
        ("all_zero", np.zeros(256, dtype=np.float32)),
        ("nan", np.full(256, np.nan, dtype=np.float32)),
    ):
        try:
            compute_fine_dpd(signal, geometry, freq_weights=bad, chunk_size=25)
        except ValueError as exc:
            rejected[name] = str(exc)
        else:
            raise AssertionError(f"非法权重{name}未被拒绝")
    try:
        compute_fine_dpd(
            signal, geometry, freq_mask=mask, freq_weights=binary, chunk_size=25,
        )
    except ValueError as exc:
        rejected["mutually_exclusive"] = str(exc)
    else:
        raise AssertionError("freq_mask与freq_weights同时输入未被拒绝")
    payload = {
        "status": "PASS",
        "stage": "s2g4r2_frequency_weight_semantics",
        "binary_weight_vs_boolean_mask_max_abs": binary_max_abs,
        "ones_weight_vs_full_band_max_abs": ones_max_abs,
        "uniform_scale_invariance_max_abs": scale_max_abs,
        "invalid_inputs_rejected": rejected,
        "device": str(device),
    }
    write_json(args.output, payload)
    return payload


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    configure_reproducibility(args.seed, True)
    device = torch.device(args.device)
    require(device.type == "cuda" and (device.index or 0) == 0, "S2-G4-R2固定cuda:0")
    torch.cuda.set_device(0)
    raw_path = args.raw_iq_dir.resolve() / f"{args.split}_data.mat"
    raw = load_ch4_mat(raw_path, include_iq=True)
    exact_index, exact_split_dir = load_exact_index(args.exact_data_dir, args.split)
    expected = args.expected_samples
    if expected is None:
        expected = 1024 if args.split == "train" else 256
    require(len(raw["src_count"]) == expected,
            f"{args.split}原始IQ样本数不是{expected}")
    limit = expected if args.max_samples is None else min(args.max_samples, expected)
    require(limit > 0, "max_samples必须为正数")
    require(int(exact_index["n_total_tasks"]) >= limit,
            f"{args.split} Exact监督不足: need={limit}, actual={exact_index['n_total_tasks']}")
    output_split = args.output_data_dir.resolve() / args.split
    if output_split.exists():
        raise FileExistsError(f"拒绝复用已有split目录: {output_split}")
    output_split.mkdir(parents=True)

    angles = np.arange(4) * 2 * np.pi / 4
    receivers = np.stack([500.0 * np.cos(angles), 500.0 * np.sin(angles)], axis=1)
    geometry = DPDGeometry(receivers, [0.0, 0.0], 2000, 10, FS, LEN, device)
    shard_files = []
    input_shards = []
    nonzero_bins = []
    equivalent_bandwidth_hz = []
    positive_weights = []
    task_seconds = []
    produced = 0

    for source_name in exact_index["shard_files"]:
        if produced >= limit:
            break
        source_path = exact_split_dir / source_name
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        required = {"gauss_label", "pos_label", "n_src", "sample_idx", "group_idx"}
        require(required.issubset(source), f"{source_path}缺少D8监督字段")
        validate_raw_positions(raw, source)
        take = min(len(source["n_src"]), limit - produced)
        values = []
        for local in range(take):
            sample = int(source["sample_idx"][local].item())
            require(sample == produced + local, f"sample_idx不连续: {sample}")
            count = int(source["n_src"][local].item())
            representation, _ = sample_representation(raw, sample, count, args.mode)
            support = representation > 0
            require(bool(support.any()), f"样本{sample}表示全零")
            nonzero_bins.append(int(support.sum()))
            equivalent_bandwidth_hz.append(float(representation.sum() * FS / LEN))
            positive_weights.extend(representation[support].tolist())
            signal = raw["sig_real"][sample] + 1j * raw["sig_imag"][sample]
            task_started = time.perf_counter()
            if args.mode == "hard_actual":
                mtr = compute_fine_dpd(
                    signal, geometry, freq_mask=support, chunk_size=args.chunk_size,
                )
            else:
                mtr = compute_fine_dpd(
                    signal, geometry, freq_weights=representation,
                    chunk_size=args.chunk_size,
                )
            task_seconds.append(time.perf_counter() - task_started)
            require(bool(torch.isfinite(mtr).all()) and bool(torch.all(mtr >= 0)),
                    f"样本{sample}细DPD非法")
            values.append(torch.log(mtr + 1.0).unsqueeze(0).half().cpu())
        payload = {
            "fine_dpd": torch.stack(values),
            "gauss_label": source["gauss_label"][:take].clone(),
            "pos_label": source["pos_label"][:take].clone(),
            "n_src": source["n_src"][:take].clone(),
            "sample_idx": source["sample_idx"][:take].clone(),
            "group_idx": source["group_idx"][:take].clone(),
            "s2g4r2_representation_mode": args.mode,
            "physical_band_definition": "BW_actual",
        }
        output_name = f"loc_{args.split}_{len(shard_files):03d}.pt"
        torch_save_new(output_split / output_name, payload)
        shard_files.append(output_name)
        input_shards.append(file_identity(source_path))
        produced += take
        print(f"[S2-G4-R2][{args.mode}][{args.split}] {produced}/{limit}", flush=True)
    require(produced == limit, f"实际生成{produced}条，不等于要求{limit}")
    index_payload = dict(exact_index)
    index_payload.update({
        "shard_files": shard_files,
        "n_total_tasks": produced,
        "n_shards": len(shard_files),
        "filter_mode": args.mode,
        "physical_band_definition": "BW_actual",
        "hard_threshold": HARD_THRESHOLD if args.mode == "hard_actual" else None,
        "soft_subband_aggregation": "max_over_sources" if args.mode == "soft19_actual" else None,
        "soft_fft_aggregation": "max_over_overlapping_subbands" if args.mode == "soft19_actual" else None,
        "soft_fft_amplitude_rule": "sqrt(weight)" if args.mode == "soft19_actual" else None,
        "source_exact_data_dir": str(args.exact_data_dir.resolve()),
        "source_raw_iq": str(raw_path),
    })
    index_path = output_split / f"loc_{args.split}_index.pt"
    torch_save_new(index_path, index_payload)
    report = {
        "status": "PASS",
        "stage": "s2g4r2_build_dataset",
        "mode": args.mode,
        "split": args.split,
        "sample_count": produced,
        "shard_count": len(shard_files),
        "physical_band_definition": "BW_actual",
        "nonzero_fft_bins": numeric_summary(np.asarray(nonzero_bins)),
        "equivalent_bandwidth_hz": numeric_summary(np.asarray(equivalent_bandwidth_hz)),
        "positive_fft_weights": numeric_summary(np.asarray(positive_weights)),
        "task_seconds": numeric_summary(np.asarray(task_seconds)),
        "raw_iq": file_identity(raw_path),
        "input_exact_shards": input_shards,
        "output_index": file_identity(index_path),
        "output_shards": [file_identity(output_split / name) for name in shard_files],
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
        "performance_interpretation_allowed": False,
    }
    write_json(args.report, report)
    return report


def load_index_and_shards(data_dir: Path, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    split_dir = data_dir.resolve() / split
    index = torch.load(split_dir / f"loc_{split}_index.pt", map_location="cpu", weights_only=False)
    shards = [
        torch.load(split_dir / name, map_location="cpu", weights_only=False)
        for name in index["shard_files"]
    ]
    return index, shards


def run_compare_data(args: argparse.Namespace) -> dict[str, Any]:
    split_reports = {}
    for split in ("train", "val"):
        hard_index, hard_shards = load_index_and_shards(args.hard_data_dir, split)
        soft_index, soft_shards = load_index_and_shards(args.soft_data_dir, split)
        require(hard_index["n_total_tasks"] == soft_index["n_total_tasks"], f"{split}任务数不一致")
        require(len(hard_shards) == len(soft_shards), f"{split}分片数不一致")
        equal_samples = 0
        total_samples = 0
        max_abs = 0.0
        absolute_sum = 0.0
        element_count = 0
        sample_mean_abs = []
        for hard, soft in zip(hard_shards, soft_shards, strict=True):
            for field in ("gauss_label", "pos_label", "n_src", "sample_idx", "group_idx"):
                require(torch.equal(hard[field], soft[field]), f"{split}监督字段{field}不一致")
            require(hard["fine_dpd"].shape == soft["fine_dpd"].shape, f"{split}DPD shape不一致")
            diff = (hard["fine_dpd"].float() - soft["fine_dpd"].float()).abs()
            flattened = diff.flatten(1)
            per_sample = flattened.mean(dim=1)
            sample_mean_abs.extend(per_sample.tolist())
            equal_samples += int((flattened.max(dim=1).values == 0).sum().item())
            total_samples += int(diff.shape[0])
            max_abs = max(max_abs, float(diff.max().item()))
            absolute_sum += float(diff.double().sum().item())
            element_count += diff.numel()
        split_reports[split] = {
            "sample_count": total_samples,
            "tensor_exact_sample_count": equal_samples,
            "tensor_different_sample_count": total_samples - equal_samples,
            "max_abs_difference": max_abs,
            "mean_abs_difference": absolute_sum / element_count,
            "per_sample_mean_abs_difference": numeric_summary(np.asarray(sample_mean_abs)),
            "supervision_fields_exact": True,
        }
    payload = {
        "status": "PASS",
        "stage": "s2g4r2_hard_soft_data_comparison",
        "splits": split_reports,
        "representations_numerically_distinct": any(
            item["tensor_different_sample_count"] > 0 for item in split_reports.values()
        ),
        "test_executed": False,
    }
    require(payload["representations_numerically_distinct"], "Hard与Soft细DPD逐样本完全相同")
    write_json(args.output, payload)
    return payload


def run_evaluate_track(args: argparse.Namespace) -> dict[str, Any]:
    namespace = SimpleNamespace(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        samples_jsonl=args.samples_jsonl,
        device=args.device,
        batch_size=args.batch_size,
        seed=args.seed,
        track=args.track,
        data_mode=args.data_mode,
    )
    return run_evaluate(namespace)


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    evaluations = {
        "EE": load_json(args.ee),
        "EHA": load_json(args.eha),
        "HHA": load_json(args.hha),
        "ESA": load_json(args.esa),
        "SSA": load_json(args.ssa),
    }
    require(all(item.get("status") == "PASS" for item in evaluations.values()), "五轨评估存在非PASS")
    require(all(item.get("evaluation_mode") == "oracle-K_ground_truth_source_count"
                for item in evaluations.values()), "存在非oracle-K评估")
    hard_summary = load_json(args.hard_summary)
    soft_summary = load_json(args.soft_summary)
    require(hard_summary.get("run_label") == "S2-G4-R2_hard_actual", "Hard训练标签错误")
    require(soft_summary.get("run_label") == "S2-G4-R2_soft19_actual", "Soft训练标签错误")
    hard_trend = trend_diagnostic(hard_summary)
    soft_trend = trend_diagnostic(soft_summary)
    metric = lambda key: evaluations[key]["metrics"]
    r_ee = float(metric("EE")["matched_errors_m"]["rmse"])
    r_hha = float(metric("HHA")["matched_errors_m"]["rmse"])
    r_ssa = float(metric("SSA")["matched_errors_m"]["rmse"])
    require(abs(r_ee - 130.72651561958395) <= 1e-9, f"EE未回归S2-G2: {r_ee}")
    hard_gap = r_hha - r_ee
    soft_recovery = None if hard_gap <= 0 else (r_hha - r_ssa) / hard_gap
    rmse_ratio = r_ssa / r_ee
    gospa_ratio = float(metric("SSA")["gospa"]["mean"]) / float(metric("EE")["gospa"]["mean"])
    recall_drop = (
        float(metric("EE")["set_detection"]["100m"]["recall"])
        - float(metric("SSA")["set_detection"]["100m"]["recall"])
    )
    checks = {
        "hard_training_trend_closed": bool(hard_trend["trend_closed"]),
        "hard_training_effective_learning": bool(hard_trend["effective_learning"]),
        "soft_training_trend_closed": bool(soft_trend["trend_closed"]),
        "soft_training_effective_learning": bool(soft_trend["effective_learning"]),
        "ssa_rmse_within_10_percent_of_ee": rmse_ratio <= 1.10,
        "ssa_gospa_within_10_percent_of_ee": gospa_ratio <= 1.10,
        "ssa_recall_100m_drop_at_most_0_05": recall_drop <= 0.05,
        "ssa_rmse_better_than_hha": r_ssa < r_hha,
    }
    training_valid = all((
        checks["hard_training_trend_closed"], checks["hard_training_effective_learning"],
        checks["soft_training_trend_closed"], checks["soft_training_effective_learning"],
    ))
    equivalent = all((
        checks["ssa_rmse_within_10_percent_of_ee"],
        checks["ssa_gospa_within_10_percent_of_ee"],
        checks["ssa_recall_100m_drop_at_most_0_05"],
    ))
    if not training_valid:
        scientific_status = "INCONCLUSIVE_SOFT19"
    elif equivalent:
        scientific_status = "SOFT19_EQUIVALENT_CANDIDATE"
    elif soft_recovery is not None and soft_recovery >= 0.50 and r_ssa < r_hha:
        scientific_status = "SOFT19_EFFECTIVE_NOT_EQUIVALENT"
    else:
        scientific_status = "SOFT19_INSUFFICIENT"

    hha_rows = load_jsonl(Path(evaluations["HHA"]["samples_jsonl"]))
    ssa_rows = load_jsonl(Path(evaluations["SSA"]["samples_jsonl"]))
    ee_rows = load_jsonl(Path(evaluations["EE"]["samples_jsonl"]))
    audit = load_json(args.representation_audit)
    comparison = load_json(args.data_comparison)
    require(audit.get("training_allowed") is True, "表示支持审计未通过")
    require(comparison.get("representations_numerically_distinct") is True, "Hard/Soft数据没有差异")
    payload = {
        "status": "PASS",
        "gate": "S2-G4-R2",
        "engineering_status": "PASS",
        "scientific_status": scientific_status,
        "scope": "BW_actual统一定义下Hard-19与Soft-19 oracle适配；缩减validation、oracle-K",
        "representation_contract": {
            "physical_band_definition": "BW_actual",
            "hard_threshold": HARD_THRESHOLD,
            "soft_subband_aggregation": "max_over_sources",
            "soft_fft_aggregation": "max_over_overlapping_subbands",
            "soft_fft_amplitude_rule": "sqrt(weight)",
        },
        "training": {"hard": hard_trend, "soft": soft_trend},
        "primary_metrics": {
            "R_EE_m": r_ee,
            "R_HHA_m": r_hha,
            "R_SSA_m": r_ssa,
            "soft_rmse_recovery_fraction_from_hard_toward_exact": soft_recovery,
            "SSA_to_EE_rmse_ratio": rmse_ratio,
            "SSA_to_EE_mean_gospa_ratio": gospa_ratio,
            "EE_minus_SSA_recall_100m": recall_drop,
        },
        "decision_checks": checks,
        "paired_bootstrap": {
            "SSA_minus_HHA": {
                "sample_rmse": paired_bootstrap(hha_rows, ssa_rows, "matched_rmse_m"),
                "gospa": paired_bootstrap(hha_rows, ssa_rows, "gospa_m"),
            },
            "SSA_minus_EE": {
                "sample_rmse": paired_bootstrap(ee_rows, ssa_rows, "matched_rmse_m"),
                "gospa": paired_bootstrap(ee_rows, ssa_rows, "gospa_m"),
            },
        },
        "evaluation_summaries": {
            key: {
                "track": value["track"],
                "data_mode": value["data_mode"],
                "checkpoint": value["checkpoint"],
                "metrics": value["metrics"],
                "stratified": value["stratified"],
                "worst_samples": value["worst_samples"],
            }
            for key, value in evaluations.items()
        },
        "evidence": {
            "representation_audit": file_identity(args.representation_audit),
            "data_comparison": file_identity(args.data_comparison),
        },
        "interpretation_boundary": (
            "结论只适用于当前1024/256缩减数据、D8、单seed和oracle-K；"
            "不证明CH3能够输出Soft-19，也不是论文正式性能或信息论上限。"
        ),
        "test_executed": False,
        "predicted_band_executed": False,
        "predicted_k_executed": False,
        "ch3_modified_or_trained": False,
        "paper_endpoint_performance_claim_allowed": False,
    }
    write_json(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S2-G4-R2 Soft-19 oracle适配诊断")
    sub = parser.add_subparsers(dest="command", required=True)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--raw_iq_dir", type=Path, required=True)
    preflight.add_argument("--exact_data_dir", type=Path, required=True)
    preflight.add_argument("--exact_checkpoint", type=Path, required=True)
    preflight.add_argument("--coarse_val_reference", type=Path, required=True)
    preflight.add_argument("--resource_baseline", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    audit = sub.add_parser("audit-representation")
    audit.add_argument("--raw_iq_dir", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    selftest = sub.add_parser("selftest-weights")
    selftest.add_argument("--device", default="cuda:0")
    selftest.add_argument("--output", type=Path, required=True)

    build = sub.add_parser("build")
    build.add_argument("--mode", choices=REPRESENTATION_MODES, required=True)
    build.add_argument("--split", choices=["train", "val"], required=True)
    build.add_argument("--raw_iq_dir", type=Path, required=True)
    build.add_argument("--exact_data_dir", type=Path, required=True)
    build.add_argument("--output_data_dir", type=Path, required=True)
    build.add_argument("--max_samples", type=int)
    build.add_argument("--expected_samples", type=int,
                       help="显式覆盖R2历史样本数门禁，供后续受控Gate复用")
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--chunk_size", type=int, default=40000)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--report", type=Path, required=True)

    compare = sub.add_parser("compare-data")
    compare.add_argument("--hard_data_dir", type=Path, required=True)
    compare.add_argument("--soft_data_dir", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--track", choices=["EE", "EHA", "HHA", "ESA", "SSA"], required=True)
    evaluate.add_argument("--data_mode", choices=["exact", *REPRESENTATION_MODES], required=True)
    evaluate.add_argument("--data_dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--samples_jsonl", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.add_argument("--batch_size", type=int, default=8)
    evaluate.add_argument("--seed", type=int, default=42)

    finalize = sub.add_parser("finalize")
    for name in ("ee", "eha", "hha", "esa", "ssa"):
        finalize.add_argument(f"--{name}", type=Path, required=True)
    finalize.add_argument("--hard_summary", type=Path, required=True)
    finalize.add_argument("--soft_summary", type=Path, required=True)
    finalize.add_argument("--representation_audit", type=Path, required=True)
    finalize.add_argument("--data_comparison", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        payload = run_preflight(args)
    elif args.command == "audit-representation":
        payload = run_representation_audit(args)
    elif args.command == "selftest-weights":
        payload = run_weight_selftest(args)
    elif args.command == "build":
        require(args.max_samples is None or args.max_samples > 0, "max_samples必须为正数")
        require(args.expected_samples is None or args.expected_samples > 0,
                "expected_samples必须为正数")
        require(args.chunk_size > 0 and args.seed == 42, "chunk_size或seed不符合固定要求")
        payload = run_build(args)
    elif args.command == "compare-data":
        payload = run_compare_data(args)
    elif args.command == "evaluate":
        require(args.batch_size == 8 and args.seed == 42, "评估固定batch=8、seed=42")
        payload = run_evaluate_track(args)
    elif args.command == "finalize":
        payload = run_finalize(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
