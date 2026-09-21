"""E2E-G2-R1：混杂消解、原结构能力恢复与条件K=1诊断。"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
CH4_DIR = PROJECT_ROOT / "第四章代码"
os.environ.setdefault(
    "SOURCECOUNT_REFERENCE_OUTPUT_ROOT",
    str((PROJECT_ROOT.parent / "SourceCount_DPD" / "outputs").resolve()),
)
for root in (PROJECT_ROOT, CH4_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gen_ch4_loc_data import group_sources_by_freq_overlap  # noqa: E402
from yolo_model import focal_loss_hm  # noqa: E402

from 统一模型代码.gates.g1 import e2e_g1 as g1  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_latent_fusion as g2  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_preflight as preflight  # noqa: E402
from 统一模型代码.models.e2e_latent_fusion import (  # noqa: E402
    CH3Features,
    FrequencySpatialSplitter,
    SourceLocalizationHead,
    SourceQueryBuilder,
    forward_ch3_features,
    forward_d8_features,
)
from 统一模型代码.common.runtime_paths import new_run_dir, validate_output_path  # noqa: E402


CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g2_r1.json"
MODEL_PATH = PACKAGE_ROOT / "models" / "e2e_latent_fusion.py"
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_RUN = (
    PROJECT_ROOT
    / "outputs_e2e"
    / "unified"
    / "e2e_g2_latent_fusion"
    / "20260903_153441"
)
SOURCE_P2_MANIFEST = SOURCE_RUN / "p2_manifest.json"
SOURCE_P2_CHECKPOINT = SOURCE_RUN / "p2_checkpoint.pt"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def code_identity() -> list[dict[str, Any]]:
    return [identity(path) for path in (SCRIPT_PATH, MODEL_PATH, CONFIG_PATH)]


def prepare(run_id: str) -> Path:
    source_manifest, _ = g2.verify_p2(SOURCE_RUN)
    files, artifacts = preflight.configure_snapshot()
    cache_files = sorted((SOURCE_RUN / "p2_cache").glob("full_*.npy"))
    if len(cache_files) != 32:
        raise RuntimeError(f"原P2全频缓存数量不是32: {len(cache_files)}")
    run_root = new_run_dir("e2e_g2_r1", run_id, create=True)
    manifest = {
        "status": "PREPARED",
        "gate": "E2E-G2-R1",
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_read_only": True,
        "test_executed": False,
        "config": read_json(CONFIG_PATH),
        "code": code_identity(),
        "source_run": str(SOURCE_RUN.resolve()),
        "source_p2_manifest": identity(SOURCE_P2_MANIFEST),
        "source_p2_checkpoint": identity(SOURCE_P2_CHECKPOINT),
        "source_p2_report": identity(SOURCE_RUN / "p2_report.json"),
        "source_fullband_cache": [identity(path) for path in cache_files],
        "inputs": {"files": files, "artifacts": artifacts},
        "records": source_manifest["records"],
    }
    write_json(run_root / "manifest.json", manifest)
    print(str(run_root), flush=True)
    return run_root


def verify_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_root / "manifest.json")
    if manifest["gate"] != "E2E-G2-R1":
        raise RuntimeError("manifest Gate错误")
    if manifest["reference_read_only"] is not True or manifest["test_executed"] is not False:
        raise RuntimeError("读写隔离或test合同错误")
    if code_identity() != manifest["code"]:
        raise RuntimeError("prepare后G2-R1代码或配置发生变化")
    source_manifest, _ = g2.verify_p2(SOURCE_RUN)
    if source_manifest["records"] != manifest["records"]:
        raise RuntimeError("原P2样本清单变化")
    if identity(SOURCE_P2_MANIFEST) != manifest["source_p2_manifest"]:
        raise RuntimeError("原P2 manifest身份变化")
    if identity(SOURCE_P2_CHECKPOINT) != manifest["source_p2_checkpoint"]:
        raise RuntimeError("原P2 checkpoint身份变化")
    if identity(SOURCE_RUN / "p2_report.json") != manifest["source_p2_report"]:
        raise RuntimeError("原P2报告身份变化")
    current_cache = [
        identity(path) for path in sorted((SOURCE_RUN / "p2_cache").glob("full_*.npy"))
    ]
    if current_cache != manifest["source_fullband_cache"]:
        raise RuntimeError("原P2全频缓存身份变化")
    return manifest, manifest["config"]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_total = sum(int(row["true_count"]) for row in rows)
    result = {
        "sample_count": len(rows),
        "source_count": source_total,
        "gospa_mean_m": float(np.mean([row["gospa_m"] for row in rows])),
    }
    for threshold in (10, 30, 50, 100):
        result[f"recall_at_{threshold}m"] = (
            sum(int(row[f"tp_at_{threshold}m"]) for row in rows)
            / max(source_total, 1)
        )
    return result


def summarize_with_k(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": summarize_rows(rows),
        "by_k": {
            str(count): summarize_rows(
                [row for row in rows if int(row["true_count"]) == count]
            )
            for count in range(4)
        },
    }


def metric_row(
    ordinal: int,
    raw_index: int,
    true_positions: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, Any]:
    gospa = g1.gospa_sample(true_positions, predicted)
    row: dict[str, Any] = {
        "ordinal": ordinal,
        "raw_index": raw_index,
        "true_count": len(true_positions),
        "predicted_positions_m": predicted.tolist(),
        "gospa_m": gospa["value_m"],
    }
    for threshold in (10, 30, 50, 100):
        row[f"tp_at_{threshold}m"] = g1.maximum_matches_within(
            true_positions, predicted, float(threshold)
        )
    return row


@dataclass
class TrainContext:
    ch3: nn.Module
    d8: nn.Module
    cached_ch3: CH3Features
    cached_d0: torch.Tensor
    query_builder: SourceQueryBuilder
    splitter: FrequencySpatialSplitter
    source_head: SourceLocalizationHead
    parameters: list[nn.Parameter]


def build_train_context(
    data: g2.CachedBatch,
    device: torch.device,
    seed: int,
    *,
    load_source_endpoint: bool = False,
    dpd_override: torch.Tensor | None = None,
) -> TrainContext:
    set_deterministic(seed)
    ch3, d8, _ = g1.build_models(device)
    ch3.eval()
    d8.eval()
    for parameter in ch3.parameters():
        parameter.requires_grad_(False)
    for parameter in d8.parameters():
        parameter.requires_grad_(False)
    for head in ch3.band_heads[:3]:
        for parameter in head.parameters():
            parameter.requires_grad_(True)

    source_dpd = data.dpd if dpd_override is None else dpd_override
    all_ch3: list[CH3Features] = []
    all_d0: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(data.counts), 4):
            stop = min(start + 4, len(data.counts))
            features = forward_ch3_features(ch3, data.coarse[start:stop].to(device))
            all_ch3.append(
                CH3Features(
                    features.spatial.cpu(),
                    features.tokens.cpu(),
                    features.global_feature.cpu(),
                    torch.empty(0),
                    torch.empty(0),
                )
            )
            d0 = forward_d8_features(
                d8,
                torch.stack(
                    [g1.d8_input(item)[0] for item in source_dpd[start:stop]]
                ).to(device),
            ).d0
            all_d0.append(d0.cpu())
    cached_ch3 = CH3Features(
        torch.cat([item.spatial for item in all_ch3]),
        torch.cat([item.tokens for item in all_ch3]),
        torch.cat([item.global_feature for item in all_ch3]),
        torch.empty(0),
        torch.empty(0),
    )
    cached_d0 = torch.cat(all_d0)
    query_builder = SourceQueryBuilder().to(device)
    splitter = FrequencySpatialSplitter().to(device)
    source_head = SourceLocalizationHead().to(device)
    nn.init.constant_(source_head.heatmap.bias, -2.19)
    nn.init.zeros_(source_head.offset.weight)
    nn.init.zeros_(source_head.offset.bias)

    if load_source_endpoint:
        checkpoint = torch.load(
            SOURCE_P2_CHECKPOINT, map_location=device, weights_only=False
        )
        ch3.band_heads[:3].load_state_dict(checkpoint["ch3_heads"], strict=True)
        query_builder.load_state_dict(checkpoint["query_builder"], strict=True)
        splitter.load_state_dict(checkpoint["splitter"], strict=True)
        source_head.load_state_dict(checkpoint["source_head"], strict=True)

    parameters = [
        parameter
        for parameter in itertools.chain(
            ch3.band_heads[:3].parameters(),
            query_builder.parameters(),
            splitter.parameters(),
            source_head.parameters(),
        )
        if parameter.requires_grad
    ]
    return TrainContext(
        ch3,
        d8,
        cached_ch3,
        cached_d0,
        query_builder,
        splitter,
        source_head,
        parameters,
    )


def features_for_indices(
    cached: CH3Features, indices: torch.Tensor, device: torch.device
) -> CH3Features:
    return CH3Features(
        cached.spatial[indices].to(device),
        cached.tokens[indices].to(device),
        cached.global_feature[indices].to(device),
        torch.empty(0, device=device),
        torch.empty(0, device=device),
    )


def forward_indices(
    context: TrainContext,
    indices: torch.Tensor,
    device: torch.device,
    *,
    stop_gradient: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    current = g2.ch3_from_cached(
        context.ch3, features_for_indices(context.cached_ch3, indices, device)
    )
    query, logits = context.query_builder(current)
    split_query = query.detach() if stop_gradient else query
    split_logits = logits.detach() if stop_gradient else logits
    spatial, attention = context.splitter(
        current.spatial, split_query, split_logits
    )
    heatmap, offset = context.source_head(
        context.cached_d0[indices].to(device), spatial, split_query
    )
    return query, logits, attention, heatmap, offset


def combine_losses(
    components: dict[str, torch.Tensor], config: dict[str, Any]
) -> torch.Tensor:
    weights = config["loss_weights"]
    if set(weights) != set(components):
        raise RuntimeError("loss_weights与loss组成不一致")
    return sum(float(weights[name]) * value for name, value in components.items())


def compute_losses(
    band_logits: torch.Tensor,
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    data: g2.CachedBatch,
    indices: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[int, int]]]:
    device = band_logits.device
    truth = data.band[indices].to(device)
    ignore = data.ignore[indices].to(device)
    positions = data.positions[indices].to(device)
    counts = data.counts[indices].to(device)
    matched = g2.assignments(
        band_logits, heatmap, truth, ignore, positions, counts
    )
    exist_target = torch.zeros_like(band_logits[..., 0])
    band_target = torch.zeros_like(band_logits)
    band_valid = torch.ones_like(band_logits)
    heat_target = torch.zeros_like(heatmap)
    offset_losses: list[torch.Tensor] = []
    for batch_index, mapping in enumerate(matched):
        for query, source in mapping.items():
            exist_target[batch_index, query] = 1.0
            band_target[batch_index, query] = truth[batch_index, source]
            band_valid[batch_index, query] = 1.0 - ignore[batch_index, source]
            target, ix, iy, delta = g2.single_source_target(
                positions[batch_index, source], device
            )
            heat_target[batch_index, query] = target
            offset_losses.append(
                torch.abs(offset[batch_index, query, :, iy, ix] - delta).sum()
            )
    slot_logits = band_logits.amax(dim=-1)
    loss_exist = F.binary_cross_entropy_with_logits(slot_logits, exist_target)
    band_element = F.binary_cross_entropy_with_logits(
        band_logits, band_target, reduction="none"
    )
    loss_band = (band_element * band_valid).sum() / band_valid.sum().clamp_min(1.0)
    loss_heatmap = focal_loss_hm(
        heatmap.reshape(-1, 1, 401, 401),
        heat_target.reshape(-1, 1, 401, 401),
    )
    loss_offset = (
        torch.stack(offset_losses).mean()
        if offset_losses
        else offset.sum() * 0.0
    )
    components = {
        "exist": loss_exist,
        "band": loss_band,
        "heatmap": loss_heatmap,
        "offset": loss_offset,
    }
    return combine_losses(components, config), components, matched


def module_gradient_norm(module: nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(torch.sum(parameter.grad.detach() ** 2).item())
    return math.sqrt(total)


def gradient_probe(
    context: TrainContext,
    data: g2.CachedBatch,
    indices: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
    loss_part: str,
    *,
    stop_gradient: bool = False,
) -> dict[str, float]:
    for parameter in context.parameters:
        parameter.grad = None
    query, logits, attention, heatmap, offset = forward_indices(
        context, indices, device, stop_gradient=stop_gradient
    )
    query.retain_grad()
    logits.retain_grad()
    attention.retain_grad()
    _, components, _ = compute_losses(
        logits, heatmap, offset, data, indices, config
    )
    if loss_part == "heatmap":
        loss = components["heatmap"]
    elif loss_part == "offset":
        loss = components["offset"]
    elif loss_part == "combined":
        loss = components["heatmap"] + components["offset"]
    else:
        raise ValueError(loss_part)
    loss.backward()

    def tensor_norm(tensor: torch.Tensor | None) -> float:
        return 0.0 if tensor is None else float(tensor.detach().norm().item())

    return {
        "loss": float(loss.detach().item()),
        "band_logits": tensor_norm(logits.grad),
        "query": tensor_norm(query.grad),
        "spatial_attention": tensor_norm(attention.grad),
        "splitter_parameters": module_gradient_norm(context.splitter),
        "source_head_parameters": module_gradient_norm(context.source_head),
    }


def all_finite(values: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in values.values())


def run_gradient_audit(
    data: g2.CachedBatch, config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    context = build_train_context(
        data,
        device,
        int(config["seed"]),
        load_source_endpoint=True,
    )
    positive = torch.tensor([8, 16, 24], dtype=torch.long)
    empty = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    positive_results = {
        name: gradient_probe(context, data, positive, config, device, name)
        for name in ("heatmap", "offset", "combined")
    }
    stop_result = gradient_probe(
        context,
        data,
        positive,
        config,
        device,
        "combined",
        stop_gradient=True,
    )
    empty_result = gradient_probe(
        context, data, empty, config, device, "heatmap"
    )
    combined = positive_results["combined"]
    positive_pass = (
        all_finite(combined)
        and all(combined[name] > 0.0 for name in (
            "band_logits",
            "query",
            "spatial_attention",
            "splitter_parameters",
            "source_head_parameters",
        ))
    )
    stop_pass = (
        all_finite(stop_result)
        and stop_result["band_logits"] == 0.0
        and stop_result["query"] == 0.0
        and stop_result["spatial_attention"] > 0.0
    )
    component_finite = all(
        all_finite(result) for result in positive_results.values()
    ) and all_finite(empty_result)
    return {
        "status": "PASS" if positive_pass and stop_pass and component_finite else "FAIL",
        "checkpoint_basis": identity(SOURCE_P2_CHECKPOINT),
        "positive_indices": positive.tolist(),
        "positive_k": data.counts[positive].tolist(),
        "empty_indices": empty.tolist(),
        "positive_e2e": positive_results,
        "positive_stop_gradient_combined": stop_result,
        "empty_heatmap": empty_result,
        "checks": {
            "positive_combined_path": positive_pass,
            "stop_gradient_boundary": stop_pass,
            "all_component_values_finite": component_finite,
        },
        "interpretation_boundary": "有限非零只证明正源定位梯度通路存在，不证明方向有利或能够改善泛化。",
    }


def run_input_comparison(
    run_root: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    _, d8, checkpoint = g1.build_models(device)
    d8.eval()
    geometry = g1.receiver_geometry(device)
    grouped_root = validate_output_path(run_root / "p0_grouped_cache")
    grouped_root.mkdir()
    rows_full: list[dict[str, Any]] = []
    rows_grouped: list[dict[str, Any]] = []
    topology: list[dict[str, Any]] = []
    grouped_files: list[dict[str, Any]] = []
    with g1.SampleStore("train") as store:
        for ordinal, record in enumerate(manifest["records"], start=1):
            sample = store.sample(record)
            count = int(sample["true_k"])
            true_positions = np.asarray(
                sample["positions_m"][:count], dtype=np.float32
            )
            full_path = SOURCE_RUN / "p2_cache" / f"full_{ordinal:03d}.npy"
            full_array = np.load(full_path, allow_pickle=False)
            if (
                full_array.shape != (401, 401)
                or full_array.dtype != np.float32
                or not np.isfinite(full_array).all()
            ):
                raise RuntimeError(f"全频缓存内容异常: {full_path}")
            full_pred = preflight.decode_known_k(
                d8, torch.from_numpy(full_array).to(device), count
            )
            rows_full.append(
                {
                    **metric_row(
                        ordinal,
                        int(record["raw_index"]),
                        true_positions,
                        full_pred,
                    ),
                    "cache": identity(full_path),
                }
            )

            raw, raw_local = store._raw(int(record["raw_index"]))
            fc = np.asarray(
                raw["fc_offset_all"][:, raw_local], dtype=np.float64
            ).reshape(-1)
            bw = np.asarray(
                raw["BW_actual_all"][:, raw_local], dtype=np.float64
            ).reshape(-1)
            groups = group_sources_by_freq_overlap(fc, bw, count)
            predictions: list[np.ndarray] = []
            group_rows: list[dict[str, Any]] = []
            for group_index, group in enumerate(groups):
                mask = preflight.frequency_mask(
                    group["freq_lo"], group["freq_hi"], device
                )
                current = preflight.dpd_map(
                    sample["signal"], mask.to(torch.float64), geometry, config
                )
                array = current.detach().cpu().numpy().astype(np.float32)
                path = grouped_root / f"group_{ordinal:03d}_{group_index:02d}.npy"
                np.save(path, array)
                description = preflight.cache_description(path, array, 0.0)
                if not description["finite"] or not description["nonconstant"]:
                    raise RuntimeError(f"分组DPD异常: {path}")
                grouped_files.append(description)
                predictions.append(
                    preflight.decode_known_k(d8, current, int(group["n_src"]))
                )
                group_rows.append(
                    {
                        "slots": [int(slot) for slot in group["slots"]],
                        "source_count": int(group["n_src"]),
                        "frequency_low_hz": float(group["freq_lo"]),
                        "frequency_high_hz": float(group["freq_hi"]),
                        "cache": description,
                    }
                )
            grouped_pred = (
                np.concatenate(predictions, axis=0)
                if predictions
                else np.empty((0, 2), dtype=np.float32)
            )
            rows_grouped.append(
                metric_row(
                    ordinal,
                    int(record["raw_index"]),
                    true_positions,
                    grouped_pred,
                )
            )
            active_fc = np.sort(fc[:count])
            topology.append(
                {
                    "ordinal": ordinal,
                    "raw_index": int(record["raw_index"]),
                    "true_count": count,
                    "group_count": len(groups),
                    "maximum_group_source_count": max(
                        [int(group["n_src"]) for group in groups], default=0
                    ),
                    "minimum_center_frequency_gap_hz": (
                        float(np.min(np.diff(active_fc))) if count >= 2 else None
                    ),
                    "groups": group_rows,
                }
            )
            print(f"[R1-P0 input] {ordinal}/32 K={count}", flush=True)
    return {
        "status": "PASS",
        "checkpoint": checkpoint,
        "fixed_fullband": summarize_with_k(rows_full),
        "grouped": summarize_with_k(rows_grouped),
        "samples_fixed_fullband": rows_full,
        "samples_grouped": rows_grouped,
        "frequency_topology": topology,
        "grouped_cache_files": grouped_files,
    }


def run_p0(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if (run_root / "p0_report.json").exists():
        raise FileExistsError("拒绝覆盖P0报告")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    comparison = run_input_comparison(run_root, manifest, config, device)
    data = g2.load_samples(SOURCE_RUN, manifest)
    gradient = run_gradient_audit(data, config, device)
    synthetic = {
        name: torch.tensor(float(index + 1))
        for index, name in enumerate(("exist", "band", "heatmap", "offset"))
    }
    weighted = float(combine_losses(synthetic, config).item())
    expected = sum(
        float(config["loss_weights"][name]) * float(value.item())
        for name, value in synthetic.items()
    )
    loss_contract = math.isclose(weighted, expected, rel_tol=0.0, abs_tol=0.0)
    status = (
        "PASS"
        if comparison["status"] == "PASS"
        and gradient["status"] == "PASS"
        and loss_contract
        else "FAIL"
    )
    report = {
        "status": status,
        "gate": "E2E-G2-R1-P0",
        "input_comparison": comparison,
        "gradient_audit": gradient,
        "loss_weight_contract": {
            "pass": loss_contract,
            "computed": weighted,
            "expected": expected,
            "weights": config["loss_weights"],
        },
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
    }
    write_json(run_root / "p0_report.json", report)
    print(json.dumps({"status": status, "duration_seconds": report["duration_seconds"]}, ensure_ascii=False), flush=True)
    return report


def assignment_signature(mapping: dict[int, int], count: int) -> list[int]:
    source_to_query = [-1] * count
    for query, source in mapping.items():
        source_to_query[source] = query
    return source_to_query


@torch.no_grad()
def evaluate_model(
    context: TrainContext,
    data: g2.CachedBatch,
    device: torch.device,
    previous_assignments: list[list[int]] | None,
) -> tuple[dict[str, Any], list[list[int]]]:
    outputs: list[tuple[torch.Tensor, ...]] = []
    for start in range(0, len(data.counts), 4):
        indices = torch.arange(start, min(start + 4, len(data.counts)))
        outputs.append(
            tuple(
                item.cpu()
                for item in forward_indices(context, indices, device)
            )
        )
    logits = torch.cat([item[1] for item in outputs])
    attention = torch.cat([item[2] for item in outputs])
    heatmap = torch.cat([item[3] for item in outputs])
    offset = torch.cat([item[4] for item in outputs])
    mappings = g2.assignments(
        logits,
        heatmap,
        data.band,
        data.ignore,
        data.positions,
        data.counts,
    )
    signatures = [
        assignment_signature(mapping, int(data.counts[index].item()))
        for index, mapping in enumerate(mappings)
    ]
    flips = 0
    flip_eligible = 0
    if previous_assignments is not None:
        for index, signature in enumerate(signatures):
            if int(data.counts[index].item()) >= 2:
                flip_eligible += 1
                flips += int(signature != previous_assignments[index])

    predicted_counts = (logits.amax(dim=-1) >= 0.0).sum(dim=-1)
    rows: list[dict[str, Any]] = []
    f1_values: list[float] = []
    target_ranks: list[int] = []
    collapse_count = 0
    offset_worsened = 0
    overlap_attention: list[float] = []
    for index, count_value in enumerate(data.counts.tolist()):
        count = int(count_value)
        active_queries = torch.nonzero(
            logits[index].amax(dim=-1) >= 0.0, as_tuple=False
        ).flatten().tolist()
        grid_predictions: list[list[float]] = []
        offset_predictions: list[list[float]] = []
        query_predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for source_query in active_queries:
            flat = torch.sigmoid(heatmap[index, source_query]).flatten()
            peak = int(torch.argmax(flat).item())
            iy, ix = divmod(peak, 401)
            delta = offset[index, source_query, :, iy, ix].clamp(-1.0, 1.0)
            grid_position = np.asarray(
                [ix * 10.0 - 2000.0, iy * 10.0 - 2000.0], dtype=np.float32
            )
            offset_position = np.asarray(
                [
                    (ix + float(delta[0])) * 10.0 - 2000.0,
                    (iy + float(delta[1])) * 10.0 - 2000.0,
                ],
                dtype=np.float32,
            )
            grid_predictions.append(grid_position.tolist())
            offset_predictions.append(offset_position.tolist())
            query_predictions[source_query] = (grid_position, offset_position)
        grid_array = np.asarray(grid_predictions, dtype=np.float32).reshape(-1, 2)
        offset_array = np.asarray(offset_predictions, dtype=np.float32).reshape(-1, 2)
        truth_positions = data.positions[index, :count].numpy()
        gospa = g1.gospa_sample(truth_positions, offset_array)
        row: dict[str, Any] = {
            "index": index,
            "true_count": count,
            "predicted_count": int(predicted_counts[index].item()),
            "grid_positions_m": grid_array.tolist(),
            "predicted_positions_m": offset_array.tolist(),
            "gospa_m": gospa["value_m"],
            "assignment": signatures[index],
        }
        for threshold in (10, 30, 50, 100):
            row[f"tp_at_{threshold}m"] = g1.maximum_matches_within(
                truth_positions, offset_array, float(threshold)
            )
            row[f"grid_tp_at_{threshold}m"] = g1.maximum_matches_within(
                truth_positions, grid_array, float(threshold)
            )
        rows.append(row)

        if count >= 2 and len(offset_array) >= 2:
            minimum_distance = min(
                np.linalg.norm(offset_array[a] - offset_array[b])
                for a in range(len(offset_array))
                for b in range(a + 1, len(offset_array))
            )
            collapse_count += int(float(minimum_distance) < 10.0)
        mapping = mappings[index]
        for source_query, source in mapping.items():
            valid = data.ignore[index, source] < 0.5
            prediction = logits[index, source_query, valid] >= 0.0
            target = data.band[index, source, valid] > 0.5
            tp = int((prediction & target).sum().item())
            fp = int((prediction & ~target).sum().item())
            fn = int((~prediction & target).sum().item())
            f1_values.append(2 * tp / max(2 * tp + fp + fn, 1))
            px = int(
                torch.round(
                    (data.positions[index, source, 0] + g1.FINE_EDGE) / g1.FINE_STEP
                ).clamp(0, 400).item()
            )
            py = int(
                torch.round(
                    (data.positions[index, source, 1] + g1.FINE_EDGE) / g1.FINE_STEP
                ).clamp(0, 400).item()
            )
            scores = torch.sigmoid(heatmap[index, source_query])
            target_score = scores[py, px]
            target_ranks.append(1 + int((scores > target_score).sum().item()))
            if source_query in query_predictions:
                grid_position, offset_position = query_predictions[source_query]
                truth = truth_positions[source]
                offset_worsened += int(
                    np.linalg.norm(grid_position - truth) <= 10.0
                    and np.linalg.norm(offset_position - truth) > 10.0
                )
        if bool(data.overlap[index]) and count >= 2:
            active = list(mapping)
            for left in range(len(active)):
                for right in range(left + 1, len(active)):
                    overlap_attention.append(
                        float(
                            torch.mean(
                                torch.abs(
                                    attention[index, active[left]]
                                    - attention[index, active[right]]
                                )
                            ).item()
                        )
                    )

    summary = summarize_with_k(rows)
    summary.update(
        {
            "exact_count": int((predicted_counts == data.counts).sum().item()),
            "exact_count_rate": float(
                (predicted_counts == data.counts).float().mean().item()
            ),
            "active_band_macro_f1": float(np.mean(f1_values)) if f1_values else 1.0,
            "collapsed_multisource_samples": collapse_count,
            "target_peak_rank1_rate": float(np.mean(np.asarray(target_ranks) == 1)) if target_ranks else 1.0,
            "target_peak_rank_median": float(np.median(target_ranks)) if target_ranks else 1.0,
            "offset_worsened_correct_grid_count": offset_worsened,
            "assignment_flip_count": flips,
            "assignment_flip_eligible_samples": flip_eligible,
            "assignment_flip_rate": flips / max(flip_eligible, 1),
            "overlap_attention_mean_l1": float(np.mean(overlap_attention)) if overlap_attention else 0.0,
            "samples": rows,
        }
    )
    return summary, signatures


def metric_snapshot(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "samples"}


def selection_score(metrics: dict[str, Any], config: dict[str, Any]) -> tuple[Any, ...]:
    auxiliary = (
        metrics["exact_count"] >= int(config["count_minimum"])
        and metrics["active_band_macro_f1"] >= float(config["band_f1_minimum"])
    )
    return (
        int(auxiliary),
        metrics["overall"]["recall_at_10m"],
        -metrics["collapsed_multisource_samples"],
        metrics["overall"]["recall_at_30m"],
        -metrics["overall"]["gospa_mean_m"],
    )


def save_context_checkpoint(
    path: Path,
    context: TrainContext,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "ch3_heads": context.ch3.band_heads[:3].state_dict(),
            "query_builder": context.query_builder.state_dict(),
            "splitter": context.splitter.state_dict(),
            "source_head": context.source_head.state_dict(),
            "metrics": metric_snapshot(metrics),
        },
        validate_output_path(path),
    )


def run_p1(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if read_json(run_root / "p0_report.json")["status"] != "PASS":
        raise RuntimeError("R1-P0未通过，禁止执行P1")
    if (run_root / "p1_report.json").exists():
        raise FileExistsError("拒绝覆盖P1报告")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = g2.load_samples(SOURCE_RUN, manifest)
    context = build_train_context(data, device, int(config["seed"]))
    optimizer = torch.optim.AdamW(
        context.parameters,
        lr=float(config["p1_initial_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    history: list[dict[str, Any]] = []
    best_metrics: dict[str, Any] | None = None
    best_row: dict[str, Any] | None = None
    best_30_metrics: dict[str, Any] | None = None
    best_30_row: dict[str, Any] | None = None
    previous_assignments: list[list[int]] | None = None
    capacity_pass = False
    timed_out = False
    started = time.perf_counter()
    for epoch in range(1, int(config["p1_max_epochs"]) + 1):
        if epoch == 121:
            optimizer.param_groups[0]["lr"] = float(config["p1_epoch_121_learning_rate"])
        elif epoch == 201:
            optimizer.param_groups[0]["lr"] = float(config["p1_epoch_201_learning_rate"])
        order = torch.randperm(len(data.counts), generator=generator)
        sums = {name: 0.0 for name in ("exist", "band", "heatmap", "offset")}
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])]
            _, logits, _, heatmap, offset = forward_indices(
                context, indices, device
            )
            total, components, _ = compute_losses(
                logits, heatmap, offset, data, indices, config
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                context.parameters, float(config["gradient_clip"])
            )
            optimizer.step()
            for name, value in components.items():
                sums[name] += float(value.detach().item()) * len(indices)
        if epoch % int(config["evaluate_every"]) == 0 or epoch == 1:
            metrics, current_assignments = evaluate_model(
                context, data, device, previous_assignments
            )
            previous_assignments = current_assignments
            row = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "losses": {
                    name: value / len(data.counts) for name, value in sums.items()
                },
                "metrics": metric_snapshot(metrics),
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            write_json(run_root / "p1_progress.json", history)
            if best_metrics is None or selection_score(metrics, config) > selection_score(best_metrics, config):
                best_metrics = metrics
                best_row = row
                save_context_checkpoint(
                    run_root / "p1_best_recall10.pt", context, epoch, metrics
                )
            score_30 = (
                metrics["overall"]["recall_at_30m"],
                metrics["overall"]["recall_at_10m"],
                -metrics["collapsed_multisource_samples"],
                -metrics["overall"]["gospa_mean_m"],
            )
            if best_30_metrics is None:
                better_30 = True
            else:
                previous_30 = (
                    best_30_metrics["overall"]["recall_at_30m"],
                    best_30_metrics["overall"]["recall_at_10m"],
                    -best_30_metrics["collapsed_multisource_samples"],
                    -best_30_metrics["overall"]["gospa_mean_m"],
                )
                better_30 = score_30 > previous_30
            if better_30:
                best_30_metrics = metrics
                best_30_row = row
                save_context_checkpoint(
                    run_root / "p1_best_recall30.pt", context, epoch, metrics
                )
            capacity_pass = (
                metrics["exact_count"] >= int(config["count_minimum"])
                and metrics["active_band_macro_f1"] >= float(config["band_f1_minimum"])
                and metrics["overall"]["recall_at_10m"] >= float(config["recall_10m_minimum"])
                and metrics["collapsed_multisource_samples"] == 0
            )
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "lr": row["learning_rate"],
                        "losses": row["losses"],
                        "r10": metrics["overall"]["recall_at_10m"],
                        "r30": metrics["overall"]["recall_at_30m"],
                        "count": metrics["exact_count"],
                        "band_f1": metrics["active_band_macro_f1"],
                        "collapsed": metrics["collapsed_multisource_samples"],
                        "flip_rate": metrics["assignment_flip_rate"],
                        "elapsed": row["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if capacity_pass:
                break
        if time.perf_counter() - started > float(config["p1_wall_limit_seconds"]):
            timed_out = True
            break
    if best_metrics is None or best_row is None or best_30_metrics is None or best_30_row is None:
        raise RuntimeError("P1没有产生评价结果")
    auxiliary_ever_passed = any(
        row["metrics"]["exact_count"] >= int(config["count_minimum"])
        and row["metrics"]["active_band_macro_f1"] >= float(config["band_f1_minimum"])
        for row in history
    )
    if timed_out:
        status = "G2_R1_STOP_RESOURCE"
    elif capacity_pass:
        status = "G2_R1_CAPACITY_PASS"
    elif not auxiliary_ever_passed:
        status = "G2_R1_AUX_REGRESSION"
    elif (
        best_30_metrics["overall"]["recall_at_30m"]
        >= float(config["recall_30m_minimum"])
        and best_30_metrics["collapsed_multisource_samples"] == 0
    ):
        status = "G2_R1_SPLIT_PASS_FINE_FAIL"
    else:
        status = "G2_R1_MULTISOURCE_UNRESOLVED"
    report = {
        "status": status,
        "gate": "E2E-G2-R1-P1",
        "epochs_completed": history[-1]["epoch"],
        "optimizer_steps": history[-1]["epoch"] * math.ceil(len(data.counts) / int(config["batch_size"])),
        "best_recall10": {**best_row, "samples": best_metrics["samples"]},
        "best_recall30": {**best_30_row, "samples": best_30_metrics["samples"]},
        "duration_seconds": time.perf_counter() - started,
        "timed_out": timed_out,
        "test_executed": False,
    }
    write_json(run_root / "p1_report.json", report)
    print(json.dumps({"status": status, "duration_seconds": report["duration_seconds"]}, ensure_ascii=False), flush=True)
    return report


def compute_k1_losses(
    logits: torch.Tensor,
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    data: g2.CachedBatch,
    indices: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = logits.device
    query_logits = logits[:, 0]
    truth = data.band[indices, 0].to(device)
    valid = 1.0 - data.ignore[indices, 0].to(device)
    loss_exist = F.binary_cross_entropy_with_logits(
        query_logits.amax(dim=-1), torch.ones_like(query_logits[:, 0])
    )
    band_element = F.binary_cross_entropy_with_logits(
        query_logits, truth, reduction="none"
    )
    loss_band = (band_element * valid).sum() / valid.sum().clamp_min(1.0)
    targets = []
    offsets = []
    for batch_index, global_index in enumerate(indices.tolist()):
        target, ix, iy, delta = g2.single_source_target(
            data.positions[global_index, 0].to(device), device
        )
        targets.append(target)
        offsets.append(torch.abs(offset[batch_index, 0, :, iy, ix] - delta).sum())
    target_tensor = torch.stack(targets)[:, None]
    loss_heatmap = focal_loss_hm(heatmap[:, 0:1], target_tensor)
    loss_offset = torch.stack(offsets).mean()
    components = {
        "exist": loss_exist,
        "band": loss_band,
        "heatmap": loss_heatmap,
        "offset": loss_offset,
    }
    return combine_losses(components, config), components


@torch.no_grad()
def evaluate_k1(
    context: TrainContext,
    data: g2.CachedBatch,
    indices: torch.Tensor,
    device: torch.device,
) -> dict[str, Any]:
    rows = []
    for start in range(0, len(indices), 4):
        current_indices = indices[start : start + 4]
        _, _, _, heatmap, offset = forward_indices(
            context, current_indices, device
        )
        for local, global_index in enumerate(current_indices.tolist()):
            scores = torch.sigmoid(heatmap[local, 0])
            peak = int(torch.argmax(scores.flatten()).item())
            iy, ix = divmod(peak, 401)
            delta = offset[local, 0, :, iy, ix].clamp(-1.0, 1.0)
            grid = np.asarray(
                [ix * 10.0 - 2000.0, iy * 10.0 - 2000.0], dtype=np.float32
            )
            predicted = np.asarray(
                [
                    (ix + float(delta[0])) * 10.0 - 2000.0,
                    (iy + float(delta[1])) * 10.0 - 2000.0,
                ],
                dtype=np.float32,
            )
            truth = data.positions[global_index, 0].numpy()
            px = int(torch.round((data.positions[global_index, 0, 0] + g1.FINE_EDGE) / g1.FINE_STEP).clamp(0, 400).item())
            py = int(torch.round((data.positions[global_index, 0, 1] + g1.FINE_EDGE) / g1.FINE_STEP).clamp(0, 400).item())
            target_score = scores[py, px]
            rank = 1 + int((scores > target_score).sum().item())
            grid_error = float(np.linalg.norm(grid - truth))
            error = float(np.linalg.norm(predicted - truth))
            rows.append(
                {
                    "global_index": global_index,
                    "target_peak_rank": rank,
                    "grid_error_m": grid_error,
                    "error_m": error,
                    "grid_within_10m": grid_error <= 10.0,
                    "within_10m": error <= 10.0,
                    "offset_worsened_correct_grid": grid_error <= 10.0 and error > 10.0,
                }
            )
    return {
        "sample_count": len(rows),
        "rank1_count": sum(int(row["target_peak_rank"] == 1) for row in rows),
        "within_10m_count": sum(int(row["within_10m"]) for row in rows),
        "grid_within_10m_count": sum(int(row["grid_within_10m"]) for row in rows),
        "offset_worsened_count": sum(int(row["offset_worsened_correct_grid"]) for row in rows),
        "mean_error_m": float(np.mean([row["error_m"] for row in rows])),
        "maximum_error_m": float(np.max([row["error_m"] for row in rows])),
        "rows": rows,
    }


def train_k1(
    run_root: Path,
    label: str,
    data: g2.CachedBatch,
    dpd: torch.Tensor,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    context = build_train_context(
        data, device, seed, dpd_override=dpd
    )
    indices = torch.arange(8, 16, dtype=torch.long)
    if not bool(torch.all(data.counts[indices] == 1)):
        raise RuntimeError("K=1条件样本索引合同失败")
    optimizer = torch.optim.AdamW(
        context.parameters,
        lr=float(config["p1_initial_learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    history = []
    best: dict[str, Any] | None = None
    started = time.perf_counter()
    timed_out = False
    passed = False
    for epoch in range(1, int(config["p2_max_epochs"]) + 1):
        if epoch == 121:
            optimizer.param_groups[0]["lr"] = float(config["p1_epoch_121_learning_rate"])
        elif epoch == 201:
            optimizer.param_groups[0]["lr"] = float(config["p1_epoch_201_learning_rate"])
        order = indices[torch.randperm(len(indices), generator=generator)]
        sums = {name: 0.0 for name in ("exist", "band", "heatmap", "offset")}
        for start in range(0, len(order), int(config["batch_size"])):
            current_indices = order[start : start + int(config["batch_size"])]
            _, logits, _, heatmap, offset = forward_indices(
                context, current_indices, device
            )
            total, components = compute_k1_losses(
                logits, heatmap, offset, data, current_indices, config
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                context.parameters, float(config["gradient_clip"])
            )
            optimizer.step()
            for name, value in components.items():
                sums[name] += float(value.detach().item()) * len(current_indices)
        if epoch % int(config["evaluate_every"]) == 0 or epoch == 1:
            metrics = evaluate_k1(context, data, indices, device)
            row = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "losses": {name: value / len(indices) for name, value in sums.items()},
                "metrics": {key: value for key, value in metrics.items() if key != "rows"},
                "elapsed_seconds": time.perf_counter() - started,
            }
            history.append(row)
            write_json(run_root / f"p2_{label}_seed{seed}_progress.json", history)
            score = (
                metrics["within_10m_count"],
                metrics["rank1_count"],
                -metrics["offset_worsened_count"],
                -metrics["mean_error_m"],
            )
            if best is None:
                better = True
            else:
                old = best["metrics"]
                better = score > (
                    old["within_10m_count"],
                    old["rank1_count"],
                    -old["offset_worsened_count"],
                    -old["mean_error_m"],
                )
            if better:
                best = {**row, "metrics": metrics}
                save_context_checkpoint(
                    run_root / f"p2_{label}_seed{seed}_best.pt",
                    context,
                    epoch,
                    metrics,
                )
            passed = (
                metrics["rank1_count"] == 8
                and metrics["within_10m_count"] == 8
                and metrics["offset_worsened_count"] == 0
            )
            print(
                json.dumps(
                    {
                        "label": label,
                        "seed": seed,
                        "epoch": epoch,
                        "rank1": metrics["rank1_count"],
                        "within10": metrics["within_10m_count"],
                        "mean_error": metrics["mean_error_m"],
                        "elapsed": row["elapsed_seconds"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if passed:
                break
        if time.perf_counter() - started > float(config["p2_wall_limit_seconds_per_run"]):
            timed_out = True
            break
    if best is None:
        raise RuntimeError("K=1运行没有评价结果")
    report = {
        "status": "PASS" if passed else ("STOP_RESOURCE" if timed_out else "FAIL"),
        "label": label,
        "seed": seed,
        "epochs_completed": history[-1]["epoch"],
        "best": best,
        "duration_seconds": time.perf_counter() - started,
        "timed_out": timed_out,
    }
    write_json(run_root / f"p2_{label}_seed{seed}_report.json", report)
    return report


def prepare_oracle_k1_cache(
    run_root: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    data: g2.CachedBatch,
    device: torch.device,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    root = validate_output_path(run_root / "p2_oracle_k1_cache")
    root.mkdir()
    geometry = g1.receiver_geometry(device)
    output = data.dpd.clone()
    descriptions = []
    with g1.SampleStore("train") as store:
        for global_index in range(8, 16):
            record = manifest["records"][global_index]
            sample = store.sample(record)
            raw, raw_local = store._raw(int(record["raw_index"]))
            fc = np.asarray(raw["fc_offset_all"][:, raw_local], dtype=np.float64).reshape(-1)
            bw = np.asarray(raw["BW_actual_all"][:, raw_local], dtype=np.float64).reshape(-1)
            groups = group_sources_by_freq_overlap(fc, bw, 1)
            if len(groups) != 1:
                raise RuntimeError("K=1 oracle频带组数不是1")
            mask = preflight.frequency_mask(
                groups[0]["freq_lo"], groups[0]["freq_hi"], device
            )
            current = preflight.dpd_map(
                sample["signal"], mask.to(torch.float64), geometry, config
            )
            array = current.detach().cpu().numpy().astype(np.float32)
            path = root / f"oracle_{global_index:02d}.npy"
            np.save(path, array)
            description = preflight.cache_description(path, array, 0.0)
            if not description["finite"] or not description["nonconstant"]:
                raise RuntimeError(f"K=1 oracle缓存异常: {path}")
            descriptions.append(description)
            output[global_index] = torch.from_numpy(array)
            print(f"[R1-P2 oracle cache] {global_index - 7}/8", flush=True)
    return output, descriptions


def run_p2(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    p1 = read_json(run_root / "p1_report.json")
    if p1["status"] != "G2_R1_MULTISOURCE_UNRESOLVED":
        raise RuntimeError(f"P1状态不要求K=1诊断: {p1['status']}")
    if (run_root / "p2_report.json").exists():
        raise FileExistsError("拒绝覆盖P2报告")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = g2.load_samples(SOURCE_RUN, manifest)
    base_seed = int(config["seed"])
    full_runs = [
        train_k1(run_root, "fullband", data, data.dpd, config, base_seed, device)
    ]
    if full_runs[0]["status"] != "PASS":
        full_runs.append(
            train_k1(
                run_root,
                "fullband",
                data,
                data.dpd,
                config,
                base_seed + 1,
                device,
            )
        )
    full_pass = any(run["status"] == "PASS" for run in full_runs)
    oracle_runs: list[dict[str, Any]] = []
    oracle_cache: list[dict[str, Any]] = []
    if not full_pass:
        oracle_dpd, oracle_cache = prepare_oracle_k1_cache(
            run_root, manifest, config, data, device
        )
        oracle_runs.append(
            train_k1(
                run_root,
                "oracle_union",
                data,
                oracle_dpd,
                config,
                base_seed,
                device,
            )
        )
        if oracle_runs[0]["status"] != "PASS":
            oracle_runs.append(
                train_k1(
                    run_root,
                    "oracle_union",
                    data,
                    oracle_dpd,
                    config,
                    base_seed + 1,
                    device,
                )
            )
    oracle_pass = any(run["status"] == "PASS" for run in oracle_runs)
    if full_pass:
        status = "G2_R1_MULTISOURCE_SPLITTER_OR_MATCHING"
    elif oracle_pass:
        status = "G2_R1_FULLBAND_INPUT_BLOCKER"
    else:
        status = "G2_R1_SINGLE_SOURCE_INTERFACE_BLOCKER"
    report = {
        "status": status,
        "gate": "E2E-G2-R1-P2",
        "fullband_runs": full_runs,
        "oracle_union_runs": oracle_runs,
        "oracle_cache": oracle_cache,
        "test_executed": False,
    }
    write_json(run_root / "p2_report.json", report)
    print(json.dumps({"status": status}, ensure_ascii=False), flush=True)
    return report


def finalize(run_root: Path) -> dict[str, Any]:
    manifest, _ = verify_run(run_root)
    p0 = read_json(run_root / "p0_report.json") if (run_root / "p0_report.json").exists() else None
    p1 = read_json(run_root / "p1_report.json") if (run_root / "p1_report.json").exists() else None
    p2 = read_json(run_root / "p2_report.json") if (run_root / "p2_report.json").exists() else None
    if p0 is None or p0["status"] != "PASS":
        status = "G2_R1_STOP_P0"
    elif p1 is None:
        status = "G2_R1_INCOMPLETE"
    elif p1["status"] == "G2_R1_MULTISOURCE_UNRESOLVED" and p2 is None:
        status = "G2_R1_INCOMPLETE"
    elif p2 is not None:
        status = p2["status"]
    else:
        status = p1["status"]
    report = {
        "status": status,
        "gate": "E2E-G2-R1",
        "run_root": str(run_root.resolve()),
        "manifest": identity(run_root / "manifest.json"),
        "p0": None if p0 is None else {"status": p0["status"]},
        "p1": None if p1 is None else {"status": p1["status"]},
        "p2": None if p2 is None else {"status": p2["status"]},
        "test_executed": False,
        "source_run_unchanged": (
            identity(SOURCE_P2_MANIFEST) == manifest["source_p2_manifest"]
            and identity(SOURCE_P2_CHECKPOINT) == manifest["source_p2_checkpoint"]
            and [identity(path) for path in sorted((SOURCE_RUN / "p2_cache").glob("full_*.npy"))]
            == manifest["source_fullband_cache"]
        ),
    }
    write_json(run_root / "final_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    for command in ("run-p0", "run-p1", "run-p2", "finalize"):
        current = sub.add_parser(command)
        current.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare":
        prepare(args.run_id)
    elif args.command == "run-p0":
        run_p0(args.run_root.resolve())
    elif args.command == "run-p1":
        run_p1(args.run_root.resolve())
    elif args.command == "run-p2":
        run_p2(args.run_root.resolve())
    else:
        finalize(args.run_root.resolve())
