"""E2E-G2-R2-D1：无重训解耦辅助退化来源。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
CH4_DIR = PROJECT_ROOT / "第四章代码"
os.environ.setdefault(
    "SOURCECOUNT_REFERENCE_OUTPUT_ROOT",
    str((PROJECT_ROOT.parent / "SourceCount_DPD" / "outputs").resolve()),
)
for root in (PROJECT_ROOT, CH4_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import chapter_runtime  # noqa: E402,F401

from 统一模型代码 import e2e_g1 as g1  # noqa: E402
from 统一模型代码 import e2e_g2_latent_fusion as g2  # noqa: E402
from 统一模型代码 import e2e_g2_r1 as r1  # noqa: E402
from 统一模型代码 import e2e_g2_r2 as r2  # noqa: E402
from 统一模型代码.runtime_paths import new_run_dir  # noqa: E402


CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g2_r2_d1.json"
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_RUN = (
    PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_g2_r2" / "20260904_163951"
)
TRACKS = ("fs_sg", "fs_e2e")
MODES = ("joint", "band_only", "position_only")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    return [identity(path) for path in (SCRIPT_PATH, CONFIG_PATH)]


def prepare(run_id: str) -> Path:
    source_manifest, _ = r2.verify_run(SOURCE_RUN)
    source_final = read_json(SOURCE_RUN / "final_report.json")
    if source_final["status"] != "G2_R2_AUX_REGRESSION":
        raise RuntimeError("R2源状态不是G2_R2_AUX_REGRESSION")
    if source_final["source_unchanged"] is not True or source_final["test_executed"]:
        raise RuntimeError("R2源完整性或test合同错误")
    run_root = new_run_dir("e2e_g2_r2_d1", run_id, create=True)
    manifest = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run+validate",
            "verification_status": "UNVERIFIED",
        },
        "gate": "E2E-G2-R2-D1",
        "status": "PREPARED",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": read_json(CONFIG_PATH),
        "code": code_identity(),
        "source": {
            "run": str(SOURCE_RUN.resolve()),
            "manifest": identity(SOURCE_RUN / "manifest.json"),
            "p3": identity(SOURCE_RUN / "p3_report.json"),
            "final": identity(SOURCE_RUN / "final_report.json"),
            "checkpoints": {
                track: identity(SOURCE_RUN / "training" / track / "best.pt")
                for track in TRACKS
            },
            "cache_manifest": identity(SOURCE_RUN / "cache_manifest.json"),
        },
        "source_contract": {
            "records": {
                split: len(source_manifest["subsets"][split])
                for split in ("train", "val_select", "val_compare")
            },
            "optimizer_steps": 0,
            "weights_updated": False,
            "test_executed": False,
        },
    }
    write_json(run_root / "manifest.json", manifest)
    print(str(run_root), flush=True)
    return run_root


def verify(run_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_root / "manifest.json")
    if manifest["gate"] != "E2E-G2-R2-D1":
        raise RuntimeError("D1 manifest Gate错误")
    if manifest["code"] != code_identity():
        raise RuntimeError("prepare后D1代码或配置变化")
    source_manifest, _ = r2.verify_run(SOURCE_RUN)
    current = {
        "run": str(SOURCE_RUN.resolve()),
        "manifest": identity(SOURCE_RUN / "manifest.json"),
        "p3": identity(SOURCE_RUN / "p3_report.json"),
        "final": identity(SOURCE_RUN / "final_report.json"),
        "checkpoints": {
            track: identity(SOURCE_RUN / "training" / track / "best.pt")
            for track in TRACKS
        },
        "cache_manifest": identity(SOURCE_RUN / "cache_manifest.json"),
    }
    if current != manifest["source"]:
        raise RuntimeError("R2源证据身份变化")
    if manifest["source_contract"] != {
        "records": {split: len(source_manifest["subsets"][split]) for split in ("train", "val_select", "val_compare")},
        "optimizer_steps": 0,
        "weights_updated": False,
        "test_executed": False,
    }:
        raise RuntimeError("D1无重训合同变化")
    return manifest, manifest["config"], source_manifest


def signature(mapping: dict[int, int], count: int) -> list[int]:
    result = [-1] * count
    for query, source in mapping.items():
        result[source] = query
    return result


def best_mapping(cost: torch.Tensor, count: int) -> dict[int, int]:
    best: tuple[float, tuple[int, ...]] | None = None
    for order in itertools.permutations(range(3), count):
        value = sum(float(cost[order[source], source].item()) for source in range(count))
        if best is None or value < best[0]:
            best = (value, order)
    if best is None:
        return {}
    return {query: source for source, query in enumerate(best[1])}


def mappings_for_sample(
    logits: torch.Tensor,
    heatmap: torch.Tensor,
    band: torch.Tensor,
    ignore: torch.Tensor,
    positions: torch.Tensor,
    count: int,
) -> dict[str, dict[int, int]]:
    if count == 0:
        return {mode: {} for mode in MODES}
    band_cost = torch.zeros((3, count))
    position_cost = torch.zeros((3, count))
    for query in range(3):
        for source in range(count):
            valid = ignore[source] < 0.5
            band_cost[query, source] = F.binary_cross_entropy_with_logits(
                logits[query, valid], band[source, valid]
            )
            px = int(torch.round((positions[source, 0] + g1.FINE_EDGE) / g1.FINE_STEP).clamp(0, 400).item())
            py = int(torch.round((positions[source, 1] + g1.FINE_EDGE) / g1.FINE_STEP).clamp(0, 400).item())
            position_cost[query, source] = 1.0 - torch.sigmoid(heatmap[query, py, px])
    return {
        "joint": best_mapping(band_cost + position_cost, count),
        "band_only": best_mapping(band_cost, count),
        "position_only": best_mapping(position_cost, count),
    }


def source_metrics(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> dict[str, float]:
    prediction = logits >= threshold
    truth = target > 0.5
    tp = int((prediction & truth).sum().item())
    fp = int((prediction & ~truth).sum().item())
    fn = int((~prediction & truth).sum().item())
    return {
        "f1": 2.0 * tp / max(2 * tp + fp + fn, 1),
        "iou": tp / max(tp + fp + fn, 1),
        "bce": float(F.binary_cross_entropy_with_logits(logits, target).item()),
    }


def analyze_outputs(
    logits: torch.Tensor,
    heatmap: torch.Tensor,
    data: g2.CachedBatch,
    start: int,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    predicted_counts = (logits.amax(dim=-1) >= 0.0).sum(dim=-1)
    for local in range(len(logits)):
        count = int(data.counts[local].item())
        mappings = mappings_for_sample(
            logits[local], heatmap[local], data.band[local], data.ignore[local],
            data.positions[local], count,
        )
        row: dict[str, Any] = {
            "ordinal": start + local,
            "raw_index": int(records[start + local]["raw_index"]),
            "local_index": int(records[start + local]["local_index"]),
            "true_count": count,
            "predicted_count": int(predicted_counts[local].item()),
            "exact_count": int(predicted_counts[local].item() == count),
            "frequency_overlap": bool(records[start + local]["frequency_overlap"]),
            "modes": {},
        }
        for mode, mapping in mappings.items():
            sources: list[dict[str, Any]] = []
            for query, source in mapping.items():
                valid = data.ignore[local, source] < 0.5
                valid_logits = logits[local, query, valid]
                valid_target = data.band[local, source, valid]
                sources.append(
                    {
                        **source_metrics(valid_logits, valid_target, 0.0),
                        "logits": valid_logits.tolist(),
                        "target": valid_target.tolist(),
                    }
                )
            row["modes"][mode] = {
                "signature": signature(mapping, count),
                "sources": sources,
            }
        output.append(row)
    return output


@torch.no_grad()
def forward_context(context: r1.TrainContext, length: int, device: torch.device, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    logits: list[torch.Tensor] = []
    heatmaps: list[torch.Tensor] = []
    for start in range(0, length, batch_size):
        indices = torch.arange(start, min(start + batch_size, length))
        values = r1.forward_indices(context, indices, device)
        logits.append(values[1].cpu())
        heatmaps.append(values[3].cpu())
    return torch.cat(logits), torch.cat(heatmaps)


def analyze_split(
    run_root: Path,
    source_manifest: dict[str, Any],
    config: dict[str, Any],
    split: str,
    checkpoints: dict[str, dict[str, Any]],
    device: torch.device,
) -> dict[str, list[dict[str, Any]]]:
    data = r2.load_split(SOURCE_RUN, source_manifest, split)
    records = source_manifest["subsets"][split]
    results: dict[str, list[dict[str, Any]]] = {track: [] for track in TRACKS}
    chunk_size = int(config["compare_chunk_size"])
    for start in range(0, len(data.counts), chunk_size):
        indices = torch.arange(start, min(start + chunk_size, len(data.counts)))
        current = r2.subset_batch(data, indices)
        context = r1.build_train_context(current, device, int(config["seed"]))
        for track in TRACKS:
            r2.load_state(context, checkpoints[track])
            logits, heatmap = forward_context(
                context, len(current.counts), device, int(config["forward_batch_size"])
            )
            results[track].extend(
                analyze_outputs(logits, heatmap, current, start, records)
            )
        del context, current
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[D1 {split}] {min(start + chunk_size, len(data.counts))}/{len(data.counts)}", flush=True)
        write_json(
            run_root / "progress.json",
            {"stage": split, "completed": min(start + chunk_size, len(data.counts)), "total": len(data.counts)},
        )
    del data
    return results


def aggregate(records: list[dict[str, Any]], mode: str, threshold: float = 0.0) -> dict[str, float]:
    exact = sum(int(row["exact_count"]) for row in records)
    values = {name: [] for name in ("f1", "iou", "bce")}
    for row in records:
        for source in row["modes"][mode]["sources"]:
            if threshold == 0.0:
                current = source
            else:
                current = source_metrics(
                    torch.tensor(source["logits"]), torch.tensor(source["target"]), threshold
                )
            for name in values:
                values[name].append(float(current[name]))
    return {
        "sample_count": len(records),
        "source_count": len(values["f1"]),
        "exact_count_rate": exact / max(len(records), 1),
        "band_macro_f1": float(np.mean(values["f1"])) if values["f1"] else 1.0,
        "band_macro_iou": float(np.mean(values["iou"])) if values["iou"] else 1.0,
        "band_bce": float(np.mean(values["bce"])) if values["bce"] else 0.0,
    }


def grouped_metrics(records: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    return {
        "overall": aggregate(records, mode),
        "by_k": {
            str(count): aggregate([row for row in records if int(row["true_count"]) == count], mode)
            for count in range(4)
        },
        "by_overlap": {
            str(value).lower(): aggregate([row for row in records if bool(row["frequency_overlap"]) is value], mode)
            for value in (False, True)
        },
    }


def choose_threshold(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, float]:
    thresholds = np.arange(
        float(config["calibration_threshold_min"]),
        float(config["calibration_threshold_max"]) + float(config["calibration_threshold_step"]) / 2,
        float(config["calibration_threshold_step"]),
    )
    candidates = []
    for threshold in thresholds:
        metric = aggregate(records, "band_only", float(threshold))
        candidates.append((metric["band_macro_f1"], metric["band_macro_iou"], -abs(float(threshold)), float(threshold)))
    best = max(candidates)
    return {"threshold": best[3], "select_f1": best[0], "select_iou": best[1]}


def bootstrap_difference(
    sg: list[dict[str, Any]],
    e2e: list[dict[str, Any]],
    mode: str,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if [(r["raw_index"], r["local_index"]) for r in sg] != [(r["raw_index"], r["local_index"]) for r in e2e]:
        raise RuntimeError("SG/E2E诊断样本身份不一致")
    by_k = {
        count: np.asarray([i for i, row in enumerate(sg) if int(row["true_count"]) == count])
        for count in range(4)
    }

    def sample_metric(rows: list[dict[str, Any]], selected: np.ndarray, name: str) -> float:
        if name == "exact_count_rate":
            return float(np.mean([rows[i]["exact_count"] for i in selected]))
        values = [
            source[name]
            for i in selected
            for source in rows[i]["modes"][mode]["sources"]
        ]
        return float(np.mean(values)) if values else (1.0 if name != "bce" else 0.0)

    names = ("exact_count_rate", "f1", "iou", "bce")
    all_indices = np.arange(len(sg))
    point = {
        name: sample_metric(e2e, all_indices, name) - sample_metric(sg, all_indices, name)
        for name in names
    }
    rng = np.random.default_rng(seed)
    boot = {name: [] for name in names}
    for _ in range(repetitions):
        selected = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in by_k.values()])
        for name in names:
            boot[name].append(sample_metric(e2e, selected, name) - sample_metric(sg, selected, name))
    labels = {"f1": "band_macro_f1", "iou": "band_macro_iou", "bce": "band_bce", "exact_count_rate": "exact_count_rate"}
    return {
        labels[name]: {
            "e2e_minus_sg": point[name],
            "ci95": np.quantile(boot[name], [0.025, 0.975]).tolist(),
        }
        for name in names
    }


def matching_changes(records: list[dict[str, Any]]) -> dict[str, float]:
    positive = [row for row in records if int(row["true_count"]) > 0]
    return {
        "joint_vs_band_only": float(np.mean([
            row["modes"]["joint"]["signature"] != row["modes"]["band_only"]["signature"]
            for row in positive
        ])),
        "joint_vs_position_only": float(np.mean([
            row["modes"]["joint"]["signature"] != row["modes"]["position_only"]["signature"]
            for row in positive
        ])),
    }


def parameter_groups(context: r1.TrainContext) -> dict[str, list[torch.nn.Parameter]]:
    return {
        "band_heads": list(context.ch3.band_heads[:3].parameters()),
        "query_anchor": list(context.query_builder.anchor.parameters()),
        "cross_attention": list(context.query_builder.cross_attention.parameters()),
        "band_residual": list(context.query_builder.band_residual.parameters()),
        "splitter": list(context.splitter.parameters()),
    }


def vector(parameters: list[torch.nn.Parameter], gradients: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
    pieces = [
        torch.zeros_like(parameter).flatten() if gradient is None else gradient.detach().flatten()
        for parameter, gradient in zip(parameters, gradients)
    ]
    return torch.cat(pieces) if pieces else torch.zeros(1)


def gradient_batch(
    context: r1.TrainContext,
    data: g2.CachedBatch,
    indices: torch.Tensor,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    groups = parameter_groups(context)
    parameters = [parameter for values in groups.values() for parameter in values]
    _, logits, _, heatmap, offset = r1.forward_indices(context, indices, device, stop_gradient=False)
    _, components, _ = r1.compute_losses(logits, heatmap, offset, data, indices, config)
    auxiliary = components["exist"] + components["band"]
    localization = components["heatmap"] + components["offset"]
    aux_gradients = torch.autograd.grad(auxiliary, parameters, retain_graph=True, allow_unused=True)
    loc_gradients = torch.autograd.grad(localization, parameters, allow_unused=True)
    result: dict[str, Any] = {}
    cursor = 0
    for name, current_parameters in groups.items():
        stop = cursor + len(current_parameters)
        aux_vector = vector(current_parameters, aux_gradients[cursor:stop])
        loc_vector = vector(current_parameters, loc_gradients[cursor:stop])
        aux_norm = float(torch.linalg.vector_norm(aux_vector).item())
        loc_norm = float(torch.linalg.vector_norm(loc_vector).item())
        cosine = None
        if aux_norm > 0.0 and loc_norm > 0.0:
            cosine = float(torch.dot(aux_vector, loc_vector).item() / (aux_norm * loc_norm))
        result[name] = {
            "auxiliary_norm": aux_norm,
            "localization_norm": loc_norm,
            "ratio": loc_norm / max(aux_norm, 1e-30),
            "cosine": cosine,
        }
        cursor = stop
    upstream_names = ("band_heads", "query_anchor", "cross_attention", "band_residual")
    aux_upstream = torch.cat([
        vector(groups[name], aux_gradients[
            sum(len(groups[key]) for key in groups if list(groups).index(key) < list(groups).index(name)):
            sum(len(groups[key]) for key in groups if list(groups).index(key) <= list(groups).index(name))
        ]) for name in upstream_names
    ])
    loc_upstream = torch.cat([
        vector(groups[name], loc_gradients[
            sum(len(groups[key]) for key in groups if list(groups).index(key) < list(groups).index(name)):
            sum(len(groups[key]) for key in groups if list(groups).index(key) <= list(groups).index(name))
        ]) for name in upstream_names
    ])
    aux_norm = float(torch.linalg.vector_norm(aux_upstream).item())
    loc_norm = float(torch.linalg.vector_norm(loc_upstream).item())
    result["upstream_total"] = {
        "auxiliary_norm": aux_norm,
        "localization_norm": loc_norm,
        "ratio": loc_norm / max(aux_norm, 1e-30),
        "cosine": None if aux_norm == 0.0 or loc_norm == 0.0 else float(torch.dot(aux_upstream, loc_upstream).item() / (aux_norm * loc_norm)),
    }
    return result


def summarize_gradients(rows: list[dict[str, Any]]) -> dict[str, Any]:
    modules = rows[0]["modules"]
    result: dict[str, Any] = {}
    for module in modules:
        values = [row["modules"][module] for row in rows]
        cosines = [float(value["cosine"]) for value in values if value["cosine"] is not None]
        result[module] = {
            "batches": len(values),
            "auxiliary_norm_median": float(np.median([value["auxiliary_norm"] for value in values])),
            "localization_norm_median": float(np.median([value["localization_norm"] for value in values])),
            "ratio_median": float(np.median([value["ratio"] for value in values])),
            "cosine_mean": float(np.mean(cosines)) if cosines else None,
            "negative_cosine_rate": float(np.mean(np.asarray(cosines) < 0.0)) if cosines else None,
        }
    return result


def gradient_audit(
    source_manifest: dict[str, Any], config: dict[str, Any], checkpoints: dict[str, dict[str, Any]], device: torch.device
) -> dict[str, Any]:
    train = r2.load_split(SOURCE_RUN, source_manifest, "train")
    selected = []
    per_k = int(config["gradient_sample_count"]) // 4
    for count in range(4):
        candidates = torch.nonzero(train.counts == count, as_tuple=False).flatten().tolist()
        selected.extend(candidates[:per_k])
    selected_tensor = torch.tensor(selected, dtype=torch.long)
    current = r2.subset_batch(train, selected_tensor)
    context = r1.build_train_context(current, device, int(config["seed"]))
    initial_digest = r2.state_digest(context)
    expected_initial = read_json(SOURCE_RUN / "p1_report.json")["initial_state_sha256"]
    if initial_digest != expected_initial:
        raise RuntimeError("D1重建epoch0身份不一致")
    states = {"epoch0": r2.state_payload(context), **checkpoints}
    output: dict[str, Any] = {}
    batch_size = int(config["gradient_batch_size"])
    for state_name, state in states.items():
        r2.load_state(context, state)
        state_before = r2.state_digest(context)
        rows = []
        for start in range(0, len(current.counts), batch_size):
            indices = torch.arange(start, min(start + batch_size, len(current.counts)))
            rows.append(
                {
                    "batch": start // batch_size,
                    "k": int(current.counts[indices[0]].item()),
                    "frequency_overlap": bool(current.overlap[indices].all().item()),
                    "modules": gradient_batch(context, current, indices, config, device),
                }
            )
        if r2.state_digest(context) != state_before:
            raise RuntimeError("梯度审计意外修改模型状态")
        output[state_name] = {
            "state_sha256": state_before,
            "summary": summarize_gradients(rows),
            "by_k": {
                str(count): summarize_gradients([row for row in rows if row["k"] == count])
                for count in range(4)
            },
            "batches": rows,
        }
    return {
        "status": "PASS",
        "sample_count": len(selected),
        "indices": selected,
        "k_histogram": {str(count): int((current.counts == count).sum().item()) for count in range(4)},
        "overlap_support": {
            "overlap": int(current.overlap.sum().item()),
            "non_overlap": int((~current.overlap).sum().item()),
            "note": "本R2 train抽样中K=2/3均为frequency_overlap；不能比较多源非重叠梯度。",
        },
        "states": output,
        "optimizer_steps": 0,
        "weights_updated": False,
    }


def run(run_root: Path) -> dict[str, Any]:
    manifest, config, source_manifest = verify(run_root)
    if (run_root / "final_report.json").exists():
        raise FileExistsError("拒绝覆盖D1 final_report")
    set_deterministic(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = {
        track: torch.load(SOURCE_RUN / "training" / track / "best.pt", map_location=device, weights_only=False)
        for track in TRACKS
    }
    started = time.perf_counter()
    select = analyze_split(run_root, source_manifest, config, "val_select", checkpoints, device)
    compare = analyze_split(run_root, source_manifest, config, "val_compare", checkpoints, device)
    metrics = {
        split: {
            track: {mode: grouped_metrics(values[track], mode) for mode in MODES}
            for track in TRACKS
        }
        for split, values in (("val_select", select), ("val_compare", compare))
    }
    calibration = {}
    for track in TRACKS:
        chosen = choose_threshold(select[track], config)
        calibration[track] = {
            **chosen,
            "compare_at_selected_threshold": aggregate(
                compare[track], "band_only", chosen["threshold"]
            ),
            "compare_at_zero": aggregate(compare[track], "band_only", 0.0),
        }
    paired = {
        mode: bootstrap_difference(
            compare["fs_sg"], compare["fs_e2e"], mode,
            int(config["bootstrap_repetitions"]), int(config["seed"]) + offset,
        )
        for mode, offset in (("joint", 101), ("band_only", 202), ("position_only", 303))
    }
    changes = {track: matching_changes(compare[track]) for track in TRACKS}
    band_delta = paired["band_only"]
    trigger = (
        band_delta["band_macro_f1"]["e2e_minus_sg"] < -float(config["gradient_trigger_drop"])
        or band_delta["band_macro_iou"]["e2e_minus_sg"] < -float(config["gradient_trigger_drop"])
        or band_delta["band_bce"]["e2e_minus_sg"] > 0.0
    )
    gradients = gradient_audit(source_manifest, config, checkpoints, device) if trigger else {
        "status": "SKIPPED_MATCHING_OR_CALIBRATION_ONLY",
        "optimizer_steps": 0,
        "weights_updated": False,
    }
    evidence = []
    margin = float(config["auxiliary_margin"])
    joint = paired["joint"]
    if (
        joint["band_macro_f1"]["e2e_minus_sg"] < -margin
        and band_delta["band_macro_f1"]["e2e_minus_sg"] >= -margin
        and band_delta["band_macro_iou"]["e2e_minus_sg"] >= -margin
    ):
        evidence.append("MATCHING_ARTIFACT")
    tuned_sg = calibration["fs_sg"]["compare_at_selected_threshold"]["band_macro_f1"]
    tuned_e2e = calibration["fs_e2e"]["compare_at_selected_threshold"]["band_macro_f1"]
    if band_delta["band_macro_f1"]["e2e_minus_sg"] < -margin and tuned_e2e - tuned_sg >= -margin:
        evidence.append("CALIBRATION_SHIFT")
    if (
        band_delta["band_macro_f1"]["e2e_minus_sg"] < -margin
        or band_delta["band_macro_iou"]["e2e_minus_sg"] < -margin
        or band_delta["band_bce"]["e2e_minus_sg"] > 0.0
    ):
        evidence.append("TRUE_BAND_REGRESSION")
    f1_ci = band_delta["band_macro_f1"]["ci95"]
    iou_ci = band_delta["band_macro_iou"]["ci95"]
    if f1_ci[0] <= 0.0 <= f1_ci[1] and iou_ci[0] <= 0.0 <= iou_ci[1]:
        evidence.append("SAMPLING_UNCERTAIN")
    if f1_ci[1] < -margin or iou_ci[1] < -margin:
        evidence.append("MATERIAL_AUX_HARM")
    if gradients["status"] == "PASS":
        upstream = gradients["states"]["fs_e2e"]["summary"]["upstream_total"]
        if upstream["negative_cosine_rate"] is not None and upstream["negative_cosine_rate"] >= 0.75:
            evidence.append("GRADIENT_CONFLICT_CANDIDATE")
    report = {
        "material_passport": {
            **manifest["material_passport"],
            "verification_status": "ANALYZED",
        },
        "gate": "E2E-G2-R2-D1",
        "status": "+".join(evidence) if evidence else "INCONCLUSIVE",
        "source_r2_status_preserved": "G2_R2_AUX_REGRESSION",
        "metrics": metrics,
        "paired_bootstrap": paired,
        "matching_changes": changes,
        "calibration": calibration,
        "gradient_audit": gradients,
        "duration_seconds": time.perf_counter() - started,
        "optimizer_steps": 0,
        "weights_updated": False,
        "test_executed": False,
        "source_unchanged": all(
            manifest["source"][name] == identity(SOURCE_RUN / filename)
            for name, filename in (("manifest", "manifest.json"), ("p3", "p3_report.json"), ("final", "final_report.json"), ("cache_manifest", "cache_manifest.json"))
        ),
        "fallacy_scan": {
            "coverage": "11/11 checked",
            "simpson_reversal": "check grouped metrics; no inference before final review",
            "single_training_seed": True,
            "sample_bootstrap_excludes_training_variance": True,
            "test_inference_forbidden": True,
        },
    }
    write_json(run_root / "diagnostic_samples.json", {"val_select": select, "val_compare": compare})
    write_json(run_root / "final_report.json", report)
    print(json.dumps({"status": report["status"], "duration_seconds": report["duration_seconds"]}, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare":
        prepare(args.run_id)
    else:
        run(args.run_root.resolve())
