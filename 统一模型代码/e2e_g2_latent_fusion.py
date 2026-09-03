"""E2E-G2 P2：32 样本逐源潜在融合强过拟合与梯度验收。"""

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

from gen_ch4_loc_data import group_sources_by_freq_overlap  # noqa: E402
from yolo_model import focal_loss_hm_hnm  # noqa: E402

from 统一模型代码 import e2e_g1 as g1  # noqa: E402
from 统一模型代码.audits import e2e_g2_preflight as preflight  # noqa: E402
from 统一模型代码.models.e2e_latent_fusion import (  # noqa: E402
    CH3Features,
    FrequencySpatialSplitter,
    SourceLocalizationHead,
    SourceQueryBuilder,
    forward_ch3_features,
    forward_d8_features,
)
from 统一模型代码.runtime_paths import validate_output_path  # noqa: E402


CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g2_latent_fusion.json"
SCRIPT_PATH = Path(__file__).resolve()
MODEL_PATH = PACKAGE_ROOT / "models" / "e2e_latent_fusion.py"
PREFLIGHT_RUN = (
    PROJECT_ROOT
    / "outputs_e2e"
    / "unified"
    / "e2e_g2_latent_fusion"
    / "20260903_153441"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def ch3_from_cached(model: nn.Module, cached: CH3Features) -> CH3Features:
    hidden = torch.stack(
        [torch.relu(head[0](cached.global_feature)) for head in model.band_heads], dim=1
    )
    logits = torch.stack(
        [head[2](hidden[:, index]) for index, head in enumerate(model.band_heads)], dim=1
    )
    return CH3Features(
        cached.spatial, cached.tokens, cached.global_feature, hidden, logits
    )


def source_is_overlap(raw: Any, raw_index: int, count: int) -> bool:
    handle, local = raw._raw(raw_index)
    fc = np.asarray(handle["fc_offset_all"][:, local], dtype=np.float64).reshape(-1)
    bw = np.asarray(handle["BW_actual_all"][:, local], dtype=np.float64).reshape(-1)
    groups = group_sources_by_freq_overlap(fc, bw, count)
    return any(int(group["n_src"]) >= 2 for group in groups)


def choose_p2_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = read_json(preflight.SOURCE_MANIFEST)["subsets"]["train"]
    per_k = int(config["p2_samples"]) // 4
    selected: list[dict[str, Any]] = []
    with g1.SampleStore("train") as store:
        for count in range(4):
            rows = [row for row in source if int(row["true_k"]) == count]
            if count >= 2:
                overlap = [
                    row for row in rows
                    if source_is_overlap(store, int(row["raw_index"]), count)
                ]
                separated = [row for row in rows if row not in overlap]
                take_overlap = min(max(per_k // 2, 1), len(overlap))
                chosen = overlap[:take_overlap] + separated[: per_k - take_overlap]
                if len(chosen) < per_k:
                    chosen.extend(overlap[take_overlap : take_overlap + per_k - len(chosen)])
            else:
                chosen = rows[:per_k]
            if len(chosen) != per_k:
                raise RuntimeError(f"P2 K={count}样本不足")
            for row in chosen:
                selected.append({**row, "frequency_overlap": bool(count >= 2 and source_is_overlap(store, int(row["raw_index"]), count))})
    return selected


def prepare_p2(run_root: Path) -> dict[str, Any]:
    run_root = validate_output_path(run_root.resolve())
    if read_json(PREFLIGHT_RUN / "p0_report.json")["status"] != "PASS":
        raise RuntimeError("正式P0未通过")
    if read_json(PREFLIGHT_RUN / "p1_report.json")["status"] != "PASS":
        raise RuntimeError("正式P1未通过")
    preflight.configure_snapshot()
    config = read_json(CONFIG_PATH)
    path = run_root / "p2_manifest.json"
    if path.exists():
        raise FileExistsError(f"拒绝覆盖P2清单: {path}")
    payload = {
        "status": "PREPARED",
        "gate": "E2E-G2-P2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "test_executed": False,
        "config": config,
        "code": [identity(SCRIPT_PATH), identity(MODEL_PATH), identity(CONFIG_PATH)],
        "preflight": {
            "p0": identity(PREFLIGHT_RUN / "p0_report.json"),
            "p1": identity(PREFLIGHT_RUN / "p1_report.json"),
        },
        "records": choose_p2_records(config),
    }
    write_json(path, payload)
    print(json.dumps({"status": "PREPARED", "records": len(payload["records"]), "overlap": sum(int(row["frequency_overlap"]) for row in payload["records"])}, ensure_ascii=False), flush=True)
    return payload


def verify_p2(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_root / "p2_manifest.json")
    current = [identity(SCRIPT_PATH), identity(MODEL_PATH), identity(CONFIG_PATH)]
    if current != manifest["code"]:
        raise RuntimeError("P2 prepare后代码或配置变化")
    for key in ("p0", "p1"):
        if identity(PREFLIGHT_RUN / f"{key}_report.json") != manifest["preflight"][key]:
            raise RuntimeError(f"{key}证据身份变化")
    preflight.configure_snapshot()
    return manifest, manifest["config"]


@dataclass
class CachedBatch:
    coarse: torch.Tensor
    dpd: torch.Tensor
    band: torch.Tensor
    ignore: torch.Tensor
    positions: torch.Tensor
    counts: torch.Tensor
    overlap: torch.Tensor


def prepare_cache(run_root: Path, manifest: dict[str, Any], config: dict[str, Any], device: torch.device) -> list[dict[str, Any]]:
    cache_root = validate_output_path(run_root / "p2_cache")
    if cache_root.exists():
        raise FileExistsError(f"拒绝复用P2缓存: {cache_root}")
    cache_root.mkdir()
    geometry = g1.receiver_geometry(device)
    weights = torch.ones(g1.N_FFT, dtype=torch.float64, device=device)
    descriptions = []
    with g1.SampleStore("train") as store:
        for ordinal, record in enumerate(manifest["records"], start=1):
            sample = store.sample(record)
            started = time.perf_counter()
            dpd = preflight.dpd_map(sample["signal"], weights, geometry, config)
            array = dpd.detach().cpu().numpy().astype(np.float32)
            path = cache_root / f"full_{ordinal:03d}.npy"
            np.save(path, array)
            description = preflight.cache_description(path, array, 0.0)
            description.update({"ordinal": ordinal, "raw_index": int(record["raw_index"]), "seconds": time.perf_counter() - started})
            if not description["finite"] or not description["nonconstant"]:
                raise RuntimeError(f"P2缓存异常: {path}")
            descriptions.append(description)
            print(f"[P2 cache] {ordinal}/{len(manifest['records'])} {description['seconds']:.1f}s", flush=True)
    write_json(run_root / "p2_cache_manifest.json", {"files": descriptions})
    return descriptions


def load_samples(run_root: Path, manifest: dict[str, Any]) -> CachedBatch:
    coarse, dpd, band, ignore, positions, counts, overlap = [], [], [], [], [], [], []
    with g1.SampleStore("train") as store:
        for ordinal, record in enumerate(manifest["records"], start=1):
            sample = store.sample(record)
            coarse.append(sample["coarse_dpd"])
            dpd.append(torch.from_numpy(np.load(run_root / "p2_cache" / f"full_{ordinal:03d}.npy", allow_pickle=False)))
            band.append(sample["band_truth"][:3])
            ignore.append(sample["ignore_truth"][:3])
            position = np.zeros((3, 2), dtype=np.float32)
            position[: int(sample["true_k"])] = sample["positions_m"][: int(sample["true_k"])]
            positions.append(torch.from_numpy(position))
            counts.append(int(sample["true_k"]))
            overlap.append(bool(record["frequency_overlap"]))
    return CachedBatch(
        torch.stack(coarse), torch.stack(dpd), torch.stack(band), torch.stack(ignore),
        torch.stack(positions), torch.tensor(counts), torch.tensor(overlap),
    )


def single_source_target(position: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, int, int, torch.Tensor]:
    px = (position[0].to(device) + g1.FINE_EDGE) / g1.FINE_STEP
    py = (position[1].to(device) + g1.FINE_EDGE) / g1.FINE_STEP
    ix = int(torch.round(px).clamp(0, 400).item())
    iy = int(torch.round(py).clamp(0, 400).item())
    axis = torch.arange(401, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * g1.GAUSS_SIGMA**2))
    target = target / target.max().clamp_min(1e-12)
    delta = torch.stack([px - ix, py - iy]).to(torch.float32)
    return target, ix, iy, delta


def assignments(
    band_logits: torch.Tensor,
    heatmap: torch.Tensor,
    band_truth: torch.Tensor,
    ignore_truth: torch.Tensor,
    positions: torch.Tensor,
    counts: torch.Tensor,
) -> list[dict[int, int]]:
    output: list[dict[int, int]] = []
    for batch_index, count_value in enumerate(counts.tolist()):
        count = int(count_value)
        if count == 0:
            output.append({})
            continue
        cost = torch.zeros((3, count), device=band_logits.device)
        for query in range(3):
            for source in range(count):
                valid = ignore_truth[batch_index, source] < 0.5
                band_cost = F.binary_cross_entropy_with_logits(
                    band_logits[batch_index, query, valid],
                    band_truth[batch_index, source, valid],
                )
                px = int(torch.round((positions[batch_index, source, 0] + g1.FINE_EDGE) / g1.FINE_STEP).clamp(0, 400).item())
                py = int(torch.round((positions[batch_index, source, 1] + g1.FINE_EDGE) / g1.FINE_STEP).clamp(0, 400).item())
                location_cost = 1.0 - torch.sigmoid(heatmap[batch_index, query, py, px])
                cost[query, source] = band_cost + location_cost
        best: tuple[float, tuple[int, ...]] | None = None
        for query_order in itertools.permutations(range(3), count):
            value = sum(float(cost[query_order[source], source].detach().item()) for source in range(count))
            if best is None or value < best[0]:
                best = (value, query_order)
        assert best is not None
        output.append({query: source for source, query in enumerate(best[1])})
    return output


def compute_losses(
    band_logits: torch.Tensor,
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    batch: CachedBatch,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[int, int]]]:
    device = band_logits.device
    truth = batch.band[indices].to(device)
    ignore = batch.ignore[indices].to(device)
    positions = batch.positions[indices].to(device)
    counts = batch.counts[indices].to(device)
    matched = assignments(band_logits, heatmap, truth, ignore, positions, counts)
    exist_target = torch.zeros_like(band_logits[..., 0])
    band_target = torch.zeros_like(band_logits)
    band_valid = torch.ones_like(band_logits)
    heat_target = torch.zeros_like(heatmap)
    offset_losses = []
    for batch_index, mapping in enumerate(matched):
        for query, source in mapping.items():
            exist_target[batch_index, query] = 1.0
            band_target[batch_index, query] = truth[batch_index, source]
            band_valid[batch_index, query] = 1.0 - ignore[batch_index, source]
            target, ix, iy, delta = single_source_target(positions[batch_index, source], device)
            heat_target[batch_index, query] = target
            offset_losses.append(torch.abs(offset[batch_index, query, :, iy, ix] - delta).sum())
    slot_logits = band_logits.amax(dim=-1)
    loss_exist = F.binary_cross_entropy_with_logits(slot_logits, exist_target)
    band_element = F.binary_cross_entropy_with_logits(band_logits, band_target, reduction="none")
    loss_band = (band_element * band_valid).sum() / band_valid.sum().clamp_min(1.0)
    loss_heatmap = focal_loss_hm_hnm(heatmap.reshape(-1, 1, 401, 401), heat_target.reshape(-1, 1, 401, 401))
    loss_offset = torch.stack(offset_losses).mean() if offset_losses else offset.sum() * 0.0
    components = {"exist": loss_exist, "band": loss_band, "heatmap": loss_heatmap, "offset": loss_offset}
    return sum(components.values()), components, matched


def decode_metrics(
    band_logits: torch.Tensor,
    heatmap: torch.Tensor,
    offset: torch.Tensor,
    attention: torch.Tensor,
    batch: CachedBatch,
) -> dict[str, Any]:
    predicted_counts = (band_logits.amax(dim=-1) >= 0.0).sum(dim=-1)
    exact = int((predicted_counts.cpu() == batch.counts).sum().item())
    f1_values, recalls, collapsed, overlap_attention_distances = [], [], [], []
    for index, count_value in enumerate(batch.counts.tolist()):
        count = int(count_value)
        active_queries = torch.nonzero(band_logits[index].amax(dim=-1) >= 0.0, as_tuple=False).flatten().tolist()
        predictions = []
        for query in active_queries:
            flat = torch.sigmoid(heatmap[index, query]).flatten()
            peak = int(torch.argmax(flat).item())
            iy, ix = divmod(peak, 401)
            delta = offset[index, query, :, iy, ix].clamp(-1.0, 1.0)
            predictions.append([(ix + float(delta[0])) * 10.0 - 2000.0, (iy + float(delta[1])) * 10.0 - 2000.0])
        predicted = np.asarray(predictions, dtype=np.float32).reshape(-1, 2)
        truth_positions = batch.positions[index, :count].numpy()
        recalls.append(g1.maximum_matches_within(truth_positions, predicted, 10.0))
        if count >= 2 and len(predicted) >= 2:
            distances = [np.linalg.norm(predicted[a] - predicted[b]) for a in range(len(predicted)) for b in range(a + 1, len(predicted))]
            collapsed.append(float(min(distances)) < 10.0)
        mapping = assignments(
            band_logits[index:index+1], heatmap[index:index+1],
            batch.band[index:index+1].to(band_logits.device), batch.ignore[index:index+1].to(band_logits.device),
            batch.positions[index:index+1].to(band_logits.device), batch.counts[index:index+1].to(band_logits.device),
        )[0]
        for query, source in mapping.items():
            valid = batch.ignore[index, source] < 0.5
            prediction = band_logits[index, query, valid] >= 0.0
            target = batch.band[index, source, valid].to(prediction.device) > 0.5
            tp = int((prediction & target).sum().item()); fp = int((prediction & ~target).sum().item()); fn = int((~prediction & target).sum().item())
            f1_values.append(2 * tp / max(2 * tp + fp + fn, 1))
        if bool(batch.overlap[index]) and count >= 2:
            active = list(mapping)
            for a in range(len(active)):
                for b in range(a + 1, len(active)):
                    overlap_attention_distances.append(float(torch.mean(torch.abs(attention[index, active[a]] - attention[index, active[b]])).item()))
    true_total = int(batch.counts.sum().item())
    return {
        "exact_count": exact,
        "exact_count_rate": exact / len(batch.counts),
        "active_band_macro_f1": float(np.mean(f1_values)) if f1_values else 1.0,
        "recall_at_10m": sum(recalls) / max(true_total, 1),
        "collapsed_multisource_samples": int(sum(collapsed)),
        "overlap_attention_mean_l1": float(np.mean(overlap_attention_distances)) if overlap_attention_distances else 0.0,
    }


def feature_batch(cached: CH3Features, indices: torch.Tensor, device: torch.device) -> CH3Features:
    return CH3Features(
        cached.spatial[indices].to(device), cached.tokens[indices].to(device),
        cached.global_feature[indices].to(device),
        torch.empty(0, device=device), torch.empty(0, device=device),
    )


def run_p2(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_p2(run_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_deterministic(int(config["seed"]))
    cache_path = run_root / "p2_cache_manifest.json"
    if not cache_path.exists():
        prepare_cache(run_root, manifest, config, device)
    data = load_samples(run_root, manifest)
    ch3, d8, checkpoint = g1.build_models(device)
    ch3.eval(); d8.eval()
    for parameter in ch3.parameters(): parameter.requires_grad_(False)
    for parameter in d8.parameters(): parameter.requires_grad_(False)
    for head in ch3.band_heads[:3]:
        for parameter in head.parameters(): parameter.requires_grad_(True)
    with torch.no_grad():
        all_ch3 = []
        all_d0 = []
        for start in range(0, len(data.counts), int(config["batch_size"])):
            stop = min(start + int(config["batch_size"]), len(data.counts))
            features = forward_ch3_features(ch3, data.coarse[start:stop].to(device))
            all_ch3.append(CH3Features(features.spatial.cpu(), features.tokens.cpu(), features.global_feature.cpu(), torch.empty(0), torch.empty(0)))
            d0 = forward_d8_features(d8, torch.stack([g1.d8_input(item)[0] for item in data.dpd[start:stop]]).to(device)).d0
            all_d0.append(d0.cpu())
    cached_ch3 = CH3Features(
        torch.cat([item.spatial for item in all_ch3]), torch.cat([item.tokens for item in all_ch3]),
        torch.cat([item.global_feature for item in all_ch3]), torch.empty(0), torch.empty(0),
    )
    cached_d0 = torch.cat(all_d0)
    query_builder = SourceQueryBuilder().to(device)
    splitter = FrequencySpatialSplitter().to(device)
    source_head = SourceLocalizationHead().to(device)
    nn.init.constant_(source_head.heatmap.bias, -2.19)
    nn.init.zeros_(source_head.offset.weight); nn.init.zeros_(source_head.offset.bias)
    parameters = [p for p in itertools.chain(ch3.band_heads[:3].parameters(), query_builder.parameters(), splitter.parameters(), source_head.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=float(config["p2_learning_rate"]), weight_decay=float(config["p2_weight_decay"]))
    order_generator = torch.Generator().manual_seed(int(config["seed"]))
    history = []
    best = None
    started = time.perf_counter()
    batch_size = int(config["batch_size"])
    for epoch in range(1, int(config["p2_max_epochs"]) + 1):
        order = torch.randperm(len(data.counts), generator=order_generator)
        sums = {name: 0.0 for name in ("exist", "band", "heatmap", "offset")}
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            current = ch3_from_cached(ch3, feature_batch(cached_ch3, indices, device))
            query, logits = query_builder(current)
            spatial, attention = splitter(current.spatial, query, logits)
            heatmap, offset = source_head(cached_d0[indices].to(device), spatial, query)
            total, components, _ = compute_losses(logits, heatmap, offset, data, indices)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(parameters, float(config["p2_gradient_clip"]))
            optimizer.step()
            for name, value in components.items(): sums[name] += float(value.detach().item()) * len(indices)
        if epoch % int(config["p2_evaluate_every"]) == 0 or epoch == 1:
            with torch.no_grad():
                outputs = []
                for start in range(0, len(data.counts), batch_size):
                    indices = torch.arange(start, min(start + batch_size, len(data.counts)))
                    current = ch3_from_cached(ch3, feature_batch(cached_ch3, indices, device))
                    query, logits = query_builder(current)
                    spatial, attention = splitter(current.spatial, query, logits)
                    heatmap, offset = source_head(cached_d0[indices].to(device), spatial, query)
                    outputs.append((logits.cpu(), heatmap.cpu(), offset.cpu(), attention.cpu()))
                metrics = decode_metrics(*(torch.cat([item[i] for item in outputs]) for i in range(4)), data)
            row = {"epoch": epoch, "losses": {name: value / len(data.counts) for name, value in sums.items()}, "metrics": metrics, "elapsed_seconds": time.perf_counter() - started}
            history.append(row); write_json(run_root / "p2_progress.json", history)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if metrics["exact_count"] >= 31 and metrics["active_band_macro_f1"] >= 0.98 and metrics["recall_at_10m"] >= 0.95 and metrics["collapsed_multisource_samples"] == 0 and metrics["overlap_attention_mean_l1"] > 1e-6:
                best = row
                break
        if time.perf_counter() - started > float(config["p2_wall_limit_seconds"]):
            break

    # 独立审计定位损失在 E2E 与 SG 接口两侧的梯度。
    indices = torch.arange(0, batch_size)
    current = ch3_from_cached(ch3, feature_batch(cached_ch3, indices, device))
    query, logits = query_builder(current); query.retain_grad(); logits.retain_grad()
    spatial, attention = splitter(current.spatial, query, logits); attention.retain_grad()
    heatmap, offset = source_head(cached_d0[indices].to(device), spatial, query)
    _, components, _ = compute_losses(logits, heatmap, offset, data, indices)
    location = components["heatmap"] + components["offset"]
    optimizer.zero_grad(set_to_none=True); location.backward()
    e2e_gradient = {"band_logits": float(logits.grad.norm().item()), "query": float(query.grad.norm().item()), "spatial_attention": float(attention.grad.norm().item())}
    current = ch3_from_cached(ch3, feature_batch(cached_ch3, indices, device))
    query, logits = query_builder(current); query.retain_grad(); logits.retain_grad()
    spatial, attention = splitter(current.spatial, query.detach(), logits.detach()); attention.retain_grad()
    heatmap, offset = source_head(cached_d0[indices].to(device), spatial, query.detach())
    _, components, _ = compute_losses(logits, heatmap, offset, data, indices)
    location = components["heatmap"] + components["offset"]
    optimizer.zero_grad(set_to_none=True); location.backward()
    sg_gradient = {"band_logits": 0.0 if logits.grad is None else float(logits.grad.norm().item()), "query": 0.0 if query.grad is None else float(query.grad.norm().item()), "spatial_attention": float(attention.grad.norm().item())}
    gradient_pass = all(math.isfinite(value) and value > 0 for value in e2e_gradient.values()) and sg_gradient["band_logits"] == 0.0 and sg_gradient["query"] == 0.0
    pass_metrics = best is not None
    duration = time.perf_counter() - started
    status = "PASS" if pass_metrics and gradient_pass and duration <= float(config["p2_wall_limit_seconds"]) else ("G2_STOP_RESOURCE" if duration > float(config["p2_wall_limit_seconds"]) else "G2_NO_GO_REPRESENTATION")
    report = {"status": status, "selected_epoch": best["epoch"] if best else None, "best": best or history[-1], "gradient_audit": {"e2e": e2e_gradient, "stop_gradient": sg_gradient, "pass": gradient_pass}, "duration_seconds": duration, "checkpoint": checkpoint, "test_executed": False}
    torch.save({"ch3_heads": ch3.band_heads[:3].state_dict(), "query_builder": query_builder.state_dict(), "splitter": splitter.state_dict(), "source_head": source_head.state_dict(), "report": report}, validate_output_path(run_root / "p2_checkpoint.pt"))
    write_json(run_root / "p2_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare-p2", "run-p2"])
    parser.add_argument("--run-root", type=Path, default=PREFLIGHT_RUN)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare-p2": prepare_p2(args.run_root)
    else: run_p2(args.run_root.resolve())
