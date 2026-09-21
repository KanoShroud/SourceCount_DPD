"""E2E-G2-R2：固定前向下FS-SG与FS-E2E定位反馈配对因果门。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

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

# 第三、四章均有同名 chapter_runtime；先固定第四章模块身份，避免后续
# 导入 CH3 时把 D8 依赖错误解析到第三章实现。
import chapter_runtime  # noqa: E402,F401

from 统一模型代码.gates.g1 import e2e_g1 as g1  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_latent_fusion as g2  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_r1 as r1  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_preflight as preflight  # noqa: E402
from 统一模型代码.common.runtime_paths import new_run_dir, validate_output_path  # noqa: E402


CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g2_r2.json"
SCRIPT_PATH = Path(__file__).resolve()
MODEL_PATH = PACKAGE_ROOT / "models" / "e2e_latent_fusion.py"
SOURCE_MANIFEST = preflight.SOURCE_MANIFEST
SOURCE_R1_RUN = (
    PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_g2_r1" / "20260903_183506"
)
TRACKS = {"fs_sg": True, "fs_e2e": False}


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


def code_identity() -> list[dict[str, Any]]:
    paths = (
        SCRIPT_PATH,
        CONFIG_PATH,
        MODEL_PATH,
        Path(r1.__file__).resolve(),
        Path(g2.__file__).resolve(),
        Path(preflight.__file__).resolve(),
        Path(g1.__file__).resolve(),
    )
    return [identity(path) for path in paths]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def annotate_records(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with g1.SampleStore(split) as store:
        for row in records:
            count = int(row["true_k"])
            result.append(
                {
                    **row,
                    "frequency_overlap": bool(
                        count >= 2
                        and g2.source_is_overlap(store, int(row["raw_index"]), count)
                    ),
                }
            )
    return result


def prepare(run_id: str) -> Path:
    files, artifacts = preflight.configure_snapshot()
    source = read_json(SOURCE_MANIFEST)
    expected = {"train": 256, "val_select": 128, "val_compare": 512}
    subsets: dict[str, list[dict[str, Any]]] = {}
    for split, count in expected.items():
        records = source["subsets"][split]
        if len(records) != count:
            raise RuntimeError(f"{split}规模错误: {len(records)} != {count}")
        histogram = {
            value: sum(int(row["true_k"]) == value for row in records)
            for value in range(4)
        }
        if any(value != count // 4 for value in histogram.values()):
            raise RuntimeError(f"{split}的K不平衡: {histogram}")
        subsets[split] = annotate_records(records, split)
    validation_raw_indices = {
        split: {int(row["raw_index"]) for row in subsets[split]}
        for split in ("val_select", "val_compare")
    }
    if validation_raw_indices["val_select"] & validation_raw_indices["val_compare"]:
        raise RuntimeError("val_select与val_compare重叠")
    run_root = new_run_dir("e2e_g2_r2", run_id, create=True)
    manifest = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "UNVERIFIED",
        },
        "status": "PREPARED",
        "gate": "E2E-G2-R2",
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_read_only": True,
        "test_executed": False,
        "config": read_json(CONFIG_PATH),
        "code": code_identity(),
        "source_manifest": identity(SOURCE_MANIFEST),
        "source_r1_final": identity(SOURCE_R1_RUN / "final_report.json"),
        "inputs": {"files": files, "artifacts": artifacts},
        "subsets": subsets,
    }
    write_json(run_root / "manifest.json", manifest)
    print(str(run_root), flush=True)
    return run_root


def verify_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_root / "manifest.json")
    if manifest["gate"] != "E2E-G2-R2":
        raise RuntimeError("manifest Gate错误")
    if manifest["reference_read_only"] is not True or manifest["test_executed"] is not False:
        raise RuntimeError("读写隔离或test合同错误")
    preflight.configure_snapshot()
    if code_identity() != manifest["code"]:
        raise RuntimeError("prepare后R2代码或配置发生变化")
    if identity(SOURCE_MANIFEST) != manifest["source_manifest"]:
        raise RuntimeError("源样本manifest身份变化")
    if identity(SOURCE_R1_RUN / "final_report.json") != manifest["source_r1_final"]:
        raise RuntimeError("R1最终报告身份变化")
    return manifest, manifest["config"]


def run_p0(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if (run_root / "p0_report.json").exists():
        raise FileExistsError("拒绝覆盖P0报告")
    checks = {
        "sizes": {
            split: len(records) == expected
            for split, records, expected in (
                ("train", manifest["subsets"]["train"], 256),
                ("val_select", manifest["subsets"]["val_select"], 128),
                ("val_compare", manifest["subsets"]["val_compare"], 512),
            )
        },
        "balanced_k": {
            split: all(
                sum(int(row["true_k"]) == count for row in records)
                == len(records) // 4
                for count in range(4)
            )
            for split, records in manifest["subsets"].items()
        },
        "loss_contract": set(config["loss_weights"])
        == {"exist", "band", "heatmap", "offset"},
        "r1_evidence": read_json(SOURCE_R1_RUN / "final_report.json")["status"]
        == "G2_R1_SPLIT_PASS_FINE_FAIL",
        "test_locked": manifest["test_executed"] is False,
    }
    status = "PASS" if all(
        value if isinstance(value, bool) else all(value.values())
        for value in checks.values()
    ) else "STOP_ENGINEERING"
    report = {
        "status": status,
        "gate": "E2E-G2-R2-P0",
        "checks": checks,
        "overlap_counts": {
            split: sum(bool(row["frequency_overlap"]) for row in records)
            for split, records in manifest["subsets"].items()
        },
        "test_executed": False,
    }
    write_json(run_root / "p0_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def cache_path(run_root: Path, split: str, ordinal: int) -> Path:
    return run_root / "fixed_fullband_cache" / split / f"full_{ordinal:04d}.npy"


def prepare_cache(
    run_root: Path, manifest: dict[str, Any], config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    root = validate_output_path(run_root / "fixed_fullband_cache")
    root.mkdir(exist_ok=True)
    geometry = g1.receiver_geometry(device)
    weights = torch.ones(g1.N_FFT, dtype=torch.float64, device=device)
    all_files: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for split in ("train", "val_select", "val_compare"):
        split_root = root / split
        split_root.mkdir(exist_ok=True)
        descriptions: list[dict[str, Any]] = []
        records = manifest["subsets"][split]
        with g1.SampleStore(split) as store:
            for ordinal, record in enumerate(records, start=1):
                path = cache_path(run_root, split, ordinal)
                sample_started = time.perf_counter()
                if path.exists():
                    array = np.load(path, allow_pickle=False)
                    reused = True
                else:
                    sample = store.sample(record)
                    dpd = preflight.dpd_map(sample["signal"], weights, geometry, config)
                    array = dpd.detach().cpu().numpy().astype(np.float32)
                    temporary = path.with_suffix(".tmp.npy")
                    np.save(temporary, array)
                    os.replace(temporary, path)
                    reused = False
                description = preflight.cache_description(path, array, 0.0)
                description.update(
                    {
                        "ordinal": ordinal,
                        "split": split,
                        "raw_index": int(record["raw_index"]),
                        "local_index": int(record["local_index"]),
                        "seconds": time.perf_counter() - sample_started,
                        "reused": reused,
                    }
                )
                if description["shape"] != [401, 401] or not description["finite"] or not description["nonconstant"]:
                    raise RuntimeError(f"缓存异常: {path}")
                descriptions.append(description)
                if ordinal % 8 == 0 or ordinal == len(records):
                    write_json(
                        run_root / "cache_progress.json",
                        {"active_split": split, "completed": ordinal, "total": len(records)},
                    )
                    print(
                        f"[R2 cache] {split} {ordinal}/{len(records)} elapsed={time.perf_counter() - started:.1f}s",
                        flush=True,
                    )
        all_files[split] = descriptions
    report = {
        "status": "PASS",
        "files": all_files,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(run_root / "cache_manifest.json", report)
    return report


def load_split(run_root: Path, manifest: dict[str, Any], split: str) -> g2.CachedBatch:
    coarse: list[torch.Tensor] = []
    dpd: list[torch.Tensor] = []
    band: list[torch.Tensor] = []
    ignore: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    counts: list[int] = []
    overlap: list[bool] = []
    records = manifest["subsets"][split]
    with g1.SampleStore(split) as store:
        for ordinal, record in enumerate(records, start=1):
            sample = store.sample(record)
            coarse.append(sample["coarse_dpd"])
            dpd.append(torch.from_numpy(np.load(cache_path(run_root, split, ordinal), allow_pickle=False)))
            band.append(sample["band_truth"][:3])
            ignore.append(sample["ignore_truth"][:3])
            position = np.zeros((3, 2), dtype=np.float32)
            count = int(sample["true_k"])
            position[:count] = sample["positions_m"][:count]
            positions.append(torch.from_numpy(position))
            counts.append(count)
            overlap.append(bool(record["frequency_overlap"]))
    return g2.CachedBatch(
        torch.stack(coarse),
        torch.stack(dpd),
        torch.stack(band),
        torch.stack(ignore),
        torch.stack(positions),
        torch.tensor(counts),
        torch.tensor(overlap),
    )


def subset_batch(data: g2.CachedBatch, indices: torch.Tensor) -> g2.CachedBatch:
    return g2.CachedBatch(
        data.coarse[indices],
        data.dpd[indices],
        data.band[indices],
        data.ignore[indices],
        data.positions[indices],
        data.counts[indices],
        data.overlap[indices],
    )


def state_payload(context: r1.TrainContext) -> dict[str, Any]:
    return {
        "ch3_heads": context.ch3.band_heads[:3].state_dict(),
        "query_builder": context.query_builder.state_dict(),
        "splitter": context.splitter.state_dict(),
        "source_head": context.source_head.state_dict(),
    }


def load_state(context: r1.TrainContext, state: dict[str, Any]) -> None:
    context.ch3.band_heads[:3].load_state_dict(state["ch3_heads"], strict=True)
    context.query_builder.load_state_dict(state["query_builder"], strict=True)
    context.splitter.load_state_dict(state["splitter"], strict=True)
    context.source_head.load_state_dict(state["source_head"], strict=True)


def state_digest(context: r1.TrainContext) -> str:
    digest = hashlib.sha256()
    for group, state in state_payload(context).items():
        digest.update(group.encode("utf-8"))
        for name, tensor in sorted(state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def add_auxiliary_metrics(
    context: r1.TrainContext,
    data: g2.CachedBatch,
    device: torch.device,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    logits_parts: list[torch.Tensor] = []
    heatmap_parts: list[torch.Tensor] = []
    for start in range(0, len(data.counts), 4):
        indices = torch.arange(start, min(start + 4, len(data.counts)))
        _, logits, _, heatmap, _ = r1.forward_indices(context, indices, device)
        logits_parts.append(logits.detach().cpu())
        heatmap_parts.append(heatmap.detach().cpu())
    logits = torch.cat(logits_parts)
    heatmap = torch.cat(heatmap_parts)
    mappings = g2.assignments(logits, heatmap, data.band, data.ignore, data.positions, data.counts)
    ious: list[float] = []
    for index, mapping in enumerate(mappings):
        for query, source in mapping.items():
            valid = data.ignore[index, source] < 0.5
            predicted = logits[index, query, valid] >= 0.0
            target = data.band[index, source, valid] > 0.5
            intersection = int((predicted & target).sum().item())
            union = int((predicted | target).sum().item())
            ious.append(intersection / max(union, 1))
    samples = metrics["samples"]
    per_k_accuracy = []
    confusion = [[0 for _ in range(4)] for _ in range(4)]
    for row in samples:
        true_count = int(row["true_count"])
        predicted_count = min(max(int(row["predicted_count"]), 0), 3)
        confusion[true_count][predicted_count] += 1
    for count in range(4):
        total = sum(confusion[count])
        per_k_accuracy.append(confusion[count][count] / max(total, 1))
    metrics["balanced_count_accuracy"] = float(np.mean(per_k_accuracy))
    metrics["count_confusion"] = confusion
    metrics["active_band_macro_iou"] = float(np.mean(ious)) if ious else 1.0
    return metrics


def evaluate_context(
    context: r1.TrainContext, data: g2.CachedBatch, device: torch.device
) -> dict[str, Any]:
    metrics, _ = r1.evaluate_model(context, data, device, None)
    return add_auxiliary_metrics(context, data, device, metrics)


def initial_forward_equal(
    context: r1.TrainContext, data: g2.CachedBatch, device: torch.device
) -> tuple[bool, list[float]]:
    indices = torch.arange(0, min(4, len(data.counts)))
    e2e = r1.forward_indices(context, indices, device, stop_gradient=False)
    sg = r1.forward_indices(context, indices, device, stop_gradient=True)
    differences = [float(torch.max(torch.abs(left - right)).item()) for left, right in zip(e2e, sg)]
    return max(differences) == 0.0, differences


def run_p1(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if read_json(run_root / "p0_report.json")["status"] != "PASS":
        raise RuntimeError("P0未通过")
    if (run_root / "p1_report.json").exists():
        raise FileExistsError("拒绝覆盖P1报告")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_report = prepare_cache(run_root, manifest, config, device)
    train = load_split(run_root, manifest, "train")
    context = r1.build_train_context(train, device, int(config["seed"]))
    equal, differences = initial_forward_equal(context, train, device)
    indices = []
    for count in (1, 2, 3):
        indices.append(int(torch.nonzero(train.counts == count, as_tuple=False)[0].item()))
    probe_indices = torch.tensor(indices)
    e2e_probe = r1.gradient_probe(
        context, train, probe_indices, config, device, "combined", stop_gradient=False
    )
    sg_probe = r1.gradient_probe(
        context, train, probe_indices, config, device, "combined", stop_gradient=True
    )
    gradient_pass = (
        e2e_probe["band_logits"] > 0.0
        and e2e_probe["query"] > 0.0
        and sg_probe["band_logits"] == 0.0
        and sg_probe["query"] == 0.0
        and all(math.isfinite(float(value)) for value in e2e_probe.values())
        and all(math.isfinite(float(value)) for value in sg_probe.values())
    )
    status = "PASS" if equal and gradient_pass else "STOP_ENGINEERING"
    report = {
        "status": status,
        "gate": "E2E-G2-R2-P1",
        "cache": {
            "duration_seconds": cache_report["duration_seconds"],
            "counts": {split: len(rows) for split, rows in cache_report["files"].items()},
        },
        "initial_state_sha256": state_digest(context),
        "forward_equal": equal,
        "forward_max_abs_differences": differences,
        "gradient_audit": {"fs_e2e": e2e_probe, "fs_sg": sg_probe, "pass": gradient_pass},
        "test_executed": False,
    }
    write_json(run_root / "p1_report.json", report)
    del context, train
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def metric_without_samples(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "samples"}


def checkpoint_payload(context: r1.TrainContext, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"epoch": epoch, **state_payload(context), "metrics": metric_without_samples(metrics)}


def transfer_trainable(source: r1.TrainContext, target: r1.TrainContext) -> None:
    load_state(target, state_payload(source))


def selection_score(metrics: dict[str, Any], epoch: int) -> tuple[float, float, int]:
    return (
        -float(metrics["overall"]["gospa_mean_m"]),
        float(metrics["overall"]["recall_at_100m"]),
        -epoch,
    )


def gradient_pair_probe(
    context: r1.TrainContext,
    data: g2.CachedBatch,
    indices: torch.Tensor,
    device: torch.device,
    *,
    stop_gradient: bool,
) -> dict[str, float]:
    upstream = list(context.ch3.band_heads[:3].parameters()) + list(context.query_builder.parameters())

    def vector_for(loss_names: set[str]) -> torch.Tensor:
        _, logits, _, heatmap, offset = r1.forward_indices(
            context, indices, device, stop_gradient=stop_gradient
        )
        _, components, _ = r1.compute_losses(logits, heatmap, offset, data, indices, read_json(CONFIG_PATH))
        loss = sum(components[name] for name in loss_names)
        gradients = torch.autograd.grad(loss, upstream, allow_unused=True)
        pieces = [
            torch.zeros_like(parameter).flatten() if gradient is None else gradient.detach().flatten()
            for parameter, gradient in zip(upstream, gradients)
        ]
        return torch.cat(pieces)

    auxiliary = vector_for({"exist", "band"})
    localization = vector_for({"heatmap", "offset"})
    aux_norm = float(torch.linalg.vector_norm(auxiliary).item())
    loc_norm = float(torch.linalg.vector_norm(localization).item())
    denominator = max(aux_norm * loc_norm, 1e-30)
    cosine = float(torch.dot(auxiliary, localization).item() / denominator) if loc_norm > 0 else 0.0
    return {"auxiliary_norm": aux_norm, "localization_norm": loc_norm, "cosine": cosine}


def train_track(run_root: Path, track: str) -> dict[str, Any]:
    if track not in TRACKS:
        raise ValueError(track)
    manifest, config = verify_run(run_root)
    if read_json(run_root / "p1_report.json")["status"] != "PASS":
        raise RuntimeError("P1未通过")
    track_root = validate_output_path(run_root / "training" / track)
    if track_root.exists():
        raise FileExistsError(f"拒绝覆盖训练目录: {track_root}")
    track_root.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(run_root, manifest, "train")
    select = load_split(run_root, manifest, "val_select")
    context = r1.build_train_context(train, device, int(config["seed"]))
    if state_digest(context) != read_json(run_root / "p1_report.json")["initial_state_sha256"]:
        raise RuntimeError("两轨初始状态不一致")
    select_context = r1.build_train_context(select, device, int(config["seed"]))
    transfer_trainable(context, select_context)
    initial_metrics = evaluate_context(select_context, select, device)
    best_metrics = initial_metrics
    best_epoch = 0
    torch.save(checkpoint_payload(context, 0, initial_metrics), track_root / "best.pt")
    history = [{"epoch": 0, "learning_rate": None, "validation": metric_without_samples(initial_metrics), "training": None}]
    optimizer = torch.optim.AdamW(
        context.parameters,
        lr=float(config["learning_rate_epoch_1"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    stop_gradient = TRACKS[track]
    optimizer_steps = 0
    gradient_samples: list[dict[str, Any]] = []
    clipped_steps = 0
    started = time.perf_counter()
    timed_out = False
    for epoch in range(1, int(config["max_epochs"]) + 1):
        if epoch == 16:
            optimizer.param_groups[0]["lr"] = float(config["learning_rate_epoch_16"])
        elif epoch == 26:
            optimizer.param_groups[0]["lr"] = float(config["learning_rate_epoch_26"])
        order = torch.randperm(len(train.counts), generator=generator)
        sums = {name: 0.0 for name in ("exist", "band", "heatmap", "offset")}
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])]
            if optimizer_steps % int(config["gradient_probe_every_steps"]) == 0:
                probe = gradient_pair_probe(
                    context, train, indices, device, stop_gradient=stop_gradient
                )
                gradient_samples.append({"step": optimizer_steps, **probe})
            _, logits, _, heatmap, offset = r1.forward_indices(
                context, indices, device, stop_gradient=stop_gradient
            )
            total, components, _ = r1.compute_losses(
                logits, heatmap, offset, train, indices, config
            )
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                context.parameters, float(config["gradient_clip"])
            )
            clipped_steps += int(float(norm) > float(config["gradient_clip"]))
            optimizer.step()
            optimizer_steps += 1
            for name, value in components.items():
                sums[name] += float(value.detach().item()) * len(indices)
        if epoch % int(config["evaluate_every"]) == 0:
            transfer_trainable(context, select_context)
            metrics = evaluate_context(select_context, select, device)
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "validation": metric_without_samples(metrics),
                    "training": {name: value / len(train.counts) for name, value in sums.items()},
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            if selection_score(metrics, epoch) > selection_score(best_metrics, best_epoch):
                best_metrics = metrics
                best_epoch = epoch
                torch.save(checkpoint_payload(context, epoch, metrics), track_root / "best.pt")
            write_json(track_root / "progress.json", history)
            print(
                json.dumps(
                    {
                        "track": track,
                        "epoch": epoch,
                        "gospa": metrics["overall"]["gospa_mean_m"],
                        "r100": metrics["overall"]["recall_at_100m"],
                        "best_epoch": best_epoch,
                        "elapsed": time.perf_counter() - started,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if time.perf_counter() - started > float(config["track_wall_limit_seconds"]):
            timed_out = True
            break
    report = {
        "status": "STOP_RESOURCE" if timed_out else "PASS",
        "gate": "E2E-G2-R2-P2",
        "track": track,
        "best_epoch": best_epoch,
        "epoch0": metric_without_samples(initial_metrics),
        "best": metric_without_samples(best_metrics),
        "epochs_completed": history[-1]["epoch"],
        "optimizer_steps": optimizer_steps,
        "duration_seconds": time.perf_counter() - started,
        "gradient_samples": gradient_samples,
        "gradient_clip_count": clipped_steps,
        "gradient_clip_rate": clipped_steps / max(optimizer_steps, 1),
        "checkpoint": identity(track_root / "best.pt"),
        "test_executed": False,
    }
    write_json(track_root / "training_report.json", report)
    del context, select_context, train, select
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps({"track": track, "status": report["status"], "best_epoch": best_epoch}, ensure_ascii=False), flush=True)
    return report


def aggregate_chunks(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [row for chunk in chunks for row in chunk["samples"]]
    result = r1.summarize_with_k(samples)
    total_sources = sum(int(chunk["overall"]["source_count"]) for chunk in chunks)
    result.update(
        {
            "exact_count": sum(int(chunk["exact_count"]) for chunk in chunks),
            "exact_count_rate": sum(int(chunk["exact_count"]) for chunk in chunks) / len(samples),
            "balanced_count_accuracy": float(np.mean([
                sum(int(row["predicted_count"]) == count for row in samples if int(row["true_count"]) == count)
                / max(sum(int(row["true_count"]) == count for row in samples), 1)
                for count in range(4)
            ])),
            "active_band_macro_f1": sum(float(chunk["active_band_macro_f1"]) * int(chunk["overall"]["source_count"]) for chunk in chunks) / max(total_sources, 1),
            "active_band_macro_iou": sum(float(chunk["active_band_macro_iou"]) * int(chunk["overall"]["source_count"]) for chunk in chunks) / max(total_sources, 1),
            "collapsed_multisource_samples": sum(int(chunk["collapsed_multisource_samples"]) for chunk in chunks),
            "offset_worsened_correct_grid_count": sum(int(chunk["offset_worsened_correct_grid_count"]) for chunk in chunks),
            "samples": samples,
        }
    )
    return result


def bootstrap_difference(
    sg_samples: list[dict[str, Any]],
    e2e_samples: list[dict[str, Any]],
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if len(sg_samples) != len(e2e_samples):
        raise RuntimeError("两轨比较样本数不同")
    by_k = {
        count: [index for index, row in enumerate(sg_samples) if int(row["true_count"]) == count]
        for count in range(4)
    }
    differences = np.asarray([e2e_samples[i]["gospa_m"] - sg_samples[i]["gospa_m"] for i in range(len(sg_samples))])
    rng = np.random.default_rng(seed)
    boot_gospa = []
    boot_recall = []
    for _ in range(repetitions):
        selected = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in by_k.values()
        ])
        boot_gospa.append(float(differences[selected].mean()))
        true_sources = sum(int(sg_samples[i]["true_count"]) for i in selected)
        sg_tp = sum(int(sg_samples[i]["tp_at_100m"]) for i in selected)
        e2e_tp = sum(int(e2e_samples[i]["tp_at_100m"]) for i in selected)
        boot_recall.append((e2e_tp - sg_tp) / max(true_sources, 1))
    point_recall_denominator = sum(int(row["true_count"]) for row in sg_samples)
    point_recall = (
        sum(int(row["tp_at_100m"]) for row in e2e_samples)
        - sum(int(row["tp_at_100m"]) for row in sg_samples)
    ) / max(point_recall_denominator, 1)
    return {
        "gospa_e2e_minus_sg_m": {
            "mean": float(differences.mean()),
            "ci95": np.quantile(boot_gospa, [0.025, 0.975]).tolist(),
            "by_k_mean": {
                str(count): float(differences[indices].mean()) for count, indices in by_k.items()
            },
        },
        "recall_100m_e2e_minus_sg": {
            "mean": float(point_recall),
            "ci95": np.quantile(boot_recall, [0.025, 0.975]).tolist(),
        },
    }


def evaluate(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if (run_root / "p3_report.json").exists():
        raise FileExistsError("拒绝覆盖P3报告")
    training = {
        track: read_json(run_root / "training" / track / "training_report.json")
        for track in TRACKS
    }
    if any(report["status"] != "PASS" for report in training.values()):
        raise RuntimeError("两轨训练未完整通过")
    checkpoints = {
        track: torch.load(run_root / "training" / track / "best.pt", map_location="cpu", weights_only=False)
        for track in TRACKS
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare = load_split(run_root, manifest, "val_compare")
    chunk_metrics = {track: [] for track in TRACKS}
    started = time.perf_counter()
    for start in range(0, len(compare.counts), 32):
        indices = torch.arange(start, min(start + 32, len(compare.counts)))
        current = subset_batch(compare, indices)
        context = r1.build_train_context(current, device, int(config["seed"]))
        for track in TRACKS:
            load_state(context, checkpoints[track])
            metrics = evaluate_context(context, current, device)
            for local_index, row in enumerate(metrics["samples"]):
                record = manifest["subsets"]["val_compare"][start + local_index]
                row.update(
                    {
                        "ordinal": start + local_index,
                        "raw_index": int(record["raw_index"]),
                        "local_index": int(record["local_index"]),
                        "frequency_overlap": bool(record["frequency_overlap"]),
                    }
                )
            chunk_metrics[track].append(metrics)
        del context, current
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[R2 compare] {min(start + 32, len(compare.counts))}/{len(compare.counts)}", flush=True)
    metrics = {track: aggregate_chunks(chunks) for track, chunks in chunk_metrics.items()}
    paired = bootstrap_difference(
        metrics["fs_sg"]["samples"],
        metrics["fs_e2e"]["samples"],
        int(config["bootstrap_repetitions"]),
        int(config["seed"]) + 77,
    )
    auxiliary_drops = {
        name: float(metrics["fs_sg"][name]) - float(metrics["fs_e2e"][name])
        for name in ("exact_count_rate", "balanced_count_accuracy", "active_band_macro_f1", "active_band_macro_iou")
    }
    guardrails = {
        "recall_100m": paired["recall_100m_e2e_minus_sg"]["ci95"][0]
        >= float(config["recall_100m_noninferiority"]),
        "auxiliary": all(value <= float(config["auxiliary_drop_maximum"]) for value in auxiliary_drops.values()),
        "e2e_trained_checkpoint": int(training["fs_e2e"]["best_epoch"]) > 0,
    }
    delta = paired["gospa_e2e_minus_sg_m"]
    if int(training["fs_sg"]["best_epoch"]) == 0 and int(training["fs_e2e"]["best_epoch"]) == 0:
        status = "G2_R2_SHARED_LOCALIZATION_LIMIT"
    elif not guardrails["auxiliary"]:
        status = "G2_R2_AUX_REGRESSION"
    elif delta["mean"] < 0 and delta["ci95"][1] < 0 and all(guardrails.values()):
        status = "G2_R2_FEEDBACK_PASS"
    elif delta["mean"] < 0 and delta["ci95"][0] <= 0 <= delta["ci95"][1] and guardrails["auxiliary"] and guardrails["e2e_trained_checkpoint"]:
        status = "G2_R2_FEEDBACK_PROMISING"
    elif delta["ci95"][0] >= 0 and all(int(training[track]["best_epoch"]) > 0 for track in TRACKS):
        status = "G2_R2_FEEDBACK_NO_BENEFIT"
    else:
        status = "G2_R2_INCONCLUSIVE"
    report = {
        "status": status,
        "gate": "E2E-G2-R2-P3",
        "training": {track: {"best_epoch": value["best_epoch"], "checkpoint": value["checkpoint"]} for track, value in training.items()},
        "metrics": {track: metric_without_samples(value) for track, value in metrics.items()},
        "samples": {track: value["samples"] for track, value in metrics.items()},
        "paired_bootstrap": paired,
        "auxiliary_drops_sg_minus_e2e": auxiliary_drops,
        "guardrails": guardrails,
        "absolute_reference": {
            "scope": "R1同32样本原分组DPD与冻结D8，以及既有Hard级联历史结果；不参与R2核心因果判定",
            "r1_p0_report": identity(SOURCE_R1_RUN / "p0_report.json"),
        },
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
    }
    write_json(run_root / "p3_report.json", report)
    print(json.dumps({"status": status, "paired": paired, "guardrails": guardrails}, ensure_ascii=False), flush=True)
    return report


def finalize(run_root: Path) -> dict[str, Any]:
    manifest, _ = verify_run(run_root)
    p0 = read_json(run_root / "p0_report.json") if (run_root / "p0_report.json").exists() else None
    p1 = read_json(run_root / "p1_report.json") if (run_root / "p1_report.json").exists() else None
    tracks = {
        track: read_json(run_root / "training" / track / "training_report.json")
        if (run_root / "training" / track / "training_report.json").exists()
        else None
        for track in TRACKS
    }
    p3 = read_json(run_root / "p3_report.json") if (run_root / "p3_report.json").exists() else None
    if p0 is None or p0["status"] != "PASS":
        status = "STOP_P0"
    elif p1 is None or p1["status"] != "PASS":
        status = "STOP_P1"
    elif any(value is None or value["status"] != "PASS" for value in tracks.values()):
        status = "INCOMPLETE_TRAINING"
    elif p3 is None:
        status = "INCOMPLETE_EVALUATION"
    else:
        status = p3["status"]
    report = {
        "status": status,
        "gate": "E2E-G2-R2",
        "run_root": str(run_root.resolve()),
        "manifest": identity(run_root / "manifest.json"),
        "p0": None if p0 is None else p0["status"],
        "p1": None if p1 is None else p1["status"],
        "tracks": {track: None if value is None else value["status"] for track, value in tracks.items()},
        "p3": None if p3 is None else p3["status"],
        "source_unchanged": identity(SOURCE_MANIFEST) == manifest["source_manifest"] and identity(SOURCE_R1_RUN / "final_report.json") == manifest["source_r1_final"],
        "test_executed": False,
    }
    write_json(run_root / "final_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    for command in ("run-p0", "run-p1", "evaluate", "finalize"):
        current = sub.add_parser(command)
        current.add_argument("--run-root", type=Path, required=True)
    train_parser = sub.add_parser("train")
    train_parser.add_argument("--run-root", type=Path, required=True)
    train_parser.add_argument("--track", choices=tuple(TRACKS), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare":
        prepare(args.run_id)
    elif args.command == "run-p0":
        run_p0(args.run_root.resolve())
    elif args.command == "run-p1":
        run_p1(args.run_root.resolve())
    elif args.command == "train":
        train_track(args.run_root.resolve(), args.track)
    elif args.command == "evaluate":
        evaluate(args.run_root.resolve())
    else:
        finalize(args.run_root.resolve())
