"""E2E-G3：定位反馈跨training seed稳定性门。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# R2先固定第四章chapter_runtime模块身份，再导入依赖它的R1/D1。
from 统一模型代码.gates.g2 import e2e_g2_r2 as r2  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_r1 as r1  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_r2_d1 as d1  # noqa: E402
from 统一模型代码.common.runtime_paths import (  # noqa: E402
    new_run_dir,
    validate_output_path,
)


CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g3.json"
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_R2_RUN = (
    PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_g2_r2" / "20260904_163951"
)
SOURCE_D1_RUN = (
    PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_g2_r2_d1" / "20260904_180714"
)
SOURCE_CACHE_ROOT = SOURCE_R2_RUN / "fixed_fullband_cache"
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
        Path(r1.__file__).resolve(),
        Path(r2.__file__).resolve(),
        Path(d1.__file__).resolve(),
        r2.MODEL_PATH,
    )
    return [identity(path) for path in paths]


def source_identity() -> dict[str, Any]:
    return {
        "r2_manifest": identity(SOURCE_R2_RUN / "manifest.json"),
        "r2_cache_manifest": identity(SOURCE_R2_RUN / "cache_manifest.json"),
        "r2_final": identity(SOURCE_R2_RUN / "final_report.json"),
        "d1_final": identity(SOURCE_D1_RUN / "final_report.json"),
    }


def prepare(run_id: str) -> Path:
    config = read_json(CONFIG_PATH)
    source_manifest = read_json(SOURCE_R2_RUN / "manifest.json")
    run_root = new_run_dir("e2e_g3", run_id, create=True)
    manifest = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run",
            "verification_status": "UNVERIFIED",
        },
        "status": "PREPARED",
        "gate": "E2E-G3",
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_read_only": True,
        "test_executed": False,
        "config": config,
        "code": code_identity(),
        "source": source_identity(),
        "source_cache_root": str(SOURCE_CACHE_ROOT.resolve()),
        "subsets": source_manifest["subsets"],
    }
    write_json(run_root / "manifest.json", manifest)
    print(str(run_root.resolve()), flush=True)
    return run_root


def verify_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_root / "manifest.json")
    if manifest["gate"] != "E2E-G3":
        raise RuntimeError("manifest Gate错误")
    if manifest["reference_read_only"] is not True or manifest["test_executed"] is not False:
        raise RuntimeError("读写隔离或test合同错误")
    current_code = code_identity()
    if manifest["code"] != current_code:
        receipt_path = run_root / "repair_receipt.json"
        if not receipt_path.exists():
            raise RuntimeError("prepare后G3代码或配置发生变化且无修复收据")
        receipt = read_json(receipt_path)
        valid_repair = (
            receipt.get("original_code") == manifest["code"]
            and receipt.get("repaired_code") == current_code
            and receipt.get("reason") == "rng_state_cpu_load"
            and receipt.get("scientific_contract_changed") is False
        )
        if not valid_repair:
            raise RuntimeError("G3工程修复收据无效")
    if manifest["source"] != source_identity():
        raise RuntimeError("R2/D1源证据身份变化")
    if Path(manifest["source_cache_root"]).resolve() != SOURCE_CACHE_ROOT.resolve():
        raise RuntimeError("全频缓存根路径变化")
    return manifest, manifest["config"]


def record_repair(run_root: Path) -> dict[str, Any]:
    """登记仅修改G3入口的高置信工程修复，不放宽其他身份检查。"""
    manifest = read_json(run_root / "manifest.json")
    receipt_path = run_root / "repair_receipt.json"
    if receipt_path.exists():
        raise FileExistsError("修复收据已存在")
    current = code_identity()
    before = {row["path"]: row for row in manifest["code"]}
    after = {row["path"]: row for row in current}
    if set(before) != set(after):
        raise RuntimeError("代码身份文件集合变化")
    changed = [path for path in before if before[path] != after[path]]
    if changed != [str(SCRIPT_PATH.resolve())]:
        raise RuntimeError(f"修复范围超出G3入口: {changed}")
    receipt = {
        "reason": "rng_state_cpu_load",
        "failure_stage": "seed_20260906_extension_before_optimizer_step",
        "scientific_contract_changed": False,
        "failed_extension_updates_reused": False,
        "completed_epoch38_evidence_preserved": True,
        "original_code": manifest["code"],
        "repaired_code": current,
        "changed_paths": changed,
    }
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return receipt


def source_cache_path(split: str, ordinal: int) -> Path:
    return SOURCE_CACHE_ROOT / split / f"full_{ordinal:04d}.npy"


def verify_source_cache(manifest: dict[str, Any]) -> dict[str, Any]:
    cache_manifest = read_json(SOURCE_R2_RUN / "cache_manifest.json")
    checked = 0
    started = time.perf_counter()
    for split in ("train", "val_select", "val_compare"):
        descriptions = cache_manifest["files"][split]
        records = manifest["subsets"][split]
        if len(descriptions) != len(records):
            raise RuntimeError(f"{split}缓存数量错误")
        for ordinal, (description, record) in enumerate(
            zip(descriptions, records), start=1
        ):
            path = source_cache_path(split, ordinal)
            current = identity(path)
            if current["size_bytes"] != int(description["size_bytes"]):
                raise RuntimeError(f"缓存大小变化: {path}")
            if current["sha256"] != description["sha256"]:
                raise RuntimeError(f"缓存SHA变化: {path}")
            if int(description["raw_index"]) != int(record["raw_index"]):
                raise RuntimeError(f"缓存样本身份错误: {path}")
            checked += 1
    return {
        "status": "PASS",
        "files_checked": checked,
        "cache_manifest": identity(SOURCE_R2_RUN / "cache_manifest.json"),
        "duration_seconds": time.perf_counter() - started,
    }


def load_split(manifest: dict[str, Any], split: str) -> r2.g2.CachedBatch:
    coarse: list[torch.Tensor] = []
    dpd: list[torch.Tensor] = []
    band: list[torch.Tensor] = []
    ignore: list[torch.Tensor] = []
    positions: list[torch.Tensor] = []
    counts: list[int] = []
    overlap: list[bool] = []
    records = manifest["subsets"][split]
    with r2.g1.SampleStore(split) as store:
        for ordinal, record in enumerate(records, start=1):
            sample = store.sample(record)
            coarse.append(sample["coarse_dpd"])
            dpd.append(
                torch.from_numpy(
                    np.load(source_cache_path(split, ordinal), allow_pickle=False)
                )
            )
            band.append(sample["band_truth"][:3])
            ignore.append(sample["ignore_truth"][:3])
            position = np.zeros((3, 2), dtype=np.float32)
            count = int(sample["true_k"])
            position[:count] = sample["positions_m"][:count]
            positions.append(torch.from_numpy(position))
            counts.append(count)
            overlap.append(bool(record["frequency_overlap"]))
    return r2.g2.CachedBatch(
        torch.stack(coarse),
        torch.stack(dpd),
        torch.stack(band),
        torch.stack(ignore),
        torch.stack(positions),
        torch.tensor(counts),
        torch.tensor(overlap),
    )


def gradient_pair_probe(
    context: r1.TrainContext,
    data: r2.g2.CachedBatch,
    indices: torch.Tensor,
    device: torch.device,
    config: dict[str, Any],
    *,
    stop_gradient: bool,
) -> dict[str, float]:
    upstream = list(context.ch3.band_heads[:3].parameters()) + list(
        context.query_builder.parameters()
    )

    def vector_for(loss_names: set[str]) -> torch.Tensor:
        _, logits, _, heatmap, offset = r1.forward_indices(
            context, indices, device, stop_gradient=stop_gradient
        )
        _, components, _ = r1.compute_losses(
            logits, heatmap, offset, data, indices, config
        )
        gradients = torch.autograd.grad(
            sum(components[name] for name in loss_names), upstream, allow_unused=True
        )
        return torch.cat(
            [
                torch.zeros_like(parameter).flatten()
                if gradient is None
                else gradient.detach().flatten()
                for parameter, gradient in zip(upstream, gradients)
            ]
        )

    auxiliary = vector_for({"exist", "band"})
    localization = vector_for({"heatmap", "offset"})
    aux_norm = float(torch.linalg.vector_norm(auxiliary).item())
    loc_norm = float(torch.linalg.vector_norm(localization).item())
    denominator = max(aux_norm * loc_norm, 1e-30)
    cosine = (
        float(torch.dot(auxiliary, localization).item() / denominator)
        if loc_norm > 0
        else 0.0
    )
    return {
        "auxiliary_norm": aux_norm,
        "localization_norm": loc_norm,
        "cosine": cosine,
    }


def run_p0(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if (run_root / "p0_report.json").exists():
        raise FileExistsError("拒绝覆盖P0报告")
    expected = {"train": 256, "val_select": 128, "val_compare": 512}
    sizes = {
        split: len(manifest["subsets"][split]) == count
        for split, count in expected.items()
    }
    cache = verify_source_cache(manifest)
    data = load_split(manifest, "train")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_checks: dict[str, Any] = {}
    for seed in config["confirmation_seeds"]:
        r2.set_deterministic(int(seed))
        left = r1.build_train_context(data, device, int(seed))
        left_digest = r2.state_digest(left)
        equal_forward, differences = r2.initial_forward_equal(left, data, device)
        sg_probe = gradient_pair_probe(
            left, data, torch.arange(4), device, config, stop_gradient=True
        )
        e2e_probe = gradient_pair_probe(
            left, data, torch.arange(4), device, config, stop_gradient=False
        )
        r2.set_deterministic(int(seed))
        right = r1.build_train_context(data, device, int(seed))
        right_digest = r2.state_digest(right)
        seed_checks[str(seed)] = {
            "paired_initial_state_equal": left_digest == right_digest,
            "initial_state_sha256": left_digest,
            "forward_equal": equal_forward,
            "forward_max_abs_differences": differences,
            "sg_upstream_localization_norm": sg_probe["localization_norm"],
            "e2e_upstream_localization_norm": e2e_probe["localization_norm"],
            "gradient_contract": sg_probe["localization_norm"] == 0.0
            and np.isfinite(e2e_probe["localization_norm"])
            and e2e_probe["localization_norm"] > 0.0,
        }
        del left, right
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    status = "PASS" if (
        all(sizes.values())
        and cache["status"] == "PASS"
        and all(
            check["paired_initial_state_equal"]
            and check["forward_equal"]
            and check["gradient_contract"]
            for check in seed_checks.values()
        )
    ) else "STOP_ENGINEERING"
    report = {
        "status": status,
        "gate": "E2E-G3-P0",
        "sizes": sizes,
        "cache": cache,
        "seed_checks": seed_checks,
        "test_executed": False,
    }
    write_json(run_root / "p0_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def learning_rate(epoch: int, config: dict[str, Any]) -> float:
    if epoch < 16:
        return float(config["learning_rate_epoch_1"])
    if epoch < 26:
        return float(config["learning_rate_epoch_16"])
    return float(config["learning_rate_epoch_26"])


def rng_payload(generator: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator": generator.get_state(),
    }


def restore_rng(payload: dict[str, Any], generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload["cuda"] is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])
    generator.set_state(payload["generator"])


def save_last(
    path: Path,
    context: r1.TrainContext,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    epoch: int,
    best_epoch: int,
    best_metrics: dict[str, Any],
    optimizer_steps: int,
    clipped_steps: int,
    gradient_samples: list[dict[str, Any]],
) -> None:
    torch.save(
        {
            "epoch": epoch,
            **r2.state_payload(context),
            "optimizer": optimizer.state_dict(),
            "rng": rng_payload(generator),
            "best_epoch": best_epoch,
            "best_metrics": r2.metric_without_samples(best_metrics),
            "optimizer_steps": optimizer_steps,
            "clipped_steps": clipped_steps,
            "gradient_samples": gradient_samples,
        },
        path,
    )


def train_track(
    run_root: Path, seed: int, track: str, target_epoch: int, resume: bool
) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if read_json(run_root / "p0_report.json")["status"] != "PASS":
        raise RuntimeError("P0未通过")
    track_root = validate_output_path(run_root / "seeds" / str(seed) / track)
    track_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = load_split(manifest, "train")
    select = load_split(manifest, "val_select")
    r2.set_deterministic(seed)
    context = r1.build_train_context(train, device, seed)
    select_context = r1.build_train_context(select, device, seed)
    optimizer = torch.optim.AdamW(
        context.parameters,
        lr=float(config["learning_rate_epoch_1"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    last_path = track_root / "last.pt"
    best_path = track_root / "best.pt"
    progress_path = track_root / "progress.json"
    if resume:
        if not last_path.exists() or not progress_path.exists():
            raise RuntimeError(f"缺少续训状态: {track_root}")
        # RNG状态必须保持CPU ByteTensor；optimizer.load_state_dict会按参数设备迁移状态。
        last = torch.load(last_path, map_location="cpu", weights_only=False)
        r2.load_state(context, last)
        optimizer.load_state_dict(last["optimizer"])
        restore_rng(last["rng"], generator)
        history = read_json(progress_path)
        start_epoch = int(last["epoch"]) + 1
        best_epoch = int(last["best_epoch"])
        best_metrics = last["best_metrics"]
        optimizer_steps = int(last["optimizer_steps"])
        clipped_steps = int(last["clipped_steps"])
        gradient_samples = list(last["gradient_samples"])
    else:
        if best_path.exists() or progress_path.exists() or last_path.exists():
            raise FileExistsError(f"拒绝覆盖训练证据: {track_root}")
        initial_metrics = r2.evaluate_context(select_context, select, device)
        best_epoch = 0
        best_metrics = r2.metric_without_samples(initial_metrics)
        torch.save(r2.checkpoint_payload(context, 0, initial_metrics), best_path)
        history = [
            {
                "epoch": 0,
                "learning_rate": None,
                "validation": best_metrics,
                "training": None,
            }
        ]
        start_epoch = 1
        optimizer_steps = 0
        clipped_steps = 0
        gradient_samples = []
    stop_gradient = TRACKS[track]
    started = time.perf_counter()
    timed_out = False
    for epoch in range(start_epoch, target_epoch + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch, config)
        order = torch.randperm(len(train.counts), generator=generator)
        sums = {name: 0.0 for name in ("exist", "band", "heatmap", "offset")}
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])]
            if optimizer_steps % int(config["gradient_probe_every_steps"]) == 0:
                probe = gradient_pair_probe(
                    context,
                    train,
                    indices,
                    device,
                    config,
                    stop_gradient=stop_gradient,
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
            r2.transfer_trainable(context, select_context)
            metrics = r2.evaluate_context(select_context, select, device)
            compact = r2.metric_without_samples(metrics)
            history.append(
                {
                    "epoch": epoch,
                    "learning_rate": learning_rate(epoch, config),
                    "validation": compact,
                    "training": {
                        name: value / len(train.counts) for name, value in sums.items()
                    },
                    "elapsed_seconds_current_phase": time.perf_counter() - started,
                }
            )
            if r2.selection_score(metrics, epoch) > r2.selection_score(
                best_metrics, best_epoch
            ):
                best_metrics = compact
                best_epoch = epoch
                torch.save(r2.checkpoint_payload(context, epoch, metrics), best_path)
            write_json(progress_path, history)
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "track": track,
                        "epoch": epoch,
                        "gospa": compact["overall"]["gospa_mean_m"],
                        "recall100": compact["overall"]["recall_at_100m"],
                        "best_epoch": best_epoch,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if time.perf_counter() - started > float(config["track_wall_limit_seconds"]):
            timed_out = True
            break
    completed_epoch = int(history[-1]["epoch"])
    save_last(
        last_path,
        context,
        optimizer,
        generator,
        completed_epoch,
        best_epoch,
        best_metrics,
        optimizer_steps,
        clipped_steps,
        gradient_samples,
    )
    report = {
        "status": "STOP_RESOURCE" if timed_out else "PASS",
        "gate": "E2E-G3-P1",
        "seed": seed,
        "track": track,
        "target_epoch": target_epoch,
        "epochs_completed": completed_epoch,
        "best_epoch": best_epoch,
        "best": best_metrics,
        "optimizer_steps": optimizer_steps,
        "gradient_samples": gradient_samples,
        "gradient_clip_count": clipped_steps,
        "gradient_clip_rate": clipped_steps / max(optimizer_steps, 1),
        "duration_seconds_current_phase": time.perf_counter() - started,
        "checkpoint": identity(best_path),
        "last_checkpoint": identity(last_path),
        "test_executed": False,
    }
    write_json(track_root / f"training_report_epoch{target_epoch}.json", report)
    write_json(track_root / "training_report.json", report)
    del context, select_context, train, select, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def train_seed(run_root: Path, seed: int) -> dict[str, Any]:
    _, config = verify_run(run_root)
    if seed not in [int(value) for value in config["confirmation_seeds"]]:
        raise ValueError(f"未预注册seed: {seed}")
    seed_root = validate_output_path(run_root / "seeds" / str(seed))
    seed_root.mkdir(parents=True, exist_ok=True)
    report_path = seed_root / "seed_report.json"
    if report_path.exists():
        raise FileExistsError(f"拒绝覆盖seed报告: {report_path}")
    first = {
        track: train_track(
            run_root, seed, track, int(config["comparison_epoch"]), resume=False
        )
        for track in TRACKS
    }
    if any(report["status"] != "PASS" for report in first.values()):
        raise RuntimeError(f"seed {seed} 的38 epoch训练未完成")
    late = {int(value) for value in config["late_best_epochs"]}
    extension_triggered = any(int(report["best_epoch"]) in late for report in first.values())
    if extension_triggered:
        final = {
            track: train_track(
                run_root, seed, track, int(config["extension_epoch"]), resume=True
            )
            for track in TRACKS
        }
    else:
        final = first
    unresolved = extension_triggered and any(
        int(report["best_epoch"])
        in {int(config["extension_epoch"]) - 2, int(config["extension_epoch"])}
        for report in final.values()
    )
    report = {
        "status": "PASS" if all(value["status"] == "PASS" for value in final.values()) else "STOP_RESOURCE",
        "gate": "E2E-G3-P1",
        "seed": seed,
        "epoch38_best": {track: value["best_epoch"] for track, value in first.items()},
        "extension_triggered": extension_triggered,
        "training_budget_status": (
            "TRAINING_BUDGET_UNRESOLVED"
            if unresolved
            else "TRAINING_BUDGET_EXTENDED"
            if extension_triggered
            else "TRAINING_BUDGET_OK"
        ),
        "final_target_epoch": int(config["extension_epoch"])
        if extension_triggered
        else int(config["comparison_epoch"]),
        "final_best": {track: value["best_epoch"] for track, value in final.items()},
        "test_executed": False,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def resume_seed_extension(run_root: Path, seed: int) -> dict[str, Any]:
    """从已完整保存的38 epoch状态恢复尚未开始的成对延长阶段。"""
    _, config = verify_run(run_root)
    if seed not in [int(value) for value in config["confirmation_seeds"]]:
        raise ValueError(f"未预注册seed: {seed}")
    seed_root = validate_output_path(run_root / "seeds" / str(seed))
    report_path = seed_root / "seed_report.json"
    if report_path.exists():
        raise FileExistsError(f"seed已经完成: {report_path}")
    first = {
        track: read_json(seed_root / track / "training_report_epoch38.json")
        for track in TRACKS
    }
    if any(
        report["status"] != "PASS"
        or int(report["epochs_completed"]) != int(config["comparison_epoch"])
        for report in first.values()
    ):
        raise RuntimeError("38 epoch配对证据不完整，拒绝恢复延长")
    late = {int(value) for value in config["late_best_epochs"]}
    if not any(int(report["best_epoch"]) in late for report in first.values()):
        raise RuntimeError("未满足预注册延长触发条件")
    final = {
        track: train_track(
            run_root, seed, track, int(config["extension_epoch"]), resume=True
        )
        for track in TRACKS
    }
    unresolved = any(
        int(report["best_epoch"])
        in {int(config["extension_epoch"]) - 2, int(config["extension_epoch"])}
        for report in final.values()
    )
    report = {
        "status": "PASS"
        if all(value["status"] == "PASS" for value in final.values())
        else "STOP_RESOURCE",
        "gate": "E2E-G3-P1",
        "seed": seed,
        "epoch38_best": {track: value["best_epoch"] for track, value in first.items()},
        "extension_triggered": True,
        "extension_recovered_after_pre_step_rng_load_error": True,
        "training_budget_status": (
            "TRAINING_BUDGET_UNRESOLVED"
            if unresolved
            else "TRAINING_BUDGET_EXTENDED"
        ),
        "final_target_epoch": int(config["extension_epoch"]),
        "final_best": {track: value["best_epoch"] for track, value in final.items()},
        "test_executed": False,
    }
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def evaluate_seed(
    run_root: Path,
    manifest: dict[str, Any],
    config: dict[str, Any],
    seed: int,
    compare: r2.g2.CachedBatch,
    device: torch.device,
) -> dict[str, Any]:
    seed_root = run_root / "seeds" / str(seed)
    checkpoints = {
        track: torch.load(
            seed_root / track / "best.pt", map_location="cpu", weights_only=False
        )
        for track in TRACKS
    }
    chunk_metrics: dict[str, list[dict[str, Any]]] = {track: [] for track in TRACKS}
    band_records: dict[str, list[dict[str, Any]]] = {track: [] for track in TRACKS}
    records = manifest["subsets"]["val_compare"]
    for start in range(0, len(compare.counts), 32):
        indices = torch.arange(start, min(start + 32, len(compare.counts)))
        current = r2.subset_batch(compare, indices)
        context = r1.build_train_context(current, device, seed)
        for track in TRACKS:
            r2.load_state(context, checkpoints[track])
            metrics = r2.evaluate_context(context, current, device)
            for local_index, row in enumerate(metrics["samples"]):
                record = records[start + local_index]
                row.update(
                    {
                        "ordinal": start + local_index,
                        "raw_index": int(record["raw_index"]),
                        "local_index": int(record["local_index"]),
                        "frequency_overlap": bool(record["frequency_overlap"]),
                    }
                )
            chunk_metrics[track].append(metrics)
            logits, heatmap = d1.forward_context(context, len(current.counts), device, 4)
            band_records[track].extend(
                d1.analyze_outputs(logits, heatmap, current, start, records)
            )
        del context, current
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[G3 seed={seed}] {min(start + 32, len(compare.counts))}/{len(compare.counts)}",
            flush=True,
        )
    metrics = {
        track: r2.aggregate_chunks(chunks) for track, chunks in chunk_metrics.items()
    }
    localization = r2.bootstrap_difference(
        metrics["fs_sg"]["samples"],
        metrics["fs_e2e"]["samples"],
        int(config["bootstrap_repetitions"]),
        seed + 77,
    )
    band_grouped = {
        track: d1.grouped_metrics(rows, "band_only")
        for track, rows in band_records.items()
    }
    auxiliary = d1.bootstrap_difference(
        band_records["fs_sg"],
        band_records["fs_e2e"],
        "band_only",
        int(config["bootstrap_repetitions"]),
        seed + 202,
    )
    reports = {
        track: read_json(seed_root / track / "training_report.json")
        for track in TRACKS
    }
    return {
        "seed": seed,
        "training": {
            track: {
                "best_epoch": report["best_epoch"],
                "target_epoch": report["target_epoch"],
                "checkpoint": report["checkpoint"],
            }
            for track, report in reports.items()
        },
        "localization_metrics": {
            track: r2.metric_without_samples(value) for track, value in metrics.items()
        },
        "localization_paired_bootstrap": localization,
        "band_only_metrics": band_grouped,
        "auxiliary_paired_bootstrap": auxiliary,
        "samples": {
            "localization": {track: value["samples"] for track, value in metrics.items()},
            "band_only": band_records,
        },
    }


def classify(results: dict[str, dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    seeds = [str(int(seed)) for seed in config["confirmation_seeds"]]
    gospa = {
        seed: results[seed]["localization_paired_bootstrap"]["gospa_e2e_minus_sg_m"]
        for seed in seeds
    }
    recall = {
        seed: results[seed]["localization_paired_bootstrap"]["recall_100m_e2e_minus_sg"]
        for seed in seeds
    }
    improved = [float(gospa[seed]["mean"]) < 0.0 for seed in seeds]
    mean_delta = float(np.mean([gospa[seed]["mean"] for seed in seeds]))
    recall_guard = all(
        float(recall[seed]["ci95"][0])
        >= float(config["recall_100m_noninferiority"])
        for seed in seeds
    )
    trained = all(
        int(results[seed]["training"]["fs_e2e"]["best_epoch"]) > 0 for seed in seeds
    )
    if all(improved) and mean_delta < 0.0 and recall_guard and trained:
        feedback = "G3_FEEDBACK_STABLE_PASS"
    elif sum(improved) == 1 and mean_delta < 0.0 and trained:
        feedback = "G3_FEEDBACK_UNSTABLE"
    else:
        feedback = "G3_FEEDBACK_NO_BENEFIT"

    metric_names = ("exact_count_rate", "band_macro_f1", "band_macro_iou")
    point_crossings: dict[str, dict[str, bool]] = {}
    supported_harm: dict[str, dict[str, bool]] = {}
    margin = float(config["auxiliary_drop_maximum"])
    for name in metric_names:
        point_crossings[name] = {}
        supported_harm[name] = {}
        for seed in seeds:
            result = results[seed]["auxiliary_paired_bootstrap"][name]
            point_crossings[name][seed] = float(result["e2e_minus_sg"]) < -margin
            supported_harm[name][seed] = point_crossings[name][seed] and float(
                result["ci95"][1]
            ) < -margin
    material = any(all(supported_harm[name].values()) for name in metric_names)
    uncertain = any(any(point_crossings[name].values()) for name in metric_names)
    auxiliary = (
        "MATERIAL_AUX_HARM"
        if material
        else "AUX_UNCERTAIN"
        if uncertain
        else "AUX_ACCEPTABLE"
    )
    final = (
        "G3_FEEDBACK_STABLE_AUX_HARM"
        if feedback == "G3_FEEDBACK_STABLE_PASS" and auxiliary == "MATERIAL_AUX_HARM"
        else feedback
    )
    return {
        "status": final,
        "feedback_status": feedback,
        "auxiliary_status": auxiliary,
        "new_seed_gospa_mean_delta_m": mean_delta,
        "per_seed_localization_improved": dict(zip(seeds, improved)),
        "recall_guard_pass": recall_guard,
        "e2e_trained_checkpoint": trained,
        "auxiliary_point_crossings": point_crossings,
        "auxiliary_supported_harm": supported_harm,
    }


def evaluate(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if (run_root / "p2_report.json").exists():
        raise FileExistsError("拒绝覆盖P2报告")
    for seed in config["confirmation_seeds"]:
        report = read_json(run_root / "seeds" / str(seed) / "seed_report.json")
        if report["status"] != "PASS":
            raise RuntimeError(f"seed {seed}训练未通过")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compare = load_split(manifest, "val_compare")
    started = time.perf_counter()
    results = {
        str(int(seed)): evaluate_seed(
            run_root, manifest, config, int(seed), compare, device
        )
        for seed in config["confirmation_seeds"]
    }
    classification = classify(results, config)
    historical_r2 = read_json(SOURCE_R2_RUN / "p3_report.json")
    historical_d1 = read_json(SOURCE_D1_RUN / "final_report.json")
    report = {
        "status": classification["status"],
        "gate": "E2E-G3-P2",
        "classification": classification,
        "confirmation_seeds": results,
        "historical_seed": {
            "seed": int(config["historical_seed"]),
            "role": "historical_support_only",
            "r2_status": historical_r2["status"],
            "localization_paired_bootstrap": historical_r2["paired_bootstrap"],
            "d1_status": historical_d1["status"],
            "d1_band_only_bootstrap": historical_d1["paired_bootstrap"]["band_only"],
        },
        "duration_seconds": time.perf_counter() - started,
        "test_executed": False,
        "sample_bootstrap_excludes_training_seed_variance": True,
    }
    write_json(run_root / "p2_report.json", report)
    print(json.dumps({"status": report["status"], **classification}, ensure_ascii=False), flush=True)
    del compare
    return report


def finalize(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    p0 = read_json(run_root / "p0_report.json")
    seeds = {
        str(int(seed)): read_json(run_root / "seeds" / str(seed) / "seed_report.json")
        for seed in config["confirmation_seeds"]
    }
    p2 = read_json(run_root / "p2_report.json")
    cache_after = verify_source_cache(manifest)
    source_unchanged = manifest["source"] == source_identity()
    status = p2["status"] if (
        p0["status"] == "PASS"
        and all(report["status"] == "PASS" for report in seeds.values())
        and cache_after["status"] == "PASS"
        and source_unchanged
    ) else "INCOMPLETE_OR_IDENTITY_FAILURE"
    report = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run+validate",
            "verification_status": "ANALYZED",
        },
        "status": status,
        "gate": "E2E-G3",
        "run_root": str(run_root.resolve()),
        "manifest": identity(run_root / "manifest.json"),
        "p0": p0["status"],
        "seeds": {seed: value["status"] for seed, value in seeds.items()},
        "p2": p2["status"],
        "classification": p2["classification"],
        "cache_after": cache_after,
        "source_unchanged": source_unchanged,
        "test_executed": False,
        "fallacy_scan": {
            "coverage": "11/11",
            "simpson": "按K报告，检查总体与分层方向",
            "ecological": "不从聚合seed结果推断单样本机制",
            "berkson": "固定平衡K validation，外推受限",
            "collider": "未加入事后控制变量",
            "base_rate": "报告平衡K先验与count指标",
            "regression_to_mean": "新seed预注册，历史seed只作支持",
            "survivorship": "完整轨道均纳入，不筛选成功seed",
            "look_elsewhere": "GOSPA为预注册主指标",
            "forking_paths": "seed、epoch延长和分流规则运行前固定",
            "correlation_causation": "因果解释限于同seed配对的梯度开关",
            "reverse_causality": "前向相同，仅定位反馈路径不同",
        },
    }
    write_json(run_root / "final_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    for command in ("run-p0", "evaluate", "finalize", "verify", "record-repair"):
        current = sub.add_parser(command)
        current.add_argument("--run-root", type=Path, required=True)
    for command in ("train-seed", "resume-seed-extension"):
        train_parser = sub.add_parser(command)
        train_parser.add_argument("--run-root", type=Path, required=True)
        train_parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "prepare":
        prepare(args.run_id)
    elif args.command == "run-p0":
        run_p0(args.run_root.resolve())
    elif args.command == "train-seed":
        train_seed(args.run_root.resolve(), args.seed)
    elif args.command == "resume-seed-extension":
        resume_seed_extension(args.run_root.resolve(), args.seed)
    elif args.command == "evaluate":
        evaluate(args.run_root.resolve())
    elif args.command == "finalize":
        finalize(args.run_root.resolve())
    elif args.command == "record-repair":
        record_repair(args.run_root.resolve())
    else:
        verify_run(args.run_root.resolve())
        print("PASS", flush=True)
