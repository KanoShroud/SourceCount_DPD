"""从完整epoch checkpoint恢复被外部中断的E2E-G1训练轨。"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 统一模型代码 import e2e_g1 as g1


def restore_rng(payload: dict[str, Any]) -> None:
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"] is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])


def preserve_partial_log(track_root: Path, completed_epoch: int) -> dict[str, Any]:
    source = track_root / "gradient_steps.jsonl"
    rows = g1.read_jsonl(source)
    partial = [row for row in rows if int(row["epoch"]) > completed_epoch]
    complete = [row for row in rows if int(row["epoch"]) <= completed_epoch]
    require_partial = len(partial) > 0
    if not require_partial:
        raise RuntimeError("没有检测到需要保留的半程梯度日志")
    preserved = track_root / f"gradient_steps_interrupted_after_epoch_{completed_epoch:03d}.jsonl"
    if preserved.exists():
        raise FileExistsError(f"中断日志已存在: {preserved}")
    source.replace(preserved)
    for row in complete:
        g1.append_jsonl(source, row)
    return {
        "preserved_log": str(preserved.resolve()),
        "complete_rows_restored": len(complete),
        "partial_rows_preserved": len(partial),
        "partial_epoch": sorted({int(row["epoch"]) for row in partial}),
    }


def resume(run_root: Path, track: str, completed_epoch: int) -> dict[str, Any]:
    if track not in {"soft_sg", "soft_e2e"}:
        raise ValueError(f"不支持的训练轨: {track}")
    manifest = g1.verify_manifest(run_root)
    config = manifest["config"]
    final_epoch = int(config["epochs"])
    if completed_epoch >= final_epoch:
        raise ValueError("恢复点已经达到最终epoch")
    track_root = g1.validate_output_path(run_root / "training" / track)
    checkpoint_path = track_root / f"epoch_{completed_epoch:03d}.pth"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"完整恢复checkpoint不存在: {checkpoint_path}")
    for epoch in range(completed_epoch + 1, final_epoch + 1):
        if (track_root / f"epoch_{epoch:03d}.pth").exists():
            raise FileExistsError(f"拒绝覆盖已有epoch checkpoint: {epoch}")
    history = g1.load_json(track_root / "history.json")
    if int(history[-1]["epoch"]) != completed_epoch:
        raise RuntimeError("history末尾不是指定完整恢复epoch")
    partial_evidence = preserve_partial_log(track_root, completed_epoch)

    seed = int(config["seed"])
    g1.set_deterministic(seed)
    device = torch.device("cuda:0")
    ch3, d8, model_info = g1.build_models(device)
    ch3_parameters, d8_parameters = g1.configure_trainable(ch3, d8, training=True)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": ch3_parameters,
                "lr": float(config["learning_rate"]),
                "weight_decay": float(config["ch3_weight_decay"]),
            },
            {
                "params": d8_parameters,
                "lr": float(config["learning_rate"]),
                "weight_decay": float(config["d8_weight_decay"]),
            },
        ]
    )
    payload = g1.load_checkpoint(checkpoint_path, ch3, d8)
    if payload["track"] != track or int(payload["epoch"]) != completed_epoch:
        raise RuntimeError("恢复checkpoint轨道或epoch不一致")
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload)

    started = time.perf_counter()
    accumulation = int(config["gradient_accumulation"])
    with g1.SampleStore("train") as store:
        lo, hi = store.subband_edges()
        matrix = g1.build_subband_fft_matrix(
            torch.from_numpy(lo),
            torch.from_numpy(hi),
            sample_rate_hz=g1.FS,
            n_fft=g1.N_FFT,
            dtype=torch.float64,
            device=device,
        )
        geometry = g1.receiver_geometry(device)
        for epoch in range(completed_epoch + 1, final_epoch + 1):
            g1.configure_trainable(ch3, d8, training=True)
            optimizer.zero_grad(set_to_none=True)
            component_sums: Counter[str] = Counter()
            step_diagnostics = []
            chapter_accum = [torch.zeros_like(parameter) for parameter in ch3_parameters]
            epoch_started = time.perf_counter()
            order = g1.sample_order(manifest["subsets"]["train"], seed, epoch)
            for ordinal, record in enumerate(order, start=1):
                sample = store.sample(record)
                logits = ch3(sample["coarse_dpd"].to(device)[None])
                total, components, _, _, _ = g1.losses(
                    logits, sample, matrix, geometry, d8, device, config, track
                )
                chapter_loss = components["band"] + components["exist"]
                chapter_grad = torch.autograd.grad(
                    chapter_loss / accumulation,
                    ch3_parameters,
                    retain_graph=True,
                    allow_unused=True,
                )
                for index, value in enumerate(chapter_grad):
                    if value is not None:
                        chapter_accum[index].add_(value.detach())
                (total / accumulation).backward()
                if not g1.finite(total):
                    raise FloatingPointError(f"恢复训练loss非有限: epoch={epoch}, sample={ordinal}")
                for name, value in components.items():
                    component_sums[name] += float(value.detach().item())
                if ordinal % accumulation == 0:
                    chapter_vector = g1.gradient_vector(ch3_parameters, chapter_accum)
                    total_vector = g1.gradient_vector(ch3_parameters)
                    localization_vector = total_vector - chapter_vector
                    diagnostics = {
                        "epoch": epoch,
                        "optimizer_step": ordinal // accumulation,
                        "chapter_grad_norm": float(torch.linalg.vector_norm(chapter_vector).item()),
                        "localization_grad_norm": float(torch.linalg.vector_norm(localization_vector).item()),
                        "total_ch3_grad_norm": float(torch.linalg.vector_norm(total_vector).item()),
                        "gradient_cosine": g1.cosine(chapter_vector, localization_vector),
                        "localization_nonzero": bool(torch.count_nonzero(localization_vector).item()),
                        "resumed_from_epoch": completed_epoch,
                    }
                    if not g1.finite(total_vector) or not g1.finite(g1.gradient_vector(d8_parameters)):
                        raise FloatingPointError("恢复训练梯度非有限")
                    diagnostics["ch3_clip_pre_norm"] = float(
                        torch.nn.utils.clip_grad_norm_(
                            ch3_parameters,
                            float(config["ch3_grad_clip"]),
                            error_if_nonfinite=True,
                        ).item()
                    )
                    diagnostics["d8_clip_pre_norm"] = float(
                        torch.nn.utils.clip_grad_norm_(
                            d8_parameters,
                            float(config["d8_grad_clip"]),
                            error_if_nonfinite=True,
                        ).item()
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    step_diagnostics.append(diagnostics)
                    g1.append_jsonl(track_root / "gradient_steps.jsonl", diagnostics)
                    chapter_accum = [torch.zeros_like(parameter) for parameter in ch3_parameters]
                if ordinal % 8 == 0 or ordinal == len(order):
                    print(f"[{track}:resume:epoch{epoch}] {ordinal}/{len(order)}", flush=True)
            g1.save_checkpoint(
                track_root / f"epoch_{epoch:03d}.pth",
                ch3,
                d8,
                optimizer,
                epoch,
                track,
                config,
            )
            validation = g1.evaluate_records(
                ch3,
                d8,
                manifest["subsets"]["val_select"],
                "val_select",
                device,
                config,
                progress_path=track_root / "validation_progress.jsonl",
            )
            validation.pop("samples")
            history.append(
                {
                    "epoch": epoch,
                    "training": {
                        "mean_losses": {
                            name: value / len(order) for name, value in component_sums.items()
                        },
                        "optimizer_steps": len(step_diagnostics),
                        "localization_nonzero_step_rate": float(
                            np.mean([row["localization_nonzero"] for row in step_diagnostics])
                        ),
                        "duration_seconds": time.perf_counter() - epoch_started,
                        "resumed_from_epoch": completed_epoch,
                    },
                    "validation": validation,
                }
            )
            g1.write_json(track_root / "history.json", history)
            print(
                f"[{track}:resume:epoch{epoch}] "
                f"val_gospa={validation['system']['gospa']['mean']:.6f}",
                flush=True,
            )

    initial_ch3 = history[0]["validation"]["ch3"]
    eligible = []
    for row in history:
        current = row["validation"]["ch3"]
        if (
            current["active_band_macro_f1"]
            >= initial_ch3["active_band_macro_f1"]
            - float(config["ch3_noninferiority_absolute"])
            and current["balanced_count_accuracy"]
            >= initial_ch3["balanced_count_accuracy"]
            - float(config["ch3_noninferiority_absolute"])
        ):
            eligible.append(row)
    if not eligible:
        raise RuntimeError("恢复训练后没有CH3非劣候选checkpoint")
    best = min(
        eligible,
        key=lambda row: (
            row["validation"]["system"]["gospa"]["mean"],
            -row["validation"]["system"]["exact_count_rate"],
            row["validation"]["system"]["gospa_components_mean_p_sum"]["missed"]
            + row["validation"]["system"]["gospa_components_mean_p_sum"]["false"],
            row["validation"]["system"].get("matched_errors_m", {}).get("p90", math.inf),
            int(row["epoch"]),
        ),
    )
    selected_path = track_root / f"epoch_{int(best['epoch']):03d}.pth"
    summary = {
        "status": "COMPLETED",
        "gate": "E2E-G1-P1",
        "track": track,
        "model_info": model_info,
        "epochs_completed": final_epoch,
        "sample_presentations": final_epoch * len(manifest["subsets"]["train"]),
        "optimizer_steps": final_epoch * len(manifest["subsets"]["train"]) // accumulation,
        "selected_epoch": int(best["epoch"]),
        "selected_checkpoint": str(selected_path.resolve()),
        "selected_checkpoint_sha256": g1.sha256_file(selected_path),
        "selection_rule": "CH3 noninferiority then lexicographic system GOSPA/count/cardinality-tail/epoch",
        "duration_seconds_after_resume": time.perf_counter() - started,
        "resume": {
            "completed_epoch": completed_epoch,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": g1.sha256_file(checkpoint_path),
            **partial_evidence,
        },
        "test_executed": False,
    }
    g1.write_json(track_root / "training_summary.json", summary)
    g1.write_json(track_root / "resume_report.json", summary["resume"])
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="恢复被外部中断的E2E-G1训练")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--track", choices=("soft_sg", "soft_e2e"), required=True)
    parser.add_argument("--completed-epoch", type=int, required=True)
    args = parser.parse_args()
    resume(args.run_root.resolve(), args.track, args.completed_epoch)


if __name__ == "__main__":
    main()
