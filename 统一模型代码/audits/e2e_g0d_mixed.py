"""E2E-G0D：FP64物理层到FP32冻结D8的完整loss确认。"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 统一模型代码.audits import e2e_g0 as g0  # noqa: E402
from 统一模型代码.audits import e2e_g0d as g0d  # noqa: E402
from 统一模型代码.audits.directional_ladder import (  # noqa: E402
    classify_derivatives,
    finite_difference_sweep,
)
from 统一模型代码.physics.fine_dpd_autograd import compute_fine_dpd_autograd  # noqa: E402
from 统一模型代码.runtime_paths import OUTPUT_ROOT, validate_output_path  # noqa: E402

SOURCE_FP64 = OUTPUT_ROOT / "smoke" / "unified" / "e2e_g0d" / "20260901_113322"
STEPS = (3e-2, 1e-2, 3e-3, 1e-3)


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有报告: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def prepare(run_id: str) -> Path:
    source_final = g0.load_json(SOURCE_FP64 / "final_report.json")
    g0.require(source_final["status"] == "PHYSICS_GRADIENT_PASS", "FP64物理层未通过")
    source_manifest = g0.load_json(SOURCE_FP64 / "manifest.json")
    run_root = validate_output_path(
        OUTPUT_ROOT / "smoke" / "unified" / "e2e_g0d" / run_id
    )
    if run_root.exists():
        raise FileExistsError(f"输出目录已存在: {run_root}")
    run_root.mkdir(parents=True)
    files = [
        Path(__file__).resolve(),
        (Path(__file__).parent / "directional_ladder.py").resolve(),
        (Path(__file__).parents[1] / "physics" / "fine_dpd_autograd.py").resolve(),
    ]
    manifest = {
        "status": "PREPARED",
        "gate": "E2E-G0D-MIXED",
        "run_id": run_id,
        "source_fp64": str(SOURCE_FP64.resolve()),
        "source_fp64_manifest_sha256": g0.sha256_file(SOURCE_FP64 / "manifest.json"),
        "source_g0": source_manifest["source_g0"],
        "backward_samples": source_manifest["backward_samples"],
        "direction_seeds": source_manifest["direction_seeds"],
        "steps": list(STEPS),
        "code": [g0.file_identity(path) for path in files],
        "training_executed": False,
        "test_executed": False,
    }
    write_json(run_root / "manifest.json", manifest)
    return run_root


def mixed_forward(
    context: g0d.DiagnosticContext,
    sample_index: int,
    point: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sample = context.samples[sample_index]
    bridge = g0.bridge_for_logits(point, context.matrix.to(torch.float64))
    dpd64 = compute_fine_dpd_autograd(
        sample["signal"],
        context.geometry(g0.GRID_SIZE),
        bridge.frequency_weights,
        fixed_support=context.fixed_support,
        grid_chunk_size=1024,
        frequency_chunk_size=128,
        eig_device="cpu",
        checkpoint_mode="reentrant",
        real_dtype=torch.float64,
    )
    normalized64 = g0.d8_input(dpd64)
    d8_input32 = normalized64.to(torch.float32)[None, None]
    heatmap, offset = context.d8(d8_input32)
    target, positions, counts = g0.gaussian_target(
        sample["positions_m"], sample["true_k"], context.device
    )
    focal = g0.focal_loss_hm(heatmap.float(), target)
    offset_loss = g0.compute_offset_loss(offset.float(), positions, counts, context.device)
    return focal + offset_loss, {
        "dpd64": dpd64,
        "normalized64": normalized64,
        "d8_input32": d8_input32,
        "heatmap": heatmap,
        "offset": offset,
        "focal": focal,
        "offset_loss": offset_loss,
    }


def fp32_baseline(
    context: g0d.DiagnosticContext,
    sample_index: int,
) -> tuple[float, dict[str, torch.Tensor]]:
    point = context.logits[sample_index].detach()
    bridge = g0.bridge_for_logits(point, context.matrix)
    dpd = compute_fine_dpd_autograd(
        context.samples[sample_index]["signal"],
        context.geometry(g0.GRID_SIZE),
        bridge.frequency_weights,
        fixed_support=context.fixed_support,
        grid_chunk_size=1024,
        frequency_chunk_size=128,
        eig_device="cpu",
        checkpoint_mode="off",
        real_dtype=torch.float32,
    )
    normalized = g0.d8_input(dpd)[None, None]
    heatmap, offset = context.d8(normalized)
    target, positions, counts = g0.gaussian_target(
        context.samples[sample_index]["positions_m"],
        context.samples[sample_index]["true_k"],
        context.device,
    )
    loss = g0.focal_loss_hm(heatmap.float(), target) + g0.compute_offset_loss(
        offset.float(), positions, counts, context.device
    )
    return float(loss.item()), {
        "normalized": normalized,
        "heatmap": heatmap,
        "offset": offset,
    }


def one_record(
    context: g0d.DiagnosticContext,
    sample_index: int,
    direction_seed: int,
) -> dict[str, Any]:
    point = context.logits[sample_index].detach().to(torch.float64).requires_grad_(True)
    direction = g0d._probe(point.shape, direction_seed, context.device).to(torch.float64)
    scalar, intermediates = mixed_forward(context, sample_index, point)
    scalar.backward()
    gradient = point.grad
    g0.require(gradient is not None, "混合精度完整loss未生成梯度")
    unit = direction / torch.linalg.vector_norm(direction)
    autograd_value = float(torch.sum(gradient * unit).item())

    def function(candidate: torch.Tensor) -> torch.Tensor:
        value, _ = mixed_forward(context, sample_index, candidate)
        return value

    sweep = finite_difference_sweep(function, point.detach(), direction, STEPS)
    classification = classify_derivatives(
        autograd_value,
        sweep,
        relative_tolerance=0.05,
        minimum_loss_span=1e-7,
    )
    return {
        "sample_local_index": sample_index,
        "true_k": int(context.samples[sample_index]["true_k"]),
        "direction_seed": direction_seed,
        "loss": float(scalar.detach().item()),
        "focal": float(intermediates["focal"].detach().item()),
        "offset_loss": float(intermediates["offset_loss"].detach().item()),
        "gradient": g0.tensor_summary(gradient),
        "classification": classification,
    }


def execute(run_root: Path) -> dict[str, Any]:
    manifest = g0.load_json(run_root / "manifest.json")
    for record in manifest["code"]:
        g0.require(
            g0.sha256_file(Path(record["path"])) == record["sha256"],
            f"代码身份已变化: {record['path']}",
        )
    context = g0d.DiagnosticContext(
        {"source_g0": manifest["source_g0"]}, torch.device("cuda:0")
    )
    records = []
    forward_comparisons = []
    for count in (1, 2, 3):
        sample_index = int(manifest["backward_samples"][str(count)])
        with torch.no_grad():
            fp32_loss, fp32 = fp32_baseline(context, sample_index)
            point64 = context.logits[sample_index].detach().to(torch.float64)
            mixed_loss, mixed = mixed_forward(context, sample_index, point64)
            forward_comparisons.append(
                {
                    "true_k": count,
                    "sample_local_index": sample_index,
                    "fp32_loss": fp32_loss,
                    "mixed_loss": float(mixed_loss.item()),
                    "loss_abs_difference": abs(fp32_loss - float(mixed_loss.item())),
                    "normalized_max_abs": float(
                        (fp32["normalized"] - mixed["d8_input32"]).abs().max().item()
                    ),
                    "heatmap_max_abs": float(
                        (fp32["heatmap"] - mixed["heatmap"]).abs().max().item()
                    ),
                    "offset_max_abs": float(
                        (fp32["offset"] - mixed["offset"]).abs().max().item()
                    ),
                }
            )
        for seed in manifest["direction_seeds"]:
            records.append(one_record(context, sample_index, int(seed)))
            torch.cuda.empty_cache()
    pass_count = sum(
        row["classification"]["status"] == "PASS_SMOOTH" for row in records
    )
    mismatch_count = sum(
        row["classification"]["status"] == "STABLE_GRADIENT_MISMATCH"
        for row in records
    )
    nonsmooth_count = sum(
        row["classification"]["status"] == "NONSMOOTH_LOCAL_POINT"
        for row in records
    )
    status = (
        "IMPLEMENTATION_FIX"
        if pass_count >= 12 and mismatch_count == 0
        else "PASS_WITH_NONSMOOTH_RISK"
        if mismatch_count == 0 and pass_count + nonsmooth_count >= 12
        else "INCONCLUSIVE_NUMERIC"
    )
    report = {
        "status": status,
        "gate": "E2E-G0D-MIXED",
        "precision_contract": "bridge_and_dpd_and_normalization_fp64_then_d8_fp32",
        "pass_count": pass_count,
        "mismatch_count": mismatch_count,
        "nonsmooth_count": nonsmooth_count,
        "forward_comparisons": forward_comparisons,
        "records": records,
        "g1_unlocked": False,
        "training_executed": False,
        "test_executed": False,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(run_root / "mixed_gradient.json", report)
    write_json(run_root / "final_report.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--run-id", required=True)
    run = sub.add_parser("execute")
    run.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        root = prepare(args.run_id)
        print(json.dumps({"status": "PREPARED", "run_root": str(root)}, ensure_ascii=False))
        return 0
    root = validate_output_path(args.run_root)
    try:
        result = execute(root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        failure = {
            "status": "EXECUTION_ERROR",
            "gate": "E2E-G0D-MIXED",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "g1_unlocked": False,
            "training_executed": False,
            "test_executed": False,
        }
        if not (root / "final_report.json").exists():
            write_json(root / "final_report.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
