"""E2E-G0D：定位FP64物理层之后的首个数值层，并执行全双精度gradcheck。"""

from __future__ import annotations

import argparse
import copy
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

SOURCE_MIXED = OUTPUT_ROOT / "smoke" / "unified" / "e2e_g0d" / "20260901_114150"
MIXED_STEPS = (3e-2, 1e-2, 3e-3, 1e-3)
DOUBLE_STEPS = (1e-3, 3e-4, 1e-4, 3e-5)
MIXED_TARGETS = (
    "L2_normalized_dpd",
    "L3_heatmap_logits",
    "L3_offset",
    "L4_focal",
    "L5_offset_loss",
    "L6_total_loss",
)


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有报告: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def prepare(run_id: str) -> Path:
    source_final = g0.load_json(SOURCE_MIXED / "final_report.json")
    g0.require(source_final["status"] == "INCONCLUSIVE_NUMERIC", "混合精度源结果不是未决")
    source_manifest = g0.load_json(SOURCE_MIXED / "manifest.json")
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
        "gate": "E2E-G0D-DOUBLE",
        "run_id": run_id,
        "source_mixed": str(SOURCE_MIXED.resolve()),
        "source_mixed_manifest_sha256": g0.sha256_file(SOURCE_MIXED / "manifest.json"),
        "source_g0": source_manifest["source_g0"],
        "backward_samples": source_manifest["backward_samples"],
        "direction_seeds": source_manifest["direction_seeds"],
        "mixed_steps": list(MIXED_STEPS),
        "double_steps": list(DOUBLE_STEPS),
        "code": [g0.file_identity(path) for path in files],
        "training_executed": False,
        "test_executed": False,
    }
    write_json(run_root / "manifest.json", manifest)
    return run_root


def physical_front(
    context: g0d.DiagnosticContext,
    sample_index: int,
    point: torch.Tensor,
) -> torch.Tensor:
    bridge = g0.bridge_for_logits(point, context.matrix.to(torch.float64))
    dpd = compute_fine_dpd_autograd(
        context.samples[sample_index]["signal"],
        context.geometry(g0.GRID_SIZE),
        bridge.frequency_weights,
        fixed_support=context.fixed_support,
        grid_chunk_size=1024,
        frequency_chunk_size=128,
        eig_device="cpu",
        checkpoint_mode="reentrant",
        real_dtype=torch.float64,
    )
    return g0.d8_input(dpd)


def mixed_scalar(
    context: g0d.DiagnosticContext,
    sample_index: int,
    point: torch.Tensor,
    target_name: str,
) -> torch.Tensor:
    normalized64 = physical_front(context, sample_index, point)
    if target_name == "L2_normalized_dpd":
        probe = g0d._probe(normalized64.shape, 7202, context.device).to(torch.float64)
        return torch.sum(normalized64 * probe)
    heatmap, offset = context.d8(normalized64.to(torch.float32)[None, None])
    if target_name == "L3_heatmap_logits":
        probe = g0d._probe(heatmap.shape, 7203, context.device)
        return torch.sum(heatmap * probe)
    if target_name == "L3_offset":
        probe = g0d._probe(offset.shape, 7204, context.device)
        return torch.sum(offset * probe)
    sample = context.samples[sample_index]
    target, positions, counts = g0.gaussian_target(
        sample["positions_m"], sample["true_k"], context.device
    )
    focal = g0.focal_loss_hm(heatmap, target)
    offset_loss = g0.compute_offset_loss(offset, positions, counts, context.device)
    if target_name == "L4_focal":
        return focal
    if target_name == "L5_offset_loss":
        return offset_loss
    if target_name == "L6_total_loss":
        return focal + offset_loss
    raise ValueError(target_name)


def focal_double(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(prediction).clamp(1e-8, 1.0 - 1e-8)
    target64 = target.to(torch.float64)
    positive = -(target64 * (1.0 - probability).square() * torch.log(probability))
    negative = -(
        (1.0 - target64).pow(4) * probability.square() * torch.log(1.0 - probability)
    )
    count = target64.gt(0.5).to(torch.float64).sum()
    return (positive.sum() + negative.sum()) / count.clamp_min(1.0)


def offset_double(
    prediction: torch.Tensor,
    positions: torch.Tensor,
    counts: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    channels = positions.shape[1]
    pos = positions.to(device=device, dtype=torch.float64)
    px = (pos[:, :, 0] * g0.FINE_EDGE + g0.FINE_EDGE) / g0.FINE_STEP
    py = (pos[:, :, 1] * g0.FINE_EDGE + g0.FINE_EDGE) / g0.FINE_STEP
    ix = px.round().long().clamp(0, width - 1)
    iy = py.round().long().clamp(0, height - 1)
    dx = px - ix.to(torch.float64)
    dy = py - iy.to(torch.float64)
    flat = iy * width + ix
    values = prediction.reshape(batch, 2, -1)
    dx_pred = values[:, 0].gather(1, flat)
    dy_pred = values[:, 1].gather(1, flat)
    mask = (
        torch.arange(channels, device=device).unsqueeze(0)
        < counts.to(device).unsqueeze(1)
    ).to(torch.float64)
    return ((dx_pred - dx).abs() + (dy_pred - dy).abs()).mul(mask).sum() / mask.sum().clamp_min(1.0)


def double_scalar(
    context: g0d.DiagnosticContext,
    d8_double: torch.nn.Module,
    sample_index: int,
    point: torch.Tensor,
) -> torch.Tensor:
    normalized64 = physical_front(context, sample_index, point)
    heatmap, offset = d8_double(normalized64[None, None])
    sample = context.samples[sample_index]
    target, positions, counts = g0.gaussian_target(
        sample["positions_m"], sample["true_k"], context.device
    )
    return focal_double(heatmap, target) + offset_double(
        offset, positions, counts, context.device
    )


def derivative_record(
    function: Any,
    point: torch.Tensor,
    direction: torch.Tensor,
    steps: tuple[float, ...],
    minimum_span: float,
) -> dict[str, Any]:
    scalar = function(point)
    scalar.backward()
    gradient = point.grad
    g0.require(gradient is not None, "目标未生成梯度")
    unit = direction / torch.linalg.vector_norm(direction)
    autograd_value = float(torch.sum(gradient * unit).item())
    sweep = finite_difference_sweep(function, point.detach(), direction, steps)
    return {
        "scalar": float(scalar.detach().item()),
        "gradient": g0.tensor_summary(gradient),
        "classification": classify_derivatives(
            autograd_value,
            sweep,
            relative_tolerance=0.05,
            minimum_loss_span=minimum_span,
        ),
    }


def execute(run_root: Path) -> dict[str, Any]:
    manifest = g0.load_json(run_root / "manifest.json")
    for record in manifest["code"]:
        g0.require(g0.sha256_file(Path(record["path"])) == record["sha256"], "代码身份已变化")
    device = torch.device("cuda:0")
    context = g0d.DiagnosticContext({"source_g0": manifest["source_g0"]}, device)
    primary = int(manifest["backward_samples"]["2"])
    primary_direction = g0d._probe(
        context.logits[primary].shape, int(manifest["direction_seeds"][0]), device
    ).to(torch.float64)
    mixed_ladder = []
    for target_name in MIXED_TARGETS:
        point = context.logits[primary].detach().to(torch.float64).requires_grad_(True)
        function = lambda value, name=target_name: mixed_scalar(
            context, primary, value, name
        )
        mixed_ladder.append(
            {
                "target": target_name,
                **derivative_record(function, point, primary_direction, MIXED_STEPS, 1e-7),
            }
        )
        torch.cuda.empty_cache()
    first_mixed_failure = next(
        (
            row["target"]
            for row in mixed_ladder
            if row["classification"]["status"] != "PASS_SMOOTH"
        ),
        None,
    )
    write_json(
        run_root / "mixed_layer_ladder.json",
        {"first_failure": first_mixed_failure, "records": mixed_ladder},
    )

    d8_double = copy.deepcopy(context.d8).to(dtype=torch.float64).eval()
    double_records = []
    cases = [(2, int(seed)) for seed in manifest["direction_seeds"]]
    cases.extend([(1, int(manifest["direction_seeds"][0])), (3, int(manifest["direction_seeds"][0]))])
    for count, seed in cases:
        sample_index = int(manifest["backward_samples"][str(count)])
        point = context.logits[sample_index].detach().to(torch.float64).requires_grad_(True)
        direction = g0d._probe(point.shape, seed, device).to(torch.float64)
        function = lambda value, index=sample_index: double_scalar(
            context, d8_double, index, value
        )
        double_records.append(
            {
                "true_k": count,
                "sample_local_index": sample_index,
                "direction_seed": seed,
                **derivative_record(function, point, direction, DOUBLE_STEPS, 1e-12),
            }
        )
        torch.cuda.empty_cache()
    double_pass = sum(
        row["classification"]["status"] == "PASS_SMOOTH" for row in double_records
    )
    double_mismatch = sum(
        row["classification"]["status"] == "STABLE_GRADIENT_MISMATCH"
        for row in double_records
    )
    if first_mixed_failure == "L3_heatmap_logits" and double_pass >= 6 and double_mismatch == 0:
        status = "IMPLEMENTATION_FIX"
        cause = "FP32_D8_FINITE_DIFFERENCE_RESOLUTION"
    elif double_mismatch >= 2:
        status = "FEASIBLE_WITH_LOCAL_REDESIGN"
        cause = "D8_OR_LOSS_NONSMOOTHNESS"
    else:
        status = "INCONCLUSIVE_NUMERIC"
        cause = "UNRESOLVED_DOWNSTREAM_NUMERIC"
    report = {
        "status": status,
        "gate": "E2E-G0D-DOUBLE",
        "cause": cause,
        "first_mixed_failure": first_mixed_failure,
        "double_pass_count": double_pass,
        "double_mismatch_count": double_mismatch,
        "double_records": double_records,
        "g1_unlocked": False,
        "training_executed": False,
        "test_executed": False,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(run_root / "double_gradient.json", report)
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
            "gate": "E2E-G0D-DOUBLE",
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
