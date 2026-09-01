"""E2E-G0D的FP64原始DPD局部梯度确认。"""

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

SOURCE_G0D = OUTPUT_ROOT / "smoke" / "unified" / "e2e_g0d" / "20260901_112416"
STEPS = (3e-2, 1e-2, 3e-3, 1e-3)


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有报告: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def prepare(run_id: str) -> Path:
    source_final = g0.load_json(SOURCE_G0D / "final_report.json")
    g0.require(source_final["status"] == "INCONCLUSIVE_NUMERIC", "源G0D不是数值未决")
    run_root = validate_output_path(
        OUTPUT_ROOT / "smoke" / "unified" / "e2e_g0d" / run_id
    )
    if run_root.exists():
        raise FileExistsError(f"输出目录已存在: {run_root}")
    run_root.mkdir(parents=True)
    files = [
        Path(__file__).resolve(),
        (Path(__file__).parent / "directional_ladder.py").resolve(),
        (Path(__file__).parent / "e2e_g0d.py").resolve(),
        (Path(__file__).parents[1] / "physics" / "fine_dpd_autograd.py").resolve(),
    ]
    source_manifest = g0.load_json(SOURCE_G0D / "manifest.json")
    manifest = {
        "status": "PREPARED",
        "gate": "E2E-G0D-FP64",
        "run_id": run_id,
        "source_g0d": str(SOURCE_G0D.resolve()),
        "source_g0d_manifest_sha256": g0.sha256_file(SOURCE_G0D / "manifest.json"),
        "source_g0": source_manifest["source_g0"],
        "backward_samples": source_manifest["backward_samples"],
        "direction_seeds": source_manifest["confirmation_direction_seeds"],
        "steps": list(STEPS),
        "code": [g0.file_identity(path) for path in files],
        "training_executed": False,
        "test_executed": False,
    }
    write_json(run_root / "manifest.json", manifest)
    return run_root


def scalar_l1(
    context: g0d.DiagnosticContext,
    sample_index: int,
    point: torch.Tensor,
) -> torch.Tensor:
    bridge = g0.bridge_for_logits(point, context.matrix.to(torch.float64))
    geometry = context.geometry(g0.GRID_SIZE)
    dpd = compute_fine_dpd_autograd(
        context.samples[sample_index]["signal"],
        geometry,
        bridge.frequency_weights,
        fixed_support=context.fixed_support,
        grid_chunk_size=1024,
        frequency_chunk_size=128,
        eig_device="cpu",
        checkpoint_mode="reentrant",
        real_dtype=torch.float64,
    )
    probe = g0d._probe(dpd.shape, 7101, context.device).to(torch.float64)
    return torch.sum(dpd * probe)


def one_record(
    context: g0d.DiagnosticContext,
    sample_index: int,
    direction_seed: int,
) -> dict[str, Any]:
    point = context.logits[sample_index].detach().to(torch.float64).requires_grad_(True)
    direction = g0d._probe(point.shape, direction_seed, context.device).to(torch.float64)
    scalar = scalar_l1(context, sample_index, point)
    scalar.backward()
    gradient = point.grad
    g0.require(gradient is not None, "FP64 L1未生成梯度")
    unit = direction / torch.linalg.vector_norm(direction)
    autograd_value = float(torch.sum(gradient * unit).item())

    def function(candidate: torch.Tensor) -> torch.Tensor:
        return scalar_l1(context, sample_index, candidate)

    sweep = finite_difference_sweep(function, point.detach(), direction, STEPS)
    classification = classify_derivatives(
        autograd_value,
        sweep,
        relative_tolerance=0.05,
        minimum_loss_span=1e-10,
    )
    return {
        "sample_local_index": sample_index,
        "true_k": int(context.samples[sample_index]["true_k"]),
        "direction_seed": direction_seed,
        "scalar": float(scalar.detach().item()),
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
    device = torch.device("cuda:0")
    source_g0_manifest = g0.load_json(Path(manifest["source_g0"]) / "manifest.json")
    context = g0d.DiagnosticContext(
        {"source_g0": manifest["source_g0"]}, device
    )
    records = []
    for count in (1, 2, 3):
        sample_index = int(source_g0_manifest["backward_samples"][str(count)])
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
    per_k = {
        str(count): {
            "pass": sum(
                row["classification"]["status"] == "PASS_SMOOTH"
                for row in records
                if row["true_k"] == count
            ),
            "mismatch": sum(
                row["classification"]["status"] == "STABLE_GRADIENT_MISMATCH"
                for row in records
                if row["true_k"] == count
            ),
        }
        for count in (1, 2, 3)
    }
    status = (
        "PHYSICS_GRADIENT_PASS"
        if pass_count >= 12 and mismatch_count == 0
        else "PHYSICS_GRADIENT_BLOCKED"
        if mismatch_count >= 6
        else "INCONCLUSIVE_NUMERIC"
    )
    report = {
        "status": status,
        "gate": "E2E-G0D-FP64",
        "dtype": "torch.float64/torch.complex128",
        "pass_count": pass_count,
        "mismatch_count": mismatch_count,
        "per_k": per_k,
        "records": records,
        "g1_unlocked": False,
        "training_executed": False,
        "test_executed": False,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(run_root / "fp64_gradient.json", report)
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
            "gate": "E2E-G0D-FP64",
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
