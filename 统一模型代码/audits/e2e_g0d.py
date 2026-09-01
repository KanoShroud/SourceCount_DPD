"""E2E-G0D：定位完整系统方向梯度不一致的首个来源。"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 统一模型代码.audits import e2e_g0 as g0  # noqa: E402
from 统一模型代码.audits.directional_ladder import (  # noqa: E402
    classify_derivatives,
    finite_difference_sweep,
    gradient_comparison,
)
from 统一模型代码.physics.band_bridge import build_subband_fft_matrix  # noqa: E402
from 统一模型代码.physics.fine_dpd_autograd import compute_fine_dpd_autograd  # noqa: E402
from 统一模型代码.runtime_paths import OUTPUT_ROOT, validate_output_path  # noqa: E402

GATE = "E2E-G0D"
SOURCE_G0 = OUTPUT_ROOT / "smoke" / "unified" / "e2e_g0" / "20260831_235442"
STEPS = (3e-2, 1e-2, 3e-3, 1e-3)
TARGETS = (
    "L0_frequency_weights",
    "L1_raw_dpd",
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
    g0.validate_split_roots()
    source_manifest = g0.load_json(SOURCE_G0 / "manifest.json")
    source_final = g0.load_json(SOURCE_G0 / "final_report.json")
    g0.require(source_final["status"] == "STOP_GRADIENT", "源G0不是STOP_GRADIENT")
    run_root = validate_output_path(
        OUTPUT_ROOT / "smoke" / "unified" / "e2e_g0d" / run_id
    )
    if run_root.exists():
        raise FileExistsError(f"输出目录已存在: {run_root}")
    run_root.mkdir(parents=True)
    code_files = [
        Path(__file__).resolve(),
        (Path(__file__).parent / "directional_ladder.py").resolve(),
        (Path(__file__).parent / "eigen_gap_probe.py").resolve(),
        (PACKAGE_ROOT / "physics" / "fine_dpd_autograd.py").resolve(),
        (PACKAGE_ROOT / "physics" / "band_bridge.py").resolve(),
    ]
    manifest = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "source_type": "frozen_validation_and_checkpoints",
            "read_status": "PREPARED",
            "verification_status": "UNVERIFIED",
        },
        "status": "PREPARED",
        "gate": GATE,
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_g0": str(SOURCE_G0.resolve()),
        "source_g0_status": source_final["status"],
        "source_g0_manifest_sha256": g0.sha256_file(SOURCE_G0 / "manifest.json"),
        "source_inputs": source_manifest["inputs"],
        "source_pilot_samples": source_manifest["pilot_samples"],
        "backward_samples": source_manifest["backward_samples"],
        "primary_sample_local_index": int(source_manifest["backward_samples"]["2"]),
        "direction_seed": g0.DIRECTION_SEED + 2,
        "confirmation_direction_seeds": [g0.DIRECTION_SEED + 2 + i for i in range(5)],
        "steps": list(STEPS),
        "targets": list(TARGETS),
        "thresholds": {
            "relative_tolerance": 0.05,
            "minimum_loss_span": 1e-6,
            "checkpoint_cosine_min": 0.9999,
            "checkpoint_relative_l2_max": 1e-3,
        },
        "code": [g0.file_identity(path) for path in code_files],
        "training_executed": False,
        "test_executed": False,
    }
    write_json(run_root / "manifest.json", manifest)
    return run_root


def verify_manifest(run_root: Path) -> dict[str, Any]:
    manifest = g0.load_json(run_root / "manifest.json")
    g0.require(manifest["status"] == "PREPARED", "G0D manifest状态错误")
    g0.require(
        g0.sha256_file(SOURCE_G0 / "manifest.json")
        == manifest["source_g0_manifest_sha256"],
        "源G0 manifest已变化",
    )
    for record in manifest["code"]:
        path = Path(record["path"])
        g0.require(g0.sha256_file(path) == record["sha256"], f"代码身份已变化: {path}")
    return manifest


class DiagnosticContext:
    def __init__(self, manifest: dict[str, Any], device: torch.device):
        source_manifest = g0.load_json(Path(manifest["source_g0"]) / "manifest.json")
        self.source_manifest = source_manifest
        self.samples = g0.load_selected_arrays(source_manifest)
        self.ch3, self.d8, self.models = g0.build_models(device)
        self.logits = g0.infer_ch3(self.ch3, self.samples, source_manifest, device)
        sub_lo = torch.from_numpy(self.samples[-1]["sub_lo_hz"]).to(device)
        sub_hi = torch.from_numpy(self.samples[-1]["sub_hi_hz"]).to(device)
        self.matrix = build_subband_fft_matrix(
            sub_lo,
            sub_hi,
            sample_rate_hz=g0.FS,
            n_fft=g0.N_FFT,
            dtype=torch.float32,
            device=device,
        )
        self.fixed_support = self.matrix.bool().any(dim=1)
        self.device = device

    def geometry(self, size: int = g0.GRID_SIZE) -> Any:
        edge = (size - 1) * g0.FINE_STEP / 2.0
        return g0.receiver_geometry(self.device, edge=edge, step=g0.FINE_STEP)


def _probe(shape: torch.Size, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    value = torch.randn(shape, generator=generator, device=device)
    return value / torch.linalg.vector_norm(value)


def scalar_forward(
    context: DiagnosticContext,
    *,
    sample_index: int,
    point: torch.Tensor,
    target: str,
    checkpoint_mode: str,
    grid_size: int = g0.GRID_SIZE,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    sample = context.samples[sample_index]
    bridge = g0.bridge_for_logits(point, context.matrix)
    intermediates: dict[str, torch.Tensor] = {
        "frequency_weights": bridge.frequency_weights
    }
    if target == "L0_frequency_weights":
        probe = _probe(bridge.frequency_weights.shape, 7100, context.device)
        return torch.sum(bridge.frequency_weights * probe), intermediates
    geometry = context.geometry(grid_size)
    dpd = compute_fine_dpd_autograd(
        sample["signal"],
        geometry,
        bridge.frequency_weights,
        fixed_support=context.fixed_support,
        grid_chunk_size=1024,
        frequency_chunk_size=128,
        eig_device="cpu",
        checkpoint_mode=checkpoint_mode,
    )
    intermediates["dpd"] = dpd
    if target == "L1_raw_dpd":
        probe = _probe(dpd.shape, 7101, context.device)
        return torch.sum(dpd * probe), intermediates
    normalized = g0.d8_input(dpd)
    intermediates["normalized"] = normalized
    if target == "L2_normalized_dpd":
        probe = _probe(normalized.shape, 7102, context.device)
        return torch.sum(normalized * probe), intermediates
    if grid_size != g0.GRID_SIZE:
        raise ValueError("D8目标只允许401×401网格")
    heatmap, offset = context.d8(normalized[None, None])
    intermediates["heatmap"] = heatmap
    intermediates["offset"] = offset
    if target == "L3_heatmap_logits":
        probe = _probe(heatmap.shape, 7103, context.device)
        return torch.sum(heatmap * probe), intermediates
    if target == "L3_offset":
        probe = _probe(offset.shape, 7104, context.device)
        return torch.sum(offset * probe), intermediates
    target_map, positions, counts = g0.gaussian_target(
        sample["positions_m"], sample["true_k"], context.device
    )
    focal = g0.focal_loss_hm(heatmap.float(), target_map)
    offset_loss = g0.compute_offset_loss(offset.float(), positions, counts, context.device)
    intermediates["focal"] = focal
    intermediates["offset_loss"] = offset_loss
    if target == "L4_focal":
        return focal, intermediates
    if target == "L5_offset_loss":
        return offset_loss, intermediates
    if target == "L6_total_loss":
        return focal + offset_loss, intermediates
    raise ValueError(f"未知目标: {target}")


def autograd_record(
    context: DiagnosticContext,
    *,
    sample_index: int,
    target: str,
    direction: torch.Tensor,
    checkpoint_mode: str,
    grid_size: int = g0.GRID_SIZE,
) -> tuple[dict[str, Any], torch.Tensor]:
    point = context.logits[sample_index].detach().clone().requires_grad_(True)
    started = time.perf_counter()
    scalar, intermediates = scalar_forward(
        context,
        sample_index=sample_index,
        point=point,
        target=target,
        checkpoint_mode=checkpoint_mode,
        grid_size=grid_size,
    )
    scalar.backward()
    gradient = point.grad
    g0.require(gradient is not None, "诊断标量未生成logits梯度")
    unit = direction / torch.linalg.vector_norm(direction)
    record = {
        "scalar": float(scalar.detach().item()),
        "autograd_directional_derivative": float(torch.sum(gradient * unit).item()),
        "gradient": g0.tensor_summary(gradient),
        "duration_seconds": time.perf_counter() - started,
        "intermediates": {
            key: g0.tensor_summary(value) for key, value in intermediates.items()
        },
    }
    return record, gradient.detach()


def directional_record(
    context: DiagnosticContext,
    *,
    sample_index: int,
    target: str,
    direction: torch.Tensor,
    checkpoint_mode: str = "reentrant",
    grid_size: int = g0.GRID_SIZE,
    steps: tuple[float, ...] = STEPS,
) -> dict[str, Any]:
    autograd, gradient = autograd_record(
        context,
        sample_index=sample_index,
        target=target,
        direction=direction,
        checkpoint_mode=checkpoint_mode,
        grid_size=grid_size,
    )
    point = context.logits[sample_index].detach().clone()

    def function(candidate: torch.Tensor) -> torch.Tensor:
        scalar, _ = scalar_forward(
            context,
            sample_index=sample_index,
            point=candidate,
            target=target,
            checkpoint_mode="off",
            grid_size=grid_size,
        )
        return scalar

    sweep = finite_difference_sweep(function, point, direction, steps)
    classification = classify_derivatives(
        autograd["autograd_directional_derivative"], sweep
    )
    return {
        "sample_local_index": sample_index,
        "target": target,
        "checkpoint_mode": checkpoint_mode,
        "grid_size": grid_size,
        "autograd": autograd,
        "classification": classification,
        "gradient_tensor": gradient.cpu(),
    }


def _json_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "gradient_tensor"}


def checkpoint_diagnostic(
    context: DiagnosticContext, sample_index: int, direction: torch.Tensor
) -> dict[str, Any]:
    full = {}
    gradients = {}
    for mode in ("reentrant", "nonreentrant"):
        record, gradient = autograd_record(
            context,
            sample_index=sample_index,
            target="L6_total_loss",
            direction=direction,
            checkpoint_mode=mode,
        )
        full[mode] = record
        gradients[mode] = gradient
        gc.collect()
        torch.cuda.empty_cache()
    small = {}
    small_gradients = {}
    for mode in ("off", "reentrant", "nonreentrant"):
        record, gradient = autograd_record(
            context,
            sample_index=sample_index,
            target="L1_raw_dpd",
            direction=direction,
            checkpoint_mode=mode,
            grid_size=21,
        )
        small[mode] = record
        small_gradients[mode] = gradient
    full_comparison = gradient_comparison(
        gradients["reentrant"], gradients["nonreentrant"]
    )
    small_comparisons = {
        mode: gradient_comparison(small_gradients["off"], small_gradients[mode])
        for mode in ("reentrant", "nonreentrant")
    }
    return {
        "status": "PASS"
        if full_comparison["pass"]
        and all(row["pass"] for row in small_comparisons.values())
        else "FAIL",
        "full_401_total_loss": full,
        "full_reentrant_vs_nonreentrant": full_comparison,
        "small_21_raw_dpd": small,
        "small_off_comparisons": small_comparisons,
        "off_401_omitted": "避免无checkpoint保存完整401×401相位图导致非必要显存风险",
    }


def execute(run_root: Path) -> dict[str, Any]:
    manifest = verify_manifest(run_root)
    g0.require(torch.cuda.is_available(), "G0D需要CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.manual_seed(g0.SELECTION_SEED)
    np.random.seed(g0.SELECTION_SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    context = DiagnosticContext(manifest, device)
    sample_index = int(manifest["primary_sample_local_index"])
    direction = _probe(
        context.logits[sample_index].shape,
        int(manifest["direction_seed"]),
        device,
    )

    reproduction = directional_record(
        context,
        sample_index=sample_index,
        target="L6_total_loss",
        direction=direction,
    )
    reproduction_json = _json_record(reproduction)
    reproduction_json["reproduced_original_mismatch"] = (
        reproduction["classification"]["status"] == "STABLE_GRADIENT_MISMATCH"
    )
    write_json(run_root / "failure_reproduction.json", reproduction_json)
    g0.require(reproduction_json["reproduced_original_mismatch"], "未复现原G0梯度异号")

    checkpoint = checkpoint_diagnostic(context, sample_index, direction)
    write_json(run_root / "checkpoint_consistency.json", checkpoint)
    if checkpoint["status"] != "PASS":
        final = {
            "status": "IMPLEMENTATION_FIX",
            "gate": GATE,
            "first_failure": "checkpoint_consistency",
            "g1_unlocked": False,
            "training_executed": False,
            "test_executed": False,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        write_json(run_root / "final_report.json", final)
        return final

    ladder = []
    raw_records: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        if target == "L6_total_loss":
            record = reproduction
        else:
            record = directional_record(
                context,
                sample_index=sample_index,
                target=target,
                direction=direction,
            )
        raw_records[target] = record
        ladder.append(_json_record(record))
        gc.collect()
        torch.cuda.empty_cache()
    first_failure = next(
        (
            target
            for target in TARGETS
            if raw_records[target]["classification"]["status"] != "PASS_SMOOTH"
        ),
        None,
    )
    ladder_report = {"status": "LOCATED", "first_failure": first_failure, "targets": ladder}
    write_json(run_root / "layer_ladder.json", ladder_report)

    if first_failure is None:
        final_status = "INCONCLUSIVE_NUMERIC"
        predecessor = None
    else:
        failure_index = TARGETS.index(first_failure)
        predecessor = TARGETS[failure_index - 1] if failure_index > 0 else None
        first_class = raw_records[first_failure]["classification"]["status"]
        if first_class == "NONSMOOTH_LOCAL_POINT":
            final_status = "PASS_WITH_NONSMOOTH_RISK"
        elif first_failure == "L0_frequency_weights":
            final_status = "IMPLEMENTATION_FIX"
        elif first_failure == "L1_raw_dpd":
            final_status = "PHYSICS_GRADIENT_BLOCKED"
        else:
            final_status = "FEASIBLE_WITH_LOCAL_REDESIGN"

    confirmation_targets = [target for target in (predecessor, first_failure) if target]
    multi_direction = []
    for seed in manifest["confirmation_direction_seeds"]:
        current_direction = _probe(context.logits[sample_index].shape, int(seed), device)
        for target in confirmation_targets:
            record = directional_record(
                context,
                sample_index=sample_index,
                target=target,
                direction=current_direction,
                steps=(3e-2, 1e-2, 3e-3),
            )
            multi_direction.append({"direction_seed": seed, **_json_record(record)})
            gc.collect()
            torch.cuda.empty_cache()
    write_json(
        run_root / "multi_direction.json",
        {"targets": confirmation_targets, "records": multi_direction},
    )

    cross_sample = []
    if first_failure is not None:
        for count in (1, 2, 3):
            current_index = int(manifest["backward_samples"][str(count)])
            current_direction = _probe(
                context.logits[current_index].shape,
                int(manifest["direction_seed"]),
                device,
            )
            for target in confirmation_targets:
                record = directional_record(
                    context,
                    sample_index=current_index,
                    target=target,
                    direction=current_direction,
                    steps=(3e-2, 1e-2, 3e-3),
                )
                cross_sample.append({"true_k": count, **_json_record(record)})
                gc.collect()
                torch.cuda.empty_cache()
    write_json(
        run_root / "cross_sample.json",
        {"targets": confirmation_targets, "records": cross_sample},
    )

    failure_confirmations = [
        row
        for row in multi_direction
        if row["target"] == first_failure
        and row["classification"]["status"] == "STABLE_GRADIENT_MISMATCH"
    ]
    cross_failures = [
        row
        for row in cross_sample
        if row["target"] == first_failure
        and row["classification"]["status"] == "STABLE_GRADIENT_MISMATCH"
    ]
    if first_failure == "L1_raw_dpd" and (
        len(failure_confirmations) < 2 or len(cross_failures) < 2
    ):
        final_status = "INCONCLUSIVE_NUMERIC"
    final = {
        "material_passport": {
            **manifest["material_passport"],
            "read_status": "COMPLETE_EXECUTION",
            "verification_status": "ANALYZED",
        },
        "status": final_status,
        "gate": GATE,
        "source_g0_status": manifest["source_g0_status"],
        "failure_reproduced": True,
        "checkpoint_status": checkpoint["status"],
        "first_failure": first_failure,
        "predecessor": predecessor,
        "multi_direction_stable_mismatch_count": len(failure_confirmations),
        "cross_sample_stable_mismatch_count": len(cross_failures),
        "g1_unlocked": False,
        "training_executed": False,
        "test_executed": False,
        "algorithm_modified": False,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(run_root / "final_report.json", final)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        run_root = prepare(args.run_id)
        print(json.dumps({"status": "PREPARED", "run_root": str(run_root)}, ensure_ascii=False))
        return 0
    run_root = validate_output_path(args.run_root)
    try:
        result = execute(run_root)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except BaseException as error:
        failure = {
            "material_passport": {
                "schema": "ARS-9-compatible-local",
                "source_type": "frozen_validation_and_checkpoints",
                "read_status": "PARTIAL_EXECUTION",
                "verification_status": "ANALYZED",
            },
            "status": "EXECUTION_ERROR",
            "gate": GATE,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "g1_unlocked": False,
            "training_executed": False,
            "test_executed": False,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        if not (run_root / "final_report.json").exists():
            write_json(run_root / "final_report.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
