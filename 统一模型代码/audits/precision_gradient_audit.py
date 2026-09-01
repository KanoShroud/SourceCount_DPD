"""E2E-G0-R1的FP64物理链与full-double完整链梯度审计。"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import torch

from 统一模型代码.audits import e2e_g0 as g0
from 统一模型代码.audits import e2e_g0d as g0d
from 统一模型代码.audits import e2e_g0d_double as double_audit
from 统一模型代码.audits import e2e_g0d_fp64 as fp64_audit


DIRECTION_SEEDS = (2026083103, 2026083104, 2026083105, 2026083106, 2026083107)


@dataclass
class AuditContext:
    samples: dict[int, dict[str, Any]]
    logits: dict[int, torch.Tensor]
    matrix: torch.Tensor
    fixed_support: torch.Tensor
    d8: torch.nn.Module
    device: torch.device

    def geometry(self, size: int = g0.GRID_SIZE) -> Any:
        step = 2.0 * g0.FINE_EDGE / (size - 1)
        return g0.receiver_geometry(self.device, edge=g0.FINE_EDGE, step=step)


def run_layered_gradient_audit(
    *,
    samples: dict[int, dict[str, Any]],
    manifest: dict[str, Any],
    logits_by_sample: dict[int, torch.Tensor],
    matrix: torch.Tensor,
    fixed_support: torch.Tensor,
    d8: torch.nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """执行预注册的15个物理方向与7个完整double方向。"""
    started = time.perf_counter()
    context = AuditContext(
        samples=samples,
        logits=logits_by_sample,
        matrix=matrix,
        fixed_support=fixed_support,
        d8=d8,
        device=device,
    )

    physical_records = []
    for count in (1, 2, 3):
        sample_index = int(manifest["backward_samples"][str(count)])
        for seed in DIRECTION_SEEDS:
            physical_records.append(fp64_audit.one_record(context, sample_index, seed))
            if device.type == "cuda":
                torch.cuda.empty_cache()
    physical_pass = sum(
        row["classification"]["status"] == "PASS_SMOOTH"
        for row in physical_records
    )
    physical_mismatch = sum(
        row["classification"]["status"] == "STABLE_GRADIENT_MISMATCH"
        for row in physical_records
    )

    d8_double = copy.deepcopy(d8).to(dtype=torch.float64).eval()
    double_records = []
    cases = [(2, seed) for seed in DIRECTION_SEEDS]
    cases.extend([(1, DIRECTION_SEEDS[0]), (3, DIRECTION_SEEDS[0])])
    for count, seed in cases:
        sample_index = int(manifest["backward_samples"][str(count)])
        point = (
            logits_by_sample[sample_index]
            .detach()
            .to(torch.float64)
            .requires_grad_(True)
        )
        direction = g0d._probe(point.shape, seed, device).to(torch.float64)
        function = lambda value, index=sample_index: double_audit.double_scalar(
            context, d8_double, index, value
        )
        double_records.append(
            {
                "true_k": count,
                "sample_local_index": sample_index,
                "direction_seed": seed,
                **double_audit.derivative_record(
                    function,
                    point,
                    direction,
                    double_audit.DOUBLE_STEPS,
                    1e-12,
                ),
            }
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
    double_pass = sum(
        row["classification"]["status"] == "PASS_SMOOTH"
        for row in double_records
    )
    double_mismatch = sum(
        row["classification"]["status"] == "STABLE_GRADIENT_MISMATCH"
        for row in double_records
    )

    passed = (
        physical_pass >= 12
        and physical_mismatch == 0
        and double_pass >= 6
        and double_mismatch == 0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "contract": "fp64_physics_multidirection_plus_full_double_spotcheck",
        "direction_seeds": list(DIRECTION_SEEDS),
        "physical": {
            "dtype": "float64/complex128",
            "required_pass": 12,
            "pass_count": physical_pass,
            "mismatch_count": physical_mismatch,
            "records": physical_records,
        },
        "full_double": {
            "dtype": "float64",
            "diagnostic_only": True,
            "required_pass": 6,
            "pass_count": double_pass,
            "mismatch_count": double_mismatch,
            "records": double_records,
        },
        "duration_seconds": time.perf_counter() - started,
    }
