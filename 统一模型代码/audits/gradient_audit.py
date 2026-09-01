"""G0 使用的方向导数与梯度有限性审计。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


def tensor_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached[finite]
    if finite_values.numel() == 0:
        minimum = maximum = mean = norm = None
    else:
        minimum = float(finite_values.min().item())
        maximum = float(finite_values.max().item())
        mean = float(finite_values.mean().item())
        norm = float(torch.linalg.vector_norm(finite_values).item())
    return {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "finite_count": int(finite.sum().item()),
        "nonfinite_count": int((~finite).sum().item()),
        "nonzero_count": int(torch.count_nonzero(detached).item()),
        "min": minimum,
        "max": maximum,
        "mean": mean,
        "l2_norm": norm,
    }


def normalize_direction(direction: torch.Tensor) -> torch.Tensor:
    if not bool(torch.isfinite(direction).all()):
        raise ValueError("方向张量含 NaN/Inf")
    norm = torch.linalg.vector_norm(direction)
    if not bool(norm > 0):
        raise ValueError("方向张量范数为零")
    return direction / norm


def autograd_directional_derivative(
    scalar: torch.Tensor,
    variable: torch.Tensor,
    direction: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    gradient = torch.autograd.grad(
        scalar, variable, retain_graph=False, create_graph=False
    )[0]
    derivative = torch.sum(gradient * normalize_direction(direction).to(gradient))
    return gradient, float(derivative.detach().item())


@torch.no_grad()
def central_finite_differences(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    direction: torch.Tensor,
    steps: Iterable[float],
) -> list[dict[str, float]]:
    unit_direction = normalize_direction(direction).to(point)
    records: list[dict[str, float]] = []
    for step in steps:
        if step <= 0:
            raise ValueError("有限差分步长必须为正")
        plus = function(point + float(step) * unit_direction)
        minus = function(point - float(step) * unit_direction)
        if plus.numel() != 1 or minus.numel() != 1:
            raise ValueError("有限差分函数必须返回标量")
        derivative = (plus - minus) / (2.0 * float(step))
        records.append(
            {
                "step": float(step),
                "plus": float(plus.item()),
                "minus": float(minus.item()),
                "directional_derivative": float(derivative.item()),
            }
        )
    return records


def compare_directional_derivatives(
    autograd_value: float,
    finite_differences: list[dict[str, float]],
    *,
    relative_tolerance: float = 0.05,
    absolute_floor: float = 1e-8,
) -> dict[str, Any]:
    comparisons = []
    for record in finite_differences:
        observed = float(record["directional_derivative"])
        denominator = max(abs(autograd_value), abs(observed), absolute_floor)
        relative_error = abs(autograd_value - observed) / denominator
        sign_match = (
            abs(autograd_value) <= absolute_floor
            and abs(observed) <= absolute_floor
        ) or (autograd_value * observed > 0)
        comparisons.append(
            {
                **record,
                "autograd_directional_derivative": float(autograd_value),
                "relative_error": float(relative_error),
                "sign_match": bool(sign_match),
                "pass": bool(sign_match and relative_error <= relative_tolerance),
            }
        )
    return {
        "status": "PASS" if any(row["pass"] for row in comparisons) else "FAIL",
        "relative_tolerance": float(relative_tolerance),
        "comparisons": comparisons,
    }
