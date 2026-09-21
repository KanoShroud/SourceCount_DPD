"""G0D逐层目标的方向导数、单边差分和判定工具。"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


def _relative_error(left: float, right: float, floor: float = 1e-8) -> float:
    return abs(left - right) / max(abs(left), abs(right), floor)


@torch.no_grad()
def finite_difference_sweep(
    function: Callable[[torch.Tensor], torch.Tensor],
    point: torch.Tensor,
    direction: torch.Tensor,
    steps: Iterable[float],
) -> list[dict[str, float]]:
    """计算中心、左侧和右侧方向差分。"""
    unit = direction / torch.linalg.vector_norm(direction)
    baseline_tensor = function(point)
    baseline = float(baseline_tensor.item())
    records = []
    for step_value in steps:
        step = float(step_value)
        plus_tensor = function(point + step * unit)
        minus_tensor = function(point - step * unit)
        plus = float(plus_tensor.item())
        minus = float(minus_tensor.item())
        if plus_tensor.dtype.is_floating_point:
            values = torch.stack(
                [baseline_tensor.detach(), plus_tensor.detach(), minus_tensor.detach()]
            ).abs()
            toward = torch.full_like(values, torch.inf)
            resolution = float(
                (torch.nextafter(values, toward) - values).abs().max().item()
            )
        else:
            resolution = 0.0
        records.append(
            {
                "step": step,
                "baseline": baseline,
                "plus": plus,
                "minus": minus,
                "center": (plus - minus) / (2.0 * step),
                "right": (plus - baseline) / step,
                "left": (baseline - minus) / step,
                "loss_span": abs(plus - minus),
                "numeric_resolution": resolution,
            }
        )
    return records


def classify_derivatives(
    autograd_value: float,
    records: list[dict[str, float]],
    *,
    relative_tolerance: float = 0.05,
    minimum_loss_span: float = 1e-6,
) -> dict[str, Any]:
    """区分光滑通过、局部非光滑、稳定不一致和数值不足。"""
    enriched = []
    smooth_passes = 0
    stable_opposites = 0
    nonsmooth_candidates = 0
    for record in records:
        center = float(record["center"])
        left = float(record["left"])
        right = float(record["right"])
        required_span = max(
            minimum_loss_span, 8.0 * float(record["numeric_resolution"])
        )
        valid = float(record["loss_span"]) >= required_span
        center_sign_match = autograd_value * center > 0
        center_error = _relative_error(autograd_value, center)
        center_pass = valid and center_sign_match and center_error <= relative_tolerance
        opposite = valid and autograd_value * center < 0
        side_separation = _relative_error(left, right) > 0.25
        side_match = (
            _relative_error(autograd_value, left) <= relative_tolerance
            or _relative_error(autograd_value, right) <= relative_tolerance
        )
        nonsmooth = valid and side_separation and side_match
        smooth_passes += int(center_pass)
        stable_opposites += int(opposite)
        nonsmooth_candidates += int(nonsmooth)
        enriched.append(
            {
                **record,
                "valid": valid,
                "required_loss_span": required_span,
                "center_sign_match": center_sign_match,
                "center_relative_error": center_error,
                "center_pass": center_pass,
                "side_relative_gap": _relative_error(left, right),
                "nonsmooth_candidate": nonsmooth,
            }
        )
    if smooth_passes >= 2:
        status = "PASS_SMOOTH"
    elif nonsmooth_candidates >= 2 and stable_opposites < 2:
        status = "NONSMOOTH_LOCAL_POINT"
    elif stable_opposites >= 2:
        status = "STABLE_GRADIENT_MISMATCH"
    else:
        status = "INCONCLUSIVE_NUMERIC"
    return {
        "status": status,
        "autograd_directional_derivative": float(autograd_value),
        "relative_tolerance": relative_tolerance,
        "minimum_loss_span": minimum_loss_span,
        "smooth_pass_count": smooth_passes,
        "stable_opposite_count": stable_opposites,
        "nonsmooth_candidate_count": nonsmooth_candidates,
        "steps": enriched,
    }


def gradient_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    """比较两种checkpoint模式的完整梯度。"""
    ref = reference.reshape(-1).to(torch.float64)
    other = candidate.reshape(-1).to(torch.float64)
    denominator = max(
        float(torch.linalg.vector_norm(ref).item()),
        float(torch.linalg.vector_norm(other).item()),
        1e-12,
    )
    cosine = float(torch.nn.functional.cosine_similarity(ref, other, dim=0).item())
    relative_l2 = float(torch.linalg.vector_norm(ref - other).item()) / denominator
    return {
        "cosine_similarity": cosine,
        "relative_l2_difference": relative_l2,
        "max_abs_difference": float((ref - other).abs().max().item()),
        "pass": bool(cosine >= 0.9999 and relative_l2 <= 1e-3),
    }
