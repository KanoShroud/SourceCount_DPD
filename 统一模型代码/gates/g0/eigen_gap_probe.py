"""最大特征值近重根的摘要工具。"""

from __future__ import annotations

from typing import Any

import torch


def summarize_relative_gaps(
    largest: torch.Tensor,
    second_largest: torch.Tensor,
    *,
    loss_gradient: torch.Tensor | None = None,
) -> dict[str, Any]:
    gap = (largest - second_largest).abs() / largest.abs().clamp_min(1e-12)
    flat = gap.detach().reshape(-1).to(torch.float64).cpu()
    report: dict[str, Any] = {
        "minimum": float(flat.min().item()),
        "quantiles": {
            str(q): float(torch.quantile(flat, q).item())
            for q in (0.001, 0.01, 0.05, 0.5)
        },
        "fractions": {
            str(threshold): float((flat < threshold).to(torch.float64).mean().item())
            for threshold in (1e-6, 1e-5, 1e-4)
        },
    }
    if loss_gradient is not None:
        mass = loss_gradient.detach().abs().reshape(-1).to(torch.float64).cpu()
        total = float(mass.sum().item())
        report["gradient_exposure"] = {
            str(threshold): (
                float(mass[flat < threshold].sum().item()) / total if total > 0 else 0.0
            )
            for threshold in (1e-6, 1e-5, 1e-4)
        }
    return report
