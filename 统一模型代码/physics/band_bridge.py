"""CH3 槽位概率到 FFT 连续权重的固定物理桥。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class BridgeOutput:
    """连续桥的全部可审计中间量。"""

    probabilities: torch.Tensor
    slot_existence: torch.Tensor
    subband_union: torch.Tensor
    frequency_weights: torch.Tensor
    cardinality_distribution: torch.Tensor


def build_subband_fft_matrix(
    subband_lo_hz: torch.Tensor,
    subband_hi_hz: torch.Tensor,
    *,
    sample_rate_hz: float,
    n_fft: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """建立冻结的矩形子带到 fftshift 频点覆盖矩阵 ``(F, B)``。"""
    if n_fft <= 0:
        raise ValueError("n_fft 必须为正整数")
    lo = torch.as_tensor(subband_lo_hz, dtype=torch.float64, device=device).reshape(-1)
    hi = torch.as_tensor(subband_hi_hz, dtype=torch.float64, device=device).reshape(-1)
    if lo.shape != hi.shape or lo.numel() == 0:
        raise ValueError("子带上下界 shape 不一致或为空")
    if not bool(torch.isfinite(lo).all() and torch.isfinite(hi).all()):
        raise ValueError("子带上下界含 NaN/Inf")
    if not bool(torch.all(hi > lo)):
        raise ValueError("每个子带上界必须大于下界")
    freq = (
        torch.arange(-n_fft // 2, n_fft // 2, dtype=torch.float64, device=lo.device)
        * (float(sample_rate_hz) / n_fft)
    )
    matrix = (freq[:, None] >= lo[None, :]) & (freq[:, None] < hi[None, :])
    if not bool(matrix.any(dim=0).all()):
        raise ValueError("至少一个子带没有覆盖任何 FFT 频点")
    return matrix.to(dtype=dtype)


def poisson_binomial_categories(
    existence: torch.Tensor,
    *,
    max_count: int = 3,
) -> torch.Tensor:
    """计算 ``P(K=0),...,P(K=max_count-1),P(K>=max_count)``。"""
    if existence.ndim < 1 or existence.shape[-1] < 1:
        raise ValueError("existence 最后一维必须是非空槽位维")
    if max_count < 1:
        raise ValueError("max_count 必须大于等于 1")
    q = existence.clamp(0.0, 1.0)
    probabilities = torch.ones(*q.shape[:-1], 1, dtype=q.dtype, device=q.device)
    for slot in range(q.shape[-1]):
        current = q[..., slot : slot + 1]
        stay = probabilities * (1.0 - current)
        advance = torch.nn.functional.pad(probabilities * current, (1, 0))
        stay = torch.nn.functional.pad(stay, (0, 1))
        probabilities = stay + advance
    if probabilities.shape[-1] <= max_count:
        probabilities = torch.nn.functional.pad(
            probabilities, (0, max_count + 1 - probabilities.shape[-1])
        )
    categories = torch.cat(
        [
            probabilities[..., :max_count],
            probabilities[..., max_count:].sum(dim=-1, keepdim=True),
        ],
        dim=-1,
    )
    return categories / categories.sum(dim=-1, keepdim=True).clamp_min(
        torch.finfo(categories.dtype).tiny
    )


def continuous_band_bridge(
    values: torch.Tensor,
    subband_fft_matrix: torch.Tensor,
    *,
    values_are_logits: bool = True,
    max_count: int = 3,
) -> BridgeOutput:
    """把槽位×子带 logits/概率映射为连续 FFT 权重。"""
    if values.ndim < 2:
        raise ValueError("values 至少需要槽位和子带两个维度")
    matrix = torch.as_tensor(
        subband_fft_matrix, dtype=values.dtype, device=values.device
    )
    if matrix.ndim != 2 or matrix.shape[1] != values.shape[-1]:
        raise ValueError(
            f"覆盖矩阵应为(F,{values.shape[-1]})，实际为{tuple(matrix.shape)}"
        )
    probabilities = torch.sigmoid(values) if values_are_logits else values
    if not bool(torch.isfinite(probabilities).all()):
        raise ValueError("频带概率含 NaN/Inf")
    if not values_are_logits and not bool(
        torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    ):
        raise ValueError("直接输入的概率必须位于[0,1]")
    probabilities = probabilities.clamp(0.0, 1.0)
    slot_existence = 1.0 - torch.prod(1.0 - probabilities, dim=-1)
    subband_union = 1.0 - torch.prod(1.0 - probabilities, dim=-2)
    covered = matrix * subband_union.unsqueeze(-2)
    frequency_weights = 1.0 - torch.prod(1.0 - covered, dim=-1)
    cardinality = poisson_binomial_categories(slot_existence, max_count=max_count)
    return BridgeOutput(
        probabilities=probabilities,
        slot_existence=slot_existence,
        subband_union=subband_union,
        frequency_weights=frequency_weights.clamp(0.0, 1.0),
        cardinality_distribution=cardinality,
    )
