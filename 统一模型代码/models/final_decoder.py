"""Soft 内部图之后的确定性 Hard 最终解码器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DecodedOutput:
    band_mask_hard: torch.Tensor
    predicted_k: int
    peak_indices: torch.Tensor
    position_set_m: torch.Tensor
    scores: torch.Tensor
    hard_count: int
    hard_count_mismatch: bool


def _stable_peak_indices(
    heatmap_logits: torch.Tensor,
    *,
    count: int,
    peak_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if count <= 0:
        return (
            torch.empty(0, dtype=torch.long, device=heatmap_logits.device),
            torch.empty(0, dtype=heatmap_logits.dtype, device=heatmap_logits.device),
        )
    if heatmap_logits.ndim == 3 and heatmap_logits.shape[0] == 1:
        heatmap_logits = heatmap_logits[0]
    if heatmap_logits.ndim != 2:
        raise ValueError("单样本 heatmap 必须为(H,W)或(1,H,W)")
    probabilities = torch.sigmoid(heatmap_logits)
    pooled = F.max_pool2d(
        probabilities[None, None], peak_size, stride=1, padding=peak_size // 2
    )[0, 0]
    suppressed = probabilities * (probabilities == pooled).to(probabilities.dtype)
    flat = suppressed.reshape(-1)
    order = torch.argsort(flat, descending=True, stable=True)
    k = min(int(count), int(flat.numel()))
    indices = order[:k]
    return indices, flat[indices]


def decode_final_output(
    *,
    band_probabilities: torch.Tensor,
    cardinality_distribution: torch.Tensor,
    heatmap_logits: torch.Tensor,
    offset: torch.Tensor,
    band_threshold: float = 0.5,
    max_count: int = 3,
    peak_size: int = 9,
    edge_m: float = 2000.0,
    grid_step_m: float = 10.0,
) -> DecodedOutput:
    """生成硬频带、整数 K 和确定长度的位置集合。"""
    if band_probabilities.ndim != 2:
        raise ValueError("band_probabilities 必须为(slot,band)")
    if cardinality_distribution.ndim != 1:
        raise ValueError("cardinality_distribution 必须为(K类别,)")
    if offset.ndim != 3 or offset.shape[0] != 2:
        raise ValueError("offset 必须为(2,H,W)")
    if heatmap_logits.shape[-2:] != offset.shape[-2:]:
        raise ValueError("heatmap 与 offset 空间 shape 不一致")
    band_mask = band_probabilities >= float(band_threshold)
    hard_count = int(band_mask.any(dim=-1).sum().item())
    predicted_k = min(
        int(torch.argmax(cardinality_distribution).item()), int(max_count)
    )
    indices, scores = _stable_peak_indices(
        heatmap_logits, count=predicted_k, peak_size=peak_size
    )
    if predicted_k == 0:
        positions = torch.empty(
            (0, 2), dtype=offset.dtype, device=offset.device
        )
    else:
        width = offset.shape[-1]
        x_index = indices.remainder(width)
        y_index = torch.div(indices, width, rounding_mode="floor")
        dx = offset[0, y_index, x_index].clamp(-1.0, 1.0)
        dy = offset[1, y_index, x_index].clamp(-1.0, 1.0)
        pixels = torch.stack(
            [x_index.to(offset.dtype) + dx, y_index.to(offset.dtype) + dy], dim=-1
        )
        positions = pixels * float(grid_step_m) - float(edge_m)
    if len(positions) != predicted_k:
        raise AssertionError("最终位置集合长度与 predicted_K 不一致")
    return DecodedOutput(
        band_mask_hard=band_mask,
        predicted_k=predicted_k,
        peak_indices=indices,
        position_set_m=positions,
        scores=scores,
        hard_count=hard_count,
        hard_count_mismatch=hard_count != predicted_k,
    )
