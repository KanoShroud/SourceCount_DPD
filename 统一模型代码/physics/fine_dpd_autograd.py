"""保持第四章物理语义的可微细 DPD 分块原型。"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint


def _validate_inputs(
    signal: torch.Tensor,
    frequency_weights: torch.Tensor,
    fixed_support: torch.Tensor,
    geometry: object,
) -> None:
    expected_signal = (int(geometry.rcv_num), int(geometry.N0))
    if tuple(signal.shape) != expected_signal:
        raise ValueError(
            f"signal 应为{expected_signal}，实际为{tuple(signal.shape)}"
        )
    if tuple(frequency_weights.shape) != (int(geometry.N0),):
        raise ValueError("frequency_weights shape 错误")
    if tuple(fixed_support.shape) != (int(geometry.N0),):
        raise ValueError("fixed_support shape 错误")
    if not bool(torch.isfinite(signal.real).all() and torch.isfinite(signal.imag).all()):
        raise ValueError("signal 含 NaN/Inf")
    if not bool(torch.isfinite(frequency_weights).all()):
        raise ValueError("frequency_weights 含 NaN/Inf")
    if not bool(
        torch.all((frequency_weights >= 0.0) & (frequency_weights <= 1.0))
    ):
        raise ValueError("frequency_weights 必须位于[0,1]")
    if not bool(fixed_support.any()):
        raise ValueError("fixed_support 为空")


def compute_fine_dpd_autograd(
    signal: np.ndarray | torch.Tensor,
    geometry: object,
    frequency_weights: torch.Tensor,
    *,
    fixed_support: torch.Tensor,
    grid_chunk_size: int = 2048,
    frequency_chunk_size: int = 256,
    eig_device: Literal["cpu", "cuda"] = "cpu",
    use_checkpoint: bool = True,
    checkpoint_mode: Literal["off", "reentrant", "nonreentrant"] | None = None,
    real_dtype: torch.dtype = torch.float32,
    diagonal_loading: float = 1e-6,
    power_floor: float = 1e-20,
    progress_callback: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """计算 ``(H,W)`` 细 DPD，并保留到 ``frequency_weights`` 的梯度。

    与历史实现一致：fftshift FFT、频域振幅乘 ``sqrt(w)``、逐站时域
    功率归一化、站间互谱、几何相位补偿以及 4×4 Hermitian 矩阵最大
    绝对特征值。连续图的频率支持只能由固定物理矩阵给出，不能由
    ``w > 0`` 动态决定。
    """
    if grid_chunk_size < 1 or frequency_chunk_size < 1:
        raise ValueError("分块大小必须为正整数")
    if checkpoint_mode is None:
        checkpoint_mode = "reentrant" if use_checkpoint else "off"
    if checkpoint_mode not in {"off", "reentrant", "nonreentrant"}:
        raise ValueError(f"未知checkpoint_mode: {checkpoint_mode}")
    if real_dtype not in {torch.float32, torch.float64}:
        raise ValueError("real_dtype只允许torch.float32或torch.float64")
    compute_device = torch.device(geometry.device)
    if eig_device == "cuda" and compute_device.type != "cuda":
        raise ValueError("eig_device=cuda 需要 CUDA geometry")
    complex_dtype = torch.complex128 if real_dtype == torch.float64 else torch.complex64
    signal_tensor = torch.as_tensor(
        signal, dtype=complex_dtype, device=compute_device
    )
    if not torch.is_complex(signal_tensor):
        raise ValueError("signal 必须是复数 IQ")
    weights = torch.as_tensor(
        frequency_weights, dtype=real_dtype, device=compute_device
    )
    support = torch.as_tensor(
        fixed_support, dtype=torch.bool, device=compute_device
    )
    _validate_inputs(signal_tensor, weights, support, geometry)

    spectrum = torch.fft.fftshift(torch.fft.fft(signal_tensor, dim=-1), dim=-1)
    weighted = spectrum * torch.sqrt(weights).unsqueeze(0)
    signal_time = torch.fft.ifft(
        torch.fft.ifftshift(weighted, dim=-1), dim=-1
    )
    power = signal_time.abs().square().mean(dim=-1, keepdim=True)
    denominator = torch.sqrt(power.clamp_min(float(power_floor)))
    normalized_time = signal_time / denominator
    normalized_spectrum = torch.fft.fftshift(
        torch.fft.fft(normalized_time, dim=-1), dim=-1
    ).to(torch.complex128)

    support_indices = torch.nonzero(support, as_tuple=False).squeeze(1)
    frequencies = geometry.f_full.index_select(0, support_indices).to(
        compute_device, dtype=torch.float64
    )
    signal_band = normalized_spectrum.index_select(1, support_indices)
    diagonal = signal_band.abs().square().sum(dim=-1).real
    cross_spectra = torch.stack(
        [
            signal_band[m1] * signal_band[m2].conj()
            for m1, m2 in geometry.pairs
        ],
        dim=-1,
    )
    dtaus = geometry.dtaus.to(compute_device, dtype=torch.float64)
    pair_count = len(geometry.pairs)
    receiver_count = int(geometry.rcv_num)
    eig_target = compute_device if eig_device == "cuda" else torch.device("cpu")

    def compute_grid_chunk(
        cross_input: torch.Tensor,
        diagonal_input: torch.Tensor,
        start: int,
        stop: int,
    ) -> torch.Tensor:
        chunk_length = stop - start
        matrix = torch.diag_embed(
            diagonal_input.to(torch.complex128)
            .unsqueeze(0)
            .expand(chunk_length, receiver_count)
        ).clone()
        chunk_dtaus = dtaus[start:stop]
        for pair_index, (m1, m2) in enumerate(geometry.pairs):
            accumulated = torch.zeros(
                chunk_length, dtype=torch.complex128, device=compute_device
            )
            for f_start in range(0, frequencies.numel(), frequency_chunk_size):
                f_stop = min(f_start + frequency_chunk_size, frequencies.numel())
                current_frequencies = frequencies[f_start:f_stop]
                phase = torch.exp(
                    chunk_dtaus[:, pair_index : pair_index + 1]
                    * (2j * math.pi * current_frequencies.unsqueeze(0))
                )
                accumulated = accumulated + phase @ cross_input[
                    f_start:f_stop, pair_index
                ]
            matrix[:, m1, m2] = accumulated
            matrix[:, m2, m1] = accumulated.conj()
        if pair_count == 0:
            raise AssertionError("DPD 至少需要一个接收站对")
        matrix = matrix.to(eig_target)
        identity = torch.eye(
            receiver_count, dtype=torch.complex128, device=eig_target
        ).unsqueeze(0)
        eigenvalues = torch.linalg.eigvalsh(
            matrix + float(diagonal_loading) * identity
        )
        maximum = eigenvalues.abs().amax(dim=-1)
        if progress_callback is not None:
            progress_callback(stop, int(geometry.num_grid))
        return maximum.to(compute_device)

    outputs: list[torch.Tensor] = []
    for grid_start in range(0, int(geometry.num_grid), grid_chunk_size):
        grid_stop = min(grid_start + grid_chunk_size, int(geometry.num_grid))
        if checkpoint_mode != "off" and (
            cross_spectra.requires_grad or diagonal.requires_grad
        ):
            current = checkpoint(
                lambda cs, diag, start=grid_start, stop=grid_stop: compute_grid_chunk(
                    cs, diag, start, stop
                ),
                cross_spectra,
                diagonal,
                use_reentrant=checkpoint_mode == "reentrant",
                preserve_rng_state=False,
            )
        else:
            current = compute_grid_chunk(
                cross_spectra, diagonal, grid_start, grid_stop
            )
        outputs.append(current)
    spectrum_map = torch.cat(outputs).reshape(
        int(geometry.num_y), int(geometry.num_x)
    )
    return spectrum_map.to(real_dtype)
