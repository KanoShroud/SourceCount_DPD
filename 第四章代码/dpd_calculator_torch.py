"""
dpd_calculator_torch.py — PyTorch 版 DPD 空间谱计算 (优化版)

翻译自 DPD_calculator_gpu_batch.m，用于第四章定位的细网格 DPD

优化：
  1. 仅在信号带宽内的频点做 matmul（~8x加速）
  2. 支持预计算几何信息（taus/dtaus），避免每样本重复计算
  3. 支持预计算网格到各站距离，加速双曲线掩码
"""

import torch
import numpy as np

from chapter_runtime import device as runtime_device


# ═══════════════════════════════════════
#  几何预计算（所有样本共享）
# ═══════════════════════════════════════
class DPDGeometry:
    """
    预计算 DPD 搜索网格的几何信息，避免每样本重复计算。

    用法:
        geo = DPDGeometry(rcvPos, init_pos, edge, lamda, fs, N0, device)
        mtr = compute_fine_dpd(sig, geo, freq_mask=mask)
    """
    def __init__(self, rcvPos, init_pos, edge, lamda, fs, N0, device):
        self.device = device
        self.fs = fs
        self.N0 = N0
        self.edge = edge
        self.lamda = lamda
        vc = 299792458.0

        if isinstance(rcvPos, np.ndarray):
            rcvPos = torch.from_numpy(rcvPos).float()
        self.rcvPos = rcvPos.to(device)
        self.rcv_num = rcvPos.shape[0]

        # 搜索网格
        num_x = int(round(2 * edge / lamda)) + 1
        self.num_x = num_x
        self.num_y = num_x
        self.num_grid = num_x * num_x

        x_vec = torch.linspace(init_pos[0] - edge, init_pos[0] + edge, num_x,
                               dtype=torch.float32, device=device)
        y_vec = torch.linspace(init_pos[1] - edge, init_pos[1] + edge, num_x,
                               dtype=torch.float32, device=device)

        yg, xg = torch.meshgrid(y_vec, x_vec, indexing='ij')
        grid_x = xg.reshape(-1)
        grid_y = yg.reshape(-1)

        # 各网格点到各站的时延 (num_grid, rcv_num)
        self.taus = torch.zeros(self.num_grid, self.rcv_num,
                                dtype=torch.float64, device=device)
        for m in range(self.rcv_num):
            dx = grid_x.double() - self.rcvPos[m, 0].double()
            dy = grid_y.double() - self.rcvPos[m, 1].double()
            self.taus[:, m] = torch.sqrt(dx**2 + dy**2) / vc

        # 站对列表
        self.pairs = []
        for m1 in range(self.rcv_num):
            for m2 in range(m1 + 1, self.rcv_num):
                self.pairs.append((m1, m2))
        self.nPairs = len(self.pairs)

        # 站对的差分时延 (num_grid, nPairs) — 预计算
        self.dtaus = torch.zeros(self.num_grid, self.nPairs,
                                 dtype=torch.float64, device=device)
        for ip, (m1, m2) in enumerate(self.pairs):
            self.dtaus[:, ip] = self.taus[:, m1] - self.taus[:, m2]

        # 频率轴 (N0,)
        self.f_full = torch.linspace(-fs/2, fs/2 - fs/N0, N0,
                                     dtype=torch.float64, device=device)

        # 双曲线掩码用的网格距离 (rcv_num, num_y, num_x)
        self.grid_dist_to_station = torch.zeros(self.rcv_num, num_x, num_x,
                                                dtype=torch.float32, device='cpu')
        for m in range(self.rcv_num):
            dx = xg - self.rcvPos[m, 0].cpu()
            dy = yg - self.rcvPos[m, 1].cpu()
            self.grid_dist_to_station[m] = torch.sqrt(dx**2 + dy**2)

        print(f"  DPDGeometry: grid {num_x}×{num_x} = {self.num_grid} points, "
              f"{self.nPairs} pairs, N0={N0}")


# ═══════════════════════════════════════
#  DPD 空间谱计算（优化版）
# ═══════════════════════════════════════
def compute_fine_dpd(sig_rcv_complex, geo, freq_mask=None, chunk_size=40000):
    """
    从 IQ 信号计算细网格 DPD 空间谱（优化版）

    参数:
        sig_rcv_complex: (rcv_num, N0) complex, 各站时域IQ信号
        geo:             DPDGeometry 预计算对象
        freq_mask:       (N0,) bool, 频域掩码（可选）
        chunk_size:      每次处理的搜索点数

    返回:
        mtr: (num_y, num_x) float32, DPD 空间谱
    """
    device = geo.device

    if isinstance(sig_rcv_complex, np.ndarray):
        sig_rcv_complex = torch.from_numpy(sig_rcv_complex)

    rcv_num = geo.rcv_num
    N0 = geo.N0

    # ── FFT + 滤波 + 功率归一化 ──
    sig_fft = torch.fft.fftshift(torch.fft.fft(sig_rcv_complex.to(device)), dim=-1)

    if freq_mask is not None:
        if isinstance(freq_mask, np.ndarray):
            freq_mask = torch.from_numpy(freq_mask).to(device)
        sig_fft = sig_fft * freq_mask.unsqueeze(0).float()

    # 时域功率归一化
    sig_time = torch.fft.ifft(torch.fft.ifftshift(sig_fft, dim=-1), dim=-1)
    for m in range(rcv_num):
        P_m = (sig_time[m].abs() ** 2).mean()
        if P_m > 0:
            sig_time[m] = sig_time[m] / torch.sqrt(P_m)
    sig_fft = torch.fft.fftshift(torch.fft.fft(sig_time, dim=-1), dim=-1)

    # ── 提取信号带宽内的频点（核心优化）──
    sig_fft_d = sig_fft.to(torch.complex128)

    if freq_mask is not None:
        band_idx = freq_mask.bool()
        if isinstance(band_idx, torch.Tensor):
            band_idx = band_idx.cpu()
        f_band = geo.f_full[band_idx]                       # (n_band,)
        sig_band = sig_fft_d[:, band_idx]                   # (rcv_num, n_band)
    else:
        f_band = geo.f_full
        sig_band = sig_fft_d

    n_band = f_band.shape[0]
    TWO_PI_F_BAND = 2j * np.pi * f_band                    # (n_band,)

    # ── 互谱（仅带内频点）──
    diag_vals = (sig_band.abs() ** 2).sum(dim=-1).real      # (rcv_num,)

    cs_mat = torch.zeros(n_band, geo.nPairs, dtype=torch.complex128, device=device)
    for ip, (m1, m2) in enumerate(geo.pairs):
        cs_mat[:, ip] = sig_band[m1] * sig_band[m2].conj()

    # ── 分块计算相关矩阵 + 最大特征值 ──
    max_eig_all = torch.zeros(geo.num_grid, dtype=torch.float64, device=device)

    for start in range(0, geo.num_grid, chunk_size):
        end = min(start + chunk_size, geo.num_grid)
        n_chunk = end - start

        R_chunk = torch.zeros(n_chunk, rcv_num, rcv_num,
                              dtype=torch.complex128, device=device)

        # 对角线
        for m in range(rcv_num):
            R_chunk[:, m, m] = diag_vals[m]

        # 非对角线（使用预计算的 dtaus，仅带内频点）
        for ip, (m1, m2) in enumerate(geo.pairs):
            dtau = geo.dtaus[start:end, ip]                            # (n_chunk,)
            phase_mat = torch.exp(dtau.unsqueeze(1) * TWO_PI_F_BAND.unsqueeze(0))  # (n_chunk, n_band)
            val = (phase_mat @ cs_mat[:, ip:ip+1]).squeeze(1)          # (n_chunk,)
            R_chunk[:, m1, m2] = val
            R_chunk[:, m2, m1] = val.conj()
            del dtau, phase_mat, val

        # 特征值分解（CPU eigvalsh，和MATLAB pageeig一致）
        R_cpu = R_chunk.cpu()
        del R_chunk
        torch.cuda.empty_cache()
        eps = 1e-6 * torch.eye(rcv_num, dtype=R_cpu.dtype).unsqueeze(0)
        R_cpu = R_cpu + eps
        try:
            eigvals = torch.linalg.eigvalsh(R_cpu)
        except torch._C._LinAlgError:
            eigvals = torch.linalg.svdvals(R_cpu)
        max_eig_all[start:end] = eigvals.abs().max(dim=-1).values.to(device)
        del R_cpu, eigvals

    mtr = max_eig_all.float().reshape(geo.num_y, geo.num_x)
    return mtr.cpu()


# ═══════════════════════════════════════
#  双曲线掩码（支持预计算距离）
# ═══════════════════════════════════════
def compute_hyperbola_mask(src_pos, rcvPos, x_vec, y_vec,
                           tolerance_m=30.0, mode='max',
                           precomputed_dists=None):
    """
    计算 TDOA 双曲线掩码

    参数:
        src_pos:           (2,) 源位置 [x, y]
        rcvPos:            (rcv_num, 2) 接收站坐标
        x_vec:             (num_x,) 网格 x 坐标
        y_vec:             (num_y,) 网格 y 坐标
        tolerance_m:       高斯衰减σ（米）
        mode:              'max'=取最大值, 'sum'=求和
        precomputed_dists: (rcv_num, num_y, num_x) 预计算的网格到各站距离

    返回:
        mask: (num_y, num_x) float32
    """
    if isinstance(src_pos, torch.Tensor):
        src_pos = src_pos.numpy()
    if isinstance(rcvPos, torch.Tensor):
        rcvPos = rcvPos.numpy()

    rcv_num = len(rcvPos)

    # 源到各站的距离
    src_dist = np.array([np.linalg.norm(src_pos - rcvPos[m]) for m in range(rcv_num)])

    # 使用预计算的网格距离
    if precomputed_dists is not None:
        if isinstance(precomputed_dists, torch.Tensor):
            precomputed_dists = precomputed_dists.numpy()
        grid_dists = precomputed_dists   # (rcv_num, num_y, num_x)
    else:
        if isinstance(x_vec, torch.Tensor):
            x_vec = x_vec.numpy()
        if isinstance(y_vec, torch.Tensor):
            y_vec = y_vec.numpy()
        yg, xg = np.meshgrid(y_vec, x_vec, indexing='ij')
        grid_dists = np.zeros((rcv_num, yg.shape[0], yg.shape[1]), dtype=np.float32)
        for m in range(rcv_num):
            grid_dists[m] = np.sqrt((xg - rcvPos[m, 0])**2 + (yg - rcvPos[m, 1])**2)

    mask = np.zeros(grid_dists.shape[1:], dtype=np.float32)
    inv_2sigma2 = -0.5 / (tolerance_m ** 2)

    for m1 in range(rcv_num):
        for m2 in range(m1 + 1, rcv_num):
            delta_d_true = src_dist[m1] - src_dist[m2]
            diff = np.abs(grid_dists[m1] - grid_dists[m2] - delta_d_true)
            hyperbola = np.exp(inv_2sigma2 * diff ** 2)
            if mode == 'sum':
                mask += hyperbola
            else:
                mask = np.maximum(mask, hyperbola)

    if mode == 'sum':
        mx = mask.max()
        if mx > 0:
            mask = mask / mx

    return mask


# ═══════════════════════════════════════
#  兼容旧接口（无预计算）
# ═══════════════════════════════════════
def compute_fine_dpd_compat(sig_rcv_complex, fs, rcvPos, init_pos, edge, lamda,
                            freq_mask=None, device=None, chunk_size=40000):
    """
    兼容旧接口：自动创建 DPDGeometry 再调用优化版

    用于 gen_exp_data.py 等不方便预计算的场景
    """
    if device is None:
        device = runtime_device()

    if isinstance(sig_rcv_complex, np.ndarray):
        N0 = sig_rcv_complex.shape[1]
    else:
        N0 = sig_rcv_complex.shape[1]

    # 每次创建（无预计算优势，但仍有带内频点和幂迭代优化）
    geo = DPDGeometry(rcvPos, init_pos, edge, lamda, fs, N0, device)
    return compute_fine_dpd(sig_rcv_complex, geo, freq_mask, chunk_size)
