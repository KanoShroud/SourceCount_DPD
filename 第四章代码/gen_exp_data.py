"""
gen_exp_data.py — 处理控制变量实验数据

将 MATLAB 生成的 .mat 文件转换为所有方法（传统+深度学习）需要的数据集

输出（按参数值分文件保存）:
  exp_4A1_snr-10.pt, exp_4A1_snr-08.pt, ..., exp_4A1_snr+10.pt
  exp_4B1_dist0200.pt, exp_4B1_dist0400.pt, ..., exp_4B1_dist2000.pt

每个 .pt 文件包含:
  共用:
    fine_dpd:          (N, 1, 401, 401)  DPD 谱（T1 和 DL 输入）
    pos_label:         (N, 3, 2)          真值坐标（归一化，评估用）
    n_src:             (N,)               源数量
  传统方法用:
    sig_filtered_real: (N, 4, 4096)       滤波后 IQ 实部（T2, T3 输入）
    sig_filtered_imag: (N, 4, 4096)       滤波后 IQ 虚部
  DL 标签:
    hyp_mask:          (N, 3, 401, 401)   距离场双曲线（每源独立通道）
    hyp_mask_single:   (N, 1, 401, 401)   距离场双曲线（所有源叠加单通道）
    gauss_label:       (N, 1, 401, 401)   高斯热力图（所有源叠加单通道）
  元信息:
    param_value:       标量，当前参数值（SNR dB 或 距离 m）

用法:
  python gen_exp_data.py --exp 4A1
  python gen_exp_data.py --exp 4B2
"""

import numpy as np
import h5py
import torch
import os
import argparse
from dpd_calculator_torch import DPDGeometry, compute_fine_dpd, compute_hyperbola_mask

from chapter_runtime import (
    DEFAULT_DEVICE,
    data_dir as runtime_data_dir,
    device as runtime_device,
)


# ═══════════════════════════════════════
#  系统参数
# ═══════════════════════════════════════
FS           = 100e6
SYMBOL_RATE  = 10e6
ARFA_V       = 0.25
BW_ACTUAL    = SYMBOL_RATE * (1 + ARFA_V * 1.2)   # = 13MHz
LEN          = 4096
EDGE         = 2000
LAMDA_FINE   = 10       # 细网格步长 10m
MAX_SRC      = 3
R_RCV        = 500.0
GAUSS_SIGMA_PX = 2.0    # 高斯热力图标签 σ（像素），与 gen_ch4_loc_data.py 一致
HYP_SIGMA_M  = 15.0     # 距离场 σ=1.5 像素=15m
HYP_MODE     = 'sum'    # 距离场叠加模式


def get_rcv_pos():
    """4站均匀圆阵坐标"""
    angles = np.arange(4) * 2 * np.pi / 4
    return np.stack([R_RCV * np.cos(angles), R_RCV * np.sin(angles)], axis=1)


def freq_range_to_mask(freq_lo, freq_hi, N0, fs):
    """从频率范围构造频域掩码"""
    f_axis = np.arange(-N0//2, N0//2) * (fs / N0)
    return (f_axis >= freq_lo) & (f_axis < freq_hi)


def filter_iq(sig_complex, freq_mask):
    """
    按频域掩码滤波 IQ 信号

    输入: sig_complex (rcv_num, len) 复信号
    返回: sig_filtered (rcv_num, len) 滤波后复信号
    """
    rcv_num, N = sig_complex.shape
    sig_filtered = np.zeros_like(sig_complex)
    for m in range(rcv_num):
        fft_m = np.fft.fftshift(np.fft.fft(sig_complex[m]))
        fft_m = fft_m * freq_mask
        sig_filtered[m] = np.fft.ifft(np.fft.ifftshift(fft_m))
    return sig_filtered


def generate_gauss_label(positions, n_src, num_xy, edge, lamda, sigma_px=GAUSS_SIGMA_PX):
    """
    生成高斯热力图标签（单通道，所有源叠加）
    标准图像惯例: gx=row=y方向, gy=col=x方向

    返回: (1, num_xy, num_xy)
    """
    label = np.zeros((num_xy, num_xy), dtype=np.float32)
    ix = np.arange(num_xy, dtype=np.float32)
    gx, gy = np.meshgrid(ix, ix, indexing='ij')

    for s in range(n_src):
        px = (positions[s, 0] * edge + edge) / lamda   # x → col
        py = (positions[s, 1] * edge + edge) / lamda   # y → row
        gaussian = np.exp(-((gx - py)**2 + (gy - px)**2) / (2 * sigma_px**2))
        label = np.maximum(label, gaussian)

    # 归一化到 [0, 1]
    mx = label.max()
    if mx > 0:
        label = label / mx
    return label[np.newaxis, :, :]   # (1, H, W)


def generate_hyp_labels(positions, n_src, rcvPos, x_vec, sigma_m=HYP_SIGMA_M, mode=HYP_MODE):
    """
    生成距离场双曲线标签

    返回:
      per_src:  (MAX_SRC, num_xy, num_xy) 每源独立通道
      single:   (1, num_xy, num_xy) 所有源叠加单通道
    """
    num_xy = len(x_vec)
    per_src = np.zeros((MAX_SRC, num_xy, num_xy), dtype=np.float32)
    combined = np.zeros((num_xy, num_xy), dtype=np.float32)

    for s in range(n_src):
        s_pos = positions[s] * EDGE   # 还原物理坐标
        hm = compute_hyperbola_mask(
            s_pos, rcvPos, x_vec, x_vec,
            tolerance_m=sigma_m, mode=mode)
        per_src[s] = hm
        combined += hm

    # 单通道归一化
    mx = combined.max()
    if mx > 0:
        combined = combined / mx

    return per_src, combined[np.newaxis, :, :]   # (MAX_SRC, H, W), (1, H, W)


def parse_exp_name(exp_name):
    """
    解析实验名称

    返回: (exp_type, exp_sub)
      exp_type: 'snr', 'dist', 'sep'
      exp_sub:  '4A1', '4A2', ... 或 '4B1', '4B2', ... 或 '4C'
    """
    if exp_name.startswith('4A'):
        return 'snr', exp_name
    elif exp_name.startswith('4B'):
        return 'dist', exp_name
    elif exp_name == '4C':
        return 'sep', exp_name
    elif exp_name == '4D':
        return 'bw', exp_name
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, required=True,
                        help='实验名称: 4A2/4A3/4B2/4B3/4C/4D')
    parser.add_argument('--data_dir', type=str, default=str(runtime_data_dir()),
                        help='控制实验 MATLAB 数据目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='处理后实验数据根目录；默认按 smoke/formal 模式隔离')
    parser.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    parser.add_argument('--hyp_sigma', type=float, default=HYP_SIGMA_M)
    parser.add_argument('--gauss_sigma', type=float, default=GAUSS_SIGMA_PX)
    args = parser.parse_args()

    exp_type, exp_sub = parse_exp_name(args.exp)
    device = runtime_device(args.device)
    output_data_dir = args.output_dir or str(runtime_data_dir(create=True))

    # ── 确定 mat 文件路径 ──
    mat_file_map = {'snr': 'snr', 'dist': 'dist', 'sep': 'sep', 'bw': 'bw'}
    mat_file = os.path.join(args.data_dir, f'exp_{mat_file_map[exp_type]}_{exp_sub}.mat')
    print(f"实验: {args.exp} ({exp_type})")
    print(f"加载: {mat_file}")
    print(f"Device: {device}")

    if not os.path.exists(mat_file):
        print(f"错误: 文件不存在 {mat_file}")
        return

    # ── 加载 mat 文件 ──
    with h5py.File(mat_file, 'r') as f:
        sig_real   = np.array(f['sig_rcv_real_all'], dtype=np.float32)
        sig_imag   = np.array(f['sig_rcv_imag_all'], dtype=np.float32)
        src_count  = np.array(f['src_count_all'], dtype=np.int64).flatten()
        src_pos    = np.array(f['src_pos_all'], dtype=np.float32)
        fc_offset  = np.array(f['fc_offset_all'], dtype=np.float32)
        Pt_W       = np.array(f['Pt_W_all'], dtype=np.float32)
        avg_snr    = np.array(f['avg_snr_all'], dtype=np.float32).flatten()

        # 实验参数
        if exp_type == 'snr':
            param_all = np.array(f['snr_param_all'], dtype=np.float32).flatten()
            param_values = np.array(f['exp_snr_range_val'], dtype=np.float32).flatten()
            n_per_param = int(f['exp_n_per_snr_val'][()].item())
        elif exp_type == 'dist':
            param_all = np.array(f['dist_param_all'], dtype=np.float32).flatten()
            param_values = np.array(f['exp_dist_range_val'], dtype=np.float32).flatten()
            n_per_param = int(f['exp_n_per_dist_val'][()].item())
        elif exp_type == 'bw':
            param_all = np.array(f['bw_param_all'], dtype=np.float32).flatten()
            param_values = np.array(f['exp_bw_range_val'], dtype=np.float32).flatten()
            n_per_param = int(f['exp_n_per_bw_val'][()].item())
        else:  # sep
            param_all = np.array(f['sep_param_all'], dtype=np.float32).flatten()
            param_values = np.array(f['exp_sep_range_val'], dtype=np.float32).flatten()
            n_per_param = int(f['exp_n_per_sep_val'][()].item())

        # 系统参数
        fs_val = float(f['fs_val'][()].item())
        symbolRate_val = float(f['symbolRate_val'][()].item())

    # h5py 转置
    sig_real  = sig_real.transpose(2, 1, 0)      # (N, rcv_num, len)
    sig_imag  = sig_imag.transpose(2, 1, 0)
    src_pos   = src_pos.transpose(2, 1, 0)       # (N, max_src, 2)
    fc_offset = fc_offset.transpose(1, 0)        # (N, max_src)
    Pt_W      = Pt_W.transpose(1, 0)             # (N, max_src)

    N_total = len(src_count)
    rcv_num = sig_real.shape[1]
    print(f"总样本: {N_total}, 站数: {rcv_num}")
    print(f"参数值: {param_values}")
    print(f"每参数: {n_per_param} 样本")

    # ── 网格 ──
    num_xy = int(round(2 * EDGE / LAMDA_FINE)) + 1
    x_vec = np.linspace(-EDGE, EDGE, num_xy)
    rcvPos = get_rcv_pos()
    print(f"DPD 网格: {num_xy}×{num_xy}")

    # ── 预计算 DPD 几何信息 ──
    print(f"预计算 DPD 几何信息...")
    geo = DPDGeometry(rcvPos, [0.0, 0.0], EDGE, LAMDA_FINE, FS, LEN, device)

    # ── 输出目录 ──
    out_dir_map = {'snr': 'exp_4A', 'dist': 'exp_4B', 'sep': 'exp_4C', 'bw': 'exp_4D'}
    out_dir = os.path.join(output_data_dir, 'exp', out_dir_map[exp_type])
    os.makedirs(out_dir, exist_ok=True)
    print(f"输出目录: {out_dir}")

    # ── 按参数值分组处理 ──
    for pi, pval in enumerate(param_values):
        idx_start = pi * n_per_param
        idx_end = idx_start + n_per_param

        if exp_type == 'snr':
            param_str = f"snr{pval:+03.0f}"
            print(f"\n{'='*50}")
            print(f"  SNR = {pval:+.0f} dB ({n_per_param} 样本)")
        elif exp_type == 'dist':
            param_str = f"dist{pval:04.0f}"
            print(f"\n{'='*50}")
            print(f"  距离 = {pval:.0f}m ({n_per_param} 样本)")
        elif exp_type == 'bw':
            param_str = f"bw{pval:04.0f}"
            print(f"\n{'='*50}")
            print(f"  源2带宽 = {pval:.0f}MHz ({n_per_param} 样本)")
            # 4D: 按源2带宽计算联合频率掩码
            ARFA_V = 0.25
            BW_src1 = 10e6 * (1 + ARFA_V * 1.2)   # 源1固定13MHz
            BW_src2 = pval * 1e6 * (1 + ARFA_V * 1.2)
            BW_ACTUAL_4D = max(BW_src1, BW_src2)
            print(f"    BW_union = {BW_ACTUAL_4D/1e6:.1f}MHz")
        else:  # sep
            param_str = f"sep{pval:04.0f}"
            print(f"\n{'='*50}")
            print(f"  间距 = {pval:.0f}m ({n_per_param} 样本)")

        # 当前组的数据容器
        dpd_list = []
        pos_list = []
        nsrc_list = []
        sig_filt_real_list = []
        sig_filt_imag_list = []
        hyp_per_src_list = []
        hyp_single_list = []
        gauss_list = []

        for i in range(idx_start, idx_end):
            if (i - idx_start) % 100 == 0:
                print(f"    [{i-idx_start}/{n_per_param}]")

            n_src = int(src_count[i])
            positions = src_pos[i, :n_src, :]   # (n_src, 2)
            fc_offs = fc_offset[i, :n_src]

            # ── 构造 IQ 复信号 ──
            sig_complex = sig_real[i] + 1j * sig_imag[i]   # (4, 4096)

            # ── 滤波：按实际频率范围 ──
            if exp_type == 'bw':
                # 4D: 使用联合带宽
                flo = -BW_ACTUAL_4D / 2
                fhi =  BW_ACTUAL_4D / 2
            elif n_src > 0:
                flo = min(fc_offs) - BW_ACTUAL / 2
                fhi = max(fc_offs) + BW_ACTUAL / 2
            else:
                flo = -BW_ACTUAL / 2
                fhi = BW_ACTUAL / 2

            freq_mask = freq_range_to_mask(flo, fhi, LEN, FS)
            sig_filtered = filter_iq(sig_complex, freq_mask)

            # ── 计算细 DPD ──
            mtr = compute_fine_dpd(sig_complex, geo, freq_mask=freq_mask)

            mtr_log = torch.log(mtr + 1.0)

            # ── 位置标签（按距中心距离排序）──
            if n_src > 0:
                dists = np.linalg.norm(positions, axis=1)
                sort_idx = np.argsort(dists)
                positions = positions[sort_idx]

            pos_label = np.zeros((MAX_SRC, 2), dtype=np.float32)
            pos_label[:n_src] = positions
            pos_label_norm = pos_label / EDGE

            # ── 距离场双曲线标签 ──
            hyp_per_src, hyp_single = generate_hyp_labels(
                pos_label_norm, n_src, rcvPos, x_vec,
                sigma_m=args.hyp_sigma, mode=HYP_MODE)

            # ── 高斯热力图标签 ──
            gauss = generate_gauss_label(
                pos_label_norm, n_src, num_xy, EDGE, LAMDA_FINE,
                sigma_px=args.gauss_sigma)

            # ── 检查点 ──
            if i == idx_start:
                mtr_np = mtr.numpy()
                peak_flat = mtr_np.argmax()
                # 标准图像惯例: row=y, col=x
                peak_row = peak_flat // num_xy
                peak_col = peak_flat % num_xy
                peak_x = x_vec[peak_col]
                peak_y = x_vec[peak_row]
                if n_src > 0:
                    tp = positions[0] * EDGE if np.abs(positions[0]).max() <= 1 else positions[0]
                    err = np.sqrt((peak_x - tp[0])**2 + (peak_y - tp[1])**2)
                    print(f"    [CHECK] DPD peak=({peak_x:.0f},{peak_y:.0f}), "
                          f"true=({tp[0]:.0f},{tp[1]:.0f}), err={err:.0f}m")

            # ── 收集数据 ──
            dpd_list.append(mtr_log.unsqueeze(0).half())
            pos_list.append(torch.from_numpy(pos_label_norm))
            nsrc_list.append(n_src)
            sig_filt_real_list.append(sig_filtered.real.astype(np.float32))
            sig_filt_imag_list.append(sig_filtered.imag.astype(np.float32))
            hyp_per_src_list.append(torch.from_numpy(hyp_per_src))
            hyp_single_list.append(torch.from_numpy(hyp_single))
            gauss_list.append(torch.from_numpy(gauss))

        # ── 保存 .pt 文件 ──
        save_path = os.path.join(out_dir, f'exp_{exp_sub}_{param_str}.pt')

        save_dict = {
            # 共用
            'fine_dpd':          torch.stack(dpd_list),                              # (N, 1, H, W)
            'pos_label':         torch.stack(pos_list),                              # (N, 3, 2)
            'n_src':             torch.tensor(nsrc_list, dtype=torch.long),           # (N,)
            'param_value':       torch.tensor(pval, dtype=torch.float32),             # 标量

            # 传统方法
            'sig_filtered_real': torch.from_numpy(np.stack(sig_filt_real_list)),      # (N, 4, 4096)
            'sig_filtered_imag': torch.from_numpy(np.stack(sig_filt_imag_list)),      # (N, 4, 4096)
            'freq_lo':           torch.tensor(flo, dtype=torch.float32),
            'freq_hi':           torch.tensor(fhi, dtype=torch.float32),

            # DL 标签
            'hyp_mask':          torch.stack(hyp_per_src_list),                      # (N, 3, H, W)
            'hyp_mask_single':   torch.stack(hyp_single_list),                       # (N, 1, H, W)
            'gauss_label':       torch.stack(gauss_list),                            # (N, 1, H, W)

            # 元信息
            'exp_name':          exp_sub,
            'exp_type':          exp_type,
            'rcvPos':            torch.from_numpy(rcvPos.astype(np.float32)),
            'BW_actual':         torch.tensor(BW_ACTUAL_4D if exp_type == 'bw' else BW_ACTUAL, dtype=torch.float32),
        }

        torch.save(save_dict, save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"  保存: {save_path} ({size_mb:.0f} MB)")

    # ── 保存索引文件 ──
    def make_param_str(v, etype):
        if etype == 'snr':
            return f"snr{v:+03.0f}"
        elif etype == 'dist':
            return f"dist{v:04.0f}"
        elif etype == 'bw':
            return f"bw{v:04.0f}"
        else:
            return f"sep{v:04.0f}"

    index_path = os.path.join(out_dir, f'exp_{exp_sub}_index.pt')
    torch.save({
        'exp_name': exp_sub,
        'exp_type': exp_type,
        'param_values': param_values.tolist(),
        'n_per_param': n_per_param,
        'param_files': [f'exp_{exp_sub}_{make_param_str(v, exp_type)}.pt'
                        for v in param_values],
        'grid_params': {'edge': EDGE, 'lamda': LAMDA_FINE, 'num_xy': num_xy},
        'hyp_sigma': args.hyp_sigma,
        'gauss_sigma': args.gauss_sigma,
    }, index_path)

    print(f"\n{'='*50}")
    print(f"实验 {args.exp} 处理完毕")
    print(f"索引: {index_path}")
    print(f"共 {len(param_values)} 个参数值 × {n_per_param} 样本 = {N_total} 样本")
    print("Done!")


if __name__ == '__main__':
    main()
