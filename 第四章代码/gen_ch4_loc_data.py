"""
gen_ch4_loc_data.py — 生成第四章定位数据集

流程:
  1. 读 IQ 信号 + fc_offset + 真值位置
  2. 根据实际频率范围分组（重叠合并，不重叠独立）
  3. 每组: IQ 按实际频率范围滤波 → 细 DPD(10m, 401×401)
  4. 每组: 计算距离场双曲线标签（σ=1.5px, sum 模式）
  5. 位置标签按距中心距离排序
  6. 每 SHARD_SIZE 个任务保存一个 .pt 文件

用法:
  python gen_ch4_loc_data.py --data_dir /mnt/data/ltzdata_loc --split train
  python gen_ch4_loc_data.py --data_dir /mnt/data/ltzdata_loc --split val
  python gen_ch4_loc_data.py --data_dir /mnt/data/ltzdata_loc --split test
"""

import numpy as np
import h5py
import torch
import os
import argparse
from dpd_calculator_torch import DPDGeometry, compute_fine_dpd, compute_hyperbola_mask


# ═══════════════════════════════════════
#  系统参数
# ═══════════════════════════════════════
FS           = 100e6
SYMBOL_RATE  = 10e6
ARFA_V       = 0.25
BW_ACTUAL    = SYMBOL_RATE * (1 + ARFA_V * 1.2)   # = 13MHz 实际信号带宽（含滚降）
LEN          = 4096
EDGE         = 2000
LAMDA_FINE   = 10       # 细网格步长
MAX_SRC      = 3        # 最大源数
SHARD_SIZE   = 2000     # 每个分片的任务数


# ═══════════════════════════════════════
#  根据实际频率范围分组（并查集）
# ═══════════════════════════════════════
def group_sources_by_freq_overlap(fc_offsets, bw_each, n_src):
    """
    按逐源实际频率范围的重叠关系分组

    输入:
        fc_offsets: (n_src,) 每个源的中心频偏
        bw_each:    (n_src,) 每个源的实际带宽
        n_src:      源数量

    返回:
        groups: list of dict
    """
    if n_src == 0:
        return []

    # 每个源的频率范围（逐源带宽）
    flo = fc_offsets[:n_src] - bw_each[:n_src] / 2
    fhi = fc_offsets[:n_src] + bw_each[:n_src] / 2

    # 并查集
    parent = list(range(n_src))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 频率范围有交集 → 合并
    for s1 in range(n_src):
        for s2 in range(s1 + 1, n_src):
            if flo[s1] < fhi[s2] and flo[s2] < fhi[s1]:
                union(s1, s2)

    # 构建分组
    group_dict = {}
    for s in range(n_src):
        root = find(s)
        if root not in group_dict:
            group_dict[root] = []
        group_dict[root].append(s)

    groups = []
    for root, slots in group_dict.items():
        group_flo = min(flo[s] for s in slots)
        group_fhi = max(fhi[s] for s in slots)
        groups.append({
            'slots': slots,
            'freq_lo': group_flo,
            'freq_hi': group_fhi,
            'n_src': len(slots),
        })

    return groups


# ═══════════════════════════════════════
#  构造频域掩码（实际频率范围）
# ═══════════════════════════════════════
def freq_range_to_mask(freq_lo, freq_hi, N0, fs):
    """从频率范围构造频域掩码"""
    f_axis = np.arange(-N0//2, N0//2) * (fs / N0)
    return (f_axis >= freq_lo) & (f_axis < freq_hi)


# ═══════════════════════════════════════
#  保存一个分片
# ═══════════════════════════════════════
def save_shard(shard_data, save_dir, split, shard_idx):
    """将一批任务保存为一个 .pt 文件"""
    split_dir = os.path.join(save_dir, split)
    os.makedirs(split_dir, exist_ok=True)
    save_path = os.path.join(split_dir, f'loc_{split}_{shard_idx:03d}.pt')

    torch.save({
        'fine_dpd':    torch.stack(shard_data['fine_dpd']),        # (K, 1, H, W)
        'hyp_mask':    torch.stack(shard_data['hyp_mask']),        # (K, 3, H, W)
        'gauss_label': torch.stack(shard_data['gauss_label']),     # (K, 1, H, W)
        'gauss_multi': torch.stack(shard_data['gauss_multi']),     # (K, 3, H, W)
        'pos_label':   torch.stack(shard_data['pos_label']),       # (K, 3, 2)
        'n_src':       torch.tensor(shard_data['n_src'], dtype=torch.long),
        'sample_idx':  torch.tensor(shard_data['sample_idx'], dtype=torch.long),
        'group_idx':   torch.tensor(shard_data['group_idx'], dtype=torch.long),
    }, save_path)

    size_mb = os.path.getsize(save_path) / 1e6
    n = len(shard_data['n_src'])
    print(f"    Saved shard {shard_idx:03d}: {n} tasks, {size_mb:.0f} MB → {save_path}")
    return save_path


def new_shard_data():
    """创建空的分片数据容器"""
    return {
        'fine_dpd': [], 'hyp_mask': [], 'gauss_label': [], 'gauss_multi': [],
        'pos_label': [], 'n_src': [], 'sample_idx': [], 'group_idx': [],
    }


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/data/ltzdata_loc')
    parser.add_argument('--split', type=str, required=True,
                        choices=['train', 'val', 'test'])
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--edge', type=float, default=EDGE)
    parser.add_argument('--lamda', type=float, default=LAMDA_FINE)
    parser.add_argument('--shard_size', type=int, default=SHARD_SIZE)
    parser.add_argument('--hyp_sigma', type=float, default=15.0,
                        help='距离场高斯σ（米），1.5像素=15m')
    parser.add_argument('--hyp_mode', type=str, default='sum',
                        choices=['max', 'sum'], help='双曲线叠加模式')
    parser.add_argument('--gauss_sigma', type=float, default=2.0,
                        help='高斯热力图σ（像素）')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Fine grid: ±{args.edge}m, step {args.lamda}m")
    print(f"Shard size: {args.shard_size} tasks per file")
    print(f"Hyperbola: σ={args.hyp_sigma}m ({args.hyp_sigma/args.lamda:.1f}px), mode={args.hyp_mode}")

    # 网格参数
    num_xy = int(round(2 * args.edge / args.lamda)) + 1
    x_vec = np.linspace(-args.edge, args.edge, num_xy)
    print(f"DPD grid: {num_xy}×{num_xy}")

    # ── 加载 IQ 信号 + 标签 ──
    mat_path = os.path.join(args.data_dir, f'{args.split}_data.mat')
    print(f"\nLoading: {mat_path}")
    with h5py.File(mat_path, 'r') as f:
        sig_real   = np.array(f['sig_rcv_real_all'], dtype=np.float32)
        sig_imag   = np.array(f['sig_rcv_imag_all'], dtype=np.float32)
        src_count  = np.array(f['src_count_all'], dtype=np.int64).flatten()
        src_pos    = np.array(f['src_pos_all'], dtype=np.float32)
        fc_offset  = np.array(f['fc_offset_all'], dtype=np.float32)

        # 逐源带宽
        if 'BW_actual_all' in f:
            bw_actual = np.array(f['BW_actual_all'], dtype=np.float32)
            has_per_src_bw = True
            print("  [v3] 逐源带宽: BW_actual_all 已加载")
        else:
            bw_actual = None
            has_per_src_bw = False
            BW_FALLBACK = 10e6 * (1 + 0.25 * 1.2)
            print(f"  [兼容] 无 BW_actual_all, 使用固定 BW={BW_FALLBACK/1e6:.1f}MHz")

    # h5py 转置（MATLAB → Python）
    sig_real  = sig_real.transpose(2, 1, 0)      # (N, rcv_num, len)
    sig_imag  = sig_imag.transpose(2, 1, 0)
    src_pos   = src_pos.transpose(2, 1, 0)       # (N, max_src, 2)
    fc_offset = fc_offset.transpose(1, 0)        # (N, max_src)
    if has_per_src_bw:
        bw_actual = bw_actual.transpose(1, 0)    # (N, max_src)

    N = len(src_count)
    rcv_num = sig_real.shape[1]
    print(f"  {N} samples, {rcv_num} stations")

    # ── 检查点: 数据完整性 ──
    print(f"  sig_real shape: {sig_real.shape}")
    print(f"  src_pos shape: {src_pos.shape}")
    print(f"  fc_offset shape: {fc_offset.shape}")
    for i in range(N):
        if src_count[i] > 0:
            iq_power = (sig_real[i]**2 + sig_imag[i]**2).mean()
            n = int(src_count[i])
            print(f"  Sample {i} ({n}src) IQ power: {iq_power:.2e}")
            print(f"  Sample {i} src_pos: {src_pos[i, :n, :]}")
            fco = fc_offset[i, :n]
            print(f"  Sample {i} fc_offset: {fco/1e6} MHz")
            if has_per_src_bw:
                bws = bw_actual[i, :n]
                flo = fco - bws/2
                fhi = fco + bws/2
                print(f"  Sample {i} BW_actual: {bws/1e6} MHz")
            else:
                flo = fco - BW_FALLBACK/2
                fhi = fco + BW_FALLBACK/2
            print(f"  Sample {i} freq range: [{flo/1e6}MHz ~ {fhi/1e6}MHz]")
            break

    # 接收站坐标
    R_rcv = 500.0
    angles = np.arange(4) * 2 * np.pi / 4
    rcvPos = np.stack([R_rcv * np.cos(angles), R_rcv * np.sin(angles)], axis=1)

    # DPD 几何信息预计算
    print(f"\nPrecomputing DPD geometry...")
    geo = DPDGeometry(rcvPos, [0.0, 0.0], args.edge, args.lamda, FS, LEN, device)

    # ── 生成定位任务（分片保存）──
    print(f"\nGenerating localization tasks (actual bandwidth filtering)...")

    shard_data = new_shard_data()
    shard_idx = 0
    shard_files = []
    n_total_tasks = 0
    n_skip_0src = 0
    task_n_src_counts = {1: 0, 2: 0, 3: 0}

    for i in range(N):
        if i % 500 == 0:
            print(f"  [{i}/{N}] tasks: {n_total_tasks}, shards: {shard_idx}")

        # 跳过 0 源
        if src_count[i] == 0:
            n_skip_0src += 1
            continue

        n_src = int(src_count[i])
        positions = src_pos[i, :n_src, :]
        fc_offs = fc_offset[i, :n_src]

        # 逐源带宽
        if has_per_src_bw:
            bw_each = bw_actual[i, :n_src]
        else:
            bw_each = np.full(n_src, BW_FALLBACK, dtype=np.float32)

        # 构造 IQ 复信号
        sig_complex = sig_real[i] + 1j * sig_imag[i]

        # 按逐源频率范围分组
        groups = group_sources_by_freq_overlap(fc_offs, bw_each, n_src)

        # ── 检查点: 前 30 个多源样本打印分组结果 ──
        if n_src > 1 and n_total_tasks < 30:
            print(f"\n  [CHECK] Sample {i}: {n_src}src")
            for s in range(n_src):
                flo_s = (fc_offs[s] - bw_each[s]/2) / 1e6
                fhi_s = (fc_offs[s] + bw_each[s]/2) / 1e6
                print(f"    src{s}: fc_off={fc_offs[s]/1e6:.1f}MHz "
                      f"BW={bw_each[s]/1e6:.1f}MHz [{flo_s:.1f}, {fhi_s:.1f}]MHz "
                      f"pos=({positions[s,0]:.0f},{positions[s,1]:.0f})")
            for gi2, g2 in enumerate(groups):
                print(f"    → Group{gi2}: slots={g2['slots']} N={g2['n_src']} "
                      f"freq=[{g2['freq_lo']/1e6:.1f}, {g2['freq_hi']/1e6:.1f}]MHz")

        for gi, group in enumerate(groups):
            slots = group['slots']
            n_src_group = group['n_src']

            # 频域掩码（实际频率范围）
            freq_mask = freq_range_to_mask(group['freq_lo'], group['freq_hi'], LEN, FS)
            if not freq_mask.any():
                continue

            # 计算细 DPD
            mtr = compute_fine_dpd(sig_complex, geo, freq_mask=freq_mask)

            # ── 检查点: 前 10 个任务验证 DPD 峰值位置 ──
            if n_total_tasks < 10:
                mtr_np = mtr.numpy()
                peak_flat = mtr_np.argmax()
                # 标准图像惯例: row=y, col=x
                peak_row = peak_flat // num_xy   # row → y
                peak_col = peak_flat % num_xy    # col → x
                peak_x = x_vec[peak_col]
                peak_y = x_vec[peak_row]
                group_pos_check = positions[slots]
                for sc in range(n_src_group):
                    tp = group_pos_check[sc]
                    err = np.sqrt((peak_x - tp[0])**2 + (peak_y - tp[1])**2)
                    print(f"  [CHECK] Task {n_total_tasks}: DPD peak=({peak_x:.0f},{peak_y:.0f}), "
                          f"true=({tp[0]:.0f},{tp[1]:.0f}), err={err:.0f}m")

            # log 变换
            mtr_log = torch.log(mtr + 1.0)

            # 位置标签（按距中心距离排序）
            group_pos = positions[slots]
            dists = np.linalg.norm(group_pos, axis=1)
            sort_idx = np.argsort(dists)
            group_pos = group_pos[sort_idx]

            pos_label = np.zeros((MAX_SRC, 2), dtype=np.float32)
            pos_label[:n_src_group] = group_pos
            pos_label_norm = pos_label / args.edge

            # ── 检查点: 坐标范围 ──
            if np.abs(pos_label_norm[:n_src_group]).max() > 1.0:
                print(f"  [WARNING] Sample {i} Group {gi}: pos out of [-1,1]: "
                      f"{pos_label_norm[:n_src_group]}")

            # 每源独立的距离场双曲线标签 (MAX_SRC, num_xy, num_xy)
            hyp_mask_per_src = np.zeros((MAX_SRC, num_xy, num_xy), dtype=np.float32)
            for s_local in range(n_src_group):
                s_pos = group_pos[s_local]
                hm = compute_hyperbola_mask(
                    s_pos, rcvPos, x_vec, x_vec,
                    tolerance_m=args.hyp_sigma, mode=args.hyp_mode,
                    precomputed_dists=geo.grid_dist_to_station)
                hyp_mask_per_src[s_local] = hm

            # 高斯热力图标签 (1, num_xy, num_xy) — 所有源叠加单通道
            # 标准图像惯例: gx=row=y方向, gy=col=x方向
            ix = np.arange(num_xy, dtype=np.float32)
            gx, gy = np.meshgrid(ix, ix, indexing='ij')
            gauss_map = np.zeros((num_xy, num_xy), dtype=np.float32)
            for s_local in range(n_src_group):
                px = (group_pos[s_local, 0] + args.edge) / args.lamda   # x → col
                py = (group_pos[s_local, 1] + args.edge) / args.lamda   # y → row
                g = np.exp(-((gx - py)**2 + (gy - px)**2) / (2 * args.gauss_sigma**2))
                gauss_map = np.maximum(gauss_map, g)
            mx = gauss_map.max()
            if mx > 0:
                gauss_map = gauss_map / mx

            # 多通道高斯标签 (MAX_SRC, num_xy, num_xy) — 每源独立通道
            gauss_multi = np.zeros((MAX_SRC, num_xy, num_xy), dtype=np.float32)
            for s_local in range(n_src_group):
                px = (group_pos[s_local, 0] + args.edge) / args.lamda
                py = (group_pos[s_local, 1] + args.edge) / args.lamda
                g = np.exp(-((gx - py)**2 + (gy - px)**2) / (2 * args.gauss_sigma**2))
                gauss_multi[s_local] = g  # 不归一化，高斯峰值自然为 1.0

            # 添加到当前分片
            shard_data['fine_dpd'].append(mtr_log.unsqueeze(0).half())
            shard_data['hyp_mask'].append(torch.from_numpy(hyp_mask_per_src).half())
            shard_data['gauss_label'].append(torch.from_numpy(gauss_map[np.newaxis, :, :]).half())
            shard_data['gauss_multi'].append(torch.from_numpy(gauss_multi).half())
            shard_data['pos_label'].append(torch.from_numpy(pos_label_norm))
            shard_data['n_src'].append(n_src_group)
            shard_data['sample_idx'].append(i)
            shard_data['group_idx'].append(gi)

            n_total_tasks += 1
            task_n_src_counts[n_src_group] = task_n_src_counts.get(n_src_group, 0) + 1

            # 分片满了 → 保存并清空
            if len(shard_data['n_src']) >= args.shard_size:
                path = save_shard(shard_data, args.data_dir, args.split, shard_idx)
                shard_files.append(path)
                shard_data = new_shard_data()
                shard_idx += 1

    # 保存剩余任务
    if len(shard_data['n_src']) > 0:
        path = save_shard(shard_data, args.data_dir, args.split, shard_idx)
        shard_files.append(path)
        shard_idx += 1

    # ── 保存索引文件 ──
    split_dir = os.path.join(args.data_dir, args.split)
    index_path = os.path.join(split_dir, f'loc_{args.split}_index.pt')
    torch.save({
        'shard_files': [os.path.basename(f) for f in shard_files],
        'n_total_tasks': n_total_tasks,
        'n_shards': shard_idx,
        'grid_params': {'edge': args.edge, 'lamda': args.lamda, 'num_xy': num_xy},
        'filter_mode': 'per_source_bandwidth' if has_per_src_bw else 'fixed_bandwidth',
        'hyp_sigma': args.hyp_sigma,
        'hyp_mode': args.hyp_mode,
        'gauss_sigma': args.gauss_sigma,
    }, index_path)

    # ── 统计 ──
    print(f"\n{'='*50}")
    print(f"  {args.split} set: {n_total_tasks} tasks in {shard_idx} shards")
    print(f"  Skipped 0-src: {n_skip_0src}")
    for ns in sorted(task_n_src_counts.keys()):
        print(f"    N={ns}: {task_n_src_counts[ns]} tasks")

    total_size = sum(os.path.getsize(f) for f in shard_files) / 1e9
    print(f"  Total size: {total_size:.2f} GB")
    print(f"  Index: {index_path}")
    print("Done!")


if __name__ == '__main__':
    main()