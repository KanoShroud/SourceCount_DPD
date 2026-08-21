"""
eval_exp.py — 评估单个方法在控制变量实验上的表现

用法:
  python eval_exp.py --exp 4A2 --method dpd_peak
  python eval_exp.py --exp 4A2 --method D1
  python eval_exp.py --exp 4A2 --method D2
  python eval_exp.py --exp 4A2 --method D3
  python eval_exp.py --exp 4A2 --method D5
  python eval_exp.py --exp 4A2 --method D6
"""

import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy.optimize import linear_sum_assignment
from yolo_config import *

from chapter_runtime import (
    checkpoint_path,
    data_dir as runtime_data_dir,
    device as runtime_device,
    output_dir as runtime_output_dir,
)


def pixel_to_phys(px):
    return px * LAMDA - EDGE


def hungarian_match_eval(pred_pos, true_pos, n_true):
    if len(pred_pos) == 0 or n_true == 0:
        return np.array([9999.0] * max(n_true, 1))
    K = len(pred_pos)
    cost = np.zeros((n_true, K))
    for t in range(n_true):
        for p in range(K):
            cost[t, p] = np.linalg.norm(pred_pos[p] - true_pos[t])
    row_ind, col_ind = linear_sum_assignment(cost)
    errors = [cost[r, c] for r, c in zip(row_ind, col_ind)]
    if len(errors) < n_true:
        errors.extend([9999.0] * (n_true - len(errors)))
    return np.array(errors)


def compute_metrics(errors):
    errors = np.array(errors)
    return {
        'rmse': float(np.sqrt((errors**2).mean())),
        'mean': float(errors.mean()),
        'median': float(np.median(errors)),
        'within_10m': float((errors < 10).mean()),
        'within_30m': float((errors < 30).mean()),
        'within_50m': float((errors < 50).mean()),
        'n_samples': len(errors),
        'errors': errors.tolist(),
    }


# ═══════════════════════════════════════
#  T1: DPD 峰值搜索
# ═══════════════════════════════════════
def nms_2d(heatmap, kernel_size):
    pad = kernel_size // 2
    if isinstance(heatmap, np.ndarray):
        heatmap = torch.from_numpy(heatmap).float()
    hm = heatmap.unsqueeze(0).unsqueeze(0)
    hmax = F.max_pool2d(hm, kernel_size, stride=1, padding=pad)
    return (heatmap * (hm.squeeze() == hmax.squeeze()).float()).numpy()


def eval_dpd_peak(fine_dpd, pos_label, n_src, peak_size):
    dpd = fine_dpd.squeeze().numpy()
    dpd_nms = nms_2d(dpd, peak_size)
    flat = dpd_nms.flatten()
    top_idx = np.argsort(flat)[::-1][:n_src]
    peak_row = top_idx // dpd.shape[1]
    peak_col = top_idx % dpd.shape[1]
    pred_phys = np.stack([peak_col * LAMDA - EDGE, peak_row * LAMDA - EDGE], axis=1)
    true_phys = pos_label[:n_src].numpy() * EDGE
    return hungarian_match_eval(pred_phys, true_phys, n_src)


# ═══════════════════════════════════════
#  T2: TDOA 两步法
# ═══════════════════════════════════════
def eval_tdoa(sig_filtered_real, sig_filtered_imag, pos_label, n_src,
              freq_lo, freq_hi, rcvPos, fs=100e6):
    try:
        from tdoa_twostep import tdoa_locate
        sig = sig_filtered_real + 1j * sig_filtered_imag
        band = freq_hi - freq_lo
        pred_pos = tdoa_locate(sig, rcvPos, fs, band)
        true_phys = pos_label[:n_src].numpy() * EDGE
        pred_phys = pred_pos.reshape(1, 2)
        return hungarian_match_eval(pred_phys, true_phys, n_src)
    except ImportError:
        return np.array([9999.0] * n_src)


# ═══════════════════════════════════════
#  T3: 子空间 DPD
# ═══════════════════════════════════════
def eval_music_dpd(sig_filtered_real, sig_filtered_imag, pos_label, n_src,
                   rcvPos, fc=5.8e9, fs=100e6, peak_size=PEAK_SIZE):
    try:
        from dpd_subspace import subspace_dpd_locate
        sig = sig_filtered_real + 1j * sig_filtered_imag
        pred_pos = subspace_dpd_locate(sig, rcvPos, n_src, fc, fs, peak_size)
        true_phys = pos_label[:n_src].numpy() * EDGE
        return hungarian_match_eval(pred_pos, true_phys, n_src)
    except ImportError:
        return np.array([9999.0] * n_src)


# ═══════════════════════════════════════
#  D1-D6: 深度学习方法
# ═══════════════════════════════════════
def eval_dl_method(model, fine_dpd, pos_label, n_src, method, peak_size, device):
    from yolo_model import extract_peaks_topn, decode_bbox_topn

    dpd = fine_dpd.unsqueeze(0).to(device).float()   # float16→float32
    mu = dpd.mean(); std = dpd.std() + 1e-6
    dpd = (dpd - mu) / std

    with torch.no_grad():
        if method == 'D1':
            cls_list, reg_list = model(dpd)
            centers, scores = decode_bbox_topn(cls_list, reg_list, n_src, model.head, peak_size)
            pred_phys = pixel_to_phys(centers).cpu().numpy()
        elif method in ('D3d', 'D4', 'D5', 'D6', 'D6s', 'D7', 'D8'):
            # D5 和 D6 推理逻辑完全一样: heatmap + offset 修正
            pred_hm, pred_offset = model(dpd)
            peaks, scores = extract_peaks_topn(pred_hm[0, 0].cpu(), n_src, peak_size)
            for i in range(peaks.shape[0]):
                ix = int(peaks[i, 0].item())
                iy = int(peaks[i, 1].item())
                ix = max(0, min(pred_offset.shape[3] - 1, ix))
                iy = max(0, min(pred_offset.shape[2] - 1, iy))
                dx = pred_offset[0, 0, iy, ix].cpu().float()
                dy = pred_offset[0, 1, iy, ix].cpu().float()
                if torch.isfinite(dx) and torch.isfinite(dy):
                    peaks[i, 0] += dx.clamp(-1, 1)
                    peaks[i, 1] += dy.clamp(-1, 1)
            pred_phys = pixel_to_phys(peaks).numpy()
        else:
            # D2, D3: 纯 heatmap
            pred_hm = model(dpd)
            peaks, scores = extract_peaks_topn(pred_hm[0, 0].cpu(), n_src, peak_size)
            pred_phys = pixel_to_phys(peaks).numpy()

    true_phys = pos_label[:n_src].numpy() * EDGE
    return hungarian_match_eval(pred_phys, true_phys, n_src)


# ═══════════════════════════════════════
#  加载实验数据
# ═══════════════════════════════════════
def load_exp_data(data_dir, exp_name):
    if exp_name.startswith('4A'):
        exp_type, sub_dir = 'snr', 'exp_4A'
    elif exp_name.startswith('4B'):
        exp_type, sub_dir = 'dist', 'exp_4B'
    elif exp_name == '4C':
        exp_type, sub_dir = 'sep', 'exp_4C'
    elif exp_name == '4D':
        exp_type, sub_dir = 'bw', 'exp_4D'
    else:
        raise ValueError(f"Unknown experiment: {exp_name}")
    index_path = os.path.join(data_dir, 'exp', sub_dir, f'exp_{exp_name}_index.pt')
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件不存在: {index_path}")
    index = torch.load(index_path, weights_only=False)
    return index, exp_type, os.path.join(data_dir, 'exp', sub_dir)


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--exp', type=str, required=True)
    pa.add_argument('--method', type=str, required=True,
                    choices=['dpd_peak', 'tdoa', 'music_dpd',
                             'D1', 'D2', 'D3', 'D3d', 'D4', 'D5', 'D6', 'D6s', 'D7', 'D8'])
    pa.add_argument('--data_dir', type=str, default=str(runtime_data_dir()))
    pa.add_argument('--model_path', type=str, default=None)
    pa.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    pa.add_argument('--peak_size', type=int, default=PEAK_SIZE)
    pa.add_argument('--results_dir', type=str, default=None)
    args = pa.parse_args()
    results_dir = args.results_dir or str(runtime_output_dir('eval_exp'))

    SAVE_NAME = {
        'dpd_peak':   'DPD_Peak_Search',
        'tdoa':       'Chan-TDOA',
        'music_dpd':  'MUSIC-DPD',
        'D1':         'DL-BBox',
        'D2':         'DL-Gaussian',
        'D3':         'DL-Single-DistField',
        'D3d':        'DL-DistField+Offset',
        'D4':         'DL-GradAtten-Offset',
        'D5':         'DL-Gaussian+Offset',
        'D6':         'DL-ConfWeight-Offset',
        'D6s':        'DL-SoftConfWeight-Offset',
        'D7':         'DL-GradAtten+ConfWeight',
        'D8':         'DL-StdCenterNet',
    }
    save_name = SAVE_NAME.get(args.method, args.method)
    print(f"实验: {args.exp}, 方法: {args.method} → 保存为 {save_name}")

    if args.method == 'tdoa' and args.exp not in ['4A1', '4B1']:
        print(f"  [INFO] TDOA 仅支持单源，跳过 {args.exp}"); return

    index, exp_type, exp_dir = load_exp_data(args.data_dir, args.exp)
    param_values = index['param_values']
    param_files = index['param_files']
    n_per_param = index['n_per_param']

    param_name_map = {'snr': 'SNR (dB)', 'dist': 'Distance (m)', 'sep': 'Separation (m)', 'bw': 'Src2 BW (MHz)'}
    param_name = param_name_map.get(exp_type, exp_type)
    print(f"参数: {param_name} = {param_values}")
    print(f"每参数 {n_per_param} 样本")

    # 加载 DL 模型
    model = None
    device = runtime_device(args.device)

    if args.method == 'music_dpd':
        from dpd_subspace import set_device
        set_device(args.device)

    if args.method in ['D1', 'D2', 'D3', 'D3d', 'D4', 'D5', 'D6', 'D6s', 'D7', 'D8']:
        method_map = {
            'D1': 'bbox', 'D2': 'heatmap', 'D3': 'heatmap', 'D3d': 'dualhead',
            'D4': 'dualhead', 'D5': 'dualhead', 'D6': 'dualhead', 'D6s': 'dualhead',
            'D7': 'dualhead', 'D8': 'dualhead',
        }
        if args.model_path is None:
            save_map = {
                'D1': 'bbox', 'D2': 'gauss', 'D3': 'distfield', 'D3d': 'distfield_dual',
                'D4': 'dualhead_ga', 'D5': 'dualhead', 'D6': 'dualhead_cw',
                'D6s': 'dualhead_cws', 'D7': 'dualhead_ga_cw', 'D8': 'dualhead_std',
            }
            args.model_path = checkpoint_path(
                'train_yolo', f'best_yolo_{save_map[args.method]}.pth'
            )
            print(f"  自动推断模型路径: {args.model_path}")

        from yolo_model import YOLOv8Loc
        model = YOLOv8Loc(method=method_map[args.method]).to(device)
        ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model.eval()
        print(f"  模型已加载: {args.model_path}")

    # 逐参数值评估
    all_results = {}

    for pi, (pval, pfile) in enumerate(zip(param_values, param_files)):
        pt_path = os.path.join(exp_dir, pfile)
        if not os.path.exists(pt_path):
            print(f"  [SKIP] 文件不存在: {pt_path}"); continue

        data = torch.load(pt_path, weights_only=False)
        fine_dpd = data['fine_dpd']
        pos_label = data['pos_label']
        n_src_all = data['n_src']
        N = len(n_src_all)

        has_iq = 'sig_filtered_real' in data
        if has_iq:
            sig_filt_r = data['sig_filtered_real']
            sig_filt_i = data['sig_filtered_imag']
            freq_lo = data['freq_lo'].item()
            freq_hi = data['freq_hi'].item()
            rcvPos = data['rcvPos'].numpy()

        errors_all = []
        for i in range(N):
            n = n_src_all[i].item()
            if n == 0: continue

            if args.method == 'dpd_peak':
                errs = eval_dpd_peak(fine_dpd[i], pos_label[i], n, args.peak_size)
            elif args.method == 'tdoa':
                errs = eval_tdoa(sig_filt_r[i].numpy(), sig_filt_i[i].numpy(),
                                 pos_label[i], n, freq_lo, freq_hi, rcvPos) if has_iq else np.array([9999.0]*n)
            elif args.method == 'music_dpd':
                errs = eval_music_dpd(sig_filt_r[i].numpy(), sig_filt_i[i].numpy(),
                                      pos_label[i], n, rcvPos, peak_size=args.peak_size) if has_iq else np.array([9999.0]*n)
            elif args.method in ['D1', 'D2', 'D3', 'D3d', 'D4', 'D5', 'D6', 'D6s', 'D7', 'D8']:
                errs = eval_dl_method(model, fine_dpd[i], pos_label[i], n,
                                      args.method, args.peak_size, device)
            else:
                errs = np.array([9999.0] * n)

            errors_all.extend(errs.tolist())

        metrics = compute_metrics(errors_all)
        all_results[pval] = metrics

        print(f"  {param_name}={pval:>6}: "
              f"RMSE={metrics['rmse']:.1f}m  mean={metrics['mean']:.1f}m  "
              f"med={metrics['median']:.1f}m  "
              f"<10m={metrics['within_10m']:.1%}  <30m={metrics['within_30m']:.1%}  <50m={metrics['within_50m']:.1%}  "
              f"({metrics['n_samples']} samples)", flush=True)

    save_dir = os.path.join(results_dir, f'exp_{args.exp}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{save_name}.pt')

    torch.save({
        'method': save_name, 'exp_name': args.exp, 'exp_type': exp_type,
        'param_name': param_name, 'param_values': param_values,
        'peak_size': args.peak_size, 'per_param': all_results,
    }, save_path)

    print(f"\n结果已保存: {save_path}")
    print("Done!")


if __name__ == '__main__':
    main()
