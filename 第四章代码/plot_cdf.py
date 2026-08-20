#!/usr/bin/env python3
"""
plot_cdf.py — 生成对比实验和消融实验的CDF曲线图（分开出图）

对比CDF (放4.3.1): Peak Search, EPN, Proposed Net — 横轴0-100m
消融CDF (放4.3.2): YOLOv8, Proposed Net — 横轴0-50m

用法:
  python plot_cdf.py --device cuda:2 --data_dir /mnt/data/ltzdata_loc
  python plot_cdf.py --device cuda:2 --data_dir /mnt/data/ltzdata_loc --exp 4A3
  python plot_cdf.py --device cuda:2 --data_dir /mnt/data/ltzdata_loc --snr_list -6 0 6
"""

import os, argparse, torch, numpy as np

from eval_exp import (eval_dpd_peak, eval_dl_method, hungarian_match_eval,
                       load_exp_data)
from yolo_config import *
from train_epn import EPNResNet, COORD_MAX, MAX_SRC

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'font.size': 12,
    'axes.labelsize': 13, 'legend.fontsize': 10,
    'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

COL_DPD = '#666666'
COL_EPN = '#E67E22'
COL_D1  = '#2980B9'
COL_D8  = '#C0392B'


def eval_epn_sample(model, fine_dpd, pos_label, n_src, device):
    dpd = fine_dpd.unsqueeze(0).to(device).float()
    mu = dpd.mean(); std = dpd.std() + 1e-6
    dpd = (dpd - mu) / std
    with torch.no_grad():
        pred = model(dpd)
    pred_m = pred[0].cpu().numpy().reshape(MAX_SRC, 2) * COORD_MAX
    true_m = pos_label[:n_src].numpy() * COORD_MAX
    return hungarian_match_eval(pred_m[:n_src], true_m, n_src)


def collect_errors(method_name, model, fine_dpd, pos_label, n_src_all,
                   peak_size, device):
    all_errors = []
    N = len(n_src_all)
    for i in range(N):
        n = n_src_all[i].item()
        if n == 0:
            continue
        if method_name == 'dpd_peak':
            errs = eval_dpd_peak(fine_dpd[i], pos_label[i], n, peak_size)
        elif method_name == 'EPN':
            errs = eval_epn_sample(model, fine_dpd[i], pos_label[i], n, device)
        else:
            errs = eval_dl_method(model, fine_dpd[i], pos_label[i], n,
                                  method_name, peak_size, device)
        all_errors.extend(errs.tolist())
    return np.array(all_errors)


def plot_single_cdf(errors_dict, method_keys, styles, snr_val, exp_name,
                    tag, max_x):
    fig, ax = plt.subplots(figsize=(5.5, 4))

    for method_name in method_keys:
        errors = errors_dict[method_name]
        sorted_errors = np.sort(errors)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        if max_x is not None and sorted_errors[-1] < max_x:
            sorted_errors = np.append(sorted_errors, max_x)
            cdf = np.append(cdf, cdf[-1])
        ax.plot(sorted_errors, cdf * 100, **styles[method_name])

    ax.set_xlabel('Localization Error (m)')
    ax.set_ylabel('CDF (%)')
    if max_x is not None:
        ax.set_xlim([0, max_x])
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc')

    fig.tight_layout()
    fname = f'fig_{exp_name}_cdf_{tag}_snr{snr_val:+d}dB'
    fig.savefig(f'{fname}.png')
    fig.savefig(f'{fname}.pdf')
    print(f'  已保存: {fname}.png')
    plt.close(fig)


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--device', default='cuda:0')
    pa.add_argument('--data_dir', required=True)
    pa.add_argument('--peak_size', type=int, default=PEAK_SIZE)
    pa.add_argument('--exp', default='4A2', choices=['4A2', '4A3'])
    pa.add_argument('--snr_list', nargs='+', type=int, default=[-6, 0, 6])
    pa.add_argument('--comparison_max_x', type=float, default=100,
                    help='对比CDF横轴上限 (m)')
    pa.add_argument('--ablation_max_x', type=float, default=50,
                    help='消融CDF横轴上限 (m)')
    args = pa.parse_args()
    device = torch.device(args.device)

    # ── 加载模型 ──
    print("加载模型...")
    from yolo_model import YOLOv8Loc

    model_d1 = YOLOv8Loc(method='bbox').to(device)
    ckpt = torch.load('best_yolo_bbox.pth', map_location=device, weights_only=False)
    model_d1.load_state_dict(ckpt['model'])
    model_d1.eval()
    print("  D1 (YOLOv8) loaded")

    model_d8 = YOLOv8Loc(method='dualhead').to(device)
    ckpt = torch.load('best_yolo_dualhead_std.pth', map_location=device, weights_only=False)
    model_d8.load_state_dict(ckpt['model'])
    model_d8.eval()
    print("  D8 (Proposed Net) loaded")

    model_epn = EPNResNet(max_src=MAX_SRC, dropout=0.0).to(device)
    ckpt = torch.load('best_epn.pth', map_location=device, weights_only=False)
    model_epn.load_state_dict(ckpt)
    model_epn.eval()
    print("  EPN loaded")

    # ── 方法列表 ──
    all_methods = [
        ('dpd_peak', None),
        ('EPN',      model_epn),
        ('D1',       model_d1),
        ('D8',       model_d8),
    ]

    # ── 样式 ──
    styles = {
        'dpd_peak': dict(color=COL_DPD, linewidth=1.5, linestyle='--',
                         label='Peak Search'),
        'EPN':      dict(color=COL_EPN, linewidth=1.5, linestyle='-.',
                         label='EPN'),
        'D1':       dict(color=COL_D1,  linewidth=1.5, linestyle='--',
                         label='YOLOv8'),
        'D8':       dict(color=COL_D8,  linewidth=2.0, linestyle='-',
                         label='Proposed Net'),
    }

    # ── 对比实验方法 / 消融实验方法 ──
    comparison_keys = ['dpd_peak', 'EPN', 'D8']
    ablation_keys   = ['D1', 'D8']

    # ── 加载实验数据索引 ──
    index, exp_type, exp_dir = load_exp_data(args.data_dir, args.exp)
    param_values = index['param_values']
    param_files = index['param_files']

    snr_file_map = {}
    for pval, pfile in zip(param_values, param_files):
        snr_file_map[int(pval)] = pfile

    # ── 日志文件 ──
    log_path = f'cdf_log_{args.exp}.txt'
    log_f = open(log_path, 'w', encoding='utf-8')

    def log(msg=''):
        print(msg)
        log_f.write(msg + '\n')

    log(f"实验: {args.exp}")
    log(f"SNR列表: {args.snr_list}")
    log(f"对比CDF横轴: 0-{args.comparison_max_x}m")
    log(f"消融CDF横轴: 0-{args.ablation_max_x}m")
    log()

    # ── 对每个 SNR 出两张图 ──
    for snr_val in args.snr_list:
        if snr_val not in snr_file_map:
            log(f"[WARN] SNR={snr_val}dB 不在数据中，跳过")
            continue

        log(f"{'='*60}")
        log(f"  SNR = {snr_val:+d} dB")
        log(f"{'='*60}")

        pt_path = os.path.join(exp_dir, snr_file_map[snr_val])
        data = torch.load(pt_path, weights_only=False)
        fine_dpd = data['fine_dpd']
        pos_label = data['pos_label']
        n_src_all = data['n_src']

        # 收集所有方法的误差
        errors_dict = {}
        for method_name, model in all_methods:
            errors = collect_errors(method_name, model, fine_dpd, pos_label,
                                    n_src_all, args.peak_size, device)
            errors_dict[method_name] = errors

        # ── 汇总表：各方法指标 ──
        log(f"\n  {'方法':<15s} {'RMSE':>8s} {'均值':>8s} {'中位数':>8s}"
            f" {'<10m':>7s} {'<20m':>7s} {'<30m':>7s} {'<50m':>7s}"
            f" {'样本数':>7s}")
        log(f"  {'-'*85}")

        method_labels = {
            'dpd_peak': 'Peak Search',
            'EPN': 'EPN',
            'D1': 'YOLOv8',
            'D8': 'Proposed Net',
        }

        for method_name in ['dpd_peak', 'EPN', 'D1', 'D8']:
            e = errors_dict[method_name]
            rmse = np.sqrt(np.mean(e**2))
            mean_e = np.mean(e)
            med = np.median(e)
            w10 = np.mean(e < 10) * 100
            w20 = np.mean(e < 20) * 100
            w30 = np.mean(e < 30) * 100
            w50 = np.mean(e < 50) * 100
            label = method_labels[method_name]
            log(f"  {label:<15s} {rmse:>7.1f}m {mean_e:>7.1f}m {med:>7.1f}m"
                f" {w10:>6.1f}% {w20:>6.1f}% {w30:>6.1f}% {w50:>6.1f}%"
                f" {len(e):>7d}")

        # ── 论文用对比摘要 ──
        log(f"\n  [论文用] SNR={snr_val:+d}dB 对比摘要:")
        e_dpd = errors_dict['dpd_peak']
        e_epn = errors_dict['EPN']
        e_d8  = errors_dict['D8']
        e_d1  = errors_dict['D1']

        log(f"    Proposed vs Peak Search: "
            f"RMSE {np.sqrt(np.mean(e_d8**2)):.1f}m vs "
            f"{np.sqrt(np.mean(e_dpd**2)):.1f}m")
        log(f"    Proposed vs EPN: "
            f"RMSE {np.sqrt(np.mean(e_d8**2)):.1f}m vs "
            f"{np.sqrt(np.mean(e_epn**2)):.1f}m")
        log(f"    Proposed vs YOLOv8 (消融): "
            f"RMSE {np.sqrt(np.mean(e_d8**2)):.1f}m vs "
            f"{np.sqrt(np.mean(e_d1**2)):.1f}m, "
            f"中位数 {np.median(e_d8):.1f}m vs {np.median(e_d1):.1f}m")

        # CDF关键百分位
        for thresh in [10, 20, 30, 50]:
            log(f"    <{thresh}m比例: "
                f"Proposed {np.mean(e_d8<thresh)*100:.1f}%, "
                f"YOLOv8 {np.mean(e_d1<thresh)*100:.1f}%, "
                f"Peak Search {np.mean(e_dpd<thresh)*100:.1f}%, "
                f"EPN {np.mean(e_epn<thresh)*100:.1f}%")

        log()

        # 对比CDF
        log(f"  [对比CDF] max_x={args.comparison_max_x}m")
        plot_single_cdf(errors_dict, comparison_keys, styles,
                        snr_val, args.exp, 'comparison',
                        max_x=args.comparison_max_x)

        # 消融CDF
        log(f"  [消融CDF] max_x={args.ablation_max_x}m")
        plot_single_cdf(errors_dict, ablation_keys, styles,
                        snr_val, args.exp, 'ablation',
                        max_x=args.ablation_max_x)

        log()

    log(f"日志已保存: {log_path}")
    log('全部完成!')
    log_f.close()


if __name__ == '__main__':
    main()