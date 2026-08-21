#!/usr/bin/env python3
"""
plot_4C.py — 生成实验 4C 的 RMSE 曲线图（论文用）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

from chapter_runtime import output_dir as runtime_output_dir

parser = argparse.ArgumentParser(description='绘制第四章实验 4C 图')
parser.add_argument('--output_dir', default=None)
args = parser.parse_args()
output_dir = args.output_dir or str(runtime_output_dir('plot_4C'))
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

COL_DPD = '#666666'; COL_EPN = '#E67E22'; COL_D1 = '#2980B9'; COL_D8 = '#C0392B'
style_dpd = dict(color=COL_DPD, marker='s', markersize=6, linewidth=1.5, linestyle='--',  label='Peak Search')
style_epn = dict(color=COL_EPN, marker='^', markersize=6, linewidth=1.5, linestyle='-.',  label='EPN')
style_d1  = dict(color=COL_D1,  marker='D', markersize=5, linewidth=1.5, linestyle='--',  label='YOLOv8')
style_d8  = dict(color=COL_D8,  marker='o', markersize=6, linewidth=2.0, linestyle='-',   label='Proposed Net')
legend_kw = dict(framealpha=0.9, edgecolor='#cccccc', handlelength=2.5, columnspacing=1.2)

# ─── 4C 数据 (从150m起) ───
sep = [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]

dpd = [99.8, 104.0, 108.9, 116.7, 124.6, 129.6, 135.5, 142.2, 148.3, 154.8, 162.0]
epn = [88.0, 85.0, 85.5, 84.0, 86.0, 84.5, 84.0, 82.5, 84.0, 83.0, 83.5]
d8  = [33.0, 28.0, 22.5, 18.0, 16.0, 15.0, 14.5, 14.0, 13.5, 13.2, 13.0]
d1  = [36.0, 32.0, 28.0, 25.0, 23.0, 21.0, 20.0, 19.5, 19.0, 18.5, 18.0]

# ═══ 4C 对比实验 ═══
fig1, ax1 = plt.subplots(figsize=(5.5, 4))
ax1.semilogy(sep, dpd, **style_dpd)
ax1.semilogy(sep, epn, **style_epn)
ax1.semilogy(sep, d8,  **style_d8)
ax1.set_xlabel('Separation (m)')
ax1.set_ylabel('RMSE (m)')
ax1.set_xticks(sep)
ax1.set_xlim([145, 255])
ax1.set_ylim([8, 300])
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='upper right', ncol=3, fontsize=9, **legend_kw)
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'fig_4C_comparison.png'))
print('已保存: fig_4C_comparison')

# ═══ 4C 消融实验 ═══
fig2, ax2 = plt.subplots(figsize=(5.5, 4))
ax2.plot(sep, d1, **style_d1)
ax2.plot(sep, d8, **style_d8)
ax2.set_xlabel('Separation (m)')
ax2.set_ylabel('RMSE (m)')
ax2.set_xticks(sep)
ax2.set_xlim([145, 255])
ax2.set_ylim([0, 45])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', ncol=1, **legend_kw)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'fig_4C_ablation.png'))
print('已保存: fig_4C_ablation')

plt.close('all')
print('Done!')
