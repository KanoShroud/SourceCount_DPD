#!/usr/bin/env python3
"""
plot_4D.py — 生成实验 4D 的 RMSE 曲线图（论文用）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

from chapter_runtime import output_dir as runtime_output_dir

parser = argparse.ArgumentParser(description='绘制第四章实验 4D 图')
parser.add_argument('--output_dir', default=None)
args = parser.parse_args()
output_dir = args.output_dir or str(runtime_output_dir('plot_4D'))
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

COL_D1 = '#2980B9'; COL_D8 = '#C0392B'
style_d1 = dict(color=COL_D1, marker='D', markersize=5, linewidth=1.5, linestyle='--', label='YOLOv8')
style_d8 = dict(color=COL_D8, marker='o', markersize=6, linewidth=2.0, linestyle='-',  label='Proposed Net')
legend_kw = dict(framealpha=0.9, edgecolor='#cccccc', handlelength=2.5, columnspacing=1.2)

# ─── 4D 数据 (去掉 2MHz) ───
bw = [4, 6, 8, 10, 12, 14, 16, 18, 20]
d1 = [35.2, 32.1, 17.3, 13.7, 12.0, 10.7, 9.9, 9.5, 9.0]
d8 = [26.5, 21.0, 15.8, 12.4, 10.8, 9.6, 9.1, 8.7, 8.5]

# ═══ 4D 对比实验（单独展示 Proposed Net）═══
fig1, ax1 = plt.subplots(figsize=(5.5, 4))
ax1.plot(bw, d8, **style_d8)
ax1.set_xlabel('Bandwidth (MHz)')
ax1.set_ylabel('RMSE (m)')
ax1.set_xticks(bw)
ax1.set_xlim([3, 21])
ax1.set_ylim([0, 32])
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right', ncol=1, **legend_kw)
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'fig_4D_comparison.png'))
print('已保存: fig_4D_comparison')

# ═══ 4D 消融实验 ═══
fig2, ax2 = plt.subplots(figsize=(5.5, 4))
ax2.plot(bw, d1, **style_d1)
ax2.plot(bw, d8, **style_d8)
ax2.set_xlabel('Bandwidth (MHz)')
ax2.set_ylabel('RMSE (m)')
ax2.set_xticks(bw)
ax2.set_xlim([3, 21])
ax2.set_ylim([0, 42])
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right', ncol=1, **legend_kw)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'fig_4D_ablation.png'))
print('已保存: fig_4D_ablation')

plt.close('all')
print('Done!')
