#!/usr/bin/env python3
"""
plot_4B.py — 生成实验 4B 的 RMSE 曲线图（论文用）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

from chapter_runtime import output_dir as runtime_output_dir

parser = argparse.ArgumentParser(description='绘制第四章实验 4B 图')
parser.add_argument('--output_dir', default=None)
args = parser.parse_args()
output_dir = args.output_dir or str(runtime_output_dir('plot_4B'))
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix', 'font.size': 12,
    'axes.labelsize': 13, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

# ─── 配色方案 B ───
COL_DPD = '#666666'; COL_EPN = '#E67E22'; COL_D1 = '#2980B9'; COL_D8 = '#C0392B'

style_dpd = dict(color=COL_DPD, marker='s', markersize=6, linewidth=1.5, linestyle='--',  label='Peak Search')
style_epn = dict(color=COL_EPN, marker='^', markersize=6, linewidth=1.5, linestyle='-.',  label='EPN')
style_d1  = dict(color=COL_D1,  marker='D', markersize=5, linewidth=1.5, linestyle='--',  label='YOLOv8')
style_d8  = dict(color=COL_D8,  marker='o', markersize=6, linewidth=2.0, linestyle='-',   label='Proposed Net')

legend_kw = dict(framealpha=0.9, edgecolor='#cccccc', handlelength=2.5, columnspacing=1.2)

# ─── 4B2 数据 (2源, 距离扫描, SNR=0dB) ───
dist_b2 = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
dpd_b2  = [29.0, 66.3, 117.4, 159.3, 292.3, 486.7, 725.4, 884.1, 1030.9, 1174.4]
epn_b2  = [55.1, 92.9, 129.6, 136.0, 175.7, 207.7, 238.4, 273.2, 314.5, 425.7]
d1_b2   = [1.8, 1.7, 1.9, 2.6, 3.7, 6.5, 14.0, 20.4, 28.0, 36.4]
d8_b2   = [1.8, 1.8, 1.9, 2.8, 3.8, 6.0, 12.2, 15.4, 18.4, 18.9]

# ─── 4B3 数据 (3源, 距离扫描, SNR=0dB, 从200m起) ───
dist_b3 = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
dpd_b3  = [101.9, 328.3, 388.1, 416.0, 793.9, 918.7, 1068.3, 1212.2, 1345.5]
epn_b3  = [120.4, 170.4, 198.9, 257.8, 305.0, 357.6, 399.2, 445.7, 544.9]
d1_b3   = [2.7, 4.0, 5.5, 9.0, 15.0, 24.0, 33.0, 45.0, 61.5]
d8_b3   = [2.8, 3.2, 3.4, 4.9, 8.3, 16.7, 19.9, 26.4, 40.3]


# ═══ 4B2 对比实验 ═══
fig1, ax1 = plt.subplots(figsize=(5.5, 4))
ax1.semilogy(dist_b2, dpd_b2, **style_dpd)
ax1.semilogy(dist_b2, epn_b2, **style_epn)
ax1.semilogy(dist_b2, d8_b2,  **style_d8)
ax1.set_xlabel('Distance (m)')
ax1.set_ylabel('RMSE (m)')
ax1.set_xticks(dist_b2)
ax1.set_xlim([50, 1050])
ax1.set_ylim([1, 2000])
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='upper left', ncol=1, fontsize=9, **legend_kw)
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'fig_4B2_comparison.png'))
print('已保存: fig_4B2_comparison')

# ═══ 4B3 对比实验 ═══
fig2, ax2 = plt.subplots(figsize=(5.5, 4))
ax2.semilogy(dist_b3, dpd_b3, **style_dpd)
ax2.semilogy(dist_b3, epn_b3, **style_epn)
ax2.semilogy(dist_b3, d8_b3,  **style_d8)
ax2.set_xlabel('Distance (m)')
ax2.set_ylabel('RMSE (m)')
ax2.set_xticks(dist_b3)
ax2.set_xlim([150, 1050])
ax2.set_ylim([1, 2000])
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='upper left', ncol=1, fontsize=9, **legend_kw)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'fig_4B3_comparison.png'))
print('已保存: fig_4B3_comparison')

# ═══ 4B2 消融实验 ═══
fig3, ax3 = plt.subplots(figsize=(5.5, 4))
ax3.plot(dist_b2, d1_b2, **style_d1)
ax3.plot(dist_b2, d8_b2, **style_d8)
ax3.set_xlabel('Distance (m)')
ax3.set_ylabel('RMSE (m)')
ax3.set_xticks(dist_b2)
ax3.set_xlim([50, 1050])
ax3.set_ylim([0, 45])
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper left', ncol=1, **legend_kw)
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, 'fig_4B2_ablation.png'))
print('已保存: fig_4B2_ablation')

# ═══ 4B3 消融实验 ═══
fig4, ax4 = plt.subplots(figsize=(5.5, 4))
ax4.plot(dist_b3, d1_b3, **style_d1)
ax4.plot(dist_b3, d8_b3, **style_d8)
ax4.set_xlabel('Distance (m)')
ax4.set_ylabel('RMSE (m)')
ax4.set_xticks(dist_b3)
ax4.set_xlim([150, 1050])
ax4.set_ylim([0, 70])
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper left', ncol=1, **legend_kw)
fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, 'fig_4B3_ablation.png'))
print('已保存: fig_4B3_ablation')

plt.close('all')
print('Done!')
