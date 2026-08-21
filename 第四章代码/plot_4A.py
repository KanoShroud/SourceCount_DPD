#!/usr/bin/env python3
"""
plot_4A.py — 生成实验 4A 的 RMSE 曲线图（论文用）
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from chapter_runtime import output_dir as runtime_output_dir

parser = argparse.ArgumentParser(description='绘制第四章实验 4A 图')
parser.add_argument('--output_dir', default=None)
args = parser.parse_args()
output_dir = args.output_dir or str(runtime_output_dir('plot_4A'))
os.makedirs(output_dir, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# ─── 数据 ───
snr = np.array([-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10])

d8_4a2  = [97.4, 22.1, 17.1, 14.7, 13.4, 12.5, 11.7, 11.7, 11.3, 11.3, 11.3]
d8_4a3  = [166.0, 36.8, 18.1, 15.7, 14.7, 13.7, 13.4, 13.1, 12.8, 12.9, 12.9]
d1_4a2  = [307.6, 138.4, 69.8, 33.2, 27.0, 17.0, 16.5, 16.2, 15.8, 15.5, 15.3]
d1_4a3  = [320.2, 90.7, 29.4, 23.0, 20.4, 18.7, 18.2, 18.0, 17.7, 17.5, 17.3]
epn_4a2 = [378.2, 310.2, 273.8, 259.2, 225.8, 188.9, 149.1, 123.7, 116.6, 104.0, 91.7]
epn_4a3 = [456.5, 398.2, 366.6, 350.1, 330.1, 317.4, 305.1, 304.7, 298.8, 294.7, 297.2]
dpd_4a2 = [1048.8, 959.0, 913.6, 889.4, 872.1, 842.9, 833.0, 829.6, 815.6, 806.9, 798.2]
dpd_4a3 = [1067, 1016, 988, 985, 981, 978, 975, 972, 969, 966, 963]

# ─── 配色 ───
COL_DPD = '#666666'
COL_EPN = '#E67E22'
COL_D1  = '#2980B9'
COL_D8  = '#C0392B'

style_dpd = dict(color=COL_DPD, marker='s', markersize=6, linewidth=1.5, linestyle='--',  label='Peak Search')
style_epn = dict(color=COL_EPN, marker='^', markersize=6, linewidth=1.5, linestyle='-.',  label='EPN')
style_d1  = dict(color=COL_D1,  marker='D', markersize=5, linewidth=1.5, linestyle='--',  label='YOLOv8')
style_d8  = dict(color=COL_D8,  marker='o', markersize=6, linewidth=2.0, linestyle='-',   label='Proposed Net')

legend_kw = dict(framealpha=0.9, edgecolor='#cccccc', handlelength=2.5, columnspacing=1.2)


# ═══ 4A2 对比实验 ═══
fig1, ax1 = plt.subplots(figsize=(5.5, 4))
ax1.semilogy(snr, dpd_4a2, **style_dpd)
ax1.semilogy(snr, epn_4a2, **style_epn)
ax1.semilogy(snr, d8_4a2,  **style_d8)
ax1.set_xlabel('SNR (dB)')
ax1.set_ylabel('RMSE (m)')
ax1.set_xticks(snr)
ax1.set_xlim([-11, 11])
ax1.set_ylim([8, 2000])
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='upper right', ncol=3, fontsize=9, **legend_kw)
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'fig_4A2_comparison.pdf'))
fig1.savefig(os.path.join(output_dir, 'fig_4A2_comparison.png'))
print('已保存: fig_4A2_comparison')

# ═══ 4A3 对比实验 ═══
fig2, ax2 = plt.subplots(figsize=(5.5, 4))
ax2.semilogy(snr, dpd_4a3, **style_dpd)
ax2.semilogy(snr, epn_4a3, **style_epn)
ax2.semilogy(snr, d8_4a3,  **style_d8)
ax2.set_xlabel('SNR (dB)')
ax2.set_ylabel('RMSE (m)')
ax2.set_xticks(snr)
ax2.set_xlim([-11, 11])
ax2.set_ylim([8, 2000])
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='upper right', ncol=3, fontsize=9, **legend_kw)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'fig_4A3_comparison.pdf'))
fig2.savefig(os.path.join(output_dir, 'fig_4A3_comparison.png'))
print('已保存: fig_4A3_comparison')

# ═══ 4A2 消融实验 ═══
fig3, ax3 = plt.subplots(figsize=(5.5, 4))
ax3.plot(snr, d1_4a2, **style_d1)
ax3.plot(snr, d8_4a2, **style_d8)
ax3.set_xlabel('SNR (dB)')
ax3.set_ylabel('RMSE (m)')
ax3.set_xticks(snr)
ax3.set_xlim([-11, 11])
ax3.set_ylim([0, 350])
ax3.grid(True, alpha=0.3)
ax3.legend(loc='upper right', ncol=1, **legend_kw)
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, 'fig_4A2_ablation.pdf'))
fig3.savefig(os.path.join(output_dir, 'fig_4A2_ablation.png'))
print('已保存: fig_4A2_ablation')

# ═══ 4A3 消融实验 ═══
fig4, ax4 = plt.subplots(figsize=(5.5, 4))
ax4.plot(snr, d1_4a3, **style_d1)
ax4.plot(snr, d8_4a3, **style_d8)
ax4.set_xlabel('SNR (dB)')
ax4.set_ylabel('RMSE (m)')
ax4.set_xticks(snr)
ax4.set_xlim([-11, 11])
ax4.set_ylim([0, 350])
ax4.grid(True, alpha=0.3)
ax4.legend(loc='upper right', ncol=1, **legend_kw)
fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, 'fig_4A3_ablation.pdf'))
fig4.savefig(os.path.join(output_dir, 'fig_4A3_ablation.png'))
print('已保存: fig_4A3_ablation')

plt.close('all')
print('Done!')
