"""
vis_feature_activation_v2.py — CNN特征激活图（Exp1A数据，支持多SNR批量）

一次加载数据，批量画多个SNR下的特征激活图。

用法:
  # 单个SNR
  python vis_feature_activation_v2.py B M10 --snr -5

  # 多个SNR一次画完（数据只加载一次）
  python vis_feature_activation_v2.py B M10 --snr -13 -9 -5 0 5

  # 指定样本索引和层
  python vis_feature_activation_v2.py B M10 --snr -13 -5 5 --sample 0 --layer deep
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import argparse


# ═══════════════════════════════════════
#  模型定义
# ═══════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return F.relu(x + res)

class SourceDetectionNet(nn.Module):
    def __init__(self, n_sub=10, max_src=3, feat_dim=128, mode='concat'):
        super().__init__()
        self.n_sub = n_sub; self.max_src = max_src
        self.feat_dim = feat_dim; self.mode = mode
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), ResBlock(32, 0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64, 0.1),
            nn.Conv2d(64, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(), ResBlock(feat_dim, 0.1),
            nn.AdaptiveAvgPool2d(1))
        if mode == 'transformer':
            self.pos_embed = nn.Parameter(torch.randn(1, n_sub, feat_dim)*0.02)
            enc = nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=4, dim_feedforward=256,
                dropout=0.1, batch_first=True)
            self.cross_attn = nn.TransformerEncoder(enc, num_layers=1)
            self.global_encoder = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3))
        else:
            self.global_encoder = nn.Sequential(
                nn.Linear(feat_dim * n_sub, 256), nn.ReLU(), nn.Dropout(0.3))
        self.band_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, n_sub))
            for _ in range(max_src)])

    def forward(self, x):
        B, S, H, W = x.shape
        feat = self.backbone(x.reshape(B*S, 1, H, W))
        feat = feat.squeeze(-1).squeeze(-1).reshape(B, S, self.feat_dim)
        if self.mode == 'transformer':
            feat = self.cross_attn(feat + self.pos_embed)
            gf = self.global_encoder(feat.mean(dim=1))
        else:
            gf = self.global_encoder(feat.reshape(B, -1))
        return torch.stack([h(gf) for h in self.band_heads], dim=1)


# ═══════════════════════════════════════
#  多层特征提取器
# ═══════════════════════════════════════
class MultiLayerExtractor:
    def __init__(self, model):
        self.features = {}
        model.backbone[3].register_forward_hook(
            lambda m, i, o: self.features.update({'shallow': o}))
        model.backbone[7].register_forward_hook(
            lambda m, i, o: self.features.update({'mid': o}))
        model.backbone[11].register_forward_hook(
            lambda m, i, o: self.features.update({'deep': o}))

    def get_activation(self, model, input_tensor, layer='deep'):
        self.features = {}
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        with torch.no_grad():
            _ = model(input_tensor)
        feat = self.features[layer]
        activation = torch.norm(feat, dim=1)
        activation = F.interpolate(
            activation.unsqueeze(1), size=(H, W),
            mode='bilinear', align_corners=False
        ).squeeze(1).cpu().numpy()
        return activation


# ═══════════════════════════════════════
#  自定义colormap
# ═══════════════════════════════════════
def make_activation_cmap():
    cdict = {
        'red':   [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.0, 0.0),
                  (0.7, 1.0, 1.0), (1.0, 0.85, 0.85)],
        'green': [(0.0, 0.0, 0.0), (0.3, 0.2, 0.2), (0.5, 0.6, 0.6),
                  (0.7, 0.85, 0.85), (1.0, 0.1, 0.1)],
        'blue':  [(0.0, 0.3, 0.3), (0.3, 0.7, 0.7), (0.5, 0.6, 0.6),
                  (0.7, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'alpha': [(0.0, 0.0, 0.0), (0.15, 0.0, 0.0), (0.3, 0.4, 0.4),
                  (0.5, 0.6, 0.6), (0.7, 0.8, 0.8), (1.0, 0.95, 0.95)],
    }
    return LinearSegmentedColormap('activation', cdict, N=256)


def make_solid_cmap():
    cdict = {
        'red':   [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.0, 0.0),
                  (0.7, 1.0, 1.0), (1.0, 0.85, 0.85)],
        'green': [(0.0, 0.0, 0.0), (0.3, 0.2, 0.2), (0.5, 0.6, 0.6),
                  (0.7, 0.85, 0.85), (1.0, 0.1, 0.1)],
        'blue':  [(0.0, 0.3, 0.3), (0.3, 0.7, 0.7), (0.5, 0.6, 0.6),
                  (0.7, 0.0, 0.0), (1.0, 0.0, 0.0)],
    }
    return LinearSegmentedColormap('act_solid', cdict, N=256)


# ═══════════════════════════════════════
#  画单个SNR的图
# ═══════════════════════════════════════
def plot_one_snr(model, extractor, spectra_raw, band_mask, src_pos_data,
                 snr_target, target_snr, sample_idx, layer, threshold,
                 cfg, device, tag, act_cmap, cmap_solid):

    N_sub = cfg['N_sub']

    # 筛选该SNR的样本
    mask = np.abs(snr_target - target_snr) < 0.5
    indices = np.where(mask)[0]
    n_match = len(indices)

    if n_match == 0:
        print(f"  SNR={target_snr:+.0f}dB: no samples found, skip")
        return
    if sample_idx >= n_match:
        print(f"  SNR={target_snr:+.0f}dB: sample {sample_idx} out of range (max {n_match-1})")
        return

    global_idx = indices[sample_idx]

    # 预处理
    spec = spectra_raw[global_idx]
    spec_log = np.log(spec + 1.0)
    mu = spec_log.mean(); sd = spec_log.std() + 1e-6
    spec_norm = (spec_log - mu) / sd

    # 真值
    bm = band_mask[global_idx]
    n_src = int((bm.sum(axis=-1) > 0).sum())
    src_pos = src_pos_data[global_idx] if src_pos_data is not None else None

    # 真值位置：有位置数据则用，否则用Exp1A固定位置
    if src_pos is not None:
        true_pos = [src_pos[s, :] for s in range(n_src)]
    elif n_src > 0:
        true_pos = [np.array([1600, 700], dtype=np.float32)]
    else:
        true_pos = []

    # 模型预测
    input_tensor = torch.from_numpy(spec_norm[None]).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits)
        band_pred = (probs > threshold).long()

    pred_count = (band_pred[0].sum(dim=-1) > 0).sum().item()
    band_pred_np = band_pred[0].cpu().numpy()

    # 特征激活
    activation = extractor.get_activation(model, input_tensor, layer)
    act_min = activation.min(); act_max = activation.max()
    if act_max > act_min:
        activation_normed = (activation - act_min) / (act_max - act_min)
    else:
        activation_normed = np.zeros_like(activation)

    print(f"  SNR={target_snr:+.0f}dB: idx={global_idx}, {n_src}src, pred={pred_count}")

    # 参数
    edge = 2000
    fs = 100e6; B_step = 5e6; B_win = 10e6
    sub_f_lo = np.array([k * B_step - fs/2 for k in range(N_sub)])
    sub_f_hi = sub_f_lo + B_win
    R_rcv = 500; N_rx = 4
    angles_rx = np.arange(N_rx) * 2 * np.pi / N_rx
    rcv_x = R_rcv * np.cos(angles_rx)
    rcv_y = R_rcv * np.sin(angles_rx)
    layer_info = {'shallow': '(32ch, 41x41)', 'mid': '(64ch, 21x21)', 'deep': '(128ch, 11x11)'}

    # 画图
    n_cols = 5
    n_rows = (N_sub + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.8, n_rows * 3.8 + 1.2))
    axes = axes.flatten()

    for k in range(N_sub):
        ax = axes[k]
        dpd = spec_norm[k]
        ax.imshow(dpd.T, origin='lower', extent=[-edge, edge, -edge, edge],
                  cmap='YlGnBu', aspect='equal')

        act = activation_normed[k]
        ax.imshow(act.T, origin='lower', extent=[-edge, edge, -edge, edge],
                  cmap=act_cmap, vmin=0, vmax=1)

        for s in range(len(true_pos)):
            ax.plot(true_pos[s][0], true_pos[s][1], '*', color='lime',
                    markersize=12, markeredgecolor='black', markeredgewidth=0.6)
        ax.plot(rcv_x, rcv_y, 'r^', markersize=4)

        active_slots = [s for s in range(cfg['max_src']) if band_pred_np[s, k] == 1]
        f_lo = sub_f_lo[k] / 1e6; f_hi = sub_f_hi[k] / 1e6
        if active_slots:
            slot_str = ','.join([f'S{s+1}' for s in active_slots])
            title_color = '#2E7D32'
        else:
            slot_str = 'noise'
            title_color = '#757575'
        ax.set_title(f'W{k+1} [{f_lo:.0f}~{f_hi:.0f}] {slot_str}',
                     fontsize=8, color=title_color, fontweight='bold')
        ax.set_xlim(-edge, edge); ax.set_ylim(-edge, edge)
        ax.set_xticks([]); ax.set_yticks([])

        if active_slots:
            for spine in ax.spines.values():
                spine.set_edgecolor('#2E7D32'); spine.set_linewidth(2.5)
        else:
            for spine in ax.spines.values():
                spine.set_edgecolor('#BDBDBD'); spine.set_linewidth(0.8)

    for k in range(N_sub, len(axes)):
        axes[k].set_visible(False)

    # colorbar
    cbar_ax = fig.add_axes([0.15, 0.02, 0.7, 0.015])
    sm = plt.cm.ScalarMappable(cmap=cmap_solid, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('CNN Feature Activation (normalized)', fontsize=10)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Low', '', 'Medium', '', 'High'])

    # 图例
    legend_handles = [
        patches.Patch(edgecolor='#2E7D32', facecolor='none', linewidth=2.5,
                      label='Signal subband'),
        patches.Patch(edgecolor='#BDBDBD', facecolor='none', linewidth=1,
                      label='Noise subband'),
        plt.Line2D([0], [0], marker='*', color='lime', linestyle='None',
                   markersize=12, markeredgecolor='black', label='True position'),
        plt.Line2D([0], [0], marker='^', color='red', linestyle='None',
                   markersize=6, label='Receiver'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, 0.045))

    fig.suptitle(f'Exp1A  SNR={target_snr:+.0f}dB  Sample #{sample_idx}   '
                 f'Truth: {n_src} src   Pred: {pred_count} src\n'
                 f'CNN {layer} layer {layer_info[layer]} feature activation',
                 fontsize=13, fontweight='bold')
    plt.subplots_adjust(top=0.92, bottom=0.10, hspace=0.35, wspace=0.15)

    save_name = f'feature_act_exp1A_snr{int(target_snr):+d}_s{sample_idx}_{layer}_{tag}.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {save_name}")


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scheme', nargs='*', default=['B', 'M10'])
    parser.add_argument('--snr', type=float, nargs='+', required=True,
                        help='目标SNR(dB)，支持多个')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.50)
    parser.add_argument('--layer', type=str, default='deep',
                        choices=['shallow', 'mid', 'deep'])
    args = parser.parse_args()

    tag = 'B'
    for a in args.scheme:
        if a == 'A': tag = 'A'
        elif a == 'B': tag = 'B'
        elif a.startswith('M') and a[1:].isdigit(): tag += f'_{a}'

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 加载模型 ──
    ckpt = torch.load(f'best_model_v26_{tag}.pth', map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model: {tag} ({cfg['mode']}) max_src={cfg['max_src']}")

    # ── 一次性加载全部Exp1A数据 ──
    exp_path = '/mnt/data/ltzdata/ctrl_exp1A.mat'
    print(f"\nLoading: {exp_path} ...")
    with h5py.File(exp_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32).transpose(3, 2, 1, 0)
        snr_target = np.array(f['snr_target_all'], dtype=np.float32).flatten()
        band_mask = np.array(f['band_mask_all'], dtype=np.float32).transpose(2, 1, 0)
        # src_pos_all 可能不存在（Exp1A是控制变量实验）
        if 'src_pos_all' in f:
            src_pos_data = np.array(f['src_pos_all'], dtype=np.float32)
            if src_pos_data.shape[0] != len(snr_target):
                src_pos_data = src_pos_data.transpose(2, 1, 0)
        else:
            src_pos_data = None
    print(f"Loaded {len(snr_target)} samples")
    print(f"Available SNRs: {sorted(np.unique(np.round(snr_target)))}")

    # ── 特征提取器 ──
    extractor = MultiLayerExtractor(model)
    act_cmap = make_activation_cmap()
    cmap_solid = make_solid_cmap()

    # ── 批量画图 ──
    print(f"\nProcessing {len(args.snr)} SNR points...")
    for snr in args.snr:
        plot_one_snr(model, extractor, spectra_raw, band_mask, src_pos_data,
                     snr_target, snr, args.sample, args.layer, args.threshold,
                     cfg, device, tag, act_cmap, cmap_solid)

    print("\nAll done!")


if __name__ == '__main__':
    main()