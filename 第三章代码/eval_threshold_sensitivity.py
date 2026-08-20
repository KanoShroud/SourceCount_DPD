"""
eval_threshold_sensitivity.py — BAND_THRESHOLD 灵敏度分析

在验证集上扫描 threshold 从 0.05 到 0.95，
计算 Count Accuracy 和 Band Accuracy，画曲线图。

用法:
  python eval_threshold_sensitivity.py B M10
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys


THRESHOLD_RANGE = np.arange(0.05, 0.96, 0.025)


# ═══════════════════════════════════════
#  模型定义
# ═══════════════════════════════════════
class ResBlock(nn.Module):
    def __init__(self, ch, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(ch)
        self.drop  = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
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
#  主函数
# ═══════════════════════════════════════
def main():
    args = sys.argv[1:]
    tag = 'A'
    for a in args:
        if a == 'B': tag = 'B'
        elif a.startswith('M') and a[1:].isdigit(): tag += f'_{a}'

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 加载模型 ──
    model_path = f'best_model_v26_{tag}.pth'
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    max_src_model = cfg['max_src']
    print(f"Model: {tag} ({cfg.get('mode','concat')}) max_src={max_src_model}")

    # ── 加载验证集 ──
    val_path = '/mnt/data/ltzdata/val_data.mat'
    print(f"Loading: {val_path}")
    with h5py.File(val_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32)
        src_count   = np.array(f['src_count_all'], dtype=np.int64).flatten()
        band_mask   = np.array(f['band_mask_all'], dtype=np.float32)
        ignore_mask = np.array(f['ignore_mask_all'], dtype=np.float32)
        N_sub       = int(np.array(f['N_sub_val']).item())
        max_src_data = int(np.array(f['max_src_val']).item())

    spectra     = spectra_raw.transpose(3, 2, 1, 0)
    band_mask   = band_mask.transpose(2, 1, 0)       # (N, max_src_data, N_sub)
    ignore_mask = ignore_mask.transpose(2, 1, 0)

    # log + z-score
    spectra_dl = np.log(spectra + 1.0)
    for i in range(len(spectra_dl)):
        s = spectra_dl[i]; mu = s.mean(); std = s.std() + 1e-6
        spectra_dl[i] = (s - mu) / std

    N = len(src_count)
    print(f"  {N} samples")

    # ── 推理（只做一次，保存 probs）──
    print("  Running inference...")
    all_probs = []
    with torch.no_grad():
        for i in range(0, N, 64):
            j = min(i + 64, N)
            x = torch.from_numpy(spectra_dl[i:j]).to(device)
            bl = model(x)
            all_probs.append(torch.sigmoid(bl).cpu().numpy())
    probs = np.concatenate(all_probs)  # (N, max_src_model, N_sub)

    # ── 扫描 threshold ──
    print(f"  Scanning {len(THRESHOLD_RANGE)} thresholds...")

    count_accs = []
    band_accs = []
    count_accs_by_src = {s: [] for s in sorted(np.unique(src_count))}

    ms = min(max_src_model, max_src_data)

    for th in THRESHOLD_RANGE:
        preds = (probs > th).astype(np.int64)

        # Count Accuracy
        count_pred = (preds.sum(axis=-1) > 0).sum(axis=-1)
        count_correct = (count_pred == src_count)
        count_accs.append(count_correct.mean())

        for s in count_accs_by_src:
            mask = (src_count == s)
            count_accs_by_src[s].append(count_correct[mask].mean())

        # Band Accuracy
        band_correct = 0
        band_total = 0
        for s in range(ms):
            for k in range(N_sub):
                ignore = (ignore_mask[:, s, k] > 0.5)
                valid = ~ignore
                true_val = (band_mask[:, s, k] > 0.5).astype(np.int64)
                pred_val = preds[:, s, k]
                band_correct += ((pred_val == true_val) & valid).sum()
                band_total += valid.sum()
        for s in range(ms, max_src_model):
            for k in range(N_sub):
                band_correct += (preds[:, s, k] == 0).sum()
                band_total += N
        band_accs.append(band_correct / max(band_total, 1))

    count_accs = np.array(count_accs)
    band_accs = np.array(band_accs)

    # ── 找最优点 ──
    best_idx = np.argmax(count_accs)
    best_th = THRESHOLD_RANGE[best_idx]
    print(f"\n  Best threshold: {best_th:.3f}")
    print(f"  Count Acc @ best: {count_accs[best_idx]:.1%}")
    print(f"  Band Acc  @ best: {band_accs[best_idx]:.1%}")

    # 标记 0.50 的位置
    idx_050 = np.argmin(np.abs(THRESHOLD_RANGE - 0.50))
    print(f"\n  @ threshold=0.50:")
    print(f"  Count Acc: {count_accs[idx_050]:.1%}")
    print(f"  Band Acc:  {band_accs[idx_050]:.1%}")

    # ── 打印表格 ──
    print(f"\n{'Threshold':>10} {'Count Acc':>12} {'Band Acc':>12}", end='')
    for s in sorted(count_accs_by_src.keys()):
        print(f" {s}src{'':>5}", end='')
    print()
    print("-" * (36 + 10 * len(count_accs_by_src)))
    for i, th in enumerate(THRESHOLD_RANGE):
        marker = " ◄" if abs(th - 0.50) < 0.013 else ""
        print(f"  {th:>8.3f} {count_accs[i]:>11.1%} {band_accs[i]:>11.1%}", end='')
        for s in sorted(count_accs_by_src.keys()):
            print(f" {count_accs_by_src[s][i]:>8.1%}", end='')
        print(marker)

    # ── 画图 ──
    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(THRESHOLD_RANGE, count_accs * 100, 'b-o', markersize=3,
             linewidth=2, label='Count Accuracy')
    ax1.plot(THRESHOLD_RANGE, band_accs * 100, 'r-s', markersize=3,
             linewidth=2, label='Band Accuracy')

    # 标记选定阈值
    ax1.axvline(x=0.50, color='gray', linestyle='--', alpha=0.7, label='Selected (0.50)')
    ax1.plot(0.50, count_accs[idx_050] * 100, 'b*', markersize=15, zorder=5)
    ax1.plot(0.50, band_accs[idx_050] * 100, 'r*', markersize=15, zorder=5)

    ax1.set_xlabel('BAND_THRESHOLD', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'BAND_THRESHOLD Sensitivity Analysis ({tag})', fontsize=13)
    ax1.legend(fontsize=10, loc='lower center')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([80, 101])

    plt.tight_layout()
    save_name = f'threshold_sensitivity_{tag}.png'
    plt.savefig(save_name, dpi=150)
    print(f"\nFigure saved: {save_name}")
    plt.close()

    # ── 按源数画图 ──
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: '#2196F3', 1: '#4CAF50', 2: '#FF9800', 3: '#E53935'}
    for s in sorted(count_accs_by_src.keys()):
        ax.plot(THRESHOLD_RANGE, np.array(count_accs_by_src[s]) * 100,
                '-o', markersize=3, linewidth=1.5, color=colors.get(s, 'gray'),
                label=f'{s} src')
    ax.axvline(x=0.50, color='gray', linestyle='--', alpha=0.7, label='Selected (0.50)')
    ax.set_xlabel('BAND_THRESHOLD', fontsize=12)
    ax.set_ylabel('Count Accuracy (%)', fontsize=12)
    ax.set_title(f'Count Accuracy by Source Count vs Threshold ({tag})', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([70, 101])

    plt.tight_layout()
    save_name2 = f'threshold_by_srccount_{tag}.png'
    plt.savefig(save_name2, dpi=150)
    print(f"Figure saved: {save_name2}")
    plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()