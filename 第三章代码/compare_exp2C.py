"""
compare_exp2C.py — Exp 2-C: 同频多源 SNR 扫描 (DL only)

3张子图 (1源/2源/3源), 每张3条曲线 (L1/L2/L3)

用法:
  python compare_exp2C.py B M10
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


BAND_THRESHOLD_DL = 0.50


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
#  公共函数
# ═══════════════════════════════════════
def dl_preprocess(spectra):
    out = np.log(spectra + 1.0)
    for i in range(len(out)):
        s = out[i]; mu = s.mean(); std = s.std() + 1e-6
        out[i] = (s - mu) / std
    return out


def predict_dl(model, spectra_dl, device, threshold):
    N = len(spectra_dl)
    N_sub = model.n_sub
    subband_count = np.zeros((N, N_sub), dtype=np.int64)
    sample_count  = np.zeros(N, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, N, 64):
            j = min(i + 64, N)
            x = torch.from_numpy(spectra_dl[i:j]).to(device)
            bl = model(x)
            bp = (torch.sigmoid(bl) > threshold).long()
            subband_count[i:j] = bp.sum(dim=1).cpu().numpy()
            sample_count[i:j]  = (bp.sum(dim=-1) > 0).sum(dim=-1).cpu().numpy()
    return sample_count, subband_count


def compute_metrics(sample_count_pred, subband_count_pred,
                    sample_count_true, subband_count_true, ignore_any):
    evaluable = ~ignore_any

    sample_correct = (sample_count_pred == sample_count_true)
    l1 = sample_correct.mean()

    subband_correct = (subband_count_pred == subband_count_true) & evaluable
    l2 = subband_correct.sum() / max(evaluable.sum(), 1)

    all_sub_ok = ((subband_count_pred == subband_count_true) | ~evaluable).all(axis=1)
    l3 = (sample_correct & all_sub_ok).mean()

    return l1, l2, l3


# ═══════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════
def load_exp2c(mat_path):
    with h5py.File(mat_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32)
        band_mask   = np.array(f['band_mask_all'], dtype=np.float32)
        ignore_mask = np.array(f['ignore_mask_all'], dtype=np.float32)
        src_count   = np.array(f['src_count_all'], dtype=np.int64).flatten()
        snr_target  = np.array(f['snr_target_all'], dtype=np.float32).flatten()
        config_id   = np.array(f['config_id_all'], dtype=np.int64).flatten()

    spectra     = spectra_raw.transpose(3, 2, 1, 0)
    band_mask   = band_mask.transpose(2, 1, 0)
    ignore_mask = ignore_mask.transpose(2, 1, 0)

    subband_count_true = (band_mask > 0.5).astype(np.int64).sum(axis=1)
    ignore_any = (ignore_mask > 0.5).any(axis=1)

    spectra_dl = dl_preprocess(spectra)

    return (spectra_dl, src_count, snr_target, config_id,
            subband_count_true, ignore_any)


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
    ckpt = torch.load(f'best_model_v26_{tag}.pth', map_location=device,
                       weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    print(f"Model: {tag} ({cfg['mode']}) max_src={cfg['max_src']}")
    print(f"DL threshold: {BAND_THRESHOLD_DL}")

    # ── 加载数据 ──
    print(f"\n--- Loading Exp 2-C data ---")
    exp_path = '/mnt/data/ltzdata/ctrl_exp2C.mat'
    (spectra_dl, src_count, snr_target, config_id,
     subband_count_true, ignore_any) = load_exp2c(exp_path)
    N = len(src_count)
    print(f"  {N} samples")

    # ── DL 推理 ──
    print("  Running DL inference...")
    sample_count_pred, subband_count_pred = predict_dl(
        model, spectra_dl, device, BAND_THRESHOLD_DL)

    # ── 逐配置逐SNR计算指标 ──
    src_nums = sorted(np.unique(config_id))
    snr_vals = sorted(np.unique(snr_target))
    print(f"  Source configs: {src_nums}")
    print(f"  SNR points: {[f'{s:+.0f}' for s in snr_vals]}")

    curves = {}
    for ns in src_nums:
        curves[ns] = {'l1': [], 'l2': [], 'l3': []}
        for snr in snr_vals:
            mask = (config_id == ns) & (np.abs(snr_target - snr) < 0.5)
            l1, l2, l3 = compute_metrics(
                sample_count_pred[mask], subband_count_pred[mask],
                src_count[mask], subband_count_true[mask], ignore_any[mask])
            curves[ns]['l1'].append(l1)
            curves[ns]['l2'].append(l2)
            curves[ns]['l3'].append(l3)

    # ── 打印表格 ──
    for ns in src_nums:
        print(f"\n===== {ns}-Source Co-frequency =====")
        print(f"{'SNR(dB)':>8} {'L1 Count':>12} {'L2 Subband':>12} {'L3 Joint':>12}")
        print("-" * 48)
        for i, snr in enumerate(snr_vals):
            print(f"  {snr:>+4.0f} dB {curves[ns]['l1'][i]:>11.1%} "
                  f"{curves[ns]['l2'][i]:>11.1%} {curves[ns]['l3'][i]:>11.1%}")

    # ── 画图: 3张子图 ──
    metric_labels = {
        'l1': 'Sample Count Accuracy',
        'l2': 'Subband Count Accuracy',
        'l3': 'Joint Accuracy',
    }
    metric_colors = {
        'l1': '#1565C0',
        'l2': '#EF6C00',
        'l3': '#C62828',
    }
    metric_markers = {
        'l1': 'o',
        'l2': 's',
        'l3': '^',
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, ns in enumerate(src_nums):
        ax = axes[idx]
        for mk in ['l1', 'l2', 'l3']:
            ax.plot(snr_vals, curves[ns][mk],
                    f'{metric_markers[mk]}-',
                    color=metric_colors[mk],
                    label=metric_labels[mk],
                    markersize=5, linewidth=2)
        ax.set_xlabel('Per-Source Subband SNR (dB)', fontsize=11)
        ax.set_ylabel('Accuracy', fontsize=11)
        ax.set_title(f'{ns}-Source Co-frequency (Δf=0)', fontsize=12)
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([min(snr_vals)-1, max(snr_vals)+1])
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.3)

    plt.suptitle(f'Exp 2-C: Co-frequency Multi-Source Detection vs SNR  (DL, {tag})',
                 fontsize=13)
    plt.tight_layout()
    save_name = f'exp2C_metrics_{tag}.png'
    plt.savefig(save_name, dpi=150)
    print(f"\nFigure saved: {save_name}")
    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()