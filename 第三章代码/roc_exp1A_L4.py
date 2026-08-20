"""
roc_exp1A_L4.py — Exp 1-A: Level 4 (Exact Match) ROC曲线

Level 4 ROC:
  Pfa: H0样本中任何子带被判有信号的比例
  Pd:  H1样本中19个子带全部预测正确的比例（严格匹配）

特征值统计量使用论文形式：
  AGM: 2*(TW-1)*log(AM/GM)
  SLE: λ_max / mean(λ)
ED-Hard: 写法B，不依赖已知噪底

用法:
  python roc_exp1A_L4.py B M10
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


# ═══════════════════════════════════════
#  系统参数
# ═══════════════════════════════════════
LEN  = 4096              # 采样点数
B_WIN = 10e6              # 子带宽度
FS    = 100e6             # 采样率
TW   = LEN * B_WIN / FS  # 时间带宽积 = 409.6


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
def count_segments(band_det):
    N, N_sub = band_det.shape
    count = np.zeros(N, dtype=np.int64)
    for i in range(N):
        segs = 0; in_seg = False
        for k in range(N_sub):
            if band_det[i, k] and not in_seg:
                segs += 1; in_seg = True
            elif not band_det[i, k]:
                in_seg = False
        count[i] = segs
    return count

def compute_eigenvalues(cov_mat):
    N, N_sub, M, _ = cov_mat.shape
    eigs = np.zeros((N, N_sub, M), dtype=np.float64)
    for i in range(N):
        for k in range(N_sub):
            ev = np.real(np.linalg.eigvalsh(cov_mat[i, k]))
            eigs[i, k] = np.sort(ev)[::-1]
    return np.clip(eigs, 1e-30, None)

def dl_preprocess(spectra_raw):
    spectra = np.log(spectra_raw + 1.0)
    for i in range(len(spectra)):
        s = spectra[i]; mu = s.mean(); std = s.std() + 1e-6
        spectra[i] = (s - mu) / std
    return spectra

def dl_get_probs(model, spectra_dl, device):
    probs_all = []
    with torch.no_grad():
        for i in range(0, len(spectra_dl), 64):
            j = min(i + 64, len(spectra_dl))
            x = torch.from_numpy(spectra_dl[i:j]).to(device)
            bl = model(x)
            probs_all.append(torch.sigmoid(bl).cpu().numpy())
    return np.concatenate(probs_all)


# ═══════════════════════════════════════
#  子带级检测函数
# ═══════════════════════════════════════
def dl_band_det_at_threshold(probs, th):
    return (probs > th).any(axis=1)

def ed_hard_band_det_at_threshold(per_sta, th, K=2):
    return (per_sta > th).sum(axis=-1) >= K

def eig_band_det_at_threshold(T_sub, th):
    return T_sub > th


# ═══════════════════════════════════════
#  Level 4 ROC
# ═══════════════════════════════════════
def compute_roc_l4_dl(probs_h0, probs_h1, band_true_h1, band_ignore_h1,
                       n_thresholds=500):
    thresholds = np.linspace(0.01, 0.999, n_thresholds)
    pfas, pds = [], []
    for th in thresholds:
        bd_h0 = dl_band_det_at_threshold(probs_h0, th)
        pfa = (bd_h0.any(axis=1)).mean()
        bd_h1 = dl_band_det_at_threshold(probs_h1, th)
        match = ((bd_h1 == band_true_h1) | band_ignore_h1).all(axis=1)
        pd = match.mean()
        pfas.append(pfa); pds.append(pd)
    return np.array(pfas), np.array(pds)

def compute_roc_l4_ed_hard(per_sta_h0, per_sta_h1,
                            band_true_h1, band_ignore_h1,
                            K=2, n_thresholds=500):
    lo = min(per_sta_h0.min(), per_sta_h1.min()) * 0.9
    hi = max(per_sta_h0.max(), per_sta_h1.max()) * 1.1
    thresholds = np.linspace(lo, hi, n_thresholds)
    pfas, pds = [], []
    for th in thresholds:
        bd_h0 = ed_hard_band_det_at_threshold(per_sta_h0, th, K)
        pfa = (bd_h0.any(axis=1)).mean()
        bd_h1 = ed_hard_band_det_at_threshold(per_sta_h1, th, K)
        match = ((bd_h1 == band_true_h1) | band_ignore_h1).all(axis=1)
        pd = match.mean()
        pfas.append(pfa); pds.append(pd)
    return np.array(pfas), np.array(pds)

def compute_roc_l4_eig(T_h0, T_h1, band_true_h1, band_ignore_h1,
                        n_thresholds=500):
    lo = min(T_h0.min(), T_h1.min()) * 0.9
    hi = max(T_h0.max(), T_h1.max()) * 1.1
    thresholds = np.linspace(lo, hi, n_thresholds)
    pfas, pds = [], []
    for th in thresholds:
        bd_h0 = eig_band_det_at_threshold(T_h0, th)
        pfa = (bd_h0.any(axis=1)).mean()
        bd_h1 = eig_band_det_at_threshold(T_h1, th)
        match = ((bd_h1 == band_true_h1) | band_ignore_h1).all(axis=1)
        pd = match.mean()
        pfas.append(pfa); pds.append(pd)
    return np.array(pfas), np.array(pds)


# ═══════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════
def load_h0_data(val_path):
    with h5py.File(val_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32).transpose(3, 2, 1, 0)
        src_count   = np.array(f['src_count_all'], dtype=np.int64).flatten()
        cov_real    = np.array(f['cov_mat_real_all'], dtype=np.float32)
        cov_imag    = np.array(f['cov_mat_imag_all'], dtype=np.float32)
    cov_mat = cov_real.transpose(3, 2, 1, 0) + 1j * cov_imag.transpose(3, 2, 1, 0)
    h0 = (src_count == 0)
    return spectra_raw[h0], cov_mat[h0]

def load_h1_data(exp_path, target_snr):
    with h5py.File(exp_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32).transpose(3, 2, 1, 0)
        cov_real    = np.array(f['cov_mat_real_all'], dtype=np.float32)
        cov_imag    = np.array(f['cov_mat_imag_all'], dtype=np.float32)
        snr_target  = np.array(f['snr_target_all'], dtype=np.float32).flatten()
        band_mask   = np.array(f['band_mask_all'], dtype=np.float32).transpose(2, 1, 0)
        ignore_mask = np.array(f['ignore_mask_all'], dtype=np.float32).transpose(2, 1, 0)
    cov_mat = cov_real.transpose(3, 2, 1, 0) + 1j * cov_imag.transpose(3, 2, 1, 0)
    band_true   = (band_mask > 0.5).any(axis=1)
    band_ignore = (ignore_mask > 0.5).any(axis=1)
    mask = np.abs(snr_target - target_snr) < 0.5
    return (spectra_raw[mask], cov_mat[mask],
            band_true[mask], band_ignore[mask])


# ═══════════════════════════════════════
#  特征值统计量计算（论文公式）
# ═══════════════════════════════════════
def compute_eig_stats(eigs):
    M = eigs.shape[-1]
    T_agm_ratio = eigs.mean(-1) / np.exp(np.log(eigs).mean(-1))
    T_agm = 2 * (TW - 1) * np.log(T_agm_ratio)
    T_mme = eigs[:,:,0] / eigs[:,:,-1]
    T_sle = eigs[:,:,0] / eigs.mean(-1)
    T_emr = (M * (eigs**2).sum(-1)) / eigs.sum(-1)**2
    return T_agm, T_mme, T_sle, T_emr


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    args = sys.argv[1:]
    tag = 'B'
    for a in args:
        if a == 'B': tag = 'B'
        elif a.startswith('M') and a[1:].isdigit(): tag += f'_{a}'

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    ckpt = torch.load(f'best_model_v26_{tag}.pth', map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    print(f"Model: {tag} ({cfg['mode']}) max_src={cfg['max_src']}")

    val_path = '/mnt/data/ltzdata/val_data.mat'
    exp_path = '/mnt/data/ltzdata/ctrl_exp1A.mat'

    # ── 加载H0 ──
    print("\nLoading H0 (val 0-source)...")
    spec_h0, cv_h0 = load_h0_data(val_path)
    n_h0 = len(spec_h0)
    print(f"  {n_h0} H0 samples")

    dl_h0 = dl_preprocess(spec_h0)
    probs_h0 = dl_get_probs(model, dl_h0, device)

    M = cv_h0.shape[2]
    per_sta_h0 = np.real(cv_h0[:, :, range(M), range(M)])

    print("  Computing H0 eigenvalues...")
    eigs_h0 = compute_eigenvalues(cv_h0)
    T_agm_h0, T_mme_h0, T_sle_h0, T_emr_h0 = compute_eig_stats(eigs_h0)

    # ── SNR点 ──
    snr_points = [-9, -7, -5]

    METHOD_NAMES  = ['DL', 'ED-Hard-K2', 'AGM', 'MME', 'SLE', 'EMR']
    METHOD_COLORS = {
        'DL':'#2196F3', 'ED-Hard-K2':'#00796B',
        'AGM':'#FF9800', 'MME':'#9C27B0',
        'SLE':'#F44336', 'EMR':'#795548'
    }
    METHOD_LINES = {
        'DL':'-', 'ED-Hard-K2':'--',
        'AGM':'-.', 'MME':':', 'SLE':'-', 'EMR':'--'
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, snr in enumerate(snr_points):
        ax = axes[idx]
        print(f"\n--- SNR = {snr:+d} dB ---")

        spec_h1, cv_h1, band_true_h1, band_ignore_h1 = load_h1_data(exp_path, snr)
        n_h1 = len(spec_h1)
        print(f"  {n_h1} H1 samples")

        # DL
        dl_h1 = dl_preprocess(spec_h1)
        probs_h1 = dl_get_probs(model, dl_h1, device)
        pfas, pds = compute_roc_l4_dl(probs_h0, probs_h1,
                                       band_true_h1, band_ignore_h1)
        auc = np.abs(np.trapezoid(pds, pfas))
        ax.plot(pfas, pds, color=METHOD_COLORS['DL'],
                linestyle=METHOD_LINES['DL'],
                linewidth=2, label=f'DL (AUC={auc:.3f})')
        print(f"  DL: AUC={auc:.3f}")

        # ED-Hard K=2
        per_sta_h1 = np.real(cv_h1[:, :, range(M), range(M)])
        pfas, pds = compute_roc_l4_ed_hard(per_sta_h0, per_sta_h1,
                                            band_true_h1, band_ignore_h1, K=2)
        auc = np.abs(np.trapezoid(pds, pfas))
        ax.plot(pfas, pds, color=METHOD_COLORS['ED-Hard-K2'],
                linestyle=METHOD_LINES['ED-Hard-K2'],
                linewidth=1.5, label=f'ED-Hard-K2 (AUC={auc:.3f})')
        print(f"  ED-Hard-K2: AUC={auc:.3f}")

        # 特征值方法
        print("  Computing H1 eigenvalues...")
        eigs_h1 = compute_eigenvalues(cv_h1)
        T_agm_h1, T_mme_h1, T_sle_h1, T_emr_h1 = compute_eig_stats(eigs_h1)

        for name, Th0, Th1 in [('AGM', T_agm_h0, T_agm_h1),
                                 ('MME', T_mme_h0, T_mme_h1),
                                 ('SLE', T_sle_h0, T_sle_h1),
                                 ('EMR', T_emr_h0, T_emr_h1)]:
            pfas, pds = compute_roc_l4_eig(Th0, Th1,
                                            band_true_h1, band_ignore_h1)
            auc = np.abs(np.trapezoid(pds, pfas))
            ax.plot(pfas, pds, color=METHOD_COLORS[name],
                    linestyle=METHOD_LINES[name],
                    linewidth=1.5, label=f'{name} (AUC={auc:.3f})')
            print(f"  {name}: AUC={auc:.3f}")

        # 装饰
        ax.plot([0, 1], [0, 1], 'k:', alpha=0.3, linewidth=0.5)
        ax.axvline(x=0.01, color='gray', linestyle=':', alpha=0.3)
        ax.set_xlabel('Pfa (any false alarm band)', fontsize=11)
        ax.set_ylabel('Pd (exact match)', fontsize=11)
        ax.set_title(f'SNR = {snr:+d} dB', fontsize=12)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.2)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.suptitle('Exp 1-A: Level 4 (Exact Match) ROC Curves', fontsize=14)
    plt.tight_layout()
    save_name = f'roc_exp1A_L4_{tag}.png'
    plt.savefig(save_name, dpi=150)
    print(f"\nFigure saved: {save_name}")
    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()