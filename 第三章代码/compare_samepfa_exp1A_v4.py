"""
compare_samepfa_exp1A_v4.py — Exp 1-A: 4个指标对比 + ED-Hard K扫描

特征值统计量使用论文形式：
  AGM: 2*(TW-1)*log(AM/GM)
  SLE: λ_max / mean(λ)
  EMR: M*Σλ²/(Σλ)²
  MME: λ_max/λ_min
ED-Hard: 写法B，直接比较物理能量，不依赖已知噪底

用法:
  python compare_samepfa_exp1A_v4.py B M10
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

from chapter_runtime import (
    checkpoint_path,
    data_dir as runtime_data_dir,
    device as runtime_device,
    output_path,
)


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


# ═══════════════════════════════════════
#  获取各方法的子带级检测结果 (N, N_sub)
# ═══════════════════════════════════════
def get_band_det_dl(model, spectra_dl, device, threshold):
    N = len(spectra_dl)
    probs_list = []
    with torch.no_grad():
        for i in range(0, N, 64):
            j = min(i+64, N)
            x = torch.from_numpy(spectra_dl[i:j]).to(device)
            bl = model(x)
            probs_list.append(torch.sigmoid(bl).cpu().numpy())
    probs = np.concatenate(probs_list)
    band_det = (probs > threshold).any(axis=1)
    return band_det

def get_band_det_ed_hard(cov_mat, phys_threshold, K=2):
    M = cov_mat.shape[2]
    per_sta = np.real(cov_mat[:, :, range(M), range(M)])
    return (per_sta > phys_threshold).sum(axis=-1) >= K

def get_band_det_eig(eigs, method, alpha):
    M = eigs.shape[-1]
    if method == 'AGM':
        am_gm = eigs.mean(-1) / np.exp(np.log(eigs).mean(-1))
        T = 2 * (TW - 1) * np.log(am_gm)
    elif method == 'MME':
        T = eigs[:,:,0] / eigs[:,:,-1]
    elif method == 'SLE':
        T = eigs[:,:,0] / eigs.mean(-1)
    elif method == 'EMR':
        T = M * (eigs**2).sum(-1) / eigs.sum(-1)**2
    return T > alpha


# ═══════════════════════════════════════
#  阈值校准
# ═══════════════════════════════════════
def calibrate_all_thresholds(val_path, model, device, target_pfa=0.01):
    print(f"Loading val set: {val_path}")
    with h5py.File(val_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32)
        src_count   = np.array(f['src_count_all'], dtype=np.int64).flatten()
        sub_energy  = np.array(f['sub_energy_all'], dtype=np.float32)
        cov_real    = np.array(f['cov_mat_real_all'], dtype=np.float32)
        cov_imag    = np.array(f['cov_mat_imag_all'], dtype=np.float32)

    spectra = spectra_raw.transpose(3, 2, 1, 0)
    if sub_energy.shape[0] != len(src_count):
        sub_energy = sub_energy.T
    cov_mat = cov_real.transpose(3, 2, 1, 0) + 1j * cov_imag.transpose(3, 2, 1, 0)

    spectra_dl = np.log(spectra + 1.0)
    for i in range(len(spectra_dl)):
        s = spectra_dl[i]; mu = s.mean(); std = s.std() + 1e-6
        spectra_dl[i] = (s - mu) / std

    h0 = (src_count == 0)
    n_h0 = h0.sum()
    print(f"  Total: {len(src_count)}, 0-source: {n_h0}")

    cv_h0 = cov_mat[h0]
    dl_h0 = spectra_dl[h0]
    M = cv_h0.shape[2]

    thresholds = {}

    # DL
    print("  Calibrating DL...")
    probs_h0_list = []
    with torch.no_grad():
        for i in range(0, len(dl_h0), 64):
            j = min(i+64, len(dl_h0))
            x = torch.from_numpy(dl_h0[i:j]).to(device)
            bl = model(x)
            probs_h0_list.append(torch.sigmoid(bl).cpu().numpy())
    probs_h0 = np.concatenate(probs_h0_list)

    best_th_dl = 0.999
    for th in np.arange(0.01, 1.0, 0.005):
        band_det = (probs_h0 > th).any(axis=1)
        pfa = (count_segments(band_det) > 0).mean()
        if pfa <= target_pfa:
            best_th_dl = th; break
    band_det = (probs_h0 > best_th_dl).any(axis=1)
    pfa_dl = (count_segments(band_det) > 0).mean()
    thresholds['DL'] = best_th_dl
    print(f"  DL: BAND_THRESHOLD={best_th_dl:.3f} (Pfa={pfa_dl:.4f})")

    # 传统方法
    print("  Computing eigenvalues...")
    eigs_h0 = compute_eigenvalues(cv_h0)
    per_sta_h0 = np.real(cv_h0[:, :, range(M), range(M)])
    # 论文公式
    T_agm_ratio = eigs_h0.mean(-1) / np.exp(np.log(eigs_h0).mean(-1))
    T_agm = 2 * (TW - 1) * np.log(T_agm_ratio)
    T_mme = eigs_h0[:,:,0] / eigs_h0[:,:,-1]
    T_sle = eigs_h0[:,:,0] / eigs_h0.mean(-1)
    T_emr = (M * (eigs_h0**2).sum(-1)) / eigs_h0.sum(-1)**2

    def scan_pfa(T_h0, name, n_steps=2000):
        lo = np.percentile(T_h0, 50); hi = T_h0.max() * 1.1
        for th in np.linspace(lo, hi, n_steps):
            pfa = (count_segments(T_h0 > th) > 0).mean()
            if pfa <= target_pfa:
                print(f"  {name}: alpha={th:.4f} (Pfa={pfa:.4f})")
                return th
        return hi

    def scan_pfa_ed_hard(per_sta_h0, K=2, n_steps=2000):
        lo = np.percentile(per_sta_h0, 50)
        hi = per_sta_h0.max() * 1.1
        for th in np.linspace(lo, hi, n_steps):
            band_det = (per_sta_h0 > th).sum(axis=-1) >= K
            pfa = (count_segments(band_det) > 0).mean()
            if pfa <= target_pfa:
                print(f"  ED-Hard(K={K}): phys_threshold={th:.4e}W (Pfa={pfa:.4f})")
                return th
        return hi

    for K in [1, 2, 3, 4]:
        thresholds[f'ED-Hard-K{K}'] = scan_pfa_ed_hard(per_sta_h0, K=K)
    thresholds['AGM'] = scan_pfa(T_agm, 'AGM')
    thresholds['MME'] = scan_pfa(T_mme, 'MME')
    thresholds['SLE'] = scan_pfa(T_sle, 'SLE')
    thresholds['EMR'] = scan_pfa(T_emr, 'EMR')
    return thresholds


# ═══════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════
def load_exp1a(mat_path):
    with h5py.File(mat_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32)
        sub_energy  = np.array(f['sub_energy_all'], dtype=np.float32)
        cov_real    = np.array(f['cov_mat_real_all'], dtype=np.float32)
        cov_imag    = np.array(f['cov_mat_imag_all'], dtype=np.float32)
        snr_target  = np.array(f['snr_target_all'], dtype=np.float32).flatten()
        Pt_target   = np.array(f['Pt_target_all'], dtype=np.float32).flatten()
        band_mask   = np.array(f['band_mask_all'], dtype=np.float32)
        ignore_mask = np.array(f['ignore_mask_all'], dtype=np.float32)

    spectra = spectra_raw.transpose(3, 2, 1, 0)
    if sub_energy.shape[0] != len(snr_target):
        sub_energy = sub_energy.T
    cov_mat = cov_real.transpose(3, 2, 1, 0) + 1j * cov_imag.transpose(3, 2, 1, 0)

    band_mask   = band_mask.transpose(2, 1, 0)
    ignore_mask = ignore_mask.transpose(2, 1, 0)
    band_true   = (band_mask > 0.5).any(axis=1)
    band_ignore = (ignore_mask > 0.5).any(axis=1)

    spectra_dl = np.log(spectra + 1.0)
    for i in range(len(spectra_dl)):
        s = spectra_dl[i]; mu = s.mean(); std = s.std() + 1e-6
        spectra_dl[i] = (s - mu) / std

    return spectra_dl, sub_energy, cov_mat, snr_target, Pt_target, band_true, band_ignore


# ═══════════════════════════════════════
#  4个指标计算
# ═══════════════════════════════════════
def compute_metrics(band_det, band_true, band_ignore=None):
    count_pred = count_segments(band_det)
    count_true = count_segments(band_true)

    metrics = {}
    metrics['pd'] = (count_pred >= 1).mean()
    metrics['count_acc'] = (count_pred == count_true).mean()

    if band_ignore is None:
        valid = np.ones_like(band_true, dtype=bool)
    else:
        valid = ~band_ignore

    tp = ((band_det == 1) & (band_true == 1) & valid).sum()
    fp = ((band_det == 1) & (band_true == 0) & valid).sum()
    fn = ((band_det == 0) & (band_true == 1) & valid).sum()
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    metrics['f1'] = f1

    match_per_pos = (band_det == band_true) | (~valid)
    exact = match_per_pos.all(axis=1)
    metrics['exact_match'] = exact.mean()
    return metrics


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    args = sys.argv[1:]
    tag = 'B'
    for a in args:
        if a == 'B': tag = 'B'
        elif a.startswith('M') and a[1:].isdigit(): tag += f'_{a}'

    device = runtime_device()
    print(f"Device: {device}")

    ckpt = torch.load(checkpoint_path(f'best_model_v26_{tag}.pth'), map_location=device,
                       weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')).to(device)
    model.load_state_dict(ckpt['model']); model.eval()
    print(f"Model: {tag} ({cfg['mode']}) max_src={cfg['max_src']}")

    print(f"\n--- Threshold calibration (all methods, Pfa=0.01) ---")
    val_path = runtime_data_dir() / 'val_data.mat'
    thresholds = calibrate_all_thresholds(val_path, model, device, target_pfa=0.01)

    print(f"\n--- Loading Exp 1-A data ---")
    exp_path = runtime_data_dir() / 'ctrl_exp1A.mat'
    spectra_dl, sub_energy, cov_mat, snr_target, Pt_target, band_true, band_ignore = load_exp1a(exp_path)
    N = len(snr_target)
    print(f"  {N} samples, {band_ignore.sum()} ignore positions total")

    snr_vals = sorted(np.unique(snr_target))
    print(f"  SNR points: {[f'{s:+.0f}' for s in snr_vals]}")

    print("  Computing eigenvalues...")
    eigs = compute_eigenvalues(cov_mat)

    print("\n--- Running all methods ---")
    band_dets = {}
    band_dets['DL'] = get_band_det_dl(model, spectra_dl, device, thresholds['DL'])
    for K in [1, 2, 3, 4]:
        band_dets[f'ED-Hard-K{K}'] = get_band_det_ed_hard(
            cov_mat, thresholds[f'ED-Hard-K{K}'], K=K)
    for m in ['AGM', 'MME', 'SLE', 'EMR']:
        band_dets[m] = get_band_det_eig(eigs, m, thresholds[m])

    METHOD_NAMES = ['DL',
                    'ED-Hard-K1', 'ED-Hard-K2', 'ED-Hard-K3', 'ED-Hard-K4',
                    'AGM', 'MME', 'SLE', 'EMR']
    METHOD_COLORS = {
        'DL':'#1565C0',
        'ED-Hard-K1':'#2E7D32', 'ED-Hard-K2':'#00ACC1',
        'ED-Hard-K3':'#66BB6A', 'ED-Hard-K4':'#795548',
        'AGM':'#EF6C00', 'MME':'#7B1FA2',
        'SLE':'#C62828', 'EMR':'#F9A825',
    }
    METHOD_MARKERS = {
        'DL':'o',
        'ED-Hard-K1':'s', 'ED-Hard-K2':'s', 'ED-Hard-K3':'s', 'ED-Hard-K4':'s',
        'AGM':'^', 'MME':'D', 'SLE':'v', 'EMR':'P',
    }

    METRIC_KEYS   = ['pd', 'count_acc', 'f1', 'exact_match']
    METRIC_TITLES = {
        'pd':          'Level 1: Sample Pd (count≥1)',
        'count_acc':   'Level 2: Count Accuracy (count==1)',
        'f1':          'Level 3: Subband F1',
        'exact_match': 'Level 4: Exact Match (all 19 bands)'
    }
    METRIC_YLABELS = {
        'pd':'Pd', 'count_acc':'Count Accuracy',
        'f1':'Subband F1', 'exact_match':'Exact Match Rate'
    }

    curves = {metric: {m: [] for m in METHOD_NAMES} for metric in METRIC_KEYS}

    for snr in snr_vals:
        mask = np.abs(snr_target - snr) < 0.5
        for m in METHOD_NAMES:
            metrics = compute_metrics(band_dets[m][mask], band_true[mask],
                                       band_ignore[mask])
            for k in METRIC_KEYS:
                curves[k][m].append(metrics[k])

    for k in METRIC_KEYS:
        print(f"\n===== {METRIC_TITLES[k]} =====")
        print(f"{'SNR(dB)':>8} {'Pt(W)':>10}", end='')
        for m in METHOD_NAMES:
            print(f" {m:>11}", end='')
        print()
        print("-" * (20 + 12*len(METHOD_NAMES)))

        for i, snr in enumerate(snr_vals):
            mask = np.abs(snr_target - snr) < 0.5
            pt = Pt_target[mask][0]
            print(f"  {snr:>+4.0f} dB  {pt:>8.6f}", end='')
            for m in METHOD_NAMES:
                print(f" {curves[k][m][i]:>10.1%}", end='')
            print()

    print(f"\n--- Thresholds (Pfa=0.01) ---")
    for m in METHOD_NAMES:
        if m.startswith('ED-Hard'):
            print(f"  {m}: {thresholds[m]:.4e} W (physical threshold)")
        else:
            print(f"  {m}: {thresholds[m]:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, k in enumerate(METRIC_KEYS):
        ax = axes[idx]
        for m in METHOD_NAMES:
            ax.plot(snr_vals, curves[k][m], f'{METHOD_MARKERS[m]}-',
                    color=METHOD_COLORS[m], label=m, markersize=5, linewidth=1.5)
        ax.set_xlabel('Subband SNR (dB)', fontsize=11)
        ax.set_ylabel(METRIC_YLABELS[k], fontsize=11)
        ax.set_title(METRIC_TITLES[k], fontsize=12)
        ax.legend(fontsize=7, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([min(snr_vals)-1, max(snr_vals)+1])
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.3)

    plt.suptitle(f'Exp 1-A: Multi-Level Comparison with ED-Hard K Sweep (Pfa=0.01)',
                 fontsize=14)
    plt.tight_layout()
    save_name = f'exp1A_metrics_{tag}.png'
    plt.savefig(output_path('compare_samepfa_exp1A_v4', save_name), dpi=150)
    print(f"\nFigure saved: {save_name}")
    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()
