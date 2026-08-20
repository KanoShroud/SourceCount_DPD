"""
compare_exp2B.py — Exp 2-B: 两源频率间距扫描 多方法对比

实验设计:
  2源，固定位置(30°/60°, 1500m)，固定每源子带SNR=+15dB
  扫描 Δf/SR = 0, 0.2, 0.4, ..., 4.0  (21点, 每点500样本)

对比方法:
  DL (BAND_THRESHOLD=0.50), ED-Hard-K1/K2/K3/K4, AGM, MME, SLE, EMR

阈值策略:
  DL:     BAND_THRESHOLD=0.50（训练验证最优）
  传统方法: val 集 H0 上校准 Pfa ≤ 0.01

指标 (3层):
  Level 1 — Sample Count Acc:   样本级 predicted_count == true_count (2)
  Level 2 — Subband Count Acc:  子带级 pred_src_count == true_src_count (跳过ignore)
  Level 3 — Joint Acc:          L1正确 且 所有可评子带的源数全部正确

DL子带源数:  Σ_s 1{slot_s 在子带k预测为1}  → 0,1,2,...
传统方法子带源数: 0 或 1 (二值检测)

用法:
  python compare_exp2B.py B M10
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
LEN = 4096
B_WIN = 10e6
FS = 100e6
TW = LEN * B_WIN / FS  # 409.6

BAND_THRESHOLD_DL = 0.50  # DL 训练最优阈值


# ═══════════════════════════════════════
#  模型定义 (和 train_v26 一致)
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
        self.n_sub = n_sub;
        self.max_src = max_src
        self.feat_dim = feat_dim;
        self.mode = mode
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), ResBlock(32, 0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), ResBlock(64, 0.1),
            nn.Conv2d(64, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(), ResBlock(feat_dim, 0.1),
            nn.AdaptiveAvgPool2d(1))
        if mode == 'transformer':
            self.pos_embed = nn.Parameter(torch.randn(1, n_sub, feat_dim) * 0.02)
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
        feat = self.backbone(x.reshape(B * S, 1, H, W))
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
def count_segments_raw(band_det):
    """简单段计数（用于Pfa校准）: (N, N_sub) bool → (N,) int"""
    N, N_sub = band_det.shape
    count = np.zeros(N, dtype=np.int64)
    for i in range(N):
        segs = 0;
        in_seg = False
        for k in range(N_sub):
            if band_det[i, k] and not in_seg:
                segs += 1;
                in_seg = True
            elif not band_det[i, k]:
                in_seg = False
        count[i] = segs
    return count


def count_segments(band_det, min_seg_width=2):
    """段计数+旁瓣过滤（用于评估）: (N, N_sub) bool → (N,) int

    去除宽度<min_seg_width的孤立检测段。
    物理含义：信号带宽≥13MHz，至少占2个连续子带（10MHz），
    孤立的单子带检测是旁瓣或虚警。
    """
    N, N_sub = band_det.shape
    count = np.zeros(N, dtype=np.int64)
    for i in range(N):
        segments = []
        k = 0
        while k < N_sub:
            if band_det[i, k]:
                start = k
                while k < N_sub and band_det[i, k]:
                    k += 1
                segments.append((start, k - start))
            else:
                k += 1
        count[i] = sum(1 for _, w in segments if w >= min_seg_width)
    return count


def compute_eigenvalues(cov_mat):
    """(N, N_sub, M, M) → (N, N_sub, M) 降序特征值"""
    N, N_sub, M, _ = cov_mat.shape
    eigs = np.zeros((N, N_sub, M), dtype=np.float64)
    for i in range(N):
        for k in range(N_sub):
            ev = np.real(np.linalg.eigvalsh(cov_mat[i, k]))
            eigs[i, k] = np.sort(ev)[::-1]
    return np.clip(eigs, 1e-30, None)


def dl_preprocess(spectra):
    """log + sample z-score (和训练一致)"""
    out = np.log(spectra + 1.0)
    for i in range(len(out)):
        s = out[i];
        mu = s.mean();
        std = s.std() + 1e-6
        out[i] = (s - mu) / std
    return out


# ═══════════════════════════════════════
#  各方法 → (sample_count, subband_count)
# ═══════════════════════════════════════
def predict_dl(model, spectra_dl, device, threshold):
    """DL: 子带源数 = 各slot预测之和, 样本源数 = 非空slot数"""
    N = len(spectra_dl)
    N_sub = model.n_sub
    subband_count = np.zeros((N, N_sub), dtype=np.int64)
    sample_count = np.zeros(N, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, N, 64):
            j = min(i + 64, N)
            x = torch.from_numpy(spectra_dl[i:j]).to(device)
            bl = model(x)
            bp = (torch.sigmoid(bl) > threshold).long()  # (B, max_src, N_sub)
            subband_count[i:j] = bp.sum(dim=1).cpu().numpy()
            sample_count[i:j] = (bp.sum(dim=-1) > 0).sum(dim=-1).cpu().numpy()
    return sample_count, subband_count


def predict_ed_hard(cov_mat, phys_threshold, K=2):
    """ED-Hard: 子带源数 = 0/1, 样本源数 = 连续段数"""
    M = cov_mat.shape[2]
    per_sta = np.real(cov_mat[:, :, range(M), range(M)])  # (N, N_sub, M)
    band_det = (per_sta > phys_threshold).sum(axis=-1) >= K  # (N, N_sub)
    return count_segments(band_det), band_det.astype(np.int64)


def predict_eig(eigs, method, alpha):
    """特征值方法: 子带源数 = 0/1, 样本源数 = 连续段数"""
    M = eigs.shape[-1]
    if method == 'AGM':
        am_gm = eigs.mean(-1) / np.exp(np.log(eigs).mean(-1))
        T = 2 * (TW - 1) * np.log(am_gm)
    elif method == 'MME':
        T = eigs[:, :, 0] / eigs[:, :, -1]
    elif method == 'SLE':
        T = eigs[:, :, 0] / eigs.mean(-1)
    elif method == 'EMR':
        T = M * (eigs ** 2).sum(-1) / eigs.sum(-1) ** 2
    band_det = T > alpha
    return count_segments(band_det), band_det.astype(np.int64)


# ═══════════════════════════════════════
#  三层指标
# ═══════════════════════════════════════
def compute_metrics(sample_count_pred, subband_count_pred,
                    sample_count_true, subband_count_true, ignore_any):
    """
    sample_count_{pred,true}: (N,) int
    subband_count_{pred,true}: (N, N_sub) int
    ignore_any: (N, N_sub) bool — 任一slot有ignore则该子带跳过
    """
    evaluable = ~ignore_any

    # Level 1: 样本级个数准确率
    sample_correct = (sample_count_pred == sample_count_true)
    l1 = sample_correct.mean()

    # Level 2: 子带级源数准确率
    subband_correct = (subband_count_pred == subband_count_true) & evaluable
    l2 = subband_correct.sum() / max(evaluable.sum(), 1)

    # Level 3: Joint (样本count对 且 所有可评子带源数全对)
    all_sub_ok = ((subband_count_pred == subband_count_true) | ~evaluable).all(axis=1)
    l3 = (sample_correct & all_sub_ok).mean()

    return l1, l2, l3


# ═══════════════════════════════════════
#  传统方法阈值校准 (val H0, Pfa ≤ 0.01)
# ═══════════════════════════════════════
def calibrate_traditional_thresholds(val_path, target_pfa=0.01):
    print(f"Loading val set for calibration: {val_path}")
    with h5py.File(val_path, 'r') as f:
        src_count = np.array(f['src_count_all'], dtype=np.int64).flatten()
        cov_real = np.array(f['cov_mat_real_all'], dtype=np.float32)
        cov_imag = np.array(f['cov_mat_imag_all'], dtype=np.float32)

    cov_mat = cov_real.transpose(3, 2, 1, 0) + 1j * cov_imag.transpose(3, 2, 1, 0)
    h0 = (src_count == 0)
    cv_h0 = cov_mat[h0]
    n_h0 = h0.sum()
    M = cv_h0.shape[2]
    print(f"  Total: {len(src_count)}, H0: {n_h0}")

    thresholds = {}

    # ── 特征值 ──
    print("  Computing H0 eigenvalues...")
    eigs_h0 = compute_eigenvalues(cv_h0)

    T_agm_ratio = eigs_h0.mean(-1) / np.exp(np.log(eigs_h0).mean(-1))
    T_agm = 2 * (TW - 1) * np.log(T_agm_ratio)
    T_mme = eigs_h0[:, :, 0] / eigs_h0[:, :, -1]
    T_sle = eigs_h0[:, :, 0] / eigs_h0.mean(-1)
    T_emr = (M * (eigs_h0 ** 2).sum(-1)) / eigs_h0.sum(-1) ** 2

    def scan_pfa(T_h0, name, n_steps=2000):
        lo = np.percentile(T_h0, 50);
        hi = T_h0.max() * 1.1
        for th in np.linspace(lo, hi, n_steps):
            pfa = (count_segments_raw(T_h0 > th) > 0).mean()
            if pfa <= target_pfa:
                print(f"  {name}: alpha={th:.4f} (Pfa={pfa:.4f})")
                return th
        return hi

    thresholds['AGM'] = scan_pfa(T_agm, 'AGM')
    thresholds['MME'] = scan_pfa(T_mme, 'MME')
    thresholds['SLE'] = scan_pfa(T_sle, 'SLE')
    thresholds['EMR'] = scan_pfa(T_emr, 'EMR')

    # ── ED-Hard ──
    per_sta_h0 = np.real(cv_h0[:, :, range(M), range(M)])

    def scan_pfa_ed_hard(per_sta_h0, K=2, n_steps=2000):
        lo = np.percentile(per_sta_h0, 50)
        hi = per_sta_h0.max() * 1.1
        for th in np.linspace(lo, hi, n_steps):
            band_det = (per_sta_h0 > th).sum(axis=-1) >= K
            pfa = (count_segments_raw(band_det) > 0).mean()
            if pfa <= target_pfa:
                print(f"  ED-Hard-K{K}: phys_threshold={th:.4e}W (Pfa={pfa:.4f})")
                return th
        return hi

    for K in [1, 2, 3, 4]:
        thresholds[f'ED-Hard-K{K}'] = scan_pfa_ed_hard(per_sta_h0, K=K)

    return thresholds


# ═══════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════
def load_exp2b(mat_path):
    with h5py.File(mat_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32)
        cov_real = np.array(f['cov_mat_real_all'], dtype=np.float32)
        cov_imag = np.array(f['cov_mat_imag_all'], dtype=np.float32)
        band_mask = np.array(f['band_mask_all'], dtype=np.float32)
        ignore_mask = np.array(f['ignore_mask_all'], dtype=np.float32)
        delta_f_all = np.array(f['delta_f_all'], dtype=np.float32).flatten()
        src_count = np.array(f['src_count_all'], dtype=np.int64).flatten()
        snr_target = np.array(f['snr_target_all'], dtype=np.float32).flatten()
        target_snr = int(snr_target[0])

    spectra = spectra_raw.transpose(3, 2, 1, 0)
    cov_mat = cov_real.transpose(3, 2, 1, 0) + 1j * cov_imag.transpose(3, 2, 1, 0)
    band_mask = band_mask.transpose(2, 1, 0)  # (N, max_src, N_sub)
    ignore_mask = ignore_mask.transpose(2, 1, 0)

    # 真值: 子带级源数 和 样本级源数
    subband_count_true = (band_mask > 0.5).astype(np.int64).sum(axis=1)  # (N, N_sub)
    sample_count_true = src_count  # (N,), 全是2

    # ignore: 任一slot有ignore则该子带跳过
    ignore_any = (ignore_mask > 0.5).any(axis=1)  # (N, N_sub)

    # DL预处理
    spectra_dl = dl_preprocess(spectra)

    return (spectra_dl, cov_mat, delta_f_all,
            sample_count_true, subband_count_true, ignore_any, target_snr)


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    args = sys.argv[1:]
    tag = 'A'
    for a in args:
        if a == 'B':
            tag = 'B'
        elif a.startswith('M') and a[1:].isdigit():
            tag += f'_{a}'

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 加载模型 ──
    ckpt = torch.load(f'best_model_v26_{tag}.pth', map_location=device,
                      weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')).to(device)
    model.load_state_dict(ckpt['model']);
    model.eval()
    print(f"Model: {tag} ({cfg['mode']}) max_src={cfg['max_src']}")
    print(f"DL threshold: {BAND_THRESHOLD_DL}")

    # ── 传统方法阈值校准 ──
    print(f"\n--- Traditional threshold calibration (Pfa=0.01) ---")
    val_path = '/mnt/data/ltzdata/val_data.mat'
    thresholds = calibrate_traditional_thresholds(val_path, target_pfa=0.01)

    # ── 加载 Exp2B 数据 ──
    print(f"\n--- Loading Exp 2-B data ---")
    exp_path = '/mnt/data/ltzdata/ctrl_exp2B.mat'
    (spectra_dl, cov_mat, delta_f_all,
     sample_count_true, subband_count_true, ignore_any, target_snr) = load_exp2b(exp_path)
    N = len(delta_f_all)
    print(f"  {N} samples")

    delta_f_vals = sorted(np.unique(delta_f_all))
    print(f"  Δf/SR points: {[f'{d:.1f}' for d in delta_f_vals]}")

    # ── 特征值预计算 ──
    print("  Computing eigenvalues...")
    eigs = compute_eigenvalues(cov_mat)

    # ── 各方法预测 ──
    print("\n--- Running all methods ---")
    predictions = {}

    print("  DL...")
    predictions['DL'] = predict_dl(model, spectra_dl, device, BAND_THRESHOLD_DL)

    for K in [1, 2, 3, 4]:
        name = f'ED-Hard-K{K}'
        print(f"  {name}...")
        predictions[name] = predict_ed_hard(cov_mat, thresholds[name], K=K)

    for m in ['AGM', 'MME', 'SLE', 'EMR']:
        print(f"  {m}...")
        predictions[m] = predict_eig(eigs, m, thresholds[m])

    # ── 方法列表与样式 ──
    METHOD_NAMES = ['DL',
                    'ED-Hard-K1', 'ED-Hard-K2', 'ED-Hard-K3', 'ED-Hard-K4',
                    'AGM', 'MME', 'SLE', 'EMR']
    METHOD_COLORS = {
        'DL': '#1565C0',  # 深蓝
        'ED-Hard-K1': '#2E7D32',  # 深绿
        'ED-Hard-K2': '#00ACC1',  # 青色
        'ED-Hard-K3': '#66BB6A',  # 浅绿
        'ED-Hard-K4': '#795548',  # 棕色
        'AGM': '#EF6C00',  # 橙色
        'MME': '#7B1FA2',  # 紫色
        'SLE': '#C62828',  # 红色
        'EMR': '#F9A825',  # 黄色
    }
    METHOD_MARKERS = {
        'DL': 'o',
        'ED-Hard-K1': 's', 'ED-Hard-K2': 's', 'ED-Hard-K3': 's', 'ED-Hard-K4': 's',
        'AGM': '^', 'MME': 'D', 'SLE': 'v', 'EMR': 'P',
    }

    METRIC_KEYS = ['l1', 'l2', 'l3']
    METRIC_TITLES = {
        'l1': 'Level 1: Sample Count Accuracy (count==2)',
        'l2': 'Level 2: Subband Source-Count Accuracy',
        'l3': 'Level 3: Joint Accuracy (L1 correct + all subbands correct)',
    }
    METRIC_YLABELS = {
        'l1': 'Sample Count Accuracy',
        'l2': 'Subband Count Accuracy',
        'l3': 'Joint Accuracy',
    }

    # ── 逐Δf计算指标 ──
    curves = {mk: {m: [] for m in METHOD_NAMES} for mk in METRIC_KEYS}

    for df in delta_f_vals:
        mask = np.abs(delta_f_all - df) < 0.05
        sct = sample_count_true[mask]
        subt = subband_count_true[mask]
        ig = ignore_any[mask]

        for m in METHOD_NAMES:
            sc_pred, sub_pred = predictions[m]
            l1, l2, l3 = compute_metrics(
                sc_pred[mask], sub_pred[mask], sct, subt, ig)
            curves['l1'][m].append(l1)
            curves['l2'][m].append(l2)
            curves['l3'][m].append(l3)

    # ── 打印表格 ──
    for mk in METRIC_KEYS:
        print(f"\n===== {METRIC_TITLES[mk]} =====")
        print(f"{'Δf/SR':>8}", end='')
        for m in METHOD_NAMES:
            print(f" {m:>11}", end='')
        print()
        print("-" * (10 + 12 * len(METHOD_NAMES)))
        for i, df in enumerate(delta_f_vals):
            print(f"  {df:>5.1f}  ", end='')
            for m in METHOD_NAMES:
                print(f" {curves[mk][m][i]:>10.1%}", end='')
            print()

    print(f"\n--- Thresholds ---")
    print(f"  DL: BAND_THRESHOLD={BAND_THRESHOLD_DL} (trained optimum)")
    for m in METHOD_NAMES:
        if m == 'DL':
            continue
        if m.startswith('ED-Hard'):
            print(f"  {m}: {thresholds[m]:.4e} W (Pfa=0.01)")
        else:
            print(f"  {m}: {thresholds[m]:.4f} (Pfa=0.01)")

    # ── 画图 ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    for idx, mk in enumerate(METRIC_KEYS):
        ax = axes[idx]
        for m in METHOD_NAMES:
            ax.plot(delta_f_vals, curves[mk][m],
                    f'{METHOD_MARKERS[m]}-',
                    color=METHOD_COLORS[m], label=m,
                    markersize=5, linewidth=1.5)
        ax.set_xlabel('Δf / Symbol Rate', fontsize=11)
        ax.set_ylabel(METRIC_YLABELS[mk], fontsize=11)
        ax.set_title(METRIC_TITLES[mk], fontsize=11)
        ax.legend(fontsize=7, loc='lower right', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlim([min(delta_f_vals) - 0.1, max(delta_f_vals) + 0.1])
        ax.axhline(y=0.9, color='gray', linestyle=':', alpha=0.3)

    plt.suptitle(f'Exp 2-B: Two-Source Frequency Separation Sweep  '
                 f'(2 sources, SNR=+{target_snr}dB, Pfa=0.01 for traditional)',
                 fontsize=13)
    plt.tight_layout()
    save_name = f'exp2B_metrics_{tag}.png'
    plt.savefig(save_name, dpi=150)
    print(f"\nFigure saved: {save_name}")
    plt.close()
    print("Done!")


if __name__ == '__main__':
    main()