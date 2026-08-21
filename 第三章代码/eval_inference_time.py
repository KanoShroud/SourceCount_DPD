"""
eval_inference_time.py — 各方法单样本推理时间对比

测量: 从输入到输出源数估计的平均耗时
  DL:     DPD谱 → CNN → sigmoid → 数非空slot → 源数
  AGM/MME/SLE/EMR: 协方差矩阵 → 特征值分解 → 统计量 → 阈值判决 → 数连续段 → 源数
  ED-Hard: 协方差矩阵对角元 → 阈值判决 → 数连续段 → 源数

用法:
  python eval_inference_time.py B M10
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

from chapter_runtime import (
    checkpoint_path,
    data_dir as runtime_data_dir,
    device as runtime_device,
)


# ═══════════════════════════════════════
#  系统参数
# ═══════════════════════════════════════
LEN   = 4096
B_WIN = 10e6
FS    = 100e6
TW    = LEN * B_WIN / FS

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
#  传统方法: 单样本完整流程
# ═══════════════════════════════════════
def count_segments_single(band_det):
    """单样本连续段计数: (N_sub,) bool → int"""
    segs = 0; in_seg = False
    for k in range(len(band_det)):
        if band_det[k] and not in_seg:
            segs += 1; in_seg = True
        elif not band_det[k]:
            in_seg = False
    return segs


def infer_eig_single(cov_sub, method, alpha):
    """单样本特征值方法: (N_sub, M, M) → 源数"""
    N_sub, M, _ = cov_sub.shape
    band_det = np.zeros(N_sub, dtype=bool)
    for k in range(N_sub):
        ev = np.sort(np.real(np.linalg.eigvalsh(cov_sub[k])))[::-1]
        ev = np.clip(ev, 1e-30, None)
        if method == 'AGM':
            am_gm = ev.mean() / np.exp(np.log(ev).mean())
            T = 2 * (TW - 1) * np.log(am_gm)
        elif method == 'MME':
            T = ev[0] / ev[-1]
        elif method == 'SLE':
            T = ev[0] / ev.mean()
        elif method == 'EMR':
            T = M * (ev**2).sum() / ev.sum()**2
        band_det[k] = (T > alpha)
    return count_segments_single(band_det)


def infer_ed_hard_single(cov_sub, phys_threshold, K=2):
    """单样本ED-Hard: (N_sub, M, M) → 源数"""
    N_sub, M, _ = cov_sub.shape
    band_det = np.zeros(N_sub, dtype=bool)
    for k in range(N_sub):
        per_sta = np.real(np.diag(cov_sub[k]))
        band_det[k] = (per_sta > phys_threshold).sum() >= K
    return count_segments_single(band_det)


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    args = sys.argv[1:]
    tag = 'A'
    for a in args:
        if a == 'B': tag = 'B'
        elif a.startswith('M') and a[1:].isdigit(): tag += f'_{a}'

    device = runtime_device()
    print(f"Device: {device}")

    # ── 加载模型 ──
    model_path = checkpoint_path(f'best_model_v26_{tag}.pth')
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Model: {tag} ({cfg.get('mode','concat')}) max_src={cfg['max_src']}")

    # ── 加载测试数据 ──
    test_path = runtime_data_dir() / 'test_data.mat'
    print(f"Loading: {test_path}")
    with h5py.File(test_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32)
        cov_real    = np.array(f['cov_mat_real_all'], dtype=np.float32)
        cov_imag    = np.array(f['cov_mat_imag_all'], dtype=np.float32)

    spectra = spectra_raw.transpose(3, 2, 1, 0)
    cov_mat = cov_real.transpose(3, 2, 1, 0) + 1j * cov_imag.transpose(3, 2, 1, 0)

    # DL 预处理
    spectra_dl = np.log(spectra + 1.0)
    for i in range(len(spectra_dl)):
        s = spectra_dl[i]; mu = s.mean(); std = s.std() + 1e-6
        spectra_dl[i] = (s - mu) / std

    N = len(spectra_dl)
    print(f"  {N} samples")

    # 阈值（直接用已知值，不重新校准）
    thresholds = {
        'AGM': 9.9241, 'MME': 1.5273, 'SLE': 1.2303, 'EMR': 1.0244,
        'ED-Hard-K1': 1.4834e-09, 'ED-Hard-K2': 1.4425e-09,
        'ED-Hard-K3': 1.3883e-09, 'ED-Hard-K4': 1.3402e-09,
    }

    N_test = min(N, 500)  # 用500样本取平均，足够稳定
    print(f"  Benchmarking on {N_test} samples\n")

    results = {}

    # ── DL (GPU, batch=1) ──
    print("  DL (GPU, batch=1)...")
    # 预热
    with torch.no_grad():
        x = torch.from_numpy(spectra_dl[0:1]).to(device)
        _ = model(x)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N_test):
            x = torch.from_numpy(spectra_dl[i:i+1]).to(device)
            bl = model(x)
            bp = (torch.sigmoid(bl) > BAND_THRESHOLD_DL).long()
            count = (bp.sum(dim=-1) > 0).sum(dim=-1).item()
    torch.cuda.synchronize()
    t_dl = (time.perf_counter() - t0) / N_test
    results['DL (GPU)'] = t_dl
    print(f"    {t_dl*1000:.3f} ms/sample")

    # ── DL (CPU, batch=1) ──
    print("  DL (CPU, batch=1)...")
    model_cpu = model.cpu()
    model_cpu.eval()
    # 预热
    with torch.no_grad():
        x = torch.from_numpy(spectra_dl[0:1])
        _ = model_cpu(x)

    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(N_test):
            x = torch.from_numpy(spectra_dl[i:i+1])
            bl = model_cpu(x)
            bp = (torch.sigmoid(bl) > BAND_THRESHOLD_DL).long()
            count = (bp.sum(dim=-1) > 0).sum(dim=-1).item()
    t_dl_cpu = (time.perf_counter() - t0) / N_test
    results['DL (CPU)'] = t_dl_cpu
    print(f"    {t_dl_cpu*1000:.3f} ms/sample")
    model.to(device)  # 移回GPU

    # ── 特征值方法 ──
    for method in ['AGM', 'MME', 'SLE', 'EMR']:
        print(f"  {method}...")
        alpha = thresholds[method]
        t0 = time.perf_counter()
        for i in range(N_test):
            _ = infer_eig_single(cov_mat[i], method, alpha)
        t_m = (time.perf_counter() - t0) / N_test
        results[method] = t_m
        print(f"    {t_m*1000:.3f} ms/sample")

    # ── ED-Hard ──
    for K in [1, 2, 3, 4]:
        name = f'ED-Hard-K{K}'
        print(f"  {name}...")
        th = thresholds[name]
        t0 = time.perf_counter()
        for i in range(N_test):
            _ = infer_ed_hard_single(cov_mat[i], th, K=K)
        t_m = (time.perf_counter() - t0) / N_test
        results[name] = t_m
        print(f"    {t_m*1000:.3f} ms/sample")

    # ── 汇总表 ──
    print("\n" + "=" * 55)
    print("  Inference Time per Sample")
    print("=" * 55)
    print(f"  {'Method':<20} {'Time (ms)':<15} {'Relative':<10}")
    print("-" * 55)

    t_base = results['DL (GPU)']
    for name, t in results.items():
        rel = t / t_base
        print(f"  {name:<20} {t*1000:<15.3f} {rel:<10.1f}x")

    print("\nDone!")


if __name__ == '__main__':
    main()
