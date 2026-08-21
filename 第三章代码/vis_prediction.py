"""
vis_prediction.py — 模型预测可视化

对选定样本画热力图：
  左: 真值 band_mask (max_src × 19)
  中: 模型输出概率 (max_src × 19)
  右: 二值判决 (threshold=0.50)

可选择按源数挑选典型样本，或指定样本索引

用法:
  python vis_prediction.py B M10              # 自动选4个典型样本(0/1/2/3源各一个)
  python vis_prediction.py B M10 --idx 42     # 指定样本索引
  python vis_prediction.py B M10 --dataset val  # 用验证集
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

from chapter_runtime import (
    checkpoint_path,
    data_dir as runtime_data_dir,
    device as runtime_device,
    output_path,
)


BAND_THRESHOLD = 0.50


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
#  数据加载
# ═══════════════════════════════════════
def load_data(mat_path):
    with h5py.File(mat_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32)
        src_count   = np.array(f['src_count_all'], dtype=np.int64).flatten()
        band_mask   = np.array(f['band_mask_all'], dtype=np.float32)
        ignore_mask = np.array(f['ignore_mask_all'], dtype=np.float32)

    spectra     = spectra_raw.transpose(3, 2, 1, 0)
    band_mask   = band_mask.transpose(2, 1, 0)       # (N, max_src_data, N_sub)
    ignore_mask = ignore_mask.transpose(2, 1, 0)

    # log + z-score
    spectra_dl = np.log(spectra + 1.0)
    for i in range(len(spectra_dl)):
        s = spectra_dl[i]; mu = s.mean(); std = s.std() + 1e-6
        spectra_dl[i] = (s - mu) / std

    return spectra_dl, src_count, band_mask, ignore_mask


# ═══════════════════════════════════════
#  单样本可视化
# ═══════════════════════════════════════
def visualize_sample(model, spectra_dl, src_count, band_mask, ignore_mask,
                     idx, device, max_src_model, save_prefix=''):
    """画一个样本的 真值 | 概率 | 判决 三栏热力图"""

    n_src = src_count[idx]
    N_sub = model.n_sub

    # 推理
    with torch.no_grad():
        x = torch.from_numpy(spectra_dl[idx:idx+1]).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]   # (max_src_model, N_sub)
    preds = (probs > BAND_THRESHOLD).astype(np.float32)

    # 真值 (对齐到 max_src_model 维度)
    max_src_data = band_mask.shape[1]
    truth = np.zeros((max_src_model, N_sub), dtype=np.float32)
    ignore = np.zeros((max_src_model, N_sub), dtype=np.float32)
    ms = min(max_src_model, max_src_data)
    truth[:ms] = (band_mask[idx, :ms] > 0.5).astype(np.float32)
    ignore[:ms] = (ignore_mask[idx, :ms] > 0.5).astype(np.float32)

    # 推断源数
    count_pred = (preds.sum(axis=1) > 0).sum()

    # 只显示有意义的 slot 行数 (最多到 max(n_src, count_pred) + 1，至少3行)
    n_show = max(int(n_src), int(count_pred), 2) + 1
    n_show = min(n_show, max_src_model)

    fig, axes = plt.subplots(1, 3, figsize=(16, 0.6 * n_show + 2.5))

    sub_labels = [f'W{k+1}' for k in range(N_sub)]
    slot_labels = [f'Slot {s+1}' for s in range(n_show)]

    # ── 左: 真值 ──
    ax = axes[0]
    # 构造显示矩阵: 1=有信号(深色), 0.5=ignore(中间色), 0=无信号(浅色)
    disp_truth = np.zeros((n_show, N_sub))
    for s in range(n_show):
        for k in range(N_sub):
            if s < ms and truth[s, k] > 0.5:
                disp_truth[s, k] = 1.0
            elif s < ms and ignore[s, k] > 0.5:
                disp_truth[s, k] = 0.5
    im0 = ax.imshow(disp_truth, cmap='Blues', aspect='auto', vmin=0, vmax=1,
                     interpolation='nearest')
    # 标注数值
    for s in range(n_show):
        for k in range(N_sub):
            if disp_truth[s, k] == 1.0:
                ax.text(k, s, '1', ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
            elif disp_truth[s, k] == 0.5:
                ax.text(k, s, 'IG', ha='center', va='center', fontsize=7,
                        color='white')
    ax.set_xticks(range(N_sub))
    ax.set_xticklabels(sub_labels, fontsize=7, rotation=45)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(slot_labels, fontsize=9)
    ax.set_title(f'真值 (band_mask)\n{n_src}源', fontsize=11)
    ax.set_xlabel('子带', fontsize=10)

    # ── 中: 概率 ──
    ax = axes[1]
    im1 = ax.imshow(probs[:n_show], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1,
                     interpolation='nearest')
    for s in range(n_show):
        for k in range(N_sub):
            v = probs[s, k]
            color = 'white' if v > 0.5 else 'black'
            ax.text(k, s, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                    color=color)
    ax.set_xticks(range(N_sub))
    ax.set_xticklabels(sub_labels, fontsize=7, rotation=45)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(slot_labels, fontsize=9)
    ax.set_title('模型输出概率\n(Sigmoid)', fontsize=11)
    ax.set_xlabel('子带', fontsize=10)
    plt.colorbar(im1, ax=ax, shrink=0.8, label='概率')

    # ── 右: 二值判决 ──
    ax = axes[2]
    im2 = ax.imshow(preds[:n_show], cmap='Greens', aspect='auto', vmin=0, vmax=1,
                     interpolation='nearest')
    for s in range(n_show):
        for k in range(N_sub):
            if preds[s, k] > 0.5:
                ax.text(k, s, '1', ha='center', va='center', fontsize=8,
                        color='white', fontweight='bold')
    ax.set_xticks(range(N_sub))
    ax.set_xticklabels(sub_labels, fontsize=7, rotation=45)
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(slot_labels, fontsize=9)
    ax.set_title(f'判决 (阈值={BAND_THRESHOLD})\n预测{int(count_pred)}源', fontsize=11)
    ax.set_xlabel('子带', fontsize=10)

    # 判断是否正确
    correct = (int(count_pred) == int(n_src))
    status = '✓ 正确' if correct else '✗ 错误'
    color = 'green' if correct else 'red'

    plt.suptitle(f'样本 #{idx}  真值: {n_src}源  预测: {int(count_pred)}源  {status}',
                 fontsize=13, color=color, fontweight='bold')
    plt.tight_layout()

    save_name = f'{save_prefix}vis_pred_{idx}.png'
    plt.savefig(save_name, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_name}")
    plt.close()
    return correct


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('scheme', nargs='*', default=['B', 'M10'])
    parser.add_argument('--idx', type=int, nargs='*', default=None,
                        help='指定样本索引')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['test', 'val'], help='数据集')
    parser.add_argument('--n_per_src', type=int, default=1,
                        help='每个源数选几个样本（自动模式）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（自动选样本用）')
    args = parser.parse_args()

    # 解析 scheme
    tag = 'A'
    for a in args.scheme:
        if a == 'B': tag = 'B'
        elif a.startswith('M') and a[1:].isdigit(): tag += f'_{a}'

    device = runtime_device()
    print(f"Device: {device}")

    # 加载模型
    model_path = checkpoint_path(f'best_model_v26_{tag}.pth')
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    max_src_model = cfg['max_src']
    print(f"Model: {tag} ({cfg['mode']}) max_src={max_src_model}")

    # 加载数据
    data_path = runtime_data_dir() / f'{args.dataset}_data.mat'
    print(f"Loading: {data_path}")
    spectra_dl, src_count, band_mask, ignore_mask = load_data(data_path)
    N = len(src_count)
    print(f"  {N} samples")

    # 选样本
    if args.idx is not None:
        indices = args.idx
        print(f"  指定样本: {indices}")
    else:
        # 自动选：每个源数选 n_per_src 个
        rng = np.random.RandomState(args.seed)
        indices = []
        for ns in sorted(np.unique(src_count)):
            pool = np.where(src_count == ns)[0]
            chosen = rng.choice(pool, size=min(args.n_per_src, len(pool)), replace=False)
            indices.extend(chosen.tolist())
        print(f"  自动选样本 (每源数{args.n_per_src}个): {indices}")
        print(f"  源数分布: {[int(src_count[i]) for i in indices]}")

    # 逐样本可视化
    print(f"\n--- 可视化 ({len(indices)} 个样本) ---")
    n_correct = 0
    for idx in indices:
        if idx >= N:
            print(f"  跳过 #{idx}: 超出范围")
            continue
        ns = src_count[idx]
        print(f"  样本 #{idx}: {ns}源", end='')
        correct = visualize_sample(
            model, spectra_dl, src_count, band_mask, ignore_mask,
            idx,
            device,
            max_src_model,
            save_prefix=str(output_path('vis_prediction', f'{args.dataset}_')),
        )
        if correct:
            n_correct += 1
        print(f"  {'✓' if correct else '✗'}")

    print(f"\n{n_correct}/{len(indices)} 正确")

    # 汇总图：所有样本拼在一起
    if len(indices) > 1 and len(indices) <= 8:
        print("\n--- 生成汇总图 ---")
        fig, all_axes = plt.subplots(len(indices), 3,
                                      figsize=(16, 2.2 * len(indices) + 1))
        if len(indices) == 1:
            all_axes = all_axes.reshape(1, -1)

        for row, idx in enumerate(indices):
            if idx >= N:
                continue
            ns = src_count[idx]
            N_sub = model.n_sub
            max_src_data = band_mask.shape[1]
            ms = min(max_src_model, max_src_data)

            with torch.no_grad():
                x = torch.from_numpy(spectra_dl[idx:idx+1]).to(device)
                logits = model(x)
                probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs > BAND_THRESHOLD).astype(np.float32)

            truth = np.zeros((max_src_model, N_sub), dtype=np.float32)
            truth[:ms] = (band_mask[idx, :ms] > 0.5).astype(np.float32)

            count_pred = (preds.sum(axis=1) > 0).sum()
            n_show = max(int(ns), int(count_pred), 2) + 1
            n_show = min(n_show, max_src_model)

            correct = (int(count_pred) == int(ns))

            # 真值
            ax = all_axes[row, 0]
            ax.imshow(truth[:n_show], cmap='Blues', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')
            ax.set_yticks(range(n_show))
            ax.set_yticklabels([f'S{s+1}' for s in range(n_show)], fontsize=7)
            if row == len(indices) - 1:
                ax.set_xticks(range(N_sub))
                ax.set_xticklabels([f'{k+1}' for k in range(N_sub)], fontsize=6)
            else:
                ax.set_xticks([])
            ax.set_ylabel(f'#{idx}\n{ns}源', fontsize=9,
                         color='green' if correct else 'red')

            # 概率
            ax = all_axes[row, 1]
            ax.imshow(probs[:n_show], cmap='YlOrRd', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')
            ax.set_yticks([])
            if row == len(indices) - 1:
                ax.set_xticks(range(N_sub))
                ax.set_xticklabels([f'{k+1}' for k in range(N_sub)], fontsize=6)
            else:
                ax.set_xticks([])

            # 判决
            ax = all_axes[row, 2]
            ax.imshow(preds[:n_show], cmap='Greens', aspect='auto',
                      vmin=0, vmax=1, interpolation='nearest')
            ax.set_yticks([])
            if row == len(indices) - 1:
                ax.set_xticks(range(N_sub))
                ax.set_xticklabels([f'{k+1}' for k in range(N_sub)], fontsize=6)
            else:
                ax.set_xticks([])
            status = '✓' if correct else '✗'
            ax.yaxis.set_label_position('right')
            ax.set_ylabel(f'→{int(count_pred)}源 {status}', fontsize=9,
                         rotation=0, labelpad=35,
                         color='green' if correct else 'red')

        all_axes[0, 0].set_title('真值 (band_mask)', fontsize=11)
        all_axes[0, 1].set_title('模型输出概率', fontsize=11)
        all_axes[0, 2].set_title(f'判决 (阈值={BAND_THRESHOLD})', fontsize=11)

        plt.suptitle(f'预测可视化汇总 ({tag}, {args.dataset}集)', fontsize=13)
        plt.tight_layout()
        save_name = f'{args.dataset}_vis_summary_{tag}.png'
        plt.savefig(output_path('vis_prediction', save_name), dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_name}")
        plt.close()

    print("\nDone!")


if __name__ == '__main__':
    main()
