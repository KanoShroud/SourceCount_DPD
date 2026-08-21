"""
train_v26.py — 多源信源检测 + 实例级频段估计

网络输出：
  band_logits: (B, max_src, N_sub) → 每个槽位的子带占用
  信源计数：从非空slot数推断（无独立count head）

Loss：
  Focal BCE（子带占用，带ignore mask）

用法：
  python train_v26.py          → 训练方案A（拼接）
  python train_v26.py B        → 训练方案B（Transformer）
  python train_v26.py B M5     → 训练方案B，max_src=5
  python train_v26.py B M10    → 训练方案B，max_src=10

  python train_v26.py test     → 测试方案A
  python train_v26.py test B   → 测试方案B
  python train_v26.py test B M5 → 测试方案B，max_src=5
  python train_v26.py test B M10 → 测试方案B，max_src=10
"""

import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

from chapter_runtime import (
    checkpoint_path,
    data_dir as runtime_data_dir,
    device as runtime_device,
    output_path,
    summary as runtime_summary,
)

BAND_THRESHOLD = 0.5


# ═══════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════
class SourceDetectionDataset(Dataset):

    def __init__(self, mat_path, augment=False, normalize='sample_zscore',
                 max_src_override=None):
        self.augment = augment

        with h5py.File(mat_path, 'r') as f:
            self.spectra = np.array(f['mtr_sub_all'], dtype=np.float32)
            self.src_count = np.array(f['src_count_all'], dtype=np.int64)
            self.band_mask = np.array(f['band_mask_all'], dtype=np.float32)
            self.ignore_mask = np.array(f['ignore_mask_all'], dtype=np.float32)
            self.N_sub = int(np.array(f['N_sub_val']).item())
            self.max_src = int(np.array(f['max_src_val']).item())
            self.num_count_classes = int(np.array(f['num_count_classes']).item())
            self.avg_snr = np.array(f['avg_snr_all'], dtype=np.float32)

        # 转置
        self.spectra = self.spectra.transpose(3, 2, 1, 0)
        self.src_count = self.src_count.flatten()
        self.band_mask = self.band_mask.transpose(2, 1, 0)
        self.ignore_mask = self.ignore_mask.transpose(2, 1, 0)
        self.avg_snr = self.avg_snr.flatten()

        # 扩展max_src（补零slot）
        if max_src_override is not None and max_src_override > self.max_src:
            N = len(self.src_count)
            pad_n = max_src_override - self.max_src
            self.band_mask = np.concatenate([
                self.band_mask,
                np.zeros((N, pad_n, self.N_sub), dtype=np.float32)], axis=1)
            self.ignore_mask = np.concatenate([
                self.ignore_mask,
                np.zeros((N, pad_n, self.N_sub), dtype=np.float32)], axis=1)
            self.max_src = max_src_override

        # log压缩
        self.spectra = np.log(self.spectra + 1.0)

        # 归一化
        if normalize == 'sample_zscore':
            for i in range(len(self.spectra)):
                s = self.spectra[i]
                mu = s.mean();
                std = s.std() + 1e-6
                self.spectra[i] = (s - mu) / std

        print(f"加载 {mat_path}: {len(self)} 样本, "
              f"N_sub={self.N_sub}, max_src={self.max_src}, "
              f"num_count_classes={self.num_count_classes}")
        for c in range(self.num_count_classes):
            cnt = (self.src_count == c).sum()
            if cnt > 0:
                print(f"  {c}源: {cnt} ({100 * cnt / len(self):.1f}%)")

    def __len__(self):
        return self.spectra.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.spectra[idx].copy())
        sc = torch.tensor(self.src_count[idx], dtype=torch.long)
        bm = torch.from_numpy(self.band_mask[idx].copy())
        ig = torch.from_numpy(self.ignore_mask[idx].copy())

        if self.augment:
            if torch.rand(1) < 0.5:
                x = x.flip(-1)
            if torch.rand(1) < 0.5:
                x = x.flip(-2)
            if torch.rand(1) < 0.3:
                x = x + 0.05 * torch.randn_like(x)

        return x, sc, bm, ig


# ═══════════════════════════════════════
#  模型（无count head）
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
    """
    CNN backbone（逐子带共享权重）→ 全局特征 → band heads
    信源计数通过非空slot数推断，无独立count head

    mode='concat':      子带特征拼接 → Linear（方案A）
    mode='transformer': 子带特征 → Transformer交互 → 池化（方案B）

    输入:  (B, N_sub, H, W)
    输出:  band_logits (B, max_src, N_sub)
    """

    def __init__(self, n_sub=10, max_src=3, feat_dim=128, mode='concat'):
        super().__init__()
        self.n_sub = n_sub
        self.max_src = max_src
        self.feat_dim = feat_dim
        self.mode = mode

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            ResBlock(32, dropout=0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            ResBlock(64, dropout=0.1),
            nn.Conv2d(64, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(),
            ResBlock(feat_dim, dropout=0.1),
            nn.AdaptiveAvgPool2d(1),
        )

        if mode == 'transformer':
            self.pos_embed = nn.Parameter(torch.randn(1, n_sub, feat_dim) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=feat_dim, nhead=4,
                dim_feedforward=256, dropout=0.1,
                batch_first=True)
            self.cross_attn = nn.TransformerEncoder(encoder_layer, num_layers=1)
            self.global_encoder = nn.Sequential(
                nn.Linear(feat_dim, 256), nn.ReLU(), nn.Dropout(0.3))
        else:
            self.global_encoder = nn.Sequential(
                nn.Linear(feat_dim * n_sub, 256), nn.ReLU(), nn.Dropout(0.3))

        self.band_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, n_sub),
            ) for _ in range(max_src)
        ])

    def forward(self, x):
        B, S, H, W = x.shape
        feat = self.backbone(x.reshape(B * S, 1, H, W))
        feat = feat.squeeze(-1).squeeze(-1).reshape(B, S, self.feat_dim)

        if self.mode == 'transformer':
            feat = self.cross_attn(feat + self.pos_embed)
            global_feat = self.global_encoder(feat.mean(dim=1))
        else:
            global_feat = self.global_encoder(feat.reshape(B, -1))

        band_logits = torch.stack(
            [head(global_feat) for head in self.band_heads], dim=1)
        return band_logits


# ═══════════════════════════════════════
#  Focal BCE Loss
# ═══════════════════════════════════════
def focal_bce(logits, targets, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p_t = p * targets + (1 - p) * (1 - targets)
    focal_weight = (1 - p_t) ** gamma
    return focal_weight * ce


# ═══════════════════════════════════════
#  Loss 计算
# ═══════════════════════════════════════
def compute_loss(band_logits, band_mask, ignore_mask, gamma=2.0):
    loss_band_raw = focal_bce(band_logits, band_mask, gamma=gamma)
    valid = 1.0 - ignore_mask
    n_valid = valid.sum().clamp(min=1)
    loss_band = (loss_band_raw * valid).sum() / n_valid
    return loss_band


# ═══════════════════════════════════════
#  评估
# ═══════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device, gamma=2.0):
    model.eval()
    total_loss = 0
    total_samples = 0

    count_correct = 0
    count_total = 0

    band_correct = 0
    band_total = 0

    # 逐类count统计（从band推断）
    max_src = model.max_src
    num_classes = min(max_src + 1, 4)  # 实际源数最多3，count类别0/1/2/3
    count_class_correct = np.zeros(num_classes)
    count_class_total = np.zeros(num_classes)

    for x, sc, bm, ig in loader:
        x = x.to(device)
        sc = sc.to(device)
        bm = bm.to(device)
        ig = ig.to(device)
        B = x.size(0)

        band_logits = model(x)
        loss = compute_loss(band_logits, bm, ig, gamma)

        total_loss += loss.item() * B
        total_samples += B

        # count（从band推断：数非空slot）
        band_pred = (torch.sigmoid(band_logits) > BAND_THRESHOLD).float()
        slot_nonempty = (band_pred.sum(dim=-1) > 0)  # (B, max_src)
        count_pred = slot_nonempty.sum(dim=-1).long()  # (B,)
        count_correct += (count_pred == sc).sum().item()
        count_total += B

        for c in range(num_classes):
            mask_c = (sc == c)
            count_class_total[c] += mask_c.sum().item()
            count_class_correct[c] += ((count_pred == c) & mask_c).sum().item()

        # band准确率（只看有源槽位的非ignore子带）
        valid = (1.0 - ig)
        for s in range(max_src):
            has_src = (sc > s)
            if has_src.sum() == 0:
                continue
            v = valid[has_src, s, :]
            p = band_pred[has_src, s, :]
            t = bm[has_src, s, :]
            band_correct += ((p == t) * v).sum().item()
            band_total += v.sum().item()

    avg_loss = total_loss / total_samples
    count_acc = count_correct / max(count_total, 1)
    band_acc = band_correct / max(band_total, 1)
    count_class_acc = {c: count_class_correct[c] / count_class_total[c]
    if count_class_total[c] > 0 else -1
                       for c in range(num_classes)}

    return {
        'loss': avg_loss,
        'count_acc': count_acc,
        'band_acc': band_acc,
        'count_class_acc': count_class_acc,
    }


# ═══════════════════════════════════════
#  详细评估（混淆矩阵 + 分析图）
# ═══════════════════════════════════════
@torch.no_grad()
def full_evaluation(model, loader, device, save_prefix='test'):
    model.eval()
    max_src = model.max_src
    n_sub = model.n_sub

    all_count_pred = []
    all_count_true = []
    all_band_pred = []
    all_band_true = []
    all_band_valid = []
    all_src_count = []

    for x, sc, bm, ig in loader:
        x = x.to(device)
        band_logits = model(x)
        band_pred = (torch.sigmoid(band_logits) > BAND_THRESHOLD).float()

        # count从band推断
        slot_nonempty = (band_pred.sum(dim=-1) > 0)
        count_pred = slot_nonempty.sum(dim=-1).long()

        all_count_pred.append(count_pred.cpu().numpy())
        all_count_true.append(sc.numpy())
        all_band_pred.append(band_pred.cpu().numpy())
        all_band_true.append(bm.numpy())
        all_band_valid.append(1.0 - ig.numpy())
        all_src_count.append(sc.numpy())

    all_count_pred = np.concatenate(all_count_pred)
    all_count_true = np.concatenate(all_count_true)
    all_band_pred = np.concatenate(all_band_pred)
    all_band_true = np.concatenate(all_band_true)
    all_band_valid = np.concatenate(all_band_valid)
    all_src_count = np.concatenate(all_src_count)

    # ── count混淆矩阵（从band推断）──
    num_classes = min(max_src + 1, 4)
    labels = list(range(num_classes))
    # clip predictions to valid range
    all_count_pred_clip = np.clip(all_count_pred, 0, num_classes - 1)
    cm = confusion_matrix(all_count_true, all_count_pred_clip, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, ttl, fmt in [
        (axes[0], cm, 'Count from Band (raw)', 'd'),
        (axes[1], cm_norm, 'Count from Band (normalized)', '.1%'),
    ]:
        im = ax.imshow(data, cmap='Blues',
                       vmin=0 if 'norm' in ttl.lower() else None,
                       vmax=1 if 'norm' in ttl.lower() else None)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(data[i, j], fmt), ha='center', va='center')
        ax.set_xlabel('Predicted');
        ax.set_ylabel('True')
        ax.set_title(ttl)
        ax.set_xticks(labels);
        ax.set_yticks(labels)
        ax.set_xticklabels([f'{c}src' for c in labels])
        ax.set_yticklabels([f'{c}src' for c in labels])
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_count_cm.png', dpi=150)
    plt.close()

    print("\n===== Count (from band) Classification Report =====")
    print(classification_report(all_count_true, all_count_pred_clip,
                                target_names=[f'{c}src' for c in labels], digits=4))

    # ── band准确率按信源数分组 ──
    print("===== Band Mask Accuracy (by src count) =====")
    for c in range(num_classes):
        mask_c = (all_src_count == c)
        if mask_c.sum() == 0:
            continue
        for s in range(max_src):
            if c <= s:
                continue
            v = all_band_valid[mask_c, s, :]
            p = all_band_pred[mask_c, s, :]
            t = all_band_true[mask_c, s, :]
            if v.sum() == 0:
                continue
            acc = ((p == t) * v).sum() / v.sum()
            pos_mask = (t == 1) & (v == 1)
            neg_mask = (t == 0) & (v == 1)
            recall = ((p == 1) & pos_mask).sum() / pos_mask.sum() if pos_mask.sum() > 0 else -1
            spec = ((p == 0) & neg_mask).sum() / neg_mask.sum() if neg_mask.sum() > 0 else -1
            print(f"  {c}src slot{s + 1}: acc={acc:.1%}  recall={recall:.1%}  specificity={spec:.1%}")

    # ── 综合指标可视化 ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # count准确率柱状图
    ax = axes[0]
    count_acc_per_class = []
    for c in labels:
        mask_c = (all_count_true == c)
        if mask_c.sum() > 0:
            count_acc_per_class.append((all_count_pred_clip[mask_c] == c).mean())
        else:
            count_acc_per_class.append(0)
    colors_bar = ['gray', 'green', 'orange', 'red'][:num_classes]
    ax.bar(labels, count_acc_per_class, color=colors_bar)
    ax.set_xlabel('Source Count');
    ax.set_ylabel('Accuracy')
    ax.set_title('Count Accuracy (from band, by class)')
    ax.set_xticks(labels);
    ax.set_ylim([0, 1.05]);
    ax.grid(True, axis='y')

    # band mask示例
    ax = axes[1]
    multi_idx = np.where(all_src_count >= 2)[0]
    if len(multi_idx) > 0:
        idx = multi_idx[0]
        n = all_src_count[idx]
        for s in range(n):
            y_true = all_band_true[idx, s, :]
            y_pred = all_band_pred[idx, s, :]
            offset = s * 0.15
            ax.bar(np.arange(n_sub) - 0.2 + offset, y_true, 0.15,
                   alpha=0.6, label=f'S{s + 1} true')
            ax.bar(np.arange(n_sub) + 0.0 + offset, y_pred, 0.15,
                   alpha=0.6, label=f'S{s + 1} pred')
        ax.set_xlabel('Subband');
        ax.set_ylabel('Occupation')
        ax.set_title(f'Band Mask Example (sample {idx}, {n}src)')
        ax.set_xticks(range(n_sub))
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, 'No multi-source samples', ha='center', va='center')

    # count预测分布
    ax = axes[2]
    for c in labels:
        cnt_true = (all_count_true == c).sum()
        cnt_pred = (all_count_pred_clip == c).sum()
        ax.bar(c - 0.15, cnt_true, 0.3, color='steelblue', alpha=0.7,
               label='True' if c == 0 else '')
        ax.bar(c + 0.15, cnt_pred, 0.3, color='coral', alpha=0.7,
               label='Pred' if c == 0 else '')
    ax.set_xlabel('Source Count');
    ax.set_ylabel('Samples')
    ax.set_title('Count Distribution (True vs Pred)')
    ax.set_xticks(labels);
    ax.legend();
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(f'{save_prefix}_analysis.png', dpi=150)
    plt.close()


# ═══════════════════════════════════════
#  训练
# ═══════════════════════════════════════
def train(mode='concat', max_src_override=None):
    data_dir = runtime_data_dir()
    max_epochs = 100
    batch_size = 64
    base_lr = 1e-3
    weight_decay = 5e-4
    warmup_epochs = 5
    patience = 25
    gamma = 2.0
    device = runtime_device()

    tag = 'B' if mode == 'transformer' else 'A'
    if max_src_override:
        tag += f'_M{max_src_override}'
    print(f"Device: {device}")
    print(runtime_summary('train_v26'))
    print(f"Scheme {tag}: mode={mode}  gamma={gamma}  max_src_override={max_src_override}")

    # ── 数据 ──
    train_set = SourceDetectionDataset(
        data_dir / 'train_data.mat',
        augment=True, normalize='sample_zscore',
        max_src_override=max_src_override)
    val_set = SourceDetectionDataset(
        data_dir / 'val_data.mat',
        augment=False, normalize='sample_zscore',
        max_src_override=max_src_override)

    N_sub = train_set.N_sub
    max_src = train_set.max_src

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # ── 模型 ──
    model = SourceDetectionNet(n_sub=N_sub, max_src=max_src, mode=mode).to(device)
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {param_count:.2f}M")

    # ── 优化器 + 调度器 ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr,
                                  weight_decay=weight_decay)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs])

    # ── 训练循环 ──
    best_val_loss = float('inf')
    no_improve = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'val_count_acc': [], 'val_band_acc': [],
        'lr': [],
    }

    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        n_batch = 0

        for x, sc, bm, ig in train_loader:
            x = x.to(device)
            bm = bm.to(device)
            ig = ig.to(device)

            band_logits = model(x)
            loss = compute_loss(band_logits, bm, ig, gamma)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batch += 1

        scheduler.step()
        avg_loss = total_loss / n_batch

        # ── 验证 ──
        val_metrics = evaluate(model, val_loader, device, gamma=gamma)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_count_acc'].append(val_metrics['count_acc'])
        history['val_band_acc'].append(val_metrics['band_acc'])
        history['lr'].append(current_lr)

        # 逐类打印
        num_classes = len(val_metrics['count_class_acc'])
        class_str = '  '.join([
            f"{c}src:{val_metrics['count_class_acc'][c]:.1%}"
            if val_metrics['count_class_acc'][c] >= 0 else f"{c}src:N/A"
            for c in range(num_classes)])

        print(f"[Epoch {epoch + 1:3d}/{max_epochs}] "
              f"train={avg_loss:.4f} | "
              f"val={val_metrics['loss']:.4f} "
              f"count={val_metrics['count_acc']:.1%} "
              f"band={val_metrics['band_acc']:.1%} | "
              f"lr={current_lr:.1e} | {class_str}")

        if epoch < warmup_epochs:
            continue

        # ── 保存最优 ──
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_count_acc': val_metrics['count_acc'],
                'val_band_acc': val_metrics['band_acc'],
                'cfg': {
                    'N_sub': N_sub, 'max_src': max_src,
                    'gamma': gamma, 'mode': mode,
                },
            }, checkpoint_path(f'best_model_v26_{tag}.pth'))
            print(f"  ★ Saved best model val_loss={val_metrics['loss']:.4f} "
                  f"count={val_metrics['count_acc']:.1%} "
                  f"band={val_metrics['band_acc']:.1%}")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stop: {patience} epochs without improvement")
            break

    # ── 训练曲线 ──
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch');
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Band Loss');
    axes[0].legend();
    axes[0].grid(True)

    axes[1].plot(epochs, history['val_count_acc'], color='green')
    axes[1].set_xlabel('Epoch');
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Val Count (from band)');
    axes[1].grid(True)

    axes[2].plot(epochs, history['val_band_acc'], color='blue')
    axes[2].set_xlabel('Epoch');
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Val Band Accuracy');
    axes[2].grid(True)

    axes[3].plot(epochs, history['lr'], color='red')
    axes[3].set_xlabel('Epoch');
    axes[3].set_ylabel('LR')
    axes[3].set_title('Learning Rate');
    axes[3].set_yscale('log');
    axes[3].grid(True)

    plt.tight_layout()
    plt.savefig(output_path('train_v26', f'training_curves_v26_{tag}.png'), dpi=150)
    plt.close()

    # ── 验证集评估 ──
    ckpt = torch.load(checkpoint_path(f'best_model_v26_{tag}.pth'), weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"\n===== Scheme {tag} Validation Evaluation =====")
    full_evaluation(
        model,
        val_loader,
        device,
        save_prefix=str(output_path('train_v26', f'val_{tag}')),
    )
    print(f"\nDone! Scheme {tag}({mode}) best val_loss={best_val_loss:.4f}")


# ═══════════════════════════════════════
#  测试
# ═══════════════════════════════════════
def test(mode='concat', max_src_override=None):
    device = runtime_device()
    tag = 'B' if mode == 'transformer' else 'A'
    if max_src_override:
        tag += f'_M{max_src_override}'

    test_set = SourceDetectionDataset(
        runtime_data_dir() / 'test_data.mat',
        normalize='sample_zscore',
        max_src_override=max_src_override)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False,
                             num_workers=4, pin_memory=True)

    ckpt = torch.load(
        checkpoint_path(f'best_model_v26_{tag}.pth'),
        map_location=device,
        weights_only=False,
    )
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')
    ).to(device)
    model.load_state_dict(ckpt['model'])

    metrics = evaluate(model, test_loader, device, gamma=cfg['gamma'])
    print(f"\n===== Test Results (Scheme {tag}: {cfg['mode']}, "
          f"max_src={cfg['max_src']}) =====")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Count (from band): {metrics['count_acc']:.1%}")
    print(f"Band:              {metrics['band_acc']:.1%}")
    for c, a in metrics['count_class_acc'].items():
        print(f"  {c}src: {a:.1%}" if a >= 0 else f"  {c}src: N/A")

    full_evaluation(
        model,
        test_loader,
        device,
        save_prefix=str(output_path('train_v26', f'test_{tag}')),
    )


if __name__ == '__main__':
    args = sys.argv[1:]
    m = 'transformer' if 'B' in args else 'concat'

    max_src_ov = None
    for a in args:
        if a.startswith('M') and a[1:].isdigit():
            max_src_ov = int(a[1:])

    if 'test' in args:
        test(mode=m, max_src_override=max_src_ov)
    else:
        train(mode=m, max_src_override=max_src_ov)
