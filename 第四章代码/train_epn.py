#!/usr/bin/env python3
"""
train_epn.py — EPN (End-to-end Positioning Network) 适配版
参考: 李嘉霖, "基于阵列接收和深度学习的直接定位方法研究"

原始 EPN: 协方差矩阵 → 1D Conv → ResNet → FC → 坐标回归
适配版:   DPD 空间谱 → 2D Conv → ResNet-18 → FC → 坐标回归

用法:
  训练:   python train_epn.py --device cuda:0 --data_dir /mnt/data/ltzdata_loc
  评估:   python train_epn.py --eval_exp 4A2 4A3 --device cuda:0 --data_dir /mnt/data/ltzdata_loc
"""

import os, sys, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
import torchvision.models as models

# ─── 常量 ───
COORD_MAX = 2000.0   # 坐标范围 ±2000m
MAX_SRC = 3           # 最大源数


# ═══════════════════════════════════════
#  数据集
# ═══════════════════════════════════════
class EPNDataset(Dataset):
    """复用现有 .pt 训练数据，只加载 dpd 和 pos_label"""

    def __init__(self, data_dir, split, max_src=MAX_SRC):
        self.max_src = max_src
        split_dir = os.path.join(data_dir, split)
        idx_path = os.path.join(split_dir, f'loc_{split}_index.pt')
        idx = torch.load(idx_path, weights_only=False)

        print(f"  Loading {len(idx['shard_files'])} shards for {split}...")
        all_dpd, all_pos, all_n = [], [], []
        for sf in idx['shard_files']:
            d = torch.load(os.path.join(split_dir, sf), weights_only=False)
            all_dpd.append(d['fine_dpd'])      # (N, 1, 401, 401) float16
            all_pos.append(d['pos_label'])      # (N, max_src, 2) normalized [-1,1]
            all_n.append(d['n_src'])            # (N,) int
            del d

        self.dpd = torch.cat(all_dpd)
        self.pos = torch.cat(all_pos).float()
        self.n_src = torch.cat(all_n)

        mem_gb = self.dpd.element_size() * self.dpd.nelement() / 1e9
        print(f"  内存占用: {mem_gb:.1f} GB")
        print(f"  Loaded {len(self.dpd)} samples")
        for ns in range(2, max_src + 1):
            cnt = (self.n_src == ns).sum().item()
            if cnt > 0:
                print(f"    N={ns}: {cnt}")

    def __len__(self):
        return len(self.dpd)

    def __getitem__(self, idx):
        dpd = self.dpd[idx].float()          # (1, 401, 401)
        pos = self.pos[idx].clone()           # (max_src, 2) normalized [-1,1]
        n = self.n_src[idx].item()

        # 归一化 DPD（per-sample zero-mean unit-variance）
        mu = dpd.mean()
        std = dpd.std() + 1e-6
        dpd = (dpd - mu) / std

        # ROLA: 按距原点距离升序排列有效源
        if n > 0:
            dists = torch.sqrt(pos[:n, 0]**2 + pos[:n, 1]**2)
            sort_idx = torch.argsort(dists)
            sorted_pos = torch.zeros_like(pos)
            sorted_pos[:n] = pos[sort_idx]
        else:
            sorted_pos = torch.zeros_like(pos)

        # 展平为 (max_src*2,) 向量
        label = sorted_pos.reshape(-1)        # (6,) for max_src=3
        return dpd, label, n


# ═══════════════════════════════════════
#  模型: ResNet-18 + FC 坐标回归
# ═══════════════════════════════════════
class EPNResNet(nn.Module):
    """
    ResNet-18 backbone (单通道输入) + FC 回归头
    输出: (batch, max_src*2) 归一化坐标
    """
    def __init__(self, max_src=MAX_SRC, dropout=0.2):
        super().__init__()
        resnet = models.resnet18(weights=None)
        # 修改第一层: 3通道 → 1通道
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 去掉原始 FC 层，保留到 avgpool
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,   # AdaptiveAvgPool2d(1)
        )
        # 回归头
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, max_src * 2),
            nn.Tanh(),        # 约束输出在 [-1, 1]
        )

    def forward(self, x):
        feat = self.features(x)     # (B, 512, 1, 1)
        feat = feat.flatten(1)      # (B, 512)
        return self.head(feat)       # (B, max_src*2)


# ═══════════════════════════════════════
#  ROLA 损失
# ═══════════════════════════════════════
def rola_loss(pred, target, n_src_batch):
    """
    只对有效源计算 L2 损失，忽略零填充部分
    pred:   (B, max_src*2)
    target: (B, max_src*2)  ROLA 排序后
    n_src_batch: (B,)
    """
    B = pred.shape[0]
    max_s = pred.shape[1] // 2
    pred_2d = pred.reshape(B, max_s, 2)
    tgt_2d = target.reshape(B, max_s, 2)

    # 构建 mask: (B, max_src)
    device = pred.device
    src_idx = torch.arange(max_s, device=device).unsqueeze(0).expand(B, -1)  # (B, max_s)
    n_src_exp = n_src_batch.to(device).unsqueeze(1)                           # (B, 1)
    mask = (src_idx < n_src_exp).float()                                      # (B, max_s)

    # L2 loss per source: (B, max_s)
    diff_sq = ((pred_2d - tgt_2d) ** 2).sum(dim=2)   # (B, max_s)
    masked_loss = (diff_sq * mask).sum()
    n_valid = mask.sum().clamp(min=1)

    return masked_loss / n_valid


# ═══════════════════════════════════════
#  匈牙利匹配评估
# ═══════════════════════════════════════
def hungarian_match_eval(pred_pos, true_pos, n_true):
    """pred_pos: (K,2) meters, true_pos: (n_true,2) meters"""
    if len(pred_pos) == 0 or n_true == 0:
        return np.array([9999.0] * max(n_true, 1))
    K = len(pred_pos)
    cost = np.zeros((n_true, K))
    for t in range(n_true):
        for p in range(K):
            cost[t, p] = np.linalg.norm(pred_pos[p] - true_pos[t])
    row_ind, col_ind = linear_sum_assignment(cost)
    errors = [cost[r, c] for r, c in zip(row_ind, col_ind)]
    if len(errors) < n_true:
        errors.extend([9999.0] * (n_true - len(errors)))
    return np.array(errors)


# ═══════════════════════════════════════
#  验证集评估
# ═══════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_errors = []
    all_nsrc = []
    total_loss, n_batches = 0.0, 0

    for dpd, target, n_src in loader:
        dpd = dpd.to(device)
        target = target.to(device)
        pred = model(dpd)

        loss = rola_loss(pred, target, n_src)
        total_loss += loss.item()
        n_batches += 1

        # 转换为物理坐标 (米) 并用匈牙利匹配
        pred_np = pred.cpu().numpy().reshape(-1, MAX_SRC, 2) * COORD_MAX
        tgt_np = target.cpu().numpy().reshape(-1, MAX_SRC, 2) * COORD_MAX

        for b in range(pred_np.shape[0]):
            n = n_src[b].item()
            if n == 0:
                continue
            errs = hungarian_match_eval(pred_np[b, :n], tgt_np[b, :n], n)
            all_errors.extend(errs.tolist())
            all_nsrc.extend([n] * len(errs))

    errors = np.array(all_errors)
    nsrc_arr = np.array(all_nsrc)

    result = {
        'val_loss': total_loss / max(n_batches, 1),
        'rmse': float(np.sqrt(np.mean(errors**2))),
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'within_10m': float(np.mean(errors < 10) * 100),
        'within_30m': float(np.mean(errors < 30) * 100),
        'within_50m': float(np.mean(errors < 50) * 100),
    }
    for ns in [2, 3]:
        mask = nsrc_arr == ns
        if mask.sum() > 0:
            e = errors[mask]
            result[f'rmse_N{ns}'] = float(np.sqrt(np.mean(e**2)))
            result[f'mean_N{ns}'] = float(np.mean(e))
            result[f'median_N{ns}'] = float(np.median(e))
            result[f'within_10m_N{ns}'] = float(np.mean(e < 10) * 100)
            result[f'within_30m_N{ns}'] = float(np.mean(e < 30) * 100)
            result[f'within_50m_N{ns}'] = float(np.mean(e < 50) * 100)

    return result


# ═══════════════════════════════════════
#  实验评估 (4A2, 4A3 等)
# ═══════════════════════════════════════
@torch.no_grad()
def eval_experiment(model, device, exp_name, data_dir, results_dir='results'):
    """评估实验数据，格式和 eval_exp.py 一致"""
    model.eval()

    # 加载索引
    if exp_name.startswith('4A'):
        exp_type, sub_dir = 'snr', 'exp_4A'
    elif exp_name.startswith('4B'):
        exp_type, sub_dir = 'dist', 'exp_4B'
    elif exp_name == '4C':
        exp_type, sub_dir = 'sep', 'exp_4C'
    elif exp_name == '4D':
        exp_type, sub_dir = 'bw', 'exp_4D'
    else:
        print(f"Unknown experiment: {exp_name}")
        return

    exp_dir = os.path.join(data_dir, 'exp', sub_dir)
    index_path = os.path.join(exp_dir, f'exp_{exp_name}_index.pt')
    if not os.path.exists(index_path):
        print(f"  索引文件不存在: {index_path}")
        return

    index = torch.load(index_path, weights_only=False)
    param_values = index['param_values']
    param_files = index['param_files']
    n_per_param = index['n_per_param']

    param_name_map = {'snr': 'SNR (dB)', 'dist': 'Distance (m)',
                      'sep': 'Separation (m)', 'bw': 'Src2 BW (MHz)'}
    param_name = param_name_map.get(exp_type, exp_type)

    save_name = 'EPN-ResNet18'
    print(f"实验: {exp_name}, 方法: {save_name}")
    print(f"参数: {param_name} = {param_values}")
    print(f"每参数 {n_per_param} 样本")

    all_results = {}

    for pi, (pval, pfile) in enumerate(zip(param_values, param_files)):
        pt_path = os.path.join(exp_dir, pfile)
        if not os.path.exists(pt_path):
            print(f"  [SKIP] 文件不存在: {pt_path}")
            continue

        data = torch.load(pt_path, weights_only=False)
        fine_dpd = data['fine_dpd']       # (N, 1, 401, 401) float16
        pos_label = data['pos_label']     # (N, max_src, 2) normalized [-1,1]
        n_src_all = data['n_src']         # (N,)
        N = len(n_src_all)

        errors_all = []
        bs = 128
        for bi in range(0, N, bs):
            be = min(bi + bs, N)
            # 准备输入
            dpd_batch = fine_dpd[bi:be].float().to(device)
            # per-sample 归一化
            for j in range(dpd_batch.shape[0]):
                mu = dpd_batch[j].mean()
                std = dpd_batch[j].std() + 1e-6
                dpd_batch[j] = (dpd_batch[j] - mu) / std

            pred = model(dpd_batch).cpu().numpy()  # (B, max_src*2)
            pred_m = pred.reshape(-1, MAX_SRC, 2) * COORD_MAX  # meters

            for b in range(pred_m.shape[0]):
                idx_global = bi + b
                n = n_src_all[idx_global].item()
                if n == 0:
                    continue

                # 真实坐标 (米)
                true_m = pos_label[idx_global, :n].numpy() * COORD_MAX  # (n, 2)
                pred_sources = pred_m[b, :n]  # (n, 2)

                errs = hungarian_match_eval(pred_sources, true_m, n)
                errors_all.extend(errs.tolist())

        errors = np.array(errors_all)
        n_total = len(errors)
        metrics = {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'mean': float(np.mean(errors)),
            'median': float(np.median(errors)),
            'within_10m': float(np.mean(errors < 10)),
            'within_30m': float(np.mean(errors < 30)),
            'within_50m': float(np.mean(errors < 50)),
            'n_samples': n_total,
        }
        all_results[pval] = metrics

        print(f"  {param_name}={pval:>6}: "
              f"RMSE={metrics['rmse']:.1f}m  mean={metrics['mean']:.1f}m  "
              f"med={metrics['median']:.1f}m  "
              f"<10m={metrics['within_10m']:.1%}  "
              f"<30m={metrics['within_30m']:.1%}  "
              f"<50m={metrics['within_50m']:.1%}  "
              f"({n_total} samples)", flush=True)

    # 保存结果
    save_dir = os.path.join(results_dir, f'exp_{exp_name}')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{save_name}.pt')
    torch.save({
        'method': save_name,
        'param_name': param_name,
        'param_values': param_values,
        'per_param': all_results,
    }, save_path)
    print(f"\n结果已保存: {save_path}")
    print("Done!")


# ═══════════════════════════════════════
#  训练主函数
# ═══════════════════════════════════════
def main():
    pa = argparse.ArgumentParser(description='EPN (ResNet-18) 训练与评估')
    pa.add_argument('--data_dir', default='data')
    pa.add_argument('--device', default='cuda:0')
    pa.add_argument('--batch_size', type=int, default=96)
    pa.add_argument('--epochs', type=int, default=200)
    pa.add_argument('--lr', type=float, default=1e-3)
    pa.add_argument('--weight_decay', type=float, default=1e-3)
    pa.add_argument('--dropout', type=float, default=0.2)
    pa.add_argument('--patience', type=int, default=30)
    pa.add_argument('--eval_exp', nargs='*', default=None,
                    help='仅评估实验，如: --eval_exp 4A2 4A3')
    pa.add_argument('--model_path', default=None)
    pa.add_argument('--results_dir', default='results')
    args = pa.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # 固定随机种子
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    # ─── 仅评估模式 ───
    if args.eval_exp is not None:
        model = EPNResNet(max_src=MAX_SRC, dropout=0.0).to(device)
        mp = args.model_path or 'best_epn.pth'
        ckpt = torch.load(mp, map_location=device, weights_only=False)
        model.load_state_dict(ckpt)
        print(f"  模型已加载: {mp}")
        for exp in args.eval_exp:
            print(f"\n{'='*50}")
            eval_experiment(model, device, exp, args.data_dir, args.results_dir)
        return

    # ─── 训练模式 ───
    print(f"Device: {device}")
    print(f"Method: EPN (ResNet-18 + FC regression)")
    print(f"Max sources: {MAX_SRC}")

    # 数据
    train_ds = EPNDataset(args.data_dir, 'train')
    val_ds = EPNDataset(args.data_dir, 'val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    # 模型
    model = EPNResNet(max_src=MAX_SRC, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params / 1e6:.2f}M")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练循环
    best_rmse = float('inf')
    patience_cnt = 0

    # 首批检查
    first_batch_done = False

    for ep in range(args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for dpd, target, n_src in train_loader:
            dpd = dpd.to(device)
            target = target.to(device)

            pred = model(dpd)
            loss = rola_loss(pred, target, n_src)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # 首批检查
            if not first_batch_done:
                print(f"\n  [CHECK] First batch:")
                print(f"    dpd: {dpd.shape}, target: {target.shape}")
                print(f"    pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
                print(f"    loss: {loss.item():.4f}")
                first_batch_done = True

        scheduler.step()
        dt = time.time() - t0

        # 验证
        vr = evaluate(model, val_loader, device)

        lr_now = optimizer.param_groups[0]['lr']
        line = (f"[{ep+1:3d}/{args.epochs}] "
                f"train={total_loss/max(n_batches,1):.4f} "
                f"val={vr['val_loss']:.4f} | "
                f"RMSE={vr['rmse']:.1f}m mean={vr['mean_error']:.1f}m "
                f"med={vr['median_error']:.1f}m "
                f"<10m={vr['within_10m']:.1f}% "
                f"<30m={vr['within_30m']:.1f}% "
                f"<50m={vr['within_50m']:.1f}% | "
                f"lr={lr_now:.1e} {dt:.0f}s")
        print(line)

        # 分 N 打印
        for ns in [2, 3]:
            k = f'rmse_N{ns}'
            if k in vr:
                print(f"    N={ns}: RMSE={vr[f'rmse_N{ns}']:.1f}m  "
                      f"mean={vr[f'mean_N{ns}']:.1f}m  "
                      f"med={vr[f'median_N{ns}']:.1f}m  "
                      f"<10m={vr[f'within_10m_N{ns}']:.1f}%  "
                      f"<30m={vr[f'within_30m_N{ns}']:.1f}%  "
                      f"<50m={vr[f'within_50m_N{ns}']:.1f}%")

        # Best 模型
        if vr['rmse'] < best_rmse:
            best_rmse = vr['rmse']
            torch.save(model.state_dict(), 'best_epn.pth')
            print(f"  ★ Best RMSE={best_rmse:.1f}m "
                  f"(mean={vr['mean_error']:.1f}m med={vr['median_error']:.1f}m "
                  f"<30m={vr['within_30m']:.1f}% <50m={vr['within_50m']:.1f}%)")
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt >= args.patience:
            print(f"\nEarly stop at epoch {ep+1}")
            break

    # 最终评估
    model.load_state_dict(torch.load('best_epn.pth', map_location=device, weights_only=False))
    print(f"\n===== Final Validation (EPN) =====")
    vr = evaluate(model, val_loader, device)
    for k, v in vr.items():
        if isinstance(v, float):
            if 'within' in k:
                print(f"  {k}: {v:.1f}%")
            elif 'loss' in k:
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v:.1f}m")

    print("\nDone!")


if __name__ == '__main__':
    main()