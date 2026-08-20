"""
eval_ablation.py — 消融实验：在测试集上评估 A_M3, A_M10, B_M3, B_M10 四个模型

输出：
  1. 总体 Count Accuracy 和 Band Accuracy
  2. 按源数分类的 Count Accuracy
  3. 参数量

用法:
  python eval_ablation.py
"""

import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_THRESHOLD = 0.50


# ═══════════════════════════════════════
#  模型定义 (和 train_v26 一致)
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
def load_test_data(mat_path):
    with h5py.File(mat_path, 'r') as f:
        spectra_raw = np.array(f['mtr_sub_all'], dtype=np.float32)
        src_count   = np.array(f['src_count_all'], dtype=np.int64).flatten()
        band_mask   = np.array(f['band_mask_all'], dtype=np.float32)
        ignore_mask = np.array(f['ignore_mask_all'], dtype=np.float32)
        N_sub       = int(np.array(f['N_sub_val']).item())
        max_src_data = int(np.array(f['max_src_val']).item())

    spectra     = spectra_raw.transpose(3, 2, 1, 0)
    band_mask   = band_mask.transpose(2, 1, 0)
    ignore_mask = ignore_mask.transpose(2, 1, 0)

    # log + z-score
    spectra_dl = np.log(spectra + 1.0)
    for i in range(len(spectra_dl)):
        s = spectra_dl[i]; mu = s.mean(); std = s.std() + 1e-6
        spectra_dl[i] = (s - mu) / std

    return spectra_dl, src_count, band_mask, ignore_mask, N_sub, max_src_data


# ═══════════════════════════════════════
#  评估函数
# ═══════════════════════════════════════
def evaluate_model(model, spectra_dl, src_count_true, band_mask_true,
                   ignore_mask_true, device, threshold=BAND_THRESHOLD):
    N = len(spectra_dl)
    N_sub = model.n_sub
    max_src = model.max_src
    max_src_data = band_mask_true.shape[1]

    # 推理
    all_probs = []
    with torch.no_grad():
        for i in range(0, N, 64):
            j = min(i + 64, N)
            x = torch.from_numpy(spectra_dl[i:j]).to(device)
            bl = model(x)
            all_probs.append(torch.sigmoid(bl).cpu().numpy())
    probs = np.concatenate(all_probs)  # (N, max_src_model, N_sub)
    preds = (probs > threshold).astype(np.int64)

    # Count: 非空slot数
    count_pred = (preds.sum(axis=-1) > 0).sum(axis=-1)  # (N,)

    # Band: 逐子带预测
    band_pred = preds  # (N, max_src_model, N_sub)

    # ── Count Accuracy ──
    count_correct = (count_pred == src_count_true)
    count_acc = count_correct.mean()

    # 按源数统计
    count_acc_by_src = {}
    for s in sorted(np.unique(src_count_true)):
        mask = (src_count_true == s)
        count_acc_by_src[int(s)] = count_correct[mask].mean()

    # ── Band Accuracy ──
    # 对齐 max_src 维度
    ms = min(max_src, max_src_data)
    band_correct_total = 0
    band_eval_total = 0
    for s in range(ms):
        for k in range(N_sub):
            ignore = (ignore_mask_true[:, s, k] > 0.5)
            valid = ~ignore
            true_val = (band_mask_true[:, s, k] > 0.5).astype(np.int64)
            pred_val = preds[:, s, k]
            band_correct_total += ((pred_val == true_val) & valid).sum()
            band_eval_total += valid.sum()
    # 多余 slot 的虚警也要算
    for s in range(ms, max_src):
        for k in range(N_sub):
            # 真值全 0，预测应为 0
            band_correct_total += (preds[:, s, k] == 0).sum()
            band_eval_total += N

    band_acc = band_correct_total / max(band_eval_total, 1)

    return count_acc, count_acc_by_src, band_acc


# ═══════════════════════════════════════
#  主函数
# ═══════════════════════════════════════
def main():
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    test_path = '/mnt/data/ltzdata/test_data.mat'
    print(f"Loading test set: {test_path}")
    spectra_dl, src_count, band_mask, ignore_mask, N_sub, max_src_data = load_test_data(test_path)
    print(f"  {len(src_count)} samples, N_sub={N_sub}")
    for s in sorted(np.unique(src_count)):
        print(f"  {s}src: {(src_count == s).sum()}")

    # 四个模型配置
    configs = [
        ('A_M3',  'best_model_v26_A.pth'),
        ('A_M10', 'best_model_v26_A_M10.pth'),
        ('B_M3',  'best_model_v26_B_M3.pth'),
        ('B_M10', 'best_model_v26_B_M10.pth'),
    ]

    results = []

    for name, path in configs:
        print(f"\n--- {name}: {path} ---")
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = ckpt['cfg']
        model = SourceDetectionNet(
            n_sub=cfg['N_sub'], max_src=cfg['max_src'],
            mode=cfg.get('mode', 'concat')).to(device)
        model.load_state_dict(ckpt['model'])
        model.eval()

        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        mode = cfg.get('mode', 'concat')
        max_src = cfg['max_src']
        print(f"  mode={mode}, max_src={max_src}, params={n_params:.2f}M")

        count_acc, count_by_src, band_acc = evaluate_model(
            model, spectra_dl, src_count, band_mask, ignore_mask, device)

        print(f"  Count Acc: {count_acc:.1%}")
        print(f"  Band Acc:  {band_acc:.1%}")
        for s, acc in count_by_src.items():
            print(f"    {s}src: {acc:.1%}")

        results.append({
            'name': name, 'mode': mode, 'max_src': max_src,
            'params': n_params,
            'count_acc': count_acc, 'band_acc': band_acc,
            'count_by_src': count_by_src,
        })

    # ── 打印消融表 ──
    print("\n" + "=" * 90)
    print("  Ablation Study — Test Set Results")
    print("=" * 90)

    print(f"\n{'Config':<12} {'Mode':<14} {'max_src':<10} {'Params':<10} "
          f"{'Count Acc':<12} {'Band Acc':<12}")
    print("-" * 80)
    for r in results:
        print(f"  {r['name']:<10} {r['mode']:<14} {r['max_src']:<10} "
              f"{r['params']:.2f}M{'':<5} "
              f"{r['count_acc']:<12.1%} {r['band_acc']:<12.1%}")

    print(f"\n{'Config':<12}", end='')
    src_nums = sorted(results[0]['count_by_src'].keys())
    for s in src_nums:
        print(f" {s}src{'':>5}", end='')
    print()
    print("-" * (12 + 10 * len(src_nums)))
    for r in results:
        print(f"  {r['name']:<10}", end='')
        for s in src_nums:
            print(f" {r['count_by_src'][s]:>8.1%}", end='')
        print()

    print("\nDone!")


if __name__ == '__main__':
    main()