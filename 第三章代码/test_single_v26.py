"""
test_single_v26.py — 单样本推理：查看模型对指定样本的预测

用法：
  python test_single_v26.py              # 方案A，测试集第0个样本
  python test_single_v26.py 42           # 方案A，测试集第42个样本
  python test_single_v26.py 100 val      # 方案A，验证集第100个样本
  python test_single_v26.py 42 test B M10   # 方案B，测试集第42个样本
  python test_single_v26.py 0 val B M10  # 方案B M10，验证集第0个样本
"""

import sys
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_THRESHOLD = 0.50


# ── 模型定义（和train_v26一致，无count head）──
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


def main():
    # ── 解析参数 ──
    args = sys.argv[1:]
    idx = 0
    dataset = 'test'
    tag = 'A'
    max_src_override = None

    for a in args:
        if a in ('B',):
            tag = 'B'
        elif a in ('val', 'test', 'train'):
            dataset = a
        elif a.startswith('M') and a[1:].isdigit():
            max_src_override = int(a[1:])
        elif a.isdigit():
            idx = int(a)

    if max_src_override:
        tag += f'_M{max_src_override}'

    mat_path = f'/mnt/data/ltzdata/{dataset}_data.mat'
    model_path = f'best_model_v26_{tag}.pth'

    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # ── 只加载第idx个样本（切片读取，不加载整个文件）──
    with h5py.File(mat_path, 'r') as f:
        N_total = f['src_count_all'].shape[-1]  # 总样本数
        N_sub = int(np.array(f['N_sub_val']).item())
        max_src = int(np.array(f['max_src_val']).item())
        sub_f_lo = np.array(f['sub_f_lo_val'], dtype=np.float32).flatten()
        sub_f_hi = np.array(f['sub_f_hi_val'], dtype=np.float32).flatten()

        # h5py切片：只读第idx列（MATLAB存储是转置的）
        spectrum_raw = np.array(f['mtr_sub_all'][:, :, :, idx], dtype=np.float32)  # (gy, gx, N_sub)
        src_count_val = int(np.array(f['src_count_all'][:, idx]).item())
        band_mask_raw = np.array(f['band_mask_all'][:, :, idx], dtype=np.float32)  # (N_sub, max_src)
        ignore_mask_raw = np.array(f['ignore_mask_all'][:, :, idx], dtype=np.float32)
        fc_offset_raw = np.array(f['fc_offset_all'][:, idx], dtype=np.float32)  # (max_src,)
        avg_snr_val = float(np.array(f['avg_snr_all'][:, idx]).item())

    # 转置为 (N_sub, gx, gy)
    spectrum = spectrum_raw.transpose(2, 1, 0)  # (N_sub, gx, gy)
    true_bm = band_mask_raw.T   # (max_src, N_sub)
    true_ig = ignore_mask_raw.T
    true_count = src_count_val
    fc_offset = fc_offset_raw

    # log + z-score（单个样本）
    spectrum = np.log(spectrum + 1.0)
    mu = spectrum.mean(); std = spectrum.std() + 1e-6
    spectrum = (spectrum - mu) / std

    print(f"Dataset: {mat_path}, {N_total} samples, viewing #{idx}")
    print(f"Model: {model_path} (scheme {tag})")
    print(f"Band threshold: {BAND_THRESHOLD}\n")

    # ── 取出单个样本 ──
    x = torch.from_numpy(spectrum).unsqueeze(0).to(device)

    # ── 加载模型 ──
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt['cfg']
    model = SourceDetectionNet(
        n_sub=cfg['N_sub'], max_src=cfg['max_src'],
        mode=cfg.get('mode', 'concat')
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    model_max_src = cfg['max_src']

    # ── 推理 ──
    with torch.no_grad():
        band_logits = model(x)
        band_probs = torch.sigmoid(band_logits)[0].cpu().numpy()  # (max_src, N_sub)

    # count从band推断
    count_pred = sum(1 for s in range(model_max_src) if (band_probs[s] > BAND_THRESHOLD).any())

    # ── 打印count结果 ──
    print("=" * 80)
    print(f"  Source count:  true={true_count}  pred={count_pred}")
    print(f"  Count method:  non-empty slots (threshold={BAND_THRESHOLD})")
    status = 'OK' if true_count == count_pred else f'WRONG (should be {true_count})'
    print(f"  Status: {status}")
    if true_count > 0:
        print(f"  Avg SNR: {avg_snr_val:.1f}dB")
    print("=" * 80)

    # ── 打印band结果 ──
    # 只打印前max(true_count, count_pred, 3)个slot
    n_show = max(true_count, count_pred, min(3, model_max_src))

    print(f"\n{'Slot':<8} {'Sub':<6} {'Freq(MHz)':<16} {'True':<6} {'Pred':<6} "
          f"{'Prob':<8} {'Status'}")
    print("-" * 70)

    for s in range(n_show):
        is_active = (s < true_count)

        for k in range(N_sub):
            f_lo = sub_f_lo[k] / 1e6
            f_hi = sub_f_hi[k] / 1e6

            true_val = int(true_bm[s, k]) if s < true_bm.shape[0] else 0
            is_ignore = int(true_ig[s, k]) == 1 if s < true_ig.shape[0] else False
            pred_val = 1 if band_probs[s, k] > BAND_THRESHOLD else 0
            prob = band_probs[s, k]

            # 只打印有意义的行
            if true_val == 0 and not is_ignore and pred_val == 0 and not is_active:
                continue

            if is_ignore:
                status_str = 'IGNORE'
            elif true_val == pred_val:
                status_str = 'OK'
            else:
                status_str = 'WRONG'

            slot_str = f'Slot{s+1}'
            print(f"  {slot_str:<6} W{k+1:<4} [{f_lo:>6.1f},{f_hi:>6.1f}] "
                  f"{'  ' + str(true_val):<6} {'  ' + str(pred_val):<6} "
                  f"{prob:<8.1%} {status_str}")

        if is_active:
            active_bands = np.where(true_bm[s] == 1)[0]
            pred_bands = np.where(band_probs[s] > BAND_THRESHOLD)[0]
            if len(active_bands) > 0:
                freq_lo = sub_f_lo[active_bands[0]] / 1e6
                freq_hi = sub_f_hi[active_bands[-1]] / 1e6
                fc_true = fc_offset[s] / 1e6
                print(f"  -> Slot{s+1} true band: [{freq_lo:.1f}, {freq_hi:.1f}]MHz  "
                      f"fc={fc_true:.1f}MHz")
            if len(pred_bands) > 0:
                freq_lo = sub_f_lo[pred_bands[0]] / 1e6
                freq_hi = sub_f_hi[pred_bands[-1]] / 1e6
                print(f"  -> Slot{s+1} pred band: [{freq_lo:.1f}, {freq_hi:.1f}]MHz")
            else:
                print(f"  -> Slot{s+1} pred band: none")
        print()

    # ── 汇总 ──
    print("=" * 80)
    n_pred = count_pred
    print(f"Final output:")
    print(f"  1. Signal present? -> {'Yes' if n_pred > 0 else 'No'}")
    print(f"  2. How many?       -> {n_pred}")
    if n_pred > 0:
        print(f"  3. Frequency bands:")
        for s in range(n_pred):
            pred_bands = np.where(band_probs[s] > BAND_THRESHOLD)[0]
            if len(pred_bands) > 0:
                freq_lo = sub_f_lo[pred_bands[0]] / 1e6
                freq_hi = sub_f_hi[pred_bands[-1]] / 1e6
                print(f"     Source {s+1}: [{freq_lo:.1f}, {freq_hi:.1f}]MHz")
            else:
                print(f"     Source {s+1}: no subbands detected")


if __name__ == '__main__':
    main()