"""
train_yolo.py — YOLOv8 + CenterNet 训练脚本（精简版）

方法:
  D1  (BBox回归):        python train_yolo.py --method bbox
  D2  (高斯热力图):       python train_yolo.py --method gauss
  D3  (距离场):           python train_yolo.py --method distfield
  D4  (高斯+偏移+梯度衰减): python train_yolo.py --method dualhead --grad_alpha 0.1
  D5  (高斯+偏移):        python train_yolo.py --method dualhead
  D6  (高斯+偏移+置信度加权): python train_yolo.py --method dualhead --conf_weight_offset

损失函数:
  D1:  CIoU + BCE + DFL
  D2:  focal + dice
  D3:  focal + dice (距离场标签)
  D5:  focal + dice + offset_L1
  D6:  focal + dice + conf_weighted_offset_L1

保存文件名:
  D1:  best_yolo_bbox.pth
  D2:  best_yolo_gauss.pth
  D3:  best_yolo_distfield.pth
  D5:  best_yolo_dualhead.pth
  D6:  best_yolo_dualhead_cw.pth
"""

import argparse
import copy
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment

from yolo_config import (
    BOX_SIZE,
    DEFAULT_BATCH,
    DEFAULT_DEVICE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    EDGE,
    GRID_SIZE,
    LAMDA,
    MAX_SRC,
    PEAK_SIZE,
    REG_MAX,
    STRIDES,
)
from yolo_model import (
    YOLOv8Loc,
    ciou_loss,
    dice_loss,
    focal_loss_hm,
    nms_heatmap,
    pixel_to_phys,
)

from chapter_runtime import (
    data_dir as runtime_data_dir,
    device as runtime_device,
    output_dir as runtime_output_dir,
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    raise TypeError(f"无法 JSON 序列化 {type(value).__name__}")


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=_json_default,
        )


def append_jsonl(path, payload):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ) + '\n')


def process_rss_bytes():
    process = psutil.Process(os.getpid())
    rss = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            rss += child.memory_info().rss
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    return int(rss)


def configure_reproducibility(seed, deterministic):
    if deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def capture_rng_state(train_generator):
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all(),
        'train_loader_generator': train_generator.get_state(),
    }


def restore_rng_state(state, train_generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])
    train_generator.set_state(state['train_loader_generator'])


def checkpoint_payload(
    *, model_state, args, epoch, best_rmse, optimizer, scheduler, scaler,
    history, train_generator, save_tag,
):
    return {
        # 保留历史评估脚本依赖的 ``model`` 键。
        'model': model_state,
        'method': args.method,
        'save_tag': save_tag,
        'epoch': int(epoch),
        'best_rmse': None if not np.isfinite(best_rmse) else float(best_rmse),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'args': vars(args).copy(),
        'history': history,
        'rng_state': capture_rng_state(train_generator),
    }


# ═══════════════════════════════════════
#  数据集
# ═══════════════════════════════════════
class LocDataset(Dataset):
    def __init__(self, data_dir, split, method='distfield', box_size=BOX_SIZE, augment=False, dist_alpha=1.0):
        self.method = method
        self.box_size = box_size
        self.augment = augment
        self.dist_alpha = dist_alpha

        split_dir = os.path.join(data_dir, split)
        idx_path = os.path.join(split_dir, f'loc_{split}_index.pt')
        idx = torch.load(idx_path, weights_only=False)

        print(f"  Loading {len(idx['shard_files'])} shards for {split}...")
        all_dpd, all_hyp, all_gauss, all_pos, all_n = [], [], [], [], []
        need_hyp = self.method in ('distfield', 'distfield_dual')
        need_gauss = self.method in ('gauss', 'dualhead')

        for sf in idx['shard_files']:
            d = torch.load(os.path.join(split_dir, sf), weights_only=False)
            all_dpd.append(d['fine_dpd'])
            if need_hyp:
                all_hyp.append(d['hyp_mask'])
            if need_gauss:
                all_gauss.append(d['gauss_label'])
            all_pos.append(d['pos_label'])
            all_n.append(d['n_src'])
            del d

        self.dpd   = torch.cat(all_dpd); del all_dpd
        self.hyp   = torch.cat(all_hyp) if need_hyp else None; del all_hyp
        self.gauss = torch.cat(all_gauss) if need_gauss else None; del all_gauss
        self.pos   = torch.cat(all_pos); del all_pos
        self.n     = torch.cat(all_n); del all_n

        mem_mb = self.dpd.element_size() * self.dpd.nelement() / 1e6
        if self.hyp is not None:
            mem_mb += self.hyp.element_size() * self.hyp.nelement() / 1e6
        if self.gauss is not None:
            mem_mb += self.gauss.element_size() * self.gauss.nelement() / 1e6
        mem_mb += self.pos.element_size() * self.pos.nelement() / 1e6
        print(f"  内存占用: {mem_mb/1e3:.1f} GB")

        print(f"  Loaded {len(self.n)} tasks")
        for ns in sorted(self.n.unique().tolist()):
            print(f"    N={ns}: {(self.n==ns).sum().item()}")

    def __len__(self):
        return len(self.n)

    def __getitem__(self, idx):
        dpd = self.dpd[idx].float()    # float16→float32
        pos = self.pos[idx]
        n   = self.n[idx]

        hyp = self.hyp[idx].float() if self.hyp is not None else torch.zeros(MAX_SRC, GRID_SIZE, GRID_SIZE)
        gauss = self.gauss[idx].float() if self.gauss is not None else torch.zeros(1, GRID_SIZE, GRID_SIZE)

        if self.augment:
            dpd, hyp, gauss, pos = self._augment(dpd, hyp, gauss, pos)

        dpd = (dpd - dpd.mean()) / (dpd.std() + 1e-6)

        if self.method == 'bbox':
            target = self._make_bbox_target(pos, n.item())
        elif self.method in ('gauss', 'dualhead'):
            target = gauss
        elif self.method == 'distfield_dual':
            target = self._make_distfield_target(hyp, n.item())
        else:  # distfield
            target = self._make_distfield_target(hyp, n.item())

        return dpd, target, pos, n

    def _augment(self, dpd, hyp, gauss, pos):
        pos = pos.clone()
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            dpd   = torch.rot90(dpd,   k, dims=(-2, -1))
            hyp   = torch.rot90(hyp,   k, dims=(-2, -1))
            gauss = torch.rot90(gauss, k, dims=(-2, -1))
            x, y = pos[:, 0].clone(), pos[:, 1].clone()
            if k == 1:   pos[:, 0], pos[:, 1] =  y, -x
            elif k == 2: pos[:, 0], pos[:, 1] = -x, -y
            elif k == 3: pos[:, 0], pos[:, 1] = -y,  x
        if torch.rand(1).item() > 0.5:
            dpd   = torch.flip(dpd,   dims=(-1,))
            hyp   = torch.flip(hyp,   dims=(-1,))
            gauss = torch.flip(gauss, dims=(-1,))
            pos[:, 0] = -pos[:, 0]
        if torch.rand(1).item() > 0.5:
            dpd   = torch.flip(dpd,   dims=(-2,))
            hyp   = torch.flip(hyp,   dims=(-2,))
            gauss = torch.flip(gauss, dims=(-2,))
            pos[:, 1] = -pos[:, 1]
        return dpd, hyp, gauss, pos

    def _make_distfield_target(self, hyp_per_src, n_src):
        if n_src == 0:
            return torch.zeros(1, GRID_SIZE, GRID_SIZE)
        powered = hyp_per_src[:n_src] ** self.dist_alpha if self.dist_alpha != 1.0 else hyp_per_src[:n_src]
        return powered.max(dim=0, keepdim=True)[0]

    def _make_bbox_target(self, pos, n_src):
        targets = []
        half = self.box_size / 2
        for s in range(n_src):
            px = (pos[s, 0].item() * EDGE + EDGE) / LAMDA
            py = (pos[s, 1].item() * EDGE + EDGE) / LAMDA
            targets.append(torch.tensor([px - half, py - half, px + half, py + half], dtype=torch.float32))
        return torch.stack(targets) if targets else torch.zeros(0, 4)


def collate_fn_bbox(batch):
    dpd = torch.stack([b[0] for b in batch])
    tgt = [b[1] for b in batch]
    pos = torch.stack([b[2] for b in batch])
    n   = torch.stack([b[3] for b in batch])
    return dpd, tgt, pos, n


def collate_fn_hm(batch):
    dpd = torch.stack([b[0] for b in batch])
    tgt = torch.stack([b[1] for b in batch])
    pos = torch.stack([b[2] for b in batch])
    n   = torch.stack([b[3] for b in batch])
    return dpd, tgt, pos, n


# ═══════════════════════════════════════
#  D1 BBox Loss
# ═══════════════════════════════════════
def compute_iou_matrix(boxes1, boxes2, eps=1e-7):
    inter_x1 = torch.max(boxes1[:, 0:1], boxes2[:, 0:1].T)
    inter_y1 = torch.max(boxes1[:, 1:2], boxes2[:, 1:2].T)
    inter_x2 = torch.min(boxes1[:, 2:3], boxes2[:, 2:3].T)
    inter_y2 = torch.min(boxes1[:, 3:4], boxes2[:, 3:4].T)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = ((boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])).clamp(0)
    area2 = ((boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])).clamp(0)
    union = area1[:, None] + area2[None, :] - inter + eps
    return inter / union


def generate_anchors(cls_list, device):
    anchor_centers, anchor_strides = [], []
    for i, cls_pred in enumerate(cls_list):
        H, W = cls_pred.shape[2:]
        stride = STRIDES[i]
        yv, xv = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
        cx = (xv.reshape(-1) + 0.5) * stride
        cy = (yv.reshape(-1) + 0.5) * stride
        anchor_centers.append(torch.stack([cx, cy], 1))
        anchor_strides.append(torch.full((H * W,), stride, device=device))
    return torch.cat(anchor_centers, 0), torch.cat(anchor_strides, 0)


def compute_bbox_loss(model, cls_list, reg_list, targets, device):
    # AMP: 输入可能是 float16，统一转 float32
    cls_list = [c.float() for c in cls_list]
    reg_list = [r.float() for r in reg_list]
    B = cls_list[0].shape[0]
    head = model.head
    pred_boxes, pred_scores = head.decode(cls_list, reg_list)
    anchor_centers, anchor_strides = generate_anchors(cls_list, device)
    N_total = anchor_centers.shape[0]
    reg_flat_all = torch.cat([r.reshape(B, 4 * REG_MAX, -1) for r in reg_list], dim=2)

    total_cls = torch.tensor(0.0, device=device)
    total_ciou = torch.tensor(0.0, device=device)
    total_dfl = torch.tensor(0.0, device=device)
    n_pos_total = 0
    TAL_TOPK, TAL_ALPHA, TAL_BETA = 10, 0.5, 6.0

    for b in range(B):
        gt = targets[b].to(device)
        n_gt = gt.shape[0]
        scores_b = pred_scores[b, :, 0]
        boxes_b = pred_boxes[b]

        if n_gt == 0:
            cls_target = torch.zeros_like(scores_b)
            total_cls += F.binary_cross_entropy_with_logits(scores_b, cls_target, reduction='mean')
            continue

        with torch.no_grad():
            iou = compute_iou_matrix(gt, boxes_b)
            pred_sig = scores_b.sigmoid().clamp(1e-6, 1.0)
            align_metric = pred_sig[None, :] ** TAL_ALPHA * (iou + 1e-6) ** TAL_BETA
            topk = min(TAL_TOPK, N_total)
            _, topk_idx = align_metric.topk(topk, dim=1)
            is_pos = torch.zeros(n_gt, N_total, dtype=torch.bool, device=device)
            is_pos.scatter_(1, topk_idx, True)
            assigned_gt = torch.full((N_total,), -1, dtype=torch.long, device=device)
            pos_any = is_pos.any(dim=0)
            if pos_any.any():
                conflict_iou = iou.clone(); conflict_iou[~is_pos] = -1
                assigned_gt[pos_any] = conflict_iou[:, pos_any].argmax(dim=0)
            cls_target = torch.zeros(N_total, device=device)
            for g in range(n_gt):
                mask_g = assigned_gt == g
                if mask_g.any():
                    am = align_metric[g, mask_g]
                    cls_target[mask_g] = (am / am.max().clamp(min=1e-8)).clamp(0, 1)
            pos_mask = assigned_gt >= 0
            n_pos = pos_mask.sum().item()
            align_weight = cls_target[pos_mask]
            target_scores_sum = max(cls_target.sum().item(), 1.0)

        l_cls = F.binary_cross_entropy_with_logits(scores_b, cls_target, reduction='sum') / target_scores_sum
        total_cls += l_cls

        if n_pos > 0:
            pos_gt_idx = assigned_gt[pos_mask]
            pos_gt_boxes = gt[pos_gt_idx]
            pos_pred_boxes = boxes_b[pos_mask]
            ciou_per = ciou_loss(pos_pred_boxes, pos_gt_boxes)
            total_ciou += (ciou_per * align_weight).sum() / target_scores_sum
            pos_indices = torch.where(pos_mask)[0]
            pos_reg = reg_flat_all[b, :, pos_indices].T.reshape(-1, 4, REG_MAX)
            pos_ac = anchor_centers[pos_indices]
            pos_stride = anchor_strides[pos_indices]
            t_ltrb = torch.stack([
                (pos_ac[:, 0] - pos_gt_boxes[:, 0]) / pos_stride,
                (pos_ac[:, 1] - pos_gt_boxes[:, 1]) / pos_stride,
                (pos_gt_boxes[:, 2] - pos_ac[:, 0]) / pos_stride,
                (pos_gt_boxes[:, 3] - pos_ac[:, 1]) / pos_stride,
            ], dim=1).clamp(0, REG_MAX - 1 - 0.01)
            tl = t_ltrb.long(); tr = tl + 1
            wl = tr.float() - t_ltrb; wr = 1.0 - wl
            log_p = F.log_softmax(pos_reg, dim=-1)
            loss_l = -log_p.gather(-1, tl.unsqueeze(-1)).squeeze(-1) * wl
            loss_r = -log_p.gather(-1, tr.unsqueeze(-1)).squeeze(-1) * wr
            dfl_per = (loss_l + loss_r).mean(dim=1)
            total_dfl += (dfl_per * align_weight).sum() / target_scores_sum
            n_pos_total += n_pos

    total_cls /= B
    if n_pos_total > 0: total_ciou /= B; total_dfl /= B
    loss = 0.5 * total_cls + 7.5 * total_ciou + 1.5 * total_dfl
    return loss, total_cls, total_ciou, total_dfl


# ═══════════════════════════════════════
#  匈牙利匹配（评估用）
# ═══════════════════════════════════════
def hungarian_match_eval(pred_pos, true_pos, n_true):
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
#  D5/D6 Offset Loss
# ═══════════════════════════════════════
def compute_offset_loss(offset_pred, pos, n_src, device, hm_pred=None, soft_conf=False):
    """
    CenterNet 式 offset L1 loss

    hm_pred: 可选 (B, 1, H, W) logits。非 None 时启用置信度加权 (D6)。
             窄带样本 heatmap 响应低 → offset 权重低，自动过滤噪声。
    """
    B, _, H, W = offset_pred.shape
    C = pos.shape[1]

    pos_dev = pos.to(device)
    px_all = (pos_dev[:, :, 0] * EDGE + EDGE) / LAMDA
    py_all = (pos_dev[:, :, 1] * EDGE + EDGE) / LAMDA

    ix_center = px_all.round().long().clamp(0, W - 1)
    iy_center = py_all.round().long().clamp(0, H - 1)

    dx_true = px_all - ix_center.float()
    dy_true = py_all - iy_center.float()

    flat_idx = iy_center * W + ix_center
    offset_flat = offset_pred.reshape(B, 2, -1)
    dx_pred = offset_flat[:, 0].gather(1, flat_idx)
    dy_pred = offset_flat[:, 1].gather(1, flat_idx)

    n_src_dev = n_src.to(device)
    mask = torch.arange(C, device=device).unsqueeze(0) < n_src_dev.unsqueeze(1)

    l1 = torch.abs(dx_pred - dx_true) + torch.abs(dy_pred - dy_true)

    if hm_pred is not None:
        # D6: 置信度加权 — 窄带峰模糊→响应低→权重低
        hm_flat = torch.sigmoid(hm_pred[:, 0]).reshape(B, -1)
        if soft_conf:
            # 软加权: 范围 [0.5, 0.95], 低置信度样本仍保留一半权重
            hm_at_gt = (0.5 + 0.5 * hm_flat.gather(1, flat_idx)).detach()
        else:
            # 硬加权: 范围 [0.1, ~1.0], 低置信度样本几乎不学
            hm_at_gt = hm_flat.gather(1, flat_idx).detach().clamp(min=0.1)
        weighted_l1 = l1 * hm_at_gt * mask.float()
        return weighted_l1.sum() / (hm_at_gt * mask.float()).sum().clamp(min=1)
    else:
        # D5: 等权
        l1 = l1 * mask.float()
        return l1.sum() / mask.sum().clamp(min=1)


# ═══════════════════════════════════════
#  评估
# ═══════════════════════════════════════
@torch.no_grad()
def evaluate(model, loader, device, method, peak_size=PEAK_SIZE, full_metrics=True,
             amp=False, offset_weight=1.0, conf_weight_offset=False, soft_conf=False, dice_weight=1.0):
    model.eval()
    errs, ns_list = [], []
    total_loss, n_batches = 0.0, 0
    _is_dualhead = method in ('dualhead', 'distfield_dual')

    for batch in loader:
        dpd = batch[0].to(device)
        tgt = batch[1]
        pos = batch[2]
        n_src = batch[3]

        with torch.amp.autocast(device_type='cuda', enabled=amp):
            if method == 'bbox':
                cls_list, reg_list = model(dpd)
            elif _is_dualhead:
                pred_hm, pred_offset = model(dpd)
            else:
                pred_hm = model(dpd)

        if method == 'bbox':
            if not all(bool(torch.isfinite(value).all()) for value in (*cls_list, *reg_list)):
                raise FloatingPointError('验证阶段 bbox 输出含 NaN/Inf')
        elif _is_dualhead:
            if not bool(torch.isfinite(pred_hm).all() and torch.isfinite(pred_offset).all()):
                raise FloatingPointError('验证阶段 heatmap/offset 输出含 NaN/Inf')
        elif not bool(torch.isfinite(pred_hm).all()):
            raise FloatingPointError('验证阶段 heatmap 输出含 NaN/Inf')

        # 所有 loss 在 autocast 外
        if method == 'bbox':
            loss, _, _, _ = compute_bbox_loss(model, cls_list, reg_list, tgt, device)
        elif _is_dualhead:
            tgt_dev = tgt.to(device)
            l_hm = focal_loss_hm(pred_hm.float(), tgt_dev)
            if dice_weight > 0:
                l_hm = l_hm + dice_weight * dice_loss(pred_hm.float(), tgt_dev)
            l_offset = compute_offset_loss(pred_offset.float(), pos, n_src, device,
                                           hm_pred=pred_hm.float() if conf_weight_offset else None,
                                           soft_conf=soft_conf)
            loss = l_hm + offset_weight * l_offset
        else:
            tgt_dev = tgt.to(device)
            loss = focal_loss_hm(pred_hm.float(), tgt_dev)
            if dice_weight > 0:
                loss = loss + dice_weight * dice_loss(pred_hm.float(), tgt_dev)

        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('验证阶段 loss 含 NaN/Inf')
        total_loss += loss.item(); n_batches += 1

        if full_metrics:
            if method == 'bbox':
                with torch.amp.autocast(device_type='cuda', enabled=amp):
                    pred_boxes, pred_scores = model.head.decode(cls_list, reg_list)
                bbox_scores = pred_scores[:, :, 0].sigmoid()
                bbox_cx = (pred_boxes[:, :, 0] + pred_boxes[:, :, 2]) / 2
                bbox_cy = (pred_boxes[:, :, 1] + pred_boxes[:, :, 3]) / 2
                K_cand = 20
                topk_sc, topk_idx = bbox_scores.topk(K_cand, dim=1)
                topk_cx = bbox_cx.gather(1, topk_idx)
                topk_cy = bbox_cy.gather(1, topk_idx)
                bbox_cand_scores = topk_sc.cpu().numpy()
                bbox_cand_cx = topk_cx.cpu().numpy()
                bbox_cand_cy = topk_cy.cpu().numpy()
                bbox_nms_dist = peak_size // 2 + 1
            else:
                hm_sig = torch.sigmoid(pred_hm)
                hm_nms = nms_heatmap(hm_sig, peak_size)
                B_cur, _, H, W = hm_nms.shape
                hm_flat = hm_nms[:, 0].reshape(B_cur, -1)
                topk_k = min(MAX_SRC, hm_flat.shape[1])
                topk_scores, topk_indices = hm_flat.topk(topk_k, dim=1)
                topk_x = (topk_indices % W).float()
                topk_y = (topk_indices // W).float()
                if _is_dualhead:
                    for b_i in range(B_cur):
                        for k_i in range(topk_k):
                            ix = int(topk_x[b_i, k_i].item())
                            iy = int(topk_y[b_i, k_i].item())
                            dx = pred_offset[b_i, 0, iy, ix].float()
                            dy = pred_offset[b_i, 1, iy, ix].float()
                            if torch.isfinite(dx) and torch.isfinite(dy):
                                topk_x[b_i, k_i] += dx.clamp(-1, 1)
                                topk_y[b_i, k_i] += dy.clamp(-1, 1)
                topk_coords = torch.stack([topk_x, topk_y], dim=-1)
                all_peaks_phys = pixel_to_phys(topk_coords).cpu().numpy()

            for b in range(dpd.size(0)):
                n = n_src[b].item()
                if n == 0: continue
                true_phys = pos[b, :n].numpy() * EDGE
                if method == 'bbox':
                    sc_b, cx_b, cy_b = bbox_cand_scores[b], bbox_cand_cx[b], bbox_cand_cy[b]
                    keep = []
                    for i in range(len(sc_b)):
                        if sc_b[i] < 1e-6: break
                        too_close = False
                        for j in keep:
                            if np.sqrt((cx_b[i]-cx_b[j])**2 + (cy_b[i]-cy_b[j])**2) < bbox_nms_dist:
                                too_close = True; break
                        if not too_close:
                            keep.append(i)
                            if len(keep) >= n: break
                    pred_phys = np.stack([cx_b[keep], cy_b[keep]], axis=1) * LAMDA - EDGE if keep else np.zeros((0,2))
                else:
                    pred_phys = all_peaks_phys[b, :n]
                errs.extend(hungarian_match_eval(pred_phys, true_phys, n).tolist())
                ns_list.extend([n] * n)

    if n_batches == 0:
        raise RuntimeError('验证 DataLoader 没有产生 batch')
    R = {'val_loss': total_loss / n_batches}
    if full_metrics and len(errs) > 0:
        errs = np.array(errs); ns_list = np.array(ns_list)
        R.update({
            'rmse': np.sqrt((errs**2).mean()), 'mean_error': errs.mean(),
            'median_error': np.median(errs),
            'within_10m': (errs < 10).mean(), 'within_30m': (errs < 30).mean(),
            'within_50m': (errs < 50).mean(),
        })
        for ns in sorted(np.unique(ns_list)):
            m = ns_list == ns; e = errs[m]
            R[f'count_N{ns}'] = int(m.sum())
            R[f'rmse_N{ns}'] = np.sqrt((e**2).mean())
            R[f'mean_N{ns}'] = e.mean(); R[f'median_N{ns}'] = np.median(e)
            R[f'within_10m_N{ns}'] = (e < 10).mean()
            R[f'within_30m_N{ns}'] = (e < 30).mean()
            R[f'within_50m_N{ns}'] = (e < 50).mean()
    return R


# ═══════════════════════════════════════
#  训练
# ═══════════════════════════════════════
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument('--data_dir', type=str, default=str(runtime_data_dir()))
    pa.add_argument('--output_dir', type=str, default=None,
                    help='权重和训练曲线目录；默认按 smoke/formal 模式隔离')
    pa.add_argument('--method', type=str, required=True,
                    choices=['bbox', 'gauss', 'distfield', 'dualhead', 'distfield_dual'])
    pa.add_argument('--device', type=str, default=DEFAULT_DEVICE)
    pa.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    pa.add_argument('--batch_size', type=int, default=DEFAULT_BATCH)
    pa.add_argument('--val_batch_size', type=int, default=None,
                    help='验证 batch；未指定时沿用历史规则 max(batch_size//2, 32)')
    pa.add_argument('--lr', type=float, default=DEFAULT_LR)
    pa.add_argument('--patience', type=int, default=DEFAULT_PATIENCE)
    pa.add_argument('--peak_size', type=int, default=PEAK_SIZE)
    pa.add_argument('--box_size', type=int, default=BOX_SIZE)
    pa.add_argument('--eval_every', type=int, default=1)
    pa.add_argument('--amp', action='store_true', default=True)
    pa.add_argument('--no_amp', dest='amp', action='store_false')
    pa.add_argument('--dist_alpha', type=float, default=2.0)
    pa.add_argument('--offset_weight', type=float, default=1.0,
                    help='D5/D6 offset loss 权重')
    pa.add_argument('--dice_weight', type=float, default=1.0)
    pa.add_argument('--weight_decay', type=float, default=5e-3)
    pa.add_argument('--dropout', type=float, default=0.4)
    pa.add_argument('--conf_weight_offset', action='store_true', default=False,
                    help='D6: 置信度加权 offset loss (窄带自动降权)')
    pa.add_argument('--soft_conf', action='store_true', default=False,
                    help='D6_soft: 软置信度加权 (floor=0.5)')
    pa.add_argument('--grad_alpha', type=float, default=1.0,
                    help='D4: offset 梯度回传到 backbone 的缩放因子 (0~1, 默认1.0=D5)')
    pa.add_argument('--seed', type=int, default=42)
    pa.add_argument('--deterministic', action='store_true', default=False,
                    help='启用严格确定性算法；不支持的算子会直接报错')
    pa.add_argument('--fail_on_nonfinite', action='store_true', default=False,
                    help='每个 epoch 额外检查全部模型参数是否有限')
    pa.add_argument('--save_last_every_epoch', action='store_true', default=False,
                    help='每个 epoch 覆盖保存可诊断的 last checkpoint')
    pa.add_argument('--require_empty_output', action='store_true', default=False,
                    help='拒绝写入已有文件的输出目录')
    pa.add_argument('--gate3_d8', action='store_true', default=False,
                    help='Gate 3 D8 固定配置强校验，不改变训练算法')
    pa.add_argument('--gate3b_d8', action='store_true', default=False,
                    help='Gate 3B D8 60 epoch 固定配置强校验，不改变训练算法')
    pa.add_argument('--s2g2_d8', action='store_true', default=False,
                    help='S2-G2 D8 200 epoch 诊断配置强校验，不改变训练算法')
    pa.add_argument('--run_label', type=str, default='')
    args = pa.parse_args()
    if args.batch_size <= 0:
        pa.error('--batch_size 必须为正整数')
    if args.val_batch_size is not None and args.val_batch_size <= 0:
        pa.error('--val_batch_size 必须为正整数')
    if args.epochs <= 0 or args.patience <= 0 or args.eval_every <= 0:
        pa.error('epochs、patience 与 eval_every 必须为正整数')
    if args.seed < 0:
        pa.error('--seed 必须为非负整数')

    if args.gate3_d8 or args.gate3b_d8 or args.s2g2_d8:
        gate3_checks = {
            'method=dualhead': args.method == 'dualhead',
            'amp=False': not args.amp,
            'dice_weight=0': args.dice_weight == 0.0,
            'grad_alpha=1': args.grad_alpha == 1.0,
            'offset_weight=1': args.offset_weight == 1.0,
            'conf_weight_offset=False': not args.conf_weight_offset,
            'soft_conf=False': not args.soft_conf,
            'batch_size=8': args.batch_size == 8,
            'val_batch_size=8': args.val_batch_size == 8,
            'lr=1e-3': args.lr == 1e-3,
            'weight_decay=5e-3': args.weight_decay == 5e-3,
            'dropout=0.4': args.dropout == 0.4,
            'eval_every=1': args.eval_every == 1,
            'deterministic=True': args.deterministic,
            'fail_on_nonfinite=True': args.fail_on_nonfinite,
            'save_last_every_epoch=True': args.save_last_every_epoch,
            'require_empty_output=True': args.require_empty_output,
        }
        failed = [name for name, passed in gate3_checks.items() if not passed]
        if failed:
            pa.error('Gate 3 D8 参数不符合冻结配置: ' + ', '.join(failed))
    if args.gate3b_d8:
        gate3b_checks = {
            'epochs=60': args.epochs == 60,
            'patience=60': args.patience == 60,
            'seed=42': args.seed == 42,
            'gate3_d8=False': not args.gate3_d8,
        }
        failed = [name for name, passed in gate3b_checks.items() if not passed]
        if failed:
            pa.error('Gate 3B D8 参数不符合冻结配置: ' + ', '.join(failed))
    if args.s2g2_d8:
        s2g2_checks = {
            'epochs=200': args.epochs == 200,
            'patience=200': args.patience == 200,
            'seed=42': args.seed == 42,
            'device=cuda:0': args.device == 'cuda:0',
            'peak_size=9': args.peak_size == 9,
            'box_size=9': args.box_size == 9,
            'gate3_d8=False': not args.gate3_d8,
            'gate3b_d8=False': not args.gate3b_d8,
        }
        failed = [name for name, passed in s2g2_checks.items() if not passed]
        if failed:
            pa.error('S2-G2 D8 参数不符合冻结配置: ' + ', '.join(failed))

    configure_reproducibility(args.seed, args.deterministic)

    device = runtime_device(args.device)
    output_dir = Path(args.output_dir or str(runtime_output_dir('train_yolo'))).resolve()
    if args.require_empty_output and output_dir.exists() and any(output_dir.iterdir()):
        pa.error(f'--require_empty_output 拒绝写入非空目录: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    print(f"Peak/NMS size: {args.peak_size}, Box size: {args.box_size}")
    if args.method in ('distfield', 'distfield_dual'):
        print(f"Dist alpha: {args.dist_alpha}")
    if args.method in ('dualhead', 'distfield_dual'):
        print(f"Offset weight: {args.offset_weight}")
        if args.conf_weight_offset:
            print("Conf-weighted offset: enabled (D6)")
        if args.grad_alpha < 1.0:
            print(f"Grad alpha: {args.grad_alpha} (D4 梯度衰减)")
    if args.weight_decay != 5e-3:
        print(f"Weight decay: {args.weight_decay}")
    if args.dropout != 0.4:
        print(f"Dropout: {args.dropout}")
    print(f"Seed/deterministic: {args.seed}/{args.deterministic}")

    method_to_model = {
        'bbox': 'bbox', 'gauss': 'heatmap', 'distfield': 'heatmap',
        'dualhead': 'dualhead', 'distfield_dual': 'dualhead',
    }
    model_method = method_to_model[args.method]
    is_dualhead = args.method in ('dualhead', 'distfield_dual')

    # soft_conf 隐含 conf_weight_offset
    if args.soft_conf:
        args.conf_weight_offset = True

    # 保存标签
    if args.method == 'distfield_dual':
        save_tag = 'distfield_dual'  # D3d (距离场 + offset)
    elif is_dualhead and args.dice_weight == 0 and args.grad_alpha >= 1.0 and not args.conf_weight_offset and not args.soft_conf:
        save_tag = 'dualhead_std'   # D8 (标准 CenterNet，无任何改进)
    elif is_dualhead and args.grad_alpha < 1.0 and args.soft_conf:
        save_tag = 'dualhead_ga_cws'  # D7_soft
    elif is_dualhead and args.grad_alpha < 1.0 and args.conf_weight_offset:
        save_tag = 'dualhead_ga_cw'  # D7
    elif is_dualhead and args.grad_alpha < 1.0:
        save_tag = 'dualhead_ga'    # D4
    elif is_dualhead and args.soft_conf:
        save_tag = 'dualhead_cws'   # D6_soft
    elif is_dualhead and args.conf_weight_offset:
        save_tag = 'dualhead_cw'    # D6
    else:
        save_tag = args.method      # D1/D2/D3/D5

    environment = {
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version(),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'pythonhashseed': os.environ.get('PYTHONHASHSEED'),
    }
    write_json(output_dir / 'run_config.json', {
        'status': 'CONFIGURED',
        'run_label': args.run_label,
        'model_label': 'D8' if (args.gate3_d8 or args.gate3b_d8 or args.s2g2_d8) else save_tag,
        'args': vars(args),
        'environment': environment,
        'performance_interpretation_allowed': False,
    })

    # 数据
    train_ds = LocDataset(args.data_dir, 'train', method=args.method, box_size=args.box_size,
                          augment=True, dist_alpha=args.dist_alpha)
    val_ds   = LocDataset(args.data_dir, 'val', method=args.method, box_size=args.box_size,
                          augment=False, dist_alpha=args.dist_alpha)
    cfn = collate_fn_bbox if args.method == 'bbox' else collate_fn_hm
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True, collate_fn=cfn,
                              generator=train_generator)
    val_batch_size = (args.val_batch_size if args.val_batch_size is not None
                      else max(args.batch_size // 2, 32))
    val_loader   = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False,
                              num_workers=0, pin_memory=True, collate_fn=cfn)
    print(f"Train/val batch size: {args.batch_size}/{val_batch_size}")

    # 模型
    model = YOLOv8Loc(method=model_method, dropout=args.dropout, grad_alpha=args.grad_alpha).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler(enabled=args.amp, init_scale=2**10)

    initial_validation_path = None
    initial_validation = None
    if args.gate3b_d8 or args.s2g2_d8:
        rng_before_initial_validation = capture_rng_state(train_generator)
        try:
            initial_validation = evaluate(
                model, val_loader, device, args.method,
                peak_size=args.peak_size, full_metrics=True, amp=args.amp,
                offset_weight=args.offset_weight,
                conf_weight_offset=args.conf_weight_offset,
                soft_conf=args.soft_conf, dice_weight=args.dice_weight,
            )
        finally:
            restore_rng_state(rng_before_initial_validation, train_generator)
        initial_validation_path = output_dir / 'initial_validation.json'
        write_json(initial_validation_path, {
            'split': 'val',
            'checkpoint_selection': 'initial_untrained_model',
            'metrics': initial_validation,
            'rng_state_restored': True,
            'performance_interpretation_allowed': False,
        })

    best_rmse = float('inf'); best_st = None; best_epoch = None; no_imp = 0
    hist = {'tl': [], 'vl': [], 'vrmse': [], 'vmean': [], 'vmed': [], 'v30': [], 'v50': [], 'vrmse_ep': []}
    epoch_records = []
    history_jsonl = output_dir / 'epoch_history.jsonl'
    history_json = output_dir / 'epoch_history.json'
    if history_jsonl.exists():
        raise FileExistsError(f'结构化历史已存在，拒绝追加覆盖: {history_jsonl}')
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    training_started = time.perf_counter()
    stopped_early = False

    for ep in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train(); tl, nb = 0, 0
        tl_focal, tl_dice, tl_offset = 0, 0, 0

        for batch_index, batch in enumerate(train_loader):
            dpd = batch[0].to(device)
            tgt = batch[1]

            with torch.amp.autocast(device_type='cuda', enabled=args.amp):
                if args.method == 'bbox':
                    cls_list, reg_list = model(dpd)
                elif is_dualhead:
                    pred_hm, pred_offset = model(dpd)
                else:
                    pred = model(dpd)

            # 所有 loss 在 autocast 外 → float32 计算
            if args.method == 'bbox':
                loss, l_cls, l_ciou, l_dfl = compute_bbox_loss(model, cls_list, reg_list, tgt, device)
            elif is_dualhead:
                tgt_dev = tgt.to(device)
                l_focal = focal_loss_hm(pred_hm.float(), tgt_dev)
                pos_batch = batch[2]; n_batch = batch[3]
                l_offset = compute_offset_loss(
                    pred_offset.float(), pos_batch, n_batch, device,
                    hm_pred=pred_hm.float() if args.conf_weight_offset else None,
                    soft_conf=args.soft_conf)
                if args.dice_weight > 0:
                    l_dice = dice_loss(pred_hm.float(), tgt_dev)
                    loss = l_focal + args.dice_weight * l_dice + args.offset_weight * l_offset
                    tl_dice += l_dice.item()
                else:
                    loss = l_focal + args.offset_weight * l_offset
                tl_focal += l_focal.item(); tl_offset += l_offset.item()
            else:
                tgt_dev = tgt.to(device)
                l_focal = focal_loss_hm(pred.float(), tgt_dev)
                if args.dice_weight > 0:
                    l_dice = dice_loss(pred.float(), tgt_dev)
                    loss = l_focal + args.dice_weight * l_dice
                    tl_dice += l_dice.item()
                else:
                    loss = l_focal
                tl_focal += l_focal.item()

            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f'训练 loss 非有限: epoch={ep + 1}, batch={batch_index + 1}'
                )

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 10.0, error_if_nonfinite=True
            )
            scaler.step(opt); scaler.update()
            tl += loss.item(); nb += 1

            if ep == 0 and nb == 1:
                print(f"\n  [CHECK] First batch: dpd={dpd.shape}")
                if args.method == 'bbox':
                    print(f"    cls: {[c.shape for c in cls_list]}, reg: {[r.shape for r in reg_list]}")
                    print(f"    loss={loss.item():.4f} (cls={l_cls.item():.4f} ciou={l_ciou.item():.4f} dfl={l_dfl.item():.4f})")
                elif is_dualhead:
                    print(f"    pred_hm: {pred_hm.shape}, offset: {pred_offset.shape}, target: {tgt_dev.shape}")
                    print(f"    sigmoid range: [{torch.sigmoid(pred_hm).min():.4f}, {torch.sigmoid(pred_hm).max():.4f}]")
                    if args.dice_weight > 0:
                        print(f"    loss={loss.item():.4f} (focal={l_focal.item():.4f} dice={l_dice.item():.4f} offset={l_offset.item():.4f})")
                    else:
                        print(f"    loss={loss.item():.4f} (focal={l_focal.item():.4f} offset={l_offset.item():.4f})")
                else:
                    print(f"    pred: {pred.shape}, target: {tgt_dev.shape}")
                    print(f"    sigmoid range: [{torch.sigmoid(pred).min():.4f}, {torch.sigmoid(pred).max():.4f}]")
                    if args.dice_weight > 0:
                        print(f"    loss={loss.item():.4f} (focal={l_focal.item():.4f} dice={l_dice.item():.4f})")
                    else:
                        print(f"    loss={loss.item():.4f} (focal={l_focal.item():.4f})")

        if nb == 0:
            raise RuntimeError(f'epoch={ep + 1} 没有产生有效训练 batch')
        sch.step(); lr = sch.get_last_lr()[0]

        do_full = (ep + 1) % args.eval_every == 0 or ep == 0 or ep >= args.epochs - 1
        torch.cuda.empty_cache()
        vr = evaluate(model, val_loader, device, args.method,
                      peak_size=args.peak_size, full_metrics=do_full, amp=args.amp,
                      offset_weight=args.offset_weight, conf_weight_offset=args.conf_weight_offset,
                      soft_conf=args.soft_conf, dice_weight=args.dice_weight)

        if args.method == 'bbox': loss_type = '(cls+ciou+dfl)'
        elif is_dualhead and args.dice_weight > 0: loss_type = '(focal+dice+offset)'
        elif is_dualhead: loss_type = '(focal+offset)'
        elif args.dice_weight > 0: loss_type = '(focal+dice)'
        else: loss_type = '(focal)'
        line = f"[{ep+1:3d}/{args.epochs}] train={tl/max(nb,1):.4f} val={vr['val_loss']:.4f} {loss_type}"
        if is_dualhead and args.dice_weight > 0:
            line += f" [focal={tl_focal/max(nb,1):.4f} dice={tl_dice/max(nb,1):.4f} offset={tl_offset/max(nb,1):.4f}]"
        elif is_dualhead:
            line += f" [focal={tl_focal/max(nb,1):.4f} offset={tl_offset/max(nb,1):.4f}]"
        elif args.method != 'bbox' and args.dice_weight > 0:
            line += f" [focal={tl_focal/max(nb,1):.4f} dice={tl_dice/max(nb,1):.4f}]"
        elif args.method != 'bbox':
            line += f" [focal={tl_focal/max(nb,1):.4f}]"
        if do_full:
            line += (f" | RMSE={vr['rmse']:.1f}m mean={vr['mean_error']:.1f}m "
                     f"med={vr['median_error']:.1f}m "
                     f"<10m={vr['within_10m']:.1%} <30m={vr['within_30m']:.1%} <50m={vr['within_50m']:.1%}")
        line += f" | lr={lr:.1e}"
        print(line)

        if do_full:
            for k in [1, 2, 3]:
                if f'rmse_N{k}' in vr:
                    print(f"    N={k}({vr.get(f'count_N{k}',0):>5}): "
                          f"RMSE={vr[f'rmse_N{k}']:.1f}m  mean={vr[f'mean_N{k}']:.1f}m  "
                          f"med={vr[f'median_N{k}']:.1f}m  "
                          f"<10m={vr[f'within_10m_N{k}']:.1%}  <30m={vr[f'within_30m_N{k}']:.1%}  <50m={vr[f'within_50m_N{k}']:.1%}")

        hist['tl'].append(tl/max(nb,1)); hist['vl'].append(vr['val_loss'])
        if do_full:
            hist['vrmse'].append(vr['rmse']); hist['vmean'].append(vr['mean_error'])
            hist['vmed'].append(vr['median_error'])
            hist['v30'].append(vr.get('within_30m', 0)); hist['v50'].append(vr.get('within_50m', 0))
            hist['vrmse_ep'].append(ep + 1)

        best_updated = False
        if do_full and ep >= 5 and vr['rmse'] < best_rmse:
            best_rmse = vr['rmse']
            best_st = copy.deepcopy(model.state_dict()); no_imp = 0
            best_epoch = ep + 1
            best_updated = True
            print(f"  ★ Best RMSE={best_rmse:.1f}m (mean={vr['mean_error']:.1f}m "
                  f"med={vr['median_error']:.1f}m <30m={vr['within_30m']:.1%} <50m={vr['within_50m']:.1%})")
        elif do_full:
            no_imp += 1

        if args.fail_on_nonfinite:
            bad_parameter = next((
                name for name, parameter in model.named_parameters()
                if not bool(torch.isfinite(parameter).all())
            ), None)
            if bad_parameter is not None:
                raise FloatingPointError(
                    f'epoch={ep + 1} 后参数 {bad_parameter} 含 NaN/Inf'
                )

        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        epoch_record = {
            'epoch': ep + 1,
            'train_loss': tl / nb,
            'train_focal_loss': tl_focal / nb if args.method != 'bbox' else None,
            'train_dice_loss': tl_dice / nb if args.method != 'bbox' and args.dice_weight > 0 else None,
            'train_offset_loss': tl_offset / nb if is_dualhead else None,
            'val_loss': vr['val_loss'],
            'full_metrics': do_full,
            'rmse': vr.get('rmse'),
            'mean_error': vr.get('mean_error'),
            'median_error': vr.get('median_error'),
            'within_10m': vr.get('within_10m'),
            'within_30m': vr.get('within_30m'),
            'within_50m': vr.get('within_50m'),
            'rmse_N2': vr.get('rmse_N2'),
            'rmse_N3': vr.get('rmse_N3'),
            'count_N2': vr.get('count_N2'),
            'count_N3': vr.get('count_N3'),
            'learning_rate': lr,
            'train_batches': nb,
            'epoch_seconds': epoch_seconds,
            'process_rss_bytes': process_rss_bytes(),
            'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0,
            'cuda_peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else 0,
            'best_rmse_so_far': None if not np.isfinite(best_rmse) else float(best_rmse),
            'no_improvement_count': no_imp,
        }
        epoch_records.append(epoch_record)
        append_jsonl(history_jsonl, epoch_record)
        write_json(history_json, epoch_records)

        if best_updated:
            torch.save(checkpoint_payload(
                model_state=best_st,
                args=args,
                epoch=ep + 1,
                best_rmse=best_rmse,
                optimizer=opt,
                scheduler=sch,
                scaler=scaler,
                history=epoch_records,
                train_generator=train_generator,
                save_tag=save_tag,
            ), output_dir / f'best_yolo_{save_tag}.pth')

        if args.save_last_every_epoch:
            torch.save(checkpoint_payload(
                model_state=model.state_dict(),
                args=args,
                epoch=ep + 1,
                best_rmse=best_rmse,
                optimizer=opt,
                scheduler=sch,
                scaler=scaler,
                history=epoch_records,
                train_generator=train_generator,
                save_tag=save_tag,
            ), output_dir / f'last_yolo_{save_tag}.pth')

        if no_imp >= args.patience:
            stopped_early = True
            print(f"\nEarly stop at epoch {ep+1}"); break

    # 画图
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ep_list = range(1, len(hist['tl']) + 1)
    ep_metric = hist['vrmse_ep']
    ax[0].plot(ep_list, hist['tl'], 'b-', label='Train')
    ax[0].plot(ep_list, hist['vl'], 'r-', label='Val')
    ax[0].set_title('Loss'); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(ep_metric, hist['vrmse'], 'r-o', markersize=3, label='RMSE')
    ax[1].plot(ep_metric, hist['vmean'], 'g--s', markersize=3, alpha=0.5, label='Mean')
    ax[1].plot(ep_metric, hist['vmed'], 'b-^', markersize=3, label='Median')
    ax[1].set_title('Val Error (m)'); ax[1].legend(); ax[1].grid(True)
    ax[2].plot(ep_metric, [v*100 for v in hist['v50']], 'g-o', markersize=3)
    ax[2].set_title('Val <50m (%)'); ax[2].grid(True)
    curves_path = os.path.join(output_dir, f'training_curves_{save_tag}.png')
    plt.tight_layout(); plt.savefig(curves_path, dpi=120)
    plt.close(fig)
    print(f"Curves saved: {curves_path}")

    if best_st is not None: model.load_state_dict(best_st)
    print(f"\n===== Final Validation ({save_tag}) =====")
    vr = evaluate(model, val_loader, device, args.method, peak_size=args.peak_size, amp=args.amp,
                  offset_weight=args.offset_weight, conf_weight_offset=args.conf_weight_offset,
                  soft_conf=args.soft_conf, dice_weight=args.dice_weight)
    for k, v in vr.items():
        if isinstance(v, float):
            if 'within' in k: print(f"  {k}: {v:.1%}")
            elif 'loss' in k: print(f"  {k}: {v:.4f}")
            else: print(f"  {k}: {v:.1f}m")
    write_json(output_dir / 'final_validation.json', {
        'split': 'val',
        'checkpoint_selection': 'best_rmse' if best_st is not None else 'last_epoch',
        'metrics': vr,
        'performance_interpretation_allowed': False,
    })
    write_json(output_dir / 'training_summary.json', {
        'status': 'PASS',
        'run_label': args.run_label,
        'model_label': 'D8' if (args.gate3_d8 or args.gate3b_d8 or args.s2g2_d8) else save_tag,
        'epochs_completed': len(epoch_records),
        'stopped_early': stopped_early,
        'best_epoch': best_epoch,
        'best_rmse': None if not np.isfinite(best_rmse) else float(best_rmse),
        'final_learning_rate': float(sch.get_last_lr()[0]),
        'total_elapsed_seconds': time.perf_counter() - training_started,
        'process_rss_bytes_end': process_rss_bytes(),
        'cuda_peak_allocated_bytes': int(torch.cuda.max_memory_allocated(device)) if device.type == 'cuda' else 0,
        'cuda_peak_reserved_bytes': int(torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else 0,
        'best_checkpoint': str((output_dir / f'best_yolo_{save_tag}.pth').resolve()) if best_st is not None else None,
        'last_checkpoint': str((output_dir / f'last_yolo_{save_tag}.pth').resolve()) if args.save_last_every_epoch else None,
        'history_json': str(history_json.resolve()),
        'history_jsonl': str(history_jsonl.resolve()),
        'initial_validation': (
            str(initial_validation_path.resolve()) if initial_validation_path is not None else None
        ),
        'initial_validation_metrics': initial_validation,
        'final_validation': str((output_dir / 'final_validation.json').resolve()),
        'performance_interpretation_allowed': False,
    })
    print("\nDone!")


if __name__ == '__main__':
    main()
