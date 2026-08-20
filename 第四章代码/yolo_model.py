"""
yolo_model.py — YOLOv8 定位模型（使用 ultralytics 标准模块）

架构 (YOLOv8s, width=0.5, depth=0.33):
  Backbone: CSPDarknet  416→208→104→52(P3)→26(P4)→13(P5+SPPF)
  PAN:      标准双向特征融合
  D1 Head:  多尺度解耦检测头 (CIoU+BCE+DFL)
  D2/D3 Head: 上采样解码器 → 401×401 热力图

输入 401→padding到416(32倍数)→处理→裁剪回401

改动说明（相对原始 D5）:
  1. offset_head 从 1×1 conv 改为 3×3 conv + ReLU + 1×1 conv
  2. 无 detach，offset 梯度正常回传到 backbone
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, SPPF

from yolo_config import *

PAD_SIZE = 416


class YOLOv8Backbone(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.conv0 = Conv(in_ch, 32, 3, 2)
        self.conv1 = Conv(32, 64, 3, 2)
        self.c2f_2 = C2f(64, 64, n=1, shortcut=True)
        self.conv3 = Conv(64, 128, 3, 2)
        self.c2f_4 = C2f(128, 128, n=2, shortcut=True)
        self.conv5 = Conv(128, 256, 3, 2)
        self.c2f_6 = C2f(256, 256, n=2, shortcut=True)
        self.conv7 = Conv(256, 512, 3, 2)
        self.c2f_8 = C2f(512, 512, n=1, shortcut=True)
        self.sppf9 = SPPF(512, 512, 5)

    def forward(self, x):
        e1 = self.conv0(x)
        x  = self.conv1(e1)
        e2 = self.c2f_2(x)
        x  = self.conv3(e2)
        p3 = self.c2f_4(x)
        x  = self.conv5(p3)
        p4 = self.c2f_6(x)
        x  = self.conv7(p4)
        x  = self.c2f_8(x)
        p5 = self.sppf9(x)
        return e1, e2, p3, p4, p5


class PAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.up      = nn.Upsample(scale_factor=2, mode='nearest')
        self.td4_c2f = C2f(512 + 256, 256, n=1, shortcut=False)
        self.td3_c2f = C2f(256 + 128, 128, n=1, shortcut=False)
        self.down3   = Conv(128, 128, 3, 2)
        self.bu4_c2f = C2f(128 + 256, 256, n=1, shortcut=False)
        self.down4   = Conv(256, 256, 3, 2)
        self.bu5_c2f = C2f(256 + 512, 512, n=1, shortcut=False)

    def forward(self, p3, p4, p5):
        td4 = self.td4_c2f(torch.cat([self.up(p5), p4], 1))
        td3 = self.td3_c2f(torch.cat([self.up(td4), p3], 1))
        bu4 = self.bu4_c2f(torch.cat([self.down3(td3), td4], 1))
        bu5 = self.bu5_c2f(torch.cat([self.down4(bu4), p5], 1))
        return td3, bu4, bu5


class DFL(nn.Module):
    def __init__(self, c1=REG_MAX):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = nn.Parameter(
            torch.arange(c1, dtype=torch.float32).view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, _, a = x.shape
        x = x.float()  # AMP 安全：确保与 conv 权重类型一致
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)


class DetectHead(nn.Module):
    def __init__(self, nc=1, reg_max=REG_MAX, pan_channels=(128, 256, 512)):
        super().__init__()
        self.nc = nc
        self.reg_max = reg_max
        self.dfl = DFL(reg_max)
        c2 = max(16, pan_channels[0] // 4, reg_max * 4)
        c3 = max(pan_channels[0], min(nc, 100))
        self.cls_heads = nn.ModuleList()
        self.reg_heads = nn.ModuleList()
        for c in pan_channels:
            self.reg_heads.append(nn.Sequential(
                Conv(c, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)))
            self.cls_heads.append(nn.Sequential(
                Conv(c, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, nc, 1)))

    def forward(self, features):
        cls_list, reg_list = [], []
        for i, feat in enumerate(features):
            cls_list.append(self.cls_heads[i](feat))
            reg_list.append(self.reg_heads[i](feat))
        return cls_list, reg_list

    def decode(self, cls_list, reg_list):
        B = cls_list[0].shape[0]
        device = cls_list[0].device
        all_boxes, all_scores = [], []
        for i, (cls_pred, reg_pred) in enumerate(zip(cls_list, reg_list)):
            H, W = cls_pred.shape[2:]
            stride = STRIDES[i]
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32), indexing='ij')
            anchor_x = (xv + 0.5) * stride
            anchor_y = (yv + 0.5) * stride
            reg_flat = reg_pred.view(B, 4 * self.reg_max, H * W)
            dist = self.dfl(reg_flat).view(B, 4, H, W)
            x1 = anchor_x - dist[:, 0] * stride
            y1 = anchor_y - dist[:, 1] * stride
            x2 = anchor_x + dist[:, 2] * stride
            y2 = anchor_y + dist[:, 3] * stride
            boxes = torch.stack([x1, y1, x2, y2], 1).view(B, 4, -1).permute(0, 2, 1)
            scores = cls_pred.view(B, self.nc, -1).permute(0, 2, 1)
            all_boxes.append(boxes)
            all_scores.append(scores)
        return torch.cat(all_boxes, 1), torch.cat(all_scores, 1)


class GradScaler(torch.autograd.Function):
    """梯度衰减层：前向传播值不变，反向传播梯度乘以 alpha"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.alpha, None


class HeatmapDecoder(nn.Module):
    def __init__(self, out_ch=1, with_offset=False, dropout=0.4, grad_alpha=1.0):
        super().__init__()
        self.with_offset = with_offset
        self.grad_alpha = grad_alpha
        self.up4 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.c4  = Conv(512 + 256, 256, 3)
        self.up3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.c3  = Conv(256 + 128, 128, 3)
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.c2  = Conv(128 + 64, 64, 3)
        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.c1  = Conv(64 + 32, 32, 3)
        self.up0  = nn.ConvTranspose2d(32, 32, 2, 2)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, out_ch, 1),
        )
        nn.init.constant_(self.head[-1].bias, -2.19)  # CenterNet 标准初始化

        # Offset 分支: 3×3 conv + ReLU + 1×1 conv（扩大感受野）
        if with_offset:
            self.offset_head = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 2, 1)
            )
            nn.init.zeros_(self.offset_head[-1].weight)
            nn.init.zeros_(self.offset_head[-1].bias)

        self.drop = nn.Dropout2d(dropout)

    def forward(self, n5, n4, n3, e2, e1):
        d4 = self.drop(self.c4(torch.cat([self.up4(n5), n4], 1)))
        d3 = self.drop(self.c3(torch.cat([self.up3(d4), n3], 1)))
        d2 = self.drop(self.c2(torch.cat([self.up2(d3), e2], 1)))
        d1 = self.c1(torch.cat([self.up1(d2), e1], 1))
        d0 = self.up0(d1)
        hm = self.head(d0)
        if self.with_offset:
            if self.grad_alpha < 1.0:
                d0_offset = GradScaler.apply(d0, self.grad_alpha)
            else:
                d0_offset = d0
            offset = self.offset_head(d0_offset)
            return hm, offset
        return hm


class YOLOv8Loc(nn.Module):
    """
    method='bbox':      Backbone + PAN + DetectHead (D1)
    method='heatmap':   Backbone + PAN + HeatmapDecoder(1ch) (D2/D3)
    method='dualhead':  Backbone + PAN + HeatmapDecoder(1ch, offset) (D4/D5/D6)
    """
    def __init__(self, method='heatmap', dropout=0.4, grad_alpha=1.0):
        super().__init__()
        self.method = method
        self.backbone = YOLOv8Backbone(in_ch=1)
        self.pan = PAN()
        if method == 'bbox':
            self.head = DetectHead()
        elif method == 'heatmap_multi':
            self.decoder = HeatmapDecoder(out_ch=MAX_SRC, dropout=dropout)
        elif method == 'dualhead':
            self.decoder = HeatmapDecoder(out_ch=1, with_offset=True, dropout=dropout, grad_alpha=grad_alpha)
        else:
            self.decoder = HeatmapDecoder(out_ch=1, dropout=dropout)

    def _pad(self, x):
        _, _, h, w = x.shape
        ph = PAD_SIZE - h
        pw = PAD_SIZE - w
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))
        return x

    def forward(self, x):
        orig_h, orig_w = x.shape[2], x.shape[3]
        x = self._pad(x)
        e1, e2, p3, p4, p5 = self.backbone(x)
        n3, n4, n5 = self.pan(p3, p4, p5)
        if self.method == 'bbox':
            cls_list, reg_list = self.head([n3, n4, n5])
            return cls_list, reg_list
        elif self.method == 'dualhead':
            hm, offset = self.decoder(n5, n4, n3, e2, e1)
            hm = hm[:, :, :orig_h, :orig_w]
            offset = offset[:, :, :orig_h, :orig_w]
            return hm, offset
        else:
            hm = self.decoder(n5, n4, n3, e2, e1)
            hm = hm[:, :, :orig_h, :orig_w]
            return hm


# ═══════════════════════════════════════
#  Loss 函数
# ═══════════════════════════════════════
def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    inter_x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
    inter_y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
    inter_x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
    inter_y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    area1 = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(0) * (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(0)
    area2 = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(0) * (target_boxes[:, 3] - target_boxes[:, 1]).clamp(0)
    union = area1 + area2 - inter + eps
    iou = inter / union
    enc_x1 = torch.min(pred_boxes[:, 0], target_boxes[:, 0])
    enc_y1 = torch.min(pred_boxes[:, 1], target_boxes[:, 1])
    enc_x2 = torch.max(pred_boxes[:, 2], target_boxes[:, 2])
    enc_y2 = torch.max(pred_boxes[:, 3], target_boxes[:, 3])
    c2 = (enc_x2 - enc_x1)**2 + (enc_y2 - enc_y1)**2 + eps
    cx1 = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    cy1 = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    cx2 = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    cy2 = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    rho2 = (cx1 - cx2)**2 + (cy1 - cy2)**2
    w1 = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=eps)
    h1 = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=eps)
    w2 = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=eps)
    h2 = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=eps)
    v = (4 / math.pi**2) * (torch.atan(w2/h2) - torch.atan(w1/h1))**2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return 1 - iou + rho2/c2 + alpha*v


def dfl_loss(pred_dist, target_dist, reg_max=REG_MAX):
    target_dist = target_dist.clamp(0, reg_max - 1 - 0.01)
    tl = target_dist.long()
    tr = tl + 1
    wl = tr.float() - target_dist
    wr = 1.0 - wl
    log_p = F.log_softmax(pred_dist, dim=-1)
    loss_l = -log_p.gather(-1, tl.unsqueeze(-1)).squeeze(-1) * wl
    loss_r = -log_p.gather(-1, tr.unsqueeze(-1)).squeeze(-1) * wr
    return (loss_l + loss_r).mean()


def focal_loss_hm(pred, target, alpha=2, beta=4):
    pred = pred.float()
    target = target.float()
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = -(target * ((1 - p)**alpha) * torch.log(p))
    neg = -(((1 - target)**beta) * (p**alpha) * torch.log(1 - p))
    n_pos = target.gt(0.5).float().sum()
    if n_pos > 0:
        return (pos.sum() + neg.sum()) / n_pos
    else:
        return neg.sum() / max(target.numel(), 1)


def focal_loss_hm_hnm(pred, target, alpha=2, beta=4, neg_ratio=3):
    pred = pred.float()
    target = target.float()
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos_loss = -(target * ((1 - p)**alpha) * torch.log(p))
    neg_loss = -(((1 - target)**beta) * (p**alpha) * torch.log(1 - p))
    pixel_loss = pos_loss + neg_loss
    pos_mask = target.gt(0.5)
    n_pos = max(pos_mask.float().sum().item(), 1)
    loss_pos = pixel_loss[pos_mask].sum()
    neg_pixel_loss = pixel_loss[~pos_mask]
    n_neg = min(int(n_pos * neg_ratio), neg_pixel_loss.numel())
    neg_topk, _ = neg_pixel_loss.topk(n_neg)
    loss_neg = neg_topk.sum()
    return (loss_pos + loss_neg) / n_pos


def weighted_focal_loss_hm(pred, target, weight, alpha=2, beta=4):
    pred = pred.float()
    target = target.float()
    weight = weight.float()
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    pos = -(target * ((1 - p)**alpha) * torch.log(p))
    neg = -(((1 - target)**beta) * (p**alpha) * torch.log(1 - p))
    loss_map = (pos + neg) * weight
    n_pos = target.gt(0.5).float().sum()
    if n_pos > 0:
        return loss_map.sum() / n_pos
    else:
        return loss_map.sum() / max(target.numel(), 1)


def dice_loss(pred, target):
    pred = pred.float()
    target = target.float()
    p = torch.sigmoid(pred).clamp(1e-4, 1 - 1e-4)
    C = pred.shape[1]
    loss = 0.0
    for c in range(C):
        pc = p[:, c]
        tc = target[:, c]
        inter = (pc * tc).sum()
        loss += 1.0 - (2.0 * inter + 1e-6) / (pc.sum() + tc.sum() + 1e-6)
    return loss / C


# ═══════════════════════════════════════
#  NMS / 检测结果提取
# ═══════════════════════════════════════
def nms_heatmap(heatmap, kernel_size=PEAK_SIZE):
    pad = kernel_size // 2
    hmax = F.max_pool2d(heatmap, kernel_size, stride=1, padding=pad)
    return heatmap * (heatmap == hmax).float()


def extract_peaks_topn(heatmap, n_det, peak_size=PEAK_SIZE):
    if heatmap.dim() == 3:
        heatmap = heatmap.squeeze(0)
    hm_sig = torch.sigmoid(heatmap)
    hm_nms = nms_heatmap(hm_sig.unsqueeze(0).unsqueeze(0), peak_size).squeeze()
    k = min(n_det, hm_nms.numel())
    scores, indices = hm_nms.view(-1).topk(k)
    xi = (indices % hm_nms.shape[1]).float()
    yi = (indices // hm_nms.shape[1]).float()
    return torch.stack([xi, yi], 1), scores


def decode_bbox_topn(cls_list, reg_list, n_det, head, peak_size=PEAK_SIZE):
    pred_boxes, pred_scores = head.decode(cls_list, reg_list)
    scores = pred_scores[0, :, 0].sigmoid()
    boxes = pred_boxes[0]
    nms_dist = peak_size // 2 + 1
    order = scores.argsort(descending=True)
    keep = []
    for i in range(len(order)):
        idx = order[i]
        if scores[idx] < 1e-6:
            break
        cx_i = (boxes[idx, 0] + boxes[idx, 2]) / 2
        cy_i = (boxes[idx, 1] + boxes[idx, 3]) / 2
        too_close = False
        for kept_idx in keep:
            cx_k = (boxes[kept_idx, 0] + boxes[kept_idx, 2]) / 2
            cy_k = (boxes[kept_idx, 1] + boxes[kept_idx, 3]) / 2
            dist = torch.sqrt((cx_i - cx_k)**2 + (cy_i - cy_k)**2)
            if dist < nms_dist:
                too_close = True
                break
        if not too_close:
            keep.append(idx.item())
            if len(keep) >= n_det:
                break
    if len(keep) == 0:
        return torch.zeros(0, 2), torch.zeros(0)
    keep_t = torch.tensor(keep, device=boxes.device)
    kept_boxes = boxes[keep_t]
    kept_scores = scores[keep_t]
    cx = (kept_boxes[:, 0] + kept_boxes[:, 2]) / 2
    cy = (kept_boxes[:, 1] + kept_boxes[:, 3]) / 2
    return torch.stack([cx, cy], 1), kept_scores


def pixel_to_phys(px):
    return px * LAMDA - EDGE


def soft_argmax(heatmap, temp=SOFTARGMAX_TEMP):
    B, C, H, W = heatmap.shape
    y_grid = torch.arange(H, device=heatmap.device, dtype=heatmap.dtype)
    x_grid = torch.arange(W, device=heatmap.device, dtype=heatmap.dtype)
    y_grid = y_grid.view(1, 1, H, 1).expand(B, C, H, W)
    x_grid = x_grid.view(1, 1, 1, W).expand(B, C, H, W)
    hm_flat = heatmap.reshape(B, C, -1)
    weights = F.softmax(hm_flat / temp, dim=-1)
    weights = weights.reshape(B, C, H, W)
    x = (weights * x_grid).sum(dim=(-2, -1))
    y = (weights * y_grid).sum(dim=(-2, -1))
    return torch.stack([x, y], dim=-1)


def extract_coords_multi(heatmap, n_src, peak_size=PEAK_SIZE):
    hm_sig = torch.sigmoid(heatmap)
    C, H, W = hm_sig.shape
    coords_px = torch.zeros(C, 2, device=heatmap.device)
    confidence = torch.zeros(C, device=heatmap.device)
    for c in range(C):
        hm_c = hm_sig[c:c+1].unsqueeze(0)
        hm_nms = nms_heatmap(hm_c, peak_size).squeeze()
        peak_val, peak_idx = hm_nms.view(-1).max(0)
        xi = (peak_idx % W).float()
        yi = (peak_idx // W).float()
        coords_px[c] = torch.stack([xi, yi])
        confidence[c] = peak_val
    if n_src < C:
        topk_conf, topk_idx = confidence.topk(n_src)
        coords_px = coords_px[topk_idx]
        confidence = topk_conf
    else:
        coords_px = coords_px[:n_src]
        confidence = confidence[:n_src]
    coords_phys = pixel_to_phys(coords_px)
    return coords_phys, confidence


def extract_coords_multi_batch(pred_hm, peak_size=PEAK_SIZE):
    B, C, H, W = pred_hm.shape
    hm_sig = torch.sigmoid(pred_hm)
    all_coords_phys = torch.zeros(B, C, 2, device=pred_hm.device)
    all_conf = torch.zeros(B, C, device=pred_hm.device)
    for c in range(C):
        hm_c = hm_sig[:, c:c+1, :, :]
        hm_nms = nms_heatmap(hm_c, peak_size)
        hm_flat = hm_nms[:, 0].reshape(B, -1)
        peak_vals, peak_indices = hm_flat.max(dim=1)
        xi = (peak_indices % W).float()
        yi = (peak_indices // W).float()
        all_coords_phys[:, c, 0] = xi * LAMDA - EDGE
        all_coords_phys[:, c, 1] = yi * LAMDA - EDGE
        all_conf[:, c] = peak_vals
    return all_coords_phys, all_conf


def hungarian_match(pred_coords, true_coords, n_src):
    from itertools import permutations
    C = pred_coords.shape[0]
    N = n_src
    cost = torch.zeros(N, C, device=pred_coords.device)
    for i in range(N):
        for j in range(C):
            cost[i, j] = torch.sqrt(((pred_coords[j] - true_coords[i]) ** 2).sum())
    best_cost = float('inf')
    best_perm = list(range(N))
    for perm in permutations(range(C)):
        c = sum(cost[i, perm[i]].item() for i in range(N))
        if c < best_cost:
            best_cost = c
            best_perm = list(perm[:N])
    return torch.tensor(best_perm, device=pred_coords.device)