"""E2E-G2 的旧模型张量接口与逐源潜在融合原型。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class CH3Features:
    spatial: torch.Tensor
    tokens: torch.Tensor
    global_feature: torch.Tensor
    head_hidden: torch.Tensor
    band_logits: torch.Tensor


@dataclass
class D8Features:
    d0: torch.Tensor
    heatmap: torch.Tensor
    offset: torch.Tensor


def forward_ch3_features(model: nn.Module, x: torch.Tensor) -> CH3Features:
    """展开 CH3 原 forward，并保留池化前空间特征及 head 隐层。"""
    batch, bands, height, width = x.shape
    spatial = model.backbone[:-1](x.reshape(batch * bands, 1, height, width))
    _, channels, out_h, out_w = spatial.shape
    spatial = spatial.reshape(batch, bands, channels, out_h, out_w)
    pooled = F.adaptive_avg_pool2d(
        spatial.reshape(batch * bands, channels, out_h, out_w), 1
    ).reshape(batch, bands, channels)
    if model.mode == "transformer":
        tokens = model.cross_attn(pooled + model.pos_embed)
        global_feature = model.global_encoder(tokens.mean(dim=1))
    else:
        tokens = pooled
        global_feature = model.global_encoder(tokens.reshape(batch, -1))
    hidden = torch.stack(
        [torch.relu(head[0](global_feature)) for head in model.band_heads], dim=1
    )
    logits = torch.stack(
        [head[2](hidden[:, index]) for index, head in enumerate(model.band_heads)],
        dim=1,
    )
    return CH3Features(spatial, tokens, global_feature, hidden, logits)


def forward_d8_features(model: nn.Module, x: torch.Tensor) -> D8Features:
    """展开 D8 原 forward，额外返回 decoder 的 d0 特征。"""
    original_h, original_w = x.shape[-2:]
    padded = model._pad(x)
    e1, e2, p3, p4, p5 = model.backbone(padded)
    n3, n4, n5 = model.pan(p3, p4, p5)
    decoder = model.decoder
    d4 = decoder.drop(decoder.c4(torch.cat([decoder.up4(n5), n4], 1)))
    d3 = decoder.drop(decoder.c3(torch.cat([decoder.up3(d4), n3], 1)))
    d2 = decoder.drop(decoder.c2(torch.cat([decoder.up2(d3), e2], 1)))
    d1 = decoder.c1(torch.cat([decoder.up1(d2), e1], 1))
    d0 = decoder.up0(d1)
    heatmap = decoder.head(d0)
    offset = decoder.offset_head(d0)
    return D8Features(
        d0[:, :, :original_h, :original_w],
        heatmap[:, :, :original_h, :original_w],
        offset[:, :, :original_h, :original_w],
    )


class SourceQueryBuilder(nn.Module):
    """由前三个 CH3 head 锚点和频带 token 构造三个 source query。"""

    def __init__(self, query_dim: int = 128, query_count: int = 3):
        super().__init__()
        self.query_count = query_count
        self.anchor = nn.Linear(64 + 256, query_dim)
        self.cross_attention = nn.MultiheadAttention(
            query_dim, num_heads=4, batch_first=True
        )
        self.band_residual = nn.Linear(query_dim, 19)
        nn.init.zeros_(self.band_residual.weight)
        nn.init.zeros_(self.band_residual.bias)

    def forward(self, features: CH3Features) -> tuple[torch.Tensor, torch.Tensor]:
        anchors = torch.cat(
            [
                features.head_hidden[:, : self.query_count],
                features.global_feature[:, None].expand(-1, self.query_count, -1),
            ],
            dim=-1,
        )
        query = self.anchor(anchors)
        query, _ = self.cross_attention(query, features.tokens, features.tokens)
        logits = features.band_logits[:, : self.query_count] + self.band_residual(query)
        return query, logits


class FrequencySpatialSplitter(nn.Module):
    """用每个 query 的 Soft 频带概率汇聚粗 DPD 空间特征。"""

    def __init__(self, query_dim: int = 128, out_channels: int = 32):
        super().__init__()
        self.spatial_projection = nn.Conv2d(128, out_channels, 1)
        self.query_projection = nn.Linear(query_dim, 128)

    def forward(
        self,
        spatial: torch.Tensor,
        query: torch.Tensor,
        band_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        probabilities = torch.sigmoid(band_logits)
        weighted = torch.einsum("bqs,bschw->bqchw", probabilities, spatial)
        weighted = weighted / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)[
            ..., None, None
        ]
        attention_logits = torch.einsum(
            "bqc,bqchw->bqhw", self.query_projection(query), weighted
        ) / (weighted.shape[2] ** 0.5)
        attention = torch.softmax(attention_logits.flatten(-2), dim=-1).reshape_as(
            attention_logits
        )
        batch, count, channels, height, width = weighted.shape
        projected = self.spatial_projection(
            weighted.reshape(batch * count, channels, height, width)
        ).reshape(batch, count, -1, height, width)
        return projected * (1.0 + attention[:, :, None]), attention


class SourceLocalizationHead(nn.Module):
    """共享逐源定位头；每个 query 输出一个 Heatmap 和二维 offset。"""

    def __init__(self, query_dim: int = 128):
        super().__init__()
        self.film = nn.Linear(query_dim, 64)
        self.fusion = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU()
        )
        self.heatmap = nn.Conv2d(32, 1, 1)
        self.offset = nn.Conv2d(32, 2, 1)

    def forward(
        self, d0: torch.Tensor, source_spatial: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, count = source_spatial.shape[:2]
        spatial = F.interpolate(
            source_spatial.reshape(batch * count, *source_spatial.shape[2:]),
            size=d0.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, count, 32, *d0.shape[-2:])
        shared = d0[:, None].expand(-1, count, -1, -1, -1)
        gamma, beta = self.film(query).chunk(2, dim=-1)
        shared = shared * (1.0 + gamma[..., None, None]) + beta[..., None, None]
        fused = self.fusion(torch.cat([shared, spatial], dim=2).reshape(
            batch * count, 64, *d0.shape[-2:]
        ))
        heatmap = self.heatmap(fused).reshape(batch, count, *d0.shape[-2:])
        offset = self.offset(fused).reshape(batch, count, 2, *d0.shape[-2:])
        return heatmap, offset
