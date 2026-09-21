"""G5 A2同前向SG/E2E、冻结范围和有界评价。"""
from __future__ import annotations

import io
import itertools
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from 统一模型代码.gates.g5.e2e_g5_features import build_models, g4
from 统一模型代码.common.g5_verified_io import RetryVerifiedCache, verified_read
from 统一模型代码.common.verified_feature_loader import VerifiedShardedArray


def build_context(run: Path, manifest: dict, seed: int, device):
    cfg = manifest["config"]
    g4.set_deterministic(seed)
    ch3, d8 = build_models(run, manifest, device)
    for p in itertools.chain(ch3.parameters(), d8.parameters()):
        p.requires_grad_(False)
    query = g4.SourceQueryBuilder().to(device)
    splitter = g4.FrequencySpatialSplitter().to(device)
    head = g4.SourceLocalizationHead().to(device)
    nn.init.constant_(head.heatmap.bias, -2.19)
    nn.init.zeros_(head.offset.weight)
    nn.init.zeros_(head.offset.bias)
    endpoint = list(itertools.chain(ch3.band_heads[:3].parameters(), query.parameters(),
                                    splitter.parameters(), head.parameters()))
    tail = list(itertools.chain(d8.decoder.c1.parameters(), d8.decoder.up0.parameters()))
    ch3_tail = list(ch3.cross_attn.parameters())
    groups = [{"params": endpoint, "lr": cfg["endpoint_learning_rate"], "name": "endpoint"},
              {"params": tail, "lr": cfg["d8_tail_learning_rate"], "name": "d8_tail"},
              {"params": ch3_tail, "lr": cfg["ch3_tail_learning_rate"], "name": "ch3_tail"}]
    parameters = endpoint + tail + ch3_tail
    for p in parameters:
        p.requires_grad_(True)
    context = g4.Context(ch3, d8, query, splitter, head, "a2_joint_tail", groups, parameters)
    g4.set_mode(context, training=True)
    return context


def forward(context, features, indices, device, *, stop_gradient):
    spatial = g4.numpy_batch(features.spatial, indices, device)
    pooled = spatial.mean(dim=(-1, -2))
    tokens = context.ch3.cross_attn(pooled + context.ch3.pos_embed)
    global_feature = context.ch3.global_encoder(tokens.mean(dim=1))
    cached = g4.CH3Features(spatial, tokens, global_feature, torch.empty(0, device=device), torch.empty(0, device=device))
    current = g4.g2.ch3_from_cached(context.ch3, cached)
    query, logits = context.query_builder(current)
    loc_query = query.detach() if stop_gradient else query
    loc_logits = logits.detach() if stop_gradient else logits
    loc_spatial = current.spatial.detach() if stop_gradient else current.spatial
    source_spatial, attention = context.splitter(loc_spatial, loc_query, loc_logits)
    e1 = g4.numpy_batch(features.e1, indices, device)
    d2 = g4.numpy_batch(features.d2, indices, device)
    with torch.no_grad():
        up1 = context.d8.decoder.up1(d2)
    d1 = context.d8.decoder.c1(torch.cat([up1, e1], 1))
    d0 = context.d8.decoder.up0(d1)[:, :, :401, :401]
    heatmap, offset = context.source_head(d0, source_spatial, loc_query)
    return query, logits, attention, heatmap, offset


def load_split(run, manifest, features_manifest, split, cache=None):
    rows = features_manifest["files"][split]
    from 统一模型代码.common.g5_runtime_v2 import enabled, load_features
    if enabled(run):
        features, cache = load_features(run, split, manifest['config']['feature_cache_bytes'], cache)
    else:
        if cache is None:
            cache = RetryVerifiedCache(manifest["config"]["feature_cache_bytes"], run / "anomalies")
        features = g4.FeatureStore(*(VerifiedShardedArray(rows[k], cache) for k in ("ch3_spatial", "d8_e1", "d8_d2")))
    payload = verified_read(rows["targets"], run / "anomalies")
    targets = g4.Targets(**torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False))
    return features, targets, cache


def band_values(logits, targets, index, mapping):
    f1, iou = [], []
    for query, source in mapping.items():
        valid = targets.ignore[index, source] < .5
        pred, truth = logits[query, valid] >= 0, targets.band[index, source, valid] > .5
        tp = int((pred & truth).sum())
        fp = int((pred & ~truth).sum())
        fn = int((~pred & truth).sum())
        f1.append(2*tp / max(2*tp+fp+fn, 1))
        iou.append(tp / max(tp+fp+fn, 1))
    return f1, iou


def summarize(rows):
    result = g4.summarize_rows(rows)
    for name in ("band_f1", "band_iou", "band_only_f1", "band_only_iou"):
        values = [v for row in rows for v in row[name]]
        result[name] = float(np.mean(values)) if values else None
    confusion = [[0]*4 for _ in range(4)]
    for row in rows:
        confusion[row["true_count"]][row["predicted_count"]] += 1
    result["count_confusion"] = confusion
    accuracies = [confusion[k][k]/sum(confusion[k]) for k in range(4) if sum(confusion[k])]
    result["balanced_count_accuracy"] = float(np.mean(accuracies)) if accuracies else None
    result['k0_false_alarm_rate'] = 1-confusion[0][0]/sum(confusion[0]) if sum(confusion[0]) else None
    return result


@torch.no_grad()
def evaluate(context, features, targets, device, metadata):
    g4.set_mode(context, training=False)
    rows = []
    for start in range(0, len(targets.counts), 4):
        indices = torch.arange(start, min(start+4, len(targets.counts)))
        _, logits, _, heatmap, offset = forward(context, features, indices, device, stop_gradient=False)
        logits, heatmap, offset = logits.cpu(), heatmap.cpu(), offset.cpu()
        mappings = g4.g2.assignments(logits, heatmap, targets.band[indices], targets.ignore[indices],
                                    targets.positions[indices], targets.counts[indices])
        for local, index in enumerate(indices.tolist()):
            count = int(targets.counts[index])
            active = torch.nonzero(logits[local].amax(-1) >= 0, as_tuple=False).flatten().tolist()
            predictions = []
            for query in active:
                peak = int(torch.argmax(torch.sigmoid(heatmap[local, query])))
                iy, ix = divmod(peak, 401)
                delta = offset[local, query, :, iy, ix].clamp(-1, 1)
                predictions.append([(ix+float(delta[0]))*10-2000, (iy+float(delta[1]))*10-2000])
            predicted = np.asarray(predictions, dtype=np.float32).reshape(-1, 2)
            truth = targets.positions[index, :count].numpy()
            gospa = g4.g1.gospa_sample(truth, predicted)
            row = {**metadata[index], "index": index, "true_count": count, "predicted_count": len(active),
                   "predicted_positions_m": predicted.tolist(), "true_positions_m": truth.tolist(),
                   "band_logits": logits[local].tolist(), "gospa_m": float(gospa["value_m"]),
                   "matched_errors_m": g4.distance_errors(truth, predicted)}
            for name in ("localization", "missed", "false"):
                row[f"gospa_{name}_p_sum"] = float(gospa[f"{name}_p_sum"])
            for threshold in (10, 30, 50, 100):
                row[f"tp_at_{threshold}m"] = g4.g1.maximum_matches_within(truth, predicted, threshold)
            row["band_f1"], row["band_iou"] = band_values(logits[local], targets, index, mappings[local])
            mapping = {}
            if count:
                cost = np.empty((3, count))
                for q in range(3):
                    for source in range(count):
                        valid = targets.ignore[index, source] < .5
                        cost[q, source] = float(torch.nn.functional.binary_cross_entropy_with_logits(
                            logits[local, q, valid], targets.band[index, source, valid]))
                left, right = linear_sum_assignment(cost)
                mapping = dict(zip(left.tolist(), right.tolist()))
            row["band_only_f1"], row["band_only_iou"] = band_values(logits[local], targets, index, mapping)
            rows.append(row)
    g4.set_mode(context, training=True)
    strata = {"by_k": {str(k): [r for r in rows if r["true_count"] == k] for k in range(4)},
              "by_snr": {"below_minus10": [r for r in rows if r["true_count"] > 0 and r["snr_db"] < -10],
                         "minus10_minus5": [r for r in rows if r["true_count"] > 0 and -10 <= r["snr_db"] < -5],
                         "minus5_0": [r for r in rows if r["true_count"] > 0 and -5 <= r["snr_db"] < 0],
                         "0_5": [r for r in rows if r["true_count"] > 0 and 0 <= r["snr_db"] < 5],
                         "5_10": [r for r in rows if r["true_count"] > 0 and 5 <= r["snr_db"] <= 10],
                         "above_10": [r for r in rows if r["true_count"] > 0 and r["snr_db"] > 10]},
              "by_overlap": {str(x): [r for r in rows if r["frequency_overlap"] == x] for x in (False, True)},
              "by_distance": {"lt500": [r for r in rows if r["min_source_distance_m"] is not None and r["min_source_distance_m"] < 500],
                              "500to1000": [r for r in rows if r["min_source_distance_m"] is not None and 500 <= r["min_source_distance_m"] < 1000],
                              "ge1000": [r for r in rows if r["min_source_distance_m"] is not None and r["min_source_distance_m"] >= 1000]}}
    return {"overall": summarize(rows), "samples": rows,
            **{name: {key: summarize(group) for key,group in groups.items() if group} for name,groups in strata.items()}}
