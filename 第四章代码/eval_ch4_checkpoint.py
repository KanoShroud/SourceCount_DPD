"""独立评估第四章 D8 checkpoint；用于 Gate 3B 的 validation/test 隔离。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for import_root in (PROJECT_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from chapter_runtime import device as runtime_device  # noqa: E402
from train_yolo import (  # noqa: E402
    LocDataset,
    collate_fn_hm,
    configure_reproducibility,
    evaluate,
    write_json,
)
from yolo_config import EDGE, MAX_SRC, PEAK_SIZE  # noqa: E402
from yolo_model import YOLOv8Loc, nms_heatmap, pixel_to_phys  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_gate3b_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get("method") != "dualhead" or checkpoint.get("save_tag") != "dualhead_std":
        raise ValueError("checkpoint 不是 Gate 3B D8 dualhead_std")
    saved = checkpoint.get("args")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint 缺少训练参数 args")
    expected = {
        "method": "dualhead",
        "amp": False,
        "dice_weight": 0.0,
        "grad_alpha": 1.0,
        "offset_weight": 1.0,
        "conf_weight_offset": False,
        "soft_conf": False,
        "batch_size": 8,
        "val_batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 5e-3,
        "dropout": 0.4,
        "eval_every": 1,
        "epochs": 60,
        "patience": 60,
        "seed": 42,
        "deterministic": True,
        "fail_on_nonfinite": True,
        "save_last_every_epoch": True,
        "require_empty_output": True,
        "gate3b_d8": True,
    }
    mismatches = [
        f"{key}={saved.get(key)!r}, expected={value!r}"
        for key, value in expected.items() if saved.get(key) != value
    ]
    if mismatches:
        raise ValueError("checkpoint Gate 3B 冻结配置不一致: " + "; ".join(mismatches))
    if not isinstance(checkpoint.get("epoch"), int) or not 6 <= checkpoint["epoch"] <= 60:
        raise ValueError(f"checkpoint epoch 非法: {checkpoint.get('epoch')!r}")
    best_rmse = checkpoint.get("best_rmse")
    if not isinstance(best_rmse, (int, float)) or not np.isfinite(best_rmse):
        raise ValueError(f"checkpoint best_rmse 非有限: {best_rmse!r}")


def require_s2g2_checkpoint(checkpoint: dict) -> None:
    if checkpoint.get("method") != "dualhead" or checkpoint.get("save_tag") != "dualhead_std":
        raise ValueError("checkpoint 不是 S2-G2 D8 dualhead_std")
    saved = checkpoint.get("args")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint 缺少训练参数 args")
    expected = {
        "method": "dualhead",
        "amp": False,
        "dice_weight": 0.0,
        "grad_alpha": 1.0,
        "offset_weight": 1.0,
        "conf_weight_offset": False,
        "soft_conf": False,
        "batch_size": 8,
        "val_batch_size": 8,
        "lr": 1e-3,
        "weight_decay": 5e-3,
        "dropout": 0.4,
        "eval_every": 1,
        "epochs": 200,
        "patience": 200,
        "seed": 42,
        "deterministic": True,
        "fail_on_nonfinite": True,
        "save_last_every_epoch": True,
        "require_empty_output": True,
        "gate3_d8": False,
        "gate3b_d8": False,
        "s2g2_d8": True,
    }
    mismatches = [
        f"{key}={saved.get(key)!r}, expected={value!r}"
        for key, value in expected.items() if saved.get(key) != value
    ]
    if mismatches:
        raise ValueError("checkpoint S2-G2 冻结配置不一致: " + "; ".join(mismatches))
    if not isinstance(checkpoint.get("epoch"), int) or not 6 <= checkpoint["epoch"] <= 200:
        raise ValueError(f"checkpoint epoch 非法: {checkpoint.get('epoch')!r}")
    best_rmse = checkpoint.get("best_rmse")
    if not isinstance(best_rmse, (int, float)) or not np.isfinite(best_rmse):
        raise ValueError(f"checkpoint best_rmse 非有限: {best_rmse!r}")


def summarize_errors(records: list[dict], source_count: int | None = None) -> dict:
    selected = [
        record for record in records
        if source_count is None or record["source_count"] == source_count
    ]
    if not selected:
        raise ValueError(f"没有可汇总的逐源误差: source_count={source_count}")
    errors = np.asarray([record["error_m"] for record in selected], dtype=np.float64)
    return {
        "source_error_count": int(errors.size),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mean_m": float(np.mean(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "max_m": float(np.max(errors)),
        "above_50m_count": int(np.sum(errors > 50.0)),
        "above_100m_count": int(np.sum(errors > 100.0)),
        "above_500m_count": int(np.sum(errors > 500.0)),
    }


@torch.no_grad()
def collect_detailed_errors(model, loader, device) -> dict:
    model.eval()
    source_records: list[dict] = []
    sample_records: list[dict] = []
    sample_offset = 0

    for batch in loader:
        dpd, _, pos, n_src = batch
        dpd = dpd.to(device)
        pred_hm, pred_offset = model(dpd)
        if not bool(torch.isfinite(pred_hm).all() and torch.isfinite(pred_offset).all()):
            raise FloatingPointError("详细评估阶段 heatmap/offset 输出含 NaN/Inf")

        hm_nms = nms_heatmap(torch.sigmoid(pred_hm), PEAK_SIZE)
        batch_size, _, height, width = hm_nms.shape
        flat = hm_nms[:, 0].reshape(batch_size, -1)
        topk_k = min(MAX_SRC, flat.shape[1])
        topk_scores, topk_indices = flat.topk(topk_k, dim=1)
        topk_x = (topk_indices % width).float()
        topk_y = (topk_indices // width).float()
        for batch_index in range(batch_size):
            for rank in range(topk_k):
                ix = int(topk_x[batch_index, rank].item())
                iy = int(topk_y[batch_index, rank].item())
                dx = pred_offset[batch_index, 0, iy, ix].float()
                dy = pred_offset[batch_index, 1, iy, ix].float()
                if torch.isfinite(dx) and torch.isfinite(dy):
                    topk_x[batch_index, rank] += dx.clamp(-1, 1)
                    topk_y[batch_index, rank] += dy.clamp(-1, 1)
        pred_phys_all = pixel_to_phys(
            torch.stack([topk_x, topk_y], dim=-1)
        ).cpu().numpy()
        score_all = topk_scores.cpu().numpy()

        for batch_index in range(batch_size):
            sample_index = sample_offset + batch_index
            count = int(n_src[batch_index].item())
            if count <= 0:
                continue
            true_phys = pos[batch_index, :count].numpy() * EDGE
            pred_phys = pred_phys_all[batch_index, :count]
            cost = np.linalg.norm(
                true_phys[:, None, :] - pred_phys[None, :, :], axis=2
            )
            true_indices, pred_indices = linear_sum_assignment(cost)
            sample_sources = []
            for true_index, pred_index in zip(true_indices, pred_indices, strict=True):
                record = {
                    "sample_index": int(sample_index),
                    "source_count": count,
                    "true_source_index": int(true_index),
                    "predicted_rank": int(pred_index) + 1,
                    "true_x_m": float(true_phys[true_index, 0]),
                    "true_y_m": float(true_phys[true_index, 1]),
                    "pred_x_m": float(pred_phys[pred_index, 0]),
                    "pred_y_m": float(pred_phys[pred_index, 1]),
                    "peak_score": float(score_all[batch_index, pred_index]),
                    "error_m": float(cost[true_index, pred_index]),
                }
                source_records.append(record)
                sample_sources.append(record)
            sample_errors = np.asarray(
                [record["error_m"] for record in sample_sources], dtype=np.float64
            )
            sample_records.append({
                "sample_index": int(sample_index),
                "source_count": count,
                "rmse_m": float(np.sqrt(np.mean(sample_errors**2))),
                "mean_m": float(np.mean(sample_errors)),
                "max_m": float(np.max(sample_errors)),
                "sources": sample_sources,
            })
        sample_offset += batch_size

    if not source_records:
        raise RuntimeError("详细评估没有产生逐源误差")
    return {
        "statistics": {
            "overall": summarize_errors(source_records),
            "N2": summarize_errors(source_records, 2),
            "N3": summarize_errors(source_records, 3),
        },
        "worst_sources": sorted(
            source_records, key=lambda record: record["error_m"], reverse=True
        )[:10],
        "worst_samples": sorted(
            sample_records, key=lambda record: record["max_m"], reverse=True
        )[:10],
        "per_source_errors": source_records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第四章 D8 独立 split 评估")
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--gate3_d8", action="store_true", default=False)
    parser.add_argument("--gate3b_d8", action="store_true", default=False)
    parser.add_argument("--s2g2_d8", action="store_true", default=False)
    parser.add_argument("--include_error_details", action="store_true", default=False)
    parser.add_argument("--expected_checkpoint_sha256")
    parser.add_argument("--require_absent_output", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch_size 必须为正整数")
    if args.gate3_d8 and (args.batch_size != 8 or not args.deterministic):
        parser.error("Gate 3 D8 独立评估固定 batch=8 且 deterministic=True")
    strict_modes = [args.gate3_d8, args.gate3b_d8, args.s2g2_d8]
    if sum(bool(mode) for mode in strict_modes) > 1:
        parser.error("--gate3_d8、--gate3b_d8 与 --s2g2_d8 不得同时使用")
    if args.gate3b_d8 or args.s2g2_d8:
        if args.batch_size != 8 or not args.deterministic or args.seed != 42:
            parser.error("Gate 3B/S2-G2 D8 独立评估固定 batch=8、seed=42、deterministic=True")
        if not args.expected_checkpoint_sha256:
            parser.error("Gate 3B/S2-G2 必须提供 --expected_checkpoint_sha256")
        if not args.require_absent_output:
            parser.error("Gate 3B/S2-G2 必须提供 --require_absent_output")
    if args.include_error_details and not (args.gate3b_d8 or args.s2g2_d8):
        parser.error("--include_error_details 仅用于 Gate 3B 或 S2-G2 严格评估")
    return args


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if args.require_absent_output and output.exists():
        raise FileExistsError(f"拒绝覆盖已有评估结果: {output}")
    configure_reproducibility(args.seed, args.deterministic)
    device = runtime_device(args.device)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if (args.expected_checkpoint_sha256 is not None
            and checkpoint_sha256.lower() != args.expected_checkpoint_sha256.lower()):
        raise ValueError(
            f"checkpoint SHA256 不一致: {checkpoint_sha256} != {args.expected_checkpoint_sha256}"
        )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model")
    if not isinstance(state, dict):
        state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise KeyError("checkpoint 缺少 model/model_state state_dict")
    if args.gate3b_d8:
        require_gate3b_checkpoint(checkpoint)
    elif args.s2g2_d8:
        require_s2g2_checkpoint(checkpoint)
    elif args.gate3_d8:
        if checkpoint.get("method") != "dualhead" or checkpoint.get("save_tag") != "dualhead_std":
            raise ValueError("checkpoint 不是 Gate 3 D8 dualhead_std")
    nonfinite_names = [
        name for name, value in state.items()
        if isinstance(value, torch.Tensor) and not bool(torch.isfinite(value).all())
    ]
    if nonfinite_names:
        raise FloatingPointError(f"checkpoint 权重含 NaN/Inf: {nonfinite_names[0]}")

    dataset = LocDataset(str(args.data_dir.resolve()), args.split, method="dualhead", augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn_hm,
    )
    model = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    started = time.perf_counter()
    metrics = evaluate(
        model,
        loader,
        device,
        "dualhead",
        full_metrics=True,
        amp=False,
        offset_weight=1.0,
        conf_weight_offset=False,
        soft_conf=False,
        dice_weight=0.0,
    )
    payload = {
        "status": "PASS",
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_best_rmse": checkpoint.get("best_rmse"),
        "model": "D8",
        "amp": False,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "evaluation_mode": "oracle-K_ground_truth_source_count",
        "task_count": len(dataset),
        "source_count_distribution": {
            str(key): int(value) for key, value in sorted(Counter(dataset.n.tolist()).items())
        },
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "performance_interpretation_allowed": False,
    }
    if args.include_error_details:
        payload["error_details"] = collect_detailed_errors(model, loader, device)
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.item()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
