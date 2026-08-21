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
from yolo_model import YOLOv8Loc  # noqa: E402


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
    parser.add_argument("--expected_checkpoint_sha256")
    parser.add_argument("--require_absent_output", action="store_true", default=False)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch_size 必须为正整数")
    if args.gate3_d8 and (args.batch_size != 8 or not args.deterministic):
        parser.error("Gate 3 D8 独立评估固定 batch=8 且 deterministic=True")
    if args.gate3b_d8:
        if args.gate3_d8:
            parser.error("--gate3b_d8 与 --gate3_d8 不得同时使用")
        if args.batch_size != 8 or not args.deterministic or args.seed != 42:
            parser.error("Gate 3B D8 独立评估固定 batch=8、seed=42、deterministic=True")
        if not args.expected_checkpoint_sha256:
            parser.error("Gate 3B 必须提供 --expected_checkpoint_sha256")
        if not args.require_absent_output:
            parser.error("Gate 3B 必须提供 --require_absent_output")
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
    write_json(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=lambda value: value.item()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
