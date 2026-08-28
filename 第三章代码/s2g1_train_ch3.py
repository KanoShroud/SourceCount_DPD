"""S2-G1 第三章受控训练、容量检查与独立评估入口。

本文件复用 ``train_v26.py`` 的 Dataset、模型和 loss，不修改师弟原训练入口。
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from chapter_runtime import device as runtime_device
from train_v26 import (
    BAND_THRESHOLD,
    SourceDetectionDataset,
    SourceDetectionNet,
    compute_loss,
)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    raise TypeError(f"无法 JSON 序列化 {type(value).__name__}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=json_default,
        )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def ensure_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空目录: {path}")
    path.mkdir(parents=True, exist_ok=True)


def configure_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def capture_rng_state(generator: torch.Generator | None = None) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if generator is not None:
        state["loader_generator"] = generator.get_state()
    return state


def restore_rng_state(state: dict[str, Any], generator: torch.Generator | None = None) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])
    if generator is not None and "loader_generator" in state:
        generator.set_state(state["loader_generator"])


def process_tree_rss_bytes() -> int:
    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return int(total)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: SourceDetectionDataset,
    batch_size: int,
    *,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
    )


def finite_tensor(tensor: torch.Tensor, name: str) -> None:
    require(torch.isfinite(tensor).all().item(), f"{name} 包含 NaN/Inf")


def reset_cuda_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device.index or 0)
        torch.cuda.reset_peak_memory_stats()


def binary_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


@torch.no_grad()
def evaluate_detailed(
    model: SourceDetectionNet,
    loader: DataLoader,
    device: torch.device,
    gamma: float,
    threshold: float,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    count_confusion = np.zeros((4, model.max_src + 1), dtype=np.int64)
    count_abs_error = 0
    under_count = 0
    over_count = 0
    active_totals = [0, 0, 0, 0]
    per_slot_totals = [[0, 0, 0, 0] for _ in range(model.max_src)]
    inactive_slots = 0
    inactive_activated = 0

    for x, src_count, band_mask, ignore_mask in loader:
        x = x.to(device, non_blocking=True)
        src_count_device = src_count.to(device, non_blocking=True)
        band_mask = band_mask.to(device, non_blocking=True)
        ignore_mask = ignore_mask.to(device, non_blocking=True)

        logits = model(x)
        finite_tensor(logits, "evaluation logits")
        loss = compute_loss(logits, band_mask, ignore_mask, gamma)
        finite_tensor(loss, "evaluation loss")
        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size

        prediction = torch.sigmoid(logits) > threshold
        slot_nonempty = prediction.any(dim=-1)
        count_prediction = slot_nonempty.sum(dim=-1).long()

        true_np = src_count.numpy().astype(np.int64)
        pred_np = count_prediction.cpu().numpy().astype(np.int64)
        for truth, pred in zip(true_np, pred_np, strict=True):
            require(0 <= truth <= 3, f"真实源数越界: {truth}")
            require(0 <= pred <= model.max_src, f"预测源数越界: {pred}")
            count_confusion[truth, pred] += 1
        delta = pred_np - true_np
        count_abs_error += int(np.abs(delta).sum())
        under_count += int((delta < 0).sum())
        over_count += int((delta > 0).sum())

        valid = ignore_mask == 0
        for slot in range(model.max_src):
            active_samples = src_count_device > slot
            inactive_samples = ~active_samples
            inactive_slots += int(inactive_samples.sum().item())
            inactive_activated += int(
                slot_nonempty[inactive_samples, slot].sum().item()
            )
            if not active_samples.any():
                continue
            current_valid = valid[active_samples, slot, :]
            current_pred = prediction[active_samples, slot, :][current_valid]
            current_true = band_mask[active_samples, slot, :][current_valid] > 0.5
            tp = int((current_pred & current_true).sum().item())
            fp = int((current_pred & ~current_true).sum().item())
            fn = int((~current_pred & current_true).sum().item())
            tn = int((~current_pred & ~current_true).sum().item())
            totals = [tp, fp, fn, tn]
            active_totals = [a + b for a, b in zip(active_totals, totals, strict=True)]
            per_slot_totals[slot] = [
                a + b for a, b in zip(per_slot_totals[slot], totals, strict=True)
            ]

    require(total_samples > 0, "评估集为空")
    diagonal = sum(count_confusion[index, index] for index in range(4))
    class_accuracy: dict[str, float] = {}
    for source_count in range(4):
        row_total = int(count_confusion[source_count].sum())
        require(row_total > 0, f"评估集缺少 {source_count} 源样本")
        class_accuracy[str(source_count)] = (
            float(count_confusion[source_count, source_count]) / row_total
        )
    per_slot_metrics = []
    slot_f1_values = []
    for slot, totals in enumerate(per_slot_totals):
        metrics = binary_metrics(*totals)
        metrics["slot"] = slot
        metrics["has_active_labels"] = sum(totals) > 0
        per_slot_metrics.append(metrics)
        if sum(totals) > 0:
            slot_f1_values.append(float(metrics["f1"]))
    active_metrics = binary_metrics(*active_totals)
    zero_total = int(count_confusion[0].sum())
    zero_false_alarm = int(count_confusion[0, 1:].sum())

    return {
        "loss": total_loss / total_samples,
        "sample_count": total_samples,
        "threshold": threshold,
        "count_accuracy": float(diagonal) / total_samples,
        "balanced_count_accuracy": float(np.mean(list(class_accuracy.values()))),
        "count_class_accuracy": class_accuracy,
        "count_mae": count_abs_error / total_samples,
        "under_count_rate": under_count / total_samples,
        "over_count_rate": over_count / total_samples,
        "zero_source_false_alarm_rate": zero_false_alarm / max(zero_total, 1),
        "count_confusion_true_0_3_pred_0_m": count_confusion.tolist(),
        "active_band": active_metrics,
        "active_band_macro_f1": float(np.mean(slot_f1_values)),
        "active_band_per_slot": per_slot_metrics,
        "inactive_slot_activation_rate": inactive_activated / max(inactive_slots, 1),
        "inactive_slot_count": inactive_slots,
        "inactive_slot_activated_count": inactive_activated,
    }


def build_model(n_sub: int, max_src: int, mode: str, device: torch.device) -> SourceDetectionNet:
    model = SourceDetectionNet(
        n_sub=n_sub,
        max_src=max_src,
        mode=mode,
    ).to(device)
    require(
        all(torch.isfinite(parameter).all().item() for parameter in model.parameters()),
        "初始化模型参数包含 NaN/Inf",
    )
    return model


def checkpoint_payload(
    *,
    epoch: int,
    model: SourceDetectionNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    metrics: dict[str, Any],
    config: dict[str, Any],
    generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "validation": metrics,
        "config": config,
        "rng_state": capture_rng_state(generator),
    }


DatasetFactory = Callable[..., SourceDetectionDataset]


def _make_dataset(
    factory: DatasetFactory | None,
    mat_path: Path,
    *,
    augment: bool,
    normalize: str,
    max_src_override: int,
) -> SourceDetectionDataset:
    dataset_factory = factory or SourceDetectionDataset
    return dataset_factory(
        mat_path,
        augment=augment,
        normalize=normalize,
        max_src_override=max_src_override,
    )


def run_capacity(
    args: argparse.Namespace,
    dataset_factory: DatasetFactory | None = None,
) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖容量检查报告: {output}")
    configure_reproducibility(args.seed, args.deterministic)
    device = runtime_device(args.device)
    reset_cuda_peak(device)
    started = time.perf_counter()
    rss_start = process_tree_rss_bytes()
    dataset = _make_dataset(
        dataset_factory,
        args.data_dir / "train_data.mat",
        augment=True,
        normalize="sample_zscore",
        max_src_override=args.max_src,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = make_loader(
        dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        generator=generator,
    )
    model = build_model(dataset.N_sub, dataset.max_src, args.mode, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    x, _, band_mask, ignore_mask = next(iter(loader))
    x = x.to(device, non_blocking=True)
    band_mask = band_mask.to(device, non_blocking=True)
    ignore_mask = ignore_mask.to(device, non_blocking=True)
    logits = model(x)
    loss = compute_loss(logits, band_mask, ignore_mask, args.gamma)
    finite_tensor(logits, "capacity logits")
    finite_tensor(loss, "capacity loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    require(gradients, "容量检查没有产生梯度")
    require(all(torch.isfinite(gradient).all().item() for gradient in gradients), "梯度包含 NaN/Inf")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item())
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    report = {
        "status": "PASS",
        "data_dir": str(args.data_dir.resolve()),
        "device": str(device),
        "mode": args.mode,
        "max_src": dataset.max_src,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "input_shape": list(x.shape),
        "output_shape": list(logits.shape),
        "loss": float(loss.item()),
        "gradient_norm_before_clip": gradient_norm,
        "duration_seconds": time.perf_counter() - started,
        "rss_start_bytes": rss_start,
        "rss_end_bytes": process_tree_rss_bytes(),
        "cuda_peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
        ),
        "cuda_peak_reserved_bytes": (
            int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0
        ),
    }
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def run_train(
    args: argparse.Namespace,
    dataset_factory: DatasetFactory | None = None,
) -> int:
    output_dir = args.output_dir.resolve()
    ensure_empty_directory(output_dir)
    configure_reproducibility(args.seed, args.deterministic)
    device = runtime_device(args.device)
    reset_cuda_peak(device)
    started = time.perf_counter()

    train_set = _make_dataset(
        dataset_factory,
        args.data_dir / "train_data.mat",
        augment=True,
        normalize="sample_zscore",
        max_src_override=args.max_src,
    )
    val_set = _make_dataset(
        dataset_factory,
        args.data_dir / "val_data.mat",
        augment=False,
        normalize="sample_zscore",
        max_src_override=args.max_src,
    )
    require(train_set.N_sub == val_set.N_sub, "train/val N_sub 不一致")
    require(train_set.max_src == val_set.max_src, "train/val max_src 不一致")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = make_loader(
        train_set,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
        generator=generator,
    )
    val_loader = make_loader(
        val_set,
        args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    model = build_model(train_set.N_sub, train_set.max_src, args.mode, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=args.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs - args.warmup_epochs,
        eta_min=args.min_lr,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[args.warmup_epochs],
    )

    config = {
        "entry": str(Path(__file__).resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "max_src": train_set.max_src,
        "n_sub": train_set.N_sub,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "val_batch_size": args.val_batch_size,
        "num_workers": args.num_workers,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "patience": args.patience,
        "gamma": args.gamma,
        "threshold": args.threshold,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "device": str(device),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(output_dir / "run_config.json", config)

    state_before_initial = capture_rng_state(generator)
    initial_validation = evaluate_detailed(
        model,
        val_loader,
        device,
        args.gamma,
        args.threshold,
    )
    restore_rng_state(state_before_initial, generator)
    write_json(output_dir / "initial_validation.json", initial_validation)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    no_improve = 0
    stopped_early = False
    tag = "B_M10" if args.mode == "transformer" and args.max_src == 10 else "S2G1"
    best_path = output_dir / f"best_model_v26_{tag}.pth"
    last_path = output_dir / f"last_model_v26_{tag}.pth"

    for epoch_index in range(args.epochs):
        epoch_started = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for x, _, band_mask, ignore_mask in train_loader:
            x = x.to(device, non_blocking=True)
            band_mask = band_mask.to(device, non_blocking=True)
            ignore_mask = ignore_mask.to(device, non_blocking=True)
            logits = model(x)
            loss = compute_loss(logits, band_mask, ignore_mask, args.gamma)
            finite_tensor(logits, "train logits")
            finite_tensor(loss, "train loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            require(gradients, "训练没有产生梯度")
            require(
                all(torch.isfinite(gradient).all().item() for gradient in gradients),
                "训练梯度包含 NaN/Inf",
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            batch_size = x.shape[0]
            train_loss_sum += float(loss.item()) * batch_size
            train_samples += batch_size

        scheduler.step()
        validation = evaluate_detailed(
            model,
            val_loader,
            device,
            args.gamma,
            args.threshold,
        )
        epoch_number = epoch_index + 1
        epoch_record = {
            "epoch": epoch_number,
            "train_loss": train_loss_sum / max(train_samples, 1),
            "validation": validation,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": time.perf_counter() - epoch_started,
            "process_rss_bytes": process_tree_rss_bytes(),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0
            ),
            "cuda_peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0
            ),
        }
        history.append(epoch_record)
        write_json(output_dir / "epoch_history.json", history)
        print(
            f"[Epoch {epoch_number:3d}/{args.epochs}] "
            f"train={epoch_record['train_loss']:.6f} "
            f"val={validation['loss']:.6f} "
            f"count={validation['count_accuracy']:.4f} "
            f"balanced={validation['balanced_count_accuracy']:.4f} "
            f"band_f1={validation['active_band_macro_f1']:.4f} "
            f"lr={epoch_record['lr']:.3e}",
            flush=True,
        )

        eligible = epoch_number > args.warmup_epochs
        if eligible and validation["loss"] < best_loss:
            best_loss = float(validation["loss"])
            best_epoch = epoch_number
            best_metrics = validation
            no_improve = 0
            torch.save(
                checkpoint_payload(
                    epoch=epoch_number,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    metrics=validation,
                    config=config,
                    generator=generator,
                ),
                best_path,
            )
        elif eligible:
            no_improve += 1

        torch.save(
            checkpoint_payload(
                epoch=epoch_number,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metrics=validation,
                config=config,
                generator=generator,
            ),
            last_path,
        )
        if eligible and no_improve >= args.patience:
            stopped_early = True
            print(f"Early stop: {args.patience} epochs without improvement", flush=True)
            break

    require(best_metrics is not None and best_path.is_file(), "没有生成best checkpoint")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    frozen_validation = evaluate_detailed(
        model,
        val_loader,
        device,
        args.gamma,
        args.threshold,
    )
    write_json(output_dir / "final_validation.json", frozen_validation)

    loss_reduction = (
        (float(initial_validation["loss"]) - float(frozen_validation["loss"]))
        / max(float(initial_validation["loss"]), 1e-12)
    )
    balanced_gain = (
        float(frozen_validation["balanced_count_accuracy"])
        - float(initial_validation["balanced_count_accuracy"])
    )
    band_f1_gain = (
        float(frozen_validation["active_band_macro_f1"])
        - float(initial_validation["active_band_macro_f1"])
    )
    learning_pass = (
        loss_reduction >= 0.20
        and balanced_gain >= 0.20
        and float(frozen_validation["balanced_count_accuracy"]) >= 0.50
        and band_f1_gain >= 0.20
        and float(frozen_validation["active_band_macro_f1"]) >= 0.50
    )
    epochs_completed = len(history)
    if stopped_early:
        convergence_pass = True
        tail_improvement = None
    else:
        cutoff = max(epochs_completed - 10, 0)
        previous_best = min(
            (float(item["validation"]["loss"]) for item in history[:cutoff]),
            default=float("inf"),
        )
        tail_best = min(float(item["validation"]["loss"]) for item in history[cutoff:])
        tail_improvement = (
            (previous_best - tail_best) / max(previous_best, 1e-12)
            if np.isfinite(previous_best)
            else None
        )
        convergence_pass = (
            best_epoch <= cutoff
            or tail_improvement is not None
            and tail_improvement < 0.01
        )
    summary = {
        "status": "TRAIN_COMPLETED",
        "epochs_completed": epochs_completed,
        "stopped_early": stopped_early,
        "best_epoch": best_epoch,
        "best_validation": frozen_validation,
        "initial_validation": initial_validation,
        "learning_gate": {
            "loss_relative_reduction": loss_reduction,
            "balanced_count_accuracy_gain": balanced_gain,
            "active_band_macro_f1_gain": band_f1_gain,
            "pass": learning_pass,
        },
        "convergence_gate": {
            "last_ten_relative_improvement": tail_improvement,
            "pass": convergence_pass,
        },
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "duration_seconds": time.perf_counter() - started,
        "maximum_epoch_rss_bytes": max(item["process_rss_bytes"] for item in history),
        "maximum_cuda_allocated_bytes": max(
            item["cuda_peak_allocated_bytes"] for item in history
        ),
        "maximum_cuda_reserved_bytes": max(
            item["cuda_peak_reserved_bytes"] for item in history
        ),
    }
    write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def run_evaluation(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖评估报告: {output}")
    configure_reproducibility(args.seed, args.deterministic)
    device = runtime_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = checkpoint["config"]
    dataset = SourceDetectionDataset(
        args.data_dir / f"{args.split}_data.mat",
        augment=False,
        normalize="sample_zscore",
        max_src_override=int(config["max_src"]),
    )
    loader = make_loader(
        dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        device=device,
    )
    model = build_model(
        int(config["n_sub"]),
        int(config["max_src"]),
        str(config["mode"]),
        device,
    )
    model.load_state_dict(checkpoint["model"])
    metrics = evaluate_detailed(
        model,
        loader,
        device,
        float(config["gamma"]),
        float(config["threshold"]),
    )
    report = {
        "status": "PASS",
        "evaluation_mode": "independent_checkpoint_reload",
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "device": str(device),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "deterministic": args.deterministic,
        "metrics": metrics,
    }
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def common_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=["concat", "transformer"], default="transformer")
    parser.add_argument("--max_src", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2-G1 第三章受控训练入口")
    subparsers = parser.add_subparsers(dest="command", required=True)

    capacity = subparsers.add_parser("capacity", help="单batch容量检查")
    common_model_arguments(capacity)
    capacity.add_argument("--output", type=Path, required=True)

    train = subparsers.add_parser("train", help="完整确定性训练")
    common_model_arguments(train)
    train.add_argument("--output_dir", type=Path, required=True)
    train.add_argument("--val_batch_size", type=int, default=64)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--warmup_epochs", type=int, default=5)
    train.add_argument("--patience", type=int, default=25)
    train.add_argument("--min_lr", type=float, default=1e-6)
    train.add_argument("--threshold", type=float, default=BAND_THRESHOLD)
    train.add_argument("--grad_clip", type=float, default=1.0)

    evaluation = subparsers.add_parser("eval", help="独立checkpoint评估")
    evaluation.add_argument("--data_dir", type=Path, required=True)
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=["val", "test"], required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--batch_size", type=int, default=64)
    evaluation.add_argument("--num_workers", type=int, default=0)
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--deterministic", action="store_true")

    args = parser.parse_args()
    for name in ("batch_size", "num_workers", "max_src"):
        if hasattr(args, name):
            value = getattr(args, name)
            if name == "num_workers":
                parser.error("--num_workers 不能为负数") if value < 0 else None
            elif value <= 0:
                parser.error(f"--{name} 必须为正数")
    if args.command == "train":
        if args.epochs <= args.warmup_epochs:
            parser.error("--epochs 必须大于 --warmup_epochs")
        if args.patience <= 0 or args.val_batch_size <= 0:
            parser.error("--patience 和 --val_batch_size 必须为正数")
        if not 0 < args.threshold < 1:
            parser.error("--threshold 必须位于 (0,1)")
    return args


def main() -> int:
    args = parse_args()
    if args.command == "capacity":
        return run_capacity(args)
    if args.command == "train":
        return run_train(args)
    if args.command == "eval":
        return run_evaluation(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
