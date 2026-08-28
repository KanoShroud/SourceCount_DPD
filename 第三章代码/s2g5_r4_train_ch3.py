"""S2-G5-R4懒加载缓存、等价性、容量与吞吐验证入口。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch

from s2g1_train_ch3 import (
    SourceDetectionDataset,
    build_model,
    compute_loss,
    configure_reproducibility,
    make_loader,
    process_tree_rss_bytes,
    run_capacity,
    run_train,
    runtime_device,
)
from s2g5_r4_lazy_dataset import (
    LazySourceDetectionDataset,
    audit_cache,
    build_cache,
    require,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "s2g5r4_ch3_scale" / "20260828_151735"
DATA_16K = RUN_ROOT / "training_views" / "data_16k"
DATA_1K = PROJECT_ROOT / "outputs" / "s2g5r2_ch3" / "20260827_191207" / "training_views" / "data_1k"
CACHE_ROOT = RUN_ROOT / "lazy_cache"
AUDIT_ROOT = RUN_ROOT / "audit"
TRAIN_16K = DATA_16K / "train_data.mat"
TRAIN_1K = DATA_1K / "train_data.mat"
CACHE_16K = CACHE_ROOT / "train_16k_sample_zscore.npy"
MANIFEST_16K = CACHE_ROOT / "train_16k_sample_zscore.json"
CACHE_1K = CACHE_ROOT / "train_1k_sample_zscore.npy"
MANIFEST_1K = CACHE_ROOT / "train_1k_sample_zscore.json"


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def lazy_factory_for(source: Path, cache: Path, manifest: Path) -> Callable[..., Any]:
    expected = source.resolve()

    def factory(mat_path: Path, **kwargs: Any) -> Any:
        if Path(mat_path).resolve() == expected:
            return LazySourceDetectionDataset(
                mat_path,
                cache_path=cache,
                manifest_path=manifest,
                **kwargs,
            )
        return SourceDetectionDataset(mat_path, **kwargs)

    return factory


def tensor_digest(tensor: torch.Tensor) -> str:
    import hashlib

    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def batch_digests(dataset: Any, seed: int, batches: int = 5) -> list[dict[str, str]]:
    configure_reproducibility(seed, True)
    generator = torch.Generator().manual_seed(seed)
    loader = make_loader(
        dataset,
        32,
        shuffle=True,
        num_workers=0,
        device=torch.device("cpu"),
        generator=generator,
    )
    result = []
    for index, batch in enumerate(loader):
        result.append({
            "x": tensor_digest(batch[0]),
            "src_count": tensor_digest(batch[1]),
            "band_mask": tensor_digest(batch[2]),
            "ignore_mask": tensor_digest(batch[3]),
        })
        if index + 1 == batches:
            break
    return result


def run_dataset_audit(output: Path) -> dict[str, Any]:
    eager_plain = SourceDetectionDataset(TRAIN_1K, augment=False, normalize="sample_zscore", max_src_override=10)
    lazy_plain = LazySourceDetectionDataset(
        TRAIN_1K,
        cache_path=CACHE_1K,
        manifest_path=MANIFEST_1K,
        augment=False,
        normalize="sample_zscore",
        max_src_override=10,
        verify_source_hash=True,
    )
    indices = [0, 1, 127, 255, 511, 767, 1022, 1023]
    exact_plain = True
    for index in indices:
        eager_item = eager_plain[index]
        lazy_item = lazy_plain[index]
        exact_plain &= all(torch.equal(left, right) for left, right in zip(eager_item, lazy_item))
    eager_aug = SourceDetectionDataset(TRAIN_1K, augment=True, normalize="sample_zscore", max_src_override=10)
    lazy_aug = LazySourceDetectionDataset(
        TRAIN_1K,
        cache_path=CACHE_1K,
        manifest_path=MANIFEST_1K,
        augment=True,
        normalize="sample_zscore",
        max_src_override=10,
    )
    eager_batches = batch_digests(eager_aug, 42)
    lazy_batches = batch_digests(lazy_aug, 42)
    require(exact_plain, "未增强Dataset不等价")
    require(eager_batches == lazy_batches, "增强或shuffle后的batch不等价")
    report = {
        "status": "PASS",
        "plain_indices": indices,
        "plain_exact": exact_plain,
        "batch_size": 32,
        "batch_count": len(eager_batches),
        "shuffle_seed": 42,
        "augmented_batch_digests": eager_batches,
        "num_workers": 0,
    }
    write_json(output, report)
    return report


def train_args(data_dir: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=data_dir,
        device="cuda:0",
        mode="transformer",
        max_src=10,
        batch_size=64,
        val_batch_size=64,
        num_workers=0,
        lr=1e-3,
        weight_decay=5e-4,
        gamma=2.0,
        seed=42,
        deterministic=True,
        output_dir=output_dir,
        epochs=3,
        warmup_epochs=1,
        patience=3,
        min_lr=1e-6,
        threshold=0.5,
        grad_clip=1.0,
    )


def formal_train_args(output_dir: Path) -> argparse.Namespace:
    args = train_args(DATA_16K, output_dir)
    args.epochs = 150
    args.warmup_epochs = 5
    args.patience = 25
    return args


def comparable_history(path: Path) -> list[dict[str, Any]]:
    history = json_load(path)
    for row in history:
        row.pop("epoch_seconds", None)
        row.pop("process_rss_bytes", None)
        row.pop("cuda_peak_allocated_bytes", None)
        row.pop("cuda_peak_reserved_bytes", None)
    return history


def compare_tensor_trees(left: Any, right: Any, path: str = "root") -> tuple[bool, float, str | None]:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            return False, float("inf"), path
        if torch.equal(left, right):
            return True, 0.0, None
        if left.is_floating_point():
            difference = float((left - right).abs().max().item())
            return difference <= 1e-7, difference, None if difference <= 1e-7 else path
        return False, float("inf"), path
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            return False, float("inf"), path
        maximum = 0.0
        for key in left:
            ok, difference, failed = compare_tensor_trees(left[key], right[key], f"{path}.{key}")
            maximum = max(maximum, difference)
            if not ok:
                return False, maximum, failed
        return True, maximum, None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            return False, float("inf"), path
        maximum = 0.0
        for index, (l_item, r_item) in enumerate(zip(left, right)):
            ok, difference, failed = compare_tensor_trees(l_item, r_item, f"{path}[{index}]")
            maximum = max(maximum, difference)
            if not ok:
                return False, maximum, failed
        return True, maximum, None
    return (left == right, 0.0 if left == right else float("inf"), None if left == right else path)


def run_short_pair(output: Path) -> dict[str, Any]:
    eager_dir = RUN_ROOT / "eager_short_train"
    lazy_dir = RUN_ROOT / "lazy_short_train"
    require(not eager_dir.exists(), f"拒绝覆盖: {eager_dir}")
    require(not lazy_dir.exists(), f"拒绝覆盖: {lazy_dir}")
    run_train(train_args(DATA_1K, eager_dir))
    run_train(
        train_args(DATA_1K, lazy_dir),
        dataset_factory=lazy_factory_for(TRAIN_1K, CACHE_1K, MANIFEST_1K),
    )
    eager_history = comparable_history(eager_dir / "epoch_history.json")
    lazy_history = comparable_history(lazy_dir / "epoch_history.json")
    histories_exact = eager_history == lazy_history
    eager_checkpoint = torch.load(eager_dir / "last_model_v26_B_M10.pth", map_location="cpu", weights_only=False)
    lazy_checkpoint = torch.load(lazy_dir / "last_model_v26_B_M10.pth", map_location="cpu", weights_only=False)
    model_ok, model_max, model_failed = compare_tensor_trees(eager_checkpoint["model"], lazy_checkpoint["model"])
    optimizer_ok, optimizer_max, optimizer_failed = compare_tensor_trees(
        eager_checkpoint["optimizer"], lazy_checkpoint["optimizer"]
    )
    scheduler_ok, _, scheduler_failed = compare_tensor_trees(
        eager_checkpoint["scheduler"], lazy_checkpoint["scheduler"]
    )
    require(histories_exact, "3 epoch训练指标不完全一致")
    require(model_ok and optimizer_ok and scheduler_ok, "3 epoch训练状态不等价")
    report = {
        "status": "PASS",
        "epochs": 3,
        "seed": 42,
        "num_workers": 0,
        "histories_exact": histories_exact,
        "model_state_ok": model_ok,
        "model_max_abs_error": model_max,
        "optimizer_state_ok": optimizer_ok,
        "optimizer_max_abs_error": optimizer_max,
        "scheduler_state_ok": scheduler_ok,
        "failed_paths": [value for value in (model_failed, optimizer_failed, scheduler_failed) if value],
        "eager_output": str(eager_dir),
        "lazy_output": str(lazy_dir),
    }
    write_json(output, report)
    return report


def capacity_args(output: Path) -> argparse.Namespace:
    return argparse.Namespace(
        data_dir=DATA_16K,
        output=output,
        device="cuda:0",
        mode="transformer",
        max_src=10,
        batch_size=64,
        num_workers=0,
        lr=1e-3,
        weight_decay=5e-4,
        gamma=2.0,
        seed=42,
        deterministic=True,
    )


def run_throughput(output: Path) -> dict[str, Any]:
    configure_reproducibility(42, True)
    device = runtime_device("cuda:0")
    dataset = LazySourceDetectionDataset(
        TRAIN_16K,
        cache_path=CACHE_16K,
        manifest_path=MANIFEST_16K,
        augment=True,
        normalize="sample_zscore",
        max_src_override=10,
        verify_source_hash=True,
    )
    generator = torch.Generator().manual_seed(42)
    loader = make_loader(dataset, 64, shuffle=True, num_workers=0, device=device, generator=generator)
    model = build_model(dataset.N_sub, dataset.max_src, "transformer", device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    started = time.perf_counter()
    loss_sum = 0.0
    samples = 0
    model.train()
    for x, _, band_mask, ignore_mask in loader:
        x = x.to(device, non_blocking=True)
        band_mask = band_mask.to(device, non_blocking=True)
        ignore_mask = ignore_mask.to(device, non_blocking=True)
        logits = model(x)
        loss = compute_loss(logits, band_mask, ignore_mask, 2.0)
        require(torch.isfinite(loss).item(), "吞吐pilot loss包含NaN/Inf")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_sum += float(loss.item()) * x.shape[0]
        samples += x.shape[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    duration = time.perf_counter() - started
    report = {
        "status": "PASS",
        "scope": "one_training_epoch_no_checkpoint_no_validation",
        "samples": samples,
        "batch_size": 64,
        "batches": len(loader),
        "duration_seconds": duration,
        "mean_train_loss": loss_sum / samples,
        "projected_150_epoch_training_seconds_without_validation": duration * 150,
        "process_tree_rss_bytes": process_tree_rss_bytes(),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else 0,
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if device.type == "cuda" else 0,
    }
    write_json(output, report)
    return report


def finalize(output: Path) -> dict[str, Any]:
    required = {
        "cache_sample": AUDIT_ROOT / "lazy_sample_equivalence.json",
        "dataset": AUDIT_ROOT / "lazy_dataset_equivalence.json",
        "short_train": AUDIT_ROOT / "lazy_short_train_equivalence.json",
        "capacity": AUDIT_ROOT / "lazy_capacity.json",
        "throughput": AUDIT_ROOT / "lazy_throughput.json",
    }
    reports = {name: json_load(path) for name, path in required.items()}
    require(all(report.get("status") == "PASS" for report in reports.values()), "存在未通过的懒加载门禁")
    capacity_monitor = json_load(RUN_ROOT / "monitor" / "14_lazy_capacity" / "stage_monitor_report.json")
    throughput_monitor = json_load(RUN_ROOT / "monitor" / "15_lazy_throughput" / "stage_monitor_report.json")
    minimum_ram = min(
        int(capacity_monitor["minimum_system_available_bytes"]),
        int(throughput_monitor["minimum_system_available_bytes"]),
    )
    maximum_rss = max(
        int(capacity_monitor["maximum_process_tree_rss_bytes"]),
        int(throughput_monitor["maximum_process_tree_rss_bytes"]),
    )
    maximum_gpu_mib = max(
        int(capacity_monitor["maximum_gpu_used_mib"]),
        int(throughput_monitor["maximum_gpu_used_mib"]),
    )
    resource_ok = minimum_ram >= 8 * 1024**3 and maximum_rss < 16 * 1024**3 and maximum_gpu_mib < 13 * 1024
    status = "LAZY_READY_FOR_TRAINING" if resource_ok else "LAZY_RESOURCE_INSUFFICIENT"
    projected = float(reports["throughput"]["projected_150_epoch_training_seconds_without_validation"])
    advisory = projected > 4 * 3600
    report = {
        "status": status,
        "lazy_io_slow_advisory": advisory,
        "formal_training_started": False,
        "reports": {name: str(path) for name, path in required.items()},
        "resource_envelope_from_capacity_and_throughput": {
            "minimum_system_available_bytes": minimum_ram,
            "maximum_process_tree_rss_bytes": maximum_rss,
            "maximum_gpu_used_mib": maximum_gpu_mib,
            "red_flags": capacity_monitor.get("red_flags", []) + throughput_monitor.get("red_flags", []),
        },
        "capacity_monitor_status": capacity_monitor.get("status"),
        "throughput_monitor_status": throughput_monitor.get("status"),
        "pycharm_entry": str(PROJECT_ROOT / "第三章代码" / "s2g5r4_pycharm_train.py"),
    }
    write_json(output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2-G5-R4懒加载验证入口")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build-16k")
    sub.add_parser("build-1k")
    sub.add_parser("audit-cache")
    sub.add_parser("audit-dataset")
    sub.add_parser("short-pair")
    sub.add_parser("capacity")
    sub.add_parser("throughput")
    sub.add_parser("finalize")
    sub.add_parser("train-16k")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-16k":
        result = build_cache(TRAIN_16K, CACHE_16K, MANIFEST_16K)
    elif args.command == "build-1k":
        result = build_cache(TRAIN_1K, CACHE_1K, MANIFEST_1K)
    elif args.command == "audit-cache":
        result = audit_cache(TRAIN_16K, CACHE_16K, MANIFEST_16K, AUDIT_ROOT / "lazy_sample_equivalence.json")
    elif args.command == "audit-dataset":
        result = run_dataset_audit(AUDIT_ROOT / "lazy_dataset_equivalence.json")
    elif args.command == "short-pair":
        result = run_short_pair(AUDIT_ROOT / "lazy_short_train_equivalence.json")
    elif args.command == "capacity":
        result_code = run_capacity(
            capacity_args(AUDIT_ROOT / "lazy_capacity.json"),
            dataset_factory=lazy_factory_for(TRAIN_16K, CACHE_16K, MANIFEST_16K),
        )
        return result_code
    elif args.command == "throughput":
        result = run_throughput(AUDIT_ROOT / "lazy_throughput.json")
    elif args.command == "finalize":
        result = finalize(RUN_ROOT / "lazy_pretrain_status.json")
    elif args.command == "train-16k":
        status = json_load(RUN_ROOT / "lazy_pretrain_status.json")
        require(status.get("status") == "LAZY_READY_FOR_TRAINING", "懒加载训练门禁未通过")
        return run_train(
            formal_train_args(RUN_ROOT / "train_16k"),
            dataset_factory=lazy_factory_for(TRAIN_16K, CACHE_16K, MANIFEST_16K),
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
