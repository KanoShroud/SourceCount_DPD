"""S2-G5-R6-A跨seed训练的身份门、初始化pilot和受控启动入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CH3_DIR = PROJECT_ROOT / "第三章代码"
CH4_DIR = PROJECT_ROOT / "第四章代码"
if str(CH4_DIR) not in sys.path:
    sys.path.insert(0, str(CH4_DIR))
from train_yolo import compute_offset_loss  # noqa: E402
from train_yolo import configure_reproducibility as configure_d8  # noqa: E402
from train_yolo import sha256_state_dict  # noqa: E402
from train_yolo import tensor_is_finite  # noqa: E402
from yolo_model import YOLOv8Loc  # noqa: E402
from yolo_model import focal_loss_hm  # noqa: E402


PYTHON = Path(sys.executable).resolve()
STAGE_RUNNER = CH4_DIR / "gate3_stage_runner.py"
CH3_TRAIN_ENTRY = CH3_DIR / "s2g5_r6_train_ch3.py"
D8_TRAIN_ENTRY = CH4_DIR / "train_yolo.py"

CH3_R4_ROOT = PROJECT_ROOT / "outputs" / "s2g5r4_ch3_scale" / "20260828_151735"
CH3_STATUS = CH3_R4_ROOT / "lazy_pretrain_status.json"
CH3_SEED42 = CH3_R4_ROOT / "train_16k" / "best_model_v26_B_M10.pth"
CH3_DATA = CH3_R4_ROOT / "training_views" / "data_16k"
CH3_SOURCE = CH3_DATA / "train_data.mat"
CH3_VALIDATION = CH3_DATA / "val_data.mat"
CH3_CACHE = CH3_R4_ROOT / "lazy_cache" / "train_16k_sample_zscore.npy"
CH3_CACHE_MANIFEST = CH3_R4_ROOT / "lazy_cache" / "train_16k_sample_zscore.json"

D8_R4_ROOT = PROJECT_ROOT / "outputs" / "s2g4r4_scale" / "20260826_132829"
D8_DATA = D8_R4_ROOT / "07_dpd" / "hard_actual"
D8_MANIFEST_ROOT = D8_R4_ROOT / "06_manifests"
D8_TRAIN_MANIFEST = D8_MANIFEST_ROOT / "train_8192.json"
D8_VAL_MANIFEST = D8_MANIFEST_ROOT / "val_select.json"
D8_SEED42 = D8_R4_ROOT / "09_training" / "n8192" / "hard_actual" / "best_yolo_dualhead_std.pth"

EXPECTED = {
    "ch3_source": "594c0fd607c6c615839a381ce26102c59b20e76d4799a9e609cc768d6c58c880",
    "ch3_validation": "d4cb6af9e3f99c2162c306105b29f947e48ac85c38075acc8226b6403046d6ad",
    "ch3_cache": "a455fe55397b312ff5b42699e6b69e6cda7d9ca3649eed6b83fa20d8ccca8dda",
    "ch3_cache_manifest": "6ed01e32276ba5f2b81507aa6ed25356f3883c0e8387f1d926e3f54949b4de30",
    "ch3_seed42": "f2f7a7c345f1866b871282670f45671d930de34bb06493a9828a9d04a38699c4",
    "d8_train_manifest": "cefe734cc6b71d1947344d67c0030d859253eabf9ab46c91e092ca750398893d",
    "d8_val_manifest": "872137b787be5f2d39ec2766cde7b6eae6f5480dcb56ded3996d4df47e0894ca",
    "d8_seed42": "4caaf2b96c2f8eb666b417f0cffe4ab90760315f9bd92c4d6ce4afcd425e0e7b",
}
APPROVED_SEEDS = (42, 1042, 2042)
NEW_SEEDS = (1042, 2042)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    path = path.resolve()
    require(path.is_file(), f"文件不存在: {path}")
    result: dict[str, Any] = {"path": str(path), "size_bytes": path.stat().st_size}
    if hash_file:
        result["sha256"] = sha256_file(path)
    return result


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_all_frozen_inputs() -> dict[str, Any]:
    paths = {
        "ch3_source": CH3_SOURCE,
        "ch3_validation": CH3_VALIDATION,
        "ch3_cache": CH3_CACHE,
        "ch3_cache_manifest": CH3_CACHE_MANIFEST,
        "ch3_seed42": CH3_SEED42,
        "d8_train_manifest": D8_TRAIN_MANIFEST,
        "d8_val_manifest": D8_VAL_MANIFEST,
        "d8_seed42": D8_SEED42,
    }
    identities = {name: identity(path) for name, path in paths.items()}
    for name, expected in EXPECTED.items():
        if name in identities:
            require(identities[name]["sha256"] == expected, f"{name} SHA变化")
    return identities


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    require(run_root.is_dir(), f"R6根目录不存在: {run_root}")
    require(not (run_root / "preflight.json").exists(), "拒绝覆盖R6 preflight")
    required = {
        "ch3_source": CH3_SOURCE,
        "ch3_validation": CH3_VALIDATION,
        "ch3_cache": CH3_CACHE,
        "ch3_cache_manifest": CH3_CACHE_MANIFEST,
        "ch3_status": CH3_STATUS,
        "ch3_seed42": CH3_SEED42,
        "d8_train_manifest": D8_TRAIN_MANIFEST,
        "d8_val_manifest": D8_VAL_MANIFEST,
        "d8_seed42": D8_SEED42,
    }
    identities = {name: identity(path) for name, path in required.items()}
    for name, expected in EXPECTED.items():
        require(identities[name]["sha256"] == expected, f"{name} SHA变化")

    cache_manifest = load_json(CH3_CACHE_MANIFEST)
    require(cache_manifest.get("status") == "PASS", "CH3缓存manifest非PASS")
    require(cache_manifest["source_meta"]["sample_count"] == 16384, "CH3训练样本数不是16k")
    require(cache_manifest["source_meta"]["source_count_histogram"] == {
        "0": 4096, "1": 4096, "2": 4096, "3": 4096,
    }, "CH3 K分层不是各4096条")
    require(load_json(CH3_STATUS).get("status") == "LAZY_READY_FOR_TRAINING", "CH3懒加载门禁未通过")
    d8_manifest = load_json(D8_TRAIN_MANIFEST)
    require(d8_manifest.get("sample_count") == 8192, "D8训练manifest不是8192条")
    require(len(d8_manifest.get("indices", [])) == 8192, "D8训练索引不是8192条")
    require(d8_manifest.get("n2") == 4096 and d8_manifest.get("n3") == 4096, "D8训练K=2/3不是各4096条")
    d8_val_manifest = load_json(D8_VAL_MANIFEST)
    require(d8_val_manifest.get("sample_count") == 1024, "D8 validation不是1024条")
    require(d8_val_manifest.get("n2") == 512 and d8_val_manifest.get("n3") == 512, "D8 validation K=2/3不是各512条")

    for split in ("train", "val"):
        index_path = D8_DATA / split / f"loc_{split}_index.pt"
        index = torch.load(index_path, map_location="cpu", weights_only=False)
        require(index.get("n_shards") == len(index.get("shard_files", [])), f"D8 {split}分片索引不完整")
        require(index.get("n_total_tasks") >= (8192 if split == "train" else 2048), f"D8 {split}样本总数异常")
        require(all((index_path.parent / name).is_file() for name in index["shard_files"]), f"D8 {split}存在缺失分片")

    ch3_config = load_json(CH3_R4_ROOT / "train_16k" / "run_config.json")
    d8_config = load_json(D8_R4_ROOT / "09_training" / "n8192" / "hard_actual" / "run_config.json")
    require(ch3_config["seed"] == 42 and ch3_config["epochs"] == 150, "seed42 CH3配置错误")
    require(ch3_config["batch_size"] == 64 and ch3_config["patience"] == 25, "seed42 CH3 batch/patience错误")
    d8_args = d8_config["args"]
    require(d8_args["seed"] == 42 and d8_args["epochs"] == 80, "seed42 D8配置错误")
    require(d8_args["batch_size"] == d8_args["val_batch_size"] == 8, "seed42 D8 batch错误")
    require(d8_args["amp"] is False and d8_args["s2g4r4_scratch"] is True, "seed42 D8身份错误")

    ch3_fixed = {
        "mode": "transformer", "max_src": 10, "n_sub": 19,
        "epochs": 150, "batch_size": 64, "val_batch_size": 64,
        "num_workers": 0, "lr": 0.001, "min_lr": 1e-6,
        "weight_decay": 0.0005, "warmup_epochs": 5, "patience": 25,
        "gamma": 2.0, "threshold": 0.5, "grad_clip": 1.0,
        "deterministic": True, "device": "cuda:0",
    }
    require(all(ch3_config.get(key) == value for key, value in ch3_fixed.items()), "CH3冻结训练配置变化")
    d8_fixed = {
        "method": "dualhead", "device": "cuda:0", "epochs": 80,
        "batch_size": 8, "val_batch_size": 8, "lr": 0.001,
        "patience": 80, "peak_size": 9, "box_size": 9, "eval_every": 1,
        "amp": False, "dist_alpha": 2.0, "offset_weight": 1.0,
        "dice_weight": 0.0, "weight_decay": 0.005, "dropout": 0.4,
        "conf_weight_offset": False, "soft_conf": False, "grad_alpha": 1.0,
        "deterministic": True, "fail_on_nonfinite": True,
        "save_last_every_epoch": True, "require_empty_output": True,
    }
    require(all(d8_args.get(key) == value for key, value in d8_fixed.items()), "D8冻结训练配置变化")

    report = {
        "status": "PASS",
        "gate": "S2-G5-R6-A",
        "run_root": str(run_root),
        "approved_seeds": list(APPROVED_SEEDS),
        "new_training_seeds": list(NEW_SEEDS),
        "frozen_inputs": identities,
        "ch3_contract": {"samples": 16384, "per_k": 4096, "epochs": 150, "batch": 64, "patience": 25},
        "d8_contract": {"samples": 8192, "per_k": 4096, "epochs": 80, "batch": 8, "patience": 80},
        "test_read": False,
    }
    write_json(run_root / "preflight.json", report)
    return report


def run_init_pilot(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    require(load_json(run_root / "preflight.json").get("status") == "PASS", "R6 preflight未通过")
    results: dict[str, Any] = {}
    for seed in APPROVED_SEEDS:
        ch3_process = subprocess.run(
            [str(PYTHON), str(CH3_TRAIN_ENTRY), "--seed", str(seed), "--initialization_only"],
            cwd=CH3_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        ch3_hash = json.loads(ch3_process.stdout.strip())["sha256"]
        configure_d8(seed, True)
        d8 = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0)
        d8_hash = sha256_state_dict(d8.state_dict())
        del d8
        results[str(seed)] = {"ch3_initial_sha256": ch3_hash, "d8_initial_sha256": d8_hash}
    require(len({row["ch3_initial_sha256"] for row in results.values()}) == 3, "CH3不同seed初始化重复")
    require(len({row["d8_initial_sha256"] for row in results.values()}) == 3, "D8不同seed初始化重复")

    ch3_pilot_path = run_root / "pilot" / "ch3_cuda_one_batch.json"
    ch3_process = subprocess.run(
        [
            str(PYTHON), str(CH3_TRAIN_ENTRY), "--seed", "1042",
            "--pilot_output", str(ch3_pilot_path),
        ],
        cwd=CH3_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8:replace"},
    )
    require(ch3_process.returncode == 0, f"CH3 CUDA pilot失败: {ch3_process.stdout[-2000:]}")
    ch3_pilot = load_json(ch3_pilot_path)
    require(ch3_pilot.get("status") == "PASS" and ch3_pilot.get("device") == "cuda:0", "CH3 CUDA pilot未通过")

    started = time.perf_counter()
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    configure_d8(1042, True)
    train_index = torch.load(D8_DATA / "train" / "loc_train_index.pt", map_location="cpu", weights_only=False)
    first_shard_path = D8_DATA / "train" / train_index["shard_files"][0]
    shard = torch.load(first_shard_path, map_location="cpu", weights_only=False)
    dpd = shard["fine_dpd"][:8].float()
    for row in range(dpd.shape[0]):
        dpd[row] = (dpd[row] - dpd[row].mean()) / (dpd[row].std() + 1e-6)
    target = shard["gauss_label"][:8].float()
    pos = shard["pos_label"][:8]
    n_src = shard["n_src"][:8]
    model = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.005)
    model.train()
    pred_hm, pred_offset = model(dpd.to(device))
    focal = focal_loss_hm(pred_hm.float(), target.to(device))
    offset = compute_offset_loss(pred_offset.float(), pos, n_src, device)
    loss = focal + offset
    require(tensor_is_finite(pred_hm) and tensor_is_finite(pred_offset), "D8 CUDA pilot输出非有限")
    require(tensor_is_finite(loss), "D8 CUDA pilot loss非有限")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradients = [value.grad for value in model.parameters() if value.grad is not None]
    require(gradients and all(tensor_is_finite(value) for value in gradients), "D8 CUDA pilot梯度非有限")
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0).item())
    optimizer.step()
    torch.cuda.synchronize(device)
    d8_pilot = {
        "status": "PASS", "device": "cuda:0", "seed": 1042,
        "source_shard": identity(first_shard_path, hash_file=False),
        "batch_size": 8, "input_shape": list(dpd.shape),
        "heatmap_shape": list(pred_hm.shape), "offset_shape": list(pred_offset.shape),
        "loss": float(loss.item()), "focal_loss": float(focal.item()),
        "offset_loss": float(offset.item()), "gradient_norm_before_clip": gradient_norm,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(run_root / "pilot" / "d8_cuda_one_batch.json", d8_pilot)
    del optimizer, model, loss, pred_hm, pred_offset, dpd, target, pos, n_src, shard
    torch.cuda.empty_cache()

    report = {
        "status": "PASS", "seeds": results, "different_seeds_distinct": True,
        "cuda_pilots": {"ch3": ch3_pilot, "d8": d8_pilot},
    }
    write_json(run_root / "initialization_pilot.json", report)
    return report


def monitor_command(run_root: Path, stage: str, child: list[str]) -> list[str]:
    return [
        str(PYTHON), str(STAGE_RUNNER),
        "--stage", stage,
        "--gate_label", "S2-G5-R6-A",
        "--output_dir", str(run_root / "monitor" / stage),
        "--scope_dir", str(run_root),
        "--working_dir", str(PROJECT_ROOT),
        "--timeout_seconds", "0",
        "--check_interval_seconds", "10",
        "--console_log_mode", "summary",
        "--scope_output_warning_gib", "8",
        "--scope_output_red_gib", "12",
        "--system_ram_warning_gib", "8",
        "--system_ram_red_gib", "6",
        "--process_rss_warning_gib", "16",
        "--process_rss_red_gib", "20",
        "--gpu_warning_gib", "13",
        "--gpu_red_gib", "15",
        "--disk_warning_gib", "40",
        "--disk_red_gib", "25",
        "--encoding", "utf-8",
        "--",
        *child,
    ]


def run_training(args: argparse.Namespace) -> int:
    run_root = args.run_root.resolve()
    require(args.seed in NEW_SEEDS, f"未批准的新训练seed: {args.seed}")
    require(load_json(run_root / "initialization_pilot.json").get("status") == "PASS", "初始化pilot未通过")
    before = verify_all_frozen_inputs()
    if args.model == "ch3":
        output = run_root / "training" / f"ch3_seed{args.seed}"
        child = [
            str(PYTHON), str(CH3_TRAIN_ENTRY),
            "--seed", str(args.seed),
            "--output_dir", str(output),
        ]
    else:
        output = run_root / "training" / f"d8_seed{args.seed}"
        child = [
            str(PYTHON), str(D8_TRAIN_ENTRY),
            "--data_dir", str(D8_DATA),
            "--output_dir", str(output),
            "--method", "dualhead", "--device", "cuda:0",
            "--epochs", "80", "--batch_size", "8", "--val_batch_size", "8",
            "--lr", "0.001", "--patience", "80", "--peak_size", "9", "--box_size", "9",
            "--eval_every", "1", "--no_amp", "--dist_alpha", "2.0",
            "--offset_weight", "1.0", "--dice_weight", "0.0",
            "--weight_decay", "0.005", "--dropout", "0.4", "--grad_alpha", "1.0",
            "--seed", str(args.seed), "--deterministic", "--fail_on_nonfinite",
            "--save_last_every_epoch", "--require_empty_output", "--s2g5r6_scratch",
            "--train_manifest", str(D8_TRAIN_MANIFEST),
            "--val_manifest", str(D8_VAL_MANIFEST),
            "--run_label", f"r6_n8192_hard_seed{args.seed}",
        ]
    require(not output.exists(), f"拒绝覆盖训练输出: {output}")
    stage = f"train_{args.model}_seed{args.seed}"
    require(not (run_root / "monitor" / stage).exists(), f"拒绝覆盖监控目录: {stage}")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8:replace"
    returncode = subprocess.run(
        monitor_command(run_root, stage, child),
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
    ).returncode
    after = verify_all_frozen_inputs()
    require(before == after, f"{stage}训练前后冻结输入身份变化")
    write_json(
        run_root / "input_checks" / f"{stage}.json",
        {"status": "PASS", "stage": stage, "before": before, "after": after},
    )
    return returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("init-pilot")
    train = sub.add_parser("train")
    train.add_argument("--model", choices=("ch3", "d8"), required=True)
    train.add_argument("--seed", type=int, choices=NEW_SEEDS, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        result = run_preflight(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "init-pilot":
        result = run_init_pilot(args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    return run_training(args)


if __name__ == "__main__":
    raise SystemExit(main())
