"""S2-G4-R4 PyCharm 点击运行入口。

当前只启动首个训练臂：Exact / 1024 / seed 42。不要添加命令行参数；在 PyCharm
中打开本文件并点击 Run 即可。该臂成功后再次运行会被 sentinel 拒绝，避免误重复。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import psutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R4_ROOT = PROJECT_ROOT / "outputs" / "s2g4r4_scale" / "20260826_132829"
REPRESENTATION = "exact"
SCALE = 1024
SEED = 42
EPOCHS_BY_SCALE = {1024: 200, 4096: 160, 8192: 80}
START_RAM_GIB_BY_SCALE = {1024: 12, 4096: 14, 8192: 16}


def main() -> int:
    expected_python = Path(r"D:\Software\anaconda3\envs\PyTorch\python.exe").resolve()
    actual_python = Path(sys.executable).resolve()
    if actual_python != expected_python:
        raise RuntimeError(
            f"PyCharm解释器不正确。期望 {expected_python}，实际 {actual_python}"
        )
    if REPRESENTATION not in {"exact", "hard_actual", "soft19_actual"}:
        raise ValueError(f"未知表示: {REPRESENTATION}")
    if SCALE not in EPOCHS_BY_SCALE:
        raise ValueError(f"未知规模: {SCALE}")
    available_ram_gib = psutil.virtual_memory().available / 2**30
    required_ram_gib = START_RAM_GIB_BY_SCALE[SCALE]
    if available_ram_gib < required_ram_gib:
        raise RuntimeError(
            f"启动可用RAM不足: {available_ram_gib:.2f} GiB < {required_ram_gib} GiB。"
            "请关闭占用内存的软件后重新点击Run。"
        )

    data_dir = R4_ROOT / "07_dpd" / REPRESENTATION
    manifests = R4_ROOT / "06_manifests"
    required = [
        data_dir / "train" / "loc_train_index.pt",
        data_dir / "val" / "loc_val_index.pt",
        manifests / f"train_{SCALE}.json",
        manifests / "val_select.json",
        R4_ROOT / "08_dpd_audit_after_equivalence_check.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("R4输入不完整:\n" + "\n".join(missing))

    run_key = f"{REPRESENTATION}_n{SCALE}_seed{SEED}"
    sentinel = R4_ROOT / "09_pycharm_state" / f"{run_key}.done.json"
    if sentinel.exists():
        raise FileExistsError(f"该训练臂已经成功完成，拒绝重复运行: {sentinel}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_output = R4_ROOT / "09_training" / f"{run_key}_{stamp}"
    monitor_output = R4_ROOT / "09_training_monitors" / f"{run_key}_{stamp}"
    if train_output.exists() or monitor_output.exists():
        raise FileExistsError("时间戳输出目录已存在，拒绝覆盖")

    epochs = EPOCHS_BY_SCALE[SCALE]
    train_command = [
        str(actual_python), str(PROJECT_ROOT / "第四章代码" / "train_yolo.py"),
        "--data_dir", str(data_dir),
        "--output_dir", str(train_output),
        "--method", "dualhead",
        "--device", "cuda:0",
        "--epochs", str(epochs),
        "--patience", str(epochs),
        "--batch_size", "8",
        "--val_batch_size", "8",
        "--lr", "0.001",
        "--peak_size", "9",
        "--box_size", "9",
        "--dice_weight", "0",
        "--offset_weight", "1",
        "--grad_alpha", "1",
        "--weight_decay", "0.005",
        "--dropout", "0.4",
        "--eval_every", "1",
        "--seed", str(SEED),
        "--no_amp",
        "--deterministic",
        "--fail_on_nonfinite",
        "--save_last_every_epoch",
        "--require_empty_output",
        "--s2g4r4_scratch",
        "--train_manifest", str(manifests / f"train_{SCALE}.json"),
        "--val_manifest", str(manifests / "val_select.json"),
        "--run_label", run_key,
    ]
    monitor_command = [
        str(actual_python), str(PROJECT_ROOT / "第四章代码" / "gate3_stage_runner.py"),
        "--stage", f"r4_e_{run_key}",
        "--gate_label", "S2-G4-R4",
        "--output_dir", str(monitor_output),
        "--scope_dir", str(R4_ROOT),
        "--working_dir", str(PROJECT_ROOT),
        "--timeout_seconds", "0",
        "--check_interval_seconds", "10",
        "--console_log_mode", "full",
        "--scope_output_warning_gib", "40",
        "--scope_output_red_gib", "60",
        "--system_ram_warning_gib", "8",
        "--system_ram_red_gib", "6",
        "--process_rss_warning_gib", "10",
        "--process_rss_red_gib", "14",
        "--gpu_warning_gib", "13",
        "--gpu_red_gib", "15",
        "--disk_warning_gib", "100",
        "--disk_red_gib", "50",
        "--",
        *train_command,
    ]
    print(json.dumps({
        "gate": "S2-G4-R4-E",
        "representation": REPRESENTATION,
        "scale": SCALE,
        "seed": SEED,
        "epochs": epochs,
        "data_dir": str(data_dir),
        "train_output": str(train_output),
        "monitor_output": str(monitor_output),
        "timeout": "disabled",
        "available_ram_gib": round(available_ram_gib, 2),
        "required_start_ram_gib": required_ram_gib,
    }, ensure_ascii=False, indent=2), flush=True)
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8:replace"
    completed = subprocess.run(monitor_command, env=environment, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"训练失败，exit={completed.returncode}；失败目录已保留: {train_output}"
        )
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(json.dumps({
        "status": "PASS",
        "run_key": run_key,
        "train_output": str(train_output),
        "monitor_output": str(monitor_output),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练完成。请让 Codex 回读: {sentinel}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
