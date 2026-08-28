"""S2-G5-R4 16k CH3正式训练的PyCharm点击入口。

只有在R4数据、审计和容量pilot均完成后才允许启动；不接受命令行改参。
"""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = PROJECT_ROOT / "outputs" / "s2g5r4_ch3_scale" / "20260828_151735"
DATA_DIR = RUN_ROOT / "training_views" / "data_16k"
LAZY_STATUS = RUN_ROOT / "lazy_pretrain_status.json"
LAZY_CACHE = RUN_ROOT / "lazy_cache" / "train_16k_sample_zscore.npy"
LAZY_MANIFEST = RUN_ROOT / "lazy_cache" / "train_16k_sample_zscore.json"
TRAIN_DIR = RUN_ROOT / "train_16k"
MONITOR_DIR = RUN_ROOT / "monitor" / "10_train_16k"
PYTHON = Path(sys.executable).resolve()
TRAIN_ENTRY = PROJECT_ROOT / "第三章代码" / "s2g5_r4_train_ch3.py"
STAGE_RUNNER = PROJECT_ROOT / "第四章代码" / "gate3_stage_runner.py"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    require(DATA_DIR.is_dir(), f"16k训练视图不存在: {DATA_DIR}")
    require(LAZY_STATUS.is_file(), f"懒加载门禁报告不存在: {LAZY_STATUS}")
    require(
        load_json(LAZY_STATUS).get("status") == "LAZY_READY_FOR_TRAINING",
        "R4懒加载等价性或资源门禁未通过",
    )
    require(LAZY_MANIFEST.is_file() and LAZY_CACHE.is_file(), "懒加载缓存或manifest不存在")
    manifest = load_json(LAZY_MANIFEST)
    require(sha256_file(DATA_DIR / "train_data.mat") == manifest["source_meta"]["sha256"], "16k源数据SHA变化")
    require(sha256_file(LAZY_CACHE) == manifest["cache_sha256"], "16k懒加载缓存SHA变化")
    require(not TRAIN_DIR.exists(), f"拒绝覆盖训练目录: {TRAIN_DIR}")
    require(not MONITOR_DIR.exists(), f"拒绝覆盖监控目录: {MONITOR_DIR}")

    command = [
        str(PYTHON),
        str(STAGE_RUNNER),
        "--stage", "r4_train_16k",
        "--gate_label", "S2-G5-R4",
        "--output_dir", str(MONITOR_DIR),
        "--scope_dir", str(RUN_ROOT),
        "--working_dir", str(PROJECT_ROOT),
        "--timeout_seconds", "0",
        "--check_interval_seconds", "10",
        "--console_log_mode", "summary",
        "--scope_output_warning_gib", "35",
        "--scope_output_red_gib", "45",
        "--system_ram_warning_gib", "10",
        "--system_ram_red_gib", "8",
        "--process_rss_warning_gib", "12",
        "--process_rss_red_gib", "16",
        "--gpu_warning_gib", "11",
        "--gpu_red_gib", "13",
        "--disk_warning_gib", "40",
        "--disk_red_gib", "25",
        "--encoding", "utf-8",
        "--",
        str(PYTHON),
        str(TRAIN_ENTRY),
        "train-16k",
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8:replace"
    print("S2-G5-R4 16k正式训练即将启动。")
    print(f"RUN_ROOT={RUN_ROOT}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT, env=environment)


if __name__ == "__main__":
    main()
