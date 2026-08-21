"""项目级运行路径、设备与 smoke/formal 隔离配置。

本模块只负责运行基础设施，不保存算法或实验超参数。环境变量：

- ``SOURCECOUNT_RUN_MODE``: ``smoke`` 或 ``formal``，默认 smoke。
- ``SOURCECOUNT_DATA_ROOT``: 正式数据根目录，默认 ``<project>/data``。
- ``SOURCECOUNT_OUTPUT_ROOT``: 输出根目录，默认 ``<project>/outputs``。
- ``SOURCECOUNT_DEVICE``: PyTorch 设备，默认 ``cuda:0``。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal


RunMode = Literal["smoke", "formal"]

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(
    os.environ.get("SOURCECOUNT_DATA_ROOT", PROJECT_ROOT / "data")
).expanduser().resolve()
OUTPUT_ROOT = Path(
    os.environ.get("SOURCECOUNT_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
).expanduser().resolve()
DEFAULT_DEVICE = os.environ.get("SOURCECOUNT_DEVICE", "cuda:0")


def get_run_mode(mode: str | None = None) -> RunMode:
    """返回经过校验的运行模式。"""
    value = (mode or os.environ.get("SOURCECOUNT_RUN_MODE", "smoke")).lower()
    if value not in {"smoke", "formal"}:
        raise ValueError(
            f"SOURCECOUNT_RUN_MODE 必须为 smoke 或 formal，当前为 {value!r}"
        )
    return value  # type: ignore[return-value]


def chapter_data_dir(
    chapter: str,
    *,
    mode: str | None = None,
    create: bool = False,
) -> Path:
    """返回章节数据目录；smoke 数据永远位于 outputs/smoke。"""
    run_mode = get_run_mode(mode)
    if run_mode == "smoke":
        path = OUTPUT_ROOT / "smoke" / chapter / "data"
    else:
        path = DATA_ROOT / chapter
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def entry_output_dir(
    chapter: str,
    entry: str,
    *,
    mode: str | None = None,
    create: bool = True,
) -> Path:
    """返回入口独占输出目录，防止不同脚本或运行模式互相覆盖。"""
    run_mode = get_run_mode(mode)
    path = OUTPUT_ROOT / run_mode / chapter / Path(entry).stem
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(
    chapter: str,
    entry: str,
    filename: str,
    *,
    mode: str | None = None,
) -> Path:
    """构造入口输出文件路径并确保父目录存在。"""
    return entry_output_dir(chapter, entry, mode=mode) / filename


def checkpoint_path(
    chapter: str,
    trainer: str,
    filename: str,
    *,
    mode: str | None = None,
) -> Path:
    """返回指定训练入口的权重路径。"""
    return entry_output_dir(chapter, trainer, mode=mode) / filename


def resolve_torch_device(requested: str | None = None):
    """校验并返回 torch.device；不会把无效 CUDA 编号静默换成别的 GPU。"""
    import torch

    value = requested or DEFAULT_DEVICE
    if not value.startswith("cuda"):
        return torch.device(value)
    if not torch.cuda.is_available():
        raise RuntimeError(f"请求设备 {value}，但当前 PyTorch 未检测到 CUDA")
    index = 0 if ":" not in value else int(value.split(":", 1)[1])
    count = torch.cuda.device_count()
    if index < 0 or index >= count:
        raise RuntimeError(f"请求设备 {value}，但当前仅检测到 {count} 张 GPU")
    return torch.device(value)


def runtime_summary(chapter: str, entry: str, *, mode: str | None = None) -> str:
    """生成便于日志记录的路径与模式摘要。"""
    run_mode = get_run_mode(mode)
    return (
        f"mode={run_mode} | data={chapter_data_dir(chapter, mode=run_mode)} | "
        f"output={entry_output_dir(chapter, entry, mode=run_mode, create=False)} | "
        f"device={DEFAULT_DEVICE}"
    )
