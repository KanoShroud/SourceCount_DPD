"""项目级运行路径、设备与 smoke/formal 隔离配置。

本模块只负责运行基础设施，不保存算法或实验超参数。环境变量：

- ``SOURCECOUNT_RUN_MODE``: ``smoke`` 或 ``formal``，默认 smoke。
- ``SOURCECOUNT_DATA_ROOT``: 正式数据根目录，默认 ``<project>/data``。
- ``SOURCECOUNT_OUTPUT_ROOT``: 输出根目录，默认 ``<project>/outputs_e2e``。
- ``SOURCECOUNT_REFERENCE_OUTPUT_ROOT``: 原项目冻结 ``outputs`` 根目录；只读。
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
    os.environ.get("SOURCECOUNT_OUTPUT_ROOT", PROJECT_ROOT / "outputs_e2e")
).expanduser().resolve()
_reference_output_value = os.environ.get("SOURCECOUNT_REFERENCE_OUTPUT_ROOT")
REFERENCE_OUTPUT_ROOT = (
    Path(_reference_output_value).expanduser().resolve()
    if _reference_output_value
    else None
)
DEFAULT_DEVICE = os.environ.get("SOURCECOUNT_DEVICE", "cuda:0")


def _is_within(path: Path, root: Path) -> bool:
    """判断解析后的 path 是否位于 root 内（含 root 本身）。"""
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def validate_split_roots(*, require_reference: bool = False) -> None:
    """校验 E2E 可写输出与原项目只读参考根严格分离。"""
    if REFERENCE_OUTPUT_ROOT is None:
        if require_reference:
            raise RuntimeError(
                "未设置 SOURCECOUNT_REFERENCE_OUTPUT_ROOT；"
                "E2E 历史产物读取必须显式配置原项目 outputs"
            )
        return
    if not REFERENCE_OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(f"只读参考根不存在: {REFERENCE_OUTPUT_ROOT}")
    if _is_within(OUTPUT_ROOT, REFERENCE_OUTPUT_ROOT) or _is_within(
        REFERENCE_OUTPUT_ROOT, OUTPUT_ROOT
    ):
        raise RuntimeError(
            "SOURCECOUNT_OUTPUT_ROOT 与 SOURCECOUNT_REFERENCE_OUTPUT_ROOT "
            "必须不同且互不嵌套"
        )


def reference_output_path(*parts: str | os.PathLike[str]) -> Path:
    """解析原项目冻结产物；只检查存在性，绝不创建目录。"""
    validate_split_roots(require_reference=True)
    assert REFERENCE_OUTPUT_ROOT is not None
    path = REFERENCE_OUTPUT_ROOT.joinpath(*parts).resolve()
    if not _is_within(path, REFERENCE_OUTPUT_ROOT):
        raise ValueError(f"参考路径越出冻结根目录: {path}")
    if not path.exists():
        raise FileNotFoundError(f"冻结参考产物不存在: {path}")
    return path


def validate_output_path(path: Path) -> Path:
    """确认写入目标位于 E2E 输出根，且不在冻结参考根内。"""
    resolved = path.expanduser().resolve()
    if not _is_within(resolved, OUTPUT_ROOT):
        raise ValueError(f"写入目标越出 E2E 输出根: {resolved}")
    validate_split_roots()
    if REFERENCE_OUTPUT_ROOT is not None and _is_within(
        resolved, REFERENCE_OUTPUT_ROOT
    ):
        raise RuntimeError(f"拒绝写入原项目冻结 outputs: {resolved}")
    return resolved


def validate_data_path(path: Path) -> Path:
    """确认可写数据目标不位于原项目冻结 outputs 内。"""
    resolved = path.expanduser().resolve()
    validate_split_roots()
    if REFERENCE_OUTPUT_ROOT is not None and _is_within(
        resolved, REFERENCE_OUTPUT_ROOT
    ):
        raise RuntimeError(f"拒绝向原项目冻结 outputs 写入数据: {resolved}")
    return resolved


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
    """返回章节数据目录；smoke 数据永远位于 outputs_e2e/smoke。"""
    run_mode = get_run_mode(mode)
    if run_mode == "smoke":
        path = validate_output_path(OUTPUT_ROOT / "smoke" / chapter / "data")
    else:
        path = validate_data_path(DATA_ROOT / chapter)
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
    path = validate_output_path(
        OUTPUT_ROOT / run_mode / chapter / Path(entry).stem
    )
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
    reference = (
        str(REFERENCE_OUTPUT_ROOT)
        if REFERENCE_OUTPUT_ROOT is not None
        else "<not configured>"
    )
    return (
        f"mode={run_mode} | data={chapter_data_dir(chapter, mode=run_mode)} | "
        f"output={entry_output_dir(chapter, entry, mode=run_mode, create=False)} | "
        f"reference_outputs={reference} | device={DEFAULT_DEVICE}"
    )
