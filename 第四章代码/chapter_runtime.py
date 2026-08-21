"""第四章脚本到项目级 ``runtime_config`` 的轻量适配层。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_config import (  # noqa: E402, F401
    DEFAULT_DEVICE,
    artifact_path,
    chapter_data_dir,
    checkpoint_path as project_checkpoint_path,
    entry_output_dir,
    resolve_torch_device,
    runtime_summary,
)


CHAPTER = "chapter4"


def data_dir(*, create: bool = False) -> Path:
    return chapter_data_dir(CHAPTER, create=create)


def output_path(entry: str, filename: str) -> Path:
    return artifact_path(CHAPTER, entry, filename)


def output_dir(entry: str, *, create: bool = True) -> Path:
    return entry_output_dir(CHAPTER, entry, create=create)


def checkpoint_path(trainer: str, filename: str) -> Path:
    return project_checkpoint_path(CHAPTER, trainer, filename)


def device(requested: str | None = None):
    return resolve_torch_device(requested)


def summary(entry: str) -> str:
    return runtime_summary(CHAPTER, entry)
