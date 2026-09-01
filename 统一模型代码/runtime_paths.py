"""统一模型的只读参考输入与 E2E 新输出路径入口。"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runtime_config import (  # noqa: E402
    OUTPUT_ROOT,
    REFERENCE_OUTPUT_ROOT,
    reference_output_path,
    validate_output_path,
    validate_split_roots,
)


def new_run_dir(
    experiment: str,
    run_id: str,
    *,
    create: bool = False,
) -> Path:
    """返回新的统一模型运行目录；拒绝复用已有目标。"""
    if not experiment or not run_id:
        raise ValueError("experiment 和 run_id 均不能为空")
    path = validate_output_path(
        OUTPUT_ROOT / "unified" / Path(experiment).name / Path(run_id).name
    )
    if path.exists():
        raise FileExistsError(f"拒绝复用已有统一模型运行目录: {path}")
    if create:
        path.mkdir(parents=True)
    return path


__all__ = [
    "OUTPUT_ROOT",
    "PROJECT_ROOT",
    "REFERENCE_OUTPUT_ROOT",
    "new_run_dir",
    "reference_output_path",
    "validate_output_path",
    "validate_split_roots",
]
