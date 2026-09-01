"""核对原项目冻结产物和 E2E 输出根隔离，不加载模型或运行实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from .reference_artifacts import ARTIFACTS, CORE_ARTIFACTS, verify_artifacts
    from .runtime_paths import (
        OUTPUT_ROOT,
        REFERENCE_OUTPUT_ROOT,
        validate_output_path,
        validate_split_roots,
    )
else:
    from reference_artifacts import ARTIFACTS, CORE_ARTIFACTS, verify_artifacts
    from runtime_paths import (
        OUTPUT_ROOT,
        REFERENCE_OUTPUT_ROOT,
        validate_output_path,
        validate_split_roots,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        choices=tuple(ARTIFACTS),
        help="只核对指定产物；可重复传入。默认核对全部核心产物。",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="可选报告路径；必须位于 outputs_e2e 且目标不存在。",
    )
    parser.add_argument("--list", action="store_true", help="列出注册产物后退出。")
    return parser.parse_args()


def build_report(names: list[str]) -> dict[str, Any]:
    validate_split_roots(require_reference=True)
    return {
        "status": "PASS",
        "stage": "e2e_reference_preflight",
        "reference_output_root": str(REFERENCE_OUTPUT_ROOT),
        "reference_read_only": True,
        "output_root": str(OUTPUT_ROOT),
        "artifacts": verify_artifacts(names),
        "model_loaded": False,
        "experiment_executed": False,
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path = validate_output_path(path)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖预检报告: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if args.list:
        print("\n".join(ARTIFACTS))
        return 0
    names = args.artifact or list(CORE_ARTIFACTS)
    report = build_report(names)
    if args.report is not None:
        write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
