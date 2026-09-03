"""将只读参考输入以 Windows 无缓冲方式固化到 outputs_e2e。

普通文件读取可能命中异常的系统缓存页；本入口使用 ``robocopy /J`` 从磁盘
读取，再以冻结 manifest 中的大小和 SHA256 验证本地副本。任何不一致都会
停止，且不会覆盖既有快照或修改参考目录。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def write_new_json(path: Path, payload: Any) -> None:
    require(not path.exists(), f"拒绝覆盖: {path}")
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def copy_unbuffered(source: Path, target_dir: Path) -> tuple[Path, dict[str, Any]]:
    require(source.is_file(), f"参考输入不存在: {source}")
    require(not target_dir.exists(), f"快照子目录已存在: {target_dir}")
    target_dir.mkdir(parents=True)
    command = [
        "robocopy",
        str(source.parent),
        str(target_dir),
        source.name,
        "/J",
        "/R:0",
        "/W:0",
        "/COPY:DAT",
        "/DCOPY:T",
        "/NFL",
        "/NDL",
        "/NP",
    ]
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8:replace"
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="gbk",
        errors="replace",
        env=environment,
    )
    log_path = target_dir / "robocopy.log"
    log_path.write_text(
        completed.stdout + completed.stderr,
        encoding="utf-8",
    )
    require(completed.returncode <= 7, f"robocopy失败({completed.returncode}): {source}")
    target = target_dir / source.name
    require(target.is_file(), f"robocopy未生成目标: {target}")
    return target, {
        "method": "robocopy /J",
        "returncode": completed.returncode,
        "log": str(log_path.resolve()),
    }


def materialize_group(
    rows: list[dict[str, Any]], group: str, snapshot_root: Path
) -> list[dict[str, Any]]:
    results = []
    for index, expected in enumerate(rows):
        source = Path(expected["path"]).resolve()
        target, copy_report = copy_unbuffered(
            source,
            snapshot_root / group / f"{index:03d}",
        )
        snapshot = identity(target)
        require(snapshot["size_bytes"] == expected["size_bytes"], f"副本大小不符: {source}")
        require(snapshot["sha256"] == expected["sha256"], f"副本SHA256不符: {source}")
        results.append(
            {
                "expected": expected,
                "snapshot": snapshot,
                "copy": copy_report,
            }
        )
        print(
            json.dumps(
                {
                    "group": group,
                    "index": index,
                    "source": str(source),
                    "sha256": snapshot["sha256"],
                    "status": "VERIFIED",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="无缓冲固化参考输入快照")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="不复制，只复核已有快照与冻结manifest；用于阶段前后门禁",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    snapshot_root = args.snapshot_root.resolve()
    require(manifest_path.is_file(), f"manifest不存在: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("reference_read_only") is True, "manifest未声明参考根只读")
    require(manifest.get("test_executed") is False, "拒绝处理已读取test的manifest")
    if args.verify_only:
        report_path = snapshot_root / "snapshot_manifest.json"
        require(report_path.is_file(), f"快照manifest不存在: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        require(report["source_manifest"] == identity(manifest_path), "快照未绑定当前manifest")
        for group in ("files", "artifacts"):
            expected_rows = manifest["inputs"][group]
            require(len(expected_rows) == len(report[group]), f"{group}快照数量不符")
            for expected, row in zip(expected_rows, report[group]):
                require(row["expected"] == expected, f"{group}冻结身份不符")
                current = identity(Path(row["snapshot"]["path"]))
                require(current == row["snapshot"], f"{group}快照身份变化")
                require(current["sha256"] == expected["sha256"], f"{group}快照SHA变化")
        print(json.dumps({"status": "PASS", "mode": "verify-only"}, ensure_ascii=False))
        return
    require(not snapshot_root.exists(), f"快照根已存在: {snapshot_root}")
    snapshot_root.mkdir(parents=True)
    files = materialize_group(manifest["inputs"]["files"], "files", snapshot_root)
    artifacts = materialize_group(
        manifest["inputs"]["artifacts"], "artifacts", snapshot_root
    )
    report = {
        "status": "PASS",
        "method": "Windows unbuffered reference materialization",
        "source_manifest": identity(manifest_path),
        "reference_read_only": True,
        "files": files,
        "artifacts": artifacts,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    report_path = snapshot_root / "snapshot_manifest.json"
    write_new_json(report_path, report)
    print(json.dumps({"status": "PASS", "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
