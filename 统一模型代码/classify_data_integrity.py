"""区分文件字节漂移、读取流不稳定与HDF5科研数据变化。

正常路径只做文件SHA256；仅当稳定的文件SHA与基线不同，才按需解码HDF5
数值dataset并计算与压缩布局无关的逻辑指纹。输入始终只读。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 统一模型代码.runtime_paths import validate_output_path  # noqa: E402


HDF5_SUFFIXES = {".h5", ".hdf5", ".mat"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repeated_file_identity(path: Path, repeat: int) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"文件不存在: {resolved}")
    started = time.perf_counter()
    hashes = [sha256_file(resolved) for _ in range(repeat)]
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256_reads": hashes,
        "read_stream_stable": len(set(hashes)) == 1,
        "duration_seconds": time.perf_counter() - started,
    }


def scientific_datasets(handle: h5py.File) -> Iterator[tuple[str, h5py.Dataset]]:
    rows: list[tuple[str, h5py.Dataset]] = []

    def visitor(name: str, value: h5py.Dataset | h5py.Group) -> None:
        if isinstance(value, h5py.Dataset) and not any(
            part.startswith("#") for part in name.split("/")
        ):
            rows.append((name, value))

    handle.visititems(visitor)
    yield from sorted(rows, key=lambda row: row[0])


def dataset_blocks(dataset: h5py.Dataset, target_bytes: int) -> Iterator[np.ndarray]:
    require(dataset.dtype.kind not in {"O", "V"} or dataset.dtype.fields is not None,
            f"不支持的dataset dtype: {dataset.name} {dataset.dtype}")
    if dataset.ndim == 0:
        yield np.asarray(dataset[()])
        return
    row_values = int(np.prod(dataset.shape[1:], dtype=np.int64)) if dataset.ndim > 1 else 1
    row_bytes = max(1, row_values * int(dataset.dtype.itemsize))
    rows_per_block = max(1, target_bytes // row_bytes)
    for start in range(0, dataset.shape[0], rows_per_block):
        yield np.asarray(dataset[start : start + rows_per_block, ...])


def hdf5_logical_identity(path: Path, target_block_bytes: int) -> dict[str, Any]:
    resolved = path.resolve()
    started = time.perf_counter()
    overall = hashlib.sha256()
    datasets = []
    logical_bytes = 0
    with h5py.File(resolved, "r") as handle:
        for name, dataset in scientific_datasets(handle):
            header = json.dumps(
                {"name": name, "shape": list(dataset.shape), "dtype": str(dataset.dtype)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            current = hashlib.sha256()
            current.update(header)
            for block in dataset_blocks(dataset, target_block_bytes):
                contiguous = np.ascontiguousarray(block)
                payload = contiguous.tobytes(order="C")
                current.update(payload)
                logical_bytes += len(payload)
            digest = current.hexdigest()
            overall.update(name.encode("utf-8"))
            overall.update(bytes.fromhex(digest))
            datasets.append(
                {
                    "name": name,
                    "shape": list(dataset.shape),
                    "dtype": str(dataset.dtype),
                    "logical_sha256": digest,
                }
            )
    require(datasets, f"未发现科研dataset: {resolved}")
    return {
        "path": str(resolved),
        "logical_sha256": overall.hexdigest(),
        "dataset_count": len(datasets),
        "logical_bytes": logical_bytes,
        "datasets": datasets,
        "duration_seconds": time.perf_counter() - started,
    }


def classify(
    candidate: Path,
    baseline: Path,
    repeat: int,
    logical_mode: str,
    target_block_bytes: int,
) -> dict[str, Any]:
    candidate_file = repeated_file_identity(candidate, repeat)
    baseline_file = repeated_file_identity(baseline, 1)
    report: dict[str, Any] = {
        "candidate_file": candidate_file,
        "baseline_file": baseline_file,
        "logical_mode": logical_mode,
        "logical_checked": False,
    }
    if not candidate_file["read_stream_stable"]:
        report.update(
            status="STOP",
            classification="READ_STREAM_UNSTABLE",
            conclusion="同一路径重复读取返回不同字节流；不能据此判定磁盘文件或科研数据已改变。",
        )
        return report

    file_match = (
        candidate_file["size_bytes"] == baseline_file["size_bytes"]
        and candidate_file["sha256_reads"][0] == baseline_file["sha256_reads"][0]
    )
    if file_match and logical_mode != "always":
        report.update(
            status="PASS",
            classification="EXACT_FILE_MATCH",
            conclusion="文件逐字节相同，因此HDF5科研数据也必然相同。",
        )
        return report

    should_check_logical = logical_mode == "always" or (
        logical_mode == "on-mismatch"
        and candidate.suffix.lower() in HDF5_SUFFIXES
        and baseline.suffix.lower() in HDF5_SUFFIXES
    )
    if not should_check_logical:
        report.update(
            status="STOP",
            classification="FILE_BYTES_CHANGED_LOGICAL_UNKNOWN",
            conclusion="文件字节不同；未执行科研逻辑数据比较。",
        )
        return report

    try:
        candidate_logical = hdf5_logical_identity(candidate, target_block_bytes)
        baseline_logical = hdf5_logical_identity(baseline, target_block_bytes)
    except (OSError, RuntimeError, ValueError, AssertionError) as error:
        report.update(
            status="STOP",
            classification="LOGICAL_READ_FAILURE",
            logical_error=f"{type(error).__name__}: {error}",
            conclusion="至少一个HDF5文件不能稳定解码，不能宣称科研数据一致。",
        )
        return report
    report["logical_checked"] = True
    report["candidate_logical"] = candidate_logical
    report["baseline_logical"] = baseline_logical
    logical_match = (
        candidate_logical["logical_sha256"] == baseline_logical["logical_sha256"]
    )
    if logical_match and file_match:
        report.update(
            status="PASS",
            classification="EXACT_FILE_AND_LOGICAL_MATCH",
            conclusion="文件逐字节相同，强制HDF5科研dataset复核也相同。",
        )
    elif logical_match:
        report.update(
            status="PASS_WITH_CONTAINER_DRIFT",
            classification="CONTAINER_DRIFT_LOGICAL_MATCH",
            conclusion="HDF5容器字节不同，但所有科研dataset的名称、shape、dtype和值相同。",
        )
    else:
        report.update(
            status="STOP",
            classification="SCIENTIFIC_DATA_CHANGED",
            conclusion="至少一个科研dataset的结构或数值发生变化。",
        )
    return report


def write_new_json(path: Path, payload: Any) -> None:
    resolved = validate_output_path(path.resolve())
    require(not resolved.exists(), f"拒绝覆盖报告: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="文件SHA与HDF5逻辑身份分级诊断")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument(
        "--logical",
        choices=("on-mismatch", "always", "never"),
        default="on-mismatch",
    )
    parser.add_argument("--target-block-mib", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(args.repeat >= 2, "repeat至少为2，才能识别读取流不稳定")
    require(args.target_block_mib >= 1, "target-block-mib必须为正数")
    started = time.perf_counter()
    report = classify(
        args.candidate.resolve(),
        args.baseline.resolve(),
        args.repeat,
        args.logical,
        args.target_block_mib * 1024 * 1024,
    )
    report.update(
        schema="sourcecount-data-integrity-v1",
        total_duration_seconds=time.perf_counter() - started,
    )
    if args.output is not None:
        write_new_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "classification": report["classification"],
                "logical_checked": report["logical_checked"],
                "total_duration_seconds": report["total_duration_seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
