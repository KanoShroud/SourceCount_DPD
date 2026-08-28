"""S2-G4轻量空载RAM基线；必须在导入PyTorch的身份预检之前运行。"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import psutil


def main() -> int:
    parser = argparse.ArgumentParser(description="S2-G4 60秒轻量资源基线")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--interval_seconds", type=float, default=5.0)
    parser.add_argument("--threshold_gib", type=float, default=14.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0 or args.interval_seconds < 0 or args.threshold_gib <= 0:
        parser.error("samples和threshold_gib必须为正，interval_seconds不得为负")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有基线: {output}")

    values = []
    for index in range(args.samples):
        values.append(int(psutil.virtual_memory().available))
        if index + 1 < args.samples and args.interval_seconds:
            time.sleep(args.interval_seconds)
    median = float(statistics.median(values))
    threshold = args.threshold_gib * 1024**3
    payload = {
        "status": "PASS" if median >= threshold else "INSUFFICIENT_RAM",
        "stage": "s2g4_lightweight_resource_baseline",
        "samples": values,
        "sample_count": len(values),
        "interval_seconds": args.interval_seconds,
        "median_available_bytes": median,
        "threshold_bytes": threshold,
        "threshold_gib": args.threshold_gib,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
