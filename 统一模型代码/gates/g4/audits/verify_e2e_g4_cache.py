"""只读复核E2E-G4固定DPD与分片特征缓存身份。"""

from __future__ import annotations

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def verify(expected: dict[str, Any]) -> dict[str, Any]:
    current = identity(Path(expected["path"]))
    return {
        "expected": {key: expected[key] for key in ("path", "size_bytes", "sha256")},
        "current": current,
        "match": current["size_bytes"] == expected["size_bytes"] and current["sha256"] == expected["sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    dpd = read_json(run_root / "dpd_cache_manifest.json")
    features = read_json(run_root / "feature_cache_v2_manifest.json")
    dpd_rows = [verify(row) for rows in dpd["files"].values() for row in rows]
    feature_rows = []
    for split in features["files"].values():
        for name, rows in split.items():
            if name == "targets":
                feature_rows.append(verify(rows))
            else:
                feature_rows.extend(verify(row) for row in rows)
    report = {
        "status": "PASS" if all(row["match"] for row in dpd_rows + feature_rows) else "STOP",
        "schema": "e2e-g4-cache-postcheck-v1",
        "dpd_file_count": len(dpd_rows),
        "feature_file_count": len(feature_rows),
        "dpd_all_match": all(row["match"] for row in dpd_rows),
        "feature_all_match": all(row["match"] for row in feature_rows),
        "test_executed": False,
        "mismatches": [row for row in dpd_rows + feature_rows if not row["match"]],
    }
    output = run_root / "cache_postcheck_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
