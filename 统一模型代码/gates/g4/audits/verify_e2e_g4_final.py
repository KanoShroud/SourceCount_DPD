"""只读审计E2E-G4最终RAM-only证据与预注册判定。"""

from __future__ import annotations

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[4]))

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


TRACKS = ("a0_large_frozen", "a1_d8_tail", "a2_joint_tail")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        while block := handle.read(8 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    config = read_json(root / "manifest.json")["config"]
    selection = read_json(root / "p1_selection_report.json")
    p2 = read_json(root / "p2_report.json")
    final = read_json(root / "final_report.json")
    rebuild = read_json(root / "in_memory_rebuild_report.json")
    prior_rebuild = read_json(root / "invalidated_v3_reports" / "in_memory_rebuild_report.json")
    expected_reports = [
        root / "training_v4" / "20260907" / track / "training_report_epoch16.json"
        for track in TRACKS
    ]
    for seed in ("20260907", "20260908"):
        for track in ("a0_large_frozen", "a2_joint_tail"):
            expected_reports.extend(
                root / "training_v4" / seed / track / f"training_report_epoch{epoch}.json"
                for epoch in (24, 32)
            )
    stable_paths = expected_reports + [
        root / "training_v4" / seed / track / filename
        for seed in ("20260907", "20260908")
        for track in ("a0_large_frozen", "a2_joint_tail")
        for filename in ("best.pt", "last.pt")
    ] + [
        root / "training_v4" / "20260907" / "a1_d8_tail" / filename
        for filename in ("best.pt", "last.pt")
    ] + [
        root / name
        for name in (
            "manifest.json",
            "p1_selection_report.json",
            "p2_report.json",
            "final_report.json",
            "in_memory_rebuild_report.json",
            "eager_loader_contract.json",
        )
    ]
    unstable = []
    for path in stable_paths:
        values = [digest(path) for _ in range(3)]
        if len(set(values)) != 1:
            unstable.append({"path": str(path), "sha256_reads": values})

    seed_checks = {}
    for seed in ("20260907", "20260908"):
        base = p2["results"][seed]["a0_large_frozen"]["metrics"]["overall"]
        candidate = p2["results"][seed]["a2_joint_tail"]["metrics"]["overall"]
        diff = p2["results"][seed]["paired_bootstrap"]["gospa_candidate_minus_a0_m"]["mean"]
        seed_checks[seed] = {
            "gospa_improved": diff < 0.0,
            "recall_noninferior": candidate["recall_at_100m"] - base["recall_at_100m"]
            >= float(config["recall_100m_noninferiority"]),
            "coverage_noninferior": candidate["matched_pair_coverage_of_true"]
            - base["matched_pair_coverage_of_true"]
            >= float(config["coverage_noninferiority"]),
            "rmse_noninferior": candidate["matched_errors_m"]["rmse"]
            / base["matched_errors_m"]["rmse"]
            <= float(config["rmse_ratio_maximum"]),
        }
    checks = {
        "candidate_is_a2": selection["candidate"] == "a2_joint_tail",
        "p2_status_pass": p2["status"] == "PASS",
        "classification_recomputed": p2["classification"] == "G4_PARTIAL_UNFREEZE_SELECTED"
        and all(all(values.values()) for values in seed_checks.values()),
        "final_consistent": final["status"] == p2["classification"]
        and final["next_gate_unlocked"] is True,
        "test_never_executed": final["test_executed"] is False
        and p2["test_executed"] is False,
        "all_expected_reports_exist": all(path.exists() for path in expected_reports),
        "all_json_numeric_values_finite": finite(selection) and finite(p2) and finite(final),
        "evidence_repeated_sha_stable": not unstable,
        "ram_rebuild_pass": rebuild["status"] == "PASS",
        "ram_rebuild_repeated_exactly": rebuild["splits"] == prior_rebuild["splits"],
        "disk_cache_runs_preserved_as_invalid": (root / "invalidated_v2_reports").exists()
        and (root / "invalidated_v3_reports").exists(),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "STOP",
        "schema": "e2e-g4-final-audit-v1",
        "checks": checks,
        "seed_gate_checks": seed_checks,
        "stable_evidence_file_count": len(stable_paths),
        "unstable_evidence": unstable,
        "test_executed": False,
    }
    output = root / "final_audit_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
