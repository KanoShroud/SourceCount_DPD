"""E2E-G1原CH3硬槽位计数最终解码诊断。

复用G1已保存的有序Top-K位置，仅把最终K从Poisson-binomial解码改为
原CH3硬槽位计数。由于诊断前强制检查hard_count不大于原predicted_count，
截取已有位置前缀与重新执行相同热力图的Top-hard_count严格等价。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import h5py
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 统一模型代码 import e2e_g1 as g1
from 统一模型代码.runtime_paths import OUTPUT_ROOT, validate_output_path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.resolve().read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def write_json(path: Path, payload: Any) -> None:
    resolved = validate_output_path(path)
    require(not resolved.exists(), f"拒绝覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    resolved = validate_output_path(path)
    require(not resolved.exists(), f"拒绝覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def verify_snapshot(g1_root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = g1_root / "manifest.json"
    manifest = load_json(manifest_path)
    snapshot_path = g1_root / "reference_snapshot" / "snapshot_manifest.json"
    snapshot = load_json(snapshot_path)
    require(manifest.get("reference_read_only") is True, "G1未声明参考输入只读")
    require(manifest.get("test_executed") is False, "G1不应读取test")
    require(snapshot.get("status") == "PASS", "参考快照未通过")
    require(snapshot["source_manifest"] == identity(manifest_path), "快照未绑定当前G1 manifest")
    for group in ("files", "artifacts"):
        require(len(snapshot[group]) == len(manifest["inputs"][group]), f"{group}快照数量变化")
        for expected, row in zip(manifest["inputs"][group], snapshot[group]):
            require(row["expected"] == expected, f"{group}冻结身份不一致")
            current = identity(Path(row["snapshot"]["path"]))
            require(current == row["snapshot"], f"{group}快照身份变化")
            require(current["sha256"] == expected["sha256"], f"{group}快照SHA变化")
    val_compare_expected = str(g1.COARSE_VAL_COMPARE.resolve())
    matches = [
        Path(row["snapshot"]["path"])
        for row in snapshot["files"]
        if str(Path(row["expected"]["path"]).resolve()) == val_compare_expected
    ]
    require(len(matches) == 1, "无法唯一定位val_compare快照")
    return snapshot, matches[0].resolve()


def source_evidence(g1_root: Path, track: str) -> dict[str, Any]:
    summary_path = g1_root / "training" / track / "training_summary.json"
    report_path = g1_root / "evaluation" / track / "evaluation_report.json"
    samples_path = g1_root / "evaluation" / track / "samples.jsonl"
    summary = load_json(summary_path)
    report = load_json(report_path)
    checkpoint_path = Path(summary["selected_checkpoint"]).resolve()
    require(summary["status"] == "COMPLETED", f"{track}训练未完成")
    require(int(summary["selected_epoch"]) == 1, f"{track}选择epoch不是1")
    require(identity(checkpoint_path)["sha256"] == summary["selected_checkpoint_sha256"], f"{track} checkpoint身份变化")
    require(report["status"] == "COMPLETED", f"{track}评价未完成")
    require(report["checkpoint_sha256"] == summary["selected_checkpoint_sha256"], f"{track}评价未绑定选择checkpoint")
    require(report.get("test_executed") is False, f"{track}评价读取了test")
    return {
        "training_summary": identity(summary_path),
        "evaluation_report": identity(report_path),
        "samples": identity(samples_path),
        "checkpoint": identity(checkpoint_path),
    }


def true_positions_by_local_index(
    coarse_path: Path, records: list[dict[str, Any]]
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    with h5py.File(coarse_path, "r") as handle:
        for record in records:
            local_index = int(record["local_index"])
            true_count = int(record["true_k"])
            stored_count = int(np.asarray(handle["src_count_all"][:, local_index]).item())
            require(stored_count == true_count, f"val_compare K不一致: {local_index}")
            positions = np.asarray(
                handle["src_pos_all"][:, :, local_index], dtype=np.float32
            ).T[:true_count]
            result[local_index] = positions
    return result


def redecode_track(
    source_rows: list[dict[str, Any]],
    truth: dict[int, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    matched_errors: list[float] = []
    for source in source_rows:
        original_count = int(source["predicted_count"])
        hard_count = int(source["hard_count"])
        positions = np.asarray(source["predicted_positions_m"], dtype=np.float32)
        require(len(positions) == original_count, "原位置集合长度与Soft K不一致")
        require(0 <= hard_count <= original_count <= g1.MAX_TRUE_SRC, "无法用已有Top-K前缀精确重解码")
        predicted = positions[:hard_count]
        true_positions = truth[int(source["local_index"])]
        gospa = g1.gospa_sample(true_positions, predicted)
        matched_errors.extend(
            item[2] for item in g1.matched_distances(true_positions, predicted)
        )
        row = deepcopy(source)
        row["source_predicted_count"] = original_count
        row["predicted_count"] = hard_count
        row["hard_count_mismatch"] = False
        row["predicted_positions_m"] = predicted.tolist()
        row["gospa_m"] = gospa["value_m"]
        row["gospa_localization_p_sum"] = gospa["localization_p_sum"]
        row["gospa_missed_p_sum"] = gospa["missed_p_sum"]
        row["gospa_false_p_sum"] = gospa["false_p_sum"]
        for threshold in (10, 30, 50, 100):
            row[f"tp_at_{threshold}m"] = g1.maximum_matches_within(
                true_positions, predicted, float(threshold)
            )
        rows.append(row)
    return g1.summarize_track(rows, matched_errors), rows


def relabel_comparison(payload: dict[str, Any], label: str) -> dict[str, Any]:
    result = deepcopy(payload)
    result["comparison"] = label
    return result


def run(g1_root: Path, run_id: str) -> Path:
    g1_root = g1_root.resolve()
    require(g1_root.is_dir(), f"G1目录不存在: {g1_root}")
    final = load_json(g1_root / "final_report.json")
    manifest = load_json(g1_root / "manifest.json")
    require(final.get("status") == "G1_NO_GO", "源G1不是NO_GO")
    require(final.get("g2_unlocked") is False, "源G1错误开放G2")
    require(final.get("test_executed") is False, "源G1读取了test")
    _, coarse_path = verify_snapshot(g1_root)
    evidence = {track: source_evidence(g1_root, track) for track in ("soft_sg", "soft_e2e")}

    output_root = validate_output_path(
        OUTPUT_ROOT / "unified" / "e2e_g1_hard_count_decode" / Path(run_id).name
    )
    require(not output_root.exists(), f"拒绝复用诊断目录: {output_root}")
    output_root.mkdir(parents=True)
    diagnostic_manifest = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run/validate",
            "verification_status": "UNVERIFIED",
        },
        "status": "PREPARED",
        "diagnostic": "E2E-G1-HARD-COUNT-DECODE",
        "source_g1_root": str(g1_root),
        "source_g1_final_report": identity(g1_root / "final_report.json"),
        "source_evidence": evidence,
        "val_compare_snapshot": identity(coarse_path),
        "code": {
            "diagnostic": identity(Path(__file__)),
            "e2e_g1": identity(Path(g1.__file__)),
            "final_decoder": identity(g1.PACKAGE_ROOT / "models" / "final_decoder.py"),
        },
        "contract": {
            "training": False,
            "new_model_inference": False,
            "test_executed": False,
            "internal_soft_dpd_unchanged": True,
            "final_k": "original CH3 hard slot count",
            "position_decode": "prefix of saved stable Top-K; require hard_count <= original soft count",
            "scope": "same fixed 512-sample val_compare; post-hoc failure attribution only",
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(output_root / "manifest.json", diagnostic_manifest)

    source_rows = {
        track: g1.read_jsonl(g1_root / "evaluation" / track / "samples.jsonl")
        for track in ("soft_sg", "soft_e2e")
    }
    expected_records = manifest["subsets"]["val_compare"]
    expected_identity = [(int(row["local_index"]), int(row["true_k"])) for row in expected_records]
    for track, rows in source_rows.items():
        require(len(rows) == 512, f"{track}原评价不是512条")
        require(
            [(int(row["local_index"]), int(row["true_count"])) for row in rows]
            == expected_identity,
            f"{track}样本身份/顺序与manifest不一致",
        )
    truth = true_positions_by_local_index(coarse_path, expected_records)

    decoded: dict[str, dict[str, Any]] = {}
    decoded_rows: dict[str, list[dict[str, Any]]] = {}
    for track in ("soft_sg", "soft_e2e"):
        system, rows = redecode_track(source_rows[track], truth)
        decoded[track] = system
        decoded_rows[track] = rows
        write_jsonl(output_root / track / "samples.jsonl", rows)
        print(
            json.dumps(
                {
                    "track": track,
                    "samples": len(rows),
                    "gospa_m": system["gospa"]["mean"],
                    "exact_count_rate": system["exact_count_rate"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    original_reports = {
        track: load_json(g1_root / "evaluation" / track / "evaluation_report.json")["result"]
        for track in ("hard_frozen", "soft_sg", "soft_e2e")
    }
    comparisons = {
        "soft_e2e_minus_soft_sg_hard_count": relabel_comparison(
            g1.paired_bootstrap(decoded_rows["soft_sg"], decoded_rows["soft_e2e"], manifest["config"]),
            "soft_e2e_minus_soft_sg_hard_count",
        ),
        "soft_sg_hard_count_minus_original": relabel_comparison(
            g1.paired_bootstrap(source_rows["soft_sg"], decoded_rows["soft_sg"], manifest["config"]),
            "soft_sg_hard_count_minus_original_poisson_binomial",
        ),
        "soft_e2e_hard_count_minus_original": relabel_comparison(
            g1.paired_bootstrap(source_rows["soft_e2e"], decoded_rows["soft_e2e"], manifest["config"]),
            "soft_e2e_hard_count_minus_original_poisson_binomial",
        ),
    }
    hard_baseline = float(original_reports["hard_frozen"]["system"]["gospa"]["mean"])
    attribution = {}
    for track in ("soft_sg", "soft_e2e"):
        original_gospa = float(original_reports[track]["system"]["gospa"]["mean"])
        decoded_gospa = float(decoded[track]["gospa"]["mean"])
        gap = original_gospa - hard_baseline
        attribution[track] = {
            "original_poisson_binomial_gospa_m": original_gospa,
            "hard_count_gospa_m": decoded_gospa,
            "hard_frozen_gospa_m": hard_baseline,
            "gospa_reduction_m": original_gospa - decoded_gospa,
            "original_soft_vs_hard_gap_m": gap,
            "gap_removed_fraction": (original_gospa - decoded_gospa) / gap if gap > 0 else None,
            "exact_count_rate_before": float(original_reports[track]["system"]["exact_count_rate"]),
            "exact_count_rate_after": float(decoded[track]["exact_count_rate"]),
        }

    verify_snapshot(g1_root)
    report = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run/validate",
            "verification_status": "VERIFIED",
        },
        "status": "DIAGNOSIS_COMPLETED",
        "diagnostic": "E2E-G1-HARD-COUNT-DECODE",
        "source_g1_status_unchanged": "G1_NO_GO",
        "g2_unlocked": False,
        "tracks": decoded,
        "comparisons": comparisons,
        "attribution": attribution,
        "integrity": {
            "snapshot_verified_before_and_after": True,
            "source_topk_prefix_contract": True,
            "source_evidence": evidence,
        },
        "scope": "post-hoc fixed-val_compare failure attribution; no training, no new inference, no test",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json(output_root / "diagnostic_report.json", report)
    print(json.dumps({"status": report["status"], "output": str(output_root)}, ensure_ascii=False), flush=True)
    return output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E-G1原CH3硬计数最终解码诊断")
    parser.add_argument("--g1-run-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.g1_run_root, args.run_id)
