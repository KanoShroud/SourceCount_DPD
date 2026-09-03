"""基于已验证的无缓冲输入快照恢复 E2E-G1 finalize。

本入口不重跑训练或评价，也不修改原参考目录。只有冻结 manifest 的全部输入
均在本地快照中逐字节匹配，且既有训练/评价证据完整时，才重新生成总报告。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import e2e_g1 as g1


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def canonical_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_new_json(path: Path, payload: Any) -> None:
    resolved = g1.validate_output_path(path)
    require(not resolved.exists(), f"拒绝覆盖恢复证据: {resolved}")
    g1.write_json(resolved, payload)


def verify_snapshot_group(
    expected_rows: list[dict[str, Any]], snapshot_rows: list[dict[str, Any]], group: str
) -> list[dict[str, Any]]:
    require(len(expected_rows) == len(snapshot_rows), f"{group}快照数量不符")
    verified = []
    for index, (expected, row) in enumerate(zip(expected_rows, snapshot_rows)):
        require(row["expected"] == expected, f"{group}[{index}]冻结身份不一致")
        current = g1.file_identity(Path(row["snapshot"]["path"]))
        require(current == row["snapshot"], f"{group}[{index}]快照保存后变化")
        require(current["size_bytes"] == expected["size_bytes"], f"{group}[{index}]大小不符")
        require(current["sha256"] == expected["sha256"], f"{group}[{index}] SHA256不符")
        require(row["copy"]["method"] == "robocopy /J", f"{group}[{index}]不是无缓冲副本")
        verified.append(current)
    return verified


def verify_hdf5_snapshots(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reports = []
    for row in rows:
        path = Path(row["snapshot"]["path"])
        datasets = []
        with h5py.File(path, "r") as handle:
            for name, value in handle.items():
                if isinstance(value, h5py.Dataset):
                    datasets.append(
                        {
                            "name": name,
                            "shape": list(value.shape),
                            "dtype": str(value.dtype),
                            "compression": value.compression,
                        }
                    )
            require(datasets, f"HDF5快照没有dataset: {path}")
        reports.append({"path": str(path.resolve()), "datasets": datasets})
    return reports


def build_integrity_audit(run_root: Path, snapshot_root: Path) -> dict[str, Any]:
    manifest = g1.load_json(run_root / "manifest.json")
    require(manifest["gate"] == "E2E-G1", "manifest Gate错误")
    require(manifest["reference_read_only"] is True, "参考根未标记只读")
    require(manifest["test_executed"] is False, "G1禁止读取test")
    require(g1.code_identity() == manifest["code"], "G1冻结代码或配置已变化")

    snapshot_manifest_path = snapshot_root / "snapshot_manifest.json"
    snapshot_manifest = g1.load_json(snapshot_manifest_path)
    require(snapshot_manifest["status"] == "PASS", "输入快照未通过")
    require(snapshot_manifest["reference_read_only"] is True, "快照未保持参考只读契约")
    require(
        snapshot_manifest["source_manifest"] == g1.file_identity(run_root / "manifest.json"),
        "输入快照未绑定当前G1 manifest",
    )
    files = verify_snapshot_group(
        manifest["inputs"]["files"], snapshot_manifest["files"], "files"
    )
    artifacts = verify_snapshot_group(
        manifest["inputs"]["artifacts"], snapshot_manifest["artifacts"], "artifacts"
    )
    hdf5_reports = verify_hdf5_snapshots(snapshot_manifest["files"])
    selected = manifest["subsets"]["train"]
    require(len(selected) == 256, "G1训练子集不是256条")
    require(len({int(row["local_index"]) for row in selected}) == 256, "G1训练索引重复")
    return {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "validate",
            "verification_status": "VERIFIED_DISK_SNAPSHOT_RECOVERY",
        },
        "status": "PASS_DISK_SNAPSHOT_RECOVERY",
        "gate": "E2E-G1-INPUT-RECOVERY",
        "run_root": str(run_root.resolve()),
        "manifest": g1.file_identity(run_root / "manifest.json"),
        "recovery_code": g1.file_identity(Path(__file__)),
        "snapshot_manifest": g1.file_identity(snapshot_manifest_path),
        "verified_files": files,
        "verified_artifacts": artifacts,
        "hdf5_open_audit": hdf5_reports,
        "selected_train_sample_count": len(selected),
        "selected_indices_sha256": canonical_sha(selected),
        "root_cause_evidence": {
            "classification": "cached_read_path_corruption; exact component not isolated",
            "disk_bytes_changed": False,
            "matlab_generation_or_python_hdf5_semantics_supported": False,
            "reason": (
                "同一参考路径的普通读取曾给出错误SHA和gzip解码失败；robocopy /J无缓冲副本"
                "逐字节匹配冻结SHA，随后普通读取也恢复。LastWriteTime未变，参考输入仅只读打开。"
            ),
        },
        "recovery_scope": (
            "全部G1参考数据和参考checkpoint均经Windows无缓冲复制到outputs_e2e，"
            "副本大小与SHA256逐项匹配冻结manifest；既有训练和评价不重跑。"
        ),
        "test_executed": False,
    }


def verify_completed_evidence(run_root: Path) -> dict[str, Any]:
    training = {}
    for track in ("soft_sg", "soft_e2e"):
        root = run_root / "training" / track
        summary = g1.load_json(root / "training_summary.json")
        require(summary["status"] == "COMPLETED", f"{track}训练未完成")
        require(summary["test_executed"] is False, f"{track}训练读取了test")
        require(summary["epochs_completed"] == 4, f"{track}训练epoch数变化")
        require(summary["optimizer_steps"] == 256, f"{track}优化步数变化")
        checkpoint = Path(summary["selected_checkpoint"])
        require(
            g1.sha256_file(checkpoint) == summary["selected_checkpoint_sha256"],
            f"{track}选定checkpoint身份变化",
        )
        gradients = g1.read_jsonl(root / "gradient_steps.jsonl")
        require(len(gradients) == 256, f"{track}正式梯度日志不是256步")
        if track == "soft_sg":
            require(not any(row["localization_nonzero"] for row in gradients), "Soft-SG出现定位反传")
        else:
            require(all(row["localization_nonzero"] for row in gradients), "Soft-E2E存在定位梯度断路")
        training[track] = {
            "summary": summary,
            "gradient_steps": len(gradients),
            "localization_nonzero_step_rate": float(
                np.mean([bool(row["localization_nonzero"]) for row in gradients])
            ),
        }

    reports = {}
    samples = {}
    for track in ("hard_frozen", "soft_sg", "soft_e2e"):
        root = run_root / "evaluation" / track
        report = g1.load_json(root / "evaluation_report.json")
        require(report["status"] == "COMPLETED", f"{track}评价未完成")
        require(report["test_executed"] is False, f"{track}评价读取了test")
        rows = g1.read_jsonl(root / "samples.jsonl")
        require(len(rows) == 512, f"{track}系统评价不是512条")
        require(len(g1.read_jsonl(root / "progress.jsonl")) == 512, f"{track}系统进度不完整")
        if track == "hard_frozen":
            require(report["oracle_result"] is None, "Hard-frozen不应包含oracle评价")
        else:
            require(report["oracle_result"] is not None, f"{track}缺少oracle评价")
            require(
                len(g1.read_jsonl(root / "oracle_progress.jsonl")) == 512,
                f"{track} oracle进度不完整",
            )
            checkpoint = Path(report["checkpoint"])
            require(g1.sha256_file(checkpoint) == report["checkpoint_sha256"], f"{track}评价checkpoint变化")
        reports[track] = report
        samples[track] = rows
    return {"training": training, "reports": reports, "samples": samples}


def finalize(run_root: Path, audit_path: Path, snapshot_root: Path) -> dict[str, Any]:
    saved_audit = g1.load_json(audit_path)
    require(saved_audit["status"] == "PASS_DISK_SNAPSHOT_RECOVERY", "恢复审计未通过")
    current_audit = build_integrity_audit(run_root, snapshot_root)
    saved_comparable = dict(saved_audit)
    saved_comparable.pop("completed_at", None)
    require(
        canonical_sha(current_audit) == canonical_sha(saved_comparable),
        "恢复审计保存后输入或恢复代码发生变化",
    )

    manifest = g1.load_json(run_root / "manifest.json")
    config = manifest["config"]
    evidence = verify_completed_evidence(run_root)
    reports = evidence["reports"]
    samples = evidence["samples"]
    comparison = g1.paired_bootstrap(samples["soft_sg"], samples["soft_e2e"], config)
    sg = reports["soft_sg"]["result"]
    e2e = reports["soft_e2e"]["result"]
    point = comparison["point"]
    intervals = comparison["bootstrap"]
    checks = {
        "gospa_point_improves": point["gospa_delta_m"] < 0.0,
        "gospa_ci_upper_below_zero": intervals["gospa_delta_m"]["ci95"][1] < 0.0,
        "recall_point_improves": point["recall_100m_delta"] > 0.0,
        "recall_ci_lower_noninferior": (
            intervals["recall_100m_delta"]["ci95"][0]
            >= -float(config["recall_noninferiority_absolute"])
        ),
        "exact_count_noninferior": (
            point["exact_count_delta"]
            >= -float(config["exact_count_noninferiority_absolute"])
        ),
        "ch3_band_noninferior": (
            e2e["ch3"]["active_band_macro_f1"]
            >= sg["ch3"]["active_band_macro_f1"]
            - float(config["ch3_noninferiority_absolute"])
        ),
        "ch3_hard_count_noninferior": (
            e2e["ch3"]["balanced_count_accuracy"]
            >= sg["ch3"]["balanced_count_accuracy"]
            - float(config["ch3_noninferiority_absolute"])
        ),
        "d8_oracle_gospa_noninferior": (
            reports["soft_e2e"]["oracle_result"]["system"]["gospa"]["mean"]
            <= reports["soft_sg"]["oracle_result"]["system"]["gospa"]["mean"]
            * float(config["d8_oracle_gospa_relative_limit"])
        ),
    }
    if all(checks.values()):
        decision = "G1_PASS_TO_G2"
        g2_unlocked = True
    elif point["gospa_delta_m"] < 0.0 and intervals["gospa_delta_m"]["ci95"][1] >= 0.0:
        decision = "G1_INCONCLUSIVE"
        g2_unlocked = False
    else:
        decision = "G1_NO_GO"
        g2_unlocked = False

    payload = {
        "material_passport": {
            "schema": "ARS-9-compatible-local",
            "origin_skill": "experiment-agent",
            "origin_mode": "run/validate",
            "verification_status": "ANALYZED_WITH_DISK_SNAPSHOT_RECOVERY",
        },
        "status": decision,
        "gate": "E2E-G1",
        "run_root": str(run_root.resolve()),
        "tracks": {track: report["result"] for track, report in reports.items()},
        "paired_comparison": comparison,
        "checks": checks,
        "g2_unlocked": g2_unlocked,
        "integrity_recovery": {
            "status": saved_audit["status"],
            "audit_report": g1.file_identity(audit_path),
            "recovery_code": g1.file_identity(Path(__file__)),
            "training_evidence": evidence["training"],
            "scope": saved_audit["recovery_scope"],
        },
        "test_executed": False,
        "scope": "single training seed; fixed K0-3 pilot validation; no test",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_new_json(run_root / "final_report.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E2E-G1输入完整性恢复与独立finalize")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("audit", "finalize"):
        current = subparsers.add_parser(command)
        current.add_argument("--run-root", type=Path, required=True)
        current.add_argument("--snapshot-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = g1.validate_output_path(args.run_root.resolve())
    snapshot_root = g1.validate_output_path(args.snapshot_root.resolve())
    if args.command == "audit":
        report = build_integrity_audit(run_root, snapshot_root)
        report["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        output = run_root / "input_integrity_recovery_report.json"
        write_new_json(output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "verified_files": len(report["verified_files"]),
                    "verified_artifacts": len(report["verified_artifacts"]),
                    "output": str(output.resolve()),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    elif args.command == "finalize":
        audit_path = run_root / "input_integrity_recovery_report.json"
        require(audit_path.is_file(), "缺少完整性恢复审计报告")
        report = finalize(run_root, audit_path, snapshot_root)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "checks": report["checks"],
                    "g2_unlocked": report["g2_unlocked"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
