"""S2-G4-R4 数据清单与三表示数据审计工具。

本文件不生成信号、不训练模型，只建立冻结样本清单并验证 R4-D 输出。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from s2g3_composability import load_ch4_mat
from s2g4_coarse_d8 import run_evaluate


MANIFEST_SEED = 20260826
REQUIRED_D8_FIELDS = {
    "fine_dpd", "gauss_label", "pos_label", "n_src", "sample_idx", "group_idx",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"拒绝覆盖已有文件: {path}")
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _round_robin_strata(raw: dict[str, Any], source_count: int, seed: int) -> list[int]:
    candidates = np.flatnonzero(raw["src_count"] == source_count)
    snr = np.asarray(raw["avg_snr"])[candidates]
    bandwidth = np.asarray(raw["bw_actual"])[candidates, :source_count]
    union_proxy = bandwidth.max(axis=1) + bandwidth.min(axis=1)

    def quartiles(values: np.ndarray) -> np.ndarray:
        edges = np.quantile(values, [0.25, 0.5, 0.75])
        return np.digitize(values, edges, right=True)

    strata: dict[tuple[int, int], list[int]] = {}
    for index, snr_bin, bw_bin in zip(candidates, quartiles(snr), quartiles(union_proxy)):
        strata.setdefault((int(snr_bin), int(bw_bin)), []).append(int(index))
    rng = np.random.default_rng(seed + source_count)
    for values in strata.values():
        rng.shuffle(values)
    ordered: list[int] = []
    keys = sorted(strata)
    while any(strata.values()):
        for key in keys:
            if strata[key]:
                ordered.append(strata[key].pop())
    require(len(ordered) == len(candidates) == len(set(ordered)), "分层排序丢失或重复样本")
    return ordered


def paired_indices(raw: dict[str, Any], total: int, seed: int) -> list[int]:
    require(total % 2 == 0, "R4清单要求N=2/N=3各半")
    two = _round_robin_strata(raw, 2, seed)
    three = _round_robin_strata(raw, 3, seed)
    half = total // 2
    require(len(two) >= half and len(three) >= half, "N=2或N=3样本不足")
    result = [value for pair in zip(two[:half], three[:half]) for value in pair]
    require(len(result) == total and len(set(result)) == total, "清单索引不唯一")
    return result


def run_manifests(args: argparse.Namespace) -> dict[str, Any]:
    raw_dir = args.raw_iq_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空manifest目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train = load_ch4_mat(raw_dir / "train_data.mat", include_iq=False)
    val = load_ch4_mat(raw_dir / "val_data.mat", include_iq=False)
    require(len(train["src_count"]) == 8192 and len(val["src_count"]) == 2048,
            "R4原始样本数不匹配")
    train_order = paired_indices(train, 8192, MANIFEST_SEED)
    val_order = paired_indices(val, 2048, MANIFEST_SEED + 100)
    common = {
        "gate": "S2-G4-R4",
        "manifest_seed": MANIFEST_SEED,
        "stratification": "N=2/3 exact balance; round-robin SNR and bandwidth quartiles",
    }
    manifests = {
        "train_1024": ("train", train_order[:1024]),
        "train_4096": ("train", train_order[:4096]),
        "train_8192": ("train", train_order),
        "val_select": ("val", val_order[:1024]),
        "val_compare": ("val", val_order[1024:]),
    }
    identities: dict[str, Any] = {}
    for name, (split, indices) in manifests.items():
        counts = np.asarray((train if split == "train" else val)["src_count"])[indices]
        payload = {
            **common,
            "name": name,
            "split": split,
            "sample_count": len(indices),
            "n2": int(np.sum(counts == 2)),
            "n3": int(np.sum(counts == 3)),
            "indices": indices,
        }
        path = output_dir / f"{name}.json"
        write_json(path, payload)
        identities[name] = {"path": str(path), "sha256": sha256(path), "sample_count": len(indices)}
    require(set(train_order[:1024]).issubset(train_order[:4096]), "1k不是4k子集")
    require(set(train_order[:4096]).issubset(train_order), "4k不是8k子集")
    require(set(val_order[:1024]).isdisjoint(val_order[1024:]), "validation两部分重叠")
    report = {"status": "PASS", "manifests": identities}
    write_json(output_dir / "manifest_audit.json", report)
    return report


def load_index(data_dir: Path, split: str) -> tuple[Path, dict[str, Any]]:
    split_dir = data_dir.resolve() / split
    index = torch.load(split_dir / f"loc_{split}_index.pt", map_location="cpu", weights_only=False)
    return split_dir, index


def run_audit_dpd(args: argparse.Namespace) -> dict[str, Any]:
    roots = {"exact": args.exact, "hard_actual": args.hard, "soft19_actual": args.soft}
    report: dict[str, Any] = {"status": "PASS", "splits": {}}
    for split, expected in (("train", 8192), ("val", 2048)):
        locations = {name: load_index(path, split) for name, path in roots.items()}
        shard_lists = {name: value[1]["shard_files"] for name, value in locations.items()}
        require(len({tuple(value) for value in shard_lists.values()}) == 1,
                f"{split}三表示分片清单不一致")
        summary = {
            name: {"count": 0, "fields": set(), "finite": True,
                   "size_bytes": sum(p.stat().st_size for p in directory.glob("*.pt"))}
            for name, (directory, _) in locations.items()
        }
        sample_indices: list[int] = []
        exact_hard_different = 0
        hard_soft_different = 0
        hard_soft_identical_samples: list[int] = []
        for names in zip(*(shard_lists[name] for name in roots)):
            shards = {
                name: torch.load(locations[name][0] / shard_name, map_location="cpu",
                                 weights_only=False)
                for name, shard_name in zip(roots, names)
            }
            reference = shards["exact"]
            for name, shard in shards.items():
                summary[name]["fields"].update(shard)
                require(REQUIRED_D8_FIELDS.issubset(shard), f"{name}/{split}缺少D8字段")
                summary[name]["finite"] &= bool(torch.isfinite(shard["fine_dpd"]).all())
                summary[name]["count"] += len(shard["n_src"])
                if name != "exact":
                    for field in ("gauss_label", "pos_label", "n_src", "sample_idx", "group_idx"):
                        require(torch.equal(shard[field], reference[field]),
                                f"{name}/{split}/{names[0]}字段{field}漂移")
            sample_indices.extend(int(value) for value in reference["sample_idx"])
            exact_hard_different += int(torch.any(
                reference["fine_dpd"] != shards["hard_actual"]["fine_dpd"], dim=(1, 2, 3)
            ).sum())
            hard_soft_different += int(torch.any(
                shards["hard_actual"]["fine_dpd"] != shards["soft19_actual"]["fine_dpd"],
                dim=(1, 2, 3),
            ).sum())
            equal_rows = torch.all(
                shards["hard_actual"]["fine_dpd"] == shards["soft19_actual"]["fine_dpd"],
                dim=(1, 2, 3),
            )
            hard_soft_identical_samples.extend(
                int(value) for value in reference["sample_idx"][equal_rows]
            )
            del shards
        for name, values in summary.items():
            require(values["count"] == expected and values["finite"],
                    f"{name}/{split}数量或有限性门禁失败")
            values["fields"] = sorted(values["fields"])
            del values["finite"]
        require(sample_indices == list(range(expected)), f"{split}样本顺序不连续")
        require(exact_hard_different == expected,
                f"{split} Exact-Hard存在意外相同样本: {exact_hard_different}/{expected}")
        summary["paired_checks"] = {
            "supervision_identical": True,
            "exact_vs_hard_different_samples": exact_hard_different,
            "hard_vs_soft_different_samples": hard_soft_different,
            "hard_vs_soft_identical_samples": hard_soft_identical_samples,
            "hard_soft_identity_interpretation": (
                "允许特定样本在唯一支持内仅相差统一频率权重；DPD功率归一化会抵消统一缩放"
            ),
        }
        report["splits"][split] = summary
    write_json(args.output.resolve(), report)
    return report


def run_r4_evaluate(args: argparse.Namespace) -> dict[str, Any]:
    payload = run_evaluate(SimpleNamespace(
        seed=42,
        device="cuda:0",
        data_dir=args.data_dir,
        manifest=args.manifest,
        expected_samples=1024,
        batch_size=8,
        checkpoint=args.checkpoint,
        track=args.track,
        data_mode=args.data_mode,
        samples_jsonl=args.samples_jsonl,
        output=args.output,
    ))
    payload["gate"] = "S2-G4-R4-E"
    payload["selection_split"] = "val_select"
    payload["evaluation_split"] = "val_compare"
    payload["performance_interpretation_allowed"] = False
    write_target = args.output.resolve()
    write_target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate_samples(rows: list[dict[str, Any]], indices: np.ndarray | None = None) -> dict[str, float]:
    selected = rows if indices is None else [rows[int(index)] for index in indices]
    errors = [float(value) for row in selected for value in row["matched_errors_m"]]
    true_count = sum(int(row["true_count"]) for row in selected)
    return {
        "sample_count": len(selected),
        "source_count": true_count,
        "rmse_m": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "median_m": float(np.median(errors)),
        "p90_m": float(np.percentile(errors, 90)),
        "p95_m": float(np.percentile(errors, 95)),
        "mean_gospa_m": float(np.mean([row["gospa_m"] for row in selected])),
        "recall_100m": sum(int(row["tp_at_100m"]) for row in selected) / true_count,
        "above_100m": sum(value > 100 for value in errors),
        "above_500m": sum(value > 500 for value in errors),
        "above_1000m": sum(value > 1000 for value in errors),
    }


def paired_bootstrap(
    left: list[dict[str, Any]], right: list[dict[str, Any]], *, seed: int, repeats: int,
) -> dict[str, Any]:
    require(len(left) == len(right), "paired bootstrap样本数不一致")
    require(
        [row["sample_index"] for row in left] == [row["sample_index"] for row in right],
        "paired bootstrap样本顺序不一致",
    )
    point_left = aggregate_samples(left)
    point_right = aggregate_samples(right)
    metrics = ("rmse_m", "mean_gospa_m", "recall_100m")
    draws = {metric: np.empty(repeats, dtype=np.float64) for metric in metrics}
    rng = np.random.default_rng(seed)
    for repeat in range(repeats):
        sampled = rng.integers(0, len(left), size=len(left))
        left_value = aggregate_samples(left, sampled)
        right_value = aggregate_samples(right, sampled)
        for metric in metrics:
            draws[metric][repeat] = right_value[metric] - left_value[metric]
    result: dict[str, Any] = {}
    for metric in metrics:
        point = point_right[metric] - point_left[metric]
        lo, hi = np.percentile(draws[metric], [2.5, 97.5])
        result[metric] = {
            "right_minus_left": point,
            "ci95": [float(lo), float(hi)],
            "probability_right_better": float(np.mean(
                draws[metric] > 0 if metric == "recall_100m" else draws[metric] < 0
            )),
        }
    return result


def union_bandwidth(fc: np.ndarray, bw: np.ndarray, count: int) -> float:
    intervals = sorted(
        (float(fc[index] - bw[index] / 2), float(fc[index] + bw[index] / 2))
        for index in range(count)
    )
    total = 0.0
    start, stop = intervals[0]
    for next_start, next_stop in intervals[1:]:
        if next_start <= stop:
            stop = max(stop, next_stop)
        else:
            total += stop - start
            start, stop = next_start, next_stop
    return total + stop - start


def scenario_values(raw: dict[str, Any], indices: list[int]) -> dict[str, np.ndarray]:
    result: dict[str, list[float]] = {
        "avg_snr_db": [], "mean_bandwidth_hz": [], "union_bandwidth_hz": [],
        "minimum_source_separation_m": [], "maximum_source_radius_m": [],
    }
    for sample_index in indices:
        count = int(raw["src_count"][sample_index])
        positions = raw["src_pos"][sample_index, :count]
        pair_distances = [
            float(np.linalg.norm(positions[left] - positions[right]))
            for left in range(count) for right in range(left + 1, count)
        ]
        result["avg_snr_db"].append(float(raw["avg_snr"][sample_index]))
        result["mean_bandwidth_hz"].append(float(np.mean(raw["bw_actual"][sample_index, :count])))
        result["union_bandwidth_hz"].append(union_bandwidth(
            raw["fc_offset"][sample_index], raw["bw_actual"][sample_index], count,
        ))
        result["minimum_source_separation_m"].append(min(pair_distances))
        result["maximum_source_radius_m"].append(float(np.max(np.linalg.norm(positions, axis=1))))
    return {name: np.asarray(values, dtype=np.float64) for name, values in result.items()}


def scenario_analysis(
    rows: dict[str, list[dict[str, Any]]], raw: dict[str, Any], indices: list[int],
) -> dict[str, Any]:
    values = scenario_values(raw, indices)
    report: dict[str, Any] = {}
    for scenario, array in values.items():
        edges = np.unique(np.quantile(array, [0.25, 0.5, 0.75]))
        bins = np.digitize(array, edges, right=True)
        groups = []
        for bin_index in range(len(edges) + 1):
            selected = np.flatnonzero(bins == bin_index)
            metrics = {mode: aggregate_samples(mode_rows, selected) for mode, mode_rows in rows.items()}
            groups.append({
                "bin": bin_index,
                "count": len(selected),
                "value_min": float(np.min(array[selected])),
                "value_max": float(np.max(array[selected])),
                "metrics": metrics,
                "rmse_delta_hard_minus_exact_m": metrics["hard_actual"]["rmse_m"] - metrics["exact"]["rmse_m"],
                "rmse_delta_soft_minus_hard_m": metrics["soft19_actual"]["rmse_m"] - metrics["hard_actual"]["rmse_m"],
            })
        report[scenario] = {"quartile_edges": edges.tolist(), "groups": groups}
    return report


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    scales = (1024, 4096, 8192)
    modes = ("exact", "hard_actual", "soft19_actual")
    mode_dir = {"exact": "exact", "hard_actual": "hard_actual", "soft19_actual": "soft19_actual"}
    report: dict[str, Any] = {
        "status": "PASS", "gate": "S2-G4-R4", "test_executed": False,
        "bootstrap": {"seed": args.bootstrap_seed, "repeats": args.bootstrap_repeats},
        "scales": {},
    }
    sample_rows: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for scale in scales:
        scale_rows: dict[str, list[dict[str, Any]]] = {}
        scale_report: dict[str, Any] = {"modes": {}, "paired_bootstrap": {}}
        for mode in modes:
            evaluation_dir = root / "10_evaluation" / f"n{scale}"
            evaluation = read_json(evaluation_dir / f"{mode}.json")
            rows = read_jsonl(evaluation_dir / f"{mode}_samples.jsonl")
            require(evaluation["status"] == "PASS" and len(rows) == 1024, f"n{scale}/{mode}评估无效")
            training_dir = root / "09_training" / f"n{scale}" / mode_dir[mode]
            training = read_json(training_dir / "training_summary.json")
            monitor = read_json(root / "09_monitors" / f"n{scale}_{'soft' if mode == 'soft19_actual' else 'hard' if mode == 'hard_actual' else 'exact'}_train" / "stage_monitor_report.json")
            require(training["status"] == monitor["status"] == "PASS", f"n{scale}/{mode}训练证据无效")
            checkpoint = evaluation["checkpoint"]
            checkpoint_path = Path(checkpoint["path"])
            checkpoint_hash = sha256(checkpoint_path)
            require(checkpoint_hash == checkpoint["sha256"], f"n{scale}/{mode} checkpoint哈希漂移")
            scale_report["modes"][mode] = {
                "aggregate": aggregate_samples(rows),
                "checkpoint_epoch": evaluation["checkpoint_epoch"],
                "val_select_best_rmse_m": training["best_rmse"],
                "epochs_completed": training["epochs_completed"],
                "checkpoint_sha256": checkpoint_hash,
                "training_seconds": training["total_elapsed_seconds"],
                "monitor_duration_seconds": monitor["duration_seconds"],
                "monitor_warnings": monitor["warnings"],
                "monitor_red_flags": monitor["red_flags"],
                "minimum_system_available_gib": monitor["minimum_system_available_bytes"] / 2**30,
                "maximum_process_tree_rss_gib": monitor["maximum_process_tree_rss_bytes"] / 2**30,
                "maximum_gpu_used_gib": monitor["maximum_gpu_used_mib"] / 1024,
            }
            scale_rows[mode] = rows
        scale_report["paired_bootstrap"]["hard_minus_exact"] = paired_bootstrap(
            scale_rows["exact"], scale_rows["hard_actual"],
            seed=args.bootstrap_seed + scale, repeats=args.bootstrap_repeats,
        )
        scale_report["paired_bootstrap"]["soft_minus_exact"] = paired_bootstrap(
            scale_rows["exact"], scale_rows["soft19_actual"],
            seed=args.bootstrap_seed + scale + 1, repeats=args.bootstrap_repeats,
        )
        scale_report["paired_bootstrap"]["soft_minus_hard"] = paired_bootstrap(
            scale_rows["hard_actual"], scale_rows["soft19_actual"],
            seed=args.bootstrap_seed + scale + 2, repeats=args.bootstrap_repeats,
        )
        report["scales"][str(scale)] = scale_report
        sample_rows[scale] = scale_rows

    manifest = read_json(root / "06_manifests" / "val_compare.json")
    raw = load_ch4_mat(args.raw_val.resolve(), include_iq=False)
    ordered_indices = [int(row["sample_index"]) for row in sample_rows[8192]["exact"]]
    require(
        set(ordered_indices) == {int(value) for value in manifest["indices"]},
        "val_compare评估样本与manifest不一致",
    )
    report["scenario_analysis_8192"] = scenario_analysis(
        sample_rows[8192], raw, ordered_indices,
    )
    monitor_reports = [
        read_json(path) for path in sorted((root / "09_monitors").glob("n*/*stage_monitor_report.json"))
    ]
    report["resource_summary"] = {
        "stage_count": len(monitor_reports),
        "pass_count": sum(item["status"] == "PASS" for item in monitor_reports),
        "warning_count": sum(len(item["warnings"]) for item in monitor_reports),
        "red_flag_count": sum(len(item["red_flags"]) for item in monitor_reports),
        "total_training_seconds": sum(
            report["scales"][str(scale)]["modes"][mode]["training_seconds"]
            for scale in scales for mode in modes
        ),
        "minimum_system_available_gib": min(
            item["minimum_system_available_bytes"] / 2**30 for item in monitor_reports
        ),
        "maximum_process_tree_rss_gib": max(
            item["maximum_process_tree_rss_bytes"] / 2**30 for item in monitor_reports
        ),
        "maximum_gpu_used_gib": max(item["maximum_gpu_used_mib"] / 1024 for item in monitor_reports),
        "minimum_disk_free_gib": min(item["minimum_disk_free_bytes"] / 2**30 for item in monitor_reports),
    }
    report["fallacy_scan"] = {
        "simpson_paradox": "CHECKED_N2_N3_AND_SCENARIO_STRATA",
        "ecological_fallacy": "MITIGATED_WITH_PAIRED_SAMPLE_ANALYSIS",
        "berkson_bias": "NOT_APPLICABLE_TO_FIXED_GENERATED_COMPARISON_SET",
        "collider_bias": "NO_POST_OUTCOME_SAMPLE_FILTERING",
        "base_rate_neglect": "N2_N3_FIXED_BALANCE_REPORTED",
        "regression_to_mean": "SEPARATE_VAL_SELECT_AND_VAL_COMPARE",
        "survivorship_bias": "ALL_1024_COMPARISON_SAMPLES_INCLUDED",
        "look_elsewhere_effect": "EXPLORATORY_MULTIPLE_METRICS_NO_PVALUE_CLAIM",
        "garden_of_forking_paths": "R4_IS_EXPLORATORY_SINGLE_SEED_SCALE_GATE",
        "correlation_not_causation": "PAIRED_CONTROLLED_INPUT_REPRESENTATION_COMPARISON",
        "reverse_causality": "NOT_APPLICABLE_TO_CONTROLLED_TRAINING_INTERVENTION",
    }
    write_json(args.output.resolve(), report)
    return report


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="S2-G4-R4清单与数据审计")
    sub = root.add_subparsers(dest="command", required=True)
    manifests = sub.add_parser("make-manifests")
    manifests.add_argument("--raw_iq_dir", type=Path, required=True)
    manifests.add_argument("--output_dir", type=Path, required=True)
    audit = sub.add_parser("audit-dpd")
    audit.add_argument("--exact", type=Path, required=True)
    audit.add_argument("--hard", type=Path, required=True)
    audit.add_argument("--soft", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--track", required=True)
    evaluate.add_argument("--data_mode", choices=["exact", "hard_actual", "soft19_actual"],
                          required=True)
    evaluate.add_argument("--data_dir", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--samples_jsonl", type=Path, required=True)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--root", type=Path, required=True)
    finalize.add_argument("--raw_val", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--bootstrap_seed", type=int, default=20260826)
    finalize.add_argument("--bootstrap_repeats", type=int, default=2000)
    return root


def main() -> int:
    args = parser().parse_args()
    if args.command == "make-manifests":
        payload = run_manifests(args)
    elif args.command == "audit-dpd":
        payload = run_audit_dpd(args)
    elif args.command == "evaluate":
        payload = run_r4_evaluate(args)
    else:
        payload = run_finalize(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
