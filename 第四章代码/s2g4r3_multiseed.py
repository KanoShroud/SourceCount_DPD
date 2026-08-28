"""S2-G4-R3 Exact/Hard/Soft配对跨seed确认。

只读复用R2三种定位数据；新增seed 1042/2042的D8训练、oracle-K验证和
三seed汇总。不得用于test、predicted band/K或第三章训练。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from s2g3_composability import file_identity
from s2g4_coarse_d8 import (
    load_json,
    load_jsonl,
    paired_bootstrap,
    require,
    run_evaluate,
    trend_diagnostic,
    write_json,
)
from train_yolo import configure_reproducibility, sha256_state_dict
from yolo_model import YOLOv8Loc


SEEDS = (42, 1042, 2042)
MODES = ("exact", "hard_actual", "soft19_actual")
EXPECTED_COUNTS = {"train": 1024, "val": 256}


def dataset_index(data_dir: Path, split: str) -> tuple[Path, dict[str, Any]]:
    split_dir = data_dir.resolve() / split
    path = split_dir / f"loc_{split}_index.pt"
    require(path.is_file(), f"缺少数据索引: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    require(int(payload["n_total_tasks"]) == EXPECTED_COUNTS[split],
            f"{path}任务数错误")
    return path, payload


def compare_supervision(data_dirs: dict[str, Path], split: str) -> dict[str, Any]:
    indexes = {mode: dataset_index(path, split) for mode, path in data_dirs.items()}
    shard_names = {mode: payload[1]["shard_files"] for mode, payload in indexes.items()}
    counts = {len(names) for names in shard_names.values()}
    require(len(counts) == 1, f"{split}三种表示分片数不一致")
    checked = 0
    dpd_distinct = {"exact_vs_hard": 0, "hard_vs_soft": 0, "exact_vs_soft": 0}
    for shard_idx in range(next(iter(counts))):
        shards = {}
        for mode, data_dir in data_dirs.items():
            path = data_dir.resolve() / split / shard_names[mode][shard_idx]
            shards[mode] = torch.load(path, map_location="cpu", weights_only=False)
        reference = shards["exact"]
        for mode in ("hard_actual", "soft19_actual"):
            for field in ("gauss_label", "pos_label", "n_src", "sample_idx", "group_idx"):
                require(torch.equal(reference[field], shards[mode][field]),
                        f"{split}分片{shard_idx}字段{field}与{mode}不一致")
        batch = int(len(reference["n_src"]))
        checked += batch
        pairs = (
            ("exact_vs_hard", reference["fine_dpd"], shards["hard_actual"]["fine_dpd"]),
            ("hard_vs_soft", shards["hard_actual"]["fine_dpd"], shards["soft19_actual"]["fine_dpd"]),
            ("exact_vs_soft", reference["fine_dpd"], shards["soft19_actual"]["fine_dpd"]),
        )
        for name, first, second in pairs:
            require(first.shape == second.shape, f"{split}分片{shard_idx} {name} shape不一致")
            different = (first != second).flatten(1).any(dim=1)
            dpd_distinct[name] += int(different.sum().item())
    require(checked == EXPECTED_COUNTS[split], f"{split}实际检查{checked}条")
    require(all(value > 0 for value in dpd_distinct.values()), f"{split}表示未形成数值差异")
    return {
        "sample_count": checked,
        "supervision_exact": True,
        "different_sample_counts": dpd_distinct,
        "indexes": {mode: file_identity(item[0]) for mode, item in indexes.items()},
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    baseline = load_json(args.resource_baseline)
    require(baseline.get("status") == "PASS", "12 GiB资源基线未通过")
    require(float(baseline["threshold_gib"]) == 12.0, "资源基线阈值不是12 GiB")
    data_dirs = {
        "exact": args.exact_data_dir,
        "hard_actual": args.hard_data_dir,
        "soft19_actual": args.soft_data_dir,
    }
    seed42 = {}
    for mode, summary_path, evaluation_path in (
        ("exact", args.seed42_exact_summary, args.seed42_exact_evaluation),
        ("hard_actual", args.seed42_hard_summary, args.seed42_hard_evaluation),
        ("soft19_actual", args.seed42_soft_summary, args.seed42_soft_evaluation),
    ):
        summary = load_json(summary_path)
        evaluation = load_json(evaluation_path)
        config = load_json(Path(summary_path).resolve().parent / "run_config.json")
        require(summary.get("status") == evaluation.get("status") == "PASS",
                f"seed42 {mode}历史证据非PASS")
        require(int(config["args"]["seed"]) == 42, f"seed42 {mode}训练seed错误")
        require(int(summary["epochs_completed"]) == 200, f"seed42 {mode}未完成200 epoch")
        require(evaluation["evaluation_mode"] == "oracle-K_ground_truth_source_count",
                f"seed42 {mode}不是oracle-K")
        seed42[mode] = {
            "summary": file_identity(summary_path),
            "evaluation": file_identity(evaluation_path),
            "checkpoint": evaluation["checkpoint"],
        }
    payload = {
        "status": "PASS",
        "gate": "S2-G4-R3",
        "stage": "identity_and_data_preflight",
        "approved_seeds": list(SEEDS),
        "resource_baseline": file_identity(args.resource_baseline),
        "data": {split: compare_supervision(data_dirs, split) for split in EXPECTED_COUNTS},
        "seed42_reused_evidence": seed42,
        "test_executed": False,
        "predicted_inputs_executed": False,
    }
    write_json(args.output, payload)
    return payload


def model_hash(seed: int) -> str:
    configure_reproducibility(seed, True)
    model = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0)
    return sha256_state_dict(model.state_dict())


def run_initialization_pilot(args: argparse.Namespace) -> dict[str, Any]:
    per_seed = {}
    for seed in SEEDS:
        hashes = [model_hash(seed) for _ in MODES]
        require(len(set(hashes)) == 1, f"seed {seed}三种表示初始化不一致")
        per_seed[str(seed)] = {"three_arm_hash": hashes[0], "all_equal": True}
    require(len({item["three_arm_hash"] for item in per_seed.values()}) == len(SEEDS),
            "不同seed产生了相同模型初始化")
    payload = {
        "status": "PASS",
        "stage": "paired_initialization_pilot",
        "seeds": per_seed,
        "different_seeds_distinct": True,
    }
    write_json(args.output, payload)
    return payload


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    require(args.seed in SEEDS, f"未批准的seed: {args.seed}")
    namespace = SimpleNamespace(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        output=args.output,
        samples_jsonl=args.samples_jsonl,
        device=args.device,
        batch_size=8,
        seed=args.seed,
        track=f"seed{args.seed}_{args.mode}",
        data_mode=args.mode,
    )
    return run_evaluate(namespace)


def run_manifest(args: argparse.Namespace) -> dict[str, Any]:
    root = args.run_root.resolve()
    payload: dict[str, Any] = {
        "status": "PASS",
        "gate": "S2-G4-R3",
        "run_root": str(root),
        "seeds": {
            "42": {
                "exact": {
                    "summary": str(args.seed42_exact_summary.resolve()),
                    "evaluation": str(args.seed42_exact_evaluation.resolve()),
                },
                "hard_actual": {
                    "summary": str(args.seed42_hard_summary.resolve()),
                    "evaluation": str(args.seed42_hard_evaluation.resolve()),
                },
                "soft19_actual": {
                    "summary": str(args.seed42_soft_summary.resolve()),
                    "evaluation": str(args.seed42_soft_evaluation.resolve()),
                },
            }
        },
    }
    for seed in (1042, 2042):
        payload["seeds"][str(seed)] = {}
        for mode in MODES:
            arm = root / f"seed_{seed}" / mode
            summary = arm / "train" / "training_summary.json"
            evaluation = arm / "evaluation.json"
            require(summary.is_file() and evaluation.is_file(),
                    f"seed {seed} {mode}结果不完整")
            payload["seeds"][str(seed)][mode] = {
                "summary": str(summary.resolve()),
                "evaluation": str(evaluation.resolve()),
            }
    write_json(args.output, payload)
    return payload


def median(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64)))


def collect_arm(seed: int, mode: str, summary_path: Path, evaluation_path: Path) -> dict[str, Any]:
    summary = load_json(summary_path)
    evaluation = load_json(evaluation_path)
    config = load_json(summary_path.resolve().parent / "run_config.json")
    require(summary.get("status") == evaluation.get("status") == "PASS",
            f"seed {seed} {mode}非PASS")
    require(int(config["args"]["seed"]) == seed, f"seed {seed} {mode}配置seed错误")
    require(int(summary["epochs_completed"]) == 200, f"seed {seed} {mode}未完成200 epoch")
    require(config["args"].get("s2g4r3_scratch") is True or seed == 42,
            f"seed {seed} {mode}不是R3严格训练")
    return {
        "summary": summary,
        "evaluation": evaluation,
        "trend": trend_diagnostic(summary),
        "config": config,
        "summary_identity": file_identity(summary_path),
        "evaluation_identity": file_identity(evaluation_path),
    }


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_json(args.manifest)
    require(manifest.get("gate") == "S2-G4-R3", "manifest gate错误")
    arms: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        seed_entry = manifest["seeds"][str(seed)]
        arms[seed] = {
            mode: collect_arm(
                seed, mode, Path(seed_entry[mode]["summary"]),
                Path(seed_entry[mode]["evaluation"]),
            )
            for mode in MODES
        }
        if seed != 42:
            init_hashes = {
                item["summary"]["initialization"]["model_state_sha256"]
                for item in arms[seed].values()
            }
            require(len(init_hashes) == 1, f"seed {seed}三臂初始权重哈希不一致")

    seed_results = {}
    hard_passes = 0
    hard_rmse_ratios = []
    hard_gospa_ratios = []
    hard_recall_drops = []
    soft_improvements = []
    soft_gospa_ratios = []
    soft_better_count = 0
    extreme_seed = False
    training_valid = True
    for seed, seed_arms in arms.items():
        metrics = {mode: item["evaluation"]["metrics"] for mode, item in seed_arms.items()}
        rmse = {mode: float(item["matched_errors_m"]["rmse"]) for mode, item in metrics.items()}
        gospa = {mode: float(item["gospa"]["mean"]) for mode, item in metrics.items()}
        recall = {mode: float(item["set_detection"]["100m"]["recall"]) for mode, item in metrics.items()}
        h_rmse_ratio = rmse["hard_actual"] / rmse["exact"]
        h_gospa_ratio = gospa["hard_actual"] / gospa["exact"]
        h_recall_drop = recall["exact"] - recall["hard_actual"]
        h_pass = h_rmse_ratio <= 1.10 and h_gospa_ratio <= 1.10 and h_recall_drop <= 0.05
        hard_passes += int(h_pass)
        hard_rmse_ratios.append(h_rmse_ratio)
        hard_gospa_ratios.append(h_gospa_ratio)
        hard_recall_drops.append(h_recall_drop)
        extreme_seed |= h_rmse_ratio > 1.25 or h_gospa_ratio > 1.25 or h_recall_drop > 0.10
        soft_improvement = (rmse["hard_actual"] - rmse["soft19_actual"]) / rmse["hard_actual"]
        soft_gospa_ratio = gospa["soft19_actual"] / gospa["hard_actual"]
        soft_improvements.append(soft_improvement)
        soft_gospa_ratios.append(soft_gospa_ratio)
        soft_better_count += int(rmse["soft19_actual"] < rmse["hard_actual"])
        training_valid &= all(
            arm["trend"]["trend_closed"] and arm["trend"]["effective_learning"]
            for arm in seed_arms.values()
        )
        sample_rows = {
            mode: load_jsonl(Path(item["evaluation"]["samples_jsonl"]))
            for mode, item in seed_arms.items()
        }
        seed_results[str(seed)] = {
            "rmse_m": rmse,
            "mean_gospa_m": gospa,
            "recall_100m": recall,
            "hard_to_exact": {
                "rmse_ratio": h_rmse_ratio,
                "gospa_ratio": h_gospa_ratio,
                "recall_drop": h_recall_drop,
                "equivalence_pass": h_pass,
            },
            "soft_to_hard": {
                "rmse_improvement_fraction": soft_improvement,
                "gospa_ratio": soft_gospa_ratio,
            },
            "stratified": {
                mode: item["evaluation"]["stratified"] for mode, item in seed_arms.items()
            },
            "extreme_error_counts": {
                mode: item["evaluation"]["metrics"]["extreme_error_counts"]
                for mode, item in seed_arms.items()
            },
            "training": {mode: item["trend"] for mode, item in seed_arms.items()},
            "paired_bootstrap": {
                "hard_minus_exact_rmse": paired_bootstrap(
                    sample_rows["exact"], sample_rows["hard_actual"], "matched_rmse_m"
                ),
                "soft_minus_hard_rmse": paired_bootstrap(
                    sample_rows["hard_actual"], sample_rows["soft19_actual"], "matched_rmse_m"
                ),
                "soft_minus_hard_gospa": paired_bootstrap(
                    sample_rows["hard_actual"], sample_rows["soft19_actual"], "gospa_m"
                ),
            },
        }

    median_hard_rmse = median(hard_rmse_ratios)
    median_hard_gospa = median(hard_gospa_ratios)
    median_hard_recall = median(hard_recall_drops)
    if not training_valid:
        scientific_status = "INCONCLUSIVE_R3"
    elif (hard_passes >= 2 and median_hard_rmse <= 1.10
          and median_hard_gospa <= 1.10 and median_hard_recall <= 0.05
          and not extreme_seed):
        scientific_status = "HARD19_STABLE_CANDIDATE"
    elif sum(value > 1.10 for value in hard_rmse_ratios) >= 2 and median_hard_rmse > 1.10:
        scientific_status = "HARD19_NOT_SUPPORTED"
    else:
        scientific_status = "HARD19_SEED_SENSITIVE"
    soft_reconsider = (
        soft_better_count >= 2
        and median(soft_improvements) >= 0.05
        and median(soft_gospa_ratios) <= 1.0
    )

    monitor_reports = []
    for path in sorted(args.run_root.resolve().rglob("stage_monitor_report.json")):
        report = load_json(path)
        require(report.get("status") == "PASS", f"监控非PASS: {path}")
        monitor_reports.append(report)
    require(len(monitor_reports) >= 14, f"R3监控报告不足: {len(monitor_reports)}")
    payload = {
        "status": "PASS",
        "gate": "S2-G4-R3",
        "engineering_status": "PASS",
        "scientific_status": scientific_status,
        "scope": "缩减validation、oracle频带、oracle-K、三seed配对诊断",
        "seed_results": seed_results,
        "cross_seed": {
            "hard_equivalence_pass_count": hard_passes,
            "hard_rmse_ratio_median": median_hard_rmse,
            "hard_rmse_ratio_range": [min(hard_rmse_ratios), max(hard_rmse_ratios)],
            "hard_gospa_ratio_median": median_hard_gospa,
            "hard_recall_drop_median": median_hard_recall,
            "hard_extreme_seed_present": extreme_seed,
            "soft_better_than_hard_seed_count": soft_better_count,
            "soft_rmse_improvement_fraction_median": median(soft_improvements),
            "soft_gospa_ratio_median": median(soft_gospa_ratios),
            "soft_reconsideration_gate_pass": soft_reconsider,
        },
        "resource_summary": {
            "monitor_count": len(monitor_reports),
            "minimum_system_available_bytes": min(
                item["minimum_system_available_bytes"] for item in monitor_reports
            ),
            "maximum_process_tree_rss_bytes": max(
                item["maximum_process_tree_rss_bytes"] for item in monitor_reports
            ),
            "maximum_gpu_used_mib": max(
                item.get("maximum_gpu_used_mib") or 0 for item in monitor_reports
            ),
            "minimum_disk_free_bytes": min(
                item["minimum_disk_free_bytes"] for item in monitor_reports
            ),
            "maximum_scope_output_bytes": max(
                item["maximum_scope_output_bytes"] for item in monitor_reports
            ),
            "hard_timeout_enabled": any(item.get("timeout_enabled") for item in monitor_reports),
        },
        "evidence": {
            "preflight": file_identity(args.preflight),
            "initialization_pilot": file_identity(args.initialization_pilot),
            "manifest": file_identity(args.manifest),
        },
        "interpretation_boundary": (
            "三个seed只用于方向和工程稳健性筛查；不进行跨seed显著性检验。"
            "结果不是test性能，不证明CH3可输出该表示。"
        ),
        "test_executed": False,
        "predicted_band_executed": False,
        "predicted_k_executed": False,
        "ch3_modified_or_trained": False,
    }
    write_json(args.output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="S2-G4-R3配对跨seed确认")
    sub = parser.add_subparsers(dest="command", required=True)
    preflight = sub.add_parser("preflight")
    for name in ("exact_data_dir", "hard_data_dir", "soft_data_dir"):
        preflight.add_argument(f"--{name}", type=Path, required=True)
    for mode in ("exact", "hard", "soft"):
        preflight.add_argument(f"--seed42_{mode}_summary", type=Path, required=True)
        preflight.add_argument(f"--seed42_{mode}_evaluation", type=Path, required=True)
    preflight.add_argument("--resource_baseline", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)

    pilot = sub.add_parser("initialization-pilot")
    pilot.add_argument("--output", type=Path, required=True)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--mode", choices=MODES, required=True)
    evaluate.add_argument("--data_dir", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--samples_jsonl", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--run_root", type=Path, required=True)
    for mode in ("exact", "hard", "soft"):
        manifest.add_argument(f"--seed42_{mode}_summary", type=Path, required=True)
        manifest.add_argument(f"--seed42_{mode}_evaluation", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    finalize = sub.add_parser("finalize")
    finalize.add_argument("--manifest", type=Path, required=True)
    finalize.add_argument("--preflight", type=Path, required=True)
    finalize.add_argument("--initialization_pilot", type=Path, required=True)
    finalize.add_argument("--run_root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "preflight":
        payload = run_preflight(args)
    elif args.command == "initialization-pilot":
        payload = run_initialization_pilot(args)
    elif args.command == "evaluate":
        payload = run_evaluation(args)
    elif args.command == "manifest":
        payload = run_manifest(args)
    elif args.command == "finalize":
        payload = run_finalize(args)
    else:
        raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
