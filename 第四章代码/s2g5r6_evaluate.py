"""S2-G5-R6-A三seed主对角与3x3交叉validation诊断。

只读取R4/R5冻结validation和R6-A权重；不读取test，不训练大模型。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CH3_DIR = PROJECT_ROOT / "第三章代码"
CH4_DIR = PROJECT_ROOT / "第四章代码"
for import_root in (CH4_DIR, CH3_DIR, PROJECT_ROOT):
    if str(import_root) in sys.path:
        sys.path.remove(str(import_root))
sys.path.insert(0, str(CH4_DIR))
sys.path.insert(1, str(CH3_DIR))
sys.path.insert(2, str(PROJECT_ROOT))

import s2g5r3_cascade as r3  # noqa: E402
import s2g5r5_candidate_k as r5  # noqa: E402


SEEDS = (42, 1042, 2042)
R6_ROOT_DEFAULT = PROJECT_ROOT / "outputs" / "s2g5r6" / "20260829_170441"
R5_ROOT = PROJECT_ROOT / "outputs" / "s2g5r5_candidate_k" / "20260828_222900"
R4_CH3_ROOT = PROJECT_ROOT / "outputs" / "s2g5r4_ch3_scale" / "20260828_151735"
R4_D8_ROOT = PROJECT_ROOT / "outputs" / "s2g4r4_scale" / "20260826_132829"
ORACLE_COMPARE_INDEX = R4_CH3_ROOT / "cascade_16k" / "03_fine_oracle" / "index.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_json(path: Path, payload: Any) -> None:
    path = path.resolve()
    require(not path.exists(), f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path = path.resolve()
    require(not path.exists(), f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checkpoint_paths(run_root: Path) -> tuple[dict[int, Path], dict[int, Path]]:
    ch3 = {
        42: R4_CH3_ROOT / "train_16k" / "best_model_v26_B_M10.pth",
        1042: run_root / "training" / "ch3_seed1042" / "best_model_v26_B_M10.pth",
        2042: run_root / "training" / "ch3_seed2042" / "best_model_v26_B_M10.pth",
    }
    d8 = {
        42: R4_D8_ROOT / "09_training" / "n8192" / "hard_actual" / "best_yolo_dualhead_std.pth",
        1042: run_root / "training" / "d8_seed1042" / "best_yolo_dualhead_std.pth",
        2042: run_root / "training" / "d8_seed2042" / "best_yolo_dualhead_std.pth",
    }
    return ch3, d8


def ch3_artifacts(run_root: Path, seed: int) -> dict[str, Path]:
    if seed == 42:
        return {
            "root": R5_ROOT,
            "select_index": R5_ROOT / "03_select_fine_predicted" / "index.json",
            "compare_index": R4_CH3_ROOT / "cascade_16k" / "04_fine_predicted" / "index.json",
        }
    root = run_root / "validation" / f"ch3_seed{seed}"
    return {
        "root": root,
        "select_index": root / "03_fine_predicted" / "val_select" / "index.json",
        "compare_index": root / "03_fine_predicted" / "val_compare" / "index.json",
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    require(load_json(run_root / "preflight.json")["status"] == "PASS", "R6训练preflight未通过")
    ch3_paths, d8_paths = checkpoint_paths(run_root)
    identities: dict[str, Any] = {}
    for family, paths in (("ch3", ch3_paths), ("d8", d8_paths)):
        for seed, path in paths.items():
            require(path.is_file(), f"缺少{family} seed {seed} checkpoint")
            identities[f"{family}_seed{seed}"] = r3.file_identity(path)
    for seed in (1042, 2042):
        require(load_json(run_root / "training" / f"ch3_seed{seed}" / "training_summary.json")["status"] == "TRAIN_COMPLETED", f"CH3 seed {seed}训练未通过")
        require(load_json(run_root / "training" / f"d8_seed{seed}" / "training_summary.json")["status"] == "PASS", f"D8 seed {seed}训练未通过")
    select = r5.load_metadata(r5.VAL_SELECT)
    compare = r5.load_metadata(r5.VAL_COMPARE)
    require(len(np.intersect1d(select["sample_idx"], compare["sample_idx"])) == 0, "val_select与val_compare重叠")
    require(r3.file_identity(r5.VAL_COMPARE)["sha256"] == "a35cb199299e17cc86d3cf9793e63e76a7c92650e35558d6d82106188ea90005", "val_compare SHA变化")
    report = {
        "status": "PASS", "gate": "S2-G5-R6-A", "stage": "validation_preflight",
        "checkpoints": identities, "seeds": list(SEEDS),
        "validation": {"val_select": 1024, "val_compare": 1024, "disjoint": True},
        "test_read": False,
    }
    write_json(run_root / "validation" / "00_preflight.json", report)
    return report


def histogram(values: np.ndarray) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(Counter(map(int, values)).items())}


def build_predicted_fine(
    arrays_path: Path,
    metadata_path: Path,
    output_dir: Path,
    *,
    chunk_size: int = 40000,
    shard_size: int = 64,
) -> dict[str, Any]:
    started = time.perf_counter()
    require(not output_dir.exists(), f"拒绝覆盖细DPD目录: {output_dir}")
    output_dir.mkdir(parents=True)
    metadata = r5.load_metadata(metadata_path)
    frozen = np.load(arrays_path)
    require(np.array_equal(frozen["source_count"], metadata["source_count"]), "细DPD源数身份错误")
    require(np.array_equal(frozen["raw_sample_idx"], metadata["sample_idx"]), "细DPD raw索引身份错误")
    counts = metadata["source_count"].astype(np.int64)
    local_indices = np.flatnonzero(np.isin(counts, r5.LOCALIZATION_COUNTS))
    raw_indices = metadata["sample_idx"][local_indices]
    raw = r3.load_ch4_mat(r5.RAW_VALIDATION, include_iq=True)
    torch.cuda.set_device(0)
    geometry = r3.receiver_geometry(torch.device("cuda:0"))
    shard_entries = []
    empty_values_all: list[bool] = []
    seconds: list[float] = []
    for shard_number, start in enumerate(range(0, len(local_indices), shard_size)):
        shard_local = local_indices[start:start + shard_size]
        shard_raw = raw_indices[start:start + shard_size]
        fine_values = []
        pos_values = []
        empty_values = []
        for local_index, raw_index in zip(shard_local, shard_raw, strict=True):
            count = int(counts[local_index])
            fine, fft_mask, elapsed = r3.one_fine_dpd(
                raw,
                int(raw_index),
                frozen["band_prediction"][local_index].astype(bool),
                geometry,
                chunk_size,
            )
            positions = raw["src_pos"][raw_index, :count]
            positions = positions[np.argsort(np.linalg.norm(positions, axis=1))]
            label = np.zeros((3, 2), dtype=np.float32)
            label[:count] = positions / r3.EDGE
            fine_values.append(fine)
            pos_values.append(torch.from_numpy(label))
            empty_values.append(not bool(fft_mask.any()))
            seconds.append(elapsed)
        payload = {
            "fine_dpd": torch.stack(fine_values),
            "pos_label": torch.stack(pos_values),
            "n_src": torch.from_numpy(counts[shard_local]),
            "local_idx": torch.from_numpy(shard_local),
            "raw_idx": torch.from_numpy(shard_raw),
            "empty_band": torch.tensor(empty_values, dtype=torch.bool),
        }
        shard_path = output_dir / f"part_{shard_number:03d}.pt"
        r3.torch_save_new(shard_path, payload)
        shard_entries.append(r3.file_identity(shard_path) | {
            "sample_count": len(shard_local),
            "local_index_first": int(shard_local[0]),
            "local_index_last": int(shard_local[-1]),
        })
        empty_values_all.extend(empty_values)
    del raw
    gc.collect()
    torch.cuda.empty_cache()
    report = {
        "status": "PASS", "gate": "S2-G5-R6-A", "stage": "predicted_fine",
        "sample_count": len(local_indices),
        "source_count_histogram": histogram(counts[local_indices]),
        "empty_band_count": int(sum(empty_values_all)), "shards": shard_entries,
        "per_sample_seconds": r3.numeric_summary(np.asarray(seconds)),
        "chunk_size": chunk_size, "shard_size": shard_size, "test_read": False,
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(output_dir / "index.json", report)
    return report


def run_prepare_ch3(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    seed = args.seed
    require(seed in (1042, 2042), "seed 42直接复用R5冻结产物")
    artifacts = ch3_artifacts(run_root, seed)
    root = artifacts["root"]
    require(not root.exists(), f"拒绝覆盖CH3 validation目录: {root}")
    (root / "00_preflight").mkdir(parents=True)
    write_json(root / "00_preflight" / "preflight.json", {"status": "PASS", "gate": "S2-G5-R6-A", "seed": seed, "test_read": False})
    ch3_paths, _ = checkpoint_paths(run_root)
    r5.CH3_CHECKPOINT = ch3_paths[seed]
    r5.run_infer(argparse.Namespace(run_root=root))
    r5.run_probe(argparse.Namespace(run_root=root))
    build_predicted_fine(root / "01_frozen_logits" / "val_select.npz", r5.VAL_SELECT, artifacts["select_index"].parent)
    build_predicted_fine(root / "01_frozen_logits" / "val_compare.npz", r5.VAL_COMPARE, artifacts["compare_index"].parent)
    report = {
        "status": "PASS", "seed": seed,
        "checkpoint": r3.file_identity(ch3_paths[seed]),
        "root": str(root), "select_index": r3.file_identity(artifacts["select_index"]),
        "compare_index": r3.file_identity(artifacts["compare_index"]), "test_read": False,
    }
    write_json(root / "prepare_report.json", report)
    return report


def fit_selector(ordinal_path: Path, peak_path: Path) -> tuple[Pipeline, dict[str, Any]]:
    ordinal = np.load(ordinal_path)
    peaks = r5.load_peak_map(peak_path)
    features = []
    targets = []
    covered = 0
    total = 0
    for local_index, truth in enumerate(ordinal["source_count"]):
        if int(truth) not in r5.LOCALIZATION_COUNTS:
            continue
        total += 1
        candidates = ordinal["candidate_k"][local_index]
        if int(truth) not in candidates:
            continue
        covered += 1
        peak = peaks[local_index]
        current = r5.selector_features(
            ordinal["k_probabilities"][local_index], ordinal["slot_scores"][local_index],
            candidates, peak["peak_positions_m"], peak["peak_scores"],
        )
        if current is not None:
            features.append(current)
            targets.append(int(truth) == int(candidates[1]))
    matrix = np.stack(features)
    target = np.asarray(targets, dtype=np.int64)
    require(len(np.unique(target)) == 2 and len(target) >= 100, "选择器校准样本不足")
    selector = Pipeline([
        ("scale", StandardScaler()),
        ("logistic", LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=1000, random_state=42)),
    ])
    selector.fit(matrix, target)
    return selector, {
        "localization_samples": total, "truth_in_candidate_pair": covered,
        "usable_training_samples": len(target), "target_larger_histogram": histogram(target),
        "training_accuracy_descriptive_only": float(selector.score(matrix, target)),
    }


def count_summary(prediction: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    confusion = np.zeros((4, max(4, int(prediction.max()) + 1)), dtype=np.int64)
    for actual, predicted in zip(truth, prediction, strict=True):
        confusion[int(actual), int(predicted)] += 1
    class_accuracy = np.diag(confusion[:, :4]) / np.maximum(confusion.sum(axis=1), 1)
    return {
        "accuracy": float(np.mean(prediction == truth)),
        "balanced_accuracy": float(class_accuracy.mean()),
        "class_accuracy": {str(index): float(value) for index, value in enumerate(class_accuracy)},
        "confusion_true_rows_pred_columns": confusion.tolist(),
    }


def band_macro_f1(arrays: Any) -> float:
    truth = arrays["band_truth"].astype(bool)
    prediction = arrays["band_prediction"].astype(bool)
    ignore = arrays["ignore_mask"].astype(bool)
    counts = arrays["source_count"].astype(np.int64)
    values = []
    for slot in range(3):
        active = counts > slot
        valid = ~ignore[active, slot]
        pred = prediction[active, slot][valid]
        true = truth[active, slot][valid]
        tp = int(np.sum(pred & true)); fp = int(np.sum(pred & ~true)); fn = int(np.sum(~pred & true))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matched = [value for row in rows for value in row["matched_errors_m"]]
    return r3.summarize_track(rows, matched)


def run_pair(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    ch3_seed = args.ch3_seed
    d8_seed = args.d8_seed
    ch3 = ch3_artifacts(run_root, ch3_seed)
    ch3_root = ch3["root"]
    ordinal_select = ch3_root / "02_ordinal_probe" / "val_select_ordinal.npz"
    ordinal_compare = ch3_root / "02_ordinal_probe" / "val_compare_ordinal.npz"
    require(all(path.is_file() for path in (ordinal_select, ordinal_compare, ch3["select_index"], ch3["compare_index"])), "CH3 validation产物不完整")
    _, d8_paths = checkpoint_paths(run_root)
    r5.D8_CHECKPOINT = d8_paths[d8_seed]
    pair_root = run_root / "validation" / "pairs" / f"ch3_{ch3_seed}__d8_{d8_seed}"
    require(not pair_root.exists(), f"拒绝覆盖pair目录: {pair_root}")
    started = time.perf_counter()
    peaks: dict[str, list[dict[str, Any]]] = {}
    for name, index_path in (
        ("val_select", ch3["select_index"]),
        ("val_compare", ch3["compare_index"]),
        ("oracle_compare", ORACLE_COMPARE_INDEX),
    ):
        rows = r5.extract_peaks_for_split(index_path)
        require(len(rows) == 512, f"{name} D8峰值样本数错误")
        path = pair_root / "peaks" / f"{name}_top4.jsonl"
        write_jsonl(path, rows)
        peaks[name] = rows
    selector, selector_fit = fit_selector(ordinal_select, pair_root / "peaks" / "val_select_top4.jsonl")
    selector_path = pair_root / "selector.joblib"
    joblib.dump(selector, selector_path)
    ordinal = np.load(ordinal_compare)
    predicted_map = {int(row["local_index"]): row for row in peaks["val_compare"]}
    oracle_map = {int(row["local_index"]): row for row in peaks["oracle_compare"]}
    tracks = {name: [] for name in ("OB-OK", "PB-BASE", "PB-SELECTED")}
    selected_k = np.full(len(ordinal["source_count"]), -1, dtype=np.int64)
    for local_index, truth in enumerate(ordinal["source_count"]):
        if int(truth) not in r5.LOCALIZATION_COUNTS:
            continue
        peak = predicted_map[local_index]
        candidates = ordinal["candidate_k"][local_index].astype(np.int64)
        feature = r5.selector_features(
            ordinal["k_probabilities"][local_index], ordinal["slot_scores"][local_index],
            candidates, peak["peak_positions_m"], peak["peak_scores"],
        )
        chosen = int(ordinal["top1_k"][local_index]) if feature is None else int(candidates[int(selector.predict(feature[None, :])[0])])
        selected_k[local_index] = chosen
        tracks["OB-OK"].append(r5.decoded_record(oracle_map[local_index], int(truth), "OB-OK"))
        tracks["PB-BASE"].append(r5.decoded_record(peak, int(ordinal["baseline_k"][local_index]), "PB-BASE"))
        tracks["PB-SELECTED"].append(r5.decoded_record(peak, chosen, "PB-SELECTED"))
    identities = [[(row["local_index"], row["raw_index"], row["true_count"]) for row in rows] for rows in tracks.values()]
    require(all(current == identities[0] for current in identities[1:]), "pair三轨身份不一致")
    for name, rows in tracks.items():
        write_jsonl(pair_root / "tracks" / f"{name.lower().replace('-', '_')}.jsonl", rows)
    bootstrap = r5.bootstrap_tracks(tracks)
    point = {name: summarize_rows(rows) for name, rows in tracks.items()}
    direct = (
        bootstrap["selected_relative_to_ob_ok"]["ci95"][1] <= 0.10
        and bootstrap["selected_recall_delta"]["ci95"][0] >= -0.05
    )
    compare_arrays = np.load(ch3_root / "01_frozen_logits" / "val_compare.npz")
    truth = compare_arrays["source_count"].astype(np.int64)
    localization = np.isin(truth, r5.LOCALIZATION_COUNTS)
    candidate_hit = np.any(ordinal["candidate_k"] == truth[:, None], axis=1)
    report = {
        "status": "PASS", "gate": "S2-G5-R6-A",
        "ch3_seed": ch3_seed, "d8_seed": d8_seed,
        "checkpoints": {"ch3": r3.file_identity(checkpoint_paths(run_root)[0][ch3_seed]), "d8": r3.file_identity(d8_paths[d8_seed])},
        "ch3": {
            "baseline": count_summary(compare_arrays["baseline_k"], truth),
            "probe": r5.count_metrics(ordinal["k_probabilities"], truth),
            "band_macro_f1": band_macro_f1(compare_arrays),
            "top2_coverage_all": float(candidate_hit.mean()),
            "top2_coverage_k2_k3": float(candidate_hit[localization].mean()),
            "selected_k_accuracy_k2_k3": float(np.mean(selected_k[localization] == truth[localization])),
        },
        "selector_fit": selector_fit, "tracks": point, "paired_bootstrap": bootstrap,
        "direct_gate": direct,
        "selected_better_than_baseline_point": point["PB-SELECTED"]["gospa"]["mean"] < point["PB-BASE"]["gospa"]["mean"],
        "test_read": False, "duration_seconds": time.perf_counter() - started,
    }
    write_json(pair_root / "pair_report.json", report)
    return report


def run_finalize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    reports = {}
    for ch3_seed in SEEDS:
        for d8_seed in SEEDS:
            key = f"{ch3_seed}/{d8_seed}"
            path = run_root / "validation" / "pairs" / f"ch3_{ch3_seed}__d8_{d8_seed}" / "pair_report.json"
            reports[key] = load_json(path)
    diagonal = [reports[f"{seed}/{seed}"] for seed in SEEDS]
    improve_count = sum(row["selected_better_than_baseline_point"] for row in diagonal)
    direct_count = sum(row["direct_gate"] for row in diagonal)
    noncatastrophic = all(
        row["tracks"]["PB-SELECTED"]["gospa"]["mean"] / row["tracks"]["OB-OK"]["gospa"]["mean"] - 1.0 <= 0.15
        and row["tracks"]["PB-SELECTED"]["set_detection"]["100m"]["recall"]
        - row["tracks"]["OB-OK"]["set_detection"]["100m"]["recall"] >= -0.05
        for row in diagonal
    )
    matrix = np.asarray([
        [reports[f"{ch3_seed}/{d8_seed}"]["selected_better_than_baseline_point"] for d8_seed in SEEDS]
        for ch3_seed in SEEDS
    ], dtype=bool)
    no_single_combo_dependency = bool(np.all(matrix.sum(axis=0) >= 1) and np.all(matrix.sum(axis=1) >= 1))
    if improve_count == 3 and direct_count >= 2 and noncatastrophic and no_single_combo_dependency:
        decision = "R6A_STABLE"
    elif improve_count >= 2 and noncatastrophic:
        decision = "R6A_MIXED"
    else:
        decision = "R6A_UNSTABLE"
    report = {
        "status": "PASS", "gate": "S2-G5-R6-A", "decision": decision,
        "diagonal": {f"{seed}/{seed}": reports[f"{seed}/{seed}"] for seed in SEEDS},
        "cross_matrix": {
            "selected_better_than_baseline": matrix.tolist(),
            "row_seeds_ch3": list(SEEDS), "column_seeds_d8": list(SEEDS),
            "no_single_row_or_column_dependency": no_single_combo_dependency,
            "all_pairs": reports,
        },
        "decision_counts": {"diagonal_improved": improve_count, "diagonal_direct_gate": direct_count},
        "noncatastrophic_diagonal": noncatastrophic,
        "test_read": False,
    }
    write_json(run_root / "validation" / "final_report.json", report)
    return report


def normalized_config(path: Path, family: str) -> dict[str, Any]:
    payload = load_json(path)
    values = payload if family == "ch3" else payload["args"]
    ignored = {"seed", "output_dir", "run_label"}
    if family == "d8":
        ignored.update({
            "gate3_d8", "gate3b_d8", "s2g2_d8", "s2g4_scratch", "s2g4_finetune",
            "s2g4r3_scratch", "s2g4r4_scratch", "s2g5r6_scratch",
        })
    return {key: value for key, value in values.items() if key not in ignored}


def run_summarize(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve()
    final = load_json(run_root / "validation" / "final_report.json")
    training: dict[str, Any] = {}
    ch3_paths, d8_paths = checkpoint_paths(run_root)
    for family, seeds, checkpoint_map in (("ch3", (1042, 2042), ch3_paths), ("d8", (1042, 2042), d8_paths)):
        for seed in seeds:
            summary = load_json(run_root / "training" / f"{family}_seed{seed}" / "training_summary.json")
            monitor = load_json(run_root / "monitor" / f"train_{family}_seed{seed}" / "stage_monitor_report.json")
            require(monitor["status"] == "PASS" and not monitor["red_flags"], f"{family} seed {seed}监控未通过")
            training[f"{family}_seed{seed}"] = {
                "epochs_completed": summary["epochs_completed"],
                "best_epoch": summary["best_epoch"],
                "best_metric": summary["best_validation"]["loss"] if family == "ch3" else summary["best_rmse"],
                "checkpoint": r3.file_identity(checkpoint_map[seed]),
                "duration_seconds": summary["duration_seconds"] if family == "ch3" else summary["total_elapsed_seconds"],
                "maximum_process_tree_rss_bytes": monitor["maximum_process_tree_rss_bytes"],
                "maximum_gpu_used_mib": monitor["maximum_gpu_used_mib"],
                "warnings": monitor["warnings"], "red_flags": monitor["red_flags"],
            }
    ch3_reference = normalized_config(R4_CH3_ROOT / "train_16k" / "run_config.json", "ch3")
    d8_reference = normalized_config(R4_D8_ROOT / "09_training" / "n8192" / "hard_actual" / "run_config.json", "d8")
    config_checks = {
        "ch3_1042_only_seed_output_changed": normalized_config(run_root / "training" / "ch3_seed1042" / "run_config.json", "ch3") == ch3_reference,
        "ch3_2042_only_seed_output_changed": normalized_config(run_root / "training" / "ch3_seed2042" / "run_config.json", "ch3") == ch3_reference,
        "d8_1042_only_seed_output_and_strict_identity_changed": normalized_config(run_root / "training" / "d8_seed1042" / "run_config.json", "d8") == d8_reference,
        "d8_2042_only_seed_output_and_strict_identity_changed": normalized_config(run_root / "training" / "d8_seed2042" / "run_config.json", "d8") == d8_reference,
    }
    require(all(config_checks.values()), f"R6新seed配置与seed42冻结配置不一致: {config_checks}")
    diagonal = {}
    cross_gospa = []
    cross_matched_rmse = []
    cross_direct = []
    for ch3_seed in SEEDS:
        gospa_row = []
        rmse_row = []
        direct_row = []
        for d8_seed in SEEDS:
            pair = final["cross_matrix"]["all_pairs"][f"{ch3_seed}/{d8_seed}"]
            gospa_row.append({
                "ob_ok": pair["tracks"]["OB-OK"]["gospa"]["mean"],
                "pb_base": pair["tracks"]["PB-BASE"]["gospa"]["mean"],
                "pb_selected": pair["tracks"]["PB-SELECTED"]["gospa"]["mean"],
            })
            rmse_row.append({
                "ob_ok": pair["tracks"]["OB-OK"]["matched_errors_m"]["rmse"],
                "pb_base": pair["tracks"]["PB-BASE"]["matched_errors_m"]["rmse"],
                "pb_selected": pair["tracks"]["PB-SELECTED"]["matched_errors_m"]["rmse"],
            })
            direct_row.append(pair["direct_gate"])
            if ch3_seed == d8_seed:
                diagonal[str(ch3_seed)] = {
                    "ch3_balanced_count_accuracy": pair["ch3"]["baseline"]["balanced_accuracy"],
                    "band_macro_f1": pair["ch3"]["band_macro_f1"],
                    "top2_coverage_k2_k3": pair["ch3"]["top2_coverage_k2_k3"],
                    "selected_k_accuracy_k2_k3": pair["ch3"]["selected_k_accuracy_k2_k3"],
                    "ob_ok_gospa_m": pair["tracks"]["OB-OK"]["gospa"]["mean"],
                    "pb_base_gospa_m": pair["tracks"]["PB-BASE"]["gospa"]["mean"],
                    "pb_selected_gospa_m": pair["tracks"]["PB-SELECTED"]["gospa"]["mean"],
                    "matched_rmse_m": {
                        "ob_ok": pair["tracks"]["OB-OK"]["matched_errors_m"]["rmse"],
                        "pb_base": pair["tracks"]["PB-BASE"]["matched_errors_m"]["rmse"],
                        "pb_selected": pair["tracks"]["PB-SELECTED"]["matched_errors_m"]["rmse"],
                    },
                    "matched_pair_coverage_of_true": {
                        "ob_ok": pair["tracks"]["OB-OK"]["matched_pair_coverage_of_true"],
                        "pb_base": pair["tracks"]["PB-BASE"]["matched_pair_coverage_of_true"],
                        "pb_selected": pair["tracks"]["PB-SELECTED"]["matched_pair_coverage_of_true"],
                    },
                    "pb_selected_recall_100m": pair["tracks"]["PB-SELECTED"]["set_detection"]["100m"]["recall"],
                    "direct_gate": pair["direct_gate"],
                    "selected_better_than_baseline": pair["selected_better_than_baseline_point"],
                    "bootstrap": pair["paired_bootstrap"],
                }
        cross_gospa.append(gospa_row)
        cross_matched_rmse.append(rmse_row)
        cross_direct.append(direct_row)
    report = {
        "status": "PASS", "gate": "S2-G5-R6-A", "experiment_id": "SYS-S2G5-R6A-20260829",
        "decision": final["decision"], "run_root": str(run_root),
        "configuration_identity": config_checks, "training": training,
        "diagonal": diagonal,
        "cross_matrix": {
            "row_seeds_ch3": list(SEEDS), "column_seeds_d8": list(SEEDS),
            "gospa_m": cross_gospa, "matched_rmse_m": cross_matched_rmse,
            "direct_gate": cross_direct,
            "selected_better_than_baseline": final["cross_matrix"]["selected_better_than_baseline"],
            "no_single_row_or_column_dependency": final["cross_matrix"]["no_single_row_or_column_dependency"],
        },
        "decision_counts": final["decision_counts"],
        "noncatastrophic_diagonal": final["noncatastrophic_diagonal"],
        "test_read": False, "r6b_started": False,
    }
    output_path = args.summary_output.resolve() if args.summary_output else run_root / "validation" / "r6a_summary.json"
    write_json(output_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=Path, default=R6_ROOT_DEFAULT)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    prepare = sub.add_parser("prepare-ch3")
    prepare.add_argument("--seed", type=int, choices=(1042, 2042), required=True)
    pair = sub.add_parser("pair")
    pair.add_argument("--ch3_seed", type=int, choices=SEEDS, required=True)
    pair.add_argument("--d8_seed", type=int, choices=SEEDS, required=True)
    sub.add_parser("finalize")
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--summary_output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "preflight":
        result = run_preflight(args)
    elif args.command == "prepare-ch3":
        result = run_prepare_ch3(args)
    elif args.command == "pair":
        result = run_pair(args)
    elif args.command == "finalize":
        result = run_finalize(args)
    else:
        result = run_summarize(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
