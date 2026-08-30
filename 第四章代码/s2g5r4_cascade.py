"""S2-G5-R4 16k CH3四轨更新与候选K oracle上限诊断。"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import s2g5r3_cascade as r3
from s2g5_r2_ch3 import metrics as ch3_metrics
from s2g4_coarse_d8 import build_model as build_d8_model
from train_yolo import configure_reproducibility


PROJECT_ROOT = Path(__file__).resolve().parents[1]
R4_ROOT = PROJECT_ROOT / "outputs" / "s2g5r4_ch3_scale" / "20260828_151735"
RUN_ROOT = R4_ROOT / "cascade_16k"
CH3_CHECKPOINT = R4_ROOT / "train_16k" / "best_model_v26_B_M10.pth"
SCALE_REPORT = R4_ROOT / "analysis" / "16k" / "compare_8k_16k.json"
R3_ROOT = PROJECT_ROOT / "outputs" / "s2g5r3_cascade" / "20260828_131043"
R3_FINAL = R3_ROOT / "final_report.json"
EXPECTED_CH3_SHA256 = "f2f7a7c345f1866b871282670f45671d930de34bb06493a9828a9d04a38699c4"
TRACKS = ("OB-OK", "OB-PK", "PB-OK", "PB-PK")
CANDIDATE_TRACKS = ("OB-CKO", "PB-CKO")
BOOTSTRAP_SEED = 20260904
BOOTSTRAP_REPETITIONS = 2000

r3.CH3_CHECKPOINT = CH3_CHECKPOINT
r3.EXPECTED_SHA256["ch3_checkpoint"] = EXPECTED_CH3_SHA256
r3.AUTHORITATIVE_FOUR_TRACK_DIR = "05_four_tracks_16k_batch8"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def input_identities() -> dict[str, Any]:
    identities = r3.verify_input_hashes()
    require(identities["ch3_checkpoint"]["sha256"] == EXPECTED_CH3_SHA256, "16k checkpoint SHA错误")
    require(SCALE_REPORT.is_file() and R3_FINAL.is_file(), "规模或R3基准报告缺失")
    return {
        **identities,
        "scale_report": r3.file_identity(SCALE_REPORT),
        "r3_final": r3.file_identity(R3_FINAL),
    }


def run_preflight() -> dict[str, Any]:
    identities = input_identities()
    checkpoint = torch.load(CH3_CHECKPOINT, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    require(checkpoint["epoch"] == 78, "16k best epoch不是78")
    require(config["mode"] == "transformer" and config["max_src"] == 10, "CH3模型契约错误")
    require(config["threshold"] == 0.5 and config["seed"] == 42, "CH3阈值或seed错误")
    metadata = r3.load_compare_metadata()
    counts = np.asarray(metadata["source_count"], dtype=np.int64)
    require(r3.histogram(counts) == {"0": 256, "1": 256, "2": 256, "3": 256}, "validation分层错误")
    report = {
        "status": "PASS",
        "gate": "S2-G5-R4",
        "stage": "cascade_preflight",
        "inputs": identities,
        "ch3_checkpoint_epoch": 78,
        "sample_count": 1024,
        "localization_count": int(np.isin(counts, (2, 3)).sum()),
        "oracle_fine_reused": True,
        "test_executed": False,
    }
    write_json(RUN_ROOT / "00_preflight" / "preflight.json", report)
    return report


def alternative_k(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    slot_scores = probabilities.max(axis=2)
    active = slot_scores > 0.5
    predicted = active.sum(axis=1).astype(np.int64)
    closest = np.argmin(np.abs(slot_scores - 0.5), axis=1)
    alternative = predicted + np.where(active[np.arange(len(active)), closest], -1, 1)
    valid = (alternative >= 0) & (alternative <= 3)
    return alternative.astype(np.int64), valid


def run_infer() -> dict[str, Any]:
    input_identities()
    arrays = r3.infer_ch3_arrays(r3.COARSE_COMPARE)
    scale = load_json(SCALE_REPORT)
    current_metrics = ch3_metrics({
        "prediction": arrays["prediction"],
        "truth": arrays["truth"],
        "ignore": arrays["ignore"],
        "source_count": arrays["source_count"],
    })
    require(r3.nested_max_abs(current_metrics, scale["metrics_16k"]) <= 1e-12, "16k指标未精确复现")
    alternative, valid = alternative_k(arrays["probabilities"])
    output = RUN_ROOT / "01_ch3_inference" / "ch3_predictions.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), f"拒绝覆盖: {output}")
    metadata = r3.load_compare_metadata()
    np.savez_compressed(
        output,
        logits=arrays["logits"],
        probabilities=arrays["probabilities"],
        band_prediction=arrays["prediction"].astype(np.uint8),
        band_truth=arrays["truth"].astype(np.uint8),
        ignore_mask=arrays["ignore"].astype(np.uint8),
        source_count=arrays["source_count"],
        k_prediction=arrays["k_prediction"],
        alternative_k=alternative,
        alternative_k_valid=valid,
        raw_sample_idx=np.asarray(metadata["sample_idx"], dtype=np.int64),
    )
    counts = arrays["source_count"]
    confusion = np.zeros((4, 4), dtype=np.int64)
    for truth, prediction in zip(counts, arrays["k_prediction"], strict=True):
        confusion[int(truth), int(prediction)] += 1
    report = {
        "status": "PASS",
        "gate": "S2-G5-R4",
        "stage": "ch3_inference_16k",
        "checkpoint": r3.file_identity(CH3_CHECKPOINT),
        "metrics": current_metrics,
        "metrics_match_scale_report_max_abs": 0.0,
        "confusion_true_rows_pred_columns_k0_k3": confusion.tolist(),
        "prediction_histogram": r3.histogram(arrays["k_prediction"]),
        "alternative_k_invalid_count": int((~valid).sum()),
        "prediction_output": r3.file_identity(output),
        "test_executed": False,
    }
    write_json(RUN_ROOT / "01_ch3_inference" / "inference_report.json", report)
    return report


def run_reuse_oracle() -> dict[str, Any]:
    source = R3_ROOT / "03_fine_oracle" / "index.json"
    index = load_json(source)
    require(index["status"] == "PASS" and index["sample_count"] == 512, "R3 Oracle索引错误")
    for shard in index["shards"]:
        require(r3.sha256_file(Path(shard["path"])) == shard["sha256"], "R3 Oracle shard SHA变化")
    report = {
        **index,
        "gate": "S2-G5-R4",
        "stage": "reuse_r3_oracle_fine",
        "reused_from": r3.file_identity(source),
        "no_payload_copy": True,
    }
    write_json(RUN_ROOT / "03_fine_oracle" / "index.json", report)
    return report


def run_build_predicted() -> dict[str, Any]:
    return r3.run_build_fine(SimpleNamespace(
        run_root=RUN_ROOT,
        mode="predicted",
        chunk_size=40000,
        shard_size=64,
    ))


def evaluate_once() -> dict[str, list[dict[str, Any]]]:
    configure_reproducibility(42, True)
    device = torch.device("cuda:0")
    model, _ = build_d8_model(r3.D8_CHECKPOINT, device)
    predictions = np.load(RUN_ROOT / "01_ch3_inference" / "ch3_predictions.npz")
    alternative = predictions["alternative_k"].astype(np.int64)
    alternative_valid = predictions["alternative_k_valid"].astype(bool)
    records = {track: [] for track in TRACKS + CANDIDATE_TRACKS}
    for mode, prefix in (("oracle", "OB"), ("predicted", "PB")):
        index_path = RUN_ROOT / ("03_fine_oracle/index.json" if mode == "oracle" else "04_fine_predicted/index.json")
        _, shards = r3.load_fine_shards(index_path)
        for shard in shards:
            fine = shard["fine_dpd"].float()
            normalized = torch.stack([(sample - sample.mean()) / (sample.std() + 1e-6) for sample in fine])
            heatmap_parts = []
            offset_parts = []
            with torch.no_grad():
                for start in range(0, len(normalized), 8):
                    current_heatmap, current_offset = model(normalized[start:start + 8].to(device))
                    heatmap_parts.append(current_heatmap.cpu())
                    offset_parts.append(current_offset.cpu())
            heatmap = torch.cat(heatmap_parts)
            offset = torch.cat(offset_parts)
            require(bool(torch.isfinite(heatmap).all() and torch.isfinite(offset).all()), "D8输出异常")
            for row in range(len(fine)):
                local = int(shard["local_idx"][row])
                raw = int(shard["raw_idx"][row])
                true_count = int(shard["n_src"][row])
                predicted_k = int(shard["predicted_k"][row])
                true_positions = shard["pos_label"][row, :true_count].numpy() * r3.EDGE
                common = dict(
                    local_index=local,
                    raw_index=raw,
                    true_count=true_count,
                    true_positions=true_positions,
                    heatmap=heatmap[row],
                    offset=offset[row],
                    empty_band=bool(shard["empty_band"][row]),
                )
                ok = r3.sample_record(track=f"{prefix}-OK", requested_count=true_count, **common)
                pk = r3.sample_record(track=f"{prefix}-PK", requested_count=predicted_k, **common)
                records[f"{prefix}-OK"].append(ok)
                records[f"{prefix}-PK"].append(pk)
                candidates = [pk]
                if alternative_valid[local] and int(alternative[local]) != predicted_k:
                    candidates.append(r3.sample_record(
                        track=f"{prefix}-ALT", requested_count=int(alternative[local]), **common,
                    ))
                chosen = min(candidates, key=lambda item: item["gospa_m"])
                candidate = {**chosen, "track": f"{prefix}-CKO"}
                candidate["candidate_k_values"] = [predicted_k] + (
                    [int(alternative[local])] if len(candidates) == 2 else []
                )
                candidate["oracle_candidate_selection"] = True
                records[f"{prefix}-CKO"].append(candidate)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def run_evaluate() -> dict[str, Any]:
    input_identities()
    primary = evaluate_once()
    replay = evaluate_once()
    difference = r3.nested_max_abs(primary, replay)
    require(difference <= 1e-12, f"D8重载不一致: {difference}")
    output_dir = RUN_ROOT / r3.AUTHORITATIVE_FOUR_TRACK_DIR
    identities = {}
    for track, rows in primary.items():
        require(len(rows) == 512, f"{track}样本数错误")
        samples = output_dir / f"{track.lower().replace('-', '_')}_samples.jsonl"
        r3.write_jsonl(samples, rows)
        report = r3.track_report(track, rows)
        report["gate"] = "S2-G5-R4"
        report["candidate_oracle_upper_bound"] = track.endswith("CKO")
        report_path = output_dir / f"{track.lower().replace('-', '_')}.json"
        r3.write_json(report_path, report)
        identities[track] = r3.file_identity(report_path)
    audit = {
        "status": "PASS",
        "gate": "S2-G5-R4",
        "stage": "four_track_and_candidate_evaluation",
        "tracks": identities,
        "checkpoint_reload_max_abs_difference": difference,
        "sample_count_per_track": 512,
        "test_executed": False,
    }
    write_json(output_dir / "evaluation_audit.json", audit)
    return audit


def bootstrap_candidate(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    counts = np.asarray([row["true_count"] for row in rows["OB-OK"]], dtype=np.int64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    recoveries = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        selected = np.concatenate([
            rng.choice(np.flatnonzero(counts == count), size=int(np.sum(counts == count)), replace=True)
            for count in (2, 3)
        ])
        ok = r3.point_track_values(rows["OB-OK"], selected)[0]
        pk = r3.point_track_values(rows["OB-PK"], selected)[0]
        cko = r3.point_track_values(rows["OB-CKO"], selected)[0]
        denominator = pk - ok
        recoveries.append((pk - cko) / denominator if denominator > 1e-12 else 0.0)
    values = np.asarray(recoveries, dtype=np.float64)
    return {"mean": float(values.mean()), "ci95": r3.ci95(values)}


def run_analyze() -> dict[str, Any]:
    standard = r3.run_analyze(SimpleNamespace(run_root=RUN_ROOT))
    output_dir = RUN_ROOT / r3.AUTHORITATIVE_FOUR_TRACK_DIR
    rows = {
        track: r3.load_jsonl(output_dir / f"{track.lower().replace('-', '_')}_samples.jsonl")
        for track in TRACKS + CANDIDATE_TRACKS
    }
    point = {track: r3.point_track_values(current) for track, current in rows.items()}
    old = load_json(R3_FINAL)
    old_count = float(old["effects"]["count_gospa_delta_ob_pk_minus_ob_ok"])
    new_count = float(standard["effects"]["count_gospa_delta_ob_pk_minus_ob_ok"])
    count_reduction = (old_count - new_count) / old_count
    scale = load_json(SCALE_REPORT)
    balanced = scale["paired_bootstrap"]["balanced_count_accuracy"]
    band = scale["paired_bootstrap"]["active_band_macro_f1"]
    class_delta = {
        key: scale["metrics_16k"]["count_class_accuracy"][key] - scale["metrics_8k"]["count_class_accuracy"][key]
        for key in ("0", "1", "2", "3")
    }
    false_alarm_delta = scale["metrics_16k"]["zero_source_false_alarm_rate"] - scale["metrics_8k"]["zero_source_false_alarm_rate"]
    scale_gate = (
        scale["metrics_16k"]["balanced_count_accuracy"] >= scale["metrics_8k"]["balanced_count_accuracy"]
        and scale["metrics_16k"]["active_band_macro_f1"] >= scale["metrics_8k"]["active_band_macro_f1"]
        and max(balanced["mean_delta_16k_minus_8k"], band["mean_delta_16k_minus_8k"]) >= 0.01
        and min(balanced["ci95"][0], band["ci95"][0]) >= -0.01
        and min(class_delta.values()) >= -0.05
        and false_alarm_delta <= 0.02
    )
    candidate_point = (point["OB-PK"][0] - point["OB-CKO"][0]) / max(point["OB-PK"][0] - point["OB-OK"][0], 1e-12)
    candidate_bootstrap = bootstrap_candidate(rows)
    top2 = scale["confidence_16k"]["top2_coverage_among_errors"]
    direct = standard["decision"]["interface_status"] == "DIRECT_CASCADE_SUPPORTED"
    if direct:
        decision = "DIRECT_CASCADE_SUPPORTED_16K"
    elif scale_gate and count_reduction >= 0.20:
        decision = "RECOMMEND_32K_SCALE"
    elif top2 >= 0.80 and candidate_point >= 0.50:
        decision = "ENTER_K_CANDIDATE_GATE"
    else:
        decision = "ENTER_CH3_MODEL_DATA_DIAGNOSIS"
    report = {
        "status": "PASS",
        "gate": "S2-G5-R4",
        "standard_four_track": standard,
        "candidate_track_point_metrics": {
            track: {"mean_gospa_m": value[0], "recall_100m": value[1]}
            for track, value in point.items() if track in CANDIDATE_TRACKS
        },
        "scale_gate_pass": scale_gate,
        "scale_gate_details": {
            "class_accuracy_delta": class_delta,
            "zero_source_false_alarm_delta": false_alarm_delta,
        },
        "count_path_loss_8k_m": old_count,
        "count_path_loss_16k_m": new_count,
        "count_path_loss_relative_reduction": count_reduction,
        "top2_coverage_among_16k_errors": top2,
        "candidate_oracle_recovery_point": candidate_point,
        "candidate_oracle_recovery_bootstrap": candidate_bootstrap,
        "decision": decision,
        "test_executed": False,
    }
    write_json(RUN_ROOT / "07_r4_analysis" / "r4_analysis.json", report)
    return report


def run_finalize() -> dict[str, Any]:
    input_identities()
    reports = {
        "preflight": RUN_ROOT / "00_preflight" / "preflight.json",
        "inference": RUN_ROOT / "01_ch3_inference" / "inference_report.json",
        "oracle": RUN_ROOT / "03_fine_oracle" / "index.json",
        "predicted": RUN_ROOT / "04_fine_predicted" / "index.json",
        "evaluation": RUN_ROOT / r3.AUTHORITATIVE_FOUR_TRACK_DIR / "evaluation_audit.json",
        "standard_analysis": RUN_ROOT / "06_analysis" / "paired_analysis.json",
        "r4_analysis": RUN_ROOT / "07_r4_analysis" / "r4_analysis.json",
    }
    payloads = {name: load_json(path) for name, path in reports.items()}
    require(all(payload["status"] == "PASS" for payload in payloads.values()), "R4级联阶段未全部通过")
    monitors = [load_json(path) for path in (RUN_ROOT / "monitor").glob("*/stage_monitor_report.json")]
    pass_monitors = [item for item in monitors if item["status"] == "PASS"]
    failed_monitors = [item for item in monitors if item["status"] != "PASS"]
    require(
        {(item["stage"], item["status"]) for item in failed_monitors}
        == {("r4_four_track_candidate_eval", "COMPLETED_WITH_RED_FLAGS")},
        "存在未解释的R4级联失败监控",
    )
    require(
        "r4_four_track_candidate_eval_batch8" in {item["stage"] for item in pass_monitors},
        "缺少batch8权威评估PASS证据",
    )
    analysis = payloads["r4_analysis"]
    report = {
        "status": "PASS",
        "gate": "S2-G5-R4",
        "experiment_id": "SYS-S2G5-R4-20260828",
        "scientific_decision": analysis["decision"],
        "analysis": analysis,
        "inputs_unchanged": input_identities(),
        "stage_reports": {name: r3.file_identity(path) for name, path in reports.items()},
        "monitor_summary": {
            "authoritative_pass_count": len(pass_monitors),
            "authoritative_pass_stages": sorted(item["stage"] for item in pass_monitors),
            "retained_failed_attempts": [
                {"stage": item["stage"], "status": item["status"], "red_flags": item.get("red_flags", [])}
                for item in failed_monitors
            ],
            "minimum_system_available_bytes": min(item["minimum_system_available_bytes"] for item in pass_monitors),
            "maximum_process_tree_rss_bytes": max(item["maximum_process_tree_rss_bytes"] for item in pass_monitors),
            "maximum_gpu_used_mib": max(item["maximum_gpu_used_mib"] or 0 for item in pass_monitors),
            "warning_union": sorted({v for item in pass_monitors for v in item.get("warnings", [])}),
            "red_flag_union": sorted({v for item in pass_monitors for v in item.get("red_flags", [])}),
        },
        "prohibitions": {
            "test_executed": False,
            "threshold_tuned": False,
            "model_or_loss_changed": False,
            "candidate_oracle_reported_as_deployable": False,
            "entered_32k_or_k_gate": False,
        },
    }
    write_json(RUN_ROOT / "final_report.json", report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="S2-G5-R4 16k四轨与候选K上限")
    parser.add_argument("command", choices=(
        "preflight", "infer", "reuse-oracle", "build-predicted", "evaluate", "analyze", "finalize",
    ))
    return parser.parse_args()


def main() -> int:
    command = parse_args().command
    handlers = {
        "preflight": run_preflight,
        "infer": run_infer,
        "reuse-oracle": run_reuse_oracle,
        "build-predicted": run_build_predicted,
        "evaluate": run_evaluate,
        "analyze": run_analyze,
        "finalize": run_finalize,
    }
    started = time.perf_counter()
    result = handlers[command]()
    print(json.dumps({"status": result["status"], "command": command, "seconds": time.perf_counter() - started}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
