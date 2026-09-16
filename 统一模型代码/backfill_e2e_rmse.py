"""为E2E-G2-R2与E2E-G3冻结预测补算匹配定位误差。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=False)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "sha256": digest.hexdigest(),
    }


def locate_val_compare(manifest: dict[str, Any]) -> Path:
    candidates = [
        Path(item["path"])
        for item in manifest["inputs"]["files"]
        if Path(item["path"]).name == "val_compare.mat"
    ]
    if len(candidates) != 1 or not candidates[0].is_file():
        raise RuntimeError(f"无法唯一定位val_compare快照: {candidates}")
    return candidates[0]


def truth_positions(handle: h5py.File, row: dict[str, Any]) -> np.ndarray:
    local = int(row["local_index"])
    count = int(row["true_count"])
    observed_count = int(np.asarray(handle["src_count_all"][:, local]).item())
    raw_index = int(np.asarray(handle["sample_idx_all"][:, local]).item())
    if observed_count != count or raw_index != int(row["raw_index"]):
        raise RuntimeError(
            f"样本身份不一致: local={local}, count={observed_count}/{count}, "
            f"raw={raw_index}/{row['raw_index']}"
        )
    return np.asarray(handle["src_pos_all"][:, :count, local], dtype=np.float64).T


def matched_errors(truth: np.ndarray, predicted: np.ndarray) -> list[float]:
    if len(truth) == 0 or len(predicted) == 0:
        return []
    distances = np.linalg.norm(truth[:, None, :] - predicted[None, :, :], axis=2)
    left, right = linear_sum_assignment(distances)
    return [float(distances[i, j]) for i, j in zip(left, right)]


def numeric(errors: list[float], true_sources: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "true_source_count": int(true_sources),
        "matched_pair_count": len(errors),
        "matched_pair_coverage_of_true": len(errors) / max(true_sources, 1),
    }
    if not errors:
        result["matched_errors_m"] = None
        return result
    values = np.asarray(errors, dtype=np.float64)
    result["matched_errors_m"] = {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [value for row in rows for value in row["matched_errors_m"]]
    result = numeric(errors, sum(int(row["true_count"]) for row in rows))
    result["sample_count"] = len(rows)
    return result


def evaluate_rows(
    rows: list[dict[str, Any]], handle: h5py.File
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        truth = truth_positions(handle, row)
        predicted = np.asarray(row["predicted_positions_m"], dtype=np.float64).reshape(-1, 2)
        errors = matched_errors(truth, predicted)
        enriched.append(
            {
                "raw_index": int(row["raw_index"]),
                "local_index": int(row["local_index"]),
                "true_count": int(row["true_count"]),
                "predicted_count": int(row["predicted_count"]),
                "frequency_overlap": bool(row["frequency_overlap"]),
                "matched_errors_m": errors,
            }
        )
    summary = {
        "overall": summarize(enriched),
        "by_k": {
            str(count): summarize(
                [row for row in enriched if int(row["true_count"]) == count]
            )
            for count in range(4)
        },
        "by_frequency_overlap": {
            str(value).lower(): summarize(
                [row for row in enriched if bool(row["frequency_overlap"]) is value]
            )
            for value in (False, True)
        },
    }
    return summary, enriched


def bootstrap_rmse_difference(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    if len(left) != len(right):
        raise RuntimeError("配对轨样本数不一致")
    for a, b in zip(left, right):
        if (a["raw_index"], a["local_index"]) != (b["raw_index"], b["local_index"]):
            raise RuntimeError("配对轨样本顺序不一致")
    by_k = {
        count: [i for i, row in enumerate(left) if int(row["true_count"]) == count]
        for count in range(4)
    }

    def rmse(rows: list[dict[str, Any]], indices: np.ndarray) -> float:
        errors = [value for index in indices for value in rows[int(index)]["matched_errors_m"]]
        if not errors:
            return math.nan
        values = np.asarray(errors, dtype=np.float64)
        return float(np.sqrt(np.mean(values**2)))

    all_indices = np.arange(len(left))
    point_left = rmse(left, all_indices)
    point_right = rmse(right, all_indices)
    rng = np.random.default_rng(seed)
    differences: list[float] = []
    for _ in range(repetitions):
        selected = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in by_k.values()]
        )
        difference = rmse(right, selected) - rmse(left, selected)
        if math.isfinite(difference):
            differences.append(difference)
    return {
        "right_minus_left_rmse_m": point_right - point_left,
        "ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "repetitions": len(differences),
        "resampling_unit": "whole_sample_stratified_by_true_k",
    }


def run(g2_run: Path, g3_run: Path, output_root: Path) -> Path:
    g2_manifest_path = g2_run / "manifest.json"
    g2_result_path = g2_run / "p3_report.json"
    g3_result_path = g3_run / "p2_report.json"
    g2_manifest = read_json(g2_manifest_path)
    g2_result = read_json(g2_result_path)
    g3_result = read_json(g3_result_path)
    val_compare = locate_val_compare(g2_manifest)
    started = time.perf_counter()
    report: dict[str, Any] = {
        "status": "PASS",
        "gate": "E2E-RMSE-BACKFILL",
        "method": "Hungarian minimum Euclidean distance over predicted hard output sets",
        "inputs": {
            "g2_manifest": identity(g2_manifest_path),
            "g2_result": identity(g2_result_path),
            "g3_result": identity(g3_result_path),
            "val_compare": identity(val_compare),
        },
        "test_executed": False,
        "g2_r2": {},
        "g3": {},
    }
    with h5py.File(val_compare, "r") as handle:
        g2_enriched: dict[str, list[dict[str, Any]]] = {}
        for track, rows in g2_result["samples"].items():
            summary, enriched = evaluate_rows(rows, handle)
            report["g2_r2"][track] = summary
            g2_enriched[track] = enriched
        report["g2_r2"]["paired_bootstrap"] = bootstrap_rmse_difference(
            g2_enriched["fs_sg"],
            g2_enriched["fs_e2e"],
            repetitions=2000,
            seed=20260905,
        )

        for seed_text, seed_report in g3_result["confirmation_seeds"].items():
            enriched_tracks: dict[str, list[dict[str, Any]]] = {}
            seed_output: dict[str, Any] = {}
            for track, rows in seed_report["samples"]["localization"].items():
                summary, enriched = evaluate_rows(rows, handle)
                seed_output[track] = summary
                enriched_tracks[track] = enriched
            seed_output["paired_bootstrap"] = bootstrap_rmse_difference(
                enriched_tracks["fs_sg"],
                enriched_tracks["fs_e2e"],
                repetitions=2000,
                seed=int(seed_text) + 303,
            )
            report["g3"][seed_text] = seed_output
    report["duration_seconds"] = time.perf_counter() - started
    run_root = output_root / time.strftime("%Y%m%d_%H%M%S")
    write_json(run_root / "rmse_backfill_report.json", report)
    return run_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-run", type=Path, required=True)
    parser.add_argument("--g3-run", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_rmse_backfill",
    )
    args = parser.parse_args()
    root = run(args.g2_run.resolve(), args.g3_run.resolve(), args.output_root.resolve())
    print(root)


if __name__ == "__main__":
    main()
