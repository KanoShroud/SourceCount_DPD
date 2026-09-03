"""E2E-G2 P0/P1：接口等价、固定全频细 DPD 兼容性与速度。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
CH4_DIR = PROJECT_ROOT / "第四章代码"
os.environ.setdefault(
    "SOURCECOUNT_REFERENCE_OUTPUT_ROOT",
    str((PROJECT_ROOT.parent / "SourceCount_DPD" / "outputs").resolve()),
)
for root in (PROJECT_ROOT, CH4_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from gen_ch4_loc_data import group_sources_by_freq_overlap  # noqa: E402
from yolo_model import extract_peaks_topn  # noqa: E402

from 统一模型代码 import e2e_g1 as g1  # noqa: E402
from 统一模型代码.models.e2e_latent_fusion import (  # noqa: E402
    FrequencySpatialSplitter,
    SourceLocalizationHead,
    SourceQueryBuilder,
    forward_ch3_features,
    forward_d8_features,
)
from 统一模型代码.runtime_paths import new_run_dir, validate_output_path  # noqa: E402


CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g2_latent_fusion.json"
SOURCE_RUN = PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_g1" / "20260901_145307"
SOURCE_MANIFEST = SOURCE_RUN / "manifest.json"
SNAPSHOT_MANIFEST = SOURCE_RUN / "reference_snapshot" / "snapshot_manifest.json"
CODE_PATHS = [Path(__file__), PACKAGE_ROOT / "models" / "e2e_latent_fusion.py", CONFIG_PATH]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def configure_snapshot() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    snapshot = read_json(SNAPSHOT_MANIFEST)
    if snapshot["status"] != "PASS":
        raise RuntimeError("冻结参考快照未通过")
    files = [row["snapshot"] for row in snapshot["files"]]
    artifacts = []
    for row in snapshot["artifacts"]:
        item = dict(row["snapshot"])
        item["name"] = row["expected"]["name"]
        artifacts.append(item)
    for expected in files + artifacts:
        current = identity(Path(expected["path"]))
        if current["size_bytes"] != expected["size_bytes"] or current["sha256"] != expected["sha256"]:
            raise RuntimeError(f"冻结输入身份变化: {expected['path']}")
    g1.USING_LOCAL_SNAPSHOT = True
    g1.COARSE_TRAIN = Path(files[0]["path"])
    g1.COARSE_VAL_SELECT = Path(files[1]["path"])
    g1.COARSE_VAL_COMPARE = Path(files[2]["path"])
    g1.RAW_VALIDATION = Path(files[3]["path"])
    g1.RAW_TRAIN_PARTS = (
        (0, 4096, Path(files[4]["path"])),
        (4096, 8192, Path(files[5]["path"])),
        (8192, 16384, Path(files[6]["path"])),
    )

    def verify(names: list[str]) -> list[dict[str, Any]]:
        indexed = {row["name"]: row for row in artifacts}
        return [{"name": name, **identity(Path(indexed[name]["path"]))} for name in names]

    g1.verify_artifacts = verify
    return files, artifacts


def choose_p1_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    source = read_json(SOURCE_MANIFEST)["subsets"]["val_select"]
    per_k = int(config["p1_samples_per_k"])
    selected = []
    for count in range(4):
        rows = [row for row in source if int(row["true_k"]) == count]
        if len(rows) < per_k:
            raise RuntimeError(f"P1 K={count}样本不足")
        selected.extend(rows[:per_k])
    return selected


def prepare(run_id: str) -> Path:
    config = read_json(CONFIG_PATH)
    files, artifacts = configure_snapshot()
    run_root = new_run_dir("e2e_g2_latent_fusion", run_id, create=True)
    manifest = {
        "status": "PREPARED",
        "gate": "E2E-G2",
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_read_only": True,
        "test_executed": False,
        "config": config,
        "code": [identity(path) for path in CODE_PATHS],
        "source_manifest": identity(SOURCE_MANIFEST),
        "snapshot_manifest": identity(SNAPSHOT_MANIFEST),
        "inputs": {"files": files, "artifacts": artifacts},
        "subsets": {"p1": choose_p1_records(config)},
    }
    write_json(run_root / "manifest.json", manifest)
    print(run_root, flush=True)
    return run_root


def verify_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_root / "manifest.json")
    config = read_json(CONFIG_PATH)
    if manifest["gate"] != "E2E-G2" or manifest["test_executed"] is not False:
        raise RuntimeError("G2 manifest契约错误")
    configure_snapshot()
    if [identity(path) for path in CODE_PATHS] != manifest["code"]:
        raise RuntimeError("prepare后代码或配置发生变化")
    return manifest, config


def run_p0(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(config["seed"]))
    ch3, d8, checkpoints = g1.build_models(device)
    ch3.eval(); d8.eval()
    records = [
        next(row for row in manifest["subsets"]["p1"] if int(row["true_k"]) == count)
        for count in range(4)
    ]
    with g1.SampleStore("val_select") as store:
        samples = [store.sample(record) for record in records]
    coarse = torch.stack([sample["coarse_dpd"] for sample in samples]).to(device)
    synthetic_dpd = torch.rand((4, 1, 401, 401), dtype=torch.float32, device=device)
    with torch.no_grad():
        old_ch3 = ch3(coarse)
        ch3_features = forward_ch3_features(ch3, coarse)
        old_heatmap, old_offset = d8(synthetic_dpd)
        d8_features = forward_d8_features(d8, synthetic_dpd)
        query_builder = SourceQueryBuilder().to(device).eval()
        splitter = FrequencySpatialSplitter().to(device).eval()
        source_head = SourceLocalizationHead().to(device).eval()
        query, refined_logits = query_builder(ch3_features)
        source_spatial, spatial_attention = splitter(
            ch3_features.spatial, query, refined_logits
        )
        source_heatmap, source_offset = source_head(
            d8_features.d0, source_spatial, query
        )
    checks = {
        "covers_k_0_1_2_3": [int(sample["true_k"]) for sample in samples] == [0, 1, 2, 3],
        "batch_larger_than_one": coarse.shape[0] > 1,
        "ch3_spatial_shape": list(ch3_features.spatial.shape) == [4, 19, 128, 11, 11],
        "ch3_token_shape": list(ch3_features.tokens.shape) == [4, 19, 128],
        "ch3_global_shape": list(ch3_features.global_feature.shape) == [4, 256],
        "ch3_hidden_shape": list(ch3_features.head_hidden.shape) == [4, 10, 64],
        "ch3_logits_shape": list(ch3_features.band_logits.shape) == [4, 10, 19],
        "d8_d0_shape": list(d8_features.d0.shape) == [4, 32, 401, 401],
        "d8_heatmap_shape": list(d8_features.heatmap.shape) == [4, 1, 401, 401],
        "d8_offset_shape": list(d8_features.offset.shape) == [4, 2, 401, 401],
        "source_query_shape": list(query.shape) == [4, 3, 128],
        "refined_logits_shape": list(refined_logits.shape) == [4, 3, 19],
        "source_spatial_shape": list(source_spatial.shape) == [4, 3, 32, 11, 11],
        "spatial_attention_shape": list(spatial_attention.shape) == [4, 3, 11, 11],
        "source_heatmap_shape": list(source_heatmap.shape) == [4, 3, 401, 401],
        "source_offset_shape": list(source_offset.shape) == [4, 3, 2, 401, 401],
    }
    differences = {
        "ch3_logits_max_abs": float((old_ch3 - ch3_features.band_logits).abs().max().item()),
        "d8_heatmap_max_abs": float((old_heatmap - d8_features.heatmap).abs().max().item()),
        "d8_offset_max_abs": float((old_offset - d8_features.offset).abs().max().item()),
    }
    limit = float(config["bypass_atol"])
    checks["bypass_equivalent"] = all(value <= limit for value in differences.values())
    checks["finite"] = all(
        bool(torch.isfinite(value).all())
        for value in (
            ch3_features.spatial,
            ch3_features.tokens,
            d8_features.d0,
            query,
            refined_logits,
            source_spatial,
            spatial_attention,
            source_heatmap,
            source_offset,
        )
    )
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "differences": differences,
        "checkpoint": checkpoints,
        "device": str(device),
    }
    write_json(run_root / "p0_report.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def frequency_mask(lo_hz: float, hi_hz: float, device: torch.device) -> torch.Tensor:
    frequency = torch.arange(-g1.N_FFT // 2, g1.N_FFT // 2, dtype=torch.float64, device=device)
    frequency = frequency * (g1.FS / g1.N_FFT)
    return (frequency >= float(lo_hz)) & (frequency < float(hi_hz))


def dpd_map(signal: np.ndarray, weights: torch.Tensor, geometry: Any, config: dict[str, Any]) -> torch.Tensor:
    return g1.compute_fine_dpd_autograd(
        signal,
        geometry,
        weights.to(torch.float64),
        fixed_support=weights.bool(),
        grid_chunk_size=int(config["grid_chunk"]),
        frequency_chunk_size=int(config["frequency_chunk"]),
        eig_device=str(config["eig_device"]),
        use_checkpoint=False,
        real_dtype=torch.float64,
    )


def decode_known_k(d8: torch.nn.Module, dpd: torch.Tensor, count: int) -> np.ndarray:
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    with torch.no_grad():
        heatmap, offset = d8(g1.d8_input(dpd))
        pixels, _ = extract_peaks_topn(heatmap[0], count, peak_size=g1.PEAK_SIZE)
        xi = pixels[:, 0].long(); yi = pixels[:, 1].long()
        delta = offset[0, :, yi, xi].T.clamp(-1.0, 1.0)
        positions = (pixels + delta) * g1.FINE_STEP - g1.FINE_EDGE
    return positions.detach().cpu().numpy().astype(np.float32)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nonzero_true = sum(int(row["true_count"]) for row in rows)
    return {
        "sample_count": len(rows),
        "gospa_mean_m": float(np.mean([row["gospa_m"] for row in rows])),
        "recall_at_100m": sum(int(row["tp_at_100m"]) for row in rows) / max(nonzero_true, 1),
        "dpd_seconds": float(sum(row["dpd_seconds"] for row in rows)),
        "d8_seconds": float(sum(row["d8_seconds"] for row in rows)),
    }


def cache_description(path: Path, array: np.ndarray, read_seconds: float) -> dict[str, Any]:
    return {
        **identity(path),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "finite": bool(np.isfinite(array).all()),
        "nonconstant": bool(float(array.std()) > 0.0),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "standard_deviation": float(array.std()),
        "read_seconds": read_seconds,
    }


def run_p1(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    p0 = read_json(run_root / "p0_report.json")
    if p0["status"] != "PASS":
        raise RuntimeError("P0未通过，禁止执行P1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, d8, _ = g1.build_models(device)
    d8.eval()
    geometry = g1.receiver_geometry(device)
    all_weights = torch.ones(g1.N_FFT, dtype=torch.float64, device=device)
    rows_grouped: list[dict[str, Any]] = []
    rows_full: list[dict[str, Any]] = []
    cache_root = validate_output_path(run_root / "cache")
    cache_root.mkdir()
    started = time.perf_counter()
    with g1.SampleStore("val_select") as store:
        for ordinal, record in enumerate(manifest["subsets"]["p1"], start=1):
            sample = store.sample(record)
            count = int(sample["true_k"])
            true_positions = np.asarray(sample["positions_m"][:count], dtype=np.float32)
            raw_index = int(sample["raw_index"])
            raw_handle, raw_local = store._raw(raw_index)
            fc = np.asarray(raw_handle["fc_offset_all"][:, raw_local], dtype=np.float64).reshape(-1)
            bw = np.asarray(raw_handle["BW_actual_all"][:, raw_local], dtype=np.float64).reshape(-1)

            full_path = cache_root / f"full_{ordinal:03d}.npy"
            dpd_started = time.perf_counter()
            full_dpd = dpd_map(sample["signal"], all_weights, geometry, config)
            full_dpd_seconds = time.perf_counter() - dpd_started
            full_array = full_dpd.detach().cpu().numpy().astype(np.float32)
            np.save(full_path, full_array)
            read_started = time.perf_counter()
            loaded_full = np.load(full_path, allow_pickle=False)
            full_read_seconds = time.perf_counter() - read_started
            full_cache = cache_description(full_path, loaded_full, full_read_seconds)
            d8_started = time.perf_counter()
            full_pred = decode_known_k(d8, full_dpd, count)
            full_d8_seconds = time.perf_counter() - d8_started
            full_gospa = g1.gospa_sample(true_positions, full_pred)
            rows_full.append({
                "ordinal": ordinal, "raw_index": raw_index, "true_count": count,
                "predicted_positions_m": full_pred.tolist(), "gospa_m": full_gospa["value_m"],
                "tp_at_100m": g1.maximum_matches_within(true_positions, full_pred, 100.0),
                "dpd_seconds": full_dpd_seconds, "d8_seconds": full_d8_seconds,
                "cache": full_cache,
            })

            grouped_predictions = []
            grouped_dpd_seconds = 0.0
            grouped_d8_seconds = 0.0
            for group_index, group in enumerate(group_sources_by_freq_overlap(fc, bw, count)):
                mask = frequency_mask(group["freq_lo"], group["freq_hi"], device)
                dpd_started = time.perf_counter()
                current_dpd = dpd_map(sample["signal"], mask.to(torch.float64), geometry, config)
                grouped_dpd_seconds += time.perf_counter() - dpd_started
                group_path = cache_root / f"group_{ordinal:03d}_{group_index:02d}.npy"
                group_array = current_dpd.detach().cpu().numpy().astype(np.float32)
                np.save(
                    group_path,
                    group_array,
                )
                d8_started = time.perf_counter()
                grouped_predictions.append(decode_known_k(d8, current_dpd, int(group["n_src"])))
                grouped_d8_seconds += time.perf_counter() - d8_started
            grouped_pred = np.concatenate(grouped_predictions, axis=0) if grouped_predictions else np.empty((0, 2), dtype=np.float32)
            grouped_gospa = g1.gospa_sample(true_positions, grouped_pred)
            rows_grouped.append({
                "ordinal": ordinal, "raw_index": raw_index, "true_count": count,
                "predicted_positions_m": grouped_pred.tolist(), "gospa_m": grouped_gospa["value_m"],
                "tp_at_100m": g1.maximum_matches_within(true_positions, grouped_pred, 100.0),
                "dpd_seconds": grouped_dpd_seconds, "d8_seconds": grouped_d8_seconds,
            })
            print(f"[P1] {ordinal}/{len(manifest['subsets']['p1'])} K={count} full={full_dpd_seconds:.1f}s grouped={grouped_dpd_seconds:.1f}s", flush=True)

    grouped = metrics(rows_grouped); full = metrics(rows_full)
    cache_checks = {
        "all_full_maps_finite": all(row["cache"]["finite"] for row in rows_full),
        "all_full_maps_nonconstant": all(row["cache"]["nonconstant"] for row in rows_full),
        "all_full_cache_shape": all(row["cache"]["shape"] == [401, 401] for row in rows_full),
        "all_full_cache_dtype_float32": all(row["cache"]["dtype"] == "float32" for row in rows_full),
    }
    ratio = full["gospa_mean_m"] / max(grouped["gospa_mean_m"], 1e-12)
    recall_drop = grouped["recall_at_100m"] - full["recall_at_100m"]
    incompatible = ratio > float(config["input_gospa_ratio_limit"]) and recall_drop > float(config["input_recall_drop_limit"])
    invalid_cache = not all(cache_checks.values())
    per_sample = (time.perf_counter() - started) / len(rows_full)
    projected = per_sample * (
        int(config["p3_train_samples"]) + int(config["p3_select_samples"]) + int(config["p3_compare_samples"])
    )
    report = {
        "status": "FAIL_CACHE" if invalid_cache else ("G2_NO_GO_INPUT" if incompatible else ("OPTIMIZE_BEFORE_P2" if projected > float(config["projected_paired_wall_limit_seconds"]) else "PASS")),
        "grouped": grouped, "fixed_fullband": full,
        "cache_checks": cache_checks,
        "gospa_ratio_full_over_grouped": ratio,
        "recall_drop_grouped_minus_full": recall_drop,
        "projected_cache_seconds_for_p3_splits": projected,
        "actual_duration_seconds": time.perf_counter() - started,
        "samples_grouped": rows_grouped, "samples_fixed_fullband": rows_full,
        "test_executed": False,
    }
    write_json(run_root / "p1_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    for name in ("p0", "p1"):
        current = sub.add_parser(name)
        current.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        prepare(args.run_id)
    elif args.command == "p0":
        run_p0(args.run_root.resolve())
    else:
        run_p1(args.run_root.resolve())


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    main()
