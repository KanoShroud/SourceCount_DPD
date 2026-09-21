"""G5固定全频DPD与冻结前缀特征；单进程小批处理，有界内存。"""
from __future__ import annotations

import argparse
import gc
import io
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_prepare import SOURCE, configure, g4, guard  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402
from 统一模型代码.common.verified_feature_loader import decode_npy  # noqa: E402
from s2g4_coarse_d8 import YOLOv8Loc  # noqa: E402


def build_models(run, manifest, device):
    rows = {row["name"]: row for row in manifest["inputs"]["artifacts"]}
    ch3_state = torch.load(io.BytesIO(verified_read(rows["ch3_seed42"], run / "anomalies")),
                           map_location=device, weights_only=False)
    cfg = ch3_state["config"]
    ch3 = g4.g1.SourceDetectionNet(n_sub=cfg["n_sub"], max_src=cfg["max_src"], mode=cfg["mode"]).to(device)
    ch3.load_state_dict(ch3_state["model"], strict=True)
    d8_state = torch.load(io.BytesIO(verified_read(rows["d8_seed42"], run / "anomalies")),
                          map_location=device, weights_only=False)
    if d8_state["method"] != "dualhead" or d8_state["save_tag"] != "dualhead_std":
        raise ValueError("Wrong D8 checkpoint")
    d8 = YOLOv8Loc(method="dualhead", dropout=0.4, grad_alpha=1.0).to(device)
    d8.load_state_dict(d8_state.get("model", d8_state.get("model_state")), strict=True)
    for model in (ch3, d8):
        model.eval()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite checkpoint")
    return ch3, d8


def save_array(path, value):
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    payload = buffer.getvalue()
    temporary = path.with_suffix(".pending")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    import hashlib
    row = {"path": str(temporary), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    readback = verified_read(row, path.parent / "write_anomalies")
    if not np.array_equal(decode_npy(readback), value):
        raise RuntimeError("Saved array differs")
    temporary.rename(path)
    return {**row, "path": str(path)}


def run_features(run):
    run = run.resolve(strict=True)
    manifest = g4.read_json(run / "manifest.json")
    if manifest["status"] != "INPUTS_PREPARED" or manifest["test_executed"]:
        raise RuntimeError("Inputs not prepared")
    configure(run, manifest["inputs"])
    root = run / "features"
    root.mkdir(exist_ok=False)
    config = manifest["config"]
    device = torch.device("cuda:0")
    g4.set_deterministic(config["data_seed"])
    ch3, d8 = build_models(run, manifest, device)
    old_manifest = g4.read_json(SOURCE / "manifest.json")
    old_dpd = g4.read_json(SOURCE / "dpd_cache_manifest.json")
    reusable = {}
    for split, rows in old_manifest["subsets"].items():
        for sample, row in zip(rows, old_dpd["files"][split]):
            if sample["raw_index"] != row["raw_index"] or sample["local_index"] != row["local_index"]:
                raise RuntimeError("Reusable DPD identity mapping mismatch")
            reusable[(split, sample["raw_index"], sample["local_index"])] = row
    geometry = g4.g1.receiver_geometry(device)
    weights = torch.ones(g4.g1.N_FFT, dtype=torch.float64, device=device)
    report = {"status": "RUNNING", "files": {}, "test_executed": False,
              "reused_dpd": 0, "generated_dpd": 0, "started_at": time.time(),
              "code": [g4.identity(Path(__file__))]}
    try:
        with torch.no_grad():
            for split, records in manifest["subsets"].items():
                destination = root / split
                destination.mkdir()
                dpd_dir = destination / "dpd"
                dpd_dir.mkdir()
                files = {name: [] for name in ("ch3_spatial", "d8_e1", "d8_d2", "dpd")}
                labels = {key: [] for key in ("band", "ignore", "positions", "counts", "overlap")}
                metadata = []
                with g4.g1.SampleStore(split) as store:
                    for start in range(0, len(records), 4):
                        guard(run)
                        if time.time() - manifest["created_at"] > config["wall_limit_seconds"]:
                            raise RuntimeError("G5 wall budget exceeded")
                        coarse, fine = [], []
                        for ordinal in range(start, min(start + 4, len(records))):
                            record = records[ordinal]
                            sample = store.sample(record)
                            if not torch.isfinite(sample["coarse_dpd"]).all():
                                raise RuntimeError("Nonfinite coarse input")
                            key = (split, record["raw_index"], record["local_index"])
                            if key in reusable:
                                row = reusable[key]
                                value = decode_npy(verified_read(row, run / "anomalies"))
                                report["reused_dpd"] += 1
                            else:
                                value = g4.preflight.dpd_map(sample["signal"], weights, geometry, config).cpu().numpy().astype(np.float32)
                                report["generated_dpd"] += 1
                            if value.shape != (401, 401) or not np.isfinite(value).all() or value.std() <= 0:
                                raise RuntimeError("Invalid fixed DPD")
                            dpd_row = save_array(dpd_dir / f"{ordinal:05d}.npy", value)
                            files["dpd"].append({"ordinal": ordinal, **record, **dpd_row})
                            fine.append(g4.g1.d8_input(torch.from_numpy(value.copy()))[0])
                            coarse.append(sample["coarse_dpd"])
                            labels["band"].append(sample["band_truth"][:3])
                            labels["ignore"].append(sample["ignore_truth"][:3])
                            positions = np.zeros((3, 2), dtype=np.float32)
                            positions[:record["true_k"]] = sample["positions_m"][:record["true_k"]]
                            labels["positions"].append(torch.from_numpy(positions))
                            labels["counts"].append(record["true_k"])
                            labels["overlap"].append(record["frequency_overlap"])
                            snr = float(store.coarse["avg_snr_all"][0, record["local_index"]])
                            pos = positions[:record["true_k"]]
                            distances = [float(np.linalg.norm(a-b)) for i,a in enumerate(pos) for b in pos[i+1:]]
                            metadata.append({**record, "snr_db": snr, "min_source_distance_m": min(distances) if distances else None})
                        inputs = torch.stack(coarse).to(device)
                        batch, bands, height, width = inputs.shape
                        spatial = ch3.backbone[:-1](inputs.reshape(batch*bands, 1, height, width)).reshape(batch, bands, 128, 11, 11)
                        e1, d2 = g4.d8_prefix(d8, torch.stack(fine).to(device))
                        for name, tensor in (("ch3_spatial", spatial), ("d8_e1", e1), ("d8_d2", d2)):
                            value = tensor.cpu().numpy()
                            if not np.isfinite(value).all():
                                raise RuntimeError("Nonfinite cached feature")
                            row = save_array(destination / f"{name}_{start:05d}.npy", value)
                            files[name].append({"start": start, "stop": start+batch, **row})
                        if (start+batch) % 32 == 0:
                            g4.write_json(run / "feature_progress.json", {"split": split, "completed": start+batch, "total": len(records),
                                                                        "elapsed": time.time()-report["started_at"]})
                            g4.write_json(destination / "partial_manifest.json", files)
                            print(f"G5 features {split} {start+batch}/{len(records)}; newDPD={report['generated_dpd']}", flush=True)
                payload = {key: torch.stack(values) if key in ("band", "ignore", "positions") else torch.tensor(values)
                           for key, values in labels.items()}
                target = destination / "targets.pt"
                torch.save(payload, target)
                row = g4.identity(target)
                torch.load(io.BytesIO(verified_read(row, run / "anomalies")), map_location="cpu", weights_only=False)
                files["targets"] = row
                files["metadata"] = metadata
                report["files"][split] = files
                g4.write_json(run / "feature_manifest.json", report)
                gc.collect()
        report.update(status="PASS", seconds=time.time()-report["started_at"])
    except Exception as exc:
        report.update(status="STOP", error=repr(exc), seconds=time.time()-report["started_at"])
        raise
    finally:
        g4.write_json(run / "feature_manifest.json", report)
    print("G5 feature preparation PASS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    run_features(parser.parse_args().run)
