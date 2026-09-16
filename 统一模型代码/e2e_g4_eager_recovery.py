"""E2E-G4完整性恢复：从登记输入重算特征，并只在内存中训练。"""

from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from 统一模型代码 import e2e_g4 as g4


def load_verified_npy(row: dict[str, Any]) -> np.ndarray:
    payload = Path(row["path"]).read_bytes()
    if len(payload) != int(row["size_bytes"]) or hashlib.sha256(payload).hexdigest() != row["sha256"]:
        raise RuntimeError(f"固定DPD身份变化: {row['path']}")
    return np.load(io.BytesIO(payload), allow_pickle=False)


def load_verified_targets(path: Path, expected: dict[str, Any]) -> g4.Targets:
    payload = path.read_bytes()
    if len(payload) != int(expected["size_bytes"]) or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
        raise RuntimeError(f"目标缓存身份变化: {path}")
    return g4.Targets(**torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False))


def build_in_memory(run_root: Path) -> tuple[dict[str, g4.FeatureStore], dict[str, g4.Targets], dict[str, Any]]:
    manifest = g4.read_json(run_root / "manifest.json")
    dpd_manifest = g4.read_json(run_root / "dpd_cache_manifest.json")
    old_features = g4.read_json(run_root / "feature_cache_manifest.json")
    snapshot = g4.read_json(g4.preflight.SNAPSHOT_MANIFEST)
    original_coarse = {
        split: snapshot["files"][index]["expected"]
        for split, index in (("train", 0), ("val_select", 1), ("val_compare", 2))
    }
    for split, expected in original_coarse.items():
        current = g4.identity(Path(expected["path"]))
        if current["size_bytes"] != expected["size_bytes"] or current["sha256"] != expected["sha256"]:
            raise RuntimeError(f"原始只读粗DPD身份变化: {split}")

    g4.configure_recorded_snapshot(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ch3, d8, _ = g4.g1.build_models(device)
    ch3.eval()
    d8.eval()
    stores: dict[str, g4.FeatureStore] = {}
    targets: dict[str, g4.Targets] = {}
    report: dict[str, Any] = {"status": "PASS", "splits": {}, "test_executed": False}
    started = time.perf_counter()
    with torch.no_grad():
        for split in ("train", "val_select", "val_compare"):
            records = manifest["subsets"][split]
            count = len(records)
            spatial_out = np.empty((count, 19, 128, 11, 11), dtype=np.float32)
            e1_out = np.empty((count, 32, 208, 208), dtype=np.float32)
            d2_out = np.empty((count, 64, 104, 104), dtype=np.float32)
            dpd_rows = dpd_manifest["files"][split]
            with h5py.File(Path(original_coarse[split]["path"]), "r") as coarse:
                for start in range(0, count, 4):
                    stop = min(start + 4, count)
                    coarse_batch = []
                    dpd_batch = []
                    for index in range(start, stop):
                        local = int(records[index]["local_index"])
                        first = np.asarray(coarse["mtr_sub_all"][:, :, :, local], dtype=np.float32)
                        second = np.asarray(coarse["mtr_sub_all"][:, :, :, local], dtype=np.float32)
                        if not np.array_equal(first, second):
                            raise RuntimeError(f"原始粗DPD双读不一致: {split}/{index}")
                        spectrum = first.transpose(2, 1, 0)
                        spectrum = np.log(spectrum + 1.0)
                        spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-6)
                        coarse_batch.append(spectrum)
                        dpd_batch.append(load_verified_npy(dpd_rows[index]))
                    coarse_tensor = torch.from_numpy(np.stack(coarse_batch)).to(device)
                    batch, bands, height, width = coarse_tensor.shape
                    spatial = ch3.backbone[:-1](coarse_tensor.reshape(batch * bands, 1, height, width))
                    spatial_out[start:stop] = spatial.reshape(batch, bands, 128, 11, 11).cpu().numpy()
                    d8_inputs = torch.stack(
                        [g4.g1.d8_input(torch.from_numpy(item))[0] for item in dpd_batch]
                    ).to(device)
                    e1, d2 = g4.d8_prefix(d8, d8_inputs)
                    e1_out[start:stop] = e1.cpu().numpy()
                    d2_out[start:stop] = d2.cpu().numpy()
                    if stop % 64 == 0 or stop == count:
                        print(f"[G4 RAM rebuild] {split} {stop}/{count}", flush=True)
            target_expected = old_features["files"][split]["targets"]
            target_file = run_root / "feature_cache" / split / "targets.pt"
            targets[split] = load_verified_targets(target_file, target_expected)
            stores[split] = g4.FeatureStore(spatial_out, e1_out, d2_out)
            report["splits"][split] = {
                "samples": count,
                "coarse_source": original_coarse[split],
                "coarse_each_sample_read_twice": True,
                "fixed_dpd_files_verified": count,
                "spatial_sha256": hashlib.sha256(memoryview(spatial_out)).hexdigest(),
                "d8_e1_sha256": hashlib.sha256(memoryview(e1_out)).hexdigest(),
                "d8_d2_sha256": hashlib.sha256(memoryview(d2_out)).hexdigest(),
            }
    report["duration_seconds"] = time.perf_counter() - started
    return stores, targets, report


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: e2e_g4_eager_recovery.py RUN_ROOT")
    run_root = Path(sys.argv[1]).resolve()
    training_root = run_root / "training_v4"
    if training_root.exists():
        raise FileExistsError("training_v4已存在，拒绝覆盖")
    stores, targets, report = build_in_memory(run_root)
    g4.write_json(run_root / "in_memory_rebuild_report.json", report)
    g4.TRAINING_ROOT_NAME = "training_v4"
    g4.load_features = lambda _root, split: stores[split]
    g4.load_targets = lambda _root, split: targets[split]
    contract = {
        "status": "PASS",
        "purpose": "invalidate disk-cache training and rerun from identical initialization using RAM-only rebuilt features",
        "training_root": g4.TRAINING_ROOT_NAME,
        "scientific_contract_changed": False,
        "old_training_reused": False,
        "test_executed": False,
        "wrapper": g4.identity(Path(__file__).resolve()),
        "in_memory_report": g4.identity(run_root / "in_memory_rebuild_report.json"),
    }
    g4.write_json(run_root / "eager_loader_contract.json", contract)
    g4.orchestrate(run_root)


if __name__ == "__main__":
    main()
