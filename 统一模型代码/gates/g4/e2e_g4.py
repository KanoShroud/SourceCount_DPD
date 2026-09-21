"""E2E-G4：扩大数据与分组解冻选择门。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = PACKAGE_ROOT.parent
CH4_DIR = PROJECT_ROOT / "第四章代码"
os.environ.setdefault(
    "SOURCECOUNT_REFERENCE_OUTPUT_ROOT",
    str((PROJECT_ROOT.parent / "SourceCount_DPD" / "outputs").resolve()),
)
for root in (PROJECT_ROOT, CH4_DIR):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import chapter_runtime  # noqa: E402,F401

from 统一模型代码.gates.g1 import e2e_g1 as g1  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_latent_fusion as g2  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_r1 as r1  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_r2 as r2  # noqa: E402
from 统一模型代码.gates.g2 import e2e_g2_preflight as preflight  # noqa: E402
from 统一模型代码.models.e2e_latent_fusion import (  # noqa: E402
    CH3Features,
    FrequencySpatialSplitter,
    SourceLocalizationHead,
    SourceQueryBuilder,
)
from 统一模型代码.common.runtime_paths import new_run_dir, validate_output_path  # noqa: E402


CONFIG_PATH = PACKAGE_ROOT / "configs" / "e2e_g4.json"
SCRIPT_PATH = Path(__file__).resolve()
MODEL_PATH = PACKAGE_ROOT / "models" / "e2e_latent_fusion.py"
SOURCE_R2_RUN = PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_g2_r2" / "20260904_163951"
SOURCE_G3_RUN = PROJECT_ROOT / "outputs_e2e" / "unified" / "e2e_g3" / "20260905_135720"
TRACKS = ("a0_large_frozen", "a1_d8_tail", "a2_joint_tail")
TRAINING_ROOT_NAME = "training_v2"
FEATURE_SHARD_SIZE = 16
G3_INITIAL = {
    20260907: SOURCE_G3_RUN / "seeds" / "20260905" / "fs_e2e" / "best.pt",
    20260908: SOURCE_G3_RUN / "seeds" / "20260906" / "fs_e2e" / "best.pt",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def code_identity() -> list[dict[str, Any]]:
    return [
        identity(path)
        for path in (
            SCRIPT_PATH,
            CONFIG_PATH,
            MODEL_PATH,
            Path(g1.__file__).resolve(),
            Path(g2.__file__).resolve(),
            Path(r1.__file__).resolve(),
            Path(r2.__file__).resolve(),
            Path(preflight.__file__).resolve(),
        )
    ]


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def all_records(split: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with g1.SampleStore(split) as store:
        size = int(store.coarse["src_count_all"].shape[1])
        for local_index in range(size):
            count = int(np.asarray(store.coarse["src_count_all"][:, local_index]).item())
            raw_index = int(np.asarray(store.coarse["sample_idx_all"][:, local_index]).item())
            records.append(
                {
                    "split": split,
                    "local_index": local_index,
                    "raw_index": raw_index,
                    "true_k": count,
                }
            )
    return records


def choose_balanced(
    pool: list[dict[str, Any]], per_k: int, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for count in range(4):
        candidates = [row for row in pool if int(row["true_k"]) == count]
        if len(candidates) < per_k:
            raise RuntimeError(f"K={count}候选不足: {len(candidates)} < {per_k}")
        order = rng.permutation(len(candidates))[:per_k]
        selected.extend(candidates[int(index)] for index in order)
    selected.sort(key=lambda row: (int(row["true_k"]), int(row["local_index"])))
    return selected


def annotate(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return r2.annotate_records(records, split)


def prepare(run_id: str) -> Path:
    files, artifacts = preflight.configure_snapshot()
    config = read_json(CONFIG_PATH)
    source_manifest = read_json(SOURCE_R2_RUN / "manifest.json")
    original_train = source_manifest["subsets"]["train"]

    train_pool = all_records("train")
    original_keys = {
        (int(row["raw_index"]), int(row["local_index"])) for row in original_train
    }
    additions_pool = [
        row
        for row in train_pool
        if (int(row["raw_index"]), int(row["local_index"])) not in original_keys
    ]
    additions = choose_balanced(additions_pool, 192, int(config["screen_seed"]))
    train = annotate([*original_train, *additions], "train")

    old_select = {
        int(row["local_index"]) for row in source_manifest["subsets"]["val_select"]
    }
    select_pool = [row for row in all_records("val_select") if int(row["local_index"]) not in old_select]
    val_select = annotate(
        choose_balanced(select_pool, 64, int(config["screen_seed"]) + 11),
        "val_select",
    )

    old_compare = {
        int(row["local_index"]) for row in source_manifest["subsets"]["val_compare"]
    }
    compare_pool = [row for row in all_records("val_compare") if int(row["local_index"]) not in old_compare]
    val_compare = annotate(
        choose_balanced(compare_pool, 128, int(config["screen_seed"]) + 23),
        "val_compare",
    )

    subsets = {"train": train, "val_select": val_select, "val_compare": val_compare}
    for split, expected in (("train", 1024), ("val_select", 256), ("val_compare", 512)):
        if len(subsets[split]) != expected:
            raise RuntimeError(f"{split}规模错误")
        histogram = {
            count: sum(int(row["true_k"]) == count for row in subsets[split])
            for count in range(4)
        }
        if any(value != expected // 4 for value in histogram.values()):
            raise RuntimeError(f"{split} K不平衡: {histogram}")
    if not original_keys.issubset(
        {(int(row["raw_index"]), int(row["local_index"])) for row in train}
    ):
        raise RuntimeError("G4 train未完整包含G2/G3原256条")

    run_root = new_run_dir("e2e_g4", run_id, create=True)
    manifest = {
        "status": "PREPARED",
        "gate": "E2E-G4",
        "run_id": run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference_read_only": True,
        "test_executed": False,
        "config": config,
        "code": code_identity(),
        "source_r2_manifest": identity(SOURCE_R2_RUN / "manifest.json"),
        "source_g3_final": identity(SOURCE_G3_RUN / "final_report.json"),
        "initial_checkpoints": {
            str(seed): identity(path) for seed, path in G3_INITIAL.items()
        },
        "inputs": {"files": files, "artifacts": artifacts},
        "subsets": subsets,
    }
    write_json(run_root / "manifest.json", manifest)
    print(run_root, flush=True)
    return run_root


def verify_run(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(run_root / "manifest.json")
    if manifest["gate"] != "E2E-G4" or manifest["test_executed"] is not False:
        raise RuntimeError("G4 manifest或test合同错误")
    current_code = code_identity()
    if manifest["code"] != current_code:
        receipt_path = run_root / "repair_receipt.json"
        if not receipt_path.exists():
            raise RuntimeError("prepare后G4代码或配置变化")
        receipt = read_json(receipt_path)
        first_repair_ok = (
            receipt.get("status") != "PASS"
            or receipt.get("original_code") != manifest["code"]
            or receipt.get("optimizer_steps_before_repair") != 0
        )
        integrity_receipt_path = run_root / "integrity_repair_receipt.json"
        integrity_repair_ok = False
        if integrity_receipt_path.exists():
            integrity_receipt = read_json(integrity_receipt_path)
            integrity_repair_ok = (
                integrity_receipt.get("status") == "PASS"
                and integrity_receipt.get("prior_code") == receipt.get("repaired_code")
                and integrity_receipt.get("repaired_code") == current_code
                and integrity_receipt.get("invalidated_optimizer_steps") == 4096
                and integrity_receipt.get("invalidated_training_reused") is False
            )
            finalize_receipt_path = run_root / "finalize_repair_receipt.json"
            if not integrity_repair_ok and finalize_receipt_path.exists():
                finalize_receipt = read_json(finalize_receipt_path)
                integrity_repair_ok = (
                    finalize_receipt.get("status") == "PASS"
                    and finalize_receipt.get("prior_code") == integrity_receipt.get("repaired_code")
                    and finalize_receipt.get("repaired_code") == current_code
                    and finalize_receipt.get("report_only_change") is True
                )
        elif receipt.get("repaired_code") == current_code:
            integrity_repair_ok = True
        if first_repair_ok or not integrity_repair_ok:
            raise RuntimeError("G4修复收据不匹配")
    configure_recorded_snapshot(manifest)
    if manifest["source_r2_manifest"] != identity(SOURCE_R2_RUN / "manifest.json"):
        raise RuntimeError("R2源manifest身份变化")
    if manifest["source_g3_final"] != identity(SOURCE_G3_RUN / "final_report.json"):
        raise RuntimeError("G3源final身份变化")
    for seed, path in G3_INITIAL.items():
        if manifest["initial_checkpoints"][str(seed)] != identity(path):
            raise RuntimeError(f"G3初始化checkpoint身份变化: {seed}")
    return manifest, manifest["config"]


def configure_recorded_snapshot(manifest: dict[str, Any]) -> None:
    """仅绑定已登记路径；G4-P0之后训练只读本门缓存，不重复读取大型HDF5。"""
    files = manifest["inputs"]["files"]
    artifacts = manifest["inputs"]["artifacts"]
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
        rows = []
        for name in names:
            current = identity(Path(indexed[name]["path"]))
            if current != {key: indexed[name][key] for key in ("path", "size_bytes", "sha256")}:
                raise RuntimeError(f"参考权重身份变化: {name}")
            rows.append({"name": name, **current})
        return rows

    g1.verify_artifacts = verify


def cache_path(run_root: Path, split: str, ordinal: int) -> Path:
    return run_root / "fixed_fullband_cache" / split / f"full_{ordinal:04d}.npy"


def prepare_dpd_cache(
    run_root: Path, manifest: dict[str, Any], config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    destination = validate_output_path(run_root / "fixed_fullband_cache")
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖DPD缓存: {destination}")
    destination.mkdir(parents=True)
    source_manifest = read_json(SOURCE_R2_RUN / "manifest.json")
    source_map = {
        (int(row["raw_index"]), int(row["local_index"])): r2.cache_path(
            SOURCE_R2_RUN, "train", ordinal
        )
        for ordinal, row in enumerate(source_manifest["subsets"]["train"], start=1)
    }
    geometry = g1.receiver_geometry(device)
    weights = torch.ones(g1.N_FFT, dtype=torch.float64, device=device)
    result: dict[str, list[dict[str, Any]]] = {}
    started = time.perf_counter()
    for split in ("train", "val_select", "val_compare"):
        split_root = destination / split
        split_root.mkdir()
        rows: list[dict[str, Any]] = []
        with g1.SampleStore(split) as store:
            for ordinal, record in enumerate(manifest["subsets"][split], start=1):
                path = cache_path(run_root, split, ordinal)
                key = (int(record["raw_index"]), int(record["local_index"]))
                reused = split == "train" and key in source_map
                sample_started = time.perf_counter()
                if reused:
                    shutil.copyfile(source_map[key], path)
                    array = np.load(path, allow_pickle=False)
                else:
                    sample = store.sample(record)
                    dpd = preflight.dpd_map(sample["signal"], weights, geometry, config)
                    array = dpd.detach().cpu().numpy().astype(np.float32)
                    temporary = path.with_suffix(".tmp.npy")
                    np.save(temporary, array)
                    os.replace(temporary, path)
                description = preflight.cache_description(path, array, 0.0)
                if description["shape"] != [401, 401] or not description["finite"] or not description["nonconstant"]:
                    raise RuntimeError(f"DPD缓存异常: {path}")
                description.update(
                    {
                        "ordinal": ordinal,
                        "raw_index": int(record["raw_index"]),
                        "local_index": int(record["local_index"]),
                        "reused": reused,
                        "seconds": time.perf_counter() - sample_started,
                    }
                )
                rows.append(description)
                if ordinal % 16 == 0 or ordinal == len(manifest["subsets"][split]):
                    write_json(
                        run_root / "dpd_cache_progress.json",
                        {"split": split, "completed": ordinal, "total": len(manifest["subsets"][split])},
                    )
                    print(f"[G4 DPD] {split} {ordinal}/{len(manifest['subsets'][split])}", flush=True)
        result[split] = rows
    report = {
        "status": "PASS",
        "duration_seconds": time.perf_counter() - started,
        "files": result,
        "reused_train": sum(bool(row["reused"]) for row in result["train"]),
    }
    write_json(run_root / "dpd_cache_manifest.json", report)
    return report


def target_path(run_root: Path, split: str) -> Path:
    repaired = run_root / "feature_cache_v2" / split / "targets.pt"
    return repaired if repaired.exists() else run_root / "feature_cache" / split / "targets.pt"


def feature_path(run_root: Path, split: str, name: str) -> Path:
    return run_root / "feature_cache" / split / f"{name}.npy"


def d8_prefix(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded = model._pad(inputs)
    e1, e2, p3, p4, p5 = model.backbone(padded)
    n3, n4, n5 = model.pan(p3, p4, p5)
    decoder = model.decoder
    d4 = decoder.drop(decoder.c4(torch.cat([decoder.up4(n5), n4], 1)))
    d3 = decoder.drop(decoder.c3(torch.cat([decoder.up3(d4), n3], 1)))
    d2 = decoder.drop(decoder.c2(torch.cat([decoder.up2(d3), e2], 1)))
    return e1, d2


def prepare_feature_cache(
    run_root: Path, manifest: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    root = validate_output_path(run_root / "feature_cache")
    if root.exists():
        raise FileExistsError(f"拒绝覆盖特征缓存: {root}")
    root.mkdir(parents=True)
    ch3, d8, _ = g1.build_models(device)
    ch3.eval()
    d8.eval()
    started = time.perf_counter()
    descriptions: dict[str, Any] = {}
    with torch.no_grad():
        for split in ("train", "val_select", "val_compare"):
            data = r2.load_split(run_root, manifest, split)
            count = len(data.counts)
            split_root = root / split
            split_root.mkdir()
            spatial_out = np.lib.format.open_memmap(
                feature_path(run_root, split, "ch3_spatial"),
                mode="w+",
                dtype=np.float32,
                shape=(count, 19, 128, 11, 11),
            )
            e1_out = np.lib.format.open_memmap(
                feature_path(run_root, split, "d8_e1"),
                mode="w+",
                dtype=np.float32,
                shape=(count, 32, 208, 208),
            )
            d2_out = np.lib.format.open_memmap(
                feature_path(run_root, split, "d8_d2"),
                mode="w+",
                dtype=np.float32,
                shape=(count, 64, 104, 104),
            )
            for start in range(0, count, 4):
                stop = min(start + 4, count)
                coarse = data.coarse[start:stop].to(device)
                batch, bands, height, width = coarse.shape
                spatial = ch3.backbone[:-1](coarse.reshape(batch * bands, 1, height, width))
                spatial = spatial.reshape(batch, bands, 128, 11, 11)
                d8_inputs = torch.stack(
                    [g1.d8_input(item)[0] for item in data.dpd[start:stop]]
                ).to(device)
                e1, d2 = d8_prefix(d8, d8_inputs)
                spatial_out[start:stop] = spatial.cpu().numpy()
                e1_out[start:stop] = e1.cpu().numpy()
                d2_out[start:stop] = d2.cpu().numpy()
                if stop % 16 == 0 or stop == count:
                    print(f"[G4 feature] {split} {stop}/{count}", flush=True)
            spatial_out.flush()
            e1_out.flush()
            d2_out.flush()
            torch.save(
                {
                    "band": data.band,
                    "ignore": data.ignore,
                    "positions": data.positions,
                    "counts": data.counts,
                    "overlap": data.overlap,
                },
                target_path(run_root, split),
            )
            descriptions[split] = {
                name: identity(feature_path(run_root, split, name))
                for name in ("ch3_spatial", "d8_e1", "d8_d2")
            }
            descriptions[split]["targets"] = identity(target_path(run_root, split))
            del data, spatial_out, e1_out, d2_out
            gc.collect()
    report = {
        "status": "PASS",
        "duration_seconds": time.perf_counter() - started,
        "files": descriptions,
    }
    write_json(run_root / "feature_cache_manifest.json", report)
    return report


def repeated_identity(path: Path, repeats: int = 3) -> list[dict[str, Any]]:
    return [identity(path) for _ in range(repeats)]


def repair_feature_cache(run_root: Path) -> dict[str, Any]:
    """从稳定逐样本DPD重建D8特征，并把全部大特征文件改为小分片。"""
    manifest = read_json(run_root / "manifest.json")
    first_receipt = read_json(run_root / "repair_receipt.json")
    if first_receipt.get("status") != "PASS":
        raise RuntimeError("首次工程修复收据无效")
    if (run_root / "feature_cache_v2").exists():
        raise FileExistsError("feature_cache_v2已存在，拒绝覆盖")
    integrity = read_json(run_root / "input_integrity_train_coarse.json")
    if integrity.get("classification") != "LOGICAL_READ_FAILURE":
        raise RuntimeError("旧快照异常尚未形成明确分类证据")
    old_manifest = read_json(run_root / "feature_cache_manifest.json")
    dpd_manifest = read_json(run_root / "dpd_cache_manifest.json")
    invalid_report = read_json(
        run_root / TRAINING_ROOT_NAME.replace("_v2", "") / "20260907" /
        "a0_large_frozen" / "training_report_epoch16.json"
    )
    if int(invalid_report.get("optimizer_steps", -1)) != 4096:
        raise RuntimeError("待作废A0训练步数与事实不符")

    checked_dpd = 0
    for rows in dpd_manifest["files"].values():
        for row in rows:
            current = identity(Path(row["path"]))
            if current["size_bytes"] != row["size_bytes"] or current["sha256"] != row["sha256"]:
                raise RuntimeError(f"固定DPD身份变化: {row['path']}")
            checked_dpd += 1

    spatial_reads: dict[str, Any] = {}
    for split in ("train", "val_select", "val_compare"):
        expected = old_manifest["files"][split]["ch3_spatial"]
        reads = repeated_identity(Path(expected["path"]))
        spatial_reads[split] = reads
        if any(
            row["size_bytes"] != expected["size_bytes"] or row["sha256"] != expected["sha256"]
            for row in reads
        ):
            raise RuntimeError(f"CH3空间特征重复读取不稳定: {split}")
        target_expected = old_manifest["files"][split]["targets"]
        target_current = identity(Path(target_expected["path"]))
        if target_current["size_bytes"] != target_expected["size_bytes"] or target_current["sha256"] != target_expected["sha256"]:
            raise RuntimeError(f"目标缓存身份变化: {split}")

    configure_recorded_snapshot(manifest)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, d8, _ = g1.build_models(device)
    d8.eval()
    root = validate_output_path(run_root / "feature_cache_v2")
    root.mkdir(parents=True)
    descriptions: dict[str, Any] = {}
    started = time.perf_counter()
    with torch.no_grad():
        for split in ("train", "val_select", "val_compare"):
            split_root = root / split
            split_root.mkdir()
            old_spatial = np.load(
                old_manifest["files"][split]["ch3_spatial"]["path"],
                mmap_mode="r",
                allow_pickle=False,
            )
            count = len(manifest["subsets"][split])
            rows = {name: [] for name in ("ch3_spatial", "d8_e1", "d8_d2")}
            for start in range(0, count, FEATURE_SHARD_SIZE):
                stop = min(start + FEATURE_SHARD_SIZE, count)
                dpd = [
                    np.load(cache_path(run_root, split, index + 1), allow_pickle=False)
                    for index in range(start, stop)
                ]
                d8_inputs = torch.stack(
                    [g1.d8_input(torch.from_numpy(item))[0] for item in dpd]
                ).to(device)
                e1, d2 = d8_prefix(d8, d8_inputs)
                values = {
                    "ch3_spatial": np.asarray(old_spatial[start:stop]).copy(),
                    "d8_e1": e1.cpu().numpy(),
                    "d8_d2": d2.cpu().numpy(),
                }
                for name, value in values.items():
                    path = split_root / f"{name}_{start:04d}_{stop:04d}.npy"
                    np.save(path, value.astype(np.float32, copy=False))
                    rows[name].append({"start": start, "stop": stop, **identity(path)})
                if stop % 64 == 0 or stop == count:
                    print(f"[G4 feature repair] {split} {stop}/{count}", flush=True)
            old_target = Path(old_manifest["files"][split]["targets"]["path"])
            new_target = split_root / "targets.pt"
            shutil.copyfile(old_target, new_target)
            descriptions[split] = {**rows, "targets": identity(new_target)}
            del old_spatial
            gc.collect()

    repaired = {
        "status": "PASS",
        "schema": "e2e-g4-sharded-feature-cache-v2",
        "shard_size": FEATURE_SHARD_SIZE,
        "checked_fixed_dpd_files": checked_dpd,
        "ch3_spatial_repeated_reads": spatial_reads,
        "d8_features_recomputed_from_fixed_dpd": True,
        "old_train_d8_e1_reused": False,
        "duration_seconds": time.perf_counter() - started,
        "files": descriptions,
    }
    write_json(run_root / "feature_cache_v2_manifest.json", repaired)
    receipt = {
        "status": "PASS",
        "cause": "large monolithic feature cache produced non-repeatable byte reads",
        "repair": "recompute D8 features from registered per-sample DPD and store all features in float32 shards",
        "scientific_contract_changed": False,
        "prior_code": first_receipt["repaired_code"],
        "repaired_code": code_identity(),
        "invalidated_optimizer_steps": 4096,
        "invalidated_training_reused": False,
        "invalidated_training_path": str(
            run_root / "training" / "20260907" / "a0_large_frozen"
        ),
        "new_training_root": TRAINING_ROOT_NAME,
    }
    write_json(run_root / "integrity_repair_receipt.json", receipt)

    features = load_features(run_root, "train")
    probe = torch.tensor([0, 511, 1023])
    repaired["loader_smoke"] = {
        "spatial_shape": list(numpy_batch(features.spatial, probe, torch.device("cpu")).shape),
        "e1_shape": list(numpy_batch(features.e1, probe, torch.device("cpu")).shape),
        "d2_shape": list(numpy_batch(features.d2, probe, torch.device("cpu")).shape),
    }
    write_json(run_root / "feature_cache_v2_manifest.json", repaired)
    return repaired


@dataclass
class Targets:
    band: torch.Tensor
    ignore: torch.Tensor
    positions: torch.Tensor
    counts: torch.Tensor
    overlap: torch.Tensor


@dataclass
class FeatureStore:
    spatial: Any
    e1: Any
    d2: Any


class ShardedArray:
    def __init__(self, rows: list[dict[str, Any]]):
        self.arrays = [np.load(row["path"], mmap_mode="r", allow_pickle=False) for row in rows]
        self.starts = [int(row["start"]) for row in rows]
        self.stops = [int(row["stop"]) for row in rows]
        self.length = self.stops[-1] if self.stops else 0

    def take(self, indices: list[int]) -> np.ndarray:
        selected = []
        for index in indices:
            for array, start, stop in zip(self.arrays, self.starts, self.stops):
                if start <= index < stop:
                    selected.append(np.asarray(array[index - start]))
                    break
            else:
                raise IndexError(index)
        return np.stack(selected)


@dataclass
class Context:
    ch3: nn.Module
    d8: nn.Module
    query_builder: SourceQueryBuilder
    splitter: FrequencySpatialSplitter
    source_head: SourceLocalizationHead
    track: str
    parameter_groups: list[dict[str, Any]]
    parameters: list[nn.Parameter]


def load_targets(run_root: Path, split: str) -> Targets:
    payload = torch.load(target_path(run_root, split), map_location="cpu", weights_only=False)
    return Targets(**payload)


def load_features(run_root: Path, split: str) -> FeatureStore:
    repaired_manifest = run_root / "feature_cache_v2_manifest.json"
    if repaired_manifest.exists():
        files = read_json(repaired_manifest)["files"][split]
        return FeatureStore(
            ShardedArray(files["ch3_spatial"]),
            ShardedArray(files["d8_e1"]),
            ShardedArray(files["d8_d2"]),
        )
    return FeatureStore(
        np.load(feature_path(run_root, split, "ch3_spatial"), mmap_mode="r"),
        np.load(feature_path(run_root, split, "d8_e1"), mmap_mode="r"),
        np.load(feature_path(run_root, split, "d8_d2"), mmap_mode="r"),
    )


def warm_state(checkpoint: Path) -> dict[str, Any]:
    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def build_context(
    track: str, seed: int, device: torch.device, config: dict[str, Any]
) -> Context:
    if track not in TRACKS:
        raise ValueError(track)
    set_deterministic(seed)
    ch3, d8, _ = g1.build_models(device)
    for parameter in itertools.chain(ch3.parameters(), d8.parameters()):
        parameter.requires_grad_(False)
    query_builder = SourceQueryBuilder().to(device)
    splitter = FrequencySpatialSplitter().to(device)
    source_head = SourceLocalizationHead().to(device)
    state = warm_state(G3_INITIAL[seed])
    ch3.band_heads[:3].load_state_dict(state["ch3_heads"], strict=True)
    query_builder.load_state_dict(state["query_builder"], strict=True)
    splitter.load_state_dict(state["splitter"], strict=True)
    source_head.load_state_dict(state["source_head"], strict=True)

    endpoint = list(
        itertools.chain(
            ch3.band_heads[:3].parameters(),
            query_builder.parameters(),
            splitter.parameters(),
            source_head.parameters(),
        )
    )
    for parameter in endpoint:
        parameter.requires_grad_(True)
    groups: list[dict[str, Any]] = [
        {"params": endpoint, "lr": float(config["endpoint_learning_rate"]), "name": "endpoint"}
    ]
    if track in ("a1_d8_tail", "a2_joint_tail"):
        d8_tail = list(itertools.chain(d8.decoder.c1.parameters(), d8.decoder.up0.parameters()))
        for parameter in d8_tail:
            parameter.requires_grad_(True)
        groups.append({"params": d8_tail, "lr": float(config["d8_tail_learning_rate"]), "name": "d8_tail"})
    if track == "a2_joint_tail":
        ch3_tail = list(ch3.cross_attn.parameters())
        for parameter in ch3_tail:
            parameter.requires_grad_(True)
        groups.append({"params": ch3_tail, "lr": float(config["ch3_tail_learning_rate"]), "name": "ch3_tail"})
    parameters = [parameter for group in groups for parameter in group["params"]]
    context = Context(ch3, d8, query_builder, splitter, source_head, track, groups, parameters)
    set_mode(context, training=True)
    return context


def set_mode(context: Context, *, training: bool) -> None:
    context.ch3.eval()
    context.d8.eval()
    context.query_builder.train(training)
    context.splitter.train(training)
    context.source_head.train(training)
    context.ch3.band_heads[:3].train(training)
    if context.track in ("a1_d8_tail", "a2_joint_tail"):
        context.d8.decoder.c1.train(training)
        context.d8.decoder.up0.train(training)
    if context.track == "a2_joint_tail":
        context.ch3.cross_attn.train(training)


def state_payload(context: Context) -> dict[str, Any]:
    return {
        "ch3_heads": context.ch3.band_heads[:3].state_dict(),
        "query_builder": context.query_builder.state_dict(),
        "splitter": context.splitter.state_dict(),
        "source_head": context.source_head.state_dict(),
        "d8_c1": context.d8.decoder.c1.state_dict(),
        "d8_up0": context.d8.decoder.up0.state_dict(),
        "ch3_cross_attn": context.ch3.cross_attn.state_dict(),
    }


def load_state(context: Context, state: dict[str, Any]) -> None:
    context.ch3.band_heads[:3].load_state_dict(state["ch3_heads"], strict=True)
    context.query_builder.load_state_dict(state["query_builder"], strict=True)
    context.splitter.load_state_dict(state["splitter"], strict=True)
    context.source_head.load_state_dict(state["source_head"], strict=True)
    context.d8.decoder.c1.load_state_dict(state["d8_c1"], strict=True)
    context.d8.decoder.up0.load_state_dict(state["d8_up0"], strict=True)
    context.ch3.cross_attn.load_state_dict(state["ch3_cross_attn"], strict=True)


def state_digest(context: Context) -> str:
    digest = hashlib.sha256()
    for group, state in state_payload(context).items():
        digest.update(group.encode("utf-8"))
        for name, tensor in sorted(state.items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def numpy_batch(array: Any, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = array.take(indices.tolist()) if isinstance(array, ShardedArray) else np.asarray(array[indices.tolist()])
    selected = np.asarray(selected).copy()
    return torch.from_numpy(selected).to(device)


def forward_indices(
    context: Context,
    features: FeatureStore,
    indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spatial = numpy_batch(features.spatial, indices, device)
    pooled = spatial.mean(dim=(-1, -2))
    if context.track == "a2_joint_tail":
        tokens = context.ch3.cross_attn(pooled + context.ch3.pos_embed)
        global_feature = context.ch3.global_encoder(tokens.mean(dim=1))
    else:
        with torch.no_grad():
            tokens = context.ch3.cross_attn(pooled + context.ch3.pos_embed)
            global_feature = context.ch3.global_encoder(tokens.mean(dim=1))
    cached = CH3Features(spatial, tokens, global_feature, torch.empty(0, device=device), torch.empty(0, device=device))
    current = g2.ch3_from_cached(context.ch3, cached)
    query, logits = context.query_builder(current)
    source_spatial, attention = context.splitter(current.spatial, query, logits)

    e1 = numpy_batch(features.e1, indices, device)
    d2 = numpy_batch(features.d2, indices, device)
    if context.track in ("a1_d8_tail", "a2_joint_tail"):
        with torch.no_grad():
            up1 = context.d8.decoder.up1(d2)
        d1 = context.d8.decoder.c1(torch.cat([up1, e1], 1))
        d0 = context.d8.decoder.up0(d1)
    else:
        with torch.no_grad():
            d1 = context.d8.decoder.c1(torch.cat([context.d8.decoder.up1(d2), e1], 1))
            d0 = context.d8.decoder.up0(d1)
    heatmap, offset = context.source_head(d0[:, :, :401, :401], source_spatial, query)
    return query, logits, attention, heatmap, offset


def as_cached_targets(targets: Targets) -> g2.CachedBatch:
    empty = torch.empty(0)
    return g2.CachedBatch(empty, empty, targets.band, targets.ignore, targets.positions, targets.counts, targets.overlap)


def distance_errors(truth: np.ndarray, predicted: np.ndarray) -> list[float]:
    if len(truth) == 0 or len(predicted) == 0:
        return []
    matrix = np.linalg.norm(truth[:, None, :] - predicted[None, :, :], axis=2)
    left, right = linear_sum_assignment(matrix)
    return [float(matrix[i, j]) for i, j in zip(left, right)]


def numeric_errors(errors: list[float], true_sources: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "true_source_count": true_sources,
        "matched_pair_count": len(errors),
        "matched_pair_coverage_of_true": len(errors) / max(true_sources, 1),
    }
    if errors:
        values = np.asarray(errors, dtype=np.float64)
        result["matched_errors_m"] = {
            "rmse": float(np.sqrt(np.mean(values**2))),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    else:
        result["matched_errors_m"] = None
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_sources = sum(int(row["true_count"]) for row in rows)
    predicted_sources = sum(int(row["predicted_count"]) for row in rows)
    errors = [value for row in rows for value in row["matched_errors_m"]]
    result = numeric_errors(errors, true_sources)
    result.update(
        {
            "sample_count": len(rows),
            "predicted_source_count": predicted_sources,
            "gospa_mean_m": float(np.mean([row["gospa_m"] for row in rows])),
            "gospa_components_mean_p_sum": {
                name: float(np.mean([row[f"gospa_{name}_p_sum"] for row in rows]))
                for name in ("localization", "missed", "false")
            },
            "exact_count_rate": float(np.mean([row["true_count"] == row["predicted_count"] for row in rows])),
            "false_positive_rate_k0": float(
                np.mean([row["predicted_count"] > 0 for row in rows if row["true_count"] == 0])
            ) if any(row["true_count"] == 0 for row in rows) else None,
        }
    )
    for threshold in (10, 30, 50, 100):
        tp = sum(int(row[f"tp_at_{threshold}m"]) for row in rows)
        result[f"recall_at_{threshold}m"] = tp / max(true_sources, 1)
        result[f"precision_at_{threshold}m"] = tp / max(predicted_sources, 1)
    return result


@torch.no_grad()
def evaluate(
    context: Context,
    features: FeatureStore,
    targets: Targets,
    device: torch.device,
) -> dict[str, Any]:
    set_mode(context, training=False)
    outputs: list[tuple[torch.Tensor, ...]] = []
    for start in range(0, len(targets.counts), 4):
        indices = torch.arange(start, min(start + 4, len(targets.counts)))
        outputs.append(tuple(value.cpu() for value in forward_indices(context, features, indices, device)))
    logits = torch.cat([row[1] for row in outputs])
    heatmap = torch.cat([row[3] for row in outputs])
    offset = torch.cat([row[4] for row in outputs])
    target_batch = as_cached_targets(targets)
    mappings = g2.assignments(logits, heatmap, targets.band, targets.ignore, targets.positions, targets.counts)
    predicted_counts = (logits.amax(dim=-1) >= 0.0).sum(dim=-1)
    rows: list[dict[str, Any]] = []
    band_f1: list[float] = []
    band_iou: list[float] = []
    for index, count_value in enumerate(targets.counts.tolist()):
        count = int(count_value)
        active = torch.nonzero(logits[index].amax(dim=-1) >= 0.0, as_tuple=False).flatten().tolist()
        predictions: list[list[float]] = []
        for query in active:
            peak = int(torch.argmax(torch.sigmoid(heatmap[index, query])).item())
            iy, ix = divmod(peak, 401)
            delta = offset[index, query, :, iy, ix].clamp(-1.0, 1.0)
            predictions.append(
                [(ix + float(delta[0])) * 10.0 - 2000.0, (iy + float(delta[1])) * 10.0 - 2000.0]
            )
        predicted = np.asarray(predictions, dtype=np.float32).reshape(-1, 2)
        truth = targets.positions[index, :count].numpy()
        gospa = g1.gospa_sample(truth, predicted)
        errors = distance_errors(truth, predicted)
        row = {
            "index": index,
            "true_count": count,
            "predicted_count": int(predicted_counts[index].item()),
            "predicted_positions_m": predicted.tolist(),
            "gospa_m": float(gospa["value_m"]),
            "gospa_localization_p_sum": float(gospa["localization_p_sum"]),
            "gospa_missed_p_sum": float(gospa["missed_p_sum"]),
            "gospa_false_p_sum": float(gospa["false_p_sum"]),
            "matched_errors_m": errors,
            "frequency_overlap": bool(targets.overlap[index].item()),
        }
        for threshold in (10, 30, 50, 100):
            row[f"tp_at_{threshold}m"] = g1.maximum_matches_within(truth, predicted, float(threshold))
        rows.append(row)
        for query, source in mappings[index].items():
            valid = targets.ignore[index, source] < 0.5
            prediction = logits[index, query, valid] >= 0.0
            target = targets.band[index, source, valid] > 0.5
            intersection = int((prediction & target).sum().item())
            union = int((prediction | target).sum().item())
            tp = intersection
            fp = int((prediction & ~target).sum().item())
            fn = int((~prediction & target).sum().item())
            band_f1.append(2 * tp / max(2 * tp + fp + fn, 1))
            band_iou.append(intersection / max(union, 1))
    result = {
        "overall": summarize_rows(rows),
        "by_k": {
            str(count): summarize_rows([row for row in rows if row["true_count"] == count])
            for count in range(4)
        },
        "by_frequency_overlap": {
            str(value).lower(): summarize_rows([row for row in rows if row["frequency_overlap"] is value])
            for value in (False, True)
        },
        "active_band_macro_f1": float(np.mean(band_f1)) if band_f1 else 1.0,
        "active_band_macro_iou": float(np.mean(band_iou)) if band_iou else 1.0,
        "samples": rows,
    }
    set_mode(context, training=True)
    del target_batch
    return result


def compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "samples"}


def checkpoint_payload(context: Context, epoch: int, metrics: dict[str, Any]) -> dict[str, Any]:
    return {"epoch": epoch, "track": context.track, "state": state_payload(context), "metrics": compact(metrics)}


def train_track(run_root: Path, seed: int, track: str, target_epoch: int) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if seed not in G3_INITIAL or track not in TRACKS:
        raise ValueError((seed, track))
    track_root = validate_output_path(run_root / TRAINING_ROOT_NAME / str(seed) / track)
    track_root.mkdir(parents=True, exist_ok=True)
    final_path = track_root / f"training_report_epoch{target_epoch}.json"
    if final_path.exists():
        raise FileExistsError(f"目标训练报告已存在: {final_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context = build_context(track, seed, device, config)
    train_features = load_features(run_root, "train")
    select_features = load_features(run_root, "val_select")
    train_targets = load_targets(run_root, "train")
    select_targets = load_targets(run_root, "val_select")
    optimizer = torch.optim.AdamW(
        context.parameter_groups, weight_decay=float(config["weight_decay"])
    )
    generator = torch.Generator().manual_seed(seed)
    last_path = track_root / "last.pt"
    history: list[dict[str, Any]] = []
    best_epoch = 0
    optimizer_steps = 0
    clipped_steps = 0
    start_epoch = 1
    if last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=False)
        if int(saved["epoch"]) >= target_epoch:
            raise RuntimeError("last checkpoint已达到或超过目标epoch")
        load_state(context, saved["state"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator_state"])
        history = saved["history"]
        best_epoch = int(saved["best_epoch"])
        optimizer_steps = int(saved["optimizer_steps"])
        clipped_steps = int(saved["clipped_steps"])
        start_epoch = int(saved["epoch"]) + 1
    else:
        initial = evaluate(context, select_features, select_targets, device)
        history = [{"epoch": 0, "validation": compact(initial), "training": None}]
        torch.save(checkpoint_payload(context, 0, initial), track_root / "best.pt")
    started = time.perf_counter()
    cached_targets = as_cached_targets(train_targets)
    for epoch in range(start_epoch, target_epoch + 1):
        set_mode(context, training=True)
        order = torch.randperm(len(train_targets.counts), generator=generator)
        sums = {name: 0.0 for name in ("exist", "band", "heatmap", "offset")}
        for start in range(0, len(order), int(config["batch_size"])):
            indices = order[start : start + int(config["batch_size"])]
            _, logits, _, heatmap, offset = forward_indices(context, train_features, indices, device)
            total, components, _ = r1.compute_losses(logits, heatmap, offset, cached_targets, indices, config)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(context.parameters, float(config["gradient_clip"]))
            clipped_steps += int(float(norm) > float(config["gradient_clip"]))
            optimizer.step()
            optimizer_steps += 1
            for name, value in components.items():
                sums[name] += float(value.detach().item()) * len(indices)
        if epoch % int(config["evaluate_every"]) == 0:
            metrics = evaluate(context, select_features, select_targets, device)
            history.append(
                {
                    "epoch": epoch,
                    "validation": compact(metrics),
                    "training": {name: value / len(train_targets.counts) for name, value in sums.items()},
                    "elapsed_seconds_this_call": time.perf_counter() - started,
                }
            )
            current_best = min(history, key=lambda row: (row["validation"]["overall"]["gospa_mean_m"], row["epoch"]))
            if int(current_best["epoch"]) == epoch:
                best_epoch = epoch
                torch.save(checkpoint_payload(context, epoch, metrics), track_root / "best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "state": state_payload(context),
                    "optimizer": optimizer.state_dict(),
                    "generator_state": generator.get_state(),
                    "history": history,
                    "best_epoch": best_epoch,
                    "optimizer_steps": optimizer_steps,
                    "clipped_steps": clipped_steps,
                },
                last_path,
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "track": track,
                        "epoch": epoch,
                        "gospa": metrics["overall"]["gospa_mean_m"],
                        "rmse": metrics["overall"]["matched_errors_m"]["rmse"],
                        "best_epoch": best_epoch,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if time.perf_counter() - started > float(config["track_wall_limit_seconds"]):
            raise RuntimeError(f"单轨超过墙钟预算: seed={seed}, track={track}")
    best = torch.load(track_root / "best.pt", map_location="cpu", weights_only=False)
    report = {
        "status": "PASS",
        "seed": seed,
        "track": track,
        "target_epoch": target_epoch,
        "best_epoch": int(best["epoch"]),
        "best": best["metrics"],
        "optimizer_steps": optimizer_steps,
        "gradient_clip_rate": clipped_steps / max(optimizer_steps, 1),
        "duration_seconds_this_call": time.perf_counter() - started,
        "checkpoint": identity(track_root / "best.pt"),
        "test_executed": False,
    }
    write_json(final_path, report)
    del context, train_targets, select_targets, cached_targets
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return report


def latest_report(run_root: Path, seed: int, track: str) -> dict[str, Any]:
    root = run_root / TRAINING_ROOT_NAME / str(seed) / track
    paths = list(root.glob("training_report_epoch*.json"))
    if not paths:
        raise RuntimeError(f"训练报告缺失: {seed}/{track}")
    return max((read_json(path) for path in paths), key=lambda row: int(row["target_epoch"]))


def select_candidate(run_root: Path) -> dict[str, Any]:
    _, config = verify_run(run_root)
    seed = int(config["screen_seed"])
    reports = {track: latest_report(run_root, seed, track) for track in TRACKS}
    if any(int(report["target_epoch"]) < int(config["screen_epoch"]) for report in reports.values()):
        raise RuntimeError("P1三轨未完成16 epoch")
    base = reports["a0_large_frozen"]["best"]["overall"]
    eligible = ["a0_large_frozen"]
    for track in ("a1_d8_tail", "a2_joint_tail"):
        current = reports[track]["best"]["overall"]
        recall_delta = current["recall_at_100m"] - base["recall_at_100m"]
        coverage_delta = current["matched_pair_coverage_of_true"] - base["matched_pair_coverage_of_true"]
        rmse_ratio = current["matched_errors_m"]["rmse"] / base["matched_errors_m"]["rmse"]
        if (
            recall_delta >= float(config["recall_100m_noninferiority"])
            and coverage_delta >= float(config["coverage_noninferiority"])
            and rmse_ratio <= float(config["rmse_ratio_maximum"])
        ):
            eligible.append(track)
    ranked = sorted(eligible, key=lambda track: TRACKS.index(track))
    candidate = ranked[0]
    for track in ranked[1:]:
        current = reports[track]["best"]["overall"]["gospa_mean_m"]
        chosen = reports[candidate]["best"]["overall"]["gospa_mean_m"]
        if current < chosen - float(config["gospa_tie_margin_m"]):
            candidate = track
    report = {
        "status": "PASS",
        "seed": seed,
        "candidate": candidate,
        "eligible": eligible,
        "screen_reports": reports,
        "selection_uses_val_compare": False,
        "test_executed": False,
    }
    write_json(run_root / "p1_selection_report.json", report)
    return report


def needs_extension(report: dict[str, Any], config: dict[str, Any]) -> bool:
    target = int(report["target_epoch"])
    return target == int(config["comparison_epoch"]) and int(report["best_epoch"]) >= target - int(config["evaluate_every"])


def paired_bootstrap(
    base: list[dict[str, Any]], candidate: list[dict[str, Any]], repetitions: int, seed: int
) -> dict[str, Any]:
    if len(base) != len(candidate):
        raise RuntimeError("配对样本数不一致")
    by_k = {count: [i for i, row in enumerate(base) if row["true_count"] == count] for count in range(4)}
    gospa_diff = np.asarray([candidate[i]["gospa_m"] - base[i]["gospa_m"] for i in range(len(base))])

    def rmse(rows: list[dict[str, Any]], selected: np.ndarray) -> float:
        errors = [value for index in selected for value in rows[int(index)]["matched_errors_m"]]
        values = np.asarray(errors, dtype=np.float64)
        return float(np.sqrt(np.mean(values**2)))

    rng = np.random.default_rng(seed)
    bg: list[float] = []
    br: list[float] = []
    for _ in range(repetitions):
        selected = np.concatenate([rng.choice(indices, len(indices), replace=True) for indices in by_k.values()])
        bg.append(float(gospa_diff[selected].mean()))
        br.append(rmse(candidate, selected) - rmse(base, selected))
    all_indices = np.arange(len(base))
    return {
        "gospa_candidate_minus_a0_m": {
            "mean": float(gospa_diff.mean()),
            "ci95": np.quantile(bg, [0.025, 0.975]).tolist(),
            "by_k_mean": {
                str(count): float(gospa_diff[indices].mean()) for count, indices in by_k.items()
            },
        },
        "rmse_candidate_minus_a0_m": {
            "mean": rmse(candidate, all_indices) - rmse(base, all_indices),
            "ci95": np.quantile(br, [0.025, 0.975]).tolist(),
        },
    }


def evaluate_final(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    selection = read_json(run_root / "p1_selection_report.json")
    candidate = selection["candidate"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = load_features(run_root, "val_compare")
    targets = load_targets(run_root, "val_compare")
    results: dict[str, Any] = {}
    seeds = (int(config["screen_seed"]), int(config["confirm_seed"]))
    tracks = ("a0_large_frozen",) if candidate == "a0_large_frozen" else ("a0_large_frozen", candidate)
    for seed in seeds:
        results[str(seed)] = {}
        for track in tracks:
            report = latest_report(run_root, seed, track)
            context = build_context(track, seed, device, config)
            checkpoint = torch.load(
                run_root / TRAINING_ROOT_NAME / str(seed) / track / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            load_state(context, checkpoint["state"])
            metrics = evaluate(context, features, targets, device)
            results[str(seed)][track] = {
                "training": report,
                "metrics": compact(metrics),
                "samples": metrics["samples"],
            }
            del context
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if candidate != "a0_large_frozen":
            results[str(seed)]["paired_bootstrap"] = paired_bootstrap(
                results[str(seed)]["a0_large_frozen"]["samples"],
                results[str(seed)][candidate]["samples"],
                int(config["bootstrap_repetitions"]),
                seed + 99,
            )
    if candidate == "a0_large_frozen":
        classification = "G4_KEEP_FROZEN"
    else:
        improved = [
            results[str(seed)]["paired_bootstrap"]["gospa_candidate_minus_a0_m"]["mean"] < 0.0
            for seed in seeds
        ]
        guards = []
        for seed in seeds:
            base = results[str(seed)]["a0_large_frozen"]["metrics"]["overall"]
            current = results[str(seed)][candidate]["metrics"]["overall"]
            guards.append(
                current["matched_pair_coverage_of_true"] - base["matched_pair_coverage_of_true"]
                >= float(config["coverage_noninferiority"])
                and current["recall_at_100m"] - base["recall_at_100m"]
                >= float(config["recall_100m_noninferiority"])
                and current["matched_errors_m"]["rmse"] / base["matched_errors_m"]["rmse"]
                <= float(config["rmse_ratio_maximum"])
            )
        classification = "G4_PARTIAL_UNFREEZE_SELECTED" if all(improved) and all(guards) else "G4_KEEP_FROZEN"
    report = {
        "status": "PASS",
        "gate": "E2E-G4-P2",
        "candidate": candidate,
        "classification": classification,
        "results": results,
        "test_executed": False,
    }
    write_json(run_root / "p2_report.json", report)
    return report


def preflight_run(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    if (run_root / "p0_report.json").exists():
        raise FileExistsError("P0报告已存在")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = prepare_dpd_cache(run_root, manifest, config, device)
    features = prepare_feature_cache(run_root, manifest, device)
    train_features = load_features(run_root, "train")
    train_targets = load_targets(run_root, "train")
    probe_indices = torch.tensor(
        [int(torch.nonzero(train_targets.counts == count, as_tuple=False)[0].item()) for count in (1, 2, 3)]
    )
    checks: dict[str, Any] = {}
    durations: dict[str, float] = {}
    peak_memory: dict[str, int] = {}
    initial_digests: dict[str, str] = {}
    for track in TRACKS:
        context = build_context(track, int(config["screen_seed"]), device, config)
        initial_digests[track] = state_digest(context)
        cached_targets = as_cached_targets(train_targets)
        # 第一次CUDA/算子调用只用于预热，不进入稳定吞吐投影。
        _, logits, _, heatmap, offset = forward_indices(context, train_features, probe_indices, device)
        total, components, _ = r1.compute_losses(logits, heatmap, offset, cached_targets, probe_indices, config)
        total.backward()
        for parameter in context.parameters:
            parameter.grad = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        repetitions: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            _, logits, _, heatmap, offset = forward_indices(context, train_features, probe_indices, device)
            total, components, _ = r1.compute_losses(logits, heatmap, offset, cached_targets, probe_indices, config)
            total.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            repetitions.append(time.perf_counter() - started)
            for parameter in context.parameters:
                parameter.grad = None
        durations[track] = float(np.median(repetitions))
        peak_memory[track] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        trainable = {name: sum(parameter.numel() for parameter in group["params"]) for name, group in zip([g["name"] for g in context.parameter_groups], context.parameter_groups)}
        finite_gradients = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all().item()
            for parameter in context.parameters
        )
        checks[track] = {
            "loss_finite": bool(torch.isfinite(total).item()),
            "component_finite": all(bool(torch.isfinite(value).item()) for value in components.values()),
            "gradients_finite": finite_gradients,
            "trainable_parameters": trainable,
        }
        del context
    batches_per_epoch = math.ceil(1024 / int(config["batch_size"]))
    projected_seconds = max(durations.values()) * batches_per_epoch * 24 * 5
    status = "PASS" if all(
        row["loss_finite"] and row["component_finite"] and row["gradients_finite"]
        for row in checks.values()
    ) and projected_seconds <= float(config["total_training_wall_limit_seconds"]) else "STOP_RESOURCE_OR_ENGINEERING"
    report = {
        "status": status,
        "gate": "E2E-G4-P0",
        "dpd_cache": {"duration_seconds": cache["duration_seconds"], "reused_train": cache["reused_train"]},
        "feature_cache": {"duration_seconds": features["duration_seconds"]},
        "checks": checks,
        "probe_seconds": durations,
        "peak_memory_bytes": peak_memory,
        "projected_five_track_24_epoch_seconds": projected_seconds,
        "initial_state_digests": initial_digests,
        "test_executed": False,
    }
    write_json(run_root / "p0_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def repair_preflight(run_root: Path) -> dict[str, Any]:
    manifest = read_json(run_root / "manifest.json")
    if read_json(run_root / "p0_report.json")["status"] != "STOP_RESOURCE_OR_ENGINEERING":
        raise RuntimeError("原P0不是可修复的投影停止状态")
    if (run_root / "p0_repair_report.json").exists() or (run_root / "repair_receipt.json").exists():
        raise FileExistsError("修复证据已存在")
    original = {row["path"]: row for row in manifest["code"]}
    repaired = {row["path"]: row for row in code_identity()}
    changed = [path for path in original if original[path] != repaired.get(path)]
    if changed != [str(SCRIPT_PATH.resolve())] or set(original) != set(repaired):
        raise RuntimeError(f"修复范围超出G4入口: {changed}")
    write_json(
        run_root / "repair_receipt.json",
        {
            "status": "PASS",
            "failure_stage": "p0_wall_projection_before_optimizer_step",
            "root_cause": "first CUDA cold-start latency was projected as stable per-batch latency",
            "repair": "one warm-up plus five synchronized repetitions and median stable latency",
            "scientific_contract_changed": False,
            "optimizer_steps_before_repair": 0,
            "original_p0_preserved": True,
            "original_code": manifest["code"],
            "repaired_code": list(repaired.values()),
        },
    )
    manifest, config = verify_run(run_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_features = load_features(run_root, "train")
    train_targets = load_targets(run_root, "train")
    probe_indices = torch.tensor(
        [int(torch.nonzero(train_targets.counts == count, as_tuple=False)[0].item()) for count in (1, 2, 3)]
    )
    checks: dict[str, Any] = {}
    durations: dict[str, float] = {}
    peak_memory: dict[str, int] = {}
    initial_digests: dict[str, str] = {}
    cached_targets = as_cached_targets(train_targets)
    for track in TRACKS:
        context = build_context(track, int(config["screen_seed"]), device, config)
        initial_digests[track] = state_digest(context)
        _, logits, _, heatmap, offset = forward_indices(context, train_features, probe_indices, device)
        total, components, _ = r1.compute_losses(logits, heatmap, offset, cached_targets, probe_indices, config)
        total.backward()
        for parameter in context.parameters:
            parameter.grad = None
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        repetitions: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            _, logits, _, heatmap, offset = forward_indices(context, train_features, probe_indices, device)
            total, components, _ = r1.compute_losses(logits, heatmap, offset, cached_targets, probe_indices, config)
            total.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            repetitions.append(time.perf_counter() - started)
            for parameter in context.parameters:
                parameter.grad = None
        durations[track] = float(np.median(repetitions))
        peak_memory[track] = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        checks[track] = {
            "loss_finite": bool(torch.isfinite(total).item()),
            "component_finite": all(bool(torch.isfinite(value).item()) for value in components.values()),
            "gradients_finite": all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in context.parameters
            ),
        }
        del context
    batches_per_epoch = math.ceil(1024 / int(config["batch_size"]))
    projected_seconds = max(durations.values()) * batches_per_epoch * 24 * 5
    status = "PASS" if all(
        row["loss_finite"] and row["component_finite"] and row["gradients_finite"]
        for row in checks.values()
    ) and projected_seconds <= float(config["total_training_wall_limit_seconds"]) else "STOP_RESOURCE_OR_ENGINEERING"
    report = {
        "status": status,
        "gate": "E2E-G4-P0-REPAIR",
        "checks": checks,
        "stable_probe_seconds_median_of_five": durations,
        "peak_memory_bytes": peak_memory,
        "projected_five_track_24_epoch_seconds": projected_seconds,
        "initial_state_digests": initial_digests,
        "test_executed": False,
    }
    write_json(run_root / "p0_repair_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


def finalize(run_root: Path) -> dict[str, Any]:
    manifest, config = verify_run(run_root)
    p0 = read_json(
        run_root / "p0_repair_report.json"
        if (run_root / "p0_repair_report.json").exists()
        else run_root / "p0_report.json"
    )
    selection = read_json(run_root / "p1_selection_report.json")
    p2 = read_json(run_root / "p2_report.json")
    cache = read_json(run_root / "dpd_cache_manifest.json")
    cache_ok = all(
        identity(Path(row["path"]))["sha256"] == row["sha256"]
        for rows in cache["files"].values()
        for row in rows
    )
    status = p2["classification"] if p0["status"] == "PASS" and cache_ok else "G4_STOP"
    report = {
        "status": status,
        "gate": "E2E-G4",
        "run_id": manifest["run_id"],
        "candidate": selection["candidate"],
        "p0_status": p0["status"],
        "cache_identity_after_training": cache_ok,
        "test_executed": False,
        "next_gate_unlocked": status in ("G4_PARTIAL_UNFREEZE_SELECTED", "G4_KEEP_FROZEN"),
        "config": config,
    }
    write_json(run_root / "final_report.json", report)
    return report


def repair_finalize(run_root: Path) -> dict[str, Any]:
    if not (run_root / "p2_report.json").exists():
        raise RuntimeError("P2尚未完成，不能执行恢复性finalize")
    integrity_receipt = read_json(run_root / "integrity_repair_receipt.json")
    prior = {row["path"]: row for row in integrity_receipt["repaired_code"]}
    current_rows = code_identity()
    current = {row["path"]: row for row in current_rows}
    changed = [path for path in prior if prior[path] != current.get(path)]
    if changed != [str(SCRIPT_PATH.resolve())] or set(prior) != set(current):
        raise RuntimeError(f"finalize修复范围超出G4入口: {changed}")
    write_json(
        run_root / "finalize_repair_receipt.json",
        {
            "status": "PASS",
            "cause": "finalize read the preserved failed P0 instead of the passed repaired P0",
            "repair": "prefer p0_repair_report when present",
            "report_only_change": True,
            "training_or_evaluation_rerun": False,
            "prior_code": integrity_receipt["repaired_code"],
            "repaired_code": current_rows,
        },
    )
    return finalize(run_root)


def orchestrate(run_root: Path) -> None:
    _, config = verify_run(run_root)
    if (run_root / "p0_repair_report.json").exists():
        p0 = read_json(run_root / "p0_repair_report.json")
    elif (run_root / "p0_report.json").exists():
        p0 = read_json(run_root / "p0_report.json")
    else:
        p0 = preflight_run(run_root)
    if p0["status"] != "PASS":
        raise RuntimeError(f"G4 P0停止: {p0['status']}")
    screen_seed = int(config["screen_seed"])
    confirm_seed = int(config["confirm_seed"])
    def ensure_track(seed: int, track: str, target_epoch: int) -> dict[str, Any]:
        path = run_root / TRAINING_ROOT_NAME / str(seed) / track / f"training_report_epoch{target_epoch}.json"
        return read_json(path) if path.exists() else train_track(run_root, seed, track, target_epoch)

    if not (run_root / "feature_cache_v2_manifest.json").exists():
        raise RuntimeError("分片特征缓存修复尚未完成")
    for track in TRACKS:
        ensure_track(screen_seed, track, int(config["screen_epoch"]))
    selection = select_candidate(run_root)
    candidate = selection["candidate"]
    for track in dict.fromkeys(("a0_large_frozen", candidate)):
        ensure_track(screen_seed, track, int(config["comparison_epoch"]))
    screen_reports = {
        track: latest_report(run_root, screen_seed, track)
        for track in dict.fromkeys(("a0_large_frozen", candidate))
    }
    if any(needs_extension(report, config) for report in screen_reports.values()):
        for track in screen_reports:
            ensure_track(screen_seed, track, int(config["extension_epoch"]))
    confirm_tracks = ("a0_large_frozen",) if candidate == "a0_large_frozen" else ("a0_large_frozen", candidate)
    for track in confirm_tracks:
        ensure_track(confirm_seed, track, int(config["comparison_epoch"]))
    confirm_reports = {track: latest_report(run_root, confirm_seed, track) for track in confirm_tracks}
    if any(needs_extension(report, config) for report in confirm_reports.values()):
        for track in confirm_tracks:
            ensure_track(confirm_seed, track, int(config["extension_epoch"]))
    evaluate_final(run_root)
    final = finalize(run_root)
    print(json.dumps(final, ensure_ascii=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--run-id", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--run-root", type=Path, required=True)
    repair_parser = sub.add_parser("repair-preflight")
    repair_parser.add_argument("--run-root", type=Path, required=True)
    feature_repair_parser = sub.add_parser("repair-feature-cache")
    feature_repair_parser.add_argument("--run-root", type=Path, required=True)
    finalize_repair_parser = sub.add_parser("repair-finalize")
    finalize_repair_parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.run_id)
    elif args.command == "repair-preflight":
        repair_preflight(args.run_root.resolve())
    elif args.command == "repair-feature-cache":
        print(json.dumps(repair_feature_cache(args.run_root.resolve()), ensure_ascii=False), flush=True)
    elif args.command == "repair-finalize":
        print(json.dumps(repair_finalize(args.run_root.resolve()), ensure_ascii=False), flush=True)
    else:
        orchestrate(args.run_root.resolve())


if __name__ == "__main__":
    main()
