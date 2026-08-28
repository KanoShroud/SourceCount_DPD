"""S2-G5-R4专用CH3只读memmap Dataset与缓存审计工具。"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


PREPROCESS_VERSION = "train_v26_sample_zscore_v1"
AUDIT_SEED = 20260903


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    if path.exists():
        raise FileExistsError(f"拒绝覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def preprocess_sample(raw: np.ndarray) -> np.ndarray:
    """逐项复现train_v26.SourceDetectionDataset的单样本运算顺序。"""
    sample = np.asarray(raw, dtype=np.float32).transpose(2, 1, 0)
    sample = np.log(sample + 1.0)
    mean = sample.mean()
    std = sample.std() + 1e-6
    return np.asarray((sample - mean) / std, dtype=np.float32)


def build_cache(source: Path, cache: Path, manifest_path: Path) -> dict[str, Any]:
    source = source.resolve()
    cache = cache.resolve()
    require(source.is_file(), f"源MAT不存在: {source}")
    require(not cache.exists(), f"拒绝覆盖缓存: {cache}")
    require(not manifest_path.exists(), f"拒绝覆盖manifest: {manifest_path}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with h5py.File(source, "r") as handle:
        spectra = handle["mtr_sub_all"]
        require(spectra.dtype == np.float32, "mtr_sub_all必须为float32")
        require(spectra.shape[:3] == (81, 81, 19), "频谱shape不符合预期")
        sample_count = int(spectra.shape[3])
        output = np.lib.format.open_memmap(
            cache,
            mode="w+",
            dtype=np.float32,
            shape=(sample_count, 19, 81, 81),
        )
        try:
            for index in range(sample_count):
                output[index] = preprocess_sample(spectra[:, :, :, index])
                if (index + 1) % 512 == 0 or index + 1 == sample_count:
                    output.flush()
                    print(f"cache {index + 1}/{sample_count}", flush=True)
        except BaseException:
            del output
            cache.unlink(missing_ok=True)
            raise
        output.flush()
        del output
        counts = np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1)
        source_meta = {
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "sample_count": sample_count,
            "source_count_histogram": {
                str(key): int(value) for key, value in sorted(Counter(map(int, counts)).items())
            },
        }
    report = {
        "status": "PASS",
        "preprocess_version": PREPROCESS_VERSION,
        "source": str(source),
        "source_meta": source_meta,
        "cache": str(cache),
        "cache_shape": [source_meta["sample_count"], 19, 81, 81],
        "cache_dtype": "float32",
        "cache_size_bytes": cache.stat().st_size,
        "cache_sha256": sha256_file(cache),
        "duration_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, report)
    return report


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class LazySourceDetectionDataset(Dataset):
    """标签常驻RAM、频谱由只读NumPy memmap按样本读取。"""

    def __init__(
        self,
        mat_path: str | Path,
        *,
        cache_path: str | Path,
        manifest_path: str | Path,
        augment: bool = False,
        normalize: str = "sample_zscore",
        max_src_override: int | None = None,
        verify_source_hash: bool = False,
    ) -> None:
        self.mat_path = Path(mat_path).resolve()
        self.cache_path = Path(cache_path).resolve()
        self.manifest_path = Path(manifest_path).resolve()
        self.augment = augment
        require(normalize == "sample_zscore", "R4懒加载仅支持sample_zscore")
        manifest = load_manifest(self.manifest_path)
        require(manifest["status"] == "PASS", "缓存manifest未通过")
        require(manifest["preprocess_version"] == PREPROCESS_VERSION, "预处理版本不一致")
        require(Path(manifest["source"]).resolve() == self.mat_path, "缓存源路径不一致")
        require(self.cache_path.is_file(), f"缓存不存在: {self.cache_path}")
        require(self.cache_path.stat().st_size == manifest["cache_size_bytes"], "缓存大小不一致")
        if verify_source_hash:
            require(sha256_file(self.mat_path) == manifest["source_meta"]["sha256"], "源MAT哈希不一致")

        self.spectra = np.load(self.cache_path, mmap_mode="r")
        require(list(self.spectra.shape) == manifest["cache_shape"], "缓存shape不一致")
        require(self.spectra.dtype == np.float32, "缓存dtype不一致")
        with h5py.File(self.mat_path, "r") as handle:
            self.src_count = np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1)
            self.band_mask = np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0)
            self.ignore_mask = np.asarray(handle["ignore_mask_all"], dtype=np.float32).transpose(2, 1, 0)
            self.N_sub = int(np.asarray(handle["N_sub_val"]).item())
            self.max_src = int(np.asarray(handle["max_src_val"]).item())
            self.num_count_classes = int(np.asarray(handle["num_count_classes"]).item())
            self.avg_snr = np.asarray(handle["avg_snr_all"], dtype=np.float32).reshape(-1)
        require(len(self.src_count) == len(self.spectra), "缓存与标签样本数不一致")
        if max_src_override is not None and max_src_override > self.max_src:
            count = len(self.src_count)
            pad = max_src_override - self.max_src
            self.band_mask = np.concatenate(
                [self.band_mask, np.zeros((count, pad, self.N_sub), dtype=np.float32)], axis=1
            )
            self.ignore_mask = np.concatenate(
                [self.ignore_mask, np.zeros((count, pad, self.N_sub), dtype=np.float32)], axis=1
            )
            self.max_src = max_src_override
        print(
            f"懒加载 {self.mat_path}: {len(self)} 样本, cache={self.cache_path}, "
            f"N_sub={self.N_sub}, max_src={self.max_src}",
            flush=True,
        )

    def __len__(self) -> int:
        return int(self.spectra.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(np.array(self.spectra[idx], dtype=np.float32, copy=True))
        source_count = torch.tensor(self.src_count[idx], dtype=torch.long)
        band_mask = torch.from_numpy(self.band_mask[idx].copy())
        ignore_mask = torch.from_numpy(self.ignore_mask[idx].copy())
        if self.augment:
            if torch.rand(1) < 0.5:
                x = x.flip(-1)
            if torch.rand(1) < 0.5:
                x = x.flip(-2)
            if torch.rand(1) < 0.3:
                x = x + 0.05 * torch.randn_like(x)
        return x, source_count, band_mask, ignore_mask


def audit_indices(counts: np.ndarray, total: int = 512) -> np.ndarray:
    rng = np.random.default_rng(AUDIT_SEED)
    selected: set[int] = set()
    midpoint = len(counts) // 2
    for half_start, half_end in ((0, midpoint), (midpoint, len(counts))):
        for source_count in range(4):
            candidates = np.flatnonzero(counts[half_start:half_end] == source_count) + half_start
            selected.update(map(int, rng.choice(candidates, size=48, replace=False)))
    boundary = list(range(0, 16)) + list(range(midpoint - 16, midpoint + 16)) + list(range(len(counts) - 16, len(counts)))
    selected.update(index for index in boundary if 0 <= index < len(counts))
    remaining = np.asarray(sorted(set(range(len(counts))) - selected), dtype=np.int64)
    if len(selected) < total:
        selected.update(map(int, rng.choice(remaining, size=total - len(selected), replace=False)))
    result = np.asarray(sorted(selected), dtype=np.int64)
    require(result.size == total, f"审计索引数量错误: {result.size}")
    return result


def audit_cache(
    source: Path,
    cache: Path,
    manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    require(sha256_file(source) == manifest["source_meta"]["sha256"], "源MAT哈希变化")
    require(sha256_file(cache) == manifest["cache_sha256"], "缓存哈希变化")
    mapped = np.load(cache, mmap_mode="r")
    with h5py.File(source, "r") as handle:
        counts = np.asarray(handle["src_count_all"], dtype=np.int64).reshape(-1)
        indices = audit_indices(counts)
        maximum = 0.0
        total_error = 0.0
        exact = 0
        for index in indices:
            expected = preprocess_sample(handle["mtr_sub_all"][:, :, :, int(index)])
            actual = np.asarray(mapped[index])
            difference = np.abs(expected - actual)
            maximum = max(maximum, float(difference.max()))
            total_error += float(difference.sum())
            exact += int(np.array_equal(expected, actual))
        mean_error = total_error / (indices.size * np.prod(mapped.shape[1:]))
    require(np.isfinite(mapped[indices]).all(), "抽样缓存包含NaN/Inf")
    require(maximum <= 1e-7 and mean_error <= 1e-9, "缓存抽样数值不等价")
    report = {
        "status": "PASS",
        "sample_count": int(indices.size),
        "indices": indices.tolist(),
        "source_count_histogram": {
            str(key): int(value) for key, value in sorted(Counter(map(int, counts[indices])).items())
        },
        "exact_sample_count": exact,
        "max_abs_error": maximum,
        "mean_abs_error": mean_error,
        "thresholds": {"max_abs_error": 1e-7, "mean_abs_error": 1e-9},
    }
    write_json(output, report)
    return report
