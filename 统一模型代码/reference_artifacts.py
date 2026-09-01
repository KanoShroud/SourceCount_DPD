"""原项目冻结模型与选择器的相对路径、大小和 SHA256 注册表。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

if __package__:
    from .runtime_paths import reference_output_path
else:
    from runtime_paths import reference_output_path


@dataclass(frozen=True)
class ArtifactSpec:
    relative_path: str
    size_bytes: int
    sha256: str


ARTIFACTS = {
    "ch3_seed42": ArtifactSpec(
        "s2g5r4_ch3_scale/20260828_151735/train_16k/"
        "best_model_v26_B_M10.pth",
        10_052_847,
        "f2f7a7c345f1866b871282670f45671d930de34bb06493a9828a9d04a38699c4",
    ),
    "ch3_seed1042": ArtifactSpec(
        "s2g5r6/20260829_170441/training/ch3_seed1042/"
        "best_model_v26_B_M10.pth",
        10_052_847,
        "baecd8b0be72259b3b37ea2381708e80015a9ddb0fa749ae685ac64ff30afaec",
    ),
    "ch3_seed2042": ArtifactSpec(
        "s2g5r6/20260829_170441/training/ch3_seed2042/"
        "best_model_v26_B_M10.pth",
        10_052_847,
        "7ed801179d29da4ec1ece0cc8d506092d537f84c5ae1d6c81fdb5a038031019b",
    ),
    "d8_seed42": ArtifactSpec(
        "s2g4r4_scale/20260826_132829/09_training/n8192/hard_actual/"
        "best_yolo_dualhead_std.pth",
        153_838_331,
        "4caaf2b96c2f8eb666b417f0cffe4ab90760315f9bd92c4d6ce4afcd425e0e7b",
    ),
    "d8_seed1042": ArtifactSpec(
        "s2g5r6/20260829_170441/training/d8_seed1042/"
        "best_yolo_dualhead_std.pth",
        153_815_035,
        "a4f71e826072e0ca47266db65d9271bc8dfcb6e27d5fa8b0e8f8942f08e10d29",
    ),
    "d8_seed2042": ArtifactSpec(
        "s2g5r6/20260829_170441/training/d8_seed2042/"
        "best_yolo_dualhead_std.pth",
        153_841_403,
        "78d74067bc09e2c7a9116ec676e55882f3cc60eed3ead36da7cf746fb9f772be",
    ),
    "r5_ordinal_probe_seed42": ArtifactSpec(
        "s2g5r5_candidate_k/20260828_222900/02_ordinal_probe/"
        "ordinal_probe.joblib",
        5_379,
        "cb926c70dda82a0cf86e3f690b0a73d0830f518413abd5afdad6203944301d23",
    ),
    "r5_d8_selector_seed42": ArtifactSpec(
        "s2g5r5_candidate_k/20260828_222900/05_selector_analysis/"
        "d8_selector.joblib",
        1_537,
        "3cf278b5ef5b4799360d9b3c1e8afc63eb8820302876d94caba3e9f87553e9d8",
    ),
}

CORE_ARTIFACTS = tuple(ARTIFACTS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(name: str) -> dict[str, str | int]:
    """解析并核对一个冻结产物，不对参考目录执行任何写操作。"""
    try:
        spec = ARTIFACTS[name]
    except KeyError as exc:
        raise KeyError(f"未知冻结产物: {name}") from exc
    path = reference_output_path(spec.relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"冻结产物不是文件: {path}")
    size_bytes = path.stat().st_size
    if size_bytes != spec.size_bytes:
        raise RuntimeError(
            f"{name} 文件大小变化: {size_bytes} != {spec.size_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != spec.sha256:
        raise RuntimeError(
            f"{name} SHA256 变化: {actual_sha256} != {spec.sha256}"
        )
    return {
        "name": name,
        "path": str(path),
        "size_bytes": size_bytes,
        "sha256": actual_sha256,
    }


def verify_artifacts(names: Iterable[str] = CORE_ARTIFACTS) -> list[dict[str, str | int]]:
    return [verify_artifact(name) for name in names]
