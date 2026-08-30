"""S2-G5-R6-A 16k CH3跨seed正式训练入口。

直接复用R4已经完成等价性、容量和正式训练验证的冻结NPY缓存。除训练seed和
独立输出目录外，配置与seed 42的16k训练完全一致；不重建MAT或缓存。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from s2g1_train_ch3 import build_model, configure_reproducibility, run_capacity, run_train
from s2g5_r4_lazy_dataset import require, sha256_file
from s2g5_r4_train_ch3 import RUN_ROOT, formal_train_args, lazy_factory_for


APPROVED_SEEDS = (1042, 2042)
DATA_16K = RUN_ROOT / "training_views" / "data_16k"
TRAIN_SOURCE = DATA_16K / "train_data.mat"
VALIDATION = DATA_16K / "val_data.mat"
CACHE = RUN_ROOT / "lazy_cache" / "train_16k_sample_zscore.npy"
MANIFEST = RUN_ROOT / "lazy_cache" / "train_16k_sample_zscore.json"
EXPECTED = {
    "train_source": "594c0fd607c6c615839a381ce26102c59b20e76d4799a9e609cc768d6c58c880",
    "validation": "d4cb6af9e3f99c2162c306105b29f947e48ac85c38075acc8226b6403046d6ad",
    "cache": "a455fe55397b312ff5b42699e6b69e6cda7d9ca3649eed6b83fa20d8ccca8dda",
    "manifest": "6ed01e32276ba5f2b81507aa6ed25356f3883c0e8387f1d926e3f54949b4de30",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=(42, *APPROVED_SEEDS), required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--initialization_only", action="store_true")
    parser.add_argument("--pilot_output", type=Path)
    return parser.parse_args()


def state_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state_dict.items():
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def verify_frozen_inputs() -> dict[str, str]:
    paths = {
        "train_source": TRAIN_SOURCE,
        "validation": VALIDATION,
        "cache": CACHE,
        "manifest": MANIFEST,
    }
    actual = {name: sha256_file(path) for name, path in paths.items()}
    require(actual == EXPECTED, f"R4冻结CH3输入SHA变化: {actual}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    require(manifest.get("status") == "PASS", "R4缓存manifest非PASS")
    require(manifest["source_meta"]["sample_count"] == 16384, "R4缓存样本数不是16k")
    require(
        manifest["source_meta"]["source_count_histogram"]
        == {"0": 4096, "1": 4096, "2": 4096, "3": 4096},
        "R4缓存K分层错误",
    )
    return actual


def main() -> int:
    cli = parse_args()
    if cli.initialization_only:
        configure_reproducibility(cli.seed, True)
        model = build_model(19, 10, "transformer", torch.device("cpu"))
        print(json.dumps({"seed": cli.seed, "sha256": state_hash(model.state_dict())}))
        return 0

    if cli.pilot_output is not None:
        verify_frozen_inputs()
        args = formal_train_args(cli.pilot_output.parent / "unused_training_output")
        args.seed = cli.seed
        args.output = cli.pilot_output
        result = run_capacity(
            args,
            dataset_factory=lazy_factory_for(TRAIN_SOURCE, CACHE, MANIFEST),
        )
        verify_frozen_inputs()
        return int(result)

    require(cli.seed in APPROVED_SEEDS, f"未批准的新训练seed: {cli.seed}")
    require(cli.output_dir is not None, "正式训练必须提供--output_dir")
    verify_frozen_inputs()
    output_dir = cli.output_dir.resolve()
    require(not output_dir.exists(), f"拒绝覆盖训练目录: {output_dir}")
    args = formal_train_args(output_dir)
    args.seed = cli.seed
    result = run_train(
        args,
        dataset_factory=lazy_factory_for(TRAIN_SOURCE, CACHE, MANIFEST),
    )
    verify_frozen_inputs()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
