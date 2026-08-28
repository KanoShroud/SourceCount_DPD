"""S2-G5-R1统一数据、Hard-19-Actual标签与分流契约审计。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


EXPECTED_PROFILE = "s2g5r1"
EXPECTED_LABEL_PROFILE = "hard19_actual_t020"
EXPECTED_CONTRACT_VERSION = "main31_v3_shared_20260827"
EXPECTED_THRESHOLD = 0.2
EXPECTED_ALPHA_FACTOR = 1.3


def route_source_count(source_count: int) -> str:
    """返回统一系统对给定源数的下游路由。"""
    if source_count == 0:
        return "stop_no_source"
    if source_count == 1:
        return "single_source_locator_pending"
    if source_count in (2, 3):
        return "d8_hard19_actual"
    raise ValueError(f"不支持的源数: {source_count}")


def locate_single_source(*_args: Any, **_kwargs: Any) -> None:
    """K=1定位预留接口；R1明确不提供临时实现。"""
    raise NotImplementedError("K=1定位方法尚未确定；S2-G5仅保留接口。")


def _matlab_text(handle: h5py.File, name: str) -> str:
    raw = np.asarray(handle[name])
    if raw.dtype.kind in "ui":
        return "".join(chr(int(value)) for value in raw.ravel(order="F") if value)
    return "".join(str(value) for value in raw.ravel(order="F"))


def _scalar(handle: h5py.File, name: str) -> float:
    return float(np.asarray(handle[name]).reshape(-1)[0])


def _recompute_labels(
    fc_offset: np.ndarray,
    bw_actual: np.ndarray,
    source_count: np.ndarray,
    sub_lo: np.ndarray,
    sub_hi: np.ndarray,
    window_width: float,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    samples, max_sources = fc_offset.shape
    positive = np.zeros((samples, max_sources, sub_lo.size), dtype=np.float32)
    ignore = np.zeros_like(positive)
    for sample_idx, count in enumerate(source_count):
        for source_idx in range(int(count)):
            actual_lo = fc_offset[sample_idx, source_idx] - bw_actual[sample_idx, source_idx] / 2
            actual_hi = fc_offset[sample_idx, source_idx] + bw_actual[sample_idx, source_idx] / 2
            overlap = np.maximum(
                0.0,
                np.minimum(actual_hi, sub_hi) - np.maximum(actual_lo, sub_lo),
            )
            coverage = overlap / window_width
            positive[sample_idx, source_idx] = coverage >= threshold
            ignore[sample_idx, source_idx] = (coverage > 0) & (coverage < threshold)
    return positive, ignore


def audit_mat(path: Path, expected_profile: str = EXPECTED_PROFILE) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        required = {
            "src_count_all",
            "band_mask_all",
            "ignore_mask_all",
            "fc_offset_all",
            "symbolRate_all",
            "BW_actual_all",
            "sig_rcv_real_all",
            "sig_rcv_imag_all",
            "sub_f_lo_val",
            "sub_f_hi_val",
            "B_win_val",
            "B_step_val",
            "fs_val",
            "thresh_val",
            "gate_profile_val",
            "band_label_profile_val",
            "physical_contract_version_val",
            "symbolRate_min_val",
            "symbolRate_max_val",
            "arfa_val",
            "snr_range_dB_val",
            "max_power_ratio_dB_val",
            "dist_range_val",
            "min_dist_src2src_val",
            "min_dist_src2rcv_val",
            "src_num_weights_val",
            "frequency_topology_val",
            "ch4_eligible_src_counts_val",
        }
        missing = sorted(required.difference(handle.keys()))
        if missing:
            raise AssertionError(f"缺少字段: {missing}")

        source_count = np.asarray(handle["src_count_all"], dtype=np.int32).reshape(-1)
        band = np.asarray(handle["band_mask_all"], dtype=np.float32).transpose(2, 1, 0)
        ignore = np.asarray(handle["ignore_mask_all"], dtype=np.float32).transpose(2, 1, 0)
        fc_offset = np.asarray(handle["fc_offset_all"], dtype=np.float64).T
        symbol_rate = np.asarray(handle["symbolRate_all"], dtype=np.float64).T
        bw_actual = np.asarray(handle["BW_actual_all"], dtype=np.float64).T
        iq_real = np.asarray(handle["sig_rcv_real_all"], dtype=np.float32)
        iq_imag = np.asarray(handle["sig_rcv_imag_all"], dtype=np.float32)
        sub_lo = np.asarray(handle["sub_f_lo_val"], dtype=np.float64).reshape(-1)
        sub_hi = np.asarray(handle["sub_f_hi_val"], dtype=np.float64).reshape(-1)
        threshold = _scalar(handle, "thresh_val")
        window_width = _scalar(handle, "B_win_val")

        profile = _matlab_text(handle, "gate_profile_val")
        label_profile = _matlab_text(handle, "band_label_profile_val")
        contract_version = _matlab_text(handle, "physical_contract_version_val")
        frequency_topology = _matlab_text(handle, "frequency_topology_val")

        if profile != expected_profile:
            raise AssertionError(f"gate profile错误: {profile!r}")
        if label_profile != EXPECTED_LABEL_PROFILE:
            raise AssertionError(f"标签profile错误: {label_profile!r}")
        if contract_version != EXPECTED_CONTRACT_VERSION:
            raise AssertionError(f"物理契约版本错误: {contract_version!r}")
        if frequency_topology != "connected_actual_overlap":
            raise AssertionError(f"频率拓扑错误: {frequency_topology!r}")
        if not np.isclose(threshold, EXPECTED_THRESHOLD, atol=1e-7):
            raise AssertionError(f"阈值错误: {threshold}")

        sample_count = source_count.size
        if band.shape != (sample_count, 3, 19) or ignore.shape != band.shape:
            raise AssertionError(f"标签shape错误: band={band.shape}, ignore={ignore.shape}")
        if fc_offset.shape != (sample_count, 3) or bw_actual.shape != fc_offset.shape:
            raise AssertionError("逐源频带参数shape错误。")
        if iq_real.shape != iq_imag.shape or iq_real.shape[-1] != sample_count:
            raise AssertionError(f"IQ shape错误: {iq_real.shape}, {iq_imag.shape}")
        if not np.isfinite(iq_real).all() or not np.isfinite(iq_imag).all():
            raise AssertionError("IQ包含NaN/Inf。")
        if not np.isin(band, [0.0, 1.0]).all() or not np.isin(ignore, [0.0, 1.0]).all():
            raise AssertionError("标签不是二值。")
        if np.any((band > 0) & (ignore > 0)):
            raise AssertionError("positive与ignore标签重叠。")

        recomputed_band, recomputed_ignore = _recompute_labels(
            fc_offset,
            bw_actual,
            source_count,
            sub_lo,
            sub_hi,
            window_width,
            threshold,
        )
        if not np.array_equal(band, recomputed_band):
            raise AssertionError("Hard-19-Actual正标签独立复算不一致。")
        if not np.array_equal(ignore, recomputed_ignore):
            raise AssertionError("Hard-19-Actual ignore标签独立复算不一致。")

        active = np.arange(3)[None, :] < source_count[:, None]
        if not np.allclose(bw_actual[active], symbol_rate[active] * EXPECTED_ALPHA_FACTOR):
            raise AssertionError("BW_actual != Rs*(1+1.2*alpha)，其中alpha=0.25。")
        if np.any(band.sum(axis=2)[active] < 1):
            raise AssertionError("至少一个有效信源没有正子带。")
        if np.any(band[~active]) or np.any(ignore[~active]):
            raise AssertionError("空源槽位存在标签。")
        if np.any(source_count == 0):
            zero_rows = source_count == 0
            if np.any(band[zero_rows]) or np.any(ignore[zero_rows]):
                raise AssertionError("K=0样本存在频带标签。")
        for row, count in zip(fc_offset, source_count, strict=True):
            if count > 1 and np.any(np.diff(row[:count]) < 0):
                raise AssertionError("有效源槽未按中心频率排序。")

        expected_contract = {
            "fs": 100e6,
            "b_win": 10e6,
            "b_step": 5e6,
            "symbol_rate_min": 2e6,
            "symbol_rate_max": 20e6,
            "alpha": 0.25,
            "max_power_ratio_db": 3.0,
            "min_dist_src2src": 150.0,
            "min_dist_src2rcv": 150.0,
        }
        observed_contract = {
            "fs": _scalar(handle, "fs_val"),
            "b_win": window_width,
            "b_step": _scalar(handle, "B_step_val"),
            "symbol_rate_min": _scalar(handle, "symbolRate_min_val"),
            "symbol_rate_max": _scalar(handle, "symbolRate_max_val"),
            "alpha": _scalar(handle, "arfa_val"),
            "max_power_ratio_db": _scalar(handle, "max_power_ratio_dB_val"),
            "min_dist_src2src": _scalar(handle, "min_dist_src2src_val"),
            "min_dist_src2rcv": _scalar(handle, "min_dist_src2rcv_val"),
        }
        if any(not np.isclose(observed_contract[key], value) for key, value in expected_contract.items()):
            raise AssertionError(f"K=2/3物理契约偏离G4-R4: {observed_contract}")
        if not np.allclose(np.asarray(handle["snr_range_dB_val"]).reshape(-1), [-10, 10]):
            raise AssertionError("SNR范围偏离G4-R4。")
        if not np.allclose(np.asarray(handle["dist_range_val"]).reshape(-1), [100, 1000]):
            raise AssertionError("空间距离范围偏离G4-R4。")
        if not np.array_equal(
            np.asarray(handle["ch4_eligible_src_counts_val"], dtype=np.int32).reshape(-1),
            [2, 3],
        ):
            raise AssertionError("CH4可消费源数契约错误。")

    count_histogram = dict(sorted(Counter(map(int, source_count)).items()))
    if set(count_histogram) != {0, 1, 2, 3} or len(set(count_histogram.values())) != 1:
        raise AssertionError(f"K=0/1/2/3未等量覆盖: {count_histogram}")
    routing = {str(count): route_source_count(count) for count in range(4)}
    try:
        locate_single_source()
    except NotImplementedError as exc:
        single_source_interface = str(exc)
    else:
        raise AssertionError("K=1定位接口不应在R1中具有实现。")

    return {
        "status": "PASS",
        "mat_path": str(path.resolve()),
        "sample_count": int(sample_count),
        "source_count_histogram": count_histogram,
        "profile": profile,
        "label_profile": label_profile,
        "physical_contract_version": contract_version,
        "threshold": threshold,
        "positive_slots": int(band.sum()),
        "ignore_slots": int(ignore.sum()),
        "routing": routing,
        "single_source_interface": single_source_interface,
        "ch4_retraining_required_for_k23": False,
        "ch4_compatibility_basis": "same conditional distribution and Hard-19-Actual contract as G4-R4",
    }


def audit_ch3_consumption(coarse_path: Path) -> dict[str, Any]:
    """验证统一数据可由CH3原Dataset、Transformer和loss直接消费。"""
    import torch

    from train_v26 import SourceDetectionDataset, SourceDetectionNet, compute_loss

    dataset = SourceDetectionDataset(
        str(coarse_path),
        augment=False,
        normalize="sample_zscore",
        max_src_override=10,
    )
    if len(dataset) < 2:
        raise AssertionError("CH3消费检查至少需要2个样本。")
    samples = [dataset[index] for index in range(2)]
    spectra = torch.stack([sample[0] for sample in samples])
    band_mask = torch.stack([sample[2] for sample in samples])
    ignore_mask = torch.stack([sample[3] for sample in samples])
    torch.manual_seed(20260827)
    model = SourceDetectionNet(n_sub=19, max_src=10, mode="transformer")
    model.eval()
    with torch.no_grad():
        logits = model(spectra)
        loss = compute_loss(logits, band_mask, ignore_mask)
    if logits.shape != (2, 10, 19) or not torch.isfinite(logits).all():
        raise AssertionError(f"CH3 forward输出无效: {tuple(logits.shape)}")
    if not torch.isfinite(loss):
        raise AssertionError("CH3 loss不是有限值。")
    return {
        "status": "PASS",
        "coarse_path": str(coarse_path.resolve()),
        "dataset_samples": len(dataset),
        "input_shape": list(spectra.shape),
        "logits_shape": list(logits.shape),
        "loss": float(loss.item()),
        "scope": "untrained forward/loss engineering check; not a performance result",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mat", type=Path, required=True, help="main31_loc_v3生成的MAT文件")
    parser.add_argument(
        "--expected-profile",
        default=EXPECTED_PROFILE,
        help="期望的MATLAB gate profile；默认s2g5r1",
    )
    parser.add_argument("--coarse", type=Path, help="可选：s2g3粗DPD MAT，用于CH3消费检查")
    parser.add_argument("--report", type=Path, required=True, help="独立JSON报告；拒绝覆盖")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.mat.is_file():
        raise FileNotFoundError(f"输入不存在: {args.mat}")
    if args.report.exists():
        raise FileExistsError(f"拒绝覆盖报告: {args.report}")
    report = audit_mat(args.mat, expected_profile=args.expected_profile)
    if args.coarse is not None:
        if not args.coarse.is_file():
            raise FileNotFoundError(f"粗DPD输入不存在: {args.coarse}")
        report["ch3_consumption"] = audit_ch3_consumption(args.coarse)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
