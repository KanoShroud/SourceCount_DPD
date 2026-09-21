"""G5 post-hoc development-only slot diagnosis; never trains or changes frozen evidence."""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_model import build_context, forward, g4  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_prepare import configure  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def cosine(array):
    flat = np.asarray(array, dtype=np.float64).reshape(len(array), -1)
    normalized = flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-20)
    return normalized @ normalized.T


def duplicate(row, radius=30):
    pos = np.asarray(row["predicted_positions_m"])
    return any(np.linalg.norm(a-b) < radius for i, a in enumerate(pos) for b in pos[i+1:])


def severe(row):
    return max(row["matched_errors_m"], default=0) > 500


def describe(rows):
    errors = [v for r in rows for v in r["matched_errors_m"]]
    true = sum(r["true_count"] for r in rows)
    return {"sample_seed_instances": len(rows), "tail500_instances": sum(map(severe, rows)),
            "duplicate30_instances": sum(map(duplicate, rows)),
            "gospa_m": float(np.mean([r["gospa_m"] for r in rows])) if rows else None,
            "recall100": sum(r["tp_at_100m"] for r in rows)/true if true else None,
            "matched_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if errors else None}


def select_cases(e2e, sg):
    first = next(r for r in e2e if r["raw_index"] == 30)
    low = [r for r in e2e if r["true_count"] == r["predicted_count"] == 3
           and r["snr_db"] < -10 and severe(r) and duplicate(r)]
    low.sort(key=lambda r: (max(r["matched_errors_m"]), r["raw_index"]))
    good = [r for r in e2e if r["true_count"] == r["predicted_count"] == r["tp_at_100m"] == 3
            and r["snr_db"] >= 0]
    good.sort(key=lambda r: (abs(r["snr_db"]-first["snr_db"]), r["raw_index"]))
    rescue = [r for r in e2e if r["true_count"] == r["predicted_count"] == r["tp_at_100m"] == 3
              and sg[r["index"]]["tp_at_100m"] < 3 and severe(sg[r["index"]])
              and r["raw_index"] != good[0]["raw_index"]]
    rescue.sort(key=lambda r: (abs(r["snr_db"]-first["snr_db"]), r["raw_index"]))
    return [("A_high_snr_duplicate", first), ("B_low_snr_duplicate", low[len(low)//2]),
            ("C_high_snr_success", good[0]), ("D_feedback_rescue", rescue[0])]


def decode(logits, heat, offset):
    active = np.flatnonzero(logits.max(-1) >= 0)
    positions = []
    for q in active:
        iy, ix = np.unravel_index(heat[q].argmax(), heat[q].shape)
        delta = offset[q, :, iy, ix].clip(-1, 1)
        positions.append([(ix+float(delta[0]))*10-2000, (iy+float(delta[1]))*10-2000])
    return np.asarray(positions).reshape(-1, 2), active


def extract(run, out):
    run = run.resolve(strict=True)
    out = out.resolve()
    if not out.is_relative_to(ROOT / "outputs_e2e") or out.is_relative_to(run) or run.is_relative_to(out):
        raise ValueError("Diagnostic output must be isolated from original run and reference inputs")
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    report = {"status": "RUNNING", "test_executed": False, "training_executed": False,
              "source_run": str(run), "seed_for_examples": 20260921,
              "selection": "A=previously discussed raw30; B=median severe error among correct-K low-SNR duplicate K3; C=successful K3 closest SNR to A; D=E2E full recall and SG severe failure closest SNR to A excluding C",
              "limits": "post-hoc descriptive development diagnosis; examples are not random; repeated seeds are not independent scenes",
              "inputs": [], "cases": []}
    write_json(out / "report.json", report)
    registry = {}

    def registered_bytes(row):
        path = Path(row["path"]).resolve(strict=True)
        if not path.is_relative_to(run):
            raise ValueError(f"Expected frozen local G5 snapshot: {path}")
        if path.stat().st_size != row["size_bytes"]:
            raise ValueError("Input size mismatch")
        registry[str(path)] = {k: row[k] for k in ("path", "sha256", "size_bytes")}
        return verified_read(row, out / "anomalies")

    m = read_json(run / "manifest.json")
    fm = read_json(run / "feature_manifest.json")
    contract = read_json(run / "engineering_v4/contract.json")
    for row in contract["files"]:
        verified_read(row, out / "anomalies")
    report["frozen_contract_files_checked"] = len(contract["files"])
    report["diagnostic_script"] = identity(__file__)
    for name in ("manifest.json", "feature_manifest.json", "comparison_report.json", "hard_reference_report.json",
                 "training_report.json", "final_audit_report.json"):
        registry[str(run / name)] = identity(run / name)
    if read_json(run / "final_audit_report.json")["status"] != "PASS" or m["test_executed"]:
        raise RuntimeError("Original G5 audit not valid")
    all_rows = []
    results = {}
    for seed in m["config"]["training_seeds"]:
        for track in ("sg", "e2e"):
            path = run / "comparison" / f"{seed}_{track}.json"
            registry[str(path)] = identity(path)
            results[seed, track] = read_json(path)["evaluation"]["samples"]
        all_rows.extend(results[seed, "e2e"])
    e2e, sg = results[20260921, "e2e"], results[20260921, "sg"]
    cases = select_cases(e2e, sg)
    tail = [r for r in all_rows if severe(r)]
    report["population"] = {
        "by_k": {str(k): describe([r for r in all_rows if r["true_count"] == k]) for k in range(4)},
        "tail_instances": len(tail), "tail_correct_count": sum(r["true_count"] == r["predicted_count"] for r in tail),
        "tail_duplicate30": sum(map(duplicate, tail)),
        "k23_duplicate": describe([r for r in all_rows if r["true_count"] >= 2 and duplicate(r)]),
        "k23_not_duplicate": describe([r for r in all_rows if r["true_count"] >= 2 and not duplicate(r)]),
        "k3_snr_below_minus10": describe([r for r in all_rows if r["true_count"] == 3 and r["snr_db"] < -10]),
        "k3_snr_at_least5": describe([r for r in all_rows if r["true_count"] == 3 and r["snr_db"] >= 5]),
        "overlap_values_by_k": {str(k): sorted(set(r["frequency_overlap"] for r in all_rows if r["true_count"] == k)) for k in range(4)}}
    indices = [r["index"] for _, r in cases]
    files = fm["files"]["val_compare"]
    for row in m["inputs"]["artifacts"]:
        registered_bytes(row)
    # Validate only relevant snapshot files, never the training or test data.
    for row in m["inputs"]["files"]:
        if Path(row["path"]).name.startswith(("files_02_", "files_03_")):
            registered_bytes(row)
    configure(out, m["inputs"])
    sample_arrays = []
    with g4.g1.SampleStore("val_compare") as store:
        lo, hi = store.subband_edges()
        for label, row in cases:
            record = m["subsets"]["val_compare"][row["index"]]
            if record["raw_index"] != row["raw_index"]:
                raise RuntimeError("Sample identity mapping changed")
            sample = store.sample(record)
            fine_row = files["dpd"][row["index"]]
            fine = np.load(io.BytesIO(registered_bytes(fine_row)), allow_pickle=False)
            spectrum = np.fft.fftshift(np.fft.fft(sample["signal"], n=g4.g1.N_FFT, axis=-1), axes=-1)
            power = np.mean(np.abs(spectrum)**2, axis=0)
            sample_arrays.append({"coarse": sample["coarse_dpd"].numpy(), "fine": fine,
                                  "truth": sample["positions_m"][:row["true_count"]],
                                  "true_band": sample["band_truth"].numpy()[:3],
                                  "ignore": sample["ignore_truth"].numpy()[:3],
                                  "frequency_mhz": np.fft.fftshift(np.fft.fftfreq(g4.g1.N_FFT, 1/g4.g1.FS))/1e6,
                                  "received_power_db": 10*np.log10(np.maximum(power/power.max(), 1e-15)),
                                  "band_lo_mhz": lo/1e6, "band_hi_mhz": hi/1e6})
            print(f"Read case {label}: raw={row['raw_index']}, SNR={row['snr_db']:.2f}", flush=True)
    values = []
    for name in ("ch3_spatial", "d8_e1", "d8_d2"):
        selected = []
        for i in indices:
            row = next(r for r in files[name] if r["start"] <= i < r["stop"])
            arr = np.load(io.BytesIO(registered_bytes(row)), allow_pickle=False)
            selected.append(arr[i-row["start"]].copy())
        values.append(np.stack(selected))
    features = g4.FeatureStore(*values)
    device = torch.device("cuda:0")
    g4.set_deterministic(20260921)
    for track in ("e2e", "sg"):
        context = build_context(out, m, 20260921, device)
        cp_row = read_json(run / "training/20260921" / track / "best.identity.json")
        cp = torch.load(io.BytesIO(registered_bytes(cp_row)), map_location="cpu", weights_only=False)
        g4.load_state(context, cp["state"])
        del cp
        g4.set_mode(context, training=False)
        captured = {}

        def capture_anchor(module, args, value):
            captured["anchor"] = value.detach().cpu().numpy()

        def capture_splitter(module, args, value):
            spatial, query, logits = args
            weights = torch.sigmoid(logits)
            weighted = torch.einsum("bqs,bschw->bqchw", weights, spatial) / weights.sum(-1)[..., None, None, None].clamp_min(1e-6)
            captured["weighted"] = weighted.detach().cpu().numpy()
            captured["source_spatial"] = value[0].detach().cpu().numpy()

        def capture_head(module, args, value):
            captured["d0_rms"] = args[0].square().mean(1).sqrt().cpu().numpy()
            captured["film"] = module.film(args[2]).cpu().numpy()

        hooks = [context.query_builder.anchor.register_forward_hook(capture_anchor),
                 context.splitter.register_forward_hook(capture_splitter),
                 context.source_head.register_forward_hook(capture_head)]
        with torch.no_grad():
            query, logits, attention, heatmap, offset = forward(context, features, torch.arange(len(indices)), device, stop_gradient=False)
        arrays = {"query": query.cpu().numpy(), "logits": logits.cpu().numpy(),
                  "attention": attention.cpu().numpy(), "heatmap": heatmap.sigmoid().cpu().numpy(),
                  "offset": offset.cpu().numpy(), **captured}
        for n, (label, row) in enumerate(cases):
            reference = results[20260921, track][row["index"]]
            pos, active = decode(arrays["logits"][n], arrays["heatmap"][n], arrays["offset"][n])
            max_delta = float(np.max(np.abs(pos - np.asarray(reference["predicted_positions_m"]))))
            logit_delta = float(np.max(np.abs(arrays["logits"][n] - np.asarray(reference["band_logits"]))))
            if max_delta > .05 or logit_delta > .001:
                raise RuntimeError(f"Replay mismatch {label}/{track}: {max_delta}, {logit_delta}")
            for key, value in arrays.items():
                sample_arrays[n][track+"_"+key] = value[n]
            sample_arrays[n][track+"_positions"] = pos
            if track == "e2e":
                a = arrays["attention"][n].reshape(3, -1)
                pairs = [(int(x), int(y)) for z, x in enumerate(active) for y in active[z+1:]
                         if np.linalg.norm(pos[z]-pos[list(active).index(y)]) < 30]
                report["cases"].append({"label": label, "raw_index": row["raw_index"], "ordinal": row["index"],
                    "snr_db": row["snr_db"], "true_k": row["true_count"], "e2e": row, "sg": results[20260921, "sg"][row["index"]],
                    "duplicate_query_pairs": pairs, "query_cosine": cosine(arrays["query"][n]).tolist(),
                    "anchor_cosine": cosine(arrays["anchor"][n]).tolist(),
                    "weighted_cosine": cosine(arrays["weighted"][n]).tolist(),
                    "source_spatial_cosine": cosine(arrays["source_spatial"][n]).tolist(),
                    "heatmap_cosine": cosine(arrays["heatmap"][n]).tolist(),
                    "attention_normalized_entropy": (-(a*np.log(np.maximum(a, 1e-30))).sum(-1)/np.log(a.shape[-1])).tolist(),
                    "attention_min": a.min(-1).tolist(), "attention_max": a.max(-1).tolist(),
                    "true_band_cosine": cosine(sample_arrays[n]["true_band"]).tolist(),
                    "replay": {}})
            report["cases"][n]["replay"][track] = {"max_position_delta_m": max_delta, "max_band_logit_delta": logit_delta,
                                                     "checkpoint": cp_row}
        for handle in hooks:
            handle.remove()
        del context, arrays, query, logits, attention, heatmap, offset
        captured.clear()
        gc.collect()
        torch.cuda.empty_cache()
    for (label, _), arrays in zip(cases, sample_arrays):
        np.savez_compressed(out / f"{label}.npz", **arrays)
    # Recheck all actually consumed inputs; use no stage outputs as replacements.
    for path, row in registry.items():
        if identity(path) != {k: row[k] for k in ("path", "sha256", "size_bytes")}:
            raise RuntimeError(f"Postcheck mismatch: {path}")
    report["inputs"] = list(registry.values())
    report["status"] = "PASS"
    report["anomaly_files"] = len(list((out / "anomalies").glob("*")))
    write_json(out / "report.json", report)
    print("Extraction and frozen-input checks PASS", flush=True)


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    style = Path("C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle")
    plt.style.use(style)
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "legend.fontsize": 9, "svg.fonttype": "none"})
    colors = ["#0072B2", "#D55E00", "#CC79A7"]
    marks = ["o", "s", "^"]
    report = read_json(out / "report.json")
    images = out / "figures"
    images.mkdir(exist_ok=False)

    def save(fig, name):
        fig.savefig(images / f"{name}.png", dpi=300)
        fig.savefig(images / f"{name}.svg")
        plt.close(fig)

    def ground(ax, truth):
        for i, p in enumerate(truth):
            ax.scatter(*p, marker=marks[i], s=85, facecolors="none", edgecolors="white", linewidths=2,
                       path_effects=[pe.withStroke(linewidth=3, foreground="black")])
            ax.annotate(f"T{i+1}", p, xytext=(5, 5), textcoords="offset points", color="white", weight="bold",
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])

    def spatial(ax, image, title, truth, *, feature=False, vmax=None, vmin=None):
        extent = [-.5, 10.5, -.5, 10.5] if feature else [-2005, 2005, -2005, 2005] if image.shape[-1] == 401 else [-2025, 2025, -2025, 2025]
        im = ax.imshow(image, origin="lower", extent=extent, cmap="viridis", aspect="equal", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(title)
        if feature:
            ax.set(xlabel="特征列索引", ylabel="特征行索引")
        else:
            ground(ax, truth)
            ax.set(xlim=(-1200, 1200), ylim=(-1200, 1200), xlabel="x (m)", ylabel="y (m)")
        return im

    for case in report["cases"]:
        label = case["label"]
        with np.load(out / f"{label}.npz") as data:
            truth = data["truth"]
            fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
            ax = axes[0, 0]
            for i, p in enumerate(truth):
                ax.scatter(*p, marker=marks[i], s=90, facecolors="none", edgecolors=colors[i], label=f"真源T{i+1}")
                ax.annotate(f"T{i+1}", p, xytext=(7, 7), textcoords="offset points")
            for name, marker, color in (("e2e", "x", "#000000"), ("sg", "+", "#E69F00")):
                p = data[name+"_positions"]
                ax.scatter(p[:, 0], p[:, 1], marker=marker, s=65, color=color, label=name.upper())
            ax.scatter([500, 0, -500, 0], [0, 500, 0, -500], marker="v", s=25, color="gray", label="接收站")
            ax.set(xlim=(-1200, 1200), ylim=(-1200, 1200), xlabel="x (m)", ylabel="y (m)", title="A 真实位置与SG/E2E输出")
            ax.set_aspect("equal")
            ax.legend(loc="upper left", fontsize=8)
            im = spatial(axes[0, 1], data["coarse"].mean(0), "B 粗DPD：19子带输入均值", truth)
            fig.colorbar(im, ax=axes[0, 1], label="log1p后整体标准化值")
            fine = data["fine"]
            im = spatial(axes[0, 2], (fine-fine.mean())/fine.std(), "C 固定全频细DPD", truth)
            fig.colorbar(im, ax=axes[0, 2], label="全图z-score")
            axes[1, 0].plot(data["frequency_mhz"], data["received_power_db"], color="#0072B2", linewidth=.65)
            axes[1, 0].set(xlabel="基带频率 (MHz)", ylabel="四站平均功率 / 峰值 (dB)", title="D 接收信号FFT功率谱（含噪声）", xlim=(-50, 50))
            probs = 1/(1+np.exp(-data["e2e_logits"]))
            valid = data["ignore"] < .5
            cost = np.array([[-np.mean(t[v]*np.log(p[v]+1e-12)+(1-t[v])*np.log(1-p[v]+1e-12))
                              for t, v in zip(data["true_band"], valid)] for p in probs])
            q, s = linear_sum_assignment(cost)
            order = [q[list(s).index(i)] for i in range(3)]
            band = np.concatenate([data["true_band"], probs[order]])
            band = np.ma.array(band, mask=np.concatenate([~valid, ~valid]))
            band_cmap = plt.get_cmap("cividis").copy()
            band_cmap.set_bad("#b0b0b0")
            im = axes[1, 1].imshow(band, vmin=0, vmax=1, aspect="auto", cmap=band_cmap, extent=[.5, 19.5, 5.5, -.5])
            axes[1, 1].set_yticks(range(6), [f"真值T{i+1}" for i in range(3)] + [f"Q{j+1}→T{i+1}" for i, j in enumerate(order)])
            axes[1, 1].set(xlabel="10 MHz子带序号（步进5 MHz；灰=忽略）", title="E 真值频带与Soft预测（仅按频带匹配）")
            fig.colorbar(im, ax=axes[1, 1], label="真值0/1或预测概率")
            im = spatial(axes[1, 2], data["e2e_d0_rms"], "F D8共享特征：通道RMS", truth)
            fig.colorbar(im, ax=axes[1, 2], label="特征RMS（不是概率）")
            fig.suptitle(f"{label} | raw={case['raw_index']} | SNR={case['snr_db']:.2f} dB\n空间面板仅显示中心±1200 m；原输入范围±2000 m", fontsize=13)
            save(fig, label+"_inputs")
            fig, axes = plt.subplots(4, 3, figsize=(12, 14), layout="constrained")
            weighted = np.sqrt(np.mean(data["e2e_weighted"]**2, axis=1))
            heatmax = max(data["e2e_heatmap"].max(), data["sg_heatmap"].max())
            for q in range(3):
                im = spatial(axes[0, q], weighted[q], f"Q{q+1} Soft频带汇聚粗特征RMS", truth, feature=True, vmax=weighted.max(), vmin=0)
                fig.colorbar(im, ax=axes[0, q], label="RMS")
                im = spatial(axes[1, q], data["e2e_attention"][q], f"Q{q+1} 空间attention（和为1）", truth, feature=True,
                             vmax=data["e2e_attention"].max(), vmin=0)
                fig.colorbar(im, ax=axes[1, q], label="attention")
                for row, track in ((2, "e2e"), (3, "sg")):
                    im = spatial(axes[row, q], data[track+"_heatmap"][q], f"{track.upper()} Q{q+1} 定位热图", truth, vmax=heatmax, vmin=0)
                    iy, ix = np.unravel_index(data[track+"_heatmap"][q].argmax(), (401, 401))
                    axes[row, q].scatter(ix*10-2000, iy*10-2000, marker="x", color="#F0E442", s=90)
                    fig.colorbar(im, ax=axes[row, q], label="sigmoid输出")
            fig.suptitle(f"{label} | 从粗特征到最终定位\n前三行属于E2E；最后一行是SG对照；SG与E2E的槽位编号不保证对应同一真实源", fontsize=12)
            save(fig, label+"_flow")
            if label.startswith("A_"):
                fig, axes = plt.subplots(4, 5, figsize=(15, 12), layout="constrained")
                low, high = data["coarse"].min(), data["coarse"].max()
                for b, ax in enumerate(axes.flat):
                    if b == 19:
                        ax.axis("off")
                        continue
                    im = spatial(ax, data["coarse"][b], f"W{b+1}: {data['band_lo_mhz'][b]:g}~{data['band_hi_mhz'][b]:g} MHz", truth, vmin=low, vmax=high)
                fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7, label="log1p后整体标准化值（19图共用色标）")
                fig.suptitle("A：19子带粗DPD；参照论文按W1—W19排列，真实源标记保持一致", fontsize=13)
                save(fig, "A_coarse_19_subbands")
    # Descriptive counts, not inferential significance tests.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    by_k = report["population"]["by_k"]
    rates = [100*by_k[str(k)]["tail500_instances"]/by_k[str(k)]["sample_seed_instances"] for k in (1, 2, 3)]
    axes[0].bar(["1源", "2源", "3源"], rates, color=colors)
    axes[0].set(ylabel="至少一个匹配误差>500 m的实例 (%)", ylim=(0, 100), title="A 多源严重长尾")
    for i, value in enumerate(rates):
        axes[0].text(i, value+2, f"{value:.1f}%", ha="center")
    pop = report["population"]
    vals = [100*pop["tail_correct_count"]/pop["tail_instances"], 100*pop["tail_duplicate30"]/pop["tail_instances"]]
    axes[1].bar(["源数正确", "两个预测相距<30 m"], vals, color=["#0072B2", "#D55E00"])
    axes[1].set(ylabel="严重长尾实例中的比例 (%)", ylim=(0, 110), title=f"B 严重长尾：{pop['tail_instances']}个样本×seed实例")
    for i, value in enumerate(vals):
        axes[1].text(i, value+2, f"{value:.1f}%", ha="center")
    fig.suptitle("开发集事后描述：同一场景跨seed重复出现，不视为独立样本；无显著性检验", fontsize=11)
    save(fig, "population_tail_summary")
    write_json(out / "figure_manifest.json", {"files": [identity(p) for p in sorted(images.glob("*"))],
                                             "data": identity(out / "report.json"), "script": identity(__file__)})
    print(f"Saved {len(list(images.glob('*.png')))} PNG figures with SVG companions", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if not args.plot_only:
        extract(args.run, args.out)
    plot(args.out)
