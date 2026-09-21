"""Small frozen-checkpoint probes for G5 slot diagnosis, not a new performance Gate."""
from __future__ import annotations

import argparse
import io
import itertools
from pathlib import Path
import sys

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.diagnostics.g5_slot_diagnosis import read_json, write_json, identity, describe  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_features import build_models, g4  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402
from s2g3_composability import decode_d8_sample  # noqa: E402


def run_probes(out):
    source = read_json(out / "report.json")
    run = Path(source["source_run"])
    dest = out / "probes"
    dest.mkdir(exist_ok=False)
    inputs = {}

    def get(row):
        p = Path(row["path"]).resolve(strict=True)
        if not p.is_relative_to(run):
            raise ValueError("Not a local frozen run input")
        inputs[str(p)] = {k: row[k] for k in ("path", "size_bytes", "sha256")}
        return verified_read(row, dest / "anomalies")

    m = read_json(run / "manifest.json")
    files = read_json(run / "feature_manifest.json")["files"]["val_compare"]
    for row in m["inputs"]["artifacts"]:
        get(row)
    targets = torch.load(io.BytesIO(get(files["targets"])), map_location="cpu", weights_only=False)
    pairs = []
    for seed in m["config"]["training_seeds"]:
        rows = read_json(run / "comparison" / f"{seed}_e2e.json")["evaluation"]["samples"]
        for r in rows:
            if r["true_count"] < 2:
                continue
            i = r["index"]
            band = targets["band"][i].numpy() > .5
            valid = targets["ignore"][i].numpy() < .5
            same = False
            for a, b in itertools.combinations(range(r["true_count"]), 2):
                common = valid[a] & valid[b]
                if np.any(band[a, common]) and np.array_equal(band[a, common], band[b, common]):
                    same = True
            pairs.append((r, same))
    group = {str(k): {"same_evaluable_band_pair": describe([r for r, same in pairs if r["true_count"] == k and same]),
                     "no_same_evaluable_band_pair": describe([r for r, same in pairs if r["true_count"] == k and not same])}
             for k in (2, 3)}
    values = []
    for name in ("d8_e1", "d8_d2"):
        selected = []
        for case in source["cases"]:
            i = case["ordinal"]
            row = next(r for r in files[name] if r["start"] <= i < r["stop"])
            array = np.load(io.BytesIO(get(row)), allow_pickle=False)
            selected.append(array[i-row["start"]].copy())
        values.append(torch.from_numpy(np.stack(selected)).cuda())
    g4.set_deterministic(20260921)
    ch3, d8 = build_models(dest, m, torch.device("cuda:0"))
    del ch3
    with torch.no_grad():
        e1, d2 = values
        d1 = d8.decoder.c1(torch.cat([d8.decoder.up1(d2), e1], 1))
        d0 = d8.decoder.up0(d1)[:, :, :401, :401]
        heat, offset = d8.decoder.head(d0), d8.decoder.offset_head(d0)
    records = []
    heat = heat.cpu()
    offset = offset.cpu()
    for n, case in enumerate(source["cases"]):
        data = np.load(out / f"{case['label']}.npz")
        truth = data["truth"]
        count = case["e2e"]["predicted_count"]
        positions, scores = decode_d8_sample(heat[n], offset[n], count)
        # Eight local maxima per active query. Select one per query, prevent pairs
        # within 30 m, maximize summed log heatmap probability. No ground truth.
        prob = data["e2e_heatmap"]
        off = data["e2e_offset"]
        active = np.flatnonzero(data["e2e_logits"].max(-1) >= 0)
        candidates = []
        for q in active:
            peaks = np.where(prob[q] == maximum_filter(prob[q], size=7, mode="constant"), prob[q], 0)
            idx = np.argsort(peaks.ravel())[-8:][::-1]
            current = []
            for flat in idx:
                iy, ix = np.unravel_index(flat, peaks.shape)
                dx, dy = off[q, :, iy, ix].clip(-1, 1)
                current.append((np.array([(ix+float(dx))*10-2000, (iy+float(dy))*10-2000]), float(peaks[iy, ix])))
            candidates.append(current)
        best = None
        for indices in itertools.product(range(8), repeat=len(active)):
            points = [candidates[q][p][0] for q, p in enumerate(indices)]
            if any(np.linalg.norm(a-b) < 30 for a, b in itertools.combinations(points, 2)):
                continue
            score = sum(np.log(max(candidates[q][p][1], 1e-20)) for q, p in enumerate(indices))
            if best is None or score > best[0]:
                best = (score, points, indices)
        joint = np.asarray(best[1]) if best else data["e2e_positions"]
        max_near_truth = []
        yy, xx = np.meshgrid(np.arange(401)*10-2000, np.arange(401)*10-2000, indexing="ij")
        for q in active:
            max_near_truth.append([float(prob[q][(xx-p[0])**2+(yy-p[1])**2 <= 30**2].max()/prob[q].max()) for p in truth])
        spatial_band = {}
        for name, pred in (("original_e2e", data["e2e_positions"]), ("joint_decode", joint)):
            ti, pi = linear_sum_assignment(np.linalg.norm(truth[:, None, :]-pred[None, :, :], axis=-1))
            f1 = []
            for t, p in zip(ti, pi):
                valid = data["ignore"][t] < .5
                a = data["true_band"][t, valid] > .5
                b = data["e2e_logits"][active[p], valid] >= 0
                f1.append(float(2*np.sum(a & b)/max(np.sum(a)+np.sum(b), 1)))
            spatial_band[name] = f1
        record = {"label": case["label"], "raw_index": case["raw_index"],
                  "original_d8_fullband_errors_m": g4.distance_errors(truth, positions),
                  "original_d8_fullband_positions": positions.tolist(),
                  "original_d8_fullband_tp100": g4.g1.maximum_matches_within(truth, positions, 100),
                  "joint_decode_errors_m": g4.distance_errors(truth, joint),
                  "joint_decode_tp100": g4.g1.maximum_matches_within(truth, joint, 100),
                  "joint_decode_positions": joint.tolist(),
                  "joint_decode_candidate_ranks": [int(i)+1 for i in best[2]] if best else None,
                  "e2e_peak_within_30m_of_truth_over_global_peak": max_near_truth,
                  "band_f1_using_spatial_assignment_diagnostic_only": spatial_band}
        records.append(record)
        np.savez_compressed(dest / f"{case['label']}.npz", original_heatmap=heat[n].sigmoid().numpy()[0],
                            original_positions=positions, joint_positions=joint,
                            original_offset=offset[n].numpy(), e2e_truth_peak_ratio=np.asarray(max_near_truth))
        print(record, flush=True)
        data.close()
    for p, row in inputs.items():
        if identity(p) != row:
            raise RuntimeError("Postcheck mismatch")
    write_json(dest / "report.json", {"status": "PASS", "inputs": list(inputs.values()), "test_executed": False,
               "diagnostic_script": identity(__file__),
               "training_executed": False, "cases": records, "band_groups": group,
               "scope": "4 selected illustrative cases; original D8 uses same fullband frozen prefix, own pretrained decoder and head, K from E2E prediction; all selected K correct. Joint decode is post-hoc feasibility probe only, no deployed changes and no validation-wide gain claim.",
               "joint_decode_protocol": "7-pixel local maximum, eight candidates per query, 30m exclusion, maximize sum log probability without ground truth"})
    plot_probes(out)


def plot_probes(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    plt.style.use("C:/Users/Administrator/.codex/skills/scientific-toolkit-skill/references/scientific-skills/scientific-visualization/assets/publication.mplstyle")
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"], "axes.unicode_minus": False,
                         "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8, "svg.fonttype": "none"})
    records = read_json(out / "probes/report.json")["cases"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
    for n, case in enumerate(records[:2]):
        with np.load(out / f"{case['label']}.npz") as data, np.load(out / "probes" / f"{case['label']}.npz") as probe:
            truth = data["truth"]
            im = axes[0, n].imshow(probe["original_heatmap"], origin="lower", extent=[-2005, 2005, -2005, 2005], cmap="viridis", vmin=0)
            fig.colorbar(im, ax=axes[0, n], label="原D8 sigmoid输出")
            p = probe["original_positions"]
            axes[0, n].scatter(p[:, 0], p[:, 1], color="#F0E442", marker="x", s=80)
            axes[0, n].set_title(f"raw={case['raw_index']}：同一全频DPD → 原D8头")
            ax = axes[1, n]
            for key, marker, color, name in (("e2e_positions", "x", "#D55E00", "原逐槽位独立最大峰"),
                                             ("joint_positions", "+", "#0072B2", "候选联合分配（仅诊断）")):
                p = data[key] if key in data else probe[key]
                ax.scatter(p[:, 0], p[:, 1], marker=marker, color=color, s=95, label=name)
                # Only label the slot that switches to a secondary peak, avoiding
                # overlapping labels where the other positions remain unchanged.
                pt = p[2]
                ax.annotate("Q3", pt, xytext=(-42, -28) if key == "e2e_positions" else (25, 32),
                            textcoords="offset points", color=color,
                            arrowprops={"arrowstyle": "-", "color": color, "linewidth": .8})
            ax.set_title(f"相同E2E热图，改成候选联合分配；命中{case['joint_decode_tp100']}/3")
            ax.legend(loc="lower left")
            for row in range(2):
                ax = axes[row, n]
                for i, pt in enumerate(truth):
                    ax.scatter(*pt, marker=["o", "s", "^"][i], facecolors="none", edgecolors="black" if row else "white", s=100)
                    ax.annotate(f"T{i+1}", pt, xytext=(9, 9), textcoords="offset points", color="black" if row else "white",
                                path_effects=[] if row else [pe.withStroke(linewidth=2, foreground="black")])
                ax.set(xlim=(-1100, 1100), ylim=(-1100, 1100), xlabel="x (m)", ylabel="y (m)")
                ax.set_aspect("equal")
    fig.suptitle("两例失败场景的隔离检查：空间信息仍在，不只是全频输入缺少可定位信息\n不是全开发集对照，不替换正式G5结果；原D8使用E2E预测K（这两例均正确）", fontsize=12)
    for suffix in ("png", "svg"):
        fig.savefig(out / "figures" / f"two_failure_probes.{suffix}", dpi=300)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), layout="constrained")
    for ax, case in zip(axes, records):
        a = np.asarray(case["e2e_peak_within_30m_of_truth_over_global_peak"])
        im = ax.imshow(a, cmap="cividis", vmin=0, vmax=1)
        for q in range(3):
            for t in range(3):
                ax.text(t, q, f"{a[q,t]:.2f}", ha="center", va="center", color="black" if a[q,t]>.5 else "white")
        ax.set_xticks(range(3), ["T1", "T2", "T3"])
        ax.set_yticks(range(3), ["Q1", "Q2", "Q3"])
        ax.set_title(f"{case['label'][0]}：raw={case['raw_index']}")
    fig.colorbar(im, ax=axes.tolist(), shrink=.75, label="真源30 m邻域峰值 / 本槽位全局峰值")
    fig.suptitle("失败例中同一列被多个槽位追逐；成功例也可有很强的次峰，因此高相似度本身不是失败判据", fontsize=11)
    for suffix in ("png", "svg"):
        fig.savefig(out / "figures" / f"query_source_peak_ratios.{suffix}", dpi=300)
    plt.close(fig)
    write_json(out / "probes/figure_manifest.json", {"files": [identity(p) for p in sorted((out / "figures").glob("*"))],
                                                     "script": identity(__file__)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run_probes(parser.parse_args().out)
