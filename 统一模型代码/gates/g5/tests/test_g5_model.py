"""K=0/1/2/3前向等价、定位梯度隔离及评价兼容性。"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_model import build_context, evaluate, forward, g4  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_prepare import configure, SOURCE  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402
from 统一模型代码.common.verified_feature_loader import decode_npy  # noqa: E402


def main(run):
    run = run.resolve(strict=True)
    manifest = g4.read_json(run / "manifest.json")
    configure(run, manifest["inputs"])
    old = g4.read_json(SOURCE / "manifest.json")
    dpd = g4.read_json(SOURCE / "dpd_cache_manifest.json")["files"]["train"]
    selected = [next((i,row) for i,row in enumerate(old["subsets"]["train"]) if row["true_k"] == k) for k in range(4)]
    device = torch.device("cuda:0")
    context = build_context(run, manifest, manifest["config"]["training_seeds"][0], device)
    coarse, fine, bands, ignores, positions, metadata = [], [], [], [], [], []
    with g4.g1.SampleStore("train") as store:
        for ordinal, row in selected:
            sample = store.sample(row)
            coarse.append(sample["coarse_dpd"])
            value = decode_npy(verified_read(dpd[ordinal], run / "anomalies"))
            fine.append(g4.g1.d8_input(torch.from_numpy(value.copy()))[0])
            bands.append(sample["band_truth"][:3])
            ignores.append(sample["ignore_truth"][:3])
            pos = torch.zeros((3,2))
            pos[:row["true_k"]] = torch.from_numpy(sample["positions_m"][:row["true_k"]])
            positions.append(pos)
            metadata.append({**row, "snr_db": float(store.coarse["avg_snr_all"][0,row["local_index"]]),
                             "min_source_distance_m": None})
    g4.set_mode(context, training=False)
    with torch.no_grad():
        x = torch.stack(coarse).to(device)
        spatial = context.ch3.backbone[:-1](x.reshape(4*19,1,81,81)).reshape(4,19,128,11,11)
        e1,d2 = g4.d8_prefix(context.d8,torch.stack(fine).to(device))
    features = g4.FeatureStore(spatial.cpu().numpy(),e1.cpu().numpy(),d2.cpu().numpy())
    targets = g4.Targets(torch.stack(bands),torch.stack(ignores),torch.stack(positions),torch.arange(4),
                         torch.tensor([row["frequency_overlap"] for _,row in selected]))
    indices = torch.arange(4)
    report = {"status":"RUNNING","test_executed":False,"samples":metadata}
    with torch.no_grad():
        a = forward(context,features,indices,device,stop_gradient=False)
        b = forward(context,features,indices,device,stop_gradient=True)
        old_output = g4.forward_indices(context,features,indices,device)
        report["sg_e2e_forward_exact"] = all(torch.equal(x,y) for x,y in zip(a,b))
        report["g4_forward_exact"] = all(torch.equal(x,y) for x,y in zip(a,old_output))
    groups = {"heads":list(context.ch3.band_heads[:3].parameters()),
              "query":list(context.query_builder.parameters()),
              "ch3_tail":list(context.ch3.cross_attn.parameters()),
              "splitter":list(context.splitter.parameters()),
              "d8_tail":list(itertools.chain(context.d8.decoder.c1.parameters(),context.d8.decoder.up0.parameters())),
              "head":list(context.source_head.parameters())}
    report["gradient"] = {}
    for track in ("sg","e2e"):
        _,logits,_,heatmap,offset = forward(context,features,indices,device,stop_gradient=track=="sg")
        _,parts,_ = g4.r1.compute_losses(logits,heatmap,offset,g4.as_cached_targets(targets),indices,manifest["config"])
        loss = parts["heatmap"]+parts["offset"]
        report["gradient"][track] = {}
        for name,parameters in groups.items():
            grads = torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
            norm = sum(float((value.detach()**2).sum()) for value in grads if value is not None)**.5
            report["gradient"][track][name] = norm
        del loss,parts,logits,heatmap,offset
    for name in ("heads","query","ch3_tail"):
        assert report["gradient"]["sg"][name] == 0,name
        assert np.isfinite(report["gradient"]["e2e"][name]) and report["gradient"]["e2e"][name]>0,name
    for track in ("sg","e2e"):
        assert all(report["gradient"][track][name]>0 for name in ("splitter","d8_tail","head"))
    reference = g4.evaluate(context,features,targets,device)
    current = evaluate(context,features,targets,device,metadata)
    assert reference["overall"] == {key:current["overall"][key] for key in reference["overall"]}
    assert reference["active_band_macro_f1"] == current["overall"]["band_f1"]
    assert reference["active_band_macro_iou"] == current["overall"]["band_iou"]
    for x,y in zip(reference["samples"],current["samples"]):
        assert all(y[key]==value for key,value in x.items())
    report["g4_evaluation_exact"] = True
    assert report["sg_e2e_forward_exact"] and report["g4_forward_exact"]
    report["status"] = "PASS"
    g4.write_json(run / "model_contract_report.json",report)
    print(report,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    main(parser.parse_args().run)
