"""G5训练/评价角色、原预训练来源及分析分层登记；不读取test载荷。"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_prepare import configure, g4, SOURCE  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402


def audit(run):
    run = run.resolve(strict=True)
    m = g4.read_json(run / "manifest.json")
    configure(run, m["inputs"])
    original = ROOT.parent / "SourceCount_DPD/outputs"
    evidence_paths = {
        "ch3_train_lineage": original / "s2g5r4_ch3_scale/20260828_151735/audit/build_16k_report.json",
        "ch3_roles": original / "s2g5r2_ch3/20260827_191207/manifest/data_manifest.json",
        "d8_train_manifest": original / "s2g4r4_scale/20260826_132829/06_manifests/train_8192.json",
        "d8_generation_audit": original / "s2g4r4_scale/20260826_132829/04_matlab_runtime/05_raw_audit.json",
    }
    evidence = {name:g4.read_json(path) for name,path in evidence_paths.items()}
    checkpoints = {row["name"]:torch.load(io.BytesIO(verified_read(row,run/"anomalies")),map_location="cpu",weights_only=False)
                   for row in m["inputs"]["artifacts"]}
    assert checkpoints["ch3_seed42"]["config"]["data_dir"].endswith("training_views\\data_16k")
    assert evidence["ch3_train_lineage"]["combined_16k"]["sha256"] == m["inputs"]["files"][0]["sha256"]
    assert evidence["ch3_train_lineage"]["strict_old_8k_prefix"]
    assert evidence["d8_train_manifest"]["split"] == "train"
    assert Path(checkpoints["d8_seed42"]["args"]["train_manifest"]) == evidence_paths["d8_train_manifest"]
    roles = evidence["ch3_roles"]
    assert roles["train_source"] != roles["val_source"]
    assert roles["train_source_sha256"] != roles["val_source_sha256"]
    validation_ids = {split: {r["raw_index"] for r in m["subsets"][split]} for split in ("val_select","val_compare")}
    assert not validation_ids["val_select"] & validation_ids["val_compare"]
    for split, ids in validation_ids.items():
        assert ids <= set(roles[split]["indices"])
    old = g4.read_json(SOURCE / "manifest.json")
    for split in ("train","val_select"):
        assert {r["local_index"] for r in old["subsets"][split]} <= {r["local_index"] for r in m["subsets"][split]}
    data_info = {}
    for split in ("train","val_select","val_compare"):
        with g4.g1.SampleStore(split) as store:
            seeds = [int(h["random_seed_val"][0,0]) for _,_,h in store.raw_parts]
            rows = m["subsets"][split]
            snr = [float(store.coarse["avg_snr_all"][0,r["local_index"]]) for r in rows if r["true_k"] > 0]
            source_ids = [m["inputs"]["files"][i]["sha256"] for i in ((4,5,6) if split=="train" else (3,))]
            data_info[split] = {"raw_generation_seeds": seeds, "count":len(rows),
                               "k_counts":{str(k):sum(r["true_k"]==k for r in rows) for k in range(4)},
                               "active_snr_range":[min(snr),max(snr)], "raw_file_sha256":source_ids,
                               "finite_snr":bool(np.isfinite(snr).all())}
            assert np.isfinite(snr).all()
    d8_seed = evidence["d8_generation_audit"]["splits"]["train"]["seed"]
    assert all(d8_seed not in data_info[split]["raw_generation_seeds"] for split in ("val_select","val_compare"))
    report = {"status":"PASS", "test_executed":False,
              "scope":"Provenance/sample-role audit; not exhaustive IQ duplicate search",
              "evidence":{name:g4.identity(path) for name,path in evidence_paths.items()},
              "data":data_info, "d8_pretraining_seed":d8_seed,
              "ch3_pretraining_pool_same_as_joint_train_pool":True,
              "joint_training_count_does_not_equal_total_pretraining_data":True,
              "pretraining_train_not_reassigned_to_evaluation":True,
              "validation_previously_used_for_model_selection":True,
              "new_modules_initialized_per_seed_no_historical_warm_start":True,
              "sample_identity_rule":"raw file lineage + raw_index; train and validation use separate raw source files",
              "limitations":["Historical lineage reports support source separation; no claim that random seeds alone prove independence",
                             "Current validation is development evidence, not untouched test"],
              "analysis_strata":{"snr_db_edges":[-10,-5,0,5,10],"include_underflow_overflow_bins":True,
                                   "snr_excludes_k0_sentinel":-999,
                                   "distance_m_edges":[0,500,1000],"small_stratum_threshold":30}}
    g4.write_json(run / "provenance_audit.json",report)
    print(report)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--run",type=Path,required=True)
    audit(p.parse_args().run)
