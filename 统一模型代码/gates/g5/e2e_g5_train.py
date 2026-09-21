"""G5六轨训练；48轮与预注册64轮成对延长，test不可访问。"""
from __future__ import annotations

import argparse
import gc
import io
import json
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_model import build_context, evaluate, forward, g4, load_split  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_prepare import guard  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402

_LAST_DISK_CHECK = {}


def compact(value):
    return {key: item for key, item in value.items() if key != "samples"}


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, value):
    temp = path.with_suffix(".pending")
    torch.save(value, temp)
    # 先登记待发布身份、回读同字节反序列化，再原子发布。
    row = g4.identity(temp)
    check = torch.load(io.BytesIO(verified_read(row, path.parent / "anomalies")), map_location="cpu", weights_only=False)
    def same(left,right):
        if isinstance(left,torch.Tensor):
            return isinstance(right,torch.Tensor) and torch.equal(left.detach().cpu(),right.detach().cpu())
        if isinstance(left,np.ndarray):
            return isinstance(right,np.ndarray) and np.array_equal(left,right)
        if isinstance(left,dict):
            return isinstance(right,dict) and left.keys()==right.keys() and all(same(v,right[k]) for k,v in left.items())
        if isinstance(left,(list,tuple)):
            return type(left) is type(right) and len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
        return left==right
    if not same(value,check):
        raise RuntimeError("Checkpoint round trip failed")
    temp.replace(path)
    row["path"] = str(path)
    g4.write_json(path.with_suffix(".identity.json"), row)
    return row


def load_checkpoint(path):
    row = g4.read_json(path.with_suffix(".identity.json"))
    return torch.load(io.BytesIO(verified_read(row, path.parent / "anomalies")), map_location="cpu", weights_only=False)


def resource_guard(run, manifest):
    from 统一模型代码.common.g5_runtime_v2 import enabled, ram_guard
    if enabled(run):
        ram_guard()
        if shutil.disk_usage(run).free < 50*2**30:
            raise RuntimeError('Disk free below 50 GiB')
    else:
        guard(run)
    cfg = manifest["config"]
    if time.time()-manifest["created_at"] > cfg["wall_limit_seconds"]:
        raise RuntimeError("G5 total wall budget exceeded")
    if torch.cuda.max_memory_allocated() > cfg["gpu_limit_gib"]*2**30:
        raise RuntimeError("G5 CUDA allocation budget exceeded")
    now = time.monotonic()
    if now-_LAST_DISK_CHECK.get(str(run), float('-inf')) >= 60:
        size = sum(path.stat().st_size for path in run.rglob('*') if path.is_file())
        if size > cfg['new_disk_limit_gib']*2**30:
            raise RuntimeError('G5 new disk budget exceeded')
        _LAST_DISK_CHECK[str(run)] = now


def must_extend(history, target_epoch, gain):
    evaluations = [r for r in history if r.get("validation") is not None and r["epoch"] <= target_epoch]
    best = min(evaluations, key=lambda r: (r["validation"]["overall"]["gospa_mean_m"], r["epoch"]))
    tail = [next(r for r in evaluations if r["epoch"] == e) for e in (target_epoch-4, target_epoch-2, target_epoch)]
    values = [r["validation"]["overall"]["gospa_mean_m"] for r in tail]
    slope = float(np.polyfit([r["epoch"] for r in tail], values, 1)[0])
    return best["epoch"] >= target_epoch-4 and slope < 0 and values[0]-values[-1] >= gain


def train_track(run, manifest, fm, seed, track, target_epoch):
    from 统一模型代码.gates.g5.e2e_g5_contract import verify_code
    verify_code(run)
    root = run / "training" / str(seed) / track
    root.mkdir(parents=True, exist_ok=True)
    selections = root / 'selections'
    selections.mkdir(exist_ok=True)
    config = manifest["config"]
    device = torch.device("cuda:0")
    torch.cuda.init()  # Statistics reset requires an initialized CUDA context.
    torch.cuda.reset_peak_memory_stats(device)
    features, targets, cache = load_split(run, manifest, fm, "train")
    selected, select_targets, _ = load_split(run, manifest, fm, "val_select", cache)
    context = build_context(run, manifest, seed, device)
    initial_digest = g4.state_digest(context)
    optimizer = torch.optim.AdamW(context.parameter_groups, weight_decay=config["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    cached_targets = g4.as_cached_targets(targets)
    last = root / "last.pt"
    if last.exists():
        checkpoint = load_checkpoint(last)
        g4.load_state(context, checkpoint["state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        generator.set_state(checkpoint["generator"])
        restore_rng(checkpoint["rng"])
        history, best_value, best_epoch = checkpoint["history"], checkpoint["best_value"], checkpoint["best_epoch"]
        start_epoch, steps = checkpoint["epoch"]+1, checkpoint["optimizer_steps"]
    else:
        evaluation = evaluate(context, selected, select_targets, device, fm["files"]["val_select"]["metadata"])
        best_value, best_epoch = evaluation["overall"]["gospa_mean_m"], 0
        history = [{"epoch": 0, "validation": compact(evaluation)}]
        save_checkpoint(selections / "epoch000.pt", {"epoch": 0, "state": g4.state_payload(context), "metrics": compact(evaluation)})
        g4.write_json(root / "initial.json", {"seed": seed, "track": track, "state_sha256": initial_digest})
        start_epoch, steps = 1, 0
    elapsed_start = time.perf_counter()
    try:
        for epoch in range(start_epoch, target_epoch+1):
            resource_guard(run, manifest)
            start = time.perf_counter()
            order = torch.randperm(len(targets.counts), generator=generator)
            g4.set_mode(context, training=True)
            sums, clipped, max_norm = [], 0, 0.0
            from 统一模型代码.common.g5_runtime_v2 import batches, enabled
            batch_ids = order.split(config['batch_size'])
            iterator = batches(features, batch_ids) if enabled(run) else ((ids,features) for ids in batch_ids)
            for batch_index, (indices, batch_features) in enumerate(iterator, 1):
                resource_guard(run, manifest)
                _, logits, _, heatmap, offset = forward(context, batch_features, indices, device, stop_gradient=track == "sg")
                loss, _, _ = g4.r1.compute_losses(logits, heatmap, offset, cached_targets, indices, config)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(context.parameters, config["gradient_clip"])
                if not torch.isfinite(loss) or not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite training loss/gradient")
                optimizer.step()
                steps += 1
                clipped += float(norm) > config["gradient_clip"]
                max_norm = max(max_norm, float(norm))
                sums.append(float(loss.detach()))
                if batch_index % 128 == 0:
                    g4.write_json(run / "training_progress.json", {"status": "RUNNING", "seed": seed, "track": track,
                        "epoch": epoch, "batch": batch_index, "batches": len(order)//4, "optimizer_steps": steps,
                        "rss_bytes": psutil.Process().memory_info().rss, "cache": cache.stats, "updated_at": time.time()})
            training_seconds = time.perf_counter()-start
            evaluation, validation_seconds = None, 0.0
            if epoch % config["evaluate_every"] == 0:
                start = time.perf_counter()
                evaluation = evaluate(context, selected, select_targets, device, fm["files"]["val_select"]["metadata"])
                validation_seconds = time.perf_counter()-start
                value = evaluation["overall"]["gospa_mean_m"]
                if value < best_value:
                    best_value, best_epoch = value, epoch
                    save_checkpoint(selections / f"epoch{epoch:03d}.pt", {"epoch": epoch, "state": g4.state_payload(context), "metrics": compact(evaluation)})
            row = {"epoch": epoch, "loss_mean": float(np.mean(sums)), "training_seconds": training_seconds,
                   "validation_seconds": validation_seconds, "validation": compact(evaluation) if evaluation else None,
                   "clipped_steps": clipped, "max_gradient_norm": max_norm,
                   "rss_bytes": psutil.Process().memory_info().rss, "cuda_peak_bytes": torch.cuda.max_memory_allocated()}
            history.append(row)
            g4.write_json(root / "history.json", history)
            save_checkpoint(last, {"epoch": epoch, "state": g4.state_payload(context), "optimizer": optimizer.state_dict(),
                                  "rng": rng_state(), "generator": generator.get_state(), "history": history,
                                  "best_value": best_value, "best_epoch": best_epoch, "optimizer_steps": steps})
            print(json.dumps({"seed": seed, "track": track, "epoch": epoch, "train_s": round(training_seconds, 2),
                              "best_epoch": best_epoch, "best_gospa": best_value}), flush=True)
        # 每个候选单独保存；中断在候选写完、last写完之前时，不让新候选覆盖旧最佳证据。
        chosen = load_checkpoint(selections / f'epoch{best_epoch:03d}.pt')
        if chosen['epoch'] != best_epoch or chosen['metrics']['overall']['gospa_mean_m'] != best_value:
            raise RuntimeError('Selected checkpoint and completed history disagree')
        save_checkpoint(root / 'best.pt', chosen)
        del chosen
        report = {"status": "PASS", "seed": seed, "track": track, "completed_epoch": target_epoch,
                  "best_epoch": best_epoch, "best_gospa": best_value, "optimizer_steps": steps,
                  "initial_digest": initial_digest, "seconds_this_call": time.perf_counter()-elapsed_start,
                  "budget_candidate": must_extend(history, target_epoch, config["extension_net_gain_m"]),
                  "best": g4.read_json((root / "best.pt").with_suffix(".identity.json")), "cache": cache.stats}
        report['engineering_contract'] = g4.identity(run / ('engineering_v2/contract.json'
            if (run/'engineering_v2/contract.json').exists() else 'training_code_contract.json'))
        g4.write_json(root / f"report_epoch{target_epoch}.json", report)
        return report
    finally:
        if 'iterator' in locals():
            iterator.close()
        del context, optimizer, features, selected, cache
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    manifest = g4.read_json(run / "manifest.json")
    fm = g4.read_json(run / "feature_manifest.json")
    pilot = g4.read_json(run / "pilot_report.json")
    provenance = g4.read_json(run / "provenance_audit.json")
    model_contract = g4.read_json(run / "model_contract_report.json")
    if any(x["status"] != "PASS" for x in (fm, pilot, provenance, model_contract)):
        raise RuntimeError("G5 preparation gates incomplete")
    outcomes = {}
    for seed in manifest["config"]["training_seeds"]:
        pairs = {}
        for track in ("sg", "e2e"):
            arm = run / "training" / str(seed) / track
            existing = [p for p in (arm / "report_epoch64.json", arm / "report_epoch48.json") if p.exists()]
            pairs[track] = g4.read_json(existing[0]) if existing else train_track(run, manifest, fm, seed, track, 48)
        if any(r["budget_candidate"] or r["completed_epoch"] == 64 for r in pairs.values()):
            pairs = {track: (row if row["completed_epoch"] == 64 else train_track(run, manifest, fm, seed, track, 64))
                     for track, row in pairs.items()}
        outcomes[str(seed)] = pairs
        g4.write_json(run / "training_report.json", {"status": "RUNNING", "seeds": outcomes, "test_executed": False})
    g4.write_json(run / "training_report.json", {"status": "PASS", "seeds": outcomes, "test_executed": False})


if __name__ == "__main__":
    main()
