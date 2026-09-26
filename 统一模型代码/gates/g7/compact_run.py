"""Small-ROI warm-start adaptation, then a separate user-owned F/S/E run."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import gc
from pathlib import Path
import shutil
import time

import numpy as np
import psutil
import torch
from scipy.optimize import linear_sum_assignment

from 统一模型代码.gates.g7.compact_data import (
    BASE, ROOT, OLD_G7, Inputs, identity, read, write, load, save)
from 统一模型代码.gates.g7.compact_model import CompactModel, candidates, windows, loss, decode
from 统一模型代码.gates.g6.coherent_dpd import AtomicDPD, geometry
from 统一模型代码.gates.g6.g6_p2_runtime import setup_environment, Progress
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as Metrics
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import summarize
from 统一模型代码.gates.g5.r2.g5_r2_train import selection_key, must_extend
from 统一模型代码.common.g5_verified_io import verified_read

CONFIG = dict(version='G7-compact-v1', edge_m=1000, coarse_size=41, fine_size=201,
    coarse_step_m=50, fine_step_m=10, window_size=41, top_k_candidate=5, comparison_top_k=8,
    baseline_decode_top_k=8,
    seeds=[20260921, 20260922], counts=dict(train=4096, val_select=512, val_compare=1024),
    batch_size=4, base_epochs=20, max_epochs=24, evaluate_every=2,
    initialization='registered_P2_B_warm_start', local_initialization='compact_baseline_best',
    precision='physics_float64_complex128_network_float32', ram_limit_percent=85,
    gpu_limit_gib=14, disk_floor_gib=50, wall_seconds=43200, lazy_cache_bytes=512*1024**2,
    global_feature_cache=False, test_executed=False)
RUN = BASE/'v1_warm_start'


class Runtime:
    def __init__(self, out=RUN):
        self.out = Path(out).resolve()
        self.inputs = Inputs(self.out)
        self.items, self.cache_bytes = OrderedDict(), 0
        self.started = time.perf_counter()
        self.budget_path = self.out/'budget.json'
        self.budget = read(self.budget_path) if self.budget_path.exists() else dict(active_seconds=0., open=False)
        self.lo, self.hi = self.inputs.edges()
        axis = torch.linspace(-1000, 1000, 41, dtype=torch.float64)
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.physics = AtomicDPD(self.lo, self.hi, geometry(torch.stack([x.flatten(), y.flatten()], -1),
                                device='cuda', shape=(41, 41)), precompute_phase=False)
        self.physics.p1_batch_mode, self.physics.p1_chunk = 'batched', 256
        self.dataset = None

    def contract(self):
        path = self.out/'contract.json'
        initial = read(OLD_G7/'contract.json')['initial']
        files = [identity(p) for p in sorted((ROOT/'统一模型代码/gates/g7').glob('compact_*.py'))]
        files += [identity(ROOT/'运行入口/E2E/G7/G7小区域一键运行.py')]
        files += self.inputs.sources + list(initial.values())
        # Stable source set across a fresh process and a resumed process.
        known = {r['path'] for r in files}
        for folder in ('统一模型代码', '第三章代码', '第四章代码'):
            for p in sorted((ROOT/folder).rglob('*.py')):
                if str(p) not in known:
                    files.append(identity(p)); known.add(str(p))
        value = dict(config=CONFIG, initial=initial, files=files,
                     material_passport=dict(mode='run', status='REGISTERED', formal_training_owner='user'))
        if path.exists():
            previous = read(path)
            if previous != value:
                if (self.out/'training').exists() or (self.out/'data').exists():
                    raise RuntimeError('小区域合同变化：不能复用已有数据/训练目录，请先核对版本')
                write(self.out/f'contract_superseded_{time.time_ns()}.json', previous)
                write(path, value)
        else:
            write(path, value)
        for row in value['files']:
            verified_read(row, self.out/'anomalies')
        self.registration = value
        return value

    def begin(self):
        if self.budget['open']:
            raise RuntimeError('上次进程被强制终止，需核对budget.json中的已用时间；不自动清零')
        self.started = time.perf_counter()
        self.budget['open'] = True
        write(self.budget_path, self.budget)

    def finish(self):
        self.budget['active_seconds'] += time.perf_counter()-self.started
        self.budget['open'] = False
        write(self.budget_path, self.budget)

    def guard(self):
        if psutil.virtual_memory().percent >= CONFIG['ram_limit_percent']:
            raise RuntimeError('RAM达到85%，保留已提交checkpoint并停止')
        if torch.cuda.memory_allocated()/2**30 > CONFIG['gpu_limit_gib']:
            raise RuntimeError('GPU超过14GiB')
        if shutil.disk_usage(self.out).free/2**30 < CONFIG['disk_floor_gib']:
            raise RuntimeError('磁盘剩余不足50GiB')
        if self.budget['active_seconds']+time.perf_counter()-self.started > CONFIG['wall_seconds']:
            raise RuntimeError('累计12小时预算已用尽；不自动续期')

    def get(self, row):
        key = (row['path'], row['sha256'])
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        value = load(row, self.out)
        while self.items and self.cache_bytes+row['size_bytes'] > CONFIG['lazy_cache_bytes']:
            _, (_, size) = self.items.popitem(last=False)
            self.cache_bytes -= size
        self.items[key] = (value, row['size_bytes'])
        self.cache_bytes += row['size_bytes']
        return self.items[key]

    def data(self, split, ids):
        records = [self.get(self.dataset[split][str(i)])[0] for i in ids]
        for i, r in zip(ids, records):
            expected = self.inputs.manifest['subsets'][split][i]
            if r['index'] != i or r['split'] != split or any(
                    r['metadata'][k] != expected[k] for k in ('raw_index', 'local_index', 'true_k')):
                raise RuntimeError('Compact cache/sample mapping changed')
        return records

    def prepare_data(self):
        progress = {}
        def update(split, n, total):
            if split not in progress:
                progress[split] = Progress(f'小区域输入 {split}', total, self.out/'progress.json')
            progress[split].update(n)
        self.dataset = self.inputs.prepare(self.out/'data',
            {s: list(range(n)) for s, n in CONFIG['counts'].items()}, self.guard, update)
        row = identity(self.out/'data/index.json')
        pointer = self.out/'dataset.identity.json'
        if pointer.exists() and read(pointer) != row:
            raise RuntimeError('Completed compact dataset index identity changed')
        if not pointer.exists():
            write(pointer, row)

    def model(self, seed, phase):
        initial = load(self.registration['initial'][str(seed)], self.out)
        model = CompactModel(self.out, self.inputs.manifest, seed, initial)
        if phase != 'baseline':
            pointer = read(self.out/f'training/{seed}/baseline/completed.json')
            cp = load(pointer['best']['checkpoint'], self.out)
            model.restore(cp['state'])
            model.selector.selector.load_state_dict(model.physical.selector.state_dict())
        return model

    def stats(self, seed, split, ids):
        index = read(self.out/f'local/{seed}/{split}/index.json')
        return [self.get(index[str(i)])[0] for i in ids]


def batches(ids):
    return [ids[i:i+CONFIG['batch_size']] for i in range(0, len(ids), CONFIG['batch_size'])]


def forward(rt, model, seed, phase, split, ids, records):
    return (model.baseline(records, rt.physics) if phase == 'baseline' else
            model.local(records, rt.physics, rt.stats(seed, split, ids), phase))


def metric_rows(result, records, phase):
    decoded = decode(result, phase)
    logits = result['band_logits'].detach().cpu()
    rows = []
    for i, r in enumerate(records):
        k, d = r['count'], decoded[i]
        row = Metrics.metric(None, r['positions'][:k].numpy(), d['positions_m'], logits[i].numpy(),
                             d['active'], r['band'].numpy(), r['ignore'].numpy(), r['metadata'])
        cost = np.zeros((3, k))
        for q in range(3):
            for t in range(k):
                valid = r['ignore'][t] < .5
                cost[q, t] = float(torch.nn.functional.binary_cross_entropy_with_logits(logits[i, q, valid], r['band'][t, valid]))
        a, b = linear_sum_assignment(cost)
        f1, iou = [], []
        for q, t in zip(a, b):
            valid = r['ignore'][t] < .5
            pred, truth = logits[i, q, valid] >= 0, r['band'][t, valid] > .5
            tp, fp, fn = int((pred & truth).sum()), int((pred & ~truth).sum()), int((~pred & truth).sum())
            f1.append(2*tp/max(2*tp+fp+fn, 1)); iou.append(tp/max(tp+fp+fn, 1))
        row.update(index=r['index'], band_only_f1=f1, band_only_iou=iou)
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate(rt, model, seed, phase, split='val_select'):
    model.mode(False)
    rows = []
    progress = Progress(f'{seed}/{phase} {split}', CONFIG['counts'][split], rt.out/'progress.json')
    for ids in batches(list(range(CONFIG['counts'][split]))):
        rt.guard()
        records = rt.data(split, ids)
        rows += metric_rows(forward(rt, model, seed, phase, split, ids, records), records, phase)
        progress.update(len(rows))
    model.mode(True)
    return summarize(rows), rows


def train(rt, seed, phase, end):
    root = rt.out/f'training/{seed}/{phase}'
    root.mkdir(parents=True, exist_ok=True)
    pointer = root/'completed.json'
    model = rt.model(seed, phase)
    optimizer = model.optimizer(phase, rt.inputs.manifest)
    generator = torch.Generator().manual_seed(seed)
    history, best = [], None
    if pointer.exists():
        previous = read(pointer)
        saved = load(previous['checkpoint'], rt.out)
        model.restore(saved['state']); optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator']); restore_rng(saved['rng'])
        history, best = saved['history'], previous['best']
    else:
        metrics, _ = evaluate(rt, model, seed, phase)
        history = [dict(epoch=0, loss=None, validation=metrics, seconds=0.)]
        cp = save(root/f'checkpoints/epoch000_{time.time_ns()}.pt', dict(
            state=model.state(), optimizer=optimizer.state_dict(), generator=generator.get_state(),
            rng=rng_state(), history=history, seed=seed, phase=phase))
        best = dict(epoch=0, metrics=metrics, checkpoint=cp)
        write(pointer, dict(checkpoint=cp, best=best, history=history))
    model.mode(True)
    for epoch in range((history[-1]['epoch']+1) if history else 1, end+1):
        started = time.perf_counter()
        order = torch.randperm(CONFIG['counts']['train'], generator=generator).tolist()
        progress = Progress(f'{seed}/{phase} epoch {epoch}/{end}', len(order), rt.out/'progress.json')
        values = []
        for n, ids in enumerate(batches(order)):
            rt.guard()
            records = rt.data('train', ids)
            result = forward(rt, model, seed, phase, 'train', ids, records)
            value, _ = loss(result, records, phase)
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            params = [p for group in optimizer.param_groups for p in group['params']]
            norm = torch.nn.utils.clip_grad_norm_(params, 10.)
            if not torch.isfinite(value) or not torch.isfinite(norm):
                raise RuntimeError('Nonfinite loss/gradient')
            optimizer.step(); values.append(float(value.detach()))
            progress.update(min((n+1)*CONFIG['batch_size'], len(order)))
        metrics = evaluate(rt, model, seed, phase)[0] if epoch % CONFIG['evaluate_every'] == 0 else None
        history.append(dict(epoch=epoch, loss=float(np.mean(values)), validation=metrics,
                            seconds=time.perf_counter()-started))
        cp = save(root/f'checkpoints/epoch{epoch:03d}_{time.time_ns()}.pt', dict(
            state=model.state(), optimizer=optimizer.state_dict(), generator=generator.get_state(),
            rng=rng_state(), history=history, seed=seed, phase=phase))
        if metrics is not None and (best is None or selection_key(metrics, epoch) < selection_key(best['metrics'], best['epoch'])):
            best = dict(epoch=epoch, metrics=metrics, checkpoint=cp)
        write(pointer, dict(checkpoint=cp, best=best, history=history))
        print(f'\n{seed}/{phase} epoch {epoch} 完成，loss={history[-1]["loss"]:.4f}，'
              f'耗时 {history[-1]["seconds"]/60:.1f} min，best={None if best is None else best["epoch"]}', flush=True)
    del model, optimizer
    gc.collect(); torch.cuda.empty_cache()
    return read(pointer)


@torch.no_grad()
def coverage(rt, seed):
    model = rt.model(seed, 'f')
    model.mode(False)
    rows = []
    for ids in batches(list(range(CONFIG['counts']['val_select']))):
        rt.guard()
        records = rt.data('val_select', ids)
        result = model.baseline(records, rt.physics)
        proposals = candidates(result['heat'], 8)
        for i, (r, proposal) in enumerate(zip(records, proposals)):
            k = r['count']; logits = result['band_logits'][i].cpu()
            cost = np.zeros((3, k))
            for q in range(3):
                for t in range(k):
                    mask = r['ignore'][t] < .5
                    cost[q, t] = float(torch.nn.functional.binary_cross_entropy_with_logits(logits[q, mask], r['band'][t, mask]))
            a, b = linear_sum_assignment(cost)
            row = dict(index=r['index'], count=k)
            for top in (5, 8):
                centers, valid, points, _ = windows(proposal, top)
                hits = [bool(((centers[q]-r['positions'][t]).abs().amax(-1) <= 200)
                             .logical_and(valid[q].flatten(1).any(-1)).any()) for q, t in zip(a, b)]
                row[str(top)] = dict(covered=sum(hits), all_covered=all(hits), unique_points=len(points))
            rows.append(row)
    report = dict(seed=seed, status='TOP5_CANDIDATE_REVIEW', independent_scenes=len(rows),
        sources=sum(r['count'] for r in rows), samples=rows,
        covered={str(t):sum(r[str(t)]['covered'] for r in rows) for t in (5, 8)},
        extra_missed_by_top5=sum(r['8']['covered']-r['5']['covered'] for r in rows), test_executed=False)
    report['by_k'] = {str(k): dict(scenes=sum(r['count'] == k for r in rows),
        sources=sum(r['count'] for r in rows if r['count'] == k),
        covered={str(t): sum(r[str(t)]['covered'] for r in rows if r['count'] == k) for t in (5, 8)},
        all_sources_covered={str(t):sum(r[str(t)]['all_covered'] for r in rows if r['count'] == k) for t in (5, 8)})
        for k in (1, 2, 3)}
    write(rt.out/f'evaluation/{seed}_top5_report.json', report)
    del model; gc.collect(); torch.cuda.empty_cache()
    return report


def run_baselines(rt):
    print('阶段1：核验快照并生成小区域原始输入；不保存D8/CH3特征体', flush=True)
    rt.inputs.audit_raw()
    rt.prepare_data()
    print('阶段2：两个seed的小区域P2-B适配重训', flush=True)
    result = {s: train(rt, s, 'baseline', CONFIG['base_epochs']) for s in CONFIG['seeds']}
    if any(must_extend(r['history'], CONFIG['base_epochs']) for r in result.values()):
        result = {s: train(rt, s, 'baseline', CONFIG['max_epochs']) for s in CONFIG['seeds']}
    print('阶段3：新201网格模型的Top5/Top8覆盖检查，不启动F/S/E', flush=True)
    reports = [coverage(rt, s) for s in CONFIG['seeds']]
    rt.inputs.audit_raw()
    for rows in rt.dataset.values():
        for row in rows.values():
            rt.guard()
            verified_read(row, rt.out/'anomalies')
    verified_read(read(rt.out/'dataset.identity.json'), rt.out/'anomalies')
    write(rt.out/'evaluation/baseline_report.json', dict(status='BASELINES_COMPLETE_TOP5_REVIEW',
          training=result, top5=reports, test_executed=False))
    (rt.out/'evaluation/运行摘要.md').write_text(
        '# 小区域基准重训完成\n\n已完成两个seed适配和Top5/8覆盖检查；尚未执行局部F/S/E六轨。\n\n'
        +'\n'.join(f'- seed {r["seed"]}：Top5覆盖{r["covered"]["5"]}/{r["sources"]}，'
                   f'Top8覆盖{r["covered"]["8"]}/{r["sources"]}。' for r in reports)
        +'\n\n请回读baseline_report.json和两个top5_report.json，决定Top5是否用于正式局部训练。\n', encoding='utf-8')
    for row in rt.registration['files']:
        verified_read(row, rt.out/'anomalies')
    write(rt.out/'evaluation/final_audit_report.json', dict(status='PASS', scope='compact_baselines_and_top5_only',
        contract=identity(rt.out/'contract.json'), dataset=read(rt.out/'dataset.identity.json'),
        outputs=[identity(p) for p in sorted((rt.out/'evaluation').iterdir())
        if p.is_file() and p.name != 'final_audit_report.json'], local_six_tracks_executed=False, test_executed=False))
    print(f'完成，请回读：{rt.out / "evaluation/运行摘要.md"}', flush=True)


class RunLock:
    def __enter__(self):
        import msvcrt
        RUN.mkdir(parents=True, exist_ok=True)
        self.handle = (RUN/'run.lock').open('a+b')
        if self.handle.tell() == 0:
            self.handle.write(b'0'); self.handle.flush()
        self.handle.seek(0)
        try:
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.handle.close()
            raise RuntimeError('已有小区域入口运行，不启动第二个进程') from None
        return self

    def __exit__(self, *_):
        import msvcrt
        self.handle.seek(0)
        msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        self.handle.close()


def main():
    raise RuntimeError('旧v1部分解冻入口已停用；请运行G7小区域一键运行.py使用完整CH3/D8适配流程')
    # Kept below as historical implementation; library helpers remain reusable.
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--run-baselines', action='store_true')
    args = parser.parse_args()
    setup_environment()
    with RunLock():
        rt = Runtime()
        rt.contract()
        prep = read(BASE/'preparation_report.json')
        report = json_load_verified(prep['report'], rt.out)
        if report['status'] != 'ENGINEERING_PASS':
            raise RuntimeError('当前小区域版本尚未通过短验证')
        checked = json_load_verified(report['contract'], rt.out)
        if checked != rt.registration:
            raise RuntimeError('工程短验证代码/配置与当前合同不一致')
        if not args.run_baselines:
            print('合同/路径/短验证检查完成；未训练。正式运行使用 --run-baselines。', flush=True)
            return
        if not rt.budget_path.exists():
            rt.budget['active_seconds'] = report['preparation_consumed_seconds']
        rt.begin()
        try:
            rt.guard()
            remaining = CONFIG['wall_seconds']-rt.budget['active_seconds']
            if not (rt.out/'training').exists() and report['forecast_baselines_remaining_seconds'] > remaining:
                raise RuntimeError('小区域基准重训预计仍超过剩余预算，未启动训练；请回读资源报告')
            required = report['projected_dataset_gib'] + CONFIG['disk_floor_gib'] + 5
            if not (rt.out/'data/index.json').exists() and shutil.disk_usage(rt.out).free/2**30 < required:
                raise RuntimeError('小区域数据准备磁盘预估不足')
            run_baselines(rt)
        finally:
            rt.finish()


def json_load_verified(row, out):
    import json
    return json.loads(verified_read(row, out/'anomalies'))


if __name__ == '__main__':
    main()
