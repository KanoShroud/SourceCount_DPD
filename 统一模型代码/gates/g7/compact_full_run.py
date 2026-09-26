"""User-owned full foundation adaptation -> candidate training -> Top5 review.

This is a new contract, not a continuation of the retired partial v1 run.
"""
from __future__ import annotations

import argparse
import gc
import shutil
import time

import numpy as np
import torch

from 统一模型代码.gates.g7 import compact_run as legacy
from 统一模型代码.gates.g7.compact_data import BASE, ROOT, OLD_G7, g4, identity, read, write, load, save
from 统一模型代码.gates.g7.compact_model import CompactModel
from 统一模型代码.gates.g7.compact_foundation import FoundationModel, build_oracle_fine
from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng

RUN = BASE / 'v2_full_adaptation'
CONFIG = dict(legacy.CONFIG, version='G7-compact-v2-full-foundations',
    initialization='registered_native_CH3_D8_full_warm_start_then_P2_B_heads',
    foundation_lr=1e-4, foundation_selection='minimum_native_val_loss',
    foundation_trainable='ALL_PARAMETERS_AND_TRAINING_BUFFERS',
    d8_training_input='BW_actual_19_window_coverage_ge_0.2_union_nonzero_source_scenes',
    candidate_initialization='adapted_full_CH3_D8_plus_registered_P2_B_heads',
    candidate_trainable='original_P2_B_groups_after_full_foundation_adaptation',
    execution_scope='foundations_then_candidates_then_top5_review',
    local_six_tracks_executed=False)


class AdaptedCandidate(CompactModel):
    """Full inherited state is carried even when prefixes are frozen downstream."""
    def state(self):
        return dict(super().state(), full_ch3=self.context.ch3.state_dict(),
                    full_d8=self.context.d8.state_dict())

    def restore(self, state):
        super().restore(state)
        # Restore the full adapted models LAST, never overwrite with historical partial heads.
        self.context.ch3.load_state_dict(state['full_ch3'], strict=True)
        self.context.d8.load_state_dict(state['full_d8'], strict=True)


class Runtime(legacy.Runtime):
    def __init__(self, out=RUN):
        super().__init__(out)
        self.oracle = {}

    def contract(self):
        initial = read(OLD_G7/'contract.json')['initial']
        paths = set()
        for folder in ('统一模型代码', '第三章代码', '第四章代码'):
            paths.update((ROOT/folder).rglob('*.py'))
        paths.add(ROOT/'运行入口/E2E/G7/G7小区域一键运行.py')
        value = dict(config=CONFIG, initial=initial,
                     files=[identity(p) for p in sorted(paths)] + self.inputs.sources + list(initial.values()))
        path = self.out/'contract.json'
        if path.exists() and read(path) != value:
            raise RuntimeError('v2代码/输入/配置合同变化；不能把已有训练当作同一运行继续')
        if not path.exists():
            write(path, value)
        for row in value['files']:
            verified_read(row, self.out/'anomalies')
        self.registration = value
        return value

    def prepare_oracle(self, selected=None):
        path = self.out/'oracle/index.json'
        self.oracle = read(path) if path.exists() else dict(train={}, val_select={})
        for split in ('train', 'val_select'):
            ids = selected[split] if selected else list(range(CONFIG['counts'][split]))
            ids = [i for i in ids if self.inputs.manifest['subsets'][split][i]['true_k'] > 0]
            bar = legacy.Progress(f'D8训练选频图 {split}', len(ids), self.out/'progress.json')
            with g4.g1.SampleStore(split) as store:
                for n, i in enumerate(ids):
                    self.guard()
                    if str(i) in self.oracle[split]:
                        verified_read(self.oracle[split][str(i)], self.out/'anomalies')
                    else:
                        entry = self.inputs.manifest['subsets'][split][i]
                        raw, j = store._raw(entry['raw_index'])
                        iq = np.asarray(raw['sig_rcv_real_all'][:, :, j], dtype=np.float64).T + 1j*np.asarray(
                            raw['sig_rcv_imag_all'][:, :, j], dtype=np.float64).T
                        record = dict(self.data(split, [i])[0],
                            actual_fc=np.asarray(raw['fc_offset_all'][:, j], dtype=np.float32),
                            actual_bw=np.asarray(raw['BW_actual_all'][:, j], dtype=np.float32),
                            b_win=float(np.asarray(raw['B_win_val']).reshape(-1)[0]))
                        fine = build_oracle_fine(iq, record, (self.lo, self.hi), device='cuda')
                        self.oracle[split][str(i)] = save(self.out/f'oracle/{split}/{i:05d}_{time.time_ns()}.pt',
                            dict(index=i, raw_index=entry['raw_index'], oracle_fine=fine.cpu()))
                        write(path, self.oracle)
                    bar.update(n+1)
        row = identity(path)
        pointer = self.out/'oracle.identity.json'
        if pointer.exists() and read(pointer) != row:
            raise RuntimeError('已完成oracle输入索引变化')
        if not pointer.exists():
            write(pointer, row)

    def foundation_data(self, kind, split, ids):
        records = self.data(split, ids)
        if kind == 'd8':
            result = []
            for i, record in zip(ids, records):
                value = self.get(self.oracle[split][str(i)])[0]
                if value['index'] != i or value['raw_index'] != record['metadata']['raw_index']:
                    raise RuntimeError('D8 oracle/sample身份不匹配')
                result.append(dict(record, oracle_fine=value['oracle_fine']))
            records = result
        return records

    def model(self, seed, phase):
        initial = load(self.registration['initial'][str(seed)], self.out)
        model = AdaptedCandidate(self.out, self.inputs.manifest, seed, initial)
        for kind in ('ch3', 'd8'):
            pointer = read(self.out/f'foundations/{seed}/{kind}/completed.json')
            cp = load(pointer['best']['checkpoint'], self.out)
            state = cp['state']
            if state['kind'] != kind:
                raise RuntimeError('Wrong foundation checkpoint kind')
            getattr(model.context, kind).load_state_dict(state['model'], strict=True)
        if phase != 'baseline':
            pointer = read(self.out/f'training/{seed}/baseline/completed.json')
            model.restore(load(pointer['best']['checkpoint'], self.out)['state'])
            model.selector.selector.load_state_dict(model.physical.selector.state_dict())
        return model


def foundation_ids(rt, kind, split):
    return [i for i in range(CONFIG['counts'][split]) if kind == 'ch3' or
            rt.inputs.manifest['subsets'][split][i]['true_k'] > 0]


@torch.no_grad()
def validate_foundation(rt, model, kind):
    model.mode(False)
    ids = foundation_ids(rt, kind, 'val_select')
    total = 0.
    for batch in legacy.batches(ids):
        rt.guard()
        value, _ = model.forward_loss(rt.foundation_data(kind, 'val_select', batch))
        total += float(value)*len(batch)
    model.mode(True)
    return total/len(ids)


def train_foundation(rt, seed, kind, end):
    root = rt.out/f'foundations/{seed}/{kind}'
    pointer = root/'completed.json'
    model = FoundationModel(rt.out, rt.inputs.manifest, seed, kind)
    optimizer = model.optimizer(lr=CONFIG['foundation_lr'])
    generator = torch.Generator().manual_seed(seed)
    history, best = [], None
    if pointer.exists():
        previous = read(pointer)
        cp = load(previous['checkpoint'], rt.out)
        model.restore(cp['state']); optimizer.load_state_dict(cp['optimizer'])
        generator.set_state(cp['generator']); restore_rng(cp['rng'])
        history, best = cp['history'], previous['best']
    ids = foundation_ids(rt, kind, 'train')
    def commit(epoch, train_loss, val_loss, seconds):
        nonlocal best
        history.append(dict(epoch=epoch, loss=train_loss, val_loss=val_loss, seconds=seconds))
        cp_row = save(root/f'checkpoints/epoch{epoch:03d}_{time.time_ns()}.pt', dict(
            state=model.state(), optimizer=optimizer.state_dict(), generator=generator.get_state(),
            rng=rng_state(), history=history, seed=seed, kind=kind))
        if val_loss is not None and (best is None or val_loss < best['val_loss']):
            best = dict(epoch=epoch, val_loss=val_loss, checkpoint=cp_row)
        write(pointer, dict(checkpoint=cp_row, best=best, history=history,
                            train_samples=len(ids), full_parameter_training=True))
    if not history:
        commit(0, None, validate_foundation(rt, model, kind), 0.)
    model.mode(True)
    for epoch in range(history[-1]['epoch']+1, end+1):
        started = time.perf_counter()
        order = [ids[i] for i in torch.randperm(len(ids), generator=generator).tolist()]
        bar = legacy.Progress(f'{seed}/{kind} 全模块 {epoch}/{end}', len(order), rt.out/'progress.json')
        total = 0.
        for n, batch in enumerate(legacy.batches(order)):
            rt.guard()
            optimizer.zero_grad(set_to_none=True)
            value, _ = model.forward_loss(rt.foundation_data(kind, 'train', batch))
            value.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.model.parameters(), 10.)
            if not torch.isfinite(value) or not torch.isfinite(norm):
                raise RuntimeError('Foundation loss/gradient nonfinite')
            optimizer.step(); total += float(value.detach())*len(batch)
            bar.update(min((n+1)*CONFIG['batch_size'], len(order)))
        val = validate_foundation(rt, model, kind) if epoch % CONFIG['evaluate_every'] == 0 else None
        commit(epoch, total/len(order), val, time.perf_counter()-started)
        print(f'\n{seed}/{kind} epoch {epoch}: loss={total/len(order):.5g}, '
              f'val={val}, best={best["epoch"]}, {history[-1]["seconds"]/60:.1f} min', flush=True)
    model = optimizer = None
    gc.collect(); torch.cuda.empty_cache()
    return read(pointer)


def foundation_needs_extension(result):
    rows = [h for h in result['history'] if h['val_loss'] is not None]
    return (result['best']['epoch'] >= CONFIG['base_epochs']-2 and len(rows) >= 3 and
            rows[-1]['val_loss'] < rows[-3]['val_loss'])


def run(rt):
    print('阶段1/3：小区域数据与CH3、D8完整适配（两个seed）', flush=True)
    rt.inputs.audit_raw()
    rt.prepare_data()
    rt.prepare_oracle()
    foundations = {}
    for kind in ('ch3', 'd8'):
        results = {seed: train_foundation(rt, seed, kind, CONFIG['base_epochs']) for seed in CONFIG['seeds']}
        if any(foundation_needs_extension(r) for r in results.values()):
            results = {seed: train_foundation(rt, seed, kind, CONFIG['max_epochs']) for seed in CONFIG['seeds']}
        foundations[kind] = results
    write(rt.out/'evaluation/foundation_report.json', dict(status='FOUNDATIONS_COMPLETE', results=foundations))
    print('阶段2/3：继承完整新权重，训练小区域统一候选生成器', flush=True)
    results = {seed: legacy.train(rt, seed, 'baseline', CONFIG['base_epochs']) for seed in CONFIG['seeds']}
    if any(legacy.must_extend(r['history'], CONFIG['base_epochs']) for r in results.values()):
        results = {seed: legacy.train(rt, seed, 'baseline', CONFIG['max_epochs']) for seed in CONFIG['seeds']}
    print('阶段3/3：新候选生成器Top5/8复核；不提前决定局部训练配置', flush=True)
    reports = [legacy.coverage(rt, seed) for seed in CONFIG['seeds']]
    rt.inputs.audit_raw()
    for row in rt.registration['files']:
        verified_read(row, rt.out/'anomalies')
    for group in (rt.dataset, rt.oracle):
        for rows in group.values():
            for row in rows.values():
                rt.guard(); verified_read(row, rt.out/'anomalies')
    for name in ('dataset', 'oracle'):
        verified_read(read(rt.out/f'{name}.identity.json'), rt.out/'anomalies')
    report = dict(status='FULL_ADAPTATION_AND_CANDIDATES_COMPLETE_TOP5_REVIEW',
                  foundations=foundations, candidates=results, top5=reports,
                  local_six_tracks_executed=False, test_executed=False,
                  next='Review Top5/8, then bind local F/S/E contract to these adapted checkpoints')
    write(rt.out/'evaluation/comparison_report.json', report)
    summary = rt.out/'evaluation/运行摘要.md'
    summary.write_text('# 小区域完整适配完成\n\nCH3、D8全模块适配和统一候选生成器训练已完成。\n\n'
        + '\n'.join(f'- {r["seed"]}: Top5 {r["covered"]["5"]}/{r["sources"]}; '
                     f'Top8 {r["covered"]["8"]}/{r["sources"]}。' for r in reports)
        + '\n\n回读comparison_report.json与final_audit_report.json，确定局部F/S/E候选配置。'
          '本入口没有执行局部六轨、硬级联正式评价或test。\n', encoding='utf-8')
    write(rt.out/'evaluation/final_audit_report.json', dict(status='PASS',
        scope=CONFIG['execution_scope'], contract=identity(rt.out/'contract.json'),
        outputs=[identity(p) for p in sorted((rt.out/'evaluation').iterdir())
                 if p.is_file() and p.name != 'final_audit_report.json'], test_executed=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    legacy.setup_environment()
    # Reuse the original trainer with explicitly registered new configuration.
    legacy.CONFIG = CONFIG
    legacy.RUN = RUN
    with legacy.RunLock():
        rt = Runtime()
        rt.contract()
        prep = read(BASE/'full_preparation_report.json')
        report = legacy.json_load_verified(prep['report'], rt.out)
        if report['status'] != 'ENGINEERING_PASS':
            raise RuntimeError('完整适配版本短验证尚未通过')
        if legacy.json_load_verified(report['contract'], rt.out) != rt.registration:
            raise RuntimeError('短验证代码/配置与当前版本不同')
        if args.check:
            print('完整适配入口检查完成；未训练。', flush=True)
            return
        audit_path = rt.out/'evaluation/final_audit_report.json'
        if audit_path.exists():
            audit = read(audit_path)
            if audit['status'] != 'PASS':
                raise RuntimeError('已有结束报告未通过，请先回读结果')
            verified_read(audit['contract'], rt.out/'anomalies')
            for row in audit['outputs']:
                verified_read(row, rt.out/'anomalies')
            print(f'本节点已完成，不重复训练或评价。回读：{audit_path.parent}', flush=True)
            return
        if not rt.budget_path.exists():
            rt.budget['active_seconds'] = report['preparation_consumed_seconds']
        remaining = CONFIG['wall_seconds']-rt.budget['active_seconds']
        if not (rt.out/'foundations').exists() and report['forecast_remaining_seconds'] > remaining:
            raise RuntimeError(f'当前节点预计{report["forecast_remaining_seconds"]/3600:.1f}小时，'
                               f'超过剩余{remaining/3600:.1f}小时预算；未启动长训练，请先调整预算审批')
        if not (rt.out/'data/index.json').exists():
            required = report['projected_dataset_gib'] + report['checkpoint_reserve_gib'] + CONFIG['disk_floor_gib']
            if shutil.disk_usage(rt.out).free/2**30 < required:
                raise RuntimeError(f'完整节点预计需要{required:.1f}GiB可用空间（含预留），未启动')
        rt.begin()
        try:
            write(rt.out/'execution_status.json', dict(status='RUNNING', started_at=time.time()))
            rt.guard()
            run(rt)
            write(rt.out/'execution_status.json', dict(status='COMPLETED', finished_at=time.time()))
        except BaseException as exc:
            write(rt.out/'execution_status.json', dict(status='STOPPED', error=repr(exc),
                  stopped_at=time.time(), next='Read progress.json and latest completed epoch; no incomplete checkpoint is used'))
            raise
        finally:
            rt.finish()


if __name__ == '__main__':
    main()
