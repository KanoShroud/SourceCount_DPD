"""G8分阶段驱动：已提交的小批记录可恢复；不训练、不读取test。"""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from .storage import BASE, Guard, source_identity, write, read, identity, checked_bytes, load_scene, save_scene
from .scenes import specifications, make_scene
from .physics import Spectrum, search
from .legacy import Sources, Models, bind
from .diagnostics import metrics, summarize, information, case_figure, cyclic_features


class Study:
    def __init__(self, out, engineering=False):
        self.out = Path(out).resolve()
        if not self.out.is_relative_to(BASE.resolve()):
            raise ValueError('Wrong G8 output root')
        self.out.mkdir(parents=True, exist_ok=True)
        self.engineering = engineering
        self.guard = Guard(self.out)
        self.guard()
        self.source = Sources(self.out)
        self.contract = dict(version='G8-scene-v2', engineering_only=engineering,
            sources=source_identity(), inputs=self.source.inputs,
            sample_seed=2026092708, bootstrap_seed=2026092709, test_read=False,
            gospa=dict(p=2, c_m=100, alpha=2, implementation='G7_s2g3_composability'),
            old_counts=dict(calibration=64, check=192),
            new_counts=dict(calibration=128, check=272),
            physical_parameters=[dict(J=j, loading=a) for j in (2, 4, 8) for a in (1e-4, 1e-2)],
            geometry_and_source_data='see_each_scene_metadata',
            coarse_native_axis='sub,x,y', fine_axis='y,x',
            supplementary_status='cyclic_features_exploratory_R04_detector_not_reproduced',
            conditional_diagnostics='65536_and_oracle_reliability_only_if_later_triggered',
            maximum_hours=12)
        path = self.out/'manifest.json'
        if path.exists():
            if read(path) != self.contract:
                raise RuntimeError('正式合同改变，不能混用历史结果；请使用新目录')
        else:
            if any(p.name != 'run.lock' for p in self.out.iterdir()):
                raise RuntimeError('新正式目标目录必须为空')
            write(path, self.contract)
        release = read(BASE/'release.json') if (BASE/'release.json').exists() else {}
        initial = 0. if engineering else release.get('engineering_active_seconds', 0.)
        self.budget = read(self.out/'budget.json') if (self.out/'budget.json').exists() else dict(active_seconds=initial)
        # Reserve a full minute before each work unit; a crash cannot erase its charge.
        self.reserved = 0.
        self.last_progress = 0.
        self.stage_times = {}

    def checkpoint_guard(self):
        self.guard()
        elapsed = time.perf_counter()-self.guard.started
        used = self.budget['active_seconds']+elapsed
        if used >= 43200:
            raise RuntimeError('G8累计12小时预算已用尽')
        if elapsed >= self.reserved:
            self.reserved = (int(elapsed//60)+1)*60.
            write(self.out/'budget.json', dict(active_seconds=self.budget['active_seconds']+self.reserved,
                accounting='active time plus at most60seconds crash reserve'))

    def finish_budget(self):
        elapsed = time.perf_counter()-self.guard.started
        write(self.out/'budget.json', dict(active_seconds=self.budget['active_seconds']+elapsed,
            accounting='completed invocation actual time'))

    def unit(self, name, callback):
        self.checkpoint_guard()
        path = self.out/name
        marker = path.with_name(path.name+'.identity.json')
        if marker.exists():
            return json.loads(checked_bytes(read(marker)))
        if path.exists():
            # An interrupted JSON commit is not evidence; retain it and recompute.
            backup = path.with_name(path.name+f'.uncommitted_{time.time_ns()}')
            if not path.resolve().is_relative_to(self.out) or not backup.resolve().is_relative_to(self.out):
                raise ValueError('Uncommitted-output recovery escaped G8 run')
            path.rename(backup)
        value = callback()
        write(path, value)
        write(marker, identity(path))
        return value

    def progress(self, stage, done, total):
        now = time.perf_counter()
        if now-self.last_progress < 10 and done != total:
            return
        self.last_progress = now
        elapsed = now-self.guard.started
        origin, initial_done = self.stage_times.setdefault(stage, (now, max(0, done-1)))
        stage_elapsed = now-origin
        eta = stage_elapsed/max(done-initial_done, 1)*(total-done)
        write(self.out/'progress.json', dict(stage=stage, done=done, total=total, invocation_seconds=elapsed,
                                           stage_seconds=stage_elapsed, eta_seconds=eta))
        print(f'\r{stage} [{"="*int(20*done/max(total,1)):20}] {done}/{total}  本次已用{elapsed/60:.1f}分  本阶段剩余约{eta/60:.1f}分',
              end='\n' if done == total else '', flush=True)

    def prepare(self):
        specs = [('old_'+str(e['index']), 'old', e) for e in self.source.selected()]
        specs += [(f'new_{i:03}', 'new', spec) for i, spec in enumerate(specifications())]
        if self.engineering:
            specs = [(f'probe_{i}', 'new', spec) for i, spec in enumerate([
                dict(group=0, dominance='same', length=4096, overlap=.5),
                dict(group=0, dominance='swapped', length=16384, overlap=.5),
                dict(group=16, dominance='same', length=4096, overlap=.5),
                dict(group=16, dominance='swapped', length=16384, overlap=.5)])]
        index = []
        for number, (sid, origin, spec) in enumerate(specs):
            def make(sid=sid, origin=origin, spec=spec):
                path = self.out/f'data/{sid}.npz'
                if path.exists():
                    path = path.with_name(path.stem+f'_retry_{time.time_ns()}.npz')
                scene = self.source.old_scene(spec) if origin == 'old' else make_scene(**spec)
                row = save_scene(path, scene)
                return dict(id=sid, identity=row, metadata=scene['metadata'])
            entry = self.unit(f'data/{sid}.json', make)
            index.append(entry)
            self.progress('数据准备', number+1, len(specs))
        write(self.out/'data_index.json', index)
        return index

    def physics_one(self, entry, method, parameters):
        start = time.perf_counter()
        scene = load_scene(entry['identity'])
        calc = Spectrum(scene['iq'], method, segments=parameters['J'], loading=parameters['loading'], guard=self.checkpoint_guard)
        pred = search(calc, scene['metadata']['count'])
        seconds = time.perf_counter()-start
        return dict(prediction=pred, info=calc.info,
                    metrics=metrics(scene['metadata'], pred['positions'], seconds=seconds))

    def systems_entry(self, entry, model, parameters, temperatures=None):
        start = time.perf_counter()
        scene = load_scene(entry['identity'])
        io_seconds = time.perf_counter()-start
        result = self.systems_one(scene, model, parameters, temperatures)
        for row in result['methods'].values():
            row['seconds'] += io_seconds
        result['read_and_verify_seconds'] = io_seconds
        return result

    def calibrate_physics(self, index):
        calibration = [e for e in index if e['metadata']['role'] == 'calibration']
        selected = {}
        for n in (4096, 16384):
            cases = [e for e in calibration if e['metadata']['samples'] == n and e['metadata']['count'] > 0]
            scores = []
            for p in self.contract['physical_parameters']:
                rows = []
                label = f"hr_J{p['J']}_a{p['loading']}"
                for i, entry in enumerate(cases):
                    row = self.unit(f'calibration/physics/{label}/{entry["id"]}.json',
                                    lambda e=entry, p=p: self.physics_one(e, 'hr', p))
                    rows.append(row['metrics'])
                    self.progress(f'物理校准 N={n} {label}', i+1, len(cases))
                scores.append(dict(parameters=p, gospa=np.mean([r['gospa_m'] for r in rows]),
                                   seconds=np.mean([r['seconds'] for r in rows])))
            selected[str(n)] = min(scores, key=lambda r:(r['gospa'],r['seconds']))['parameters']
        return selected

    def systems_one(self, scene, model, hr_parameters, temperatures=None):
        start = time.perf_counter()
        result = model.infer(scene['iq'], self.checkpoint_guard)
        output = {}
        for kind in ('hard', 'candidate'):
            pred = result[kind]
            logits = np.asarray(pred['logits'])
            probabilities = 1/(1+np.exp(-np.clip(logits, -50, 50)))
            bands = probabilities[pred['active']]
            if kind == 'candidate':
                output['B1'] = metrics(scene['metadata'], pred['positions'], bands, pred['seconds'])
                candidates = [p for c in pred['candidates'] for p in c['positions']]
                output['B1']['candidate_oracle'] = metrics(scene['metadata'], candidates)
            variants = [(0, 1)]+[(3, t) for t in ((1, 2) if temperatures is None else (temperatures[kind],))]
            for iterations, temperature in variants:
                t = time.perf_counter()
                binding = bind(scene['iq'], pred['positions'], bands, guard=self.checkpoint_guard,
                               iterations=iterations, temperature=temperature)
                key = ('B0' if kind == 'hard' else 'B1_rebind') if iterations == 0 else f'B3_{kind}_T{temperature}'
                chosen = bands[binding['slot_indices']]
                row = metrics(scene['metadata'], binding['positions'], chosen, pred['seconds']+time.perf_counter()-t)
                row['binding'] = binding
                output[key] = row
        # Hybrid backend only: predicted K from native CH3, never truth.
        native = result['hard']
        logits = np.asarray(native['logits'])
        bands = (1/(1+np.exp(-np.clip(logits, -50, 50))))[native['active']]
        t = time.perf_counter()
        calc = Spectrum(scene['iq'], 'hr', segments=hr_parameters['J'], loading=hr_parameters['loading'], guard=self.checkpoint_guard)
        p = search(calc, len(native['active']))
        binding = bind(scene['iq'], p['positions'], bands, guard=self.checkpoint_guard)
        output['B2'] = metrics(scene['metadata'], binding['positions'], bands[binding['slot_indices']],
                               result['ch3_seconds']+time.perf_counter()-t)
        output['B2']['binding'] = binding
        return dict(methods=output, wall_seconds=time.perf_counter()-start)

    def run(self):
        index = self.prepare()
        parameters = self.unit('calibration/physical_selection.json', lambda:self.calibrate_physics(index))
        def cycle_calibration():
            by_length = {}
            seen = set()
            for e in index:
                m = e['metadata']
                key = (m['group'], m['samples'])
                if m['origin'] != 'new' or m['role'] != 'calibration' or key in seen:
                    continue
                seen.add(key)
                features = cyclic_features(load_scene(e['identity'])['noise'])
                by_length.setdefault(str(m['samples']), []).append(max(max(f['strengths']) for f in features))
            return {n:dict(threshold=float(np.quantile(v, .95)), independent_noise_records=len(v),
                rule='95th_percentile_of_max_over_registered_cycle_features', exploratory=True) for n,v in by_length.items()}
        cycles = self.unit('calibration/cyclic_noise_thresholds.json', cycle_calibration)
        # Complete all calibration selection BEFORE any checking metrics are consumed.
        temperatures = {}
        for seed in self.source.seeds:
            model = Models(self.source, seed)
            calibration = [e for e in index if e['metadata']['role'] == 'calibration']
            rows = []
            for i, entry in enumerate(calibration):
                row = self.unit(f'calibration/systems/{seed}/{entry["id"]}.json',
                    lambda e=entry, model=model:self.systems_entry(e, model, parameters[str(e['metadata']['samples'])]))
                rows.append(row)
                self.progress(f'系统校准 {seed}', i+1, len(calibration))
            temperatures[str(seed)] = {kind:min((1, 2), key=lambda t:
                -sum(r['methods'][f'B3_{kind}_T{t}']['joint_tp'] for r in rows)) for kind in ('hard', 'candidate')}
            del model
            torch.cuda.empty_cache()
        selection = dict(physical=parameters, temperatures=temperatures, cyclic_noise=cycles)
        frozen = self.unit('selection_frozen.json', lambda:selection)
        if frozen != selection:
            raise RuntimeError('校准选择发生变化')
        checking = [e for e in index if e['metadata']['role'] == 'check']
        for i, entry in enumerate(checking):
            for method in ('dpd', 'hr'):
                self.unit(f'check/physics/{method}/{entry["id"]}.json',
                    lambda e=entry, m=method:self.physics_one(e, m, parameters[str(e['metadata']['samples'])]))
            self.unit(f'check/information/{entry["id"]}.json', lambda e=entry:information(
                load_scene(e['identity']), cycles[str(e['metadata']['samples'])]['threshold']))
            self.progress('独立物理比较与信息诊断', i+1, len(checking))
        for seed in self.source.seeds:
            model = Models(self.source, seed)
            for i, entry in enumerate(checking):
                self.unit(f'check/systems/{seed}/{entry["id"]}.json',
                    lambda e=entry, model=model:self.systems_entry(e, model, parameters[str(e['metadata']['samples'])], temperatures[str(seed)]))
                self.progress(f'完整系统比较 {seed}', i+1, len(checking))
            del model
            torch.cuda.empty_cache()
        # True components are accessed only here, not by any deployment method.
        for i, entry in enumerate(checking):
            if entry['metadata']['origin'] == 'new':
                self.unit(f'check/single_source/{entry["id"]}.json',
                          lambda e=entry:self.single_source(e, parameters))
                self.progress('单源反事实诊断', i+1, len(checking))
        self.report(checking, frozen)

    def single_source(self, entry, parameters):
        scene = load_scene(entry['identity'])
        p = parameters[str(scene['metadata']['samples'])]
        output = []
        for s, component in enumerate(scene['components']):
            m = dict(scene['metadata'], count=1, positions=[scene['metadata']['positions'][s]])
            methods = {}
            for method in ('dpd', 'hr'):
                calc = Spectrum(component+scene['noise'], method, segments=p['J'], loading=p['loading'], guard=self.checkpoint_guard)
                pred = search(calc, 1)
                methods[method] = metrics(m, pred['positions'])
            output.append(methods)
        return dict(scope='oracle_component_plus_same_noise_not_deployment', sources=output)

    def report(self, checking, selection):
        strata = {}
        for entry in checking:
            m = entry['metadata']
            key = f"old_K{m['count']}" if m['origin'] == 'old' else f"new_{m['dominance']}_N{m['samples']}_IoU{m['overlap_iou']}_control{m['total_power_control']}"
            strata.setdefault(key, []).append(entry)
        result = {}
        for key, entries in strata.items():
            result[key] = {}
            for method in ('dpd', 'hr'):
                rows = [json.loads(checked_bytes(read(self.out/f'check/physics/{method}/{e["id"]}.json.identity.json')))['metrics'] for e in entries]
                result[key]['knownK_'+method] = summarize(rows)
            for seed in self.source.seeds:
                records = [json.loads(checked_bytes(read(self.out/f'check/systems/{seed}/{e["id"]}.json.identity.json')))['methods'] for e in entries]
                for method in ('B0', 'B1', 'B2', 'B3_hard', 'B3_candidate'):
                    selected = method if not method.startswith('B3') else method+'_T'+str(selection['temperatures'][str(seed)][method[3:]])
                    result[key][f'{seed}_{method}'] = summarize([r[selected] for r in records])
        # Four predeclared examples: same/swapped, short/long, first checking layout.
        examples = [e for e in checking if e['metadata']['origin'] == 'new' and
                    e['metadata']['group'] == 16 and e['metadata']['overlap_iou'] == .5 and not e['metadata']['total_power_control']]
        for e in examples:
            scene = load_scene(e['identity'])
            pred = {m:read(self.out/f'check/physics/{m}/{e["id"]}.json')['prediction'] for m in ('dpd', 'hr')}
            case_figure(scene, pred, self.out/f'figures/{e["id"]}.png', self.checkpoint_guard)
        intervals = self.bootstrap(checking, selection)
        report = dict(status='ENGINEERING_FLOW_PASS' if self.engineering else 'COMPLETED_AWAITING_SCIENTIFIC_REVIEW',
            performance_evidence=not self.engineering,
            scope='G8_first_round_no_training_no_test', strata=result, selection=selection, test_read=False,
            paired_intervals=intervals,
            limitations=['循环统计是线索描述，不是R04方法复现或排除依据',
                         '场景实际需求仍需结合资料审核；不据本轮仿真确认研究空白',
                         '条件触发的65536点/真值可靠性诊断未自动启动',
                         '冻结模型分布失配不能替代常规适配结果'])
        write(self.out/'report.json', report)
        text = '# G8运行摘要\n\n'+('仅工程微型流程，不作为性能证据。' if self.engineering else '首轮核验计算完成，待回读分析。')+'未训练、未读取test。\n\n'
        text += '按report.json的分层结果判断场景和方法，不将全部条件混成平均数。\n\n'
        text += '| 条件 | 方法 | GOSPA(m) | 匹配RMSE(m) | Recall100 |\n|---|---|---|---|---|\n'
        for key, methods in result.items():
            for method, row in methods.items():
                text += f"| {key} | {method} | {row['gospa_m']:.3f} | {row['rmse_m']} | {row['recall']['100']} |\n"
        (self.out/'运行摘要.md').write_text(text, encoding='utf-8')

    def bootstrap(self, checking, selection):
        result = {}
        for origin in ('old', 'new'):
            entries = [e for e in checking if e['metadata']['origin'] == origin and not e['metadata'].get('total_power_control', False)]
            if not entries:
                continue
            groups = sorted({e['metadata']['group'] for e in entries})
            # Same bootstrap draws across seeds/methods. Old samples stratified by K.
            rng = np.random.default_rng(2026092709)
            if origin == 'old':
                strata = [[j for j, g in enumerate(groups) if next(e for e in entries if e['metadata']['group'] == g)['metadata']['count'] == k] for k in range(4)]
                draws = np.concatenate([rng.choice(s, size=(2000, len(s)), replace=True) for s in strata], axis=1)
            else:
                draws = rng.integers(len(groups), size=(2000, len(groups)))
            vectors = {}
            for method in ('dpd', 'hr'):
                vectors['knownK_'+method] = {e['id']:read(self.out/f'check/physics/{method}/{e["id"]}.json')['metrics']['gospa_m'] for e in entries}
            for seed in self.source.seeds:
                for method in ('B0', 'B1', 'B2', 'B3_hard', 'B3_candidate'):
                    selected = method if not method.startswith('B3') else method+'_T'+str(selection['temperatures'][str(seed)][method[3:]])
                    vectors[f'{seed}_{method}'] = {e['id']:read(self.out/f'check/systems/{seed}/{e["id"]}.json')['methods'][selected]['gospa_m'] for e in entries}
            pairs = [('knownK_hr', 'knownK_dpd')]
            for seed in self.source.seeds:
                pairs += [(f'{seed}_{a}', f'{seed}_{b}') for a,b in [('B2','B0'),('B1','B0'),('B3_hard','B0'),('B3_candidate','B1')]]
            for a, b in pairs:
                values = np.array([np.mean([vectors[a][e['id']]-vectors[b][e['id']] for e in entries if e['metadata']['group'] == g]) for g in groups])
                distribution = values[draws].mean(-1)
                result[f'{origin}:{a}-{b}'] = dict(mean_delta_gospa_m=float(values.mean()),
                    paired_group_95_interval=np.quantile(distribution,[.025,.975]).tolist(), groups=len(groups), draws=2000)
        return result


def run_formal(out):
    out = Path(out).resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError('Wrong G8 output root')
    out.mkdir(parents=True, exist_ok=True)
    # OS-held byte lock is released even after forced termination; no stale PID guessing.
    import msvcrt
    lock = (out/'run.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    study = None
    try:
        study = Study(out)
        study.run()
        for row in study.source.inputs:
            checked_bytes(row)
        if source_identity() != study.contract['sources']:
            raise RuntimeError('正式运行期间源码改变')
        for p in out.rglob('*.json.identity.json'):
            checked_bytes(read(p))
        write(out/'final_audit_report.json', dict(status='PASS', test_read=False,
            scope='first_round_outputs_not_research_gap_verdict',
            outputs=[identity(out/n) for n in ('report.json', '运行摘要.md', 'selection_frozen.json')]))
    except BaseException as exc:
        write(out/f'failure_{time.time_ns()}.json', dict(status='FAILED_PRESERVED', error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if study is not None:
            study.finish_budget()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()
