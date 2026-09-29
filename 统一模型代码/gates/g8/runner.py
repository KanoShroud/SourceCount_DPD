"""G8工程短测入口；正式运行需通过完整度及预算检查，不自动接续。"""
from __future__ import annotations

import argparse
import datetime
import io
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import unittest
import numpy as np
import torch

from .physics import Spectrum, search
from .scenes import make_scene
from .storage import BASE, Guard, write, source_identity, save_scene, load_scene, identity, checked_bytes


def progress(label, done, total, start):
    elapsed = time.perf_counter()-start
    eta = elapsed/max(done, 1)*(total-done)
    print(f'\r{label} [{"="*int(20*done/total):20}] {done}/{total}  已用{elapsed/60:.1f}分  剩余约{eta/60:.1f}分', end='\n' if done == total else '', flush=True)


def short_run(out):
    out = Path(out).resolve()
    if not out.is_relative_to(BASE.resolve()):
        raise ValueError('G8 output isolation violation')
    out.mkdir(parents=True, exist_ok=False)
    guard = Guard(out, seconds=3600)
    guard()
    sources_before = source_identity()
    manifest = dict(mode='engineering_short', performance_evidence=False, test_read=False,
        source_files=sources_before, python=sys.version, numpy=np.__version__, torch=torch.__version__,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        plan='G8-scene-v2', wall_limit_seconds=3600, input_root='new_synthetic_and_registered_local_G7_snapshot')
    write(out/'manifest.json', manifest)
    started = time.perf_counter()
    try:
        print('阶段1/4：工程等价测试、中文子进程日志检查', flush=True)
        log = io.StringIO()
        suite = unittest.defaultTestLoader.loadTestsFromName('统一模型代码.gates.g8.test_g8')
        test_result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
        write(out/'unit_tests.json', dict(success=test_result.wasSuccessful(), count=test_result.testsRun, log=log.getvalue()))
        if not test_result.wasSuccessful():
            raise RuntimeError(log.getvalue())
        env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
        captured = subprocess.run([sys.executable, '-c', 'print("G8中文路径与日志：通过")'],
            capture_output=True, text=True, encoding='utf-8', errors='replace', env=env, check=True)
        if '中文路径与日志：通过' not in captured.stdout:
            raise RuntimeError('非ASCII捕获不一致')
        write(out/'encoding_check.json', dict(stdout=captured.stdout, stderr=captured.stderr, status='PASS'))

        print('阶段2/4：8条代表记录的全频物理搜索与6组HR参数短测', flush=True)
        specs = [dict(group=0, dominance=d, length=n, overlap=o)
                 for n in (4096, 16384) for d in ('same', 'swapped') for o in (.5, 1.)]
        scenes, timings, registered = [], [], []
        for ordinal, spec in enumerate(specs):
            guard()
            t0 = time.perf_counter()
            scene = make_scene(**spec)
            row = save_scene(out/f'data/scene_{ordinal:02d}.npz', scene)
            scene = load_scene(row)
            registered.append(row)
            scenes.append(scene)
            generation_seconds = time.perf_counter()-t0
            for method in ('dpd', 'hr'):
                params = [(4, 1e-4)] if method == 'dpd' or ordinal not in (0, 4) else [(j, a) for j in (2, 4, 8) for a in (1e-4, 1e-2)]
                for j, loading in params:
                    t0 = time.perf_counter()
                    calc = Spectrum(scene['iq'], method, segments=j, loading=loading, guard=guard)
                    prediction = search(calc, 2)
                    timings.append(dict(ordinal=ordinal, **spec, method=method, J=j, loading=loading,
                        seconds=time.perf_counter()-t0, generation_seconds=generation_seconds,
                        info=calc.info, count_returned=len(prediction['positions'])))
            write(out/'physics_timings.json', timings)
            progress('物理短测', ordinal+1, len(specs), started)
        write(out/'data_index.json', registered)

        # Verify atomic recovery without a formal comparison: the callback runs once.
        from .study import Study
        recovery = object.__new__(Study)
        recovery.out = out/'recovery_probe'
        recovery.checkpoint_guard = guard
        calls = []
        def once():
            calls.append(1)
            return dict(message='恢复流程：只执行一次', value=42)
        a = recovery.unit('committed.json', once)
        b = recovery.unit('committed.json', once)
        if a != b or len(calls) != 1:
            raise RuntimeError('已提交小批恢复检查失败')
        write(out/'recovery_check.json', dict(status='PASS', callback_calls=len(calls)))
        write(recovery.out/'interrupted.json', {'uncommitted':True})
        recovered = recovery.unit('interrupted.json', lambda:dict(recomputed=True))
        if not recovered.get('recomputed') or not list(recovery.out.glob('*.uncommitted_*')):
            raise RuntimeError('未提交记录的保留与重算检查失败')

        print('阶段3/4：核对冻结模型身份，测量完整系统及旧数据读取', flush=True)
        from .legacy import Sources, Models, bind
        sources = Sources(out)
        write(out/'frozen_inputs.json', sources.inputs)
        old = []
        for k in range(4):
            entry = next(r for r in sources.selected() if r['count'] == k and r['role'] == 'calibration')
            t0 = time.perf_counter()
            scene = sources.old_scene(entry)
            old.append(scene)
            row = save_scene(out/f'data/old_k{k}.npz', scene)
            registered.append(row)
            write(out/f'old_k{k}_read.json', dict(seconds=time.perf_counter()-t0, metadata=scene['metadata']))
        net_times = []
        for seed in sources.seeds:
            model = Models(sources, seed)
            model_start = time.perf_counter()
            # Both lengths; all old K controls. No outcome-dependent selection.
            for number, scene in enumerate([scenes[0], scenes[4], *old]):
                guard()
                result = model.infer(scene['iq'], guard)
                delta = None
                if 'cached_coarse' in scene:
                    a, b = result['coarse'].astype(float), scene['cached_coarse'].astype(float)
                    delta = float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))
                    if delta > 1e-5:
                        raise RuntimeError(f'重算CH3输入不匹配冻结缓存：{delta}')
                for kind in ('hard', 'candidate'):
                    t0 = time.perf_counter()
                    pred = result[kind]
                    logits = np.asarray(pred['logits'])
                    probs = 1/(1+np.exp(-np.clip(logits, -50, 50)))
                    binding = bind(scene['iq'], pred['positions'], probs[pred['active']], guard=guard)
                    iterative = bind(scene['iq'], pred['positions'], probs[pred['active']], guard=guard, iterations=3)
                    net_times.append(dict(seed=seed, number=number, origin=scene['metadata']['origin'],
                        count=scene['metadata']['count'], samples=scene['iq'].shape[1], method=kind,
                        seconds=pred['seconds'], binding_two_rules_seconds=time.perf_counter()-t0,
                        predicted_count=len(pred['positions']), coarse_cache_relative_error=delta,
                        binding_count=len(binding['positions']), iterative_binding_count=len(iterative['positions'])))
                write(out/'network_timings.json', net_times)
                progress(f'模型 {seed}', number+1, 6, model_start)
            del model
            torch.cuda.empty_cache()

        print('阶段4/4：身份复核与成本汇总', flush=True)
        print('附加工程检查：4条记录走通校准、冻结、比较、图表与汇总（非性能证据）', flush=True)
        flow = Study(out/'flow_probe', engineering=True)
        flow.run()
        flow.finish_budget()
        del flow
        for row in registered:
            checked_bytes(row)
        # Snapshot consumption already uses per-block SHA; do not read unrelated train files.
        for row in sources.inputs:
            if 'blocks' not in row:
                checked_bytes(row)
        if source_identity() != sources_before:
            raise RuntimeError('短测期间源码身份改变')
        estimate = estimate_cost(timings, net_times, registered)
        write(out/'cost_estimate.json', estimate)
        write(out/'data_index.json', registered)
        report = dict(status='SHORT_COMPLETED', performance_evidence=False, test_read=False,
            seconds=time.perf_counter()-started, unit_tests=test_result.testsRun,
            peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30, peak_ram_percent=guard.peak_ram_percent,
            cost=estimate, formal_run_started=False)
        write(out/'report.json', report)
        write(out/'final_audit_report.json', dict(status='PASS', scope='engineering_short_only',
            source_identity_unchanged=True, consumed_data_verified=True, test_read=False,
            outputs=[identity(out/n) for n in ('report.json', 'cost_estimate.json', 'physics_timings.json', 'network_timings.json')]))
        engineering_seconds = 0.
        from .storage import read
        for folder in BASE.glob('short_*'):
            if (folder/'failure.json').exists():
                engineering_seconds += float(read(folder/'failure.json')['elapsed_seconds'])
            elif (folder/'report.json').exists():
                engineering_seconds += float(read(folder/'report.json')['seconds'])
        write(BASE/'release.json', dict(source_files=sources_before,
            short_audit=identity(out/'final_audit_report.json'), cost=estimate,
            engineering_active_seconds=engineering_seconds,
            formal_entry_ready=bool(estimate['provisional_with_50_percent_reserve_hours']+engineering_seconds/3600 <= 12)))
        print(f'短测完成：{out}', flush=True)
    except BaseException as exc:
        write(out/'failure.json', dict(status='FAILED_PRESERVED', error=repr(exc), traceback=traceback.format_exc(),
                                      elapsed_seconds=time.perf_counter()-started))
        write(out/'final_audit_report.json', dict(status='FAIL', scope='engineering_short_only', error=repr(exc)))
        raise


def estimate_cost(physical, networks, records):
    # 64 old +128 new calibration;192 old+272 new checking (including16controls).
    total = 0.
    pieces = {}
    for n, calibration, checking in [(4096, 128, 336), (16384, 64, 128)]:
        dpd = np.mean([r['seconds'] for r in physical if r['method'] == 'dpd' and r['length'] == n])
        hr = np.mean([r['seconds'] for r in physical if r['method'] == 'hr' and r['length'] == n])
        physics_seconds = (calibration+checking)*dpd+(6*calibration+checking)*hr
        # Full system: two existing seeds. Candidate/hard are independently timed incl coarse prep.
        net = np.mean([r['seconds']+r['binding_two_rules_seconds'] for r in networks if r['samples'] == n])*2
        system_seconds = (calibration+checking)*2*net
        pieces[str(n)] = dict(physics_seconds=float(physics_seconds), system_seconds=float(system_seconds))
        total += physics_seconds+system_seconds
    # Extra diagnostics/reporting and CPU/GPU contention are not directly measured.
    return dict(measured_path_extrapolation_hours=total/3600,
        provisional_with_50_percent_reserve_hours=total*1.5/3600,
        not_a_runtime_promise=True, pieces=pieces,
        sample_iq_storage_bytes=sum(r['size_bytes'] for r in records),
        persistent_scene_estimate_gib=sum(r['size_bytes'] for r in records[:8])/8*400/2**30,
        unmeasured=['conditional65536diagnostics', 'full_group_statistics_and_figures', 'B2_predictedK_search_cost'],
        formal_ready=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['short', 'formal'], default='formal')
    parser.add_argument('--run-id')
    args = parser.parse_args()
    args.run_id = args.run_id or ('short_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') if args.mode == 'short' else 'formal_20260928_v2')
    if Path(args.run_id).name != args.run_id:
        raise ValueError('run-id must be one directory name')
    torch.set_num_threads(4)
    if args.mode == 'short':
        short_run(BASE/args.run_id)
    else:
        from .storage import read
        release = read(BASE/'release.json')
        checked_bytes(release['short_audit'])
        if release['source_files'] != source_identity() or not release['formal_entry_ready']:
            raise RuntimeError('当前源码未通过最终短测，或预计超12小时；不能启动正式比较')
        from .study import run_formal
        run_formal(BASE/args.run_id)


if __name__ == '__main__':
    main()
