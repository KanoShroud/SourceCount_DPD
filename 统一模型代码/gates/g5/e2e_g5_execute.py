"""串行推进已批准G5；等待已确认的特征进程，不自动重启任何失败阶段。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_train import g4, resource_guard  # noqa: E402


def run_stage(run, manifest, name, arguments):
    logs = run / 'execution_logs'
    logs.mkdir(exist_ok=True)
    path = logs / f'{name}.log'
    if path.exists():
        raise FileExistsError(f'Stage evidence already exists, inspect before any recovery: {path}')
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
    started = time.time()
    with path.open('xb') as log:
        process = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT, env=env,
                                   stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        g4.write_json(run / 'execution_status.json', {'status': 'RUNNING', 'stage': name,
                      'pid': process.pid, 'started_at': started, 'log': str(path), 'test_executed': False})
        print(f'G5 stage {name}: pid={process.pid}, log={path}', flush=True)
        try:
            while process.poll() is None:
                resource_guard(run, manifest)
                time.sleep(10)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            raise
    result = {'stage': name, 'returncode': process.returncode, 'seconds': time.time()-started, 'log': str(path)}
    g4.write_json(logs / f'{name}.json', result)
    if process.returncode != 0:
        raise RuntimeError(f'Stage failed: {name}; inspect {path}')
    if name == 'pilot' and g4.read_json(run / 'pilot_report.json')['status'] != 'PASS':
        raise RuntimeError('Pilot resource projection did not pass; no formal training started')
    return result


def main(run, feature_pid):
    run = run.resolve(strict=True)
    manifest = g4.read_json(run / 'manifest.json')
    results = []
    try:
        process = psutil.Process(feature_pid)
        command = process.cmdline()
        if not any('e2e_g5_features.py' in part for part in command) or str(run.name) not in ' '.join(command):
            raise RuntimeError('Feature process identity mismatch')
        created = process.create_time()
        g4.write_json(run / 'execution_status.json', {'status': 'WAITING_LIVE_FEATURE_PROCESS',
                      'pid': feature_pid, 'process_created_at': created, 'command': command, 'test_executed': False})
        print(f'Waiting for confirmed feature process {feature_pid}', flush=True)
        while process.is_running() and process.create_time() == created:
            resource_guard(run, manifest)
            time.sleep(10)
        if g4.read_json(run / 'feature_manifest.json')['status'] != 'PASS':
            raise RuntimeError('Feature process ended without PASS; will not restart')
        stages = [
            ('freeze', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'freeze']),
            ('identity_before_pilot', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'verify-inputs', '--phase', 'before_pilot']),
            ('pilot', ['统一模型代码/gates/g5/audits/e2e_g5_pilot.py']),
            ('training', ['统一模型代码/gates/g5/e2e_g5_train.py']),
            ('identity_after_training', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'verify-inputs', '--phase', 'after_training']),
            ('comparison', ['统一模型代码/gates/g5/e2e_g5_evaluate.py', '--stage', 'compare']),
            ('hard_reference', ['统一模型代码/gates/g5/e2e_g5_evaluate.py', '--stage', 'hard']),
            ('final_audit', ['统一模型代码/gates/g5/audits/verify_e2e_g5_final.py']),
        ]
        for name, arguments in stages:
            results.append(run_stage(run, manifest, name, [*arguments, '--run', str(run)]))
        g4.write_json(run / 'execution_status.json', {'status': 'EXECUTION_COMPLETE_DOC_SYNC_PENDING',
                      'stages': results, 'test_executed': False})
    except Exception as exc:
        g4.write_json(run / 'execution_status.json', {'status': 'STOP', 'error': repr(exc),
                      'completed_stages': results, 'test_executed': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--feature-pid', type=int, required=True)
    args = parser.parse_args()
    main(args.run, args.feature_pid)
