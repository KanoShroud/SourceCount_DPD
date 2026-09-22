"""R2独立编排。默认长运行只由用户的一键入口启动。"""
from __future__ import annotations

import argparse
import gc
import hashlib
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
import os  # noqa: E402
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch  # noqa: E402
from 统一模型代码.gates.g5.r2.g5_r2_runtime import (  # noqa: E402
    BASE, SOURCE, CONFIG, SEEDS, ARMS, Runtime, RunLock, register, read, write, safe_print, setup_environment, identity)
from 统一模型代码.gates.g5.r2.g5_r2_prepare import prepare  # noqa: E402
from 统一模型代码.gates.g5.r2.g5_r2_train import train_all, restore, load_checkpoint  # noqa: E402
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import evaluate, build_report, save_rows  # noqa: E402


class Log:
    def __init__(self, console, handle):
        self.console,self.handle = console,handle
        self.progress = False
    @property
    def encoding(self):
        return 'utf-8'
    def write(self,text):
        self.console.write(text.encode(getattr(self.console,'encoding',None) or 'utf-8',errors='replace').decode(
            getattr(self.console,'encoding',None) or 'utf-8'))
        self.console.flush()
        if text.startswith('\r'):
            self.progress = True
        elif self.progress and text == '\n':
            self.progress = False
        elif not self.progress:
            self.handle.write(text)
            self.handle.flush()
        return len(text)
    def flush(self):
        self.console.flush()
        self.handle.flush()


def check_ready(runtime):
    runtime.preflight()
    prep = read(BASE/'preparation_report.json')
    sha = hashlib.sha256((BASE/'contract.json').read_bytes()).hexdigest()
    if prep['status']!='ENGINEERING_PASS' or prep['contract_sha256']!=sha:
        raise RuntimeError('缺少当前合同的有效工程测试')
    if not prep['forecast_within_budget']:
        raise RuntimeError('FORECAST_EXCEEDS_BUDGET：不启动正式训练，需审批预算')
    return prep


def execute(runtime):
    training = train_all(runtime)
    results = {}
    for seed in SEEDS:
        for arm in ARMS:
            root = runtime.out/'evaluation'/f'{seed}_{arm}'
            root.mkdir(parents=True,exist_ok=True)
            done = root/'complete.json'
            if done.exists():
                marker = read(done)
                from 统一模型代码.common.g5_verified_io import verified_read
                import json
                data = verified_read(marker['samples'],runtime.out/'anomalies').decode('utf-8')
                if marker['checkpoint'] != training['seeds'][str(seed)][arm]['best']['checkpoint']:
                    raise RuntimeError('Saved evaluation checkpoint mismatch')
                results[seed,arm] = [json.loads(line) for line in data.splitlines()]
                continue
            attempt = root/f'attempt_{time.time_ns()}'
            attempt.mkdir()
            context,head,optimizer,parameters = runtime.context(seed,arm)
            best = training['seeds'][str(seed)][arm]['best']['checkpoint']
            if read(Path(best['path']).with_suffix('.identity.json')) != best:
                raise RuntimeError('Selected checkpoint identity changed')
            saved = load_checkpoint(Path(best['path']))
            restore(context,head,saved['state'])
            bundle = runtime.features('val_compare')
            metrics,rows = evaluate(runtime,context,head,arm,bundle,final=True,label=f'{seed}/{arm} 最终比较')
            path = attempt/'samples.jsonl'
            save_rows(path,rows)
            runtime.postcheck(f'compare_{seed}_{arm}')
            write(done,{'status':'PASS','samples':identity(path),'checkpoint':best,'metrics':metrics})
            results[seed,arm] = rows
            del context,head,optimizer,parameters,bundle,saved
            gc.collect()
            torch.cuda.empty_cache()
    runtime.guard()
    out = runtime.out/'evaluation'
    build_report(results,out,training)
    # Reuse frozen references with explicit scope; no hard-cascade rerun.
    write(out/'historical_references.json',{'hard_reference':read(SOURCE/'hard_reference_report.json'),
          'hard_reference_scope':'Original shared K2/K3 subset, not overall 1024-scene performance',
          'g5_r1_reference':str(ROOT/'outputs_e2e/unified/e2e_g5_r1/20260920_approved/evaluation/comparison_report.json')})
    from 统一模型代码.gates.g5.r2.g5_r2_figures import plot_cases
    plot_cases(results,out)
    runtime.postcheck('final')
    runtime.guard()
    outputs = [identity(p) for p in sorted(out.rglob('*')) if p.is_file()]
    write(out/'final_audit_report.json',{'status':'PASS','scientific_status':'COMPLETE_FOR_REVIEW',
          'six_tracks_complete':len(results)==6,'test_executed':False,'outputs':outputs,
          'peak_system_ram_percent':runtime.peak_ram,'contract':identity(BASE/'contract.json'),
          'run_wall_seconds':time.time()-read(runtime.out/'budget.json')['started_at'],
          'training_seconds':sum(r['total_training_seconds'] for s in training['seeds'].values() for r in s.values())})
    safe_print(f'G5-R2完成。回读：{out/"运行摘要.md"}、comparison_report.json、final_audit_report.json')


def main():
    setup_environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--run',action='store_true')
    args = parser.parse_args()
    with RunLock():
        if args.register:
            register()
            return
        if args.prepare:
            try:
                prepare()
            except BaseException as exc:
                write(BASE/f'engineering/preparation_failure_{time.time_ns()}.json',
                      {'status':'ENGINEERING_FAILED','error':repr(exc),'traceback':traceback.format_exc()})
                raise
            return
        if args.check_only:
            check_ready(Runtime(BASE/'precheck'))
            safe_print('当前合同、准备测试和预算检查通过；未启动正式训练。')
            return
        if not args.run:
            parser.error('请从运行入口/E2E/G5_R2/G5_R2一键运行.py启动')
        out = BASE/'run'
        if (out/'evaluation/final_audit_report.json').exists():
            safe_print(f'已完成，不重复运行。请回读：{out/"evaluation"}')
            return
        attempt = out/f'logs/attempt_{time.time_ns()}'
        attempt.mkdir(parents=True)
        with (attempt/'run.log').open('x',encoding='utf-8') as log:
            stdout,stderr = sys.stdout,sys.stderr
            sys.stdout,sys.stderr = Log(stdout,log),Log(stderr,log)
            started = time.time()
            try:
                runtime = Runtime(out)
                prep = check_ready(runtime)
                budget = out/'budget.json'
                if budget.exists():
                    deadline = read(budget)['deadline']
                else:
                    deadline = started+CONFIG['wall_seconds']-prep['seconds']
                    write(budget,{'started_at':started,'deadline':deadline,'preparation_seconds':prep['seconds'],
                                  'policy':'absolute deadline; interruption does not reset budget'})
                runtime.deadline = deadline
                runtime.guard()
                safe_print('G5-R2启动：三组×两seed；20秒单行进度；完整epoch恢复；test封存。')
                execute(runtime)
                write(attempt/'exit_receipt.json',{'status':'COMPLETED','seconds':time.time()-started})
                safe_print(f'本次运行耗时 {(time.time()-started)/3600:.2f} 小时。')
            except BaseException as exc:
                write(attempt/'failure_report.json',{'status':'STOPPED','error':repr(exc),'traceback':traceback.format_exc()})
                write(attempt/'exit_receipt.json',{'status':'STOPPED','seconds':time.time()-started})
                traceback.print_exc()
                raise
            finally:
                sys.stdout,sys.stderr = stdout,stderr


if __name__=='__main__':
    main()
