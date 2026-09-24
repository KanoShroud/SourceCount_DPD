"""G6-P1 orchestration. Only --run starts the user-operated formal experiment."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.r2.e2e_g5_r2 import Log
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import save_rows
from 统一模型代码.gates.g5.e2e_g5_train import load_checkpoint
from 统一模型代码.gates.g6.g6_p1_runtime import (
    BASE, CONFIG, ARMS, SEEDS, Runtime, RunLock, setup_environment, register, read, write, safe_print, identity)
from 统一模型代码.gates.g6.g6_p1_prepare import prepare
from 统一模型代码.gates.g6.g6_p1_train import train_all, evaluate, restore, build_report


def verify_cache_extension(pilot, current):
    if any(current.get(key) != row for key,row in pilot.items()):
        raise RuntimeError('Pilot cache identities changed during cache extension')


def check_ready(runtime):
    runtime.preflight()
    prep = read(BASE/'preparation_report.json')
    verified_read(prep['report'],runtime.out/'anomalies')
    if prep['status'] != 'ENGINEERING_PASS' or prep['contract_sha256'] != identity(BASE/'contract.json')['sha256']:
        raise RuntimeError('No valid engineering result for current contract')
    if not prep['forecast_within_budget']:
        raise RuntimeError('FORECAST_EXCEEDS_12H：未启动训练；请先审批预算或新方案')
    remaining = sum(n-len(runtime.cache_rows(s)) for s,n in [('train',4096),('val_select',512),('val_compare',1024)])
    if shutil.disk_usage(BASE).free < remaining*prep['cache_bytes_per_sample']+55*1024**3:
        raise RuntimeError('Insufficient disk forecast')
    for split,rows in prep['pilot_cache_rows'].items():
        verify_cache_extension(rows,runtime.cache_rows(split))
    return prep


def execute(runtime):
    runtime.audit_raw()
    for split in ('train','val_select','val_compare'):
        bundle = runtime.features(split)
        runtime.ensure_cache(split,range(len(bundle[1].counts)),bundle[1])
        del bundle
        gc.collect()
    runtime.audit_raw()
    # Freeze generated cache indexes before training, then always verify them on resume.
    indexes = {s:identity(BASE/f'physical_cache/{s}/index.json') for s in ('train','val_select','val_compare')}
    frozen = runtime.out/'cache_contract.json'
    if frozen.exists() and read(frozen) != indexes:
        raise RuntimeError('Physical cache index changed after formal start')
    if not frozen.exists():
        write(frozen,indexes)
    training = train_all(runtime)
    results = {}
    out = runtime.out/'evaluation'
    out.mkdir(exist_ok=True)
    for seed in SEEDS:
        for arm in ARMS:
            best = training['seeds'][str(seed)][arm]['best']['checkpoint']
            done = out/f'{seed}_{arm}_complete.json'
            if done.exists():
                marker = read(done)
                if marker['checkpoint'] != best:
                    raise RuntimeError('Evaluation checkpoint differs')
                results[seed,arm] = [json.loads(s) for s in verified_read(marker['samples'],runtime.out/'anomalies').decode('utf-8').splitlines()]
                continue
            context,head,optimizer,params = runtime.context(seed,arm)
            verified_read(best,runtime.out/'anomalies')
            saved = load_checkpoint(Path(best['path']))
            restore(context,head,saved['state'])
            bundle = runtime.features('val_compare')
            metrics,rows = evaluate(runtime,context,head,arm,bundle,split='val_compare',label=f'{seed}/{arm} 最终比较')
            path = out/f'{seed}_{arm}_{time.time_ns()}.jsonl'
            save_rows(path,rows)
            runtime.postcheck(f'compare_{seed}_{arm}')
            write(done,{'checkpoint':best,'samples':identity(path),'metrics':metrics})
            results[seed,arm] = rows
            del context,head,optimizer,params,bundle,saved
            gc.collect(); torch.cuda.empty_cache()
    build_report(results,out,training)
    runtime.postcheck('final')
    files = [identity(p) for p in sorted(out.iterdir()) if p.is_file()]
    for row in files:
        verified_read(row,runtime.out/'anomalies')
    write(out/'final_audit_report.json',dict(status='PASS',six_tracks_complete=len(results)==6,
        outputs=files,contract=identity(BASE/'contract.json'),test_executed=False,
        peak_ram_percent=runtime.peak_ram,peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
    safe_print(f'G6-P1完成。请回读：{out/"运行摘要.md"}、comparison_report.json、final_audit_report.json')


def main():
    setup_environment()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('register','prepare','check-only','run'):
        group.add_argument('--'+flag,action='store_true')
    args = parser.parse_args()
    with RunLock():
        if args.register:
            register(); return
        if args.prepare:
            try:
                prepare()
            except BaseException:
                write(BASE/f'engineering/preparation_failure_{time.time_ns()}.json',
                      {'status':'FAILED','traceback':traceback.format_exc()})
                raise
            return
        if args.check_only:
            check_ready(Runtime(BASE/'precheck'))
            safe_print('合同、工程检查和资源预测通过；未启动六轨。'); return
        out = BASE/'run'
        if (out/'evaluation/final_audit_report.json').exists():
            safe_print(f'已完成，不重复运行：{out/"evaluation"}'); return
        # Check readiness outside the formal tree so a blocked click creates no training state.
        prep = check_ready(Runtime(BASE/'precheck'))
        out.mkdir(exist_ok=True)
        attempt = out/f'logs/{time.time_ns()}'
        attempt.mkdir(parents=True)
        budget_path = out/'budget.json'
        start = time.time()
        budget = read(budget_path) if budget_path.exists() else {
            'started_at':start,'deadline':start+CONFIG['wall_seconds']-prep['preparation_consumed_seconds'],
            'preparation_seconds':prep['preparation_consumed_seconds'],'policy':'absolute deadline; resume never resets budget'}
        write(budget_path,budget)
        runtime = Runtime(out,deadline=budget['deadline'])
        with (attempt/'run.log').open('x',encoding='utf-8') as f:
            stdout,stderr = sys.stdout,sys.stderr
            sys.stdout,sys.stderr = Log(stdout,f),Log(stderr,f)
            try:
                runtime.preflight()
                safe_print('G6-P1：缓存准备→三轨×两seed→统一评价；20秒单行进度，完整epoch恢复。')
                execute(runtime)
            except BaseException:
                write(attempt/'failure.json',{'status':'STOPPED','traceback':traceback.format_exc()})
                raise
            finally:
                write(attempt/'timing.json',{'seconds':time.time()-start})
                sys.stdout,sys.stderr = stdout,stderr


if __name__ == '__main__':
    main()
