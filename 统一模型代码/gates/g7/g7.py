"""G7 orchestration. Formal training starts only with --run after preparation passes."""
import argparse
import gc
import json
import sys
import time
import traceback

import torch

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.r2.e2e_g5_r2 import Log
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import save_rows
from 统一模型代码.gates.g7.g7_runtime import (
    BASE, CONFIG, ARMS, SEEDS, Runtime, RunLock, setup_environment, register,
    read, write, safe_print, identity)
from 统一模型代码.gates.g7.g7_train import train_all, evaluate, restore, load_registered, build_report


def check_ready(runtime):
    runtime.preflight()
    prep = read(BASE/'preparation_report.json')
    verified_read(prep['report'], runtime.out/'anomalies')
    if prep['status'] != 'ENGINEERING_PASS' or prep['contract_sha256'] != identity(BASE/'contract.json')['sha256']:
        raise RuntimeError('当前合同没有通过第一步准备验证；不启动训练')
    if not all(prep.get(k, False) for k in ('forecast_within_budget', 'input_check_pass', 'resources_pass')):
        raise RuntimeError(f'G7未启动：输入检查={prep.get("input_check_pass")}；'
            f'20/24轮预计{prep["forecast_base_seconds"]/3600:.2f}/{prep["forecast_max_seconds"]/3600:.2f}h，'
            f'上限12h；磁盘通过={prep.get("resources_pass")}。请先确认资源方案，不自动降配置。')
    return prep


def execute(runtime):
    runtime.audit_raw()
    cache_reports = {}
    for seed in SEEDS:
        safe_print(f'G7阶段：核对并准备{seed}冻结候选与局部统计缓存')
        cache_reports[str(seed)] = runtime.ensure_all_cache(seed)
    write(runtime.out/'cache_preparation_report.json', cache_reports)
    training = train_all(runtime)
    results = {}
    out = runtime.out/'evaluation'
    out.mkdir(exist_ok=True)
    safe_print('G7阶段：六轨训练结束，开始固定checkpoint开发集比较')
    for seed in SEEDS:
        for arm in ARMS:
            best = training['seeds'][str(seed)][arm]['best']['checkpoint']
            done = out/f'{seed}_{arm}_complete.json'
            if done.exists():
                marker = read(done)
                if marker['checkpoint'] != best:
                    raise RuntimeError('Evaluation checkpoint differs from selected checkpoint')
                verified_read(best, runtime.out/'anomalies')
                results[seed, arm] = [json.loads(line) for line in
                    verified_read(marker['samples'], runtime.out/'anomalies').decode('utf-8').splitlines()]
                continue
            context, head, optimizer, params = runtime.context(seed, arm)
            saved = load_registered(best, runtime.out)
            restore(context, head, saved['state'])
            bundle = runtime.features('val_compare')
            metrics, rows = evaluate(runtime, context, head, arm, bundle, seed,
                                     split='val_compare', label=f'{seed}/{arm} 最终比较')
            path = out/f'{seed}_{arm}_{time.time_ns()}.jsonl'
            save_rows(path, rows)
            runtime.postcheck(f'compare_{seed}_{arm}')
            write(done, {'checkpoint': best, 'samples': identity(path), 'metrics': metrics})
            results[seed, arm] = rows
            del context, head, optimizer, params, saved, bundle
            gc.collect()
            torch.cuda.empty_cache()
    references = {int(seed): [json.loads(line) for line in
        verified_read(row, runtime.out/'anomalies').decode('utf-8').splitlines()]
        for seed, row in runtime.g7_contract['frozen_p2_b_samples'].items()}
    hard = json.loads(verified_read(runtime.g7_contract['hard_reference_report'], runtime.out/'anomalies'))
    build_report(results, out, training, references, hard['references'])
    runtime.postcheck('final')
    files = [identity(p) for p in sorted(out.iterdir()) if p.is_file()]
    for row in files:
        verified_read(row, runtime.out/'anomalies')
    write(out/'final_audit_report.json', dict(status='PASS', six_tracks_complete=len(results) == 6,
        outputs=files, contract=identity(BASE/'contract.json'), test_executed=False,
        peak_ram_percent=runtime.peak_ram, peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
    safe_print(f'G7完成。请回读：{out/"运行摘要.md"}、comparison_report.json、final_audit_report.json')


def run_formal(prep):
    out = BASE/'run'
    out.mkdir(exist_ok=True)
    attempt = out/f'logs/{time.time_ns()}'
    attempt.mkdir(parents=True)
    budget_path = out/'budget.json'
    budget = read(budget_path) if budget_path.exists() else {
        'preparation_seconds': prep['preparation_consumed_seconds'], 'active_seconds': 0.0,
        'policy': 'Cumulative active process time, including cache/training/evaluation/interrupted attempts; offline time excluded'}
    remaining = CONFIG['wall_seconds']-budget['preparation_seconds']-budget['active_seconds']
    if remaining <= 0:
        raise RuntimeError('G7累计12小时预算已用尽；未自动续期')
    # An unclosed reservation means the process was killed; conservatively charge its remaining allowance.
    if budget.get('open_attempt'):
        raise RuntimeError('上次进程未正常关闭预算记录；需回读日志核实已用时间，禁止静默重置预算')
    start = time.time()
    budget['open_attempt'] = {'path': str(attempt), 'started_at': start, 'remaining_seconds': remaining}
    write(budget_path, budget)
    runtime = Runtime(out, deadline=start+remaining)
    with (attempt/'run.log').open('x', encoding='utf-8') as stream:
        stdout, stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Log(stdout, stream), Log(stderr, stream)
        try:
            runtime.preflight()
            safe_print(f'G7：F/S/E × 两seed；20轮/条件24轮；本次剩余预算{remaining/3600:.2f}h。')
            execute(runtime)
        except BaseException:
            write(attempt/'failure.json', {'status': 'STOPPED', 'traceback': traceback.format_exc()})
            raise
        finally:
            elapsed = time.time()-start
            budget['active_seconds'] += elapsed
            budget.pop('open_attempt', None)
            write(budget_path, budget)
            write(attempt/'timing.json', {'seconds': elapsed})
            sys.stdout, sys.stderr = stdout, stderr


def main():
    setup_environment()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ('register', 'prepare', 'check-only', 'run'):
        group.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    with RunLock():
        if args.register:
            register()
            return
        if args.prepare:
            from 统一模型代码.gates.g7.g7_prepare import prepare
            prepare()
            return
        if args.check_only:
            check_ready(Runtime(BASE/'precheck'))
            safe_print('G7准备、合同和资源检查通过；没有启动正式训练。')
            return
        audit_path = BASE/'run/evaluation/final_audit_report.json'
        if audit_path.exists():
            audit = read(audit_path)
            if audit.get('status') != 'PASS' or not audit.get('six_tracks_complete'):
                raise RuntimeError('已有不完整审计，不得当作完成')
            for row in audit['outputs']:
                verified_read(row, BASE/'precheck/anomalies')
            safe_print(f'G7已完成，不重复训练：{audit_path.parent}')
            return
        prep = check_ready(Runtime(BASE/'precheck'))
        run_formal(prep)


if __name__ == '__main__':
    main()
