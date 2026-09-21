"""PyCharm直接运行：恢复本次G5，不重新生成数据，不读取test。

仅在首个SG停止审计完成后可用。--check-only只检查入口，不训练。
不改变原始48小时预算；若预算不足，停止并申请新的明确授权。
"""

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[3]))
import argparse
from contextlib import contextmanager
import json
import msvcrt
import os
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'outputs_e2e/unified/e2e_g5/20260919_approved'
PYTHON = Path('D:/Software/anaconda3/envs/PyTorch/python.exe')


@contextmanager
def execution_lock():
    """OS lock is released even after process termination; no stale PID lock to delete."""
    with (RUN/'manual_execution.lock').open('a+b') as handle:
        if handle.tell()==0:
            handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:
            raise RuntimeError('另一份一键入口持有运行锁，禁止重复启动') from exc
        try:
            yield
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def finishing_mode():
    if not (RUN/'engineering_v4/contract.json').exists():
        return None
    training=read(RUN/'training_report.json')
    comparison=read(RUN/'comparison_report.json')
    if training['status']!='PASS' or comparison['status'] not in (
            'FEEDBACK_SUPPORTED_ON_DEV','DIRECTIONAL_BUT_INCONCLUSIVE','TRADEOFF_OR_REGRESSION'):
        raise RuntimeError('恢复收尾的既有训练/比较证据不完整')
    if read(RUN/'identity_after_training.json')['status']!='PASS':
        raise RuntimeError('训练后输入身份检查未通过')
    if (RUN/'final_audit_report.json').exists():
        if read(RUN/'final_audit_report.json')['status']!='PASS':
            raise RuntimeError('已有最终审计未通过，请先回读')
        return 'complete'
    if not (RUN/'hard_reference_report.json').exists() and (RUN/'hard_reference_recovery_v4').exists():
        raise RuntimeError('修复后的Hard参考已有未完成尝试，请先回读，不覆盖')
    return 'finishing'


def preflight():
    if Path(sys.executable).resolve() != PYTHON.resolve():
        raise RuntimeError(f'请在PyCharm选择解释器：{PYTHON}')
    mode=finishing_mode()
    for name in ('comparison', 'hard_reference', 'comparison_report.json', 'hard_reference_report.json', 'final_audit_report.json'):
        if (RUN / name).exists() and mode is None:
            raise RuntimeError(f'已有评价/审计证据：{name}；先由Codex回读，不自动覆盖或重评')
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid == os.getpid():
            continue
        args = process.info['cmdline'] or []
        if any(Path(arg).name in ('e2e_g5_train.py', 'G5一键继续.py') for arg in args):
            raise RuntimeError(f'已有训练或继续入口运行，PID={process.pid}；禁止重复启动')
    receipt = read(RUN / 'user_pause_audit.json')
    if receipt.get('status') != 'SG48_COMPLETE_STOPPED' or receipt.get('test_executed') is not False:
        raise RuntimeError('缺少首个SG完整停止审计，不启动后续任务')
    if not (RUN/'engineering_v2/contract.json').is_file():
        raise RuntimeError('缺少工程v2冻结合同；禁止回退到旧加载路径')
    from 统一模型代码.gates.g5.e2e_g5_contract import verify_code
    from 统一模型代码.common.g5_verified_io import verified_read
    verify_code(RUN)
    ready = read(RUN/'engineering_v2/recovery_validation.identity.json')
    if json.loads(verified_read(ready,RUN/'anomalies'))['status']!='PASS':
        raise RuntimeError('恢复流程验证未通过')
    manifest = read(RUN / 'manifest.json')
    if mode:
        if mode=='finishing':
            from 统一模型代码.gates.g5.e2e_g5_train import resource_guard
            resource_guard(RUN,manifest)
            if manifest['created_at']+manifest['config']['wall_limit_seconds']-time.time()<600:
                raise RuntimeError('收尾剩余预算不足10分钟，需审批预算修订')
        print('G5已完成，禁止重复训练/评价。' if mode=='complete' else
              '六轨训练和比较已完成；仅恢复Hard参考核对与最终审计，不重训、不重评比较集。',flush=True)
        return {**manifest,'_entry_mode':mode}
    sg = read(RUN / 'training/20260921/sg/report_epoch48.json')
    if sg['status'] != 'PASS' or sg['completed_epoch'] != 48 or sg['optimizer_steps'] != 49152:
        raise RuntimeError('首个SG48结果不完整')
    verified_read(sg['best'],RUN/'anomalies')
    verified_read(read(RUN/'training/20260921/sg/last.identity.json'),RUN/'anomalies')
    remaining = manifest['created_at'] + manifest['config']['wall_limit_seconds'] - time.time()
    pilot = read(RUN / 'pilot_report.json')
    # 保守预计所有轨道延长64轮，已完成轮次不重算；不重置原始预算。
    seconds = 0.0
    for seed in manifest['config']['training_seeds']:
        for track in ('sg', 'e2e'):
            history_path = RUN / 'training' / str(seed) / track / 'history.json'
            done = max((r['epoch'] for r in read(history_path)), default=0) if history_path.exists() else 0
            arm = pilot['tracks'][track]
            per_epoch = arm['train_128_seconds'] * 8 + arm['validation_512_seconds'] / 2
            seconds += max(0, 64 - done) * per_epoch
    required = seconds * 1.15 + 600
    from 统一模型代码.common.g5_live_log import estimate_remaining
    estimate=estimate_remaining(RUN,manifest)
    print(f"新加载器短测规划估计：基础48轮剩余约 {estimate['48']['planning_hours']:.1f} 小时；"
          f"全部延长64轮约 {estimate['64']['planning_hours']:.1f} 小时（非保证）。",flush=True)
    print(f'原预算剩余 {remaining/3600:.2f} 小时；保守恢复需求 {required/3600:.2f} 小时', flush=True)
    if remaining < required:
        raise RuntimeError('暂停后剩余预算不足；请让Codex核对并获批预算修订，不要手改manifest')
    from 统一模型代码.common.g5_runtime_v2 import ram_guard
    ram_guard()
    print('工程v2：按样本校验＋单batch预取；batch=4；整机RAM 85%预警即停止。',flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.check_only:
        preflight()
        print('入口预检通过；未训练、未评价。', flush=True)
        return
    with execution_lock():
        execute(preflight())


def execute(manifest):
    from 统一模型代码.gates.g5.e2e_g5_execute import g4
    from 统一模型代码.common.g5_live_log import run_stage
    if manifest.get('_entry_mode')=='complete':
        print(f'全部完成，请回读：{RUN / "final_audit_report.json"}',flush=True)
        return
    tag = 'manual_' + time.strftime('%Y%m%d_%H%M%S')
    stages = [
        ('input_check', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'verify-inputs', '--phase', tag]),
        ('training', ['统一模型代码/gates/g5/e2e_g5_train.py']),
        ('post_check', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'verify-inputs', '--phase', 'after_training']),
        ('comparison', ['统一模型代码/gates/g5/e2e_g5_evaluate.py', '--stage', 'compare']),
        ('hard_reference', ['统一模型代码/gates/g5/e2e_g5_evaluate.py', '--stage', 'hard']),
        ('final_audit', ['统一模型代码/gates/g5/audits/verify_e2e_g5_final.py']),
    ]
    if manifest.get('_entry_mode')=='finishing':
        stages=[stage for stage in stages if stage[0]=='final_audit' or
                (stage[0]=='hard_reference' and not (RUN/'hard_reference_report.json').exists())]
    receipt_path = RUN / f'{tag}_receipt.json'
    completed = []
    try:
        for number,(name, command) in enumerate(stages,1):
            print(f'总阶段 {number}/{len(stages)}：{name}；日志目录：{RUN / "execution_logs"}', flush=True)
            completed.append(run_stage(RUN, manifest, tag + '_' + name, [*command, '--run', str(RUN)]))
    except BaseException as exc:
        result = {'status': 'STOP', 'error': repr(exc), 'completed': completed, 'test_executed': False}
        g4.write_json(receipt_path, result)
        g4.write_json(RUN / 'execution_status.json', result)
        raise
    result = {'status': 'EXECUTION_COMPLETE_ANALYSIS_PENDING', 'completed': completed, 'test_executed': False}
    g4.write_json(receipt_path, result)
    g4.write_json(RUN / 'execution_status.json', result)
    print(f'运行完成，请让Codex读取：{receipt_path}，以及同目录最终审计、比较和训练报告。', flush=True)


if __name__ == '__main__':
    main()
