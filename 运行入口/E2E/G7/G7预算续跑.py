"""Approved budget-only continuation; frozen scientific sources remain unchanged."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'outputs_e2e/unified/e2e_g7_compact/formal_v1_40h'
AMENDMENT=RUN/'budget_extension_60h_recovery_pathfix.json'
RECOVERY=RUN/'interruption_recovery_20260926.json'
ENTRY=ROOT/'运行入口/E2E/G7/G7正式训练一键运行.py'
LIMIT=60*3600
sys.path.insert(0,str(ROOT))
os.environ['PYTHONUTF8']='1'
os.environ['PYTHONIOENCODING']='utf-8:replace'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):
        stream.reconfigure(encoding='utf-8',errors='replace')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(path):
    path=Path(path).resolve()
    data=path.read_bytes()
    return dict(path=str(path),size_bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def check_row(row):
    if identity(row['path'])!=row:
        raise RuntimeError(f'Identity changed: {row["path"]}')


def status(name,**fields):
    path=RUN/'budget_extension_status.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(status=name,time=time.time(),**fields),
                              ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def validate():
    amendment=read(AMENDMENT)
    if amendment['effective_wall_seconds']!=LIMIT or amendment['original_wall_seconds']!=144000:
        raise RuntimeError('Unapproved budget amendment')
    check_row(amendment['original_contract']);check_row(amendment['runner'])
    check_row(amendment['previous_amendment']);check_row(amendment['recovery'])
    contract=read(RUN/'contract.json')
    if contract['config']['wall_seconds']!=144000:
        raise RuntimeError('Unexpected original budget')
    for row in contract['files']:
        check_row(row)
    if not (RUN/'training').is_dir():
        raise RuntimeError('This continuation only resumes the existing formal training')
    return amendment


def atomic_json(path,value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(temp,path)


def recovery_values(plan,budget,execution):
    """Accept only the approved snapshot or either half of its atomic-file commit."""
    if budget not in (plan['budget_before'],plan['budget_after']):
        raise RuntimeError('预算已发生其他变化，不能再次补记本次中断')
    if execution not in (plan['execution_before'],plan['execution_after']):
        raise RuntimeError('执行状态不属于本次获批中断，请先核对')
    return plan['budget_after'],plan['execution_after']


def check_no_training_process():
    import psutil
    for proc in psutil.process_iter(['pid','name','cmdline']):
        if proc.pid==os.getpid() or 'python' not in (proc.info['name'] or '').lower():
            continue
        command=' '.join(proc.info['cmdline'] or [])
        if str(ROOT).lower() in command.lower() and any(
                name in command for name in (ENTRY.name,Path(__file__).name,'formal_run')):
            raise RuntimeError(f'已有G7进程PID={proc.pid}，不启动第二份')


def prepare_recovery(apply=False):
    import io
    import torch
    check_no_training_process()
    plan=read(RECOVERY)
    budget,execution=recovery_values(plan,read(RUN/'budget.json'),read(RUN/'execution_status.json'))
    for row in plan['pointers']:
        check_row(row)
        pointer=read(row['path'])
        check_row(pointer['checkpoint']);check_row(pointer['best']['checkpoint'])
    row=plan['resume_checkpoint']
    payload=Path(row['path']).read_bytes()
    if len(payload)!=row['size_bytes'] or hashlib.sha256(payload).hexdigest()!=row['sha256']:
        raise RuntimeError('恢复checkpoint身份不匹配')
    cp=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
    pointer=read(RUN/'training/20260922/s/completed.json')
    if (cp['seed'],cp['phase'],cp['history'][-1]['epoch'])!=(20260922,'s',17):
        raise RuntimeError('恢复轨道或epoch不匹配')
    if cp['history']!=pointer['history'] or cp['steps']!=pointer['steps']:
        raise RuntimeError('Checkpoint与提交指针不一致')
    for key in ('state','optimizer','scheduler','generator','rng'):
        if key not in cp:raise RuntimeError(f'恢复状态缺失：{key}')
    if apply:
        # Plan retains original states. Reapplying either partial commit never adds time again.
        atomic_json(RUN/'budget.json',budget)
        atomic_json(RUN/'execution_status.json',execution)
    print(f'恢复检查通过：S轨从第18轮开始；累计已用{budget["active_seconds"]/3600:.3f}小时，'
          f'剩余{(LIMIT-budget["active_seconds"])/3600:.3f}小时。',flush=True)
    return budget,execution


def allowed_resume(execution,budget):
    return (execution.get('status')=='STOPPED' and not budget.get('open',True)
            and '累计40小时预算到期' in execution.get('error','')
            and 144000<=budget.get('active_seconds',0)<LIMIT)


def extended_guard(original):
    def guard(runtime):
        original_config=runtime.config
        runtime.config={**original_config,'wall_seconds':LIMIT}
        try:
            return original(runtime)
        finally:
            runtime.config=original_config
    return guard


def check_disk():
    import psutil
    if psutil.disk_usage(str(RUN)).free<50*2**30:
        raise RuntimeError('Disk still below original 50 GiB floor')


def continue_run():
    validate()
    execution,budget=read(RUN/'execution_status.json'),read(RUN/'budget.json')
    if execution.get('status')=='COMPLETED':
        status('ALREADY_COMPLETED');return
    check_disk()
    from 统一模型代码.gates.g7 import formal_runtime as runtime
    from 统一模型代码.gates.g7 import formal_run as runner
    from 统一模型代码.gates.g7 import compact_run
    compact_run.RUN=RUN
    with compact_run.RunLock():
        if allowed_resume(execution,budget):
            check_no_training_process()
        else:
            budget,execution=prepare_recovery(apply=True)
    original_guard=runtime.Runtime.guard
    original_comparison=runner.final_comparison
    amendment_row=identity(AMENDMENT)

    def comparison(rt,training):
        result=original_comparison(rt,training)
        result['approved_budget_extension']=amendment_row
        result['effective_wall_seconds']=LIMIT
        result['interruption_recovery']=identity(RECOVERY)
        runtime.write(rt.out/'evaluation/comparison_report.json',result)
        with (rt.out/'evaluation/运行摘要.md').open('a',encoding='utf-8') as handle:
            handle.write('\n预算追加：累计60小时，保留原40小时合同与全部实际耗时；'
                         '科学配置未变。见budget_extension_60h_recovery_pathfix.json及中断恢复记录。\n')
        return result

    runtime.Runtime.guard=extended_guard(original_guard)
    runner.final_comparison=comparison
    sys.argv=[str(ENTRY)]
    try:
        status('RESUMING',effective_wall_seconds=LIMIT,prior_active_seconds=budget['active_seconds'])
        print('开始按累计60小时续跑；仅重放未提交的epoch，原训练进度条将在预检后显示。',flush=True)
        runner.main()
        if read(RUN/'execution_status.json')['status']!='COMPLETED':
            raise RuntimeError('Continuation returned without completion')
        status('COMPLETED',final_audit=identity(RUN/'evaluation/final_audit_report.json'),
               actual_budget=read(RUN/'budget.json'),approved_amendment=amendment_row)
    finally:
        runtime.Runtime.guard=original_guard
        runner.final_comparison=original_comparison


def watch(pid):
    import psutil
    validate()
    process=psutil.Process(pid)
    if not any(Path(arg).resolve()==ENTRY.resolve() for arg in process.cmdline()[1:] if not arg.startswith('-')):
        raise RuntimeError('PID is not the original G7 entry')
    creation=process.create_time()
    status('WAITING_CURRENT_RUN',pid=pid,process_created_at=creation,effective_wall_seconds=LIMIT)
    print(f'已接管预算续跑等待：原PID={pid}。不打断当前训练，仅在40小时正常退出后接续。',flush=True)
    deadline=time.monotonic()+24*3600
    while process.is_running():
        if process.create_time()!=creation:
            break
        if time.monotonic()>deadline:
            raise RuntimeError('Original process did not exit within bounded wait; inspect manually')
        time.sleep(30)
    continue_run()


def self_test():
    from types import SimpleNamespace
    from unittest.mock import patch
    from 统一模型代码.gates.g7 import formal_runtime as rt
    instance=SimpleNamespace(config=dict(rt.CONFIG),peak_ram_percent=0,out=RUN,
        budget={'active_seconds':41*3600},started=time.perf_counter())
    config=instance.config
    with patch.object(rt.psutil,'virtual_memory',return_value=SimpleNamespace(percent=20)),\
         patch.object(rt.torch.cuda,'memory_allocated',return_value=0),\
         patch.object(rt.shutil,'disk_usage',return_value=SimpleNamespace(free=100*2**30)):
        try:
            rt.Runtime.guard(instance)
        except RuntimeError as exc:
            assert '40小时' in str(exc)
        else:
            raise AssertionError('Original time gate did not stop')
        extended_guard(rt.Runtime.guard)(instance)
        assert instance.config is config and config['wall_seconds']==144000
        instance.budget['active_seconds']=61*3600
        try:
            extended_guard(rt.Runtime.guard)(instance)
        except RuntimeError as exc:
            assert '60小时' in str(exc)
        else:
            raise AssertionError('Extended gate did not stop')
        instance.budget['active_seconds']=41*3600
        with patch.object(rt.psutil,'virtual_memory',return_value=SimpleNamespace(percent=86)):
            try: extended_guard(rt.Runtime.guard)(instance)
            except RuntimeError as exc: assert 'RAM' in str(exc)
            else: raise AssertionError('RAM safeguard lost')
    assert instance.config is config
    assert allowed_resume(dict(status='STOPPED',error='累计40小时预算到期'),dict(open=False,active_seconds=144001))
    assert not allowed_resume(dict(status='STOPPED',error='Disk below50GiB reserve'),dict(open=False,active_seconds=144001))
    assert not allowed_resume(dict(status='STOPPED',error='累计40小时预算到期'),dict(open=True,active_seconds=144001))
    plan=dict(budget_before=dict(active_seconds=100,open=True),
              budget_after=dict(active_seconds=200,open=False),
              execution_before=dict(status='RUNNING'),execution_after=dict(status='RECOVERY_READY'))
    for b in (plan['budget_before'],plan['budget_after']):
        for e in (plan['execution_before'],plan['execution_after']):
            assert recovery_values(plan,b,e)==(plan['budget_after'],plan['execution_after'])
    try:recovery_values(plan,dict(active_seconds=300,open=True),plan['execution_after'])
    except RuntimeError:pass
    else:raise AssertionError('Unexpected new run was accepted')
    print('PASS：时间门、RAM限制、恢复记账幂等及半完成写入恢复检查。',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--wait-pid',type=int)
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if args.check:
        validate();check_disk();prepare_recovery(apply=False)
        print('原合同和恢复检查通过；未改预算、未启动训练。',flush=True);return
    # A separate Windows lock prevents two waiting helpers. Original training lock remains untouched.
    import msvcrt
    with (RUN/'budget_extension.lock').open('a+b') as handle:
        if handle.tell()==0:handle.write(b'0');handle.flush()
        handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
        try:
            if args.wait_pid:watch(args.wait_pid)
            else:continue_run()
        except BaseException as exc:
            status('NEEDS_ATTENTION',error=repr(exc));raise
        finally:
            handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)


if __name__=='__main__':main()
