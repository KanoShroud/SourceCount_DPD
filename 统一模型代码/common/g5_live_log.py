"""Tail child UTF-8 file logs into PyCharm without a stdout pipe or logging thread."""
import codecs
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from 统一模型代码.gates.g5.e2e_g5_train import g4, resource_guard

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT/'outputs_e2e/unified/g5_prefetch_probe/20260919_230910/report.json'


def estimate_remaining(run, manifest):
    """Short-probe projection for display only; never replaces the approved budget gate."""
    bench=g4.read_json(BENCHMARK)
    pilot=g4.read_json(run/'pilot_report.json')
    result={}
    for target in (48,64):
        seconds=0.0
        for track in ('sg','e2e'):
            rows=[r for r in bench['rows'] if r['track']==track and r['batch']==4 and r['prefetch']]
            if len(rows)!=2:
                raise RuntimeError('Missing registered batch4 prefetch timing evidence')
            epoch_seconds=manifest['config']['counts']['train']*sum(r['seconds'] for r in rows)/1024
            for seed in manifest['config']['training_seeds']:
                path=run/'training'/str(seed)/track/'history.json'
                done=max((r['epoch'] for r in g4.read_json(path)),default=0) if path.exists() else 0
                remaining=max(0,target-done)
                seconds+=remaining*(epoch_seconds+pilot['tracks'][track]['validation_512_seconds']/2)
        result[str(target)]=dict(training_and_selection_hours=seconds/3600,
                               planning_hours=seconds*1.25/3600+1)
    return result


def emit(text):
    try:
        sys.stdout.write(text)
    except UnicodeEncodeError:
        sys.stdout.write(text.encode(sys.stdout.encoding or 'utf-8',errors='replace').decode(
            sys.stdout.encoding or 'utf-8',errors='replace'))
    sys.stdout.flush()


class Progress:
    def __init__(self):
        self.previous = None

    def update(self, row, now):
        key=(row['seed'],row['track'],row['epoch'],row['batch'])
        if self.previous and self.previous[0]==key:
            return
        eta='估算中'
        if self.previous and self.previous[0][:3]==key[:3]:
            delta=key[3]-self.previous[0][3]
            if delta>0:
                eta=f"{(row['batches']-row['batch'])*(now-self.previous[1])/delta:.0f}秒"
        self.previous=(key,now)
        emit(f"[训练] seed={row['seed']} {row['track'].upper()} | epoch={row['epoch']} "
             f"| batch={row['batch']}/{row['batches']} ({100*row['batch']/row['batches']:.1f}%) "
             f"| 累计更新={row['optimizer_steps']} | 本轮训练剩余≈{eta}\n")


def run_stage(run, manifest, name, arguments):
    logs=run/'execution_logs';logs.mkdir(exist_ok=True)
    path=logs/f'{name}.log'
    env=dict(os.environ,PYTHONUTF8='1',PYTHONIOENCODING='utf-8:replace')
    started=time.time();clock=time.monotonic();last_heartbeat=clock;last_guard=float('-inf')
    progress=Progress();error=None
    with path.open('xb') as log:
        process=subprocess.Popen([sys.executable,'-u',*arguments],cwd=ROOT,env=env,
                                 stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        g4.write_json(run/'execution_status.json',dict(status='RUNNING',stage=name,pid=process.pid,
                      started_at=started,log=str(path),test_executed=False))
        emit(f'[阶段开始] {name} | PID={process.pid} | 日志：{path}\n')
        with path.open('rb') as reader:
            decoder=codecs.getincrementaldecoder('utf-8')(errors='replace')
            def drain(final=False):
                while chunk := reader.read(65536):
                    emit(decoder.decode(chunk))
                if final:
                    emit(decoder.decode(b'',final=True))
            try:
                while True:
                    drain()
                    now=time.monotonic()
                    if name.endswith('_training'):
                        progress_path=run/'training_progress.json'
                        try:
                            row=json.loads(progress_path.read_text(encoding='utf-8'))
                        except (FileNotFoundError,json.JSONDecodeError):
                            row=None  # Writer can be between truncate and write; retry next poll.
                        if row and row.get('updated_at',0)>=started:
                            progress.update(row,now)
                    if process.poll() is not None:
                        drain(final=True)
                        break
                    if now-last_guard>=10:
                        resource_guard(run,manifest);last_guard=now
                    if now-last_heartbeat>=60:
                        emit(f'[运行中] {name} | 本阶段已耗时 {(now-clock)/60:.1f} 分钟\n')
                        last_heartbeat=now
                    time.sleep(1)
            except BaseException as exc:
                error=repr(exc)
                if process.poll() is None:
                    process.terminate()
                    try:process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill();process.wait(timeout=10)
                drain(final=True)
                raise
            finally:
                g4.write_json(logs/f'{name}.json',dict(stage=name,returncode=process.poll(),
                    seconds=time.time()-started,log=str(path),error=error))
    result=g4.read_json(logs/f'{name}.json')
    if result['returncode']!=0:
        raise RuntimeError(f'Stage failed: {name}; inspect {path}')
    emit(f"[阶段完成] {name} | 耗时 {result['seconds']/60:.1f} 分钟\n")
    return result
