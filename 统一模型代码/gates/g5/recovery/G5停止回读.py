"""首个SG48完成且进程停止后的一次性证据审计；不启动训练、不读取test。"""

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[4]))
import json
from pathlib import Path
import time

import psutil

from 统一模型代码.gates.g5.e2e_g5_contract import verify_code, verify_inputs
from 统一模型代码.gates.g5.e2e_g5_train import load_checkpoint

RUN = Path(__file__).resolve().parents[4] / 'outputs_e2e/unified/e2e_g5/20260919_approved'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    destination = RUN / 'user_pause_audit.json'
    if destination.exists():
        raise FileExistsError('停止审计已存在，不覆盖')
    for process in psutil.process_iter(['pid', 'cmdline']):
        if any(Path(arg).name == 'e2e_g5_train.py' for arg in (process.info['cmdline'] or [])):
            raise RuntimeError(f'训练进程仍运行：{process.pid}')
    arm = RUN / 'training/20260921/sg'
    report = read(arm / 'report_epoch48.json')
    history = read(arm / 'history.json')
    assert report['status'] == 'PASS' and report['completed_epoch'] == 48
    assert report['optimizer_steps'] == 49152
    assert [r['epoch'] for r in history] == list(range(49))
    assert [r['epoch'] for r in history if r.get('validation') is not None] == list(range(0, 49, 2))
    best = min((r for r in history if r.get('validation') is not None),
               key=lambda r: (r['validation']['overall']['gospa_mean_m'], r['epoch']))
    assert best['epoch'] == report['best_epoch']
    assert best['validation']['overall']['gospa_mean_m'] == report['best_gospa']
    last = load_checkpoint(arm / 'last.pt')
    assert last['epoch'] == 48 and last['optimizer_steps'] == 49152
    assert last['history'] == history
    assert last['best_epoch'] == best['epoch']
    del last
    checkpoint = load_checkpoint(arm / 'best.pt')
    assert checkpoint['epoch'] == best['epoch']
    assert checkpoint['metrics'] == best['validation']
    del checkpoint
    other_files = [str(p.relative_to(RUN)) for p in (RUN / 'training').rglob('*')
                   if p.is_file() and not p.is_relative_to(arm)]
    if other_files:
        raise RuntimeError(f'发现其他轨道文件，需人工核查是否发生更新：{other_files}')
    verify_code(RUN)
    identity = verify_inputs(RUN, 'user_pause')
    events = []
    for path in RUN.rglob('events.jsonl'):
        events.extend(json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line)
    # 有异常时交由人工逐项核验首次字节及恢复事件，不给出假通过。
    if events:
        raise RuntimeError('存在异常事件，需回读捕获字节和恢复记录后单独完成审计')
    result = {'status': 'SG48_COMPLETE_STOPPED', 'created_at': time.time(),
              'seed': 20260921, 'track': 'sg', 'epoch': 48, 'optimizer_steps': 49152,
              'best_epoch': report['best_epoch'], 'best_gospa': report['best_gospa'],
              'budget_candidate': report['budget_candidate'], 'other_track_files': other_files,
              'checkpoint_identity_and_history_verified': True, 'input_identity_status': identity['status'],
              'read_anomaly_events': len(events), 'test_executed': False,
              'note': '用户要求暂停；不是六轨G5完成，也不作SG/E2E比较结论。'}
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
