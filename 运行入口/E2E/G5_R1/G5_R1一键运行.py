"""PyCharm直接运行，无需参数。独立G5-R1评价，不训练，不修改原G5。

--check-only 只检查登记合同。已有运行目录时禁止自动覆盖或重跑。
"""

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[3]))
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
PYTHON=Path('D:/Software/anaconda3/envs/PyTorch/python.exe')
BASE=ROOT/'outputs_e2e/unified/e2e_g5_r1/20260920_approved'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    if Path(sys.executable).resolve()!=PYTHON.resolve():
        raise RuntimeError(f'请选择项目解释器：{PYTHON}')
    if not (BASE/'contract.json').is_file():
        raise RuntimeError('缺少预登记合同，请交由Codex检查，不手改配置')
    command=[str(PYTHON),'-u',str(ROOT/'统一模型代码/gates/g5/r1/e2e_g5_r1.py')]
    env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8:replace','CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    if args.check_only:
        subprocess.run(command+['--check-only'],cwd=ROOT,env=env,check=True)
        return
    # Exclusive file creation: never reset a failed/aborted attempt or its four-hour budget.
    started=time.time()
    receipt=BASE/'launch_receipt.json'
    with receipt.open('x',encoding='utf-8') as handle:
        json.dump({'status':'LAUNCHED','started_at':started,'wall_limit_seconds':14400},handle)
    if (BASE/'evaluation').exists():
        raise RuntimeError('已有评价目录，禁止自动覆盖。请回读现有结果。')
    print('启动G5-R1：六轨固定模型评价，四小时上限；可在PyCharm实时查看每batch进度。',flush=True)
    process=subprocess.Popen(command+['--worker','--started',str(started)],cwd=ROOT,env=env)
    try:
        code=process.wait(timeout=max(1,14400-(time.time()-started)))
        status='COMPLETED' if code==0 else 'STOPPED'
    except (subprocess.TimeoutExpired,KeyboardInterrupt) as exc:
        process.terminate()
        process.wait(timeout=30)
        code=process.returncode
        status='BUDGET_EXCEEDED' if isinstance(exc,subprocess.TimeoutExpired) else 'USER_INTERRUPTED'
    (BASE/'exit_receipt.json').write_text(json.dumps({'status':status,'returncode':code,
        'elapsed_seconds':time.time()-started},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'状态：{status}；结果目录：{BASE / "evaluation"}',flush=True)
    if status!='COMPLETED':
        raise RuntimeError('运行已停止；保留现场，请回读failure_report.json/run.log及exit_receipt.json，不重复点击')


if __name__=='__main__':
    main()
