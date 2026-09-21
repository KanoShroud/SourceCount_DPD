"""只读观察正在运行的G5，不启动/终止训练；在PyCharm直接运行即可。"""

import sys as _path_sys
from pathlib import Path as _PathRoot
_path_sys.path.insert(0, str(_PathRoot(__file__).resolve().parents[3]))
from pathlib import Path
import json
import os
import shutil
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'outputs_e2e/unified/e2e_g5/20260919_approved'


def read(name):
    return json.loads((RUN / name).read_text(encoding='utf-8'))


def main():
    status = read('execution_status.json')
    if status.get('stage') != 'training_cuda_init' or status.get('status') != 'RUNNING':
        raise RuntimeError('当前不是预期的训练阶段；不采样其他进程')
    process = psutil.Process(status['pid'])
    if not any('e2e_g5_train.py' in arg for arg in process.cmdline()):
        raise RuntimeError('进程身份不匹配')
    stamp = time.strftime('%Y%m%d_%H%M%S')
    output = RUN / 'performance_observations' / stamp
    output.mkdir(parents=True, exist_ok=False)
    gpu = shutil.which('nvidia-smi')
    rows = []
    env = dict(os.environ, PYTHONUTF8='1', PYTHONIOENCODING='utf-8:replace')
    for index in range(12):
        cpu = process.cpu_times()
        row = {'time': time.time(), 'process_created': process.create_time(),
               'cpu_seconds': cpu.user + cpu.system,
               'logical_cpus': psutil.cpu_count(), 'rss': process.memory_info().rss,
               'io': process.io_counters()._asdict(), 'progress': read('training_progress.json')}
        if gpu:
            result = subprocess.run([gpu, '--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                    encoding='utf-8', errors='replace', timeout=5, env=env,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
            row['gpu_csv'] = result.stdout.strip()
            row['gpu_returncode'] = result.returncode
        with (output / 'samples.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        rows.append(row)
        print(f'采样 {index+1}/12', flush=True)
        if index < 11:
            time.sleep(5)
    elapsed = rows[-1]['time'] - rows[0]['time']
    cpu_delta = rows[-1]['cpu_seconds'] - rows[0]['cpu_seconds']
    summary = {'status': 'OBSERVATION_ONLY', 'seconds': elapsed,
               'process_cpu_one_core_percent': 100 * cpu_delta / elapsed,
               'process_cpu_machine_percent': 100 * cpu_delta / elapsed / psutil.cpu_count(),
               'gpu_samples': [r.get('gpu_csv') for r in rows],
               'note': '短窗口采样，不是完整训练平均；进程IO和加载器字节均不等于物理磁盘流量。',
               'training_modified': False, 'test_executed': False}
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f'结果：{output}', flush=True)


if __name__ == '__main__':
    main()
