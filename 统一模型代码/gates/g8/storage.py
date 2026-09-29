"""G8限定输出、同字节校验、原子报告与资源保护。"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'outputs_e2e/unified/e2e_g8'
FROZEN = Path('F:/SourceCount_DPD/outputs').resolve()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return dict(path=str(path), size_bytes=path.stat().st_size, sha256=h.hexdigest())


def checked_bytes(row):
    path = Path(row['path']).resolve(strict=True)
    payload = path.read_bytes()
    if len(payload) != row['size_bytes'] or digest(payload) != row['sha256']:
        raise RuntimeError(f'输入身份不一致，停止使用：{path}')
    return payload


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE.resolve()) or path.is_relative_to(FROZEN):
        raise ValueError('G8输出路径越界')
    return path


def write(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temporary.open('xb') as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def save_scene(path, scene):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        np.savez(f, **{k: v for k, v in scene.items() if k != 'metadata'},
                 metadata=np.array(json.dumps(scene['metadata'], ensure_ascii=False)))
    return identity(path)


def load_scene(row):
    with np.load(io.BytesIO(checked_bytes(row)), allow_pickle=False) as values:
        result = {k: values[k].copy() for k in values.files if k != 'metadata'}
        result['metadata'] = json.loads(str(values['metadata']))
    return result


class Guard:
    def __init__(self, out, seconds=43200):
        self.out = output_path(out)
        self.started, self.seconds = time.perf_counter(), seconds
        self.peak_ram_percent = 0.

    def __call__(self):
        usage = psutil.virtual_memory().percent
        self.peak_ram_percent = max(self.peak_ram_percent, usage)
        if usage >= 85 or shutil.disk_usage(ROOT).free < 50*2**30:
            raise RuntimeError('RAM达到85%或磁盘剩余不足50GiB')
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 14*2**30:
            raise RuntimeError('显存超过14GiB')
        if time.perf_counter()-self.started > self.seconds:
            raise RuntimeError('本次短测/运行预算到期')


def source_identity():
    paths = sorted((ROOT/'统一模型代码/gates/g8').glob('*.py'))
    paths += sorted((ROOT/'运行入口/E2E/G8').glob('*.py'))
    paths += [ROOT/'DPD_MVDR/DPD_MVDR.py', ROOT/'第四章代码/s2g3_composability.py']
    paths += sorted((ROOT/'统一模型代码/gates/g7').glob('*.py'))
    paths += [ROOT/'统一模型代码/gates/g6/coherent_dpd.py',
              ROOT/'统一模型代码/gates/g6/g6_p1_speed.py',
              ROOT/'统一模型代码/physics/fine_dpd_autograd.py']
    return [identity(path) for path in paths]
