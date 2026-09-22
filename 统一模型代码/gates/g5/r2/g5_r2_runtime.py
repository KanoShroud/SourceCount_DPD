"""R2只读输入、独立合同、有限内存与低频进度。"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time

import psutil
import torch

from 统一模型代码.common.g5_sample_range import SampleRangeArray, SampleRangeCache
from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.e2e_g5_model import build_context, g4
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import identity
from 统一模型代码.gates.g5.r2.g5_r2_model import AssociationHead

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT/'outputs_e2e/unified/e2e_g5/20260919_approved'
BASE = ROOT/'outputs_e2e/unified/e2e_g5_r2/20260921_approved'
ARMS = ('c0', 'c1', 'c2')
SEEDS = (20260921, 20260922)
CONFIG = {'gate': 'E2E-G5-R2', 'seeds': list(SEEDS), 'arms': list(ARMS),
          'batch_size': 4, 'base_epochs': 20, 'extension_epochs': 24, 'evaluate_every': 2,
          'wall_seconds': 43200, 'pilot_batches': 32, 'pilot_validation_samples': 32,
          'association_lr': 1e-4, 'association_loss_weight': .2, 'association_decode_weight': 1,
          'cache_bytes': 2*1024**3, 'ram_stop_percent': 85, 'precision': 'FP32',
          'positive_radius_m': 100, 'nearest_truth_owner': True, 'tie_policy': 'exclude_from_assoc',
          'separation_m': 30, 'top_k': 8, 'nms_window': 7,
          'feature_sampling': 'bilinear_center_align_corners_border',
          'progress_interval_seconds': 20, 'bootstrap_repeats': 2000, 'test_executed': False}
NAMES = ('ch3_spatial', 'd8_e1', 'd8_d2')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def safe_print(text, end='\n'):
    encoding = getattr(sys.stdout, 'encoding', None) or 'utf-8'
    print(text.encode(encoding, errors='replace').decode(encoding), end=end, flush=True)


class Progress:
    """One updating console line; disk log contains epoch/stage summaries only."""
    def __init__(self, label, total, path=None):
        self.label, self.total, self.path = label, total, path
        self.start = self.last = time.perf_counter()

    def update(self, done, force=False):
        now = time.perf_counter()
        if not force and done < self.total and now-self.last < CONFIG['progress_interval_seconds']:
            return
        elapsed = now-self.start
        remaining = elapsed/max(done, 1)*(self.total-done)
        n = int(20*done/max(self.total, 1))
        safe_print(f'\r{self.label} [{"#"*n}{"-"*(20-n)}] {done}/{self.total} '
                   f'已用 {elapsed/60:.1f}m 剩余约 {remaining/60:.1f}m     ',
                   end='\n' if force or done >= self.total else '')
        if self.path:
            write(self.path, {'stage': self.label, 'done': done, 'total': self.total,
                             'elapsed_seconds': elapsed, 'remaining_seconds': remaining, 'time': time.time()})
        self.last = now


class Runtime:
    def __init__(self, out, deadline=None):
        self.out = Path(out).resolve()
        if not self.out.is_relative_to(BASE.resolve()) or self.out.is_relative_to(SOURCE.resolve()):
            raise ValueError('R2 output isolation violation')
        self.out.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.inputs, self.ranges = {}, {}
        self.peak_ram = 0
        self.manifest = self.fm = None

    def guard(self):
        ram = psutil.virtual_memory().percent
        self.peak_ram = max(self.peak_ram, ram)
        if ram >= 85:
            raise RuntimeError('RAM_WARNING_85_PERCENT')
        if self.deadline and time.time() >= self.deadline:
            raise RuntimeError('R2运行预算到期；禁止自动续期')
        if shutil.disk_usage(self.out).free < 50*1024**3:
            raise RuntimeError('剩余磁盘低于50 GiB')
        if torch.cuda.is_initialized() and torch.cuda.max_memory_allocated() > 14*1024**3:
            raise RuntimeError('GPU allocation above 14 GiB')

    def get(self, row):
        self.guard()
        path = Path(row['path']).resolve(strict=True)
        if not path.is_relative_to(SOURCE.resolve(strict=True)):
            raise ValueError(f'Input escaped frozen G5: {path}')
        self.inputs[str(path)] = row
        return verified_read(row, self.out/'anomalies')

    def preflight(self):
        contract = read(BASE/'contract.json')
        if contract['config'] != CONFIG:
            raise RuntimeError('R2 configuration contract changed')
        for row in contract['files']:
            self.guard()
            verified_read(row, self.out/'anomalies')
        self.manifest = read(SOURCE/'manifest.json')
        self.fm = read(SOURCE/'feature_manifest.json')
        if read(SOURCE/'final_audit_report.json')['status'] != 'PASS' or self.manifest['test_executed']:
            raise RuntimeError('G5 evidence not complete or test not sealed')
        for split, n in (('train',4096), ('val_select',512), ('val_compare',1024)):
            if len(self.manifest['subsets'][split]) != n:
                raise RuntimeError('Unexpected split size')
        # train and validation raw_index belong to different original MAT files.
        sets = [set(r['raw_index'] for r in self.manifest['subsets'][s])
                for s in ('val_select','val_compare')]
        if sets[0] & sets[1]:
            raise RuntimeError('Overlapping splits')
        if read(SOURCE/'provenance_audit.json')['status'] != 'PASS':
            raise RuntimeError('G5 data provenance audit missing')
        for row in self.manifest['inputs']['artifacts']:
            self.get(row)
        for seed in SEEDS:
            row = read(SOURCE/f'training/{seed}/sg/best.identity.json')
            if row != read(SOURCE/f'comparison/{seed}_sg.json')['checkpoint']:
                raise RuntimeError('Historical best identity mismatch')
            self.get(row)
        if not torch.cuda.is_available():
            raise RuntimeError('Requires registered CUDA environment')
        self.guard()

    def features(self, split, cache=None):
        if split not in ('train','val_select','val_compare'):
            raise ValueError('test is forbidden')
        registry = read(SOURCE/'engineering_v2/index_registry.json')
        index = json.loads(self.get(registry['indexes'][split]))
        files = self.fm['files'][split]
        for name in NAMES:
            parents = {str(Path(r['path']).resolve(strict=True)): r for r in files[name]}
            for i, row in enumerate(index[name]):
                path = Path(row['path']).resolve(strict=True)
                parent = parents[str(path)]
                if (row['index'] != i or row['parent_sha256'] != parent['sha256']
                        or not path.is_relative_to(SOURCE/'features'/split)
                        or path.stat().st_size != parent['size_bytes']
                        or row['offset']+row['size_bytes'] > parent['size_bytes']):
                    raise RuntimeError('Sample range identity/geometry mismatch')
        cache = cache if cache is not None else SampleRangeCache(CONFIG['cache_bytes'], self.out/'anomalies')
        features = g4.FeatureStore(*(SampleRangeArray(index[n], cache) for n in NAMES))
        targets = g4.Targets(**torch.load(io.BytesIO(self.get(files['targets'])), map_location='cpu', weights_only=False))
        return features, targets, files['metadata'], index, cache

    def consumed(self, index, ids):
        for name in NAMES:
            for i in ids.tolist():
                row = index[name][i]
                self.ranges[(row['path'],row['offset'])] = row

    def postcheck(self, label):
        progress = Progress('输入阶段后核对', len(self.ranges), self.out/'progress.json')
        for i, row in enumerate(self.ranges.values(), 1):
            self.guard()
            verified_read(row, self.out/'anomalies', offset=row['offset'], length=row['size_bytes'])
            progress.update(i)
        for row in self.inputs.values():
            self.get(row)
        for row in read(BASE/'contract.json')['files']:
            verified_read(row, self.out/'anomalies')
        write(self.out/f'{label}_input_audit.json', {'status':'PASS', 'sample_ranges':len(self.ranges),
              'files':list(self.inputs.values()), 'same_bytes_verified_and_consumed':True})
        self.ranges.clear()

    def context(self, seed, arm):
        device = torch.device('cuda:0')
        context = build_context(self.out, self.manifest, seed, device)
        row = read(SOURCE/f'training/{seed}/sg/best.identity.json')
        payload = torch.load(io.BytesIO(self.get(row)), map_location='cpu', weights_only=False)
        g4.load_state(context, payload['state'])
        rates = read(BASE/'contract.json')['group_learning_rates'][str(seed)]
        for group in context.parameter_groups:
            group['lr'] = rates[group['name']]
        torch.manual_seed(seed+1000)
        head = AssociationHead().to(device) if arm != 'c0' else None
        groups = list(context.parameter_groups)
        parameters = list(context.parameters)
        if head is not None:
            params = list(head.parameters())
            groups.append({'params':params, 'lr':CONFIG['association_lr'], 'name':'association'})
            parameters += params
        optimizer = torch.optim.AdamW(groups, weight_decay=self.manifest['config']['weight_decay'])
        g4.set_deterministic(seed)
        return context, head, optimizer, parameters


def digest_state(state):
    h = hashlib.sha256()
    def walk(value):
        if isinstance(value, torch.Tensor):
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(str(key).encode('utf-8'))
                walk(value[key])
    walk(state)
    return h.hexdigest()


def register():
    if (BASE/'run/budget.json').exists():
        raise FileExistsError('正式运行已启动，禁止重新登记')
    BASE.mkdir(parents=True, exist_ok=True)
    files = {Path(__file__), ROOT/'运行入口/E2E/G5_R2/G5_R2一键运行.py'}
    files.update((ROOT/'统一模型代码/gates/g5/r2').rglob('*.py'))
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and Path(name).is_absolute():
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'outputs_e2e'):
                files.add(path)
    for name in ('manifest.json','feature_manifest.json','final_audit_report.json','provenance_audit.json',
                 'engineering_v2/index_registry.json','hard_reference_report.json'):
        files.add(SOURCE/name)
    rates = {}
    for seed in SEEDS:
        for name in ('best.identity.json', 'last.identity.json'):
            files.add(SOURCE/f'training/{seed}/sg/{name}')
        files.add(SOURCE/f'comparison/{seed}_sg.json')
        row = read(SOURCE/f'training/{seed}/sg/last.identity.json')
        payload = torch.load(io.BytesIO(verified_read(row,BASE/'registration_anomalies')), map_location='cpu', weights_only=False)
        rates[str(seed)] = {g['name']:g['lr'] for g in payload['optimizer']['param_groups']}
        del payload
    old_cfg = read(SOURCE/'manifest.json')['config']
    for seed_rates in rates.values():
        for name, value in seed_rates.items():
            if value != old_cfg[{'endpoint':'endpoint_learning_rate','d8_tail':'d8_tail_learning_rate',
                                 'ch3_tail':'ch3_tail_learning_rate'}[name]]:
                raise RuntimeError('G5 saved final LR differs from expected constant LR')
    previous = None
    if (BASE/'contract.json').exists():
        archive = BASE/'engineering'/f'contract_superseded_{time.time_ns()}.json'
        archive.parent.mkdir(parents=True,exist_ok=True)
        archive.write_bytes((BASE/'contract.json').read_bytes())
        previous = identity(archive)
    write(BASE/'contract.json', {'status':'REGISTERED', 'config':CONFIG, 'supersedes':previous,
          'group_learning_rates':rates, 'files':[identity(p) for p in sorted(files)],
          'source':str(SOURCE), 'python':sys.version, 'torch':torch.__version__})


class RunLock:
    def __enter__(self):
        import msvcrt
        BASE.mkdir(parents=True, exist_ok=True)
        self.handle = (BASE/'run.lock').open('a+b')
        try:
            if (BASE/'run.lock').stat().st_size == 0:
                self.handle.write(b'0')
                self.handle.flush()
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.handle.close()
            raise RuntimeError('已有G5-R2运行进程，请勿重复点击') from None
        return self

    def __exit__(self, *exc):
        import msvcrt
        self.handle.seek(0)
        msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        self.handle.close()


def setup_environment():
    os.environ['PYTHONUTF8'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8:replace'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
