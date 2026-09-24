"""G6-P1 isolated contract, lazy physical cache, and registered G5 inputs."""
import hashlib
import io
from pathlib import Path
import shutil
import sys
import time

import h5py
import numpy as np
import torch

from 统一模型代码.common.g5_verified_io import VerifiedFile, verified_read
from 统一模型代码.gates.g5.e2e_g5_model import build_context, g4
from 统一模型代码.gates.g5.e2e_g5_features import configure
from 统一模型代码.gates.g5.r2.g5_r2_runtime import (
    Runtime as OldRuntime, Progress, read, write, safe_print as safe_print, SOURCE, ROOT,
    setup_environment as setup_environment, digest_state as digest_state,
)
from 统一模型代码.gates.g6.coherent_dpd import AtomicDPD, geometry, grid_points
from 统一模型代码.gates.g6.g6_p0 import identity
from 统一模型代码.gates.g6.g6_p1_model import PhysicalResidual

BASE = ROOT/'outputs_e2e/unified/e2e_g6_p1/20260922_approved'
P0 = ROOT/'outputs_e2e/unified/e2e_g6_p0/20260922_165854'
SEEDS = (20260921, 20260922)
ARMS = ('c0', 'c1', 'c2')
CONFIG = dict(gate='E2E-G6-P1', seeds=list(SEEDS), arms=list(ARMS), batch_size=4,
              base_epochs=20, extension_epochs=24, evaluate_every=2, wall_seconds=43200,
              ram_stop_percent=85, gpu_limit_gib=14, disk_floor_gib=50, cache_bytes=2*1024**3,
              physical_precision='complex128/float64', network_precision='float32', evd_grid_chunk=256,
              physical_batch_mode='batched', physical_prefetch_batches=1, physical_pinned_staging=True,
              physical_lr=1e-4, pilot_batches=32, pilot_validation_samples=32,
              bootstrap_repeats=2000, separation_m=30, top_k=8, test_executed=False)


def register():
    if (BASE/'run').exists():
        raise RuntimeError('Formal output already exists; contract cannot change')
    BASE.mkdir(parents=True, exist_ok=True)
    p0audit = read(P0/'final_audit.json')
    if p0audit['status'] != 'PASS':
        raise RuntimeError('P0 is not PASS')
    source_audit = next(r for r in p0audit['files'] if Path(r['path']).name == 'input_audit.json')
    import json
    p0inputs = json.loads(verified_read(source_audit, BASE/'anomalies'))
    manifest_row = next(r for r in p0inputs['files'] if Path(r['path']).resolve() == SOURCE/'manifest.json')
    verified_read(manifest_row, BASE/'anomalies')
    files = set((ROOT/'统一模型代码/gates/g6').glob('*.py'))
    files.add(ROOT/'运行入口/E2E/G6/G6_P1一键运行.py')
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and Path(name).is_absolute():
            p = Path(name).resolve()
            if p.suffix == '.py' and p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'outputs_e2e'):
                files.add(p)
    for name in ('manifest.json', 'feature_manifest.json', 'final_audit_report.json',
                 'provenance_audit.json', 'engineering_v2/index_registry.json'):
        files.add(SOURCE/name)
    for seed in SEEDS:
        files.add(SOURCE/f'training/{seed}/sg/best.identity.json')
        files.add(SOURCE/f'comparison/{seed}_sg.json')
    if (BASE/'contract.json').exists():
        shutil.copy2(BASE/'contract.json', BASE/f'contract_superseded_{time.time_ns()}.json')
    write(BASE/'contract.json', {'config': CONFIG, 'files': [identity(p) for p in sorted(files)],
        'p0': identity(P0/'final_audit.json'), 'manifest': manifest_row,
        'torch': torch.__version__, 'python': sys.version,
        'scope': 'C1-C0 bypass package; C2-C1 only new physical feedback; no test'})


class Runtime(OldRuntime):
    def __init__(self, out, deadline=None):
        self.out = Path(out).resolve()
        if not self.out.is_relative_to(BASE.resolve()) or self.out.is_relative_to(SOURCE.resolve()):
            raise ValueError('G6 output isolation violation')
        self.out.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.inputs, self.ranges, self.physical_consumed = {}, {}, {}
        self.peak_ram = 0
        self.manifest = self.fm = self.physics = None
        self._cache_indexes = {}

    def preflight(self):
        contract = read(BASE/'contract.json')
        if contract['config'] != CONFIG:
            raise RuntimeError('Contract/config mismatch')
        for row in contract['files']:
            self.guard()
            verified_read(row, self.out/'anomalies')
        self.manifest = read(SOURCE/'manifest.json')
        self.fm = read(SOURCE/'feature_manifest.json')
        if read(SOURCE/'final_audit_report.json')['status'] != 'PASS':
            raise RuntimeError('G5 audit failed')
        for split, size in [('train',4096), ('val_select',512), ('val_compare',1024)]:
            if len(self.manifest['subsets'][split]) != size:
                raise RuntimeError('Split count changed')
        if {r['raw_index'] for r in self.manifest['subsets']['val_select']} & {
                r['raw_index'] for r in self.manifest['subsets']['val_compare']}:
            raise RuntimeError('Overlapping validation roles')
        for row in self.manifest['inputs']['artifacts']:
            self.get(row)
        for seed in SEEDS:
            row = read(SOURCE/f'training/{seed}/sg/best.identity.json')
            registered = next(r['best'] for r in read(SOURCE/'final_audit_report.json')['tracks']
                              if r['seed'] == seed and r['track'] == 'sg')
            if row != registered:
                raise RuntimeError('G5 checkpoint registration mismatch')
            self.get(row)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        coarse = self.manifest['inputs']['files'][1]
        with VerifiedFile(coarse, self.out/'anomalies') as f, h5py.File(f, 'r') as h:
            lo = torch.as_tensor(h['sub_f_lo_val'][()].reshape(-1))
            hi = torch.as_tensor(h['sub_f_hi_val'][()].reshape(-1))
        self.physics = AtomicDPD(lo, hi, geometry(grid_points(), device='cuda', shape=(81,81)),
                                 precompute_phase=False)
        self.physics.p1_batch_mode = CONFIG['physical_batch_mode']
        self.physics.p1_chunk = CONFIG['evd_grid_chunk']
        configure(self.out, self.manifest['inputs'])
        self.guard()

    def context(self, seed, arm):
        context = build_context(self.out, self.manifest, seed, torch.device('cuda:0'))
        row = read(SOURCE/f'training/{seed}/sg/best.identity.json')
        state = torch.load(io.BytesIO(self.get(row)), map_location='cpu', weights_only=False)
        g4.load_state(context, state['state'])
        torch.manual_seed(seed+1000)
        head = PhysicalResidual().cuda() if arm != 'c0' else None
        groups, params = list(context.parameter_groups), list(context.parameters)
        if head is not None:
            groups.append({'params': list(head.parameters()), 'lr': CONFIG['physical_lr'], 'name':'physical'})
            params += list(head.parameters())
        optimizer = torch.optim.AdamW(groups, weight_decay=self.manifest['config']['weight_decay'])
        g4.set_deterministic(seed)
        return context, head, optimizer, params

    def audit_raw(self):
        for row in self.manifest['inputs']['files']:
            path = Path(row['path']).resolve(strict=True)
            if path.parent != (SOURCE/'input_snapshot').resolve() or path.stat().st_size != row['size_bytes']:
                raise RuntimeError('Snapshot path/size mismatch')
            digest = hashlib.sha256()
            for i, b in enumerate(row['blocks']):
                self.guard()
                digest.update(verified_read({'path':str(path), **b}, self.out/'anomalies',
                              offset=i*row['block_size'], length=b['size_bytes']))
            if digest.hexdigest() != row['sha256']:
                raise RuntimeError('Snapshot full SHA mismatch')

    def cache_rows(self, split):
        path = BASE/f'physical_cache/{split}/index.json'
        if split not in self._cache_indexes:
            frozen = BASE/'run/cache_contract.json'
            if frozen.exists():
                verified_read(read(frozen)[split], self.out/'anomalies')
            self._cache_indexes[split] = read(path) if path.exists() else {}
        return self._cache_indexes[split]

    def ensure_cache(self, split, ids, targets):
        rows = self.cache_rows(split)
        missing = [int(i) for i in ids if str(int(i)) not in rows]
        if not missing:
            return {'split':split, 'count':0, 'seconds':0, 'bytes':0}
        start = time.perf_counter()
        # Preallocate P0 phases only while generating IQ statistics, never during training.
        physics = AtomicDPD(self.physics.lo, self.physics.hi, self.physics.geo)
        progress = Progress(f'生成物理缓存 {split}', len(missing), self.out/'progress.json')
        with g4.g1.SampleStore(split) as store, torch.no_grad():
            lo,hi = store.subband_edges()
            np.testing.assert_array_equal(lo, self.physics.lo.cpu().numpy())
            np.testing.assert_array_equal(hi, self.physics.hi.cpu().numpy())
            for n, i in enumerate(missing, 1):
                self.guard()
                record = self.manifest['subsets'][split][i]
                raw, raw_local = store._raw(record['raw_index'])
                local = record['local_index']
                k = int(targets.counts[i])
                if int(raw['src_count_all'][0,raw_local]) != k or int(store.coarse['src_count_all'][0,local]) != k:
                    raise RuntimeError('Cache target count mismatch')
                positions = np.asarray(raw['src_pos_all'][:,:,raw_local]).T[:k]
                bands = np.asarray(raw['band_mask_all'][:,:,raw_local]).T[:k]
                torch.testing.assert_close(torch.as_tensor(positions).float(), targets.positions[i,:k], rtol=0, atol=0)
                torch.testing.assert_close(torch.as_tensor(bands).float(), targets.band[i,:k].float(), rtol=0, atol=0)
                signal = (np.asarray(raw['sig_rcv_real_all'][:,:,raw_local],dtype=np.float64).T
                          +1j*np.asarray(raw['sig_rcv_imag_all'][:,:,raw_local],dtype=np.float64).T)
                stats = {key:v.cpu() for key,v in physics.statistics(signal).items()}
                folder = BASE/f'physical_cache/{split}'
                folder.mkdir(parents=True, exist_ok=True)
                path = folder/f'{i:05d}_{time.time_ns()}.pt'
                torch.save(stats, path)
                row = identity(path)
                restored = torch.load(io.BytesIO(verified_read(row,self.out/'anomalies')), weights_only=True)
                if not all(torch.equal(stats[key],restored[key]) for key in stats):
                    raise RuntimeError('Physical cache round trip mismatch')
                rows[str(i)] = row
                write(folder/'index.json', rows)
                progress.update(n)
        del physics
        torch.cuda.empty_cache()
        return {'split':split, 'count':len(missing), 'seconds':time.perf_counter()-start,
                'bytes':sum(rows[str(i)]['size_bytes'] for i in missing)}

    def stats(self, split, ids):
        rows = self.cache_rows(split)
        result = []
        for i in ids.tolist():
            row = rows[str(i)]
            path = Path(row['path']).resolve(strict=True)
            if not path.is_relative_to((BASE/'physical_cache'/split).resolve()):
                raise RuntimeError('Cache path escaped')
            result.append(torch.load(io.BytesIO(verified_read(row,self.out/'anomalies')), weights_only=True))
            self.physical_consumed[str(path)] = row
        return result

    def postcheck(self, label):
        for row in self.ranges.values():
            self.guard()
            verified_read(row, self.out/'anomalies', offset=row['offset'], length=row['size_bytes'])
        for row in self.physical_consumed.values():
            self.guard()
            verified_read(row,self.out/'anomalies')
        for row in list(self.inputs.values())+read(BASE/'contract.json')['files']:
            verified_read(row,self.out/'anomalies')
        write(self.out/f'{label}_input_audit.json', {'status':'PASS','feature_ranges':len(self.ranges),
              'physical_files':len(self.physical_consumed),'same_bytes_verified_consumed':True})
        self.ranges.clear()
        self.physical_consumed.clear()


class RunLock:
    def __enter__(self):
        import msvcrt
        BASE.mkdir(parents=True, exist_ok=True)
        self.f = (BASE/'run.lock').open('a+b')
        if self.f.tell() == 0:
            self.f.write(b'0')
            self.f.flush()
        self.f.seek(0)
        try:
            msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            self.f.close()
            raise RuntimeError('已有G6-P1进程运行') from None
        return self

    def __exit__(self,*args):
        import msvcrt
        self.f.seek(0)
        msvcrt.locking(self.f.fileno(),msvcrt.LK_UNLCK,1)
        self.f.close()
