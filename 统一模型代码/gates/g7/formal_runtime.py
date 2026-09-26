"""Clean G7 data contract and bounded lazy runtime. No historical model loads."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import psutil
import torch

from 统一模型代码.gates.g7.compact_data import BASE, ROOT, Inputs, g4, read, write, save, load, identity
from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g6.coherent_dpd import AtomicDPD, geometry
from 统一模型代码.gates.g6.g6_p2_runtime import Progress
from 统一模型代码.gates.g7.compact_model import candidates, windows, local_maps, loss
from 统一模型代码.gates.g7.compact_run import metric_rows
from 统一模型代码.gates.g7.g7_physics import statistics

SOURCE = BASE/'v2_full_adaptation'
RUN = BASE/'formal_v1_40h'
PHASES = ('ch3', 'd8', 'candidate', 'f', 's', 'e')
CONFIG = dict(version='G7_FORMAL_RANDOM_V1', seeds=[20260921, 20260922],
    counts=dict(train=4096, val_select=512, val_compare=1024), test_target_count=2048,
    test_status='SEALED_NOT_REGISTERED', edge_m=1000, coarse_size=41, fine_size=201,
    coarse_step_m=50, fine_step_m=10, top_k=5, window_size=41,
    initialization='random_native_models_and_new_modules_only',
    reuse='verified_model_independent_IQ_DPD_statistics_ONLY',
    ram_limit_percent=85, gpu_limit_gib=14, disk_floor_gib=50, wall_seconds=144000,
    cache_bytes=1024**3, bootstrap_repeats=2000,
    precision='network_FP32_physics_FP64_complex128', test_executed=False,
    phases={p: dict(batch_size=8 if p in ('ch3','candidate') else 4,
                    base_epochs=80 if p in ('ch3','d8') else 40,
                    max_epochs=120 if p in ('ch3','d8') else 60,
                    evaluate_every=2, warmup_epochs=5, min_lr_factor=.01) for p in PHASES})


def verified_json(row, out):
    return json.loads(verified_read(row, Path(out)/'anomalies'))


class Runtime:
    def __init__(self, out=RUN, config=None):
        self.out = Path(out).resolve()
        if not self.out.is_relative_to(BASE.resolve()) or self.out == SOURCE.resolve():
            raise ValueError('Formal output isolation violation')
        self.out.mkdir(parents=True, exist_ok=True)
        self.config = CONFIG if config is None else config
        self.inputs = Inputs(self.out)
        self.manifest = self.inputs.manifest
        self.items, self.cache_bytes = OrderedDict(), 0
        self.sources = list(self.inputs.sources)
        # Bind data-only pointers, not warm-start weights or optimizer history.
        self.dataset = self.read_source_index('dataset', 'data/index.json')
        self.oracle = self.read_source_index('oracle', 'oracle/index.json')
        self.local_indices = {}
        self.stores = {}
        self.scene_hashes = {}
        self.lo, self.hi = self.inputs.edges()
        axis = torch.linspace(-1000,1000,41,dtype=torch.float64)
        y,x = torch.meshgrid(axis,axis,indexing='ij')
        self.physics = AtomicDPD(self.lo,self.hi,geometry(torch.stack([x.flatten(),y.flatten()],-1),
                                 device='cuda',shape=(41,41)),precompute_phase=False)
        self.physics.p1_batch_mode, self.physics.p1_chunk = 'batched',256
        self.started = time.perf_counter()
        self.budget_path = self.out/'budget.json'
        self.budget = read(self.budget_path) if self.budget_path.exists() else dict(active_seconds=0.,open=False)
        self.peak_ram_percent = psutil.virtual_memory().percent

    def read_source_index(self, name, relative):
        pointer = SOURCE/f'{name}.identity.json'
        row = read(pointer)
        if Path(row['path']).resolve() != (SOURCE/relative).resolve():
            raise ValueError('Data identity points to an unexpected source')
        value = verified_json(row,self.out)
        self.sources.extend([identity(pointer),row])
        return value

    def contract(self):
        paths = set()
        for folder in ('统一模型代码','第三章代码','第四章代码'):
            paths.update((ROOT/folder).rglob('*.py'))
        paths.update((ROOT/'运行入口/E2E/G7').glob('G7正式*.py'))
        value = dict(config=self.config,files=[identity(p) for p in sorted(paths)]+self.sources,
            historical_checkpoint_inputs=[], data_roles='Existing validation remains development data',
            test_permission=False, proposed_test_count=2048)
        path = self.out/'contract.json'
        if path.exists() and read(path) != value:
            raise RuntimeError('正式合同已变化；不复用已有训练')
        if not path.exists():
            write(path,value)
        for row in value['files']:
            verified_read(row,self.out/'anomalies')
        self.registration = value
        return value

    def progress(self,label,total):
        return Progress(label,total,self.out/'progress.json')

    def begin(self, initial_seconds=0.):
        if self.budget['open']:
            raise RuntimeError('上次被强制结束；先核对预算记录，不自动重置')
        if not self.budget_path.exists():
            self.budget['active_seconds'] = float(initial_seconds)
        self.started = time.perf_counter()
        self.budget['open'] = True
        write(self.budget_path,self.budget)

    def finish(self):
        self.budget['active_seconds'] += time.perf_counter()-self.started
        self.budget['open'] = False
        write(self.budget_path,self.budget)

    def guard(self):
        self.peak_ram_percent = max(self.peak_ram_percent,psutil.virtual_memory().percent)
        if psutil.virtual_memory().percent >= self.config['ram_limit_percent']:
            raise RuntimeError('RAM reached 85%')
        if torch.cuda.memory_allocated()/2**30 > self.config['gpu_limit_gib']:
            raise RuntimeError('GPU allocation exceeded14GiB')
        if shutil.disk_usage(self.out).free/2**30 < self.config['disk_floor_gib']:
            raise RuntimeError('Disk below50GiB reserve')
        if self.budget['active_seconds']+time.perf_counter()-self.started > self.config['wall_seconds']:
            raise RuntimeError(f'累计{self.config["wall_seconds"]/3600:g}小时预算到期；已完成epoch保留')

    def get(self,row):
        key=(row['path'],row['sha256'])
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key][0]
        value=load(row,self.out)
        while self.items and self.cache_bytes+row['size_bytes'] > self.config['cache_bytes']:
            _,(_,n)=self.items.popitem(last=False); self.cache_bytes-=n
        self.items[key]=(value,row['size_bytes']); self.cache_bytes+=row['size_bytes']
        return value

    def clear_cache(self):
        self.items.clear(); self.cache_bytes=0

    def ids(self,phase,split):
        if split not in self.config['counts']:
            raise ValueError('Unknown/test split forbidden')
        return [i for i in range(self.config['counts'][split]) if phase!='d8' or
                self.manifest['subsets'][split][i]['true_k']>0]

    def data(self,phase,seed,split,ids):
        if split not in self.config['counts']:
            raise ValueError('test input is sealed')
        output=[]
        for i in ids:
            r=self.get(self.dataset[split][str(i)])
            expected=self.manifest['subsets'][split][i]
            if r['index']!=i or r['split']!=split or any(r['metadata'][k]!=expected[k]
                    for k in ('raw_index','local_index','true_k')):
                raise RuntimeError('Formal sample mapping changed')
            if phase=='d8':
                oracle=self.get(self.oracle[split][str(i)])
                if oracle['index']!=i or oracle['raw_index']!=expected['raw_index']:
                    raise RuntimeError('Oracle/sample mapping differs')
                r=dict(r,oracle_fine=oracle['oracle_fine'])
            if phase in ('f','s','e'):
                index=self.local_index(seed,split)
                local=self.get(index['samples'][str(i)])
                if local['index']!=i or local['split']!=split or local['candidate_sha256']!=index['candidate']['sha256']:
                    raise RuntimeError('Local candidates/cache mapping differs')
                r=dict(r,_local=local)
            output.append(r)
        return output

    def signal(self,split,index):
        if split not in self.config['counts']:
            raise ValueError('test is sealed')
        if split not in self.stores:
            self.stores[split]=g4.g1.SampleStore(split)
        entry=self.manifest['subsets'][split][index]
        raw,j=self.stores[split]._raw(entry['raw_index'])
        real=np.asarray(raw['sig_rcv_real_all'][:,:,j],dtype=np.float32).T.copy()
        imag=np.asarray(raw['sig_rcv_imag_all'][:,:,j],dtype=np.float32).T.copy()
        expected=self.scene_hashes.get((split,index))
        if expected and hashlib.sha256(np.stack([real,imag]).astype('<f4').tobytes()).hexdigest()!=expected:
            raise RuntimeError('Consumed IQ differs from registered scene audit')
        return real.astype(np.float64)+1j*imag.astype(np.float64)

    def bind_scene_audit(self,row):
        report=verified_json(row,self.out)
        if report['status']!='PASS' or report['scenes']!=sum(self.config['counts'].values()):
            raise RuntimeError('Scene identity audit does not cover this split')
        self.scene_hashes={(r['split'],r['index']):r['iq_sha256'] for r in report['samples']}

    def close(self):
        for store in self.stores.values():
            store.close()
        self.stores.clear()

    def audit_scenes(self):
        """Canonical IQ bytes, independent of parent filename/local row numbering."""
        seen,rows={},[]
        for split in self.config['counts']:
            bar=self.progress(f'IQ场景去重 {split}',self.config['counts'][split])
            for i in self.ids('ch3',split):
                self.guard()
                signal=self.signal(split,i)
                payload=np.stack([signal.real,signal.imag]).astype('<f4').tobytes()
                sha=hashlib.sha256(payload).hexdigest()
                if sha in seen:
                    raise RuntimeError(f'Duplicate IQ scenes: {seen[sha]} and {(split,i)}')
                seen[sha]=(split,i)
                rows.append(dict(split=split,index=i,iq_sha256=sha,
                    raw_index=self.manifest['subsets'][split][i]['raw_index']))
                bar.update(i+1)
        self.close()
        report=dict(status='PASS',scenes=len(rows),samples=rows,test_read=False,
                    existing_validation_is_development=True)
        write(self.out/'scene_identity_report.json',report)
        self.bind_scene_audit(identity(self.out/'scene_identity_report.json'))
        return report

    def best(self,seed,phase):
        return read(self.out/f'training/{seed}/{phase}/completed.json')['best']['checkpoint']

    def model(self,seed,phase):
        from 统一模型代码.gates.g7.formal_model import FormalFoundation,FormalCandidate
        if phase in ('ch3','d8'):
            return FormalFoundation(self.out,self.manifest,seed,phase)
        model=FormalCandidate(self.out,self.manifest,seed)
        if phase=='candidate':
            states=[load(self.best(seed,k),self.out)['state'] for k in ('ch3','d8')]
            model.load_foundations(*states)
        else:
            candidate=self.best(seed,'candidate')
            pointer=self.out/f'training/{seed}/local_initial.identity.json'
            if pointer.exists():
                saved=load(read(pointer),self.out)
                if saved['candidate']!=candidate:
                    raise RuntimeError('Shared local initialization refers to another candidate')
                model.restore(saved['state'])
            else:
                model.restore(load(candidate,self.out)['state'])
                model.initialize_local()
                row=save(self.out/f'training/{seed}/local_initial_{time.time_ns()}.pt',
                         dict(state=model.state(),candidate=candidate))
                write(pointer,row)
        return model

    def optimizer(self,model,phase):
        return model.optimizer(lr=1e-3) if phase in ('ch3','d8') else model.optimizer(
            'baseline' if phase=='candidate' else 'local',self.manifest)

    def forward(self,model,phase,batch):
        if phase=='candidate':
            return model.baseline(batch,self.physics)
        return model.local(batch,self.physics,[r['_local'] for r in batch],phase)

    def loss(self,output,batch,phase):
        return loss(output,batch,'baseline' if phase=='candidate' else phase)

    def metric_rows(self,output,batch,phase):
        return metric_rows(output,batch,'baseline' if phase=='candidate' else phase)

    def local_index(self,seed,split):
        key=(seed,split)
        if key not in self.local_indices:
            pointer=self.out/f'local/{seed}/{split}/index.identity.json'
            self.local_indices[key]=verified_json(read(pointer),self.out)
        return self.local_indices[key]

    @torch.no_grad()
    def local_record(self,model,seed,split,i,candidate_sha):
        record=self.data('candidate',seed,split,[i])[0]
        output=model.baseline([record],self.physics)
        proposed=candidates(output['heat'],8)[0]
        centers,valid,points,inverse=windows(proposed,self.config['top_k'])
        stat=statistics(self.physics,self.signal(split,i),points,guard=self.guard)
        stat.update(centers_m=centers,valid_mask=valid,inverse=inverse,points=points,
                    index=i,split=split,candidate_sha256=candidate_sha,proposals=proposed,
                    candidate_band_logits=output['band_logits'][0].cpu())
        stat['full_maps']=local_maps(self.physics,torch.ones(1,3,19,device='cuda'),[stat],full=True)[0].cpu()
        return stat

    def prepare_local(self,seed,split):
        candidate=self.best(seed,'candidate')
        root=self.out/f'local/{seed}/{split}'
        path=root/'index.json'; pointer=root/'index.identity.json'
        if pointer.exists():
            idx=verified_json(read(pointer),self.out)
            if idx['candidate']!=candidate or len(idx['samples'])!=self.config['counts'][split]:
                raise RuntimeError('Completed local cache contract differs')
            self.local_indices[(seed,split)]=idx
            return
        index=read(path) if path.exists() else dict(candidate=candidate,samples={})
        if index['candidate']!=candidate:
            raise RuntimeError('Partial local cache candidate changed')
        model=self.model(seed,'f'); model.mode(False)
        bar=self.progress(f'{seed} 局部物理缓存 {split}',self.config['counts'][split])
        for i in self.ids('candidate',split):
            self.guard()
            if str(i) not in index['samples']:
                stat=self.local_record(model,seed,split,i,candidate['sha256'])
                index['samples'][str(i)]=save(root/f'{i:05d}_{time.time_ns()}.pt',stat)
                write(path,index)
            else:
                verified_read(index['samples'][str(i)],self.out/'anomalies')
            bar.update(i+1)
        write(pointer,identity(path)); self.local_indices[(seed,split)]=index
        self.close(); del model
        torch.cuda.empty_cache()
