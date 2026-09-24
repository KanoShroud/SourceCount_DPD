"""P2 writes are isolated; P1 code, C1 weights, and physical cache are frozen inputs."""
import io
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g6 import g6_p1_runtime as p1
from 统一模型代码.gates.g6.g6_p1_runtime import (
    SEEDS, ROOT, Progress as Progress, read, write, safe_print as safe_print,
    identity, digest_state as digest_state, setup_environment as setup_environment, g4)
from 统一模型代码.gates.g6.g6_p2_model import SplitPhysical

BASE = ROOT/'outputs_e2e/unified/e2e_g6_p2/20260923_approved'
ARMS = ('a','b','c')
CONFIG = {**p1.CONFIG,'gate':'E2E-G6-P2','arms':list(ARMS),'selector_lr':1e-4,
          'selector_parameters':2451,'band_f1_tolerance':.01,'initialization':'P1 C1 best per seed'}


def register():
    if (BASE/'run').exists():
        raise RuntimeError('P2 formal output exists; no new contract')
    BASE.mkdir(parents=True,exist_ok=True)
    audit_path=p1.BASE/'run/evaluation/final_audit_report.json'
    audit=read(audit_path)
    if audit['status']!='PASS' or not audit['six_tracks_complete']:
        raise RuntimeError('P1 must be complete')
    verified_read(audit['contract'],BASE/'anomalies')
    comp_row=next(r for r in audit['outputs'] if Path(r['path']).name=='comparison_report.json')
    comp=json.loads(verified_read(comp_row,BASE/'anomalies'))
    training=read(p1.BASE/'run/training_report.json')
    if training != comp['training']:
        raise RuntimeError('P1 training report mismatch')
    initial={str(s):training['seeds'][str(s)]['c1']['best']['checkpoint'] for s in SEEDS}
    for row in initial.values():
        verified_read(row,BASE/'anomalies')
    references={}
    for s in SEEDS:
        marker=next(r for r in audit['outputs'] if Path(r['path']).name==f'{s}_c1_complete.json')
        row=json.loads(verified_read(marker,BASE/'anomalies'))
        if row['checkpoint']!=initial[str(s)] or row['samples'] not in audit['outputs']:
            raise RuntimeError('Frozen C1 evaluation registration mismatch')
        references[str(s)]=row['samples']
    files={Path(__file__).resolve(),ROOT/'运行入口/E2E/G6/G6_P2一键运行.py',audit_path,
           p1.BASE/'contract.json',p1.BASE/'run/cache_contract.json',p1.BASE/'run/training_report.json'}
    files.update((ROOT/'统一模型代码/gates/g6').glob('g6_p2*.py'))
    files.add(ROOT/'统一模型代码/gates/g6/test_g6_p2.py')
    files.update(Path(r['path']) for r in initial.values())
    files.update(Path(r['path']) for r in references.values())
    for row in read(p1.BASE/'run/cache_contract.json').values():
        verified_read(row,BASE/'anomalies'); files.add(Path(row['path']))
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if isinstance(filename,str) and Path(filename).is_absolute():
            path=Path(filename).resolve()
            if path.suffix=='.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'outputs_e2e'):
                files.add(path)
    if (BASE/'contract.json').exists():
        shutil.copy2(BASE/'contract.json',BASE/f'contract_superseded_{time.time_ns()}.json')
    write(BASE/'contract.json',dict(config=CONFIG,files=[identity(p) for p in sorted(files)],
        initial=initial,frozen_c1_samples=references,p1_contract=audit['contract'],
        torch=torch.__version__,python=sys.version,scope='B-A separation; C-B only query feedback; no test'))


class Runtime(p1.Runtime):
    def __init__(self,out,deadline=None):
        self.out=Path(out).resolve()
        if not self.out.is_relative_to(BASE.resolve()) or self.out.is_relative_to(p1.BASE.resolve()):
            raise ValueError('P2 output isolation violation')
        self.out.mkdir(parents=True,exist_ok=True)
        self.deadline=deadline
        self.inputs,self.ranges,self.physical_consumed={},{},{}
        self.peak_ram=0
        self.manifest=self.fm=self.physics=None
        self._cache_indexes={}

    def preflight(self):
        self.contract=read(BASE/'contract.json')
        if self.contract['config']!=CONFIG:
            raise RuntimeError('P2 contract/config mismatch')
        for row in self.contract['files']:
            self.guard(); verified_read(row,self.out/'anomalies')
        super().preflight()
        for split,size in [('train',4096),('val_select',512),('val_compare',1024)]:
            if set(self.cache_rows(split))!={str(i) for i in range(size)}:
                raise RuntimeError('P1 physical cache incomplete; do not generate or mutate it')

    def ensure_cache(self,*args,**kwargs):
        raise RuntimeError('P2 may only consume the frozen P1 cache')

    def context(self,seed,arm):
        context,old_head,old_optimizer,old_params=super().context(seed,'c1')
        del old_head,old_optimizer,old_params
        row=self.contract['initial'][str(seed)]
        path=Path(row['path']).resolve(strict=True)
        if not path.is_relative_to(p1.BASE/'run/training'/str(seed)/'c1'):
            raise RuntimeError('Initialization escaped C1 training root')
        saved=torch.load(io.BytesIO(verified_read(row,self.out/'anomalies')),map_location='cpu',weights_only=False)
        g4.load_state(context,saved['state']['base'])
        torch.manual_seed(seed+1000)
        head=SplitPhysical(arm).cuda()
        head.physical.load_state_dict(saved['state']['physical'],strict=True)
        groups=list(context.parameter_groups)+[{'params':list(head.physical.parameters()),'lr':CONFIG['physical_lr'],'name':'physical'}]
        if head.selector is not None:
            groups.append({'params':list(head.selector.parameters()),'lr':CONFIG['selector_lr'],'name':'selector'})
        params=list(context.parameters)+list(head.parameters())
        optimizer=torch.optim.AdamW(groups,weight_decay=self.manifest['config']['weight_decay'])
        g4.set_deterministic(seed)
        return context,head,optimizer,params

    def postcheck(self,label):
        for row in self.contract['files']:
            self.guard(); verified_read(row,self.out/'anomalies')
        super().postcheck(label)


class RunLock(p1.RunLock):
    def __enter__(self):
        import msvcrt
        BASE.mkdir(parents=True,exist_ok=True)
        self.f=(BASE/'run.lock').open('a+b')
        if self.f.tell()==0:
            self.f.write(b'0'); self.f.flush()
        self.f.seek(0)
        try:
            msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:
            self.f.close()
            raise RuntimeError('已有G6-P2进程运行') from None
        return self
