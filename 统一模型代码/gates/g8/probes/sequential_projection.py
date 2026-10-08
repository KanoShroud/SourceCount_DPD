"""G8-R5 matched-propagation sequential least squares; no overlap labels in solver."""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter

from .peak_diagnosis import evaluate_positions, local_ids
from ..physics import FS, C, RECEIVERS, Spectrum, grid, peaks
from ..storage import BASE, Guard, read, write, identity, checked_bytes, load_scene
from ..diagnostics import summarize


class ProjectedScore:
    """Profiled likelihood increment for adding a steering vector to existing sources.

    A_f[m,k] = (1/d_mk)*exp(-2j*pi*f*d_mk/c), normalized per column.
    Score = sum_f |(P a)^H(P x)|^2 / ||P a||^2, P=I-A A^+.
    Both data and candidate are projected. No per-station IQ renormalization.
    """
    def __init__(self, iq, guard=lambda: None):
        self.n=iq.shape[-1]
        self.x=torch.as_tensor(np.fft.fftshift(np.fft.fft(iq,axis=-1),axes=-1).T/self.n,device='cuda')
        self.f=torch.as_tensor(np.fft.fftshift(np.fft.fftfreq(self.n,1/FS)),device='cuda')
        self.stations=torch.as_tensor(RECEIVERS,dtype=torch.float64,device='cuda')
        self.guard=guard
        self.set_sources([])

    def steering(self, points, freq):
        p=torch.as_tensor(np.asarray(points).reshape(-1,2),dtype=torch.float64,device='cuda')
        distance=torch.linalg.vector_norm(p[:,None]-self.stations[None],dim=-1).clamp_min(1.)
        amplitude=1/distance
        amplitude=amplitude/torch.linalg.vector_norm(amplitude,dim=-1,keepdim=True)
        return amplitude[:,None,:]*torch.exp(-2j*np.pi*freq[None,:,None]*distance[:,None,:]/C)

    def set_sources(self, positions):
        self.sources=np.asarray(positions,dtype=float).reshape(-1,2)
        eye=torch.eye(4,dtype=torch.complex128,device='cuda')
        if len(self.sources):
            a=self.steering(self.sources,self.f).permute(1,2,0)
            self.projector=eye[None]-a@torch.linalg.pinv(a,rtol=1e-10)
            self.projector=(self.projector+self.projector.mH)/2
            self.residual=torch.einsum('fmn,fn->fm',self.projector,self.x)
        else:
            self.projector=None
            self.residual=self.x

    @torch.no_grad()
    def evaluate(self, points):
        points=np.asarray(points,dtype=float).reshape(-1,2)
        values=[]
        for begin in range(0,len(points),32):
            self.guard()
            p=points[begin:begin+32]
            total=torch.zeros(len(p),dtype=torch.float64,device='cuda')
            for lo in range(0,self.n,4096):
                a=self.steering(p,self.f[lo:lo+4096])
                if self.projector is not None:
                    a=torch.einsum('fmn,pfn->pfm',self.projector[lo:lo+4096],a)
                denom=a.abs().square().sum(-1)
                dot=(a.conj()*self.residual[None,lo:lo+4096]).sum(-1)
                total+=torch.where(denom>1e-10,dot.abs().square()/denom.clamp_min(1e-10),0.).sum(-1)
            values.append(total.cpu().numpy())
        return np.concatenate(values)


def window(center, radius, step):
    axis=np.arange(-radius,radius+step/2,step)
    x,y=np.meshgrid(axis,axis)
    pts=np.asarray(center)+np.stack([x.ravel(),y.ravel()],-1)
    return pts[(abs(pts)<=1000).all(-1)]


def allowed(points, excluded):
    if not len(excluded):
        return np.ones(len(points),bool)
    return (np.linalg.norm(points[:,None]-np.asarray(excluded)[None],axis=-1)>=30).all(-1)


def refine(calc, center, excluded, radius=10, step=1):
    pts=window(center,radius,step)
    pts=pts[allowed(pts,excluded)]
    if not len(pts):
        return np.asarray(center)
    return pts[np.argmax(calc.evaluate(pts))]


def discover(calc, excluded):
    pts,shape=grid()
    v=calc.evaluate(pts)
    v[~allowed(pts,excluded)]=-np.inf
    lm=v.reshape(shape)==maximum_filter(v.reshape(shape),3,mode='constant',cval=-np.inf)
    ids=np.flatnonzero(lm.ravel() & np.isfinite(v))
    chosen=ids[peaks(pts[ids],v[ids],5)]
    candidates=np.unique(np.concatenate([window(pts[i],50,10) for i in chosen]),axis=0)
    candidates=candidates[allowed(candidates,excluded)]
    coarse=candidates[np.argmax(calc.evaluate(candidates))]
    return refine(calc,coarse,excluded)


def sequential(calc, initial, count=2):
    locations=[np.asarray(initial)]
    while len(locations)<count:
        calc.set_sources(locations)
        locations.append(discover(calc,locations))
    return np.asarray(locations)


def joint(calc, locations, rounds=2):
    locations=np.asarray(locations).copy()
    history=[]
    for _ in range(rounds):
        for index in range(len(locations)):
            others=np.delete(locations,index,axis=0)
            calc.set_sources(others)
            p=refine(calc,locations[index],others,50,10)
            locations[index]=refine(calc,p,others)
        calc.set_sources(locations)
        history.append(float(calc.residual.abs().square().sum().cpu()))
    return locations,history


def tests():
    rng=np.random.default_rng(2026100605)
    iq=rng.normal(size=(4,64))+1j*rng.normal(size=(4,64))
    calc=ProjectedScore(iq)
    checks=[]
    locations=np.array([[210.,120.],[-350.,220.],[180.,-360.]])
    candidates=np.array([[650.,550.],[-620.,-180.]])
    for k in range(4):
        calc.set_sources(locations[:k])
        a=calc.steering(locations[:k],calc.f).cpu().numpy() if k else None
        b=calc.steering(candidates,calc.f).cpu().numpy()
        x=calc.x.cpu().numpy()
        ref=[]
        for candidate in b:
            score=0.
            for f in range(64):
                p=np.eye(4,dtype=complex) if not k else np.eye(4)-a[:,f,:].T@np.linalg.pinv(a[:,f,:].T,rcond=1e-10)
                np.testing.assert_allclose(p@p,p,atol=1e-9)
                if k:
                    np.testing.assert_allclose(p@a[:,f,:].T,0,atol=1e-9)
                u=p@candidate[f]
                den=np.vdot(u,u).real
                score+=abs(np.vdot(u,p@x[f]))**2/den if den>1e-10 else 0.
            ref.append(score)
        actual=calc.evaluate(candidates)
        np.testing.assert_allclose(actual,ref,rtol=1e-8,atol=1e-9)
        checks.append(float(np.max(abs(actual-ref))))
    return dict(status='PASS',source_counts=[0,1,2,3],max_absolute_error=max(checks),
        scope='algebra and variable source dimensions, not K3 performance')


def run_case(scene, dpd_map, guard, oracle=False):
    start=time.perf_counter()
    truth=scene['metadata']['positions']
    points,_=grid(10)
    calc=ProjectedScore(scene['components'].sum(0)+scene['noise'],guard)
    first=refine(calc,points[np.argmax(dpd_map)],[])
    ids=local_ids(dpd_map)
    erased=ids[np.linalg.norm(points[ids]-first,axis=-1)>=100]
    second=points[erased[np.argmax(dpd_map.ravel()[erased])]]
    result=dict(**scene['metadata'],initial=first.tolist(),methods={})
    result['methods']['erase_peak']=evaluate_positions(truth,[first,second],time.perf_counter()-start)
    t0=time.perf_counter()
    calc.set_sources([])
    direct=discover(calc,[first])
    result['methods']['no_projection']=evaluate_positions(truth,[first,direct],time.perf_counter()-t0)
    t0=time.perf_counter()
    initial=sequential(calc,first,2)
    result['methods']['projection']=evaluate_positions(truth,initial,time.perf_counter()-t0)
    final,history=joint(calc,initial)
    result['methods']['projection_joint']=evaluate_positions(truth,final,time.perf_counter()-t0)
    result['joint_residual_energy']=history
    # Diagnostics only: fractions of true components remaining after the estimated first projection.
    calc.set_sources([first])
    fractions=[]
    for component in scene['components']:
        x=torch.as_tensor(np.fft.fftshift(np.fft.fft(component,axis=-1),axes=-1).T/calc.n,device='cuda')
        residual=torch.einsum('fmn,fn->fm',calc.projector,x)
        fractions.append(float((residual.abs().square().sum()/x.abs().square().sum()).cpu()))
    result['component_retained_fraction_diagnostic']=fractions
    if oracle:
        t0=time.perf_counter()
        assisted=sequential(calc,truth[0],2)
        result['oracle_first_position']=evaluate_positions(truth,assisted,time.perf_counter()-t0)
    result['seconds']=time.perf_counter()-start
    return result


def aggregate(rows):
    result={}
    for role in ('calibration','check'):
        for overlap in (.5,1.):
            selected=[r for r in rows if r['role']==role and r['overlap']==overlap]
            if not selected:
                continue
            variants={}
            for method in ('erase_peak','no_projection','projection','projection_joint','oracle_first_position'):
                data=([r[method] for r in selected if method in r] if method=='oracle_first_position' else [r['methods'][method] for r in selected])
                if data:
                    v=summarize(data)
                    v.update(strong_recall100=float(np.mean([r['source_recall']['100'][0] for r in data])),
                        weak_recall100=float(np.mean([r['source_recall']['100'][1] for r in data])))
                    variants[method]=v
            result[f'{role}_{overlap}']=variants
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--short',action='store_true')
    args=parser.parse_args()
    out=(BASE/args.run_id).resolve()
    if not out.is_relative_to(BASE.resolve()) or out==BASE.resolve():
        raise ValueError('Output root')
    out.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter()
    torch.set_num_threads(4)
    resolved={}
    for p in BASE.glob('*/final_audit_report.json'):
        a=read(p)
        if a.get('status')=='PASS':
            for gate,scope in [('r2','G8_R2_development_knownK'),('r3','G8_R3_development_diagnostic'),('r4','G8_R4_oracle_band_diagnostic')]:
                if a.get('scope')==scope:
                    if gate in resolved:
                        raise RuntimeError('Ambiguous prior')
                    resolved[gate]=(p.parent,a)
    if len(resolved)!=3:
        raise RuntimeError('Missing audited prerequisite')
    inputs=[]
    for folder,a in resolved.values():
        inputs += [identity(folder/'final_audit_report.json')]+a['outputs']+a['sources']
    for item in inputs:
        checked_bytes(item)
    records=read(resolved['r2'][0]/'data_index.json')
    oldrows=read(resolved['r3'][0]/'rows.json')
    previous=sum(read(p)['seconds'] for p in BASE.glob('*/report.json') if read(p).get('gate')=='G8-R5')
    used=read(resolved['r4'][0]/'report.json')['cumulative_G8_seconds']
    guard=Guard(out,min(7200-previous,43200-used-previous))
    sources=resolved['r4'][1]['sources']+[identity(Path(__file__))]
    groups=[0] if args.short else list(range(48))
    for g in groups:
        for overlap in (.5,1.):
            inputs.append(records[f'{g}_{overlap}'])
            checked_bytes(inputs[-1])
    guard()
    write(out/'manifest.json',dict(gate='G8-R5',short=args.short,groups=groups,N=65536,
        initialized_by='fullband DPD highest point + shared LS refine',propagation='estimated-distance 1/d amplitude, calibrated coherent stations',
        first_refine=[10,1],global_search=[50,5,50,10],joint_rounds=2,joint_refine=[[50,10],[10,1]],
        erase_radius_m=100,separation_m=30,pinv_rtol=1e-10,den_floor=1e-10,
        no_training=True,test_read=False,known_K=2,oracle_calibration_groups=list(range(8)),
        prior_G8_seconds=used,prior_R5_seconds=previous,seconds_limit=guard.seconds,inputs=inputs,sources=sources))
    rows=[]
    try:
        write(out/'algebra_tests.json',tests())
        print('阶段1：共同配置校准/短测；投影与NumPy参考式核验通过',flush=True)
        for g in groups:
            if g==16:
                print('阶段2：配置不变，进入32个检查布局',flush=True)
            for overlap in (.5,1.):
                scene=load_scene(records[f'{g}_{overlap}'])
                if g>=16:
                    old=next(r for r in oldrows if r['group']==g and r['overlap']==overlap and r['method']=='dpd')
                    dpd_map=load_scene(old['map_identity'])['mixed']
                    inputs.append(old['map_identity'])
                else:
                    points,shape=grid(10)
                    dpd_map=Spectrum(scene['components'].sum(0)+scene['noise'],guard=guard).evaluate(points).reshape(shape)
                result=run_case(scene,dpd_map,guard,oracle=g<8)
                result['role']='calibration' if g<16 else 'check'
                rows.append(result)
                write(out/'rows.json',rows)
                print(f'布局{g} overlap={overlap}完成；累计{(time.perf_counter()-start)/60:.1f}分钟',flush=True)
        for item in inputs+sources:
            checked_bytes(item)
        seconds=time.perf_counter()-start
        write(out/'report.json',dict(gate='G8-R5',status='SHORT_COMPLETED' if args.short else 'COMPLETED',
            seconds=seconds,cumulative_G8_seconds=used+previous+seconds,summary=aggregate(rows),
            no_training=True,test_read=False,peak_ram_percent=guard.peak_ram_percent,
            peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30))
        outputs=[identity(p) for p in out.rglob('*') if p.is_file()]
        write(out/'final_audit_report.json',dict(status='PASS',scope='G8_R5_short' if args.short else 'G8_R5_matched_model_knownK',
            inputs=inputs,sources=sources,outputs=outputs))
        print(f'完成：{out}；{seconds/60:.2f}分钟',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':
    main()
