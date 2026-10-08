"""G8-R4: direct oracle weak-band comparison against audited R3 full maps."""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .peak_diagnosis import spectrum, trace, nearby, evaluate_positions
from ..physics import grid, FS
from ..scenes import BW
from ..diagnostics import summarize
from ..storage import BASE, Guard, read, write, identity, checked_bytes, load_scene, save_scene


def energies(scene, band):
    n = scene['noise'].shape[-1]
    freq = np.fft.fftshift(np.fft.fftfreq(n, 1/FS))
    mask = (freq >= band[0]) & (freq <= band[1])
    signals = np.concatenate([scene['components'], scene['noise'][None]], axis=0)
    power = abs(np.fft.fftshift(np.fft.fft(signals, axis=-1), axes=-1))**2/n**2
    full = power.sum(-1).mean(-1)
    kept = power[..., mask].sum(-1).mean(-1)
    return dict(order=['strong', 'weak', 'noise'], retained_fraction=(kept/full).tolist(),
        full_power=full.tolist(), band_power=kept.tolist(),
        full_strong_over_weak_db=float(10*np.log10(full[0]/full[1])),
        band_strong_over_weak_db=float(10*np.log10(kept[0]/kept[1])))


def summary(rows):
    result = {}
    for method in ('dpd', 'hr'):
        for overlap in (.5, 1., 'all'):
            selected = [r for r in rows if r['method']==method and (overlap=='all' or r['overlap']==overlap)]
            variants = {}
            for variant in ('full', 'weak_band'):
                data = [r[variant] for r in selected]
                item = summarize(data)
                item.update(weak_recall100=float(np.mean([r['source_recall']['100'][1] for r in data])),
                    strong_recall100=float(np.mean([r['source_recall']['100'][0] for r in data])),
                    ranks=[r[variant+'_peak']['best_rank'] for r in selected])
                variants[variant] = item
            # Counts are paired on the same IQ; no independent-sample interpretation.
            variants['weak_rescued'] = sum(not r['full']['source_recall']['100'][1] and r['weak_band']['source_recall']['100'][1] for r in selected)
            variants['weak_lost'] = sum(r['full']['source_recall']['100'][1] and not r['weak_band']['source_recall']['100'][1] for r in selected)
            result[f'{method}_{overlap}'] = variants
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    out = (BASE/args.run_id).resolve()
    if not out.is_relative_to(BASE.resolve()) or out==BASE.resolve():
        raise ValueError('Output root')
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(4)
    found = [(p.parent,read(p)) for p in BASE.glob('*/final_audit_report.json')
        if read(p).get('scope')=='G8_R3_development_diagnostic' and read(p).get('status')=='PASS']
    if len(found)!=1:
        raise RuntimeError('Resolve unique audited R3')
    r3,audit = found[0]
    inputs = [identity(r3/'final_audit_report.json')]+audit['outputs']+audit['inputs']+audit['sources']
    for item in inputs:
        checked_bytes(item)
    data_indexes = [Path(r['path']) for r in audit['inputs'] if Path(r['path']).name=='data_index.json']
    if len(data_indexes)!=1:
        raise RuntimeError('Resolve unique R2 data index')
    records = read(data_indexes[0])
    baseline = read(r3/'rows.json')
    cfg = read(r3/'manifest.json')['hr_config']
    previous = sum(read(p)['seconds'] for p in BASE.glob('*/report.json') if read(p).get('gate')=='G8-R4')
    used = read(r3/'report.json')['cumulative_G8_seconds']
    guard = Guard(out, min(1800-previous,43200-used-previous))
    sources = audit['sources']+[identity(Path(__file__))]
    groups = [16] if args.short else list(range(16,48))
    guard()
    write(out/'manifest.json',dict(gate='G8-R4',short=args.short,groups=groups,N=65536,
        hr_config=cfg,frequency_mask='oracle weak center +/- BW/2',known_K=2,
        no_training=True,test_read=False,prior_G8_seconds=used,prior_R4_seconds=previous,
        seconds_limit=guard.seconds,inputs=inputs,sources=sources))
    rows=[]
    try:
        points,shape = grid(10)
        print('阶段1：同一混合IQ，仅限制真实弱源频带；全频结果按审计身份复用',flush=True)
        for index,g in enumerate(groups):
            for overlap in (.5,1.):
                scene = load_scene(records[f'{g}_{overlap}'])
                band = (scene['metadata']['centers_hz'][1]-BW/2,scene['metadata']['centers_hz'][1]+BW/2)
                power = energies(scene,band)
                for method in ('dpd','hr'):
                    old = next(r for r in baseline if r['group']==g and r['overlap']==overlap and r['method']==method)
                    t0=time.perf_counter()
                    _,evaluate = spectrum(scene['components'].sum(0)+scene['noise'],method,*cfg,guard,band)
                    z=evaluate(points).reshape(shape)
                    tr=trace(z)
                    seconds=time.perf_counter()-t0
                    metric=evaluate_positions(scene['metadata']['positions'],tr['full_positions'],seconds)
                    row=dict(group=g,overlap=overlap,method=method,band_hz=list(band),energy=power,
                        full=old['predictions']['full_local'],full_peak=old['peak_detail'],
                        weak_band=metric,weak_band_peak=nearby(z,tr,np.asarray(scene['metadata']['positions'][1])))
                    row['map_identity']=save_scene(out/f'maps/{g}_{overlap}_{method}.npz',
                        dict(spectrum=z,metadata=dict(group=g,overlap=overlap,method=method,band_hz=list(band))))
                    rows.append(row)
            write(out/'rows.json',rows)
            if args.short or (index+1)%4==0:
                print(f'频带配对 {index+1}/{len(groups)}布局；累计{(time.perf_counter()-start)/60:.1f}分钟',flush=True)
        print('阶段2：分重叠条件汇总、身份复核',flush=True)
        for item in inputs+sources+[r['map_identity'] for r in rows]:
            checked_bytes(item)
        seconds=time.perf_counter()-start
        write(out/'report.json',dict(gate='G8-R4',status='SHORT_COMPLETED' if args.short else 'COMPLETED',
            seconds=seconds,cumulative_G8_seconds=used+previous+seconds,summary=summary(rows),
            no_training=True,test_read=False,peak_ram_percent=guard.peak_ram_percent,
            peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30))
        outputs=[identity(p) for p in out.rglob('*') if p.is_file()]
        write(out/'final_audit_report.json',dict(status='PASS',scope='G8_R4_short' if args.short else 'G8_R4_oracle_band_diagnostic',
            inputs=inputs,sources=sources,outputs=outputs))
        print(f'完成：{out}；{seconds/60:.2f}分钟',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':
    main()
