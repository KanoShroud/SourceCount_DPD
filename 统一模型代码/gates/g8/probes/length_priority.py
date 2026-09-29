"""G8-R1：先查IQ长度；固定4096点排查弱源。无训练，输出独立。"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
from pathlib import Path
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

from ..physics import Spectrum, FS, C, RECEIVERS, grid, search
from ..scenes import components
from ..storage import BASE, ROOT, Guard, read, write, checked_bytes, identity, save_scene, load_scene
from ..legacy import Sources, Models
from DPD_MVDR.DPD_MVDR import DPD_MVDR


def hr(iq, segments, loading, guard=lambda: None, frequency_range=None, device='cuda'):
    """Same HR equation; extend only J and original FrequencyRange capability.

    Spectrum.evaluate is reused unmodified. No frequency subsampling or segment overlap.
    """
    iq = np.asarray(iq, dtype=np.complex128)
    if iq.ndim != 2 or iq.shape[0] != 4 or not np.isfinite(iq).all():
        raise ValueError('Invalid IQ')
    n = iq.shape[1]
    if segments < 1 or n % segments or loading <= 0:
        raise ValueError('Registered records must divide exactly into positive segments')
    nf = n//segments
    f = np.fft.fftshift(np.fft.fftfreq(nf, 1/FS))
    mask = np.ones(nf, bool) if frequency_range is None else (f >= frequency_range[0]) & (f <= frequency_range[1])
    x = np.fft.fftshift(np.fft.fft(iq.reshape(4, segments, nf), axis=-1), axes=-1)[..., mask]
    covariance = np.einsum('mjf,njf->fmn', x, x.conj())/segments
    covariance = (covariance+covariance.conj().transpose(0, 2, 1))/2
    loads = loading*np.trace(covariance, axis1=1, axis2=2).real/4
    loaded = covariance+loads[:, None, None]*np.eye(4)
    eigenvalues = np.linalg.eigvalsh(loaded)
    ratios = eigenvalues[:, 0]/eigenvalues[:, -1]
    if not np.all(ratios > 64*np.finfo(float).eps):
        raise ValueError('Covariance not positive definite')
    calc = object.__new__(Spectrum)
    calc.device, calc.method, calc.guard = device, 'hr', guard
    calc.fs, calc.n = FS, n
    calc.receivers = torch.as_tensor(RECEIVERS, dtype=torch.float64, device=device)
    calc.pairs = list(itertools.combinations(range(4), 2))
    calc.inverse = torch.as_tensor(np.linalg.inv(loaded), device=device)
    calc.freq = torch.as_tensor(f[mask], device=device)
    delay = np.linalg.norm(RECEIVERS[:, None]-RECEIVERS, axis=-1).max()/C
    calc.info = dict(N_total=n, J=segments, N_fft=nf, frequencies=int(mask.sum()),
        segment_margin=(nf/FS)/delay, loading=loading, condition_max=float(1/ratios.min()),
        frequency_range_hz=None if frequency_range is None else list(frequency_range))
    return calc


def equivalence():
    rng = np.random.default_rng(2026092708)
    iq = rng.normal(size=(4, 4096))+1j*rng.normal(size=(4, 4096))
    points, shape = grid(50, 50)
    errors = []
    for j in (4, 32):
        for band in (None, (-5e6, 7e6)):
            for loading in (1e-4, 1e-2):
                _, expected, _ = DPD_MVDR(RECEIVERS, iq, [0, 0], 50, 50, FS, FS, 0,
                    dict(J=j, DiagLoad=loading, FrequencyRange=band))
                actual = hr(iq, j, loading, frequency_range=band).evaluate(points).reshape(shape)
                np.testing.assert_allclose(actual, expected.T, rtol=2e-8, atol=1e-15)
                errors.append(float(np.max(abs(actual-expected.T))/np.max(abs(expected))))
    return dict(status='PASS', configurations=8, max_relative_error=max(errors))


def measured(scene, method, n, j, loading, guard, frequency_range=None):
    t0 = time.perf_counter()
    iq = scene['iq'][:, :n]
    calc = (Spectrum(iq, guard=guard) if method == 'dpd' else
            hr(iq, j, loading, guard, frequency_range))
    pred = search(calc, 1)['positions'][0]
    err = float(np.linalg.norm(np.asarray(pred)-scene['metadata']['positions'][0]))
    return dict(group=scene['metadata']['group'], method=method, N=n, J=j, loading=loading,
        error_m=err, recall100=err <= 100, position=pred, seconds=time.perf_counter()-t0,
        frequency_mode='all' if frequency_range is None else 'oracle_band', info=calc.info)


def aggregate(rows):
    groups = {}
    for row in rows:
        key = f"{row['method']}_N{row['N']}_J{row['J']}_load{row['loading']}_{row['frequency_mode']}"
        groups.setdefault(key, []).append(row)
    return {k:dict(cases=len(v), recall100=float(np.mean([r['recall100'] for r in v])),
        rmse_m=float(np.sqrt(np.mean([r['error_m']**2 for r in v]))),
        median_error_m=float(np.median([r['error_m'] for r in v])),
        mean_seconds=float(np.mean([r['seconds'] for r in v])),
        segment_margin=v[0]['info'].get('segment_margin')) for k,v in groups.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', default='r1_length_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()
    out = (BASE/args.run_id).resolve()
    if not out.is_relative_to(BASE.resolve()) or out == BASE.resolve():
        raise ValueError('Wrong output root')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    started = time.perf_counter()
    # Discover completed first-round evidence, excluding short probes and this new run.
    found = [(p.parent, read(p)) for p in BASE.glob('*/final_audit_report.json')
             if read(p).get('scope') == 'first_round_outputs_not_research_gap_verdict' and read(p).get('status') == 'PASS']
    if len(found) != 1:
        raise RuntimeError('Resolve unique completed G8 run before diagnostic execution')
    prior, audit = found[0]
    inputs = [identity(prior/'final_audit_report.json'), identity(prior/'manifest.json'), identity(prior/'data_index.json')]
    for row in audit['outputs']:
        checked_bytes(row)
        inputs.append(row)
    used = float(read(prior/'budget.json')['active_seconds'])
    guard = Guard(out, seconds=min(1800, 43200-used))
    sources = [identity(Path(__file__)), identity(ROOT/'DPD_MVDR/DPD_MVDR.py')]
    sources += [r for r in read(prior/'manifest.json')['sources'] if Path(r['path']).exists()]
    for row in sources:
        checked_bytes(row)
    guard()
    write(out/'manifest.json', dict(gate='G8-R1', groups=list(range(8)), source_index=1,
        N=[4096, 16384, 65536], J_primary=[2,4,8], J_extra_65536=[32], loads=[1e-4,1e-2],
        fixed_segment_chain=[[4096,2],[16384,8],[65536,32]], prior_active_seconds=used,
        maximum_seconds=guard.seconds, inputs=inputs, sources=sources,
        network_N=4096, no_training=True, test_read=False,
        fallback_rule='best_fullband_65536_recall_below_half',
        long_record_note='new65536 realization with exact short prefixes, not old G8 bytes'))
    try:
        print('阶段1：任意分段/频段扩展与原DPD-HR公式等价检查', flush=True)
        write(out/'equivalence.json', equivalence())
        scenes, rows, scene_inputs = [], [], []
        for group in range(8):
            pos, gain, power, centers, unit, noise, _ = components(group, .5, maximum=65536)
            weak = unit[1]*np.sqrt(power[0,1]*gain[1,:,None])
            scene = dict(iq=weak+noise, components=weak[None], noise=noise,
                metadata=dict(group=group, count=1, positions=[pos[1].tolist()],
                              center_hz=float(centers[1]), bandwidth_hz=12.5e6, fs_hz=FS, samples=65536))
            record = save_scene(out/f'data/long_{group}.npz', scene)
            scene_inputs.append(record)
            scenes.append(load_scene(record))
        write(out/'long_records.json', scene_inputs)
        print('阶段2：全频长度优先比较；固定J与固定每段长度', flush=True)
        for i, scene in enumerate(scenes):
            for n in (4096,16384,65536):
                guard()
                rows.append(measured(scene, 'dpd', n, None, None, guard))
                for j in ((2,4,8) if n != 65536 else (2,4,8,32)):
                    for loading in (1e-4,1e-2):
                        rows.append(measured(scene, 'hr', n, j, loading, guard))
            write(out/'length_rows.json', rows)
            print(f'长度比较 {i+1}/8，累计{(time.perf_counter()-started)/60:.1f}分钟', flush=True)
        primary = aggregate(rows)
        write(out/'primary_summary.json', primary)
        best_long = max(v['recall100'] for k,v in primary.items() if k.startswith('hr_N65536'))
        fallback = best_long < .5
        if fallback:
            print('阶段3：长记录全频仍失常，追加真实频带诊断（额外信息，不是部署排名）', flush=True)
            for scene in scenes:
                m = scene['metadata']
                band = (m['center_hz']-m['bandwidth_hz']/2, m['center_hz']+m['bandwidth_hz']/2)
                for n in (4096,16384,65536):
                    for loading in (1e-4,1e-2):
                        rows.append(measured(scene, 'hr', n, 4, loading, guard, band))
            write(out/'all_length_rows.json', rows)
        print('阶段4：固定4096点，冻结网络对比混合与单独弱源', flush=True)
        frozen = Sources(out)
        index = read(prior/'data_index.json')
        chosen = [next(e for e in index if e['metadata']['origin']=='new' and e['metadata']['group']==g
            and e['metadata']['dominance']=='same' and e['metadata']['samples']==4096
            and e['metadata']['overlap_iou']==.5 and not e['metadata']['total_power_control']) for g in range(8)]
        network = []
        for seed in frozen.seeds:
            model = Models(frozen, seed)
            for e in chosen:
                marker = read(prior/f'data/{e["id"]}.json.identity.json')
                assert json.loads(checked_bytes(marker)) == e
                scene = load_scene(e['identity'])
                guard()
                result = model.infer(scene['components'][1]+scene['noise'], guard)
                marker2 = read(prior/f'calibration/systems/{seed}/{e["id"]}.json.identity.json')
                mixed = json.loads(checked_bytes(marker2))['methods']
                target = np.asarray(scene['metadata']['positions'][1])
                for kind, key in [('hard','B0'),('candidate','B1')]:
                    single = result[kind]['positions']
                    old = mixed[key]['positions']
                    def nearest(points):
                        return float(np.min(np.linalg.norm(np.asarray(points).reshape(-1,2)-target,axis=-1))) if len(points) else None
                    network.append(dict(seed=seed, group=e['metadata']['group'], model=key,
                        mixed_count=len(old), isolated_count=len(single),
                        mixed_weak_error_m=nearest(old), isolated_weak_error_m=nearest(single),
                        frozen_N=4096, raw_identity=e['identity']))
                inputs.extend([marker, marker2, e['identity']])
                write(out/'network_rows.json', network)
            del model
            torch.cuda.empty_cache()
            print(f'冻结网络seed {seed}完成', flush=True)
        # Digest checks cover consumed bytes and unchanged historical implementations.
        for row in inputs+sources+scene_inputs+frozen.inputs:
            if 'blocks' not in row:
                checked_bytes(row)
        summary = {}
        for seed in frozen.seeds:
            for model in ('B0','B1'):
                selected = [r for r in network if r['seed']==seed and r['model']==model]
                for condition in ('mixed','isolated'):
                    errors = [r[f'{condition}_weak_error_m'] for r in selected]
                    finite = [v for v in errors if v is not None]
                    summary[f'{seed}_{model}_{condition}'] = dict(cases=len(selected),
                        weak_recall100=sum(v is not None and v<=100 for v in errors)/len(errors),
                        nearest_weak_rmse_m=float(np.sqrt(np.mean(np.square(finite)))) if finite else None,
                        output_nonempty=len(finite)/len(errors),
                        counts=[r[f'{condition}_count'] for r in selected])
        seconds = time.perf_counter()-started
        report = dict(status='COMPLETED', no_training=True, test_read=False, seconds=seconds,
            cumulative_G8_seconds=used+seconds, primary=primary, band_diagnostic_triggered=fallback,
            all_physics=aggregate(rows), network=summary, sources_unchanged=True,
            scope='8_calibration_layouts_diagnostic_not_generalization_or_method_gap_proof')
        write(out/'report.json', report)
        write(out/'final_audit_report.json', dict(status='PASS', test_read=False,
            scope='G8_R1_length_and_fixed4096_diagnostic',
            report=identity(out/'report.json'), consumed_inputs=inputs,
            sources=sources, new_data=scene_inputs))
        print(f'完成：{out}；{seconds/60:.1f}分钟', flush=True)
    except BaseException as exc:
        write(out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                                      seconds=time.perf_counter()-started))
        raise


if __name__ == '__main__':
    main()
