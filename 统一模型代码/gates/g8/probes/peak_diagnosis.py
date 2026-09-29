"""G8-R3: full fine spectrum, search attribution, conditional oracle-band probe."""
from __future__ import annotations

import argparse
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.optimize import linear_sum_assignment

from .length_priority import hr
from ..physics import Spectrum, grid, peaks, search, FS
from ..scenes import BW
from ..diagnostics import metrics, summarize
from ..storage import BASE, Guard, read, write, identity, checked_bytes, save_scene, load_scene


class MapSpectrum:
    """Exact lookup for existing search's 50m/10m lattice; no interpolation."""
    def __init__(self, z):
        self.z = z

    def evaluate(self, points):
        p = np.asarray(points)
        indices = np.rint((p+1000)/10).astype(int)
        np.testing.assert_allclose(indices*10-1000, p, atol=1e-10)
        return self.z[indices[:, 1], indices[:, 0]]


def local_ids(z):
    maximum = maximum_filter(z, 3, mode='constant', cval=-np.inf)
    minimum = minimum_filter(z, 3, mode='constant', cval=np.inf)
    return np.flatnonzero(((z == maximum) & (z > minimum)).ravel())


def trace(z):
    fine, _ = grid(10)
    coarse, shape = grid()
    values = MapSpectrum(z).evaluate(coarse)
    coarse_lm = values.reshape(shape) == maximum_filter(values.reshape(shape), 3, mode='constant', cval=-np.inf)
    ids = np.flatnonzero(coarse_lm.ravel())
    chosen = ids[peaks(coarse[ids], values[ids], 5)]
    centers = coarse[chosen]
    covered = np.any(np.all(abs(fine[:, None]-centers[None]) <= 50, axis=-1), axis=1)
    local = local_ids(z)
    inside = local[covered[local]]
    def decode(indices):
        if not len(indices):
            return []
        return fine[indices[peaks(fine[indices], z.ravel()[indices], 2)]].tolist()
    original = search(MapSpectrum(z), 2)
    return dict(original=original, window_positions=decode(inside), full_positions=decode(local),
        centers=centers, covered=covered, local_ids=local, points=fine)


def evaluate_positions(truth, positions, seconds):
    truth = np.asarray(truth)
    predicted = np.asarray(positions).reshape(-1, 2)
    result = metrics(dict(positions=truth.tolist()), positions, seconds=seconds)
    d = np.linalg.norm(truth[:, None]-predicted, axis=-1)
    recalls = {}
    for threshold in (30, 50, 100):
        a, b = linear_sum_assignment((d > threshold)*1e6+d)
        flags = [False]*len(truth)
        for i, j in zip(a, b):
            flags[i] = bool(d[i, j] <= threshold)
        recalls[str(threshold)] = flags
    result['source_recall'] = recalls
    return result


def nearby(z, traced, truth):
    ids = traced['local_ids']
    points = traced['points']
    distances = np.linalg.norm(points[ids]-truth, axis=-1)
    near = ids[distances <= 100]
    best = int(near[np.argmax(z.ravel()[near])]) if len(near) else None
    result = dict(counts={str(t):int(np.sum(distances <= t)) for t in (30, 50, 100)},
        covered_count=int(np.sum(traced['covered'][near])), best_position=None, best_rank=None,
        relative_peak_db=None, local_contrast_db=None)
    if best is not None:
        value = z.ravel()[best]
        ring_distance = np.linalg.norm(points-points[best], axis=-1)
        ring = z.ravel()[(ring_distance >= 30) & (ring_distance <= 60)]
        result.update(best_position=points[best].tolist(), best_rank=int(np.sum(z.ravel()[ids] > value)+1),
            relative_peak_db=float(10*np.log10(value/z.max())),
            local_contrast_db=float(10*np.log10(value/np.median(ring))))
    return result


def spectrum(iq, method, j, load, guard, band=None):
    if method == 'hr':
        calc = hr(iq, j, load, guard, frequency_range=band)
        return calc, lambda p: calc.evaluate(p)
    calc = Spectrum(iq, guard=guard)
    mask = None
    if band is not None:
        f = np.fft.fftshift(np.fft.fftfreq(iq.shape[1], 1/FS))
        mask = (f >= band[0]) & (f <= band[1])
    return calc, lambda p: calc.evaluate(p, mask=mask)


def run_case(scene, method, cfg, baseline, guard):
    started = time.perf_counter()
    iq = scene['components'].sum(0)+scene['noise']
    truth = np.asarray(scene['metadata']['positions'])
    points, shape = grid(10)
    _, evaluator = spectrum(iq, method, *cfg, guard)
    z = evaluator(points).reshape(shape)
    full_seconds = time.perf_counter()-started
    traced = trace(z)
    # Replay unchanged search from the dense map; verify actual R2 outputs.
    np.testing.assert_allclose(traced['original']['positions'], baseline['predicted_positions'], rtol=0, atol=1e-8)
    predictions = dict(original=evaluate_positions(truth, traced['original']['positions'], baseline['seconds']),
        window_local=evaluate_positions(truth, traced['window_positions'], full_seconds),
        full_local=evaluate_positions(truth, traced['full_positions'], full_seconds))
    detail = nearby(z, traced, truth[1])
    if predictions['original']['source_recall']['100'][1]:
        category = 'original_recovers_weak'
    elif detail['counts']['100'] == 0:
        category = 'no_nearby_local_peak'
    elif detail['covered_count'] == 0:
        category = 'outside_top5_windows'
    else:
        category = 'covered_but_not_selected'
    local_set = set(traced['local_ids'].tolist())
    original_points = np.asarray(traced['original']['positions'])
    ij = np.rint((original_points+1000)/10).astype(int)
    nonpeak = [int(y*201+x) not in local_set for x, y in ij]
    # Oracle-centered patch is solely a visualization/shape diagnostic, never decoded as a method.
    center = np.rint(truth[1]/10)*10
    axis = np.arange(-150, 151, 10)
    xaxis = (center[0]+axis)[abs(center[0]+axis) <= 1000]
    yaxis = (center[1]+axis)[abs(center[1]+axis) <= 1000]
    xx, yy = np.meshgrid(xaxis, yaxis)
    patch_points = np.stack([xx.ravel(), yy.ravel()], -1)
    _, isolated_eval = spectrum(scene['components'][1]+scene['noise'], method, *cfg, guard)
    isolated = isolated_eval(patch_points).reshape(xx.shape)
    mixed_patch = MapSpectrum(z).evaluate(patch_points).reshape(xx.shape)
    arrays = dict(mixed=z, patch_x=xaxis, patch_y=yaxis, isolated_patch=isolated, mixed_patch=mixed_patch)
    result = dict(**scene['metadata'], method=method, N=65536, category=category,
        peak_detail=detail, predictions=predictions, original_nonlocalmax=nonpeak,
        original_candidates=traced['original']['candidates'], original_scores=traced['original']['scores'],
        window_centers=traced['centers'].tolist(), full_map_seconds=full_seconds,
        full_local_peaks=traced['points'][traced['local_ids']].tolist(),
        full_local_scores=z.ravel()[traced['local_ids']].tolist(), oracle_band=None)
    if category == 'no_nearby_local_peak':
        band = (scene['metadata']['centers_hz'][1]-BW/2, scene['metadata']['centers_hz'][1]+BW/2)
        t0 = time.perf_counter()
        _, band_eval = spectrum(iq, method, *cfg, guard, band)
        band_map = band_eval(points).reshape(shape)
        band_trace = trace(band_map)
        result['oracle_band'] = dict(band_hz=list(band), peak_detail=nearby(band_map, band_trace, truth[1]),
            metrics=evaluate_positions(truth, band_trace['full_positions'], time.perf_counter()-t0))
        arrays['oracle_band_map'] = band_map
    result['total_seconds'] = time.perf_counter()-started
    return result, arrays


def aggregate(rows):
    report = {}
    for method in ('dpd', 'hr'):
        selected = [r for r in rows if r['method'] == method]
        variants = {}
        for name in ('original', 'window_local', 'full_local'):
            data = [r['predictions'][name] for r in selected]
            v = summarize(data)
            v.update(weak_recall100=float(np.mean([r['source_recall']['100'][1] for r in data])),
                strong_recall100=float(np.mean([r['source_recall']['100'][0] for r in data])))
            variants[name] = v
        oracle = [r['oracle_band'] for r in selected if r['oracle_band'] is not None]
        report[method] = dict(cases=len(selected), categories=dict(Counter(r['category'] for r in selected)),
            variants=variants, nonlocalmax_output_cases=sum(any(r['original_nonlocalmax']) for r in selected),
            nearby_peak_ranks=[r['peak_detail']['best_rank'] for r in selected if r['peak_detail']['best_rank'] is not None],
            oracle_cases=len(oracle), oracle_weak_success=sum(r['metrics']['source_recall']['100'][1] for r in oracle),
            oracle_peak_restored=sum(r['peak_detail']['counts']['100'] > 0 for r in oracle))
    return report


def draw(out, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    folder = out/'figures'
    folder.mkdir()
    selected = []
    for method in ('dpd', 'hr'):
        for category in ('no_nearby_local_peak', 'outside_top5_windows', 'covered_but_not_selected'):
            choices = [r for r in rows if r['method'] == method and r['category'] == category]
            if choices:
                selected.append(choices[0])
    for r in selected:
        arrays = load_scene(r['map_identity'])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        z = arrays['mixed']
        ax = axes[0]
        im = ax.imshow(10*np.log10(z/z.max()), origin='lower', extent=[-1000,1000,-1000,1000], cmap='viridis')
        for x, y in r['window_centers']:
            ax.add_patch(Rectangle((x-50,y-50),100,100, fill=False, edgecolor='white', linewidth=.6))
        truth = np.asarray(r['positions'])
        ax.scatter(*truth[0], marker='*', c='red', label='Strong truth')
        ax.scatter(*truth[1], marker='o', facecolors='none', edgecolors='red', label='Weak truth')
        for name, marker, color in [('original','x','white'), ('full_local','+','orange')]:
            p = np.asarray(r['predictions'][name]['positions']).reshape(-1,2)
            ax.scatter(p[:,0],p[:,1],marker=marker,c=color,label=name)
        ax.set_title('Mixed full map + Top5 windows')
        ax.legend(fontsize=6)
        fig.colorbar(im, ax=ax, label='Relative dB')
        for ax, key, title in zip(axes[1:], ('isolated_patch','mixed_patch'), ('Weak alone: diagnostic patch','Mixed: same patch')):
            z = arrays[key]
            im = ax.imshow(10*np.log10(z/z.max()), origin='lower',
                extent=[arrays['patch_x'][0],arrays['patch_x'][-1],arrays['patch_y'][0],arrays['patch_y'][-1]], cmap='viridis')
            ax.scatter(*truth[1],marker='o',facecolors='none',edgecolors='red')
            ax.set_title(title)
            fig.colorbar(im, ax=ax, label='Relative dB')
        for ax in axes:
            ax.set(xlabel='x (m)',ylabel='y (m)')
        fig.suptitle(f"{r['method'].upper()}, layout {r['group']}, overlap {r['overlap']}: {r['category']}\nEach panel normalized independently; patches are not deployable predictions")
        stem = folder/f"{r['method']}_{r['group']}_{r['overlap']}"
        fig.savefig(str(stem)+'.png',dpi=160)
        fig.savefig(str(stem)+'.svg')
        plt.close(fig)
    write(folder/'selection.json', [dict(group=r['group'],overlap=r['overlap'],method=r['method'],category=r['category']) for r in selected])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    out = (BASE/args.run_id).resolve()
    if not out.is_relative_to(BASE.resolve()) or out == BASE.resolve():
        raise ValueError('Output root')
    out.mkdir(parents=True,exist_ok=False)
    start = time.perf_counter()
    torch.set_num_threads(4)
    found = [(p.parent,read(p)) for p in BASE.glob('*/final_audit_report.json')
        if read(p).get('scope') == 'G8_R2_development_knownK' and read(p).get('status') == 'PASS']
    if len(found) != 1:
        raise RuntimeError('Expected unique R2')
    prior, audit = found[0]
    inputs = [identity(prior/'final_audit_report.json')]+audit['sources']+audit['outputs']
    for item in inputs:
        checked_bytes(item)
    records = read(prior/'data_index.json')
    baselines = read(prior/'check_rows.json')
    cfg = read(prior/'selection_frozen.json')['65536']
    previous = sum(read(p)['seconds'] for p in BASE.glob('*/report.json') if read(p).get('gate') == 'G8-R3')
    used = read(prior/'report.json')['cumulative_G8_seconds']
    guard = Guard(out, min(7200-previous,43200-used-previous))
    sources = audit['sources']+[identity(Path(__file__))]
    cases = [(16,.5)] if args.short else [(g,o) for g in range(16,48) for o in (.5,1.)]
    for g,o in cases:
        checked_bytes(records[f'{g}_{o}'])
        inputs.append(records[f'{g}_{o}'])
    guard()
    write(out/'manifest.json',dict(gate='G8-R3',short=args.short,cases=cases,N=65536,hr_config=cfg,
        grid_step_m=10,peak_neighborhood=3,separation_m=30,oracle_trigger='no_nearby_local_peak',
        no_training=True,test_read=False,prior_G8_seconds=used,prior_R3_seconds=previous,
        budget_seconds=guard.seconds,inputs=inputs,sources=sources))
    rows=[]
    try:
        print('阶段1：全区域细谱、原搜索回放与局部峰解码；缺峰时追加真实频带诊断',flush=True)
        for index,(g,o) in enumerate(cases):
            scene=load_scene(records[f'{g}_{o}'])
            for method in ('dpd','hr'):
                baseline=next(r for r in baselines if r['group']==g and r['overlap']==o and r['method']==method and r['kind']=='mixed' and r['N']==65536)
                result,arrays=run_case(scene,method,cfg,baseline,guard)
                result['map_identity']=save_scene(out/f'maps/{g}_{o}_{method}.npz',dict(**arrays,metadata=dict(group=g,overlap=o,method=method)))
                rows.append(result)
                write(out/'rows.json',rows)
            print(f'全图诊断 {index+1}/{len(cases)}；累计{(time.perf_counter()-start)/60:.1f}分钟',flush=True)
        print('阶段2：汇总、典型图与身份审计',flush=True)
        draw(out,rows)
        for item in inputs+sources+[r['map_identity'] for r in rows]:
            checked_bytes(item)
        seconds=time.perf_counter()-start
        report=dict(gate='G8-R3',status='SHORT_COMPLETED' if args.short else 'COMPLETED',
            seconds=seconds,cumulative_G8_seconds=used+previous+seconds,summary=aggregate(rows),
            peak_ram_percent=guard.peak_ram_percent,peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
            no_training=True,test_read=False,known_K=2,full_map_cost_note='window_local timings include diagnostic full map, not optimized deployment')
        write(out/'report.json',report)
        outputs=[identity(p) for p in out.rglob('*') if p.is_file()]
        write(out/'final_audit_report.json',dict(status='PASS',scope='G8_R3_short' if args.short else 'G8_R3_development_diagnostic',
            inputs=inputs,sources=sources,outputs=outputs))
        print(f'完成：{out}；{seconds/60:.2f}分钟',flush=True)
    except BaseException as exc:
        write(out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),seconds=time.perf_counter()-start))
        raise


if __name__=='__main__':
    main()
