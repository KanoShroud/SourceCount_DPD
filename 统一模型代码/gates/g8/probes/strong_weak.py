"""G8-R2: paired isolated/mixed physical reference, no network training."""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .length_priority import hr, equivalence
from ..physics import Spectrum, grid, search
from ..scenes import components
from ..storage import BASE, ROOT, Guard, identity, checked_bytes, read, write, save_scene, load_scene
from ..diagnostics import metrics, summarize

NS = (4096, 16384, 65536)
KINDS = ('weak', 'strong', 'mixed')


def configs(n):
    return [(j, load) for j in ((2, 8) if n != 65536 else (8, 32)) for load in (1e-4, 1e-2)]


def make(group, overlap):
    pos, gain, powers, centers, unit, noise, _ = components(group, overlap, maximum=65536)
    signal = unit*np.sqrt(powers[0, :, None, None]*gain[..., None])
    rx = powers[0, :, None]*gain
    assert np.all(rx[0] > rx[1])
    return dict(components=signal, noise=noise, metadata=dict(group=group, overlap=overlap,
        positions=pos.tolist(), centers_hz=centers.tolist(),
        power_gap_db=float(10*np.log10(rx[0].mean()/rx[1].mean()))))


def evaluate(scene, kind, n, method, j, load, guard, with_map=False):
    ids = [1] if kind == 'weak' else [0] if kind == 'strong' else [0, 1]
    iq = scene['components'][ids, :, :n].sum(0)+scene['noise'][:, :n]
    truth = np.asarray(scene['metadata']['positions'])[ids]
    torch.cuda.synchronize()
    start = time.perf_counter()
    calc = Spectrum(iq, guard=guard) if method == 'dpd' else hr(iq, j, load, guard)
    result = search(calc, len(ids))
    torch.cuda.synchronize()
    seconds = time.perf_counter()-start
    row = metrics(dict(positions=truth.tolist()), result['positions'], seconds=seconds)
    # Maximize valid matches first, then minimize distance; a prediction cannot serve two sources.
    d = np.linalg.norm(truth[:, None]-np.asarray(result['positions']).reshape(-1, 2), axis=-1)
    source_recall = {}
    for threshold in (30, 50, 100):
        a, b = linear_sum_assignment((d > threshold)*1e6+d)
        flags = {source: False for source in ids}
        for x, y in zip(a, b):
            flags[ids[x]] = bool(d[x, y] <= threshold)
        source_recall[str(threshold)] = flags
    row.update(scene['metadata'], kind=kind, N=n, method=method, J=j, loading=load,
        predicted_positions=result['positions'], source_recall=source_recall, info=calc.info)
    # Metadata positions denote truth; preserve predictions separately.
    if with_map:
        points, shape = grid()
        row['map'] = calc.evaluate(points).reshape(shape).tolist()
    return row


def summary(rows):
    result = summarize(rows)
    for source, name in [(0, 'strong'), (1, 'weak')]:
        flags = [r['source_recall']['100'][str(source)] for r in rows
                 if str(source) in r['source_recall']['100']]
        result[name+'_recall100'] = float(np.mean(flags)) if flags else None
    return result


def report_results(rows):
    table, strata, drops = {}, {}, {}
    for method in ('dpd', 'hr'):
        for n in NS:
            subset = [r for r in rows if r['method'] == method and r['N'] == n]
            for kind in KINDS:
                table[f'{method}_{n}_{kind}'] = summary([r for r in subset if r['kind'] == kind])
            mixed = [r for r in subset if r['kind'] == 'mixed']
            for overlap in (.5, 1.):
                strata[f'{method}_{n}_overlap{overlap}'] = summary([r for r in mixed if r['overlap'] == overlap])
            for lo, hi in ((0, 6), (6, 12), (12, 21)):
                selected = [r for r in mixed if lo <= r['power_gap_db'] < hi]
                if selected:
                    strata[f'{method}_{n}_gap{lo}_{hi}'] = summary(selected)
            deltas = []
            for group in sorted({r['group'] for r in subset}):
                isolated = [r['source_recall']['100']['1'] for r in subset if r['group'] == group and r['kind'] == 'weak']
                combined = [r['source_recall']['100']['1'] for r in mixed if r['group'] == group]
                deltas.append(float(np.mean(isolated)-np.mean(combined)))
            rng = np.random.default_rng(2026092902)
            boot = np.asarray(deltas)[rng.integers(0, len(deltas), (2000, len(deltas)))].mean(-1)
            drops[f'{method}_{n}'] = dict(isolated_minus_mixed=float(np.mean(deltas)),
                layout_bootstrap95=np.quantile(boot, [.025, .975]).tolist(), layouts=len(deltas))
    return dict(table=table, strata=strata, weak_recovery_drop=drops)


def figures(out, rows, records, selected, guard):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = out/'figures'
    folder.mkdir()
    # Representative cases chosen transparently: first long-HR failure and first success.
    mixed = [r for r in rows if r['method'] == 'hr' and r['N'] == 65536 and r['kind'] == 'mixed']
    chosen = []
    for success in (False, True):
        candidates = [r for r in mixed if r['both_recovered100'] == success]
        if candidates:
            chosen.append(candidates[0])
    if not chosen:
        chosen = mixed[:1]
    for example in chosen:
        g, overlap = example['group'], example['overlap']
        scene = load_scene(records[f'{g}_{overlap}'])
        panels = []
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for row_index, method in enumerate(('dpd', 'hr')):
            j, load = selected['65536'] if method == 'hr' else (None, None)
            for col, kind in enumerate(KINDS):
                r = evaluate(scene, kind, 65536, method, j, load, guard, True)
                panels.append(r)
                ax = axes[row_index, col]
                z = np.asarray(r['map'])
                z = 10*np.log10(np.maximum(z/z.max(), 1e-12))
                im = ax.imshow(z, origin='lower', extent=[-1000, 1000, -1000, 1000], cmap='viridis')
                for source, marker, label in [(0, '*', 'Strong truth'), (1, 'o', 'Weak truth')]:
                    if kind == 'mixed' or kind == ('strong' if source == 0 else 'weak'):
                        x, y = scene['metadata']['positions'][source]
                        ax.scatter(x, y, marker=marker, s=90, facecolors='none', edgecolors='red', label=label)
                p = np.asarray(r['predicted_positions'])
                ax.scatter(p[:, 0], p[:, 1], marker='x', c='white', label='Prediction')
                ax.set(title=f'{method.upper()} / {kind}', xlabel='x (m)', ylabel='y (m)')
                ax.legend(fontsize=7)
                fig.colorbar(im, ax=ax, label='Relative spectrum (dB)')
        fig.suptitle(f'Layout {g}, overlap={overlap}, N=65536; each panel normalized independently')
        stem = folder/f'layout_{g}_overlap_{overlap}'
        fig.savefig(str(stem)+'.png', dpi=160)
        fig.savefig(str(stem)+'.svg')
        plt.close(fig)
        write(str(stem)+'.json', dict(selection='first failure/success in group order, explanatory only', panels=panels))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    out = (BASE/args.run_id).resolve()
    if not out.is_relative_to(BASE.resolve()) or out == BASE.resolve():
        raise ValueError('Output root')
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    start = time.perf_counter()
    prior = []
    for p in BASE.glob('*/final_audit_report.json'):
        a = read(p)
        if a.get('status') == 'PASS' and a.get('scope') == 'G8_R1_length_and_fixed4096_diagnostic':
            prior.append((p.parent, a))
    if len(prior) != 1:
        raise RuntimeError('Expected one audited R1')
    r1, audit = prior[0]
    inputs = [identity(r1/'final_audit_report.json'), audit['report'], identity(r1/'long_records.json')]
    for item in inputs:
        checked_bytes(item)
    used = read(r1/'report.json')['cumulative_G8_seconds']
    short_seconds = 0.
    for p in BASE.glob('*/report.json'):
        r = read(p)
        if r.get('gate') == 'G8-R2':
            short_seconds += r['seconds']
    guard = Guard(out, min(7200-short_seconds, 43200-used-short_seconds))
    sources = [identity(Path(__file__)), identity(Path(__file__).with_name('length_priority.py'))]
    sources += [identity(ROOT/'统一模型代码/gates/g8'/name) for name in ('physics.py', 'scenes.py', 'storage.py', 'diagnostics.py', 'legacy.py')]
    sources += [identity(ROOT/'第四章代码/s2g3_composability.py'), identity(ROOT/'DPD_MVDR/DPD_MVDR.py')]
    for item in audit['sources']:
        checked_bytes(item)
    inputs += audit['sources']
    groups = [0, 16] if args.short else list(range(48))
    write(out/'manifest.json', dict(gate='G8-R2', short=args.short, groups=groups, N=NS,
        configs={n:configs(n) for n in NS}, kinds=KINDS, selection='mixed both100 desc, GOSPA asc, config order',
        inputs=inputs, sources=sources, no_training=True, test_read=False, prior_G8_seconds=used,
        prior_R2_seconds=short_seconds, budget_seconds=guard.seconds))
    try:
        guard()
        write(out/'equivalence.json', equivalence())
        # Metric test: one coincident prediction must not count as recovering two sources.
        test = metrics(dict(positions=[[0, 0], [200, 0]]), [[0, 0], [0, 0]])
        assert test['recall_counts']['100'] == 1
        records, calibration, checks = {}, [], []
        r1_records = read(r1/'long_records.json')
        print('阶段1：生成配对长记录并核对R1分量；只持久化分量与噪声', flush=True)
        for group in groups:
            for overlap in (.5, 1.):
                guard()
                scene = make(group, overlap)
                if group < 8 and overlap == .5:
                    old = load_scene(r1_records[group])
                    np.testing.assert_array_equal(scene['components'][1], old['components'][0])
                    np.testing.assert_array_equal(scene['noise'], old['noise'])
                    inputs.append(r1_records[group])
                records[f'{group}_{overlap}'] = save_scene(out/f'data/{group}_{overlap}.npz', scene)
        write(out/'data_index.json', records)
        print('阶段2：仅校准组双源选参', flush=True)
        for group in [g for g in groups if g < 16]:
            for overlap in (.5, 1.):
                scene = load_scene(records[f'{group}_{overlap}'])
                for n in NS:
                    for j, load in configs(n):
                        calibration.append(evaluate(scene, 'mixed', n, 'hr', j, load, guard))
            write(out/'calibration.json', calibration)
            if group % 4 == 3:
                print(f'校准 {group+1}/16；累计{(time.perf_counter()-start)/60:.1f}分钟', flush=True)
        selected = {}
        for n in NS:
            def key(cfg):
                subset = [r for r in calibration if r['N'] == n and (r['J'], r['loading']) == cfg]
                return (-np.mean([r['both_recovered100'] for r in subset]), np.mean([r['gospa_m'] for r in subset]))
            selected[str(n)] = list(min(configs(n), key=key))
        write(out/'selection_frozen.json', selected)
        selection_identity = identity(out/'selection_frozen.json')
        print(f'阶段3：配置冻结，检查组单/双源比较：{selected}', flush=True)
        for group in [g for g in groups if g >= 16]:
            for overlap in (.5, 1.):
                scene = load_scene(records[f'{group}_{overlap}'])
                for n in NS:
                    for kind in KINDS:
                        checks.append(evaluate(scene, kind, n, 'dpd', None, None, guard))
                        j, load = selected[str(n)]
                        checks.append(evaluate(scene, kind, n, 'hr', j, load, guard))
            write(out/'check_rows.json', checks)
            if group % 4 == 3:
                print(f'检查 {group-15}/32；累计{(time.perf_counter()-start)/60:.1f}分钟', flush=True)
        checks = read(out/'check_rows.json')
        print('阶段4：配对统计、空间谱图与最终审计', flush=True)
        result = report_results(checks)
        figures(out, checks, records, selected, guard)
        for item in inputs+sources+list(records.values())+[selection_identity]:
            checked_bytes(item)
        seconds = time.perf_counter()-start
        result.update(gate='G8-R2', status='SHORT_COMPLETED' if args.short else 'COMPLETED',
            seconds=seconds, cumulative_G8_seconds=used+short_seconds+seconds,
            peak_ram_percent=guard.peak_ram_percent, peak_cuda_GiB=torch.cuda.max_memory_allocated()/2**30,
            selection=selected, no_training=True, test_read=False, checks=len(checks), calibration=len(calibration))
        write(out/'report.json', result)
        outputs = [identity(p) for p in out.rglob('*') if p.is_file() and 'data' not in p.relative_to(out).parts]
        write(out/'final_audit_report.json', dict(status='PASS', scope='G8_R2_short' if args.short else 'G8_R2_development_knownK',
            sources=sources, inputs=inputs, data=list(records.values()), outputs=outputs))
        print(f'完成：{out}；{seconds/60:.2f}分钟', flush=True)
    except BaseException as exc:
        write(out/'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), seconds=time.perf_counter()-start))
        raise


if __name__ == '__main__':
    main()
