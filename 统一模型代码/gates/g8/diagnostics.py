"""G8可追溯诊断与指标；真实分量仅用于诊断，不进入部署推理。"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import stft
from .physics import FS, C, RECEIVERS, grid, Spectrum


def metrics(metadata, positions, bands=None, seconds=0.):
    truth = np.asarray(metadata['positions'], dtype=float).reshape(-1, 2)
    predicted = np.asarray(positions, dtype=float).reshape(-1, 2)
    # Same registered G7 GOSPA implementation, including its component convention.
    from .legacy import ROOT  # imports native module paths without creating a Runtime
    from s2g3_composability import gospa_sample, matched_distances, maximum_matches_within
    del ROOT
    g = gospa_sample(truth, predicted)
    matches = matched_distances(truth, predicted)
    errors = [d for _, _, d in matches]
    row = dict(gospa_m=g['value_m'], gospa_components=g, errors_m=errors,
        true_count=len(truth), predicted_count=len(predicted), exact_count=len(truth) == len(predicted),
        seconds=seconds, positions=predicted.tolist(),
        recall_counts={str(t):maximum_matches_within(truth, predicted, t) for t in (10, 30, 50, 100)},
        both_recovered100=bool(len(truth) and maximum_matches_within(truth, predicted, 100) == len(truth)),
        joint_tp=None, band_f1=None, band_iou=None)
    if bands is not None:
        bands = np.asarray(bands, dtype=float).reshape(-1, 19)
        if len(bands) != len(predicted):
            raise ValueError('One reported band per predicted position required')
        labels = np.asarray(metadata['bands'], dtype=float).reshape(-1, 19)
        ignore = np.asarray(metadata['ignore'], dtype=float).reshape(-1, 19)
        f1 = np.zeros((len(truth), len(predicted)))
        iou = np.zeros_like(f1)
        for t in range(len(truth)):
            for p in range(len(predicted)):
                valid = ignore[t] < .5
                a, b = labels[t, valid] > .5, bands[p, valid] > .5
                tp, fp, fn = (a & b).sum(), (~a & b).sum(), (a & ~b).sum()
                f1[t, p] = 2*tp/max(2*tp+fp+fn, 1)
                iou[t, p] = tp/max(tp+fp+fn, 1)
        row['joint_tp'] = 0
        if f1.size:
            eligible = (np.linalg.norm(truth[:, None]-predicted, axis=-1) <= 100) & (f1 >= .8)
            t, p = linear_sum_assignment(-eligible.astype(float))
            row['joint_tp'] = int(eligible[t, p].sum())
            t, p = linear_sum_assignment(-f1)
            row['band_f1'] = float(f1[t, p].sum()/max(len(truth), len(predicted)))
            row['band_iou'] = float(iou[t, p].sum()/max(len(truth), len(predicted)))
        elif len(truth):
            row['band_f1'], row['band_iou'] = 0., 0.
    return row


def summarize(rows):
    truth = sum(r['true_count'] for r in rows)
    errors = np.asarray([e for r in rows for e in r['errors_m']])
    joint = [r for r in rows if r['joint_tp'] is not None]
    return dict(scenes=len(rows), gospa_m=float(np.mean([r['gospa_m'] for r in rows])),
        rmse_m=float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
        matched_coverage=len(errors)/truth if truth else None,
        median_p90_p95_m=np.quantile(errors, [.5, .9, .95]).tolist() if len(errors) else None,
        exact_count=float(np.mean([r['exact_count'] for r in rows])),
        recall={str(t):sum(r['recall_counts'][str(t)] for r in rows)/truth if truth else None for t in (10, 30, 50, 100)},
        joint_recall=sum(r['joint_tp'] for r in joint)/sum(r['true_count'] for r in joint) if joint and sum(r['true_count'] for r in joint) else None,
        tail500_scenes=sum(any(e > 500 for e in r['errors_m']) for r in rows),
        both_recovered100=float(np.mean([r['both_recovered100'] for r in rows if r['true_count'] == 2])) if any(r['true_count'] == 2 for r in rows) else None,
        online_seconds=float(np.mean([r['seconds'] for r in rows])),
        band_f1=float(np.mean([r['band_f1'] for r in rows if r['band_f1'] is not None])) if any(r['band_f1'] is not None for r in rows) else None)


def cyclic_features(iq):
    """Blind exploratory cycle-frequency energy peaks, NOT a validated R04 detector.

    Search the same nonzero cycle range for every sample; do not insert true rates.
    Report conjugate/nonconjugate lag0/1/2/4/8 strengths. No independence claim.
    """
    x = np.asarray(iq)
    n = x.shape[-1]
    scores = []
    for lag in (0, 1, 2, 4, 8):
        a, b = x[:, lag:], x[:, :n-lag]
        den = np.mean(abs(a)**2, axis=-1).clip(1e-30)
        freq = np.fft.fftfreq(a.shape[-1], 1/FS)
        ids = np.flatnonzero((abs(freq) >= .5e6) & (abs(freq) <= 25e6))
        for conjugate in (False, True):
            product = a*(b.conj() if conjugate else b)
            spec = abs(np.fft.fft(product, axis=-1))/product.shape[-1]/den[:, None]
            score = spec[:, ids].mean(0)
            best = ids[np.argsort(-score, kind='stable')[:5]]
            scores.append(dict(lag_samples=lag, conjugate=conjugate,
                frequencies_hz=freq[best].tolist(), strengths=spec[:, best].mean(0).tolist()))
    return scores


def information(scene, cycle_threshold=None):
    meta, iq = scene['metadata'], scene['iq']
    pos = np.asarray(meta['positions']).reshape(-1, 2)
    delays = np.linalg.norm(pos[:, None]-RECEIVERS, axis=-1)/C
    out = dict(mixed_cyclic_features=cyclic_features(iq),
        cycle_status='EXPLORATORY_FEATURES_NOT_METHOD_REJECTION',
        delays_s=delays.tolist(), components_available='components' in scene)
    if cycle_threshold is not None:
        out['cycle_noise_threshold'] = cycle_threshold
        out['above_calibrated_noise'] = [bool(max(r['strengths']) > cycle_threshold)
                                         for r in out['mixed_cyclic_features']]
    if 'components' not in scene:
        return out
    signal, noise = scene['components'], scene['noise']
    args = dict(fs=FS, window='hann', nperseg=1024, noverlap=512,
                boundary=None, padded=False, return_onesided=False, axis=-1)
    _, _, z = stft(signal, **args)
    _, _, zn = stft(noise, **args)
    energy, ne = abs(z)**2, abs(zn)**2
    ratio = energy/np.maximum(energy.sum(0)[None]-energy, 1e-30)
    snr = energy/np.maximum(ne[None], 1e-30)
    dominant = (ratio >= 10**.6) & (snr >= 1)
    out.update(single_source_dominant_fraction=dominant.mean((-1, -2)).tolist(),
        same_source_dominant_at_least3stations=(dominant.sum(1) >= 3).mean((-1, -2)).tolist(),
        sir_db_quantiles=np.quantile(10*np.log10(ratio.clip(1e-30)), [.1, .5, .9], axis=(-1, -2)).tolist(),
        stft_frames=z.shape[-1], stft_frequency_resolution_hz=FS/1024,
        oracle_component_cyclic=[cyclic_features(s) for s in signal])
    return out


def case_figure(scene, predictions, path, guard=lambda: None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    meta = scene['metadata']
    pos = np.asarray(meta['positions']).reshape(-1, 2)
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(RECEIVERS[:, 0], RECEIVERS[:, 1], marker='^', label='Receivers')
    if len(pos):
        ax.scatter(pos[:, 0], pos[:, 1], marker='*', s=120, label='Truth')
    for name, row in predictions.items():
        p = np.asarray(row['positions']).reshape(-1, 2)
        if len(p):
            ax.scatter(p[:, 0], p[:, 1], marker='x', label=name)
    ax.set(xlim=(-1000, 1000), ylim=(-1000, 1000), xlabel='x (m)', ylabel='y (m)', title='Geometry and predictions')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    x = scene['iq']
    f = np.fft.fftshift(np.fft.fftfreq(x.shape[-1], 1/FS))/1e6
    for i, spectrum in enumerate(abs(np.fft.fftshift(np.fft.fft(x), axes=-1))**2):
        axes[0, 1].plot(f, 10*np.log10(spectrum.clip(1e-30)), linewidth=.6, label=f'RX{i+1}')
    axes[0, 1].set(xlabel='Baseband frequency (MHz)', ylabel='FFT power (dB, relative)', title='Measured mixture spectra')
    axes[0, 1].legend(fontsize=8)
    pts, shape = grid()
    raw = Spectrum(x, guard=guard).evaluate(pts).reshape(shape)
    image = axes[1, 0].imshow(np.log1p(raw), origin='lower', extent=(-1000, 1000, -1000, 1000), cmap='viridis')
    axes[1, 0].set(xlabel='x (m)', ylabel='y (m)', title='Full-band DPD, log(1+P)')
    fig.colorbar(image, ax=axes[1, 0])
    power = meta.get('model_snr_db')
    if power is not None:
        for i, values in enumerate(power):
            axes[1, 1].plot(range(1, 5), values, '-o', label=f'Source{i+1}')
        axes[1, 1].set(xlabel='Receiver', ylabel='Model in-band SNR (dB)', xticks=range(1, 5), title='Source-wise received quality')
        axes[1, 1].legend()
    else:
        axes[1, 1].text(.1, .5, 'No per-source waveform in old data', transform=axes[1, 1].transAxes)
    fig.suptitle(f"G8 {meta['origin']} group={meta['group']} N={x.shape[-1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
