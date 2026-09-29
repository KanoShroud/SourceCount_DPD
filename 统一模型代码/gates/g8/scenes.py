"""有限接收窗双源场景；真值分量与部署输入分离。

RRC脉冲按连续时刻求值，信号在观测窗前后均有独立符号；不循环移位。
功率按模型平均功率定标，不对长短接收窗分别归一化。
"""
from __future__ import annotations

import numpy as np
from .physics import FS, C, RECEIVERS, LO, HI

SEED = 2026092708
RATE = 10e6
BETA = .25
BW = RATE*(1+BETA)


def rrc(t, beta=BETA):
    t = np.asarray(t, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        h = (np.sin(np.pi*t*(1-beta))+4*beta*t*np.cos(np.pi*t*(1+beta)))/(np.pi*t*(1-(4*beta*t)**2))
    h = np.where(abs(t) < 1e-10, 1-beta+4*beta/np.pi, h)
    special = beta/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*beta))+(1-2/np.pi)*np.cos(np.pi/(4*beta)))
    h = np.where(abs(abs(t)-1/(4*beta)) < 1e-10, special, h)
    return np.where(abs(t) <= 10, h, 0.)


def layout(group):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, group, 1]))
    for attempt in range(10000):
        radii = np.sqrt(rng.uniform(100**2, 1000**2, 2))
        angles = rng.uniform(-np.pi, np.pi, 2)
        pos = np.stack([radii*np.cos(angles), radii*np.sin(angles)], -1)
        distances = np.linalg.norm(pos[:, None]-RECEIVERS, axis=-1)
        if distances.min() < 150 or np.linalg.norm(pos[0]-pos[1]) < 150:
            continue
        gain = 1/distances**2
        difference = 10*np.log10(gain[0]/gain[1])
        ratios = np.array([3.01-difference.min(), -(difference.min()+difference.max())/2])
        if np.ptp(difference) < 6.02:
            continue
        powers = []
        for ratio in ratios:
            tx = np.array([10**(ratio/10), 1.])
            mean_rx = (tx[:, None]*gain).mean(-1)
            tx *= (BW/FS)/mean_rx.min()
            powers.append(tx)
        powers = np.asarray(powers)
        avg = (powers[..., None]*gain).mean(-1)
        if np.max(abs(10*np.log10(avg[:, 0]/avg[:, 1]))) <= 20:
            return pos, gain, powers, attempt+1
    raise RuntimeError('Geometry cannot satisfy registered dominance pairing; no artificial fading fallback')


def components(group, overlap, maximum=16384):
    pos, gain, powers, attempts = layout(group)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, group, 2]))
    shift = BW*(1-overlap)/(1+overlap)
    center = rng.uniform(-FS/2+BW/2+BW/6, FS/2-BW/2-BW/6)
    centers = center+np.array([-shift/2, shift/2])
    if rng.random() < .5:
        centers = centers[::-1].copy()
    out = np.empty((2, 4, maximum), complex)
    # A common ensemble normalization, not a received-record normalization.
    dense = np.arange(-10, 10, .001)
    pulse_energy = np.sum(rrc(dense)**2)*.001
    for source in range(2):
        bits = rng.choice([-1., 1.], size=int(maximum*RATE/FS)+512)
        symbol_origin = -256
        timing, initial = rng.random(), rng.uniform(-np.pi, np.pi)
        for station in range(4):
            tau = np.linalg.norm(pos[source]-RECEIVERS[station])/C
            times = np.arange(maximum)/FS-tau
            u = times*RATE-timing
            base = np.floor(u).astype(int)
            value = np.zeros(maximum)
            for delta in range(-10, 11):
                symbol = base+delta
                value += bits[symbol-symbol_origin]*rrc(u-symbol)
            out[source, station] = value/np.sqrt(pulse_energy)*np.exp(2j*np.pi*centers[source]*times+1j*initial)
    noise_rng = np.random.default_rng(np.random.SeedSequence([SEED, group, 3]))
    noise = (noise_rng.normal(size=(4, maximum))+1j*noise_rng.normal(size=(4, maximum)))/np.sqrt(2)
    return pos, gain, powers, centers, out, noise, attempts


def make_scene(group, dominance, length, overlap, *, total_power_control=False):
    if length not in (4096, 16384) or dominance not in ('same', 'swapped') or overlap not in (.5, 1.):
        raise ValueError('Only registered primary scene conditions are implemented')
    pos, gain, powers, centers, unit, noise, attempts = components(group, overlap)
    branch = 0 if dominance == 'same' else 1
    tx = powers[branch].copy()
    if total_power_control:
        tx *= (powers[1, :, None]*gain).sum()/(tx[:, None]*gain).sum()
    signal = unit*np.sqrt(tx[:, None, None]*gain[..., None])
    signal, noise = signal[..., :length].copy(), noise[:, :length].copy()
    rx_power = tx[:, None]*gain
    ratio = 10*np.log10(rx_power[0]/rx_power[1])
    if dominance == 'same' and not (ratio >= 3).all():
        raise AssertionError('Same-source dominance failed')
    if dominance == 'swapped' and not (ratio.max() >= 3 and ratio.min() <= -3):
        raise AssertionError('Swapped-source dominance failed')
    full_overlap = np.maximum(0, np.minimum(centers[:, None]+BW/2, HI)-np.maximum(centers[:, None]-BW/2, LO))
    # Frozen G7 raw MAT declares hard19_actual_t020, not the generator default branch.
    bands = (full_overlap/10e6 >= .2).astype(float)
    ignore = ((full_overlap > 0) & (bands == 0)).astype(float)
    role = 'calibration' if group < 16 else 'check'
    meta = dict(group=group, role=role, origin='new', count=2, fs_hz=FS,
        samples=length, dominance=dominance, overlap_iou=overlap,
        total_power_control=total_power_control, positions=pos.tolist(),
        frequency_centers_hz=centers.tolist(), bandwidth_hz=[BW]*2,
        symbol_rate_hz=[RATE]*2, bands=bands.tolist(), ignore=ignore.tolist(),
        power_ratio_db=ratio.tolist(), rx_power=rx_power.tolist(),
        model_snr_db=(10*np.log10(rx_power/(BW/FS))).tolist(),
        measured_power=np.mean(abs(signal)**2, axis=-1).tolist(),
        noise_fullband_power=1., layout_draws=attempts,
        amplitude_model='relative_free_space_1_over_distance_squared',
        generation='finite_receive_window_continuous_RRC_span20symbols',
        label_profile='hard19_actual_t020', seed=SEED)
    return dict(iq=signal.sum(0)+noise, components=signal, noise=noise, metadata=meta)


def specifications():
    for group in range(48):
        for dominance in ('same', 'swapped'):
            for n in (4096, 16384):
                for overlap in (.5, 1.):
                    yield dict(group=group, dominance=dominance, length=n, overlap=overlap)
    for group in range(16, 24):
        for dominance in ('same', 'swapped'):
            yield dict(group=group, dominance=dominance, length=4096, overlap=.5, total_power_control=True)
