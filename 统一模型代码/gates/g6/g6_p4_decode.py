"""Frozen shared proposals and two scores; deployable decoding never sees labels."""
import itertools

import numpy as np
from scipy.special import logsumexp

from 统一模型代码.gates.g6.g6_p3_candidates import THRESHOLDS, decode, matching_counts

MODES = ('native', 'shared_final', 'shared_factorized')


def prepare(logits, record, final_probability, residual):
    active = np.flatnonzero(logits.max(-1) >= 0).tolist()
    pool = [dict(donor=q, rank=r+1, flat=int(flat), position=point)
            for q, c in enumerate(record['candidates'])
            for r, (flat, point) in enumerate(zip(c['flat_indices'], c['positions']))]
    points = np.asarray([p['position'] for p in pool], dtype=float).reshape(-1, 2)
    flats = [p['flat'] for p in pool]
    probability = np.asarray(final_probability).reshape(3, -1)[:, flats].astype(float)
    correction = np.asarray(residual).reshape(3, -1)[:, flats].astype(float)
    if not np.isfinite(probability).all() or not np.isfinite(correction).all():
        raise ValueError('Nonfinite candidate score')
    k, n = len(active), len(pool)
    combos = np.asarray(list(itertools.permutations(range(n), k)), dtype=int).reshape(-1, k) if k else np.empty((1, 0), int)
    feasible = np.ones(len(combos), bool)
    for a, b in itertools.combinations(range(k), 2):
        feasible &= np.linalg.norm(points[combos[:, a]]-points[combos[:, b]], axis=1) >= 30
    combos = combos[feasible]
    native_score = np.log(np.maximum(probability[active], 1e-20))
    log_q = np.log(np.maximum(probability.max(0), 1e-20))
    selected = correction[active]
    factorized = log_q[None]+selected-logsumexp(selected, axis=0, keepdims=True) if k else np.empty((0,n))
    return dict(active=active, pool=pool, points=points, combos=combos,
                final_probabilities=probability, residual=correction,
                scores={'shared_final': native_score, 'shared_factorized': factorized})


def decode_shared(logits, record, prepared, mode):
    if mode not in MODES:
        raise ValueError(mode)
    native = decode(logits, record)
    active, combos = prepared['active'], prepared['combos']
    if mode == 'native' or len(active) < 2 or not len(combos):
        result = dict(native)
        result.update(selected_pool=None, cross_slot_count=0,
                      shared_fallback=mode != 'native' and len(active) >= 2 and not len(combos))
        return result
    scores = prepared['scores'][mode]
    values = sum(scores[j, combos[:, j]] for j in range(len(active)))
    chosen = combos[int(np.argmax(values))]
    return dict(active=active, joint=prepared['points'][chosen].tolist(),
                fallback=False, shared_fallback=False, selected_pool=chosen.tolist(),
                cross_slot_count=sum(prepared['pool'][p]['donor'] != q for q,p in zip(active,chosen)),
                score=float(values.max()))


def upper_bounds(truth, logits, bands, ignore, prepared, native_upper):
    """Separate diagnostic only: exact nested spatial and identity upper bounds."""
    truth = np.asarray(truth).reshape(-1, 2)
    active, combos = prepared['active'], prepared['combos']
    k = len(active)
    points = prepared['points'][combos]
    dist = np.linalg.norm(truth[None, :, None]-points[:, None], axis=-1)
    separated = {str(d): int(matching_counts(dist <= d).max(initial=0)) for d in THRESHOLDS}
    band_ok = np.zeros((len(truth), k), bool)
    for t in range(len(truth)):
        mask = ignore[t] < .5
        y = bands[t, mask] > .5
        for j, q in enumerate(active):
            p = logits[q, mask] >= 0
            band_ok[t, j] = 2*np.sum(y & p)/max(int(y.sum()+p.sum()), 1) >= .8
    joint = int(matching_counts((dist <= 100) & band_ok[None]).max(initial=0))
    result = dict(union=native_upper['union'],
        pred_k_union={str(d): min(k, native_upper['union'][str(d)]) for d in THRESHOLDS},
        shared_separated=separated, native_feasible=native_upper['feasible'],
        shared_joint100=joint, native_joint100=native_upper['joint100'],
        shared_feasible_combinations=len(combos))
    for d in THRESHOLDS:
        name = str(d)
        assert result['native_feasible'][name] <= separated[name] <= result['pred_k_union'][name] <= result['union'][name]
    assert result['native_joint100'] <= joint <= separated['100']
    return result
