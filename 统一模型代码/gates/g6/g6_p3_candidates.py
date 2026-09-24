"""Frozen candidate diagnostics; labels are never passed to deployable decoding."""
import copy
import itertools

import numpy as np
from scipy.ndimage import maximum_filter

from 统一模型代码.gates.g5.r1.g5_r1_decode import maximum_valid_pairs, positions_from_indices
from 统一模型代码.gates.g5.r2.g5_r2_model import decode_r2

THRESHOLDS = (10, 30, 50, 100)


def rescore_base(record, final_probability):
    result = copy.deepcopy(record)
    for q, candidate in enumerate(result['candidates']):
        ids = candidate['flat_indices']
        if not ids:
            raise ValueError('Empty base candidate set')
        candidate['scores'] = final_probability[q].ravel()[ids].astype(float).tolist()
        rank = int(np.argmax(candidate['scores']))
        result['original'][q] = candidate['positions'][rank]
    return result


def decode(logits, record):
    result = decode_r2(logits, record, None)
    if len(result['active']) == 1:
        q = result['active'][0]
        result['selected_ranks'] = [int(np.argmax(record['candidates'][q]['scores']))+1]
    return result


def matching_counts(valid):
    """Exact maximum cardinality for batches of matrices with <=3 rows/columns."""
    n, k, p = valid.shape
    counts = np.zeros(n, dtype=np.int64)
    for size in range(1, min(k, p)+1):
        for rows in itertools.combinations(range(k), size):
            for cols in itertools.permutations(range(p), size):
                good = valid[:, rows, cols].all(axis=1)
                counts[good] = size
    return counts


def ceilings(truth, record, logits, bands, ignore):
    truth = np.asarray(truth).reshape(-1, 2)
    all_candidates = record['candidates']
    active = np.flatnonzero(logits.max(-1) >= 0).tolist()
    candidates = [all_candidates[q] for q in active]
    # Identical coordinates across slots represent one union candidate, not two sources.
    union = np.unique(np.asarray([p for c in all_candidates for p in c['positions']]).reshape(-1, 2), axis=0)
    distance = np.linalg.norm(truth[:, None]-union[None], axis=-1)
    result = {'union': {str(d): maximum_valid_pairs(distance <= d) for d in THRESHOLDS}}
    ranks = list(itertools.product(*(range(len(c['scores'])) for c in candidates)))
    points = np.asarray([[c['positions'][r] for c, r in zip(candidates, combo)] for combo in ranks], dtype=float)
    points = points.reshape(len(ranks), len(active), 2)
    feasible = np.ones(len(ranks), dtype=bool)
    for a, b in itertools.combinations(range(len(active)), 2):
        feasible &= np.linalg.norm(points[:, a]-points[:, b], axis=-1) >= 30
    points = points[feasible]
    dist = np.linalg.norm(truth[None, :, None]-points[:, None], axis=-1)
    result['feasible_combinations'] = int(feasible.sum())
    result['feasible'] = {str(d): int(matching_counts(dist <= d).max(initial=0)) for d in THRESHOLDS}
    band_ok = np.zeros((len(truth), len(active)), dtype=bool)
    for t in range(len(truth)):
        mask = ignore[t] < .5
        y = bands[t, mask] > .5
        for j, q in enumerate(active):
            p = logits[q, mask] >= 0
            band_ok[t, j] = 2*np.sum(y & p)/max(int(y.sum()+p.sum()), 1) >= .8
    result['joint100'] = int(matching_counts((dist <= 100) & band_ok[None]).max(initial=0))
    return result


def missing_ranks(truth, record, probability, offset):
    """Only missing Top-8 targets: first accurate peak rank in the SAME heatmap."""
    truth = np.asarray(truth).reshape(-1, 2)
    union = np.asarray([p for c in record['candidates'] for p in c['positions']]).reshape(-1, 2)
    dist = np.linalg.norm(truth[:, None]-union[None], axis=-1)
    missing = [(t, d) for t in range(len(truth)) for d in THRESHOLDS if not (dist[t] <= d).any()]
    if not missing:
        return []
    peaks = []
    for q in range(3):
        heat = probability[q]
        mask = (heat == maximum_filter(heat, size=7, mode='constant')) & (heat > 0)
        ids = np.flatnonzero(mask)
        ids = ids[np.argsort(-heat.ravel()[ids], kind='stable')]
        peaks.append(positions_from_indices(ids, offset[q], 401))
    result = []
    for t, d in missing:
        choices = []
        for q, points in enumerate(peaks):
            distances = np.linalg.norm(points-truth[t], axis=-1)
            found = np.flatnonzero(distances <= d)
            if len(found):
                j = int(found[0])
                choices.append((j+1, q, float(distances[j])))
        best = min(choices) if choices else None
        result.append(dict(source=t, threshold=d, rank=best[0] if best else None,
                           slot=best[1] if best else None, error=best[2] if best else None))
    return result


def summarize_candidates(rows):
    total = sum(r['true_count'] for r in rows)
    result = {'samples': len(rows), 'true_sources': total}
    for source in ('base', 'final'):
        entries = [r['ceilings'][source] for r in rows]
        ranks = [x for r in rows for x in r['missing_ranks'][source]]
        result[source] = {
            level: {str(d): sum(c[level][str(d)] for c in entries)/max(total, 1) for d in THRESHOLDS}
            for level in ('union', 'feasible')}
        result[source].update(joint100=sum(c['joint100'] for c in entries)/max(total, 1),
            no_feasible_scenes=sum(c['feasible_combinations'] == 0 for c in entries),
            missing={str(d): dict(top8_absent=sum(x['threshold'] == d for x in ranks),
                lower_rank_exists=sum(x['threshold'] == d and x['rank'] is not None for x in ranks),
                no_local_peak=sum(x['threshold'] == d and x['rank'] is None for x in ranks)) for d in THRESHOLDS})
    return result
