"""G5-R1固定候选解码；不接受真值，评价函数与解码严格分离。"""
from __future__ import annotations

import itertools
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment


def positions_from_indices(indices, offset, width):
    positions = []
    for flat in indices:
        iy, ix = divmod(int(flat), width)
        dx, dy = offset[:, iy, ix].clip(-1, 1)
        positions.append([(ix + float(dx))*10-2000, (iy + float(dy))*10-2000])
    return np.asarray(positions, dtype=np.float64).reshape(-1, 2)


def choose_joint(candidates, original, radius=30.0):
    """候选已按概率降序、网格索引升序排列；相同总分选首个组合。"""
    best = None
    for ranks in itertools.product(*(range(len(c['scores'])) for c in candidates)):
        points = np.asarray([c['positions'][r] for c, r in zip(candidates, ranks)]).reshape(-1, 2)
        if any(np.linalg.norm(a-b) < radius for a, b in itertools.combinations(points, 2)):
            continue
        score = sum(np.log(max(c['scores'][r], 1e-20)) for c, r in zip(candidates, ranks))
        if best is None or score > best[0]:
            best = (score, points, ranks)
    if best is None:
        return original.copy(), None, True
    return best[1], [int(r)+1 for r in best[2]], False


def decode(logits, heat, offset):
    """heat为FP32 sigmoid概率；物理坐标与原G5一致。"""
    if logits.shape != (3, 19) or heat.shape != (3, 401, 401) or offset.shape != (3, 2, 401, 401):
        raise ValueError('Unexpected logits/heatmap/offset shape')
    if not all(np.isfinite(x).all() for x in (logits, heat, offset)):
        raise ValueError('Nonfinite model output')
    active = np.flatnonzero(logits.max(-1) >= 0).tolist()
    original, candidates = [], []
    for q in active:
        original.append(positions_from_indices([heat[q].argmax()], offset[q], 401)[0])
        mask = (heat[q] == maximum_filter(heat[q], size=7, mode='constant')) & (heat[q] > 0)
        ids = np.flatnonzero(mask)
        # Stable descending sort preserves ascending flat-index ties; no artificial zero peaks.
        ids = ids[np.argsort(-heat[q].ravel()[ids], kind='stable')[:8]]
        candidates.append({'query': q, 'flat_indices': ids.tolist(),
                           'scores': heat[q].ravel()[ids].astype(float).tolist(),
                           'positions': positions_from_indices(ids, offset[q], 401).tolist()})
    original = np.asarray(original, dtype=np.float32).reshape(-1, 2)
    if len(active) < 2:
        joint, ranks, fallback = original.copy(), [1]*len(active), False
    else:
        joint, ranks, fallback = choose_joint(candidates, original)
    return {'active': active, 'original': original.tolist(), 'joint': np.asarray(joint, dtype=np.float32).tolist(),
            'candidates': candidates, 'selected_ranks': ranks, 'fallback': fallback}


def maximum_valid_pairs(valid):
    if 0 in valid.shape:
        return 0
    a, b = linear_sum_assignment(-valid.astype(np.int64))
    return int(valid[a, b].sum())


def association(truth, pred, logits, active, bands, ignore):
    """真值只用于评价。空间匹配F1保留连续值，联合命中采用独立一对一匹配。"""
    dist = np.linalg.norm(truth[:, None, :]-pred[None, :, :], axis=-1)
    f1 = np.zeros(dist.shape)
    for t in range(len(truth)):
        valid = ignore[t] < .5
        for j, q in enumerate(active):
            a, b = bands[t, valid] > .5, logits[q, valid] >= 0
            f1[t, j] = 2*np.sum(a & b)/max(int(a.sum()+b.sum()), 1)
    ti, pi = linear_sum_assignment(dist)
    return {'joint_tp': maximum_valid_pairs((dist <= 100) & (f1 >= .8)),
            'spatial_band_f1': f1[ti, pi].tolist(),
            'spatial_pairs': [[int(t), int(p)] for t, p in zip(ti, pi)]}


def candidate_ceiling(truth, candidates, logits, bands, ignore):
    """仅评价：最多3槽位×8候选的可行组合上限，保留槽位和30m约束。"""
    location, joint = 0, 0
    valid_location, valid_joint = [], []
    for c in candidates:
        points = np.asarray(c['positions']).reshape(-1, 2)
        valid = np.linalg.norm(truth[:, None, :]-points[None, :, :], axis=-1) <= 100
        valid_location.append(valid)
        band_ok = []
        for t in range(len(truth)):
            mask = ignore[t] < .5
            a, b = bands[t, mask] > .5, logits[c['query'], mask] >= 0
            band_ok.append(2*np.sum(a & b)/max(int(a.sum()+b.sum()),1) >= .8)
        valid_joint.append(valid & np.asarray(band_ok,dtype=bool)[:,None])
    for ranks in itertools.product(*(range(len(c['scores'])) for c in candidates)):
        points = np.asarray([c['positions'][r] for c, r in zip(candidates, ranks)]).reshape(-1, 2)
        if any(np.linalg.norm(a-b) < 30 for a, b in itertools.combinations(points, 2)):
            continue
        loc = np.stack([v[:,r] for v,r in zip(valid_location,ranks)],axis=1) if candidates else np.empty((len(truth),0),dtype=bool)
        both = np.stack([v[:,r] for v,r in zip(valid_joint,ranks)],axis=1) if candidates else loc
        location = max(location, maximum_valid_pairs(loc))
        joint = max(joint, maximum_valid_pairs(both))
        if joint == min(len(truth),len(candidates)):
            break
    # Union coverage ignores the separation constraint; distinguishes missing candidates from conflicts.
    union = np.asarray([p for c in candidates for p in c['positions']]).reshape(-1, 2)
    covered = int((np.linalg.norm(truth[:, None, :]-union[None, :, :], axis=-1) <= 100).any(1).sum())
    return {'truth_with_candidate100': covered, 'feasible_location_tp100': location, 'feasible_joint_tp100': joint}


def duplicate(pred):
    return any(np.linalg.norm(a-b) < 30 for a, b in itertools.combinations(pred, 2))
