"""Bounded execution-only optimizations; optimizer batch and FP64 stay fixed."""
from concurrent.futures import ThreadPoolExecutor

import torch

from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.gates.g6.coherent_dpd import normalize_maps, spatial_evd


def physical_maps(physics, probabilities, stats):
    if isinstance(stats, list):
        stats = {k: torch.stack([r[k] for r in stats]) for k in stats[0]}
    current = {k: v.to(probabilities.device, non_blocking=v.is_pinned()) for k,v in stats.items()}
    if getattr(physics, 'p1_batch_mode', 'serial') == 'serial':
        return torch.stack([normalize_maps(physics.evaluate(
            {k:v[i] for k,v in current.items()}, physics.weights(p.double())))[0]
            for i,p in enumerate(probabilities)])
    a = physics.weights(probabilities.double())
    energy = torch.bmm(a, current['energy'])
    denom = (energy/physics.geo.N0**2).clamp_min(1e-20).sqrt()
    coherent = torch.einsum('bqr,brgp->bqgp', a.to(torch.complex128), current['coherent'])
    matrix = torch.diag_embed((energy/denom.square()).to(torch.complex128))[:, :, None].expand(
        -1, -1, physics.geo.num_grid, -1, -1).clone()
    for p,(m,n) in enumerate(physics.geo.pairs):
        value = coherent[...,p]/(denom[...,m]*denom[...,n])[...,None]
        matrix[...,m,n] = value
        matrix[...,n,m] = value.conj()
    matrix = matrix + 1e-6*torch.eye(4,dtype=torch.complex128,device=a.device)
    b,q = a.shape[:2]
    raw = spatial_evd(matrix.reshape(b*q,physics.geo.num_grid,4,4),
                      chunk_size=getattr(physics,'p1_chunk',64))
    return normalize_maps(raw.reshape(b,q,physics.geo.num_y,physics.geo.num_x))[0]


def physical_batches(runtime, features, indices, split, enabled, prefetch=True):
    """One producer owns feature cache and verified physical reads; one batch ahead."""
    indices = list(indices)
    def read(ids):
        iterator = batches(features,[ids],prefetch=False)
        try:
            _, data = next(iterator)
        finally:
            iterator.close()
        stats = None
        if enabled:
            records = runtime.stats(split,ids)
            stats = {k:torch.stack([r[k] for r in records]).pin_memory() for k in records[0]}
        return data,stats
    if not prefetch:
        for ids in indices:
            data,stats = read(ids)
            yield ids,data,stats
        return
    if not indices:
        return
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(read,indices[0])
    try:
        for i,ids in enumerate(indices):
            data,stats = future.result()
            future = pool.submit(read,indices[i+1]) if i+1 < len(indices) else None
            yield ids,data,stats
    finally:
        try:
            if future is not None:
                future.result()
        finally:
            pool.shutdown(wait=True,cancel_futures=True)
