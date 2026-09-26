"""Small-ROI inputs: crop raw maps before normalization; no feature-volume cache."""
from __future__ import annotations

import io
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from 统一模型代码.common.g5_verified_io import VerifiedFile, verified_read
from 统一模型代码.gates.g5.e2e_g5_prepare import configure, g4
from 统一模型代码.gates.g6.g6_p0 import identity

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs_e2e/unified/e2e_g7_compact'
SOURCE = ROOT / 'outputs_e2e/unified/e2e_g5/20260919_approved'
P1 = ROOT / 'outputs_e2e/unified/e2e_g6_p1/20260922_approved'
OLD_G7 = ROOT / 'outputs_e2e/unified/e2e_g7/20260924_approved'
SPLITS = ('train', 'val_select', 'val_compare')
EDGE, COARSE_N, FINE_N, STEP = 1000., 41, 201, 10.


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE.resolve()):
        raise ValueError('Compact output isolation violation')
    g4.write_json(path, value)


def load(row, out):
    return torch.load(io.BytesIO(verified_read(row, out / 'anomalies')),
                      map_location='cpu', weights_only=False)


def save(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(BASE.resolve()) or path.exists():
        raise ValueError('Output must be new and inside compact root')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        torch.save(value, f)
    row = identity(path)
    verified_read(row, path.parent / 'anomalies')
    return row


def crop_maps(coarse, fine):
    """Input axes are channel/y/x and y/x; no image resize/interpolation."""
    if coarse.shape != (19, 81, 81) or fine.shape != (401, 401):
        raise ValueError('Unexpected raw map shape')
    c = np.array(coarse[:, 20:61, 20:61], dtype=np.float32, copy=True)
    f = np.array(fine[100:301, 100:301], dtype=np.float32, copy=True)
    if not np.isfinite(c).all() or not np.isfinite(f).all() or min(c.min(), f.min()) < 0:
        raise ValueError('Invalid physical map values')
    return c, f


def network_inputs(coarse, fine):
    # Keep each original input convention: CH3 numpy population std; D8 torch sample std.
    c = np.log1p(coarse)
    c = (c - c.mean()) / (c.std() + 1e-6)
    f = torch.from_numpy(np.array(fine, copy=True)).log1p()
    f = (f - f.mean()) / (f.std() + 1e-6)
    return torch.from_numpy(c.copy()), f[None]


def crop_statistics(stats):
    energy, coherent = stats['energy'], stats['coherent']
    if coherent.shape[1:] != (81 * 81, 6):
        raise ValueError('Unexpected physical grid')
    return dict(energy=energy.clone(), coherent=coherent.reshape(-1, 81, 81, 6)
                [:, 20:61, 20:61].contiguous().reshape(-1, 41 * 41, 6))


class Inputs:
    def __init__(self, out):
        self.out = Path(out).resolve()
        if not self.out.is_relative_to(BASE.resolve()):
            raise ValueError('Wrong output root')
        self.out.mkdir(parents=True, exist_ok=True)
        # The approved G6 contract binds the G5 manifest and feature manifest.
        contract = read(P1 / 'contract.json')
        registered = {str(Path(r['path']).resolve()): r for r in contract['files']}
        self.manifest = json.loads(verified_read(registered[str(SOURCE / 'manifest.json')], self.out/'anomalies'))
        self.fm = json.loads(verified_read(registered[str(SOURCE / 'feature_manifest.json')], self.out/'anomalies'))
        self.cache_contract = read(P1 / 'run/cache_contract.json')
        self.stats_index = {s: json.loads(verified_read(self.cache_contract[s], self.out/'anomalies')) for s in SPLITS}
        for s, count in zip(SPLITS, (4096, 512, 1024)):
            if len(self.manifest['subsets'][s]) != count:
                raise ValueError('Split size changed')
        a = {r['raw_index'] for r in self.manifest['subsets']['val_select']}
        b = {r['raw_index'] for r in self.manifest['subsets']['val_compare']}
        if a & b:
            raise ValueError('Validation roles overlap')
        for row in self.manifest['inputs']['files']:
            path = Path(row['path']).resolve(strict=True)
            if path.parent != (SOURCE/'input_snapshot').resolve() or path.stat().st_size != row['size_bytes']:
                raise ValueError('Snapshot identity/path invalid')
        configure(self.out, self.manifest['inputs'])
        self.sources = [registered[str(SOURCE/'manifest.json')], registered[str(SOURCE/'feature_manifest.json')],
                        identity(P1/'contract.json'), identity(P1/'run/cache_contract.json')]

    def audit_raw(self):
        import hashlib
        for row in self.manifest['inputs']['files']:
            digest = hashlib.sha256()
            for n, block in enumerate(row['blocks']):
                digest.update(verified_read({'path': row['path'], **block}, self.out/'anomalies',
                    offset=n*row['block_size'], length=block['size_bytes']))
            if digest.hexdigest() != row['sha256']:
                raise RuntimeError('Snapshot whole-file SHA mismatch')

    def edges(self):
        row = self.manifest['inputs']['files'][0]
        with VerifiedFile(row, self.out/'anomalies') as stream, h5py.File(stream, 'r') as h:
            return tuple(torch.from_numpy(h[k][()].reshape(-1).copy()).double()
                         for k in ('sub_f_lo_val', 'sub_f_hi_val'))

    def sample(self, store, split, index):
        entry = self.manifest['subsets'][split][index]
        i = int(entry['local_index'])
        coarse = np.asarray(store.coarse['mtr_sub_all'][:, :, :, i], dtype=np.float32).transpose(2, 1, 0)
        fine_row = self.fm['files'][split]['dpd'][index]
        if any(fine_row[k] != entry[k] for k in ('raw_index', 'local_index')):
            raise ValueError('DPD/sample identity differs')
        fine = np.load(io.BytesIO(verified_read(fine_row, self.out/'anomalies')), allow_pickle=False)
        coarse, fine = crop_maps(coarse, fine)
        k = int(store.coarse['src_count_all'][0, i])
        if k != entry['true_k']:
            raise ValueError('Count differs')
        pos = np.asarray(store.coarse['src_pos_all'][:, :, i], dtype=np.float32).T.copy()
        if np.any(np.abs(pos[:k]) > EDGE):
            raise ValueError('True source lies outside small ROI; never discard that sample')
        raw, ri = store._raw(entry['raw_index'])
        raw_pos = np.asarray(raw['src_pos_all'][:, :, ri], dtype=np.float32).T
        if int(raw['src_count_all'][0, ri]) != k or not np.array_equal(raw_pos[:k], pos[:k]):
            raise ValueError('IQ/coarse labels differ')
        statistics_row = self.stats_index[split][str(index)]
        stats = crop_statistics(load(statistics_row, self.out))
        result = dict(coarse=torch.from_numpy(coarse), fine=torch.from_numpy(fine),
            band=torch.from_numpy(np.asarray(store.coarse['band_mask_all'][:, :, i], dtype=np.float32).T.copy()),
            ignore=torch.from_numpy(np.asarray(store.coarse['ignore_mask_all'][:, :, i], dtype=np.float32).T.copy()),
            positions=torch.from_numpy(pos), count=k, statistics=stats, index=index, split=split,
            metadata={**entry, 'snr_db': float(store.coarse['avg_snr_all'][0, i])},
            sources=dict(fine=fine_row, statistics=statistics_row))
        return result

    def prepare(self, root, indices, guard=lambda: None, progress=lambda *args: None):
        root = Path(root)
        registry = root/'index.json'
        rows = read(registry) if registry.exists() else {s: {} for s in SPLITS}
        for split, ids in indices.items():
            if split not in SPLITS:
                raise ValueError('Unknown/test split')
            with g4.g1.SampleStore(split) as store:
                for n, i in enumerate(ids):
                    guard()
                    if str(i) in rows[split]:
                        verified_read(rows[split][str(i)], self.out/'anomalies')
                    else:
                        value = self.sample(store, split, i)
                        rows[split][str(i)] = save(root/split/f'{i:05d}_{time.time_ns()}.pt', value)
                        write(registry, rows)
                    progress(split, n+1, len(ids))
        return rows
