"""G7 isolated runtime, frozen proposals and verified lazy local statistics."""
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from 统一模型代码.common.g5_runtime_v2 import batches as feature_batches
from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.r2.g5_r2_model import candidate_batch
from 统一模型代码.gates.g6 import g6_p2_runtime as p2
from 统一模型代码.gates.g6.g6_p1_speed import physical_batches
from 统一模型代码.gates.g6.g6_p2_model import forward as proposal_forward
from 统一模型代码.gates.g6.g6_p2_train import restore as restore_p2
from 统一模型代码.gates.g6.g6_p2_runtime import (
    ROOT, SEEDS, Progress, read, write, identity, safe_print as safe_print,
    setup_environment as setup_environment, g4)
from 统一模型代码.gates.g7 import g7_physics as local
from 统一模型代码.gates.g7.g7_model import LocalFrequencySelector, LocalRefiner, local_loss, decode_local

BASE = ROOT/'outputs_e2e/unified/e2e_g7/20260924_approved'
ARMS = ('f', 's', 'e')
CONFIG = {**p2.CONFIG, 'gate':'E2E-G7', 'arms':list(ARMS), 'window_size':41,
          'step_m':10., 'gaussian_sigma':2., 'local_lr':1e-4,
          'loss_weights':dict(exist=1., band=1., heatmap=1., offset=1., candidate=1.),
          'proposal':'frozen_P2_B_final_per_slot_top8_grid_centers',
          'pilot_samples_per_split':32, 'initialization':'P2 B best per seed',
          'selector_parameters':2451, 'statistics':'deduplicated_exact_local_complex128',
          'candidate_gradient':False, 'existence_head':False}


def register():
    if (BASE/'run').exists():
        raise RuntimeError('Formal directory exists; contract is immutable')
    BASE.mkdir(parents=True, exist_ok=True)
    audit_path = p2.BASE/'run/evaluation/final_audit_report.json'
    audit = read(audit_path)
    if audit['status'] != 'PASS' or not audit['six_tracks_complete']:
        raise RuntimeError('P2 is not complete')
    verified_read(audit['contract'], BASE/'anomalies')
    comp_row = next(r for r in audit['outputs'] if Path(r['path']).name == 'comparison_report.json')
    comparison = json.loads(verified_read(comp_row, BASE/'anomalies'))
    initial = {str(s):comparison['training']['seeds'][str(s)]['b']['best']['checkpoint'] for s in SEEDS}
    refs = {}
    for seed in SEEDS:
        marker = next(r for r in audit['outputs'] if Path(r['path']).name == f'{seed}_b_complete.json')
        complete = json.loads(verified_read(marker, BASE/'anomalies'))
        if complete['checkpoint'] != initial[str(seed)]:
            raise RuntimeError('Frozen P2-B reference identity mismatch')
        refs[str(seed)] = complete['samples']
    files = [identity(audit_path), audit['contract'], comp_row, *initial.values(), *refs.values()]
    r1 = read(ROOT/'outputs_e2e/unified/e2e_g5_r1/20260920_approved/contract.json')
    hard = next(r for r in r1['files'] if Path(r['path']).name == 'hard_reference_report.json')
    files.append(hard)
    for row in files:
        verified_read(row, BASE/'anomalies')
    files += [identity(p) for p in sorted((ROOT/'统一模型代码/gates/g7').glob('*.py'))]
    files += [identity(p) for p in sorted((ROOT/'运行入口/E2E/G7').glob('*.py'))]
    if (BASE/'contract.json').exists():
        shutil.copy2(BASE/'contract.json', BASE/f'contract_superseded_{time.time_ns()}.json')
    write(BASE/'contract.json', dict(config=CONFIG, files=files, initial=initial,
        frozen_p2_b_samples=refs, hard_reference_report=hard,
        scope='F/S input contrast; E/S only local physical feedback; no test',
        precision='physical FP64/complex128; neural FP32'))


class LocalHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.selector = LocalFrequencySelector()
        self.refiner = LocalRefiner()


class Runtime(p2.Runtime):
    def __init__(self, out, deadline=None):
        self.out = Path(out).resolve()
        if not self.out.is_relative_to(BASE.resolve()) or self.out.is_relative_to(Path('F:/SourceCount_DPD/outputs').resolve()):
            raise ValueError('G7 output isolation violation')
        self.out.mkdir(parents=True, exist_ok=True)
        self.deadline = deadline
        self.inputs, self.ranges, self.physical_consumed = {}, {}, {}
        self.peak_ram = 0
        self.manifest = self.fm = self.physics = None
        self._cache_indexes, self._local_indexes, self._local_used = {}, {}, {}

    def preflight(self):
        self.g7_contract = read(BASE/'contract.json')
        self.cache_root = BASE/'local_cache'/identity(BASE/'contract.json')['sha256'][:16]
        if self.g7_contract['config'] != CONFIG:
            raise RuntimeError('G7 config differs from registered contract')
        for row in self.g7_contract['files']:
            verified_read(row, self.out/'anomalies')
        super().preflight()

    def frozen_model(self, seed):
        context, head, optimizer, params = p2.Runtime.context(self, seed, 'b')
        del optimizer, params
        row = self.g7_contract['initial'][str(seed)]
        cp = torch.load(io.BytesIO(verified_read(row, self.out/'anomalies')), map_location='cpu', weights_only=False)
        restore_p2(context, head, cp['state'])
        g4.set_mode(context, training=False)
        head.eval()
        return context, head

    def context(self, seed, arm):
        if arm not in ARMS:
            raise ValueError(arm)
        context, prior = self.frozen_model(seed)
        torch.manual_seed(seed+2000)
        head = LocalHead().cuda()
        head.selector.selector.load_state_dict(prior.selector.state_dict())
        del prior
        groups = list(context.parameter_groups)+[
            dict(params=list(head.refiner.parameters()), lr=CONFIG['local_lr'], name='local'),
            dict(params=list(head.selector.parameters()), lr=CONFIG['selector_lr'], name='selector')]
        params = list(context.parameters)+list(head.parameters())
        optimizer = torch.optim.AdamW(groups, weight_decay=self.manifest['config']['weight_decay'])
        g4.set_mode(context, training=True)
        g4.set_deterministic(seed)
        return context, head, optimizer, params

    @staticmethod
    def query_forward(context, batch, ids):
        spatial = g4.numpy_batch(batch.spatial, ids, torch.device('cuda:0'))
        pooled = spatial.mean(dim=(-1, -2))
        tokens = context.ch3.cross_attn(pooled+context.ch3.pos_embed)
        global_feature = context.ch3.global_encoder(tokens.mean(dim=1))
        cached = g4.CH3Features(spatial, tokens, global_feature,
            torch.empty(0, device=spatial.device), torch.empty(0, device=spatial.device))
        current = g4.g2.ch3_from_cached(context.ch3, cached)
        return context.query_builder(current)

    def forward(self, context, head, batch, ids, arm, stats):
        query, logits = self.query_forward(context, batch, ids)
        weights = head.selector.weights(query, logits, arm)
        maps = local.local_maps(self.physics, weights, stats, full=arm == 'f')
        centers = torch.stack([r['centers_m'] for r in stats]).to(query.device).float()
        valid = torch.stack([r['valid_mask'] for r in stats]).to(query.device)
        output = head.refiner(maps, query, valid)
        return dict(query=query, band_logits=logits, maps=maps, weights=weights,
                    centers_m=centers, valid_mask=valid, output=output)

    def loss(self, result, targets, ids):
        ks = [int(targets.counts[i]) for i in ids.tolist()]
        fields = [[getattr(targets, key)[i, :k] for i, k in zip(ids.tolist(), ks)]
                  for key in ('positions', 'band', 'ignore')]
        return local_loss(result['output'], result['band_logits'], result['centers_m'],
                          result['valid_mask'], *fields, config=CONFIG)

    def decode(self, result, counts=None):
        return decode_local(result['output'], result['centers_m'], result['valid_mask'],
                            result['band_logits'], step_m=10., separation_m=30., counts=counts)

    def local_rows(self, seed, split):
        if split not in ('train', 'val_select', 'val_compare') or seed not in SEEDS:
            raise ValueError('Unregistered split or seed')
        key = (seed, split)
        if key not in self._local_indexes:
            path = self.cache_root/f'{seed}/{split}/index.json'
            self._local_indexes[key] = read(path) if path.exists() else {}
        return self._local_indexes[key]

    def stats(self, *args):
        # Frozen proposal generation still calls P2's coarse-cache API.
        if len(args) == 2:
            return p2.Runtime.stats(self, *args)
        seed, split, ids = args
        result = []
        for i in ids.tolist():
            row = self.local_rows(seed, split)[str(i)]
            path = Path(row['path']).resolve(strict=True)
            if not path.is_relative_to(self.cache_root/str(seed)/split):
                raise RuntimeError('Local cache path escaped')
            value = torch.load(io.BytesIO(verified_read(row, self.out/'anomalies')), weights_only=True)
            if value['contract_sha256'] != identity(BASE/'contract.json')['sha256']:
                raise RuntimeError('Local cache contract mismatch')
            if (value['seed'], value['split'], value['index']) != (seed, split, i):
                raise RuntimeError('Local cache sample identity mismatch')
            result.append(value)
            self._local_used[str(path)] = row
        return result

    def batches(self, features, order, split, seed):
        order = list(order)
        if not order:
            return
        def load(ids):
            it = feature_batches(features, [ids], prefetch=False)
            try:
                _, batch = next(it)
            finally:
                it.close()
            return batch, self.stats(seed, split, ids)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(load, order[0])
            try:
                for j, ids in enumerate(order):
                    batch, stats = future.result()
                    future = pool.submit(load, order[j+1]) if j+1 < len(order) else None
                    yield ids, batch, stats
            finally:
                if future is not None:
                    future.result()

    def ensure_local_cache(self, seed, split, ids, bundle):
        rows = self.local_rows(seed, split)
        missing = [int(i) for i in ids if str(int(i)) not in rows]
        if not missing:
            return dict(count=0, seconds=0., bytes=0)
        begin = time.perf_counter()
        folder = self.cache_root/f'{seed}/{split}'
        folder.mkdir(parents=True, exist_ok=True)
        context, head = self.frozen_model(seed)
        features, targets, _, index, _ = bundle
        it = physical_batches(self, features, [torch.tensor(missing[i:i+4])
            for i in range(0, len(missing), 4)], split, True)
        progress = Progress(f'G7局部统计 {seed}/{split}', len(missing), self.out/'progress.json')
        contract_hash = identity(BASE/'contract.json')['sha256']
        n = 0
        try:
            with g4.g1.SampleStore(split) as store, torch.no_grad():
                for batch_ids, batch, coarse_stats in it:
                    self.guard(); self.consumed(index, batch_ids)
                    output, _ = proposal_forward(context, head, batch, batch_ids, 'b', self.physics, coarse_stats)
                    records = candidate_batch(output[3], output[4])
                    weights = head.weights(output[0], output[1], 'b').cpu()
                    for j, i in enumerate(batch_ids.tolist()):
                        entry = self.manifest['subsets'][split][i]
                        raw, raw_i = store._raw(entry['raw_index'])
                        k = int(targets.counts[i])
                        if int(raw['src_count_all'][0, raw_i]) != k:
                            raise RuntimeError('Raw/target count mismatch')
                        positions = np.asarray(raw['src_pos_all'][:, :, raw_i]).T[:k]
                        torch.testing.assert_close(torch.as_tensor(positions).float(), targets.positions[i, :k], rtol=0, atol=0)
                        signal = (np.asarray(raw['sig_rcv_real_all'][:, :, raw_i], dtype=np.float64).T
                                  +1j*np.asarray(raw['sig_rcv_imag_all'][:, :, raw_i], dtype=np.float64).T)
                        centers, valid, points, inverse = local.windows(records[j])
                        stats = local.statistics(self.physics, signal, points, guard=self.guard)
                        stats.update(centers_m=centers, valid_mask=valid, points=points, inverse=inverse,
                            seed=seed, split=split, index=i, raw_index=int(entry['raw_index']),
                            proposal_record=records[j], predicted_weights=weights[j],
                            frozen_logits=output[1][j].cpu(), contract_sha256=contract_hash)
                        path = folder/f'{i:05d}_{time.time_ns()}.pt'
                        torch.save(stats, path)
                        row = identity(path)
                        saved = torch.load(io.BytesIO(verified_read(row, self.out/'anomalies')), weights_only=True)
                        if not torch.equal(saved['coherent'], stats['coherent']):
                            raise RuntimeError('Local cache write/read mismatch')
                        rows[str(i)] = row
                        write(folder/'index.json', rows)
                        n += 1; progress.update(n)
                        del stats, saved
        finally:
            it.close()
            del context, head
            gc.collect(); torch.cuda.empty_cache()
        return dict(count=len(missing), seconds=time.perf_counter()-begin,
                    bytes=sum(rows[str(i)]['size_bytes'] for i in missing))

    def ensure_all_cache(self, seed):
        reports = {}
        for split in ('train', 'val_select', 'val_compare'):
            bundle = self.features(split)
            reports[split] = self.ensure_local_cache(seed, split, range(len(bundle[1].counts)), bundle)
            del bundle
        return reports

    def postcheck(self, label):
        for row in self._local_used.values():
            self.guard(); verified_read(row, self.out/'anomalies')
        for row in self.g7_contract['files']:
            verified_read(row, self.out/'anomalies')
        super().postcheck(label)
        self._local_used.clear()


class RunLock(p2.RunLock):
    def __enter__(self):
        import msvcrt
        BASE.mkdir(parents=True, exist_ok=True)
        self.f = (BASE/'run.lock').open('a+b')
        if self.f.tell() == 0:
            self.f.write(b'0'); self.f.flush()
        self.f.seek(0)
        try:
            msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            self.f.close()
            raise RuntimeError('已有G7进程运行') from None
        return self
