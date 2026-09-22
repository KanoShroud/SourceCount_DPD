"""Frozen positions and weights; calibrate one correction threshold per method."""
from __future__ import annotations

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np
import torch
from torch.nn import functional as F

from 统一模型代码.common.g5_verified_io import VerifiedFile, verified_read
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import identity
from 统一模型代码.gates.g5.r1.g5_r1_report import summarize
from 统一模型代码.gates.g5.r2.g5_r2_runtime import ROOT, SOURCE as G5, SEEDS, Runtime, write, safe_print, setup_environment
from 统一模型代码.gates.g5.r4.g5_r4 import ExplicitMatcher, scores_for

R4 = ROOT/'outputs_e2e/unified/e2e_g5_r4/20260922_131120'
BASE = ROOT/'outputs_e2e/unified/e2e_g5_r5'
A, B, C = 'A_original_binding', 'B_r4_explicit_conservative', 'C_coarse_dpd_conservative'
CONFIG = {'gate': 'G5-R5', 'seeds': list(SEEDS), 'split_seed': 20260922,
          'calibration_scenes': 256, 'check_scenes': 256, 'epoch': 20,
          'patch_offsets_m': [-50., 0., 50.], 'sampling': 'bilinear_border_align_corners',
          'degenerate_relative_norm': 1e-6, 'tie_atol': 1e-12,
          'score_accumulation': 'float64_mean', 'threshold_candidates': 'observed_margin_plus_0_inf',
          'wall_seconds': 3600, 'training_executed': False, 'test_executed': False,
          'val_compare_executed': False}


def preprocess(raw):
    """Exactly the coarse-only operations of G1 SampleStore.sample; once."""
    x = np.asarray(raw, dtype=np.float32).transpose(2, 1, 0)
    x = np.log(x + 1.0)
    x = (x - x.mean()) / (x.std() + 1e-6)
    if x.shape != (19, 81, 81) or not np.isfinite(x).all():
        raise ValueError('Invalid CH3 input')
    return torch.from_numpy(x.copy())


def local_signature(coarse, points):
    offsets = torch.tensor([[x, y] for y in CONFIG['patch_offsets_m']
                            for x in CONFIG['patch_offsets_m']], dtype=coarse.dtype)
    grid = (torch.as_tensor(points, dtype=coarse.dtype)[:, None] + offsets).div(2000)
    values = F.grid_sample(coarse[None], grid[None], mode='bilinear',
                           padding_mode='border', align_corners=True)
    return values[0].mean(-1).T.double()


def centered_cosine(bands, response):
    vectors = [bands.double(), response.double()]
    normalized = []
    for vector in vectors:
        centered = vector - vector.mean(0, keepdim=True)
        norms = centered.norm(dim=-1)
        threshold = 1e-6 * vector.norm(dim=-1).clamp_min(1)
        if (norms <= threshold).any():
            return None
        normalized.append(centered / norms[:, None])
    return normalized[0] @ normalized[1].T


def proposal(scores, perms, reason=None):
    k = perms.shape[1]
    if k < 2 or scores is None:
        return {'best': 0, 'margin': 0., 'reason': reason or 'K_LE_1', 'values': []}
    if not torch.isfinite(scores).all():
        raise ValueError('Nonfinite matching scores')
    values = scores.double()[torch.arange(k)[None], perms].mean(-1).cpu().numpy()
    winners = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=CONFIG['tie_atol']))
    if len(winners) != 1:
        return {'best': 0, 'margin': 0., 'reason': 'TIED_BEST', 'values': values.tolist()}
    best = int(winners[0])
    return {'best': best, 'margin': max(0., float(values[best] - values[0])),
            'reason': 'ORIGINAL_BEST' if best == 0 else 'ALTERNATIVE_BEST', 'values': values.tolist()}


def split_scenes(records):
    rng = np.random.default_rng(CONFIG['split_seed'])
    groups = {'calibration': [], 'check': []}
    for k in range(4):
        ids = sorted(r['raw_index'] for r in records if r['true_k'] == k)
        if len(ids) != 128:
            raise ValueError('Expected 128 scenes per K')
        ids = rng.permutation(ids).tolist()
        groups['calibration'] += ids[:64]
        groups['check'] += ids[64:]
    return {key: sorted(ids) for key, ids in groups.items()}


def decision(row, method, tau):
    p = row['scores'][method]
    return p['best'] if p['margin'] > tau else 0


def calibrate(rows, method):
    candidates = sorted({0., float('inf')} | {r['scores'][method]['margin'] for r in rows})
    trials = []
    for tau in candidates:
        scene_deltas = {}
        changed = 0
        for r in rows:
            chosen = decision(r, method, tau)
            delta = r['metrics'][chosen]['joint_tp'] - r['metrics'][0]['joint_tp']
            scene_deltas[r['raw_index']] = scene_deltas.get(r['raw_index'], 0) + delta
            changed += chosen != 0
        trials.append({'tau': None if np.isinf(tau) else tau,
                       'net_joint_hits': sum(scene_deltas.values()), 'changed_scene_seed_pairs': changed})
    best = max(range(len(trials)), key=lambda i: (trials[i]['net_joint_hits'],
               -trials[i]['changed_scene_seed_pairs'], candidates[i]))
    return candidates[best], {'selected': trials[best], 'trials': trials,
                             'tau_null_means': 'infinity / no correction',
                             'distinct_scene_count': len({r['raw_index'] for r in rows})}


def transitions(rows, method, tau):
    changed = beneficial = harmful = neutral = gained = lost = 0
    for r in rows:
        chosen = decision(r, method, tau) if method != A else 0
        delta = r['metrics'][chosen]['joint_tp'] - r['metrics'][0]['joint_tp']
        changed += chosen != 0
        beneficial += delta > 0
        harmful += delta < 0
        neutral += chosen != 0 and delta == 0
        gained += max(delta, 0)
        lost += max(-delta, 0)
    return {'scenes': len(rows), 'changed': changed, 'beneficial': beneficial, 'harmful': harmful,
            'neutral_changes': neutral, 'gained_joint_hits': gained, 'lost_joint_hits': lost,
            'net_joint_hits': gained - lost, 'change_rate': changed / len(rows) if rows else None,
            'beneficial_per_change': beneficial / changed if changed else None,
            'net_hit_per_change': (gained - lost) / changed if changed else None}


def evaluate(rows, method, tau):
    base = [r['metrics'][0] for r in rows]
    selected = [r['metrics'][decision(r, method, tau) if method != A else 0] for r in rows]
    result, baseline = summarize(selected), summarize(base)
    for key in ('gospa_m', 'matched_rmse_m', 'matched_coverage', 'count_accuracy',
                'recall10', 'recall30', 'recall50', 'recall100'):
        if result[key] is None and baseline[key] is None:
            continue
        if not np.isclose(result[key], baseline[key], rtol=0, atol=1e-5):
            raise AssertionError(f'Location invariant failed: {key}')
    result['joint_recall_delta'] = result['joint_recall100_f1_08'] - baseline['joint_recall100_f1_08']
    result['transitions'] = transitions(rows, method, tau)
    count_ok = [r for r in rows if r['metrics'][0]['true_count'] == r['metrics'][0]['predicted_count']]
    result['count_correct_transitions'] = transitions(count_ok, method, tau)
    if method != A:
        result['proposal_reasons'] = dict(Counter(r['scores'][method]['reason'] for r in rows))
    return result


class Run:
    guard = Runtime.guard

    def __init__(self, out):
        self.out = out.resolve(strict=True)
        assert self.out.parent == BASE.resolve() and not any(self.out.iterdir())
        self.deadline = time.time() + CONFIG['wall_seconds']
        self.peak_ram = 0
        self.inputs, self.ranges = {}, []

    def get(self, row, part=False):
        self.guard()
        p = Path(row['path']).resolve(strict=True)
        if not p.is_relative_to(ROOT) or p.is_relative_to(self.out):
            raise ValueError('Input/output roots overlap or invalid reference root')
        if part:
            if not p.is_relative_to(R4):
                raise ValueError('Cache outside approved R4')
            self.ranges.append(row)
            return verified_read(row, self.out/'anomalies', offset=row['offset'], length=row['size_bytes'])
        self.inputs[str(p)] = row
        return verified_read(row, self.out/'anomalies')

    def bootstrap(self, path):
        return json.loads(self.get(identity(path.resolve(strict=True))))

    def coarse_audit(self, row):
        p = Path(row['path']).resolve(strict=True)
        if not p.is_relative_to(G5/'input_snapshot') or 'val_select' not in p.name:
            raise ValueError('Only local validation-select coarse snapshot is allowed')
        if p.stat().st_size != row['size_bytes']:
            raise ValueError('Coarse file size changed')
        digest = hashlib.sha256()
        for i, block in enumerate(row['blocks']):
            self.guard()
            payload = verified_read({'path': str(p), **block}, self.out/'anomalies',
                                    offset=i * row['block_size'], length=block['size_bytes'])
            digest.update(payload)
        if digest.hexdigest() != row['sha256']:
            raise ValueError('Coarse full-file SHA does not match registered blocks')


def code_files():
    files = {Path(__file__).resolve(), Path(__file__).with_name('test_g5_r5.py').resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if isinstance(name, str) and Path(name).is_absolute():
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'outputs_e2e'):
                files.add(path)
    return [identity(p) for p in sorted(files)]


@torch.no_grad()
def run():
    setup_environment()
    torch.set_num_threads(1)
    began = time.time()
    out = BASE/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    runtime = Run(out)
    safe_print(f'G5-R5 开始：{out}')
    audit = runtime.bootstrap(R4/'final_audit.json')
    if audit['status'] != 'PASS':
        raise RuntimeError('R4 audit must pass')
    registered = {str(Path(r['path']).resolve()): r for r in audit['files']}

    def r4_json(path):
        return json.loads(runtime.get(registered[str(path.resolve(strict=True))]))

    contract4 = r4_json(R4/'contract.json')
    for row in contract4['files']:
        runtime.get(row)
    manifest = runtime.bootstrap(G5/'manifest.json')
    coarse_rows = [r for r in manifest['inputs']['files'] if Path(r['path']).name.endswith('val_select.mat')]
    if len(coarse_rows) != 1:
        raise ValueError('Ambiguous coarse source')
    coarse_row = coarse_rows[0]
    records = manifest['subsets']['val_select']
    partitions = split_scenes(records)
    ids = {r['raw_index']: r for r in records}
    if len(ids) != 512:
        raise ValueError('Scene IDs must be unique')
    contract = {'config': CONFIG, 'scene_partitions': partitions, 'scenes': records,
                'coarse_dpd_source': coarse_row, 'preprocessed': False,
                'preprocessing': 'G1 SampleStore.sample coarse transform reproduced once in preprocess()',
                'checkpoint_role': 'fixed R4 explicit epoch20 probe, not selected best',
                'files': code_files(), 'software': {'python': sys.version, 'torch': torch.__version__,
                                                 'numpy': np.__version__, 'h5py': h5py.__version__},
                'evidence_scope': 'Previously inspected development data; no independent test claim'}
    # Split and protocol committed before computing any B/C score.
    write(out/'contract.json', contract)
    write(out/'progress.json', {'stage': 'PREFLIGHT'})
    runtime.coarse_audit(coarse_row)
    samples = {}
    for seed in SEEDS:
        index = r4_json(R4/f'cache/{seed}/val_select/index.json')
        if len(index) != 512:
            raise ValueError('Unexpected R4 cache length')
        samples[seed] = [torch.load(io.BytesIO(runtime.get(row, True)), map_location='cpu', weights_only=False)
                         for row in index]
        if [s['index'] for s in samples[seed]] != [r['raw_index'] for r in records]:
            raise ValueError('R4/G5 scene order mismatch')
        for s in samples[seed]:
            base = s['metrics'][0]
            record = ids[s['index']]
            if base['true_count'] != record['true_k'] or base['local_index'] != record['local_index']:
                raise ValueError('R4 metadata differs from G5')
            for m in s['metrics']:
                for key in ('gospa_m', 'true_count', 'predicted_count', 'tp_at_100m'):
                    if m[key] != base[key]:
                        raise AssertionError('Cached permutation changed location/count')
                np.testing.assert_allclose(sorted(m['matched_errors_m']), sorted(base['matched_errors_m']), atol=1e-5, rtol=0)
    safe_print('阶段完成：输入身份、固定位置及256/256场景划分已登记；开始B/C评分。')
    heads = {}
    for seed in SEEDS:
        row = r4_json(R4/f'training/{seed}/explicit/epoch20.identity.json')
        cp = torch.load(io.BytesIO(runtime.get(row)), map_location='cpu', weights_only=False)
        heads[seed] = ExplicitMatcher().eval()
        heads[seed].load_state_dict(cp['state'])
        for p in heads[seed].parameters():
            p.requires_grad_(False)
    rows = []
    score_start = time.time()
    pilot = None
    with VerifiedFile(coarse_row, out/'anomalies') as stream:
        with h5py.File(stream, 'r') as handle:
            for i, record in enumerate(records):
                runtime.guard()
                local = record['local_index']
                if int(handle['src_count_all'][0, local]) != record['true_k']:
                    raise ValueError('Coarse label/scene mismatch')
                coarse = preprocess(handle['mtr_sub_all'][:, :, :, local])
                for seed in SEEDS:
                    s = samples[seed][i]
                    if len(s['active']) < 2:
                        sb = sc = proposal(None, s['perms'])
                    else:
                        sb = proposal(scores_for(heads[seed], s, 'cpu'), s['perms'])
                        response = local_signature(coarse, s['points'])
                        bands = s['logits'][s['active']].sigmoid()
                        sc = proposal(centered_cosine(bands, response), s['perms'], 'DEGENERATE_SIGNATURE')
                    rows.append({'seed': seed, 'raw_index': s['index'],
                                 'partition': 'calibration' if s['index'] in partitions['calibration'] else 'check',
                                 'metrics': s['metrics'], 'permutations': s['perms'].tolist(),
                                 'positions_m': s['points'].tolist(), 'band_logits': s['logits'].tolist(),
                                 'scores': {B: sb, C: sc}})
                if i == 31:
                    elapsed = time.time() - score_start
                    estimate = elapsed / 32 * 512 * 1.5 + 120
                    pilot = {'scenes': 32, 'seconds': elapsed, 'estimated_remaining_bound_seconds': estimate,
                             'includes_50_percent_margin_and_120s_postcheck': True}
                    write(out/'pilot.json', pilot)
                    if time.time() + estimate > runtime.deadline:
                        raise RuntimeError('Short probe predicts approved budget exceeded')
                if (i + 1) % 128 == 0:
                    write(out/'progress.json', {'stage': 'SCORING', 'scenes': i + 1, 'total': 512})
    safe_print('阶段完成：两seed的B/C评分已完成；仅使用校准半集选阈值。')
    calibration = {}
    thresholds = {}
    for method in (B, C):
        thresholds[method], calibration[method] = calibrate([r for r in rows if r['partition'] == 'calibration'], method)
    write(out/'calibration.json', calibration)
    # Persist thresholds before consulting check labels for evaluation.
    report = {'status': 'COMPLETED', 'config': CONFIG, 'thresholds': {m: v['selected']['tau'] for m, v in calibration.items()},
              'tau_null_means': 'infinity / no correction', 'results': {}, 'pilot': pilot}
    for partition in ('calibration', 'check'):
        report['results'][partition] = {}
        for seed in SEEDS:
            subset = [r for r in rows if r['partition'] == partition and r['seed'] == seed]
            report['results'][partition][str(seed)] = {A: evaluate(subset, A, float('inf'))}
            for method in (B, C):
                report['results'][partition][str(seed)][method] = {
                    'tau0': evaluate(subset, method, 0.), 'calibrated': evaluate(subset, method, thresholds[method])}
    for r in rows:
        r['selected'] = {m: {'tau0': decision(r, m, 0.), 'calibrated': decision(r, m, thresholds[m])} for m in (B, C)}
    with (out/'samples.jsonl').open('x', encoding='utf-8') as handle:
        for r in rows:
            handle.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n')
    report['scientific_status'] = {}
    for method in (B, C):
        gains = [report['results']['check'][str(s)][method]['calibrated']['transitions']['net_joint_hits'] for s in SEEDS]
        report['scientific_status'][method] = {'seed_net_hits': gains,
            'decision': 'DIRECTIONAL_POSITIVE' if min(gains) > 0 else 'NO_CHECK_GAIN' if max(gains) <= 0 else 'INCONCLUSIVE'}
    safe_print('阶段完成：阈值已冻结、检查集评价完成；正在复核输入与输出。')
    for row in runtime.ranges:
        verified_read(row, out/'anomalies', offset=row['offset'], length=row['size_bytes'])
    runtime.coarse_audit(coarse_row)
    for row in list(runtime.inputs.values()) + contract['files']:
        runtime.get(row)
    report.update(wall_seconds=time.time() - began, peak_ram_percent=runtime.peak_ram,
                  fixed_location_invariants=True, inputs_pre_post_verified=True)
    write(out/'comparison_report.json', report)
    lines = ['# G5-R5运行结论', '', '已完成；无网络训练，未读取val_compare或test。', '',
             '| 方法 | seed | 校准阈值 | 检查集净命中 | 联合Recall变化/pp | 修改/有益/有害 |', '|---|---:|---:|---:|---:|---|']
    for method in (B, C):
        for seed in SEEDS:
            v = report['results']['check'][str(seed)][method]['calibrated']
            t = v['transitions']
            lines.append(f"| {method} | {seed} | {report['thresholds'][method]} | {t['net_joint_hits']} | {v['joint_recall_delta']*100:.4f} | {t['changed']}/{t['beneficial']}/{t['harmful']} |")
    lines += ['', '阈值null表示∞，即保留全部原绑定。结果仅为开发集内部诊断。',
              f"总墙钟{report['wall_seconds']:.2f}秒；RAM采样峰值{runtime.peak_ram:.1f}%。",
              '两seed同向改善只提供候选方向，需结合实际命中数判断；不自动开始下一实验。']
    (out/'运行结论.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write(out/'input_audit.json', {'status': 'PASS', 'files': list(runtime.inputs.values()),
          'cache_ranges_verified': len(runtime.ranges), 'coarse_full_sha_pre_post': coarse_row['sha256'],
          'same_bytes_verified_and_consumed': True, 'band_logits_and_positions_not_modified': True})
    write(out/'progress.json', {'stage': 'COMPLETED', 'scenes': 512, 'seeds': list(SEEDS)})
    outputs = [identity(p) for p in sorted(out.rglob('*')) if p.is_file() and 'anomalies' not in p.parts]
    for row in outputs:
        verified_read(row, out/'anomalies')
    write(out/'final_audit.json', {'status': 'PASS', 'files': outputs, 'scientific_status': report['scientific_status']})
    safe_print(f'G5-R5 完成：{out / "comparison_report.json"}')
    return out


if __name__ == '__main__':
    run()
