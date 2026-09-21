"""G5六轨开发实验完成性审计；科学结论可为不确定或退步。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_contract import verify_code  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_model import g4, summarize  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_train import load_checkpoint, must_extend  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402


def main(run):
    run = run.resolve(strict=True)
    verify_code(run)
    manifest = g4.read_json(run / 'manifest.json')
    cfg = manifest['config']
    assert not manifest['test_executed'] and cfg['counts'] == {'train': 4096, 'val_select': 512, 'val_compare': 1024}
    for name in ('feature_manifest', 'provenance_audit', 'model_contract_report', 'pilot_report',
                 'training_report', 'hard_reference_report', 'identity_before_pilot', 'identity_after_training'):
        assert g4.read_json(run / f'{name}.json')['status'] == 'PASS', name
    training = g4.read_json(run / 'training_report.json')
    assert set(training['seeds']) == {str(seed) for seed in cfg['training_seeds']}
    expected = [(r['raw_index'], r['true_k']) for r in manifest['subsets']['val_compare']]
    checked = []
    for seed in cfg['training_seeds']:
        pair = training['seeds'][str(seed)]
        assert set(pair) == {'sg', 'e2e'}
        assert pair['sg']['initial_digest'] == pair['e2e']['initial_digest']
        assert pair['sg']['completed_epoch'] == pair['e2e']['completed_epoch']
        triggered = False
        for track in ('sg', 'e2e'):
            root = run / 'training' / str(seed) / track
            row = pair[track]
            completed = row['completed_epoch']
            assert row['status'] == 'PASS' and completed in (48, 64)
            history = g4.read_json(root / 'history.json')
            assert [r['epoch'] for r in history] == list(range(completed+1))
            assert [r['epoch'] for r in history if r.get('validation') is not None] == list(range(0, completed+1, 2))
            best = min((r for r in history if r.get('validation') is not None),
                       key=lambda r: (r['validation']['overall']['gospa_mean_m'], r['epoch']))
            assert row['best_epoch'] == best['epoch']
            assert row['best_gospa'] == best['validation']['overall']['gospa_mean_m']
            assert row['optimizer_steps'] == completed * 1024
            triggered |= must_extend(history, 48, cfg['extension_net_gain_m'])
            assert row['budget_candidate'] == must_extend(history, completed, cfg['extension_net_gain_m'])
            verified_read(row['best'], run / 'anomalies')
            selected = load_checkpoint(root / 'best.pt')
            assert selected['epoch'] == best['epoch'] and selected['metrics'] == best['validation']
            del selected
            payload = g4.read_json(run / 'comparison' / f'{seed}_{track}.json')
            assert payload['status'] == 'PASS' and not payload['test_executed']
            assert payload['checkpoint'] == row['best']
            evaluation = payload['evaluation']
            assert [(r['raw_index'], r['true_count']) for r in evaluation['samples']] == expected
            assert summarize(evaluation['samples']) == evaluation['overall']
            for sample in evaluation['samples']:
                truth = np.asarray(sample['true_positions_m'], dtype=np.float32).reshape(-1, 2)
                predicted = np.asarray(sample['predicted_positions_m'], dtype=np.float32).reshape(-1, 2)
                assert np.isfinite(predicted).all()
                assert len(truth) == sample['true_count'] and len(predicted) == sample['predicted_count']
                assert np.isclose(g4.g1.gospa_sample(truth, predicted)['value_m'], sample['gospa_m'])
                assert g4.distance_errors(truth, predicted) == sample['matched_errors_m']
                for threshold in (10, 30, 50, 100):
                    assert g4.g1.maximum_matches_within(truth, predicted, threshold) == sample[f'tp_at_{threshold}m']
            checked.append({'seed': seed, 'track': track, 'epochs': completed, 'best_epoch': best['epoch'],
                            'samples': len(evaluation['samples']), 'best': row['best']})
        assert pair['sg']['completed_epoch'] == (64 if triggered else 48)
    comparison = g4.read_json(run / 'comparison_report.json')
    assert comparison['status'] in ('FEEDBACK_SUPPORTED_ON_DEV', 'DIRECTIONAL_BUT_INCONCLUSIVE', 'TRADEOFF_OR_REGRESSION')
    assert not comparison['test_executed']
    event_files = list(run.rglob('events.jsonl'))
    events = [json.loads(line) for path in event_files for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    # 第二次失败不允许以最终PASS掩盖。首次失败必须找到同一路径/偏移的恢复记录。
    failures = [r for r in events if r.get('status') == 'READ_FAILED']
    for failure in failures:
        assert failure['attempt'] == 1
        assert any(r.get('status') == 'RECOVERED_ONCE' and r['path'] == failure['path']
                   and r.get('offset') == failure.get('offset') and r['time_ns'] > failure['time_ns'] for r in events)
    disk_bytes = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    elapsed = time.time()-manifest['created_at']
    assert disk_bytes <= cfg['new_disk_limit_gib']*2**30 and elapsed <= cfg['wall_limit_seconds']
    result = {'status': 'PASS', 'scientific_status': comparison['status'], 'tracks': checked,
              'recovered_read_failures': len(failures), 'disk_bytes': disk_bytes, 'total_wall_seconds': elapsed,
              'test_executed': False, 'scope': 'Full six-track development experiment, not frozen-test or publication completion'}
    g4.write_json(run / 'final_audit_report.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
