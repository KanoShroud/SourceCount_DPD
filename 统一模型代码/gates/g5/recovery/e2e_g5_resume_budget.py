"""Apply the user-approved 36h -> 48h wall-only amendment, retaining prior evidence."""
import argparse
import copy
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_contract import freeze, verify_code  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_execute import g4, run_stage  # noqa: E402


def main(run):
    run = run.resolve(strict=True)
    verify_code(run)
    manifest = g4.read_json(run / 'manifest.json')
    old = copy.deepcopy(manifest)
    pilot = g4.read_json(run / 'pilot_report.json')
    assert manifest['config']['wall_limit_seconds'] == 129600
    assert pilot['status'] == 'STOP_RESOURCE'
    assert all(t['checkpoint_resume_exact'] for t in pilot['tracks'].values())
    assert len({t['initial_digest'] for t in pilot['tracks'].values()}) == 1
    assert not (run / 'training').exists()
    archive = run / 'budget36_evidence'
    archive.mkdir(exist_ok=False)
    for name in ('manifest.json', 'pilot_report.json', 'training_code_contract.json', 'execution_status.json'):
        shutil.copy2(run / name, archive / name)
        assert (run / name).read_bytes() == (archive / name).read_bytes()
    manifest['config']['wall_limit_seconds'] = 172800
    unchanged = copy.deepcopy(manifest)
    unchanged['config']['wall_limit_seconds'] = 129600
    assert unchanged == old  # Includes the original start time: no budget clock reset.
    elapsed = time.time() - manifest['created_at']
    remaining_projection = pilot['projected_with_reserve_seconds'] - pilot['elapsed_preparation_seconds']
    revised_projection = elapsed + remaining_projection
    disk = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    assert revised_projection <= 172800 and disk < manifest['config']['new_disk_limit_gib'] * 2**30
    pilot.update(status='PASS', budget_amendment='User approved 48h; original pilot retained in budget36_evidence',
                 original_projected_with_reserve_seconds=pilot['projected_with_reserve_seconds'],
                 projected_with_reserve_seconds=revised_projection,
                 budget_reassessment_elapsed_seconds=elapsed, wall_limit_seconds=172800)
    g4.write_json(run / 'budget_amendment.json', {
        'authorized_change': {'wall_limit_seconds': [129600, 172800]},
        'original_created_at_preserved': manifest['created_at'],
        'reused_pilot_measurements': True, 'scientific_configuration_unchanged': True,
        'archive': str(archive), 'projected_with_reserve_seconds': revised_projection,
        'test_executed': False})
    g4.write_json(run / 'manifest.json', manifest)
    g4.write_json(run / 'pilot_report.json', pilot)
    # Move only this explicitly identified contract; original bytes are also copied above.
    (run / 'training_code_contract.json').rename(archive / 'training_code_contract_original.json')
    freeze(run, 'training_source_snapshot_budget48')
    stages = [
        ('identity_budget48', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'verify-inputs', '--phase', 'budget48']),
        ('training', ['统一模型代码/gates/g5/e2e_g5_train.py']),
        ('identity_after_training', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'verify-inputs', '--phase', 'after_training']),
        ('comparison', ['统一模型代码/gates/g5/e2e_g5_evaluate.py', '--stage', 'compare']),
        ('hard_reference', ['统一模型代码/gates/g5/e2e_g5_evaluate.py', '--stage', 'hard']),
        ('final_audit', ['统一模型代码/gates/g5/audits/verify_e2e_g5_final.py']),
    ]
    results = []
    try:
        for name, arguments in stages:
            results.append(run_stage(run, manifest, name, [*arguments, '--run', str(run)]))
        g4.write_json(run / 'execution_status.json', {'status': 'EXECUTION_COMPLETE_DOC_SYNC_PENDING', 'stages': results, 'test_executed': False})
    except Exception as exc:
        g4.write_json(run / 'execution_status.json', {'status': 'STOP', 'error': repr(exc), 'completed_stages': results, 'test_executed': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    main(parser.parse_args().run)
