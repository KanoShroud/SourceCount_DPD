"""Explicit one-time recovery from the virtual-module archive failure; retain old evidence."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_execute import g4, run_stage  # noqa: E402


def main(run):
    run = run.resolve(strict=True)
    previous = g4.read_json(run / 'execution_status.json')
    assert previous['status'] == 'STOP' and previous['completed_stages'] == []
    assert g4.read_json(run / 'execution_logs/freeze.json')['returncode'] != 0
    assert g4.read_json(run / 'feature_manifest.json')['status'] == 'PASS'
    assert not (run / 'training_code_contract.json').exists()
    receipt = run / 'freeze_recovery_receipt.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    g4.write_json(receipt, {'previous_status': previous,
        'cause': 'torch.ops/classes synthetic relative __file__ mistaken for repository source',
        'fix': 'Exclude non-file dynamic module names; preserve original snapshot and logs',
        'scientific_configuration_changed': False, 'test_executed': False})
    manifest = g4.read_json(run / 'manifest.json')
    stages = [
        ('freeze_recovery', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'freeze', '--snapshot-name', 'training_source_snapshot_recovery']),
        ('identity_before_pilot', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'verify-inputs', '--phase', 'before_pilot']),
        ('pilot', ['统一模型代码/gates/g5/audits/e2e_g5_pilot.py']),
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
