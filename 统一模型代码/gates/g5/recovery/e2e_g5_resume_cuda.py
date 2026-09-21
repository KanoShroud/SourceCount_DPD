"""One-time recovery of CUDA-statistics initialization, before any training update."""
import argparse
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_contract import freeze  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_execute import g4, run_stage  # noqa: E402


def main(run):
    run = run.resolve(strict=True)
    manifest = g4.read_json(run / 'manifest.json')
    assert manifest['config']['wall_limit_seconds'] == 172800
    assert g4.read_json(run / 'execution_status.json')['status'] == 'STOP'
    assert g4.read_json(run / 'pilot_report.json')['status'] == 'PASS'
    assert 'reset_peak_memory_stats' in (run / 'execution_logs/training.log').read_text(encoding='utf-8')
    training = (run / 'training').resolve(strict=True)
    assert training.parent == run and not any(p.is_file() for p in training.rglob('*'))
    archive = run / 'cuda_init_failure_evidence'
    archive.mkdir(exist_ok=False)
    for name in ('execution_status.json', 'training_code_contract.json'):
        shutil.copy2(run / name, archive / name)
    contract = g4.read_json(run / 'training_code_contract.json')
    # Every old contract byte must still match, except precisely the one-line fix.
    from 统一模型代码.common.g5_verified_io import verified_read
    for row in contract['files']:
        path = Path(row['path'])
        if path == ROOT / '统一模型代码/gates/g5/e2e_g5_train.py':
            original = (run / 'training_source_snapshot_budget48/统一模型代码/e2e_g5_train.py').read_text(encoding='utf-8')
            current = path.read_text(encoding='utf-8')
            assert current.replace('    torch.cuda.init()  # Statistics reset requires an initialized CUDA context.\n', '', 1) == original
        else:
            verified_read(row, run / 'anomalies')
    training.rename(archive / 'empty_training')
    (run / 'training_code_contract.json').rename(archive / 'original_contract.json')
    g4.write_json(archive / 'receipt.json', {'cause': 'CUDA statistics reset before initialization',
        'fix': 'Explicit torch.cuda.init before statistics reset', 'training_updates_before_failure': 0,
        'scientific_configuration_changed': False, 'test_executed': False})
    freeze(run, 'training_source_snapshot_cuda_init')
    stages = [
        ('identity_cuda_init', ['统一模型代码/gates/g5/e2e_g5_contract.py', '--action', 'verify-inputs', '--phase', 'cuda_init']),
        ('training_cuda_init', ['统一模型代码/gates/g5/e2e_g5_train.py']),
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
