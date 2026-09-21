"""G5阶段边界身份复核与正式训练代码冻结；不读取test。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_model import g4  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402


def rows_in(value):
    if isinstance(value, dict):
        if {'path', 'sha256', 'size_bytes'} <= value.keys():
            yield value
        else:
            for item in value.values():
                yield from rows_in(item)
    elif isinstance(value, list):
        for item in value:
            yield from rows_in(item)


def verify_inputs(run, phase):
    """完整阶段检查；大快照按已登记8MiB块读取，校验和消费同字节。"""
    start = time.time()
    destination = run / f'identity_{phase}.json'
    if destination.exists():
        raise FileExistsError(destination)
    manifests = [run / 'input_manifest.json', run / 'feature_manifest.json']
    seen, checked = set(), []
    for manifest_path in manifests:
        manifest = g4.read_json(manifest_path)
        if manifest_path.name == 'feature_manifest.json' and manifest['status'] != 'PASS':
            raise RuntimeError('Feature preparation has not passed')
        registered_inputs = manifest['files'] if manifest_path.name == 'feature_manifest.json' else manifest
        for row in rows_in(registered_inputs):
            path = Path(row['path']).resolve(strict=True)
            if not path.is_relative_to(run.resolve()) or path.is_relative_to(ROOT.parent / 'SourceCount_DPD/outputs'):
                raise RuntimeError(f'Unexpected G5 input path: {path}')
            key = (str(path), row['sha256'])
            if key in seen:
                continue
            seen.add(key)
            if 'blocks' in row:
                digest, offset = hashlib.sha256(), 0
                for block in row['blocks']:
                    data = verified_read({**block, 'path': str(path)}, run / 'anomalies',
                                         offset=offset, length=block['size_bytes'])
                    digest.update(data)
                    offset += len(data)
                if offset != row['size_bytes'] or path.stat().st_size != offset or digest.hexdigest() != row['sha256']:
                    raise RuntimeError(f'Block and file identity disagree: {path}')
            else:
                verified_read(row, run / 'anomalies')
            checked.append({'path': str(path), 'sha256': row['sha256'], 'size_bytes': row['size_bytes']})
    report = {'status': 'PASS', 'phase': phase, 'files': checked, 'seconds': time.time()-start,
              'test_executed': False}
    g4.write_json(destination, report)
    return report


def source_paths():
    # 本Gate入口、历史复用模块与实际导入的仓库内模型源码均登记。
    paths = set((ROOT / '统一模型代码').rglob('*.py'))
    paths.update((ROOT / '统一模型代码/configs').glob('e2e_g5.json'))
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if filename:
            # torch.ops/classes expose synthetic relative __file__ values.
            if getattr(getattr(module, '__spec__', None), 'origin', None) is None and not Path(filename).is_absolute():
                continue
            path = Path(filename).resolve()
            if path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / 'outputs_e2e'):
                paths.add(path)
    return sorted(paths)


def freeze(run, snapshot_name='training_source_snapshot'):
    destination = run / 'training_code_contract.json'
    if destination.exists():
        raise FileExistsError(destination)
    if Path(snapshot_name).name != snapshot_name:
        raise ValueError('Snapshot name must be a single directory name')
    snapshot = run / snapshot_name
    snapshot.mkdir(exist_ok=False)
    rows = []
    for path in source_paths():
        payload = path.read_bytes()
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(payload)
        rows.append({'path': str(path), 'size_bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    for name in ('manifest.json', 'input_manifest.json', 'feature_manifest.json', 'provenance_audit.json'):
        rows.append(g4.identity(run / name))
    g4.write_json(destination, {'status': 'FROZEN_BEFORE_PILOT_AND_TRAINING', 'created_at': time.time(), 'files': rows,
        'scope': 'Training contract; not a retroactive claim that feature-generation dependencies were frozen at launch',
        'feature_dependency_note': 'During preparation, read-failure metadata and missing-file retry handling were improved; normal successful-read semantics unchanged.'})
    return verify_code(run)


def verify_code(run):
    active = run / 'engineering_v4/contract.json'
    if not active.exists():
        active = run / 'engineering_v3/contract.json'
    if not active.exists():
        active = run / 'engineering_v2/contract.json'
    contract = g4.read_json(active if active.exists() else run / 'training_code_contract.json')
    for row in contract['files']:
        verified_read(row, run / 'anomalies')
    return {'status': 'PASS', 'files': len(contract['files'])}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--action', choices=('freeze', 'verify-code', 'verify-inputs'), required=True)
    parser.add_argument('--phase', default='before_pilot')
    parser.add_argument('--snapshot-name', default='training_source_snapshot')
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    result = freeze(run, args.snapshot_name) if args.action == 'freeze' else verify_code(run) if args.action == 'verify-code' else verify_inputs(run, args.phase)
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}), flush=True)
