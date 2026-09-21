"""轻量验证G5延长规则、配对统计与checkpoint完整往返，不产生模型性能证据。"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from 统一模型代码.gates.g5.e2e_g5_evaluate import metrics, paired  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_train import must_extend, save_checkpoint, load_checkpoint  # noqa: E402


def history(values):
    return [{'epoch': epoch, 'validation': {'overall': {'gospa_mean_m': value}}}
            for epoch, value in values]


def main():
    assert must_extend(history([(0, 100), (44, 10), (46, 9.8), (48, 9.4)]), 48, .5)
    assert not must_extend(history([(0, 100), (44, 10), (46, 9.8), (48, 9.6)]), 48, .5)
    assert not must_extend(history([(0, 8), (44, 10), (46, 9.8), (48, 9.4)]), 48, .5)
    assert must_extend(history([(0, 100), (60, 10), (62, 9.8), (64, 9.4)]), 64, .5)
    row = dict(raw_index=1, true_count=1, predicted_count=0, matched_errors_m=[], gospa_m=100.,
               tp_at_100m=0, band_f1=[0.], band_iou=[0.], band_only_f1=[0.], band_only_iou=[0.])
    assert np.isnan(metrics([row], np.arange(1))['rmse'])
    assert 'rmse' not in paired([row], [row], 20, 42)
    matched = dict(row, predicted_count=1, matched_errors_m=[5.], gospa_m=5., tp_at_100m=1)
    assert metrics([matched], np.arange(1))['rmse'] == 5
    assert paired([matched], [matched], 20, 42)['rmse_ratio']['ci95'] == [1., 1.]
    better = dict(matched, matched_errors_m=[4.], gospa_m=4.)
    assert paired([matched], [better], 20, 42)['gospa']['ci95'] == [-1., -1.]
    folder = Path(tempfile.mkdtemp(prefix='g5_training_contract_', dir=ROOT / 'outputs_e2e'))
    value = {'epoch': 0, 'nested': {'tensor': torch.arange(4), 'array': np.arange(4),
                                  'tuple': (1, '中文路径与日志')}, 'optimizer': {'state': {}, 'param_groups': []}}
    checkpoint = folder / 'roundtrip.pt'
    save_checkpoint(checkpoint, value)
    loaded = load_checkpoint(checkpoint)
    assert torch.equal(loaded['nested']['tensor'], value['nested']['tensor'])
    assert np.array_equal(loaded['nested']['array'], value['nested']['array'])
    assert loaded['nested']['tuple'] == value['nested']['tuple']
    report = {'status': 'PASS', 'scope': 'engineering_only', 'extension_rule_cases': 4,
              'undefined_rmse_not_zero': True, 'paired_direction_exact': True,
              'checkpoint_nested_roundtrip': True, 'non_ascii_roundtrip': True}
    (folder / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(folder / 'report.json'), **report}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
