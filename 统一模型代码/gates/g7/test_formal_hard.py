"""PB-BASE pure CPU contract checks; no model checkpoint or data reads."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from 统一模型代码.gates.g7 import formal_hard as hard
import s2g3_composability as native


class FormalHardTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_all_ten_slots_strict_threshold_and_union(self):
        logits = torch.zeros(10, 19)
        active, selected = hard.predicted_selection(logits)
        self.assertEqual(len(active), 0)
        logits[[0, 4, 7, 9], [1, 3, 5, 7]] = 1
        active, selected = hard.predicted_selection(logits)
        self.assertEqual(active.tolist(), [0, 4, 7, 9])
        self.assertEqual(selected.shape, (4, 19))
        with patch.object(hard, 'build_oracle_fine', return_value=torch.zeros(201, 201)) as build:
            hard.build_predicted_fine('synthetic_iq', selected, ('lo', 'hi'), 'cpu')
        passed = build.call_args.args[1]
        self.assertEqual(passed['count'], 4)
        torch.testing.assert_close(passed['oracle_slots'], selected)
        self.assertEqual(set(passed), {'count', 'oracle_slots'})

    def test_native_nms_topk_offset_equivalence_and_no_count_cap(self):
        heat = torch.randn(1, 201, 201)
        offset = torch.randn(2, 201, 201)*3
        with patch.object(native, 'pixel_to_phys', lambda xy:xy*10-1000):
            expected, expected_scores = native.decode_d8_sample(heat, offset, 10)
        actual, scores = hard.decode_hard(heat, offset, 10)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(scores, expected_scores)
        self.assertEqual(len(actual), 10)
        self.assertEqual(hard.PEAK_SIZE, 9)
        self.assertEqual(hard.decode_hard(heat, offset, 0)[0].shape, (0, 2))

    def test_spatial_metrics_do_not_fabricate_binding(self):
        logits = torch.full((10, 19), -4.)
        logits[9, :3] = 4
        band = torch.zeros(3, 19)
        band[0, :3] = 1
        record = dict(positions=torch.tensor([[0., 0.], [0., 0.], [0., 0.]]),
            count=1, band=band, ignore=torch.zeros_like(band), index=0,
            metadata=dict(raw_index=23, true_k=1))
        row = hard.spatial_row(record, [[3., 4.]], logits, [9], .2)
        self.assertEqual(row['band_only_f1'], [1.])
        self.assertEqual(row['matched_errors_m'], [5.])
        self.assertNotIn('joint_tp', row)
        self.assertIsNone(row['slot_position_binding'])
        summary = hard.summarize_hard([row])
        self.assertIsNone(summary['joint_recall100_f1_08'])
        self.assertEqual(summary['matched_rmse_m'], 5.)
        self.assertEqual(summary['recall10'], 1.)

    def test_test_split_rejected_before_loading(self):
        with self.assertRaises(ValueError):
            hard.evaluate_hard(None, 17, 'test')


if __name__ == '__main__':
    unittest.main()
