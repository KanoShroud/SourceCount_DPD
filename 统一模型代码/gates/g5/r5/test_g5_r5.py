"""Numerics, geometry and calibration tests; no research inputs."""
import itertools
import unittest

import numpy as np
import torch

from 统一模型代码.gates.g5.r5.g5_r5 import (
    B, calibrate, centered_cosine, decision, local_signature,
    preprocess, proposal, split_scenes, transitions,
)


class R5Tests(unittest.TestCase):
    def test_preprocess_matches_ch3(self):
        raw = np.random.default_rng(42).uniform(0, 10, (81, 81, 19)).astype(np.float32)
        spectrum = np.asarray(raw, dtype=np.float32).transpose(2, 1, 0)
        spectrum = np.log(spectrum + 1.0)
        spectrum = (spectrum - spectrum.mean()) / (spectrum.std() + 1e-6)
        np.testing.assert_array_equal(preprocess(raw).numpy(), spectrum)

    def test_coordinate_patch_and_border(self):
        y, x = torch.meshgrid(torch.linspace(-2000, 2000, 81), torch.linspace(-2000, 2000, 81), indexing='ij')
        coarse = torch.stack([x + 2*y + b for b in range(19)])
        result = local_signature(coarse, np.array([[75., -125.], [2000., 2000.]]))
        torch.testing.assert_close(result[0], torch.arange(19).double() - 175, atol=1e-3, rtol=0)
        torch.testing.assert_close(result[1], torch.arange(19).double() + 5950, atol=1e-3, rtol=0)

    def test_cosine_swap_and_common_component(self):
        bands = torch.tensor([[1., 0.], [0., 1.]])
        response = torch.tensor([[0., 3.], [3., 0.]])
        scores = centered_cosine(bands, response)
        p = proposal(scores, torch.tensor([[0, 1], [1, 0]]))
        self.assertEqual(p['best'], 1)
        self.assertAlmostEqual(p['margin'], 2.)
        torch.testing.assert_close(scores, centered_cosine(bands, response + torch.tensor([5., 7.])))

    def test_degenerate_ties_and_k01(self):
        self.assertIsNone(centered_cosine(torch.ones(2, 19), torch.randn(2, 19)))
        self.assertIsNone(centered_cosine(torch.randn(2, 19), torch.ones(2, 19)))
        self.assertEqual(proposal(torch.ones(3, 3), torch.tensor(list(itertools.permutations(range(3)))))['reason'], 'TIED_BEST')
        for k in (0, 1):
            self.assertEqual(proposal(None, torch.empty((1, k), dtype=torch.long))['best'], 0)

    def test_partition_shared_and_balanced(self):
        records = [{'raw_index': i, 'true_k': i//128} for i in range(512)]
        parts = split_scenes(records)
        self.assertEqual(parts, split_scenes(records[::-1]))
        self.assertFalse(set(parts['calibration']) & set(parts['check']))
        for ids in parts.values():
            self.assertEqual(len(ids), 256)
            self.assertEqual([sum(i//128 == k for i in ids) for k in range(4)], [64]*4)

    def row(self, raw_index, margin, delta):
        return {'raw_index': raw_index, 'metrics': [{'joint_tp': 1}, {'joint_tp': 1 + delta}],
                'scores': {B: {'best': 1, 'margin': margin}}}

    def test_calibration_strict_threshold_and_no_correction(self):
        rows = [self.row(1, .1, -1), self.row(2, .4, 1)] * 2
        tau, report = calibrate(rows, B)
        self.assertEqual(tau, .1)
        self.assertEqual(report['selected']['net_joint_hits'], 2)
        self.assertEqual(decision(rows[0], B, tau), 0)
        self.assertEqual(report['distinct_scene_count'], 2)
        tau, report = calibrate([self.row(1, .2, -1)], B)
        self.assertTrue(np.isinf(tau))
        self.assertIsNone(report['selected']['tau'])
        self.assertIsNone(transitions(rows, B, float('inf'))['beneficial_per_change'])

    def test_three_source_permutation(self):
        bands = torch.eye(3)
        response = bands[[2, 0, 1]]
        perms = torch.tensor(list(itertools.permutations(range(3))))
        p = proposal(centered_cosine(bands, response), perms)
        self.assertEqual(perms[p['best']].tolist(), [1, 2, 0])


if __name__ == '__main__':
    unittest.main()
