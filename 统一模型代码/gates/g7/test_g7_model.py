"""CPU-only synthetic engineering checks; not a performance experiment."""
import unittest

import torch

from 统一模型代码.gates.g7.g7_model import (
    LocalFrequencySelector, LocalRefiner, decode_local, local_loss,
)


class G7ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        torch.set_num_threads(2)

    def test_gradient_routing_and_shapes(self):
        selector = LocalFrequencySelector()
        torch.nn.init.normal_(selector.selector.weight, std=.02)
        head = LocalRefiner()
        for arm in ('S', 'E'):
            q = torch.randn(1, 3, 128, requires_grad=True)
            logits = torch.randn(1, 3, 19, requires_grad=True)
            weights = selector.weights(q, logits, arm)
            maps = torch.randn(1, 3, 8, 9, 9) * weights.mean(-1)[..., None, None, None]
            out = head(maps, q, torch.ones_like(maps, dtype=torch.bool))
            out['candidate_logits'].sum().backward()
            self.assertIsNone(logits.grad)
            if arm == 'S':
                self.assertIsNone(q.grad)
            else:
                self.assertGreater(float(q.grad.abs().sum()), 0)
            self.assertEqual(out['offset'].shape, (1, 3, 8, 2, 9, 9))

    def test_uncovered_source_keeps_semantic_not_local_targets(self):
        heat = torch.zeros(1, 3, 2, 9, 9, requires_grad=True)
        score = torch.zeros(1, 3, 2, requires_grad=True)
        offset = torch.zeros(1, 3, 2, 2, 9, 9, requires_grad=True)
        logits = torch.full((1, 3, 19), -2., requires_grad=True)
        outputs = dict(heat_logits=heat, offset=offset, candidate_logits=score)
        total, parts, info = local_loss(outputs, logits, torch.zeros(1, 3, 2, 2),
            torch.ones_like(heat, dtype=torch.bool), [torch.tensor([[1000., 1000.]])],
            [torch.ones(1, 19)])
        total.backward()
        q = next(iter(info['mappings'][0]))
        self.assertEqual(info['covered_sources'], 0)
        self.assertEqual(float(heat.grad[0, q].abs().sum()), 0)
        self.assertEqual(float(score.grad[0, q].abs().sum()), 0)
        self.assertGreater(float(logits.grad[0, q].abs().sum()), 0)
        self.assertTrue(torch.isfinite(parts['offset']))

    def test_decode_coordinates_mask_and_separation(self):
        heat = torch.full((1, 3, 2, 9, 9), -10.)
        heat[..., 4, 4] = 5
        score = torch.zeros(1, 3, 2)
        score[..., 0] = 2
        offset = torch.zeros(1, 3, 2, 2, 9, 9)
        offset[0, 0, 0, :, 4, 4] = torch.tensor([.2, -.3])
        centers = torch.tensor([[[[0., 0.], [100., 0.]], [[0., 0.], [200., 0.]], [[0., 0.], [300., 0.]]]])
        logits = torch.full((1, 3, 19), -2.)
        logits[0, :2, 0] = 2
        outputs = dict(heat_logits=heat, offset=offset, candidate_logits=score)
        valid = torch.ones_like(heat, dtype=torch.bool)
        row = decode_local(outputs, centers, valid, logits)[0]
        self.assertFalse(row['fallback'])
        self.assertEqual(len(row['joint']), 2)
        self.assertGreaterEqual(float(torch.dist(torch.tensor(row['joint'][0]), torch.tensor(row['joint'][1]))), 30.)
        torch.testing.assert_close(torch.tensor(row['candidates'][0]['positions'][0]),
                                   torch.tensor([2., -3.]), atol=3e-6, rtol=0)
        logits.fill_(-2)
        self.assertEqual(decode_local(outputs, centers, valid, logits)[0]['joint'], [])

    def test_same_assignment_permutation_and_finite_backward(self):
        head = LocalRefiner()
        maps = torch.randn(1, 3, 2, 9, 9, requires_grad=True)
        out = head(maps, torch.randn(1, 3, 128), torch.ones_like(maps, dtype=torch.bool))
        logits = torch.full((1, 3, 19), -5.)
        logits[0, 0, :5] = 5
        logits[0, 1, 10:] = 5
        bands = torch.zeros(2, 19)
        bands[0, 10:] = 1
        bands[1, :5] = 1
        loss, _, info = local_loss(out, logits, torch.zeros(1, 3, 2, 2),
            torch.ones_like(maps, dtype=torch.bool), [torch.tensor([[20., 0.], [-20., 0.]])], [bands])
        self.assertEqual(info['mappings'][0], {0: 1, 1: 0})
        self.assertEqual(info['covered_sources'], 2)
        loss.backward()
        self.assertTrue(torch.isfinite(maps.grad).all())


if __name__ == '__main__':
    unittest.main()
