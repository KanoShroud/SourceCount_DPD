"""Synthetic engineering tests: clean initialization, full-state handoff, routing."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

import torch

from 统一模型代码.gates.g7.formal_model import FormalFoundation, FormalCandidate


def no_historical_loading():
    stack = ExitStack()
    for target in (
        'torch.load',
        '统一模型代码.gates.g5.e2e_g5_features.build_models',
        '统一模型代码.gates.g5.e2e_g5_model.build_context',
        '统一模型代码.gates.g7.compact_foundation.build_models',
        '统一模型代码.gates.g7.compact_model.build_context',
    ):
        stack.enter_context(patch(target, side_effect=AssertionError('Historical loader called')))
    return stack


class FormalModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_no_loading_and_deterministic_native_initialization(self):
        with no_historical_loading():
            for kind in ('ch3', 'd8'):
                a = FormalFoundation('.', {}, 11, kind, 'cpu')
                b = FormalFoundation('.', {}, 11, kind, 'cpu')
                c = FormalFoundation('.', {}, 12, kind, 'cpu')
                for key, value in a.model.state_dict().items():
                    torch.testing.assert_close(value, b.model.state_dict()[key], rtol=0, atol=0)
                self.assertTrue(any(not torch.equal(p, q) for p, q in zip(a.model.parameters(), c.model.parameters())))
                self.assertTrue(all(p.requires_grad for p in a.model.parameters()))
                opt = a.optimizer()
                self.assertEqual(opt.param_groups[0]['lr'], 1e-3)
                self.assertEqual(opt.param_groups[0]['weight_decay'], 5e-4 if kind == 'ch3' else 5e-3)
                if kind == 'ch3':
                    self.assertEqual(a.model.max_src, 10)

    def test_foundation_handoff_and_full_buffer_restore(self):
        with no_historical_loading():
            ch3 = FormalFoundation('.', {}, 11, 'ch3', 'cpu')
            d8 = FormalFoundation('.', {}, 11, 'd8', 'cpu')
            candidate = FormalCandidate('.', {}, 11, dict(ch3=ch3.state(), d8=d8.state()), 'cpu')
            self.assertTrue(candidate.foundations_loaded)
            for native, target in [(ch3.model, candidate.context.ch3), (d8.model, candidate.context.d8)]:
                for key, value in native.state_dict().items():
                    torch.testing.assert_close(value, target.state_dict()[key], rtol=0, atol=0)
            checkpoint = candidate.state()
            with torch.no_grad():
                candidate.context.ch3.backbone[1].running_mean.add_(3)
                next(candidate.context.d8.parameters()).add_(2)
            candidate.restore(checkpoint)
            for name, model in [('full_ch3', candidate.context.ch3), ('full_d8', candidate.context.d8)]:
                for key, value in model.state_dict().items():
                    torch.testing.assert_close(value, checkpoint[name][key], rtol=0, atol=0)
            with self.assertRaises(ValueError):
                candidate.restore(dict(base=checkpoint['base']))
            with self.assertRaises(ValueError):
                candidate.load_foundations(dict(kind='ch3', model=ch3.state()['model']), d8.state())

    def test_paired_local_initialization_and_optimizer_groups(self):
        with no_historical_loading():
            a = FormalCandidate('.', {}, 11, device='cpu')
            b = FormalCandidate('.', {}, 11, device='cpu')
            torch.nn.init.constant_(a.physical.selector.weight, .125)
            checkpoint = a.state()
            b.restore(checkpoint)
            a.initialize_local()
            torch.randn(71)
            b.initialize_local()
            for key, value in a.refiner.state_dict().items():
                torch.testing.assert_close(value, b.refiner.state_dict()[key], rtol=0, atol=0)
            torch.testing.assert_close(a.selector.selector.weight, a.physical.selector.weight, rtol=0, atol=0)
            candidate_lrs = {g['name']:g['lr'] for g in a.optimizer('baseline').param_groups}
            self.assertEqual(candidate_lrs, dict(new_candidate=1e-3, band_heads=1e-4,
                d8_tail=2e-5, ch3_tail=1e-5, physical=1e-3))
            local_lrs = {g['name']:g['lr'] for g in a.optimizer('local').param_groups}
            self.assertEqual(local_lrs, dict(query=1e-4, ch3_tail=1e-5, local=1e-3))
            for arm in ('S', 'E'):
                q = torch.randn(1, 3, 128, requires_grad=True)
                logits = torch.randn(1, 3, 19, requires_grad=True)
                a.selector.weights(q, logits, arm).sum().backward()
                self.assertIsNone(logits.grad)
                if arm == 'S':
                    self.assertIsNone(q.grad)
                else:
                    self.assertGreater(float(q.grad.abs().sum()), 0.)

    def test_zero_selector_needs_its_first_update_before_query_feedback(self):
        with no_historical_loading():
            candidate = FormalCandidate('.', {}, 11, device='cpu')
        selector = candidate.selector
        optimizer = torch.optim.AdamW(selector.parameters(), lr=1e-3)
        query = torch.randn(1, 3, 128, requires_grad=True)
        logits = torch.randn(1, 3, 19, requires_grad=True)
        selector.weights(query, logits, 'E').square().mean().backward()
        self.assertEqual(float(query.grad.abs().sum()), 0.)
        self.assertGreater(float(selector.selector.weight.grad.abs().sum()), 0.)
        optimizer.step()
        query.grad = None
        selector.weights(query, logits, 'E').square().mean().backward()
        self.assertGreater(float(query.grad.abs().sum()), 0.)
        query.grad = None
        selector.weights(query, logits, 'S').square().mean().backward()
        self.assertIsNone(query.grad)


if __name__ == '__main__':
    unittest.main()
