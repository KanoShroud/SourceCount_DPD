"""CPU tests for exact epoch recovery, maximum-horizon schedule, and extension rules."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from 统一模型代码.gates.g7 import compact_data
from 统一模型代码.gates.g7.formal_train import train, lr_factor, needs_extension


class _Model:
    def __init__(self, seed):
        torch.manual_seed(seed)
        self.network = torch.nn.Sequential(torch.nn.Linear(2, 5), torch.nn.ReLU(),
                                           torch.nn.Dropout(.25), torch.nn.Linear(5, 1))

    def mode(self, training):
        self.network.train(training)

    def state(self):
        return copy.deepcopy(self.network.state_dict())

    def restore(self, state):
        self.network.load_state_dict(state, strict=True)

    def forward_loss(self, batch):
        value = (self.network(batch['x'])-batch['y']).square().mean()
        return value, {'mse': value}


class _Progress:
    def update(self, done):
        pass


class _Runtime:
    def __init__(self, path, fail_batch=None):
        self.out = path
        self.fail_batch, self.training_batches = fail_batch, 0
        self.config = {'phases': {'ch3': dict(batch_size=2, base_epochs=2, max_epochs=4,
            warmup_epochs=1, min_lr_factor=.01, evaluate_every=1)}}

    def model(self, seed, phase):
        return _Model(seed)

    def optimizer(self, model, phase):
        return torch.optim.AdamW(model.network.parameters(), lr=.003)

    def ids(self, phase, split):
        return list(range(6 if split == 'train' else 4))

    def data(self, phase, seed, split, ids):
        if split == 'train':
            self.training_batches += 1
            if self.training_batches == self.fail_batch:
                raise RuntimeError('INJECTED_PARTIAL_EPOCH')
        x = torch.tensor([[i/10, (i+1)/10] for i in ids])
        return {'x': x, 'y': x.sum(-1, keepdim=True)}

    def guard(self):
        pass

    def progress(self, label, total):
        return _Progress()


def _assert_nested(test, a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a:
            _assert_nested(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            _assert_nested(test, x, y)
    else:
        test.assertEqual(a, b)


class FormalTrainerTests(unittest.TestCase):
    def test_schedule_uses_max_horizon(self):
        cfg = dict(max_epochs=120, warmup_epochs=5, min_lr_factor=.01)
        self.assertEqual(lr_factor(0, cfg), .2)
        self.assertEqual(lr_factor(4, cfg), 1.)
        self.assertGreater(lr_factor(79, cfg), .01)
        self.assertAlmostEqual(lr_factor(119, cfg), .01)
        self.assertAlmostEqual(lr_factor(120, cfg), .01)

    def test_partial_epoch_resume_and_extension_exact(self):
        with tempfile.TemporaryDirectory(prefix='g7_formal_cpu_') as folder:
            base = Path(folder)
            with patch.object(compact_data, 'BASE', base):
                uninterrupted = _Runtime(base/'uninterrupted')
                train(uninterrupted, 17, 'ch3', 2)
                expected = train(uninterrupted, 17, 'ch3', 4)
                failed = _Runtime(base/'resumed', fail_batch=5)
                with self.assertRaisesRegex(RuntimeError, 'INJECTED_PARTIAL_EPOCH'):
                    train(failed, 17, 'ch3', 2)
                committed = compact_data.read(failed.out/'training/17/ch3/completed.json')
                self.assertEqual(committed['completed_epoch'], 1)
                resumed = _Runtime(failed.out)
                train(resumed, 17, 'ch3', 2)
                actual = train(resumed, 17, 'ch3', 4)
                left = compact_data.load(expected['checkpoint'], uninterrupted.out)
                right = compact_data.load(actual['checkpoint'], resumed.out)
                for key in ('state', 'optimizer', 'scheduler', 'generator', 'steps'):
                    _assert_nested(self, left[key], right[key])
                self.assertEqual(left['steps'], 12)
                self.assertEqual(left['scheduler']['last_epoch'], 4)
                for a, b in zip(left['history'], right['history']):
                    for key in ('loss', 'validation', 'learning_rates', 'next_learning_rates'):
                        _assert_nested(self, a[key], b[key])

    def test_extension_requires_late_actual_improvement(self):
        cfg = {'phases': {'ch3': dict(base_epochs=80, max_epochs=120,
                warmup_epochs=5, evaluate_every=2, min_lr_factor=.01)}}
        def result(values):
            return {'history': [{'epoch': e, 'validation': {'val_loss': v}}
                                for e, v in zip((76, 78, 80), values)]}
        self.assertTrue(needs_extension(result((3., 2., 1.)), 'ch3', cfg))
        self.assertFalse(needs_extension(result((1., 1., 1.)), 'ch3', cfg))
        self.assertFalse(needs_extension(result((1., 2., 3.)), 'ch3', cfg))


if __name__ == '__main__':
    unittest.main()
