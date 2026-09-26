"""Synthetic CPU recovery test; never consumes research data or starts formal training."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np

from 统一模型代码.gates.g5.r2.g5_r2_prepare import same
from 统一模型代码.gates.g7 import g7_train as train
from 统一模型代码.gates.g7.g7_runtime import read


class FakeRuntime:
    def __init__(self, out):
        self.out = Path(out)
        self.manifest = {'config': {}}

    def features(self, split, cache=None):
        return None, SimpleNamespace(counts=torch.ones(8, dtype=torch.long)), [], {}, None

    def context(self, seed, arm):
        torch.manual_seed(seed)
        model, head = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        params = list(model.parameters())+list(head.parameters())
        return model, head, torch.optim.AdamW(params, lr=.01), params

    def batches(self, features, order, split, seed):
        for ids in order:
            yield ids, ids.float()[:, None]/8, None

    def forward(self, model, head, batch, ids, arm, stats):
        return head(model(batch+torch.rand_like(batch)*.01))

    def loss(self, result, targets, ids):
        loss = (result-1).square().mean()
        return loss, {'heatmap':loss}, {}

    def guard(self):
        pass

    def consumed(self, *args):
        pass

    def postcheck(self, *args):
        pass


class RunnerTests(unittest.TestCase):
    def test_epoch_resume_matches_uninterrupted(self):
        def payload(model, head):
            return {'base':model.state_dict(), 'local':head.state_dict()}
        def restore(model, head, state):
            model.load_state_dict(state['base']); head.load_state_dict(state['local'])
        with TemporaryDirectory(prefix='g7_cpu_resume_') as directory, \
                patch.object(train, 'payload', payload), patch.object(train, 'restore', restore), \
                patch.object(train.g4, 'set_mode', lambda m, training:m.train(training)), \
                patch.object(train, 'must_extend', lambda *a, **kw:False), \
                patch.object(train, 'evaluate', lambda *a, **kw:({'joint_recall100_f1_08':0., 'gospa_m':100.}, [])):
            a, b = FakeRuntime(Path(directory)/'whole'), FakeRuntime(Path(directory)/'resumed')
            train.train_track(a, 71, 'f', 2)
            train.train_track(b, 71, 'f', 1)
            train.train_track(b, 71, 'f', 2)
            def saved(runtime):
                row = read(runtime.out/'training/71/f/completed.json')['checkpoint']
                return train.load_registered(row, runtime.out)
            left, right = saved(a), saved(b)
            for key in ('state', 'optimizer', 'generator', 'steps'):
                self.assertTrue(same(left[key], right[key]), key)
            def normalize(value):
                if isinstance(value, (np.ndarray, torch.Tensor)):
                    return value.tolist()
                if isinstance(value, dict):
                    return {k:normalize(v) for k, v in value.items()}
                if isinstance(value, (tuple, list)):
                    return [normalize(v) for v in value]
                return value
            self.assertEqual(normalize(left['rng']), normalize(right['rng']))


if __name__ == '__main__':
    unittest.main()
