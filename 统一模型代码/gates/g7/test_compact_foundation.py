"""Native-task contract tests using synthetic tensors, without checkpoint writes."""
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from 统一模型代码.gates.g7 import compact_foundation as cf
from train_v26 import compute_loss as ch3_loss
import train_yolo
from s2g4r2_soft19 import hard_actual_mask


def record(count=1):
    bands = torch.zeros(3, 19)
    bands[0, 9] = 1
    return dict(coarse=torch.arange(19*41*41, dtype=torch.float32).reshape(19, 41, 41),
        oracle_fine=torch.rand(201, 201), band=bands, ignore=torch.zeros_like(bands),
        count=count, positions=torch.tensor([[13., -17.], [501., 3.], [-302., 104.]]),
        actual_fc=torch.tensor([0., 25e6, -20e6]), actual_bw=torch.tensor([3e6, 7e6, 11e6]), b_win=10e6)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(19)
        self.out = nn.Linear(19, 5*19)

    def forward(self, x):
        return self.out(self.bn(x).mean((-2, -1))).reshape(len(x), 5, 19)


class CompactFoundationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_native_preprocessing_and_targets(self):
        r = record(2)
        raw = r['coarse'].numpy()
        logged = np.log(raw+1)
        expected = (logged-logged.mean())/(logged.std()+1e-6)
        np.testing.assert_array_equal(cf.ch3_input(r).numpy(), expected)
        fine = torch.log(r['oracle_fine']+1)
        torch.testing.assert_close(cf.d8_input(r)[0], (fine-fine.mean())/(fine.std()+1e-6), rtol=0, atol=0)
        target, xy = cf.d8_targets([r], 'cpu')
        self.assertEqual(target.shape, (1, 1, 201, 201))
        self.assertEqual(float(target.max()), 1.)
        torch.testing.assert_close(xy[0][0], torch.tensor([101.3, 98.3]))

    def test_native_offset_loss_equivalence(self):
        records = [record(2), record(0)]
        heat = torch.randn(2, 1, 201, 201, requires_grad=True)
        offset = torch.randn(2, 2, 201, 201, requires_grad=True)
        loss, parts = cf.d8_loss(heat, offset, records)
        normalized = torch.stack([r['positions'] for r in records])/1000
        with patch.object(train_yolo, 'EDGE', 1000.), patch.object(train_yolo, 'LAMDA', 10.):
            expected = train_yolo.compute_offset_loss(offset, normalized, torch.tensor([2, 0]), 'cpu')
        torch.testing.assert_close(parts['offset'], expected, atol=1e-5, rtol=1e-5)
        loss.backward()
        self.assertTrue(torch.isfinite(offset.grad).all())
        self.assertTrue(torch.isfinite(heat.grad).all())

    def test_hard_actual_rule_matches_original(self):
        geo = cf._small_geometry('cpu')
        lo = np.arange(-50e6, 45e6, 5e6)
        hi = lo+10e6
        r = record(3)
        raw = dict(fc_offset=r['actual_fc'].numpy()[None], bw_actual=r['actual_bw'].numpy()[None],
                   b_win=r['b_win'], sub_f_lo=lo, sub_f_hi=hi)
        expected, slots = hard_actual_mask(raw, 0, 3)
        np.testing.assert_array_equal(cf.oracle_fft_mask(r, (lo, hi), geo).numpy(), expected)
        np.testing.assert_array_equal(cf.oracle_fft_mask(dict(count=3, oracle_slots=slots), (lo, hi), geo).numpy(), expected)
        # Existing semantic labels must never silently substitute hard_actual.
        with self.assertRaises(KeyError):
            cf.oracle_fft_mask(dict(count=1, band=r['band']), (lo, hi), geo)
        empty = cf.build_oracle_fine(torch.zeros(4, 4096, dtype=torch.complex64), dict(count=0), (lo, hi), 'cpu')
        self.assertEqual(empty.shape, (201, 201))
        self.assertEqual(float(empty.abs().sum()), 0.)

    def test_full_state_includes_bn_and_all_parameters(self):
        with patch.object(cf, 'build_models', return_value=(Tiny(), Tiny())):
            model = cf.FoundationModel('.', {}, 23, 'ch3', 'cpu')
        state = model.state()
        before = state['model']['bn.running_mean'].clone()
        opt = model.optimizer(lr=1e-4)
        self.assertEqual(sum(p.numel() for g in opt.param_groups for p in g['params']),
                         sum(p.numel() for p in model.model.parameters()))
        records = [record(), record()]
        loss, parts = model.forward_loss(records)
        loss.backward()
        opt.step()
        self.assertTrue(all(p.requires_grad for p in model.model.parameters()))
        torch.testing.assert_close(state['model']['bn.running_mean'], before, rtol=0, atol=0)
        model.restore(state)
        for key, value in model.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), state['model'][key], rtol=0, atol=0)
        model.mode(False)
        logits = model._forward(records)
        target, ignore = cf._semantic_targets(records, logits)
        torch.testing.assert_close(model.forward_loss(records)[0], ch3_loss(logits, target, ignore), rtol=0, atol=0)
        self.assertEqual(model.validation_statistics(records)['samples'], 2)
        self.assertEqual(set(parts), {'band'})

    def test_native_architectures_small_roi_full_backward(self):
        from train_v26 import SourceDetectionNet
        from yolo_model import YOLOv8Loc
        records = [record(), record(2)]
        for kind, network in [('ch3', SourceDetectionNet(n_sub=19, max_src=5, mode='transformer')),
                              ('d8', YOLOv8Loc(method='dualhead', dropout=.4, grad_alpha=1.))]:
            with patch.object(cf, 'build_models', return_value=(network, network)):
                model = cf.FoundationModel('.', {}, 23, kind, 'cpu')
            total, _ = model.forward_loss(records)
            total.backward()
            self.assertTrue(torch.isfinite(total))
            missing = [name for name, p in model.model.named_parameters() if p.grad is None]
            self.assertEqual(missing, [], f'{kind}: unused trainable parameters')
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.model.parameters()))


if __name__ == '__main__':
    unittest.main()
