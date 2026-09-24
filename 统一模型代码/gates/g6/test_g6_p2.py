"""P2 separation and reporting invariants; actual model replay lives in --prepare."""
import unittest

import torch

from 统一模型代码.gates.g6.g6_p2_model import SplitPhysical, subband_counts
from 统一模型代码.gates.g6.g6_p2_train import band_guard


class Tests(unittest.TestCase):
    def test_zero_initialization_and_size(self):
        q,z = torch.randn(2,3,128),torch.randn(2,3,19)
        for arm in ('a','b','c'):
            head = SplitPhysical(arm)
            self.assertTrue(torch.equal(head.weights(q,z,arm),z.sigmoid()))
            count = sum(p.numel() for p in head.selector.parameters()) if head.selector else 0
            self.assertEqual(count,0 if arm == 'a' else 2451)

    def test_branch_gradients(self):
        for arm in ('b','c'):
            q = torch.randn(2,3,128,requires_grad=True)
            z = torch.randn(2,3,19,requires_grad=True)
            head = SplitPhysical(arm)
            head.weights(q,z,arm).sum().backward()
            self.assertIsNone(z.grad)
            self.assertTrue(q.grad is None or bool((q.grad==0).all()))
            self.assertGreater(float(head.selector.weight.grad.norm()),0)
            with torch.no_grad():
                head.selector.weight.add_(-.001*head.selector.weight.grad)
            q.grad = None
            head.weights(q,z,arm).sum().backward()
            self.assertIsNone(z.grad)
            if arm == 'b':
                self.assertIsNone(q.grad)
            else:
                self.assertGreater(float(q.grad.norm()),0)

    def test_b_c_identical_forward(self):
        b,c = SplitPhysical('b'),SplitPhysical('c')
        with torch.no_grad():
            b.selector.weight.normal_()
        c.load_state_dict(b.state_dict())
        q,z = torch.randn(2,3,128),torch.randn(2,3,19)
        self.assertTrue(torch.equal(b.weights(q,z,'b'),c.weights(q,z,'c')))

    def test_counts_use_same_assignment_and_ignore(self):
        bands = torch.tensor([[1.,1,0],[0,1,1]])
        ignore = torch.tensor([[0.,0,0],[0,0,1]])
        logits = torch.tensor([[-1.,-1,1],[1,1,-1]])
        weights = torch.full((2,3),.6)
        result = subband_counts(logits,weights,bands,ignore,{1:0,0:1})
        self.assertEqual(result['shared_positive'],2)
        self.assertEqual(result['exclusive_positive'],1)
        self.assertEqual(result['semantic_shared_missed'],1)
        self.assertEqual(result['semantic_exclusive_missed'],0)
        self.assertEqual(result['physical_shared_missed'],0)
        self.assertTrue(all(v==0 for v in subband_counts(logits,weights,bands,ignore,{}).values()))

    def test_band_tolerance(self):
        for ci,expected in [([-.01,.005],'MAINTAINED'),([-.03,-.011],'CLEAR_COST'),([-.02,0],'UNCERTAIN')]:
            self.assertEqual(band_guard({'band_only_f1':{'ci95':ci}}),expected)


if __name__ == '__main__':
    unittest.main()
