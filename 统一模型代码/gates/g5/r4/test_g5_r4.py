"""Synthetic checks for the fixed-position experiment."""
import unittest
import numpy as np
import torch
from 统一模型代码.gates.g5.r4.g5_r4 import ExplicitMatcher, fixed_sample, scores_for, permutation_scores, permutation_loss
from 统一模型代码.gates.g5.r2.g5_r2_model import AssociationHead


class R4Tests(unittest.TestCase):
    def sample(self):
        logits=torch.full((3,19),-5.)
        logits[0,0]=5
        logits[1,1]=5
        truth=np.array([[0.,0.],[500.,0.]],dtype=np.float32)
        bands=np.zeros((3,19),dtype=np.float32)
        bands[0,0]=bands[1,1]=1
        record={'candidates':[{'query':q,'positions':truth.tolist(),'scores':[.8,.7]} for q in range(3)],
                'original':[[0.,0.]]*3}
        return {'logits':logits,'query':torch.randn(3,128),'local':torch.randn(3,2,19,128),
                'record':record,'truth':truth,'bands':bands,'ignore':np.zeros_like(bands),'metadata':{'raw_index':0}}

    def test_fixed_geometry_and_labels(self):
        s=fixed_sample(self.sample())
        self.assertEqual(s['positive'].tolist(),[True,False])
        self.assertEqual(s['metrics'][0]['gospa_m'],s['metrics'][1]['gospa_m'])
        self.assertEqual(s['metrics'][0]['joint_tp'],2)
        self.assertEqual(s['metrics'][1]['joint_tp'],0)
        for head in (AssociationHead(),ExplicitMatcher()):
            self.assertEqual(int(permutation_scores(scores_for(head,s,'cpu'),s['perms']).argmax()),0)

    def test_common_preferences_cancel(self):
        a=torch.tensor([[3.,5.],[4.,6.]])
        torch.testing.assert_close(permutation_scores(a,torch.tensor([[0,1],[1,0]])),torch.tensor([9.,9.]))

    def test_gamma_then_projection_gradient(self):
        torch.manual_seed(42)
        s=fixed_sample(self.sample())
        h=ExplicitMatcher()
        opt=torch.optim.SGD(h.parameters(),lr=.1)
        loss=permutation_loss(scores_for(h,s,'cpu'),s['perms'],s['positive'])
        loss.backward()
        self.assertGreater(abs(float(h.gamma.grad)),0)
        self.assertEqual(float(h.slot.weight.grad.abs().sum()),0)
        opt.step()
        opt.zero_grad()
        permutation_loss(scores_for(h,s,'cpu'),s['perms'],s['positive']).backward()
        self.assertGreater(float(h.slot.weight.grad.abs().sum()),0)
        self.assertGreater(float(h.project.weight.grad.abs().sum()),0)

    def test_empty(self):
        s=self.sample()
        s['logits'].fill_(-5)
        v=fixed_sample(s)
        self.assertEqual(v['perms'].shape,(1,0))
        self.assertTrue(v['positive'].all())


if __name__=='__main__':
    unittest.main()
