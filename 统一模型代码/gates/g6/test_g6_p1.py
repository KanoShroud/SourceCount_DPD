import unittest
import io
from unittest.mock import patch

import numpy as np
import torch

from 统一模型代码.gates.g6.g6_p1_model import PhysicalResidual, identity_hits
from 统一模型代码.gates.g5.r2.g5_r2_train import must_extend


class Tests(unittest.TestCase):
    def test_prefetch_order_and_worker_error(self):
        from 统一模型代码.gates.g6.g6_p1_speed import physical_batches
        def fake_batches(features,indices,prefetch):
            for ids in indices:
                if int(ids[0]) == 9:
                    raise ValueError('worker error must reach caller')
                yield ids,ids.clone()
        ids = [torch.tensor([3,1]),torch.tensor([2,0])]
        with patch('统一模型代码.gates.g6.g6_p1_speed.batches',fake_batches):
            sync = list(physical_batches(None,None,ids,'train',False,False))
            pref = list(physical_batches(None,None,ids,'train',False,True))
            for (a,x,_),(b,y,_) in zip(sync,pref):
                self.assertTrue(torch.equal(a,b) and torch.equal(x,y))
            iterator = physical_batches(None,None,[ids[0],torch.tensor([9])],'train',False)
            next(iterator)
            with self.assertRaisesRegex(ValueError,'worker error'):
                iterator.close()

    def test_zero_init_and_gradient_onset(self):
        torch.manual_seed(42)
        head = PhysicalResidual()
        x = torch.randn(1,3,81,81,requires_grad=True)
        y = head(x)
        self.assertEqual(tuple(y.shape),(1,3,401,401))
        self.assertEqual(float(y.detach().abs().max()),0)
        loss = (y-1).square().mean()
        loss.backward()
        self.assertEqual(float(x.grad.abs().max()),0)
        self.assertGreater(float(head.net[-1].weight.grad.abs().max()),0)
        optimizer = torch.optim.AdamW(head.parameters(),lr=1e-4)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        x.grad = None
        head(x).square().mean().backward()
        self.assertGreater(float(x.grad.abs().max()),0)

    def test_spatial_matching_does_not_rebind_using_bands(self):
        truth = np.array([[0.,0.],[10.,0.]])
        pred = truth.copy()
        bands = np.eye(2)
        logits = np.array([[-1,1],[1,-1]])
        result = identity_hits(truth,pred,logits,[0,1],bands,np.zeros_like(bands))
        self.assertEqual(result,dict(spatial_identity_denominator=2,spatial_identity_numerator=0))
        result = identity_hits(truth,[],logits,[],bands,np.zeros_like(bands))
        self.assertEqual(result['spatial_identity_denominator'],0)

    def test_extension(self):
        h = [{'epoch':e,'validation':{'joint_recall100_f1_08':v,'gospa_m':20}}
             for e,v in [(0,.7),(16,.71),(18,.72),(20,.73)]]
        self.assertTrue(must_extend(h,end=20))
        h[0]['validation']['joint_recall100_f1_08'] = .8
        self.assertFalse(must_extend(h,end=20))

    def test_physical_grid_endpoint_alignment(self):
        head = PhysicalResidual()
        with torch.no_grad():
            for layer in (head.net[0],head.net[2],head.net[4]):
                layer.weight.zero_(); layer.bias.zero_()
            head.net[0].weight[0,0,1,1] = 1
            head.net[2].weight[0,0,1,1] = 1
            head.net[4].weight[0,0,0,0] = 1
            x = torch.linspace(0,400,81)[None,None,None].expand(1,1,81,81)
            y = head(x)
            torch.testing.assert_close(y[0,0,200],torch.arange(401).float())

    def test_chinese_console_capture(self):
        from 统一模型代码.gates.g5.r2.e2e_g5_r2 import Log
        console,log = io.StringIO(),io.StringIO()
        tee = Log(console,log)
        tee.write('阶段：中文路径检查通过\n')
        tee.write('\r训练进度 50%'); tee.write('\n')
        tee.write('结果写入完成\n'); tee.flush()
        self.assertIn('中文路径检查通过',log.getvalue())
        self.assertNotIn('训练进度',log.getvalue())
        self.assertIn('结果写入完成',console.getvalue())

    def test_partial_cache_resume(self):
        from 统一模型代码.gates.g6.g6_p1 import verify_cache_extension
        pilot = {'0':{'sha256':'same','path':'old'}}
        verify_cache_extension(pilot,{**pilot,'1':{'sha256':'new','path':'new'}})
        with self.assertRaises(RuntimeError):
            verify_cache_extension(pilot,{'0':{'sha256':'changed','path':'old'}})


if __name__ == '__main__':
    unittest.main()
