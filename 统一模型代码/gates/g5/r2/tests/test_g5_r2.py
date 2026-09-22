"""Synthetic protocol and recovery checks; no scientific data or training."""
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch

from 统一模型代码.gates.g5.r2.g5_r2_model import (
    AssociationHead, sample_centers, positive_mask, association_loss, decode_r2, candidate_batch)
from 统一模型代码.gates.g5.r1.g5_r1_decode import decode
from 统一模型代码.gates.g5.r2.g5_r2_runtime import Progress, write, read, Runtime, BASE, RunLock
from 统一模型代码.gates.g5.r2.g5_r2_train import selection_key, must_extend, train_all, commit
from 统一模型代码.gates.g5.r2.e2e_g5_r2 import Log


class R2Tests(unittest.TestCase):
    def test_coordinate_field(self):
        y,x = torch.meshgrid(torch.linspace(-2000,2000,11),torch.linspace(-2000,2000,11),indexing='ij')
        field = torch.stack((x,y))[None].expand(19,-1,-1,-1)
        points = torch.tensor([[-2000.,-2000.],[0,0],[123,456],[2000,2000]])
        sampled = sample_centers(field,points)
        torch.testing.assert_close(sampled[:,0],points,atol=.001,rtol=0)
        # Outside fine-grid offsets use a defined border, not wrapped indexing.
        torch.testing.assert_close(sample_centers(field,torch.tensor([[2010.,-2010.]]))[0,0],
                                   torch.tensor([2000.,-2000.]))

    def test_positive_owner_and_tie(self):
        truth = torch.tensor([[0.,0.],[150.,0.]])
        pos,valid = positive_mask(torch.tensor([[50.,0.],[75.,0.],[100.,0.],[300.,0.]]),truth,0)
        self.assertEqual(pos.tolist(),[True,False,False,False])
        self.assertEqual(valid.tolist(),[True,False,True,True])

    def test_multi_positive_and_empty(self):
        s = torch.tensor([0.,0.,0.],requires_grad=True)
        target = SimpleNamespace(positions=torch.tensor([[[0.,0.]]]),counts=torch.tensor([1]))
        candidates = [{'candidates':[{'positions':[[0,0],[10,0],[200,0]]}]}]
        loss,stats = association_loss([[s]],candidates,[{0:0}],target,torch.tensor([0]),s)
        self.assertAlmostEqual(float(loss.detach()),np.log(1.5),places=6)
        self.assertEqual(stats['positive_candidates'],2)
        loss.backward()
        self.assertLess(float(s.grad[0]),0)
        self.assertGreater(float(s.grad[2]),0)
        zero,_ = association_loss([[s]],candidates,[{}],target,torch.tensor([0]),s)
        self.assertEqual(float(zero.detach()),0)
        zero.backward()
        target.positions += 5000
        empty,stats = association_loss([[s]],candidates,[{0:0}],target,torch.tensor([0]),s)
        self.assertEqual(stats['covered_slots'],0)
        self.assertEqual(float(empty.detach()),0)

    def test_initial_decode_matches_r1_all_k(self):
        rng = np.random.default_rng(42)
        heat = rng.random((3,401,401),dtype=np.float32)
        offset = rng.uniform(-2,2,(3,2,401,401)).astype(np.float32)
        full = decode(np.zeros((3,19),dtype=np.float32),heat,offset)
        for k in range(4):
            logits = np.full((3,19),-1,dtype=np.float32)
            logits[:k] = 1
            ref = decode(logits,heat,offset)
            for scores in (None,[torch.zeros(len(c['scores'])) for c in full['candidates']]):
                got = decode_r2(logits,full,scores)
                self.assertEqual(got['joint'],ref['joint'])
                self.assertEqual(got['active'],ref['active'])

    def test_zero_head_and_feedback(self):
        torch.manual_seed(42)
        head = AssociationHead()
        spatial = torch.randn(19,128,11,11)
        query = torch.randn(128,requires_grad=True)
        logits = torch.randn(19,requires_grad=True)
        points = torch.tensor([[0.,0.],[500.,500.]],requires_grad=True)
        s = head(spatial,query,logits,points)
        self.assertTrue(torch.equal(s,torch.zeros_like(s)))
        optimizer = torch.optim.AdamW(head.parameters(),lr=.001)
        (-s.log_softmax(0)[0]).backward()
        optimizer.step()
        for detach in (True,False):
            q,b = (query.detach(),logits.detach()) if detach else (query,logits)
            s = head(spatial,q,b,points)
            grad = torch.autograd.grad(-s.log_softmax(0)[0],(query,logits,points),allow_unused=True)
            self.assertIsNone(grad[2])
            self.assertEqual([g is None for g in grad[:2]],[detach,detach])
            if not detach:
                self.assertTrue(all(float(g.norm())>0 for g in grad[:2]))

    def test_fast_candidates_equal_r1(self):
        torch.manual_seed(21)
        heat = torch.randn(1,3,401,401)
        offset = torch.randn(1,3,2,401,401)
        fast = candidate_batch(heat,offset)[0]
        old = decode(np.zeros((3,19),dtype=np.float32),heat[0].sigmoid().numpy(),offset[0].numpy())
        self.assertEqual(fast['candidates'],old['candidates'])
        np.testing.assert_array_equal(np.asarray(fast['original'],dtype=np.float32),old['original'])

    def test_selection_extension(self):
        history = [{'epoch':e,'validation':{'joint_recall100_f1_08':v,'gospa_m':20.}}
                   for e,v in ((0,.5),(2,.51),(4,.52),(6,.53),(16,.54),(18,.55),(20,.56))]
        self.assertTrue(must_extend(history))
        history[-1]['validation']['joint_recall100_f1_08'] = .53
        self.assertFalse(must_extend(history))
        self.assertLess(selection_key({'joint_recall100_f1_08':.6,'gospa_m':30},2),
                        selection_key({'joint_recall100_f1_08':.5,'gospa_m':20},0))

    def test_progress_rate_and_utf8_log(self):
        console,log = io.StringIO(),io.StringIO()
        stream = Log(console,log)
        stream.write('中文开始\n')
        stream.write('\r进度')
        stream.write('\n')
        stream.write('完成\n')
        self.assertEqual(log.getvalue(),'中文开始\n完成\n')
        with patch('统一模型代码.gates.g5.r2.g5_r2_runtime.safe_print') as output:
            p = Progress('测试',100)
            for i in range(1,100):
                p.update(i)
            output.assert_not_called()
            p.update(100)
            output.assert_called_once()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'中文.json'
            write(path,{'内容':'频带—位置'})
            self.assertEqual(read(path),{'内容':'频带—位置'})

    def test_output_boundary_and_lock(self):
        with self.assertRaises(ValueError):
            Runtime(BASE.parent/'wrong')
        with RunLock():
            with self.assertRaises(RuntimeError):
                with RunLock():
                    pass

    def test_atomic_checkpoint_roundtrip(self):
        from 统一模型代码.gates.g5.e2e_g5_train import save_checkpoint,load_checkpoint
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'中文.pt'
            value = {'state':{'x':torch.arange(4)},'epoch':1}
            save_checkpoint(p,value)
            got = load_checkpoint(p)
            self.assertTrue(torch.equal(got['state']['x'],value['state']['x']))
            self.assertEqual(got['epoch'],1)
            p.write_bytes(b'bad')
            with self.assertRaises(RuntimeError):
                load_checkpoint(p)

    def test_paired_extension_and_completed_skip(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = SimpleNamespace(out=Path(folder))
            calls = []
            def fake_train(rt,seed,arm,end):
                calls.append((seed,arm,end))
                value = {'status':'PASS','extend':seed==20260921 and arm=='c2',
                         'completed_epoch':end}
                write(rt.out/f'training/{seed}/{arm}/report_epoch{end}.json',value)
                return value
            with patch('统一模型代码.gates.g5.r2.g5_r2_train.train_track',side_effect=fake_train):
                report = train_all(runtime)
                self.assertEqual(len(calls),9)
                self.assertEqual(sum(x[2]==20 for x in calls),6)
                self.assertEqual([x[1] for x in calls if x[2]==24],['c0','c1','c2'])
                self.assertEqual(report['status'],'PASS')
                calls.clear()
                train_all(runtime)
                self.assertEqual(calls,[])

    def test_failed_checkpoint_does_not_advance_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write(root/'completed.json',{'epoch':0})
            model = torch.nn.Linear(2,1)
            optimizer = torch.optim.AdamW(model.parameters())
            with patch('统一模型代码.gates.g5.r2.g5_r2_train.payload',return_value={'x':torch.ones(1)}), \
                 patch('统一模型代码.gates.g5.r2.g5_r2_train.rng_state',return_value={}), \
                 patch('统一模型代码.gates.g5.r2.g5_r2_train.save_checkpoint',side_effect=OSError('simulated interrupted save')):
                with self.assertRaises(OSError):
                    commit(root,None,None,optimizer,torch.Generator(),[{'epoch':1}],{},1)
            self.assertEqual(read(root/'completed.json'),{'epoch':0})

    def test_synthetic_comparison_report(self):
        from 统一模型代码.gates.g5.r2.g5_r2_evaluate import build_report
        rows = []
        for k in range(4):
            row = {'raw_index':k,'true_count':k,'predicted_count':k,'gospa_m':1.,
                   'matched_errors_m':[1.]*k,'spatial_band_f1':[1.]*k,'joint_tp':k,
                   'duplicate30':False,'min_source_distance_m':200. if k>1 else None,
                   'snr_db':0.,'band_only_f1':[1.]*k,'band_only_iou':[1.]*k,
                   'ceiling':{'truth_with_candidate100':k,'feasible_location_tp100':k,'feasible_joint_tp100':k}}
            for t in (10,30,50,100):
                row[f'tp_at_{t}m'] = k
            for component in ('localization','missed','false'):
                row[f'gospa_{component}_p_sum'] = 0.
            rows.append(row)
        results = {(s,a):rows for s in (20260921,20260922) for a in ('c0','c1','c2')}
        with tempfile.TemporaryDirectory() as folder:
            report = build_report(results,Path(folder),{'status':'SYNTHETIC_ONLY'})
            self.assertEqual(report['paired']['feedback_c2_minus_c1']['combined']['gospa_m']['mean'],0)
            self.assertEqual(report['tracks']['20260921_c0']['overall']['matched_rmse_m'],1)
            self.assertTrue((Path(folder)/'运行摘要.md').is_file())


if __name__=='__main__':
    unittest.main()
