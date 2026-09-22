"""Small synthetic checks; no research inputs."""
import unittest
import numpy as np
import torch
from 统一模型代码.gates.g5.r3.g5_r3 import combinations, loss_terms, cached_scores
from 统一模型代码.gates.g5.r2.g5_r2_model import AssociationHead, decode_r2


class R3Tests(unittest.TestCase):
    def fixture(self):
        logits=np.full((3,19),-5.,dtype=np.float32)
        logits[0,0]=5
        logits[1,1]=5
        truth=np.asarray([[0.,0.],[500.,0.]],dtype=np.float32)
        bands=np.zeros((3,19),dtype=np.float32)
        bands[0,0]=1
        bands[1,1]=1
        record={'candidates':[{'query':q,'scores':[.8,.7], 'positions':truth.tolist()} for q in range(3)],
                'original':[[0.,0.]]*3}
        return logits,truth,bands,record

    def test_joint_labels_and_permutation(self):
        logits,truth,bands,record=self.fixture()
        ranks,heat,pos=combinations(record,logits,truth,bands,np.zeros_like(bands))
        self.assertEqual(ranks.tolist(),[[0,1],[1,0]])
        self.assertEqual(pos.tolist(),[True,False])
        other=combinations(record,logits,truth[::-1].copy(),bands[[1,0,2]],np.zeros_like(bands))
        np.testing.assert_array_equal(pos,other[2])
        sample={'ranks':torch.tensor(ranks),'heat_sum':torch.tensor(heat),
                'positive':torch.tensor(pos),'active':[0,1]}
        scores=torch.zeros((3,2),requires_grad=True)
        loss=loss_terms(scores,sample,'joint')[0]
        loss.backward()
        self.assertLess(float(scores.grad[0,0]),0)
        self.assertGreater(float(scores.grad[0,1]),0)
        sample['positive']=torch.ones(2,dtype=torch.bool)
        self.assertEqual(loss_terms(scores,sample,'joint'),[])

    def test_zero_and_project_gradient(self):
        logits,_,_,record=self.fixture()
        head=AssociationHead()
        sample={'local':torch.randn(3,2,19,128),'query':torch.randn(3,128),'logits':torch.tensor(logits)}
        scores=cached_scores(head,sample,'cpu')
        self.assertEqual(decode_r2(logits,record)['joint'],decode_r2(logits,record,list(scores))['joint'])
        with torch.no_grad():
            head.mlp[-1].weight.fill_(.01)
        cached_scores(head,sample,'cpu').sum().backward()
        self.assertGreater(float(head.project.weight.grad.abs().sum()),0)

    def test_no_sources(self):
        logits,_,bands,record=self.fixture()
        result=combinations(record,logits,np.empty((0,2),dtype=np.float32),bands,np.zeros_like(bands))
        self.assertTrue(result[2].all())
        logits[:]=-5
        self.assertEqual(len(combinations(record,logits,np.empty((0,2)),bands,bands)[0]),0)


if __name__=='__main__':
    unittest.main()
