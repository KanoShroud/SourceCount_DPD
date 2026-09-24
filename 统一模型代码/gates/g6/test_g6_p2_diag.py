import unittest

import numpy as np
import torch

from 统一模型代码.gates.g6.g6_p2_diagnose import coarse_constant, selected_points, local_changes
from 统一模型代码.gates.g5.r2.g5_r2_model import candidate_batch, decode_r2


class Tests(unittest.TestCase):
    def test_coordinate_centers_and_blocks(self):
        small=torch.arange(81*81).reshape(1,1,81,81)
        out=coarse_constant(small)
        self.assertEqual(tuple(out.shape),(1,1,401,401))
        self.assertTrue(torch.equal(out[...,::5,::5],small))
        self.assertEqual(int(out[0,0,0,2]),0)
        self.assertEqual(int(out[0,0,0,3]),1)
        self.assertEqual(int(out[0,0,400,400]),6560)

    def test_decode_grid_offset_reconstruction(self):
        h=torch.full((1,3,401,401),-20.); o=torch.zeros(1,3,2,401,401)
        for q,x in enumerate((100,200,300)):
            h[0,q,200,x]=10.; o[0,q,:,200,x]=torch.tensor([.3,-.2])
        cand=candidate_batch(h,o)[0]
        for k in (0,1,2,3):
            z=np.full((3,19),-10.); z[:k]=10.
            dec=decode_r2(z,cand,None)
            selected=selected_points(dec,h.sigmoid()[0].numpy(),o[0].numpy())
            self.assertEqual(len(selected),k)
            for p in selected.values():
                np.testing.assert_allclose(np.array(p['position'])-p['grid'],[3,-2],atol=1e-4)

    def test_local_reversal_tracks_native_slot(self):
        h=np.zeros((3,401,401)); r=h.copy()
        h[0,0,0]=2; h[0,0,1]=1; r[0,0,1]=2
        native={'truth':[[0,0]],'sources':{'0':{'slot':0,'flat':1,'grid':[10,0],'position':[10,0],'error':10,'coarse_boundary':False}}}
        other={'selected':{'0':{'flat':0,'grid':[0,0],'position':[0,0],'coarse_boundary':False}}}
        self.assertTrue(local_changes(native,other,h,r)[0]['local_order_reversed'])


if __name__=='__main__':
    unittest.main()
