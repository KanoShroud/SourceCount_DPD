"""CPU invariants for raw ROI crops, coordinates and dynamic candidate counts."""
import unittest
import numpy as np
import torch

from 统一模型代码.gates.g7.compact_data import crop_maps, crop_statistics, network_inputs
from 统一模型代码.gates.g7.compact_model import candidates, windows, baseline_loss, decode


class CompactTests(unittest.TestCase):
    def test_crop_raw_and_renormalize(self):
        coarse = np.arange(19*81*81, dtype=np.float32).reshape(19, 81, 81)
        fine = np.arange(401**2, dtype=np.float32).reshape(401, 401)
        c, f = crop_maps(coarse, fine)
        self.assertEqual(c.shape, (19, 41, 41))
        self.assertEqual(f.shape, (201, 201))
        self.assertEqual(c[0, 0, 0], coarse[0, 20, 20])
        self.assertEqual(f[-1, -1], fine[300, 300])
        a, b = network_inputs(c, f)
        self.assertLess(abs(float(a.mean())), 1e-5)
        self.assertLess(abs(float(b.mean())), 1e-5)

    def test_statistics_order(self):
        raw = torch.arange(81**2, dtype=torch.float64)[None, :, None].expand(20, -1, 6).to(torch.complex128)
        cropped = crop_statistics(dict(energy=torch.ones(20, 4), coherent=raw))
        self.assertEqual(cropped['coherent'].shape, (20, 41**2, 6))
        self.assertEqual(cropped['coherent'][0, 0, 0], raw[0, 20*81+20, 0])
        self.assertEqual(cropped['coherent'][0, -1, 0], raw[0, 60*81+60, 0])

    def test_candidates_and_boundary(self):
        heat = torch.full((1, 3, 201, 201), -90.)
        for j in range(8):
            heat[0, :, 5*j, 7*j] = 8-j
        rec = candidates(heat, 5)[0]
        centers, valid, points, inv = windows(rec, 5)
        self.assertEqual(centers.shape, (3, 5, 2))
        self.assertEqual(inv.shape, (3, 5, 41, 41))
        self.assertEqual(centers[0, 0].tolist(), [-1000., -1000.])
        self.assertEqual(int(valid[0, 0].sum()), 21**2)
        self.assertLessEqual(float(points.abs().max()), 1000.)
        torch.testing.assert_close(points[inv[0, 0, 20, 20]], centers[0, 0])

    def test_global_decode_offset(self):
        heat = torch.full((1, 3, 201, 201), -30.)
        heat[0, 0, 100, 110] = 15
        logits = torch.full((1, 3, 19), -20.)
        logits[0, 0] = 20
        offset = torch.zeros(1, 3, 2, 201, 201)
        offset[0, 0, :, 100, 110] = torch.tensor([.8, -.9])
        r = decode(dict(heat=heat, offset=offset, band_logits=logits), 'baseline')[0]
        np.testing.assert_allclose(r['positions_m'], [[108., -9.]], atol=1e-3)

    def test_loss_zero_and_edge_source(self):
        for count in (0, 1):
            out = dict(band_logits=torch.zeros(1, 3, 19, requires_grad=True),
                       heat=torch.zeros(1, 3, 201, 201, requires_grad=True),
                       offset=torch.zeros(1, 3, 2, 201, 201, requires_grad=True))
            record = dict(count=count, positions=torch.tensor([[1000., -1000.], [0., 0.], [0., 0.]]),
                          band=torch.ones(3, 19), ignore=torch.zeros(3, 19))
            value, _ = baseline_loss(out, [record])
            self.assertTrue(torch.isfinite(value))
            value.backward()
            self.assertTrue(torch.isfinite(out['heat'].grad).all())


if __name__ == '__main__':
    unittest.main()
