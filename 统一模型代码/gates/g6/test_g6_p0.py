"""Synthetic tests only: no project datasets or historical results."""
import unittest

import numpy as np
import torch

from 统一模型代码.gates.g6.coherent_dpd import (
    AtomicDPD, band_assignments, geometry, grid_points, identity_scores,
    normalize_maps, sample_map, spatial_evd,
)
from 统一模型代码.gates.g6.g6_p0 import overlap_category, select_scenes


class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.lo, self.hi = [-32., -16., 0.], [0., 16., 32.]
        self.geo = geometry([[-100., 50.], [50., 200.], [100., -250.]], n_fft=64, fs=64.)
        self.physics = AtomicDPD(self.lo, self.hi, self.geo)
        rng = np.random.default_rng(123)
        self.signal = rng.normal(size=(4, 64))+1j*rng.normal(size=(4, 64))

    def test_intervals_union_and_slot_separation(self):
        p = torch.tensor([[1., 0., 0.], [0., 0., 1.]], dtype=torch.float64)
        w = self.physics.weights(p)
        torch.testing.assert_close(w, torch.tensor([[1., 1., 0., 0.], [0., 0., 1., 1.]], dtype=torch.float64))
        soft = self.physics.weights(torch.tensor([.2, .7, .5], dtype=torch.float64))
        torch.testing.assert_close(soft, torch.tensor([.2, .76, .85, .5], dtype=torch.float64))

    def test_bounded_phase_matches_direct(self):
        reference = torch.stack([torch.exp(2j*np.pi*self.geo.dtaus[:, p, None]*self.geo.f_full[None])
                                 for p in range(6)])
        torch.testing.assert_close(self.physics.phase, reference, atol=0, rtol=0)

    def test_chunked_evd_forward_and_gradient(self):
        generator = torch.Generator().manual_seed(20260922)
        a = torch.randn(3, 130, 4, 4, dtype=torch.complex128, generator=generator, requires_grad=True)
        matrix = a @ a.mH + .01*torch.eye(4)
        old = torch.linalg.eigvalsh(matrix).abs().amax(-1)
        new = spatial_evd(matrix)
        g_old = torch.autograd.grad(old.square().mean(), a, retain_graph=True)[0]
        g_new = torch.autograd.grad(new.square().mean(), a)[0]
        torch.testing.assert_close(new, old, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(g_new, g_old, atol=1e-12, rtol=1e-12)

    def test_cached_direct_precision_and_zero(self):
        stats = self.physics.statistics(self.signal)
        for weights in [torch.zeros(4), torch.tensor([1., 0., 0., 0.]),
                        torch.tensor([.1, .2, .7, .8]), torch.ones(4)]:
            expected = self.physics.direct(self.signal, weights.double())
            torch.testing.assert_close(self.physics.evaluate(stats, weights)[0], expected, atol=1e-9, rtol=1e-12)

    def test_probability_gradient(self):
        stats = self.physics.statistics(self.signal)
        p = torch.tensor([.2, .7, .5], dtype=torch.float64, requires_grad=True)
        def f(x, direct=False):
            w = self.physics.weights(x)
            y = self.physics.direct(self.signal, w) if direct else self.physics.evaluate(stats, w)
            return y.log1p().mean()
        cached = torch.autograd.grad(f(p), p)[0]
        direct = torch.autograd.grad(f(p, True), p)[0]
        torch.testing.assert_close(cached, direct, atol=1e-10, rtol=1e-7)
        finite = (f(p+1e-4)-f(p-1e-4))/(2e-4)
        torch.testing.assert_close(cached.sum(), finite, atol=1e-8, rtol=1e-3)

    def test_sampling_coordinates_and_normalization(self):
        xy = grid_points()
        maps = (xy[:, 0]+2*xy[:, 1]).reshape(1, 81, 81)
        value = sample_map(maps, [[75., -125.], [2000., 2000.]])
        torch.testing.assert_close(value, torch.tensor([[-175., 6000.]], dtype=torch.float64))
        norm, mean, std = normalize_maps(torch.ones(2, 81, 81))
        self.assertTrue(torch.isfinite(norm).all())
        self.assertLess(float(norm.abs().max()), .2)
        self.assertEqual(mean.shape, (2, 1, 1))
        self.assertTrue((std > 0).all())

    def test_joint_beats_single_row_rule(self):
        truth = np.eye(2)
        perms, correct, _ = band_assignments(truth, truth, np.zeros_like(truth))
        result = identity_scores([[10., 3.], [9., 8.]], perms, correct)
        self.assertEqual(result['accuracy'], 1.)
        self.assertLess(result['pair_positive_fraction'], 1.)
        tied = identity_scores(np.ones((2, 2)), perms, correct)
        self.assertEqual(tied['accuracy'], tied['chance'])

    def test_same_band_equivalence_and_ignore(self):
        truth = np.ones((3, 3))
        perms, correct, _ = band_assignments(truth, truth, np.zeros_like(truth))
        self.assertEqual(len(perms), 6)
        self.assertTrue(correct.all())
        self.assertFalse(identity_scores(np.eye(3), perms, correct)['informative'])
        truth = np.array([[1., 0.], [0., 1.]])
        _, _, cost = band_assignments([[.8, .01], [.2, .99]], truth, [[0, 1], [0, 0]])
        self.assertAlmostEqual(cost[0, 0], -np.log(.8))
        self.assertEqual(overlap_category(truth), 'distinct')

    def test_selection_fixed_and_balanced(self):
        records, labels, predictions = [], {}, {}
        for k in range(4):
            for i in range(128):
                rid = k*128+i
                records.append({'raw_index': rid, 'true_k': k})
                labels[rid] = {'overlap': ['distinct', 'partial', 'identical'][i % 3]}
                predictions[(20260921, rid)] = {'metrics': [{'snr_db': float(i % 30-15)}]}
        a = select_scenes(records, labels, predictions)
        self.assertEqual(a, select_scenes(records, labels, predictions))
        self.assertEqual([sum(r['true_k'] == k for r in a) for k in range(4)], [8, 8, 56, 56])
        self.assertEqual(len({r['raw_index'] for r in a}), 128)


if __name__ == '__main__':
    unittest.main()
