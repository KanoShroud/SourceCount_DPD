"""工程等价/边界测试；不是G8性能证据。"""
import unittest
import numpy as np
import torch

from .physics import Spectrum, RECEIVERS, FS, grid, peaks
from .scenes import make_scene, specifications, layout
from DPD_MVDR.DPD_MVDR import DPD_MVDR
from 统一模型代码.gates.g6.coherent_dpd import geometry
from 统一模型代码.physics.fine_dpd_autograd import compute_fine_dpd_autograd


class PhysicsTests(unittest.TestCase):
    def test_hr_equivalence(self):
        rng = np.random.default_rng(0)
        iq = rng.normal(size=(4, 256))+1j*rng.normal(size=(4, 256))
        points, shape = grid(50, 50)
        for j in (2, 4, 8):
            for loading in (1e-4, 1e-2):
                _, expected, info = DPD_MVDR(RECEIVERS, iq, [0, 0], 50, 50, FS, FS, 0,
                    dict(J=j, DiagLoad=loading))
                calc = Spectrum(iq, 'hr', segments=j, loading=loading, device='cpu')
                actual = calc.evaluate(points).reshape(shape)
                np.testing.assert_allclose(actual, expected.T, rtol=2e-9, atol=1e-15)
                self.assertEqual(calc.info['N_fft'], info['N_fft'])

    def test_dpd_equivalence_and_length(self):
        rng = np.random.default_rng(1)
        for n in (256, 1024):
            iq = rng.normal(size=(4, n))+1j*rng.normal(size=(4, n))
            points, shape = grid(50, 50)
            geo = geometry(points, n_fft=n, shape=shape)
            mask = torch.arange(n) % 3 != 0
            expected = compute_fine_dpd_autograd(iq, geo, mask.double(), fixed_support=mask,
                checkpoint_mode='off', real_dtype=torch.float64).numpy()
            actual = Spectrum(iq, device='cpu').evaluate(points, mask=mask).reshape(shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-7)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_cpu(self):
        iq = make_scene(0, 'swapped', 4096, .5)['iq']
        points, _ = grid(50, 50)
        for method in ('dpd', 'hr'):
            a = Spectrum(iq, method, device='cpu').evaluate(points)
            b = Spectrum(iq, method, device='cuda').evaluate(points)
            np.testing.assert_allclose(a, b, rtol=2e-8, atol=1e-12)

    def test_peaks_distance(self):
        points = np.array([[0., 0.], [10., 0.], [40., 0.]])
        self.assertEqual(peaks(points, np.array([3., 2., 1.]), 2), [0, 2])


class SceneTests(unittest.TestCase):
    def test_all_layouts_feasible(self):
        for group in range(48):
            pos, gain, powers, _ = layout(group)
            self.assertLessEqual(abs(pos).max(), 1000)
            self.assertTrue((powers > 0).all() and (gain > 0).all())

    def test_pairing_and_prefix(self):
        short = make_scene(0, 'swapped', 4096, .5)
        long = make_scene(0, 'swapped', 16384, .5)
        np.testing.assert_array_equal(short['iq'], long['iq'][:, :4096])
        np.testing.assert_array_equal(short['components'], long['components'][..., :4096])
        np.testing.assert_array_equal(short['iq'], short['components'].sum(0)+short['noise'])
        same = make_scene(0, 'same', 4096, .5)
        np.testing.assert_array_equal(short['noise'], same['noise'])
        self.assertEqual(short['metadata']['positions'], same['metadata']['positions'])

    def test_control_and_roles(self):
        a = make_scene(16, 'same', 4096, .5, total_power_control=True)
        b = make_scene(16, 'swapped', 4096, .5, total_power_control=True)
        self.assertAlmostEqual(np.sum(a['metadata']['rx_power']), np.sum(b['metadata']['rx_power']))
        specs = list(specifications())
        self.assertEqual(len(specs), 400)
        self.assertEqual(sum(s['group'] < 16 for s in specs), 128)

    def test_truth_label_profile(self):
        scene = make_scene(0, 'same', 4096, .5)
        from .physics import LO, HI
        m = scene['metadata']
        center = np.asarray(m['frequency_centers_hz'])[:, None]
        width = np.asarray(m['bandwidth_hz'])[:, None]
        coverage = np.maximum(0, np.minimum(center+width/2, HI)-np.maximum(center-width/2, LO))/10e6
        np.testing.assert_array_equal(np.asarray(m['bands']), coverage >= .2)
        self.assertEqual(m['label_profile'], 'hard19_actual_t020')

    def test_invalid_duration_not_silently_cropped(self):
        with self.assertRaises(ValueError):
            make_scene(0, 'same', 65536, .5)


if __name__ == '__main__':
    unittest.main()
