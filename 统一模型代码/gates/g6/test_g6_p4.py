"""Exact small P4 checks without project datasets or checkpoints."""
import itertools
import unittest

import numpy as np

from 统一模型代码.gates.g5.r1.g5_r1_decode import maximum_valid_pairs
from 统一模型代码.gates.g6.g6_p3_candidates import ceilings, decode
from 统一模型代码.gates.g6.g6_p4_decode import MODES, decode_shared, prepare, upper_bounds


def example():
    # Different donor coordinates at the same grid index model different offsets.
    record = dict(candidates=[dict(query=q,flat_indices=[q],positions=[[q*100.,0.]],scores=[.5])
                              for q in range(3)],original=[[q*100.,0.] for q in range(3)])
    logits = np.ones((3,19)); logits[2] = -1
    probability = np.array([[.5,.9,.2],[.8,.5,.3],[.1,.1,.5]])
    residual = np.array([[0.,2.,0.],[3.,0.,0.],[100.,100.,100.]])
    return logits,record,probability,residual


class SharedTests(unittest.TestCase):
    def test_native_exact(self):
        z,r,p,v = example(); prepared = prepare(z,r,p,v)
        actual = decode_shared(z,r,prepared,'native')
        expected = decode(z,r)
        for field in expected:
            self.assertEqual(actual[field],expected[field])

    def test_cross_slot_preserves_donor_coordinates(self):
        z,r,p,v = example(); prepared = prepare(z,r,p,v)
        out = decode_shared(z,r,prepared,'shared_final')
        self.assertEqual(out['joint'],[[100.,0.],[0.,0.]])
        self.assertEqual(out['cross_slot_count'],2)
        self.assertEqual(prepared['pool'][out['selected_pool'][0]]['donor'],1)

    def test_k_zero_one_unchanged(self):
        z,r,p,v = example()
        for k in (0,1):
            z[:] = -1; z[:k] = 1
            prepared = prepare(z,r,p,v)
            for mode in MODES:
                self.assertEqual(decode_shared(z,r,prepared,mode)['joint'],decode(z,r)['joint'])

    def test_duplicate_and_separation(self):
        z,r,p,v = example()
        r['candidates'][1]['positions'] = [[0.,0.]]
        prepared = prepare(z,r,p,v)
        for combo in prepared['combos']:
            self.assertGreaterEqual(np.linalg.norm(prepared['points'][combo[0]]-prepared['points'][combo[1]]),30)
        r['candidates'][2]['positions'] = [[1.,0.]]
        prepared = prepare(z,r,p,v)
        out = decode_shared(z,r,prepared,'shared_factorized')
        self.assertTrue(out['fallback']); self.assertTrue(out['shared_fallback'])
        self.assertEqual(out['joint'],decode(z,r)['joint'])

    def test_active_only_softmax_and_numerical_stability(self):
        z,r,p,v = example(); v *= 1000
        prepared = prepare(z,r,p,v)
        score = prepared['scores']['shared_factorized']
        self.assertTrue(np.isfinite(score).all())
        v[2] *= -100
        np.testing.assert_array_equal(score,prepare(z,r,p,v)['scores']['shared_factorized'])

    def test_factorized_binding_is_residual_binding(self):
        z,r,p,v = example(); prepared = prepare(z,r,p,v)
        score = prepared['scores']['shared_factorized']
        difference = score[0,0]+score[1,1]-score[0,1]-score[1,0]
        self.assertAlmostEqual(difference,v[0,0]+v[1,1]-v[0,1]-v[1,0])

    def test_shared_score_bruteforce_and_stable_ties(self):
        z,r,p,v = example(); prepared = prepare(z,r,p,v)
        for mode in MODES[1:]:
            best = max(itertools.permutations(range(3),2),key=lambda ids:sum(prepared['scores'][mode][j,c] for j,c in enumerate(ids)))
            self.assertEqual(decode_shared(z,r,prepared,mode)['selected_pool'],list(best))
        p[:] = .5; v[:] = 0
        prepared = prepare(z,r,p,v)
        self.assertEqual(decode_shared(z,r,prepared,'shared_final')['selected_pool'],[0,1])

    def test_nested_bounds_and_joint_bruteforce(self):
        rng = np.random.default_rng(63)
        for k in range(4):
            z,r,p,v = example(); z[:] = -1; z[:k] = 1
            truth = np.array([[0.,0.],[100.,0.],[200.,0.]])
            bands = rng.integers(0,2,(3,19)); ignore = np.zeros_like(bands)
            native = ceilings(truth,r,z,bands,ignore)
            prepared = prepare(z,r,p,v)
            upper = upper_bounds(truth,z,bands,ignore,prepared,native)
            for d in (10,30,50,100):
                best = 0
                for combo in prepared['combos']:
                    points = prepared['points'][combo]
                    best = max(best,maximum_valid_pairs(np.linalg.norm(truth[:,None]-points[None],axis=-1)<=d))
                self.assertEqual(upper['shared_separated'][str(d)],best)

    def test_empty_truth(self):
        z,r,p,v = example(); z[:] = -1
        truth = np.empty((0,2)); bands = np.empty((0,19)); ignore = bands.copy()
        native = ceilings(truth,r,z,bands,ignore)
        upper = upper_bounds(truth,z,bands,ignore,prepare(z,r,p,v),native)
        self.assertEqual(upper['shared_joint100'],0)
        np.testing.assert_array_equal(truth,np.asarray(truth.tolist()).reshape(-1,2))


if __name__ == '__main__':
    unittest.main()
