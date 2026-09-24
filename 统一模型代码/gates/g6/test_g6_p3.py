"""Small exact checks; no project data or model checkpoints consumed."""
import itertools
import unittest

import numpy as np

from 统一模型代码.gates.g5.r1.g5_r1_decode import maximum_valid_pairs
from 统一模型代码.gates.g6.g6_p3_candidates import (
    THRESHOLDS, ceilings, decode, matching_counts, missing_ranks, rescore_base)


def record(points, scores=None):
    candidates=[]
    for q,p in enumerate(points):
        s=scores[q] if scores is not None else [.9]*len(p)
        candidates.append(dict(query=q,positions=p,scores=s,flat_indices=list(range(len(p)))))
    return dict(candidates=candidates,original=[p[0] if p else [0,0] for p in points])


class CandidateTests(unittest.TestCase):
    def test_matching_exact(self):
        rng=np.random.default_rng(42)
        for k in range(4):
            for p in range(4):
                valid=rng.random((128,k,p))>.5
                np.testing.assert_array_equal(matching_counts(valid),[maximum_valid_pairs(x) for x in valid])

    def test_union_is_one_to_one_and_deduplicated(self):
        rec=record([[[0,0]],[[0,0]],[[0,0]]])
        out=ceilings(np.array([[0,0],[5,0]]),rec,np.ones((3,19)),np.ones((2,19)),np.zeros((2,19)))
        self.assertEqual(out['union']['10'],1)
        self.assertEqual(out['feasible_combinations'],0)

    def test_empty_truth_and_k0(self):
        rec=record([[[0,0]],[[100,0]],[[200,0]]])
        z=-np.ones((3,19))
        out=ceilings(np.empty((0,2)),rec,z,np.empty((0,19)),np.empty((0,19)))
        self.assertEqual(out['joint100'],0)
        self.assertEqual(decode(z,rec)['joint'],[])

    def test_exact_ceiling_against_bruteforce(self):
        rng=np.random.default_rng(54)
        for _ in range(15):
            points=rng.uniform(-100,100,(3,3,2)).tolist()
            rec=record(points); truth=rng.uniform(-100,100,(3,2))
            z=np.ones((3,19)); bands=np.ones((3,19)); ignore=np.zeros((3,19))
            out=ceilings(truth,rec,z,bands,ignore)
            brute={str(d):0 for d in THRESHOLDS}
            for ranks in itertools.product(range(3),repeat=3):
                p=np.array([points[q][r] for q,r in enumerate(ranks)])
                if any(np.linalg.norm(a-b)<30 for a,b in itertools.combinations(p,2)):
                    continue
                dist=np.linalg.norm(truth[:,None]-p[None],axis=-1)
                for d in THRESHOLDS:
                    brute[str(d)]=max(brute[str(d)],maximum_valid_pairs(dist<=d))
            self.assertEqual(out['feasible'],brute)
            self.assertEqual(out['joint100'],brute['100'])

    def test_log_probability_not_sum_logits(self):
        rec=record([[[0,0],[100,0]],[[0,0],[100,0]],[[200,0]]],[[.99,.8],[.8,.5],[.1]])
        z=np.ones((3,19));z[2]=-1
        self.assertEqual(decode(z,rec)['joint'],[[100.,0.],[0.,0.]])

    def test_fixed_coordinates_single_and_fallback(self):
        rec=record([[[0,0],[10,0]],[[0,1],[10,1]],[[200,0]]])
        p=np.array([[[.1,.9]],[[.8,.1]],[[.1,.2]]])
        changed=rescore_base(rec,p)
        self.assertEqual(rec['candidates'][0]['scores'],[.9,.9])
        self.assertEqual(changed['candidates'][0]['positions'],rec['candidates'][0]['positions'])
        z=-np.ones((3,19));z[0]=1
        self.assertEqual(decode(z,changed)['joint'],[[10.,0.]])
        self.assertEqual(decode(z,changed)['selected_ranks'],[2])
        z[1]=1
        out=decode(z,changed)
        self.assertTrue(out['fallback'])
        self.assertEqual(out['joint'],[[10.,0.],[0.,1.]])

    def test_joint_identity_limit(self):
        rec=record([[[0,0]],[[200,0]],[[400,0]]])
        z=np.ones((3,19));z[2]=-1
        bands=np.ones((2,19));bands[1]=0
        out=ceilings(np.array([[0,0],[200,0]]),rec,z,bands,np.zeros((2,19)))
        self.assertEqual(out['feasible']['100'],2)
        self.assertEqual(out['joint100'],1)

    def test_missing_ninth_peak(self):
        probability=np.zeros((3,401,401));offset=np.zeros((3,2,401,401))
        points=[]
        for n in range(9):
            x=20+n*10;probability[0,100,x]=.99-n*.01
            points.append([x*10-2000,-1000])
        rec=record([points[:8],[],[]])
        out=missing_ranks(np.array([points[8]]),rec,probability,offset)
        self.assertEqual(next(r['rank'] for r in out if r['threshold']==10),9)


if __name__=='__main__':
    unittest.main()
