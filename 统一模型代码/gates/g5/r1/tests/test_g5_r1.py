"""G5-R1边界、源身份、统计与中文日志轻量检查；不读取科研数据。"""
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT))
import numpy as np  # noqa: E402
from 统一模型代码.gates.g5.r1.g5_r1_decode import decode, choose_joint, association, candidate_ceiling  # noqa: E402
from 统一模型代码.gates.g5.r1.g5_r1_report import paired_many, summarize, build_report, plot_cases  # noqa: E402


class DecodeTests(unittest.TestCase):
    def arrays(self):
        return np.full((3,19),-1,dtype=np.float32),np.zeros((3,401,401),dtype=np.float32),np.zeros((3,2,401,401),dtype=np.float32)

    def test_empty_single_ties(self):
        logits,heat,offset=self.arrays()
        self.assertEqual(decode(logits,heat,offset)['joint'],[])
        logits[0,0]=1
        heat[0,200,200]=.9
        heat[0,200,240]=.9
        result=decode(logits,heat,offset)
        self.assertEqual(result['original'],result['joint'])
        self.assertEqual(result['candidates'][0]['flat_indices'][0],200*401+200)
        self.assertEqual(len(result['candidates'][0]['scores']),2)

    def test_conflict_fallback_and_boundary(self):
        c=[{'scores':[.9],'positions':[[0,0]]},{'scores':[.8],'positions':[[0,0]]}]
        original=np.zeros((2,2))
        self.assertTrue(choose_joint(c,original)[2])
        c[1]['positions']=[[30,0]]
        self.assertFalse(choose_joint(c,original)[2])
        c[1]={'scores':[.9,.8],'positions':[[0,0],[100,0]]}
        points,ranks,fallback=choose_joint(c,original)
        self.assertEqual(ranks,[1,2])
        self.assertFalse(fallback)
        self.assertEqual(points.tolist(),[[0,0],[100,0]])

    def test_identity_binding_and_ceiling(self):
        truth=np.array([[0,0],[1000,0]])
        bands=np.zeros((2,19));bands[0,0]=1;bands[1,1]=1
        logits=np.full((3,19),-1.);logits[0,0]=1;logits[1,1]=1
        ignore=np.zeros_like(bands)
        self.assertEqual(association(truth,truth,logits,[0,1],bands,ignore)['joint_tp'],2)
        self.assertEqual(association(truth,truth[::-1],logits,[0,1],bands,ignore)['joint_tp'],0)
        c=[{'query':0,'scores':[.9,.8],'positions':[[0,0],[1000,0]]},
           {'query':1,'scores':[.9,.8],'positions':[[0,0],[1000,0]]}]
        upper=candidate_ceiling(truth,c,logits,bands,ignore)
        self.assertEqual(upper['feasible_joint_tp100'],2)
        self.assertEqual(association(truth,np.zeros((0,2)),logits,[],bands,ignore)['joint_tp'],0)

    def test_reports_and_seed_pairing(self):
        def rows(delta):
            return [dict(raw_index=i,true_count=k,predicted_count=k,gospa_m=10.+delta,
                         matched_errors_m=[10.+delta]*k,joint_tp=k,tp_at_100m=k,
                         tp_at_10m=k,tp_at_30m=k,tp_at_50m=k,
                         spatial_band_f1=[1.]*k,spatial_pairs=[],duplicate30=False,
                         min_source_distance_m=200 if k>1 else None,snr_db=5,
                         gospa_localization_p_sum=1.,gospa_missed_p_sum=0.,gospa_false_p_sum=0.)
                    for i,k in enumerate([0,1,2,3])]
        a,b=rows(0),rows(-2)
        report=paired_many([(a,b),(a,b),(a,b)],repeats=30)
        self.assertEqual(report['gospa_m']['mean'],-2)
        self.assertEqual(report['gospa_m']['ci95'],[-2.,-2.])
        self.assertIsNone(summarize([a[0]])['matched_rmse_m'])
        records=[{'original':x,'joint':y,'truth':[[0,0]]*x['true_count'],
                  'decode':{'original':[[0,0]]*x['true_count'],'joint':[[0,0]]*x['true_count'],
                            'active':list(range(x['true_count'])),'fallback':False},
                  'ceiling':dict(truth_with_candidate100=x['true_count'],feasible_location_tp100=x['true_count'],feasible_joint_tp100=x['true_count'])}
                 for x,y in zip(a,b)]
        results={(s,t):records for s in [20260921,20260922,20260923] for t in ['sg','e2e']}
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            build_report(results,out)
            plot_cases(results,out)
            self.assertTrue((out/'运行摘要.md').exists())
            self.assertEqual(json.loads((out/'comparison_report.json').read_text(encoding='utf-8'))['status'],'COMPLETE_FOR_REVIEW')

    def test_utf8(self):
        env={**os.environ,'PYTHONUTF8':'1','PYTHONIOENCODING':'utf-8:replace'}
        result=subprocess.run([sys.executable,'-c',"print('中文路径：联合选峰✓')"],env=env,
                              capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
        self.assertIn('中文路径：联合选峰✓',result.stdout)


if __name__=='__main__':
    unittest.main()
