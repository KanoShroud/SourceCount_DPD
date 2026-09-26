"""Synthetic final-report integration; no training, real dataset or test access."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from 统一模型代码.gates.g7 import formal_run as run
from 统一模型代码.gates.g7 import formal_hard as hard
from 统一模型代码.gates.g7 import formal_latency as latency


class FormalReportTests(unittest.TestCase):
    def test_complete_report_and_zero_source_stratum(self):
        rows=[]
        for k in range(4):
            for _ in range(8):
                i=len(rows)
                band=torch.zeros(3,19);band[:k,:2]=1
                positions=torch.tensor([[0.,0.],[100.,200.],[300.,500.]])
                logits=torch.full((10,19),-4.);logits[:k,:2]=4.
                record=dict(count=k,band=band,ignore=torch.zeros_like(band),positions=positions,
                    index=i,metadata=dict(raw_index=i,true_k=k))
                row=hard.spatial_row(record,positions[:k].numpy(),logits,list(range(k)),.2)
                row.update(joint_tp=k,spatial_band_f1=[1.]*k)
                rows.append(row)
        def write(path,value):
            path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')
        class Model:
            def restore(self,state): pass
        with TemporaryDirectory() as folder:
            rt=SimpleNamespace(out=Path(folder),config=dict(seeds=[17,18],bootstrap_repeats=10),
                manifest=dict(subsets={'val_compare':[dict(true_k=r['true_count']) for r in rows]}),
                prepare_local=lambda *args:None,best=lambda seed,phase:dict(seed=seed,phase=phase),
                model=lambda *args:Model())
            with patch.object(run,'write',write),patch.object(run,'load',return_value={'state':{}}),\
                 patch.object(run,'evaluate',return_value=(run.summarize(rows),rows)),\
                 patch.object(hard,'evaluate_hard',return_value=(hard.summarize_hard(rows),rows)),\
                 patch.object(latency,'profile_online',side_effect=lambda rt,seed,phase,ids:
                     dict(indices=ids,mean_seconds=.5,p95_seconds=.5)):
                report=run.final_comparison(rt,{})
            self.assertEqual(report['status'],'COMPLETED_FORMAL_DEVELOPMENT')
            self.assertEqual(len(report['tracks']),10)
            self.assertIsNone(report['tracks']['17_hard']['joint_recall100_f1_08'])
            self.assertIsNone(report['by_k']['17_e']['K0']['matched_rmse_m'])
            self.assertEqual(report['paired']['feedback_e_minus_s']['gospa_m']['mean'],0.)
            self.assertEqual(len(report['online_timing']['18_e']['indices']),32)
            self.assertIn('回读',(rt.out/'evaluation/运行摘要.md').read_text(encoding='utf-8'))


if __name__=='__main__':unittest.main()
