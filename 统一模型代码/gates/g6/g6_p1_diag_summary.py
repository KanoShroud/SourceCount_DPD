"""Read-only decomposition of saved diagnostic predictions; writes new summary only."""
import argparse
import io
from pathlib import Path

import numpy as np
import torch

from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g6.g6_p1_runtime import BASE, SOURCE, SEEDS, read, write, identity


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('source',type=Path)
    root=parser.parse_args().source.resolve(strict=True)
    if not root.is_relative_to((BASE/'diagnosis').resolve()) or (root/'decomposition.json').exists():
        raise RuntimeError('Require existing diagnosis and a fresh summary output')
    final=read(root/'final_audit.json')
    verified_read(final['report'],root/'anomalies')
    report=read(root/'report.json')
    contract=read(BASE/'contract.json')
    fm_row=next(r for r in contract['files'] if Path(r['path']).resolve()==(SOURCE/'feature_manifest.json').resolve())
    import json
    fm=json.loads(verified_read(fm_row,root/'anomalies'))
    target_row=fm['files']['val_select']['targets']
    targets=torch.load(io.BytesIO(verified_read(target_row,root/'anomalies')),map_location='cpu',weights_only=False)
    result={'status':'PASS','target_identity':target_row,'source_report':final['report'],
            'code':identity(__file__),'files':[],'seeds':{},'training_executed':False,'test_executed':False}
    def rows(name):
        p=root/f'{name}_samples.json'
        row=identity(p); result['files'].append(row)
        return json.loads(verified_read(row,root/'anomalies'))
    for seed in SEEDS:
        early,late,decode=[rows(f'{seed}_{tag}') for tag in ('c2_e4','c2_e20','decode_early')]
        true=sum(x['true_count'] for x in late)
        np.testing.assert_allclose(sum(x['joint_tp'] for x in late)/true,
                                   report['tracks'][f'{seed}_c2_e20']['joint_recall100_f1_08'])
        stable=[]
        for a,b in zip(late,decode):
            assert a['index']==b['index']
            active_a=np.max(a['logits'],axis=1)>=0
            active_b=np.max(b['logits'],axis=1)>=0
            if np.array_equal(active_a,active_b):
                assert a['predicted_positions_m']==b['predicted_positions_m']
                stable.append((a,b))
        ntrue=sum(a['true_count'] for a,b in stable)
        gain_all=sum(b['joint_tp']-a['joint_tp'] for a,b in zip(late,decode))
        gain_stable=sum(b['joint_tp']-a['joint_tp'] for a,b in stable)
        changes={'shared':{'positive_bins':0,'removed_bins':0},'exclusive':{'positive_bins':0,'removed_bins':0}}
        for a,b in zip(early,late):
            assert a['index']==b['index']
            i=a['index']; k=a['true_count']
            inv_a={t:int(q) for q,t in a['mapping'].items()}
            inv_b={t:int(q) for q,t in b['mapping'].items()}
            bands=targets['band'][i,:k].numpy()>.5
            ignore=targets['ignore'][i,:k].numpy()>=.5
            for t in range(k):
                positive=bands[t]&~ignore[t]
                other=np.any(np.delete(bands&~ignore,t,axis=0),axis=0)
                removed=(np.asarray(a['logits'][inv_a[t]])>=0)&(np.asarray(b['logits'][inv_b[t]])<0)
                for group,mask in [('shared',positive&other),('exclusive',positive&~other)]:
                    changes[group]['positive_bins']+=int(mask.sum())
                    changes[group]['removed_bins']+=int((mask&removed).sum())
        result['seeds'][str(seed)]={'same_active_and_positions_scenes':len(stable),
            'same_active_true_sources':ntrue,'late_joint_hits_same_active':sum(a['joint_tp'] for a,b in stable),
            'decode_joint_hits_same_active':sum(b['joint_tp'] for a,b in stable),
            'joint_gain_all_sources':gain_all,'joint_gain_same_active':gain_stable,
            'removed_true_subbands':changes}
    verified_read(final['report'],root/'anomalies')
    write(root/'decomposition.json',result)
    print(json.dumps(result['seeds'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
