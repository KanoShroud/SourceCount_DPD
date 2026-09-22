"""已批准R2评分开关诊断：固定checkpoint/候选，不训练、不读test。"""
from __future__ import annotations

import gc
import io
import time
from pathlib import Path

import numpy as np
import torch

from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.common.g5_verified_io import verified_read
from 统一模型代码.gates.g5.r1.e2e_g5_r1 import Run as R1Run, identity
from 统一模型代码.gates.g5.r1.g5_r1_report import summarize
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import save_rows
from 统一模型代码.gates.g5.r2.g5_r2_model import forward_r2, decode_r2
from 统一模型代码.gates.g5.r2.g5_r2_runtime import (
    BASE, SEEDS, Runtime, Progress, read, write, setup_environment, safe_print, g4,
)
from 统一模型代码.gates.g5.r2.g5_r2_train import restore


def candidate_scores(logits, record, scores, truth, bands, ignore):
    """真值仅用于事后标注：该槽位频带和该位置能否联合命中任一真实源。"""
    rows = []
    for q in np.flatnonzero(logits.max(-1) >= 0):
        c = record['candidates'][q]
        positions = np.asarray(c['positions'])
        good = np.zeros(len(positions), dtype=bool)
        for t, point in enumerate(truth):
            valid = ignore[t] < .5
            predicted, target = logits[q, valid] >= 0, bands[t, valid] > .5
            denominator = int(predicted.sum() + target.sum())
            f1 = 2*int((predicted & target).sum())/max(denominator, 1)
            if f1 >= .8:
                good |= np.linalg.norm(positions-point, axis=1) <= 100
        heat = np.log(np.maximum(c['scores'], 1e-20))
        assoc = scores[q].detach().log_softmax(0).cpu().numpy().astype(float)
        row = {'query':int(q), 'positions':c['positions'], 'joint_good':good.tolist(),
               'heat_log':heat.tolist(), 'association_log_probability':assoc.tolist(),
               'combined_score':(heat+assoc).tolist()}
        if good.any() and (~good).any():
            row['margins'] = {k:float(v[good].max()-v[~good].max())
                              for k,v in [('heat',heat),('association',assoc),('combined',heat+assoc)]}
        rows.append(row)
    return rows


@torch.no_grad()
def run():
    setup_environment()
    start = time.time()
    out = BASE/'diagnostics'/f'score_toggle_{time.strftime("%Y%m%d_%H%M%S")}'
    out.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(out, deadline=start+3600)
    safe_print(f'评分开关诊断；输出：{out}')
    runtime.preflight()
    bundle = runtime.features('val_select')
    features, targets, metadata, index, _ = bundle
    assert len(targets.counts) == 512
    checkpoints = {}
    for seed in SEEDS:
        for arm in ('c1','c2'):
            root = BASE/f'run/training/{seed}/{arm}'
            for epoch in (0,2,20):
                paths = [root/'initial.pt'] if epoch == 0 else list((root/'checkpoints').glob(f'epoch{epoch:03d}_*/state.pt'))
                if len(paths) != 1:
                    raise RuntimeError(f'Ambiguous checkpoint: {seed}/{arm}/{epoch}')
                path = paths[0].resolve(strict=True)
                row = read(path.with_suffix('.identity.json'))
                if Path(row['path']).resolve(strict=True) != path:
                    raise RuntimeError('Checkpoint identity path mismatch')
                checkpoints[f'{seed}_{arm}_{epoch}'] = row
    write(out/'protocol.json', {'status':'REGISTERED','split':'val_select','samples':512,
          'epochs':[0,2,20],'seeds':list(SEEDS),'arms':['c1','c2'],
          'training':False,'test_executed':False,'single_forward_two_decodes':True,
          'source':identity(Path(__file__)), 'checkpoints':checkpoints})
    reports = {}
    for seed in SEEDS:
        for arm in ('c1','c2'):
            context, head, optimizer, parameters = runtime.context(seed,arm)
            del optimizer, parameters
            history = read(BASE/f'run/training/{seed}/{arm}/history.json')
            for epoch in (0,2,20):
                key = f'{seed}_{arm}_{epoch}'
                cp = torch.load(io.BytesIO(verified_read(checkpoints[key],out/'anomalies')),
                                map_location='cpu',weights_only=False)
                assert cp['epoch'] == epoch
                restore(context,head,cp['state'])
                del cp
                g4.set_mode(context,training=False)
                head.eval()
                rows, off_rows, on_rows = [], [], []
                progress = Progress(key,512,out/'progress.json')
                iterator = batches(features,list(torch.arange(512).split(4)),prefetch=False)
                try:
                    for ids,batch in iterator:
                        runtime.guard()
                        runtime.consumed(index,ids)
                        outputs,candidates,scores = forward_r2(context,head,batch,ids,torch.device('cuda:0'),arm)
                        logits = outputs[1].cpu().numpy()
                        for local,i in enumerate(ids.tolist()):
                            truth = targets.positions[i,:int(targets.counts[i])].numpy()
                            bands,ignore = targets.band[i].numpy(),targets.ignore[i].numpy()
                            decoded = {mode:decode_r2(logits[local],candidates[local],s)
                                       for mode,s in [('off',None),('on',scores[local])]}
                            metrics = {mode:R1Run.metric(None,truth,d['joint'],logits[local],d['active'],bands,ignore,metadata[i])
                                       for mode,d in decoded.items()}
                            if epoch == 0 and metrics['off'] != metrics['on']:
                                raise AssertionError('Initial zero scorer is not equivalent')
                            off_rows.append(metrics['off'])
                            on_rows.append(metrics['on'])
                            rows.append({'index':i,'truth':truth.tolist(),'band_logits':logits[local].tolist(),
                                         'off':metrics['off'],'on':metrics['on'],'decode':decoded,
                                         'score_details':candidate_scores(logits[local],candidates[local],scores[local],truth,bands,ignore)})
                        progress.update(len(rows))
                finally:
                    iterator.close()
                off,on = summarize(off_rows),summarize(on_rows)
                expected = next(r['validation'] for r in history if r['epoch']==epoch)
                for metric in ('joint_recall100_f1_08','gospa_m','matched_rmse_m'):
                    if not np.isclose(on[metric],expected[metric],rtol=0,atol=1e-5):
                        raise AssertionError(f'Historical validation mismatch {key}/{metric}')
                margins = [s['margins'] for r in rows if r['off']['predicted_count']>=2
                           for s in r['score_details'] if 'margins' in s]
                report = {'off':off,'on':on,'historical_validation_reproduced':True,
                          'joint_improved_scenes':sum(r['on']['joint_tp']>r['off']['joint_tp'] for r in rows),
                          'joint_worsened_scenes':sum(r['on']['joint_tp']<r['off']['joint_tp'] for r in rows),
                          'mixed_good_bad_slots':len(margins),
                          'score_margins':{k:{'mean':float(np.mean([m[k] for m in margins])),
                                             'positive_fraction':float(np.mean([m[k]>0 for m in margins]))}
                                           for k in ('heat','association','combined')}}
                save_rows(out/f'{key}_samples.jsonl',rows)
                reports[key] = report
                write(out/'report.json',{'status':'RUNNING','tracks':reports})
                safe_print(f'{key}: 联合Recall off={off["joint_recall100_f1_08"]:.4f}, on={on["joint_recall100_f1_08"]:.4f}')
            del context,head
            gc.collect()
            torch.cuda.empty_cache()
    runtime.postcheck('diagnostic')
    for row in checkpoints.values():
        verified_read(row,out/'anomalies')
    write(out/'report.json',{'status':'PASS','test_executed':False,'tracks':reports,
          'wall_seconds':time.time()-start,'peak_ram_percent':runtime.peak_ram,
          'input_audit':str(out/'diagnostic_input_audit.json'),
          'artifacts':[identity(p) for p in sorted(out.glob('*_samples.jsonl'))]})
    safe_print(f'完成；报告：{out / "report.json"}')


if __name__ == '__main__':
    run()
