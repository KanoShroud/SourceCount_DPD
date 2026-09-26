"""Bounded preparation only: no epoch loop, no optimizer step, no test split."""
import gc
import time

import torch

from 统一模型代码.gates.g7.compact_data import BASE, SPLITS, identity, load, read, save, write
from 统一模型代码.gates.g7.compact_model import candidates, windows, loss, decode
from 统一模型代码.gates.g7.compact_run import Runtime, CONFIG, setup_environment


def main():
    setup_environment()
    out = BASE/'engineering'/time.strftime('%Y%m%d_%H%M%S')
    rt = Runtime(out)
    started = time.perf_counter()
    report = dict(status='RUNNING', formal_training_executed=False, test_executed=False,
                  optimizer_steps=0, material_passport=dict(mode='run', verification_status='RUNNING'))
    write(out/'report.json', report)
    try:
        rt.contract()
        print('小区域准备：核对快照身份', flush=True)
        rt.inputs.audit_raw()
        selected = {}
        for split in ('train', 'val_select'):
            entries = rt.inputs.manifest['subsets'][split]
            selected[split] = sorted(i for k in range(4) for i in
                [j for j, r in enumerate(entries) if r['true_k'] == k][:4])
        before = time.perf_counter()
        rt.dataset = rt.inputs.prepare(out/'data', selected, rt.guard)
        data_seconds = time.perf_counter()-before
        print('小区域准备：201网格前后向、坐标与保存恢复', flush=True)
        model = rt.model(CONFIG['seeds'][0], 'baseline')
        model.mode(False)
        ids = selected['train'][:4]
        records = rt.data('train', ids)
        with torch.no_grad():
            result = model.baseline(records, rt.physics)
        if result['heat'].shape != (4, 3, 201, 201):
            raise AssertionError('Small network still emits large grid')
        cp = save(out/'model_replay.pt', dict(state=model.state()))
        first = result['heat'].detach().clone()
        model.restore(load(cp, out)['state'])
        with torch.no_grad():
            replay = model.baseline(records, rt.physics)['heat']
        torch.testing.assert_close(first, replay, rtol=0, atol=0)
        timings = []
        for start in range(0, len(selected['train']), 4):
            rt.guard()
            batch_ids = selected['train'][start:start+4]
            torch.cuda.synchronize(); t = time.perf_counter()
            batch = rt.data('train', batch_ids)
            model.mode(True)
            result = model.baseline(batch, rt.physics)
            total, parts = loss(result, batch, 'baseline')
            for group in model.context.parameter_groups:
                for p in group['params']:
                    p.grad = None
            model.physical.zero_grad(set_to_none=True)
            total.backward()
            grads = [p.grad for group in model.context.parameter_groups for p in group['params'] if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):
                raise AssertionError('Nonfinite/missing small model gradients')
            torch.cuda.synchronize(); timings.append(time.perf_counter()-t)
        model.mode(False)
        torch.cuda.synchronize(); validation_start = time.perf_counter()
        with torch.no_grad():
            result = model.baseline(records, rt.physics)
            pred = decode(result, 'baseline')
            geometry = [windows(r, 5) for r in candidates(result['heat'], 5)]
        torch.cuda.synchronize(); validation_seconds = time.perf_counter()-validation_start
        if any(abs(v)>1000.001 for row in pred for xy in row['positions_m'] for v in xy):
            raise AssertionError('Decoded position escaped ROI')
        # End-to-end physical local path on a real predicted compact window, no parameter update.
        from 统一模型代码.gates.g7 import g7_physics
        from 统一模型代码.gates.g7.compact_data import g4
        with g4.g1.SampleStore('train') as store:
            entry = rt.inputs.manifest['subsets']['train'][ids[0]]
            raw, ri = store._raw(entry['raw_index'])
            import numpy as np
            signal = np.asarray(raw['sig_rcv_real_all'][:, :, ri], dtype=np.float64).T + 1j*np.asarray(raw['sig_rcv_imag_all'][:, :, ri], dtype=np.float64).T
        centers, valid, points, inverse = geometry[0]
        local = g7_physics.statistics(rt.physics, signal, points, guard=rt.guard)
        local.update(centers_m=centers, valid_mask=valid, points=points, inverse=inverse)
        from 统一模型代码.gates.g7.compact_model import local_maps
        padded = {**local, 'inverse': torch.cat((inverse, inverse[:, :3]), 1),
                  'valid_mask': torch.cat((valid, valid[:, :3]), 1)}
        weights = torch.full((1, 3, 19), .5, device='cuda', dtype=torch.float64)
        with torch.no_grad():
            old_maps = g7_physics.local_maps(rt.physics, weights, [padded])[:, :, :5]
            new_maps = local_maps(rt.physics, weights, [local])
            torch.testing.assert_close(old_maps, new_maps, rtol=0, atol=0)
        one = records[:1]
        gradients = {}
        for arm in ('s', 'e'):
            model.mode(False)
            model.context.query_builder.zero_grad(set_to_none=True)
            model.selector.zero_grad(set_to_none=True)
            output = model.local(one, rt.physics, [local], arm)
            output['query'].retain_grad(); output['band_logits'].retain_grad()
            _, parts = loss(output, one, arm)
            (parts['heatmap']+parts['offset']+parts['candidate']).backward()
            gradients[arm] = dict(query=0. if output['query'].grad is None else float(output['query'].grad.norm()),
                semantic=0. if output['band_logits'].grad is None else float(output['band_logits'].grad.norm()),
                selector=sum(float(p.grad.norm()) for p in model.selector.parameters() if p.grad is not None))
        if gradients['s']['query'] != 0 or gradients['e']['query'] <= 0 or any(r['semantic'] != 0 for r in gradients.values()):
            raise AssertionError('S/E physical feedback isolation failed')
        bytes_total = sum(r['size_bytes'] for s in SPLITS for r in rt.dataset[s].values())
        rt.inputs.audit_raw()
        elapsed = time.perf_counter()-started
        previous_seconds = sum(read(p).get('seconds', 0.) for p in (BASE/'engineering').glob('*/report.json')
                               if p.parent != out)
        preparation_seconds = elapsed+previous_seconds
        # Small warm-cache timing is an estimate: include 50% reserve and 20 minutes for audit/I/O.
        forecast = preparation_seconds + 1.5*(
            data_seconds/sum(map(len, selected.values()))*sum(CONFIG['counts'].values())
            + max(timings)*1024*24*2 + validation_seconds*128*(13*2+2)) + 1200
        report.update(status='ENGINEERING_PASS', samples=sum(map(len, selected.values())), selected=selected,
            network_shape=list(first.shape), replay_exact=True, physical_feedback=gradients,
            dynamic_top5_matches_original_physics=True, validation_batch_seconds=validation_seconds,
            forecast_baselines_max_seconds=forecast, preparation_consumed_seconds=preparation_seconds,
            forecast_baselines_remaining_seconds=forecast-preparation_seconds,
            baseline_batch_seconds=timings, data_seconds=data_seconds, data_bytes=bytes_total,
            projected_dataset_gib=bytes_total/sum(map(len, selected.values()))*sum(CONFIG['counts'].values())/2**30,
            gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30, seconds=elapsed,
            contract=identity(out/'contract.json'), local_points=len(points),
            next='User baseline adaptation then new-model Top5 review; no local six-track approval inferred')
        report['material_passport']['verification_status'] = 'VERIFIED_ENGINEERING_ONLY'
        write(out/'report.json', report)
        write(BASE/'preparation_report.json', dict(status=report['status'], report=identity(out/'report.json')))
        print(f'小区域短验证完成：{out / "report.json"}', flush=True)
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc), seconds=time.perf_counter()-started)
        write(out/'report.json', report)
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
