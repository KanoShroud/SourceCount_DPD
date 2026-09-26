"""Short full-module update/resume/handoff checks; never enters an epoch loop."""
import copy
import gc
import time

import torch

from 统一模型代码.gates.g7 import compact_run as legacy
from 统一模型代码.gates.g7.compact_data import BASE, write, read, save, load, identity
from 统一模型代码.gates.g7.compact_full_run import Runtime, CONFIG, AdaptedCandidate
from 统一模型代码.gates.g7.compact_foundation import FoundationModel
from 统一模型代码.gates.g7.compact_model import loss
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng


def tensor_equal(a, b):
    if a.keys() != b.keys():
        raise AssertionError('State keys differ')
    for k in a:
        torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)


def main():
    legacy.setup_environment()
    legacy.CONFIG = CONFIG
    out = BASE/'full_engineering'/time.strftime('%Y%m%d_%H%M%S')
    rt = Runtime(out)
    started = time.perf_counter()
    report = dict(status='RUNNING', formal_training_executed=False, test_executed=False,
                  short_optimizer_steps=0)
    write(out/'report.json', report)
    try:
        rt.contract()
        print('完整适配短测：输入身份、K分层样本和D8原生选频', flush=True)
        rt.inputs.audit_raw()
        selected = {s: sorted(next(i for i, r in enumerate(rt.inputs.manifest['subsets'][s])
                    if r['true_k'] == k) for k in range(4)) for s in ('train', 'val_select')}
        t = time.perf_counter()
        rt.dataset = rt.inputs.prepare(out/'data', selected, rt.guard)
        data_seconds = time.perf_counter()-t
        t = time.perf_counter()
        rt.prepare_oracle(selected)
        oracle_seconds = time.perf_counter()-t
        checks, adapted = {}, {}
        for kind in ('ch3', 'd8'):
            print(f'完整适配短测：{kind} 全模块更新与optimizer/RNG恢复', flush=True)
            model = FoundationModel(out, rt.inputs.manifest, CONFIG['seeds'][0], kind)
            optimizer = model.optimizer(lr=CONFIG['foundation_lr'])
            ids = [i for i in selected['train'] if kind == 'ch3' or
                   rt.inputs.manifest['subsets']['train'][i]['true_k'] > 0]
            records = rt.foundation_data(kind, 'train', ids)
            # Repeat one record for the same formal batch shape, not as extra evidence samples.
            records = (records+records)[:CONFIG['batch_size']]
            model.mode(True)
            all_params = list(model.model.parameters())
            if not all(p.requires_grad for p in all_params):
                raise AssertionError('Frozen foundation parameter')
            optimized = {id(p) for group in optimizer.param_groups for p in group['params']}
            if optimized != {id(p) for p in all_params}:
                raise AssertionError('Optimizer omits foundation parameters')
            original = copy.deepcopy(model.model.state_dict())
            def step():
                optimizer.zero_grad(set_to_none=True)
                value, _ = model.forward_loss(records)
                value.backward()
                grad = torch.nn.utils.clip_grad_norm_(all_params, 10.)
                if not torch.isfinite(value) or not torch.isfinite(grad):
                    raise AssertionError('Nonfinite native gradient')
                optimizer.step()
                return float(value.detach())
            torch.cuda.synchronize(); t = time.perf_counter()
            step(); report['short_optimizer_steps'] += 1
            torch.cuda.synchronize(); first_seconds = time.perf_counter()-t
            cp = save(out/f'{kind}_resume.pt', dict(state=model.state(),
                      optimizer=optimizer.state_dict(), rng=rng_state()))
            torch.cuda.synchronize(); t = time.perf_counter()
            next_loss = step(); report['short_optimizer_steps'] += 1
            torch.cuda.synchronize(); train_seconds = time.perf_counter()-t
            expected = copy.deepcopy(model.model.state_dict())
            restored = load(cp, out)
            model.restore(restored['state']); optimizer.load_state_dict(restored['optimizer'])
            restore_rng(restored['rng'])
            resumed_loss = step(); report['short_optimizer_steps'] += 1
            tensor_equal(expected, model.model.state_dict())
            if resumed_loss != next_loss:
                raise AssertionError('Resumed next-step loss differs')
            changed = [k for k, v in expected.items() if not torch.equal(v, original[k])]
            if not changed:
                raise AssertionError('No foundation weights/buffers changed')
            model.mode(False)
            torch.cuda.synchronize(); t = time.perf_counter()
            with torch.no_grad():
                model.forward_loss(records)
            torch.cuda.synchronize(); val_seconds = time.perf_counter()-t
            adapted[kind] = {k: v.detach().cpu().clone() for k, v in expected.items()}
            checks[kind] = dict(all_parameters_trainable=True, optimizer_complete=True,
                changed_keys=changed, exact_next_update_replay=True, first_batch_seconds=first_seconds,
                train_batch_seconds=train_seconds, val_batch_seconds=val_seconds,
                checkpoint_bytes=cp['size_bytes'])
            model = optimizer = all_params = None
            del original, expected, restored
            gc.collect(); torch.cuda.empty_cache()
        print('完整适配短测：新完整权重进入统一候选生成器并保存回放', flush=True)
        initial = load(rt.registration['initial'][str(CONFIG['seeds'][0])], out)
        model = AdaptedCandidate(out, rt.inputs.manifest, CONFIG['seeds'][0], initial)
        for kind, state in adapted.items():
            getattr(model.context, kind).load_state_dict(state, strict=True)
            tensor_equal(state, {k:v.cpu() for k,v in getattr(model.context, kind).state_dict().items()})
        records = rt.data('train', selected['train'])
        model.mode(False)
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            result = model.baseline(records, rt.physics)
            from 统一模型代码.gates.g7.compact_run import metric_rows
            metric_rows(result, records, 'baseline')
        torch.cuda.synchronize(); candidate_val_seconds = time.perf_counter()-t
        cp = save(out/'candidate_full_resume.pt', dict(state=model.state()))
        expected = result['heat'].detach().clone()
        model.restore(load(cp, out)['state'])
        with torch.no_grad():
            torch.testing.assert_close(expected, model.baseline(records, rt.physics)['heat'], rtol=0, atol=0)
        model.mode(True)
        optimizer = model.optimizer('baseline', rt.inputs.manifest)
        torch.cuda.synchronize(); t = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total, _ = loss(model.baseline(records, rt.physics), records, 'baseline')
        total.backward()
        torch.cuda.synchronize(); candidate_seconds = time.perf_counter()-t
        # No candidate optimizer update: foundation restore/update is the changed engineering path.
        rt.inputs.audit_raw()
        elapsed = time.perf_counter()-started
        previous = sum(read(p).get('seconds', 0.) for folder in ('engineering', 'full_engineering')
                       for p in (BASE/folder).glob('*/report.json') if p.parent != out)
        consumed = elapsed+previous
        counts = CONFIG['counts']
        positive = {s: sum(r['true_k'] > 0 for r in rt.inputs.manifest['subsets'][s])
                    for s in ('train', 'val_select')}
        iterations = 0.
        for kind, check in checks.items():
            ntrain = counts['train'] if kind == 'ch3' else positive['train']
            nval = counts['val_select'] if kind == 'ch3' else positive['val_select']
            iterations += 2*(24*((ntrain+3)//4)*check['train_batch_seconds']
                             + 13*((nval+3)//4)*check['val_batch_seconds'])
        iterations += 2*(24*1024*candidate_seconds + 14*128*candidate_val_seconds)
        materialization = data_seconds/8*sum(counts.values()) + oracle_seconds/6*sum(positive.values())
        remaining = 1.5*(materialization+iterations)+1200
        data_bytes = sum(row['size_bytes'] for rows in rt.dataset.values() for row in rows.values())
        checkpoint_gib = (2*25*sum(c['checkpoint_bytes'] for c in checks.values())
                          + 2*25*cp['size_bytes']*3)/2**30
        report.update(status='ENGINEERING_PASS', selected=selected, foundation_checks=checks,
            full_candidate_inheritance=True, candidate_full_checkpoint_replay=True,
            candidate_train_batch_seconds=candidate_seconds, candidate_val_batch_seconds=candidate_val_seconds,
            data_seconds=data_seconds, oracle_seconds=oracle_seconds,
            projected_dataset_gib=data_bytes/8*sum(counts.values())/2**30 +
                sum(positive.values())*201*201*4/2**30,
            checkpoint_reserve_gib=checkpoint_gib,
            forecast_remaining_seconds=remaining, preparation_consumed_seconds=consumed,
            resource_status='WITHIN_NODE_BUDGET' if remaining+consumed <= CONFIG['wall_seconds'] else 'RESOURCE_HOLD',
            forecast_scope='foundations_and_candidates_top5_only_NOT_FSE',
            gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30, seconds=elapsed,
            contract=identity(out/'contract.json'))
        write(out/'report.json', report)
        write(BASE/'full_preparation_report.json', dict(status='ENGINEERING_PASS', report=identity(out/'report.json')))
        print(f'短测完成：{out / "report.json"}', flush=True)
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc), seconds=time.perf_counter()-started)
        write(out/'report.json', report)
        raise


if __name__ == '__main__':
    main()
