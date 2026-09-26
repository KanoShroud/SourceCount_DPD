"""Clean G7 epoch training: fixed maximum-horizon schedule and exact committed resume."""
from __future__ import annotations

import gc
import math
import time

import torch

from 统一模型代码.gates.g7.compact_data import read, write, save, load
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng
from 统一模型代码.gates.g5.r2.g5_r2_evaluate import summarize

NATIVE = ('ch3', 'd8')
PHASES = (*NATIVE, 'candidate', 'f', 's', 'e')


def phase_config(config, phase):
    if phase not in PHASES:
        raise ValueError(f'Unknown phase: {phase}')
    result = config['phases'][phase]
    if not 0 <= result['warmup_epochs'] < result['max_epochs']:
        raise ValueError('Warmup must be shorter than the registered maximum horizon')
    if result['base_epochs'] > result['max_epochs'] or result['evaluate_every'] < 1:
        raise ValueError('Invalid epoch configuration')
    return result


def lr_factor(index, config):
    """Index 0 is epoch 1; epoch max reaches min factor. Extension never resets cosine."""
    maximum, warmup = config['max_epochs'], config['warmup_epochs']
    floor = config['min_lr_factor']
    if not 0 < floor <= 1:
        raise ValueError('Invalid minimum learning-rate factor')
    if warmup and index < warmup:
        return (index+1)/warmup
    start = max(warmup-1, 0)
    progress = min(max((index-start)/max(maximum-1-start, 1), 0.), 1.)
    return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))


def make_scheduler(optimizer, config):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda index: lr_factor(index, config))


def selection_key(metrics, phase, epoch):
    if phase in NATIVE:
        return (metrics['val_loss'], epoch)
    return (-metrics['joint_recall100_f1_08'], metrics['gospa_m'], epoch)


def needs_extension(result, phase, config):
    cfg = phase_config(config, phase)
    base, interval = cfg['base_epochs'], cfg['evaluate_every']
    rows = [r for r in result['history'] if r['epoch'] <= base and r.get('validation') is not None]
    if len(rows) < 3:
        return False
    best = min(rows, key=lambda r: selection_key(r['validation'], phase, r['epoch']))
    if best['epoch'] not in (base-interval, base):
        return False
    tail = rows[-3:]
    first, last = tail[0]['validation'], tail[-1]['validation']
    # Remove epoch tie-break: only actual metric improvement can extend the budget.
    return selection_key(last, phase, 0) < selection_key(first, phase, 0)


def _batches(ids, size):
    return [ids[i:i+size] for i in range(0, len(ids), size)]


def _loss(rt, model, phase, batch):
    if phase in NATIVE:
        return model.forward_loss(batch)
    result = rt.forward(model, phase, batch)
    value = rt.loss(result, batch, phase)
    return value[0], value[1]


@torch.no_grad()
def evaluate(rt, model, seed, phase, split='val_select'):
    if split not in ('val_select', 'val_compare'):
        raise ValueError('Evaluation accepts only development validation splits; no test')
    if phase in NATIVE and split != 'val_select':
        raise ValueError('Native foundation selection uses val_select only')
    cfg = phase_config(rt.config, phase)
    ids = list(rt.ids(phase, split))
    if not ids:
        raise ValueError('Empty validation split')
    model.mode(False)
    bar = rt.progress(f'{seed}/{phase} {split}', len(ids))
    total, rows = 0., []
    try:
        done = 0
        for indices in _batches(ids, cfg['batch_size']):
            rt.guard()
            batch = rt.data(phase, seed, split, indices)
            if phase in NATIVE:
                value, _ = model.forward_loss(batch)
            else:
                output = rt.forward(model, phase, batch)
                value = rt.loss(output, batch, phase)[0]
                rows.extend(rt.metric_rows(output, batch, phase))
            if not torch.isfinite(value):
                raise RuntimeError('Nonfinite validation loss')
            total += float(value)*len(indices)
            done += len(indices)
            bar.update(done)
    finally:
        model.mode(True)
    metrics = {'val_loss': total/len(ids), 'scenes': len(ids)}
    if phase not in NATIVE:
        if len(rows) != len(ids):
            raise RuntimeError('Evaluation row count does not match requested samples')
        metrics.update(summarize(rows))
    return metrics, rows


def train(rt, seed, phase, end):
    cfg = phase_config(rt.config, phase)
    if end not in (cfg['base_epochs'], cfg['max_epochs']):
        raise ValueError('Unregistered training horizon')
    root = rt.out/f'training/{seed}/{phase}'
    root.mkdir(parents=True, exist_ok=True)
    pointer = root/'completed.json'
    model = rt.model(seed, phase)
    optimizer = rt.optimizer(model, phase)
    scheduler = make_scheduler(optimizer, cfg)
    parameters = [p for group in optimizer.param_groups for p in group['params']]
    generator = torch.Generator().manual_seed(seed)
    history, best, steps = [], None, 0

    def commit(epoch, loss_value, validation, seconds, train_lr, components=None):
        nonlocal best
        history.append(dict(epoch=epoch, loss=loss_value, validation=validation,
            seconds=seconds, learning_rates=train_lr, next_learning_rates=scheduler.get_last_lr(),
            loss_components=components))
        checkpoint = save(root/f'checkpoints/epoch{epoch:03d}_{time.time_ns()}.pt', dict(
            state=model.state(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            generator=generator.get_state(), rng=rng_state(), history=history, steps=steps,
            seed=seed, phase=phase, phase_config=cfg))
        if validation is not None and (best is None or selection_key(validation, phase, epoch)
                                      < selection_key(best['metrics'], phase, best['epoch'])):
            best = dict(epoch=epoch, metrics=validation, checkpoint=checkpoint)
        write(pointer, dict(checkpoint=checkpoint, best=best, history=history, steps=steps,
            completed_epoch=epoch, seed=seed, phase=phase))

    try:
        if pointer.exists():
            previous = read(pointer)
            cp = load(previous['checkpoint'], rt.out)
            if cp['phase_config'] != cfg or (cp['seed'], cp['phase']) != (seed, phase):
                raise RuntimeError('Resume configuration/track mismatch')
            model.restore(cp['state'])
            optimizer.load_state_dict(cp['optimizer'])
            scheduler.load_state_dict(cp['scheduler'])
            generator.set_state(cp['generator'])
            restore_rng(cp['rng'])
            history, best, steps = cp['history'], previous['best'], cp['steps']
            if previous['completed_epoch'] != history[-1]['epoch'] or previous['steps'] != steps:
                raise RuntimeError('Epoch commit pointer mismatch')
            del cp
        else:
            metrics, _ = evaluate(rt, model, seed, phase)
            commit(0, None, metrics, 0., scheduler.get_last_lr())
        ids = list(rt.ids(phase, 'train'))
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('Training identifiers are empty or duplicated')
        for epoch in range(history[-1]['epoch']+1, end+1):
            rt.guard()
            model.mode(True)
            started = time.perf_counter()
            order = [ids[i] for i in torch.randperm(len(ids), generator=generator).tolist()]
            bar = rt.progress(f'{seed}/{phase} epoch {epoch}/{end}', len(order))
            train_lr = [group['lr'] for group in optimizer.param_groups]
            total, count, component_sums = 0., 0, {}
            for indices in _batches(order, cfg['batch_size']):
                rt.guard()
                batch = rt.data(phase, seed, 'train', indices)
                optimizer.zero_grad(set_to_none=True)
                value, parts = _loss(rt, model, phase, batch)
                value.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, rt.config.get('gradient_clip', 10.))
                if not torch.isfinite(value) or not torch.isfinite(norm):
                    raise RuntimeError('Nonfinite training loss/gradient')
                optimizer.step()
                steps += 1
                count += len(indices)
                total += float(value.detach())*len(indices)
                for name, part in parts.items():
                    number = float(part.detach()) if torch.is_tensor(part) else float(part)
                    component_sums[name] = component_sums.get(name, 0.)+number*len(indices)
                bar.update(count)
            metrics = evaluate(rt, model, seed, phase)[0] if epoch % cfg['evaluate_every'] == 0 else None
            # Exactly one scheduler step per committed epoch, never per batch or eval.
            scheduler.step()
            commit(epoch, total/count, metrics, time.perf_counter()-started, train_lr,
                   {k: v/count for k, v in component_sums.items()})
            print(f'\n{seed}/{phase} 第{epoch}/{end}轮完成：loss={total/count:.5g}，'
                  f'耗时{history[-1]["seconds"]/60:.1f}min，best={best["epoch"]}', flush=True)
        result = read(pointer)
        result.update(extend=needs_extension(result, phase, rt.config),
            budget_unresolved=history[-1]['epoch'] >= cfg['max_epochs'] and
                best['epoch'] in (cfg['max_epochs']-cfg['evaluate_every'], cfg['max_epochs']),
            checkpoint_disk_bytes=sum(p.stat().st_size for p in (root/'checkpoints').glob('*.pt')))
        write(root/f'report_epoch{end}.json', result)
        return result
    finally:
        model = optimizer = scheduler = parameters = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
