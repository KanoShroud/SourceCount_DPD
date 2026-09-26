"""Bounded real-data engineering preparation for the clean formal protocol."""
import argparse
import gc
import hashlib
import statistics
import time

import torch

from 统一模型代码.gates.g7.formal_runtime import Runtime,BASE,SOURCE,CONFIG,identity,read,write,save,load
from 统一模型代码.gates.g7.formal_model import FormalFoundation,FormalCandidate
from 统一模型代码.gates.g7.formal_train import make_scheduler
from 统一模型代码.gates.g6.g6_p2_runtime import setup_environment
from 统一模型代码.gates.g5.e2e_g5_train import rng_state,restore_rng


def tree_equal(a,b):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: tree_equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): tree_equal(x,y)
    else:
        assert a==b,(a,b)


def fingerprint(state):
    h=hashlib.sha256()
    def visit(value):
        if torch.is_tensor(value):h.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value,dict):
            for key in sorted(value):h.update(key.encode());visit(value[key])
    visit(state)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reuse-scene-audit',type=str)
    args=parser.parse_args()
    setup_environment()
    out=BASE/'formal_engineering'/time.strftime('%Y%m%d_%H%M%S')
    rt=Runtime(out); started=time.perf_counter()
    report=dict(status='RUNNING',formal_training_executed=False,test_read=False,optimizer_steps=0)
    write(out/'report.json',report)
    try:
        rt.contract()
        print('正式版短测1/4：快照身份与全部开发场景IQ去重',flush=True)
        rt.inputs.audit_raw()
        if args.reuse_scene_audit:
            from pathlib import Path
            previous=Path(args.reuse_scene_audit).resolve()
            prior_contract=read(previous.parent/'contract.json')
            if any(row not in prior_contract['files'] for row in rt.sources):
                raise RuntimeError('Scene audit cannot be reused across data contracts')
            rt.bind_scene_audit(identity(previous))
            write(out/'scene_identity_report.json',read(previous))
        else:
            rt.audit_scenes()
        selected={s:sorted(i for k in range(4) for i in [j for j,r in enumerate(rt.manifest['subsets'][s])
                          if r['true_k']==k][:4]) for s in ('train','val_select')}
        local_ids=[i for k in range(4) for i in [j for j in selected['train']
                   if rt.manifest['subsets']['train'][j]['true_k']==k][:2]]
        seed=CONFIG['seeds'][0]
        timings={};states={};cp_sizes={}
        for phase in ('ch3','d8'):
            print(f'正式版短测2/4：{phase}随机初始化与完整恢复',flush=True)
            model=FormalFoundation(out,rt.manifest,seed,phase)
            initial=model.state()
            ids=[i for i in selected['train'] if phase!='d8' or rt.manifest['subsets']['train'][i]['true_k']]
            timings[phase]={}
            for size in (4,8):
                model.restore(initial);model.mode(True)
                opt=rt.optimizer(model,phase)
                times=[]
                for n in range(4):
                    rt.clear_cache();torch.cuda.synchronize();t=time.perf_counter()
                    batch=rt.data(phase,seed,'train',ids[n:n+size])
                    opt.zero_grad(set_to_none=True);v,_=model.forward_loss(batch)
                    v.backward(); norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),10.)
                    if not torch.isfinite(v) or not torch.isfinite(norm):raise AssertionError('Native nonfinite')
                    opt.step();report['optimizer_steps']+=1
                    torch.cuda.synchronize()
                    if n: times.append(time.perf_counter()-t)
                model.mode(False);t=time.perf_counter()
                with torch.no_grad():model.forward_loss(batch)
                torch.cuda.synchronize()
                timings[phase][str(size)]=dict(train_seconds=statistics.mean(times),val_seconds=time.perf_counter()-t)
            model.mode(True)
            scheduler=make_scheduler(opt,CONFIG['phases'][phase])
            row=save(out/f'{phase}_resume.pt',dict(state=model.state(),optimizer=opt.state_dict(),
                     scheduler=scheduler.state_dict(),rng=rng_state()))
            def update():
                opt.zero_grad(set_to_none=True);value,_=model.forward_loss(batch)
                value.backward();torch.nn.utils.clip_grad_norm_(model.model.parameters(),10.)
                opt.step();scheduler.step()
                return float(value.detach())
            first=update();report['optimizer_steps']+=1;expected=model.state()
            cp=load(row,out);model.restore(cp['state']);opt.load_state_dict(cp['optimizer'])
            scheduler.load_state_dict(cp['scheduler']);restore_rng(cp['rng'])
            second=update();report['optimizer_steps']+=1
            assert first==second;tree_equal(expected,model.state())
            states[phase]=model.state();cp_sizes[phase]=row['size_bytes']
            model=opt=None;gc.collect();torch.cuda.empty_cache()
        print('正式版短测3/4：新基础→候选→局部输入；不使用历史候选',flush=True)
        model=FormalCandidate(out,rt.manifest,seed)
        model.load_foundations(states['ch3'],states['d8'])
        base=model.state();timings['candidate']={}
        for size in (4,8):
            model.restore(base);model.mode(True);opt=rt.optimizer(model,'candidate')
            times=[]
            for n in range(4):
                rt.clear_cache();torch.cuda.synchronize();t=time.perf_counter()
                batch=rt.data('candidate',seed,'train',selected['train'][n:n+size])
                opt.zero_grad(set_to_none=True);o=rt.forward(model,'candidate',batch);v,_=rt.loss(o,batch,'candidate')
                v.backward();norm=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],10.)
                if not torch.isfinite(v) or not torch.isfinite(norm):raise AssertionError('Candidate nonfinite')
                opt.step();report['optimizer_steps']+=1;torch.cuda.synchronize()
                if n: times.append(time.perf_counter()-t)
            model.mode(False);torch.cuda.synchronize();t=time.perf_counter()
            with torch.no_grad():
                o=rt.forward(model,'candidate',batch);rt.metric_rows(o,batch,'candidate')
            torch.cuda.synchronize();timings['candidate'][str(size)]=dict(
                train_seconds=statistics.mean(times),val_seconds=time.perf_counter()-t)
        cp=save(out/'candidate_resume.pt',dict(state=model.state(),optimizer=opt.state_dict()))
        cp_sizes['candidate']=cp['size_bytes']
        model.restore(load(cp,out)['state']);model.initialize_local();model.mode(False)
        local_start=model.state();local_rows=[];local_seconds=[]
        for i in local_ids:
            torch.cuda.synchronize();t=time.perf_counter()
            value=rt.local_record(model,seed,'train',i,'ENGINEERING_RANDOM_NOT_FORMAL_CANDIDATE')
            torch.cuda.synchronize();local_seconds.append(time.perf_counter()-t)
            local_rows.append(save(out/f'local/{i:05d}.pt',value))
        timings.update({p:{} for p in ('f','s','e')}); shared={}
        for phase in ('f','s','e'):
            shared[phase]=fingerprint(local_start)
            for size in (4,8):
                model.restore(local_start);model.mode(True);opt=rt.optimizer(model,phase);times=[]
                for n in range(4):
                    rt.clear_cache();torch.cuda.synchronize();t=time.perf_counter()
                    batch=rt.data('candidate',seed,'train',local_ids[:size])
                    batch=[dict(r,_local=rt.get(row)) for r,row in zip(batch,local_rows[:size])]
                    opt.zero_grad(set_to_none=True);o=rt.forward(model,phase,batch);v,_=rt.loss(o,batch,phase)
                    v.backward();norm=torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g['params']],10.)
                    if not torch.isfinite(v) or not torch.isfinite(norm):raise AssertionError('Local nonfinite')
                    opt.step();report['optimizer_steps']+=1;torch.cuda.synchronize()
                    if n: times.append(time.perf_counter()-t)
                model.mode(False);torch.cuda.synchronize();t=time.perf_counter()
                with torch.no_grad():
                    o=rt.forward(model,phase,batch);rt.metric_rows(o,batch,phase)
                torch.cuda.synchronize();timings[phase][str(size)]=dict(
                    train_seconds=statistics.mean(times),val_seconds=time.perf_counter()-t)
        cp=save(out/'local_resume.pt',dict(state=model.state(),optimizer=opt.state_dict()))
        cp_sizes['local']=cp['size_bytes']
        print('正式版短测4/4：物理梯度隔离、资源测量和输入复核',flush=True)
        gradients={}
        for phase in ('s','e'):
            model.restore(local_start);model.mode(False)
            model.context.query_builder.zero_grad(set_to_none=True);model.selector.zero_grad(set_to_none=True)
            o=rt.forward(model,phase,batch[-1:]);o['query'].retain_grad();o['band_logits'].retain_grad()
            probe=o['output']['heat_logits'].sigmoid().sum()
            probe.backward()
            gradients[phase]=dict(query=0. if o['query'].grad is None else float(o['query'].grad.norm()),
                semantic=0. if o['band_logits'].grad is None else float(o['band_logits'].grad.norm()))
        assert gradients['s']['query']==0 and gradients['e']['query']>0,gradients
        assert all(v['semantic']==0 for v in gradients.values())
        # One timing-only scene verifies the online path; no validation accuracy is evaluated.
        from 统一模型代码.gates.g7.formal_latency import profile_online
        candidate_cp=save(out/'online_candidate.pt',dict(state=model.state()))
        local_cp=save(out/'online_local.pt',dict(state=local_start))
        native_rows={p:save(out/f'online_{p}.pt',dict(state=states[p])) for p in ('ch3','d8')}
        original_best=rt.best
        rt.best=lambda _seed,phase: native_rows[phase] if phase in native_rows else (
            candidate_cp if phase=='candidate' else local_cp)
        try:
            online_smoke={phase:profile_online(rt,seed,phase,[0]) for phase in ('candidate','e')}
        finally:
            rt.best=original_best
        rt.inputs.audit_raw()
        for row in rt.registration['files']:
            from 统一模型代码.common.g5_verified_io import verified_read
            verified_read(row,out/'anomalies')
        ntrain=CONFIG['counts']['train'];nval=CONFIG['counts']['val_select'];ncompare=CONFIG['counts']['val_compare']
        positive={s:len(rt.ids('d8',s)) for s in ('train','val_select')}
        def forecast(maximum):
            total=0.
            for phase in ('ch3','d8','candidate','f','s','e'):
                cfg=CONFIG['phases'][phase];size=cfg['batch_size'];t=timings[phase][str(size)]
                epochs=cfg['max_epochs' if maximum else 'base_epochs']
                tr=positive['train'] if phase=='d8' else ntrain
                va=positive['val_select'] if phase=='d8' else nval
                total+=2*(epochs*((tr+size-1)//size)*t['train_seconds']+
                          (1+epochs//2)*((va+size-1)//size)*t['val_seconds'])
                if phase not in ('ch3','d8'):total+=2*((ncompare+size-1)//size)*t['val_seconds']
            total+=2*(ntrain+nval+ncompare)*statistics.mean(local_seconds)
            # Real hard cascade is not benchmarked here: measured fine-map generation plus both networks.
            from 统一模型代码.gates.g7.compact_foundation import build_oracle_fine
            r=rt.data('candidate',seed,'train',[selected['train'][0]])[0]
            t=time.perf_counter()
            build_oracle_fine(rt.signal('train',r['index']),dict(count=3,oracle_slots=torch.ones(3,19)),
                              (rt.lo,rt.hi),'cuda')
            torch.cuda.synchronize();hard_seconds=time.perf_counter()-t+.1
            total+=2*ncompare*hard_seconds
            return 1.5*total+1200
        base_forecast=forecast(False);max_forecast=forecast(True)
        elapsed=time.perf_counter()-started
        previous=sum(read(p).get('seconds',0.) for p in (BASE/'formal_engineering').glob('*/report.json') if p.parent!=out)
        inherited=read(SOURCE/'budget.json')['active_seconds']
        charged=inherited+previous+elapsed
        cache_gib=sum(row['size_bytes'] for row in local_rows)/len(local_rows)*2*(ntrain+nval+ncompare)/2**30
        checkpoints_gib=(2*121*(cp_sizes['ch3']+cp_sizes['d8'])+2*61*cp_sizes['candidate']+6*61*cp_sizes['local'])/2**30
        required=cache_gib+checkpoints_gib+CONFIG['disk_floor_gib']
        import shutil
        time_ok=max_forecast<=CONFIG['wall_seconds']-charged
        disk_ok=required<=shutil.disk_usage(out).free/2**30
        report.update(status='ENGINEERING_PASS',selected=selected,local_indices=local_ids,timings=timings,
            native_full_state_next_update_replay=True,local_initial_fingerprints=shared,feedback=gradients,
            online_timing_smoke=online_smoke,
            scene_audit=identity(out/'scene_identity_report.json'),contract=identity(out/'contract.json'),
            candidate_independent_of_historical_models=True,local_statistics_mean_seconds=statistics.mean(local_seconds),
            local_cache_estimate_gib=cache_gib,checkpoint_estimate_gib=checkpoints_gib,required_free_gib=required,
            forecast_base_seconds=base_forecast,forecast_remaining_seconds=max_forecast,
            inherited_prior_seconds=inherited,charged_seconds_before_training=charged,
            resource_status='WITHIN_BUDGET' if time_ok and disk_ok else 'RESOURCE_HOLD',
            time_ok=time_ok,disk_ok=disk_ok,gpu_peak_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_ram_percent=rt.peak_ram_percent,seconds=elapsed,
            benchmark_method='Each phase/batch: one excluded warmup update plus three timed updates; same bounded lazy loader',
            timing_scope='small real-data pilot, untrained proposals; estimates not full-run measurements')
        write(out/'report.json',report)
        write(BASE/'formal_preparation_report.json',dict(status=report['status'],report=identity(out/'report.json')))
        print(f'正式版工程检查完成：{out / "report.json"}；资源={report["resource_status"]}',flush=True)
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc),seconds=time.perf_counter()-started)
        write(out/'report.json',report);raise
    finally:
        rt.close()


if __name__=='__main__':main()
