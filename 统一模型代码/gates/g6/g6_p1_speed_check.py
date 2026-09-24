"""Engineering-only equivalence and throughput sweep; never starts formal tracks."""
import copy
import argparse
import gc
import os
import time
import traceback

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from 统一模型代码.common.g5_runtime_v2 import batches
from 统一模型代码.gates.g5.e2e_g5_train import rng_state, restore_rng
from 统一模型代码.gates.g6.g6_p1_model import forward
from 统一模型代码.gates.g6.g6_p1_runtime import BASE, Runtime, SEEDS, RunLock, register, write, identity, safe_print
from 统一模型代码.gates.g6.g6_p1_speed import physical_maps, physical_batches
from 统一模型代码.gates.g6.g6_p1_train import payload, restore, train_step


def cpu_state(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_state(v) for k,v in value.items()}
    if isinstance(value,list):
        return [cpu_state(v) for v in value]
    return copy.deepcopy(value)


def compare(a,b,rtol=1e-4,atol=1e-6):
    if isinstance(a,torch.Tensor):
        torch.testing.assert_close(a,b,rtol=rtol,atol=atol)
    elif isinstance(a,dict):
        assert a.keys() == b.keys()
        for k in a:
            compare(a[k],b[k],rtol,atol)
    elif isinstance(a,list):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            compare(x,y,rtol,atol)
    else:
        assert a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--larger',action='store_true')
    larger = parser.parse_args().larger
    chunks = (256,) if larger else (64,128)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with RunLock():
        register()
        out = BASE/f'speed_check/{time.strftime("%Y%m%d_%H%M%S")}'
        out.mkdir(parents=True,exist_ok=False)
        start = time.perf_counter()
        runtime = Runtime(out,deadline=time.time()+1800)
        report = {'formal_training_executed':False,'test_executed':False,
                  'contract':identity(BASE/'contract.json'),'equivalence':{},'timings':{}}
        try:
            runtime.preflight()
            bundle = runtime.features('train')
            ids = torch.tensor([torch.nonzero(bundle[1].counts==k).flatten()[0] for k in range(4)])
            records = runtime.stats('train',ids)
            safe_print('速度优化阶段1：真实物理缓存的前向与梯度等价。')
            torch.manual_seed(31)
            p = torch.rand(4,3,19,device='cuda',dtype=torch.float64,requires_grad=True)
            runtime.physics.p1_batch_mode = 'serial'
            reference = physical_maps(runtime.physics,p,records)
            objective = torch.randn_like(reference)
            gradient = torch.autograd.grad((reference*objective).sum(),p)[0]
            reference = reference.detach()
            for chunk in chunks:
                torch.cuda.reset_peak_memory_stats()
                runtime.physics.p1_batch_mode = 'batched'; runtime.physics.p1_chunk = chunk
                actual = physical_maps(runtime.physics,p,records)
                grad = torch.autograd.grad((actual*objective).sum(),p)[0]
                torch.testing.assert_close(actual,reference,rtol=1e-9,atol=1e-9)
                torch.testing.assert_close(grad,gradient,rtol=1e-6,atol=1e-8)
                runtime.guard()
                report['equivalence'][str(chunk)] = dict(
                    max_map_difference=float((actual.detach()-reference).abs().max()),
                    max_gradient_difference=float((grad-gradient).abs().max()),
                    peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
                del actual,grad
            del p,reference,gradient,objective,records
            gc.collect(); torch.cuda.empty_cache()
            safe_print('速度优化阶段2：同一非零物理头起点，比较三步更新及优化器状态。')
            context,head,optimizer,params = runtime.context(SEEDS[0],'c2')
            iterator = batches(bundle[0],[ids],prefetch=False)
            _,batch = next(iterator); iterator.close()
            runtime.consumed(bundle[3],ids)
            runtime.physics.p1_batch_mode = 'serial'
            for _ in range(3):
                train_step(runtime,context,head,optimizer,params,'c2',batch,bundle[1],ids)
            initial, opt, rng = cpu_state(payload(context,head)),cpu_state(optimizer.state_dict()),rng_state()
            reference_state = reference_opt = reference_outputs = None
            for mode,chunk in [('serial',64)]+[('batched',c) for c in chunks]:
                runtime.physics.p1_batch_mode = mode; runtime.physics.p1_chunk = chunk
                restore(context,head,initial); optimizer.load_state_dict(copy.deepcopy(opt)); restore_rng(rng)
                losses=[]
                for _ in range(3):
                    losses.append(train_step(runtime,context,head,optimizer,params,'c2',batch,bundle[1],ids)[0])
                state,optim = cpu_state(payload(context,head)),cpu_state(optimizer.state_dict())
                with torch.no_grad():
                    outputs = cpu_state(list(forward(context,head,batch,ids,'c2',runtime.physics,runtime.stats('train',ids))[0]))
                if reference_state is None:
                    reference_state,reference_opt,reference_outputs = state,optim,outputs
                else:
                    compare(state,reference_state); compare(optim,reference_opt); compare(outputs,reference_outputs)
                report['equivalence'][f'{mode}{chunk}_three_updates'] = {'pass':True,'losses':losses}
            del context,head,optimizer,params,batch,initial,opt,rng,state,optim,outputs
            del reference_state,reference_opt,reference_outputs
            gc.collect(); torch.cuda.empty_cache()
            order = torch.randperm(4096,generator=torch.Generator().manual_seed(SEEDS[0]))[:128]
            safe_print('速度优化阶段3：相同128条样本，两次交错顺序比较；不增加训练数据。')
            configs = [('serial',64,False),('batched',64,False),('batched',128,False),('batched',128,True)]
            if larger:
                configs = [('batched',128,True),('batched',256,True)]
            for repeat in range(2):
                for mode,chunk,prefetch in configs[::1 if repeat == 0 else -1]:
                    for arm in ('c1','c2'):
                        context,head,optimizer,params = runtime.context(SEEDS[0],arm)
                        runtime.physics.p1_batch_mode = mode; runtime.physics.p1_chunk = chunk
                        iterator = batches(bundle[0],[ids],prefetch=False)
                        _,batch = next(iterator); iterator.close()
                        train_step(runtime,context,head,optimizer,params,arm,batch,bundle[1],ids)
                        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                        begin = time.perf_counter()
                        if prefetch:
                            iterator = physical_batches(runtime,bundle[0],order.split(4),'train',True)
                        else:
                            iterator = batches(bundle[0],list(order.split(4)),prefetch=True)
                        try:
                            for item in iterator:
                                current,data = item[:2]
                                runtime.consumed(bundle[3],current)
                                stats = item[2] if prefetch else None
                                train_step(runtime,context,head,optimizer,params,arm,data,bundle[1],current,stats)
                        finally:
                            iterator.close()
                        torch.cuda.synchronize()
                        seconds=time.perf_counter()-begin
                        key=f'{mode}{chunk}_prefetch{prefetch}_{arm}'
                        report['timings'].setdefault(key,[]).append(dict(seconds=seconds,
                            peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3))
                        safe_print(f'{repeat+1}/2 {key}: {seconds:.2f}s/32batch')
                        del context,head,optimizer,params,batch,data,item
                        gc.collect(); torch.cuda.empty_cache()
                        write(out/'partial.json',report)
            runtime.postcheck('speed_check')
            report.update(status='PASS',seconds=time.perf_counter()-start,peak_ram_percent=runtime.peak_ram)
            write(out/'report.json',report)
            safe_print(f'速度筛选完成：{out}')
        except BaseException:
            report.update(status='FAILED',seconds=time.perf_counter()-start,traceback=traceback.format_exc())
            write(out/'failure.json',report)
            raise


if __name__ == '__main__':
    main()
