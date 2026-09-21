"""G5-R1独立运行器。原G5只读，所有日志、异常和报告写入独立目录。"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
from 统一模型代码.common.g5_runtime_v2 import batches  # noqa: E402
from 统一模型代码.common.g5_sample_range import SampleRangeArray, SampleRangeCache  # noqa: E402
from 统一模型代码.gates.g5.e2e_g5_model import build_context, forward, g4  # noqa: E402
from 统一模型代码.common.g5_verified_io import verified_read  # noqa: E402
from 统一模型代码.gates.g5.r1.g5_r1_decode import decode, association, candidate_ceiling, duplicate  # noqa: E402
from 统一模型代码.gates.g5.r1.g5_r1_report import write, build_report, plot_cases  # noqa: E402

SOURCE = ROOT/'outputs_e2e/unified/e2e_g5/20260919_approved'
BASE = ROOT/'outputs_e2e/unified/e2e_g5_r1/20260920_approved'
CONFIG = {'gate': 'E2E-G5-R1', 'wall_seconds': 14400, 'batch_size': 4,
          'cache_bytes': 2*1024**3, 'prefetch_batches': 1, 'ram_warning_percent': 85,
          'local_max_size': 7, 'candidates_per_slot': 8, 'separation_m': 30,
          'joint_distance_m': 100, 'joint_band_f1': .8, 'pilot_count': 32,
          'compare_count': 1024, 'seeds': [20260921,20260922,20260923],
          'bootstrap_repetitions': 2000, 'bootstrap_seed': 20260920,
          'test_executed': False, 'training_executed': False}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identity(path):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            digest.update(block)
    return {'path': str(path), 'size_bytes': path.stat().st_size, 'sha256': digest.hexdigest()}


def source_path(path):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(SOURCE.resolve(strict=True)):
        raise ValueError(f'Input escaped frozen G5: {path}')
    return path


def register():
    """只登记当前源码和小型元数据；不运行模型、不读取特征或test。"""
    BASE.mkdir(parents=True, exist_ok=True)
    destination = BASE/'contract.json'
    if destination.exists() or (BASE/'evaluation').exists():
        raise FileExistsError('已有合同/运行，禁止覆盖或重置预算')
    old = read(SOURCE/'engineering_v4/contract.json')
    for row in old['files']:
        verified_read(row, BASE/'preparation_anomalies')
    paths = {ROOT/'运行入口/E2E/G5_R1/G5_R1一键运行.py', Path(__file__), ROOT/'统一模型代码/gates/g5/r1/g5_r1_decode.py',
             ROOT/'统一模型代码/gates/g5/r1/g5_r1_report.py', ROOT/'统一模型代码/gates/g5/r1/tests/test_g5_r1.py'}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not isinstance(filename, str) or not Path(filename).is_absolute():
            continue
        path = Path(filename).resolve()
        if path.suffix == '.py' and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT/'outputs_e2e'):
            paths.add(path)
    for name in ('manifest.json','feature_manifest.json','training_report.json','comparison_report.json',
                 'hard_reference_report.json','final_audit_report.json','engineering_v4/contract.json',
                 'engineering_v2/index_registry.json'):
        paths.add(SOURCE/name)
    for seed in CONFIG['seeds']:
        for track in ('sg','e2e'):
            paths.add(SOURCE/f'training/{seed}/{track}/best.identity.json')
            paths.add(SOURCE/f'comparison/{seed}_{track}.json')
    rows = [identity(p) for p in sorted(paths)]
    snapshot = BASE/'source_snapshot'
    snapshot.mkdir()
    for p in sorted(paths):
        if p.suffix == '.py':
            target = snapshot/p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
    write(destination, {'status':'REGISTERED_BEFORE_EXECUTION', 'config':CONFIG,
                         'files':rows, 'historical_contract':old['files'], 'source':str(SOURCE)})
    print(f'合同已登记，未运行评价：{destination}', flush=True)


class Run:
    def __init__(self, out, started=None):
        self.out = out
        self.started = time.time() if started is None else started
        self.inputs = {}
        self.ranges = {}
        self.peak_ram = 0.0
        self.guard()

    def guard(self):
        ram = psutil.virtual_memory().percent
        self.peak_ram = max(self.peak_ram, ram)
        if ram >= 85:
            raise RuntimeError(f'RAM_WARNING_85_PERCENT: {ram:.1f}%')
        if time.time()-self.started >= CONFIG['wall_seconds']:
            raise RuntimeError('G5-R1四小时预算到期，不自动续跑')

    def get(self, row):
        self.guard()
        source_path(row['path'])
        self.inputs[row['path']] = row
        return verified_read(row, self.out/'anomalies')

    def preflight(self):
        contract = read(BASE/'contract.json')
        if contract['config'] != CONFIG:
            raise RuntimeError('配置与批准合同不一致')
        for row in contract['files'] + contract['historical_contract']:
            self.guard()
            verified_read(row, self.out/'anomalies')
        self.contract = contract
        self.manifest = read(SOURCE/'manifest.json')
        self.fm = read(SOURCE/'feature_manifest.json')
        if read(SOURCE/'final_audit_report.json')['status'] != 'PASS' or self.manifest['test_executed']:
            raise RuntimeError('原G5证据未通过')
        if self.manifest['config']['training_seeds'] != CONFIG['seeds'] or len(self.manifest['subsets']['val_compare']) != 1024:
            raise RuntimeError('源数/seed/划分合同不符')
        if not torch.cuda.is_available():
            raise RuntimeError('需要原CUDA环境，禁止自动切换CPU')
        for row in self.manifest['inputs']['artifacts']:
            source_path(row['path'])
        for seed in CONFIG['seeds']:
            for track in ('sg','e2e'):
                row = read(SOURCE/f'training/{seed}/{track}/best.identity.json')
                source_path(row['path'])
                prior = read(SOURCE/f'comparison/{seed}_{track}.json')
                if prior['checkpoint'] != row:
                    raise RuntimeError('最佳checkpoint身份与正式G5评价不一致')
        print('预检通过：六轨固定checkpoint；test封存；原G5只读。', flush=True)

    def load_features(self, split):
        if split not in ('val_select','val_compare'):
            raise ValueError('Only approved development splits')
        registry = read(SOURCE/'engineering_v2/index_registry.json')
        index = json.loads(self.get(registry['indexes'][split]))
        files = self.fm['files'][split]
        self.guard()
        cache = SampleRangeCache(CONFIG['cache_bytes'], self.out/'anomalies')
        for name in ('ch3_spatial','d8_e1','d8_d2'):
            parents = {str(source_path(r['path'])): r for r in files[name]}
            for row in index[name]:
                p = source_path(row['path'])
                parent = parents[str(p)]
                if row['parent_sha256'] != parent['sha256'] or not p.is_relative_to(SOURCE/'features'/split):
                    raise RuntimeError('样本范围与父分片登记不符')
                if p.stat().st_size != parent['size_bytes'] or row['offset']+row['size_bytes']>parent['size_bytes']:
                    raise RuntimeError('分片大小/范围变化')
        features = g4.FeatureStore(*(SampleRangeArray(index[n],cache) for n in ('ch3_spatial','d8_e1','d8_d2')))
        targets = g4.Targets(**torch.load(io.BytesIO(self.get(files['targets'])), map_location='cpu', weights_only=False))
        return features, targets, files['metadata'], index, cache

    def metric(self, truth, pred, logits, active, bands, ignore, meta):
        pred = np.asarray(pred, dtype=np.float32).reshape(-1,2)
        g = g4.g1.gospa_sample(truth, pred)
        result = {**meta, 'true_count':len(truth), 'predicted_count':len(active),
                  'predicted_positions_m':pred.tolist(), 'gospa_m':float(g['value_m']),
                  'matched_errors_m':g4.distance_errors(truth,pred), 'duplicate30':duplicate(pred),
                  **association(truth,pred,logits,active,bands,ignore)}
        for name in ('localization','missed','false'):
            result[f'gospa_{name}_p_sum'] = float(g[f'{name}_p_sum'])
        for t in (10,30,50,100):
            result[f'tp_at_{t}m'] = g4.g1.maximum_matches_within(truth,pred,t)
        return result

    def evaluate(self, seed, track, split, indices, label):
        self.guard()
        device = torch.device('cuda:0')
        for artifact in self.manifest['inputs']['artifacts']:
            self.get(artifact)
        context = build_context(self.out,self.manifest,seed,device)
        cp = read(SOURCE/f'training/{seed}/{track}/best.identity.json')
        payload = torch.load(io.BytesIO(self.get(cp)), map_location='cpu', weights_only=False)
        g4.load_state(context,payload['state'])
        del payload
        g4.set_mode(context,training=False)
        features,targets,metadata,index,cache = self.load_features(split)
        old = read(SOURCE/f'comparison/{seed}_{track}.json')['evaluation']['samples'] if split=='val_compare' else None
        rows = []
        path = self.out/f'{label}_samples.jsonl'
        begin = time.time()
        batches_ids = [torch.tensor(indices[i:i+4]) for i in range(0,len(indices),4)]
        iterator = batches(features,batches_ids,prefetch=True)
        try:
            with path.open('x',encoding='utf-8') as handle, torch.no_grad():
                for ids,batch in iterator:
                    self.guard()
                    _,logits,_,heat,offset = forward(context,batch,ids,device,stop_gradient=False)
                    logits,heat,offset = logits.cpu().numpy(),heat.cpu().sigmoid().numpy(),offset.cpu().numpy()
                    for local,i in enumerate(ids.tolist()):
                        for name in ('ch3_spatial','d8_e1','d8_d2'):
                            row = index[name][i]
                            if row['index'] != i:
                                raise RuntimeError('Range order mismatch')
                            self.ranges[(row['path'],row['offset'])] = row
                        count = int(targets.counts[i])
                        truth = targets.positions[i,:count].numpy()
                        bands,ignore = targets.band[i].numpy(),targets.ignore[i].numpy()
                        decoded = decode(logits[local],heat[local],offset[local])
                        record = {'index':i, 'truth':truth.tolist(), 'logits':logits[local].tolist(),
                                  'true_band':bands.tolist(), 'ignore':ignore.tolist(), 'decode':decoded}
                        for mode in ('original','joint'):
                            record[mode] = self.metric(truth,decoded[mode],logits[local],decoded['active'],bands,ignore,metadata[i])
                        if old is not None:
                            prior = old[i]
                            if prior['raw_index'] != metadata[i]['raw_index'] or prior['true_count'] != count:
                                raise RuntimeError('Historical sample identity mismatch')
                            if not np.array_equal(np.asarray(prior['band_logits'],dtype=np.float32),logits[local]):
                                raise RuntimeError('原G5 logits未精确复现，停止，不比较新解码')
                            if not np.allclose(np.asarray(prior['predicted_positions_m']).reshape(-1,2),np.asarray(decoded['original']).reshape(-1,2),atol=5e-5,rtol=0):
                                raise RuntimeError('原G5位置未复现')
                            if abs(prior['gospa_m']-record['original']['gospa_m'])>1e-4:
                                raise RuntimeError('原G5 GOSPA未复现')
                            record['original']['historical_band_only_f1'] = prior['band_only_f1']
                            record['joint']['historical_band_only_f1'] = prior['band_only_f1']
                        record['ceiling'] = candidate_ceiling(truth,decoded['candidates'],logits[local],bands,ignore)
                        handle.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')
                        rows.append(record)
                    handle.flush()
                    elapsed = time.time()-begin
                    done = len(rows)
                    progress = {'stage':label,'done':done,'total':len(indices),'elapsed_seconds':elapsed,
                                'estimated_remaining_seconds':elapsed/done*(len(indices)-done),'ram_percent':psutil.virtual_memory().percent}
                    write(self.out/'progress.json',progress)
                    print(f"[{label}] {done}/{len(indices)}；已用{elapsed/60:.1f}分钟；本轨预计剩余{progress['estimated_remaining_seconds']/60:.1f}分钟；RAM={progress['ram_percent']:.1f}%",flush=True)
        finally:
            iterator.close()
        elapsed = time.time()-begin
        stats = {'status':'COMPLETE','checkpoint':cp,'seconds':elapsed,'rows':len(rows),'cache':cache.stats,
                 'max_cuda_allocated_bytes':torch.cuda.max_memory_allocated(), 'max_cuda_reserved_bytes':torch.cuda.max_memory_reserved()}
        write(self.out/f'{label}_receipt.json',stats)
        del context,features,targets,cache,iterator,batch,logits,heat,offset
        gc.collect()
        torch.cuda.empty_cache()
        return rows,elapsed

    def postcheck(self):
        # Stream only the consumed registered ranges; no reread of unrelated train/test data.
        for n,row in enumerate(self.ranges.values(),1):
            self.guard()
            verified_read(row,self.out/'anomalies',offset=row['offset'],length=row['size_bytes'])
            if n%128==0:
                print(f'[阶段后身份复核] {n}/{len(self.ranges)} 个样本范围',flush=True)
        for row in self.inputs.values():
            self.get(row)
        for row in self.contract['files']+self.contract['historical_contract']:
            self.guard()
            verified_read(row,self.out/'anomalies')
        write(self.out/'input_audit.json',{'status':'PASS','files':list(self.inputs.values()),
              'sample_ranges':list(self.ranges.values()),'same_bytes_verified_and_consumed':True})


def execute(out,started):
    runtime = Run(out,started)
    runtime.preflight()
    start = time.time()
    metadata = runtime.fm['files']['val_select']['metadata']
    pilot = sorted(i for k in range(4) for i in [j for j,r in enumerate(metadata) if r['true_k']==k][:8])
    if len(pilot)!=32:
        raise RuntimeError('Pilot must contain eight samples per K')
    write(out/'protocol.json',{'config':CONFIG,'pilot_indices':pilot,'contract':identity(BASE/'contract.json'),
                             'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0)})
    _,seconds = runtime.evaluate(CONFIG['seeds'][0],'e2e','val_select',pilot,'pilot')
    # Conservative allowance for six tracks, postchecks, statistics and figures; never shrink the experiment.
    estimate = seconds/32*1024*6*1.35+900
    write(out/'pilot_report.json',{'status':'ENGINEERING_PASS','seconds':seconds,'forecast_seconds':estimate,
                                 'remaining_budget_seconds':14400-(time.time()-started)})
    print(f'32条工程核对完成；六轨及收尾保守估计{estimate/3600:.2f}小时（非保证）。',flush=True)
    if estimate > 14400-(time.time()-started):
        raise RuntimeError('FORECAST_EXCEEDS_BUDGET：暂停，禁止缩样本/seed或自动延时')
    results = {}
    for seed in CONFIG['seeds']:
        for track in ('sg','e2e'):
            print(f'开始正式开发评价：{seed}/{track}，不训练',flush=True)
            records,_ = runtime.evaluate(seed,track,'val_compare',list(range(1024)),f'{seed}_{track}')
            results[seed,track] = records
    runtime.guard()
    build_report(results,out)
    # Existing Hard/R5a metrics retain their own 512-sample scope.
    write(out/'hard_reference_readonly.json',{'source':identity(SOURCE/'hard_reference_report.json'),
                                            'reference':read(SOURCE/'hard_reference_report.json'),
                                            'scope':'Original shared K2/K3 reference; not a 1024-scene comparison'})
    runtime.guard()
    plot_cases(results,out)
    runtime.postcheck()
    runtime.guard()
    products = [identity(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ('run.log','final_audit_report.json') and 'anomalies' not in p.parts]
    write(out/'final_audit_report.json',{'status':'PASS','scientific_status':'COMPLETE_FOR_REVIEW',
         'test_executed':False,'training_executed':False,'six_tracks_complete':len(results)==6,
         'historical_replay':'all 6144 logits exact; coordinates within 5e-5m; GOSPA within 1e-4m',
         'seconds':time.time()-started,'evaluation_seconds':time.time()-start,'peak_system_ram_percent':runtime.peak_ram,
         'outputs':products})
    print(f'G5-R1完成。请回读：{out / "运行摘要.md"} 和 comparison_report.json、final_audit_report.json',flush=True)


class Tee:
    def __init__(self,console,log):
        self.console,self.log=console,log
    def write(self,text):
        self.log.write(text)
        self.log.flush()
        self.console.write(text)
        self.console.flush()
        return len(text)
    def flush(self):
        self.log.flush()
        self.console.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='仅4条旧案例工程回放，不作为性能结果')
    parser.add_argument('--started',type=float)
    args=parser.parse_args()
    if args.register:
        register()
        return
    if args.check_only:
        Run(BASE/'precheck').preflight()
        print('仅预检通过；未读取科研特征、未推理、未训练。',flush=True)
        return
    if args.smoke:
        out=BASE/('smoke_'+time.strftime('%Y%m%d_%H%M%S'))
        out.mkdir(exist_ok=False)
        runtime=Run(out)
        runtime.preflight()
        rows,_=runtime.evaluate(20260921,'e2e','val_compare',[770,793,940,1019],'smoke')
        runtime.postcheck()
        write(out/'smoke_report.json',{'status':'ENGINEERING_PASS','samples':len(rows),
              'formal_evaluation':False,'test_executed':False,'training_executed':False})
        print(f'四例工程回放通过：{out}',flush=True)
        return
    if not args.worker or args.started is None:
        raise RuntimeError('请运行项目内的运行入口/E2E/G5_R1/G5_R1一键运行.py')
    out=BASE/'evaluation'
    out.mkdir(exist_ok=False)
    if not out.resolve().is_relative_to((ROOT/'outputs_e2e').resolve()) or out.resolve().is_relative_to(SOURCE.resolve()):
        raise RuntimeError('Output isolation violation')
    with (out/'run.log').open('x',encoding='utf-8') as log:
        stdout,stderr=sys.stdout,sys.stderr
        sys.stdout,sys.stderr=Tee(stdout,log),Tee(stderr,log)
        try:
            print('G5-R1启动：中文日志UTF-8；固定模型；联合选峰；test不读取。',flush=True)
            execute(out,args.started)
        except BaseException as exc:
            write(out/'failure_report.json',{'status':'STOPPED','error':repr(exc),'traceback':traceback.format_exc(),
                  'test_executed':False,'training_executed':False,'elapsed_seconds':time.time()-args.started})
            traceback.print_exc()
            raise
        finally:
            sys.stdout,sys.stderr=stdout,stderr


if __name__=='__main__':
    main()
