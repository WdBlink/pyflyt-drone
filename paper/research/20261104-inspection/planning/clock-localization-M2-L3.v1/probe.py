"""One prospective, unscored B3-M2 development L3 clock diagnostic.

Reuses the original actor/coordinator/service. The explicit worker variant
adds clock/rusage telemetry; it never fills or retries an SDK/pilot slot.
"""
import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=next(parent for parent in Path(__file__).resolve().parents if (parent/'envs/inspection/mission.py').is_file())
sys.path.insert(0,str(ROOT))
from eval.inspection_model_receipts import validate_model_operation
from eval.inspection_model_service import operation_directory
from eval.inspection_process_cleanup import SubprocessTree, inventory_diagnostics

ROLE='instrumented_adaptive_B3_M2_existing_development_L3_not_pilot_retry'
CASE='qualification/dev-cases.v2/dev-L3-00.json'
SAMPLER_SEED=1096691202
EXTRA_SOURCES=('paper/research/20261104-inspection/planning/clock-localization-M2-L3.v1/probe.py','eval/inspection_diagnostic_worker.py',
               'eval/inspection_clock_profile.py','eval/inspection_clock_profile_rusage.py',
               'eval/inspection_process_cleanup.py','eval/inspection_darwin_inventory.py')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def admission(run,reservation):
    registry=run/'planning/local-model-registry.v1.json'
    model=json.loads(registry.read_text())['models']['M2'];receipt=run/model['download_receipt']
    if sha(receipt)!=model['download_receipt_sha256']:raise ValueError('frozen model receipt changed')
    owner=json.loads((run/'.paper/control.json').read_text());entry=owner['usage_ledger'][reservation]
    purpose=json.loads(entry['purpose']);limits=model['generation_profile']
    required={'kind':'inspection_local_generation','phase':'development','diagnostic_role':ROLE,
        'model_registry_sha256':sha(registry),'model_receipt_sha256':sha(receipt),
        'case_file_sha256':sha(run/CASE),'sampler_seed':SAMPLER_SEED,
        'worker_variant_source_hashes':{str(ROOT/p):sha(ROOT/p) for p in EXTRA_SOURCES}}
    if (entry.get('status')!='reserved' or any(purpose.get(k)!=v for k,v in required.items())
            or any(type(entry['reserved'].get(k)) is not int or entry['reserved'][k]<limits[b]
                   for k,b in [('calls','max_calls'),('tokens','max_total_tokens')])):
        raise ValueError('specific sufficient live diagnostic reservation required')
    # Shared service independently checks accepted-plan/registry/active-owner
    # binding before model import/load and every real generation.
    return {'reservation':reservation,'owner_frozen_plan':owner['frozen_plan'],
        'model_receipt_sha256':sha(receipt),'service_sha256':sha(ROOT/'eval/inspection_model_service.py'),
        'reserved':entry['reserved'],'generation_profile':limits}


class ObservedTree(SubprocessTree):
    def __init__(self):
        self.observations=[];self.overflow=0
        super().__init__()
    def observe(self,current=None):
        start,cpu=time.monotonic(),time.process_time()
        try:return super().observe(current)
        finally:
            value={'start_monotonic':start,'wall_seconds':time.monotonic()-start,
                   'process_cpu_seconds':time.process_time()-cpu}
            if len(self.observations)<12000:self.observations.append(value)
            else:self.overflow+=1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--reservation',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--instrumented-child',action='store_true')
    args=parser.parse_args();args.run=args.run.resolve();args.out=args.out.resolve()
    if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')):
        parser.error('all four declared thread environment variables must be1')
    expected=admission(args.run,args.reservation)
    if args.instrumented_child:
        from eval import run_inspection_llm_development as coordinator
        from eval.inspection_diagnostic_worker import worker
        from eval.inspection_clock_profile_rusage import RusageClockProfile
        coordinator.worker=partial(worker,profile_factory=RusageClockProfile)
        sys.argv=[str(ROOT/'eval/run_inspection_llm_development.py'),'--run',str(args.run),
            '--reservation',args.reservation,'--model','M2','--architecture','B3',
            '--case',str(args.run/CASE),'--sampler-seed',str(SAMPLER_SEED),'--out',str(args.out/'flight')]
        return coordinator.main()
    if args.out.exists():parser.error('fresh engineering output required; no retries')
    args.out.mkdir(parents=True)
    sources={str(ROOT/p):sha(ROOT/p) for p in EXTRA_SOURCES}
    result={'role':ROLE,'no_SDK_or_pilot_slot_admission':True,'sampler_seed':SAMPLER_SEED,
        'case_file_sha256':sha(args.run/CASE),'extra_worker_variant_sources':sources,
        'expected_model_meter':expected,'started_monotonic':time.monotonic(),
        'status':'intent_only','errors':[],
        'supervision_scope':'same existing SubprocessTree .1s observation loop, plus bounded wall/CPU observations; not a kernel subreaper'}
    def save():(args.out/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    save();watch=ObservedTree();process=awake=None
    try:
        awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
        result['awake_assertion']={'pid':awake.pid,'supervisor_pid':os.getpid(),'flags':'-i -w'}
        with (args.out/'coordinator.stdout-stderr.log').open('xb') as stream:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                '--instrumented-child','--run',str(args.run),'--reservation',args.reservation,'--out',str(args.out)],
                stdout=stream,stderr=subprocess.STDOUT)
            watch.attach(process);result['coordinator_pid']=process.pid
            result['returncode']=watch.wait(timeout=900)
        if result['returncode']!=0:raise RuntimeError('instrumented coordinator failed')
        run=json.loads((args.out/'flight/run.json').read_text());world=json.loads((args.out/'flight/world.json').read_text())
        if (run['status']!='completed' or not all(run.get(k) is True for k in
                ('private_case_unchanged','sources_unchanged','dependencies_unchanged'))
                or world['status']!='completed' or world['simulated_seconds']!=600
                or world['physics_steps_executed']!=144000 or world['clock_profile']['rows']!=72000
                or world['clock_profile']['rusage_profile']['rows']!=72000):
            raise ValueError('complete unchanged engineering denominator required')
        result['resource_recount']=validate_model_operation(operation_directory(args.reservation),expected)
        result['max_clock_lag_seconds']=world['max_clock_lag_seconds']
        result['lag_threshold_exceeded']=world['max_clock_lag_seconds']>.1
        result['status']='completed_engineering_not_qualification'
    except BaseException:
        result['errors'].append(traceback.format_exc());result['status']='operational_failure_or_unknown'
    finally:
        if process is not None:
            try:
                result['supervision']=watch.close()
                if result['supervision']['status']!='closed':raise RuntimeError('supervision cleanup unknown')
            except BaseException:result['errors'].append(traceback.format_exc())
        result['supervisor_observations']=watch.observations
        result['supervisor_observation_overflow']=watch.overflow
        result['supervisor_inventory']=inventory_diagnostics()
        if awake is not None and awake.poll() is None:
            try:awake.terminate();awake.wait(timeout=10)
            except BaseException:result['errors'].append(traceback.format_exc())
        try:
            result['worker_variant_sources_unchanged']=all(sha(p)==h for p,h in sources.items())
            result['original_case_unchanged']=sha(args.run/CASE)==result['case_file_sha256']
        except BaseException:
            result['errors'].append(traceback.format_exc())
            result['worker_variant_sources_unchanged']=result['original_case_unchanged']=False
        if result['errors'] or not result['worker_variant_sources_unchanged'] or not result['original_case_unchanged']:
            result['status']='operational_failure_or_unknown'
        result['finished_monotonic']=time.monotonic();save()
    print(json.dumps({'status':result['status'],'max_clock_lag_seconds':result.get('max_clock_lag_seconds'),
        'usage':result.get('resource_recount',{}).get('usage')}))
    return 0 if result['status']=='completed_engineering_not_qualification' else 1


if __name__=='__main__':raise SystemExit(main())
