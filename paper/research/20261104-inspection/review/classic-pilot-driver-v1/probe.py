"""Pure-SDK/JSON pilot-driver v1 probes; real physics CLI is intercepted."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[5]
RESEARCH=REPO/'paper/research/20261104-inspection'
sys.path.insert(0,str(REPO))
from eval.inspection_episode_receipts import execute_local_episode,digest

WORKER=r'''
import hashlib,json,sys
from pathlib import Path
out,plan_path,case_path,counter_path=map(Path,sys.argv[1:5]);value=float(sys.argv[5])
plan=json.loads(plan_path.read_text());out.mkdir();case=case_path.read_bytes()
counter_path.write_text(str(int(counter_path.read_text())+1 if counter_path.exists() else 1))
(out/'private-case.json').write_bytes(case)
world={'status':'completed','clock_mode':'wallclock_1x','simulated_seconds':600,'physics_steps_executed':144000,'physics_hz':240,'control_hz':120,'max_clock_lag_seconds':.02,'private_case_file_sha256':plan['case_file_sha256'],'score':{'C':value,'S':value,'D':value,'Q':value,'Y':value,'H':1,'E':1,'SR':int(value>=.8)}}
(out/'world.json').write_text(json.dumps(world))
run={'status':'completed','source_hashes':plan['source_hashes'],'dependency_hashes':plan['dependency_hashes'],'sources_unchanged':True,'dependencies_unchanged':True,'private_case_sha256':hashlib.sha256(case).hexdigest(),'private_case_unchanged':True,'policy':plan['condition'],'world_sha256':hashlib.sha256((out/'world.json').read_bytes()).hexdigest(),'requested_seconds':600,'realtime':True,'runtime_thread_environment':plan['thread_environment']}
(out/'run.json').write_text(json.dumps(run))
'''


def save(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2,allow_nan=False);f.write('\n')


def load_driver():
    path=RESEARCH/'qualification/source/classic-pilot-driver-v1/eval/run_inspection_classic_pilot.py'
    assert digest(path)=='f03ad63c236f267ef59f6ce28d753d0ab9b178415f7193915411b11936ae0d5e'
    spec=importlib.util.spec_from_file_location('review_classic_pilot_v1',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ROOT=REPO
    return module,path


def setup(base):
    base.mkdir();suite=base/'suite';suite.mkdir();plans=base/'plans';plans.mkdir()
    records=[]
    for level in (1,2,3):
        for i in (0,1):
            case=f'dev-L{level}-{i:02d}';p=suite/(case+'.json')
            p.write_text(json.dumps({'role':'synthetic fixture; never a CaseSpec or flight','case_id':case}))
            records.append(dict(case_id=case,lineage_group=case,first_pilot=True,file=p.name,file_sha256=digest(p),case_sha256=digest(p)))
    (suite/'manifest.json').write_text(json.dumps(dict(split='dev',main_and_holdout_instantiated=False,records=records)))
    (base/'worker.py').write_text(WORKER);(base/'dependency-fixture.bin').write_bytes(b'initial synthetic dependency')
    return suite


def run(module,base,suite,outname,mode,values):
    sdk=RESEARCH/'runtime/autoresearch-state-reader-v1';store=base/'flights';out=base/outname
    calls=[];plans=[];original_run=module.subprocess.run;original_argv=sys.argv[:]
    physics_executed=0
    def fake_cli(argv,**kwargs):
        assert Path(argv[1]).resolve()==REPO/'eval/run_inspection_dev.py'
        case=argv[argv.index('--case-id')+1];condition=argv[argv.index('--policy')+1]
        calls.append((case,condition));number=len(calls)
        if mode=='operational-failure':return subprocess.CompletedProcess(argv,7,'synthetic supervisor failure','')
        if mode=='interrupt':raise KeyboardInterrupt('synthetic driver interruption, no physics')
        manifest_path=suite/'manifest.json';manifest=json.loads(manifest_path.read_text())
        row=next(r for r in manifest['records'] if r['case_id']==case)
        if number==18 and mode=='suite-final-race':
            case_file=suite/row['file'];case_file.write_text(json.dumps({'role':'changed synthetic input after driver precheck','case_id':case}))
            row['file_sha256']=row['case_sha256']=digest(case_file)
            manifest['revision']='changed during final CLI handoff';manifest_path.write_text(json.dumps(manifest))
        if number==18 and mode=='dependency-drift':(base/'dependency-fixture.bin').write_bytes(b'changed synthetic dependency')
        source_paths=[*sorted((REPO/'envs/inspection').glob('*.py')),REPO/'envs/__init__.py',REPO/'eval/run_inspection_reference.py',REPO/'eval/inspection_cases.py',REPO/'eval/inspection_scenarios.py']
        if condition!='B0':source_paths.append(REPO/'eval/inspection_classic.py')
        plan=dict(role='synthetic SDK admission fixture; not physical evidence',condition=condition,case_id=case,repeat_ordinal=0,max_attempts=2,max_clock_lag_seconds=.1,
            source_hashes={str(p):digest(p) for p in source_paths},dependency_hashes={str(base/'dependency-fixture.bin'):digest(base/'dependency-fixture.bin')},
            case_file_sha256=row['file_sha256'],suite_manifest_sha256=digest(manifest_path),
            thread_environment={k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            supervisor_sha256=digest(REPO/'eval/run_inspection_dev.py'),receipt_adapter_sha256=digest(REPO/'eval/inspection_episode_receipts.py'))
        plan_path=base/'plans'/f'{case}-{condition}.json';plan_path.write_text(json.dumps(plan))
        y,evidence=execute_local_episode(store/f'{case}-{condition}-0',plan,
            lambda flight:[sys.executable,str(base/'worker.py'),str(flight),str(plan_path),str(suite/row['file']),str(base/'worker-count'),str(values[condition])],timeout_seconds=2)
        if number==1 and mode=='receipt-chain-race':
            (Path(evidence['episode_receipt']).parent/'stdout.log').write_text('post-CLI corruption of bound stdout\n')
        plans.append(plan)
        return subprocess.CompletedProcess(argv,0,json.dumps({'Y':y,**evidence})+'\n','')
    module.subprocess.run=fake_cli
    sys.argv=['run_inspection_classic_pilot.py','--suite',str(suite),'--sdk-root',str(sdk),'--flight-receipts',str(store),'--out',str(out)]
    captured=io.StringIO();error=None
    try:
        with redirect_stdout(captured):module.main()
    except BaseException as e:error=type(e).__name__+': '+str(e)
    finally:module.subprocess.run=original_run;sys.argv=original_argv
    (base/(outname+'-stdout.jsonl')).write_text(captured.getvalue())
    results=json.loads((out/'results.json').read_text())
    selection=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else None
    record_hash_errors=0
    for p in store.glob('*/attempt-*/terminal.json'):
        terminal=json.loads(p.read_text())
        for name,h in terminal['record_hashes'].items():
            if h is not None and digest(p.parent/name)!=h:record_hash_errors+=1
    summary=dict(mode=mode,error=error,logical_cli_calls=len(calls),physics_cli_executed=physics_executed,
        synthetic_worker_process_calls=int((base/'worker-count').read_text()) if (base/'worker-count').exists() else 0,
        sdk_reported_denominator=results['reported_denominator'],sdk_rows=len(results['runs']),sdk_execution_status=results['execution_status'],
        completed=sum(r['status']=='completed' for r in results['runs']),attempted=results['attempted_count'],selection=selection,
        suite_hashes_in_plans=sorted({p['suite_manifest_sha256'] for p in plans}),dependency_hash_sets=len({json.dumps(p['dependency_hashes'],sort_keys=True) for p in plans}),
        mismatching_receipt_record_hashes=record_hash_errors)
    save(base/(outname+'-probe-summary.json'),summary)
    return summary


def main():
    root=Path(__file__).parent/'results';root.mkdir(exist_ok=False)
    module,path=load_driver();sdk=RESEARCH/'runtime/autoresearch-state-reader-v1'
    snapshots={str(path):path.read_text(),str(sdk/'lib/experiments.py'):(sdk/'lib/experiments.py').read_text()}
    save(root/'source-snapshot.json',snapshots)
    summaries={}
    base=root/'baseline';suite=setup(base);manifest=json.loads((suite/'manifest.json').read_text())
    assert module.development_schedule(manifest)==module.development_schedule({**manifest,'records':list(reversed(manifest['records']))})
    summaries['baseline']=run(module,base,suite,'recording-a','normal',{'B0':0.,'B1':.5,'B2':.5+5e-13})
    assert summaries['baseline']['selection']['classic_best']=='B1' and summaries['baseline']['completed']==18
    summaries['cache-reuse']=run(module,base,suite,'recording-b','normal',{'B0':0.,'B1':.5,'B2':.5+5e-13})
    assert summaries['cache-reuse']['synthetic_worker_process_calls']==18
    assert all(json.loads(x)['cache_reused'] for x in (base/'recording-b-stdout.jsonl').read_text().splitlines())
    for mode in ('operational-failure','interrupt','suite-final-race','dependency-drift','receipt-chain-race'):
        base=root/mode;suite=setup(base)
        summaries[mode]=run(module,base,suite,'recording',mode,{'B0':.1,'B1':.4,'B2':.6})
    assert summaries['operational-failure']['sdk_rows']==18 and summaries['operational-failure']['completed']==0
    assert summaries['interrupt']['sdk_rows']==18 and summaries['interrupt']['completed']==0
    assert summaries['suite-final-race']['completed']==18 and len(summaries['suite-final-race']['suite_hashes_in_plans'])==2
    assert summaries['dependency-drift']['completed']==18 and summaries['dependency-drift']['dependency_hash_sets']==2
    assert summaries['receipt-chain-race']['completed']==18 and summaries['receipt-chain-race']['mismatching_receipt_record_hashes']==1
    save(root/'report.json',dict(role='independent_synthetic_SDK_driver_probe_not_physical_or_pilot_results',utc=datetime.now(timezone.utc).isoformat(),
        driver_snapshot_sha256=digest(path),sdk_experiments_sha256=digest(sdk/'lib/experiments.py'),script_sha256=digest(Path(__file__)),summaries=summaries))
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
