"""Independent current driver probes in copied sources/dependencies; no physics."""
from contextlib import redirect_stdout
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from types import SimpleNamespace

REPO=Path(__file__).resolve().parents[5]
RESEARCH=REPO/'paper/research/20261104-inspection'
sys.path.insert(0,str(REPO))
from eval.inspection_episode_receipts import digest,execute_local_episode

OLD=runpy.run_path(str(RESEARCH/'review/classic-pilot-driver-v1/probe.py'))
WORKER=OLD['WORKER']
SOURCE_FILES=('run_inspection_reference.py','run_inspection_dev.py','inspection_cases.py','inspection_scenarios.py','inspection_classic.py','inspection_episode_receipts.py')


def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def setup(base,driver_text):
    base.mkdir();suite=OLD['setup'](base/'data')
    sandbox=base/'repo';(sandbox/'envs/inspection').mkdir(parents=True);(sandbox/'eval').mkdir()
    for p in (REPO/'envs/inspection').glob('*.py'):shutil.copyfile(p,sandbox/'envs/inspection'/p.name)
    shutil.copyfile(REPO/'envs/__init__.py',sandbox/'envs/__init__.py')
    for name in SOURCE_FILES:shutil.copyfile(REPO/'eval'/name,sandbox/'eval'/name)
    driver_path=base/'driver-snapshot.py';driver_path.write_text(driver_text)
    spec=importlib.util.spec_from_file_location('review_pilot_repair_'+base.name.replace('-','_'),driver_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=sandbox
    package=base/'dependencies/PyFlyt';package.mkdir(parents=True);(package/'__init__.py').write_text('# synthetic dependency\n')
    for i in range(59):(package/f'asset-{i:02d}.yaml').write_text(f'synthetic_asset: {i}\n')
    bullet=base/'dependencies/pybullet.synthetic.so';bullet.write_bytes(b'synthetic dependency; never loaded')
    return module,suite,sandbox,package,bullet


def run(base,config,outname,mode,values):
    module,suite,sandbox,package,bullet=config;sdk=RESEARCH/'runtime/autoresearch-state-reader-v1'
    data=base/'data';store=data/'flights';out=data/outname;plans=[];calls=[];post_cache_audits=0
    def sources(condition):
        ps=[*sorted((sandbox/'envs/inspection').glob('*.py')),sandbox/'envs/__init__.py',sandbox/'eval/run_inspection_reference.py',sandbox/'eval/inspection_cases.py',sandbox/'eval/inspection_scenarios.py']
        if condition!='B0':ps.append(sandbox/'eval/inspection_classic.py')
        return {str(p):digest(p) for p in ps}
    def deps():return {str(p):digest(p) for p in [*package.rglob('*.py'),*package.rglob('*.yaml'),bullet]}
    expected_sources={c:sources(c) for c in ('B0','B1','B2')}
    original_run,original_spec,original_exec=module.subprocess.run,module.importlib.util.find_spec,module.execute_local_episode
    original_argv=sys.argv[:]
    def find_spec(name,*a,**kw):
        if name=='PyFlyt':return SimpleNamespace(origin=str(package/'__init__.py'))
        if name=='pybullet':return SimpleNamespace(origin=str(bullet))
        return original_spec(name,*a,**kw)
    def audit(*a,**kw):
        nonlocal post_cache_audits
        post_cache_audits+=1
        return original_exec(*a,**kw)
    def fake_cli(argv,**kw):
        assert Path(argv[1])==sandbox/'eval/run_inspection_dev.py'
        case=argv[argv.index('--case-id')+1];condition=argv[argv.index('--policy')+1];calls.append((case,condition));n=len(calls)
        if mode=='operational-failure':return subprocess.CompletedProcess(argv,7,'synthetic error','')
        manifest_path=suite/'manifest.json';manifest=json.loads(manifest_path.read_text());row=next(x for x in manifest['records'] if x['case_id']==case)
        restore=None
        if n==18 and mode=='suite-final-race':
            p=suite/row['file'];p.write_text(json.dumps({'role':'changed synthetic case','case_id':case}));row['file_sha256']=row['case_sha256']=digest(p);manifest['changed']=True;manifest_path.write_text(json.dumps(manifest))
        if n==18 and mode=='dependency-drift':(package/'asset-00.yaml').write_text('changed synthetic dependency\n')
        if n==18 and mode in ('source-final-race','source-ABA'):
            p=sandbox/'envs/inspection/__init__.py';before=p.read_bytes();p.write_bytes(before+b'\n# synthetic source changed during child\n')
            if mode=='source-ABA':restore=(p,before)
        plan=dict(role='synthetic driver repair fixture; not physical evidence',condition=condition,case_id=case,repeat_ordinal=0,max_attempts=2,max_clock_lag_seconds=.1,
            source_hashes=sources(condition),dependency_hashes=deps(),case_file_sha256=row['file_sha256'],case_sha256=row['case_sha256'],suite_manifest_sha256=digest(manifest_path),
            thread_environment={k:'1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
            supervisor_sha256=digest(sandbox/'eval/run_inspection_dev.py'),receipt_adapter_sha256=digest(sandbox/'eval/inspection_episode_receipts.py'))
        p=data/'plans'/f'{case}-{condition}.json';p.write_text(json.dumps(plan))
        y,evidence=execute_local_episode(store/f'{case}-{condition}-0',plan,
            lambda flight:[sys.executable,str(data/'worker.py'),str(flight),str(p),str(suite/row['file']),str(data/'worker-count'),str(values[condition])],timeout_seconds=2)
        if restore:restore[0].write_bytes(restore[1])
        if n==1 and mode=='receipt-chain-race':(Path(evidence['episode_receipt']).parent/'stdout.log').write_text('changed after CLI response\n')
        plans.append(plan)
        return subprocess.CompletedProcess(argv,0,json.dumps({'Y':y,**evidence})+'\n','')
    module.subprocess.run=fake_cli;module.importlib.util.find_spec=find_spec;module.execute_local_episode=audit
    sys.argv=['run_inspection_classic_pilot.py','--suite',str(suite),'--sdk-root',str(sdk),'--flight-receipts',str(store),'--out',str(out)]
    output=io.StringIO();error=None
    try:
        with redirect_stdout(output):module.main()
    except BaseException as e:error=type(e).__name__+': '+str(e)
    finally:module.subprocess.run=original_run;module.importlib.util.find_spec=original_spec;module.execute_local_episode=original_exec;sys.argv=original_argv
    (base/(outname+'-stdout.jsonl')).write_text(output.getvalue())
    results=json.loads((out/'results.json').read_text());selection=json.loads((out/'selection.json').read_text()) if (out/'selection.json').exists() else None
    source_mismatch=sum(p['source_hashes']!=expected_sources[p['condition']] for p in plans)
    summary=dict(mode=mode,error=error,physics_cli_executed=0,logical_cli_calls=len(calls),post_cache_audits=post_cache_audits,
        synthetic_workers=int((data/'worker-count').read_text()) if (data/'worker-count').exists() else 0,
        denominator=results['reported_denominator'],rows=len(results['runs']),attempted=results['attempted_count'],completed=sum(x['status']=='completed' for x in results['runs']),
        sdk_execution_status=results['execution_status'],selection=selection,dependency_count=len(json.loads((out/'dependency-hashes.json').read_text())),
        receipt_source_maps_differing_from_initial=source_mismatch)
    save(base/(outname+'-summary.json'),summary);return summary


def stable_live_audit():
    old=RESEARCH/'qualification/classic-pilot-generation1.v1';r=json.loads((old/'results.json').read_text());meaning=json.loads((old/'meaning.json').read_text())
    rows=r['runs'];assert r['execution_status']=='failed' and len(rows)==r['reported_denominator']==18
    assert sum(x['status']=='failed' and x.get('attempted') for x in rows)==1
    assert sum(x['status']=='stopped' and x.get('attempted') is False for x in rows)==17
    assert not (old/'selection.json').exists()
    terminal_path=Path(meaning['original_terminal']);terminal=json.loads(terminal_path.read_text());plan=json.loads((terminal_path.parent.parent/'plan.json').read_text())
    assert digest(terminal_path)==meaning['terminal_sha256'] and terminal['status']=='completed' and terminal['returncode']==0
    assert len(list(terminal_path.parent.parent.glob('attempt-*')))==1
    for p,h in terminal['record_hashes'].items():assert digest(terminal_path.parent/p)==h
    from eval.inspection_episode_receipts import validate_flight
    y,e=validate_flight(terminal_path.parent/'flight',plan,terminal['returncode']);assert y==terminal['Y'] and e==terminal['evidence']
    raw=[];live=REPO/'logs/inspection-classic-dev-recording.v2'
    for p in sorted((live/'raw').glob('*.json')):
        try:d=json.loads(p.read_text())
        except (ValueError,OSError):continue
        if d.get('status')=='completed':
            raw.append(dict(path=str(p),sha256=digest(p),case_id=d['case_id'],condition=d['condition'],value=d['value'],cache_reused=d['evidence'].get('cache_reused'),episode_receipt_sha256=d['evidence'].get('episode_receipt_sha256')))
    return dict(role='read_only_stable_evidence_and_completed_v2_slots_not_matrix_completion',generation1_results_sha256=digest(old/'results.json'),generation1_full_denominator=18,
        generation1_failed=1,generation1_unstarted=17,actual_completed_terminal_sha256=digest(terminal_path),actual_Y=y,actual_attempts=1,
        v2_completed_raw_observed=raw,v2_selection_exists=(live/'selection.json').exists(),v2_results_exists=(live/'results.json').exists())


def main():
    root=Path(__file__).parent/'results';root.mkdir(exist_ok=False)
    source=REPO/'eval/run_inspection_classic_pilot.py';text=source.read_text();source_hash=digest(source)
    save(root/'driver-source-snapshot.json',{'path':str(source),'source':text,'sha256':source_hash})
    summaries={};base=root/'normal';config=setup(base,text)
    summaries['normal']=run(base,config,'recording-a','normal',{'B0':0.,'B1':.5,'B2':.5+5e-13})
    summaries['cache-reuse']=run(base,config,'recording-b','normal',{'B0':0.,'B1':.5,'B2':.5+5e-13})
    assert summaries['normal']['completed']==18 and summaries['cache-reuse']['synthetic_workers']==18
    assert summaries['cache-reuse']['post_cache_audits']==18
    for mode in ('operational-failure','suite-final-race','dependency-drift','source-final-race','receipt-chain-race','source-ABA'):
        base=root/mode;config=setup(base,text);summaries[mode]=run(base,config,'recording',mode,{'B0':.1,'B1':.4,'B2':.6})
    for mode in ('suite-final-race','dependency-drift','source-final-race'):
        assert summaries[mode]['completed']==17 and summaries[mode]['rows']==18 and summaries[mode]['selection'] is None
    assert summaries['receipt-chain-race']['completed']==0 and summaries['receipt-chain-race']['rows']==18
    assert summaries['source-ABA']['completed']==18 and summaries['source-ABA']['receipt_source_maps_differing_from_initial']==1
    stable=stable_live_audit();save(root/'stable-readonly-audit.json',stable)
    save(root/'report.json',dict(role='independent_synthetic_driver_repair_audit_not_physical_results',utc=datetime.now(timezone.utc).isoformat(),
        driver_sha256=source_hash,snapshotted_source_tested_not_modified_live_source=True,summaries=summaries,stable_readonly=stable,script_sha256=digest(Path(__file__))))
    print(json.dumps(summaries,indent=2));print(json.dumps(stable,indent=2))


if __name__=='__main__':main()
