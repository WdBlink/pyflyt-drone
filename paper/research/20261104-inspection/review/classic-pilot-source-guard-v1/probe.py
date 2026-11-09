"""Last source guard probes in isolated SDK/JSON fixtures; no physics CLI."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[5]
RESEARCH=REPO/'paper/research/20261104-inspection'
sys.path.insert(0,str(REPO))
from eval.inspection_episode_receipts import digest,validate_flight


def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')


def stable_audit():
    store=REPO/'logs/inspection-classic-dev-flights.v1';recording=REPO/'logs/inspection-classic-dev-recording.v2'
    results=json.loads((recording/'results.json').read_text());rows=results['runs']
    assert results['execution_status']=='failed' and len(rows)==results['reported_denominator']==18
    assert sum(r['status']=='completed' for r in rows)==1 and sum(r['status']=='failed' for r in rows)==1
    assert sum(r['status']=='stopped' and r.get('attempted') is False for r in rows)==16
    assert not (recording/'selection.json').exists()
    frozen=json.loads((recording/'source-hashes.json').read_text());deps=json.loads((recording/'dependency-hashes.json').read_text());schedule=json.loads((recording/'schedule.json').read_text())
    receipts=[]
    for condition in ('B0','B2'):
        root=store/f'dev-L3-00-{condition}-0';terminal_path=root/'attempt-00/terminal.json';t=json.loads(terminal_path.read_text());plan=json.loads((root/'plan.json').read_text())
        assert t['status']=='completed' and t['returncode']==0 and len(list(root.glob('attempt-*')))==1
        assert all(digest(terminal_path.parent/p)==h for p,h in t['record_hashes'].items())
        assert all(frozen[p]==h for p,h in plan['source_hashes'].items())
        assert plan['supervisor_sha256']==frozen[str(REPO/'eval/run_inspection_dev.py')]
        assert plan['receipt_adapter_sha256']==frozen[str(REPO/'eval/inspection_episode_receipts.py')]
        assert plan['dependency_hashes']==deps and plan['suite_manifest_sha256']==schedule['suite_sha256']
        y,e=validate_flight(root/'attempt-00/flight',plan,t['returncode']);assert y==t['Y'] and e==t['evidence']
        receipts.append(dict(condition=condition,terminal_sha256=digest(terminal_path),Y=y,physical_attempts=1,worker_source_count=len(plan['source_hashes']),source_and_versions_match_generation2=True))
    invocation=json.loads((recording/'invocations/01.json').read_text());response=json.loads(invocation['stdout'].splitlines()[-1])
    assert invocation['returncode']==0 and response['episode_receipt_sha256']==receipts[1]['terminal_sha256'] and response['actual_attempts']==1
    return dict(role='read_only_completed_v2_evidence_not_V3_execution',generation2_results_sha256=digest(recording/'results.json'),
        full_denominator=18,completed=1,failed=1,unstarted=16,selection_exists=False,receipts=receipts,V3_executed_by_review=False)


def main():
    root=Path(__file__).parent/'results';root.mkdir(exist_ok=False)
    path=REPO/'eval/run_inspection_classic_pilot.py';driver=path.read_text();driver_sha=digest(path)
    assert driver_sha=='b59263b31b413caaee0924600e1e8974b104ee47a15199ca1d6e89ad36267464'
    save(root/'driver-source-snapshot.json',dict(source=driver,sha256=driver_sha))
    previous=RESEARCH/'review/classic-pilot-driver-repair-v1/probe.py';helper=previous.read_text()
    anchor="        p=data/'plans'/f'{case}-{condition}.json';p.write_text(json.dumps(plan))"
    assert helper.count(anchor)==1
    injected="""        if n==18 and mode=='supervisor-mismatch':plan['supervisor_sha256']='0'*64
        if n==18 and mode=='adapter-mismatch':plan['receipt_adapter_sha256']='0'*64
        if n==18 and mode=='source-map-mismatch':
            key=next(iter(plan['source_hashes']));plan['source_hashes'][key]='0'*64
"""
    helper=helper.replace(anchor,injected+anchor)
    (root/'fixture-helper.py').write_text(helper)
    namespace={'__file__':str(previous),'__name__':'isolated_source_guard_fixture'}
    exec(compile(helper,str(previous),'exec'),namespace)
    setup,run=namespace['setup'],namespace['run'];summaries={}
    base=root/'normal';config=setup(base,driver)
    summaries['normal']=run(base,config,'recording-a','normal',{'B0':0.,'B1':.5,'B2':.5+5e-13})
    summaries['cache']=run(base,config,'recording-b','normal',{'B0':0.,'B1':.5,'B2':.5+5e-13})
    assert summaries['normal']['completed']==18 and summaries['normal']['selection']['classic_best']=='B1'
    assert summaries['cache']['completed']==18 and summaries['cache']['synthetic_workers']==18 and summaries['cache']['post_cache_audits']==18
    for mode in ('source-ABA','source-map-mismatch','supervisor-mismatch','adapter-mismatch','suite-final-race','dependency-drift','receipt-chain-race','operational-failure'):
        base=root/mode;config=setup(base,driver);summaries[mode]=run(base,config,'recording',mode,{'B0':.1,'B1':.4,'B2':.6})
        assert summaries[mode]['rows']==summaries[mode]['denominator']==18 and summaries[mode]['selection'] is None
        assert summaries[mode]['sdk_execution_status']=='failed'
        expected=0 if mode in ('receipt-chain-race','operational-failure') else 17
        assert summaries[mode]['completed']==expected
    stable=stable_audit();save(root/'stable-readonly-audit.json',stable)
    report=dict(role='independent_synthetic_source_guard_review_not_physical_pilot_results',utc=datetime.now(timezone.utc).isoformat(),
        driver_sha256=driver_sha,fixture_helper_original_sha256=digest(previous),fixture_helper_used_sha256=digest(root/'fixture-helper.py'),
        source_guard_scope='exact worker map and frozen supervisor/adapter versions; no live modifications',summaries=summaries,stable=stable,script_sha256=digest(Path(__file__)))
    save(root/'report.json',report);print(json.dumps(summaries,indent=2));print(json.dumps(stable,indent=2))


if __name__=='__main__':main()
