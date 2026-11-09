"""Sequential low-priority read-only audit, no generation/physics/replay."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import stat
import subprocess
import sys
import zipfile
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
RUN=REPO/'paper/research/20261104-inspection';BASE=Path('/tmp/inspection-event-scope-pair-review-v1')
os.nice(10)
from eval.inspection_episode_receipts import validate_flight,_bindings
from eval.inspection_population import load_development_case
from eval.inspection_common_calendar import rescore_common_calendar
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
def load(name,p):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ep=load('scoped_endpoint',BASE/'source/inspection_event_endpoints.py')
pair=read(BASE/'source/pair.json');candidate=read(BASE/'source/event-endpoints-ood-B0.candidate.v2.json')
event_root=Path(pair['event_episode']['episode_receipt']).parent.parent
clean_root=Path(pair['clean_episode']['episode_receipt']).parent.parent
event_plan=read(event_root/'plan.json');clean_plan=read(clean_root/'plan.json')
root_checks={'event_exact_one_attempt':[p.name for p in sorted(event_root.glob('attempt-*'))]==['attempt-00'],
    'clean_exact_two_attempts':[p.name for p in sorted(clean_root.glob('attempt-*'))]==['attempt-00','attempt-01'],
    'same17source61dependencies':event_plan['source_hashes']==clean_plan['source_hashes'] and len(event_plan['source_hashes'])==17
        and event_plan['dependency_hashes']==clean_plan['dependency_hashes'] and len(event_plan['dependency_hashes'])==61,
    'same_actor_config_repeat_clock_threads':all(event_plan[k]==clean_plan[k] for k in ('condition','repeat_ordinal','max_attempts','max_clock_lag_seconds','thread_environment'))
        and event_plan['condition']=='B0' and event_plan['max_attempts']==2 and event_plan['max_clock_lag_seconds']==.1}
cases={};worlds={};attempts=[];inputs={}
entry=RUN/'experiment/evaluator.v2.py';evaluator_hash=sha(entry)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
keep={'private_case','case_sha256','score','safety','sensor_audit','delivered_observations','reports','required_cells','coverage_evidence','jobs'}
for label,root,ordinal,plan,expect_admitted in [('event',event_root,0,event_plan,True),('clean_failed',clean_root,0,clean_plan,False),('clean',clean_root,1,clean_plan,True)]:
    attempt=root/f'attempt-{ordinal:02d}';flight=attempt/'flight';terminal=read(attempt/'terminal.json');run=read(flight/'run.json');world=read(flight/'world.json')
    admitted,evidence=validate_flight(flight,plan,terminal['returncode'])
    command=[str(REPO/'.venv/bin/python'),'-B',str(entry),'--world',str(flight/'world.json'),'--expected-world-sha256',terminal['record_hashes']['flight/world.json']]
    ret=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True,timeout=30)
    (BASE/(label+'-frozen.stdout')).write_text(ret.stdout);(BASE/(label+'-frozen.stderr')).write_text(ret.stderr)
    frozen=json.loads(next(line for line in reversed(ret.stdout.splitlines()) if line.startswith('{'))) if ret.returncode==0 else None
    rc={'record_hashes':terminal['record_hashes']==_bindings(attempt),'plan_canonical_binding':terminal['plan_sha256']==canonical(plan),
        'stdout_binding':terminal['stdout_sha256']==sha(attempt/'stdout.log'),
        'fresh_validation_matches_terminal':evidence==terminal['evidence'] and admitted==terminal['Y'],
        'expected_admission':(admitted is not None)==expect_admitted,
        'embedded_sources_match':set(run['sources'])==set(plan['source_hashes']) and all(hashlib.sha256(v.encode()).hexdigest()==plan['source_hashes'][k] for k,v in run['sources'].items()),
        'same_planned_source_dep':run['source_hashes']==plan['source_hashes'] and run['dependency_hashes']==plan['dependency_hashes'],
        'completed600_144k':run['status']==world['status']=='completed' and world['simulated_seconds']==600 and world['physics_steps_executed']==144000,
        'frozen_full_score_exact':ret.returncode==0 and frozen['score_equal'] is True and frozen['score']==world['score'] and frozen['evaluator_sha256']==evaluator_hash}
    if label=='clean_failed':
        rc['only_known_clock_failure']=terminal['status']=='known_operational_failure' and terminal['Y'] is None and evidence['failure_reasons']==['clock_lag_exceeded'] and terminal['retry_allowed'] is True and terminal['retry_budget_remaining']==1
    else:rc['accepted_status_no_retry']=terminal['status']=='completed' and evidence['failure_reasons']==[] and terminal['retry_allowed'] is False
    if label=='clean':rc['retry_consumed_remaining_zero']=terminal['retry_budget_remaining']==0
    if expect_admitted:
        worlds[label]={k:world[k] for k in keep};cases[label]=load_development_case(world['private_case'])
    attempts.append({'label':label,'attempt':str(attempt),'status':terminal['status'],'Y':terminal['Y'],'raw_world_Y':world['score']['Y'],
        'maxlag':world['max_clock_lag_seconds'],'world_sha256':sha(flight/'world.json'),'terminal_sha256':sha(attempt/'terminal.json'),
        'plan_file_sha256':sha(root/'plan.json'),'plan_canonical_sha256':canonical(plan),'checks':rc,'passed':all(rc.values())})
    for p in [root/'plan.json',attempt/'terminal.json',flight/'world.json',flight/'run.json',flight/'private-case.json']:inputs[str(p)]=sha(p)
    print(label,'frozen_score_equal',rc['frozen_full_score_exact'],'admitted',expect_admitted,flush=True)
    del world
geometry_fields=('case_id','lineage_group','split','seed','roi_polygon','wind_xy','entities','schema','domain')
root_checks['same_private_geometry_motion_noise']=all(getattr(cases['event'],k)==getattr(cases['clean'],k) for k in geometry_fields)
calendar_fields=('priority_at','link_drone','link_start','link_end','camera_drone','camera_start','camera_end','retire_drone','retire_at')
root_checks['only_explicit_calendar_and_level_change']=all(getattr(cases['clean'],k) is None for k in calendar_fields) and cases['clean'].level=='L2' and cases['event'].level=='L3'
root_checks['paired_terminal_receipts']=sha(Path(pair['event_episode']['episode_receipt']))==pair['event_episode']['episode_receipt_sha256'] and sha(Path(pair['clean_episode']['episode_receipt']))==pair['clean_episode']['episode_receipt_sha256']
root_checks['paired_semantic_plan_receipts']=canonical(event_plan)==pair['event_episode']['plan_sha256'] and canonical(clean_plan)==pair['clean_episode']['plan_sha256']
common=rescore_common_calendar(worlds['clean'],cases['event'])
root_checks['common_calendar_full_exact']=common==pair['common_calendar_recount'] and common['original_score_exact'] is True
delta=worlds['event']['score']['Y']-common['score']['Y']
root_checks['scalar_delta_exact']=pair['event_Y']==worlds['event']['score']['Y'] and pair['clean_actual_Y']==worlds['clean']['score']['Y'] and pair['clean_common_demand_Y']==common['score']['Y'] and pair['event_minus_clean_common_demand_Y']==delta
new=ep.analyze_event_trace(worlds['event'],cases['event'])
root_checks['actual_candidate_v2_exact']=all(json.dumps(new[k],sort_keys=True)==json.dumps(candidate[k],sort_keys=True) for k in ('role','rows','event_count','case_sha256')) and candidate['source_sha256']==sha(BASE/'source/inspection_event_endpoints.py') and candidate['world_sha256']==attempts[0]['world_sha256']
archive=RUN/'qualification/event-clean-pair-OOD-L3-B0.v1/complete-clean-operation.zip';archive_checks={}
with zipfile.ZipFile(archive) as z:
    mapping=json.loads(z.read('archive-manifest.json'));names=z.namelist()
    archive_checks['exact_unique_members']=len(names)==len(set(names)) and set(names)==set(mapping)|{'archive-manifest.json'}
    archive_checks['safe_members']=all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names) and all(not stat.S_ISLNK(i.external_attr>>16) for i in z.infolist())
    matches={}
    for name,h in mapping.items():
        v=hashlib.sha256()
        with z.open(name) as f:
            for block in iter(lambda:f.read(1024*1024),b''):v.update(block)
        matches[name]=v.hexdigest()==h and (not name.startswith('logs/') or sha(REPO/name)==h)
    archive_checks['every_member_sha_and_raw_original']=all(matches.values())
    derivation=json.loads(z.read('clean-sibling-derivation.json'))
    archive_checks['derivation_bindings']=derivation['original_case_sha256']==cases['event'].digest() and derivation['clean_case_sha256']==cases['clean'].digest() and derivation['original_file_sha256']==event_plan['case_file_sha256'] and derivation['clean_file_sha256']==clean_plan['case_file_sha256'] and derivation['independent_root_count_added']==0
archive_checks['archive_sha_bound']=sha(archive)==pair['archive_sha256'];inputs[str(archive)]=sha(archive)
token=read(BASE/'source/model-context-tokenization.v4.json');paragraph='These stripes partition the ROI by nearest y center (-300,-100,100,300)m;\nequal-distance boundaries belong to the lower index. This is survey responsibility.'
token_checks={'same_actor_source':token['actor_source_sha256']==sha(REPO/'eval/inspection_llm_actor.py'),
    'all_five_contexts_have_same_public_partition':len(token['requests'])==5 and all(paragraph in x['messages'][0]['content'] for x in token['requests']),
    'fifteen_tokenizer_only_counts':len(token['results'])==15 and token['actual_generation_calls']==0 and all(x['input_tokens']<=8192 for x in token['results'])}
doc=RUN/'qualification/event-obligation-scope.v1.md';cand=RUN/'qualification/event-endpoints-ood-B0.candidate.v2.json'
root_checks['scope_registered_before_candidate_file']=doc.stat().st_mtime<=cand.stat().st_mtime
data={'role':'bounded readonly n1 development scope/pair audit; no new physics/model, no main/G4 inference','audit_nice':os.nice(0),
    'root_checks':root_checks,'attempts':attempts,'archive_checks':archive_checks,'tokenizer_only_checks':token_checks,
    'pair_scalars':{'event_Y':worlds['event']['score']['Y'],'clean_actual_Y':worlds['clean']['score']['Y'],'clean_common_Y':common['score']['Y'],'delta':delta},
    'candidate_summary':{'rows':len(new['rows']),'eligible':sum(x['eligible'] for x in new['rows']),'response_censored':sum(x['response']['censored'] for x in new['rows'] if x['eligible']),'recovery_censored':sum(x['recovery']['censored'] for x in new['rows'] if x['eligible'])},
    'tokenizer_counts':[x['input_tokens'] for x in token['results']],'input_sha256':inputs,
    'public_case_field_difference':['level'],'private_calendar_fields_removed':list(calendar_fields)}
data['passed']=all(root_checks.values()) and all(x['passed'] for x in attempts) and all(archive_checks.values()) and all(token_checks.values())
(BASE/'pair-audit.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({k:v for k,v in data.items() if k not in ('input_sha256','attempts','tokenizer_counts')},indent=2),flush=True)
