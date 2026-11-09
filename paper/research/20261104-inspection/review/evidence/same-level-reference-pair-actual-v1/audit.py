"""Sequential low-priority evidence audit; no physics, replay, model or generation."""
from collections import Counter
import hashlib,json,os,stat,subprocess,sys,zipfile
from pathlib import Path,PurePosixPath
from datetime import datetime
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
R=REPO/'paper/research/20261104-inspection'; D=R/'qualification/dev-same-level-event-pair.v2'
OUT=Path(__file__).resolve().parent
os.nice(10)
read=lambda p:json.loads(Path(p).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def semantic(v):return hashlib.sha256(json.dumps(v,sort_keys=True,allow_nan=False).encode()).hexdigest()
assert sha(D/'pair.json')=='6ff386e880379d9b88e3382bb6af9714e1365c0eb30e187f5bc7a6819253ac84'
assert sha(D/'binding-final.json')=='4a240b353b796fa0433edb3c04f6c0cf5f0b0b6b923b34129f6c86506c295508'
pair=read(D/'pair.json');binding=read(D/'binding-final.json')
assert all(sha(D/name)==digest for name,digest in binding['files'].items())
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
entry=R/'experiment/evaluator.v2.py';assert sha(entry)=='de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b'
rows={};worlds={};runs={};plans={};inputs={};commands=[]
for label,summary in pair['flights'].items():
    root=Path(summary['root']);a=root/'attempt-00';flight=a/'flight'
    assert [p.name for p in root.glob('attempt-*')]==['attempt-00']
    plan=read(root/'plan.json'); terminal=read(a/'terminal.json');run=read(flight/'run.json');world=read(flight/'world.json')
    intent=read(a/'intent.json');started=read(a/'started.json')
    assert terminal['status']=='completed' and terminal['returncode']==0 and terminal['Y']==world['score']['Y']
    assert terminal['evidence']['failure_reasons']==[] and terminal['retry_allowed'] is False
    assert terminal['plan_sha256']==intent['plan_sha256']==started['plan_sha256']==semantic(plan)
    assert intent['attempt_ordinal']==0 and type(started['pid']) is int and started['pid']>0
    assert datetime.fromisoformat(intent['utc'])<=datetime.fromisoformat(started['utc'])
    argv=intent['argv'];assert argv[argv.index('--policy')+1]=='B0' and '--realtime' in argv and argv[argv.index('--duration')+1]=='600'
    assert Path(argv[argv.index('--case')+1]).resolve()==(D/label/'devext-OOD-L3-00.json').resolve()
    names=('intent.json','started.json','stdout.log','flight/run.json','flight/world.json','flight/private-case.json')
    assert terminal['record_hashes']=={name:sha(a/name) for name in names}
    assert terminal['stdout_sha256']==sha(a/'stdout.log')
    assert all(sha(a/name)==summary[key] for name,key in [('terminal.json','terminal_sha256'),('flight/run.json','run_sha256'),('flight/world.json','world_sha256')])
    assert run['status']==world['status']=='completed' and run['requested_seconds']==world['simulated_seconds']==600
    assert run['realtime'] is True and world['clock_mode']=='wallclock_1x' and world['physics_steps_executed']==144000 and world['physics_hz']==240 and world['control_hz']==120
    assert world['max_clock_lag_seconds']==summary['max_clock_lag_seconds']<=.1 and world['clock_lag_over_0p1_seconds_ticks']==0
    assert all(run[k] is True for k in ('sources_unchanged','dependencies_unchanged','private_case_unchanged'))
    assert run['source_hashes']==plan['source_hashes'] and run['dependency_hashes']==plan['dependency_hashes']
    assert run['runtime_thread_environment']==plan['thread_environment']
    assert set(run['sources'])==set(plan['source_hashes']) and all(hashlib.sha256(v.encode()).hexdigest()==plan['source_hashes'][k] for k,v in run['sources'].items())
    assert all(sha(D/'flight-source'/Path(k).relative_to(REPO))==v for k,v in run['source_hashes'].items())
    assert sha(flight/'private-case.json')==sha(D/label/'devext-OOD-L3-00.json')==plan['case_file_sha256']==run['private_case_sha256']==world['private_case_file_sha256']
    assert sha(D/label/'manifest.json')==plan['suite_manifest_sha256'] and semantic(world['private_case'])==plan['case_sha256']==world['case_sha256']
    assert run['world_sha256']==sha(flight/'world.json') and run['policy']==plan['condition']=='B0'
    score=world['score'];covered={c for row in world['coverage_evidence'] if row['arrived_at']<=600 for c in row['cells']}
    assert covered<=set(world['required_cells']) and len(covered)/len(world['required_cells'])==score['C']
    static=score['static_entities'];dynamic=score['dynamic_entities'];entities={e['key']:e for e in world['private_case']['entities']}
    assert len(static)==3 and len(dynamic)==2 and len(entities)==11
    sw=sum(x['weight'] for x in static);dw=sum(b['weight'] for x in dynamic for b in x['bins'])
    assert sw==score['static_denominator_weight'] and dw==score['dynamic_denominator_weight']
    sn=sum(x['weight']*x['quality'] for x in static);dn=sum(b['weight']*b['quality'] for x in dynamic for b in x['bins'])
    assert sn/sw==score['S'] and dn/dw==score['D']
    crossing=[]
    for x in dynamic:
        assert len(x['bins'])==60
        for i,b in enumerate(x['bins']):
            start=entities[x['entity_key']]['appeared_at']+3*i
            assert b['index']==i+1 and b['starts_at']==start and b['ends_at']==start+3
            priority=world['private_case']['priority_at'];weight=2. if priority is not None and start>=priority else 1.
            assert b['weight']==weight
            if priority is not None and start<priority<start+3:crossing.append({'target':x['entity_key'],**b})
    assert score['Q']==(score['C']+score['S']+score['D'])/3
    h=int(not world['safety']['violations']); e=int(all(v>=20-1e-6 for v in world['safety']['exit_dwell_seconds']) and all(v>=1000 for v in score['remaining_range_m']))
    assert score['H']==h and score['E']==e and score['Y']==h*e*score['Q']
    assert score['SR']==int(h==e==1 and score['C']>=.8 and score['S']>=.6 and score['D']>=.5)
    assert world['safety']['physics_audits']==world['safety']['physics_steps']==144000
    for k,v in summary['score'].items():assert score[k]==v
    command=[str(REPO/'.venv/bin/python'),'-B',str(entry),'--world',str(flight/'world.json'),'--expected-world-sha256',summary['world_sha256']]
    commands.append(command);ret=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True,timeout=30)
    (OUT/(label+'-frozen.stdout.json')).write_text(ret.stdout);(OUT/(label+'-frozen.stderr.txt')).write_text(ret.stderr)
    assert ret.returncode==0
    frozen=json.loads(next(s for s in reversed(ret.stdout.splitlines()) if s.startswith('{')))
    assert frozen['score_equal'] is True and frozen['score']==score and frozen==read(D/(label+'-frozen-score-recount.json'))
    replay=read(D/(label+'-execution-replay.json'))
    assert replay['input_run_sha256']==sha(flight/'run.json') and replay['input_world_sha256']==sha(flight/'world.json')
    assert replay['replayer_sha256']==sha(D/'execution-replayer.py')==pair['replayer_sha256']
    assert replay['passed'] is True and replay['response_discrepancies']==[] and replay['simulated_seconds']==600
    assert replay['tool_calls_replayed']==replay['tool_calls_expected']==len(world['tool_calls'])==pair['replays'][label]['tool_calls']
    assert len(replay['exact_matches'])==25 and set(replay['exact_matches'])<=set(world) and all(v is True for v in replay['exact_matches'].values())
    sensor={x['evidence_id']:x for x in world['sensor_audit']};obs=world['delivered_observations']
    assert len({x['evidence_id'] for x in obs})==len(obs)
    assert all(x['evidence_id'] in sensor and x['measured_at']==sensor[x['evidence_id']]['t'] and x['arrived_at']>=x['measured_at']+.5-1e-8 for x in obs)
    sealed_ids={r['request']['report_id'] for r in world['reports'] if r['response']['status']=='accepted'}
    assert len(sealed_ids)<=10
    counts={k:len(world[k]) for k in ('tool_calls','trace','reports','delivered_observations','sensor_opportunities','background_trials','coverage_evidence')}
    retired={};link={}
    if label=='event':
        c=world['private_case'];d=c['retire_drone'];t=c['retire_at']
        retired={k:sum(x['drone_id']==d and x[stamp]>=t for x in world[k]) for k,stamp in [('sensor_audit','t'),('sensor_opportunities','t'),('background_trials','t'),('coverage_evidence','measured_at')]}
        assert all(v==0 for v in retired.values())
        retired['pre_retire_observations_arriving_after']=sum(x['drone_id']==d and x['measured_at']<t<=x['arrived_at'] for x in obs)
        buffered=[x for x in obs if x['drone_id']==c['link_drone'] and c['link_start']<=x['measured_at']<c['link_end']]
        assert all(x['arrived_at']>=c['link_end'] for x in buffered)
        assert all(not(c['link_start']<=x['arrived_at']<c['link_end']) for x in obs if x['drone_id']==c['link_drone'])
        link={'window':[c['link_start'],c['link_end']],'buffered_observations':len(buffered),'arrivals_at_recovery':sum(x['arrived_at']==c['link_end'] for x in buffered),
            'report_entries_at_recovery':sum(r['t']==c['link_end'] for r in world['reports'])}
        assert world['public_events']==[{'t':c['priority_at'],'type':'priority_update','task_card_version':2},{'t':c['retire_at'],'type':'exogenous_capacity_exit','drone_id':d}]
    else:assert world['public_events']==[]
    rows[label]={'status':terminal['status'],'attempts':1,'terminal_Y':terminal['Y'],'retry_allowed':terminal['retry_allowed'],'maxlag':world['max_clock_lag_seconds'],
        'counts':counts,'tool_names':dict(Counter(x['name'] for x in world['tool_calls'])),'job_statuses':dict(Counter(x['status'] for x in world['jobs'])),
        'coverage':[len(covered),len(world['required_cells'])],'static_denominator':sw,'static_numerator':sn,'dynamic_denominator':dw,'dynamic_numerator':dn,
        'dynamic_targets':[{'key':x['entity_key'],'appearance':entities[x['entity_key']]['appeared_at'],'bin_count':len(x['bins']),'weight_sum':sum(b['weight'] for b in x['bins']),
            'positive_bins':sum(b['quality']>0 for b in x['bins'])} for x in dynamic],
        'priority_crossing_bins':crossing,'sealed_report_slots':len(sealed_ids),'remaining_range_m':score['remaining_range_m'],'exit_dwell_seconds':world['safety']['exit_dwell_seconds'],
        'retirement':retired,'link':link,'run_events':run['events'],'world_events':world['events'],'replay_exact_fields':list(replay['exact_matches']),
        'frozen_full_score_exact':True,'score':{k:score[k] for k in ('C','S','D','Q','H','E','Y','SR')}}
    inputs[label]={str(p):sha(p) for p in (root/'plan.json',a/'terminal.json',flight/'run.json',flight/'world.json',a/'stdout.log')}
    worlds[label]=world;runs[label]=run;plans[label]=plan
    print(label,'raw bindings, full score and replay receipt verified',flush=True)
assert runs['event']['source_hashes']==runs['clean']['source_hashes'] and len(runs['event']['source_hashes'])==17
assert runs['event']['dependency_hashes']==runs['clean']['dependency_hashes'] and len(runs['event']['dependency_hashes'])==61
assert all(sha(Path(k))==v for k,v in runs['event']['dependency_hashes'].items())
assert worlds['event']['trace'][0]==worlds['clean']['trace'][0]
p=worlds['event']['private_case'];c=worlds['clean']['private_case']
fields=('priority_at','link_drone','link_start','link_end','camera_drone','camera_start','camera_end','retire_drone','retire_at')
assert all(c[k] is None for k in fields) and c['parent_sha256']==worlds['event']['case_sha256'] and c['parent_case']==p
assert all(p[k]==c[k] for k in p if k not in ('schema',*fields)) and p['level']==c['level']=='L3'
command=[str(REPO/'.venv/bin/python'),'-B',str(OUT/'common_recount.py'),str(Path(pair['flights']['clean']['root'])/'attempt-00/flight/world.json'),str(Path(pair['flights']['event']['root'])/'attempt-00/flight/world.json')]
commands.append(command);ret=subprocess.run(command,cwd='/tmp',env=env,capture_output=True,text=True,timeout=30)
(OUT/'common-frozen.stdout.json').write_text(ret.stdout);(OUT/'common-frozen.stderr.txt').write_text(ret.stderr)
assert ret.returncode==0
common=json.loads(next(s for s in reversed(ret.stdout.splitlines()) if s.startswith('{')))['score']
assert common==read(D/'clean-on-original-demand-calendar.json')['score']
assert all(common[k]==v for k,v in pair['common_demand_clean_score'].items())
delta=worlds['event']['score']['Y']-common['Y'];assert delta==pair['event_minus_common_demand_clean_Y']
archive=D/'complete-reference-pair.zip';assert sha(archive)==pair['archive_sha256']=='266fcdc253c5f7e6177109951e95d3c77103f6c302122d966a77f59a3a59c1b5'
with zipfile.ZipFile(archive) as z:
    names=z.namelist();assert len(names)==len(set(names))
    assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
    assert all(not stat.S_ISLNK(i.external_attr>>16) for i in z.infolist())
    expected={str(Path(label)/p.relative_to(Path(v['root']))) for label,v in pair['flights'].items() for p in Path(v['root']).rglob('*') if p.is_file()}
    assert set(names)==expected
    total=0
    for name in names:
        h=hashlib.sha256()
        with z.open(name) as f:
            for block in iter(lambda:f.read(1024*1024),b''):h.update(block);total+=len(block)
        label,tail=name.split('/',1);assert h.hexdigest()==sha(Path(pair['flights'][label]['root'])/tail)
result={'role':'read-only bounded n1 completed pair evidence audit; no new physics/replay/model/main generation','passed':True,
    'binding_file_count':len(binding['files']),'source_count':17,'dependency_count':61,'current_dependency_hashes_match':True,
    'attempts':rows,'common_demand_clean_score':{k:common[k] for k in ('C','S','D','Q','H','E','Y','SR')},'delta':delta,
    'archive':{'sha256':sha(archive),'members':len(names),'uncompressed_bytes':total,'every_member_matches_original':True},
    'frozen_evaluator_sha256':sha(entry),'commands':commands,'input_hashes':inputs}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','commands')},indent=2))
