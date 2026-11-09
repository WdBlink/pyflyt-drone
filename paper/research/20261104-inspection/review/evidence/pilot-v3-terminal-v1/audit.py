"""Offline stdlib recount; no source imports/RNG/model/physics/native mutation."""
from pathlib import Path
import array,base64,collections,hashlib,json,math,struct,sys,zipfile,zlib
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection';Q=R/'qualification/eight-condition-pilot-terminated.v3';SDK=REPO/'logs/inspection-eight-condition-pilot.v3';PHYS=REPO/'logs/inspection-eight-condition-dev-flights.v3';SLOT=PHYS/'dev-L3-00-B3-M2-1';OUT=Path('/tmp/inspection-pilot-v3-terminal-review')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();hbytes=lambda b:hashlib.sha256(b).hexdigest();read=lambda p:json.loads(Path(p).read_text());canonical=lambda v:hbytes(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
b=read(Q/'binding.json');members=read(Q/'archive-members.json');assert sha(Q/'binding.json')=='ea78536750d84959f8ecf9f6afa5a3745bb87268d4a6abb68e732355e9d0c011'
archives={}
for name,root in [('complete-pilot-sdk.zip',SDK),('complete-physical-attempts.zip',PHYS)]:
 p=Q/name;assert sha(p)==b['archives'][name]['sha256']
 with zipfile.ZipFile(p) as z:
  names=z.namelist();assert len(names)==len(set(names))==b['archives'][name]['members'];assert set(names)==set(members[name]);assert set(names)=={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
  for n in names:
   raw=z.read(n);assert hbytes(raw)==members[name][n] and (root/n).read_bytes()==raw
  archives[name]={'sha256':sha(p),'members':len(names),'uncompressed_bytes':sum(x.file_size for x in z.infolist()),'exact_current_tree':True}
prepared=REPO/'logs/inspection-eight-condition-pilot-preparation.v3';launch=read(R/'review/evidence/pilot-v3-launch-v1/recount.json')
for name,count in [('source-hashes.json',103),('dependency-hashes.json',61),('input-hashes.json',10)]:
 table=read(SDK/name);assert len(table)==count and (SDK/name).read_bytes()==(prepared/name).read_bytes()
 for p,h in table.items():assert sha(p)==h,p
sources=read(SDK/'source-hashes.json');deps=read(SDK/'dependency-hashes.json');inputs=read(SDK/'input-hashes.json');texts=read(SDK/'source-files.json');assert set(texts)==set(sources)
for p,t in texts.items():assert hbytes(t.encode())==sources[p] and Path(p).read_text()==t
assert (SDK/'episode-order.json').read_bytes()==(prepared/'prepared-episode-order.json').read_bytes();assert (SDK/'sampler-seeds.json').read_bytes()==(prepared/'sampler-seeds.json').read_bytes()
order=read(SDK/'episode-order.json');seeds=read(SDK/'sampler-seeds.json');assert order['seed']==2026110621 and seeds['master']==2026110622
assert order['slots'][0]['case_id']=='dev-L3-00' and order['slots'][0]['condition']=='B3-M2' and order['slots'][0]['repeat_ordinal']==1
conditions={'B0':1,'B1':1,'B2':1,'B3-M1':3,'B3-M2':3,'B3-M3':3,'B4-M1':3,'B4-M2':3};cases=[f'dev-L{l}-{i:02d}' for l in (1,2,3) for i in (0,1)];sdk_summary={};valid=set()
for group,n in [('classic',18),('llm',90),('root-means',48)]:
 p=SDK/group/'results.json';v=read(p);assert v['reported_denominator']==len(v['runs'])==n and v['execution_status']=='failed' and v['outcome']=='inconclusive' and sha(p)==b['sdk'][group]['sha256']
 expected={(c,k,a) for c in cases for a,nr in conditions.items() if group=='root-means' or (nr==1 if group=='classic' else nr==3) for k in (range(1) if group=='root-means' else range(nr))}
 actual={(x['case_id'],x['seed'],x['condition']) for x in v['runs']};assert actual==expected and len(actual)==n
 statuses=collections.Counter(x['status'] for x in v['runs']);assert dict(statuses)==b['sdk'][group]['statuses'];assert sum(x['attempted'] for x in v['runs'])==v['attempted_count']==b['sdk'][group]['attempted_count']
 for x in v['runs']:
  raw=read(SDK/x['raw']);assert all(raw[k]==x[k] for k in ['case_id','seed','condition','status','value','attempted','failure_kind','reason']);assert x['value'] is None
  if x['status']=='completed':valid.add((x['case_id'],x['seed'],x['condition']))
 sdk_summary[group]={'planned':n,'statuses':dict(statuses),'attempted':v['attempted_count'],'execution_status':v['execution_status'],'values_all_null':True,'results_sha256':sha(p)}
assert not valid;assert {str(x.relative_to(SDK)) for x in (SDK/'starts').glob('*.json')}=={'starts/dev-L3-00-B3-M2-1.json'};assert not (SDK/'order-receipts').exists();assert not (SDK/'pilot-completion.json').exists();assert not (SDK/'recording-completion-index.json').exists()
assert {x.name for x in SLOT.glob('attempt-*')}=={'attempt-00','attempt-01'} and not (SLOT/'attempt-02').exists();plan=read(SLOT/'plan.json');assert plan['max_attempts']==2 and plan['max_clock_lag_seconds']==.1 and plan['sampler_seed']==seeds['by_order_index']['0'];plan_sha=hbytes(json.dumps(plan,sort_keys=True,allow_nan=False).encode())
owner=read(R/'.paper/control.json');attempts=[];call_rows=[]
for ordinal in (0,1):
 attempt=SLOT/f'attempt-{ordinal:02d}';flight=attempt/'flight';rid=f'dev-all8-pilot-v3-000-a{ordinal}';op=REPO/'logs/inspection-model-resource-operations.v1'/rid;resource=R/'qualification/local-model-episodes'/rid
 terminal=read(attempt/'terminal.json');run=read(flight/'run.json');world=read(flight/'world.json');receipt=read(resource/'usage-receipt.json');intent=read(op/'service-intent.json');completion=read(op/'service-completion.json')
 assert terminal['status']=='known_operational_failure' and terminal['Y'] is None and terminal['returncode']==0 and terminal['plan_sha256']==plan_sha==receipt['plan_sha256'] and terminal['physical_validation']['failure_reasons']==['clock_lag_exceeded']
 assert sha(attempt/'terminal.json')==b['attempts'][ordinal]['terminal_sha256']
 assert terminal['physical_validation']['run_sha256']==sha(flight/'run.json')==run.get('unused',sha(flight/'run.json')) and terminal['physical_validation']['world_sha256']==sha(flight/'world.json')==run['world_sha256']
 assert run['sources_unchanged'] is True and run['dependencies_unchanged'] is True and run['private_case_unchanged'] is True and run['status']=='completed'
 assert run['source_hashes']==plan['source_hashes'] and run['dependency_hashes']==plan['dependency_hashes']==deps
 assert set(run['sources'])==set(plan['source_hashes'])
 for p,t in run['sources'].items():assert hbytes(t.encode())==plan['source_hashes'][p]==sources[p]
 assert sha(flight/'private-case.json')==plan['case_file_sha256']==run['private_case_sha256']==world['private_case_file_sha256']
 assert run['sampler_base_seed']==plan['sampler_seed'] and run['reservation']==rid and run['requested_seconds']==600 and run['realtime'] is True
 assert world['status']=='completed' and world['simulated_seconds']==600 and world['physics_steps_executed']==144000 and world['physics_hz']==240 and world['control_hz']==120 and world['clock_mode']=='wallclock_1x'
 assert world['safety']['physics_audits']==world['safety']['physics_steps']==world['execution_tracking_240hz']['physics_samples']==144000
 tr=world['execution_tracking_trace'];raw=zlib.decompress(base64.b64decode(tr['data']));assert hbytes(raw)==tr['raw_sha256'] and tr['rows']==144000 and len(raw)==144000*len(tr['columns'])*8
 for i,row in enumerate(struct.iter_unpack('<'+'d'*len(tr['columns']),raw)):
  assert all(math.isfinite(v) for v in row) and math.isclose(row[0],(i+1)/240,abs_tol=1e-8)
 lag=world['max_clock_lag_seconds'];assert lag==terminal['physical_validation']['max_clock_lag_seconds']==b['attempts'][ordinal]['physical_validation']['max_clock_lag_seconds'] and lag>.1
 witness=world['maximum_clock_lag_witness'];assert witness['lag_seconds']==lag and 0<=witness['sim_time']<=600
 progress=[json.loads(s) for s in (flight/'progress.jsonl').read_text().splitlines()];assert len(progress)==600 and [x['t'] for x in progress]==list(range(1,601));assert all(math.isfinite(x['wall_elapsed']) for x in progress)
 residual=[x['wall_elapsed']-x['t'] for x in progress]
 tree=read(attempt/'process-tree.json');started=read(attempt/'started.json');returned=read(attempt/'process-return.json');assert tree['status']=='closed' and not tree['remaining_owned'] and not tree['unclaimed_new_python_orphans'] and tree['root_pid']==started['pid'] and returned['returncode']==0 and returned['stdout_sha256']==sha(attempt/'stdout.log')
 for group,base in [('payload_hashes',attempt),('model_payload_hashes',op)]:
  actual={str(p.relative_to(base)):sha(p) for p in base.rglob('*') if p.is_file() and (group!='payload_hashes' or p.name!='terminal.json')};assert actual==terminal[group]
 for p,h in terminal['settlement_file_hashes'].items():assert sha(p)==h
 reserve=read(attempt/'reserve-owner.json');settled=read(attempt/'settle-owner.json');assert reserve['status']=='reserved' and reserve['reservation']==rid and settled['status']=='known' and settled['reservation']==rid and not settled['budget']['unknown_usage'] and not settled['budget']['unresolved_usage'] and settled['budget']['reserved']['calls']==settled['budget']['reserved']['tokens']==0
 settle_packet=read(R/'resource-requests'/f'{rid}.settle.json');assert settle_packet['status']=='known' and settle_packet['receipt']==str((resource/'usage-receipt.json').relative_to(R))
 ledger=owner['usage_ledger'][rid];assert ledger['status']=='known' and ledger['unreserved'] is False and ledger['receipt_sha256']==sha(resource/'usage-receipt.json')
 purpose=json.loads(ledger['purpose']);assert purpose['phase']=='development' and purpose['case_id']==plan['case_id'] and purpose['condition']==plan['condition'] and purpose['attempt_ordinal']==ordinal and purpose['episode_plan_sha256']==plan_sha
 assert intent['reservation']==rid and intent['owner_frozen_plan']==plan['owner_frozen_plan'] and intent['reserved']==plan['reservation_usage'] and intent['source_sha256']==plan['service_sha256'] and intent['model_receipt_sha256']==plan['model_receipt_sha256']
 with zipfile.ZipFile(resource/'complete-attempt.zip') as z:
  names=z.namelist();m=json.loads(z.read('manifest.json'));assert len(names)==len(set(names))==107 and set(names)==set(m)|{'manifest.json'}
  for n,h in m.items():
   data=z.read(n);assert hbytes(data)==h
   actual=attempt/n[len('attempt/'):] if n.startswith('attempt/') else op/n[len('model-operation/'):] if n.startswith('model-operation/') else SLOT/'plan.json'
   assert actual.read_bytes()==data
  assert receipt['archive_sha256']==sha(resource/'complete-attempt.zip')
 calls=sorted(op.glob('call-*'));assert [p.name for p in calls]==[f'call-{k:04d}' for k in range(1,31)]
 actor=[json.loads(s) for s in (flight/'actor.jsonl').read_text().splitlines()];requests=[x for x in actor if x['kind']=='public_model_request'];responses=[x for x in actor if x['kind']=='actual_model_response'];assert len(requests)==len(responses)==30
 inputs_total=generated_total=0
 for i,call in enumerate(calls,1):
  req=read(call/'request.json');res=read(call/'response.json');tokens=[json.loads(s) for s in (call/'tokens.jsonl').read_text().splitlines()];a=requests[i-1];response=responses[i-1]['response']
  assert req['messages']==a['messages'] and canonical(req['messages'])==a['messages_sha256'] and req['seed']==a['sampler_seed']==(plan['sampler_seed']+i-1)%2**32
  assert all(response.get(k)==v for k,v in res.items()) and res['status']=='completed' and res['kind']=='response'
  assert res['generated_tokens_observed']==res['emitted_tokens_journaled']==len(tokens) and res['input_tokens_submitted']>0
  assert [x['runtime_generation_tokens'] for x in tokens]==list(range(1,len(tokens)+1)) and all(type(x['token']) is int and x['runtime_prompt_tokens']==res['input_tokens_submitted'] for x in tokens)
  assert res['usage']=={'calls':1,'tokens':res['input_tokens_submitted']+len(tokens)}
  mc=receipt['resource_recount']['calls'][i-1];assert mc['request_sha256']==sha(call/'request.json') and mc['response_sha256']==sha(call/'response.json') and mc['tokens_sha256']==sha(call/'tokens.jsonl') and mc['reported_unjournaled_observations']==0
  inputs_total+=res['input_tokens_submitted'];generated_total+=len(tokens);call_rows.append({'attempt':ordinal,'call':i,'input_tokens':res['input_tokens_submitted'],'observed_generated':len(tokens),'status':res['status'],'request_sha256':sha(call/'request.json'),'response_sha256':sha(call/'response.json'),'tokens_sha256':sha(call/'tokens.jsonl')})
 usage={'calls':len(calls),'tokens':inputs_total+generated_total};assert usage=={'calls':30,'tokens':165378}==receipt['usage']==receipt['resource_recount']['usage']==completion['usage']==ledger['reported_usage']==settle_packet['usage']==b['attempts'][ordinal]['usage']
 assert receipt['resource_recount']['generated_tokens_runtime_observed']==receipt['resource_recount']['tokens_durably_journaled']==generated_total
 assert receipt['resource_recount']['intent_sha256']==sha(op/'service-intent.json') and receipt['resource_recount']['completion_sha256']==sha(op/'service-completion.json')
 assert run['model_operation']['usage']==usage and run['model_operation']['sha256']==sha(op/'service-completion.json')
 attempts.append({'ordinal':ordinal,'terminal_sha256':sha(attempt/'terminal.json'),'run_sha256':sha(flight/'run.json'),'world_sha256':sha(flight/'world.json'),'status':terminal['status'],'admitted_Y':terminal['Y'],'raw_Y':world['score']['Y'],'simulated_seconds':world['simulated_seconds'],'physics_steps':world['physics_steps_executed'],'tracking_rows_recounted':tr['rows'],'tracking_raw_sha256':tr['raw_sha256'],'reported_max_lag':lag,'reported_p95_lag':world['clock_lag_p95_seconds'],'reported_over_0p1_ticks':world['clock_lag_over_0p1_seconds_ticks'],'reported_maximum_witness':witness,'progress_rows':len(progress),'one_hz_residual_max':max(residual),'one_hz_residual_over_0p1':sum(x>.1 for x in residual),'cleanup_status':tree['status'],'actual_model_usage':usage,'input_tokens_recounted':inputs_total,'generated_tokens_journaled':generated_total,'resource_archive_members':107,'resource_archive_sha256':sha(resource/'complete-attempt.zip'),'usage_receipt_sha256':sha(resource/'usage-receipt.json'),'native_settlement_status':settled['status'],'source_texts_verified':len(run['sources'])})
assert len(attempts)==2 and not any(x['admitted_Y'] is not None for x in attempts)
usage_statuses=dict(collections.Counter(x['status'] for x in owner['usage_ledger'].values()));all_usage={k:sum(x['reported_usage'].get(k,0) for x in owner['usage_ledger'].values() if x['status']=='known') for k in ('calls','tokens')};assert usage_statuses=={'known':22} and all_usage=={'calls':474,'tokens':3004256}
assert b['resource_total']['known_usage']==all_usage and b['resource_total']['status_counts']==usage_statuses
result={'terminal_binding_sha256':sha(Q/'binding.json'),'archive_members_sha256':sha(Q/'archive-members.json'),'archives':archives,'sdk':sdk_summary,'valid_callbacks':0,'complete_roots':0,'attempt_count':2,'no_attempt02':True,'schedule_seed':order['seed'],'sampler_master':seeds['master'],'source_count_verified':103,'dependencies_verified':61,'inputs_verified':10,'source_matches_launch_preparation':True,'plan_sha256':plan_sha,'attempts':attempts,'new_usage_total':{'calls':sum(x['actual_model_usage']['calls'] for x in attempts),'tokens':sum(x['actual_model_usage']['tokens'] for x in attempts)},'native_current_usage_statuses':usage_statuses,'native_cumulative_known_usage':all_usage,'owner_snapshot_sha256':sha(R/'.paper/control.json'),'software_concurrency_attributed_to_root_metadata':b['software_review_concurrency'],'limitations':['clock aggregates/witness authenticated, not independently recomputed from absent72000 lag trace','tracking error/z trace is not full physical state or wallclock vector','no exact UTC physical epoch; no peak UTC or unique cause','concurrent software tests reported by root/agent metadata, not full independent host activity telemetry','raw Y retained but not accepted or regraded']}
(OUT/'recount.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(OUT/'model-calls.json').write_text(json.dumps(call_rows,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['software_concurrency_attributed_to_root_metadata']},ensure_ascii=False,indent=2))
