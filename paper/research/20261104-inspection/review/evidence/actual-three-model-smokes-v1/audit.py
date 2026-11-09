"""Read-only audit of three actual smokes; no model, tokenizer or physics imports."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import zipfile

REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
RUN=REPO/'paper/research/20261104-inspection'
OUT=Path('/tmp/inspection-actual-three-model-smokes-review-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def unique(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError('duplicate field '+k)
        d[k]=v
    return d
def parse(s):return json.loads(s,object_pairs_hook=unique,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def read(p):return parse(Path(p).read_text())
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
source=RUN/'qualification/source/model-orchestrator-v2'
meter=load('frozen_actual_meter',source/'inspection_model_receipts.py')
wire=load('frozen_actual_wire',source/'inspection_llm_protocol.py')
state_bytes=(RUN/'.paper/control.json').read_bytes();state=parse(state_bytes)
regpath=RUN/'planning/local-model-registry.v1.json';registry=read(regpath)
rows=[];input_bindings={};ledger_excerpt={};history_excerpt=[];operations_excerpt={}
for i,alias in enumerate(('M1','M2','M3')):
    checks={}
    def check(k,v):checks[k]=bool(v)
    q=RUN/'qualification/local-model-generation'/(alias+'-smoke.v1')
    probe=read(q/'probe.json');receipt=read(q/'usage-receipt.json');rid=probe['expected']['reservation']
    op=REPO/'logs/inspection-model-resource-operations.v1'/rid
    transport=REPO/'logs'/('inspection-model-smoke-'+alias+'.v1')/'transport'
    ledger=state['usage_ledger'][rid];purpose=parse(ledger['purpose']);profile=registry['models'][alias]
    intent=read(op/'service-intent.json');request=read(op/'call-0001/request.json');response=read(op/'call-0001/response.json')
    tokens=[parse(line) for line in (op/'call-0001/tokens.jsonl').read_text().splitlines()]
    completion=read(op/'service-completion.json');download=read(RUN/profile['download_receipt'])
    budget={**profile['generation_profile'],'max_calls':1,'max_total_tokens':16384,'max_output_tokens':64}
    expected={'reservation':rid,'owner_frozen_plan':state['frozen_plan'],'model_receipt_sha256':sha(RUN/profile['download_receipt']),
        'service_sha256':sha(source/'inspection_model_service.py'),'reserved':{'calls':1,'tokens':16384},'generation_profile':budget}
    check('expected_prebound_config',probe['expected']==expected)
    check('registry_alias_revision_receipt',probe['model_alias']==receipt['model_alias']==alias
        and receipt['repo_id']==profile['repo_id']==download['repo_id'] and profile['revision']==download['revision']
        and len(download['files'])==profile['file_count'] and expected['model_receipt_sha256']==profile['download_receipt_sha256']
        and probe['registry_sha256']==sha(regpath))
    declared_sources=probe['sources'];source_checks={}
    for path,h in declared_sources.items():
        archived=source/Path(path).name
        source_checks[Path(path).name]={'expected_sha256':h,'archived_sha256':sha(archived),'current_sha256':sha(path)}
    check('all_five_sources_snapshotted',len(source_checks)==5 and all(v['expected_sha256']==v['archived_sha256']==v['current_sha256'] for v in source_checks.values()) and probe['sources_unchanged'] is True)
    binding={'kind':'inspection_local_generation','phase':'development','model_alias':alias,'model_registry_sha256':sha(regpath),
        'model_receipt_sha256':expected['model_receipt_sha256'],'owner_frozen_plan':state['frozen_plan'],'service_source_sha256':expected['service_sha256']}
    check('purpose_identity_actual_config',all(purpose.get(k)==v for k,v in binding.items())
        and json.dumps(purpose['actual_request_budget'],sort_keys=True,allow_nan=False)==json.dumps(budget,sort_keys=True,allow_nan=False))
    check('intent',all(intent[k]==expected[k] for k in ('reservation','owner_frozen_plan','reserved','model_receipt_sha256'))
        and intent['source_sha256']==expected['service_sha256'] and intent['request_budget']==budget
        and type(intent['pid']) is int and intent['pid']>0 and intent['pid']==intent['process_group'])
    check('one_call_only',sorted(p.name for p in op.glob('call-*'))==['call-0001'])
    check('same_public_prompt_seed',request['messages']==probe['messages'] and type(request['seed']) is int
        and request['seed']==probe['seed']==2026110605+i)
    inp=request['input_tokens_submitted'];gen=len(tokens);usage={'calls':1,'tokens':inp+gen}
    check('token_rows_independent_recount',type(inp) is int and inp>0
        and all(type(x['token']) is int and x['token']>=0 and x['runtime_prompt_tokens']==inp
            and type(x['runtime_generation_tokens']) is int and x['runtime_generation_tokens']==j+1 for j,x in enumerate(tokens))
        and response['generated_tokens_observed']==response['emitted_tokens_journaled']==gen
        and response['input_tokens_submitted']==inp and response['text']==''.join(x['text'] for x in tokens))
    check('profile_prefix_usage',request['max_output_tokens']==64 and inp+64<=16384
        and request['temperature']==.2 and request['top_p']==.95 and request['enable_thinking'] is False
        and response['usage']==completion['usage']==usage and response['status']=='completed'
        and response['deadline_exceeded'] is False and 0<=response['wall_seconds']<=60)
    check('transport_probe_durable_response',all(probe['response'].get(k)==v for k,v in response.items())
        and probe['response']['parent_end_to_end_wall_seconds']>=response['wall_seconds'])
    stdout=[parse(x) for x in (transport/'stdout.log').read_text().splitlines()]
    check('transport_two_exact_envelopes',len(stdout)==2 and stdout[0]=={'kind':'ready','reservation':rid} and stdout[1]==response)
    reply,commands=wire.parse_reply(response['text'])
    check('strict_json_reply',reply==probe['parsed_reply']=={'commands':[],'note':'ready'} and commands==[] and receipt['policy_reply_valid'] is True)
    recount=meter.validate_model_operation(op,expected)
    check('frozen_meter_matches_independent_and_receipts',recount['usage']==usage and recount['generated_tokens_runtime_observed']==gen
        and recount==probe['resource_recount']==receipt['resource_recount']
        and receipt['usage']==usage and receipt['generated_tokens_runtime_observed']==receipt['tokens_durably_journaled']==gen)
    members={}
    for p in op.rglob('*'):
        if p.is_file():members['operation/'+str(p.relative_to(op))]=p
    for p in transport.iterdir():
        if p.is_file():members['transport/'+p.name]=p
    with zipfile.ZipFile(q/'complete-operation.zip') as z:
        check('archive_exact_member_set_no_duplicates',len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(members)
            and all(not stat.S_ISLNK(x.external_attr>>16) for x in z.infolist()))
        check('archive_all_original_bytes_exact',all(z.read(name)==p.read_bytes() for name,p in members.items()))
    check('receipt_probe_archive_bindings',receipt['archive_sha256']==sha(q/'complete-operation.zip') and receipt['probe_sha256']==sha(q/'probe.json')
        and receipt['archive']==str((q/'complete-operation.zip').relative_to(RUN))
        and (REPO/'logs'/('inspection-model-smoke-'+alias+'.v1')/'probe.json').read_bytes()==(q/'probe.json').read_bytes())
    native_receipt=str((q/'usage-receipt.json').relative_to(RUN));rh=sha(q/'usage-receipt.json')
    check('ledger_known_exact_unique_receipt',ledger['status']=='known' and ledger['unreserved'] is False
        and ledger['reserved']==expected['reserved'] and ledger['reported_usage']==usage
        and ledger['receipt']==native_receipt and ledger['receipt_sha256']==rh
        and ledger['receipts']==[{'path':native_receipt,'sha256':rh}])
    reserve_packet=read(RUN/'resource-requests'/('model-smoke-'+alias+'.v1.json'))
    settle_packet=read(RUN/'resource-requests'/('model-smoke-'+alias+'-settle.v1.json'))
    reserve_request={'command':'budget-reserve','packet':reserve_packet}
    settle_request={'command':'budget-settle','reservation':rid,'packet':settle_packet}
    reserve_op=state['operations'][rid];settle_op=state['operations']['settle-'+rid]
    hs=[h for h in state['history'] if isinstance(h,dict) and h.get('operation') in (rid,'settle-'+rid)]
    check('native_reserve_settle_request_digests',reserve_op['request_sha256']==digest(reserve_request)
        and settle_op['request_sha256']==digest(settle_request) and reserve_packet['purpose']==ledger['purpose']
        and reserve_packet['usage']==expected['reserved'] and settle_packet=={'receipt':native_receipt,'status':'known','usage':usage})
    check('one_native_reserve_one_settle',len(hs)==2 and [h['operation'] for h in hs]==[rid,'settle-'+rid]
        and hs[0]['result']['status']=='reserved' and hs[1]['result']['status']=='known'
        and all(state['operations'][h['operation']]['request_sha256']==h['request_sha256'] for h in hs))
    def ts(v):return datetime.fromisoformat(v).timestamp()
    times={'reserved':ledger['created_at'],'probe_started':probe['started_at'],
        'intent_file_mtime':datetime.fromtimestamp((op/'service-intent.json').stat().st_mtime,timezone.utc).isoformat(),
        'request_file_mtime':datetime.fromtimestamp((op/'call-0001/request.json').stat().st_mtime,timezone.utc).isoformat(),
        'response_file_mtime':datetime.fromtimestamp((op/'call-0001/response.json').stat().st_mtime,timezone.utc).isoformat(),
        'completion_file_mtime':datetime.fromtimestamp((op/'service-completion.json').stat().st_mtime,timezone.utc).isoformat(),
        'settled':ledger['settled_at']}
    check('recorded_reserve_probe_generate_settle_order',all(ts(a)<=ts(b) for a,b in zip(list(times.values()),list(times.values())[1:])))
    rows.append({'alias':alias,'reservation':rid,'repo_id':profile['repo_id'],'revision':profile['revision'],'input_tokens':inp,
        'output_tokens':gen,'usage':usage,'seed':request['seed'],'checks':checks,'passed':all(checks.values()),
        'source_bindings':source_checks,'times':times,'native_receipt_sha256':rh,'probe_sha256':sha(q/'probe.json'),
        'archive_sha256':sha(q/'complete-operation.zip'),'stderr_bytes':(transport/'stderr.log').stat().st_size})
    for p in [q/'probe.json',q/'usage-receipt.json',q/'complete-operation.zip',RUN/profile['download_receipt'],regpath,
            *members.values(),RUN/'resource-requests'/('model-smoke-'+alias+'.v1.json'),
            RUN/'resource-requests'/('model-smoke-'+alias+'-settle.v1.json')]:input_bindings[str(p)]=sha(p)
    ledger_excerpt[rid]=ledger;history_excerpt+=hs;operations_excerpt[rid]=reserve_op;operations_excerpt['settle-'+rid]=settle_op

total={'calls':sum(x['usage']['calls'] for x in rows),'tokens':sum(x['usage']['tokens'] for x in rows),
    'input_tokens':sum(x['input_tokens'] for x in rows),'output_tokens':sum(x['output_tokens'] for x in rows)}
last=history_excerpt[-1]['result']['budget']
overall={'all_case_checks':all(x['passed'] for x in rows),'three_only_known_ledger_entries':len(state['usage_ledger'])==3 and all(v['status']=='known' for v in state['usage_ledger'].values()),
    'unique_receipts_and_reservations':len({x['native_receipt_sha256'] for x in rows})==len({x['reservation'] for x in rows})==3,
    'native_final_usage_equal':last['known_usage']['calls']==total['calls'] and last['known_usage']['tokens']==total['tokens'],
    'no_remaining_reserved_or_unknown':last['reserved']['calls']==last['reserved']['tokens']==0 and last['unknown_usage']==last['unresolved_usage']==[]}
result={'role':'read-only actual text-smoke audit, not a new generation/flight/attestation','rows':rows,'totals':total,'overall':overall,
    'passed':all(overall.values()),'native_final_budget':last,'control_sha256_at_read':hashlib.sha256(state_bytes).hexdigest(),'input_sha256':input_bindings}
(OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'native-ledger-excerpt.json').write_text(json.dumps({'control_sha256_at_read':result['control_sha256_at_read'],'frozen_plan':state['frozen_plan'],
    'usage_ledger':ledger_excerpt,'history':history_excerpt,'operations':operations_excerpt},indent=2)+'\n')
print(json.dumps({'passed':result['passed'],'overall':overall,'totals':total,'rows':[{'alias':x['alias'],'passed':x['passed'],'failed_checks':[k for k,v in x['checks'].items() if not v]} for x in rows]},indent=2))
