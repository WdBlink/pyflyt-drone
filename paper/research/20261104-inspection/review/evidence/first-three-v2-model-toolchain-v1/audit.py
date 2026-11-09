"""Stored causal I/O only; pure parser and ledger, no model/mission imports."""
from collections import Counter
from dataclasses import asdict
import hashlib,importlib.util,json,math,os,sys
from pathlib import Path
from types import ModuleType
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');R=REPO/'paper/research/20261104-inspection'
OUT=Path(__file__).resolve().parent;S=OUT/'source';os.nice(10)
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canon(v):return json.dumps(v,sort_keys=True,allow_nan=False)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
for name in ('envs','envs.inspection','eval'):
    m=ModuleType(name);m.__path__=[];sys.modules[name]=m
scoring=load('envs.inspection.scoring',S/'scoring.py')
protocol=load('envs.inspection.protocol',S/'protocol.py')
wire=load('eval.inspection_llm_protocol',S/'inspection_llm_protocol.py')
result={};input_hashes={};source_sets=[]
for label,episode,attempt in [('M3','dev-L1-00-B3-M3-0','attempt-01'),('M2','dev-L1-00-B3-M2-2','attempt-00'),('M1','dev-L1-00-B3-M1-2','attempt-00')]:
    root=REPO/'logs/inspection-eight-condition-dev-flights.v2'/episode;a=root/attempt;f=a/'flight'
    terminal=read(a/'terminal.json');plan=read(root/'plan.json');run=read(f/'run.json');w=read(f/'world.json')
    logs=[json.loads(line) for line in (f/'actor.jsonl').read_text().splitlines()]
    assert terminal['status']=='completed' and terminal['Y']==w['score']['Y'] and terminal['evidence']['flight']['failure_reasons']==[]
    assert terminal['plan_sha256']==hashlib.sha256(canon(plan).encode()).hexdigest()
    assert terminal['evidence']['actor_sha256']==sha(f/'actor.jsonl')
    assert all(sha(a/name)==digest for name,digest in terminal['payload_hashes'].items())
    assert terminal['evidence']['flight']['run_sha256']==sha(f/'run.json') and terminal['evidence']['flight']['world_sha256']==sha(f/'world.json')
    assert run['world_sha256']==sha(f/'world.json') and run['status']==w['status']=='completed'
    assert run['sources_unchanged'] is run['dependencies_unchanged'] is run['private_case_unchanged'] is True
    assert run['source_hashes']==plan['source_hashes'] and run['dependency_hashes']==plan['dependency_hashes']
    assert all(hashlib.sha256(text.encode()).hexdigest()==run['source_hashes'][name] for name,text in run['sources'].items())
    source_sets.append(run['source_hashes'])
    for path in S.iterdir():
        key=next(k for k in run['sources'] if Path(k).name==path.name)
        assert sha(path)==run['source_hashes'][key]
    assert w['physics_steps_executed']==144000 and w['simulated_seconds']==600 and w['max_clock_lag_seconds']<=.1
    assert len(logs)%3==0
    op=Path(run['model_operation']['completion']).parent;assert sha(op/'service-completion.json')==run['model_operation']['sha256']
    assert all(sha(op/name)==digest for name,digest in terminal['model_payload_hashes'].items())
    delivered={o['evidence_id']:o for o in w['delivered_observations']};sensor={s['evidence_id']:s for s in w['sensor_audit']}
    dispatched=[];enqueues=[];requests=[];formats=[];reply_statuses=Counter();feedback=0;observations_shown=0
    worker_index={}
    for call in w['tool_calls']:worker_index.setdefault(canon({k:call[k] for k in ('name','args','response')}),[]).append(call['t'])
    for i in range(0,len(logs),3):
        request,response,decoded=logs[i:i+3]
        assert [x['kind'] for x in (request,response,decoded)]==['public_model_request','actual_model_response','decoded_reply']
        ordinal=request['request_ordinal'];raw=response['response'];audit=decoded['audit'];reply_statuses[audit['status']]+=1
        assert ordinal==i//3+1 and raw['submitted_ordinal']==ordinal
        messages=request['messages']; frame=json.loads(messages[-1]['content']);pub=frame['public_state']
        assert hashlib.sha256(json.dumps(messages,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()==request['messages_sha256']
        assert request['source_sim_time']==pub['sim_time']<=request['worker_before_request'][0]
        assert wire.PUBLIC_TOOL_CONTRACT.strip() in messages[0]['content']
        wire.serialize_public_frame(pub)
        for o in pub['observations']:
            assert o==delivered[o['evidence_id']] and o['measured_at']<=o['arrived_at']<=pub['sim_time']
            observations_shown+=1
        for tool in frame['recent_tool_results']:
            assert any(t<=request['worker_before_request'][0]+1e-8 for t in worker_index[canon(tool)])
            feedback+=1
        durable_req=read(op/f'call-{ordinal:04d}/request.json');durable_resp=read(op/f'call-{ordinal:04d}/response.json')
        assert durable_req['messages']==messages and durable_req['seed']==request['sampler_seed']
        assert all(raw[k]==v for k,v in durable_resp.items())
        assert audit['request']['messages']==messages and audit['request']['source_sim_time']==request['source_sim_time']
        if audit['status']=='decoded_not_tool_acceptance':
            reply,commands=wire.parse_reply(raw['text']);assert reply==audit['reply']
            expected=[{'name':name,'args':args} for name,args in commands]
        elif audit['status']=='policy_format_failure':
            try:wire.parse_reply(raw['text'])
            except wire.ModelReplyError as error:assert str(error)==audit['reason']
            else:raise AssertionError('recorded format failure does not reproduce')
            expected=[];formats.append({'source_time':request['source_sim_time'],'worker_time':decoded['worker_progress_at_enqueue'][0],'reason':audit['reason']})
        else:raise AssertionError('new status outside this bounded audit')
        if response['final_after_flight'] or decoded['worker_progress_at_enqueue'][0]>=600:expected=[]
        assert expected==decoded['commands_enqueued_not_accepted']
        for command in expected:
            dispatched.append(command);enqueues.append({'ordinal':ordinal,'source_time':request['source_sim_time'],
                'enqueue_time':decoded['worker_progress_at_enqueue'][0],'visible_aircraft_versions':{r['drone_id']:r['job_version'] for r in pub['aircraft']},
                'visible_task_card':pub['task_card']['task_card_version'],'visible_regions':[r['region_id'] for r in pub['task_card']['regions']],
                'visible_rules':[r['rule_id'] for r in pub['task_card']['rules']],'visible_evidence':{o['evidence_id'] for o in pub['observations']},
                'recent_rejections':[x['response'].get('reason') for x in frame['recent_tool_results'] if x['response'].get('status')=='rejected']})
        requests.append({'source_time':request['source_sim_time'],'reply_sim_time':decoded['worker_progress_at_enqueue'][0],
            'reply_status':audit['status'],'commands_enqueued':len(expected),'visible_observation_count':len(pub['observations'])})
    assert dispatched==[{'name':c['name'],'args':c['args']} for c in w['tool_calls']]
    assert len(dispatched)==run['actor_mutating_calls'] and all(c['t']>=e['enqueue_time']-1e-8 and c['t']<600 for c,e in zip(w['tool_calls'],enqueues))
    assert len(requests)==run['model_operation']['usage']['calls']
    representatives={};accepted=[];duration_rejections=[]
    for call,context in zip(w['tool_calls'],enqueues):
        reason=call['response'].get('reason',call['response']['status'])
        item={'t':call['t'],'source_time':context['source_time'],'ordinal':context['ordinal'],'name':call['name'],'args':call['args'],'response':call['response'],
              'visible_aircraft_versions':context['visible_aircraft_versions'],'visible_task_card':context['visible_task_card'],
              'recent_rejections':context['recent_rejections']}
        representatives.setdefault(reason,item)
        if call['response']['status']=='accepted':accepted.append(item)
        if reason=='duration exceeds remaining mission deadline':
            duration_rejections.append({'source_time':context['source_time'],'worker_time':call['t'],'duration':call['args']['duration'],
                'source_slack_seconds':600-context['source_time']-call['args']['duration'],
                'worker_slack_seconds':600-call['t']-call['args']['duration']})
    entities=w['private_case']['entities']
    def truth_at(t):return {e['key']:tuple(e['origin_xy']) for e in entities if t>=e['appeared_at']}
    ledger=protocol.ReportLedger(10,set(w['private_case'].get('classes',('road_obstruction','bank_anomaly','infrastructure_damage','inspection_vehicle','normal'))),truth_at)
    observations=sorted(delivered.values(),key=lambda o:(o['arrived_at'],o['evidence_id']));oi=0;report_diagnostics=[]
    for entry in w['reports']:
        while oi<len(observations) and observations[oi]['arrived_at']<=entry['t']:
            o=observations[oi];ledger.register(protocol.Evidence(protocol.Observation(**o),sensor[o['evidence_id']]['entity_key']));oi+=1
        req=entry['request'];assert ledger.submit(req,entry['t'])==entry['response']
        r=ledger.reports[req['report_id']];v=r.versions[-1]
        report_diagnostics.append({'t':entry['t'],'report_id':r.report_id,'candidate_id':req['candidate_id'],'associated_key':r.associated_key,
            'measured_at':req['measured_at'],'provenance_valid':v.provenance_valid,'quality_valid':v.quality_valid,'identity_correct':v.identity_correct,
            'missing_evidence_ids':[i for i in req['evidence_ids'] if i not in delivered],
            'evidence_after_claimed_stamp':[i for i in req['evidence_ids'] if i in delivered and delivered[i]['measured_at']>req['measured_at']],
            'mixed_candidate_ids':[i for i in req['evidence_ids'] if i in delivered and delivered[i]['candidate_id']!=req['candidate_id']],
            'no_evidence_at_claimed_stamp':not any(i in delivered and delivered[i]['measured_at']==req['measured_at'] for i in req['evidence_ids'])})
    static=[]
    for e in entities:
        if not e['rule_id']:continue
        assert e['object_type']=='static'
        report=next((r for r in ledger.reports.values() if r.associated_key==e['key'] and not r.withdrawn and r.object_type=='static'),None)
        static.append({'entity_key':e['key'],'weight':1,'quality':scoring.final_static_quality(report.versions if report else [],tuple(e['origin_xy']),e['true_class'],600)})
    assert static==w['score']['static_entities']
    ss=sum(x['quality'] for x in static)/len(static);assert ss==w['score']['S']==0 and w['score']['Y']==(w['score']['C']+ss)/2
    source_hashes={Path(k).name:v for k,v in run['source_hashes'].items() if Path(k).name in [p.name for p in S.iterdir()]}
    result[label]={'episode':episode,'attempt':attempt,'status':terminal['status'],'Y':terminal['Y'],'max_clock_lag':w['max_clock_lag_seconds'],
        'requests':len(requests),'decoded_statuses':dict(reply_statuses),'formats':formats,'enqueued_and_worker_count':len(dispatched),'all_commands_exact_in_order':True,
        'feedback_entries_joined':feedback,'public_observations_joined':observations_shown,'tool_names':dict(Counter(c['name'] for c in w['tool_calls'])),
        'tool_rejections':dict(Counter(c['response'].get('reason') for c in w['tool_calls'] if c['response']['status']=='rejected')),
        'tool_acceptances':dict(Counter(c['name'] for c in w['tool_calls'] if c['response']['status']=='accepted')),'representatives':representatives,
        'duration_rejections':duration_rejections,'duration_rejections_admissible_at_source_time':sum(d['source_slack_seconds']>=0 for d in duration_rejections),
        'fences_removed':sum(x['audit'].get('wire_framing',{}).get('json_fence_removed',False) for x in logs if x['kind']=='decoded_reply'),
        'accepted_commands':accepted,'report_diagnostics':report_diagnostics,'report_slots':len(ledger.reports),
        'inspect_jobs':[{k:j[k] for k in ('job_id','drone_id','kind','status','created_at','started_at','finished_at','reason')} for j in w['jobs'] if j['kind']=='inspect'],
        'static_denominator':len(static),'static_qualities':static,'request_timing':requests,'source_hashes':source_hashes}
    input_hashes[label]={str(p):sha(p) for p in (root/'plan.json',a/'terminal.json',f/'run.json',f/'world.json',f/'actor.jsonl',op/'service-completion.json')}
assert source_sets[0]==source_sets[1]==source_sets[2]
old=REPO/'logs/inspection-eight-condition-dev-flights.v2/dev-L1-00-B3-M3-0/attempt-00'
t=read(old/'terminal.json');assert t['status']=='known_operational_failure' and t['Y'] is None and t['evidence']['flight']['failure_reasons']==['clock_lag_exceeded'] and t['retry_allowed'] is True
oldw=read(old/'flight/world.json');assert sha(old/'flight/world.json')==t['evidence']['flight']['world_sha256'] and oldw['max_clock_lag_seconds']==t['evidence']['flight']['max_clock_lag_seconds']
prior={'status':t['status'],'Y':t['Y'],'retry_allowed':t['retry_allowed'],'max_clock_lag':t['evidence']['flight']['max_clock_lag_seconds'],'terminal_sha256':sha(old/'terminal.json')}
data={'role':'bounded stored causal toolchain audit, no new model/physics/policy replay, no method optimization','passed':True,'same_embedded_source_count':len(source_sets[0]),
      'prior_M3_attempt00':prior,'episodes':result,'input_hashes':input_hashes}
(OUT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
print(json.dumps({'passed':True,'same_embedded_source_count':len(source_sets[0]),'counts':{k:{s:v[s] for s in ('requests','enqueued_and_worker_count','tool_rejections','tool_acceptances','report_slots','static_denominator')} for k,v in result.items()}},indent=2))
