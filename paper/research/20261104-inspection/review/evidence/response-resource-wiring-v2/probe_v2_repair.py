from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json,hashlib,tempfile,copy,traceback
R=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection');W=R/'planning/formal-admission-candidate.v3';F=W/'response-resource-freeze.v2';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(F/'binding.json')=='8441e871ceb1c2b95602eaa45d1b430a6b0bcb5dceeec03dbb360374c13356e0'
b=json.loads((F/'binding.json').read_bytes())
for rel,item in b['members'].items():assert sha(F/rel)==item['sha256']
sys.path.insert(0,str(R.parents[2]))
spec=importlib.util.spec_from_file_location('independent_response_fixture',F/'tests/test_response_diagnostics.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);a=h.diagnostics
assert Path(a.__file__)==F/'source/eval/inspection_response_diagnostics.py'
result={'subject_binding_sha256':sha(F/'binding.json'),'scope':'v2 reader repair with original v1 inputs, synthetic closed fixture and trusted-evidence stubs only; no actor validator/native/model/physics/formal RNG'}

def rejected(action):
    try:action()
    except (ValueError,KeyError,FileNotFoundError) as error:return {'type':type(error).__name__,'message':str(error)}
    raise AssertionError('expected refusal')

def rehash(snapshot,project):snapshot['files']={str(p.relative_to(project)):sha(p) for p in project.rglob('*') if p.is_file()}

with tempfile.TemporaryDirectory(prefix='response-wiring-') as tmp:
    base=Path(tmp).resolve();single=base/'single';single.mkdir();p,request,snapshot,evidence=h.fixture(single,retry=True,context=True)
    one=a.episode_response_diagnostics(snapshot,request,evidence,p)
    assert one['valid_flight_usage']=={'calls':1,'tokens':102} and one['all_known_attempt_usage']=={'calls':2,'tokens':110}
    assert one['context_rejections']==1 and one['money_cost'] is None and one['energy_cost'] is None
    result['one_call_retry_valid_vs_allknown_counts']=True
    repeats=[{**copy.deepcopy(one),'repeat_ordinal':i,'method':'delay-20'} for i in range(3)]
    summary=a.summarize_response_resources(repeats);row=summary['root_method_rows'][0]
    assert row['valid_flight_usage']=={'calls':1,'tokens':102} and row['all_known_attempt_usage']=={'calls':2,'tokens':110} and row['known_attempt_count']==6
    result['complete_original_repeat_means_and_retry_count']=True
    result['missing_repeat_rejected']=rejected(lambda:a.summarize_response_resources(repeats[:-1]))
    wrong=copy.deepcopy(repeats);wrong[-1]['repeat_ordinal']=1
    result['duplicate_repeat_rejected']=rejected(lambda:a.summarize_response_resources(wrong))

    td=base/'two';td.mkdir();p,request,snapshot,evidence=h.fixture(td)
    slot,ep,before=snapshot;reservation='formal-latency-aaaaaaaaaaaa-0000-a0';op=p/'logs/inspection-model-resource-operations.v1'/reservation
    first=json.loads((op/'call-0001/response.json').read_bytes());second=copy.deepcopy(first);second['receipt']=str(op/'call-0002/response.json');h.write(op/'call-0002/response.json',second)
    usage=p/'run/qualification/local-model-episodes'/reservation/'usage-receipt.json';h.write(usage,{'usage':{'calls':2,'tokens':204},'role':'synthetic known usage, not actual settlement'})
    actor=ep/'attempt-00/flight/actor.jsonl';entries=[json.loads(line) for line in actor.read_text().splitlines()];more=copy.deepcopy(entries)
    more[0]['request_ordinal']=2;more[0]['source_sim_time']=30.
    more[1]['response']['submitted_ordinal']=2;more[1]['response']['receipt']=second['receipt'];more[1]['worker_before_request'][0]=30.;more[1]['worker_at_model_reply'][0]=36.;more[1]['worker_after_response'][0]=56.05
    more[2]['audit']['request']['request_ordinal']=2;more[2]['dispatch_age_sim_seconds']=56.05-30.
    actor.write_text('\n'.join(json.dumps(entry) for entry in entries+more)+'\n')
    join=copy.deepcopy(evidence['causal_request_joins'][0]);join['decision_attempt']=2;join['admitted_call_ordinal']=2;evidence['causal_request_joins'].append(join);rehash(before,p)
    assert sum(json.loads((op/f'call-{i:04d}/response.json').read_bytes())['usage']['tokens'] for i in (1,2))==204
    original_before=copy.deepcopy(before);actor_before=actor.read_bytes();evidence_before=copy.deepcopy(evidence)
    value=a.episode_response_diagnostics(snapshot,request,evidence,p)
    assert value['valid_flight_usage']=={'calls':2,'tokens':204} and value['all_known_attempt_usage']=={'calls':2,'tokens':204}
    assert len(value['valid_flight_responses'])==2 and [r['service_generation_and_journaling_wall_seconds'] for r in value['valid_flight_responses']]==[5.,5.]
    assert before==original_before and evidence==evidence_before and actor.read_bytes()==actor_before
    new_inputs={'slot':slot.__dict__,'evidence':evidence,'actor':entries+more,'original_calls':[first,second],'native_usage':{'calls':2,'tokens':204},'snapshot_file_count':len(before['files'])}
    old_path=R/'review/evidence/response-resource-wiring-v1/two-call-inputs.json';old_inputs=json.loads(old_path.read_bytes())
    old_root=old_inputs['original_calls'][0]['receipt'].split('/logs/inspection-model-resource-operations.v1/')[0]
    def relocated(value,root):
        if isinstance(value,str):return value.replace(root,'<PROJECT>')
        if isinstance(value,list):return [relocated(v,root) for v in value]
        if isinstance(value,dict):return {k:relocated(v,root) for k,v in value.items()}
        return value
    assert relocated(old_inputs,old_root)==relocated(new_inputs,str(p))
    result['original_v1_two_call_input_replayed_only_project_path_relocation']=True
    result['original_v1_input_sha256']=sha(old_path)
    result['two_generated_calls_accepted']={'calls':2,'tokens':204,'response_count':2,'service_intervals':[5.,5.],'original_snapshot_and_actor_evidence_unchanged':True}
    (OUT/'two-call-replayed-inputs.json').write_text(json.dumps(new_inputs,indent=2)+'\n')

    # A declined reply still must leave the original snapshot available for its later generated reply.
    td=base/'declined_then_generated';td.mkdir();p,request,snapshot,evidence=h.fixture(td,non_generation=True)
    slot,ep,before=snapshot;reservation='formal-latency-aaaaaaaaaaaa-0000-a0';op=p/'logs/inspection-model-resource-operations.v1'/reservation
    generated={'kind':'response','status':'completed','usage':{'calls':1,'tokens':102},'wall_seconds':5.,'deadline_exceeded':False,'receipt':str(op/'call-0001/response.json')}
    h.write(op/'call-0001/response.json',generated);h.write(p/'run/qualification/local-model-episodes'/reservation/'usage-receipt.json',{'usage':{'calls':1,'tokens':102},'role':'synthetic known usage'})
    actor=ep/'attempt-00/flight/actor.jsonl';entries=[json.loads(line) for line in actor.read_text().splitlines()]
    entries.extend([{'kind':'public_model_request','request_ordinal':2,'source_sim_time':30.},
        {'kind':'actual_model_response','response':{**generated,'submitted_ordinal':2,'parent_end_to_end_wall_seconds':6.},'injected_delay_requested_seconds':20,'injected_delay_wall_seconds':20.05,'final_after_flight':False,'worker_before_request':[30.,0.,0.,96.],'worker_at_model_reply':[36.,0.,0.,96.],'worker_after_response':[56.05,0.,0.,96.]},
        {'kind':'decoded_reply','audit':{'request':{'request_ordinal':2},'status':'decoded_not_tool_acceptance'},'dispatch_age_sim_seconds':56.05-30.,'dispatch_age_wall_seconds':26.1,'commands_enqueued_not_accepted':[]}])
    actor.write_text('\n'.join(json.dumps(row) for row in entries)+'\n');evidence['causal_request_joins'].append({'decision_attempt':2,'kind':'response','admitted_call_ordinal':1,'decode_status':'decoded_not_tool_acceptance'});rehash(before,p)
    original_before=copy.deepcopy(before);value=a.episode_response_diagnostics(snapshot,request,evidence,p)
    assert value['valid_flight_usage']=={'calls':1,'tokens':102} and len(value['valid_flight_responses'])==2 and value['non_generation_responses']==1
    assert [r['service_generation_and_journaling_wall_seconds'] for r in value['valid_flight_responses']]==[None,5.]
    assert before==original_before
    result['not_started_then_generated_accepted_without_zero_imputation']={'calls':1,'tokens':102,'response_count':2,'service_intervals':[None,5.],'original_snapshot_unchanged':True}
    call=op/'call-0001/response.json';call.write_bytes(call.read_bytes()+b' ')
    result['old_original_call_SHA_drift_still_rejected']=rejected(lambda:a.episode_response_diagnostics(snapshot,request,evidence,p))

    negatives={}
    for fault in ('missing_actor','unknown_terminal','wrong_kind','mismatched_tokens','changed_actor_bytes','missing_usage','missing_join'):
        td=base/fault;td.mkdir();p,request,snapshot,evidence=h.fixture(td);slot,ep,before=snapshot
        actor=ep/'attempt-00/flight/actor.jsonl'
        if fault=='missing_actor':actor.unlink();before['files'].pop(str(actor.relative_to(p)))
        elif fault=='unknown_terminal':path=ep/'attempt-00/terminal.json';h.write(path,{'status':'unknown'});rehash(before,p)
        elif fault=='wrong_kind':
            entries=[json.loads(line) for line in actor.read_text().splitlines()];entries[1]['response']['kind']='not_started';actor.write_text('\n'.join(json.dumps(e) for e in entries)+'\n');rehash(before,p)
        elif fault=='mismatched_tokens':
            path=next((p/'run/qualification').rglob('usage-receipt.json'));v=json.loads(path.read_bytes());v['usage']['tokens']=103;h.write(path,v);rehash(before,p)
        elif fault=='changed_actor_bytes':actor.write_bytes(actor.read_bytes()+b' ')
        elif fault=='missing_usage':
            path=next((p/'run/qualification').rglob('usage-receipt.json'));path.unlink();before['files'].pop(str(path.relative_to(p)))
        else:evidence['causal_request_joins']=[]
        negatives[fault]=rejected(lambda:a.episode_response_diagnostics(snapshot,request,evidence,p))
    result['independent_reader_negatives']=negatives
for rel,item in b['members'].items():assert sha(F/rel)==item['sha256']
result['frozen_members_preserved']=True
(OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
