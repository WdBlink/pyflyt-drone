from pathlib import Path
from types import SimpleNamespace
import importlib.util,sys,json,hashlib,tempfile,copy,traceback
R=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection');W=R/'planning/formal-admission-candidate.v3';F=W/'response-resource-freeze.v1';OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(F/'binding.json')=='2cbc108e4e7550aa0abbb4b904f07b00d98256c561f3434497e5c21aa89b31a3'
b=json.loads((F/'binding.json').read_bytes())
for rel,item in b['members'].items():assert sha(F/rel)==item['sha256']
sys.path.insert(0,str(R.parents[2]))
spec=importlib.util.spec_from_file_location('independent_response_fixture',F/'tests/test_response_diagnostics.py');h=importlib.util.module_from_spec(spec);sys.modules[spec.name]=h;spec.loader.exec_module(h);a=h.diagnostics
assert Path(a.__file__)==F/'source/eval/inspection_response_diagnostics.py'
result={'subject_binding_sha256':sha(F/'binding.json'),'scope':'frozen reader, synthetic closed fixture and trusted-evidence stub only; no actor validator/native/model/physics/formal RNG'}

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
    try:a.episode_response_diagnostics(snapshot,request,evidence,p)
    except TypeError as error:
        assert str(error)=="'float' object is not subscriptable"
        result['two_valid_call_failure']={'type':type(error).__name__,'message':str(error),'expected_calls':2,'expected_tokens':204,'actor_requests':2,'actor_responses':2,'causal_joins':2}
        (OUT/'two-call-traceback.txt').write_text(traceback.format_exc())
    else:raise AssertionError('v1 shadowing bug not reproduced')
    # Retain only the small synthetic proof inputs; no real private/run material.
    (OUT/'two-call-inputs.json').write_text(json.dumps({'slot':slot.__dict__,'evidence':evidence,'actor':entries+more,'original_calls':[first,second],'native_usage':{'calls':2,'tokens':204},'snapshot_file_count':len(before['files'])},indent=2)+'\n')

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
