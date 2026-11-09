"""Actual subprocess/cache faults over declared synthetic receipts, no flight/MLX."""

import fcntl
import hashlib
import json
from pathlib import Path
import sys

import pytest

from eval.inspection_episode_receipts import UnresolvedEpisode, canonical_digest, durable_once
import eval.run_inspection_llm_episode as module


CHILD = r'''
from pathlib import Path
import hashlib,json,sys,subprocess
flight,op,planfile,reservation,mode,counter=map(Path,sys.argv[1:])
p=json.loads(planfile.read_text());reservation=str(reservation);mode=str(mode)
count=int(counter.read_text())+1 if counter.exists() else 1;counter.write_text(str(count))
if mode=='orphan':
 leak=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 Path(str(counter)+'.orphan').write_text(str(leak.pid))
flight.mkdir();op.mkdir()
def put(path,v):path.write_text(json.dumps(v)+'\n')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
(flight/'private-case.json').write_bytes(Path(p['case_path']).read_bytes())
world={'status':'completed','clock_mode':'wallclock_1x','simulated_seconds':600,
 'physics_steps_executed':144000,'physics_hz':240,'control_hz':120,'max_clock_lag_seconds':.02,
 'private_case_file_sha256':p['case_file_sha256'],
 'score':{'C':.5,'S':.5,'D':0,'Q':.3333333333333333,'Y':0.,'H':1,'E':0,'SR':0}}
badclock=mode in ('always-clock','clock-and-seed') or mode=='first-clock' and count==1
if badclock:world['max_clock_lag_seconds']=.2
put(flight/'world.json',world)
put(op/'service-intent.json',{'reservation':reservation,'owner_frozen_plan':p['owner_frozen_plan'],
 'model_receipt_sha256':p['model_receipt_sha256'],'reserved':p['reservation_usage'],
 'source_sha256':p['service_sha256'],'request_budget':p['generation_profile']})
state={'sim_time':0,'task_card':{'task_card_version':1},'aircraft':[],'jobs':[],'events':[],
 'coverage':[],'observations':[],'report_slots_used':0}
messages=[{'role':'system','content':'synthetic unit fixture only'},
 {'role':'user','content':json.dumps({'public_state':state,'recent_tool_results':[],'memory':{}})}]
mh=hashlib.sha256(json.dumps(messages,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
before=[.1,0,0,80];after=[1.,20,0,80]
attempt=2 if mode=='context-gap' else 1
request={'request_ordinal':attempt,'source_sim_time':0,'messages':messages,'messages_sha256':mh}
public={'kind':'public_model_request',**request,'sampler_seed':p['sampler_seed']+attempt-1,'worker_before_request':before}
usage={'calls':0,'tokens':0}
if mode=='not-started':
 response={'kind':'not_started','reason':'frozen local request budget/context exhausted'}
 audit={'status':'ignored_noncompleted_or_late','request':request}
else:
 call=op/'call-0001';call.mkdir();text='{"commands":[]}'
 put(call/'request.json',{'messages':messages,'seed':p['sampler_seed']+attempt-1+(mode in ('wrong-seed','clock-and-seed')),
  'input_tokens_submitted':3,'max_output_tokens':p['generation_profile']['max_output_tokens'],
  'temperature':.2,'top_p':.95,'enable_thinking':False})
 (call/'tokens.jsonl').write_text(json.dumps({'token':1,'text':text,'runtime_prompt_tokens':3,'runtime_generation_tokens':1})+'\n')
 late=mode=='policy-timeout'
 response={'kind':'response','status':'policy_timeout' if late else 'completed','text':text,
  'generated_tokens_observed':1,'emitted_tokens_journaled':1,'input_tokens_submitted':3,
  'usage':{'calls':1,'tokens':4},'wall_seconds':61. if late else 1.,'deadline_exceeded':late,
  'receipt':str((call/'response.json').resolve())}
 put(call/'response.json',response)
 usage={'calls':1,'tokens':4}
 audit={'status':'ignored_noncompleted_or_late','request':request} if late else {
  'status':'decoded_not_tool_acceptance','request':request,'reply':{'commands':[]},'wire_framing':{'json_fence_removed':False}}
if mode!='missing-completion':put(op/'service-completion.json',{'usage':usage})
rows=[]
if mode=='context-gap':rows.append({'kind':'policy_context_failure','decision_attempt':1,'model_request_dispatched':False})
rows += [public,{'kind':'actual_model_response','response':{**response,'submitted_ordinal':1,'parent_end_to_end_wall_seconds':1.1},
 'worker_before_request':before,'worker_after_response':after,'final_after_flight':False},
 {'kind':'decoded_reply','audit':audit,'commands_enqueued_not_accepted':[],'worker_progress_at_enqueue':after}]
if mode=='bad-enqueue':rows[-1]['worker_progress_at_enqueue']=[601,0,0,80]
(flight/'actor.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
run={'status':'completed','source_hashes':p['source_hashes'],'dependency_hashes':p['dependency_hashes'],
 'sources':{name:Path(name).read_text() for name in p['source_hashes']},
 'sources_unchanged':mode!='source-fault','dependencies_unchanged':True,
 'private_case_sha256':p['case_file_sha256'],'private_case_unchanged':True,'policy':p['condition'],
 'runtime_thread_environment':p['thread_environment'],'requested_seconds':600,'realtime':True,
 'world_sha256':sha(flight/'world.json'),'reservation':reservation,'sampler_base_seed':p['sampler_seed'],
 'ablation':None,'model_registry_sha256':p['model_registry_sha256'],'model_receipt_sha256':p['model_receipt_sha256'],
 'model_generation_profile':p['generation_profile'],'actor_mutating_calls':0,
 'model_operation':{'usage':usage,'sha256':sha(op/'service-completion.json') if (op/'service-completion.json').exists() else None}}
if mode=='embedded-source':run['sources']={name:'mismatched embedded source' for name in p['source_hashes']}
put(flight/'run.json',run)
'''


def fixture(tmp_path, monkeypatch):
    script = tmp_path/"synthetic-child.py"
    script.write_text(CHILD)
    case = tmp_path/"fake-private.json"
    case.write_text('synthetic fixture, not a scientific case')
    evaluator = tmp_path/"fake-evaluator.py"
    evaluator.write_text('print(\'{"score_equal":true,"score":{"Y":0.0}}\')\n')
    protocol = Path(module.__file__).with_name("inspection_llm_protocol.py").resolve()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    profile = {"max_calls":32,"max_total_tokens":180000,"max_input_tokens":8192,"max_output_tokens":1024,
        "max_request_seconds":60,"temperature":.2,"top_p":.95,"enable_thinking":False}
    plan = {"condition":"B3-M1","repeat_ordinal":0,"max_attempts":2,"max_clock_lag_seconds":.1,
        "case_id":"dev-fixture-only","case_path":str(case),"case_file_sha256":sha(case),
        "source_hashes":{str(script):sha(script),str(protocol):sha(protocol)},"dependency_hashes":{},
        "thread_environment":{},"sampler_seed":42,"model_registry_sha256":"a"*64,
        "model_receipt_sha256":"b"*64,"owner_frozen_plan":"c"*64,"service_sha256":"d"*64,
        "generation_profile":profile,"reservation_usage":{"calls":32,"tokens":180000},
        "evaluator":str(evaluator),"evaluator_sha256":sha(evaluator),"physics_python":sys.executable}
    directory=tmp_path/"episode"; directory.mkdir()
    planfile=tmp_path/"plan-input.json";planfile.write_text(json.dumps(plan))
    run=tmp_path/"owner";(run/"resource-requests").mkdir(parents=True)
    operations=tmp_path/"operations";operations.mkdir()
    monkeypatch.setattr(module,"operation_directory",lambda reservation:operations/reservation)
    counter=tmp_path/"dispatch-count";effects=[]
    def owner(run, action, packet, operation_id, log, reservation=None):
        value=json.loads(packet.read_text());effects.append((action,value))
        durable_once(log,{"role":"fake owner effect only, no actual paper state","action":action})
        return {"status":"reserved" if action=="budget-reserve" else value["status"]}
    def command(mode):
        return lambda out,reservation:[sys.executable,str(script),str(out),str(operations/reservation),str(planfile),reservation,mode,str(counter)]
    def execute(mode="ok", **kw):
        return module.execute_development_llm_episode(directory,plan,command(mode),run=run,
            reservation_prefix="fixture-M1",call_owner=owner,**kw)
    return plan,directory,counter,effects,execute


@pytest.mark.parametrize("mode", ["ok","not-started","policy-timeout","context-gap"])
def test_completed_policy_zero_is_preserved_and_never_repeated(tmp_path,monkeypatch,mode):
    _,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    y,receipt=execute(mode)
    assert y==0 and not receipt["cache_reused"]
    assert [a for a,_ in effects]==["budget-reserve","budget-settle"]
    assert effects[-1][1]["status"]=="known"
    y,cached=execute("always-clock")
    assert y==0 and cached["cache_reused"] and counter.read_text()=="1" and len(effects)==2
    assert json.loads((directory/"attempt-00/terminal.json").read_text())["retry_allowed"] is False


def test_one_known_clock_failure_and_finite_retry(tmp_path,monkeypatch):
    _,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    _,receipt=execute("first-clock")
    assert receipt["actual_attempts"]==2 and counter.read_text()=="2" and len(effects)==4
    assert json.loads((directory/"attempt-00/terminal.json").read_text())["retry_allowed"] is True


def test_exhaustion_preserves_both_and_blocks_more(tmp_path,monkeypatch):
    _,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    for mode in ("always-clock","ok"):
        with pytest.raises(UnresolvedEpisode,match="exhausted"):execute(mode)
    assert counter.read_text()=="2" and len(effects)==4


@pytest.mark.parametrize("mode,settlement", [("missing-completion","unknown"),("wrong-seed","known"),("source-fault","known"),
    ("clock-and-seed","known"),("bad-enqueue","known"),("embedded-source","known")])
def test_unknown_and_causal_source_faults_are_settled_truthfully_and_block_retry(tmp_path,monkeypatch,mode,settlement):
    _,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    with pytest.raises(UnresolvedEpisode,match="blocks"):execute(mode)
    with pytest.raises(UnresolvedEpisode,match="blocks"):execute("ok")
    assert counter.read_text()=="1" and len(effects)==2
    assert effects[-1][1]["status"]==settlement
    assert effects[-1][1]["usage"]==({} if settlement=="unknown" else {"calls":1,"tokens":4})


def test_missing_terminal_blocks_without_dispatch_or_reservation(tmp_path,monkeypatch):
    plan,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    durable_once(directory/"plan.json",plan);(directory/"attempt-00").mkdir()
    with pytest.raises(UnresolvedEpisode,match="reconciliation"):execute()
    assert not counter.exists() and not effects


def test_modified_payload_or_other_plan_cannot_reuse_completed_slot(tmp_path,monkeypatch):
    plan,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    execute();(directory/"attempt-00/flight/actor.jsonl").write_text('')
    with pytest.raises(UnresolvedEpisode,match="payload changed"):execute()
    assert counter.read_text()=="1" and len(effects)==2
    plan["sampler_seed"]+=1
    with pytest.raises(UnresolvedEpisode,match="identity changed"):execute()


def test_concurrent_slot_owner_cannot_reserve_or_dispatch(tmp_path,monkeypatch):
    _,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    with (directory/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(UnresolvedEpisode,match="another coordinator"):execute()
    assert not counter.exists() and not effects


def test_settled_archive_missing_blocks_cache_without_new_dispatch(tmp_path,monkeypatch):
    _,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    execute()
    terminal=json.loads((directory/"attempt-00/terminal.json").read_text())
    archive=next(Path(p) for p in terminal["settlement_file_hashes"] if p.endswith('.zip'))
    archive.unlink()
    with pytest.raises(UnresolvedEpisode,match="settled archive"):execute()
    assert counter.read_text()=="1" and len(effects)==2


def test_started_receipt_failure_still_stops_spawned_process(tmp_path,monkeypatch):
    _,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    original=module.durable_once
    observed=[]
    def broken(path,value):
        if Path(path).name=='started.json':
            observed.append(value['pid'])
            raise OSError('injected durable start write failure')
        return original(path,value)
    monkeypatch.setattr(module,'durable_once',broken)
    with pytest.raises(OSError,match='start write'):execute()
    from eval.inspection_process_cleanup import _inventory
    assert observed and all(pid not in _inventory() for pid in observed)
    assert len(effects)==1 and not (directory/'attempt-00/terminal.json').exists()
    with pytest.raises(UnresolvedEpisode,match='reconciliation'):execute()


def test_normal_exit_cannot_admit_a_live_new_python_orphan(tmp_path,monkeypatch):
    _,directory,counter,effects,execute=fixture(tmp_path,monkeypatch)
    from eval.inspection_process_cleanup import _inventory
    import os,signal
    try:
        try:
            value,_=execute('orphan')
        except UnresolvedEpisode:
            value=None
        pid=int(Path(str(counter)+'.orphan').read_text())
        row=_inventory().get(pid)
        alive=row is not None and not row['status'].startswith('Z')
        assert value is None or not alive
        if value is None:
            assert json.loads((directory/'attempt-00/terminal.json').read_text())['retry_allowed'] is False
            with pytest.raises(UnresolvedEpisode,match='blocks'):execute()
        assert counter.read_text()=='1' and len(effects)==2
    finally:
        orphan=Path(str(counter)+'.orphan')
        if orphan.exists():
            try:os.killpg(int(orphan.read_text()),signal.SIGKILL)
            except ProcessLookupError:pass
