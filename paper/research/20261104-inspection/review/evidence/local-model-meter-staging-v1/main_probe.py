"""Independent CPU-only integration probes. No service/MLX/world is invoked."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import queue
import shutil
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
sys.path.insert(0, str(REPO))
from eval.inspection_llm_actor import PublicLLMActor
import eval.run_inspection_llm_development as runner

BASE = Path('/tmp/inspection-model-meter-staging-review-v1')
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frame(t=0):
    return {'sim_time':t,'task_card':{'task_card_version':1},'aircraft':[
        {'drone_id':0,'retired':False,'link_available':True,'camera_quality':1}],
        'jobs':[],'events':[],'coverage':[],'observations':[],'report_slots_used':0}
def response(text='{"commands":[]}'):
    return {'kind':'response','status':'completed','wall_seconds':1,
        'deadline_exceeded':False,'text':text,'parent_end_to_end_wall_seconds':1}

def main_probe(mode, ablation=None):
    root = Path(tempfile.mkdtemp(prefix=mode+'-', dir=BASE))
    (root/'envs/inspection').mkdir(parents=True)
    (root/'envs/__init__.py').write_text('# fake env\n')
    (root/'eval').mkdir()
    for name in ('inspection_cases.py','inspection_scenarios.py','run_inspection_reference.py',
        'inspection_llm_actor.py','inspection_llm_protocol.py','inspection_model_client.py','inspection_model_service.py','inspection_process_cleanup.py'):
        (root/'eval'/name).write_text('# frozen fake source\n')
    dep = root/'fake-dependencies';dep.mkdir()
    (dep/'__init__.py').write_text('# fake dependency\n')
    (dep/'bullet.so').write_bytes(b'fixture')
    run = root/'run'; (run/'planning').mkdir(parents=True)
    (run/'receipt.json').write_text('{}')
    limits={'max_calls':2,'max_request_seconds':1,'max_total_tokens':20,
        'max_input_tokens':10,'max_output_tokens':2}
    (run/'planning/local-model-registry.v1.json').write_text(json.dumps({'models':{'M1':{
        'generation_profile':limits,'download_receipt':'receipt.json',
        'download_receipt_sha256':sha(run/'receipt.json'),'tokenizer_config':{}}}}))
    case = root/'case.json';case.write_text('{}')
    out=root/'out';operation=root/'staging/synthetic';operation.mkdir(parents=True)
    progress=[0.,0.,0.,0.]
    updates=queue.Queue();commands=queue.Queue()
    is_response_mode=mode in ('normal_reply','late_reply','after600_reply','known_runtime','unknown_transport','not_started')
    class Array(list):
        def get_lock(self): return contextlib.nullcontext()
    shared=Array(progress)
    class FakeProcess:
        pid=None;exitcode=None
        def start(self):
            self.pid=987654;self.exitcode=0
            shared[0]=0. if is_response_mode else 600.
            if mode=='private_deleted': (out/'private-case.json').unlink()
            if mode=='source_deleted': (root/'eval/inspection_cases.py').unlink()
            if mode=='dependency_deleted': (dep/'bullet.so').unlink()
            if mode=='dependency_changed': (dep/'bullet.so').write_bytes(b'changed')
            if mode=='source_changed': (root/'eval/inspection_cases.py').write_text('# changed source\n')
            if mode!='world_missing':
                (out/'world.json').write_text(json.dumps({'status':'completed','simulated_seconds':600,'physics_steps_executed':144000,'fixture':True}))
            if mode=='world_list': (out/'world.json').write_text('[]')
            if mode=='world_null': (out/'world.json').write_text('null')
            if mode=='world_incomplete': (out/'world.json').write_text(json.dumps({'status':'completed','simulated_seconds':599,'physics_steps_executed':143760,'fixture':True}))
            if mode=='bad_completion': (operation/'service-completion.json').write_text('{')
            else: (operation/'service-completion.json').write_text(json.dumps({'usage':{'calls':0 if mode=='not_started' or not is_response_mode else 1,'tokens':0},'fixture':True}))
            if is_response_mode: updates.put({'kind':'state','state':frame()})
            if mode=='missing_completion': (operation/'service-completion.json').unlink()
            if mode=='bad_usage': (operation/'service-completion.json').write_text(json.dumps({'usage':{'calls':True,'tokens':0}}))
            updates.put({'kind':'done','status':'completed'})
        def is_alive(self): return False
        def join(self,timeout=None): pass
    proc=FakeProcess()
    class Context:
        calls=0
        def Queue(self):
            self.calls+=1
            return updates if self.calls==2 else commands
        def Array(self,*a): return shared
        def Process(self,*a,**k): return proc
    class FakeClient:
        def __init__(self,*a,**k): self.ready={'reservation':'synthetic'};self.process=SimpleNamespace(poll=lambda:0);self.pending=False
        def submit(self,*a): self.pending=True
        def poll(self):
            if not self.pending:return None
            self.pending=False;shared[0]=600. if mode=='after600_reply' else 5.
            if mode=='unknown_transport':raise runner.ModelOperationUnknown('pure fixture unknown transport')
            if mode=='not_started':return {'kind':'not_started','parent_end_to_end_wall_seconds':.01}
            r=response('{"commands":[{"name":"enter_safe_exit","args":{"drone_id":0,"expected_version":0}}]}')
            if mode=='known_runtime':r['status']='runtime_failure'
            if mode=='late_reply':r['parent_end_to_end_wall_seconds']=2
            return r
        def finish(self): return None
    def spec(name):
        return SimpleNamespace(origin=str(dep/'__init__.py' if name=='PyFlyt' else dep/'bullet.so'))
    argv=['runner','--run',str(run),'--reservation','synthetic','--model','M1','--architecture','B4',
        '--case',str(case),'--sampler-seed','1','--out',str(out)]
    if ablation is not None: argv+=['--ablation',ablation]
    captured=io.StringIO();raised=None;code=None;actor_args={}
    real_actor=runner.PublicLLMActor
    def actor_factory(*a,**k):
        actor_args.update(k);return real_actor(*a,**k)
    with patch.object(runner,'ROOT',root),patch.object(runner.mp,'get_context',return_value=Context()),\
        patch.object(runner,'LocalModelClient',FakeClient),patch.object(runner,'operation_directory',return_value=operation),patch.object(runner,'PublicLLMActor',side_effect=actor_factory),patch.object(runner.importlib.util,'find_spec',side_effect=spec),\
        patch.object(runner.importlib.metadata,'version',return_value='pure-fake'),patch.object(sys,'argv',argv),\
        contextlib.redirect_stdout(captured):
        try: code=runner.main()
        except BaseException as e: raised=type(e).__name__+': '+str(e)
    manifest=json.loads((out/'run.json').read_text())
    return {'fixture':True,'mode':mode,'return_code':code,'raised':raised,
        'status':manifest['status'],'private_case_unchanged':manifest.get('private_case_unchanged'),
        'sources_unchanged':manifest.get('sources_unchanged'),'dependencies_unchanged':manifest.get('dependencies_unchanged'),
        'model_operation_unknown':'model_operation_unknown' in manifest,'ablation_recorded':manifest.get('ablation'),
        'actor_ablation':actor_args.get('ablation'),'has_world_hash':'world_sha256' in manifest,
        'path':str(root),'stdout':captured.getvalue(),'commands_enqueued':list(commands.queue),
        'error_tail':(manifest.get('operational_error') or '').splitlines()[-1:]}

def actor_probes():
    rows={}
    actors=[PublicLLMActor('B3'),PublicLLMActor('B4'),PublicLLMActor('B4',ablation='periodic')]
    for a in actors:
        a.ingest(frame());a.start_request();a.finish_request(response())
        state=frame(10);state['aircraft'][0]['link_available']=False;a.ingest(state)
    rows['event_due_at10']=[a.due() for a in actors]
    a=PublicLLMActor('B4');a.ingest(frame());a.start_request();a.finish_request(response())
    state=frame(10);state['jobs']=[{'job_id':'job-1','status':'succeeded'}];a.ingest(state)
    rows['success_due_at10']=a.due();rows['success_monitor_triggers']=sorted(a.pending_triggers)
    a=PublicLLMActor('B4',ablation='no-cross-aircraft-memory')
    state=frame();state['observations']=[{'evidence_id':'ev-a','candidate_id':'local-a','drone_id':0,
        'measured_at':0,'object_type':'static','estimate_xy':[0,0]},
        {'evidence_id':'ev-b','candidate_id':'local-b','drone_id':1,
        'measured_at':0,'object_type':'static','estimate_xy':[1,0]}]
    a.ingest(state);content=json.loads(a.start_request()['messages'][1]['content'])
    rows['no_cross_memory_keys']=list(content['memory']);rows['no_cross_raw_drone_ids']=sorted({o['drone_id'] for o in content['public_state']['observations']})
    a.finish_request(response());a.ingest(frame(20));content=json.loads(a.start_request()['messages'][1]['content'])
    rows['no_cross_retained_cross_aircraft_evidence_ids']=[o['evidence_id'] for o in content['public_state']['observations']]
    deadline=[]
    command='{"commands":[{"name":"enter_safe_exit","args":{"drone_id":0,"expected_version":0}}]}'
    for t,change in [(599,{'wall_seconds':61}), (600,{})]:
        a=PublicLLMActor('B4');a.ingest(frame());a.start_request();a.ingest(frame(t))
        cmd,audit=a.finish_request({**response(command),**change});deadline.append({'at':t,'commands':cmd,'audit':audit['status']})
    rows['late_and_after600']=deadline
    return rows

if __name__=='__main__':
    result={'role':'synthetic independent 0 physics / 0 real generation review',
        'main':[main_probe(m) for m in ['valid_fake','private_deleted','source_changed','world_missing','bad_completion',
            'normal_reply','late_reply','after600_reply','known_runtime','unknown_transport','not_started',
            'source_deleted','dependency_deleted','dependency_changed','missing_completion','bad_usage',
            'world_incomplete','world_list','world_null']],
        'cli_ablations':[main_probe('valid_fake',flag) for flag in ('periodic','no-cross-aircraft-memory','no-progressive-confirmation-guidance')],
        'actor':actor_probes()}
    (BASE/'probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
