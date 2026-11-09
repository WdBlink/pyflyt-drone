from pathlib import Path
import ast,copy,json,types,sys,contextlib,io,hashlib,difflib
ROOT=Path.cwd().resolve();sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
from eval import run_inspection_llm_development as m
from eval.inspection_llm_actor import PublicLLMActor
T=Path('/tmp/inspection-injected-delay-review')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=ROOT/'eval/run_inspection_llm_development.py'
assert H(source)=='c3bed0403ca4a8e7f5d79c0db3df5da0aeb27e46e37423e94474447fbd4da552'
tree=ast.parse(source.read_text());consume=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='consume_response')
# Compile the actual unmodified nested body in a minimal lexical fixture; no main worker/client is constructed.
wrapper=ast.parse('def make_consumer():\n mutation_count=0\n def counts(): return mutation_count\n return None\n').body[0]
wrapper.body=[wrapper.body[0],consume,wrapper.body[1],ast.Return(ast.Tuple(elts=[ast.Name(id='consume_response',ctx=ast.Load()),ast.Name(id='counts',ctx=ast.Load())],ctx=ast.Load()))]
module=ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[]))
class Clock:
 def __init__(self):self.now=0.;self.sleeps=[]
 def monotonic(self):return self.now
 def sleep(self,n):self.sleeps.append(n);self.now+=n
class Commands:
 def __init__(self):self.values=[]
 def put(self,v):self.values.append(copy.deepcopy(v))
raw={'kind':'response','status':'completed','text':'{"commands":[{"name":"enter_safe_exit","args":{"drone_id":0,"expected_version":0}}]}','wall_seconds':1.,'parent_end_to_end_wall_seconds':1.1,'deadline_exceeded':False,'usage':{'calls':1,'tokens':9}}
results=[]
cases=[('zero',0,100,False,{}),('delay2',2,100,False,{}),('delay5',5,100,False,{}),('delay10',10,100,False,{}),('delay20',20,100,False,{}),
('exact60',20,100,False,{'wall_seconds':60.,'parent_end_to_end_wall_seconds':60.}),('over60',20,100,False,{'wall_seconds':60.,'parent_end_to_end_wall_seconds':60.000001}),
('policy_timeout',20,100,False,{'status':'policy_timeout','deadline_exceeded':True}),('not_started',20,100,False,{'kind':'not_started','status':'not_started'}),
('runtime_failure',20,100,False,{'status':'runtime_failure'}),('physical600_stale_actor',20,600,False,{}),('cross600',2,599,False,{}),('final',20,590,True,{})]
original_time=m.time
for name,delay,start,final,changes in cases:
 clock=Clock();m.time=clock;logs=[];commands=Commands();actor=PublicLLMActor('B4');actor.state={'sim_time':min(start,580)};actor.inflight={'source_sim_time':start,'request_ordinal':1}
 progress=lambda:[min(start+clock.now,600),0,0,80]
 env={'time':clock,'inject_reply_wait':m.inject_reply_wait,'physical_progress':progress,'log':lambda r:logs.append(copy.deepcopy(r)),
 'actor':actor,'args':types.SimpleNamespace(injected_delay_seconds=delay),'limits':{'max_request_seconds':60},'last_start_progress':[start,0,0,80],'commands':commands}
 exec(compile(module,str(source),'exec'),env);fn,count=env['make_consumer']();response={**copy.deepcopy(raw),**changes};before=copy.deepcopy(response);error=None
 try:fn(response,final=final)
 except Exception as exc:error={'type':type(exc).__name__,'message':str(exc)}
 expectwait=delay if delay and not final and start<600 and response['kind']=='response' and response['status'] in ('completed','policy_timeout') else 0
 assert clock.sleeps==([expectwait] if expectwait else [])
 assert response==before and logs[0]['response']==before
 if name=='runtime_failure':assert error and error['type']=='RuntimeError' and not commands.values
 else:
  assert error is None
  wantdispatch=name in ('zero','delay2','delay5','delay10','delay20','exact60')
  assert bool(commands.values)==wantdispatch and count()==int(wantdispatch)
  assert logs[-1]['dispatch_age_wall_seconds']==response['parent_end_to_end_wall_seconds']+expectwait
  assert logs[-1]['worker_progress_at_enqueue'][0]==min(start+expectwait,600)
 results.append({'case':name,'sleep_calls':clock.sleeps,'raw_response_unchanged':response==before,'commands':commands.values,'error':error,'logs':logs})
m.time=original_time
# Exercise real main argparse/guards. Accepted options stop at the first registry read, before any output/worker/model path.
class InputsReached(Exception):pass
old_read=Path.read_text;oldargv=sys.argv;cli=[]
def read(path,*a,**k):
 if str(path)=='/tmp/delay-review-no-run/planning/local-model-registry.v1.json':raise InputsReached()
 return old_read(path,*a,**k)
Path.read_text=read
try:
 for arch in ('B3','B4'):
  for model in ('M1','M2','M3'):
   for ablation in (None,'periodic','no-cross-aircraft-memory','no-progressive-confirmation-guidance'):
    for delay in (0,2,5,10,20):
     sys.argv=['runner','--run','/tmp/delay-review-no-run','--reservation','never-reserve','--model',model,'--architecture',arch,'--case','/tmp/not-read.json','--sampler-seed','1','--out','/tmp/delay-review-no-output','--injected-delay-seconds',str(delay)]+(['--ablation',ablation] if ablation else [])
     expected=not (arch=='B4' and model=='M3') and (ablation is None or arch=='B4') and (not delay or (arch=='B4' and model=='M1' and ablation is None))
     status=None
     try:
      with contextlib.redirect_stderr(io.StringIO()):m.main()
     except InputsReached:status='admitted_to_first_read'
     except SystemExit as e:status='rejected_'+str(e.code)
     assert (status=='admitted_to_first_read')==expected
     cli.append({'architecture':arch,'model':model,'ablation':ablation,'delay':delay,'result':status})
finally:Path.read_text=old_read;sys.argv=oldargv
old=ROOT/'paper/research/20261104-inspection/review/eight-condition-development-driver-repair-v1-evidence/run_inspection_eight_condition_pilot.py';new=ROOT/'eval/run_inspection_eight_condition_pilot.py'
expected=old.read_text().replace('                "service_sha256": frozen[str(ROOT/"eval/inspection_model_service.py")], "evaluator": str(evaluator),','                "injected_delay_seconds": 0,\n                "service_sha256": frozen[str(ROOT/"eval/inspection_model_service.py")], "evaluator": str(evaluator),')
assert new.read_text()==expected
out={'runner_sha256':H(source),'driver_sha256':H(new),'original_driver_sha256':H(old),'exact_driver_only_change':'one field injected_delay_seconds=0','consume_function_ast_sha256':hashlib.sha256(ast.dump(consume).encode()).hexdigest(),'cli_cases':len(cli),'cli_admitted':sum(x['result']=='admitted_to_first_read' for x in cli),'cli':cli,'consume_cases':results}
(T/'independent-results.json').write_text(json.dumps(out,indent=2,sort_keys=True));print(json.dumps({'cli_cases':len(cli),'consume_cases':len(results),'driver_exact_one_field':True,'all_assertions_passed':True}))
