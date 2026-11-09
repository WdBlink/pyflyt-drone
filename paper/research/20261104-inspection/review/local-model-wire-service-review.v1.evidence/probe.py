"""Isolated FakeMLX fixtures only. No real model/MLX/GPU/physics operation."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import sys,json,hashlib,io,contextlib
import eval.inspection_model_service as service
from eval.inspection_llm_protocol import PUBLIC_FIELDS,serialize_public_frame
ROOT=Path('/tmp/local-model-wire-review');out={'real_model_generation':0,'real_gpu_operations':0}
def state():
 return {'frozen_plan':{'fixture':True},'receipts':[{'stage':'plan'}],'active_round':{},'task_stop':None,'usage_ledger':{'fixture':{'status':'reserved','reserved':{'calls':1,'tokens':50}}}}
p=ROOT/'admission';(p/'.paper').mkdir(parents=True);s=state();(p/'.paper/control.json').write_text(json.dumps(s));out['incomplete_plan_and_empty_round_admitted']=service.admitted_limits(p,'fixture',1,20)['status']=='reserved'
s['usage_ledger']['fixture']['reserved']={'calls':float('nan'),'tokens':float('nan')};(p/'.paper/control.json').write_text(json.dumps(s));out['nan_owner_caps_admitted']=service.admitted_limits(p,'fixture',100,100000)['status']=='reserved'
frame={n:[] for n in PUBLIC_FIELDS};frame['sim_time']=1.;frame['task_card']={'score':{'Y':1},'truth':{'target_xy':[1,2]},'grader_output':'hidden verdict'};out['nested_evaluator_fields_serialized']=json.loads(serialize_public_frame(frame))['task_card']
real=Path('paper/research/20261104-inspection');before=hashlib.sha256((real/'.paper/control.json').read_bytes()).hexdigest()
try:service.admitted_limits(real,'review-negative-no-reservation',1,1);out['actual_negative_refused']=False
except ValueError:out['actual_negative_refused']=True
out['actual_controller_unchanged']=before==hashlib.sha256((real/'.paper/control.json').read_bytes()).hexdigest()
# Inject modules before main can import any MLX implementation.
modules={name:sys.modules.get(name) for name in ['mlx','mlx.core','mlx_lm','mlx_lm.sample_utils']}
mx=ModuleType('mlx.core');mx._review_fake_backend=True;mx.random=SimpleNamespace(seed=lambda n:None)
mlx=ModuleType('mlx');mlx.__path__=[];mlx.core=mx
lm=ModuleType('mlx_lm');lm.__path__=[];lm._review_fake_backend=True
su=ModuleType('mlx_lm.sample_utils');su.make_sampler=lambda **k:k
sys.modules.update({'mlx':mlx,'mlx.core':mx,'mlx_lm':lm,'mlx_lm.sample_utils':su})
class Tokenizer:
 def apply_chat_template(self,messages,**kwargs):assert kwargs['enable_thinking'] is False;return 'prompt'
 def encode(self,prompt):return [10,11]
orig_write,orig_time,orig_stdin,orig_argv=service.write_once,service.time.monotonic,sys.stdin,sys.argv
try:
 for mode in ['normal','journal_io_failure','late_empty','empty_model_file_manifest']:
  run=ROOT/mode;(run/'.paper').mkdir(parents=True);control=state();(run/'.paper/control.json').write_text(json.dumps(control));model=run/'fake-model';model.mkdir();blob=model/'fixture.bin';blob.write_bytes(b'abcd');files=[{'path':'fixture.bin','bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}]
  receipt=run/'model.json';receipt.write_text(json.dumps({'destination':str(model),'repo_id':'fake-only','revision':'fixture','files':[] if mode=='empty_model_file_manifest' else files}))
  loaded=[];clock={'now':0.}
  lm.load=lambda *a,**k:(loaded.append({'path':a[0],'kwargs':k}) or (object(),Tokenizer()))
  def fake_stream(*a,**k):
   assert lm._review_fake_backend and mx._review_fake_backend
   if mode=='late_empty':clock['now']=10.;return
   yield SimpleNamespace(text='X',token=123,prompt_tokens=2,generation_tokens=1)
  lm.stream_generate=fake_stream;service.time.monotonic=lambda:clock['now']
  def writer(path,value):
   if mode=='journal_io_failure' and Path(path).name.startswith('token-'):raise OSError(28,'fixture ENOSPC before token write')
   orig_write(path,value)
  service.write_once=writer
  request=json.dumps({'messages':[{'role':'user','content':'fixture'}],'seed':1})+'\n';sys.stdin=io.StringIO(request);sys.argv=['fixture-service','--run',str(run),'--reservation','fixture','--model-receipt',str(receipt),'--max-calls','1','--max-total-tokens','20','--max-output-tokens','2','--max-request-seconds','1']
  output=io.StringIO()
  with contextlib.redirect_stdout(output):service.main()
  op=run/'resource-operations/fixture';resp=json.loads((op/'call-0001/response.json').read_text());completion=json.loads((op/'service-completion.json').read_text());out[mode]={'fake_load_count':len(loaded),'load_config':loaded[0]['kwargs'],'response':resp,'token_files':len(list((op/'call-0001').glob('token-*'))),'completion':completion,'controller_same':json.loads((run/'.paper/control.json').read_text())==control}
  if mode=='normal':
   # A duplicate admission cannot reuse the same reservation operation folder.
   sys.stdin=io.StringIO(request)
   try:
    with contextlib.redirect_stdout(io.StringIO()):service.main()
    out['same_reservation_second_start_rejected']=False
   except FileExistsError:out['same_reservation_second_start_rejected']=True
finally:
 service.write_once,service.time.monotonic,sys.stdin,sys.argv=orig_write,orig_time,orig_stdin,orig_argv
 for name,prior in modules.items():
  if prior is None:sys.modules.pop(name,None)
  else:sys.modules[name]=prior
assert out['normal']['response']['usage']=={'calls':1,'tokens':3} and out['same_reservation_second_start_rejected']
assert out['journal_io_failure']['token_files']==0 and out['journal_io_failure']['response']['emitted_tokens_journaled']==1
assert out['late_empty']['response']['status']=='completed' and out['late_empty']['response']['wall_seconds']==10.
(ROOT/'probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
