"""v2 independent FakeMLX closure. Never import real MLX or load weights."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
import ast,hashlib,json,sys,io,contextlib,copy
import eval.inspection_model_service as s
from eval.inspection_llm_protocol import PUBLIC_FIELDS,serialize_public_frame
R=Path('paper/research/20261104-inspection');T=Path('/tmp/local-model-service-repair-review-v2b');out={'real_generation':0,'real_GPU':0,'real_physics':0}
# Extract only the actual owner digest pure function; no owner module/state load.
owner_path=R/'runtime/autoresearch-state-reader-v1/lib/runtime.py';tree=ast.parse(owner_path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='digest');ns={'hashlib':hashlib,'json':json};exec(compile(ast.Module(body=[node],type_ignores=[]),str(owner_path),'exec'),ns);owner_digest=ns['digest']
value={'z':[1,'中文'],'a':False};literal=b'{"a":false,"z":[1,"\\u4e2d\\u6587"]}';oracle=hashlib.sha256(literal).hexdigest();assert owner_digest(value)==s._digest(value)==oracle;out['canonical']={'owner_file_sha':hashlib.sha256(owner_path.read_bytes()).hexdigest(),'owner_service_oracle_exact':True,'unicode_case_digest':oracle}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build(run,empty=False,mistral=False):
 (run/'.paper').mkdir(parents=True);model=run/'fake-model';model.mkdir();(model/'blob.bin').write_bytes(b'abcd');files=[] if empty else [{'path':'blob.bin','bytes':4,'sha256':hashlib.sha256(b'abcd').hexdigest()}];receipt=run/'download.json';receipt.write_text(json.dumps({'destination':str(model),'repo_id':'fake-only','revision':'fixture','files':files}));profile={'download_receipt_sha256':sha(receipt),'file_count':len(files),'tokenizer_config':{'local_files_only':True,**({'fix_mistral_regex':True} if mistral else {})},'generation_profile':{'max_input_tokens':8,'max_output_tokens':2,'max_request_seconds':1.,'max_calls':2,'max_total_tokens':20,'enable_thinking':False,'temperature':.2,'top_p':.95}};registry=run/'registry.json';registry.write_text(json.dumps({'models':{'fixture':profile}}));plan={'stage':'plan','local_model_registry':registry.name,'local_model_registry_sha256':sha(registry)};purpose={'kind':'inspection_local_generation','phase':'development','model_registry_sha256':sha(registry),'model_receipt_sha256':sha(receipt)};state={'frozen_plan':owner_digest(plan),'receipts':[{'stage':'plan','packet':plan}],'active_round':{'id':'round-fixture','stage':'plan'},'task_stop':None,'usage_ledger':{'fixture':{'status':'reserved','reserved':{'calls':2,'tokens':20},'purpose':json.dumps(purpose)}}};(run/'.paper/control.json').write_text(json.dumps(state));return receipt,profile,state,model
# Admission faults retain a valid independent owner digest except the target fault.
admission={}
for fault in ['valid','fake_plan','wrong_digest','empty_round','empty_stage','nan_calls','nan_tokens','bool_calls','float_tokens','wrong_purpose','wrong_phase','wrong_model']:
 run=T/('admit-'+fault);receipt,profile,state,model=build(run)
 if fault=='fake_plan':state['frozen_plan']={'fixture':True}
 if fault=='wrong_digest':state['frozen_plan']='0'*64
 if fault=='empty_round':state['active_round']={}
 if fault=='empty_stage':state['active_round']['stage']=''
 if fault=='nan_calls':state['usage_ledger']['fixture']['reserved']['calls']=float('nan')
 if fault=='nan_tokens':state['usage_ledger']['fixture']['reserved']['tokens']=float('nan')
 if fault=='bool_calls':state['usage_ledger']['fixture']['reserved']['calls']=True
 if fault=='float_tokens':state['usage_ledger']['fixture']['reserved']['tokens']=20.
 if fault in ['wrong_purpose','wrong_phase']:
  purpose=json.loads(state['usage_ledger']['fixture']['purpose']);purpose['kind' if fault=='wrong_purpose' else 'phase']='other';state['usage_ledger']['fixture']['purpose']=json.dumps(purpose)
 if fault=='wrong_model':receipt.write_text(receipt.read_text()+' ')
 (run/'.paper/control.json').write_text(json.dumps(state))
 try:p=s.admitted_limits(run,'fixture',1,10,receipt);admission[fault]={'accepted':True}
 except Exception as e:admission[fault]={'accepted':False,'error':type(e).__name__+': '+str(e)}
assert admission['valid']['accepted'] and all(not x['accepted'] for k,x in admission.items() if k!='valid');out['admission']=admission
frame={f:[] for f in PUBLIC_FIELDS};frame['sim_time']=1.;frame['task_card']={};assert serialize_public_frame(frame)
out['private_fields']={}
for field in ['score','truth','ground_truth','grader_output']:
 frame['observations']=[{'candidate_id':'c',field:{'hidden':1}}]
 try:serialize_public_frame(frame);out['private_fields'][field]=False
 except ValueError:out['private_fields'][field]=True
assert all(out['private_fields'].values())
# Fake modules are injected before the service's only MLX import site.
prior={n:sys.modules.get(n) for n in ['mlx','mlx.core','mlx_lm','mlx_lm.sample_utils']};mx=ModuleType('mlx.core');mx._fake=True;mx.random=SimpleNamespace(seed=lambda n:None);mlx=ModuleType('mlx');mlx.__path__=[];mlx.core=mx;lm=ModuleType('mlx_lm');lm.__path__=[];lm._fake=True;su=ModuleType('mlx_lm.sample_utils');su.make_sampler=lambda **kw:kw;sys.modules.update({'mlx':mlx,'mlx.core':mx,'mlx_lm':lm,'mlx_lm.sample_utils':su})
class Tokenizer:
 def apply_chat_template(self,messages,**kw):assert kw['enable_thinking'] is False;return 'prompt'
 def encode(self,prompt):return [10,11]
oldj,oldv,oldt,oldin,oldargv=s.journal_token,s.verify_model_files,s.time.monotonic,sys.stdin,sys.argv
try:
 for mode in ['normal','enospc','late_empty','empty_manifest','unlisted_blob','same_size_mutation','over_profile','mistral_without_flag','mistral_with_flag','owner_revoked_after_verify']:
  run=T/mode;receipt,profile,state,model=build(run,empty=mode=='empty_manifest',mistral=mode.startswith('mistral_'));control_before=(run/'.paper/control.json').read_bytes();clock={'now':0.};loads=[]
  if mode=='unlisted_blob':(model/'extra.bin').write_bytes(b'x')
  if mode=='same_size_mutation':(model/'blob.bin').write_bytes(b'abce')
  if mode=='normal':(model/'.cache').mkdir();(model/'.cache/ignored').write_bytes(b'x');(model/'.DS_Store').write_bytes(b'x')
  lm.load=lambda *a,**k:(loads.append(k) or (object(),Tokenizer()))
  def fake_stream(*a,**k):
   assert lm._fake and mx._fake
   if mode=='late_empty':clock['now']=10.;return
   yield SimpleNamespace(text='X',token=123,prompt_tokens=2,generation_tokens=1)
  lm.stream_generate=fake_stream;s.time.monotonic=lambda:clock['now']
  def journal(stream,value):
   if mode=='enospc':raise OSError(28,'fixture ENOSPC before JSONL write')
   return oldj(stream,value)
  def verify(path,files):
   oldv(path,files)
   if mode=='owner_revoked_after_verify':
    revoked=copy.deepcopy(state);revoked['task_stop']={'reason':'fixture owner revocation'};(run/'.paper/control.json').write_text(json.dumps(revoked))
  s.journal_token=journal;s.verify_model_files=verify
  request=json.dumps({'messages':[{'role':'user','content':'fixture'}],'seed':1})+'\n';sys.stdin=io.StringIO('' if mode=='owner_revoked_after_verify' else request*2);sys.argv=['fixture','--run',str(run),'--reservation','fixture','--model-receipt',str(receipt),'--max-calls','1','--max-total-tokens','10','--max-input-tokens','8','--max-output-tokens','3' if mode=='over_profile' else '2','--max-request-seconds','1']
  if mode=='mistral_with_flag':sys.argv.append('--fix-mistral-regex')
  stdout=io.StringIO();error=None
  try:
   with contextlib.redirect_stdout(stdout):s.main()
  except Exception as e:error=type(e).__name__+': '+str(e)
  op=run/'resource-operations/fixture';responses=list(op.glob('call-*/response.json'));records=[json.loads(p.read_text()) for p in responses];token_lines=[line for p in op.glob('call-*/tokens.jsonl') for line in p.read_text().splitlines()];completion=json.loads((op/'service-completion.json').read_text()) if (op/'service-completion.json').exists() else None
  out[mode]={'error':error,'fake_load_count':len(loads),'load_config':loads,'responses':records,'journal_lines':len(token_lines),'completion':completion,'controller_unchanged':(run/'.paper/control.json').read_bytes()==control_before,'stdout_kinds':[json.loads(z)['kind'] for z in stdout.getvalue().splitlines()]}
  if mode=='normal':
   sys.stdin=io.StringIO(request)
   try:
    with contextlib.redirect_stdout(io.StringIO()):s.main()
    out['duplicate_reservation_rejected']=False
   except FileExistsError:out['duplicate_reservation_rejected']=True
finally:
 s.journal_token,s.verify_model_files,s.time.monotonic,sys.stdin,sys.argv=oldj,oldv,oldt,oldin,oldargv
 for n,v in prior.items():
  if v is None:sys.modules.pop(n,None)
  else:sys.modules[n]=v
assert out['normal']['responses'][0]['usage']=={'calls':1,'tokens':3} and out['normal']['journal_lines']==1 and 'not_started' in out['normal']['stdout_kinds'] and out['duplicate_reservation_rejected']
a=out['enospc']['responses'][0];assert a['generated_tokens_observed']==1 and a['emitted_tokens_journaled']==0 and out['enospc']['journal_lines']==0 and a['usage']['tokens']==3
assert out['late_empty']['responses'][0]['status']=='policy_timeout' and out['late_empty']['responses'][0]['deadline_exceeded']
assert all(out[m]['fake_load_count']==0 and out[m]['error'] for m in ['empty_manifest','unlisted_blob','same_size_mutation','over_profile','mistral_without_flag'])
assert out['mistral_with_flag']['fake_load_count']==1 and out['mistral_with_flag']['load_config'][0]['tokenizer_config']['fix_mistral_regex'] is True
(T/'probe.json').write_text(json.dumps(out,indent=2));print('canonical',out['canonical']);print('admission',admission);print('private_fields',out['private_fields']);
for m in ['normal','enospc','late_empty','empty_manifest','unlisted_blob','same_size_mutation','over_profile','mistral_without_flag','mistral_with_flag','owner_revoked_after_verify']:print(m,out[m])

assert out['owner_revoked_after_verify']['fake_load_count']==0 and out['owner_revoked_after_verify']['error'] is not None
