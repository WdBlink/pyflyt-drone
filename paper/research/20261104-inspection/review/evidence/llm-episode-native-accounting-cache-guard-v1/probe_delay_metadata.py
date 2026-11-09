"""Synthetic metadata/causal probes only; no real wait, model, physics or owner."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source-final'))
spec=importlib.util.spec_from_file_location('delay_metadata_fixture',HERE/'source-final/tests/test_inspection_llm_episode_receipts.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);module=f.module
results=[]
def save(label,**v):
 row={'probe':label,'role':'synthetic metadata, not actual model/physics/owner evidence',**v};results.append(row);print(json.dumps(row),flush=True)
def setup(name,delay,mode='meta-arm'):
 d=HERE/'delay-probes'/name;d.mkdir(parents=True,exist_ok=False);mp=pytest.MonkeyPatch()
 p,ep,counter,effects,execute=f.fixture(d,mp);p.update(condition='B4-M1',injected_delay_seconds=delay)
 s=d/'synthetic-child.py';text=s.read_text();assert "('injected2','injected20','delay-underwait')" in text
 text=text.replace("('injected2','injected20','delay-underwait')","('injected2','injected20','delay-underwait','meta-arm')")
 text=text.replace("59. if mode=='injected20' else 1.1","59. if mode in ('injected20','meta-arm') else 1.1")
 s.write_text(text);p['source_hashes'][str(s)]=module.digest(s)
 value,receipt=execute(mode);assert value==0 and counter.read_text()=='1' and len(effects)==2
 return d,mp,p,ep,counter,effects,receipt

base=None
for delay in (0,2,5,10,20):
 d,mp,p,ep,counter,effects,receipt=setup('arm-'+str(delay),delay)
 try:
  rows=[json.loads(x) for x in (ep/'attempt-00/flight/actor.jsonl').read_text().splitlines()]
  response,decoded=rows[1],rows[2]
  save('legal-arm-'+str(delay),accepted=True,Y=0,condition=p['condition'],run_delay=json.loads((ep/'attempt-00/flight/run.json').read_text())['injected_delay_seconds'],
       service_wall=response['response']['wall_seconds'],parent_compute_wall=response['response']['parent_end_to_end_wall_seconds'],
       injected_wall=response['injected_delay_wall_seconds'],reply_sim=response['worker_at_model_reply'][0],after_sim=response['worker_after_response'][0],
       dispatch_age_sim=decoded['dispatch_age_sim_seconds'],dispatch_age_wall=decoded['dispatch_age_wall_seconds'],dispatches=int(counter.read_text()))
  if delay==20:base=(d,copy.deepcopy(p),ep)
 finally:mp.undo()

d,mp,p,ep,counter,effects,receipt=setup('not-started-no-injection',20,'not-started')
try:
 rows=[json.loads(x) for x in (ep/'attempt-00/flight/actor.jsonl').read_text().splitlines()]
 save('not-started-no-injection',accepted=True,injected_wall=rows[1]['injected_delay_wall_seconds'],model_usage=json.loads((d/'operations/fixture-M1-a0/service-completion.json').read_text())['usage'])
finally:mp.undo()

d,plan,ep=base;flight=ep/'attempt-00/flight';op=d/'operations/fixture-M1-a0';actor=flight/'actor.jsonl';runfile=flight/'run.json';requestfile=op/'call-0001/request.json'
original_actor_bytes,original_run_bytes,original_request_bytes=actor.read_bytes(),runfile.read_bytes(),requestfile.read_bytes()
original_rows=[json.loads(x) for x in actor.read_text().splitlines()];original_run=json.loads(runfile.read_text());original_request=json.loads(requestfile.read_text())

def check(label,change,expected_reject=True):
 p=copy.deepcopy(plan);r=copy.deepcopy(original_run);rows=copy.deepcopy(original_rows);request=copy.deepcopy(original_request)
 change(p,r,rows,request)
 actor.write_text(''.join(json.dumps(x)+'\n' for x in rows));runfile.write_text(json.dumps(r));requestfile.write_text(json.dumps(request))
 artifact=HERE/'counterfactuals'/label;artifact.mkdir(parents=True,exist_ok=False)
 for name,value in (('plan.json',p),('run.json',r),('request.json',request)): (artifact/name).write_text(json.dumps(value,indent=2)+'\n')
 (artifact/'actor.jsonl').write_text(actor.read_text())
 try:
  value,_=module.validate_llm_episode(flight,op,p,'fixture-M1-a0',0)
 except (ValueError,OSError,KeyError,TypeError) as error:
  save(label,accepted=False,error=str(error),expected_reject=expected_reject)
  if not expected_reject:raise
 else:
  save(label,accepted=True,Y=value,expected_reject=expected_reject,source_sim=rows[0]['source_sim_time'],before_sim=rows[1]['worker_before_request'][0],
       after_sim=rows[1]['worker_after_response'][0],age_sim=rows[2]['dispatch_age_sim_seconds'],age_wall=rows[2]['dispatch_age_wall_seconds'])
 finally:
  actor.write_bytes(original_actor_bytes);runfile.write_bytes(original_run_bytes);requestfile.write_bytes(original_request_bytes)

check('plan-default0-run20',lambda p,r,a,q:p.pop('injected_delay_seconds'))
check('plan20-run2',lambda p,r,a,q:r.__setitem__('injected_delay_seconds',2))
for condition in ('B3-M1','B4-M2'):
 check('illegal-delay-'+condition,lambda p,r,a,q,c=condition:(p.__setitem__('condition',c),r.__setitem__('policy',c)))
check('illegal-mixed-ablation',lambda p,r,a,q:p.__setitem__('ablation','periodic'))
for label,value in (('bool',True),('float',20.),('outside',3)):
 check('illegal-delay-'+label,lambda p,r,a,q,v=value:(p.__setitem__('injected_delay_seconds',v),r.__setitem__('injected_delay_seconds',v)))
check('response-requested-wrong',lambda p,r,a,q:a[1].__setitem__('injected_delay_requested_seconds',2))
check('injection-underwait',lambda p,r,a,q:a[1].__setitem__('injected_delay_wall_seconds',19.5))
check('missing-physical-advance',lambda p,r,a,q:(a[1].__setitem__('worker_after_response',a[1]['worker_at_model_reply']),a[2].__setitem__('worker_progress_at_enqueue',a[1]['worker_at_model_reply']),a[2].__setitem__('dispatch_age_sim_seconds',1)))
check('wallage-omits-injection',lambda p,r,a,q:a[2].__setitem__('dispatch_age_wall_seconds',59))
check('simage-omits-injection',lambda p,r,a,q:a[2].__setitem__('dispatch_age_sim_seconds',1))
check('parent-compute61-still-soft-timeout',lambda p,r,a,q:(a[1]['response'].__setitem__('parent_end_to_end_wall_seconds',61.),a[2].__setitem__('dispatch_age_wall_seconds',81.)))
check('parent-compute60-plus20-is-not-timeout',lambda p,r,a,q:(a[1]['response'].__setitem__('parent_end_to_end_wall_seconds',60.),a[2].__setitem__('dispatch_age_wall_seconds',80.)),expected_reject=False)

def after_episode(p,r,a,q):
 a[1].update(final_after_flight=True,worker_at_model_reply=[600.,0,0,80],worker_after_response=[600.,0,0,80],injected_delay_wall_seconds=0.)
 a[2].update(worker_progress_at_enqueue=[600.,0,0,80],dispatch_age_sim_seconds=600.,dispatch_age_wall_seconds=59.)
 a[2]['audit']['status']='ignored_after_episode_deadline'
check('final600-no-inject-no-enqueue',after_episode,expected_reject=False)

# Check the source-frame clock against the recorded real-worker before sample.
def future_prompt(p,r,a,q,stamp):
 a[0]['source_sim_time']=stamp
 content=json.loads(a[0]['messages'][1]['content']);content['public_state']['sim_time']=stamp
 a[0]['messages'][1]['content']=json.dumps(content)
 a[0]['messages_sha256']=hashlib.sha256(json.dumps(a[0]['messages'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 a[2]['audit']['request']={k:a[0][k] for k in ('request_ordinal','source_sim_time','messages','messages_sha256')}
 q['messages']=copy.deepcopy(a[0]['messages'])
 a[2]['dispatch_age_sim_seconds']=a[1]['worker_after_response'][0]-stamp
for stamp in (5.,580.):
 check('future-public-clock-'+str(int(stamp)),lambda p,r,a,q,t=stamp:future_prompt(p,r,a,q,t))

(HERE/'delay-metadata-results.json').write_text(json.dumps(results,indent=2)+'\n')
