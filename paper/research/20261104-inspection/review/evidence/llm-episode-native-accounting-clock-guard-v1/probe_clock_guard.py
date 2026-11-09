# Synthetic-only readonly checks of two clock guards; no orchestration/owner/physics.
import copy,hashlib,importlib.util,json,os,subprocess,sys
from pathlib import Path
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source'))
spec=importlib.util.spec_from_file_location('clock_fixture',HERE/'source/tests/test_inspection_llm_episode_receipts.py')
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f);module=f.module
results=[]
def record(label,**v):
 row={'probe':label,'role':'synthetic readonly metadata, no real model/physics/native owner',**v};results.append(row);print(json.dumps(row),flush=True)
bases={}
for delay,mode in ((0,'ok'),(20,'injected20')):
 d=HERE/'fixtures'/('arm'+str(delay));d.mkdir(parents=True,exist_ok=False);mp=pytest.MonkeyPatch()
 try:
  plan,episode,counter,effects,execute=f.fixture(d,mp);plan.update(condition='B4-M1',injected_delay_seconds=delay)
  planfile=d/'plan-input.json';planfile.write_text(json.dumps(plan));flight=d/'flight';op=d/'operation'
  child=subprocess.run([sys.executable,str(d/'synthetic-child.py'),str(flight),str(op),str(planfile),'fixture-M1-a0',mode,str(counter)],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
  assert child.returncode==0,child.stderr
  value,evidence=module.validate_llm_episode(flight,op,plan,'fixture-M1-a0',0)
  assert value==0 and effects==[] and counter.read_text()=='1'
  rows=[json.loads(x) for x in (flight/'actor.jsonl').read_text().splitlines()]
  record('normal-arm'+str(delay),accepted=True,Y=value,public_sim=rows[0]['source_sim_time'],before_sim=rows[1]['worker_before_request'][0],reply_sim=rows[1]['worker_at_model_reply'][0],after_sim=rows[1]['worker_after_response'][0],age_sim=rows[2]['dispatch_age_sim_seconds'],age_wall=rows[2]['dispatch_age_wall_seconds'],owner_effects=0)
  bases[delay]=(d,plan,flight,op)
 finally:mp.undo()
def reject(label,delay,change):
 d,plan,flight,op=bases[delay];actor=flight/'actor.jsonl';req=op/'call-0001/request.json'
 ab,rb=actor.read_bytes(),req.read_bytes();rows=[json.loads(x) for x in actor.read_text().splitlines()];request=json.loads(req.read_text())
 change(rows,request);actor.write_text(''.join(json.dumps(x)+'\n' for x in rows));req.write_text(json.dumps(request))
 artifact=HERE/'counterfactuals'/label;artifact.mkdir(parents=True,exist_ok=False)
 (artifact/'actor.jsonl').write_text(actor.read_text());(artifact/'request.json').write_text(req.read_text());(artifact/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 try:
  try:value,_=module.validate_llm_episode(flight,op,plan,'fixture-M1-a0',0)
  except (ValueError,OSError,TypeError,KeyError) as error:record(label,rejected=True,error=str(error),public_sim=rows[0]['source_sim_time'],before_sim=rows[1]['worker_before_request'][0],after_sim=rows[1]['worker_after_response'][0],age_sim=rows[2]['dispatch_age_sim_seconds'])
  else:raise AssertionError('invalid metadata accepted: '+label)
 finally:actor.write_bytes(ab);req.write_bytes(rb)
def future(rows,request,stamp):
 public=rows[0];public['source_sim_time']=stamp;c=json.loads(public['messages'][1]['content']);c['public_state']['sim_time']=stamp
 public['messages'][1]['content']=json.dumps(c);public['messages_sha256']=hashlib.sha256(json.dumps(public['messages'],sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
 rows[2]['audit']['request']={k:public[k] for k in ('request_ordinal','source_sim_time','messages','messages_sha256')}
 request['messages']=copy.deepcopy(public['messages']);rows[2]['dispatch_age_sim_seconds']=rows[1]['worker_after_response'][0]-stamp
reject('future-public5',20,lambda a,q:future(a,q,5.))
reject('future-public580-negative-age',20,lambda a,q:future(a,q,580.))
reject('bool-age-equals-native-one',0,lambda a,q:a[2].__setitem__('dispatch_age_sim_seconds',True))
(HERE/'results.json').write_text(json.dumps(results,indent=2)+'\n')
