"""Integration-only Fake mission/native/collector; synthetic receipt subprocess only."""
from pathlib import Path
from types import SimpleNamespace
import json,queue,subprocess
import pytest
import envs.inspection.mission as missions
import eval.run_inspection_reference as runner
import eval.inspection_worker_runtime as helper
import eval.inspection_episode_receipts as receipts
from tests.test_inspection_worker_runtime import Native,Collector
from tests.test_inspection_episode_receipts import inputs
OBS=[]
@pytest.mark.parametrize('mode',['normal','start_failure','restore_failure','accelerated'])
def test_real_helper_fake_native_is_closed_and_audited(tmp_path,monkeypatch,mode):
 created=[];windows=[];native=Native();collector=Collector()
 if mode=='start_failure':native.fail=2
 if mode=='restore_failure':native.fail=1
 class Mission:
  def __init__(self,*a,**k):
   self.t=0.;self.warmup=not created;created.append(self);self.executor=SimpleNamespace(infrastructure_errors=[]);self.flight=SimpleNamespace(world=SimpleNamespace(physics_steps=1,physics_hz=240,step_period=1/120))
  def step(self):self.t+=1.
  def close(self):
   if not self.warmup and mode!='accelerated':assert windows[0].closed
  def public_state(self,**k):return {}
  def audit(self):return {'role':'Fake mission; no flight'}
  def business_audit(self):return {'partial':'literal'}
 def factory():
  assert len(created)==2
  if mode=='accelerated':raise AssertionError('accelerated diagnostic touched native helper')
  w=helper.PhysicsRuntimeWindow(native=native,collector=collector);windows.append(w);return w
 monkeypatch.setattr(missions,'InspectionMission',Mission);monkeypatch.setattr(runner,'PhysicsRuntimeWindow',factory);monkeypatch.setattr(runner.time,'sleep',lambda n:None)
 updates=queue.Queue();runner.worker(0,True,mode!='accelerated',1,queue.Queue(),updates,tmp_path)
 world=json.loads((tmp_path/'world.json').read_text())
 if mode=='accelerated':assert not windows and 'physics_runtime'not in world and world['clock_mode']=='accelerated_diagnostic'
 else:
  assert windows[0].closed and world['physics_runtime']==windows[0].audit and collector.enabled
  assert receipts._runtime_window_valid(world['physics_runtime'])==(mode=='normal')
 assert world['status']==('completed' if mode in ('normal','accelerated') else 'operational_failure')
 OBS.append({'mode':mode,'status':world['status'],'audit_present':'physics_runtime'in world,'audit_errors':world.get('physics_runtime',{}).get('errors'),'real_native_or_system_gc_changed':False})

def test_restore_failure_receipt_is_nonretryable_and_legacy_scope_unchanged(tmp_path):
 legacy=tmp_path/'legacy';legacy.mkdir();plan,command,counter=inputs(legacy);out=legacy/'flight';assert subprocess.run(command('ok')(out),check=False).returncode==0
 assert receipts.validate_flight(out,plan,0)[0]==0
 current=tmp_path/'new';current.mkdir();plan,command,counter=inputs(current);plan['source_hashes']['/fixture/eval/inspection_worker_runtime.py']='c'*64
 (current/'fixture-plan.json').write_text(json.dumps(plan));script=current/'fixture.py'
 script.write_text(script.read_text()+"\nworld=json.loads((out/'world.json').read_text());world['status']='operational_failure';world['physics_runtime']={'errors':['literal restoration failure']};(out/'world.json').write_text(json.dumps(world));run=json.loads((out/'run.json').read_text());run['world_sha256']=hashlib.sha256((out/'world.json').read_bytes()).hexdigest();(out/'run.json').write_text(json.dumps(run))\n")
 episode=current/'episode'
 with pytest.raises(receipts.UnresolvedEpisode):receipts.execute_local_episode(episode,plan,command('ok'))
 terminal=json.loads((episode/'attempt-00/terminal.json').read_text());assert terminal['retry_allowed'] is False and 'runtime_window_invalid'in terminal['evidence']['failure_reasons']
 with pytest.raises(receipts.UnresolvedEpisode):receipts.execute_local_episode(episode,plan,command('ok'))
 assert counter.read_text()=='1' and not(episode/'attempt-01').exists()
 OBS.append({'receipt':'runtime restoration failure rejected, terminal nonretryable and no second child','legacy_no_helper_no_audit_still_valid_under_old_scope':True,'synthetic_child_processes':2,'physical_flights':0})
def teardown_module():Path('/tmp/runtime-production-integration-review-v1/independent-probes.json').write_text(json.dumps(OBS,indent=2)+'\n')
