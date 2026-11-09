"""Snapshot-v1 control-flow counterexamples. Literal cache, Fake validators; no flight/model."""
from pathlib import Path
import json,sys,hashlib
import pytest
PROJECT=Path('/tmp/formal-primary-consumer-review-v1/project');sys.path.insert(0,str(PROJECT))
import eval.run_inspection_llm_episode as supervisor
import eval.inspection_formal_dispatch as dispatch
import eval.inspection_measurement_loader as measurement
from eval.inspection_episode_receipts import canonical_digest,digest,_bindings
from types import SimpleNamespace
OBS=[]
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def cache(tmp_path,monkeypatch,trailing=False):
 episode=tmp_path/'episode';run=tmp_path/'run';op=tmp_path/'operation';op.mkdir();episode.mkdir()
 plan={'condition':'B4-M1','case_id':'main-literal','repeat_ordinal':0,'max_attempts':2,'max_clock_lag_seconds':.1,'formal_dispatch':{'run':str(run),'packet':{'path':'literal-packet.json','sha256':'a'*64}}}
 write(episode/'plan.json',plan);attempt=episode/'attempt-00';attempt.mkdir();reservation='literal-a0';ph=canonical_digest(plan)
 write(attempt/'intent.json',{'plan_sha256':ph,'reservation':reservation,'attempt_ordinal':0,'argv':['literal-no-process']})
 write(attempt/'started.json',{'plan_sha256':ph,'pid':12345});(attempt/'stdout.log').write_text('literal-no-process')
 write(attempt/'process-return.json',{'plan_sha256':ph,'returncode':0,'stdout_sha256':digest(attempt/'stdout.log')})
 write(attempt/'process-tree.json',{'status':'closed','root_pid':12345})
 for n in ('reserve-owner.json','settle-owner.json'):write(attempt/n,{'role':'Fake prior known acknowledgement'})
 write(attempt/'flight/world.json',{'score':{'Q':.4,'C':.4,'S':.4}})
 settled=[run/'qualification/local-model-episodes'/reservation/n for n in ('complete-attempt.zip','usage-receipt.json')]+[run/'resource-requests'/(reservation+'.settle.json')]
 for p in settled:p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'literal-placeholder-assumed-validated')
 meter={'usage':{'calls':1,'tokens':2}};evidence={'model_meter':meter}
 payload,model_payload=supervisor._bindings(attempt,op)
 write(attempt/'terminal.json',{'status':'completed','returncode':0,'plan_sha256':ph,'Y':.4,'evidence':evidence,'payload_hashes':payload,'model_payload_hashes':model_payload,'settlement_file_hashes':{str(p.resolve()):digest(p) for p in settled}})
 if trailing:
  write(episode/'attempt-01/terminal.json',{'status':'completed','role':'deliberately incomplete trailing attempt'})
  write(episode/'attempt-01/flight/world.json',{'score':{'Q':.9,'C':.9,'S':.9}})
 validated=[];settled_ids=[]
 monkeypatch.setattr(supervisor,'_formal',lambda plan:SimpleNamespace(binding=lambda:{'phase':'main'}))
 monkeypatch.setattr(supervisor,'operation_directory',lambda reservation:op)
 def valid(flight,*args,**kwargs):validated.append(str(flight));return .4,evidence
 monkeypatch.setattr(supervisor,'validate_llm_episode',valid)
 monkeypatch.setattr(dispatch,'admit_model_settlement',lambda plan,project,reservation,*a:(settled_ids.append(reservation) or None))
 monkeypatch.setattr(measurement,'_recount_score',lambda world:world['score'])
 return episode,plan,validated,settled_ids

def test_readonly_completed_llm_measurement_creates_missing_lock(tmp_path,monkeypatch):
 episode,plan,validated,settled=cache(tmp_path,monkeypatch)
 assert not(episode/'.lock').exists();value,evidence=measurement._measure_completed_llm(episode,plan,'main')
 assert value==.4 and (episode/'.lock').exists()
 OBS.append({'counterexample':'readonly measurement creates missing .lock','lock_created':True,'validator_scope':'first cached attempt and settlement assumed valid via Fake; actual supervisor/cache/hash control flow and IO run','no_reserve_settle_or_spawn_called':True})

def test_extra_attempt_after_completed_is_ignored_and_measurement_mixes_last_world(tmp_path,monkeypatch):
 episode,plan,validated,settled=cache(tmp_path,monkeypatch,True)
 value,evidence=measurement._measure_completed_llm(episode,plan,'main')
 assert value==.4 and evidence['score']['Q']==.9 and evidence['retained_attempts']==2
 assert validated==[str(episode/'attempt-00/flight')] and settled==['literal-a0']
 assert not(episode/'attempt-01/intent.json').exists()
 OBS.append({'counterexample':'trailing incomplete attempt after completed first cache ignored','returned_Y':value,'returned_score_Q':evidence['score']['Q'],'receipt_is_attempt00':evidence['episode_receipt'].endswith('attempt-00/terminal.json'),'world_is_attempt01':evidence['world_sha256']==digest(episode/'attempt-01/flight/world.json'),'retained_attempts':2,'validated_flights':validated,'native_settlement_checked_only':settled,'trailing_intent_missing':True,'Fake_first_flight_and_meter_only':True})

def test_classic_measurement_relabels_stored_formal_plan_development(tmp_path,monkeypatch):
 episode=tmp_path/'classic';attempt=episode/'attempt-00';attempt.mkdir(parents=True)
 private={'level':'L1','priority_at':None,'entities':[]};score={'C':.6,'S':.4,'Q':.5};source='literal frozen source'
 plan={'formal_dispatch':{'run':str(tmp_path/'run'),'packet':{'path':'literal.json','sha256':'a'*64}},'case_id':'main-literal','condition':'B0','repeat_ordinal':0,'case_file_sha256':'1'*64,'case_sha256':hashlib.sha256(json.dumps(private,sort_keys=True,allow_nan=False).encode()).hexdigest(),'suite_manifest_sha256':'2'*64,'source_hashes':{'literal.py':hashlib.sha256(source.encode()).hexdigest()},'dependency_hashes':{},'supervisor_sha256':'3'*64,'receipt_adapter_sha256':'4'*64,'thread_environment':{},'max_clock_lag_seconds':.1,'max_attempts':2}
 write(episode/'plan.json',plan);ph=canonical_digest(plan);write(attempt/'intent.json',{'plan_sha256':ph,'attempt_ordinal':0});write(attempt/'started.json',{'plan_sha256':ph});(attempt/'stdout.log').write_text('literal')
 write(attempt/'flight/run.json',{'sources':{'literal.py':source}});write(attempt/'flight/world.json',{'private_case':private,'score':score,'hidden_entities':[]})
 evidence={'role':'Fake valid physical/source receipt'}
 write(attempt/'terminal.json',{'status':'completed','plan_sha256':ph,'record_hashes':_bindings(attempt),'stdout_sha256':digest(attempt/'stdout.log'),'returncode':0,'Y':.5,'evidence':evidence})
 seen=[]
 monkeypatch.setattr(measurement,'validate_flight',lambda flight,actual,*a:(seen.append('formal_dispatch'in actual) or (.5,evidence)))
 monkeypatch.setattr(measurement,'_recount_score',lambda world:score)
 expected={k:v for k,v in plan.items() if k!='formal_dispatch'};expected['phase']='development'
 y,out=measurement.measure_completed_episode(episode,expected)
 assert y==.5 and out['phase']=='development' and out['case_id'].startswith('main-') and seen==[True]
 OBS.append({'counterexample':'stored formal classic plan accepted under development expected role','returned_phase':out['phase'],'case_id':out['case_id'],'stored_plan_has_formal_dispatch':True,'Fake_physical_validator_only':True,'actual_phase_and_plan_branch_run':True})

def teardown_module():Path('/tmp/formal-primary-consumer-review-v1/independent-cache-phase-probes.json').write_text(json.dumps(OBS,indent=2)+'\n')
