from pathlib import Path
import json,sys,hashlib,shutil
from types import SimpleNamespace
import pytest
PROJECT=Path('/tmp/formal-primary-consumer-repair-review-v2/project');sys.path.insert(0,str(PROJECT))
import eval.run_inspection_llm_episode as supervisor
import eval.inspection_formal_dispatch as dispatch
import eval.inspection_measurement_loader as measurement
from eval.inspection_episode_receipts import canonical_digest,digest,_bindings
OBS=[]
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

def old_classic_relabel_counter(tmp_path,monkeypatch):
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
def hashes(path):return {str(p.relative_to(path)):digest(p) for p in path.rglob('*') if p.is_file()}
def setup(tmp_path,monkeypatch,trailing=False,lock=True):
 result=cache(tmp_path,monkeypatch,trailing)
 monkeypatch.setattr(dispatch,'admit_episode_plan',lambda *a:SimpleNamespace(binding=lambda:{'phase':'main'}))
 if lock:(result[0]/'.lock').touch()
 return result

def test_PC1_trailing_attempt_now_rejected(tmp_path,monkeypatch):
 episode,plan,_,_=setup(tmp_path,monkeypatch,True);before=hashes(episode)
 with pytest.raises(supervisor.UnresolvedEpisode,match='last and unique'):measurement._measure_completed_llm(episode,plan,'main')
 assert hashes(episode)==before;OBS.append({'PC1':'old trailing attempt rejected; no Y/world mix returned'})

def test_PC2_missing_lock_not_created(tmp_path,monkeypatch):
 episode,plan,_,_=setup(tmp_path,monkeypatch,lock=False);before=hashes(episode)
 with pytest.raises(supervisor.UnresolvedEpisode,match='existing plan and lock'):measurement._measure_completed_llm(episode,plan,'main')
 assert hashes(episode)==before and not(episode/'.lock').exists();OBS.append({'PC2':'no new lock or file bytes'})

def test_valid_readonly_final_cache_Y_world_same_attempt(tmp_path,monkeypatch):
 episode,plan,validated,settled=setup(tmp_path,monkeypatch);before=hashes(episode)
 y,e=measurement._measure_completed_llm(episode,plan,'main');assert y==e['score']['Q']==.4
 assert e['episode_receipt'].endswith('attempt-00/terminal.json') and e['world_sha256']==digest(episode/'attempt-00/flight/world.json')
 assert hashes(episode)==before;OBS.append({'positive':'valid final00 cache readonly; Y/world same accepted00'})

def test_valid_second_cache_after_known_failure_same_Y_world(tmp_path,monkeypatch):
 episode,plan,validated,settled=setup(tmp_path,monkeypatch);a=episode/'attempt-00';b=episode/'attempt-01';shutil.copytree(a,b)
 intent=json.loads((b/'intent.json').read_text());intent.update(reservation='literal-a1',attempt_ordinal=1);write(b/'intent.json',intent)
 write(b/'flight/world.json',{'score':{'Q':.9,'C':.9,'S':.9}})
 term=json.loads((b/'terminal.json').read_text());term['Y']=.9
 paths=[Path(plan['formal_dispatch']['run'])/'qualification/local-model-episodes/literal-a1'/n for n in ('complete-attempt.zip','usage-receipt.json')]+[Path(plan['formal_dispatch']['run'])/'resource-requests/literal-a1.settle.json']
 for path in paths:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'Fake known settlement')
 term['settlement_file_hashes']={str(p.resolve()):digest(p) for p in paths};term['payload_hashes']=supervisor._bindings(b,tmp_path/'operation')[0];write(b/'terminal.json',term)
 first=json.loads((a/'terminal.json').read_text());physical={'failure_reasons':['clock_lag_exceeded']};first.update(status='known_operational_failure',Y=None,retry_allowed=True,physical_validation=physical);write(a/'terminal.json',first)
 meter={'model_meter':{'usage':{'calls':1,'tokens':2}}}
 monkeypatch.setattr(supervisor,'validate_llm_episode',lambda flight,*aa,**kk:((None if Path(flight).parent.name=='attempt-00' else .9),meter))
 monkeypatch.setattr(supervisor,'validate_flight',lambda *a:(None,physical));before=hashes(episode)
 y,e=measurement._measure_completed_llm(episode,plan,'main');assert y==e['score']['Q']==.9 and e['episode_receipt'].endswith('attempt-01/terminal.json')
 assert hashes(episode)==before and settled==['literal-a0','literal-a1'];OBS.append({'positive':'known failed00 + accepted01, Y/world same01; no file changes'})

def test_PC3_stored_formal_relabel_now_rejected(tmp_path,monkeypatch):
 with pytest.raises(ValueError,match='stored formal'):old_classic_relabel_counter(tmp_path,monkeypatch)
 OBS.append({'PC3':'stored formal cannot become development'})
def teardown_module():Path('/tmp/formal-primary-consumer-repair-review-v2/independent-cache-probes.json').write_text(json.dumps(OBS,indent=2)+'\n')
