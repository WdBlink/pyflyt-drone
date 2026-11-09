"""Snapshot-v1 lightweight service/CLI guard counterexamples; no MLX/model/child worker."""
from pathlib import Path
import importlib.util,json,sys,copy
from types import SimpleNamespace
import pytest
PROJECT=Path('/tmp/formal-primary-consumer-review-v1/project');sys.path.insert(0,str(PROJECT));T=PROJECT/'paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests'
s=importlib.util.spec_from_file_location('independent_service_fixtures',T/'test_formal_model_service.py');fxt=importlib.util.module_from_spec(s);sys.modules[s.name]=fxt;s.loader.exec_module(fxt);m=fxt.module
OBS=[]
def test_service_accepts_bool_repeat_as_native_zero(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch)
 purpose['repeat_ordinal']=False;state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
 (tmp_path/'.paper/control.json').write_text(json.dumps(state))
 assert m.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)==profile
 OBS.append({'counterexample':'formal service accepts bool repeat_ordinal False as integer slot0','purpose_repeat_type':'bool','Fake_dispatch_only':True,'no_MLX_or_model_effect':True})

def test_service_uses_replaced_raw_state_instead_of_gate_snapshot(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch)
 old=copy.deepcopy(state);old['usage_ledger']={}
 (tmp_path/'.paper/control.json').write_text(json.dumps(old));(tmp_path/'.paper/checkpoint.json').write_text(json.dumps({'state':old,'sha256':m._digest(old)}))
 slot=SimpleNamespace(condition='B4-M1',case_id='main-literal',repeat_ordinal=0,sampler_seed=70)
 binding={'phase':'main','sampler_seed':70,'packet':packet}
 admitted=SimpleNamespace(binding=lambda:binding,episode=SimpleNamespace(slot=slot,execution=SimpleNamespace(owner_snapshot=old)))
 def gate(*args):
  (tmp_path/'.paper/control.json').write_text(json.dumps(state))
  return admitted
 monkeypatch.setattr(fxt.helper.module,'admit_dispatch',gate)
 assert m.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)==profile
 assert json.loads((tmp_path/'.paper/checkpoint.json').read_text())['state']['usage_ledger']=={}
 OBS.append({'counterexample':'service raw-state second read accepts reservation absent from snapshot/checkpoint','validated_gate_snapshot_has_reservation':False,'accepted_raw_state_has_reservation':True,'Fake_gate_interleaving_not_actual_native_transition':True,'no_generation_executed':True})

def test_other_model_receipt_rejected_control(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch)
 registry_path=tmp_path/'registry.json';registry=json.loads(registry_path.read_text());other=tmp_path/'other-model.json';other.write_text('{"other":true}')
 registry['models']['M2']={'download_receipt_sha256':m.hashlib.sha256(other.read_bytes()).hexdigest(),'file_count':2};registry_path.write_text(json.dumps(registry))
 plan=state['receipts'][0]['packet'];plan['local_model_registry_sha256']=m.hashlib.sha256(registry_path.read_bytes()).hexdigest();state['frozen_plan']=m._digest(plan)
 purpose['model_registry_sha256']=plan['local_model_registry_sha256'];purpose['model_receipt_sha256']=registry['models']['M2']['download_receipt_sha256'];state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
 (tmp_path/'.paper/control.json').write_text(json.dumps(state))
 with pytest.raises(ValueError,match='primary slot'):m.admitted_limits(tmp_path,'literal',2,1000,other,formal_dispatch=packet)
 OBS.append({'control':'purpose/model M2 receipt for M1 slot rejected','no_model_effect':True})

def test_classical_main_case_without_dispatch_flags_reaches_output_creation(tmp_path,monkeypatch):
 import eval.run_inspection_reference as reference
 out=tmp_path/'flight';case=tmp_path/'run/datasets/main.v1/cases/main-ID-L1-00.json';case.parent.mkdir(parents=True);case.write_text(json.dumps({'schema':'inspection-population-case-v2','split':'main','case_id':'main-ID-L1-00'}))
 argv=['reference','--run',str(tmp_path/'run'),'--case',str(case),'--layout','generated','--realtime','--duration','600','--out',str(out)]
 monkeypatch.setattr(sys,'argv',argv);real=Path.mkdir
 def observe(path,*a,**kw):
  value=real(path,*a,**kw)
  if path==out:raise RuntimeError('independent stop immediately after output effect; no process')
  return value
 monkeypatch.setattr(Path,'mkdir',observe)
 with pytest.raises(RuntimeError,match='after output effect'):reference.main()
 assert out.is_dir()
 OBS.append({'counterexample':'formal-role canonical case without formal flags takes development branch before output','actual_argparse_main':True,'output_created':True,'stopped_before_metadata_or_process':True,'case_only_header_literal_not_real_formal_instance':True,'not_a_proven_physics_bypass':True})

def teardown_module():Path('/tmp/formal-primary-consumer-review-v1/independent-service-cli-probes.json').write_text(json.dumps(OBS,indent=2)+'\n')
