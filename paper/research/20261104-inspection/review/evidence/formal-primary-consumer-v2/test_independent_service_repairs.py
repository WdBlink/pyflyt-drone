from pathlib import Path
import importlib.util,json,sys,copy
from types import SimpleNamespace
import pytest
PROJECT=Path('/tmp/formal-primary-consumer-repair-review-v2/project');sys.path.insert(0,str(PROJECT));T=PROJECT/'paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests'
s=importlib.util.spec_from_file_location('repair_service_fixtures',T/'test_formal_model_service.py');fxt=importlib.util.module_from_spec(s);sys.modules[s.name]=fxt;s.loader.exec_module(fxt);m=fxt.module
OBS=[]
def old_model_negative_control(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch)
 registry_path=tmp_path/'registry.json';registry=json.loads(registry_path.read_text());other=tmp_path/'other-model.json';other.write_text('{"other":true}')
 registry['models']['M2']={'download_receipt_sha256':m.hashlib.sha256(other.read_bytes()).hexdigest(),'file_count':2};registry_path.write_text(json.dumps(registry))
 plan=state['receipts'][0]['packet'];plan['local_model_registry_sha256']=m.hashlib.sha256(registry_path.read_bytes()).hexdigest();state['frozen_plan']=m._digest(plan)
 purpose['model_registry_sha256']=plan['local_model_registry_sha256'];purpose['model_receipt_sha256']=registry['models']['M2']['download_receipt_sha256'];state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
 (tmp_path/'.paper/control.json').write_text(json.dumps(state))
 with pytest.raises(ValueError,match='primary slot'):m.admitted_limits(tmp_path,'literal',2,1000,other,formal_dispatch=packet)
 OBS.append({'control':'purpose/model M2 receipt for M1 slot rejected','no_model_effect':True})

def old_cli_counter(tmp_path,monkeypatch):
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
def test_PC4_selfconsistent_raw_state_cannot_override_snapshot(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch);admitted=fxt.helper.module.admit_dispatch(tmp_path,tmp_path,packet)
 old=copy.deepcopy(state);old['usage_ledger']={};admitted.episode.execution.owner_snapshot=old
 (tmp_path/'.paper/control.json').write_text(json.dumps(state));(tmp_path/'.paper/checkpoint.json').write_text(json.dumps({'state':state,'sha256':m._digest(state)}))
 with pytest.raises(ValueError,match='sufficient live owner reservation'):m.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)
 OBS.append({'PC4':'selfconsistent raw pair has reservation but verified snapshot lacks it -> rejected'})

def test_valid_snapshot_ignores_bad_raw_control(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch);state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
 (tmp_path/'.paper/control.json').write_text('invalid raw JSON ignored in formal path')
 assert m.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)==profile
 OBS.append({'positive':'valid source/profile/reservation snapshot accepted despite unusable raw control'})

def test_PC5_original_bool_repeat_now_rejected(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch);purpose['repeat_ordinal']=False;state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
 with pytest.raises(ValueError,match='primary slot'):m.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)
 OBS.append({'PC5 original':'outer repeat False rejected'})

def test_PC5_nested_binding_bool_repeat_should_reject(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch);admitted=fxt.helper.module.admit_dispatch(tmp_path,tmp_path,packet)
 admitted.binding()['repeat_ordinal']=0;purpose['formal_dispatch_binding']=copy.deepcopy(admitted.binding());purpose['formal_dispatch_binding']['repeat_ordinal']=False
 state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
 result=m.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)
 OBS.append({'PC5 residual':'nested formal_dispatch_binding.repeat_ordinal=False accepted against integer0','outer_repeat_native_int':type(purpose['repeat_ordinal'])is int,'accepted_profile':result==profile})
 assert False,'nested canonical purpose False/0 still admitted'

def test_model_receipt_negative_control(tmp_path,monkeypatch):old_model_negative_control(tmp_path,monkeypatch)
def test_C1_canonical_path_no_flags_now_before_output(tmp_path,monkeypatch):
 with pytest.raises(ValueError,match='canonical formal case'):old_cli_counter(tmp_path,monkeypatch)
 assert not(tmp_path/'flight').exists();OBS.append({'C1':'actual main()/argparse canonical flags refusal before output'})
def teardown_module():Path('/tmp/formal-primary-consumer-repair-review-v2/independent-service-probes.json').write_text(json.dumps(OBS,indent=2)+'\n')
