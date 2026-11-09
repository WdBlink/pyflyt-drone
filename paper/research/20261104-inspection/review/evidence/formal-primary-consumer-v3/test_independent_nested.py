"""Only PC5-b canonical deltas: literal service/meter data, no generation/native action."""
from pathlib import Path
import importlib.util,json,sys,copy
import pytest
HOME=Path('/tmp/formal-primary-nested-binding-review-v3');PROJECT=HOME/'project';sys.path.insert(0,str(PROJECT));T=PROJECT/'paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests'
BASE=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/formal-admission-candidate.v3/primary-consumer-repair-freeze.v2/source/eval')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
fxt=load('independent_nested_service_fixture',T/'test_formal_model_service.py');service=fxt.module
old_service=load('old_frozen_v2_service',BASE/'inspection_model_service.py');old_meter=load('old_frozen_v2_meter',BASE/'inspection_model_receipts.py');new_meter=load('qualified_nested_meter',PROJECT/'eval/inspection_model_receipts.py');meter_helper=load('independent_meter_literal_fixture',PROJECT/'tests/test_inspection_model_receipts.py')
OBS=[]
@pytest.mark.parametrize('field,value',[('repeat_ordinal',False),('order_index',False),('sampler_seed',70.0)])
def test_service_old_type_alias_now_blocks_before_setup(tmp_path,monkeypatch,field,value):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch);admitted=fxt.helper.module.admit_dispatch(tmp_path,tmp_path,packet)
 purpose['formal_dispatch_binding']=copy.deepcopy(admitted.binding());purpose['formal_dispatch_binding'][field]=value;state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
 assert old_service.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)==profile
 with pytest.raises(ValueError,match='primary slot'):service.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)
 effects=[]
 def forbidden(*a,**k):effects.append(True);raise AssertionError('effect after rejected typed identity')
 monkeypatch.setattr(service,'verify_model_files',forbidden);monkeypatch.setattr(service,'operation_directory',forbidden)
 monkeypatch.setattr(sys,'argv',['service','--run',str(tmp_path),'--reservation','literal','--model-receipt',str(receipt),'--max-calls','2','--max-total-tokens','1000','--formal-dispatch',packet['path'],'--formal-dispatch-sha256',packet['sha256']])
 with pytest.raises(ValueError,match='primary slot'):service.main()
 assert effects==[]
 OBS.append({'area':'service','field':field,'actual_type':type(value).__name__,'old_v2_accepted':True,'new_admission_rejected':True,'actual_main_rejected_before_file_or_operation_hooks':True,'no_model_loaded':True})

def test_service_valid_typed_binding_key_order_independent(tmp_path,monkeypatch):
 receipt,profile,packet,purpose,state,seen=fxt.setup_service(tmp_path,monkeypatch);admitted=fxt.helper.module.admit_dispatch(tmp_path,tmp_path,packet)
 purpose['formal_dispatch_binding']=dict(reversed(list(admitted.binding().items())));state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
 assert service.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)==profile
 OBS.append({'area':'service','legitimate_native_int_binding_accepted':True,'object_key_order_irrelevant':True})

@pytest.mark.parametrize('field,value',[('repeat_ordinal',False),('order_index',False),('sampler_seed',70.0)])
def test_meter_old_alias_now_rejected(tmp_path,field,value):
 expected=meter_helper.fixture(tmp_path);expected['formal_dispatch_binding']={'order_index':0,'repeat_ordinal':0,'sampler_seed':70}
 p=tmp_path/'service-intent.json';intent=json.loads(p.read_text());intent['formal_dispatch_binding']=copy.deepcopy(expected['formal_dispatch_binding']);p.write_text(json.dumps(intent))
 assert new_meter.validate_model_operation(tmp_path,expected)['usage']=={'calls':1,'tokens':4}
 intent['formal_dispatch_binding'][field]=value;p.write_text(json.dumps(intent));before={str(q.relative_to(tmp_path)):q.read_bytes() for q in tmp_path.rglob('*') if q.is_file()}
 assert old_meter.validate_model_operation(tmp_path,expected)['usage']=={'calls':1,'tokens':4}
 with pytest.raises(ValueError,match='formal dispatch differs'):new_meter.validate_model_operation(tmp_path,expected)
 assert {str(q.relative_to(tmp_path)):q.read_bytes() for q in tmp_path.rglob('*') if q.is_file()}==before
 OBS.append({'area':'meter','field':field,'actual_type':type(value).__name__,'old_v2_accepted':True,'new_rejected':True,'read_only_bytes_unchanged':True})

def test_meter_valid_typed_binding_and_key_order(tmp_path):
 expected=meter_helper.fixture(tmp_path);binding={'order_index':0,'repeat_ordinal':0,'sampler_seed':70};expected['formal_dispatch_binding']=binding
 p=tmp_path/'service-intent.json';intent=json.loads(p.read_text());intent['formal_dispatch_binding']=dict(reversed(list(binding.items())));p.write_text(json.dumps(intent));assert new_meter.validate_model_operation(tmp_path,expected)['usage']=={'calls':1,'tokens':4}
 OBS.append({'area':'meter','legitimate_native_int_binding_accepted':True,'object_key_order_irrelevant':True})
def teardown_module():HOME.joinpath('independent-probes.json').write_text(json.dumps(OBS,indent=2)+'\n')
