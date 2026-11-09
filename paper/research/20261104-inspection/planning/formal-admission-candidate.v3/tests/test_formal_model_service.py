"""Synthetic model-service guards; never import MLX, load weights or generate."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('service_dispatch_helpers',HERE/'test_formal_dispatch.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
module=helper.helper.load('formal_model_service_candidate',HERE.parent/'source/eval/inspection_model_service.py')


def setup_service(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'eval.inspection_formal_dispatch',helper.module)
    receipt=tmp_path/'model.json';receipt.write_text('{"software_fixture":true}')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    profile={'download_receipt_sha256':sha(receipt),'file_count':1}
    registry={'models':{'M1':profile}};path=tmp_path/'registry.json';path.write_text(json.dumps(registry))
    plan={'local_model_registry':'registry.json','local_model_registry_sha256':sha(path)}
    packet={'path':'dispatch/main.v1/literal.json','sha256':'a'*64}
    binding={'phase':'main','sampler_seed':70,'packet':packet,'order_index':0,'repeat_ordinal':0}
    slot=SimpleNamespace(condition='B4-M1',case_id='main-literal',repeat_ordinal=0,sampler_seed=70)
    admitted=SimpleNamespace(binding=lambda:binding,episode=SimpleNamespace(slot=slot,execution=SimpleNamespace()))
    seen=[]
    def admit(*args):seen.append(args);return admitted
    monkeypatch.setattr(helper.module,'admit_dispatch',admit)
    purpose={'kind':'inspection_local_generation','phase':'main','model_registry_sha256':sha(path),
        'model_receipt_sha256':sha(receipt),'formal_dispatch':packet,'formal_dispatch_binding':binding,
        'condition':slot.condition,'case_id':slot.case_id,'repeat_ordinal':0}
    state={'frozen_plan':module._digest(plan),'receipts':[{'stage':'plan','packet':plan}],
        'active_round':{'id':'literal','stage':'literature'},'task_stop':None,
        'usage_ledger':{'literal':{'status':'reserved','reserved':{'calls':2,'tokens':1000},
                                'purpose':json.dumps(purpose)}}}
    admitted.episode.execution.owner_snapshot=state
    (tmp_path/'.paper').mkdir()
    return receipt,profile,packet,purpose,state,seen


@pytest.mark.parametrize('fault',['none','phase','packet','binding','condition','case_id','repeat_ordinal','repeat_bool',
                                 'nested_repeat_bool','nested_order_bool','nested_sampler_float',
                                 'reservation','missing_formal_ref','gate'])
def test_service_revalidates_typed_phase_and_slot_before_model_effects(tmp_path,monkeypatch,fault):
    receipt,profile,packet,purpose,state,seen=setup_service(tmp_path,monkeypatch)
    if fault=='phase':purpose['phase']='development'
    elif fault=='packet':purpose['formal_dispatch']={**packet,'sha256':'b'*64}
    elif fault=='binding':purpose['formal_dispatch_binding']={}
    elif fault=='condition':purpose['condition']='B3-M1'
    elif fault=='case_id':purpose['case_id']='main-other'
    elif fault=='repeat_ordinal':purpose['repeat_ordinal']=1
    elif fault=='repeat_bool':purpose['repeat_ordinal']=False
    elif fault=='nested_repeat_bool':purpose['formal_dispatch_binding']={**purpose['formal_dispatch_binding'],'repeat_ordinal':False}
    elif fault=='nested_order_bool':purpose['formal_dispatch_binding']={**purpose['formal_dispatch_binding'],'order_index':False}
    elif fault=='nested_sampler_float':purpose['formal_dispatch_binding']={**purpose['formal_dispatch_binding'],'sampler_seed':70.}
    elif fault=='reservation':state['usage_ledger']['literal']['status']='known'
    elif fault=='gate':
        def reject(*args):raise ValueError('revoked parent')
        monkeypatch.setattr(helper.module,'admit_dispatch',reject)
    state['usage_ledger']['literal']['purpose']=json.dumps(purpose)
    (tmp_path/'.paper/control.json').write_text(json.dumps(state))
    ref=None if fault=='missing_formal_ref' else packet
    if fault=='none':
        assert module.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=ref)==profile
        assert module.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=ref)==profile
        assert len(seen)==2
    else:
        with pytest.raises(ValueError):module.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=ref)


def test_raw_control_cannot_replace_the_verified_owner_snapshot(tmp_path,monkeypatch):
    from copy import deepcopy
    receipt,profile,packet,purpose,state,seen=setup_service(tmp_path,monkeypatch)
    admitted=helper.module.admit_dispatch(tmp_path,tmp_path,packet)
    snapshot=deepcopy(state);snapshot['usage_ledger']={}
    admitted.episode.execution.owner_snapshot=snapshot
    (tmp_path/'.paper/control.json').write_text(json.dumps(state))
    with pytest.raises(ValueError,match='sufficient live owner reservation'):
        module.admitted_limits(tmp_path,'literal',2,1000,receipt,formal_dispatch=packet)
