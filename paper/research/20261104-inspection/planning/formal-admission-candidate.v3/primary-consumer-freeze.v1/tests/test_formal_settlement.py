"""Native settlement content fixtures, no real owner/resource mutation."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('settlement_dispatch_helpers',HERE/'test_formal_dispatch.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
module=helper.module


@pytest.mark.parametrize('fault',['none','status','unreserved','reserved','usage','purpose','phase',
    'case','condition','repeat','attempt','receipt_path','receipt_sha','receipt_usage','receipt_meter','frozen_plan'])
def test_formal_receipt_requires_identical_known_native_settlement(tmp_path,monkeypatch,fault):
    packet={'path':'literal-dispatch.json','sha256':'a'*64};binding={'phase':'main','packet':packet}
    plan={'formal_dispatch':{'run':str(tmp_path),'packet':packet},'reservation_usage':{'calls':32,'tokens':180000},
        'model_registry_sha256':'b'*64,'model_receipt_sha256':'c'*64,'condition':'B4-M1',
        'case_id':'main-literal','repeat_ordinal':2,'owner_frozen_plan':'d'*64}
    meter={'usage':{'calls':1,'tokens':4},'role':'software fixture'}
    reservation='literal-a0';path='qualification/local-model-episodes/'+reservation+'/usage-receipt.json'
    value={'reservation':reservation,'plan_sha256':module.canonical_digest(plan),
           'usage':meter['usage'],'resource_recount':meter}
    purpose={'kind':'inspection_local_generation','phase':'main','condition':'B4-M1','case_id':'main-literal',
        'repeat_ordinal':2,'episode_plan_sha256':module.canonical_digest(plan),'attempt_ordinal':0,
        'model_registry_sha256':'b'*64,'model_receipt_sha256':'c'*64,
        'formal_dispatch':packet,'formal_dispatch_binding':binding}
    entry={'status':'known','unreserved':False,'reserved':plan['reservation_usage'],
        'reported_usage':meter['usage'],'receipt':path}
    if fault=='status':entry['status']='reserved'
    elif fault=='unreserved':entry['unreserved']=True
    elif fault=='reserved':entry['reserved']={}
    elif fault=='usage':entry['reported_usage']={'calls':0,'tokens':0}
    elif fault=='purpose':purpose['kind']='untyped'
    elif fault=='phase':purpose['phase']='development'
    elif fault=='case':purpose['case_id']='main-other'
    elif fault=='condition':purpose['condition']='B3-M1'
    elif fault=='repeat':purpose['repeat_ordinal']=1
    elif fault=='attempt':purpose['attempt_ordinal']=1
    elif fault=='receipt_path':entry['receipt']='foreign.json'
    elif fault=='receipt_usage':value['usage']={}
    elif fault=='receipt_meter':value['resource_recount']={}
    target=tmp_path/path;target.parent.mkdir(parents=True);target.write_text(json.dumps(value))
    entry['receipt_sha256']='e'*64 if fault=='receipt_sha' else module.file_sha(target)
    entry['purpose']=json.dumps(purpose)
    owner={'frozen_plan':'f'*64 if fault=='frozen_plan' else plan['owner_frozen_plan'],
           'usage_ledger':{reservation:entry}}
    admitted=SimpleNamespace(binding=lambda:binding,packet=SimpleNamespace(json=lambda run:
        {'runtime':str(tmp_path/'runtime'),'sdk':str(tmp_path/'sdk')}))
    monkeypatch.setattr(module,'admit_episode_plan',lambda *args:admitted)
    monkeypatch.setattr(module,'context_from_environment',lambda *args:SimpleNamespace(owner_reader=lambda run:owner))
    if fault=='none':assert module.admit_model_settlement(plan,tmp_path,reservation,0,meter).path==path
    else:
        with pytest.raises(ValueError):module.admit_model_settlement(plan,tmp_path,reservation,0,meter)
