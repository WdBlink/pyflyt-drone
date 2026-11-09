"""Dispatch admission fixtures; no formal entropy, model or physical effects."""
import importlib.util
from pathlib import Path
import sys

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('dispatch_execution_helpers',HERE/'test_formal_execution.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
module=helper.load('eval.inspection_formal_dispatch',HERE.parent/'source/eval/inspection_formal_dispatch.py')


def prepared(tmp_path,monkeypatch,condition='B4-M1'):
    f=helper.ExecutionFixture(tmp_path);refs,execution=f.execution()
    monkeypatch.setattr(module,'context_from_environment',lambda *args:f.verification)
    slot=next(row for row in f.execution_proof(refs,execution).slots if row.condition==condition)
    packet=module.write_primary_dispatch(f.run,f.project,f.acceptance,*refs,execution,
        slot.case_id,slot.condition,slot.repeat_ordinal,runtime=f.project/'runtime',sdk=f.project/'sdk',
        verification=f.verification)
    return f,refs,execution,slot,packet


@pytest.mark.parametrize('condition',['B0','B1','B2','B3-M1','B3-M2','B3-M3','B4-M1','B4-M2'])
def test_dispatch_binds_exact_primary_slot_before_any_effect(tmp_path,monkeypatch,condition):
    f,refs,execution,slot,packet=prepared(tmp_path,monkeypatch,condition)
    result=module.admit_dispatch(f.run,f.project,packet)
    assert result.episode.slot==slot
    assert result.worker_input['case_id']==slot.case_id
    assert result.worker_input['dataset']==refs[1]
    assert module.write_primary_dispatch(f.run,f.project,f.acceptance,*refs,execution,
        slot.case_id,slot.condition,slot.repeat_ordinal,runtime=f.project/'runtime',sdk=f.project/'sdk',
        verification=f.verification)==packet


@pytest.mark.parametrize('fault',['condition','repeat','phase','schema','extra','revocation','packet_drift'])
def test_changed_dispatch_or_parent_proof_is_rejected(tmp_path,monkeypatch,fault):
    f,refs,execution,slot,packet=prepared(tmp_path,monkeypatch)
    value=f.read(packet)
    if fault=='condition':value['condition']='B0'
    elif fault=='repeat':value['repeat_ordinal']=True
    elif fault=='phase':value['phase']='holdout'
    elif fault=='schema':value['schema']='untrusted'
    elif fault=='extra':value['sampler_seed']=0
    elif fault=='revocation':
        f.bind('synthetic-revoke',[f.catalog],revises=helper.module.EXECUTION_FINDINGS['main'])
    else:
        original=module.admit_primary_episode
        def change(*args,**kwargs):
            result=original(*args,**kwargs)
            path=f.run/packet['path'];path.write_bytes(path.read_bytes()+b' ')
            return result
        monkeypatch.setattr(module,'admit_primary_episode',change)
    if fault not in ('revocation','packet_drift'):packet=f.put(packet['path'],value)
    with pytest.raises(ValueError):module.admit_dispatch(f.run,f.project,packet)


def classical_plan(f,slot,packet):
    proof=module.admit_dispatch(f.run,f.project,packet);case=proof.episode.case
    return {'formal_dispatch':{'run':str(f.run),'packet':packet},
        'case_id':slot.case_id,'condition':slot.condition,'repeat_ordinal':slot.repeat_ordinal,
        'sampler_seed':None,'case_path':str(f.run/case.case.path),
        'case_file_sha256':case.case.sha256,'case_sha256':case.case_sha256,
        'suite_manifest_sha256':proof.episode.execution.dataset.manifest.sha256,
        'max_clock_lag_seconds':.1,'max_attempts':2,
        'thread_environment':{key:'1' for key in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}}


@pytest.mark.parametrize('field,value',[
    ('condition','B1'),('case_id','main-other'),('repeat_ordinal',1),('sampler_seed',1),
    ('case_path','/tmp/foreign'),('case_file_sha256','0'*64),('case_sha256','0'*64),
    ('suite_manifest_sha256','0'*64),('max_clock_lag_seconds',.2),('max_attempts',3),
    ('thread_environment',{}),('ablation','periodic'),('injected_delay_seconds',20)])
def test_actual_episode_plan_cannot_change_the_admitted_primary_protocol(tmp_path,monkeypatch,field,value):
    f,refs,execution,slot,packet=prepared(tmp_path,monkeypatch,'B0')
    plan=classical_plan(f,slot,packet)
    assert module.admit_episode_plan(plan,f.project).episode.slot==slot
    plan[field]=value
    with pytest.raises(ValueError):module.admit_episode_plan(plan,f.project)


@pytest.mark.parametrize('condition,sampler',[('B1',None),('B0',1)])
def test_cli_identity_is_checked_before_returning_worker_input(tmp_path,monkeypatch,condition,sampler):
    f,refs,execution,slot,packet=prepared(tmp_path,monkeypatch,'B0')
    path=f.run/module.admit_dispatch(f.run,f.project,packet).episode.case.case.path
    with pytest.raises(ValueError):module.cli_dispatch(f.run,f.project,packet['path'],packet['sha256'],
        case_path=path,condition=condition,sampler_seed=sampler)
