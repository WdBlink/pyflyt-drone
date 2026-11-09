"""CLI refusal and actual SDK/full-shape Fake adapter tests; no model/flight."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent;STAGE=HERE.parent/'source/eval'
spec=importlib.util.spec_from_file_location('runner_dispatch_helpers',HERE/'test_formal_dispatch.py')
helper=importlib.util.module_from_spec(spec);sys.modules[spec.name]=helper;spec.loader.exec_module(helper)
load=helper.helper.load
load('eval.inspection_population',STAGE/'inspection_population.py')
driver=load('formal_primary_driver_candidate',STAGE/'run_inspection_formal.py')
PROJECT=HERE.parents[5]
SDK=PROJECT/'paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1'


@pytest.mark.parametrize('runner',['run_inspection_reference.py','run_inspection_llm_development.py'])
def test_formal_gate_refusal_precedes_output_client_or_process_construction(tmp_path,monkeypatch,runner):
    monkeypatch.setitem(sys.modules,'eval.inspection_formal_dispatch',helper.module)
    module=load('cli_'+runner[:-3],STAGE/runner);out=tmp_path/'no-output';seen=[]
    def reject(*args,**kwargs):seen.append('gate');raise ValueError('unadmitted formal input')
    def forbidden(*args,**kwargs):raise AssertionError('effect before admission')
    monkeypatch.setattr(helper.module,'cli_dispatch',reject)
    monkeypatch.setattr(module.mp,'get_context',forbidden)
    if runner=='run_inspection_llm_development.py':monkeypatch.setattr(module,'LocalModelClient',forbidden)
    argv=[runner,'--run',str(tmp_path),'--case',str(tmp_path/'private.json'),
          '--formal-dispatch','dispatch/main.v1/literal.json','--formal-dispatch-sha256','a'*64,'--out',str(out)]
    if runner=='run_inspection_llm_development.py':
        argv+=['--reservation','literal','--model','M1','--architecture','B4','--sampler-seed','1']
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(ValueError,match='unadmitted formal'):module.main()
    assert seen==['gate'] and not out.exists()


def test_classical_plan_and_actual_argv_preserve_the_proof_input(tmp_path,monkeypatch):
    f,refs,execution,slot,packet=helper.prepared(tmp_path,monkeypatch,'B0')
    admission=helper.module.admit_dispatch(f.run,f.project,packet)
    plan=driver.episode_plan(admission,f.run,f.project,sys.executable)
    assert plan['case_sha256']==admission.episode.case.case_sha256
    assert plan['formal_dispatch']['packet']==packet
    assert plan['sampler_seed'] is None
    command=driver.episode_command(plan,tmp_path/'flight',project_root=f.project)
    assert command[command.index('--formal-dispatch')+1]==packet['path']
    assert command[command.index('--policy')+1]=='B0'
    assert command[command.index('--layout')+1]=='generated'
    assert command[command.index('--duration')+1]=='600' and '--realtime' in command


def test_formal_classical_cli_passes_admitted_case_protocol_to_spawn(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'eval.inspection_formal_dispatch',helper.module)
    module=load('positive_classical_cli_candidate',STAGE/'run_inspection_reference.py')
    project=tmp_path/'project';project.mkdir();case=project/'case.json';case.write_text('{"fake":true}')
    out=project/'flight';calls=[]
    admitted=SimpleNamespace(binding=lambda:{'phase':'main'},worker_input={'software':'fixture'},
        frozen_sources=lambda *args:{str(case):case.read_text()})
    monkeypatch.setattr(helper.module,'cli_dispatch',lambda *args,**kwargs:admitted)
    class StopAtSpawn:
        def Queue(self):return object()
        def Array(self,*args):return object()
        def Process(self,**kwargs):calls.append(kwargs);raise RuntimeError('intentional fake spawn boundary')
    monkeypatch.setattr(module.mp,'get_context',lambda *args:StopAtSpawn())
    monkeypatch.setattr(sys,'argv',['reference','--run',str(project),'--case',str(case),
        '--formal-dispatch','literal','--formal-dispatch-sha256','a'*64,'--layout','generated',
        '--realtime','--duration','600','--out',str(out)])
    with pytest.raises(RuntimeError,match='fake spawn'):module.main()
    assert len(calls)==1 and calls[0]['args'][:4]==(0,False,True,600.)
    assert calls[0]['args'][7] is False and calls[0]['kwargs']=={'formal_input':admitted.worker_input}
    assert json.loads((out/'run.json').read_text())['formal_dispatch_binding']=={'phase':'main'}


def fake_driver(tmp_path,monkeypatch,phase):
    f=helper.helper.ExecutionFixture(tmp_path)
    opening=None
    if phase=='holdout':
        main=f.dataset();opening=f.typed_main_opening(main[1])
    refs,execution=f.execution(phase,opening);proof=f.execution_proof(refs,execution,opening)
    request={'schema':'inspection-primary-run-v1','run':str(f.run),'runtime':str(f.project/'runtime'),
        'sdk':str(SDK),'g4':f.acceptance,'namespace':refs[0],'dataset':refs[1],
        'dataset_review':refs[2],'execution':execution,'opening':opening}
    seen=[];finished={};pending={}
    monkeypatch.setattr(driver,'admitted_request',lambda *args:(f.verification,proof))
    def dispatch(*args,**kwargs):
        identity=(args[7],args[8],args[9]);slot=next(row for row in proof.slots
            if (row.case_id,row.condition,row.repeat_ordinal)==identity)
        assert slot.order_index==len(seen)
        pending['slot']=slot
        return {'path':'literal-software-packet','sha256':'a'*64}
    monkeypatch.setattr(driver,'write_primary_dispatch',dispatch)
    def admit(*args):
        slot=pending['slot']
        return SimpleNamespace(binding=lambda:{'case_id':slot.case_id,'condition':slot.condition,
            'repeat_ordinal':slot.repeat_ordinal,'order_index':slot.order_index,'phase':phase},
            episode=SimpleNamespace(slot=slot))
    monkeypatch.setattr(driver,'admit_dispatch',admit)
    def plan(admission,*args):
        slot=admission.episode.slot
        return {'case_id':slot.case_id,'condition':slot.condition,'repeat_ordinal':slot.repeat_ordinal,
                'sampler_seed':slot.sampler_seed}
    monkeypatch.setattr(driver,'episode_plan',plan)
    def execute(directory,plan,command,**kwargs):
        slot=pending['slot'];seen.append(slot)
        finished[str(directory)]=slot.repeat_ordinal/4
    monkeypatch.setattr(driver,'execute_local_episode',execute)
    monkeypatch.setattr(driver,'execute_development_llm_episode',execute)
    monkeypatch.setattr(driver,'measure_completed_episode',lambda directory,expected:
        (finished[str(directory)],{'role':'Fake software value, never physical or method evidence'}))
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        monkeypatch.setenv(key,'1')
    return f,request,proof,seen


@pytest.mark.parametrize('phase,count',[('main',1296),('holdout',216)])
def test_actual_sdk_preserves_complete_order_and_root_means_with_fake_effects(tmp_path,monkeypatch,phase,count):
    f,request,proof,seen=fake_driver(tmp_path,monkeypatch,phase);out=f.project/'logs/sdk'
    result=driver.run_primary(request,out,project_root=f.project)
    assert result['actual_episode_denominator']==count and seen==list(proof.slots)
    assert result['outer_aggregate_rows']==count//18*8
    for group,multiple in [('classic',3),('llm',15),('root-means',8)]:
        value=json.loads((out/group/'results.json').read_text())
        assert value['execution_status']=='completed' and len(value['runs'])==count//18*multiple
    outer=json.loads((out/'root-means/results.json').read_text())
    assert all(row['value']==(0 if row['condition'] in ('B0','B1','B2') else .25) for row in outer['runs'])


def test_operational_failure_retains_all_three_sdk_denominators(tmp_path,monkeypatch):
    f,request,proof,seen=fake_driver(tmp_path,monkeypatch,'main');out=f.project/'logs/failed-sdk'
    def fail(*args,**kwargs):raise RuntimeError('intentional software infrastructure failure')
    monkeypatch.setattr(driver,'execute_local_episode',fail)
    with pytest.raises(RuntimeError,match='intentional software'):driver.run_primary(request,out,project_root=f.project)
    for group,count in [('classic',216),('llm',1080),('root-means',576)]:
        value=json.loads((out/group/'results.json').read_text())
        assert value['execution_status']=='failed' and len(value['runs'])==count
    assert not (out/'primary-callbacks-complete.json').exists()
