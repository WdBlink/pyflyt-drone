"""Warmup-order and branch checks with fake constructors; no physical world."""
import ast
import importlib.util
import json
from pathlib import Path
import queue
import sys
import types
from dataclasses import replace

import pytest

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent/'source/eval'
spec = importlib.util.spec_from_file_location('worker_content_helpers', HERE/'test_formal_case.py')
helper = importlib.util.module_from_spec(spec); sys.modules[spec.name] = helper
spec.loader.exec_module(helper)
load = helper.load


def worker_fixture(tmp_path, monkeypatch):
    f, refs, row, private, case = helper.selected_fixture(tmp_path)
    events = []
    class NoPhysicsMission:
        def __init__(self, *args, **kwargs):
            events.append('warmup-constructor')
            raise RuntimeError('intentional stop before any physical world')
    fake = types.ModuleType('envs.inspection.mission'); fake.InspectionMission = NoPhysicsMission
    monkeypatch.setitem(sys.modules, 'envs.inspection.mission', fake)
    monkeypatch.setitem(sys.modules, 'eval.inspection_formal_case', helper.module)
    worker = load('formal_reference_worker_candidate', STAGE/'run_inspection_reference.py')
    monkeypatch.setattr(worker, 'ROOT', f.project)
    def verifier(*args): events.append('verifier'); return f.verification
    monkeypatch.setattr(helper.module, 'context_from_environment', verifier)
    original = helper.module.load_formal_worker_input
    def observe(*args):
        events.append('admission')
        result = original(*args)
        assert result.private_copy_sha256 == row['case']['sha256']
        events.append('content-verified')
        return result
    monkeypatch.setattr(helper.module, 'load_formal_worker_input', observe)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        monkeypatch.setenv(key,'1')
    return f, refs, row, private, helper.worker_payload(f,refs,row), worker, events


@pytest.mark.parametrize('fault', ['none', 'private_changed', 'g4_changed', 'payload_shape',
                                  'duration', 'realtime', 'threads', 'copy_location', 'legacy'])
def test_formal_admission_and_content_checks_precede_warmup(tmp_path, monkeypatch, fault):
    f, refs, row, private, payload, worker, events = worker_fixture(tmp_path,monkeypatch)
    duration, realtime = 600, True
    if fault == 'private_changed': private.write_text('{}')
    elif fault == 'g4_changed': payload['g4'] = {**payload['g4'], 'sha256': '0'*64}
    elif fault == 'payload_shape': payload.pop('sdk')
    elif fault == 'duration': duration = 599
    elif fault == 'realtime': realtime = False
    elif fault == 'threads': monkeypatch.setenv('OMP_NUM_THREADS','2')
    elif fault == 'copy_location':
        other = f.project/'other-private.json'; other.write_bytes(private.read_bytes()); private = other
    elif fault == 'legacy': payload = None
    updates = queue.Queue(); out = f.project/'flight'
    worker.worker(0,False,realtime,duration,queue.Queue(),updates,out,False,None,private,
                  row['case']['sha256'],formal_input=payload)
    world = json.loads((out/'world.json').read_text())
    assert world['status'] == 'operational_failure'
    if fault == 'none':
        assert events == ['admission','verifier','content-verified','warmup-constructor']
    elif fault == 'legacy': assert events == ['warmup-constructor']
    else: assert 'warmup-constructor' not in events
    assert updates.get_nowait() == {'kind':'done','status':'operational_failure'}


@pytest.mark.parametrize('mode', ['main', 'legacy', 'foreign_phase', 'bad_geometry'])
def test_scenario_content_guard_precedes_physical_parent_constructor(tmp_path, monkeypatch, mode):
    f, refs, row, private, case = helper.selected_fixture(tmp_path); events=[]
    class NoPhysicalParent:
        def __init__(self,*args,**kwargs):
            events.append('physical-parent'); raise RuntimeError('fake parent boundary')
    fake=types.ModuleType('envs.inspection.mission');fake.InspectionMission=NoPhysicalParent
    monkeypatch.setitem(sys.modules,'envs.inspection.mission',fake)
    previous=sys.modules.get('eval.inspection_population')
    monkeypatch.setitem(sys.modules,'eval.inspection_population',helper.shared)
    scenario=load('formal_scenario_candidate',STAGE/'inspection_scenarios.py')
    if mode=='bad_geometry':case=replace(case,wind_xy=(9.,0.))
    phase=None if mode=='legacy' else 'holdout' if mode=='foreign_phase' else 'main'
    if mode=='main':
        with pytest.raises(RuntimeError,match='fake parent boundary'):
            scenario.ScenarioMission(case,formal_phase=phase)
        assert events==['physical-parent']
    else:
        with pytest.raises(ValueError):scenario.ScenarioMission(case,formal_phase=phase)
        assert events==[]


def test_worker_control_loop_and_existing_scenario_methods_are_ast_unchanged():
    project=HERE.parents[5]
    old=ast.parse((project/'eval/run_inspection_reference.py').read_text())
    new=ast.parse((STAGE/'run_inspection_reference.py').read_text())
    def loop(tree):
        worker=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='worker')
        return next(node for node in ast.walk(worker) if isinstance(node,ast.While))
    assert ast.dump(loop(old))==ast.dump(loop(new))
    old=ast.parse((project/'eval/inspection_scenarios.py').read_text())
    new=ast.parse((STAGE/'inspection_scenarios.py').read_text())
    def methods(tree):
        cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='ScenarioMission')
        return {node.name:ast.dump(node) for node in cls.body if isinstance(node,ast.FunctionDef)
                and node.name not in ('__init__','business_audit')}
    assert methods(old)==methods(new)
