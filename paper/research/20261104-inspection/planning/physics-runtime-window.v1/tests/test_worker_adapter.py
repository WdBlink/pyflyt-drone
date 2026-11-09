"""Final policy-error propagation with a skipped Fake loop; no aircraft."""
import ast
import importlib.util
import json
from pathlib import Path
import queue
import sys
from types import ModuleType,SimpleNamespace

import pytest

HERE=Path(__file__).resolve().parent;SOURCE=HERE.parent/'source';PROJECT=HERE.parents[5]
spec=importlib.util.spec_from_file_location('candidate_timed_worker',SOURCE/'timed_worker.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


@pytest.mark.parametrize('errors',[[],['intentional restore failure']])
def test_policy_errors_make_the_saved_world_operational_failure(tmp_path,monkeypatch,errors):
    created=[]
    class Mission:
        def __init__(self,*args,**kwargs):
            self.t=0. if not created else 600.;created.append(self)
            self.executor=SimpleNamespace(infrastructure_errors=[])
            self.flight=SimpleNamespace(world=SimpleNamespace(physics_steps=144000,physics_hz=240,step_period=1/120))
        def step(self):self.t+=1.
        def close(self):pass
        def public_state(self):return {}
        def audit(self):return {'score':{'role':'software Fake, not a physical score'}}
        def business_audit(self):return {}
    fake=ModuleType('envs.inspection.mission');fake.InspectionMission=Mission
    monkeypatch.setitem(sys.modules,'envs.inspection.mission',fake)
    class Profile:
        def __init__(self,duration):assert len(created)==2 and duration==600
        def anchor(self,epoch):pass
        def initial_publication_end(self):pass
        def stop(self):pass
        def save(self,out):return {'runtime_policy':{'errors':errors}}
    updates=queue.Queue()
    module.worker(0,True,False,600,queue.Queue(),updates,tmp_path,profile_factory=Profile)
    value=json.loads((tmp_path/'world.json').read_text())
    assert value['status']==('operational_failure' if errors else 'completed')
    assert value['clock_profile']['runtime_policy']['errors']==errors


def test_original_physics_loop_is_ast_unchanged():
    loop=lambda text:ast.dump(next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.While)))
    assert loop((PROJECT/'eval/inspection_diagnostic_worker.py').read_text())==loop((SOURCE/'timed_worker.py').read_text())
