"""Independent frozen-source CPU audit. No flight/model imports or generation."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
import traceback
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

REPO = Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone')
R = REPO/'paper/research/20261104-inspection'
OUT = Path(__file__).resolve().parent
V1 = R/'qualification/public-contract-clean-candidate.v1'
V2 = R/'qualification/public-contract-clean-candidate.v2'
PAIR = R/'qualification/dev-same-level-event-pair.v2'
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

for name in ('eval', 'envs', 'envs.inspection'):
    module = ModuleType(name); module.__path__ = []; sys.modules[name] = module
scenario = load('envs.inspection.scenario', V2/'source/envs/inspection/scenario.py')
cases = load('eval.inspection_cases', V2/'source/eval/inspection_cases.py')
v1 = load('frozen_population_v1', V1/'source/eval/inspection_population.py')
p = load('eval.inspection_population', V2/'source/eval/inspection_population.py')
binding = json.loads((V2/'source-binding.json').read_text())
source_checks = {name: {'expected': digest, 'snapshot': sha(V2/'source'/name),
    'current': sha(REPO/name)} for name, digest in binding['sources'].items()}
assert all(x['expected'] == x['snapshot'] for x in source_checks.values())
parent_path = R/'qualification/dev-population.v1/devext-OOD-L3-00.json'
parent_value = json.loads(parent_path.read_text())
old_parent = v1.load_development_case(parent_value)
old_clean = v1.event_clean_sibling(old_parent)
old_loaded = v1.load_development_case(json.loads(json.dumps(old_clean.as_dict())))
old_failure = {'objects_equal': old_clean == old_loaded,
    'semantic_digest_equal': old_clean.digest() == old_loaded.digest(),
    'parent_roi_generated_type': type(old_clean.parent_case['roi_polygon']).__name__,
    'parent_roi_loaded_type': type(old_loaded.parent_case['roi_polygon']).__name__}
assert old_failure['objects_equal'] is False and old_failure['semantic_digest_equal'] is True
parent = p.load_development_case(parent_value)
clean = p.event_clean_sibling(parent)
value = json.loads(json.dumps(clean.as_dict()))
restored = p.load_development_case(value)
assert type(clean) is p.EventCleanSibling and restored == clean and restored.digest() == clean.digest()
assert clean.parent_sha256 == parent.digest() and clean.level == parent.level == 'L3'
base = parent.as_dict()
differences = sorted(k for k in base if base[k] != clean.as_dict()[k])
assert set(differences) == {'schema', *p.EVENT_FIELDS}
assert all(getattr(clean, k) is None for k in p.EVENT_FIELDS)
for entity, other in zip(parent.entities, clean.entities):
    assert all(entity.position(t) == other.position(t) for t in (0, 120, 180, 240, 360, 600))

rejections = []
def reject(name, mutate, original=value):
    candidate = copy.deepcopy(original); mutate(candidate)
    try: p.load_development_case(candidate)
    except (ValueError, TypeError): rejections.append(name)
    else: raise AssertionError('unexpected admission: '+name)
for key, replacement in [('level','L2'),('split','main'),('split','holdout'),('seed',True),
        ('case_id','devext-other'),('lineage_group','devext-other'),('domain','ID'),
        ('parent_sha256','0'*64),('retire_at',parent.retire_at),('schema','undeclared')]:
    reject('root_'+key+'_'+str(replacement), lambda x, k=key, v=replacement: x.__setitem__(k,v))
reject('root_unknown', lambda x:x.__setitem__('unknown',1))
reject('wind', lambda x:x['wind_xy'].__setitem__(0,x['wind_xy'][0]+.01))
reject('geometry', lambda x:x['roi_polygon'][0].__setitem__(0,x['roi_polygon'][0][0]+.1))
reject('motion', lambda x:x['entities'][-1].__setitem__('speed',1.3))
reject('entity_identity', lambda x:x['entities'][0].__setitem__('key','other'))
reject('entity_unknown', lambda x:x['entities'][0].__setitem__('unknown',1))
reject('parent_unknown', lambda x:x['parent_case'].__setitem__('unknown',1))
reject('parent_unknown_entity', lambda x:x['parent_case']['entities'][0].__setitem__('unknown',1))
reject('parent_main', lambda x:x['parent_case'].__setitem__('split','main'))
reject('parent_holdout', lambda x:x['parent_case'].__setitem__('split','holdout'))
reject('parent_L2', lambda x:x['parent_case'].__setitem__('level','L2'))
reject('parent_no_calendar', lambda x:x['parent_case'].update(dict.fromkeys(p.EVENT_FIELDS)))
reject('parent_invalid_calendar', lambda x:x['parent_case'].__setitem__('link_end',x['parent_case']['link_start']+100))
reject('parent_bad_geometry', lambda x:x['parent_case']['roi_polygon'][0].__setitem__(0,999))
reject('ordinary_L3_no_calendar', lambda x:x.update(dict.fromkeys(p.EVENT_FIELDS)), parent_value)
for key in p.EVENT_FIELDS:
    if key != 'retire_at':
        reject('restored_event_'+key, lambda x, k=key: x.__setitem__(k,base[k]))

class ProxyWorld:
    GEOM_SPHERE=2
    def __init__(self): self.positions={}; self.counter=1000
    def createCollisionShape(self,*a,**k): return 1
    def createMultiBody(self,*a,**k): self.counter+=1; return self.counter
    def resetBasePositionAndOrientation(self,body,pos,orn): self.positions[body]=pos
    def setCollisionFilterGroupMask(self,*a): pass
    def register_all_new_bodies(self): pass
class Parent:
    def __init__(self,seed,**kwargs):
        self.parent_input=(seed,kwargs); self.flight=SimpleNamespace(world=ProxyWorld())
        self.entity_bodies={}; self.retired=set(); self.t=0
        self.card={'task_card_version':1,'rules':[{'rule_id':'static-'+str(i),'object_type':'static','priority':1} for i in range(3)]+
            [{'rule_id':'dynamic-inspection','object_type':'dynamic','priority':1}]}
        self.public_events=[]; self.jobs=SimpleNamespace(versions=[0]*4); self.submitted=[]
        self.executor=SimpleNamespace(submit=lambda *a,**k:self.submitted.append((a,k)))
    def _refresh_public(self): pass
tree=ast.parse((V2/'source/eval/inspection_scenarios.py').read_text())
klass=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='ScenarioMission')
space={'InspectionMission':Parent,'validate_development_case':p.validate_development_case,'coverage_cells':cases.coverage_cells}
exec(compile(ast.Module(body=[klass],type_ignores=[]),'frozen-scenarios','exec'),space)
event_m=space['ScenarioMission'](parent,start_planner=False)
clean_m=space['ScenarioMission'](restored,start_planner=False)
assert event_m.card == clean_m.card and clean_m.card['level']=='L3'
assert event_m.parent_input==clean_m.parent_input
assert event_m.flight.world.positions==clean_m.flight.world.positions
for drone in range(4):
    for stamp in (0,parent.link_start,parent.link_end,parent.camera_start,parent.camera_end,600):
        assert clean_m.online(drone,stamp) and clean_m.camera_quality(drone,stamp)==1
clean_m.t=600; clean_m._events()
assert not clean_m.retired and not clean_m.public_events and not clean_m.submitted
assert not event_m.online(parent.link_drone,parent.link_start)
assert event_m.online(parent.link_drone,parent.link_end)
assert event_m.camera_quality(parent.camera_drone,parent.camera_start)==.55
event_m.t=600; event_m._events()
assert len(event_m.submitted)==1 and parent.retire_drone in event_m.retired and len(event_m.public_events)==2

# Actual worker body, explicit fake backend, zero mission steps on selected input.
captured=[]
class FakeMission:
    def __init__(self,*a,**k):
        self.t=0.; self.executor=SimpleNamespace(infrastructure_errors=[])
        self.flight=SimpleNamespace(world=SimpleNamespace(physics_steps=0,physics_hz=240,step_period=1/120))
    def step(self): self.t+=1/120
    def close(self): pass
    def public_state(self,**k): return {'fixture':True}
    def audit(self): return {'fixture':True}
    def business_audit(self): return {'fixture':True}
class FakeScenario(FakeMission):
    def __init__(self,case): super().__init__(); captured.append(case)
tree=ast.parse((V2/'source/eval/run_inspection_reference.py').read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='worker')
ws={'Path':Path,'hashlib':hashlib,'json':json,'time':time,'queue':queue,'traceback':traceback}
exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen-worker','exec'),ws)
mods={name:ModuleType(name) for name in ('envs.inspection.mission','envs.inspection.protocol','eval.inspection_scenarios')}
mods['envs.inspection.mission'].InspectionMission=FakeMission
mods['envs.inspection.protocol'].audit_value=lambda x:x
mods['eval.inspection_scenarios'].ScenarioMission=FakeScenario
worker_checks=[]
with tempfile.TemporaryDirectory(prefix='same-level-clean-review-') as temp, patch.dict(sys.modules,mods):
    for label in ('event','clean'):
        path=PAIR/label/'devext-OOD-L3-00.json'; out=Path(temp)/label; out.mkdir()
        ws['worker'](0,False,False,0,queue.Queue(),queue.Queue(),out,False,None,path,sha(path))
        result=json.loads((out/'world.json').read_text())
        expected=p.load_development_case(json.loads(path.read_text()))
        assert captured[-1]==expected and result['status']=='completed' and result['physics_steps_executed']==0
        worker_checks.append({'label':label,'loaded_type':type(captured[-1]).__name__,
            'case_digest':captured[-1].digest(),'status':result['status'],'fake_physics_steps':0,
            'world_events_metadata':result['events']})

pair_binding=json.loads((PAIR/'binding.json').read_text())
assert pair_binding['parent_file_sha256']==sha(parent_path)
assert pair_binding['parent_case_sha256']==parent.digest()
assert pair_binding['clean_case_sha256']==clean.digest()
pair_inputs={}
for label,expected in [('event',parent),('clean',clean)]:
    path=PAIR/label/'devext-OOD-L3-00.json'; manifest=json.loads((PAIR/label/'manifest.json').read_text())
    row=manifest['records'][0]
    assert sha(path)==row['file_sha256'] and expected.digest()==row['case_sha256']
    assert p.load_development_case(json.loads(path.read_text()))==expected
    pair_inputs[label]={'file_sha256':sha(path),'case_sha256':expected.digest()}
checks=json.loads((V2/'checks-binding.json').read_text())
assert sha(V2/'tests.v2.log')==checks['checks']['108-tests']['sha256']
assert sha(PAIR/'binding.json')==checks['checks']['initial-public-card-pair']['sha256']
tree=ast.parse((V2/'source/eval/inspection_common_calendar.py').read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='rescore_common_calendar')
cs={'validate_development_case':p.validate_development_case,'load_development_case':p.load_development_case,
    '_score':lambda world,geometry,demand: world['score']}
exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen-common-calendar','exec'),cs)
calendar_result=cs['rescore_common_calendar']({'private_case':clean.as_dict(),'case_sha256':clean.digest(),
    'score':{'fixture':'not a physical score'}},parent)
assert calendar_result['geometry_case_sha256']==clean.digest() and calendar_result['demand_case_sha256']==parent.digest()
result={'role':'independent CPU/fake backend only; no PyFlyt/model imports or actual physics; not qualification',
    'v1_roundtrip_failure_reproduced':old_failure,'v2_roundtrip_equal':True,
    'base_field_differences':differences,'rejected_mutations':rejections,
    'fake_adapter_initial_card_seed_wind_proxies_equal':True,'fake_adapter_event_boundaries':True,
    'worker_checks':worker_checks,'common_calendar_schema_entry_only':True,'pair_inputs':pair_inputs,'sources':source_checks,
    'source_binding_sha256':sha(V2/'source-binding.json'),'v1_source_binding_sha256':sha(V1/'source-binding.json'),
    'external_test_log_sha256':sha(V2/'tests.v2.log'),'passed':True}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='sources'},indent=2))
