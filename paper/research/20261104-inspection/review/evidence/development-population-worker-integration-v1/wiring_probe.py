"""Execute actual adapter/worker AST against explicit fake parents, 0 physics."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import queue
import sys
import tempfile
import time
import traceback
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
from eval.inspection_population import load_development_case,validate_development_case
from eval.inspection_cases import coverage_cells
BASE=Path('/tmp/inspection-development-population-review-v1');RUN=REPO/'paper/research/20261104-inspection'
fresh=RUN/'qualification/dev-population.v1';manifest=json.loads((fresh/'manifest.json').read_text())

class ProxyWorld:
    GEOM_SPHERE=2
    def __init__(self):self.positions={};self.next=1000;self.registered=0
    def createCollisionShape(self,*a,**k):return 1
    def createMultiBody(self,*a,**k):self.next+=1;return self.next
    def resetBasePositionAndOrientation(self,body,pos,orn):self.positions[body]=pos
    def setCollisionFilterGroupMask(self,*a):pass
    def register_all_new_bodies(self):self.registered+=1
class Parent:
    def __init__(self,seed,**kwargs):
        self.parent_input=(seed,kwargs);self.flight=SimpleNamespace(world=ProxyWorld())
        self.entity_bodies={str(i):100+i for i in range(5)};self.retired=set();self.t=0
        self.card={'task_card_version':1,'rules':[{'rule_id':'static-'+str(i),'object_type':'static','priority':1} for i in range(3)]+
            [{'rule_id':'dynamic-inspection','object_type':'dynamic','priority':1}]}
        self.public_events=[];self.jobs=SimpleNamespace(versions=[0]*4);self.submitted=[]
        self.executor=SimpleNamespace(submit=lambda *a,**k:self.submitted.append((a,k)))
    def _refresh_public(self):pass
source=ast.parse((REPO/'eval/inspection_scenarios.py').read_text())
klass=next(x for x in source.body if isinstance(x,ast.ClassDef) and x.name=='ScenarioMission')
space={'InspectionMission':Parent,'validate_development_case':validate_development_case,'coverage_cells':coverage_cells}
exec(compile(ast.Module(body=[klass],type_ignores=[]),str(REPO/'eval/inspection_scenarios.py'),'exec'),space)
Scenario=space['ScenarioMission'];adapter=[]
for row in manifest['records']:
    case=load_development_case(json.loads((fresh/row['file']).read_text()));s=Scenario(case,start_planner=False)
    p=s.flight.world
    positions=all(p.positions[s.entity_bodies[e.key]]==[*(e.position(0) or (0,0)),.6 if e.position(0) else -10] for e in case.entities)
    basic=s.parent_input==(case.seed,{'events':False,'start_planner':False,'wind_xy':case.wind_xy})
    calendar=True
    if case.level=='L3':
        calendar=(s.online(case.link_drone,case.link_start-1e-6) and not s.online(case.link_drone,case.link_start)
            and s.online(case.link_drone,case.link_end) and s.camera_quality(case.camera_drone,case.camera_start)==.55
            and s.camera_quality(case.camera_drone,case.camera_end)==1)
        s.t=case.retire_at-1e-6;s._events();before=len(s.submitted);s.t=case.retire_at;s._events();s._events()
        calendar=calendar and before==0 and len(s.submitted)==1 and case.retire_drone in s.retired and s.card['task_card_version']==2
    adapter.append({'case_id':case.case_id,'wind_seed_parent_exact':basic,'proxy_positions_at0':positions,
        'grid_exact':s.cells==coverage_cells(case.roi_polygon),'calendar_boundaries':calendar,'proxy_registered_once':p.registered==1})

captured=[]
class FakeMission:
    def __init__(self,*a,**k):
        self.t=0.;self.executor=SimpleNamespace(infrastructure_errors=[])
        self.flight=SimpleNamespace(world=SimpleNamespace(physics_steps=0,physics_hz=240,step_period=1/120),step_period=1/120)
    def step(self):self.t+=1/120
    def close(self):pass
    def public_state(self,**k):return {'fixture':'not an actual state'}
    def audit(self):return {'fixture':True}
    def business_audit(self):return {'fixture':True}
class FakeScenario(FakeMission):
    def __init__(self,case):super().__init__();captured.append(case);self.case=case
worker_tree=ast.parse((REPO/'eval/run_inspection_reference.py').read_text())
worker_node=next(x for x in worker_tree.body if isinstance(x,ast.FunctionDef) and x.name=='worker')
wspace={'Path':Path,'hashlib':hashlib,'json':json,'time':time,'queue':queue,'traceback':traceback}
exec(compile(ast.Module(body=[worker_node],type_ignores=[]),str(REPO/'eval/run_inspection_reference.py'),'exec'),wspace)
mods={}
for name in ('envs.inspection.mission','envs.inspection.protocol','eval.inspection_scenarios'):mods[name]=ModuleType(name)
mods['envs.inspection.mission'].InspectionMission=FakeMission
mods['envs.inspection.protocol'].audit_value=lambda x:x
mods['eval.inspection_scenarios'].ScenarioMission=FakeScenario
worker=[]
with patch.dict(sys.modules,mods):
    for row in manifest['records']:
        path=fresh/row['file'];out=Path(tempfile.mkdtemp(prefix='fake-worker-',dir=BASE));updates=queue.Queue()
        before=len(captured);wspace['worker'](0,False,False,0,queue.Queue(),updates,out,False,None,path,hashlib.sha256(path.read_bytes()).hexdigest())
        result=json.loads((out/'world.json').read_text())
        worker.append({'case_id':row['case_id'],'status':result['status'],'physics_steps':result.get('physics_steps_executed'),
            'loaded_exact_case':len(captured)==before+1 and captured[-1].as_dict()==load_development_case(json.loads(path.read_text())).as_dict()})
    path=fresh/manifest['records'][0]['file'];out=Path(tempfile.mkdtemp(prefix='fake-bad-sha-',dir=BASE));before=len(captured)
    wspace['worker'](0,False,False,0,queue.Queue(),queue.Queue(),out,False,None,path,'0'*64)
    bad_sha_rejected=json.loads((out/'world.json').read_text())['status']=='operational_failure' and len(captured)==before

calendar_tree=ast.parse((REPO/'eval/inspection_common_calendar.py').read_text());node=next(x for x in calendar_tree.body if isinstance(x,ast.FunctionDef) and x.name=='rescore_common_calendar')
cspace={'validate_development_case':validate_development_case,'load_development_case':load_development_case,
    '_score':lambda world,geometry,demand:world['score']}
exec(compile(ast.Module(body=[node],type_ignores=[]),str(REPO/'eval/inspection_common_calendar.py'),'exec'),cspace)
calendar=[]
for row in manifest['records']:
    case=load_development_case(json.loads((fresh/row['file']).read_text()));world={'private_case':case.as_dict(),'case_sha256':case.digest(),'score':{'fixture':'not a score'}}
    result=cspace['rescore_common_calendar'](world,case);calendar.append({'case_id':case.case_id,'schema_entry_accepts':result['geometry_case_sha256']==case.digest()})
data={'role':'actual constructor/method/worker AST against explicit fakes; no PyFlyt imports, no physical step/model generation or score qualification',
    'adapter':adapter,'worker':worker,'bad_private_sha_rejected':bad_sha_rejected,'common_calendar_schema_entry':calendar}
data['passed']=all(all(v for k,v in x.items() if k!='case_id') for x in adapter) and all(x['status']=='completed' and x['physics_steps']==0 and x['loaded_exact_case'] for x in worker) and bad_sha_rejected and all(x['schema_entry_accepts'] for x in calendar)
(BASE/'wiring-probe.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
