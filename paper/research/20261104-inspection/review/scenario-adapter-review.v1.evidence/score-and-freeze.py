"""Zero-physics doubles for grade inheritance and input freeze boundary."""
from dataclasses import replace
import hashlib,json,queue
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from envs.inspection.protocol import Evidence,Observation,ReportLedger
from envs.inspection.scenario import CLASSES,public_card
from eval.inspection_cases import CaseSpec,coverage_cells,weight_at
from eval.inspection_scenarios import ScenarioMission
D=Path('paper/research/20261104-inspection/qualification/dev-cases.v1');out={'role':'synthetic_semantic_doubles_no_actual_physics'}
for level in ['L1','L2','L3']:
 case=CaseSpec.from_dict(json.loads((D/f'dev-{level}-00.json').read_text()));m=ScenarioMission.__new__(ScenarioMission);m.case=case;m.entities=list(case.entities);m.events=False;m.cells=coverage_cells(case.roi_polygon);m.covered=set(m.cells);m.flight=SimpleNamespace(violations=[],exit_dwell=np.array([20.]*4));m.distance_m=np.array([10000.]*4);m.card=public_card();m.ledger=ReportLedger(10,set(CLASSES),m.truth_at)
 targets=[e for e in case.entities if e.rule_id]
 for i,e in enumerate(targets):
  ts=[1.] if e.object_type=='static' else [e.appeared_at+3*k for k in range(1,61)]
  for k,t in enumerate(ts):
   xy=e.position(t);o=Observation(f'{i}-{k}',f'candidate-{i}',0,t,t+.5,xy,e.true_class,e.object_type,1.,(1.,1.),'inspect',(0.,0.));m.ledger.register(Evidence(o,e.key));assert m.ledger.submit(m.report_request(o,f'r-{i}'),t+.5)['status']=='accepted'
 s=m.grade();static=[e for e in targets if e.object_type=='static'];dynamic=[e for e in targets if e.object_type=='dynamic'];expected_s=sum(weight_at(case,e,600) for e in static);expected_d=sum(weight_at(case,e,e.appeared_at+3*k) for e in dynamic for k in range(60))
 assert s['static_denominator_weight']==expected_s and s['dynamic_denominator_weight']==expected_d
 assert s['C']==s['S']==s['Q']==s['Y']==s['SR']==1 and s['D']==(None if level=='L1' else 1)
 out[level]={'all_perfect':{k:s[k] for k in ['C','S','D','Q','H','E','Y','SR','static_denominator_weight','dynamic_denominator_weight']},'static_entities':len(static),'dynamic_entities':len(dynamic)}
 if level=='L1':
  idx=next(i for i,e in enumerate(targets) if e.object_type=='static');m.ledger.withdraw(f'r-{idx}');partial=m.grade();assert partial['static_denominator_weight']==expected_s and abs(partial['S']-(len(static)-1)/len(static))<1e-12;out[level]['withdraw_keeps_denominator']={'S':partial['S'],'denominator':partial['static_denominator_weight']}
  m.flight.violations=[{'hard':True}];assert m.grade()['Y']==0;m.flight.violations=[];m.flight.exit_dwell[0]=0;assert m.grade()['E']==0;m.flight.exit_dwell[0]=20;m.distance_m[0]=17500;assert m.grade()['E']==0
# Exercise the real worker deserialization while replacing all flight objects.
import envs.inspection.mission as mm
import eval.inspection_scenarios as sm
import eval.run_inspection_reference as rr
class FakeMission:
 def __init__(self,*a,**kw):
  self.t=0.;self.executor=SimpleNamespace(infrastructure_errors=[]);self.flight=SimpleNamespace(world=SimpleNamespace(physics_steps=0,physics_hz=240,step_period=1/120),summary=lambda:{})
 def step(self):self.t+=1/120;self.flight.world.physics_steps+=2
 def public_state(self,**kw):return {'sim_time':self.t}
 def audit(self):return {'score':{},'case_sha256':self.case.digest(),'private_case':self.case.as_dict()}
 def close(self):pass
class FakeScenario(FakeMission):
 def __init__(self,case):super().__init__();self.case=case
p=Path('/tmp/scenario-adapter-review/mutated-copy');p.mkdir(exist_ok=True);f=p/'private-case.json';original=(D/'dev-L1-00.json').read_bytes();f.write_bytes(original);expected=hashlib.sha256(original).hexdigest();c=json.loads(original);c['seed']+=1;f.write_text(json.dumps(c));assert hashlib.sha256(f.read_bytes()).hexdigest()!=expected
old_m,old_s=mm.InspectionMission,sm.ScenarioMission
try:
 mm.InspectionMission,sm.ScenarioMission=FakeMission,FakeScenario
 rr.worker(0,True,False,1/120,queue.SimpleQueue(),queue.SimpleQueue(),p,True,None,f)
finally:mm.InspectionMission,sm.ScenarioMission=old_m,old_s
w=json.loads((p/'world.json').read_text());out['freeze_mutation']={'expected_copied_file_sha':expected,'actual_worker_file_sha':hashlib.sha256(f.read_bytes()).hexdigest(),'worker_status':w['status'],'worker_case_digest':w['case_sha256'],'original_case_digest':CaseSpec.from_dict(json.loads(original)).digest(),'no_actual_physics':True,'scope':'worker file read with flight double; not an actual CLI run or scientific sample'}
assert w['status']=='completed' and w['case_sha256']!=out['freeze_mutation']['original_case_digest']
Path('/tmp/scenario-adapter-review/score-and-freeze.json').write_text(json.dumps(out,indent=2,allow_nan=False));print(json.dumps(out,indent=2))
