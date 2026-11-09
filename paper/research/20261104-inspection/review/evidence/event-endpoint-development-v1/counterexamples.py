"""Hand-computed development endpoint counterexamples, zero physics/model."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
BASE=Path('/tmp/inspection-event-endpoint-review-v1')
s=importlib.util.spec_from_file_location('frozen_endpoint',BASE/'source/inspection_event_endpoints.py');ep=importlib.util.module_from_spec(s);s.loader.exec_module(ep)
from envs.inspection.scoring import GradingMeasurement
from eval.inspection_cases import CaseEntity
entity=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
def version(i,t,correct):return GradingMeasurement(str(i),t-.5,t,(0.,0.),'road_obstruction' if correct else 'bank_anomaly',True,True,True,i)
versions=[version(1,10,True),version(2,20,False),version(3,20,True)]
static={'actual':ep.static_recovery(versions,entity,5),'expected_first_stable_start':20,
    'reason':'quality0 version at20/order2 breaks10..40 even though20/order3 restores quality'}
case=SimpleNamespace(case_id='fixture',lineage_group='fixture',level='L3',entities=(entity,),priority_at=5.,link_drone=0,link_start=10.,link_end=15.,
    camera_drone=1,camera_start=20.,camera_end=25.,retire_drone=2,retire_at=30.)
case.as_dict=lambda:{'fixture':'eligibility tie only'};case.digest=lambda:'f'*64
obs={'evidence_id':'at-event','candidate_id':'c','drone_id':0,'measured_at':9.,'arrived_at':10.,'quality':.9,'sensor_mode':'inspect'}
job={'job_id':'old-running','drone_id':0,'kind':'inspect','created_at':1.,'started_at':2.,'finished_at':100.,
    'params':{'candidate_id':'c','rule_id':'static-0','task_card_version':1}}
world={'private_case':case.as_dict(),'case_sha256':case.digest(),'score':{'fixture':True},'reports':[],
    'sensor_audit':[{'evidence_id':'at-event','entity_key':'s'}],'delivered_observations':[],'jobs':[job]}
with patch.object(ep,'validate_development_case',lambda c:None),patch.object(ep,'_score',lambda *a:world['score']),\
    patch.object(ep,'reconstruct_reports',lambda *a:SimpleNamespace(reports={})):
    before=next(r for r in ep.analyze_event_trace(world,case)['rows'] if r['event_type']=='link_start')
    world['delivered_observations']=[obs]
    at=next(r for r in ep.analyze_event_trace(world,case)['rows'] if r['event_type']=='link_start')
data={'role':'pure hand fixture, not actual world admission or qualification','same_timestamp_static_recovery':static,
    'association_at_event':{'no_pre_event_observation':before,'only_observation_arrival_equals_event':at,
        'expected_under_claimed_left_limit':'both no_affected_active_workload; equality is not delivery before event'}}
(BASE/'counterexamples.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
