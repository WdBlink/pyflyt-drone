"""Compare bounded endpoint fixes with archived originals, no flight/model."""
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
BASE=Path('/tmp/inspection-event-endpoint-review-v1')
def load(p):
    s=importlib.util.spec_from_file_location('endpoint_repair',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
ep=load(BASE/'source-repaired/inspection_event_endpoints.py')
from envs.inspection.scoring import GradingMeasurement
from eval.inspection_cases import CaseEntity
from eval.inspection_population import load_development_case
e=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
def version(i,t,ok):return GradingMeasurement(str(i),t-.5,t,(0.,0.),'road_obstruction' if ok else 'bank_anomaly',True,True,True,i)
rows={}
rows['same_timestamp_bad_then_good']=ep.static_recovery([version(1,10,True),version(2,20,False),version(3,20,True)],e,5)
old=SimpleNamespace(created_at=1,versions=[version(1,2,False),version(2,10,True)])
rows['same_clock_revision_response']=ep.static_revision_response(old,e,10)
rows['same_clock_recovery']=ep.static_recovery(old.versions,e,10)
rows['old_good_without_new_version']=ep.static_recovery([version(1,2,True)],e,10)
rows['new_id_recovery_allowed']=ep.static_recovery([version(1,10,True)],e,10)
case=SimpleNamespace(priority_at=10,retire_drone=3,retire_at=100,link_drone=1,link_start=20,link_end=30)
obs={'evidence_id':'e','candidate_id':'c','drone_id':0,'measured_at':12.,'arrived_at':12.5,'quality':.9,'sensor_mode':'inspect'}
job={'job_id':'j','kind':'inspect','drone_id':0,'created_at':10.,'started_at':10.5,'finished_at':15.,'params':{'candidate_id':'c','rule_id':'static-0','task_card_version':2}}
world={'sensor_audit':[{'evidence_id':'e','entity_key':'s'}],'delivered_observations':[obs],'jobs':[job]}
rows['same_clock_job_packet']=ep.packet_response(world,case,e,10,600)
job['params']['task_card_version']=1;rows['same_clock_old_card_rejected']=ep.packet_response(world,case,e,10,600)

case=SimpleNamespace(case_id='fixture',lineage_group='fixture',level='L3',entities=(e,),priority_at=5.,link_drone=0,link_start=10.,link_end=15.,camera_drone=1,camera_start=20.,camera_end=25.,retire_drone=2,retire_at=30.)
case.as_dict=lambda:{'fixture':'left limit'};case.digest=lambda:'f'*64
world={'private_case':case.as_dict(),'case_sha256':case.digest(),'score':{},'reports':[],
    'sensor_audit':[{'evidence_id':'at','entity_key':'s'}],
    'delivered_observations':[{'evidence_id':'at','candidate_id':'c','drone_id':0,'measured_at':9.,'arrived_at':10.,'quality':.9,'sensor_mode':'inspect'}],
    'jobs':[{'job_id':'old','kind':'inspect','drone_id':0,'created_at':1.,'started_at':2.,'finished_at':100.,'params':{'candidate_id':'c','rule_id':'static-0','task_card_version':1}}]}
with patch.object(ep,'validate_development_case',lambda c:None),patch.object(ep,'_score',lambda *a:{}),patch.object(ep,'reconstruct_reports',lambda *a:SimpleNamespace(reports={})):
    rows['only_at_event_observation_eligible']=next(r for r in ep.analyze_event_trace(world,case)['rows'] if r['event_type']=='link_start')['eligible']
flight=REPO/'logs/inspection-population-dev-flights.v1/devext-OOD-L3-00-B0-0/attempt-00/flight'
actual=json.loads((flight/'world.json').read_text());case=load_development_case(actual['private_case']);result=ep.analyze_event_trace(actual,case)
old_result=json.loads((BASE/'source/event-endpoints-ood-B0.candidate.v1.json').read_text())
changed=[]
for a,b in zip(result['rows'],old_result['rows']):
    diff=[k for k in a if json.dumps(a[k],sort_keys=True)!=json.dumps(b[k],sort_keys=True)]
    if diff:changed.append({'event_type':a['event_type'],'entity_key':a['entity_key'],'changed_fields':diff})
data={'role':'pure fixtures and offline existing event trace; not a new run/pair/Y','fixtures':rows,
    'actual_rows':len(result['rows']),'actual_eligible':sum(r['eligible'] for r in result['rows']),
    'actual_response_censored':sum(r['response']['censored'] for r in result['rows'] if r['eligible']),
    'actual_recovery_censored':sum(r['recovery']['censored'] for r in result['rows'] if r['eligible']),
    'actual_changed_row_fields_vs_v1':changed}
(BASE/'repaired-probe.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
