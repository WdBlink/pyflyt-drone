"""Independent hand fixtures, no scientific draws/model/physics."""
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace
from unittest.mock import patch
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
BASE=Path('/tmp/inspection-event-scope-pair-review-v1')
s=importlib.util.spec_from_file_location('scope6ac',BASE/'source/inspection_event_endpoints.py');ep=importlib.util.module_from_spec(s);s.loader.exec_module(ep)
from envs.inspection.scoring import GradingMeasurement
from eval.inspection_cases import CaseEntity
regions=[]
for y,expected in [(-200-1e-7,0),(-200,0),(-200+1e-7,1),(-1e-7,1),(0,1),(1e-7,2),(200-1e-7,2),(200,2),(200+1e-7,3)]:
    e=CaseEntity('s','static','road_obstruction',(0.,y),rule_id='static-0');got=ep.survey_region(e,200)
    regions.append({'y':y,'expected':expected,'actual':got,'passed':got==f'stripe-{expected}'})
entities=(CaseEntity('a','static','road_obstruction',(0.,-300.),rule_id='static-0'),CaseEntity('b','static','bank_anomaly',(0.,300.),rule_id='static-1'))
case=SimpleNamespace(case_id='fixture',lineage_group='fixture',level='L3',entities=entities,priority_at=160.,link_drone=0,link_start=200.,link_end=230.,camera_drone=1,camera_start=300.,camera_end=335.,retire_drone=2,retire_at=370.)
case.as_dict=lambda:{'fixture':'scope eligibility'};case.digest=lambda:'f'*64
def eligibility(kind,arrival=None):
    world={'private_case':case.as_dict(),'case_sha256':case.digest(),'score':{},'reports':[],
        'sensor_audit':[],'delivered_observations':[],
        'jobs':[{'job_id':'j','drone_id':0,'kind':kind,'created_at':190.,'started_at':190.5,'finished_at':210.,
            'params':{'region_id':'stripe-0','candidate_id':'c','rule_id':'static-0','task_card_version':2}}]}
    if arrival is not None:
        world['sensor_audit']=[{'evidence_id':'e','entity_key':'a'}]
        world['delivered_observations']=[{'evidence_id':'e','candidate_id':'c','drone_id':0,'measured_at':195.,'arrived_at':arrival,'quality':.9,'sensor_mode':'inspect'}]
    with patch.object(ep,'validate_development_case',lambda x:None),patch.object(ep,'_score',lambda *x:{}),patch.object(ep,'reconstruct_reports',lambda *x:SimpleNamespace(reports={})):
        return [r for r in ep.analyze_event_trace(world,case)['rows'] if r['event_type']=='link_start']
survey=eligibility('survey');inspects={str(t):eligibility('inspect',t)[0]['eligible'] for t in (199.5,200.,200.5)}
entity=entities[0]
def v(i,t,correct=True,x=0):return GradingMeasurement(str(i),t-.5,t,(float(x),-300.),'road_obstruction' if correct else 'bank_anomaly',True,True,True,i)
versions=[v(1,2,x=6),v(2,11,False),v(3,12,x=6)]
restored=ep.static_revision_response(SimpleNamespace(created_at=1,versions=versions),entity,10)
improved=ep.static_revision_response(SimpleNamespace(created_at=1,versions=versions+[v(4,13,x=4)]),entity,10)
case2=SimpleNamespace(priority_at=20,retire_drone=3,retire_at=100,link_drone=1,link_start=50,link_end=60)
obs={'evidence_id':'e','candidate_id':'c','drone_id':0,'measured_at':22.,'arrived_at':22.5,'quality':.9,'sensor_mode':'inspect'}
job={'job_id':'j','kind':'inspect','drone_id':0,'created_at':12.,'started_at':12.5,'finished_at':25.,'params':{'candidate_id':'c','rule_id':'static-0','task_card_version':1}}
w={'sensor_audit':[{'evidence_id':'e','entity_key':'a'}],'delivered_observations':[obs],'jobs':[job]}
old_card=ep.packet_response(w,case2,entity,10,600)
checks={'region_boundaries':all(x['passed'] for x in regions),
    'unseen_inregion_eligible_and_missing_packet_censored':survey[0]['eligible'] and survey[0]['response']['censored'],
    'other_region_NA':not survey[1]['eligible'] and survey[1]['NA_reason']=='no_affected_active_workload',
    'inspect_only_preevent_association':inspects=={'199.5':True,'200.0':False,'200.5':False},
    'bad_restore_baseline_not_response':restored is None,'beyond_baseline_response':improved==13,
    'legal_created_before_later_priority_response':old_card==(22.,'e','j'),
    'same_time_bad_good_hold':ep.static_recovery([v(1,10),v(2,20,False),v(3,20)],entity,5)['time']==20,
    'sameclock_new_revision_response':ep.static_revision_response(SimpleNamespace(created_at=1,versions=[v(1,2,False),v(2,10)]),entity,10)==10}
result={'role':'pure hand fixtures for frozen dev scope, no physics/model/score','regions':regions,'survey_rows':survey,'inspect_association':inspects,
    'checks':checks,'passed':all(checks.values())}
(BASE/'scope-probe.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'checks':checks},indent=2))
