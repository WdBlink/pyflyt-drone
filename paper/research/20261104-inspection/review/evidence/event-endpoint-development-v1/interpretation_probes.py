"""Expose two current operational choices; not a substitute for contract freeze."""
from pathlib import Path
import importlib.util
import json
import sys
from types import SimpleNamespace
REPO=Path('/Users/wdblink/.codex/worktrees/538e/pyflyt-drone');sys.path.insert(0,str(REPO))
BASE=Path('/tmp/inspection-event-endpoint-review-v1')
s=importlib.util.spec_from_file_location('ep924',BASE/'source-repaired/inspection_event_endpoints.py');ep=importlib.util.module_from_spec(s);s.loader.exec_module(ep)
from envs.inspection.scoring import GradingMeasurement
from eval.inspection_cases import CaseEntity
entity=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
def v(i,t,correct):return GradingMeasurement(str(i),t-.5,t,(6.,0.),'road_obstruction' if correct else 'bank_anomaly',True,True,True,i)
versions=[v(1,2,True),v(2,11,False),v(3,12,True)]
revision={'pre_event_quality':ep.final_static_quality(versions[:1],entity.origin_xy,entity.true_class,10),
    'revised_quality':ep.final_static_quality(versions,entity.origin_xy,entity.true_class,12),
    'response_under_current_previous_prefix_baseline':ep.static_revision_response(SimpleNamespace(created_at=1,versions=versions),entity,10)}
case=SimpleNamespace(priority_at=20,retire_drone=3,retire_at=100,link_drone=1,link_start=50,link_end=60)
obs={'evidence_id':'e','candidate_id':'c','drone_id':0,'measured_at':22.,'arrived_at':22.5,'quality':.9,'sensor_mode':'inspect'}
job={'job_id':'j','kind':'inspect','drone_id':0,'created_at':12.,'started_at':12.5,'finished_at':25.,'params':{'candidate_id':'c','rule_id':'static-0','task_card_version':1}}
world={'sensor_audit':[{'evidence_id':'e','entity_key':'s'}],'delivered_observations':[obs],'jobs':[job]}
data={'role':'pure CPU witnesses for unresolved interpretation, not actual trace or approval',
    'return_to_event_baseline':revision,
    'accepted_job_before_later_priority':{'event_time':10,'job_created':12,'card_legal_at_creation':1,'later_priority':20,'new_measurement':22,
        'response_under_current_measurement_card_anchor':ep.packet_response(world,case,entity,10,600)}}
(BASE/'interpretation-probes.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data,indent=2))
