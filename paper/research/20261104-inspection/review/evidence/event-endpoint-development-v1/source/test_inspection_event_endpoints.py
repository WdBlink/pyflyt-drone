"""Hand-computed original-window endpoints; no model, mission or new physics."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from envs.inspection.scoring import GradingMeasurement, grade_dynamic_bins
from eval.inspection_cases import CaseEntity
from eval.inspection_event_endpoints import (
    dynamic_recovery, endpoint, packet_response, static_recovery, static_revision_response,
)


def static_version(i, stamp, correct=True):
    return GradingMeasurement(str(i), max(0, stamp-.5), stamp, (0.,0.),
        'road_obstruction' if correct else 'bank_anomaly', True, True, True, i)


def test_static_recovery_requires_the_full_next_thirty_seconds_with_no_quality_regression():
    entity=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
    versions=[static_version(i,t,correct) for i,(t,correct) in enumerate(
        [(10,True),(20,False),(30,True),(55,False),(60,True)],1)]
    value=static_recovery(versions,entity,5)
    assert value=={'time':60,'elapsed':55,'censored':False,'followup_horizon':595}
    late=static_recovery([static_version(1,590)],entity,580)
    assert late=={'time':600,'elapsed':20,'censored':True,'followup_horizon':20}


def dynamic_fixture():
    entity=CaseEntity('d','dynamic','inspection_vehicle',(0.,0.),100,1.2,'dynamic-inspection')
    versions=[GradingMeasurement(f'v{i}',100+3*i-.5,100+3*i+.5,entity.position(100+3*i-.5),
        entity.true_class,True,True,True,i) for i in range(1,61)]
    _,bins=grade_dynamic_bins(versions,entity.position,entity.true_class,100,600)
    return versions,[asdict(b) for b in bins]


def test_dynamic_recovery_uses_ten_original_bins_and_actual_upload_not_reset_wallclock():
    versions,bins=dynamic_fixture()
    value=dynamic_recovery(bins,versions,101,286)
    assert value=={'time':133.5,'elapsed':32.5,'censored':False,'followup_horizon':185}
    short=dynamic_recovery(bins,versions,260,286)
    assert short=={'time':286,'elapsed':26,'censored':True,'followup_horizon':26}


@pytest.mark.parametrize('fault',['nine_bins','nan','reset_grid','missing_measurement'])
def test_changed_or_incomplete_original_dynamic_grid_is_not_accepted(fault):
    versions,bins=dynamic_fixture()
    if fault=='nine_bins':bins=bins[:9]
    elif fault=='nan':bins[0]['quality']=float('nan')
    elif fault=='reset_grid':bins[5]['starts_at']+=.1
    else:versions=[]
    with pytest.raises(ValueError):dynamic_recovery(bins,versions,100,286)


def test_static_response_is_an_existing_id_positive_improvement_not_an_ack_or_repetition():
    entity=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
    report=SimpleNamespace(created_at=1,versions=[static_version(1,2,False),static_version(2,12)])
    assert static_revision_response(report,entity,10)==12
    assert static_revision_response(SimpleNamespace(created_at=11,versions=report.versions),entity,10) is None
    assert static_revision_response(SimpleNamespace(created_at=1,versions=[static_version(1,2),static_version(2,12)]),entity,10) is None


def packet_fixture():
    case=SimpleNamespace(priority_at=10,retire_drone=3,retire_at=100,link_drone=1,link_start=20,link_end=30)
    entity=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
    obs={'evidence_id':'e','candidate_id':'c','drone_id':0,'measured_at':12.,'arrived_at':12.5,'quality':.9,'sensor_mode':'inspect'}
    job={'job_id':'j','kind':'inspect','drone_id':0,'created_at':11,'started_at':11.5,'finished_at':15,
        'params':{'candidate_id':'c','rule_id':'static-0','task_card_version':2}}
    return case,entity,{'sensor_audit':[{'evidence_id':'e','entity_key':'s'}],'delivered_observations':[obs],'jobs':[job]}


def test_response_is_measured_packet_after_a_new_correct_job_even_when_local_job_failed():
    case,entity,world=packet_fixture();world['jobs'][0]['status']='failed'
    assert packet_response(world,case,entity,10,600)==(12.,'e','j')
    world['jobs'][0]['finished_at']=11.75
    assert packet_response(world,case,entity,10,600) is None


@pytest.mark.parametrize('fault',['old_job','old_card','wrong_rule','pre_event_measurement','low_quality','offline','retired','wrong_candidate'])
def test_ack_old_measurement_and_ineligible_aircraft_do_not_count_as_response(fault):
    case,entity,world=packet_fixture();job=world['jobs'][0];obs=world['delivered_observations'][0]
    if fault=='old_job':job['created_at']=9
    elif fault=='old_card':job['params']['task_card_version']=1
    elif fault=='wrong_rule':job['params']['rule_id']='static-1'
    elif fault=='pre_event_measurement':obs['measured_at']=10
    elif fault=='low_quality':obs['quality']=.79
    elif fault=='offline':obs['drone_id']=job['drone_id']=1;obs['measured_at']=22;job['finished_at']=25
    elif fault=='retired':obs['drone_id']=job['drone_id']=3;obs['measured_at']=101;job['finished_at']=105
    else:obs['candidate_id']='unrelated'
    assert packet_response(world,case,entity,10,600) is None


def test_each_censor_keeps_its_target_deadline_and_nonnegative_horizon():
    assert endpoint(270,286,None)['followup_horizon']==16
    assert endpoint(270,600,None)['followup_horizon']==330
    with pytest.raises(ValueError):endpoint(300,286,None)
