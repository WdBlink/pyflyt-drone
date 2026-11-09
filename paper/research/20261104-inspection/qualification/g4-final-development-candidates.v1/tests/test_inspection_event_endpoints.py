"""Hand-computed original-window endpoints; no model, mission or new physics."""

from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest

from envs.inspection.scoring import GradingMeasurement, grade_dynamic_bins
from eval.inspection_cases import CaseEntity
from eval.inspection_event_endpoints import (
    dynamic_recovery, endpoint, packet_response, static_recovery, static_revision_response, survey_region,
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


def test_same_clock_bad_then_good_report_cannot_erase_the_bad_holding_version():
    entity=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
    versions=[static_version(1,10),static_version(2,20,False),static_version(3,20)]
    assert static_recovery(versions,entity,5)['time']==20


def test_same_event_clock_new_version_is_postevent_but_eligibility_is_pre_event():
    entity=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
    versions=[static_version(1,2,False),static_version(2,10)]
    report=SimpleNamespace(created_at=1,versions=versions)
    assert static_revision_response(report,entity,10)==10
    assert static_recovery(versions,entity,10)['elapsed']==0
    # Recovery allows a new report ID; the revision response exception needs
    # an ID already present strictly before the event.
    fresh=SimpleNamespace(created_at=10,versions=[static_version(1,10)])
    assert static_revision_response(fresh,entity,10) is None
    assert static_recovery(fresh.versions,entity,10)['elapsed']==0


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


def test_static_bad_then_restore_baseline_is_not_new_business_quality():
    entity=CaseEntity('s','static','road_obstruction',(0.,0.),rule_id='static-0')
    baseline=replace(static_version(1,2),estimate_xy=(6.,0.))
    restored=replace(static_version(3,12),estimate_xy=(6.,0.))
    report=SimpleNamespace(created_at=1,versions=[baseline,static_version(2,11,False),restored])
    assert static_revision_response(report,entity,10) is None
    report.versions.append(replace(static_version(4,13),estimate_xy=(4.,0.)))
    assert static_revision_response(report,entity,10)==13


@pytest.mark.parametrize('y,region',[(-399,0),(-200,0),(-199,1),(0,1),(1,2),(200,2),(201,3),(399,3)])
def test_fixed_survey_scope_is_public_geometry_and_not_detection_or_fov(y,region):
    entity=CaseEntity('s','static','road_obstruction',(0.,float(y)),rule_id='static-0')
    assert survey_region(entity,200)==f'stripe-{region}'


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


def test_new_card_job_at_event_clock_is_postevent_but_measurement_must_be_later():
    case,entity,world=packet_fixture();world['jobs'][0]['created_at']=10
    assert packet_response(world,case,entity,10,600)==(12.,'e','j')
    world['delivered_observations'][0]['measured_at']=10
    assert packet_response(world,case,entity,10,600) is None


def test_correct_card_is_checked_at_job_creation_not_a_later_priority_change():
    case,entity,world=packet_fixture();case.priority_at=12
    world['jobs'][0]['params']['task_card_version']=1
    world['delivered_observations'][0]['measured_at']=13
    assert packet_response(world,case,entity,10,600)==(13,'e','j')


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


def eligibility_fixture(monkeypatch, kind='survey', arrival=None):
    # Isolate eligibility only; source/score admission has separate real traces.
    import eval.inspection_event_endpoints as module
    from eval.inspection_cases import generate_dev_case
    case=generate_dev_case('dev-eligibility-fixture','L3',2026110604)
    entities=(CaseEntity('a','static','road_obstruction',(0.,-300.),rule_id='static-0'),
        CaseEntity('b','static','bank_anomaly',(0.,300.),rule_id='static-1'))
    case=replace(case,entities=entities,priority_at=160.,link_drone=0,link_start=200.,link_end=230.,
        camera_drone=1,camera_start=300.,camera_end=335.,retire_drone=2,retire_at=370.)
    world={'private_case':case.as_dict(),'case_sha256':case.digest(),'score':{'dynamic_entities':[]},
        'reports':[],'sensor_audit':[],'delivered_observations':[],
        'jobs':[{'job_id':'j','drone_id':0,'kind':kind,'created_at':190.,'started_at':190.5,'finished_at':210.,
            'params':{'region_id':'stripe-0','candidate_id':'c','rule_id':'static-0','task_card_version':2}}]}
    if arrival is not None:
        world['delivered_observations']=[{'evidence_id':'e','candidate_id':'c','drone_id':0,'measured_at':195.,
            'arrived_at':arrival,'quality':.9,'sensor_mode':'inspect'}]
        world['sensor_audit']=[{'evidence_id':'e','entity_key':'a'}]
    monkeypatch.setattr(module,'validate_development_case',lambda _:None)
    monkeypatch.setattr(module,'_score',lambda *_:world['score'])
    monkeypatch.setattr(module,'reconstruct_reports',lambda *_:SimpleNamespace(reports={}))
    return module,world,case


def test_unseen_in_region_target_stays_eligible_while_other_region_is_unaffected(monkeypatch):
    module,world,case=eligibility_fixture(monkeypatch)
    rows=[r for r in module.analyze_event_trace(world,case)['rows'] if r['event_type']=='link_start']
    assert rows[0]['entity_key']=='a' and rows[0]['eligible'] and rows[0]['response']['censored']
    assert rows[1]['entity_key']=='b' and not rows[1]['eligible'] and rows[1]['NA_reason']=='no_affected_active_workload'


@pytest.mark.parametrize('arrival,eligible',[(199.5,True),(200.,False)])
def test_inspect_eligibility_cannot_use_an_event_clock_new_identity_association(monkeypatch,arrival,eligible):
    module,world,case=eligibility_fixture(monkeypatch,'inspect',arrival)
    row=next(r for r in module.analyze_event_trace(world,case)['rows'] if r['event_type']=='link_start' and r['entity_key']=='a')
    assert row['eligible'] is eligible
