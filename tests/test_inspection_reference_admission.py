"""Counterexamples for complete dependencies and a joined physical task chain."""

from copy import deepcopy
import hashlib

import pytest

import eval.qualify_inspection_reference as admission


def chain_fixture():
    def view(identity, measured, x):
        return dict(evidence_id=identity,candidate_id="c",drone_id=0,quality=.95,
                    sensor_mode="inspect",measured_at=measured,arrived_at=measured+.5,aircraft_xy=[x,0.])
    return dict(simulated_seconds=600.,delivered_observations=[view("a",1.5,0.),view("b",2.5,25.),view("local",4.5,50.)],
        jobs=[dict(job_id="inspect",drone_id=0,kind="inspect",status="succeeded",created_at=1.,finished_at=3.,
                   params=dict(candidate_id="c")),
              dict(job_id="observe",drone_id=0,kind="observe",status="running",created_at=4.,
                   params=dict(candidate_id="c",report_id="r",service_started_at=4.5))],
        reports=[dict(t=3.5,request=dict(candidate_id="c",report_id="r",object_type="dynamic",
                      measured_at=2.5,evidence_ids=["b"]),response=dict(status="accepted"))])


def test_complete_dependencies_cannot_be_replaced_by_a_matching_subset(tmp_path,monkeypatch):
    paths=[tmp_path/"a",tmp_path/"b"]
    for path in paths:path.write_text(path.name)
    monkeypatch.setattr(admission,"expected_dependency_paths",lambda:{str(p) for p in paths})
    recorded={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    assert admission.verify_dependencies(dict(dependencies_unchanged=True,dependency_hashes=recorded))
    assert not admission.verify_dependencies(dict(dependencies_unchanged=True,dependency_hashes=dict(list(recorded.items())[:1])))
    paths[1].write_text("changed")
    assert not admission.verify_dependencies(dict(dependencies_unchanged=True,dependency_hashes=recorded))


def test_causal_task_chain_accepts_the_joined_evidence():
    assert admission.causal_chain_witness(chain_fixture())==dict(candidate_id="c",inspect_job="inspect",
        report_id="r",observe_job="observe",first_local_evidence="local")


@pytest.mark.parametrize("fault",["candidate","report","local_drone","future_report","insufficient_baseline",
                                 "future_measurement","unbound_measurement","pre_inspection_survey"])
def test_unrelated_or_acausal_jobs_do_not_prove_the_reference_chain(fault):
    world=deepcopy(chain_fixture())
    if fault=="candidate":world['jobs'][1]['params']['candidate_id']='unrelated'
    elif fault=="report":world['jobs'][1]['params']['report_id']='unrelated'
    elif fault=="local_drone":world['delivered_observations'][-1]['drone_id']=1
    elif fault=="future_report":world['reports'][0]['t']=5.
    elif fault=="insufficient_baseline":world['delivered_observations'][1]['aircraft_xy']=[5.,0.]
    elif fault=="future_measurement":world['reports'][0]['request']['measured_at']=999.
    elif fault=="unbound_measurement":world['reports'][0]['request']['measured_at']=3.
    else:
        old=deepcopy(world['delivered_observations'][0])
        old.update(evidence_id='old',sensor_mode='survey',measured_at=.5,arrived_at=1.)
        world['delivered_observations'].append(old)
        world['reports'][0]['request'].update(evidence_ids=['old'],measured_at=.5)
    assert admission.causal_chain_witness(world) is None
