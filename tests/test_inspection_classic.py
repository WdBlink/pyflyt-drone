"""Public-only classical coordination examples, not policy performance data."""

from copy import deepcopy

import pytest

from envs.inspection.scenario import public_card
from eval.inspection_classic import EventStatePolicy, UtilityAuctionPolicy


def observation(candidate, drone, stamp, xy, *, kind="static", narrow=False, aircraft_xy=None):
    return dict(evidence_id=f"{candidate}-{drone}-{stamp}", candidate_id=candidate, drone_id=drone,
                measured_at=stamp, arrived_at=stamp+.5, estimate_xy=list(xy),
                estimated_class="inspection_vehicle" if kind == "dynamic" else "road_obstruction",
                object_type=kind, quality=.95, sensor_mode="inspect" if narrow else "survey",
                aircraft_xy=list(aircraft_xy or xy))


def snapshot():
    return dict(sim_time=20., task_card=public_card(), observations=[], jobs=[], events=[], report_slots_used=0,
                aircraft=[dict(drone_id=i, measured_at=20., position=[-200.+400*i, -300.+200*i, 96.],
                    camera_quality=1., retired=i > 1, link_available=True, remaining_range_m=17000.,
                    job_version=0) for i in range(4)], coverage=dict(qualified_cells=0, required_cells=100))


@pytest.mark.parametrize("policy_type", [UtilityAuctionPolicy, EventStatePolicy])
def test_public_only_dispatch_assigns_each_candidate_and_aircraft_once(policy_type):
    state=snapshot()
    for candidate, drone, xy in (("left", 0, (-210., -300.)), ("right", 1, (210., -100.))):
        state["observations"].extend(observation(candidate, drone, t, xy) for t in (19., 19.5))
    actions=policy_type().decide(state)
    inspect=[args for name,args in actions if name == "inspect_candidate"]
    assert len(inspect)==2 and {a["candidate_id"] for a in inspect}=={"left", "right"}
    assert {a["drone_id"] for a in inspect}=={0,1}
    assert all(a["task_card_version"]==1 and a["expected_version"]==0 for a in inspect)
    poisoned=deepcopy(state)
    poisoned.update(seed=99, hidden_entities=[dict(truth="should never be read")], score=dict(Y=1))
    assert policy_type().decide(poisoned)==actions


@pytest.mark.parametrize("policy_type", [UtilityAuctionPolicy, EventStatePolicy])
def test_unhealthy_observer_handoff_uses_public_source_and_receiver_versions(policy_type):
    policy=policy_type()
    state=snapshot()
    state["aircraft"][1].update(camera_quality=.55, job_version=4)
    state["jobs"]=[dict(job_id="source", drone_id=1, kind="observe", version=4, status="running",
        started_at=10., created_at=9., params=dict(candidate_id="moving", rule_id="dynamic-inspection"))]
    state["observations"]=[observation("moving",1,t,(-100.,-100.),kind="dynamic") for t in (19.,19.5)]
    policy.reports["moving"]="report-moving"
    policy.observed.add("moving")
    actions=policy.decide(state)
    name,args=next((name,args) for name,args in actions if name=="handoff_observation")
    assert args["from_drone"]==1 and args["to_drone"]==0
    assert args["expected_from_version"]==4 and args["expected_to_version"]==0
    assert 0 in policy.pending and "drone_id" not in args
    policy.result(name,args,dict(status="accepted",job_id="receiver"))
    assert 0 not in policy.pending


def test_confirmation_deduplicates_same_batch_without_refunding_report_slots():
    policy=UtilityAuctionPolicy()
    state=snapshot()
    state["report_slots_used"]=9
    for candidate,drone in (("a",0),("b",1)):
        state["observations"].extend(observation(candidate,drone,t,(10.,10.),narrow=True,
            aircraft_xy=(x,0.)) for t,x in ((18.,0.),(19.,25.)))
    reports=[(name,args) for name,args in policy.decide(state) if name=="submit_or_revise_report"]
    assert len(reports)==1
    name,args=reports[0]
    policy.result(name,args,dict(status="accepted"))
    assert policy.reports["a"]==policy.reports["b"]


def test_late_packet_does_not_rewind_latest_measurement_or_permanently_poison_failed_candidate():
    policy=EventStatePolicy()
    state=snapshot()
    state["observations"]=[observation("c",0,t,(-200.,-300.)) for t in (19.5,19.)]
    state["jobs"]=[dict(job_id="failed",drone_id=0,kind="inspect",status="failed",params=dict(candidate_id="c"))]
    assert not any(name=="inspect_candidate" for name,args in policy.decide(state))
    assert policy.history["c"][-1]["measured_at"]==19.5 and "c" not in policy.rejected
    policy.pending.clear()
    state["sim_time"]=51.
    for aircraft in state["aircraft"]:aircraft["measured_at"]=51.
    state["observations"]=[observation("c",0,50.,(-200.,-300.))]
    assert any(name=="inspect_candidate" for name,args in policy.decide(state))


@pytest.mark.parametrize("policy_type", [UtilityAuctionPolicy, EventStatePolicy])
def test_accepted_handoff_reserves_candidate_until_receiver_terminal(policy_type):
    policy=policy_type()
    state=snapshot()
    state["aircraft"][1].update(camera_quality=.55,job_version=4)
    source=dict(job_id="source",drone_id=1,kind="observe",version=4,status="running",
                started_at=10.,created_at=9.,params=dict(candidate_id="moving",rule_id="dynamic-inspection"))
    state["jobs"]=[source]
    state["observations"]=[observation("moving",1,t,(-100.,-100.),kind="dynamic") for t in (19.,19.5)]
    policy.reports["moving"]="r"
    name,args=next((name,args) for name,args in policy.decide(state) if name=="handoff_observation")
    policy.result(name,args,dict(status="accepted",job_id="receiver"))
    state["sim_time"]=21.
    state["aircraft"][2].update(retired=False,measured_at=21.,job_version=1)
    for aircraft in state["aircraft"]:aircraft["measured_at"]=21.
    state["jobs"].append(dict(job_id="survey2",drone_id=2,kind="survey",status="running",
        started_at=0.,created_at=0.,params=dict(region_id="stripe-2")))
    # ACK can precede receiver telemetry: the explicit reservation covers it.
    first=policy.decide(state)
    assert not any(name in ("handoff_observation","cancel_job") for name,args in first)
    assert not any(name=="assign_survey" and args['drone_id']==0 for name,args in first)
    receiver=dict(job_id="receiver",drone_id=0,kind="observe",status="queued",created_at=20.,
                  started_at=None,params=dict(candidate_id="moving"))
    state["jobs"].append(receiver)
    state["aircraft"][0]["job_version"]=1
    assert not any(name in ("handoff_observation","cancel_job") for name,args in policy.decide(state))
    receiver.update(status="failed",finished_at=21.)
    state["sim_time"]=22.
    state["aircraft"][0]["link_available"]=False
    for aircraft in state["aircraft"]:aircraft["measured_at"]=22.
    policy.decide(state)
    state["sim_time"]=52.
    state["aircraft"][0]["link_available"]=True
    for aircraft in state["aircraft"]:aircraft["measured_at"]=52.
    state["observations"]=[observation("moving",1,51.,(-100.,-100.),kind="dynamic")]
    actions=policy.decide(state)
    assert "moving" not in policy.handoff_reservations
    assert any(name in ("handoff_observation","cancel_job") for name,args in actions)


@pytest.mark.parametrize("policy_type", [UtilityAuctionPolicy, EventStatePolicy])
def test_class_tie_after_physical_inspection_has_a_finite_reconfirmation_exit(policy_type):
    policy=policy_type()
    state=snapshot()
    views=[observation("tie",0,t,(-200.,-300.),narrow=True,aircraft_xy=(x,0.))
           for t,x in ((18.,0.),(19.,25.))]
    views[1]["estimated_class"]="bank_anomaly"
    state["observations"]=views
    state["jobs"]=[dict(job_id="done",drone_id=0,kind="inspect",status="succeeded",
                        finished_at=19.,params=dict(candidate_id="tie"))]
    assert not any(name=="inspect_candidate" for name,args in policy.decide(state))
    policy.pending.clear()
    state["sim_time"]=23.
    for aircraft in state["aircraft"]:aircraft["measured_at"]=23.
    actions=policy.decide(state)
    assert "tie" not in policy.confirmed
    assert any(name=="inspect_candidate" and args["candidate_id"]=="tie" for name,args in actions)
