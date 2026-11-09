"""Independent positive/negative tool and grading fixtures, not task episodes."""

from dataclasses import FrozenInstanceError, replace
import json
import queue

import numpy as np

import pytest

from envs.inspection.mission import InspectionMission
from envs.inspection.protocol import Evidence, JobBook, Observation, ReportLedger
from envs.inspection.reference import ReferencePolicy
from envs.inspection.scenario import Entity
from envs.inspection.scoring import final_static_quality, grade_dynamic_bins


def observation(eid="e1", candidate="c1", t=1, arrival=2, xy=(0., 0.), cls="risk", kind="static", drone=0):
    return Observation(eid, candidate, drone, t, arrival, xy, cls, kind, .95, (1., 1.), "inspect", (0., 0.))


def request(obs, report_id="r1", **changes):
    data = InspectionMission.report_request(obs, report_id)
    data.update(changes)
    return data


@pytest.fixture
def ledger():
    return ReportLedger(2, {"risk", "normal", "other"}, lambda t: {"a": (0, 0), "b": (100, 0), "normal": (-100, 0)})


def test_correct_and_alternative_static_estimates_are_accepted_without_truth_feedback(ledger):
    obs = observation()
    ledger.register(Evidence(obs, "a"))
    response = ledger.submit(request(obs, estimate_xy=[3, 0]), 2)
    assert response == {"status": "accepted", "report_id": "r1", "slots_used": 1, "slots_remaining": 1}
    assert final_static_quality(ledger.reports["r1"].versions, (0, 0), "risk", 600) == 1
    ledger.submit(request(obs, estimate_xy=[5, 0]), 3)
    assert len(ledger.reports) == 1
    assert final_static_quality(ledger.reports["r1"].versions, (0, 0), "risk", 600) == 2/3


def test_schema_rejection_does_not_consume_capacity_but_unsupported_claim_does(ledger):
    obs = observation()
    assert ledger.submit(request(obs, measured_at=float("nan")), 2)["status"] == "rejected"
    assert len(ledger.reports) == 0
    assert ledger.submit(request(obs), 2)["status"] == "accepted"
    assert len(ledger.reports) == 1
    assert not ledger.reports["r1"].versions[0].provenance_valid


def test_withdrawal_and_duplicate_ids_do_not_refund_capacity(ledger):
    obs = observation()
    ledger.register(Evidence(obs, "a"))
    ledger.submit(request(obs), 2)
    ledger.withdraw("r1")
    ledger.submit(request(obs, "duplicate"), 2)
    assert ledger.reports["duplicate"].associated_key is None
    assert ledger.submit(request(obs, "third"), 3)["status"] == "rejected"
    assert len(ledger.reports) == 2


def test_first_association_cannot_be_retargeted_or_rescued_with_future_truth(ledger):
    a, b = observation(), observation("e2", "c2", xy=(100, 0))
    ledger.register(Evidence(a, "a")); ledger.register(Evidence(b, "b"))
    ledger.submit(request(a), 2)
    ledger.submit(request(b), 3)
    assert ledger.reports["r1"].associated_key == "a"
    assert final_static_quality(ledger.reports["r1"].versions, (0, 0), "risk", 600) == 0
    ledger.submit(request(a, "unmatched", estimate_xy=[500, 0]), 2)
    ledger.submit(request(a, "unmatched"), 3)
    assert ledger.reports["unmatched"].associated_key is None


def test_matching_gate_inclusive_tie_stable_and_class_independent():
    ledger = ReportLedger(2, {"wrong"}, lambda t: {"b": (20, 0), "a": (-20, 0)})
    obs = observation(cls="wrong")
    ledger.register(Evidence(obs, "a"))
    ledger.submit(request(obs), 2)
    assert ledger.reports["r1"].associated_key == "a"
    assert ledger.association_ambiguities == 1


@pytest.mark.parametrize("variant", ["future_measurement", "future_arrival", "wrong_candidate", "wrong_entity", "low_quality"])
def test_structurally_valid_but_invalid_evidence_cannot_score(ledger, variant):
    obs = observation()
    key = "a"
    if variant == "future_measurement":
        obs = replace(obs, measured_at=3, arrived_at=4)
    elif variant == "future_arrival":
        obs = replace(obs, arrived_at=3)
    elif variant == "low_quality":
        obs = replace(obs, quality=.5)
    elif variant == "wrong_entity":
        key = "b"
    ledger.register(Evidence(obs, key))
    req = request(obs, candidate_id="wrong" if variant == "wrong_candidate" else obs.candidate_id)
    assert ledger.submit(req, 2)["status"] == "accepted"
    assert final_static_quality(ledger.reports["r1"].versions, (0, 0), "risk", 600) == 0


def test_dynamic_measurement_is_sealed_and_later_new_points_keep_slot():
    ledger = ReportLedger(2, {"risk", "wrong"}, lambda t: {"moving": (10*t, 0)})
    first = observation(t=1, xy=(10, 0), kind="dynamic")
    ledger.register(Evidence(first, "moving"))
    ledger.submit(request(first), 2)
    assert ledger.submit(request(first, claimed_class="wrong"), 3)["status"] == "rejected"
    second = observation("e2", t=4, arrival=5, xy=(40, 0), kind="dynamic")
    ledger.register(Evidence(second, "moving"))
    ledger.submit(request(second), 5)
    assert len(ledger.reports) == 1
    score, _ = grade_dynamic_bins(ledger.reports["r1"].versions, lambda t: (10*t, 0), "risk", 0, 200)
    assert score == pytest.approx(2/60)


def test_evidence_cannot_be_edited_after_minting(ledger):
    obs = observation()
    ledger.register(Evidence(obs, "a"))
    with pytest.raises(FrozenInstanceError):
        obs.measured_at = 10
    with pytest.raises(ValueError):
        ledger.register(Evidence(replace(obs, estimate_xy=(100, 0)), "a"))


def test_acknowledgement_is_queued_not_success_and_stale_cancel_preserves_job():
    jobs = JobBook(1)
    ack = jobs.create(0, "survey", {}, 0, 0)
    job = jobs.jobs[ack["job_id"]]
    assert job.status == "queued" and job.finished_at is None
    assert jobs.create(0, "inspect", {}, 1, 0)["status"] == "rejected"
    jobs.transition(job, "running", .1)
    assert jobs.cancel(job.job_id, 0, .2)["status"] == "rejected"
    assert job.status == "running"
    assert jobs.cancel(job.job_id, 1, .3)["status"] == "accepted"
    assert job.status == "cancelled" and jobs.versions[0] == 2
    assert jobs.cancel(job.job_id, 2, .4)["status"] == "rejected"


@pytest.fixture
def mission():
    value = InspectionMission(0, reference=True)
    try:
        yield value
    finally:
        value.close()


def test_public_snapshot_excludes_truth_and_is_a_copy(mission):
    snapshot = mission.public_state()
    forbidden = {"seed", "truth", "truth_id", "entity_key", "hidden_entities", "pending", "sensor_audit", "score"}
    def keys(value):
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(v) for v in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(keys(v) for v in value))
        return set()
    assert not keys(snapshot) & forbidden
    snapshot["task_card"]["deadline"] = 1
    assert mission.card["deadline"] == 600
    assert mission.public_state()["observations"] == []
    for name in ["get_truth", "__dict__", "set_seed", "set_retired", "grade"]:
        assert mission.tool(name, {})["status"] == "rejected"


def test_unobserved_candidate_and_stale_card_are_rejected(mission):
    args = {"drone_id": 0, "candidate_id": "truth-000", "expected_version": 0,
            "rule_id": "static-0", "task_card_version": 1}
    assert mission.tool("inspect_candidate", args)["status"] == "rejected"
    args = {"drone_id": 0, "region_id": "stripe-0", "duration": 100, "expected_version": 0,
            "rule_id": "static-0", "task_card_version": 2}
    assert mission.tool("assign_survey", args)["status"] == "rejected"


def test_sensor_trackers_do_not_share_undelivered_cross_aircraft_measurements(mission):
    a = mission._candidate((0, 0), "static", 0, 0)
    b = mission._candidate((0, 0), "static", 0, 1)
    assert a != b
    assert a not in mission.trackers[1]


def test_late_observation_keeps_original_stamp_and_uploads_after_job_end(mission, monkeypatch):
    # Explicit synthetic clock fixture; this is not a physical reference run.
    obs = observation("late", "c", t=211, arrival=211.5, xy=(0, 0), cls="inspection_vehicle", kind="dynamic", drone=1)
    mission.ledger.truth_at = lambda t: {"moving": (0, 0)}
    earlier = replace(obs, evidence_id="earlier", measured_at=200, arrived_at=200.5)
    mission.ledger.register(Evidence(earlier, "moving"))
    mission.ledger.submit(request(earlier), 201)
    mission.pending.append((211.5, Evidence(obs, "moving"), "r1"))
    monkeypatch.setattr(type(mission), "t", property(lambda self: 220.0))
    mission._deliver()
    assert not mission.delivered and len(mission.pending) == 1
    monkeypatch.setattr(type(mission), "t", property(lambda self: 240.0))
    mission._deliver()
    assert mission.delivered[0].measured_at == 211
    assert mission.delivered[0].arrived_at == 240
    assert mission.ledger.reports["r1"].versions[-1].arrived_at == 240
    assert mission.ledger.reports["r1"].versions[-1].measured_at == 211


def test_missing_entities_stay_in_full_denominator_and_exit_ack_is_insufficient(mission):
    result = mission.grade()
    assert len(result["static_entities"]) == 6
    assert sum(len(x["bins"]) for x in result["dynamic_entities"]) == 180
    assert result["S"] == result["D"] == result["Y"] == 0
    ack = mission.tool("enter_safe_exit", {"drone_id": 0, "expected_version": 0})
    assert ack["status"] == "accepted"
    assert mission.grade()["E"] == 0
    assert mission.jobs.jobs[ack["job_id"]].status == "queued"


def test_cancellation_reply_releases_only_its_aircraft_in_reference_policy():
    policy = ReferencePolicy()
    policy.pending = {0, 1}
    policy.pending_cancellations["j"] = 0
    policy.result("cancel_job", {"job_id": "j", "expected_version": 1}, {"status": "accepted"})
    assert policy.pending == {1}


def test_static_revision_reception_order_breaks_same_tick_ties(ledger):
    first = observation("a")
    last = observation("z", cls="wrong")
    ledger.classes.add("wrong")
    ledger.register(Evidence(first, "a")); ledger.register(Evidence(last, "a"))
    ledger.submit(request(first), 2)
    ledger.submit(request(last), 2)
    assert final_static_quality(ledger.reports["r1"].versions, (0, 0), "risk", 600) == 0


def test_survey_evidence_can_score_when_its_actual_quality_is_sufficient(ledger):
    obs = replace(observation(), sensor_mode="survey")
    ledger.register(Evidence(obs, "a"))
    ledger.submit(request(obs), 2)
    assert final_static_quality(ledger.reports["r1"].versions, (0, 0), "risk", 600) == 1


def test_coverage_is_visible_only_after_its_packet_arrives(mission, monkeypatch):
    mission.pending_coverage.append((211.5, 1, 211.0, [next(iter(mission.cells))]))
    monkeypatch.setattr(type(mission), "t", property(lambda self: 220.0))
    mission._deliver()
    assert not mission.covered
    monkeypatch.setattr(type(mission), "t", property(lambda self: 240.0))
    mission._deliver()
    assert len(mission.covered) == 1
    assert mission.coverage_evidence[0]["measured_at"] == 211
    assert mission.coverage_evidence[0]["arrived_at"] == 240


def test_offline_cancellation_cannot_stop_onboard_execution(mission, monkeypatch):
    ack = mission.jobs.create(1, "survey", {"duration": 100}, 0, 200)
    job = mission.jobs.jobs[ack["job_id"]]
    mission.jobs.transition(job, "running", 200)
    monkeypatch.setattr(type(mission), "t", property(lambda self: 220.0))
    response = mission.tool("cancel_job", {"job_id": job.job_id, "expected_version": 1})
    assert response["status"] == "rejected"
    assert job.status == "running"


def test_physical_handoff_ack_waits_for_new_aircraft_qualified_observation(mission):
    # Controlled one-object fixture uses the actual four-aircraft physics,
    # frustum, ray casting, queued tools and measured views. Only ground proxy
    # placement changes; aircraft states are never reset or teleported.
    body = next(iter(mission.entity_bodies.values()))
    for other in mission.entity_bodies.values():
        mission.flight.world.resetBasePositionAndOrientation(other, [0, 0, -10], [0, 0, 0, 1])
    mission.entities = [Entity("handoff", "dynamic", "inspection_vehicle", (-240, 114), 0, 20, "dynamic-inspection")]
    mission.entity_bodies = {"handoff": body}
    while mission.t < 1.0:
        mission.step()
    candidate = next(o.candidate_id for o in mission.delivered if o.drone_id == 2 and o.object_type == "dynamic")
    args = {"drone_id": 2, "candidate_id": candidate, "expected_version": 0,
            "rule_id": "dynamic-inspection", "task_card_version": 1}
    inspect_ack = mission.tool("inspect_candidate", args)
    assert inspect_ack["status"] == "accepted"
    while mission.t < 6 and mission.jobs.jobs[inspect_ack["job_id"]].status != "succeeded":
        mission.step()
    assert mission.jobs.jobs[inspect_ack["job_id"]].status == "succeeded"
    obs = next(o for o in reversed(mission.delivered) if o.candidate_id == candidate and o.sensor_mode == "inspect")
    assert mission.tool("submit_or_revise_report", request(obs))["status"] == "accepted"
    args.update(expected_version=mission.jobs.versions[2], duration=30, report_id="r1")
    source_ack = mission.tool("observe_track", args)
    assert source_ack["status"] == "accepted"
    source = mission.jobs.jobs[source_ack["job_id"]]
    while mission.t < 6 and source.status == "queued":
        mission.step()
    assert source.status == "running"
    handoff = mission.tool("handoff_observation", {"from_drone": 2, "to_drone": 3,
        "candidate_id": candidate, "duration": 30, "report_id": "r1",
        "expected_from_version": mission.jobs.versions[2], "expected_to_version": 0,
        "rule_id": "dynamic-inspection", "task_card_version": 1})
    assert handoff["status"] == "accepted"
    assert source.status == "running"
    destination = mission.jobs.jobs[handoff["job_id"]]
    assert destination.status == "queued"
    while mission.t < 7 and source.status == "running":
        mission.step()
    assert source.status == "cancelled"
    assert destination.params["handoff_effective_at"] > destination.created_at
    assert mission.flight.physics_audits == mission.flight.world.physics_steps


def test_dynamic_seal_cannot_be_bypassed_by_evidence_reordering():
    ledger = ReportLedger(2, {"risk", "wrong"}, lambda t: {"moving": (0, 0)})
    old = observation("e1", t=1, arrival=1.5, kind="dynamic")
    new = observation("e2", t=2, arrival=2.5, kind="dynamic")
    ledger.register(Evidence(old, "moving")); ledger.register(Evidence(new, "moving"))
    bad = request(new, evidence_ids=["e2", "e1"], claimed_class="wrong")
    assert ledger.submit(bad, 3)["status"] == "accepted"
    changed = {**bad, "evidence_ids": ["e1", "e2"], "claimed_class": "risk"}
    assert ledger.submit(changed, 4)["status"] == "rejected"
    assert len(ledger.reports["r1"].versions) == 1
    assert ledger.reports["r1"].versions[0].evidence_id == "e2"


def test_belly_camera_can_observe_a_clear_nadir_proxy(mission):
    position = mission.flight.world.state(0)[3]
    body = next(iter(mission.entity_bodies.values()))
    ground = tuple(position[:2])
    mission.flight.world.resetBasePositionAndOrientation(body, [*ground, .6], [0, 0, 0, 1])
    origin = mission.camera_origin(0)
    assert origin[2] < position[2]-.25
    assert mission._visible(origin, ground, np.array([0., 0., -1.]), np.deg2rad(55), body)


def test_event_is_published_at_the_same_tick_its_weight_becomes_effective(mission, monkeypatch):
    # Boundary fixture isolates event ordering; it does not claim 180 s flight.
    mission.flight.world.physics_steps = 43198
    mission.flight.world.elapsed_time = 179.99166666666667
    def integrate_one_tick(*args,**kwargs):
        mission.flight.world.physics_steps = 43200
        mission.flight.world.elapsed_time = 180.0
    monkeypatch.setattr(mission.flight, "step", integrate_one_tick)
    monkeypatch.setattr(mission, "_sample", lambda: None)
    mission.step()
    assert mission.public_state()["task_card"]["task_card_version"] == 2
    assert mission.card["effective_from"] == 180


def test_sr_requires_component_thresholds_and_exit_requires_range(mission, monkeypatch):
    from envs.inspection import mission as module
    mission.covered = set(mission.cells)
    mission.flight.exit_dwell[:] = 21
    monkeypatch.setattr(module, "final_static_quality", lambda *args: 1.)
    def low_dynamic(*args, **kwargs):
        from envs.inspection.scoring import BinResult
        return .01, [BinResult(i, 3*i, 3*(i+1), 1., None, .01, None) for i in range(60)]
    monkeypatch.setattr(module, "grade_dynamic_bins", low_dynamic)
    score = mission.grade()
    assert score["Y"] > .6
    assert score["SR"] == 0
    assert score["E"] == 1
    mission.distance_m[:] = 17500
    assert mission.grade()["E"] == 0


def test_invalid_numeric_tool_input_can_be_retained_as_strict_json(ledger, mission):
    ledger.submit(request(observation(), measured_at=float("nan")), 2)
    json.dumps(ledger.audit, allow_nan=False)
    mission.tool("assign_survey", {"drone_id": 0, "region_id": "stripe-0", "duration": float("nan"),
                                  "expected_version": 0, "rule_id": "static-0", "task_card_version": 1})
    json.dumps(mission.tool_audit, allow_nan=False)


def test_worker_failure_preserves_business_trace(tmp_path, monkeypatch):
    from eval.run_inspection_reference import worker
    original_init, original_step = InspectionMission.__init__, InspectionMission.step
    instances = []
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        instances.append(self)
    def fail_after_warmup(self):
        if len(instances) == 2 and self.t >= .1:
            raise RuntimeError("injected world failure")
        return original_step(self)
    monkeypatch.setattr(InspectionMission, "__init__", initialize)
    monkeypatch.setattr(InspectionMission, "step", fail_after_warmup)
    commands, updates = queue.Queue(), queue.Queue()
    commands.put({"name": "assign_survey", "args": {"drone_id": 0, "region_id": "stripe-0", "duration": 100,
                 "expected_version": 0, "rule_id": "static-0", "task_card_version": 1}})
    worker(0, True, False, 1, commands, updates, tmp_path)
    result = json.loads((tmp_path/"world.json").read_text())
    assert result["status"] == "operational_failure"
    assert "injected world failure" in result["operational_error"]
    assert len(result["partial_business_audit"]["tool_calls"]) == 1
    assert result["partial_business_audit"]["job_transitions"] == [{"t":0,"job_id":"job-0001","status":"queued"}]
    assert any(row["type"] == "plan_request" for row in result["partial_business_audit"]["execution_audit"])


def test_reference_waits_for_delayed_confirmation_proof_without_reinspection():
    from envs.inspection.scenario import public_card
    policy = ReferencePolicy()
    a = replace(observation("a", cls="inspection_vehicle", kind="dynamic"), sensor_mode="survey", quality=.9)
    b = replace(a, evidence_id="b", measured_at=2, arrived_at=2.5)
    state = {"observations": [a.public(), b.public()], "sim_time": 3.0, "task_card": public_card(),
             "aircraft": [{"drone_id": 0, "job_version": 1, "retired": False, "link_available": True, "camera_quality": 1.0}],
             "jobs": [{"job_id": "j", "drone_id": 0, "kind": "inspect", "status": "succeeded", "params": {"candidate_id": "c1"}}],
             "report_slots_used": 0}
    commands = policy.decide(state)
    assert not any(name == "inspect_candidate" for name, args in commands)
    first = replace(b, evidence_id="c", sensor_mode="inspect", aircraft_xy=(0, 0), quality=.95)
    last = replace(first, evidence_id="d", aircraft_xy=(25, 0), measured_at=3, arrived_at=3.5)
    state.update(observations=[first.public(), last.public()], sim_time=3.5)
    commands = policy.decide(state)
    report = next(args for name, args in commands if name == "submit_or_revise_report")
    assert report["evidence_ids"] == ["d"]


def test_retired_sensor_stops_new_acquisitions_but_preserves_pre_event_upload(mission,monkeypatch):
    # Boundary fixture only: no claim that this test physically flew to360s.
    mission.retired.add(3)
    mission._sample()
    assert not any(r['drone_id']==3 for r in mission.sensor_audit+mission.sensor_opportunities+mission.background_trials)
    assert not any(r[1]==3 for r in mission.pending_coverage)
    old=observation('pre-retirement','old',t=359.5,arrival=360.,drone=3)
    mission.pending.append((360.,Evidence(old,None),None))
    monkeypatch.setattr(type(mission),'t',property(lambda self:360.))
    mission._deliver()
    received=next(o for o in mission.delivered if o.evidence_id=='pre-retirement')
    assert received.measured_at==359.5 and received.arrived_at==360.
