"""Literal event-phase witnesses, no flight/model/population sampling."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "eval/inspection_event_endpoints.py"
spec = importlib.util.spec_from_file_location("formal_event_endpoints_candidate", path)
events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events)
from eval.inspection_cases import CaseEntity


def commit_fixture():
    case = SimpleNamespace(priority_at=10.)
    entity = CaseEntity("s", "static", "road_obstruction", (0., 0.), rule_id="static-0")
    job = {"job_id": "j", "version": 1, "kind": "survey", "drone_id": 0, "created_at": 10.,
        "started_at": 11., "finished_at": 30., "status": "succeeded",
        "params": {"region_id": "stripe-1", "duration": 20., "rule_id": "static-0", "task_card_version": 2}}
    call = {"t": 10., "name": "assign_survey", "args": {**job["params"], "drone_id": 0, "expected_version": 0},
        "response": {"status": "accepted", "job_id": "j", "version": 1}}
    audit = [{"type": "plan_request", "serial": 1, "revision": 0, "now": 10., "commit_at": 11.,
        "job_id": "j", "drone": 0, "tokens": [[1, "j"], [0, None], [0, None], [0, None]]},
        {"type": "plan_response", "serial": 1, "received_at": 10.5, "fresh": True, "program_sha256": "a" * 64},
        {"type": "program_committed", "serial": 1, "revision": 1, "at": 11., "drone": 0,
            "job_id": "j", "program_sha256": "a" * 64}]
    world = {"simulated_seconds": 600., "jobs": [job], "tool_calls": [call], "execution_audit": audit,
        "job_transitions": [{"t": 10., "job_id": "j", "status": "queued"},
            {"t": 11., "job_id": "j", "status": "running", "reason": "physical program adopted"},
            {"t": 30., "job_id": "j", "status": "succeeded"}], "sensor_audit": [], "delivered_observations": []}
    row = {"event_time": 10., "deadline": 600.}
    return world, case, entity, row


def test_program_adoption_is_control_tick_inference_with_two_real_step_bounds():
    world, case, entity, row = commit_fixture()
    result = events._adoption_response(world, case, entity, row, {"physics_samples": 144000})
    assert result["status"] == "identified" and result["endpoint"]["elapsed"] == 1.
    witness = result["witness"]
    assert witness["kind"] == "program_commit_control_tick_inference"
    assert witness["first_subsequent_physics_hook"] == pytest.approx(11 + 1 / 240)
    assert witness["control_step_end"] == pytest.approx(11 + 1 / 120)
    assert world["jobs"][0]["status"] == "succeeded"


@pytest.mark.parametrize("field", ["drone", "expected_version", "response_version", "card_version"])
def test_bool_or_float_identity_fields_do_not_alias_native_integer_tool_fields(field):
    world, case, entity, row = commit_fixture()
    call = world["tool_calls"][0]
    if field == "drone":
        call["args"]["drone_id"] = False
    elif field == "expected_version":
        call["args"]["expected_version"] = False
    elif field == "response_version":
        call["response"]["version"] = True
    else:
        call["args"]["task_card_version"] = 2.
    result = events._adoption_response(world, case, entity, row, {"physics_samples": 144000})
    assert result["status"] == "adoption_not_identifiable_from_trace" and result["endpoint"] is None


@pytest.mark.parametrize("part,field,value", [(0, "drone", False), (2, "drone", False),
    (0, "serial", True), (1, "serial", True), (2, "serial", True),
    (0, "revision", False), (2, "revision", True), (0, "token_version", True)])
def test_program_chain_and_tokens_preserve_native_integer_identities(part, field, value):
    world, case, entity, row = commit_fixture()
    if field == "token_version":
        world["execution_audit"][part]["tokens"][0][0] = value
    else:
        world["execution_audit"][part][field] = value
    result = events._adoption_response(world, case, entity, row, {"physics_samples": 144000})
    assert result["status"] == "adoption_not_identifiable_from_trace" and result["endpoint"] is None


@pytest.mark.parametrize("fault,expected", [("ack_only", "right_censored"), ("same_advance_terminal", "right_censored"),
    ("no_subsequent_step", "right_censored"), ("hash_mismatch", "adoption_not_identifiable_from_trace"),
    ("missing_request", "adoption_not_identifiable_from_trace"), ("old_card", "adoption_not_identifiable_from_trace"),
    ("duplicate_commit", "adoption_not_identifiable_from_trace"),
    ("old_job", "right_censored"), ("wrong_region", "right_censored")])
def test_nonadoption_and_unidentifiable_traces_are_distinct(fault, expected):
    world, case, entity, row = commit_fixture()
    if fault == "ack_only":
        world["execution_audit"].pop()
        world["job_transitions"] = world["job_transitions"][:1]
        world["jobs"][0].update(status="queued", started_at=None, finished_at=None)
    elif fault == "same_advance_terminal":
        world["job_transitions"][-1].update(t=11., status="failed")
        world["jobs"][0].update(status="failed", finished_at=11.)
    elif fault == "no_subsequent_step":
        world["execution_audit"][0].update(now=599., commit_at=600.)
        world["execution_audit"][1]["received_at"] = 599.5
        world["execution_audit"][2]["at"] = 600.
        world["job_transitions"][1]["t"] = 600.
        world["job_transitions"].pop()
        world["jobs"][0].update(started_at=600., finished_at=None, status="running")
    elif fault == "hash_mismatch":
        world["execution_audit"][1]["program_sha256"] = "b" * 64
    elif fault == "missing_request":
        world["execution_audit"].pop(0)
    elif fault == "duplicate_commit":
        world["execution_audit"].append(dict(world["execution_audit"][-1]))
    elif fault == "old_card":
        world["jobs"][0]["params"]["task_card_version"] = 1
        world["tool_calls"][0]["args"]["task_card_version"] = 1
    elif fault == "old_job":
        world["jobs"][0]["created_at"] = 9.
    else:
        world["jobs"][0]["params"]["region_id"] = "stripe-3"
    result = events._adoption_response(world, case, entity, row, {"physics_samples": 144000})
    assert result["status"] == expected
    assert (result["endpoint"] is None) == (expected == "adoption_not_identifiable_from_trace")
    if result["endpoint"]:
        assert result["endpoint"]["censored"] and result["endpoint"]["elapsed"] == 590.


def local_fixture():
    world, case, entity, row = commit_fixture()
    job = world["jobs"][0]
    job.update(kind="inspect", created_at=11., started_at=12., finished_at=13.)
    job["params"] = {"candidate_id": "c", "rule_id": "static-0", "task_card_version": 2,
        "object_type": "static", "service_started_at": 12.}
    world["tool_calls"] = [{"t": 11., "name": "inspect_candidate", "args": {"candidate_id": "c", "rule_id": "static-0",
        "task_card_version": 2, "drone_id": 0, "expected_version": 0},
        "response": {"status": "accepted", "job_id": "j", "version": 1}}]
    world["execution_audit"] = []
    world["job_transitions"] = [{"t": 11., "job_id": "j", "status": "queued"},
        {"t": 12., "job_id": "j", "status": "running", "reason": "actual local acquisition on committed flight"},
        {"t": 13., "job_id": "j", "status": "succeeded"}]
    world["sensor_audit"] = [{"evidence_id": "seed", "entity_key": "s", "t": 9., "drone_id": 0, "mode": "survey", "quality": .9},
        {"evidence_id": "local", "entity_key": "s", "t": 12., "drone_id": 0, "mode": "inspect", "quality": .9}]
    world["delivered_observations"] = [{"evidence_id": "seed", "candidate_id": "c", "arrived_at": 9.5, "measured_at": 9.}]
    return world, case, entity, row


def test_local_acquisition_on_existing_continuation_needs_no_new_commit_or_upload():
    world, case, entity, row = local_fixture()
    result = events._adoption_response(world, case, entity, row, {"physics_samples": 144000})
    assert result["status"] == "identified" and result["endpoint"]["time"] == 12.
    assert result["witness"]["kind"] == "local_acquisition_on_existing_continuation"
    assert result["witness"]["evidence_id"] == "local" and world["jobs"][0]["status"] == "succeeded"
    assert "local" not in {o["evidence_id"] for o in world["delivered_observations"]}


@pytest.mark.parametrize("fault", [None, "from_drone", "expected_from_version"])
def test_handoff_forwarded_identity_keeps_native_integer_types(fault):
    world, case, entity, row = local_fixture()
    job = world["jobs"][0]
    job.update(kind="observe", status="running", finished_at=None)
    job["params"].update(duration=30., report_id="r")
    call = world["tool_calls"][0]
    call["name"] = "handoff_observation"
    call["args"] = {"from_drone": 1, "to_drone": 0, "expected_from_version": 3, "expected_to_version": 0,
        "candidate_id": "c", "duration": 30., "report_id": "r", "rule_id": "static-0", "task_card_version": 2}
    if fault:
        call["args"][fault] = False if fault == "from_drone" else 3.
    result = events._adoption_response(world, case, entity, row, {"physics_samples": 144000})
    assert result["status"] == ("identified" if fault is None else "adoption_not_identifiable_from_trace")


@pytest.mark.parametrize("fault", ["same_clock_creation", "ambiguous_frame", "future_seed", "off_camera_grid"])
def test_local_phase_or_candidate_gaps_do_not_turn_into_policy_censor(fault):
    world, case, entity, row = local_fixture()
    if fault == "same_clock_creation":
        world["jobs"][0]["created_at"] = 12.
        world["tool_calls"][0]["t"] = 12.
    elif fault == "ambiguous_frame":
        world["sensor_audit"].append({**world["sensor_audit"][-1], "evidence_id": "other"})
    elif fault == "off_camera_grid":
        world["jobs"][0]["params"]["service_started_at"] = 12.25
        world["job_transitions"][1]["t"] = 12.25
        world["sensor_audit"][-1]["t"] = 12.25
    else:
        world["delivered_observations"][0]["arrived_at"] = 11.5
    result = events._adoption_response(world, case, entity, row, {"physics_samples": 144000})
    assert result["status"] == "adoption_not_identifiable_from_trace" and result["endpoint"] is None


def test_nominal_event_clock_and_inferred_application_clock_are_distinct():
    value = events._event_apply_time(None, {"type": "priority", "time": 10.001})
    assert value["t_apply_tick"] == pytest.approx(1201 / 120)
    assert value["t_apply_tick"] > 10.001
    assert events._event_apply_time(None, {"type": "camera_start", "time": 10.001})["t_apply_tick"] is None


def test_summary_keeps_NA_policy_censor_and_trace_gap_denominators():
    rows = []
    for state in ("identified", "right_censored", "adoption_not_identifiable_from_trace", "NA"):
        row = {"event_type": "priority", "event_time": 10., "deadline": 600., "eligible": state != "NA",
            "NA_reason": "pre_event_threshold_met" if state == "NA" else None}
        if state != "NA":
            indicator = {"identified": 1, "right_censored": 0, "adoption_not_identifiable_from_trace": None}[state]
            row["job_adoption"] = {"status": state}
            if indicator == 1:
                row["job_adoption"]["witness"] = {"kind": "local_acquisition_on_existing_continuation"}
            row["timing_records"] = {"job_adoption": {"U": 2 if indicator == 1 else (590 if indicator == 0 else None),
                "H": 590., "indicator": indicator}, **dict.fromkeys(("information_response", "recovery"),
                {"U": 590., "H": 590., "indicator": 0})}
        rows.append(row)
    result = events.summarize_formal_events([{"case_id": "main-ID-L3-00", "domain": "ID", "method": "event:B4-M1",
        "events": {"rows": rows}}])
    assert len(result["groups"]) == 2
    for group in result["groups"]:
        assert group["full_event_target_repeat_denominator"] == 4 and group["eligible"] == 3
        assert group["NA_counts"] == {"pre_event_threshold_met": 1}
        timing = group["timing"]["job_adoption"]
        assert timing["identified_observed"] == timing["right_censored"] == timing["not_identifiable"] == 1
        assert timing["observed_only_conditional_elapsed_seconds"]["median"] == 2
        assert timing["right_censored_rate_among_eligible"] == pytest.approx(1 / 3)


def literal_main_case():
    from eval.inspection_population import decode_formal_population_case
    root = Path(__file__).resolve().parents[1]/"paper/research/20261104-inspection"
    body = json.loads((root / "qualification/dev-population.v1/devext-ID-L3-00.json").read_text())
    body.update(split="main", case_id="main-ID-L3-00", lineage_group="main-ID-L3-00")
    return decode_formal_population_case(body, "main", body["case_id"])


def test_formal_content_guard_requires_exact_parent_before_tracking(monkeypatch):
    import eval.inspection_tracking as tracking
    case = literal_main_case()
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid parent must fail before physical-trace reads")
    monkeypatch.setattr(tracking, "recount_tracking", forbidden)
    with pytest.raises(ValueError, match="admitted original L3"):
        events.analyze_formal_event_trace({}, case, "0" * 64)


def test_formal_wrapper_preserves_static_confirmation_and_NA_semantics(monkeypatch):
    # Real typed case validation; grade and physics are explicit software stubs.
    import eval.inspection_tracking as tracking
    case = literal_main_case()
    entity = next(entity for entity in case.entities if entity.object_type == "static" and entity.rule_id)
    t = case.priority_at
    row = {"entity_key": entity.key, "object_type": "static", "event_type": "priority", "event_time": t,
        "deadline": 600., "eligible": True, "NA_reason": None,
        "response": events.endpoint(t, 600., t + 2), "recovery": events.endpoint(t, 600., t + 5)}
    na = {**row, "eligible": False, "NA_reason": "already_satisfied", "response": None, "recovery": None}
    monkeypatch.setattr(events, "_analyze_event_trace", lambda *args: {"rows": [row, na]})
    monkeypatch.setattr(events, "reconstruct_reports", lambda *args: SimpleNamespace(reports={}))
    monkeypatch.setattr(events, "_adoption_response", lambda *args: {
        "status": "right_censored", "endpoint": events.endpoint(t, 600., None)})
    monkeypatch.setattr(tracking, "recount_tracking", lambda *args, **kwargs: {"physics_samples": 144000})
    world = {"simulated_seconds": 600., "physics_steps_executed": 144000, "physics_hz": 240, "control_hz": 120.,
        "execution_tracking_trace": {"raw_sha256": "b" * 64}}
    result = events.analyze_formal_event_trace(world, case, case.digest())
    assert row["recovery_witness"]["recovery_start"] == t + 5
    assert row["recovery_witness"]["confirmed_at"] == t + 35
    assert row["information_response"]["time"] == t + 2 and "response" not in row
    assert row["timing_records"]["job_adoption"] == {"U": 600 - t, "H": 600 - t, "indicator": 0}
    assert row["t_actor_known"] is None
    assert na["NA_reason"] == "pre_event_threshold_met" and na["job_adoption"] is None
    assert all(value is None for value in na["timing_records"].values())
    assert result["tracking_raw_sha256"] == "b" * 64


def test_dynamic_confirmation_witness_keeps_ten_original_bins(monkeypatch):
    # Isolate rich recovery extraction using the actual pure grading function.
    from envs.inspection.scoring import GradingMeasurement, grade_dynamic_bins
    population = sys.modules["eval.inspection_population"]
    import eval.inspection_tracking as tracking
    entity = CaseEntity("d", "dynamic", "inspection_vehicle", (0., 0.), 100, 1.2, "dynamic-inspection")
    versions = [GradingMeasurement(f"v{i}", 100 + 3 * i - .5, 100 + 3 * i + .5,
        entity.position(100 + 3 * i - .5), entity.true_class, True, True, True, i) for i in range(1, 61)]
    _, bins = grade_dynamic_bins(versions, entity.position, entity.true_class, 100, 600)
    bins = [asdict(entry) for entry in bins]
    case = SimpleNamespace(case_id="main-ID-L3-00", level="L3", digest=lambda: "a" * 64, entities=(entity,),
        priority_at=101., link_start=150., link_end=160., link_drone=0, camera_start=170., camera_end=180.,
        camera_drone=1, retire_at=190., retire_drone=2)
    row = {"entity_key": "d", "object_type": "dynamic", "event_type": "priority", "event_time": 101.,
        "deadline": 286., "eligible": True, "NA_reason": None, "response": events.endpoint(101., 286., None),
        "recovery": events.dynamic_recovery(bins, versions, 101., 286.)}
    monkeypatch.setattr(population, "validate_formal_population_case", lambda *args: None)
    monkeypatch.setattr(tracking, "recount_tracking", lambda *args, **kwargs: {"physics_samples": 144000})
    monkeypatch.setattr(events, "_analyze_event_trace", lambda *args: {"rows": [row]})
    monkeypatch.setattr(events, "_adoption_response", lambda *args: {
        "status": "right_censored", "endpoint": events.endpoint(101., 286., None)})
    report = SimpleNamespace(associated_key="d", object_type="dynamic", withdrawn=False, versions=versions)
    monkeypatch.setattr(events, "reconstruct_reports", lambda *args: SimpleNamespace(reports={"report": report}))
    world = {"simulated_seconds": 600., "physics_steps_executed": 144000, "physics_hz": 240, "control_hz": 120.,
        "execution_tracking_trace": {"raw_sha256": "b" * 64}, "score": {"dynamic_entities": [{"entity_key": "d", "bins": bins}]}}
    events.analyze_formal_event_trace(world, case, "a" * 64)
    assert row["recovery_witness"]["original_bin_indices"] == list(range(2, 12))
    assert row["recovery_witness"]["recovery_start"] == 103.
    assert row["recovery_witness"]["confirmed_at"] == 133.5
