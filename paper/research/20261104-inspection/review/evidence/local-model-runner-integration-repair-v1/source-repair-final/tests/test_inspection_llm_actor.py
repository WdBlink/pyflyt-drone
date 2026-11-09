"""Public clock and reply boundaries, without model or flight execution."""

import copy
import json

import pytest

from eval.inspection_llm_actor import PublicLLMActor


def frame(stamp=0):
    return {"sim_time": stamp, "task_card": {"task_card_version": 1},
        "aircraft": [{"drone_id": 0, "retired": False, "link_available": True, "camera_quality": 1}],
        "jobs": [], "events": [], "coverage": [], "observations": [], "report_slots_used": 0}


def response(text='{"commands":[]}'):
    return {"status": "completed", "wall_seconds": 1, "deadline_exceeded": False, "text": text}


def test_only_one_inflight_and_fixed_period_vs_event_trigger():
    b3, b4 = PublicLLMActor("B3"), PublicLLMActor("B4")
    for actor in (b3, b4):
        actor.ingest(frame())
        assert actor.due()
        actor.start_request()
        actor.ingest(frame(30))
        assert not actor.due()
        with pytest.raises(ValueError, match="another request"):
            actor.start_request()
        actor.finish_request(response())
        actor.start_request()
        actor.finish_request(response())
    changed = frame(40)
    changed["aircraft"][0]["link_available"] = False
    b3.ingest(changed)
    b4.ingest(changed)
    assert not b3.due() and b4.due()


@pytest.mark.parametrize("change", [{"wall_seconds": 61}, {"wall_seconds": float("nan")},
    {"deadline_exceeded": True}, {"status": "runtime_failure"}, {"text": "```json\n{}\n```"}])
def test_failed_late_or_malformed_reply_never_dispatches_or_retries(change):
    actor = PublicLLMActor("B4", max_calls=1)
    actor.ingest(frame())
    actor.start_request()
    actor.ingest(frame(70))
    commands, audit = actor.finish_request({**response('{"commands":[{"name":"enter_safe_exit","args":{"drone_id":0,"expected_version":0}}]}'), **change})
    assert commands == [] and audit["status"] != "decoded_not_tool_acceptance"
    assert actor.attempts == 1 and not actor.due()


def test_model_commands_remain_worker_requests_not_ack_or_executed_plan():
    actor = PublicLLMActor("B4")
    actor.ingest(frame())
    actor.start_request()
    commands, audit = actor.finish_request(response('{"commands":[{"name":"enter_safe_exit","args":{"drone_id":0,"expected_version":987}}],"plan":["exit"]}'))
    assert commands == [("enter_safe_exit", {"drone_id": 0, "expected_version": 987})]
    assert audit["status"] == "decoded_not_tool_acceptance" and actor.plan == ["exit"]
    actor.result(*commands[0], {"status": "rejected", "reason": "version mismatch"})
    assert "tool_rejection" in actor.pending_triggers


def test_incremental_observations_are_retained_causally_and_private_state_refused():
    actor = PublicLLMActor("B3")
    state = frame()
    state["observations"] = [{"evidence_id": "visible-1", "measured_at": 0, "candidate_id": "local-0"}]
    actor.ingest(state)
    actor.ingest(frame(1))
    assert "visible-1" in actor.start_request()["messages"][1]["content"]
    private = copy.deepcopy(state)
    private["observations"][0]["truth_xy"] = [10, 20]
    with pytest.raises(ValueError, match="private/evaluator"):
        actor.ingest(private)


def test_identical_public_projection_for_b3_b4_and_no_hidden_case_input():
    actors = [PublicLLMActor(x) for x in ("B3", "B4")]
    for actor in actors:
        actor.ingest(frame())
    contents = [json.loads(a.start_request()["messages"][1]["content"]) for a in actors]
    assert contents[0]["public_state"] == contents[1]["public_state"]
    assert contents[0]["recent_tool_results"] == contents[1]["recent_tool_results"]


def test_clock_regression_rejected_before_new_state_is_retained():
    actor = PublicLLMActor("B4")
    actor.ingest(frame(10))
    with pytest.raises(ValueError, match="regressed"):
        actor.ingest(frame(9))
    assert actor.state["sim_time"] == 10


@pytest.mark.parametrize("event", ["succeeded_job", "accepted_report"])
def test_b4_monitors_completed_service_and_report_receipts(event):
    actor = PublicLLMActor("B4")
    state = frame()
    state["jobs"] = [{"job_id": "j1", "status": "running"}]
    actor.ingest(state)
    actor.start_request()
    actor.finish_request(response())
    state["sim_time"] = 10
    if event == "succeeded_job":
        state["jobs"][0]["status"] = "succeeded"
    else:
        actor.result("submit_or_revise_report", {}, {"status": "accepted"})
    actor.ingest(state)
    assert actor.due()


def test_three_single_factor_ablations_change_the_declared_mechanism_only():
    actors = {flag: PublicLLMActor("B4", ablation=flag) for flag in
        (None, "periodic", "no-cross-aircraft-memory", "no-progressive-confirmation-guidance")}
    requests = {}
    for flag, actor in actors.items():
        actor.plan = ["persistent plan"]
        actor.note = "bounded memory"
        actor.ingest(frame())
        requests[flag] = actor.start_request()
        actor.finish_request(response())
        actor.result("submit_or_revise_report", {}, {"status": "accepted"})
        actor.ingest(frame(10))
    assert actors[None].due() and not actors["periodic"].due()
    memories = {k: json.loads(v["messages"][1]["content"])["memory"] for k, v in requests.items()}
    assert "public_static_alias_groups" in memories[None] and "plan" in memories[None]
    assert "public_static_alias_groups" not in memories["no-cross-aircraft-memory"]
    assert "plan" not in memories["no-cross-aircraft-memory"]
    assert "two high-quality distinct-view confirmations" in requests[None]["messages"][0]["content"]
    assert "two high-quality distinct-view confirmations" not in requests["no-progressive-confirmation-guidance"]["messages"][0]["content"]
