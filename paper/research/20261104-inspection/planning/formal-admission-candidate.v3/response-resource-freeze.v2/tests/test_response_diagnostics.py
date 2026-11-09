"""Synthetic closed files and trusted-evidence stubs; no model/native effects."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

path = Path(__file__).resolve().parents[1] / "source/eval/inspection_response_diagnostics.py"
sys.path.insert(0, str(path.parents[1]))
spec = importlib.util.spec_from_file_location("response_diagnostics_candidate", path)
diagnostics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostics)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def fixture(tmp_path, *, retry=False, context=False, non_generation=False, final=False):
    project = (tmp_path / "project").resolve(); project.mkdir()
    root = project / "run"; root.mkdir()
    slot = SimpleNamespace(case_id="literal-software-case", domain="ID", level=3, condition="B4-M1",
        repeat_ordinal=0, sampler_seed=7, order_index=0, track="latency", arm="delay-20")
    request = {"run": str(root), "g4": {"sha256": "a" * 64}, "opening": None}
    episode = project / "logs/episode"
    for ordinal in range(2 if retry else 1):
        last = ordinal == (1 if retry else 0)
        attempt = episode / f"attempt-{ordinal:02d}"
        write(attempt / "terminal.json", {"status": "completed" if last else "known_operational_failure"})
        reservation = f"formal-latency-aaaaaaaaaaaa-0000-a{ordinal}"
        operation = project / "logs/inspection-model-resource-operations.v1" / reservation
        response_path = operation / "call-0001/response.json"
        tokens = 102 if last else 8
        usage = {"calls": 0, "tokens": 0} if last and non_generation else {"calls": 1, "tokens": tokens}
        write(root / "qualification/local-model-episodes" / reservation / "usage-receipt.json", {
            "role": "literal fixture, not actual native settlement", "usage": usage})
        response = {"kind": "response", "status": "completed" if last else "runtime_failure",
            "usage": {"calls": 1, "tokens": tokens}, "wall_seconds": (15. if final else 5.) if last else 2.,
            "deadline_exceeded": False, "receipt": str(response_path)}
        if usage["calls"]:
            write(response_path, response)
        if not last:
            continue
        decision = 2 if context else 1
        kind = "not_started" if non_generation else "response"
        status = "ignored_noncompleted_or_late" if non_generation else (
            "ignored_after_episode_deadline" if final else "decoded_not_tool_acceptance")
        join = {"decision_attempt": decision, "kind": kind, "admitted_call_ordinal": None if non_generation else 1,
            "decode_status": status}
        returned = {"kind": "not_started", "reason": "frozen local request budget/context exhausted"} if non_generation else response
        source_time = 589. if final else .5
        before = 589.5 if final else .5
        reply = 600. if final else 6.5
        after = 600. if final else (6.5 if non_generation else 26.55)
        rows = ([{"kind": "policy_context_failure", "decision_attempt": 1}] if context else []) + [
            {"kind": "public_model_request", "request_ordinal": decision, "source_sim_time": source_time},
            {"kind": "actual_model_response", "response": {**returned, "submitted_ordinal": 1,
                "parent_end_to_end_wall_seconds": 16. if final else 6.}, "injected_delay_requested_seconds": 20,
                "injected_delay_wall_seconds": 0. if non_generation or final else 20.05, "final_after_flight": final,
                "worker_before_request": [before, 0., 0., 96.], "worker_at_model_reply": [reply, 0., 0., 96.],
                "worker_after_response": [after, 0., 0., 96.]},
            {"kind": "decoded_reply", "audit": {"request": {"request_ordinal": decision}, "status": status},
                "dispatch_age_sim_seconds": after - source_time,
                "dispatch_age_wall_seconds": 16.1 if final else (6.1 if non_generation else 26.1),
                "commands_enqueued_not_accepted": []}]
        actor = attempt / "flight/actor.jsonl"; actor.parent.mkdir()
        actor.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    snapshot = {"files": {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in project.rglob("*") if path.is_file()}}
    return project, request, (slot, episode, snapshot), {"causal_request_joins": [join]}


@pytest.mark.parametrize("retry,context", [(False, False), (True, False), (False, True)])
def test_measured_intervals_stay_distinct_and_retry_cost_is_not_a_repeat(tmp_path, retry, context):
    project, request, snapshot, evidence = fixture(tmp_path, retry=retry, context=context)
    value = diagnostics.episode_response_diagnostics(snapshot, request, evidence, project)
    assert value["valid_flight_usage"] == {"calls": 1, "tokens": 102}
    assert value["all_known_attempt_usage"] == {"calls": 2 if retry else 1, "tokens": 110 if retry else 102}
    response = value["valid_flight_responses"][0]
    assert response["service_generation_and_journaling_wall_seconds"] == 5.
    assert response["client_end_to_end_before_injection_wall_seconds"] == 6.
    assert response["injected_wait_measured_wall_seconds"] == 20.05
    assert response["recorded_dispatch_age_sim_seconds"] == 26.55 - .5
    assert response["client_roundtrip_plus_consume_wall_seconds"] == 26.1
    assert response["decision_attempt"] == (2 if context else 1)
    assert value["context_rejections"] == int(context)
    assert value["money_cost"] is None and value["energy_cost"] is None


@pytest.mark.parametrize("non_generation,final", [(True, False), (False, True)])
def test_declined_or_final_replies_do_not_fabricate_twenty_seconds_wait(tmp_path, non_generation, final):
    project, request, snapshot, evidence = fixture(tmp_path, non_generation=non_generation, final=final)
    value = diagnostics.episode_response_diagnostics(snapshot, request, evidence, project)
    response = value["valid_flight_responses"][0]
    assert response["injected_wait_requested_seconds"] == 20
    assert response["injected_wait_measured_wall_seconds"] == 0
    assert (response["service_generation_and_journaling_wall_seconds"] is None) == non_generation
    assert value["non_generation_responses"] == int(non_generation)


@pytest.mark.parametrize("declined_first", [False, True])
def test_multiple_generated_joins_keep_original_file_snapshot(tmp_path, declined_first):
    project, request, snapshot, evidence = fixture(tmp_path, non_generation=declined_first)
    slot, episode, before = snapshot
    reservation = "formal-latency-aaaaaaaaaaaa-0000-a0"
    operation = project / "logs/inspection-model-resource-operations.v1" / reservation
    number = 1 if declined_first else 2
    path = operation / f"call-{number:04d}/response.json"
    response = {"kind": "response", "status": "completed", "usage": {"calls": 1, "tokens": 102},
        "wall_seconds": 7., "deadline_exceeded": False, "receipt": str(path)}
    write(path, response)
    usage = project / "run/qualification/local-model-episodes" / reservation / "usage-receipt.json"
    write(usage, {"role": "literal fixture, no native operation", "usage": {"calls": number, "tokens": 102 * number}})
    actor = episode / "attempt-00/flight/actor.jsonl"
    rows = [json.loads(line) for line in actor.read_text().splitlines()]
    rows.extend([{"kind": "public_model_request", "request_ordinal": 2, "source_sim_time": 30.},
        {"kind": "actual_model_response", "response": {**response, "submitted_ordinal": 2,
            "parent_end_to_end_wall_seconds": 8.}, "injected_delay_requested_seconds": 20,
            "injected_delay_wall_seconds": 20.05, "final_after_flight": False,
            "worker_before_request": [30., 0., 0., 96.], "worker_at_model_reply": [38., 0., 0., 96.],
            "worker_after_response": [58.05, 0., 0., 96.]},
        {"kind": "decoded_reply", "audit": {"request": {"request_ordinal": 2}, "status": "decoded_not_tool_acceptance"},
            "dispatch_age_sim_seconds": 58.05 - 30., "dispatch_age_wall_seconds": 28.1,
            "commands_enqueued_not_accepted": []}])
    actor.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    evidence["causal_request_joins"].append({"decision_attempt": 2, "kind": "response",
        "admitted_call_ordinal": number, "decode_status": "decoded_not_tool_acceptance"})
    # Complete the synthetic fixture before collecting its ORIGINAL read snapshot.
    before["files"] = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in project.rglob("*") if path.is_file()}
    value = diagnostics.episode_response_diagnostics(snapshot, request, evidence, project)
    assert value["valid_flight_usage"] == {"calls": number, "tokens": 102 * number}
    assert len(value["valid_flight_responses"]) == 2
    assert value["valid_flight_responses"][-1]["service_generation_and_journaling_wall_seconds"] == 7.


@pytest.mark.parametrize("fault", ["unknown", "changed_bytes", "wrong_actor_kind", "missing_join", "bool_call_id", "mismatched_tokens"])
def test_unresolved_mismatched_or_changed_data_cannot_supply_diagnostics(tmp_path, fault):
    project, request, snapshot, evidence = fixture(tmp_path)
    slot, episode, before = snapshot
    if fault == "missing_join":
        evidence["causal_request_joins"] = []
    elif fault == "bool_call_id":
        evidence["causal_request_joins"][0]["admitted_call_ordinal"] = True
    else:
        if fault == "unknown":
            path = episode / "attempt-00/terminal.json"; value = json.loads(path.read_text()); value["status"] = "unknown"; write(path, value)
        elif fault == "mismatched_tokens":
            path = next((project / "run/qualification").rglob("usage-receipt.json")); value = json.loads(path.read_text()); value["usage"]["tokens"] = 103; write(path, value)
        else:
            path = episode / "attempt-00/flight/actor.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[1]["response"]["kind"] = "not_started"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        if fault != "changed_bytes":
            before["files"][str(path.relative_to(project))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError):
        diagnostics.episode_response_diagnostics(snapshot, request, evidence, project)


def test_resource_summary_uses_complete_original_repeats_and_no_missing_time_imputation(tmp_path):
    project, request, snapshot, evidence = fixture(tmp_path, retry=True)
    value = diagnostics.episode_response_diagnostics(snapshot, request, evidence, project)
    rows = [{**deepcopy(value), "repeat_ordinal": repeat, "method": "delay-20"} for repeat in range(3)]
    summary = diagnostics.summarize_response_resources(rows)
    row = summary["root_method_rows"][0]
    assert row["planned_repeat_denominator"] == 3 and row["known_attempt_count"] == 6
    assert row["valid_flight_usage"]["tokens"] == 102 and row["all_known_attempt_usage"]["tokens"] == 110
    with pytest.raises(ValueError, match="all original repeats"):
        diagnostics.summarize_response_resources(rows[:-1])


def test_classic_has_zero_model_calls_but_no_imputed_money_or_latency(tmp_path):
    project = tmp_path.resolve()
    root = project / "run"; root.mkdir()
    episode = project / "logs/classic"
    terminal = episode / "attempt-00/terminal.json"; write(terminal, {"status": "completed"})
    before = {"files": {str(terminal.relative_to(project)): hashlib.sha256(terminal.read_bytes()).hexdigest()}}
    slot = SimpleNamespace(case_id="literal-classic", domain="ID", level=1, condition="B1", repeat_ordinal=0,
        sampler_seed=None, order_index=0)
    request = {"run": str(root), "opening": None, "g4": {"sha256": "a" * 64}}
    value = diagnostics.episode_response_diagnostics((slot, episode, before), request, {}, project)
    assert value["valid_flight_usage"] == value["all_known_attempt_usage"] == {"calls": 0, "tokens": 0}
    assert value["valid_flight_responses"] == [] and value["money_cost"] is None
    assert diagnostics.summarize_response_resources([value])["root_method_rows"][0]["planned_repeat_denominator"] == 1
    slot.condition = "B4-M1"
    with pytest.raises(ValueError, match="condition and sampler"):
        diagnostics.episode_response_diagnostics((slot, episode, before), request, {}, project)
