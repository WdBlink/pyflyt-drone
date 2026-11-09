"""Read-only admission of local LLM flight, request and score receipts.

No dispatch, retry, budget mutation or G4 decision is performed here. Expected
bindings must come from the caller's pre-dispatch frozen episode plan.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

from eval.inspection_episode_receipts import digest, validate_flight
from eval.inspection_llm_protocol import ModelReplyError, parse_reply, serialize_public_frame, unwrap_json_fence
from eval.inspection_model_receipts import _parse_object, validate_model_operation


def read(path):
    return _parse_object(Path(path).read_text())


def validate_llm_episode(flight, operation, plan, reservation, returncode):
    """Return a source-bound scored flight, or raise without imputing Y=0."""
    flight, operation = Path(flight), Path(operation)
    if plan["condition"] not in ("B3-M1", "B3-M2", "B3-M3", "B4-M1", "B4-M2"):
        raise ValueError("undeclared LLM condition")
    y, flight_evidence = validate_flight(flight, plan, returncode)
    if y is None:
        raise ValueError("physical/source admission failed: "+str(flight_evidence["failure_reasons"]))
    run = read(flight/"run.json")
    if (run.get("reservation") != reservation or run.get("sampler_base_seed") != plan["sampler_seed"]
            or run.get("ablation") != plan.get("ablation")
            or run.get("model_registry_sha256") != plan["model_registry_sha256"]
            or run.get("model_receipt_sha256") != plan["model_receipt_sha256"]
            or run.get("model_generation_profile") != plan["generation_profile"]):
        raise ValueError("model identity/input/profile differs")
    expected = {k: plan[k] for k in ("owner_frozen_plan", "model_receipt_sha256", "generation_profile")}
    expected.update(reservation=reservation, reserved=plan["reservation_usage"], service_sha256=plan["service_sha256"])
    meter = validate_model_operation(operation, expected)
    if (run["model_operation"]["usage"] != meter["usage"]
            or run["model_operation"]["sha256"] != digest(operation/"service-completion.json")):
        raise ValueError("parent model usage differs from original operation")
    protocol_path = Path(__file__).with_name("inspection_llm_protocol.py").resolve()
    if digest(protocol_path) != plan["source_hashes"][str(protocol_path)]:
        raise ValueError("current parser is not this flight's frozen parser")
    rows = [_parse_object(line) for line in (flight/"actor.jsonl").read_text().splitlines()]
    pending, response_row, admitted, attempts, commands_count, joins = None, None, 0, 0, 0, []
    for row in rows:
        kind = row.get("kind")
        if kind == "policy_context_failure":
            if pending is not None or row.get("model_request_dispatched") is not False:
                raise ValueError("context rejection cannot dispatch or replace a pending call")
            attempts += 1
            if row.get("decision_attempt") != attempts:
                raise ValueError("decision attempts are not causal")
            continue
        if kind == "public_model_request":
            attempts += 1
            if pending is not None or row.get("request_ordinal") != attempts:
                raise ValueError("overlapping or reordered model request")
            if (type(row["source_sim_time"]) not in (int, float) or not math.isfinite(row["source_sim_time"])
                    or not 0 <= row["source_sim_time"] < 590
                    or row["sampler_seed"] != (plan["sampler_seed"]+attempts-1) % 2**32):
                raise ValueError("model request clock/seed differs")
            messages = row["messages"]
            canonical = json.dumps(messages, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
            if hashlib.sha256(canonical).hexdigest() != row["messages_sha256"]:
                raise ValueError("public messages digest differs")
            content = _parse_object(messages[1]["content"])
            serialize_public_frame(content["public_state"])
            if content["public_state"]["sim_time"] != row["source_sim_time"]:
                raise ValueError("prompt public clock differs from request clock")
            pending, response_row = row, None
            continue
        if kind == "actual_model_response":
            if pending is None or response_row is not None:
                raise ValueError("response has no unique preceding public request")
            result = row["response"]
            if result["submitted_ordinal"] != len(joins)+1:
                raise ValueError("transport submission sequence differs")
            if pending["worker_before_request"] != row["worker_before_request"]:
                raise ValueError("physical request witness differs")
            before, after = row["worker_before_request"], row["worker_after_response"]
            if (len(before) != 4 or len(after) != 4 or any(type(v) not in (int, float) or not math.isfinite(v)
                    for v in before+after) or not 0 <= before[0] <= after[0] <= 600):
                raise ValueError("invalid physical concurrency witnesses")
            if result["kind"] == "response":
                admitted += 1
                call = operation/f"call-{admitted:04d}"
                request, original = read(call/"request.json"), read(call/"response.json")
                if request["messages"] != pending["messages"] or request["seed"] != pending["sampler_seed"]:
                    raise ValueError("actual generation request differs from public request")
                if any(result.get(k) != v for k, v in original.items()):
                    raise ValueError("transport changed the original generated response")
            elif result != {"kind": "not_started", "reason": "frozen local request budget/context exhausted",
                    "parent_end_to_end_wall_seconds": result.get("parent_end_to_end_wall_seconds"),
                    "submitted_ordinal": result.get("submitted_ordinal")}:
                raise ValueError("unknown non-generation response")
            if type(result["parent_end_to_end_wall_seconds"]) not in (int, float) or not math.isfinite(result["parent_end_to_end_wall_seconds"]) or result["parent_end_to_end_wall_seconds"] < 0:
                raise ValueError("invalid parent latency")
            response_row = row
            continue
        if kind != "decoded_reply" or pending is None or response_row is None:
            raise ValueError("unexpected/incomplete actor audit")
        audit, result = row["audit"], response_row["response"]
        if audit["request"] != {k: pending[k] for k in ("request_ordinal", "source_sim_time", "messages", "messages_sha256")}:
            raise ValueError("decoded request binding differs")
        decoded = []
        if audit["status"] == "decoded_not_tool_acceptance":
            if (result["kind"] != "response" or result["status"] != "completed"
                    or result["deadline_exceeded"] or max(result["wall_seconds"], result["parent_end_to_end_wall_seconds"]) > plan["generation_profile"]["max_request_seconds"]):
                raise ValueError("late/noncompleted response was decoded")
            reply, decoded = parse_reply(result["text"])
            json.dumps(reply, ensure_ascii=False, allow_nan=False).encode("utf-8")
            if reply != audit["reply"] or audit.get("wire_framing") != {"json_fence_removed": unwrap_json_fence(result["text"])[1]}:
                raise ValueError("decoded reply/framing differs from original text")
        elif audit["status"] == "policy_format_failure":
            try:
                reply, _ = parse_reply(result.get("text"))
                json.dumps(reply, ensure_ascii=False, allow_nan=False).encode("utf-8")
            except (ModelReplyError, UnicodeError):
                pass
            else:
                raise ValueError("reported format failure is parseable under frozen parser")
        elif audit["status"] == "ignored_after_episode_deadline":
            if not response_row["final_after_flight"] and response_row["worker_after_response"][0] < 600:
                raise ValueError("episode-deadline ignore before deadline")
        elif audit["status"] == "ignored_noncompleted_or_late":
            if (result["kind"] == "response" and result["status"] == "completed"
                    and not result["deadline_exceeded"]
                    and max(result["wall_seconds"], result["parent_end_to_end_wall_seconds"]) <= plan["generation_profile"]["max_request_seconds"]):
                raise ValueError("completed on-time reply mislabeled as late")
        else:
            raise ValueError("unknown decoder disposition")
        expected_commands = [] if response_row["final_after_flight"] or response_row["worker_after_response"][0] >= 600 else [
            {"name": name, "args": args} for name, args in decoded][:max(0, 2048-commands_count)]
        if row["commands_enqueued_not_accepted"] != expected_commands:
            raise ValueError("dispatch does not follow original decoded commands/deadline")
        commands_count += len(expected_commands)
        joins.append({"decision_attempt": pending["request_ordinal"], "kind": result["kind"],
            "decode_status": audit["status"], "admitted_call_ordinal": admitted if result["kind"] == "response" else None})
        pending = response_row = None
    if (pending is not None or admitted != meter["usage"]["calls"] or attempts > plan["generation_profile"]["max_calls"]
            or commands_count != run["actor_mutating_calls"]):
        raise ValueError("unreconciled requests, call/decision cap or command counts")
    evaluator = Path(plan["evaluator"])
    if digest(evaluator) != plan["evaluator_sha256"]:
        raise ValueError("frozen evaluator entry differs")
    process = subprocess.run([plan["physics_python"], str(evaluator), "--world", str(flight/"world.json"),
        "--expected-world-sha256", digest(flight/"world.json")], capture_output=True, text=True, timeout=120)
    if process.returncode != 0:
        raise ValueError("frozen evaluator refused the world: "+process.stderr[-2000:])
    score = _parse_object(process.stdout)
    if score.get("score_equal") is not True or score["score"]["Y"] != y:
        raise ValueError("full frozen score differs")
    return y, {"role": "source-bound completed development LLM episode, not scientific acceptance",
        "flight": flight_evidence, "model_meter": meter, "causal_request_joins": joins,
        "decision_attempts": attempts, "commands_enqueued": commands_count, "actor_sha256": digest(flight/"actor.jsonl"),
        "frozen_score_recount": score}
