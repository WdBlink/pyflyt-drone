"""Offline resource/response records from already admitted closed receipts.

The physical/native/actor validator remains authoritative. This reader does
not reserve, settle, generate, dispatch, or turn missing latency into zero.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from statistics import mean

from eval.inspection_formal_admission import Artifact, bounded_bytes, canonical_sha, parse_json_bytes
from eval.inspection_recording import CONDITION_REPEATS


def _bytes(path, snapshot, project):
    name = str(path.relative_to(project))
    ref = Artifact(name, snapshot["files"][name])
    raw = bounded_bytes(ref.path_in(project), 128 * 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != ref.sha256:
        raise ValueError("closed response/resource bytes changed after admission")
    return raw


def _usage(value):
    if (type(value) is not dict or set(value) != {"calls", "tokens"}
            or any(type(value[key]) is not int or value[key] < 0 for key in value)):
        raise ValueError("known native-counted calls/tokens required, not missing or estimated usage")
    return value


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative measured response interval required")
    return value


def episode_response_diagnostics(receipt_snapshot, request, evidence, project_root):
    """Consume fresh measure_completed_episode evidence and its ORIGINAL snapshot."""
    slot, directory, before = receipt_snapshot
    project, root = Path(project_root).resolve(), Path(request["run"]).resolve()
    if (slot.condition not in CONDITION_REPEATS
            or (slot.sampler_seed is None) != (slot.condition in ("B0", "B1", "B2"))):
        raise ValueError("admitted condition and sampler identity must agree before resource lookup")
    attempts = sorted(directory.glob("attempt-*"))
    records = []
    for ordinal, attempt in enumerate(attempts):
        terminal = parse_json_bytes(_bytes(attempt / "terminal.json", before, project))
        expected_status = "completed" if ordinal == len(attempts) - 1 else "known_operational_failure"
        if terminal["status"] != expected_status:
            raise ValueError("only the admitted finite known-attempt chain supplies resource diagnostics")
        calls = []
        if slot.sampler_seed is None:
            usage = {"calls": 0, "tokens": 0}
        else:
            kind = getattr(slot, "track", None)
            if kind is not None and kind not in ("clean", "ablation", "latency"):
                raise ValueError("only admitted primary or fixed intervention tracks have resource identities")
            if kind is None:
                kind = "main" if request["opening"] is None else "holdout"
            reservation = f"formal-{kind}-{request['g4']['sha256'][:12]}-{slot.order_index:04d}-a{ordinal}"
            receipt = root / "qualification/local-model-episodes" / reservation / "usage-receipt.json"
            usage = _usage(parse_json_bytes(_bytes(receipt, before, project))["usage"])
            operation = project / "logs/inspection-model-resource-operations.v1" / reservation
            for number in range(1, usage["calls"] + 1):
                path = operation / f"call-{number:04d}/response.json"
                response = parse_json_bytes(_bytes(path, before, project))
                call_usage = _usage(response["usage"])
                if (call_usage["calls"] != 1 or response["kind"] != "response"
                        or response["status"] not in ("completed", "policy_timeout", "runtime_failure")
                        or response["receipt"] != str(path)):
                    raise ValueError("resource interval must come from its original admitted call receipt")
                calls.append({"ordinal": number, "status": response["status"], "tokens": call_usage["tokens"],
                    "service_generation_and_journaling_wall_seconds": _seconds(response["wall_seconds"]),
                    "response_path": str(path.relative_to(project)), "response_sha256": before["files"][str(path.relative_to(project))],
                    "deadline_exceeded": response["deadline_exceeded"]})
            if sum(call["tokens"] for call in calls) != usage["tokens"]:
                raise ValueError("all call token receipts must match the known attempt settlement")
        records.append({"attempt_ordinal": ordinal, "status": terminal["status"], "usage": usage, "calls": calls})
    if not records:
        raise ValueError("an actual admitted completed attempt is required")
    responses, context_rejections = [], 0
    if slot.sampler_seed is not None:
        actor_path = attempts[-1] / "flight/actor.jsonl"
        actor_name = str(actor_path.relative_to(project))
        if actor_name in before["files"]:
            actor = [parse_json_bytes(line) for line in _bytes(actor_path, before, project).splitlines()]
        elif records[-1]["usage"]["calls"]:
            raise ValueError("admitted nonzero model calls require their validated actor journal")
        else:
            actor = []
        requests, replies, decoded = {}, {}, {}
        for entry in actor:
            kind = entry["kind"]
            if kind == "policy_context_failure":
                context_rejections += 1
                continue
            if kind not in ("public_model_request", "actual_model_response", "decoded_reply"):
                raise ValueError("only the previously validated actor journal format is analyzed")
            if kind == "public_model_request":
                key, table = entry["request_ordinal"], requests
            elif kind == "actual_model_response":
                key, table = entry["response"]["submitted_ordinal"], replies
            else:
                key, table = entry["audit"]["request"]["request_ordinal"], decoded
            if type(key) is not int or key < 1 or key in table:
                raise ValueError("native unique actor request/response identities required")
            table[key] = entry
        joins = evidence.get("causal_request_joins", [])
        if (len(replies) != len(joins) or len(requests) != len(joins) or len(decoded) != len(joins)
                or [join["admitted_call_ordinal"] for join in joins if join["kind"] == "response"]
                   != list(range(1, records[-1]["usage"]["calls"] + 1))):
            raise ValueError("diagnostic timing must retain all freshly revalidated causal request joins")
        for submitted, join in enumerate(joins, 1):
            decision = join["decision_attempt"]
            if type(decision) is not int or decision not in requests or decision not in decoded or submitted not in replies:
                raise ValueError("diagnostic join does not identify the original request and response")
            source, received, dispatched = requests[decision], replies[submitted], decoded[decision]
            result = received["response"]
            number = join["admitted_call_ordinal"]
            if result["kind"] != join["kind"]:
                raise ValueError("response kind differs from the freshly validated causal join")
            service_wall = None
            if join["kind"] == "response":
                if type(number) is not int or not 1 <= number <= len(records[-1]["calls"]):
                    raise ValueError("diagnostic generation must bind a native admitted call ordinal")
                call = records[-1]["calls"][number - 1]
                original = parse_json_bytes(_bytes(project / call["response_path"], before, project))
                if canonical_sha({key: result.get(key) for key in original}) != canonical_sha(original):
                    raise ValueError("actor timing changed the original service response")
                service_wall = call["service_generation_and_journaling_wall_seconds"]
            elif join["kind"] != "not_started" or number is not None:
                raise ValueError("declined requests cannot be reclassified as generated model calls")
            if dispatched["audit"]["status"] != join["decode_status"]:
                raise ValueError("diagnostic decoder disposition differs from the admitted causal join")
            delay = int(slot.arm[6:]) if getattr(slot, "track", None) == "latency" else 0
            if type(received["injected_delay_requested_seconds"]) is not int or received["injected_delay_requested_seconds"] != delay:
                raise ValueError("recorded requested wait differs from the fixed admitted delay arm")
            source_time = _seconds(source["source_sim_time"])
            before = _seconds(received["worker_before_request"][0])
            reply = _seconds(received["worker_at_model_reply"][0])
            after = _seconds(received["worker_after_response"][0])
            if (not source_time <= before <= reply <= after <= 600
                    or dispatched["dispatch_age_sim_seconds"] != after - source_time):
                raise ValueError("recorded response progress must retain the admitted causal simulation clocks")
            responses.append({"decision_attempt": decision, "kind": join["kind"], "admitted_call_ordinal": number,
                "decode_status": join["decode_status"], "source_sim_time": source_time,
                "worker_before_request_sim_time": before,
                "worker_before_injected_wait_sim_time": reply, "worker_after_response_sim_time": after,
                "service_generation_and_journaling_wall_seconds": service_wall,
                "client_end_to_end_before_injection_wall_seconds": _seconds(result["parent_end_to_end_wall_seconds"]),
                "injected_wait_requested_seconds": received["injected_delay_requested_seconds"],
                "injected_wait_measured_wall_seconds": _seconds(received["injected_delay_wall_seconds"]),
                "recorded_dispatch_age_sim_seconds": _seconds(dispatched["dispatch_age_sim_seconds"]),
                "client_roundtrip_plus_consume_wall_seconds": _seconds(dispatched["dispatch_age_wall_seconds"]),
                "final_after_flight": received["final_after_flight"],
                "commands_enqueued_not_accepted_count": len(dispatched["commands_enqueued_not_accepted"])})
    return {"case_id": slot.case_id, "domain": slot.domain, "level": slot.level,
        "condition": slot.condition, "repeat_ordinal": slot.repeat_ordinal,
        "track": getattr(slot, "track", None), "arm": getattr(slot, "arm", None),
        "source_episode": str(directory.relative_to(project)),
        "valid_flight_usage": records[-1]["usage"],
        "all_known_attempt_usage": {key: sum(record["usage"][key] for record in records) for key in ("calls", "tokens")},
        "attempts": records, "valid_flight_responses": responses, "context_rejections": context_rejections,
        "non_generation_responses": sum(row["kind"] == "not_started" for row in responses),
        "money_cost": None, "energy_cost": None, "capital_energy_cost_unmetered": True,
        "timing_scope": "service includes generation/journaling; client ends at poll before extra wait; sim age uses response-handling progress reused at enqueue, before decoding; wall sum combines roundtrip and consume and omits unsampled poll-to-consume gap, not source-information wall age or exact job adoption",
        "retry_scope": "all known attempts count resources; only admitted valid-flight actor timing, failed-attempt service intervals retained separately"}


def summarize_response_resources(episodes):
    """Repeat means within each original root; request distributions stay descriptive."""
    grouped = {}
    for episode in episodes:
        key = episode["case_id"], episode["domain"], episode["level"], episode.get("method", episode["condition"])
        grouped.setdefault(key, []).append(episode)
    rows = []
    for (case, domain, level, method), values in sorted(grouped.items()):
        conditions = {row["condition"] for row in values}
        if len(conditions) != 1:
            raise ValueError("a resource method has one original condition and its full planned repeats")
        count = CONDITION_REPEATS[next(iter(conditions))]
        if (any(type(row["repeat_ordinal"]) is not int for row in values)
                or sorted(row["repeat_ordinal"] for row in values) != list(range(count))):
            raise ValueError("resource summaries require all original repeats exactly once")
        rows.append({"case_id": case, "domain": domain, "level": level, "method": method,
            "planned_repeat_denominator": len(values),
            **{scope: {key: mean(row[scope][key] for row in values) for key in ("calls", "tokens")}
               for scope in ("valid_flight_usage", "all_known_attempt_usage")},
            "known_attempt_count": sum(len(row["attempts"]) for row in values),
            "non_generation_response_count": sum(row["non_generation_responses"] for row in values),
            "context_rejection_count": sum(row["context_rejections"] for row in values)})
    return {"role": "descriptive_resource_repeat_means_with_all_known_retry_overhead", "root_method_rows": rows,
        "money_cost": None, "energy_cost": None, "scope": "calls/tokens are observed local model usage; no request-level inferential intervals or unmetered cost imputation"}
