"""Candidate LLM wire boundary; no model invocation or paper acceptance."""

from __future__ import annotations

import json
import math


TOOL_FIELDS = {
    "assign_survey": ("drone_id", "region_id", "duration", "expected_version", "rule_id", "task_card_version"),
    "inspect_candidate": ("drone_id", "candidate_id", "expected_version", "rule_id", "task_card_version"),
    "observe_track": ("drone_id", "candidate_id", "duration", "report_id", "expected_version", "rule_id", "task_card_version"),
    "handoff_observation": ("from_drone", "to_drone", "candidate_id", "duration", "report_id",
                            "expected_from_version", "expected_to_version", "rule_id", "task_card_version"),
    "cancel_job": ("job_id", "expected_version"),
    "submit_or_revise_report": ("report_id", "object_type", "candidate_id", "measured_at", "estimate_xy", "claimed_class", "evidence_ids"),
    "enter_safe_exit": ("drone_id", "expected_version"),
}
PUBLIC_FIELDS = {"sim_time", "task_card", "aircraft", "jobs", "events", "coverage", "observations", "report_slots_used"}
FORBIDDEN_FIELDS = {"private_case", "hidden_entities", "entity_key", "truth_xy", "seed", "sensor_audit",
                    "sensor_opportunities", "background_trials", "pending_packets", "future_events",
                    "score", "truth", "ground_truth", "grader_output"}


class ModelReplyError(ValueError):
    """A policy-format failure, not an infrastructure retry reason."""


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ModelReplyError("duplicate JSON field")
        result[name] = value
    return result


def _nonfinite(value):
    raise ModelReplyError("nonfinite JSON constant")


def unwrap_json_fence(text):
    """Remove one whole-output JSON fence; never extract or repair JSON."""
    if not isinstance(text, str):
        return text, False
    stripped = text.strip()
    lines = stripped.split("\n")
    if (len(lines) >= 3 and lines[0].strip().casefold() in ("```", "```json")
            and lines[-1].strip() == "```"
            and not any(line.strip().startswith("```") for line in lines[1:-1])):
        return "\n".join(lines[1:-1]), True
    return text, False


def parse_reply(text):
    if not isinstance(text, str) or not text or len(text) > 24000:
        raise ModelReplyError("reply length is invalid")
    text, _ = unwrap_json_fence(text)
    try:
        reply = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_nonfinite)
        json.dumps(reply, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as error:
        raise ModelReplyError("reply must be one strict JSON object") from error
    if (not isinstance(reply, dict) or "commands" not in reply or set(reply)-{"commands", "note", "plan"}
            or not isinstance(reply["commands"], list) or len(reply["commands"]) > 8):
        raise ModelReplyError("commands schema is invalid")
    if "note" in reply and (not isinstance(reply["note"], str) or len(reply["note"]) > 512):
        raise ModelReplyError("brief note is invalid")
    if "plan" in reply and (not isinstance(reply["plan"], list) or len(reply["plan"]) > 8
                            or any(not isinstance(x, str) or len(x) > 256 for x in reply["plan"])):
        raise ModelReplyError("plan summary is invalid")
    commands = []
    for command in reply["commands"]:
        if (not isinstance(command, dict) or set(command) != {"name", "args"}
                or not isinstance(command["name"], str) or command["name"] not in TOOL_FIELDS
                or not isinstance(command["args"], dict)
                or set(command["args"]) != set(TOOL_FIELDS[command["name"]])):
            raise ModelReplyError("unknown tool or incorrect fields")
        commands.append((command["name"], command["args"]))
    # Numerical/type/version/evidence semantics remain the shared worker's
    # authority; decoder acceptance never means a job succeeded or report scored.
    return reply, commands


def serialize_public_frame(state):
    if not isinstance(state, dict) or set(state) != PUBLIC_FIELDS:
        raise ValueError("only the declared public state may enter an LLM prompt")
    stamp = state["sim_time"]
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or not 0 <= stamp <= 600:
        raise ValueError("invalid public clock")
    def check(value):
        if isinstance(value, dict):
            if any(not isinstance(key, str) or key in FORBIDDEN_FIELDS or key.startswith("_") for key in value):
                raise ValueError("private/evaluator field at the prompt boundary")
            for child in value.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)
    check(state)
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
