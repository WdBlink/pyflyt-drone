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
PUBLIC_TOOL_CONTRACT = """
Shared public tool contract v2:
For a first report choose a new report_id of 1..64 characters; for revisions
reuse that accepted ID. Each new ID permanently consumes one report slot,
including an unsupported claim. Its object_type and initial target association
are fixed: revising does not redirect it to another target or reset association.
ACK confirms storage only, without revealing association or correctness.
Effective evidence must be a nonempty unique list of received evidence_ids,
all from the request's candidate_id and all arrived by submission. Use a finite
measured_at in [0,now]: every cited measurement must be at or before that stamp
and at least one must be exactly at it. All cited observations need quality>=0.8
for effective delivery. A schema-valid unsupported/low-quality claim may still
be ACKed and consume a slot. Correct class, identity and location are evaluated
offline; no correctness oracle is provided. Different local candidate handles
cannot be mixed in one evidence reference even if they may denote one object.
inspect_candidate succeeds after two actual qualified views with aircraft
positions at least20m apart. This is a job completion condition, not a hard
report admission rule; reports need not wait for an inspect success ACK.
observe_track and handoff_observation require an already accepted report_id.
Each new matched observation acquired by that observe job is routed to its
report at acquisition and automatically appended when its packet reaches
ground; no manual report per bin is required. Offline packets wait for link
restoration and may arrive too late. Job failure does not erase delivered data.
A dynamic measurement is sealed on its first submission. Repeating the same
measurement/time/estimate/class is idempotent; changing a sealed measurement's
estimate or class is rejected. New evidence may append a new measurement.
The global dynamic demand window is anchored at target appearance, not at
job start, discovery or first local acquisition. Its180s window has3s bins;
timely delivery requires the report and its evidence at ground by bin end+6s.
The measured_at stamp selects the original bin; late upload cannot create a
new measurement or restart demand. Observed data must still reach ground by
the episode deadline. The job's requested service duration instead starts at
its first new qualified local measurement. Hidden appearance times and future
events are unavailable; current public observations are the only evidence.
"""
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
