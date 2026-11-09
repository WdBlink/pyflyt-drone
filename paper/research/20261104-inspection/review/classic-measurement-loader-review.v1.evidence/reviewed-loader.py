"""Read genuine completed episode receipts for the recording SDK.

No flight, generation, imputation or G4 acceptance occurs here. The caller must
bind ``expected`` to its frozen design/dataset/method manifest before recording.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from eval.inspection_cases import CaseEntity
from eval.inspection_common_calendar import _score
from eval.inspection_episode_receipts import _bindings, canonical_digest, digest, validate_flight
from eval.inspection_recording import CONDITION_REPEATS


def _json(path):
    return json.loads(Path(path).read_text())


def _recount_score(world):
    private = world["private_case"]
    case = SimpleNamespace(level=private["level"], priority_at=private["priority_at"],
        entities=tuple(CaseEntity(**e) for e in private["entities"]))
    if list(private["entities"]) != world["hidden_entities"]:
        raise ValueError("hidden world entities differ from frozen case")
    return _score(world, case, case)


def measure_completed_episode(episode, expected):
    """Return Y plus exact file bindings; missing/operational rows raise."""
    episode = Path(episode)
    condition, repeat = expected["condition"], expected["repeat_ordinal"]
    if (condition not in CONDITION_REPEATS or type(repeat) is not int
            or not 0 <= repeat < CONDITION_REPEATS[condition]):
        raise ValueError("episode is outside the preregistered eight-condition repetition grid")
    if condition not in ("B0", "B1", "B2"):
        raise ValueError("model receipt adapter is not qualified; a physics trace alone is not model evidence")
    phase = expected["phase"]
    if phase not in ("development", "main", "holdout"):
        raise ValueError("declare the actual data role")
    if phase != "development":
        gate = expected["g4_receipt"]
        if digest(gate["path"]) != gate["sha256"] or _json(gate["path"]).get("status") != "qualified":
            raise ValueError("formal measurement remains blocked before source-bound G4 qualification")
    plan_path = episode/"plan.json"
    plan = _json(plan_path)
    for field in ("case_id", "condition", "repeat_ordinal", "case_file_sha256", "case_sha256",
                  "suite_manifest_sha256", "source_hashes", "dependency_hashes", "supervisor_sha256",
                  "receipt_adapter_sha256", "thread_environment", "max_clock_lag_seconds"):
        if plan[field] != expected[field]:
            raise ValueError("episode differs from the frozen input/source/method manifest: "+field)
    attempts = sorted(episode.glob("attempt-*"))
    if (type(plan.get("max_attempts")) is not int or plan["max_attempts"] != 2
            or not 1 <= len(attempts) <= plan["max_attempts"]
            or [p.name for p in attempts] != [f"attempt-{i:02d}" for i in range(len(attempts))]
            or not all(p.is_dir() for p in attempts)):
        raise ValueError("attempt sequence differs from the finite retry policy")
    completed = []
    for ordinal, attempt in enumerate(attempts):
        terminal = _json(attempt/"terminal.json")  # a missing terminal is unknown, never zero
        if terminal["plan_sha256"] != canonical_digest(plan) or terminal["record_hashes"] != _bindings(attempt):
            raise ValueError("attempt record chain differs")
        intent = _json(attempt/"intent.json")
        if (intent.get("plan_sha256") != canonical_digest(plan) or intent.get("attempt_ordinal") != ordinal
                or terminal.get("stdout_sha256") != terminal["record_hashes"]["stdout.log"]):
            raise ValueError("attempt intent or output binding differs")
        if not terminal.get("not_started") and _json(attempt/"started.json").get("plan_sha256") != canonical_digest(plan):
            raise ValueError("attempt start binding differs")
        if terminal["status"] == "completed":
            if terminal.get("not_started") or ordinal != len(attempts)-1:
                raise ValueError("completed flight must be the final dispatched attempt")
            value, validation = validate_flight(attempt/"flight", plan, terminal["returncode"])
            if value is None or value != terminal["Y"] or validation != terminal["evidence"]:
                raise ValueError("completed physical flight validation differs")
            completed.append((attempt, terminal, value))
        elif terminal["status"] != "known_operational_failure" or not terminal.get("retry_allowed"):
            raise ValueError("nonretryable/unknown attempt blocks measurement")
    if len(completed) != 1:
        raise ValueError("exactly one first valid flight within the finite retry policy is required")
    attempt, terminal, value = completed[0]
    world = _json(attempt/"flight/world.json")
    private_sha = hashlib.sha256(json.dumps(world["private_case"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    if private_sha != plan["case_sha256"]:
        raise ValueError("worker semantic case identity differs")
    recount = _recount_score(world)
    if recount != world["score"]:
        raise ValueError("complete raw report/coverage/denominator score recount differs")
    level, score = world["private_case"]["level"], recount
    components = [score["C"], score["S"]]+([] if level == "L1" else [score["D"]])
    if (not all(type(x) in (int, float) and math.isfinite(x) and 0 <= x <= 1 for x in components)
            or not math.isclose(score["Q"], sum(components)/len(components), abs_tol=1e-12)):
        raise ValueError("quality aggregation differs from the declared task level")
    return value, {"role": "genuine_completed_source_bound_episode_not_new_run",
        "phase": phase, "case_id": plan["case_id"], "condition": condition, "repeat_ordinal": repeat,
        "plan_sha256": digest(plan_path), "terminal_path": str((attempt/"terminal.json").resolve()),
        "terminal_sha256": digest(attempt/"terminal.json"), "world_sha256": digest(attempt/"flight/world.json"),
        "score_recount_exact": True, "score": score, "retained_attempts": len(attempts)}
