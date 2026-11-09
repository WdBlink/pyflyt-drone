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
    phase = expected["phase"]
    if phase not in ("development", "main", "holdout"):
        raise ValueError("declare the actual data role")
    if phase != "development":
        from eval.inspection_formal_dispatch import admit_episode_plan
        admission = admit_episode_plan(expected, Path(__file__).resolve().parents[1])
        if admission.binding()["phase"] != phase:
            raise ValueError("formal measurement phase differs from its admitted namespace")
    elif "formal_dispatch" in expected:
        raise ValueError("formal data cannot be relabeled development")
    plan_path = episode/"plan.json"
    plan = _json(plan_path)
    if phase=='development' and ('formal_dispatch' in plan or plan.get('case_id','').startswith(('main-','holdout-'))):
        raise ValueError('stored formal episode cannot be relabeled development')
    if phase != "development" and plan != {key:value for key,value in expected.items() if key != "phase"}:
        raise ValueError("formal measurement must bind the entire actual episode plan")
    if condition not in ("B0", "B1", "B2"):
        if phase == "development":
            raise ValueError("development LLM measurement uses its original pilot receipt adapter")
        return _measure_completed_llm(episode, plan, phase)
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
                or terminal["record_hashes"]["stdout.log"] is None
                or terminal.get("stdout_sha256") != terminal["record_hashes"]["stdout.log"]):
            raise ValueError("attempt intent or output binding differs")
        if not terminal.get("not_started") and _json(attempt/"started.json").get("plan_sha256") != canonical_digest(plan):
            raise ValueError("attempt start binding differs")
        run_path = attempt/"flight/run.json"
        if run_path.is_file():
            run = _json(run_path)
            embedded = run.get("sources")
            if (not isinstance(embedded, dict) or not embedded
                    or any(not isinstance(source, str) for source in embedded.values())
                    or {name: hashlib.sha256(source.encode()).hexdigest() for name, source in embedded.items()}
                    != plan["source_hashes"]):
                raise ValueError("embedded worker source bytes differ from the frozen manifest")
        if terminal["status"] == "completed":
            if terminal.get("not_started") or ordinal != len(attempts)-1:
                raise ValueError("completed flight must be the final dispatched attempt")
            value, validation = validate_flight(attempt/"flight", plan, terminal["returncode"])
            if value is None or value != terminal["Y"] or validation != terminal["evidence"]:
                raise ValueError("completed physical flight validation differs")
            completed.append((attempt, terminal, value))
        elif terminal["status"] != "known_operational_failure" or not terminal.get("retry_allowed"):
            raise ValueError("nonretryable/unknown attempt blocks measurement")
        else:
            failed_value, validation = validate_flight(attempt/"flight", plan, terminal["returncode"])
            allowed = {"worker_operational_failure", "clock_lag_exceeded", "incomplete_or_wrong_clock"}
            if (failed_value is not None or terminal.get("Y") is not None
                    or not validation["failure_reasons"] or not set(validation["failure_reasons"]) <= allowed
                    or (not terminal.get("not_started") and validation != terminal.get("evidence"))):
                raise ValueError("prior failure is not an independently recomputed retryable infrastructure outcome")
    if len(completed) != 1:
        raise ValueError("exactly one first valid flight within the finite retry policy is required")
    attempt, terminal, value = completed[0]
    world = _json(attempt/"flight/world.json")
    private_sha = hashlib.sha256(json.dumps(world["private_case"], sort_keys=True, allow_nan=False).encode()).hexdigest()
    clean=phase!='development' and admission.binding().get('track')=='clean'
    if clean:
        from eval.inspection_formal_intervention import check_clean_world_identity
        check_clean_world_identity(world,plan['case_sha256'])
    elif private_sha != plan["case_sha256"]:
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


def _measure_completed_llm(episode, plan, phase):
    """Reuse the completed-cache validator with all effect hooks forbidden."""
    from eval.run_inspection_llm_episode import execute_development_llm_episode
    attempts=sorted(episode.glob("attempt-*"))
    if (not attempts or len(attempts)>2 or not (attempts[-1]/"terminal.json").is_file()
            or [path.name for path in attempts]!=[f'attempt-{i:02d}' for i in range(len(attempts))]
            or _json(attempts[-1]/"terminal.json").get("status") != "completed"):
        raise ValueError("measurement requires an already completed finite LLM receipt chain")
    def forbidden(*args,**kwargs):
        raise ValueError("read-only measurement cannot reserve, settle or dispatch")
    dispatch=plan["formal_dispatch"]
    prefix=_json(attempts[0]/"intent.json")["reservation"]
    if not prefix.endswith("-a0"):
        raise ValueError("original attempt reservation identity is malformed")
    y,evidence=execute_development_llm_episode(episode,plan,forbidden,run=dispatch["run"],
        reservation_prefix=prefix[:-3],call_owner=forbidden,read_only=True)
    world_path=attempts[-1]/"flight/world.json";world=_json(world_path)
    from eval.inspection_formal_dispatch import admit_episode_plan
    if admit_episode_plan(plan,Path(__file__).resolve().parents[1]).binding().get('track')=='clean':
        from eval.inspection_formal_intervention import check_clean_world_identity
        check_clean_world_identity(world,plan['case_sha256'])
    if _recount_score(world)!=world["score"]:
        raise ValueError("LLM full report/coverage/denominator score recount differs")
    return y,{**evidence,"role":"genuine_completed_source_bound_llm_episode_not_new_run",
        "phase":phase,"case_id":plan["case_id"],"condition":plan["condition"],
        "repeat_ordinal":plan["repeat_ordinal"],"score_recount_exact":True,
        "world_sha256":digest(world_path),"score":world["score"],"retained_attempts":len(attempts)}
