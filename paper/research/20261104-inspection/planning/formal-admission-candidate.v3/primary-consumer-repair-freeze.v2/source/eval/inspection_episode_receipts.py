"""Durable local-flight receipts and conservative recovery for G4 development.

This wraps real CLI runs. It does not own paper state, assign a scientific
acceptance status, recover paid requests, or turn unknown attempts into zeroes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from datetime import datetime, timezone


class UnresolvedEpisode(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir(path):
    path = Path(path)
    if not path.exists():
        _mkdir(path.parent)
        try:
            path.mkdir()
        except FileExistsError:
            pass
        _sync_directory(path.parent)


def durable_once(path, value):
    """Exclusive create + fsync; readers require the complete final JSON."""
    path = Path(path)
    data = (json.dumps(value, indent=2, allow_nan=False)+"\n").encode()
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    _sync_directory(path.parent)


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise UnresolvedEpisode(f"missing or incomplete receipt: {path}") from error


def validate_flight(out, plan, returncode):
    """Admit a completed *local classical* flight, including policy Y=0."""
    formal = None
    if "formal_dispatch" in plan:
        from eval.inspection_formal_dispatch import admit_episode_plan
        formal = admit_episode_plan(plan, Path(__file__).resolve().parents[1])
    out = Path(out)
    def optional(name):
        try:
            return _read(out/name)
        except UnresolvedEpisode:
            return None
    run, world = optional("run.json"), optional("world.json")
    reasons = []
    if (returncode != 0 or run is None or world is None
            or run.get("status") != "completed" or world.get("status") != "completed"):
        reasons.append("worker_operational_failure")
    if run is not None and (run.get("source_hashes") != plan["source_hashes"]
            or run.get("dependency_hashes") != plan["dependency_hashes"]
            or run.get("sources_unchanged") is not True or run.get("dependencies_unchanged") is not True):
        reasons.append("source_or_dependency_mismatch")
    case_hash = digest(out/"private-case.json") if (out/"private-case.json").is_file() else None
    if case_hash is not None and case_hash != plan["case_file_sha256"]:
        reasons.append("input_or_condition_mismatch")
    if run is not None and (run.get("private_case_sha256") != plan["case_file_sha256"]
            or case_hash != plan["case_file_sha256"] or run.get("private_case_unchanged") is not True
            or run.get("policy") != plan["condition"]
            or run.get("runtime_thread_environment") != plan.get("thread_environment")):
        reasons.append("input_or_condition_mismatch")
    world_hash = digest(out/"world.json") if (out/"world.json").is_file() else None
    if world is not None and world.get("private_case_file_sha256") != plan["case_file_sha256"]:
        reasons.append("world_binding_mismatch")
    if run is not None and world is not None and run.get("world_sha256") != world_hash:
        reasons.append("world_binding_mismatch")
    if run is not None and world is not None and (run.get("requested_seconds") != 600 or run.get("realtime") is not True
            or world.get("clock_mode") != "wallclock_1x" or world.get("simulated_seconds") != 600
            or world.get("physics_steps_executed") != 144000
            or world.get("physics_hz") != 240 or world.get("control_hz") != 120):
        reasons.append("incomplete_or_wrong_clock")
    lag = world.get("max_clock_lag_seconds") if world is not None else None
    if world is not None and (type(lag) not in (int, float) or not math.isfinite(lag) or lag > plan["max_clock_lag_seconds"]):
        reasons.append("clock_lag_exceeded")
    score = world.get("score", {}) if world is not None else {}
    if world is not None and (any(type(score.get(k)) not in (int, float) or not math.isfinite(score[k])
            or not 0 <= score[k] <= 1 for k in ("C", "S", "Q", "Y"))
            or any(type(score.get(k)) is not int or score[k] not in (0, 1) for k in ("H", "E", "SR"))
            or not math.isclose(score.get("Y", -1), score.get("H", 0)*score.get("E", 0)*score.get("Q", 0), abs_tol=1e-12)):
        reasons.append("invalid_score")
    if formal is not None and (run is None or world is None
            or run.get("formal_dispatch_binding") != formal.binding()
            or world.get("formal_input_binding") != formal.case_binding()):
        reasons.append("formal_dispatch_binding_mismatch")
    evidence = {"run_sha256": digest(out/"run.json") if (out/"run.json").is_file() else None,
                "world_sha256": world_hash, "copied_case_sha256": case_hash, "returncode": returncode,
                "max_clock_lag_seconds": lag, "failure_reasons": reasons}
    return (None if reasons else score["Y"]), evidence


def _bindings(attempt):
    names = ("intent.json", "started.json", "stdout.log", "flight/run.json", "flight/world.json", "flight/private-case.json")
    return {name: digest(attempt/name) if (attempt/name).is_file() else None for name in names}


def _cleanup(process):
    # Also terminate orphaned worker/planner processes after a supervisor exits.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute_local_episode(directory, plan, command_factory, *, timeout_seconds=780):
    """Rebuild a new SDK envelope from verified cache, never replay an unknown.

    At most one retry is allowed, with identical plan and only for a known
    worker failure or drift. Source/input/binding/score faults block immediately.
    A crash between spawn and terminal fsync remains unknown on the next call.
    """
    if (plan.get("condition") not in ("B0", "B1", "B2")
            or plan.get("max_clock_lag_seconds") != .1 or type(plan.get("max_attempts")) is not int
            or plan["max_attempts"] != 2 or type(plan.get("case_id")) is not str or not plan["case_id"]
            or type(plan.get("repeat_ordinal")) is not int
            or plan["repeat_ordinal"] != 0):
        raise ValueError("this adapter admits only preregistered local classical episodes")
    if "formal_dispatch" in plan:
        from eval.inspection_formal_dispatch import admit_episode_plan
        admit_episode_plan(plan, Path(__file__).resolve().parents[1])
    elif plan["case_id"].startswith(("main-", "holdout-")):
        raise ValueError("formal classical input requires typed dispatch admission")
    directory = Path(directory)
    _mkdir(directory)
    plan_hash = canonical_digest(plan)
    with (directory/".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise UnresolvedEpisode("another supervisor owns this episode") from error
        plan_path = directory/"plan.json"
        if plan_path.exists():
            if canonical_digest(_read(plan_path)) != plan_hash:
                raise UnresolvedEpisode("episode identity changed; preserve old attempts")
        else:
            durable_once(plan_path, plan)
        for ordinal in range(plan["max_attempts"]):
            if "formal_dispatch" in plan:
                admit_episode_plan(plan, Path(__file__).resolve().parents[1])
            attempt = directory/f"attempt-{ordinal:02d}"
            terminal_path = attempt/"terminal.json"
            if attempt.exists():
                if not terminal_path.exists():
                    raise UnresolvedEpisode(f"unknown attempt requires source-bound reconciliation: {attempt}")
                terminal = _read(terminal_path)
                if terminal.get("plan_sha256") != plan_hash or terminal.get("record_hashes") != _bindings(attempt):
                    raise UnresolvedEpisode("terminal belongs to a different plan")
                if terminal.get("stdout_sha256") != terminal["record_hashes"]["stdout.log"]:
                    raise UnresolvedEpisode("terminal output hash differs")
                intent = _read(attempt/"intent.json")
                if intent.get("plan_sha256") != plan_hash or intent.get("attempt_ordinal") != ordinal:
                    raise UnresolvedEpisode("attempt intent differs")
                if not terminal.get("not_started"):
                    if _read(attempt/"started.json").get("plan_sha256") != plan_hash:
                        raise UnresolvedEpisode("attempt start differs")
                if terminal.get("status") == "completed":
                    if terminal.get("not_started"):
                        raise UnresolvedEpisode("completed receipt cannot also claim no dispatch")
                    value, evidence = validate_flight(attempt/"flight", plan, terminal["returncode"])
                    if value is None or evidence != terminal["evidence"] or terminal.get("Y") != value:
                        raise UnresolvedEpisode("cached flight no longer matches its terminal receipt")
                    return value, {"episode_receipt": str(terminal_path.resolve()),
                                   "episode_receipt_sha256": digest(terminal_path), "cache_reused": True,
                                   "actual_attempts": ordinal+1, "plan_sha256": plan_hash}
                if terminal.get("status") != "known_operational_failure" or not terminal.get("retry_allowed"):
                    raise UnresolvedEpisode("nonretryable or uncertain prior attempt")
                # Known failures are retained; continue only within the frozen cap.
                continue
            attempt.mkdir()
            _sync_directory(directory)
            command = command_factory(attempt/"flight")
            if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
                raise ValueError("command factory must return an argv list")
            durable_once(attempt/"intent.json", {"role": "intent_not_proof_of_start", "plan_sha256": plan_hash,
                "argv": command, "attempt_ordinal": ordinal, "utc": datetime.now(timezone.utc).isoformat()})
            started = time.monotonic()
            with (attempt/"stdout.log").open("xb") as output:
                _sync_directory(attempt)
                try:
                    process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except OSError as error:
                    output.flush()
                    os.fsync(output.fileno())
                    durable_once(terminal_path, {"status": "known_operational_failure", "plan_sha256": plan_hash,
                        "returncode": None, "retry_allowed": True, "spawn_error": str(error), "not_started": True,
                        "stdout_sha256": digest(attempt/"stdout.log"),
                        "record_hashes": _bindings(attempt)})
                    continue
                try:
                    durable_once(attempt/"started.json", {"role": "observed_process_spawn_not_flight_or_model_start",
                        "pid": process.pid, "plan_sha256": plan_hash, "utc": datetime.now(timezone.utc).isoformat()})
                    returncode = process.wait(timeout=timeout_seconds)
                finally:
                    # No automatic success/zero/retry after a supervisor interruption.
                    _cleanup(process)
                output.flush()
                os.fsync(output.fileno())
            value, evidence = validate_flight(attempt/"flight", plan, returncode)
            retry = value is None and set(evidence["failure_reasons"]) <= {"worker_operational_failure", "clock_lag_exceeded", "incomplete_or_wrong_clock"}
            durable_once(terminal_path, {"status": "completed" if value is not None else "known_operational_failure",
                "plan_sha256": plan_hash, "returncode": returncode, "Y": value, "evidence": evidence,
                "stdout_sha256": digest(attempt/"stdout.log"), "elapsed_seconds": time.monotonic()-started,
                "record_hashes": _bindings(attempt),
                "retry_allowed": retry, "retry_budget_remaining": plan["max_attempts"]-ordinal-1})
            if value is not None:
                return value, {"episode_receipt": str(terminal_path.resolve()),
                               "episode_receipt_sha256": digest(terminal_path), "cache_reused": False,
                               "actual_attempts": ordinal+1, "plan_sha256": plan_hash}
            if not retry:
                raise UnresolvedEpisode("source/input/score defect blocks the episode; see terminal receipt")
        raise UnresolvedEpisode("finite symmetric infrastructure retry budget exhausted")
