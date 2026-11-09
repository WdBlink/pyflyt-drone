"""Recount the two last v4 attempts and bind exact037 external evidence."""
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import zipfile

from launchd_provenance import match_lifetime, match_sidecar

RUN = Path(__file__).resolve().parents[3]
PROJECT = RUN.parents[2]
BUNDLE = Path(__file__).resolve().parents[1]
FLIGHTS = PROJECT / "logs/inspection-eight-condition-dev-flights.v4"
OLD = PROJECT / "logs/inspection-eight-condition-pilot.v4.continuation.v2"
SOURCE = RUN / "recovery/pilot-v4-cleanup-030.v1/source/continue_pilot.py"
SOURCE_SHA = "4c78d5b130f27f15f34483f7f88bf5a7c348cef2399d272faae3c63ba3502a8f"
PREVIOUS_ACCEPTANCE = SOURCE.parents[1] / "acceptance.v1.json"
EPISODE = FLIGHTS / "dev-L2-00-B3-M3-1"
PLAN_SHA = "94545338d18d94fb66ad278578bdf9a86a05abd5ff6fb3508f9ed0792c53f232"
TERMINAL_SHA = "2183b93170d7362349bcfe33e452809d989ab8f0a212d9848436e0e4c4a6aaf0"
TREE_SHA = "a4f78cb975844c88ad90e7fb65f92b151110d92b20e7bf90b03b534ceff36ba2"
SIDE = OLD / "cleanup-observations/root-60623-48545371959708.json"
SIDE_SHA = "6edd60553b11cf9032bf67e7a5f05f22e05550769c9856c624c3e2b5fa25cee8"
RESERVATION = "dev-all8-pilot-v4-037-a0"

import hashlib

if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != SOURCE_SHA:
    raise ValueError("previous accepted reader changed")
spec = importlib.util.spec_from_file_location("_inspection_v4_previous030", SOURCE)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
checked, read, require = previous.checked, previous.read, previous.require


def context(plan, prefix):
    return {**{key: plan[key] for key in ("case_id", "condition", "repeat_ordinal")},
            "plan_sha256": previous.canonical_digest(plan), "reservation_prefix": prefix}


def inspect_attempt(directory, ordinal, reservation, owner):
    """Read original complete flight, model, dispatch and known receipt bindings."""
    attempt = directory / f"attempt-{ordinal:02d}"
    operation = previous.operation_directory(reservation)
    resource = RUN / "qualification/local-model-episodes" / reservation
    settled = (resource / "complete-attempt.zip", resource / "usage-receipt.json",
               RUN / "resource-requests" / (reservation + ".settle.json"))
    plan, terminal, tree, intent, start, returned, ack = [read(path) for path in (
        directory / "plan.json", *(attempt / name for name in ("terminal.json", "process-tree.json",
        "intent.json", "started.json", "process-return.json", "settle-owner.json")))]
    key = previous.canonical_digest(plan)
    require(terminal["plan_sha256"] == intent["plan_sha256"] == start["plan_sha256"]
            == returned["plan_sha256"] == key
            and intent["reservation"] == ack["reservation"] == reservation
            and ack["status"] == "known" and intent["attempt_ordinal"] == ordinal
            and start["pid"] == tree["root_pid"] and terminal["returncode"] == returned["returncode"] == 0
            and returned["stdout_sha256"] == previous.digest(attempt / "stdout.log")
            and not (directory / f"attempt-{ordinal + 1:02d}").exists(),
            "original zero-return dispatch and terminal proof differ")
    refs = [checked(directory / "plan.json"), checked(attempt / "terminal.json")]
    payload, model = previous._bindings(attempt, operation)
    require((payload, model) == (terminal["payload_hashes"], terminal["model_payload_hashes"])
            and set(terminal["settlement_file_hashes"]) == {str(p) for p in settled},
            "original attempt, model or settlement file set differs")
    refs.extend(checked(attempt / name, sha) for name, sha in payload.items())
    refs.extend(checked(operation / name, sha) for name, sha in model.items())
    refs.extend(checked(p, terminal["settlement_file_hashes"][str(p)]) for p in settled)
    with zipfile.ZipFile(settled[0]) as archive:
        require(archive.testzip() is None, "original complete archive failed CRC")
    y, evidence = previous.validate_llm_episode(attempt / "flight", operation, plan, reservation, 0)
    usage = evidence["model_meter"]["usage"]
    require(type(y) in (int, float) and 0 <= y <= 1 and evidence == terminal["evidence"]
            and terminal["physical_validation"]["failure_reasons"] == [],
            "original physical, frozen grade, causality or model recount differs")
    ledger = owner["usage_ledger"][reservation]
    require(ledger["status"] == "known" and ledger["receipt_sha256"] == previous.digest(settled[1])
            and ledger["reported_usage"] == usage and owner["frozen_plan"] == plan["owner_frozen_plan"]
            and owner["task_stop"] is None,
            "actual sole-owner known settlement differs")
    purpose = json.loads(ledger["purpose"])
    require(purpose["phase"] == "development" and purpose["episode_plan_sha256"] == key
            and purpose["attempt_ordinal"] == ordinal
            and all(purpose[k] == plan[k] for k in ("case_id", "condition", "repeat_ordinal",
                    "model_registry_sha256", "model_receipt_sha256")), "native reservation identity differs")
    for ref in refs:
        checked(PROJECT / ref["path"], ref["sha256"])
    return plan, terminal, tree, y, refs


def exact037(owner):
    refs = [checked(EPISODE / "attempt-00/terminal.json", TERMINAL_SHA),
            checked(EPISODE / "attempt-00/process-tree.json", TREE_SHA), checked(SIDE, SIDE_SHA)]
    plan, terminal, tree, y, payload = inspect_attempt(EPISODE, 0, RESERVATION, owner)
    require(previous.canonical_digest(plan) == PLAN_SHA and terminal["status"] == "unknown_process_cleanup"
            and terminal["Y"] is None and terminal["retry_allowed"] is False
            and tree["root_pid"] == 60623 and tree["unclaimed_new_python_orphans"] == [65758]
            and set(tree["owned_births"]) == {"60623", "60625", "60626", "60660", "60721"}
            and y == .3314065510597302
            and terminal["evidence"]["model_meter"]["usage"] == {"calls": 22, "tokens": 160664},
            "only exact original037 cleanup-only unknown is eligible")
    events = BUNDLE / "launchd.v1/slot037.json"
    refs.append(checked(events, "a8e1f13ba8d39995ea9b7c33e0dddb6a54a6ff27f4903e5f0daf6c8ad973d5d4"))
    proof = match_sidecar(read(SIDE), tree, context(plan, RESERVATION[:-3]),
                          previous.OBSERVER_SHA, json.loads(events.read_bytes()), 501)
    require(len(proof["original_scan_rows"]) == 2, "both original037 rows are required")
    return plan, terminal, tree, y, proof, [*refs, *payload]


def capture():
    result_dir = BUNDLE / "evidence.v1"
    require(not result_dir.exists(), "never overwrite frozen incident evidence")
    owner = previous.owner()
    refs = previous.live_frozen_sources() + previous.observer_refs()
    refs += [checked(path) for path in (SOURCE, Path(__file__),
              Path(__file__).with_name("launchd_provenance.py"), *sorted((BUNDLE / "launchd.v1").iterdir()))]
    plan, terminal, tree, y, proof, more = exact037(owner)
    refs += more
    current_rows = previous._inventory()
    remaining = sorted(pid for pid, row in tree["owned_births"].items() if int(pid) in current_rows
                       and current_rows[int(pid)]["born"] == row["born"]
                       and not current_rows[int(pid)]["status"].startswith("Z"))
    later = {"at": datetime.now(timezone.utc).isoformat(), "original_owned_remaining": remaining,
             "unclaimed_pid_present": 65758 in current_rows, "scope": "later only; no process actions"}
    require(not remaining and not later["unclaimed_pid_present"], "original identities remain unresolved")
    abort = {}
    for group, denominator, count in (("classic", 18, 6), ("llm", 90, 32), ("root-means", 48, 0)):
        path = OLD / group / "results.json"
        result = read(path)
        require(result["reported_denominator"] == denominator and result["attempted_count"] == count
                and result["execution_status"] == "failed", "original abort denominator differs")
        abort[group] = dict(Counter(row["status"] for row in result["runs"]))
        refs.append(checked(path))
    require(abort == {"classic": {"completed": 6, "stopped": 12},
            "llm": {"completed": 31, "failed": 1, "stopped": 58}, "root-means": {"stopped": 48}},
            "original abort statuses differ")
    require(len(list((OLD / "order-receipts").glob("*.json"))) == 37
            and len(list(FLIGHTS.rglob("attempt-*"))) == 38
            and not list((RUN / "resource-requests").glob("dev-all8-pilot-v4-038*")),
            "physical attempts or next original slot differ")
    slot36 = read(OLD / "order-receipts/000036.json")
    raw_path = OLD / slot36["raw"]
    raw = read(raw_path)
    plan36, terminal36, tree36, y36, more = inspect_attempt(
        FLIGHTS / "dev-L2-00-B3-M1-2", 0, "dev-all8-pilot-v4-036-a0", owner)
    close_ref, = raw["evidence"]["passive_close_observations"]
    close_path = PROJECT / close_ref["path"]
    close = read(close_path)
    require(raw["status"] == terminal36["status"] == "completed" and raw["attempted"] is True
            and raw["value"] == terminal36["Y"] == y36 and tree36["status"] == "closed"
            and close["original_result"] == tree36 and close["observation_errors"] == []
            and close["original_exception"] is None
            and close["context"] == context(plan36, "dev-all8-pilot-v4-036"), "original036 callback differs")
    refs += more + [checked(raw_path), checked(OLD / "order-receipts/000036.json"),
                    checked(close_path, close_ref["sha256"])]
    cases = {}
    for index, episode, actual, original in ((36, FLIGHTS / "dev-L2-00-B3-M1-2", y36, terminal36),
                                              (37, EPISODE, y, terminal)):
        flight = episode / "attempt-00/flight"
        world = json.loads((flight / "world.json").read_bytes())
        last = json.loads((flight / "progress.jsonl").read_text().splitlines()[-1])
        ids = {r["response"]["report_id"] for r in world["reports"] if r["response"].get("status") == "accepted"}
        require(world["score"] == original["evidence"]["frozen_score_recount"]["score"]
                and actual == world["score"]["Y"] and last["t"] == 600 and last["reports"] == len(ids)
                and world["simulated_seconds"] == 600 and world["physics_steps_executed"] == 144000
                and world["clock_lag_over_0p1_seconds_ticks"] == 0 and world["physics_runtime"]["errors"] == [],
                "original complete clock/score/report recount differs")
        cases[str(index)] = {"terminal_status": original["status"], "terminal_Y": original["Y"],
             "score_before_cleanup_gate": world["score"], "meter": original["evidence"]["model_meter"]["usage"],
             "causal_joins": len(original["evidence"]["causal_request_joins"]),
             "max_lag_seconds": world["max_clock_lag_seconds"], "p95_lag_seconds": world["clock_lag_p95_seconds"],
             "report_audit_entries": len(world["reports"]), "accepted_unique_ids": len(ids)}
    historical = []
    for item in json.loads((BUNDLE / "launchd.v1/file-times.json").read_bytes()):
        index = item["order_index"]
        refs.append(checked(PROJECT / item["path"], item["sha256"]))
        historical.append({"order_index": index, "scope": item["scope"],
            "match": match_lifetime(json.loads((BUNDLE / f"launchd.v1/slot{index:03d}.json").read_bytes()), 501,
                          item["unclaimed_pid"], (item["file_birth_utc"], item["file_mtime_utc"]))})
    review = RUN / "review/evidence/pilot-v4-foreign-cleanup-evidence-v1/binding.json"
    refs.append(checked(review, "9fc8deb230a1fcc0f9e724e24fbdd9dd14b5c5dd2c7139032ca6c09505412643"))
    refs.append(checked(RUN / "review/pilot-v4-foreign-cleanup-evidence-review.v1.md",
                        "003206f7f3aa555b4ac7406833566395e3336aaba8c583bac16ef541ba4d2645"))
    status_path = Path("/tmp/inspection-v4-third-cleanup-native-status.v1.json")
    status = json.loads(status_path.read_bytes())
    budget = status["budget"]
    require(budget["known_usage"] == {"calls": 1314, "tokens": 8480064, "cost": {}}
            and budget["reserved"] == {"calls": 0, "tokens": 0, "cost": {}}
            and budget["unknown_usage"] == budget["unresolved_usage"] == status["stale"] == [],
            "abort owner status differs")
    for ref in refs:
        checked(PROJECT / ref["path"], ref["sha256"])
    result_dir.mkdir()
    captures = []
    for original, name in ((EPISODE / "attempt-00/terminal.json", "original037-terminal.json"),
            (EPISODE / "attempt-00/process-tree.json", "original037-tree.json"), (SIDE, "original037-sidecar.json"),
            (close_path, "original036-sidecar.json"), (status_path, "native-status.json"),
            (Path("/tmp/inspection-pilot-v4-continuation.v2.stdout"), "aborted-supervisor.stdout")):
        target = result_dir / name
        with target.open("xb") as stream:
            stream.write(original.read_bytes())
        captures.append(checked(target))
    result = {"role": "recount and attribution evidence only; original037 still unknown",
              "at": datetime.now(timezone.utc).isoformat(), "callbacks": 37, "attempted_slots": 38, "denominator": 108,
              "sdk_abort_counts": abort, "last_slots": cases, "original037_external": proof,
              "later_observation": later, "tree_filetime_corroboration_only": historical,
              "native_usage": budget["known_usage"], "new_resource_operations": 0,
              "files": list({ref["path"]: ref for ref in refs}.values()), "captures": captures}
    out = result_dir / "binding.json"
    previous.durable_once(out, result)
    print(json.dumps({"binding": checked(out), "last_slots": cases, "sdk_abort_counts": abort,
                      "later_observation": later}, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    capture()
