"""Prospective v4 disposition after the original terminal and known settlement."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

from disposition037 import receipt_snapshot
from inspect_original import (
    BUNDLE, FLIGHTS, PROJECT, RUN, checked, context, inspect_attempt, previous, read, require,
)
from launchd_provenance import match_sidecar, utc

OUTPUT = PROJECT / "logs/inspection-eight-condition-pilot.v4.continuation.v3"
POLICY = BUNDLE / "prospective-policy.v2.json"
POLICY_REVIEW = RUN / "review/evidence/pilot-v4-external-policy-v2/review.json"
POLICY_FINDING = "g4-dev-v4-exact-external-job-admission-v1"
OBSERVER = OUTPUT / "cleanup-observations"


def policy_refs(owner):
    policy_ref, review_ref = checked(POLICY), checked(POLICY_REVIEW)
    policy, review = read(POLICY), read(POLICY_REVIEW)
    require(owner["active_round"] and owner["task_stop"] is None
            and policy["schema"] == "inspection-v4-external-cleanup-policy-v1"
            and policy["sdk_view"] == str(OUTPUT.relative_to(PROJECT))
            and policy["first_original_order_index"] == 38
            and review["schema"] == "inspection-v4-external-cleanup-policy-review-v1"
            and review["decision"] == "accept_prospective_development_only"
            and type(review["reviewer"]) is str and review["reviewer"] not in ("", "/root", "codex-primary")
            and review["open_findings"] == [] and review["policy_sha256"] == policy_ref["sha256"],
            "matching independent prospective v4 review required")
    sources = policy["sources"]
    directory = Path(__file__).resolve().parent
    required = {str((directory / name).relative_to(PROJECT)) for name in (
        "admit_external.py", "continue_pilot.py", "launchd_provenance.py", "inspect_original.py", "disposition037.py")}
    require(set(sources) == required, "exact prospective executable source set required")
    refs = [checked(PROJECT / path, sha) for path, sha in sources.items()]
    admission = previous.admission_module()
    native = admission.require_native_binding(owner, POLICY_FINDING, [
        admission.Artifact(str(path.relative_to(RUN)), ref["sha256"])
        for path, ref in ((POLICY, policy_ref), (POLICY_REVIEW, review_ref))])
    return {"policy": policy_ref, "review": review_ref, "native_request_sha256": native, "sources": refs}


def check_policy(refs):
    for ref in (refs["policy"], refs["review"], *refs["sources"]):
        checked(PROJECT / ref["path"], ref["sha256"])


def freeze_json(path, value):
    expected = hashlib.sha256((json.dumps(value, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    previous.durable_once(path, value)
    return checked(path, expected)


def _boot():
    output = subprocess.check_output(["/usr/sbin/sysctl", "-n", "kern.bootsessionuuid"], timeout=3, text=True).strip()
    require(re.fullmatch("[0-9A-F]{8}(-[0-9A-F]{4}){3}-[0-9A-F]{12}", output) is not None,
            "native current boot identity unavailable")
    return output


def _events(pid, sidecar, destination, verify_match):
    require(type(pid) is int and pid > 1, "exact PID required for the OS query")
    close = utc(sidecar["close_started_at"])
    observed = datetime.now(timezone.utc)
    require(timedelta(0) <= observed - close <= timedelta(seconds=90),
            "prospective lookup is restricted to the just-settled attempt")
    # log show parses start/end as local time; its JSON includes explicit offsets.
    start = (close - timedelta(seconds=2)).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    end = (close + timedelta(seconds=10)).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    predicate = (f'process == "launchd" AND (eventMessage == "xpcproxy spawned with pid {pid}" OR '
                 f'eventMessage == "Successfully spawned python[{pid}] because semaphore" OR '
                 f'eventMessage == "removing child: pid/{pid}")')
    argv = ["/usr/bin/log", "show", "--style", "json", "--info", "--start", start, "--end", end,
            "--predicate", predicate]
    boot = _boot()
    destination.mkdir()
    failure = None
    for i in range(2):
        log_path = destination / f"launchd-query-{i}.json"
        queried = datetime.now(timezone.utc).isoformat()
        proc = subprocess.run(argv, capture_output=True, timeout=8)
        require(proc.returncode == 0 and not proc.stderr and len(proc.stdout) <= 1024 * 1024
                and _boot() == boot, "bounded same-boot launchd query did not complete")
        with log_path.open("xb") as stream:
            stream.write(proc.stdout)
            stream.flush()
            os.fsync(stream.fileno())
        ref = checked(log_path, hashlib.sha256(proc.stdout).hexdigest())
        meta = {"argv": argv, "queried_at_utc": queried, "after_utc": datetime.now(timezone.utc).isoformat(),
                "boot": boot, "returncode": proc.returncode, "events": ref}
        query = freeze_json(destination / f"query-{i}.json", meta)
        try:
            match = verify_match(json.loads(proc.stdout))
            require(match["python_spawn"]["bootUUID"] == boot, "launchd events are not on the live host boot")
        except (KeyError, TypeError, ValueError) as error:
            failure = error
            if i == 0:
                time.sleep(.4)
            continue
        checked(log_path, ref["sha256"])
        checked(PROJECT / query["path"], query["sha256"])
        return match, query
    raise ValueError("exact launchd lifetime proof unavailable after two bounded queries") from failure


def reconcile_new(directory, plan, prefix, sidecar_ref, refs):
    """No writes to attempt/model/owner files; return the frozen original grade."""
    require(re.fullmatch(r"dev-all8-pilot-v4-\d{3}", prefix) is not None, "original v4 prefix required")
    index = int(prefix[-3:])
    require(38 <= index < 108 and Path(directory) == FLIGHTS / (
        f"{plan['case_id']}-{plan['condition']}-{plan['repeat_ordinal']}"), "new original development slot required")
    check_policy(refs)
    directory = Path(directory)
    reservation = prefix + "-a0"
    snap = receipt_snapshot(directory, reservation)
    owner = previous.owner()
    require(policy_refs(owner) == refs, "prospective native policy changed")
    original, terminal, tree, y, originals = inspect_attempt(directory, 0, reservation, owner)
    require(plan == original and terminal["status"] == "unknown_process_cleanup"
            and terminal["admission_error"] == "process cleanup requires source-bound reconciliation"
            and terminal["Y"] is None and terminal["retry_allowed"] is False,
            "only a complete first attempt blocked solely by original cleanup may continue")
    require(tree["status"] == "unknown" and tree["remaining_owned"] == []
            and str(tree["root_pid"]) in tree["owned_births"]
            and len(tree["unclaimed_new_python_orphans"]) == 1,
            "original cleanup must leave only one unclaimed process")
    pid, = tree["unclaimed_new_python_orphans"]
    side_path = PROJECT / sidecar_ref["path"]
    require(side_path.parent == OBSERVER, "sidecar belongs to a different SDK view")
    side = checked(side_path, sidecar_ref["sha256"])
    sidecar = read(side_path)
    verify_match = lambda events: match_sidecar(sidecar, tree, context(plan, prefix), previous.OBSERVER_SHA,
                                                 events, os.getuid())
    base = OUTPUT / "external-cleanup-dispositions"
    require(not base.is_symlink(), "symbolic proof directory")
    base.mkdir(exist_ok=True)
    target = base / f"{index:03d}"
    match, query = _events(pid, sidecar, target, verify_match)
    # Later observation is an additional safety check, never the ownership proof.
    rows = previous._inventory()
    still_live = [key for key, old in tree["owned_births"].items() if int(key) in rows
                  and rows[int(key)]["born"] == old["born"] and not rows[int(key)]["status"].startswith("Z")]
    require(not still_live and pid not in rows, "later process state is still unresolved")
    query_path = PROJECT / query["path"]
    checked(query_path, query["sha256"])
    query_meta = read(query_path)
    raw = query_meta["events"]
    checked(PROJECT / raw["path"], raw["sha256"])
    checked(query_path, query["sha256"])
    attempt = directory / "attempt-00"
    bound = {ref["path"]: ref["sha256"] for ref in originals}
    def original_ref(path):
        return checked(path, bound[str(path.relative_to(PROJECT))])
    terminal_ref = original_ref(attempt / "terminal.json")
    proof = {"schema": "inspection-v4-external-cleanup-disposition-v1", "order_index": index,
             "reservation": reservation, "original_terminal": terminal_ref,
             "original_tree": original_ref(attempt / "process-tree.json"),
             "plan_sha256": previous.canonical_digest(plan), "original_status": terminal["status"],
             "policy": refs, "observer": side, "query": query, "attribution": match,
             "later_observation_utc": datetime.now(timezone.utc).isoformat(), "later_owned_live": still_live,
             "later_external_pid_present": False, "original_inputs": snap, "original_refs": originals,
             "new_resource_operations_for_disposition": 0, "Y": y}
    record = target / "proof.json"
    proof_ref = freeze_json(record, proof)
    return y, {"episode_receipt": str(attempt / "terminal.json"), "episode_receipt_sha256": terminal_ref["sha256"],
               "cache_reused": False, "actual_attempts": 1, "plan_sha256": proof["plan_sha256"],
               "reservation": reservation, "cleanup_disposition": {
                   "proof": proof_ref, "original_status_retained": terminal["status"],
                   "native_request_sha256": refs["native_request_sha256"]}}


def check_disposition(directory, plan, prefix, y, evidence, refs):
    """Last caller check after observer and SDK companion validation."""
    disposition = evidence["cleanup_disposition"]
    ref = disposition["proof"]
    index = int(prefix[-3:])
    path = OUTPUT / "external-cleanup-dispositions" / f"{index:03d}" / "proof.json"
    require(checked(path, ref["sha256"]) == ref, "disposition belongs to a different callback")
    proof = read(path)
    side, = evidence["passive_close_observations"]
    require(proof["schema"] == "inspection-v4-external-cleanup-disposition-v1"
            and proof["order_index"] == index and proof["reservation"] == evidence["reservation"] == prefix + "-a0"
            and proof["plan_sha256"] == evidence["plan_sha256"] == previous.canonical_digest(plan)
            and proof["original_status"] == disposition["original_status_retained"] == "unknown_process_cleanup"
            and proof["Y"] == y and proof["policy"] == refs
            and proof["original_terminal"]["sha256"] == evidence["episode_receipt_sha256"]
            and PROJECT / proof["original_terminal"]["path"] == Path(evidence["episode_receipt"])
            and all(proof["observer"][key] == side[key] for key in ("path", "sha256"))
            and refs["native_request_sha256"] == disposition["native_request_sha256"],
            "sealed disposition does not match this original callback")
    query = proof["query"]
    query_path = PROJECT / query["path"]
    require(query_path.parent == path.parent and query_path.name in ("query-0.json", "query-1.json"),
            "launchd query belongs to a different callback")
    checked(query_path, query["sha256"])
    raw = read(query_path)["events"]
    require(PROJECT / raw["path"] == path.parent / query_path.name.replace("query-", "launchd-query-"),
            "launchd events belong to a different callback")
    require(receipt_snapshot(Path(directory), proof["reservation"]) == proof["original_inputs"],
            "original attempt input set changed")
    check_policy(refs)
    for bound in (*proof["original_refs"], proof["original_terminal"], proof["original_tree"],
                  proof["observer"], raw, query, ref):
        checked(PROJECT / bound["path"], bound["sha256"])
