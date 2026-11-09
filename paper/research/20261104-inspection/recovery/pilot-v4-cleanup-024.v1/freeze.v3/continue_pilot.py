"""One append-only development recovery; never rewrite or redispatch slot024."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile

RUN = Path(__file__).resolve().parents[3]
PROJECT = RUN.parents[2]
BUNDLE = Path(__file__).resolve().parents[1]
PILOT = PROJECT / "logs/inspection-eight-condition-pilot.v4"
EPISODE = PROJECT / "logs/inspection-eight-condition-dev-flights.v4/dev-L3-01-B3-M1-2"
ATTEMPT = EPISODE / "attempt-00"
RESERVATION = "dev-all8-pilot-v4-024-a0"
FINDING = "g4-dev-v4-slot024-late-cleanup-disposition-v1"
PLAN_CANON = "0ab1224d422a4bee4e3934b22f34a2a36d8bfa21f1886004b8ba54be8a38b4ae"
TERMINAL_SHA = "86f0189e000815abc250df4092bc264bbb7fa79dcca65a032edec88d68c3f6cd"
TREE_SHA = "ae19d0f443f3093bcee565e45214e23ff63e3f366b71507fc588c7ff64fa6291"
ADMISSION = RUN / "planning/formal-admission-candidate.v3/response-resource-freeze.v2/source/eval/inspection_formal_admission.py"
ADMISSION_SHA = "3272e19e4ec4d3ad4802ff5a939feea8cf0a258f6777ee60a5b21ef2d758c5fa"
sys.path.insert(0, str(PROJECT))
from eval.inspection_episode_receipts import canonical_digest, digest, durable_once
from eval.inspection_llm_episode_receipts import read as original_read, validate_llm_episode
from eval.inspection_model_service import operation_directory
from eval.inspection_process_cleanup import _inventory
from eval.run_inspection_llm_episode import _bindings


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(Path(path).stat().st_size <= 16 * 1024 * 1024, "bounded recovery JSON required")
    return original_read(path)


def checked(path, expected=None):
    path = Path(path).absolute()
    require(path.is_relative_to(PROJECT) and ".." not in path.parts, "project file required")
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), "symbolic alias in recovery input")
        if parent == PROJECT:
            break
    require(path.is_file() and path.stat().st_nlink == 1, "one regular recovery input required")
    sha = digest(path)
    require(expected is None or sha == expected, "recovery input changed: " + str(path))
    return {"path": str(path.relative_to(PROJECT)), "sha256": sha}


def admission_module():
    checked(ADMISSION, ADMISSION_SHA)
    name = "_inspection_v4_recovery_native_reader"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ADMISSION)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def owner():
    return admission_module().native_owner_reader(RUN / "runtime/autoresearch-document-snapshots-v2")(RUN)


def quiet_observation(tree):
    rows = _inventory()
    live = [pid for pid, old in tree["owned_births"].items()
            if int(pid) in rows and rows[int(pid)]["born"] == old["born"]
            and not rows[int(pid)]["status"].startswith("Z")]
    orphans = {str(pid): row for pid, row in rows.items()
               if row["uid"] == os.getuid() and row["parent"] == 1
               and "python" in row["command"].lower() and not row["status"].startswith("Z")}
    require(not live and not orphans and 26337 not in rows, "cleanup remains unresolved")
    return {"at": datetime.now(timezone.utc).isoformat(), "remaining_original_live": live,
            "current_same_user_python_orphans": orphans, "original_unclaimed_pid_present": False,
            "scope": "later quiet observation; not original ownership attribution or historical closed"}


def target(plan, terminal, tree):
    require(canonical_digest(plan) == PLAN_CANON, "only the original slot024 plan is admissible")
    require(terminal["status"] == "unknown_process_cleanup" and terminal["Y"] is None
            and terminal["retry_allowed"] is False and terminal["returncode"] == 0
            and terminal["plan_sha256"] == PLAN_CANON, "not the original cleanup-only unknown")
    require(tree["status"] == "unknown" and tree["root_pid"] == 21710
            and tree["remaining_owned"] == [] and tree["unclaimed_new_python_orphans"] == [26337]
            and set(tree["owned_births"]) == {"21710", "21712", "21713", "21757", "21768"},
            "other cleanup uncertainty cannot use this disposition")
    require(not (EPISODE / "attempt-01").exists(), "original single attempt required")


def live_frozen_sources():
    pins = {"source-hashes.json": "abd4b12aba595746e9cc97eb44ff359fc64da1fa79c64eb160e720b0dc873560",
            "dependency-hashes.json": "d90279e0d9a3c26fb2fe5070c40c495cea0145f277808c32e01f7957f69e7d20",
            "input-hashes.json": "7c2d224ef66d1e4bddc18c7f323074bf245b0600eafde622f2d57c9c99a2c0dd"}
    refs = []
    for name, pin in pins.items():
        refs.append(checked(PILOT / name, pin))
        for path, sha in read(PILOT / name).items():
            checked(path, sha)
    texts = read(PILOT / "source-files.json")
    hashes = read(PILOT / "source-hashes.json")
    import hashlib
    require(set(texts) == set(hashes) and all(isinstance(texts[key], str)
            and hashlib.sha256(texts[key].encode()).hexdigest() == sha for key, sha in hashes.items()),
            "original frozen source text differs")
    refs.append(checked(PILOT / "source-files.json"))
    return refs


def closed_inputs():
    """Same entry-type/directory rule as the reviewed primary receipt snapshot."""
    directories, files = set(), {}
    resource = RUN / "qualification/local-model-episodes" / RESERVATION
    for base in (EPISODE, operation_directory(RESERVATION), resource):
        require(base.is_dir() and not base.is_symlink(), "closed input directory required")
        for path in (base, *base.rglob("*")):
            require(not path.is_symlink(), "closed input cannot use a symbolic alias")
            if path.is_dir():
                directories.add(str(path.relative_to(PROJECT)))
            elif path.is_file():
                ref = checked(path)
                files[ref["path"]] = ref["sha256"]
            else:
                raise ValueError("closed input must be a regular file or directory")
    return {"directories": sorted(directories), "files": files}


def inspect_original():
    """Only original validators recount quality/resources; no effects or source changes."""
    plan = read(EPISODE / "plan.json")
    checked(ATTEMPT / "terminal.json", TERMINAL_SHA)
    checked(ATTEMPT / "process-tree.json", TREE_SHA)
    terminal, tree = read(ATTEMPT / "terminal.json"), read(ATTEMPT / "process-tree.json")
    target(plan, terminal, tree)
    files = [checked(EPISODE / "plan.json"), checked(ATTEMPT / "terminal.json", TERMINAL_SHA),
             checked(ATTEMPT / "process-tree.json", TREE_SHA), checked(Path(__file__)),
             checked(ADMISSION, ADMISSION_SHA)]
    files.extend(live_frozen_sources())
    operation = operation_directory(RESERVATION)
    payload, model_payload = _bindings(ATTEMPT, operation)
    require(payload == terminal["payload_hashes"] and model_payload == terminal["model_payload_hashes"],
            "original attempt/model file set or bytes changed")
    for relative, sha in payload.items():
        files.append(checked(ATTEMPT / relative, sha))
    for relative, sha in model_payload.items():
        files.append(checked(operation / relative, sha))
    for path, sha in terminal["settlement_file_hashes"].items():
        files.append(checked(path, sha))
    resource = RUN / "qualification/local-model-episodes" / RESERVATION
    with zipfile.ZipFile(resource / "complete-attempt.zip") as archive:
        require(archive.testzip() is None, "original archive CRC failed")
    y, evidence = validate_llm_episode(ATTEMPT / "flight", operation, plan, RESERVATION, 0)
    require(type(y) in (int, float) and 0 <= y <= 1 and evidence == terminal["evidence"],
            "original frozen quality/actor/meter recount differs")
    current = owner()
    entry = current["usage_ledger"][RESERVATION]
    receipt = resource / "usage-receipt.json"
    require(entry["status"] == "known" and entry["receipt_sha256"] == digest(receipt)
            and entry["reported_usage"] == {"calls": 25, "tokens": 166000}
            and current["frozen_plan"] == plan["owner_frozen_plan"] and current["task_stop"] is None,
            "actual sole-owner known settlement differs")
    purpose = json.loads(entry["purpose"])
    require(purpose["episode_plan_sha256"] == PLAN_CANON and purpose["attempt_ordinal"] == 0
            and purpose["case_id"] == plan["case_id"] and purpose["condition"] == plan["condition"]
            and purpose["repeat_ordinal"] == 2, "native reservation identity differs")
    return plan, tree, y, evidence, files


def prepare(proposal):
    before = closed_inputs()
    plan, tree, y, evidence, files = inspect_original()
    proposal = Path(proposal).absolute()
    require(proposal.parent == BUNDLE, "new proposal must be in this recovery bundle")
    value = {"schema": "inspection-dev-late-cleanup-proposal-v1", "status": "proposed_not_accepted",
             "disposition": "late_cleanup_observed_complete_not_original_closed",
             "target": {"order_index": 24, "case_id": "dev-L3-01", "condition": "B3-M1",
                        "repeat_ordinal": 2, "attempt_ordinal": 0, "reservation": RESERVATION},
             "original_plan_canonical_sha256": PLAN_CANON, "original_terminal_sha256": TERMINAL_SHA,
             "original_tree_sha256": TREE_SHA, "Y": y,
             "evidence_sha256": canonical_digest(evidence), "files": files,
             "observations": [quiet_observation(tree)], "closed_inputs": before, "new_resource_operations": 0,
             "limits": "Original unknown/ownership gap retained. One development cache only; no G4/formal permission."}
    live_frozen_sources()
    require(closed_inputs() == before, "original closed input changed during proposal validation")
    for ref in files:
        checked(PROJECT / ref["path"], ref["sha256"])
    durable_once(proposal, value)
    return {"proposal": str(proposal.relative_to(RUN)), "sha256": digest(proposal), "Y": y,
            "status": "proposal_only_no_acceptance"}


def verify(acceptance):
    acceptance = Path(acceptance).absolute()
    acceptance_ref = checked(acceptance)
    value = read(acceptance)
    require(set(value) == {"schema", "proposal", "review", "finding_id"}
            and value["schema"] == "inspection-dev-late-cleanup-acceptance-v1"
            and value["finding_id"] == FINDING, "explicit late-cleanup acceptance required")
    module = admission_module()
    refs = [module.Artifact.parse(value[key]) for key in ("proposal", "review")]
    for ref in refs:
        checked(RUN / ref.path, ref.sha256)
    review = read(RUN / refs[1].path)
    require(review.get("schema") == "inspection-dev-late-cleanup-review-v1"
            and review.get("decision") == "accept_late_cleanup_only"
            and review.get("proposal_sha256") == refs[0].sha256
            and type(review.get("reviewer")) is str
            and review["reviewer"] not in ("", "codex-primary", "/root"),
            "actual independent scoped acceptance required")
    proposal = read(RUN / refs[0].path)
    require(proposal["schema"] == "inspection-dev-late-cleanup-proposal-v1"
            and proposal["disposition"] == "late_cleanup_observed_complete_not_original_closed"
            and proposal["new_resource_operations"] == 0, "wrong recovery proposal")
    require(proposal["original_plan_canonical_sha256"] == PLAN_CANON
            and proposal["original_terminal_sha256"] == TERMINAL_SHA
            and proposal["original_tree_sha256"] == TREE_SHA
            and proposal["target"] == {"order_index": 24, "case_id": "dev-L3-01", "condition": "B3-M1",
                                       "repeat_ordinal": 2, "attempt_ordinal": 0, "reservation": RESERVATION}
            and proposal["observations"] and all(row["remaining_original_live"] == []
                and row["current_same_user_python_orphans"] == {}
                and row["original_unclaimed_pid_present"] is False for row in proposal["observations"]),
            "wrong target or missing accepted late-cleanup observation")
    for ref in proposal["files"]:
        checked(PROJECT / ref["path"], ref["sha256"])
    require(closed_inputs() == proposal["closed_inputs"], "accepted closed input tree changed")
    source = module.Artifact(str(Path(__file__).relative_to(RUN)), digest(Path(__file__)))
    current = owner()
    request_sha = module.require_native_binding(current, FINDING, [*refs, source])
    checked(ATTEMPT / "terminal.json", TERMINAL_SHA)
    checked(ATTEMPT / "process-tree.json", TREE_SHA)
    plan, terminal, tree = [read(path) for path in
                            (EPISODE / "plan.json", ATTEMPT / "terminal.json", ATTEMPT / "process-tree.json")]
    target(plan, terminal, tree)
    payload, model_payload = _bindings(ATTEMPT, operation_directory(RESERVATION))
    require(payload == terminal["payload_hashes"] and model_payload == terminal["model_payload_hashes"],
            "accepted original file set changed")
    # Accepted immutable evidence remains readable after later source installs;
    # unrelated future processes cannot undo the recorded cleanup observation.
    evidence = terminal["evidence"]
    y = evidence["frozen_score_recount"]["score"]["Y"]
    require(y == proposal["Y"] and type(y) in (int, float) and 0 <= y <= 1
            and canonical_digest(evidence) == proposal["evidence_sha256"],
            "accepted original validator result changed")
    resource = RUN / "qualification/local-model-episodes" / RESERVATION / "usage-receipt.json"
    entry = current["usage_ledger"][RESERVATION]
    require(entry["status"] == "known" and entry["receipt_sha256"] == digest(resource)
            and entry["reported_usage"] == {"calls": 25, "tokens": 166000}, "accepted settlement changed")
    for ref in proposal["files"]:
        checked(PROJECT / ref["path"], ref["sha256"])
    require(closed_inputs() == proposal["closed_inputs"], "accepted input changed after validation")
    for ref in refs:
        checked(RUN / ref.path, ref.sha256)
    checked(acceptance, acceptance_ref["sha256"])
    return y, {"episode_receipt": str((ATTEMPT / "terminal.json").resolve()),
               "episode_receipt_sha256": TERMINAL_SHA, "cache_reused": True, "actual_attempts": 1,
               "plan_sha256": PLAN_CANON, "reservation": RESERVATION,
               "cleanup_disposition": {"acceptance": acceptance_ref, "proposal": value["proposal"],
                                       "review": value["review"], "native_request_sha256": request_sha,
                                       "original_status_retained": "unknown_process_cleanup"}}


def run_continuation(acceptance, argv):
    """Delegate all scientific operations to the byte-identical v4 pilot."""
    y, recovery_evidence = verify(acceptance)
    live_frozen_sources()
    parser = argparse.ArgumentParser(add_help=False)
    for flag in ("run", "out", "flight-receipts", "reservation-prefix", "schedule-seed", "sampler-seed"):
        parser.add_argument("--" + flag, required=True)
    args, _ = parser.parse_known_args(argv)
    require(Path(args.run).resolve() == RUN and Path(args.flight_receipts).resolve() == EPISODE.parent
            and args.reservation_prefix == "dev-all8-pilot-v4"
            and args.schedule_seed == "2026110641" and args.sampler_seed == "2026110642",
            "only the same v4 cohort/order/samplers/reservations may continue")
    output = Path(args.out).absolute()
    require(output.parent == PROJECT / "logs" and not output.exists(), "fresh project-log SDK view required")
    original = PROJECT / "eval/run_inspection_eight_condition_pilot.py"
    spec = importlib.util.spec_from_file_location("_inspection_v4_original_pilot", original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    execute = module.execute_development_llm_episode
    source_sha = digest(Path(__file__))
    context = {"role": "same v4 queue continuation, not new independent episodes",
               "adapter_source": checked(Path(__file__)), "original_pilot": checked(original),
               "original_sdk_view": str(PILOT.relative_to(PROJECT)),
               "original_episode_order_sha256": digest(PILOT / "episode-order.json"),
               "original_sampler_seeds_sha256": digest(PILOT / "sampler-seeds.json"),
               "late_cleanup_evidence": recovery_evidence, "physical_denominator": 108,
               "reuse_before_new_effects": "24 strict caches plus one independently reconciled cache; next new order25"}
    def recovered(directory, plan, command_factory, **kwargs):
        checked(Path(__file__), source_sha)
        companion = output / "recovery-adapter.json"
        if not companion.exists():
            require(digest(output / "episode-order.json") == context["original_episode_order_sha256"]
                    and digest(output / "sampler-seeds.json") == context["original_sampler_seeds_sha256"],
                    "continuation changed original order/samplers")
            durable_once(companion, context)
        require(read(companion) == context, "continuation companion changed")
        if Path(directory).resolve() == EPISODE:
            require(canonical_digest(plan) == PLAN_CANON and Path(kwargs["run"]).resolve() == RUN
                    and kwargs["reservation_prefix"] == "dev-all8-pilot-v4-024", "wrong reconciliation callback")
            actual_y, actual_evidence = verify(acceptance)
            require(actual_y == y and actual_evidence == recovery_evidence, "reconciliation changed during replay")
            durable_once(output / "checkpoint.request", {
                "role": "pause before order25 and any new effect after exact25-cache prefix reconstruction"})
            return actual_y, actual_evidence
        return execute(directory, plan, command_factory, **kwargs)
    module.execute_development_llm_episode = recovered
    previous = sys.argv
    try:
        sys.argv = [str(original), *argv]
        module.main()
    finally:
        sys.argv = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify", "run"))
    parser.add_argument("--proposal", type=Path)
    parser.add_argument("--acceptance", type=Path, default=BUNDLE / "acceptance.v1.json")
    args, forwarded = parser.parse_known_args()
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    if args.mode == "prepare":
        require(args.proposal is not None and not forwarded, "prepare needs one new proposal path")
        print(json.dumps(prepare(args.proposal), allow_nan=False))
    elif args.mode == "verify":
        require(not forwarded, "verify performs no pilot dispatch")
        y, evidence = verify(args.acceptance)
        print(json.dumps({"role": "read-only late-cleanup verification, no effects", "Y": y,
                          "evidence": evidence}, allow_nan=False))
    else:
        run_continuation(args.acceptance, forwarded)


if __name__ == "__main__":
    main()
