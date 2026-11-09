"""Source-bound append-only admission for the exact original development slot037."""
import argparse
import json
from pathlib import Path

from inspect_original import (
    BUNDLE, EPISODE, PLAN_SHA, PROJECT, RESERVATION, RUN, SIDE, SIDE_SHA, TERMINAL_SHA, TREE_SHA,
    checked, exact037, previous, read, require,
)

FINDING = "g4-dev-v4-slot037-external-cleanup-disposition-v1"
FACTS = BUNDLE / "evidence.v1/binding.json"
FACTS_SHA = "1d69331ce39b41a92d73d39808449439c5761c21b3fa48b9010eb70110379406"
PROPOSAL = BUNDLE / "proposal.v2.json"
ACCEPTANCE = BUNDLE / "acceptance.v1.json"
TARGET = {"order_index": 37, "case_id": "dev-L2-00", "condition": "B3-M3",
          "repeat_ordinal": 1, "attempt_ordinal": 0, "reservation": RESERVATION}
DISPOSITION = "source_bound_external_process_after_known_settlement"


def receipt_snapshot(directory, reservation):
    roots = (directory, previous.operation_directory(reservation),
             RUN / "qualification/local-model-episodes" / reservation)
    files, directories = {}, []
    for root in roots:
        require(root.is_dir() and not root.is_symlink(), "one original closed directory required")
        for path in (root, *root.rglob("*")):
            require(not path.is_symlink(), "symbolic alias in original closed input")
            if path.is_dir():
                directories.append(str(path.relative_to(PROJECT)))
            else:
                ref = checked(path)
                files[ref["path"]] = ref["sha256"]
    return {"directories": sorted(directories), "files": files}


def prepare():
    facts_ref = checked(FACTS, FACTS_SHA)
    source_ref = checked(Path(__file__))
    snapshot = receipt_snapshot(EPISODE, RESERVATION)
    facts = read(FACTS)
    for ref in [*facts["files"], *facts["captures"]]:
        checked(PROJECT / ref["path"], ref["sha256"])
    _, terminal, _, y, proof, refs = exact037(previous.owner())
    require(proof == facts["original037_external"] and receipt_snapshot(EPISODE, RESERVATION) == snapshot,
            "exact037 proof or input set changed")
    proposal = {"schema": "inspection-dev-external-cleanup-proposal-v1", "target": TARGET,
        "disposition": DISPOSITION, "new_resource_operations": 0,
        "plan_canonical_sha256": PLAN_SHA, "original_terminal_sha256": TERMINAL_SHA,
        "original_tree_sha256": TREE_SHA, "observer": checked(SIDE, SIDE_SHA), "facts": facts_ref,
        "source": source_ref, "source_attribution": proof,
        "Y": y, "evidence_sha256": previous.canonical_digest(terminal["evidence"]),
        "original_inputs": snapshot, "refs": refs,
        "limits": "Exact037 only; original unknown/Y=null/retry=false retained; no new attempt or G4/formal authority."}
    for ref in [*facts["files"], *facts["captures"], *refs, facts_ref, source_ref]:
        checked(PROJECT / ref["path"], ref["sha256"])
    previous.durable_once(PROPOSAL, proposal)
    return checked(PROPOSAL)


def verify():
    acceptance_ref = checked(ACCEPTANCE)
    value = read(ACCEPTANCE)
    require(set(value) == {"schema", "finding_id", "proposal", "review"}
            and value["schema"] == "inspection-dev-external-cleanup-acceptance-v1"
            and value["finding_id"] == FINDING, "exact037 independently accepted disposition required")
    admission = previous.admission_module()
    p, r = [admission.Artifact.parse(value[key]) for key in ("proposal", "review")]
    require(RUN / p.path == PROPOSAL, "only the exact037 proposal may be used")
    checked(PROPOSAL, p.sha256)
    review_path = RUN / r.path
    checked(review_path, r.sha256)
    proposal, review = read(PROPOSAL), read(review_path)
    require(review["schema"] == "inspection-dev-external-cleanup-review-v1"
            and review["decision"] == "accept_exact037_only" and review["open_findings"] == []
            and type(review["reviewer"]) is str and review["reviewer"] not in ("", "/root", "codex-primary")
            and review["proposal_sha256"] == p.sha256
            and review["disposition_source_sha256"] == proposal["source"]["sha256"],
            "matching independent exact037 acceptance required")
    facts_ref = checked(FACTS, FACTS_SHA)
    facts = read(FACTS)
    source_ref = checked(Path(__file__), proposal["source"]["sha256"])
    require(proposal["schema"] == "inspection-dev-external-cleanup-proposal-v1"
            and proposal["disposition"] == DISPOSITION and proposal["target"] == TARGET
            and proposal["plan_canonical_sha256"] == PLAN_SHA and proposal["original_terminal_sha256"] == TERMINAL_SHA
            and proposal["original_tree_sha256"] == TREE_SHA and proposal["new_resource_operations"] == 0
            and proposal["facts"] == facts_ref and proposal["source"] == source_ref
            and proposal["observer"] == checked(SIDE, SIDE_SHA),
            "exact037 proposal is not bound to the original evidence")
    owner = previous.owner()
    binding_refs = [p, r, admission.Artifact(str(FACTS.relative_to(RUN)), FACTS_SHA),
                    admission.Artifact(str(Path(__file__).relative_to(RUN)), source_ref["sha256"])]
    native_sha = admission.require_native_binding(owner, FINDING, binding_refs)
    require(receipt_snapshot(EPISODE, RESERVATION) == proposal["original_inputs"],
            "accepted original closed input tree differs")
    plan, terminal, tree, y, proof, refs = exact037(owner)
    require(proof == proposal["source_attribution"] and refs == proposal["refs"] and y == proposal["Y"]
            and previous.canonical_digest(terminal["evidence"]) == proposal["evidence_sha256"],
            "accepted original attribution or scientific recount differs")
    require(receipt_snapshot(EPISODE, RESERVATION) == proposal["original_inputs"],
            "accepted original inputs changed during verification")
    for ref in [*facts["files"], *facts["captures"], *refs, facts_ref, source_ref]:
        checked(PROJECT / ref["path"], ref["sha256"])
    checked(ACCEPTANCE, acceptance_ref["sha256"])
    checked(PROPOSAL, p.sha256)
    checked(review_path, r.sha256)
    return y, {"episode_receipt": str(EPISODE / "attempt-00/terminal.json"), "episode_receipt_sha256": TERMINAL_SHA,
               "cache_reused": True, "actual_attempts": 1, "plan_sha256": PLAN_SHA, "reservation": RESERVATION,
               "cleanup_disposition": {"acceptance": acceptance_ref, "proposal": value["proposal"],
                   "review": value["review"], "native_request_sha256": native_sha,
                   "original_status_retained": "unknown_process_cleanup", "external_attribution": facts_ref}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    args = parser.parse_args()
    if args.mode == "prepare":
        print(json.dumps(prepare(), allow_nan=False))
    else:
        y, evidence = verify()
        print(json.dumps({"role": "same original037; no physical/model effect", "Y": y, "evidence": evidence},
                         allow_nan=False))
