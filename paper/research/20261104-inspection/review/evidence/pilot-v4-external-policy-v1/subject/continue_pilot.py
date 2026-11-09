"""Continue the same v4 pilot after the three exact historical dispositions."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from admit_external import OBSERVER, OUTPUT, check_policy, freeze_json, policy_refs, reconcile_new
import disposition037
from inspect_original import FLIGHTS, OLD, PREVIOUS_ACCEPTANCE, PROJECT, RUN, checked, context, previous, read, require
from eval.inspection_episode_receipts import UnresolvedEpisode

MANIFESTS = ("source-hashes.json", "source-files.json", "dependency-hashes.json", "input-hashes.json",
             "episode-order.json", "sampler-seeds.json")


def run_observed(execute, observer, directory, plan, command, kwargs, refs):
    prefix = kwargs["reservation_prefix"]
    info = context(plan, prefix)
    has_attempt = (Path(directory) / "attempt-00").exists()
    with observer.CloseObserver(info) as observation:
        try:
            y, evidence = execute(directory, plan, command, **kwargs)
        except UnresolvedEpisode as original:
            if has_attempt or str(original) != "unknown or source/input/score/causal fault blocks this LLM slot":
                raise
            # The original supervisor has already persisted its unknown and settled usage.
            records = observation.check()
            require(len(records) == 1, "prospective disposition requires exactly one original close")
            y, evidence = reconcile_new(directory, plan, prefix, records[0], refs)
        records = observation.check()
        require(len(records) == (0 if evidence["cache_reused"] else evidence["actual_attempts"]),
                "one fixed sidecar per newly executed original attempt required")
    if records:
        evidence = {**evidence, "passive_close_observations": records}
    elif has_attempt:
        # Keep actual past sidecars with cached callbacks; no new scan is implied.
        raw_path = OLD / "llm/raw" / f"{plan['case_id']}-{plan['condition']}-{plan['repeat_ordinal']}.json"
        prior = read(raw_path)
        old_evidence = prior["evidence"]
        require(prior["status"] == "completed" and prior["value"] == y
                and all(old_evidence[key] == evidence[key] for key in ("episode_receipt_sha256", "plan_sha256",
                                                                      "reservation", "actual_attempts")),
                "strict cache differs from its original SDK callback")
        if past := old_evidence.get("passive_close_observations"):
            require(len(past) == 1, "one original observer sidecar required")
            side = PROJECT / past[0]["path"]
            require(side.parent == OLD / "cleanup-observations", "original sidecar escaped its SDK view")
            checked(side, past[0]["sha256"])
            original = read(side)
            tree = read(Path(evidence["episode_receipt"]).with_name("process-tree.json"))
            require(original["context"] == info and original["original_result"] == tree
                    and original["adapter_source_sha256"] == previous.OBSERVER_SHA
                    and original["original_exception"] is None and original["observation_errors"] == [],
                    "cached original observer sidecar differs")
            checked(side, past[0]["sha256"])
            evidence = {**evidence, "passive_close_observations": past}
    return y, evidence


def run(argv):
    expected = {
        "run": str(RUN), "suite": str(RUN / "qualification/dev-cases.v2"),
        "selection": str(RUN / "qualification/classic-six-root-cohort.v3/selection.json"),
        "selection-sha256": "75f57580f0f5371d91a741fd4f9573e52192d019e9730335123a7ae67897a32e",
        "sdk-root": str(RUN / "runtime/autoresearch-state-reader-v1"),
        "flight-receipts": str(FLIGHTS), "out": str(OUTPUT), "schedule-seed": "2026110641",
        "sampler-seed": "2026110642", "reservation-prefix": "dev-all8-pilot-v4",
    }
    require(argv == [arg for key, value in expected.items() for arg in ("--" + key, value)]
            and not OUTPUT.exists() and not OUTPUT.is_symlink(),
            "only the exact original v4 queue in the new declared SDK view is admitted")
    live = previous.live_frozen_sources()
    refs = policy_refs(previous.owner())
    old = previous.older()
    dispositions = {
        24: (old.EPISODE, old.PLAN_CANON, lambda: old.verify(previous.OLDER_ACCEPTANCE)),
        30: (previous.EPISODE, previous.PLAN_CANON, lambda: previous.verify(PREVIOUS_ACCEPTANCE)),
        37: (disposition037.EPISODE, disposition037.PLAN_SHA, disposition037.verify),
    }
    initial = {index: verify() for index, (_, _, verify) in dispositions.items()}
    observer = previous.observer_module()
    # Rebind only the in-process destination; the accepted observer source and original scans stay fixed.
    observer.OUTPUT = OBSERVER
    pilot = PROJECT / "eval/run_inspection_eight_condition_pilot.py"
    spec = importlib.util.spec_from_file_location("_inspection_v4_original_pilot", pilot)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    execute = original.execute_development_llm_episode
    companion = OUTPUT / "recovery-adapter.json"
    companion_value = {"role": "same v4 episode queue; original unknowns and physical attempts retained",
        "sdk_predecessors": [str(path.relative_to(PROJECT)) for path in (previous.PILOT, OLD)],
        "reconciled": {str(index): evidence for index, (_, evidence) in initial.items()},
        "prior_manifests": [checked(OLD / name) for name in MANIFESTS],
        "passive_observer": previous.observer_refs(),
        "prospective_policy": refs, "observer_output": str(OBSERVER.relative_to(PROJECT)),
        "physical_denominator": 108, "cache_barrier_before_order": 38}
    companion_ref = None

    def fixed_metadata():
        check_policy(refs)
        require(observer.OUTPUT == OBSERVER and previous.observer_refs() == companion_value["passive_observer"],
                "bound observer sources or destination changed")
        checked(companion, companion_ref["sha256"])

    def measured(directory, plan, command, **kwargs):
        nonlocal companion_ref
        if companion_ref is None:
            for ref in companion_value["prior_manifests"]:
                checked(OUTPUT / Path(ref["path"]).name, ref["sha256"])
                checked(PROJECT / ref["path"], ref["sha256"])
            companion_ref = freeze_json(companion, companion_value)
        fixed_metadata()
        order = int(kwargs["reservation_prefix"][-3:])
        if order in dispositions:
            episode, plan_sha, verify = dispositions[order]
            require(Path(directory) == episode and previous.canonical_digest(plan) == plan_sha
                    and Path(kwargs["run"]) == RUN and kwargs["reservation_prefix"] == f"dev-all8-pilot-v4-{order:03d}",
                    "historical disposition cannot bind another callback")
            answer = verify()
            require(answer == initial[order], "historical disposition changed within replay")
            if order == 37:
                previous.durable_once(OUTPUT / "checkpoint.request", {
                    "role": "pause before any new effect after exact38-cache reconstruction"})
        else:
            answer = run_observed(execute, observer, directory, plan, command, kwargs, refs)
        fixed_metadata()
        return answer

    for ref in live:
        checked(PROJECT / ref["path"], ref["sha256"])
    original.execute_development_llm_episode = measured
    old_argv = sys.argv
    try:
        sys.argv = [str(pilot), *argv]
        original.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    supplied = parser.parse_args().args
    run(supplied[1:] if supplied[:1] == ["--"] else supplied)
