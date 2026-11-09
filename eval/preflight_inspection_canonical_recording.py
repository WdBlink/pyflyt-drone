"""Synthetic bridge into the exact prospective outer design, never flights."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.inspection_recording import CONDITION_REPEATS, episode_schedule, record_matrix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("preserve prior preflight evidence")
    design_bytes = (args.run/args.design).read_bytes()
    design = json.loads(design_bytes)
    expected = hashlib.sha256(design_bytes).hexdigest()
    # Exact bytes and evaluator closure, independently of the caller's context.
    for name in (args.design, design["evaluator"]["path"], "experiment/evaluator-source-manifest.v2.json"):
        out = args.out/name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes((args.run/name).read_bytes())
    shutil.copytree(args.run/"experiment/evaluator-source.v2", args.out/"experiment/evaluator-source.v2")
    sdk = args.run/"runtime/autoresearch-state-reader-v1"
    sys.path.insert(0, str(sdk.resolve()))
    from lib.experiments import ExperimentRecorder
    from lib.production import json_bytes, write_once
    from lib import science
    def write(root, name, value):
        write_once(root, name, json_bytes(value))
    calls = []
    def synthetic(case, repeat, condition):
        calls.append((case, repeat, condition))
        return (0. if CONDITION_REPEATS[condition] == 1 else repeat/4), {
            "role": "synthetic callback fixture, no aircraft or model executed"}
    context = {"experiment_id": "synthetic-inner-only", "claim_id": "SDK-exact-design-bridge",
        "hypothesis": "SYNTHETIC software preflight, no empirical hypothesis outcome",
        "independent_variable": "fixture", "dependent_variable": "synthetic Y",
        "control_variables": ["no physics", "no model", "full unequal repetition grid"],
        "data_roles": [{"id": "synthetic", "role": "development", "used_for_selection": False}],
        "planned_at": datetime.now(timezone.utc).isoformat()}
    schedule = episode_schedule(design["recording_semantics"]["root_metadata"], 2026110602)
    result = record_matrix(args.out, design["cases"], synthetic, context=context,
        classic_best="B1", execution_order=schedule, recorder_type=ExperimentRecorder,
        write_json=write, frozen_outer_design=args.design)
    saved = json.loads((args.out/"root-means/results.json").read_text())
    if saved["design"] != args.design or saved["design_sha256"] != expected:
        raise AssertionError("canonical outer design was silently regenerated")
    if len(calls) != 1296 or len(set(calls)) != 1296 or len(saved["runs"]) != 576:
        raise AssertionError("full denominator differs")
    if any(row["value"] != (0. if CONDITION_REPEATS[row["condition"]] == 1 else .25) for row in saved["runs"]):
        raise AssertionError("inner repeats were not averaged within root")
    summary = "Synthetic SDK compatibility only. These planned slot names do not denote generated test instances or flown data.\n"
    (args.out/"synthetic-summary.md").write_text(summary)
    packet = {"design": args.design, "results": "root-means/results.json", "summary": "synthetic-summary.md",
        "artifacts": [str(p.relative_to(args.out)) for p in args.out.rglob("*") if p.is_file()],
        "_plan": {"experiment_design_sha256": expected}}
    issues = science.validate("experiment", packet, args.out)
    if issues:
        raise AssertionError(issues)
    report = {"role": "synthetic exact-design SDK compatibility, not main instances, flights or G4 qualification",
        "passed": True, "design_sha256": expected, "design_bytes_unchanged": (args.out/args.design).read_bytes() == design_bytes,
        "actual_model_calls": 0, "actual_flights": 0, "callback_count": len(calls), "outer_rows": len(saved["runs"]),
        "owner_science_experiment_validation": issues, "fixture_classic_B1_is_not_dev_selection": True,
        "recording_summary": result,
        "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (Path(__file__), ROOT/"eval/inspection_recording.py")}}
    (args.out/"report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
