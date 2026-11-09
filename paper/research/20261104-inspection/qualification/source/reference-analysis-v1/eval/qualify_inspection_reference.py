"""Audit a saved physical reference; never run or silently repair an episode.

Adds complete-step, sensor-opportunity and task-chain checks to the original
quality/clock calibration. G1 composition and G2/G4 remain separate decisions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.analyze_inspection_qualification import analyze
from eval.inspection_tracking import recount_tracking


def recount_sensor_opportunities(world):
    """Separate raw detection misses, association suppression and background."""
    audit = world["sensor_audit"]
    emitted = {r["evidence_id"]: r for r in audit}
    if len(emitted) != len(audit):
        raise ValueError("duplicate emitted sensor evidence")
    referenced = set()
    groups = defaultdict(lambda: dict(visible_opportunities=0, raw_detection_misses=0,
                                     emitted_measurements=0, association_suppressed=0))
    keys = set()
    for row in world["sensor_opportunities"]:
        key = row["t"], row["drone_id"], row["entity_key"]
        if key in keys or type(row["missed"]) is not bool:
            raise ValueError("invalid/duplicate visible opportunity")
        keys.add(key)
        summary = groups[row["mode"]]
        summary["visible_opportunities"] += 1
        summary["raw_detection_misses"] += row["missed"]
        evidence = row["emitted_evidence_id"]
        if evidence is not None:
            actual = emitted[evidence]
            if (row["missed"] or evidence in referenced or any(actual[k] != row[k]
                    for k in ("t", "drone_id", "entity_key", "mode"))):
                raise ValueError("emitted evidence does not match its raw opportunity")
            referenced.add(evidence)
            summary["emitted_measurements"] += 1
        elif not row["missed"]:
            if row["mode"] != "inspect":
                raise ValueError("unexplained suppression of a wide-frame detection")
            summary["association_suppressed"] += 1
    frames = set()
    backgrounds = dict(wide_frame_trials=0, raw_triggers=0, emitted_measurements=0)
    for row in world["background_trials"]:
        key = row["t"], row["drone_id"]
        if key in frames or type(row["triggered"]) is not bool:
            raise ValueError("invalid/duplicate background frame trial")
        frames.add(key)
        backgrounds["wide_frame_trials"] += 1
        backgrounds["raw_triggers"] += row["triggered"]
        evidence = row["emitted_evidence_id"]
        if evidence is not None:
            actual = emitted[evidence]
            if (not row["triggered"] or evidence in referenced or actual["entity_key"] is not None
                    or actual["mode"] != "survey"
                    or any(actual[k] != row[k] for k in ("t", "drone_id"))):
                raise ValueError("background evidence does not match its frame trial")
            referenced.add(evidence)
            backgrounds["emitted_measurements"] += 1
    if referenced != set(emitted):
        raise ValueError("emitted evidence lacks its acquisition opportunity")
    for values in groups.values():
        values["raw_miss_fraction"] = values["raw_detection_misses"]/values["visible_opportunities"]
    backgrounds["raw_trigger_fraction"] = (backgrounds["raw_triggers"]/backgrounds["wide_frame_trials"]
                                              if backgrounds["wide_frame_trials"] else None)
    return dict(sensor_modes=dict(groups), background=backgrounds,
                emitted_evidence_reconciled=len(referenced),
                limitations=["conditional raw misses differ from final association suppression",
                             "observed fractions do not certify real vision or population reliability"])


def qualify(directory, contract_path, source_root=None):
    run = json.loads((directory/"run.json").read_text())
    world = json.loads((directory/"world.json").read_text())
    contract = json.loads(contract_path.read_text())
    component = analyze([directory], contract_path, source_root)
    checks = {
        "source_bound_quality_and_clock": component["reference_and_clock_evidence_pass"],
        "normal_sensor_point_thresholds": component["sensor_point_thresholds_pass"],
        "recorded_dependency_closure_unchanged": bool(run.get("dependency_hashes"))
            and run.get("dependencies_unchanged") is True,
    }
    errors = []
    try:
        checks["recorded_dependencies_match_local_files"] = bool(run.get("dependency_hashes")) and all(
            hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
            for name, digest in run.get("dependency_hashes", {}).items())
        counts = recount_tracking(world, expected_xy_tube_m=contract["executor_qualification"]["xy_tracking_tube_m"])
        summary = world["execution_tracking_240hz"]
        checks["complete_actual_240hz_coverage"] = (
            world["physics_steps_executed"] == world["safety"]["physics_steps"]
            == world["safety"]["physics_audits"] == counts["physics_samples"] == 600*240
            and world["physics_hz"] == 240 and world["control_hz"] == 120
            and all(summary[k] == value for k, value in counts.items()))
        checks["physical_tracking_and_seams"] = (counts["horizontal_tube_breach_steps"]
            == counts["altitude_envelope_breach_steps"] == summary["seam_capture_failures"] == 0)
        opportunities = recount_sensor_opportunities(world)
        checks["opportunity_denominators_present"] = bool(opportunities["sensor_modes"])
        checks["background_trials_reconciled"] = opportunities["background"]["wide_frame_trials"] > 0
        accepted = {row["name"] for row in world["tool_calls"] if row["response"]["status"] == "accepted"}
        jobs = world["jobs"]
        checks["search_confirm_report_observe_exit_chain"] = (
            {"assign_survey", "inspect_candidate", "submit_or_revise_report", "observe_track",
             "enter_safe_exit"} <= accepted
            and any(j["kind"] == "inspect" and j["status"] == "succeeded" for j in jobs)
            and any(j["kind"] == "observe" and "service_started_at" in j["params"] for j in jobs)
            and {j["drone_id"] for j in jobs if j["kind"] == "exit" and j["status"] == "succeeded"}
            == set(range(4)) and bool(world["coverage_evidence"]))
        checks["infrastructure_clean"] = not world["execution_infrastructure_errors"]
    except (KeyError, TypeError, ValueError, OSError, zlib.error) as error:
        counts, opportunities = None, None
        errors.append(type(error).__name__+": "+str(error))
        checks["complete_saved_evidence"] = False
    return {"role": "dev_reference_G3_component_audit_not_overall_platform_acceptance",
            "passed": not errors and all(checks.values()), "checks": checks, "errors": errors,
            "run_sha256": hashlib.sha256((directory/"run.json").read_bytes()).hexdigest(),
            "world_sha256": hashlib.sha256((directory/"world.json").read_bytes()).hexdigest(),
            "analysis_sources_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (Path(__file__), ROOT/"eval/analyze_inspection_qualification.py",
                          ROOT/"eval/inspection_tracking.py", ROOT/"eval/inspection_artifacts.py")},
            "original_quality_clock_calibration": component, "tracking_recount": counts,
            "sensor_opportunity_recount": opportunities,
            "limitations": ["requires independent review of frozen sources, evidence and reference trajectory",
                            "no full state physics replay is inferred from the error trace",
                            "G1 source composition, G2 protocol semantics and G4 development freeze remain separate",
                            "no model latency, policy comparison or human domain validation is implied"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output")
    result = qualify(args.run, args.contract, args.source_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: result[k] for k in ("passed", "checks", "errors")}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
