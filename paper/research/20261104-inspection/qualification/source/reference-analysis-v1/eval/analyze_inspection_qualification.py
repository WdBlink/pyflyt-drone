"""Audit recorded dev sensor/clock evidence against the prospective gate.

This reports qualification evidence, not formal policy comparisons. It does not
infer unobserved detection opportunities from detected-only sensor records.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.inspection_artifacts import verify_source_bundle


def analyze(runs, contract_path, source_root=None):
    contract = json.loads(contract_path.read_text())
    groups, attempts, backgrounds = defaultdict(list), [], 0
    for directory in runs:
        run_path, world_path = directory/"run.json", directory/"world.json"
        run, world = json.loads(run_path.read_text()), json.loads(world_path.read_text())
        bound = hashlib.sha256(world_path.read_bytes()).hexdigest() == run.get("world_sha256")
        embedded = all(hashlib.sha256(source.encode()).hexdigest() == run["source_hashes"][name]
                       for name, source in run["sources"].items())
        try:
            verify_source_bundle(run, source_root)
            current = True
        except (OSError, ValueError):
            current = False
        valid = run["status"] == world["status"] == "completed" and run["sources_unchanged"] and bound and embedded and current
        record = {"directory": str(directory), "world_sha256": run.get("world_sha256"),
                  "source_bound_completed": valid, "clock_mode": world.get("clock_mode"),
                  "simulated_seconds": world.get("simulated_seconds")}
        if not valid:
            record["admitted"] = False
            attempts.append(record)
            continue
        score = world["score"]
        block = run.get("coordinator_block_probe")
        record.update(
            admitted=True,
            reference_quality_pass=(score["Y"] >= contract["G3"]["reference_minimum_Y"]
                                    and score["static_recall"] >= contract["G3"]["reference_minimum_static_recall"]
                                    and score["H"] == score["E"] == 1),
            realtime_clock_pass=(world["clock_mode"] == "wallclock_1x" and world["simulated_seconds"] == 600
                                 and world["max_clock_lag_seconds"] <= contract["G3"]["maximum_clock_lag_seconds"]),
            coordinator_wait_pass=(block is not None and block["displacement_m"] > 1
                                   and abs(block["sim_advance_seconds"]-block["actual_wait_seconds"])
                                   <= contract["G3"]["maximum_wait_advance_error_seconds"]),
            score={k: score[k] for k in ("C", "S", "D", "Q", "H", "E", "Y", "SR", "static_recall")},
            max_clock_lag_seconds=world.get("max_clock_lag_seconds"),
            clock_lag_p95_seconds=world.get("clock_lag_p95_seconds"))
        attempts.append(record)
        delivered = {o["evidence_id"]: o for o in world["delivered_observations"]}
        for row in world["sensor_audit"]:
            if row["entity_key"] is None:
                backgrounds += 1
                continue
            covariance = delivered.get(row["evidence_id"], {}).get("covariance_xy")
            degraded = bool(run["events"] and row["drone_id"] == 2 and 300 <= row["t"] < 345)
            key = row["mode"]+("_degraded" if degraded else "_normal")
            dx = row["estimate_xy"][0]-row["truth_xy"][0]
            dy = row["estimate_xy"][1]-row["truth_xy"][1]
            groups[key].append({"error_squared": dx*dx+dy*dy,
                                "correct": row["predicted_class"] == row["true_class"],
                                "normal_distractor": row["true_class"] == "normal",
                                "false_candidate": row["true_class"] == "normal" and row["predicted_class"] != "normal",
                                "normalized_squared": dx*dx/covariance[0]+dy*dy/covariance[1] if covariance else None})
    summary = {}
    for key, rows in groups.items():
        count = len(rows)
        normalized = [r["normalized_squared"] for r in rows if r["normalized_squared"] is not None]
        distractors = sum(r["normal_distractor"] for r in rows)
        summary[key] = {"measurements": count,
                        "position_vector_rmse_m": math.sqrt(sum(r["error_squared"] for r in rows)/count),
                        "class_accuracy": sum(r["correct"] for r in rows)/count,
                        "distractor_measurements": distractors,
                        "distractor_false_candidate_fraction": sum(r["false_candidate"] for r in rows)/distractors if distractors else None,
                        "covariance_95_percent_ellipse_coverage": sum(v <= 5.991464547 for v in normalized)/len(normalized) if normalized else None}
    normal = summary.get("inspect_normal")
    sensor_pass = bool(normal and normal["position_vector_rmse_m"] <= contract["G3"]["normal_sensor_inspect_position_rmse_max_m"]
                       and normal["class_accuracy"] >= contract["G3"]["normal_sensor_confirmed_class_accuracy_min"]
                       and backgrounds > 0)
    return {"role": "dev_qualification_calibration_not_formal_method_statistics",
            "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
            "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "source_root": str(source_root) if source_root is not None else None,
            "attempts": attempts, "sensor_groups": summary, "background_candidates": backgrounds,
            "sensor_point_thresholds_pass": sensor_pass and all(a["admitted"] for a in attempts),
            "reference_and_clock_evidence_pass": (all(a["admitted"] for a in attempts)
                and any(a.get("reference_quality_pass") and a.get("realtime_clock_pass")
                        and a.get("coordinator_wait_pass") for a in attempts)),
            "limitations": ["Detected-only records do not estimate detection miss probability or visibility opportunity denominators.",
                            "Synthetic truth comparisons do not validate real imagery or field sensing.",
                            "Point threshold checks and one reference episode do not establish population reliability.",
                            "Full tool capability, development freeze, model access and statistical adapter remain separate gates."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output path")
    result = analyze(args.runs, args.contract, args.source_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result, indent=2))
