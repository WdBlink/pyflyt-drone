"""From-zero physical G2 fixture for exogenous retirement and safe exit.

Uses the production event and executor with controlled next-tick planning.
This is neither a policy comparison nor a wall-clock latency measurement.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
import PyFlyt
import pybullet

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import envs.inspection.mission as mission_module
from envs.inspection.execution import program_hash
from eval.inspection_tracking import recount_tracking
from eval.qualify_inspection_executor import ControlledPlannerExecutor
from eval.qualify_inspection_g1 import write_json_atomic


def probe():
    original = mission_module.FlightExecutor
    mission_module.FlightExecutor = ControlledPlannerExecutor
    try:
        mission = mission_module.InspectionMission(0, reference=True, events=True)
    finally:
        mission_module.FlightExecutor = original
    result = {"role": "controlled_physical_retirement_fixture_not_policy_or_1x_evidence",
              "initial_sim_epoch": 0., "planner_transport": "controlled next-tick"}
    try:
        while mission.t < 280.-1e-8:
            mission.step()
        response = mission.tool("assign_survey", {
            "drone_id": 3, "region_id": "stripe-3", "duration": 160.,
            "expected_version": mission.jobs.versions[3], "rule_id": "static-0",
            "task_card_version": mission.card["task_card_version"]})
        if response["status"] != "accepted":
            raise RuntimeError("retirement fixture could not submit pre-event task")
        old_job = mission.jobs.jobs[response["job_id"]]
        while mission.t < 360.-1/120-1e-8:
            mission.step()
        before = {"t": mission.t, "job": asdict(old_job),
                  "program_hash": program_hash(mission.executor.programs[3]),
                  "positions": [s[3].tolist() for s in mission.flight.world.all_states],
                  "version": mission.jobs.versions[3]}
        mission.step()
        after = {"t": mission.t, "job": asdict(old_job),
                 "program_hash": program_hash(mission.executor.programs[3]),
                 "positions": [s[3].tolist() for s in mission.flight.world.all_states],
                 "version": mission.jobs.versions[3]}
        exit_jobs = [j for j in mission.jobs.jobs.values()
                     if j.drone_id == 3 and j.params.get("origin") == "exogenous_capacity_exit"]
        if len(exit_jobs) != 1:
            raise RuntimeError("expected one actual production retirement event")
        exiting = exit_jobs[0]
        denied = mission.tool("assign_survey", {
            "drone_id": 3, "region_id": "stripe-3", "duration": 10.,
            "expected_version": mission.jobs.versions[3], "rule_id": "static-0",
            "task_card_version": mission.card["task_card_version"]})
        stale = mission.tool("cancel_job", {
            "job_id": old_job.job_id, "expected_version": before["version"]})
        while mission.t < 500.-1e-8:
            mission.step()
        business = mission.business_audit()
        tracking = recount_tracking(business, expected_xy_tube_m=mission.executor.xy_tube_m)
        steps = mission.flight.world.physics_steps
        displacements = np.linalg.norm(np.asarray(after["positions"])-before["positions"], axis=1)
        commits = [r for r in business["execution_audit"]
                   if r["type"] == "program_committed" and r["job_id"] == exiting.job_id]
        remaining = mission.card["range_budget_m_per_aircraft"]-mission.distance_m
        new_acquisitions = {
            "target_observations": sum(r["drone_id"] == 3 and r["t"] >= 360.
                                       for r in business["sensor_audit"]),
            "coverage_packets": sum(r["drone_id"] == 3 and r["measured_at"] >= 360.
                                    for r in business["coverage_evidence"]),
            "visible_opportunities": sum(r["drone_id"] == 3 and r["t"] >= 360.
                                         for r in business["sensor_opportunities"]),
            "background_trials": sum(r["drone_id"] == 3 and r["t"] >= 360.
                                      for r in business["background_trials"]),
        }
        old_uploads = [o for o in business["delivered_observations"]
                       if o["drone_id"] == 3 and o["measured_at"] < 360. <= o["arrived_at"]]
        checks = {
            "actual_continuous_500s_coverage": steps == 500*240 == mission.flight.physics_audits
                == tracking["physics_samples"],
            "task_running_immediately_before_event": before["job"]["status"] == "running",
            "event_exactly_360s": before["t"] < 360. == after["t"] == exiting.created_at,
            "old_task_cancelled_at_event": old_job.status == "cancelled" and old_job.finished_at == 360.,
            "logical_preemption_retains_old_physical_program": before["program_hash"] == after["program_hash"],
            "no_aircraft_jump_at_event": bool(np.all((displacements > .01) & (displacements < 27/120))),
            "retired_work_rejected": denied["status"] == "rejected" and "retired" in denied.get("reason", ""),
            "stale_old_cancellation_rejected": stale["status"] == "rejected" and "stale" in stale.get("reason", ""),
            "retirement_stops_new_mission_acquisition": not any(new_acquisitions.values()),
            "pre_retirement_late_upload_preserved": bool(old_uploads),
            "exit_is_physical_terminal": exiting.status == "succeeded"
                and exiting.finished_at is not None and exiting.finished_at > 380.
                and mission.flight.exit_dwell[3] >= 20.,
            "exit_commits_use_future_capture": bool(commits) and all(r["at"] >= 360.5-1e-8 for r in commits),
            "hard_safety_clean": not mission.flight.violations,
            "tracking_envelope_clean": tracking["horizontal_tube_breach_steps"]
                == tracking["altitude_envelope_breach_steps"] == 0,
            "seam_capture_clean": business["execution_tracking_240hz"]["seam_capture_failures"] == 0,
            "all_aircraft_physically_exited": bool(np.all(mission.flight.exit_dwell >= 20.)),
            "all_aircraft_reserve_retained": bool(np.all(remaining >= mission.card["exit_range_reserve_m"])),
            "infrastructure_clean": not business["execution_infrastructure_errors"],
        }
        checks = {name: bool(value) for name, value in checks.items()}
        result.update(status="completed", passed=all(checks.values()), checks=checks,
                      final_sim_epoch=mission.t, physics_steps_executed=steps,
                      before_retirement=before, after_retirement=after,
                      event_tick_displacement_m=displacements.tolist(), retirement_exit_job=asdict(exiting),
                      retired_work_response=denied, stale_cancellation_response=stale,
                      post_retirement_acquisition_counts=new_acquisitions,
                      pre_retirement_late_observations=old_uploads,
                      remaining_range_m=remaining.tolist(), safety=mission.flight.summary(),
                      tracking_recount=tracking, public_events=mission.public_events,
                      trace=mission.trace, business_audit=business)
    except Exception:
        result.update(status="operational_failure", passed=False, error=traceback.format_exc(),
                      final_sim_epoch=mission.t, physics_steps_executed=mission.flight.world.physics_steps,
                      partial_business_audit=mission.business_audit())
    finally:
        mission.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output")
    paths = [*sorted((ROOT/"envs/inspection").glob("*.py")), ROOT/"envs/__init__.py",
             Path(__file__), ROOT/"eval/qualify_inspection_executor.py",
             ROOT/"eval/qualify_inspection_g1.py", ROOT/"eval/inspection_tracking.py"]
    sources = {str(p.resolve()): p.read_text() for p in paths}
    package = Path(PyFlyt.__file__).resolve().parent
    dependencies = [p for p in package.rglob("*")
                    if p.suffix.lower() in (".py", ".yaml", ".urdf", ".obj", ".stl", ".dae")]
    dependencies.append(Path(pybullet.__file__))
    fingerprints = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    try:
        result = probe()
    except Exception:
        result = {"status": "operational_failure", "passed": False,
                  "error": traceback.format_exc()}
    result.update(source_snapshots=sources,
                  source_hashes={p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()},
                  sources_unchanged=all(Path(p).read_text() == s for p, s in sources.items()),
                  dependency_hashes=fingerprints,
                  dependencies_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                                             for p, h in fingerprints.items()))
    write_json_atomic(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k in
                      ("status", "passed", "checks", "error", "final_sim_epoch", "physics_steps_executed")}, indent=2))
    return 0 if result["passed"] and result["sources_unchanged"] and result["dependencies_unchanged"] else 1


if __name__ == "__main__":
    sys.exit(main())
