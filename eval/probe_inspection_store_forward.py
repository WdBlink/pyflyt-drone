"""Actual flight from epoch0 through a target-evidence link outage.

Controlled G2 fixture; no clock/aircraft reset, no1x or policy comparison claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import envs.inspection.mission as mission_module
from eval.qualify_inspection_executor import ControlledPlannerExecutor
from envs.inspection.scenario import Entity
from envs.inspection.scoring import grade_dynamic_bins


def probe():
    original = mission_module.FlightExecutor
    mission_module.FlightExecutor = ControlledPlannerExecutor
    try:
        mission = mission_module.InspectionMission(0, reference=True)
    finally:
        mission_module.FlightExecutor = original
    appearance = 206.0
    try:
        body = next(iter(mission.entity_bodies.values()))
        for other in mission.entity_bodies.values():
            mission.flight.world.resetBasePositionAndOrientation(other, [0, 0, -10], [0, 0, 0, 1])
        mission.entities, mission.entity_bodies = [], {}
        while mission.t < appearance-1e-8:
            mission.step()
        position = mission.flight.world.state(1)[3]
        entity = Entity("fixture-moving", "dynamic", "inspection_vehicle", tuple(position[:2]), appearance,
                        1.5, "dynamic-inspection")
        mission.entities, mission.entity_bodies = [entity], {entity.key: body}
        while mission.t < appearance+1.-1e-8:
            mission.step()
        obs = next(o for o in reversed(mission.delivered) if o.drone_id == 1 and o.object_type == "dynamic")
        inspect_ack = mission.tool("inspect_candidate", {"drone_id": 1, "candidate_id": obs.candidate_id,
            "expected_version": 0, "rule_id": "dynamic-inspection", "task_card_version": mission.card["task_card_version"]})
        job = mission.jobs.jobs[inspect_ack["job_id"]]
        while mission.t < 210.-1e-8 and (job.status != "succeeded" or not any(
                o.candidate_id == obs.candidate_id and o.sensor_mode == "inspect" and o.quality >= .8
                and job.finished_at is not None and o.measured_at >= job.finished_at for o in mission.delivered)):
            mission.step()
        qualified = [o for o in mission.delivered if o.candidate_id == obs.candidate_id
                     and o.sensor_mode == "inspect" and o.quality >= .8]
        if job.status != "succeeded" or not qualified or mission.t >= 210.:
            raise RuntimeError("controlled fixture did not confirm before the original link outage")
        latest = qualified[-1]
        report_ack = mission.tool("submit_or_revise_report", mission.report_request(latest,"fixture-report"))
        observe_ack = mission.tool("observe_track", {"drone_id": 1, "candidate_id": obs.candidate_id,
            "duration": 60.0, "report_id": "fixture-report", "expected_version": mission.jobs.versions[1],
            "rule_id": "dynamic-inspection", "task_card_version": mission.card["task_card_version"]})
        pending_at_239 = []
        while mission.t < 241.-1e-8:
            if abs(mission.t-239.)<1e-8:
                pending_at_239 = [e.observation.public() for ready,e,route in mission.pending
                    if e.observation.drone_id == 1 and 210 <= e.observation.measured_at < 239 and route == "fixture-report"]
            mission.step()
        late = [o for o in mission.delivered if o.drone_id == 1 and o.candidate_id == obs.candidate_id and 210 <= o.measured_at < 240]
        versions = mission.ledger.reports["fixture-report"].versions
        delivered, bins = grade_dynamic_bins(versions, entity.position, entity.true_class, appearance, 600)
        observed, _ = grade_dynamic_bins(versions, entity.position, entity.true_class, appearance, 600, track="observed")
        actual_steps = mission.flight.world.physics_steps
        passed = (inspect_ack["status"] == report_ack["status"] == observe_ack["status"] == "accepted" and bool(late) and bool(pending_at_239)
                  and mission.flight.world.physics_steps == round(mission.t*240)
                  and all(o.arrived_at >= 240 and o.measured_at < o.arrived_at for o in late)
                  and all(any(v.evidence_id == o.evidence_id and v.measured_at == o.measured_at
                              and v.arrived_at == o.arrived_at for v in versions) for o in late)
                  and mission.flight.physics_audits == actual_steps and not mission.flight.violations)
        return {"role": "controlled_physical_store_forward_fixture_not_full_episode_or_policy_sample",
                "initial_sim_epoch": 0., "target_appearance_epoch": appearance, "final_sim_epoch": mission.t,
                "physically_integrated_seconds": mission.t,
                "planner_transport": "controlled next-tick, not1x or model latency",
                "pending_target_packets_at_239": pending_at_239,
                "physics_steps_executed": actual_steps, "passed": passed,
                "late_target_observations": [o.public() for o in late],
                "single_target_D_delivered_diagnostic": delivered, "single_target_D_observed_diagnostic": observed,
                "selected_evidence_by_bin": [b.evidence_id for b in bins],
                "safety": mission.flight.summary(), "tool_calls": mission.tool_audit,
                "report_versions": [{"evidence_id": v.evidence_id, "measured_at": v.measured_at,
                                     "arrived_at": v.arrived_at} for v in versions],
                "trace": mission.trace, "business_audit":mission.business_audit()}
    except Exception:
        return {"passed":False,"status":"operational_failure","error":traceback.format_exc(),
                "initial_sim_epoch":0.,"final_sim_epoch":mission.t,
                "physics_steps_executed":mission.flight.world.physics_steps,
                "partial_business_audit":mission.business_audit()}
    finally:
        mission.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new output")
    sources = {str(p.resolve()): p.read_text() for p in [*Path("envs/inspection").glob("*.py"), Path(__file__),
        Path("eval/qualify_inspection_executor.py"),Path("eval/qualify_inspection_g1.py"),Path("eval/inspection_tracking.py")]}
    try:
        result = probe()
    except Exception:
        result = {"passed": False, "status": "operational_failure", "error": traceback.format_exc()}
    result["source_hashes"] = {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()}
    result["source_snapshots"] = sources
    result["sources_unchanged"] = all(Path(p).read_text() == source for p, source in sources.items())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k in ("passed", "error", "initial_sim_epoch", "physically_integrated_seconds", "physics_steps_executed", "single_target_D_delivered_diagnostic", "single_target_D_observed_diagnostic")}, indent=2))
    sys.exit(0 if result["passed"] and result["sources_unchanged"] else 1)
