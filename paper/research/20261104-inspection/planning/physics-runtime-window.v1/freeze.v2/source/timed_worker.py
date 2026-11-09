"""Frozen-worker derivative for one explicitly unscored clock localization.

Original loop/epoch/order/limits retained; numeric phase markers are additional
work. This derivative is not a completed pilot retry or policy comparison.
"""
import hashlib
import json
from pathlib import Path
import queue
import time
import traceback
from eval.inspection_clock_profile import ClockProfile

def worker(seed, events, realtime, duration, commands, updates, out, reference_layout=True,
           shared_progress=None, private_case_path=None, expected_case_sha256=None,
           *, profile_factory=ClockProfile):
    from envs.inspection.mission import InspectionMission
    from envs.inspection.protocol import audit_value
    mission, profile = None, None
    result = {"status": "operational_failure"}
    try:
        warmup = InspectionMission(seed, reference=reference_layout, events=False,start_planner=False)
        for _ in range(120):
            warmup.step()
        warmup.close()
        if private_case_path is None:
            mission = InspectionMission(seed, reference=reference_layout, events=events)
        else:
            from eval.inspection_population import load_development_case
            from eval.inspection_scenarios import ScenarioMission
            case_bytes = Path(private_case_path).read_bytes()
            if hashlib.sha256(case_bytes).hexdigest() != expected_case_sha256:
                raise ValueError("private case changed before worker read")
            case = load_development_case(json.loads(case_bytes))
            mission = ScenarioMission(case)
        profile = profile_factory(duration)
        epoch = time.monotonic()
        profile.anchor(epoch)
        lags, published_at = [], -1.0
        max_lag,lag_witness,previous_tail = float("-inf"),None,0.
        updates.put({"kind": "state", "state": mission.public_state()})
        profile.initial_publication_end()
        with (out/"progress.jsonl").open("w") as stream:
            while mission.t < duration-1e-8:
                profile.begin(mission.t)
                remaining = 0.
                if realtime:
                    remaining = epoch+mission.t+1/120-time.monotonic()
                    profile.requested_sleep(remaining)
                    if remaining > 0:
                        time.sleep(remaining)
                profile.mark("wake")
                work_started = time.monotonic()
                rpc_count = 0
                for _ in range(16):
                    try:
                        command = commands.get_nowait()
                        rpc_count += 1
                    except queue.Empty:
                        break
                    if not isinstance(command, dict) or set(command) != {"name", "args"} or not isinstance(command["name"], str) or not isinstance(command["args"], dict):
                        mission.tool_audit.append({"t": mission.t, "invalid_rpc": audit_value(command)})
                        updates.put({"kind": "tool_result", "name": "invalid_rpc", "args": {},
                                     "response": {"status": "rejected", "reason": "malformed actor RPC envelope"}})
                        continue
                    response = mission.tool(command["name"], command["args"])
                    updates.put({"kind": "tool_result", **command, "response": response})
                profile.rpc(rpc_count)
                profile.mark("rpc_end")
                mission.step()
                profile.mark("step_end")
                if shared_progress is not None:
                    with shared_progress.get_lock():
                        shared_progress[:] = [mission.t, *mission.flight.world.state(0)[3].tolist()]
                profile.mark("progress_end")
                lag = time.monotonic()-epoch-mission.t
                lags.append(lag)
                profile.lag(lag, mission.flight.world.physics_steps%120 == 0)
                if lag > max_lag:
                    max_lag = lag
                    lag_witness = {"sim_time":mission.t,"lag_seconds":lag,
                        "current_step_work_seconds":time.monotonic()-work_started,
                        "sensor_tick":mission.flight.world.physics_steps%120 == 0,
                        "previous_publication_work_seconds":previous_tail}
                profile.mark("stats_end")
                tail_started = time.monotonic()
                if mission.flight.world.physics_steps % 120 == 0:
                    updates.put({"kind": "state", "state": mission.public_state(since=published_at)})
                    published_at = mission.t
                profile.mark("pub_end")
                if mission.flight.world.physics_steps % 240 == 0:
                    stream.write(json.dumps({"t": mission.t, "wall_elapsed": time.monotonic()-epoch,
                                             "safety": mission.flight.summary(),
                                             "reports": len(mission.ledger.reports), "coverage_cells": len(mission.covered)})+"\n")
                    stream.flush()
                profile.mark("log_end")
                previous_tail = time.monotonic()-tail_started
                profile.end(mission.t)
        profile.stop()
        result = {"status": "completed" if not mission.executor.infrastructure_errors else "operational_failure",
                  "role": "non_llm_dev_reference_not_formal_comparison",
                  "seed": seed if private_case_path is None else None,
                  "events": events if private_case_path is None else case.level == "L3",
                  "clock_mode": "wallclock_1x" if realtime else "accelerated_diagnostic",
                  "simulated_seconds": mission.t, "wall_seconds": time.monotonic()-epoch,
                  "physics_steps_executed": mission.flight.world.physics_steps,
                  "physics_hz": mission.flight.world.physics_hz,
                  "control_hz": 1/mission.flight.world.step_period,
                  "private_case_file_sha256": expected_case_sha256,
                  "max_clock_lag_seconds": max(lags) if realtime else None,
                  "clock_lag_p95_seconds": sorted(lags)[int(.95*(len(lags)-1))] if realtime else None,
                  "maximum_clock_lag_witness":lag_witness if realtime else None,
                  "clock_lag_over_0p1_seconds_ticks":sum(x > .1 for x in lags) if realtime else None,
                  **mission.audit()}
    except Exception:
        result["operational_error"] = traceback.format_exc()
        if mission is not None:
            result["simulated_seconds"] = mission.t
            result["partial_business_audit"] = mission.business_audit()
    finally:
        if profile is not None:
            profile.stop()
        if mission is not None:
            try:
                mission.close()
            except Exception:
                result["status"] = "operational_failure"
                result["cleanup_error"] = traceback.format_exc()
        if profile is not None:
            try:
                result["clock_profile"] = profile.save(out)
                if result["clock_profile"].get("runtime_policy",{}).get("errors"):
                    result["status"] = "operational_failure"
            except Exception:
                result["clock_profile_error"] = traceback.format_exc()
                result["status"] = "operational_failure"
        (out/"world.json").write_text(json.dumps(audit_value(result), indent=2, allow_nan=False)+"\n")
        updates.put({"kind": "done", "status": result["status"]})
