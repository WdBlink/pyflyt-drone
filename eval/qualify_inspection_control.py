"""Save real-physics development flights; this does not certify G1-G4."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Callable

import numpy as np
from PyFlyt.core import Aviary
from PyFlyt.core.drones import Fixedwing

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from envs.inspection.control import ControlGains, FixedwingAutopilot, FlightTarget, orbit_target


def fly(duration: float, seed: int, task: str, count: int, *, realtime: bool = False,
        on_step: Callable[[Aviary], None] | None = None) -> dict:
    source_paths = [Path(__file__), Path(inspect.getfile(Fixedwing)),
                    Path(inspect.getfile(FixedwingAutopilot)), Path(inspect.getfile(Aviary))]
    source_bytes = {str(p.relative_to(Path(__file__).resolve().parents[1])): p.read_bytes() for p in source_paths}
    centers = np.array([[0.0, 280.0 * i] for i in range(count)])
    radius = 90.0
    starts = np.array([[center[0], center[1] - radius, 80.0] for center in centers])
    world = Aviary(start_pos=starts, start_orn=np.zeros((count, 3)), drone_type="fixedwing",
                   np_random=np.random.default_rng(seed))
    world.set_mode(-1)
    controllers = [FixedwingAutopilot() for _ in range(count)]
    trace = []
    violations = set()
    min_separation = float("inf")
    min_altitude, max_altitude = 80.0, 80.0
    min_speed, max_speed, max_roll = 20.0, 20.0, 0.0
    wall_start = time.monotonic()
    operational_error = None
    terminal_state = None
    valid_steps = 0
    clock_lags = []
    try:
        while world.elapsed_time < duration - 1e-9:
            if realtime:
                remaining = wall_start + world.elapsed_time + world.step_period - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
            for i, controller in enumerate(controllers):
                state = world.state(i)
                target = (orbit_target(state[3], centers[i], radius) if task == "orbit"
                          else FlightTarget(0.0))
                world.set_setpoint(i, controller.step(state, target, world.step_period))
            world.step()
            clock_lags.append(time.monotonic() - wall_start - world.elapsed_time)
            if on_step is not None:
                on_step(world)
            states = np.array(world.all_states)
            altitude = states[:, 3, 2]
            speed = np.linalg.norm(states[:, 2], axis=1)
            roll = np.abs(states[:, 1, 0])
            if not np.isfinite(states).all():
                violations.add("nonfinite_state")
                terminal_state = {
                    "t": world.elapsed_time,
                    "states": np.where(np.isfinite(states), states, np.nan).tolist(),
                    "nonfinite_indices": np.argwhere(~np.isfinite(states)).tolist(),
                }
                terminal_state["states"] = [
                    [[value if np.isfinite(value) else None for value in row] for row in drone]
                    for drone in terminal_state["states"]
                ]
                break
            valid_steps += 1
            min_altitude = min(min_altitude, float(altitude.min()))
            max_altitude = max(max_altitude, float(altitude.max()))
            min_speed, max_speed = min(min_speed, float(speed.min())), max(max_speed, float(speed.max()))
            max_roll = max(max_roll, float(roll.max()))
            if count > 1:
                distances = np.linalg.norm(states[:, None, 3] - states[None, :, 3], axis=-1)
                distances[np.diag_indices(count)] = np.inf
                min_separation = min(min_separation, float(distances.min()))
            if np.any(world.contact_array):
                violations.add("collision")
            if np.any((altitude < 60.0) | (altitude > 100.0)):
                violations.add("altitude_envelope")
            if np.any((speed < 15.0) | (speed > 25.0)):
                violations.add("speed_envelope")
            if np.any(roll > np.deg2rad(45.0)):
                violations.add("bank_envelope")
            if world.aviary_steps % 12 == 0:
                trace.append({"t": world.elapsed_time, "states": states.tolist(),
                              "radial_errors": (np.linalg.norm(states[:, 3, :2] - centers, axis=1) - radius).tolist()})
            if np.any((altitude < 20.0) | (altitude > 140.0)) or np.any(roll > 1.2):
                violations.add("development_emergency_stop")
                break
    except Exception:
        operational_error = traceback.format_exc()
        try:
            states = np.array(world.all_states)
            terminal_state = {
                "t": world.elapsed_time,
                "states": [[[float(value) if np.isfinite(value) else None for value in row]
                            for row in drone] for drone in states],
                "nonfinite_indices": np.argwhere(~np.isfinite(states)).tolist(),
            }
        except Exception:
            terminal_state = {"t": world.elapsed_time, "snapshot_error": traceback.format_exc()}
    finally:
        wall_seconds = time.monotonic() - wall_start
        simulated = world.elapsed_time
        physics_steps = world.physics_steps
        try:
            world.disconnect()
        except Exception:
            operational_error = (operational_error or "") + "\nCleanup failure:\n" + traceback.format_exc()
    if any(p.read_bytes() != source_bytes[str(p.relative_to(Path(__file__).resolve().parents[1]))] for p in source_paths):
        operational_error = (operational_error or "") + "\nSource changed during development run."
    return {
        "role": "accelerated_development_control_flight_not_formal_benchmark_or_full_qualification",
        "seed": seed, "task": task, "aircraft_count": count,
        "requested_seconds": duration, "simulated_seconds": simulated,
        "physics_steps": physics_steps, "physics_hz": 240, "control_hz": 120,
        "clock_mode": "wallclock_1x_development" if realtime else "accelerated_development",
        "clock_lag_seconds": {"max": max(clock_lags, default=None),
                              "p95": float(np.quantile(clock_lags, .95)) if clock_lags else None} if realtime else None,
        "wall_seconds_stepping_only": wall_seconds, "stepping_rtf": simulated / wall_seconds,
        "gains": asdict(ControlGains()), "centers": centers.tolist(), "radius": radius,
        "versions": {p: importlib.metadata.version(p) for p in ("PyFlyt", "pybullet", "numpy")},
        "source_hashes": {key: hashlib.sha256(value).hexdigest() for key, value in source_bytes.items()},
        "source_snapshots": {key: value.decode("utf-8") for key, value in source_bytes.items()},
        "operational_error": operational_error, "terminal_state": terminal_state,
        "envelope_pass": operational_error is None and not violations and simulated >= duration - 1e-8,
        "violations": sorted(violations),
        "summary": {"min_altitude": min_altitude, "max_altitude": max_altitude,
                    "min_speed": min_speed, "max_speed": max_speed, "max_abs_roll": max_roll,
                    "min_separation": min_separation if count > 1 and np.isfinite(min_separation) else None},
        "summary_scope": "initial state plus finite completed steps; excludes any invalid terminal state",
        "valid_completed_control_steps": valid_steps,
        "trace_hz": 10, "trace": trace,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--aircraft", type=int, choices=(1, 4), default=1)
    parser.add_argument("--task", choices=("straight", "orbit"), default="straight")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0 or args.out.exists():
        parser.error("duration must be positive and output must be a new path")
    try:
        result = fly(args.duration, args.seed, args.task, args.aircraft, realtime=args.realtime)
    except Exception:
        result = {"role": "development_operational_failure_not_model_performance",
                  "seed": args.seed, "task": args.task, "aircraft_count": args.aircraft,
                  "requested_seconds": args.duration, "envelope_pass": False,
                  "operational_error": traceback.format_exc(), "trace": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "trace"}, indent=2))
    sys.exit(0 if result["envelope_pass"] else 1)
