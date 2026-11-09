"""Execute the prospective G1 development grid, preserving every slot."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import PyFlyt
import pybullet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_source_paths = [Path(__file__).resolve().parents[1]/"envs/inspection/control.py",
                 Path(__file__).resolve().parents[1]/"envs/inspection/flight.py", Path(__file__).resolve()]
_import_sources = {str(p): p.read_text() for p in _source_paths}
from envs.inspection.control import ControlGains,FixedwingAutopilot,FlightTarget,SurveyRoute,orbit_target,wrap_angle
from envs.inspection.flight import FlightLimits, FlightWorld


def write_json_atomic(path, value):
    temporary = path.with_name("."+path.name+".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_case(task, root, duration, contract):
    rng = np.random.default_rng(root)
    count = 1 if root % 2 == 0 and task == "straight" else 4
    wind = contract["wind_max_m_s"]*np.array([math.cos(root), math.sin(root)]) if root % 2 else np.zeros(2)
    radius = float(rng.uniform(*contract["radius_range_m"]))
    flight = FlightWorld(root, count=count, wind_xy=wind, task=task, radius=radius,
                         half_length=float(rng.uniform(*contract["survey_half_length_range_m"])), limits=FlightLimits.from_contract(contract),
                         bound=contract["flight_bounds_xy_m"][task])
    if "control_gains" in contract:
        flight.controllers=[FixedwingAutopilot(ControlGains(**contract["control_gains"])) for _ in range(count)]
    nominal_airspeed=contract.get("nominal_airspeed_m_s",20.)
    trace, errors, heading_errors = [], [], []
    operational_error = None
    phase_start = 0
    requested_heading = 0.0
    routes = [SurveyRoute(float(c),flight.half_length,flight.radius) for c in flight.centers]
    modes = ["survey" if task == "sortie_exit" else task]*count
    switch_at = duration*contract["sortie_exit_switch_fraction"]
    started = time.monotonic()
    try:
        while flight.t < duration-1e-8:
            if task == "heading_changes":
                phase = max((x for x in contract["heading_schedule"] if flight.t >= x["from"]), key=lambda x: x["from"])
                phase_start, requested_heading = phase["from"], math.radians(phase["degrees"])
            if task == "sortie_exit" and flight.t >= switch_at:
                modes = ["exit"] * count
            targets = []
            for i,mode in enumerate(modes):
                position = flight.world.state(i)[3]
                if mode in ("exit","loiter"):
                    target = orbit_target(position,np.array([0.,flight.centers[i]]),radius,float(flight.altitudes[i]),nominal_airspeed)
                elif mode == "survey":
                    target = routes[i].target(position,float(flight.altitudes[i]),nominal_airspeed)
                else:
                    target = FlightTarget(requested_heading,float(flight.altitudes[i]),nominal_airspeed)
                targets.append(target)
            flight.step(targets,exit_flags=[m == "exit" for m in modes])
            if flight.world.aviary_steps % 12 == 0:
                states = np.array(flight.world.all_states)
                current_errors, current_heading = [], []
                for i, state in enumerate(states):
                    if modes[i] in ("exit", "loiter"):
                        error = abs(np.linalg.norm(state[3, :2]-[0, flight.centers[i]])-radius)
                    elif task in ("survey", "sortie_exit"):
                        error = routes[i].cross_track_error(state[3])
                    else:
                        error = abs(state[3, 1]-(flight.centers[i]-radius)) if task == "straight" else 0
                    velocity, _ = flight.world.getBaseVelocity(flight.world.drones[i].Id)
                    course = math.atan2(velocity[1], velocity[0])
                    current_errors.append(float(error))
                    # The acceptance schedule is independent of target()/last_targets.
                    current_heading.append(abs(wrap_angle(course-requested_heading)))
                phase_time = flight.t-switch_at if task == "sortie_exit" and flight.t >= switch_at else flight.t
                if phase_time >= contract["settle_seconds"]:
                    errors.extend(current_errors)
                if task in ("straight", "heading_changes") and flight.t-phase_start >= contract["settle_seconds"]:
                    heading_errors.extend(current_heading)
                trace.append({"t": flight.t, "positions": states[:, 3].tolist(),
                              "tracking_error_m": current_errors, "course_error_rad": current_heading})
        summary = flight.summary()
        rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        maximum = max(errors, default=None)
        heading_max = max(heading_errors, default=0)
        tracking_pass = rmse is not None and rmse <= contract["tracking_rmse_m_after_settle"] and maximum <= contract["tracking_max_m_after_settle"]
        if task == "heading_changes":
            tracking_pass = bool(heading_errors) and heading_max <= math.radians(contract["settled_course_error_degrees"])
        if task == "straight":
            tracking_pass = tracking_pass and bool(heading_errors) and heading_max <= math.radians(contract["settled_course_error_degrees"])
        exit_pass = task != "sortie_exit" or all(x >= contract["exit_dwell_seconds"]-1e-6 for x in flight.exit_dwell)
        passed = not summary["violations"] and tracking_pass and exit_pass and summary["physics_audits"] == summary["physics_steps"]
    except Exception:
        operational_error = traceback.format_exc()
        summary = flight.summary()
        passed, rmse, maximum, heading_max = False, None, None, None
    finally:
        simulated = flight.t
        try:
            flight.close()
        except Exception:
            operational_error = (operational_error or "")+"\nCleanup failure:\n"+traceback.format_exc()
            passed = False
    return {"task": task, "root": root, "role": "control_development_qualification",
            "aircraft_count": count, "wind_xy_m_s": wind.tolist(), "radius_m": radius,
            "requested_seconds": duration, "simulated_seconds": simulated,
            "wall_seconds": time.monotonic()-started, "passed": bool(passed),
            "tracking_rmse_m": rmse, "tracking_max_m": maximum,
            "settled_max_course_error_rad": heading_max,
            "operational_error": operational_error, "safety": summary, "trace": trace}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roots", type=int, default=None, help="development diagnostic subset, not full qualification")
    parser.add_argument("--families", nargs="+", default=None)
    args = parser.parse_args()
    if args.out.exists() or (args.roots is not None and args.roots <= 0):
        parser.error("choose a new output directory and positive roots")
    args.out.mkdir(parents=True)
    contract_bytes = args.contract.read_bytes()
    contract = json.loads(contract_bytes)
    sources = [Path("envs/inspection/control.py"), Path("envs/inspection/flight.py"), Path(__file__)]
    frozen = {str(p): p.read_text() for p in sources}
    if any(p.read_text() != _import_sources[str(p)] for p in _source_paths):
        raise RuntimeError("source changed since module import")
    families = args.families or list(contract["G1"]["families"])
    roots = args.roots or contract["G1"]["roots_per_family"]
    root_ids = contract["G1"]["root_ids"][:roots]
    grid = [{"task": family, "root": root, "status": "unstarted"} for family in families for root in root_ids]
    package = Path(PyFlyt.__file__).resolve().parent
    dependencies = [p for p in package.rglob("*") if p.suffix.lower() in (".py", ".yaml", ".urdf", ".obj", ".stl", ".dae")]
    dependencies.append(Path(pybullet.__file__))
    dependency_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    manifest = {"contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
                "contract": contract, "sources": frozen, "grid": grid,
                "dependency_hashes": dependency_hashes,
                "versions": {p: importlib.metadata.version(p) for p in ("PyFlyt", "pybullet", "numpy", "numba")},
                "full_grid": args.roots is None and args.families is None}
    def save():
        write_json_atomic(args.out/"manifest.json", manifest)
    save()
    for slot in grid:
        slot["status"] = "running"
        slot["started_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()
        try:
            result = run_case(slot["task"], slot["root"], contract["G1"]["families"][slot["task"]], contract["G1"])
            slot["status"] = "completed" if not result["operational_error"] else "operational_failure"
            slot["passed"] = result["passed"]
        except Exception:
            result = {"passed": False, "operational_error": traceback.format_exc()}
            slot["status"], slot["passed"] = "operational_failure", False
        path = args.out/f"{slot['task']}-{slot['root']:02d}.json"
        write_json_atomic(path, result)
        slot["file"] = path.name
        slot["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        save()
        print(slot["task"], slot["root"], slot["status"], slot["passed"], flush=True)
    manifest["sources_unchanged"] = all(Path(p).read_text() == source for p, source in frozen.items())
    manifest["contract_unchanged"] = args.contract.read_bytes() == contract_bytes
    manifest["dependencies_unchanged"] = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p, sha in dependency_hashes.items())
    manifest["family_rates"] = {f: sum(s["passed"] for s in grid if s["task"] == f)/roots for f in families}
    manifest["qualified"] = (manifest["full_grid"] and manifest["sources_unchanged"] and manifest["dependencies_unchanged"] and manifest["contract_unchanged"]
                             and all(r >= contract["G1"]["success_fraction_per_family"] for r in manifest["family_rates"].values())
                             and all(s["status"] == "completed" for s in grid))
    save()
    return 0 if (manifest["qualified"] if manifest["full_grid"] else all(s["passed"] for s in grid)) else 1


if __name__ == "__main__":
    sys.exit(main())
