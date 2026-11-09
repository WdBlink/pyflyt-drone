"""Physical analytic transfers with complete target-independent exit continuations.

Development falsification only. Geometry screening uses a provisional6m tube;
it does not confer qualification. No target, mission truth, actor or report API.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.inspection.control import ControlGains, FixedwingAutopilot
from envs.inspection.flight import FlightWorld
from envs.inspection.paths import (Pose, Primitive, RouteProgram, TimedPath, capsule_from_bottom,
                                   csc_connectors, fleet_clearance, stripe_transfer)


def make_program(source, destination, delay, wind, airspeed=20.):
    start = Pose(-250., source-80., 0.)
    prefix = []
    if delay:
        prefix.append(Primitive(start, 20*delay, 0.))
        start = prefix[-1].end
    if source != destination:
        prefix.extend(stripe_transfer(start, destination-80.))
        start = prefix[-1].end
    survey = capsule_from_bottom(start, destination)
    prefix.extend(survey)
    start = prefix[-1].end
    # Choose only a finite CSC recovery candidate whose entire ideal path
    # lies inside the declared fence with provisional tube allowance.
    # A40m tangent runout gives the aircraft time to roll level between a
    # possible clockwise recovery arc and the CCW home circle. Without it
    # the real fourth aircraft exceeded the provisional6m tube at reversal.
    finish = Pose(-40., source-80., 0.)
    candidates = csc_connectors(start, finish)
    selected = None
    for path in candidates:
        points = [s.pose(float(d)) for s in path for d in np.linspace(0,s.length,max(2,math.ceil(s.length/2)))]
        if all(max(abs(p.x), abs(p.y)) <= 540 for p in points):
            selected = path
            break
    if selected is None:
        raise ValueError("no fenced finite CSC recovery candidate")
    prefix.extend(selected)
    prefix.append(Primitive(prefix[-1].end,40.,0.))
    tail = (Primitive(prefix[-1].end, 2*math.pi*80., 1/80.),)
    return RouteProgram(TimedPath(prefix, wind, airspeed), TimedPath(tail, wind, airspeed))


def search_delays(sources, destinations, wind, airspeed=20.):
    """Finite offline planning search; no simulator/actor or hidden state."""
    options = list(range(13))
    candidates = [[make_program(a,b,d,wind,airspeed) for d in options] for a,b in zip(sources,destinations)]
    horizon = max(p.prefix.duration for row in candidates for p in row)+candidates[0][0].tail.duration
    times = np.r_[np.arange(0.,horizon,.05),horizon]
    xy = [np.array([[p.at(float(t))[:2] for t in times] for p in row]) for row in candidates]
    pair_minima = {}
    for i,j in itertools.combinations(range(4),2):
        pair_minima[i,j] = np.sqrt(((xy[i][:,None]-xy[j][None,:])**2).sum(axis=-1)).min(axis=-1)-1.35
    attempts = 0
    for values in sorted(itertools.product(options,repeat=4),key=lambda d:(sum(d),d)):
        attempts += 1
        if all(pair_minima[i,j][values[i],values[j]] >= 27 for i,j in pair_minima):
            return list(values), {"candidate_combinations_checked":attempts,"options_seconds":options,
                                  "precomputed_individual_programs":52,"geometry_horizon_seconds":horizon,
                                  "selection":"minimum total declared delay then lexicographic; no physics or mission score"}
    return None, {"candidate_combinations_checked":attempts,"options_seconds":options,
                  "precomputed_individual_programs":52,"geometry_horizon_seconds":horizon}


def run(sources, destinations, delays, wind, geometry_only, search=False, airspeed=20.):
    started = time.monotonic()
    search_audit = None
    if search:
        delays,search_audit = search_delays(sources,destinations,wind,airspeed)
        if delays is None:
            return {"status":"finite_search_exhausted","search_audit":search_audit,
                    "source_stripe_centers":sources,"destination_stripe_centers":destinations,
                    "wall_seconds":time.monotonic()-started}
    programs = [make_program(a,b,d,wind,airspeed) for a,b,d in zip(sources,destinations,delays)]
    geometry = fleet_clearance(programs) if len(programs) == 4 else None
    candidate_pass = (geometry is None or geometry["continuous_lower_bound_horizontal_m"] >= 27)
    base = {"role": "analytic_transfer_physical_development_probe_not_qualification",
            "source_stripe_centers": sources, "destination_stripe_centers": destinations,
            "initial_eastbound_delay_seconds_at_20m_s": delays, "wind_xy_m_s": wind,
            "initial_eastbound_distances_m": [20*d for d in delays], "nominal_airspeed_m_s":airspeed,
            "geometry": geometry, "provisional_nominal_6m_tube_screen_passed": candidate_pass,
            "prefix_seconds": [p.prefix.duration for p in programs], "tail_period_seconds": [p.tail.duration for p in programs],
            "search_audit":search_audit}
    if geometry_only:
        return {**base, "status": "geometry_only", "wall_seconds": time.monotonic()-started}
    if not candidate_pass:
        return {**base, "status": "screen_rejected_before_physics", "wall_seconds": time.monotonic()-started}
    count = len(programs)
    flight = FlightWorld(0, count=count, task="survey", jitter=False, wind_xy=wind)
    gains = replace(ControlGains(), pitch_p=4.5, elevator_trim=-.1)
    flight.controllers = [FixedwingAutopilot(gains) for _ in programs]
    for i, source in enumerate(sources):
        flight.centers[i] = source
        drone = flight.world.drones[i]
        flight.world.resetBasePositionAndOrientation(drone.Id, [-250, source-80, float(flight.altitudes[i])], [0,0,0,1])
        flight.world.resetBaseVelocity(drone.Id, [20+wind[0],wind[1],0], [0,0,0])
        drone.update_state()
    tracking_audit = {"physics_samples":0,"maximum_horizontal_error_m":0.,
                      "settled_maximum_altitude_error_m":0.,"horizontal_6m_breach_steps":0,
                      "settled_altitude_2m_breach_steps":0}
    original_audit = flight.world.after_physics
    def audit_tracking(stamp):
        original_audit(stamp)
        positions = [flight.world.getBasePositionAndOrientation(d.Id)[0] for d in flight.world.drones]
        horizontal = max(math.dist(pos[:2],p.at(stamp)[:2]) for pos,p in zip(positions,programs))
        vertical = max(abs(pos[2]-alt) for pos,alt in zip(positions,flight.altitudes))
        tracking_audit["physics_samples"] += 1
        tracking_audit["maximum_horizontal_error_m"] = max(tracking_audit["maximum_horizontal_error_m"],horizontal)
        tracking_audit["horizontal_6m_breach_steps"] += int(horizontal > 6)
        if stamp >= 20:
            tracking_audit["settled_maximum_altitude_error_m"] = max(tracking_audit["settled_maximum_altitude_error_m"],vertical)
            tracking_audit["settled_altitude_2m_breach_steps"] += int(vertical > 2)
    flight.world.after_physics = audit_tracking
    end = max(p.prefix.duration for p in programs)+max(p.tail.duration for p in programs)+10
    trace, error = [], None
    try:
        while flight.t < end-1e-8:
            targets = [p.target(flight.t,flight.world.state(i)[3],float(flight.altitudes[i])) for i,p in enumerate(programs)]
            flight.step(targets,exit_flags=[flight.t >= p.prefix.duration for p in programs])
            if flight.world.physics_steps % 24 == 0:
                states = np.array(flight.world.all_states)
                planned = [p.at(flight.t)[:2].tolist() for p in programs]
                trace.append({"t": flight.t, "positions": states[:,3].tolist(), "planned_xy": planned,
                              "horizontal_errors_m": [math.dist(s[3,:2], xy) for s,xy in zip(states,planned)],
                              "altitude_errors_m": [float(abs(s[3,2]-a)) for s,a in zip(states,flight.altitudes)]})
    except Exception:
        error = traceback.format_exc()
    finally:
        safety, simulated = flight.summary(), flight.t
        try:
            flight.close()
        except Exception:
            error = (error or "")+traceback.format_exc()
    max_horizontal = max(max(r["horizontal_errors_m"]) for r in trace) if trace else None
    settled = [r for r in trace if r["t"] >= 20]
    max_vertical = max(max(r["altitude_errors_m"]) for r in settled) if settled else None
    passed = (not error and not safety["violations"] and tracking_audit["horizontal_6m_breach_steps"] == 0
              and tracking_audit["settled_altitude_2m_breach_steps"] == 0
              and tracking_audit["physics_samples"] == safety["physics_audits"]
              and all(t >= 20 for t in safety["exit_dwell_seconds"]))
    return {**base, "status": "completed" if not error else "operational_failure", "passed_provisional_physical_probe": bool(passed),
            "simulated_seconds": simulated, "wall_seconds": time.monotonic()-started, "safety": safety,
            "maximum_horizontal_phase_and_cross_track_error_m": max_horizontal,
            "settled_maximum_altitude_error_m": max_vertical, "tracking_audit_240hz":tracking_audit,
            "operational_error": error, "trace": trace}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sources", type=float, nargs="+", default=[-300.,-100.,100.,300.])
    parser.add_argument("--destinations", type=float, nargs="+", default=[-100.,100.,300.,-300.])
    parser.add_argument("--delays", type=float, nargs="+", default=[0.,0.,0.,8.])
    parser.add_argument("--wind", type=float, nargs=2, default=[0.,0.])
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--airspeed",type=float,default=20.)
    parser.add_argument("--search-delays", action="store_true")
    args = parser.parse_args()
    if (args.out.exists() or len(args.sources) not in (1,4) or len(args.sources) != len(args.destinations)
            or len(args.sources) != len(args.delays) or any(d < 0 or d > 12 for d in args.delays)):
        parser.error("new output, one/four matching aircraft and delays0..12s required")
    source_paths = [*sorted((ROOT/"envs/inspection").glob("*.py")), Path(__file__)]
    sources = {str(p.relative_to(ROOT)):p.read_text() for p in source_paths}
    try:
        result = run(args.sources, args.destinations, args.delays, args.wind, args.geometry_only, args.search_delays,args.airspeed)
    except Exception:
        result = {"status":"operational_failure", "operational_error":traceback.format_exc()}
    result["source_hashes"] = {p:hashlib.sha256(s.encode()).hexdigest() for p,s in sources.items()}
    result["source_snapshots"] = sources
    result["sources_unchanged"] = all((ROOT/p).read_text() == s for p,s in sources.items())
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("source_hashes","source_snapshots","trace")},indent=2))
    sys.exit(1 if result["status"] == "operational_failure" else 0)
