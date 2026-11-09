"""Physical falsification of causal moving-circle guidance; not a full episode.

One airborne aircraft is initialized before integration. A separate hidden
point moves at 1.5 m/s. The controller receives only noisy, actually visible
2 Hz local samples (10% seeded misses), never that point's future or velocity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import traceback
from dataclasses import replace, asdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from envs.inspection.flight import FlightWorld
from envs.inspection.control import ControlGains, FixedwingAutopilot
from envs.inspection.guidance import LocalTrack, moving_orbit_target


def run(seed, duration, initial_altitude, initial_airspeed, wind, gains=None,
        turn_at=None, blind_interval=None):
    flight = FlightWorld(seed, count=1, task="loiter", radius=90, jitter=False, wind_xy=wind)
    gains = gains or ControlGains()
    flight.controllers = [FixedwingAutopilot(gains)]
    world = flight.world
    drone = world.drones[0]
    # Initialization only, before any recorded physics integration.
    world.resetBasePositionAndOrientation(drone.Id, [0, -90, initial_altitude], [0, 0, 0, 1])
    world.resetBaseVelocity(drone.Id, [initial_airspeed+wind[0], wind[1], 0], [0, 0, 0])
    drone.update_state()
    flight.min_altitude = flight.max_altitude = initial_altitude
    flight.min_airspeed = flight.max_airspeed = initial_airspeed
    rng = np.random.default_rng(seed)
    initial_sample = rng.normal(0, 1, 2)
    track = LocalTrack.first(initial_sample, 0)
    measurements = [{"measured_at": 0., "estimate_xy": initial_sample.tolist(), "initial_public_sample": True}]
    trace, frames = [], []
    def target(_):
        return moving_orbit_target(world.state(0)[3, :2], track.predict(flight.t), track.velocity, flight.wind[:2])
    started = time.monotonic()
    error = None
    try:
        while flight.t < duration-1e-8:
            flight.step([target(0)])
            stamp = flight.t
            # Hidden motion belongs only to this sensor/auditor closure.
            hidden_xy = (np.array([1.5*stamp, 0.0]) if turn_at is None or stamp <= turn_at else
                         np.array([1.5*turn_at, 0.0])+1.5*(stamp-turn_at)*np.array([math.sqrt(.5), math.sqrt(.5)]))
            state = world.state(0)
            if world.physics_steps % 120 == 0:
                pos, quat = world.getBasePositionAndOrientation(drone.Id)
                origin = np.array(pos)+np.array(world.getMatrixFromQuaternion(quat)).reshape(3, 3)@np.array([0., 0., -.3])
                vector = np.r_[hidden_xy, .6]-origin
                aim = np.r_[track.predict(stamp), .6]-origin
                off_nadir = math.degrees(math.atan2(np.linalg.norm(vector[:2]), -vector[2]))
                angle = math.degrees(math.acos(float(np.clip(vector@aim/np.linalg.norm(vector)/np.linalg.norm(aim), -1, 1))))
                visible = off_nadir <= 50 and angle <= 10
                blind = blind_interval is not None and blind_interval[0] <= stamp < blind_interval[1]
                detected = visible and not blind and rng.random() >= .1
                if detected:
                    sample = hidden_xy+rng.normal(0, 1, 2)
                    track.update(sample, stamp)
                    measurements.append({"measured_at": stamp, "estimate_xy": sample.tolist()})
                frames.append({"t": stamp, "visible": bool(visible), "detected": bool(detected),
                               "off_nadir_degrees": off_nadir, "pointing_error_degrees": angle})
            if world.physics_steps % 24 == 0:
                trace.append({"t": stamp, "position": state[3].tolist(), "hidden_xy_for_audit_only": hidden_xy.tolist(),
                              "attitude": state[1].tolist(), "body_velocity": state[2].tolist(),
                              "estimated_center": track.predict(stamp).tolist(), "estimated_velocity": track.velocity.tolist(),
                              "center_estimation_error_m": float(np.linalg.norm(track.predict(stamp)-hidden_xy)),
                              "radial_error_to_true_center_m": float(abs(np.linalg.norm(state[3, :2]-hidden_xy)-90)),
                              "radial_error_to_estimated_center_m": float(abs(np.linalg.norm(state[3, :2]-track.predict(stamp))-90)),
                              "altitude_error_m": float(abs(state[3, 2]-96))})
    except Exception:
        error = traceback.format_exc()
    finally:
        safety = flight.summary()
        simulated = flight.t
        try:
            flight.close()
        except Exception:
            error = (error or "")+traceback.format_exc()
    stable = [r for r in trace if r["t"] >= 20]
    stamps = [r["measured_at"] for r in measurements]
    max_gap = max(b-a for a, b in zip(stamps, stamps[1:]+[simulated]))
    visible_fraction = sum(f["visible"] for f in frames)/len(frames) if frames else 0
    tube_pass = (bool(stable) and all(r["radial_error_to_true_center_m"] <= 6 and r["altitude_error_m"] <= 2 for r in stable))
    local_bins = [{"start": float(s), "end": min(float(s+3), simulated),
                   "has_actual_measurement": any(s < stamp <= min(s+3, simulated) for stamp in stamps)}
                  for s in range(0, math.ceil(simulated), 3)]
    return {"role": "single_aircraft_causal_guidance_falsification_not_qualification", "seed": seed,
            "requested_seconds": duration, "simulated_seconds": simulated, "wall_seconds": time.monotonic()-started,
            "initial_altitude_m": initial_altitude, "initial_airspeed_m_s": initial_airspeed, "wind_xy_m_s": wind,
            "control_gains": asdict(gains),
            "hidden_motion_fixture_turn_at": turn_at, "forced_sensor_blind_interval": blind_interval,
            "sensor_scope": "2Hz local noisy measurements, no transport/occlusion/classification in this isolated probe",
            "passed_physical_and_view_probe": not error and not safety["violations"] and visible_fraction >= .95 and max_gap <= 3,
            "provisional_6m_radial_2m_altitude_tube_passed": tube_pass,
            "visible_fraction": visible_fraction, "maximum_measurement_gap_seconds": max_gap,
            "settled_max_radial_error_m": max((r["radial_error_to_true_center_m"] for r in stable), default=None),
            "settled_max_altitude_error_m": max((r["altitude_error_m"] for r in stable), default=None),
            "settled_max_center_estimation_error_m": max((r["center_estimation_error_m"] for r in stable), default=None),
            "local_3s_measurement_bins": local_bins,
            "safety": safety, "operational_error": error, "trace": trace, "sensor_frames": frames, "measurements": measurements}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--initial-altitude", type=float, default=96)
    parser.add_argument("--initial-airspeed", type=float, default=18)
    parser.add_argument("--wind", type=float, nargs=2, default=[-2., 0.])
    parser.add_argument("--pitch-p", type=float, default=1.5)
    parser.add_argument("--elevator-trim", type=float, default=-.03)
    parser.add_argument("--altitude-i", type=float, default=.015)
    parser.add_argument("--turn-at", type=float)
    parser.add_argument("--blind-interval", type=float, nargs=2)
    args = parser.parse_args()
    if args.out.exists() or args.seconds <= 20:
        parser.error("choose a new output and a duration over20s")
    sources = {str(p.relative_to(ROOT)): p.read_text() for p in [*sorted((ROOT/"envs/inspection").glob("*.py")), Path(__file__)]}
    try:
        gains = replace(ControlGains(), pitch_p=args.pitch_p, elevator_trim=args.elevator_trim, altitude_i=args.altitude_i)
        result = run(args.seed, args.seconds, args.initial_altitude, args.initial_airspeed, args.wind, gains,
                     args.turn_at, args.blind_interval)
    except Exception:
        result = {"passed_physical_and_view_probe": False, "operational_error": traceback.format_exc()}
    result["source_snapshots"] = sources
    result["source_hashes"] = {p: hashlib.sha256(s.encode()).hexdigest() for p, s in sources.items()}
    result["sources_unchanged"] = all((ROOT/p).read_text() == s for p, s in sources.items())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("source_snapshots", "source_hashes", "trace", "sensor_frames", "measurements")}, indent=2))
    sys.exit(0 if result["passed_physical_and_view_probe"] and result["sources_unchanged"] else 1)
