"""One bounded physical falsifier; not fleet, sensor, tube, or 1x qualification.

Post-run review found a frame error in predictor airspeed initialization below.
The experiment is retained, not silently rerun. See rationale.md. Physical
control uses FlightWorld's independent world-velocity airdata and is unaffected.
Do not reuse this predictor as a validated implementation.
"""

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))

from envs.inspection.control import FixedwingAutopilot, air_heading_for_course, inspection_gains, wrap_angle
from envs.inspection.flight import FlightWorld
from envs.inspection.guidance import LocalTrack, moving_orbit_target


def law(xy, center, velocity, wind):
    return moving_orbit_target(xy, center, velocity, wind, radius=80., altitude=96., airspeed=18.)


def reduced_step(state, anchor, at, wind, dt):
    xy, yaw, roll, speed = state
    pos, vel, stamp = anchor
    target = law(xy, pos + vel * (at-stamp), vel, wind)
    air_heading = air_heading_for_course(target.heading, wind, target.airspeed)
    bank = np.clip(-.8*wrap_angle(air_heading-yaw)+target.bank_feedforward, -math.pi/6, math.pi/6)
    roll += (bank-roll)/.45*dt
    yaw += -9.81*math.tan(roll)/max(speed, 5.)*dt
    speed += (target.airspeed-speed)/2.*dt
    xy = xy+(speed*np.array([math.cos(yaw), math.sin(yaw)])+wind)*dt
    return xy, yaw, roll, speed


def main():
    started = time.monotonic()
    flight = FlightWorld(217, count=1, task="survey", wind_xy=(2., 0.), altitudes=[96.], jitter=True)
    flight.controllers = [FixedwingAutopilot(inspection_gains())]
    rng = np.random.default_rng(90217)
    wind = np.array([2., 0.])
    track = None
    anchor = None
    predicted = None
    one_step_error = []
    visible_count = sample_count = 0
    accepted_stamps = []
    rows = []
    try:
        while flight.t < 55.-1e-8:
            stamp = flight.t
            state = flight.world.state(0)
            actual = state[3]
            # Only this synthetic sensor/auditor can evaluate the fixture motion.
            truth = np.array([-250.+80.*math.sin(1.5*stamp/80.), 0.])
            if round(stamp*120) % 60 == 0:
                sample_count += 1
                drone = flight.world.drones[0]
                base, quaternion = flight.world.getBasePositionAndOrientation(drone.Id)
                rotation = np.array(flight.world.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
                camera = np.array(base)+rotation@np.array([0., 0., -.3])
                off_nadir = math.atan2(np.linalg.norm(truth-camera[:2]), camera[2]-.6)
                actual_axis = np.r_[truth-camera[:2], .6-camera[2]]
                aimed_axis = np.r_[track.predict(stamp)-camera[:2], -camera[2]] if track else np.array([0.,0.,-1.])
                cosine = float(actual_axis@aimed_axis)/(np.linalg.norm(actual_axis)*np.linalg.norm(aimed_axis))
                visible = off_nadir <= math.radians(50.) and (track is None or cosine >= math.cos(math.radians(10.)))
                visible_count += int(visible)
                if visible and rng.random() >= .1:
                    measured = truth+rng.normal(0., 1. if track else 4., 2)
                    if track is None:
                        track = LocalTrack.first(measured, stamp)
                    else:
                        track.update(measured, stamp)
                    accepted_stamps.append(stamp)
            if track is None:
                raise RuntimeError("fixture failed initial actual wide frame")
            if anchor is None or round(stamp*120) % 120 == 0:
                anchor = (track.position.copy(), track.velocity.copy(), track.measured_at)
                # Known experiment defect: state[2] is BODY-frame velocity,
                # but wind is WORLD-frame. Retained to identify the old result.
                airspeed = np.linalg.norm(state[2]-np.r_[wind, 0.])
                predicted = (actual[:2].copy(), state[1, 2], state[1, 0], airspeed)
            pos, vel, measured_at = anchor
            target = law(actual[:2], pos+vel*(stamp-measured_at), vel, wind)
            predicted = reduced_step(predicted, anchor, stamp, wind, 1/120)
            flight.step([target], exit_flags=[False])
            end_actual = flight.world.state(0)[3]
            error = float(np.linalg.norm(end_actual[:2]-predicted[0]))
            one_step_error.append(error)
            if round(flight.t*120) % 12 == 0:
                rows.append({"t":flight.t,"position":end_actual.tolist(),"predictor_xy_error_m":error})
        gaps = np.diff(np.r_[0., accepted_stamps, flight.t])
        result = {
            "role":"single_fixture_falsifier_not_admission_sensor_tube_or_1x_evidence",
            "seconds":flight.t,"wall_seconds":time.monotonic()-started,
            "fixture":"initial offset80m, sinusoidal1.5m/s proxy,2m/s wind, causal noisy2Hz,10percent misses",
            "navigation":"same frozen radius80 vector field and own measured state; refreshed once per simulated second without planner/admission",
            "predictor":"unfitted roll_tau.45s/speed_tau2s, reset each1s; no z model",
            "visible_fraction":visible_count/sample_count,
            "maximum_new_measurement_gap_s":float(gaps.max()),
            "max_1s_predictor_xy_error_m":max(one_step_error),
            "safety":flight.summary(),
            "limitations":["no obstacles or ray test","initial association supplied by fixture","no fleet admission or recovery","not180s grading","not frozen source qualification","no scheduling latency"],
            "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ["envs/inspection/control.py","envs/inspection/flight.py","envs/inspection/guidance.py"]},
            "trace":rows,
        }
        (Path(__file__).parent/"probe_field.json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("trace","source_sha256")},indent=2))
    finally:
        flight.close()


if __name__ == "__main__":
    main()
