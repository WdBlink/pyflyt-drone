"""One necessary-geometry falsifier, NOT a physical/qualification run.

Run once from repository root with .venv/bin/python. Only this directory is
written. No world, target generator, archived hidden_entities or paid API.
The target line below is an explicit navigation test input, not evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from envs.inspection.paths import Pose, Primitive, TimedPath


def main() -> None:
    started = time.monotonic()
    radius, airspeed, target_speed = 80.0, 18.0, 1.5
    wind = np.array([-2.0, 0.0])
    circle = TimedPath(
        (Primitive(Pose(0.0, -radius, 0.0), 2 * math.pi * radius, 1 / radius),),
        wind,
        airspeed,
    )
    # At the eastbound bottom phase, the straight advances the fixed circle's
    # center exactly. Include target motion DURING the straight in its length.
    straight_ground_speed = airspeed + wind[0]
    shift = target_speed * circle.duration / (1 - target_speed / straight_ground_speed)
    line_seconds = shift / straight_ground_speed
    period = circle.duration + line_seconds
    target_initial_x = -target_speed * circle.duration / 2
    stamps = np.r_[np.arange(0, 60.0, 0.01), 60.0]
    phase_positions, frozen_positions = [], []
    banks = []
    for stamp in stamps:
        lap = int(stamp // period)
        local = stamp - lap * period
        if local <= circle.duration:
            row = circle.at(local)
            position = row[:2] + np.array([lap * shift, 0.0])
            banks.append(abs(float(row[4])))
        else:
            position = np.array([lap * shift + straight_ground_speed * (local - circle.duration), -radius])
            banks.append(0.0)
        phase_positions.append(position)
        frozen_positions.append(circle.at(stamp % circle.duration)[:2])
    target = np.column_stack((target_initial_x + target_speed * stamps, np.zeros(len(stamps))))
    phase_dist = np.linalg.norm(np.array(phase_positions) - target, axis=1)
    frozen_dist = np.linalg.norm(np.array(frozen_positions) - target, axis=1)
    # Deliberately more conservative than the already-failed 6m/2m tube.
    # These are illustrative UNQUALIFIED allowances, not fitted guarantees.
    xy_allowance, altitude_allowance, target_allowance = 10.0, 3.0, 4.0
    optical_height = 96.0 - altitude_allowance - 0.3 - 0.6
    reach = optical_height * math.tan(math.radians(50))
    # |aircraft velocity| <=20 and explicit target input speed1.5. The nearest
    # 0.01s sample is <=0.005s away, so this bounds ideal curve sample omission.
    interpolation_allowance = (20.0 + target_speed) * 0.005
    horizontal_optical_origin_allowance = 0.3
    allowance = (xy_allowance + target_allowance + interpolation_allowance
                 + horizontal_optical_origin_allowance)
    bounded_phase_max = float(phase_dist.max()) + allowance
    # A candidate already seen to the north must be turned toward immediately.
    # Compare early and late gate capture from the SAME explicit initial state.
    acquisition = []
    for gate in (1.5, 3.5):
        center_x = straight_ground_speed * gate
        future = np.r_[np.arange(0, circle.duration, 0.01), circle.duration]
        own = np.array([circle.at(float(t))[:2] for t in future]) + np.array([center_x, 0.0])
        aim = np.column_stack((target_speed * (gate + future), np.zeros(len(future))))
        maximum = float(np.linalg.norm(own - aim, axis=1).max())
        acquisition.append({"capture_at_s": gate, "ideal_max_distance_m": maximum,
                            "with_illustrative_allowance_m": maximum + allowance,
                            "optical_necessary_screen_passes": maximum + allowance <= reach})
    output = {
        "role": "single_analytic_necessary_geometry_falsifier_not_physics_not_qualification",
        "input": {"radius_m": radius, "airspeed_m_s": airspeed, "wind_xy_m_s": wind.tolist(),
                  "explicit_navigation_target_speed_m_s": target_speed, "seconds": 60,
                  "initial_target_offset_m": [target_initial_x, 0.0]},
        "geometry": {"circle_period_s": circle.duration, "center_shift_m": shift,
                     "straight_seconds": line_seconds, "repeat_period_s": period,
                     "phase_gated_max_ideal_distance_m": float(phase_dist.max()),
                     "fixed_circle_max_ideal_distance_m": float(frozen_dist.max()),
                     "max_nominal_feedforward_bank_degrees": math.degrees(max(banks))},
        "illustrative_unqualified_allowances": {"own_xy_m": xy_allowance,
                     "own_altitude_m": altitude_allowance, "target_error_m": target_allowance,
                     "optical_origin_and_target_height_m": 0.9,
                     "horizontal_optical_origin_allowance_m": horizontal_optical_origin_allowance,
                     "ideal_sampling_omission_bound_m": interpolation_allowance,
                     "reduced_horizontal_reach_m": reach,
                     "phase_gated_distance_upper_m": bounded_phase_max},
        "acquisition_delay_sensitivity": acquisition,
        "necessary_geometry_rejected": bounded_phase_max > reach,
        "excluded": ["curvature-ramp geometry and roll transient", "actual navigation capture",
                     "Gaussian error support or unknown target turns", "narrow-axis pointing error",
                     "ray occlusion/health/misses/classification/association/upload",
                     "fleet reservation/home recovery/tail proof", "1x wall clock"],
        "interpretation": "The phase matters: a full-circle radius+center-offset screen is overly pessimistic. "
                          "This ideal input survives a conservative off-nadir screen; the delayed initial capture does not. "
                          "It is not evidence that an aircraft or dynamic task passes.",
        "wall_seconds": time.monotonic() - started,
    }
    destination = Path(__file__).with_name("analytic_falsifier.json")
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
