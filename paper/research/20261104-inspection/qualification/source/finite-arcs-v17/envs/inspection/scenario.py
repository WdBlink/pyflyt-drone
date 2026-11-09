"""Dev generator and hidden truth; only public_card is an actor input."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


CLASSES = ("road_obstruction", "bank_anomaly", "infrastructure_damage", "inspection_vehicle", "normal")
ROI = ((-380, -380), (340, -380), (380, -160), (310, 10), (380, 260),
       (270, 380), (-300, 380), (-380, 150), (-300, -30))


def in_polygon(x, y, polygon=ROI):
    inside = False
    for (ax, ay), (bx, by) in zip(polygon, (*polygon[1:], polygon[0])):
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            inside = not inside
    return inside


@dataclass(frozen=True)
class Entity:
    key: str
    object_type: str
    true_class: str
    origin_xy: tuple[float, float]
    appeared_at: float = 0.0
    speed: float = 0.0
    rule_id: str = ""

    def position(self, t):
        if t < self.appeared_at:
            return None
        x, y = self.origin_xy
        return (x + (80*math.sin(self.speed*(t-self.appeared_at)/80) if self.speed else 0), y)


def generate_dev(seed, reference=False):
    rng = np.random.default_rng(seed)
    entities = []
    counts = [2, 2, 2] if reference else rng.integers(1, 4, 3).tolist()
    for class_index, count in enumerate(counts):
        for _ in range(count):
            for attempt in range(10000):
                stripe = int(rng.integers(0, 4))
                point = (float(rng.uniform(-220, 220)), float(-300+200*stripe+rng.uniform(-55, 55)))
                if in_polygon(*point) and all(math.dist(point, e.origin_xy) > 45 for e in entities):
                    break
            else:
                raise RuntimeError("unable to place a separated static entity")
            entities.append(Entity(f"truth-{len(entities):03d}", "static", CLASSES[class_index], point,
                                   rule_id=f"static-{class_index}"))
    for index in range(3 if reference else int(rng.integers(2, 5))):
        stripe = index % 4
        entities.append(Entity(f"truth-{len(entities):03d}", "dynamic", "inspection_vehicle",
                               (float(rng.uniform(-100, 100)), float(-300+200*stripe+rng.uniform(-15, 15))),
                               120+60*index, 1.5, "dynamic-inspection"))
    for _ in range(6 if reference else int(rng.integers(4, 9))):
        for attempt in range(10000):
            point = tuple(float(x) for x in rng.uniform(-340, 340, 2))
            if in_polygon(*point) and all(math.dist(point, e.origin_xy) > 45 for e in entities):
                break
        else:
            raise RuntimeError("unable to place a separated distractor")
        entities.append(Entity(f"truth-{len(entities):03d}", "static", "normal", point))
    return entities


def public_card():
    return {"task_card_version": 1, "effective_from": 0.0, "deadline": 600.0,
            "roi_polygon": ROI, "flight_bounds_xy": [-550.0, 550.0], "terrain_elevation_m": 0.0,
            "altitude_limits_agl_m": [60.0, 100.0], "minimum_separation_m": 15.0,
            "report_budget": 10, "classes": CLASSES,
            "range_budget_m_per_aircraft": 18000.0, "exit_range_reserve_m": 1000.0,
            "regions": [{"region_id": f"stripe-{i}", "center_y": -300+200*i,
                         "lower_y": -400+200*i, "upper_y": -200+200*i} for i in range(4)],
            "rules": [{"rule_id": f"static-{i}", "object_type": "static", "classes": [c], "priority": 1}
                      for i, c in enumerate(CLASSES[:3])]
                     + [{"rule_id": "dynamic-inspection", "object_type": "dynamic",
                         "classes": ["inspection_vehicle"], "priority": 1}],
            "dynamic_requirement": {"window_seconds": 180, "bin_seconds": 3, "grace_seconds": 6},
            "exit_condition": {"radius_m": 80.0, "radius_tolerance_m": 10.0,
                               "course_tolerance_degrees": 25, "continuous_dwell_seconds": 20},
            "sensor": {"kind": "structured synthetic", "gimbal": "ideal stabilized; nadir survey, causal candidate-pointed inspection",
                       "survey_fov_degrees": 110, "inspect_fov_degrees": 20,
                       "optical_origin_body_m": [0, 0, -.3],
                       "maximum_ground_off_nadir_degrees": 50, "sample_hz": 2},
            "tool_schema_version": "inspection-executor-dev-v7",
            "execution_semantics": {"nominal_airspeed_m_s":17,
                "survey_duration_start":"actual route capture, plus a completed physical pass",
                "observe_duration_start":"first new qualified local measurement",
                "acquisition_timeout_seconds":90,
                "local_observation_bins_seconds":3,
                "observation_gaps":"retained; predictions are not measurements",
                "local_stale_track_seconds":3,
                "stale_camera":"wide actual sensor reacquisition, covariance-scale position gate",
                "vertical_command_limit_m_s":1.2,
                "plan_capture_lead_seconds":.5,
                "inspect_camera_trial_seconds":0,
                "observation_navigation":"finite fixed arcs/lines with2s curvature ramps and reserved return",
                "observation_nominal_radius_options_m":[70,80],
                "shared_initial_altitude_agl_m":96,
                "delivery_grace_seconds":6,
                "qualification_status":"development_unqualified_new_modes"}}
