"""Small analytical falsification probe. No simulator, dataset, or research state.

This is necessary geometry, not physical qualification. Run with Python 3.10+.
The moving-center calculation uses synthetic velocity, not scenario metadata.
"""
from __future__ import annotations

import json
import math


def add(a, b):
    return a[0] + b[0], a[1] + b[1]


def scale(a, k):
    return a[0] * k, a[1] * k


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def stripe_connector(source_y, target_y, radius=100.0):
    """Two quarter arcs plus an optional vertical straight, exact endpoints."""
    sign = 1.0 if target_y > source_y else -1.0
    assert abs(target_y - source_y) >= 2 * radius
    start = (-250.0, source_y - 80.0)
    p = start
    heading = 0.0
    points = [start]
    for curvature, length in ((sign / radius, math.pi * radius / 2),
                              (0.0, abs(target_y - source_y) - 2 * radius),
                              (-sign / radius, math.pi * radius / 2)):
        if not length:
            continue
        segments = max(1, math.ceil(length / 0.5))
        for _ in range(segments):
            ds = length / segments
            if curvature:
                new_heading = heading + curvature * ds
                p = add(p, ((math.sin(new_heading) - math.sin(heading)) / curvature,
                            (-math.cos(new_heading) + math.cos(heading)) / curvature))
                heading = new_heading
            else:
                p = add(p, scale((math.cos(heading), math.sin(heading)), ds))
            points.append(p)
    expected = (-50.0, target_y - 80.0)
    assert math.dist(p, expected) < 1e-8
    assert abs(heading) < 1e-10
    assert all(-250.000001 <= x <= -49.999999 and -380.000001 <= y <= 220.000001
               for x, y in points)
    return {"from_y": source_y, "to_y": target_y,
            "length_m": math.pi * radius + abs(target_y - source_y) - 2 * radius,
            "time_s_at_20": (math.pi * radius + abs(target_y - source_y) - 2 * radius) / 20,
            "endpoint": p, "max_abs_xy": max(abs(x) for point in points for x in point)}


def moving_orbit_banks(radius, center_velocity=(1.5, 0.0), wind=(-2.0, 0.0), airspeed=20.0):
    """Exact constant airspeed for a circle translating at constant velocity.

    r=(cos(theta),sin(theta)), t=(-sin(theta),cos(theta));
    air velocity u=d+s*t, d=center_velocity-wind, |u|=airspeed.
    s=-d.t+sqrt((d.t)^2+airspeed^2-|d|^2), theta_dot=s/radius.
    Differentiate u with theta to obtain exact air heading rate and bank.
    """
    d = add(center_velocity, scale(wind, -1))
    banks = []
    for k in range(3600):
        theta = 2 * math.pi * k / 3600
        radial = (math.cos(theta), math.sin(theta))
        tangent = (-radial[1], radial[0])
        a, b = dot(d, tangent), dot(d, radial)
        q = math.sqrt(a*a + airspeed*airspeed - dot(d, d))
        s = -a + q
        derivative_s = b * (1-a/q)
        u = add(d, scale(tangent, s))
        du_dt = scale(add(scale(tangent, derivative_s), scale(radial, -s)), s/radius)
        rate = cross(u, du_dt)/(airspeed*airspeed)
        banks.append(math.degrees(math.atan(airspeed*rate/9.81)))
        assert abs(math.hypot(*u)-airspeed) < 1e-10
    return {"radius_m": radius, "airspeed_m_s": airspeed, "center_velocity_m_s": center_velocity,
            "wind_m_s": wind, "bank_degrees_min_max": [min(banks), max(banks)]}


def main():
    centers = (-300, -100, 100, 300)
    connectors = [stripe_connector(a, b) for a in centers for b in centers if a != b]
    moving = [moving_orbit_banks(r) for r in (80, 90, 100, 105)]
    selected = moving_orbit_banks(90, airspeed=18)
    # This negative is the point: a reused 80 m circle cannot cover the demand.
    assert moving[0]["bank_degrees_min_max"][1] > 30
    assert moving[2]["bank_degrees_min_max"][1] < 30
    assert selected["bank_degrees_min_max"][1] < 28
    visibility = {str(h): h*math.tan(math.radians(50)) for h in (72, 80, 88, 96)}
    result = {
        "kind": "analytical_only_not_physics_or_qualification",
        "connectors": connectors,
        "moving_orbit_at_20m_s": moving,
        "selected_observe_18m_s_90m": selected,
        "horizontal_view_limit_m": visibility,
        "nominal_observe_radial_margin_m_at_96m": visibility["96"]-90,
        "observe_view_limit_with_2m_vertical_error_and_0_3m_camera_offset_m": (96-2-.3)*math.tan(math.radians(50)),
        "observe_remaining_view_margin_after_6m_xy_tube_and_5m_estimate_radius_m": (96-2-.3)*math.tan(math.radians(50))-90-6-5,
        "survey_capsule_length_m": 1000+2*math.pi*80,
        "survey_lap_s_at_20": (1000+2*math.pi*80)/20,
        "adjacent_capsules_nominal_gap_m": 200-2*80,
        "tube_m_provisional_not_qualified": 6,
        "required_nominal_pair_clearance_m": 15+2*6,
        "continuous_sampling_extra_margin_m_at_50ms_vmax27": 2*27*.05/2,
        "limitations": ["No controller, airframe, roll transient, estimator, occlusion, or traffic execution is run.",
                        "1.5 m/s is a synthetic probe value, not an authorized target speed bound.",
                        "A 6 m tracking tube is a hypothesis requiring new physical qualification."]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
