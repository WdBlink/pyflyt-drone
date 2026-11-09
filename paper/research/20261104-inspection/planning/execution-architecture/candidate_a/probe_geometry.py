"""Small analytic geometry falsifier; no PyFlyt, mission, truth or research state.

This is an ideal constant-speed point-mass witness, not a flight qualification.
It deliberately does not implement a production path planner or controller.
"""

from __future__ import annotations

import itertools
import json
import math


SPEED = 20.0
HALF_LENGTH = 250.0
SURVEY_RADIUS = 80.0
TRANSIT_RADIUS = 90.0
CENTERS = (-300.0, -100.0, 100.0, 300.0)
ALTITUDES = (72.0, 80.0, 88.0, 96.0)
LAP_LENGTH = 4 * HALF_LENGTH + 2 * math.pi * SURVEY_RADIUS
LAP_TIME = LAP_LENGTH / SPEED
DT = 0.05
TUBE_RADIUS = 6.0
NOMINAL_CLEARANCE = 15 + 2 * TUBE_RADIUS


def capsule(center_y: float, distance: float) -> tuple[float, float]:
    """Exact CCW capsule point, distance from lower-left eastbound gate."""
    distance %= LAP_LENGTH
    if distance < 2 * HALF_LENGTH:
        return -HALF_LENGTH + distance, center_y - SURVEY_RADIUS
    distance -= 2 * HALF_LENGTH
    if distance < math.pi * SURVEY_RADIUS:
        angle = -math.pi / 2 + distance / SURVEY_RADIUS
        return HALF_LENGTH + SURVEY_RADIUS * math.cos(angle), center_y + SURVEY_RADIUS * math.sin(angle)
    distance -= math.pi * SURVEY_RADIUS
    if distance < 2 * HALF_LENGTH:
        return HALF_LENGTH - distance, center_y + SURVEY_RADIUS
    distance -= 2 * HALF_LENGTH
    angle = math.pi / 2 + distance / SURVEY_RADIUS
    return -HALF_LENGTH + SURVEY_RADIUS * math.cos(angle), center_y + SURVEY_RADIUS * math.sin(angle)


def transfer(source: float, destination: float, distance: float, origin_x: float = -HALF_LENGTH) -> tuple[float, float]:
    """LSL gate connector: total turns 2 pi, straight abs(destination-source)."""
    y0, y1 = source - SURVEY_RADIUS, destination - SURVEY_RADIUS
    first_angle = math.pi / 2 if y1 > y0 else 3 * math.pi / 2
    first_length = TRANSIT_RADIUS * first_angle
    delta = abs(y1 - y0)
    if distance <= first_length:
        angle = -math.pi / 2 + distance / TRANSIT_RADIUS
        return origin_x + TRANSIT_RADIUS * math.cos(angle), y0 + TRANSIT_RADIUS * (1 + math.sin(angle))
    distance -= first_length
    tangent_x = origin_x + (TRANSIT_RADIUS if y1 > y0 else -TRANSIT_RADIUS)
    if distance <= delta:
        return tangent_x, y0 + TRANSIT_RADIUS + math.copysign(distance, y1 - y0)
    distance -= delta
    angle = -math.pi / 2 + first_angle + distance / TRANSIT_RADIUS
    return origin_x + TRANSIT_RADIUS * math.cos(angle), y1 + TRANSIT_RADIUS * (1 + math.sin(angle))


def transfer_length(source: float, destination: float) -> float:
    return 2 * math.pi * TRANSIT_RADIUS + abs(destination - source)


def aircraft_position(aircraft: int, destination: int, depart: float | None, t: float) -> tuple[float, float, float]:
    source = CENTERS[aircraft]
    if depart is None or t < depart:
        xy = capsule(source, SPEED * t)
    else:
        # Any eastbound bottom-leg point is a valid tangent gate. The phase
        # is a reservation choice; the aircraft actually flies until then.
        phase = depart % LAP_TIME
        if phase > 2 * HALF_LENGTH / SPEED + 1e-9:
            raise ValueError("departure must occur on a bottom leg")
        origin_x = -HALF_LENGTH + SPEED * phase
        distance = SPEED * (t - depart)
        length = transfer_length(source, CENTERS[destination])
        xy = transfer(source, CENTERS[destination], distance, origin_x) if distance < length else capsule(CENTERS[destination], distance - length + SPEED * phase)
    return *xy, ALTITUDES[aircraft]


def separation_witness(destinations: tuple[int, ...], departures: tuple[float | None, ...]) -> dict:
    """Sample a finite non-physics route set, conservatively correcting gaps.

    After the last arrival every route has the same period. One extra period
    therefore checks the entire ideal periodic tail. A pair's distance is
    40 m/s Lipschitz; nearest sample correction is 40*DT/2 = 1 m.
    """
    last_arrival = max((depart + transfer_length(CENTERS[i], CENTERS[destinations[i]]) / SPEED
                        for i, depart in enumerate(departures) if depart is not None), default=0)
    end = last_arrival + LAP_TIME
    minimum, witness = math.inf, None
    for tick in range(math.ceil(end / DT) + 1):
        t = min(tick * DT, end)
        positions = [aircraft_position(i, destinations[i], departures[i], t) for i in range(4)]
        for i, j in itertools.combinations(range(4), 2):
            distance = math.dist(positions[i], positions[j])
            if distance < minimum:
                minimum, witness = distance, (round(t, 2), i, j)
    return {"sample_min_m": minimum, "continuous_nominal_lower_bound_m": minimum - SPEED * DT,
            "witness_t_and_pair": witness, "analytic_seconds_checked": end,
            "passes_nominal_tube_test": minimum - SPEED * DT >= NOMINAL_CLEARANCE}


def main():
    max_abs_x = max_abs_y = 0.0
    maximum_endpoint_error = 0.0
    maximum_transfer_seconds = 0.0
    for source, destination in itertools.permutations(CENTERS, 2):
        length = transfer_length(source, destination)
        maximum_transfer_seconds = max(maximum_transfer_seconds, length / SPEED)
        for sample in range(2001):
            x, y = transfer(source, destination, length * sample / 2000)
            max_abs_x, max_abs_y = max(max_abs_x, abs(x)), max(max_abs_y, abs(y))
        maximum_endpoint_error = max(maximum_endpoint_error, math.dist(transfer(source, destination, length),
                                                                      (-HALF_LENGTH, destination - SURVEY_RADIUS)))
    assert maximum_endpoint_error < 1e-10
    assert max_abs_x < 341 and max_abs_y < 401

    # A cyclic permutation actually assigns every aircraft to another stripe.
    # This single geometric stress witness is not a benchmark or a grid run.
    destinations = (1, 2, 3, 0)
    simultaneous = separation_witness(destinations, (0.0, 0.0, 0.0, 0.0))
    # Chosen deterministic second witness: all three upward motions run in
    # parallel; the long downward crossing waits one actual capsule lap.
    staggered = separation_witness(destinations, (0.0, 0.0, 0.0, LAP_TIME))
    # A constructive feasible *ideal* witness: leave the first three gates
    # unchanged; aircraft 3 flies eight seconds further along its bottom leg.
    phased_departures = (0.0, 0.0, 0.0, 8.0)
    phased = separation_witness(destinations, phased_departures)
    assert phased["passes_nominal_tube_test"]
    def orbit_geometry(radius, altitude, altitude_error, uncertainty, flight_error, relative_drift):
        camera_min_z = altitude - altitude_error - .3
        visible_radius = camera_min_z * math.tan(math.radians(50))
        optical_xy_offset = .3  # conservative norm bound for rotated body offset
        # For uniform relative drift U and fixed airspeed V, max acceleration
        # on a translating circle is (V+U)^2/R. At the down-drift tangent, the
        # changing crab angle contributes: V*(V+U)/R would underestimate it.
        # Center acceleration, airspeed error and roll-in are NOT included.
        bank = math.degrees(math.atan((SPEED + relative_drift) ** 2 / (9.81 * radius)))
        return {"radius_m": radius, "commanded_altitude_m": altitude,
                "assumed_altitude_error_m": altitude_error,
                "assumed_min_camera_z_m": camera_min_z,
                "50deg_horizontal_reach_m": visible_radius,
                "radius_plus_uncertainty_flight_and_optical_offset_m": radius + uncertainty + flight_error + optical_xy_offset,
                "remaining_horizontal_margin_m": visible_radius - radius - uncertainty - flight_error - optical_xy_offset,
                "max_bank_deg_uniform_relative_drift_only": bank,
                "ideal_orbit_period_seconds_stationary_center": 2 * math.pi * radius / SPEED,
                "radius_interval_for_28deg_bank_and_visibility_m": [
                    (SPEED + relative_drift) ** 2 / (9.81 * math.tan(math.radians(28))),
                    visible_radius - uncertainty - flight_error - optical_xy_offset]}
    output = {
        "scope": "ideal analytic geometry only; no physical or stochastic guarantees",
        "all_12_ordered_stripe_pairs": {"max_abs_x_m_sampled": max_abs_x,
                                          "max_abs_y_m_sampled": max_abs_y,
                                          "max_endpoint_error_m": maximum_endpoint_error,
                                          "max_transfer_seconds": maximum_transfer_seconds,
                                          "analytic_envelope_xy_m": [[-340, -160], [-380, 400]]},
        "capsule_lap_seconds": LAP_TIME,
        "cyclic_permutation_simultaneous": simultaneous,
        "cyclic_permutation_one_lap_stagger": staggered,
        "cyclic_permutation_distinct_tangent_gates": {"departure_seconds": phased_departures, **phased},
        "nominal_required_separation_for_6m_tubes_m": NOMINAL_CLEARANCE,
        "observe_geometry_moderate_relative_drift_1p5m_s": orbit_geometry(95, 96, 2, 6, 6, 1.5),
        "observe_geometry_full_corner_relative_drift_3p5m_s": orbit_geometry(106, 98, 1, 4, 4, 3.5),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
