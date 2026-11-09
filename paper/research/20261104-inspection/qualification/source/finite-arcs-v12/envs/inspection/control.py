"""Conventional fixedwing guidance using only own-aircraft navigation state.

PyFlyt uses an ENU world and body x forward, y left, z up. Positive pitch is
nose down; positive roll banks right. Commands address mode -1 surfaces
explicitly, avoiding the upstream mode 0 tail-surface coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class FlightTarget:
    heading: float
    altitude: float = 80.0
    airspeed: float = 20.0
    bank_feedforward: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (
            self.heading, self.altitude, self.airspeed, self.bank_feedforward
        )) or self.altitude <= 0.0 or self.airspeed <= 0.0:
            raise ValueError("flight target must be finite with positive altitude and airspeed")


@dataclass(frozen=True)
class ControlGains:
    heading_p: float = 0.8
    roll_p: float = 1.5
    roll_d: float = 0.3
    pitch_p: float = 1.5
    pitch_d: float = 0.4
    altitude_p: float = 0.3
    altitude_i: float = 0.015
    speed_p: float = 0.08
    speed_i: float = 0.015
    pitch_trim: float = -0.04
    elevator_trim: float = -0.03
    throttle_trim: float = 0.4
    altitude_throttle_p: float = 0.0
    max_vertical_speed: float | None = None
    max_bank: float = math.radians(30.0)
    max_pitch: float = math.radians(12.0)


class FixedwingAutopilot:
    """Nested PI/PD loops, shared by every prospective mission policy."""

    def __init__(self, gains: ControlGains | None = None):
        self.gains = gains or ControlGains()
        self.altitude_integral = 0.0
        self.speed_integral = 0.0

    def step(self, state: np.ndarray, target: FlightTarget, dt: float,
             measured_airspeed: float | None = None) -> np.ndarray:
        if state.shape != (4, 3) or not np.isfinite(state).all():
            raise ValueError("navigation state must be a finite (4, 3) array")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        rates, attitude, velocity, position = state
        roll, pitch, heading = attitude
        g = self.gains
        speed = float(np.linalg.norm(velocity)) if measured_airspeed is None else measured_airspeed
        if not math.isfinite(speed) or speed < 0:
            raise ValueError("measured airspeed must be finite and nonnegative")
        height_error = target.altitude - position[2]
        speed_error = target.airspeed - speed
        proposed_integral = float(np.clip(self.altitude_integral + height_error * dt, -20.0, 20.0))
        proposed_vertical = g.altitude_p*height_error+g.altitude_i*proposed_integral
        saturated = (g.max_vertical_speed is not None and abs(proposed_vertical) > g.max_vertical_speed
                     and height_error*proposed_vertical > 0)
        if not saturated:
            self.altitude_integral = proposed_integral
        self.speed_integral = float(np.clip(self.speed_integral + speed_error * dt, -15.0, 15.0))
        bank = float(np.clip(
            -g.heading_p * wrap_angle(target.heading - heading) + target.bank_feedforward,
            -g.max_bank, g.max_bank,
        ))
        vertical_command = g.altitude_p*height_error+g.altitude_i*self.altitude_integral
        if g.max_vertical_speed is not None:
            vertical_command = float(np.clip(vertical_command,-g.max_vertical_speed,g.max_vertical_speed))
        pitch_target = float(np.clip(
            g.pitch_trim - math.atan2(vertical_command, max(speed, 5.0)),
            -g.max_pitch, g.max_pitch,
        ))
        aileron = float(np.clip(g.roll_p * (bank - roll) - g.roll_d * rates[0], -1.0, 1.0))
        elevator = float(np.clip(g.elevator_trim + g.pitch_p * (pitch_target - pitch) - g.pitch_d * rates[1], -1.0, 1.0))
        turn_rate = -9.81 * math.tan(bank) / max(speed, 5.0)
        rudder = float(np.clip(0.4 * (rates[2] - turn_rate * math.cos(roll)), -0.5, 0.5))
        throttle = float(np.clip(g.throttle_trim + g.speed_p * speed_error + g.speed_i * self.speed_integral
                                 + g.altitude_throttle_p * height_error, 0.0, 1.0))
        return np.array([aileron, -aileron, elevator, rudder, 0.0, throttle])


def inspection_gains():
    """Development profile with bounded vertical command and anti-windup.

    The1.2 value limits the desired vertical command, not measured climb rate.
    It requires new qualification; legacy ControlGains defaults are retained.
    """
    return ControlGains(pitch_p=4.5,elevator_trim=-.1,speed_p=.12,max_vertical_speed=1.2)


def orbit_target(position: np.ndarray, center: np.ndarray, radius: float, altitude: float = 80.0,
                 airspeed: float = 20.0, direction: int = 1) -> FlightTarget:
    """Tangent vector field; direction +1 is counterclockwise in ENU."""
    if radius <= 0.0 or not math.isfinite(radius) or direction not in (-1, 1):
        raise ValueError("radius must be positive and direction must be +/-1")
    offset = position[:2] - center[:2]
    distance = float(np.linalg.norm(offset))
    radial = math.atan2(offset[1], offset[0])
    heading = radial + direction * (math.pi / 2.0 + math.atan(2.0 * (distance - radius) / radius))
    feedforward = -direction * math.atan2(airspeed ** 2, 9.81 * radius)
    return FlightTarget(heading, altitude, airspeed, feedforward)


def air_heading_for_course(course: float, wind_xy: np.ndarray, airspeed: float) -> float:
    """Crab correction from shared ideal local airdata; no mission truth input."""
    if wind_xy.shape != (2,) or not np.isfinite(wind_xy).all() or not math.isfinite(airspeed) or airspeed <= 0:
        raise ValueError("finite planar wind and positive airspeed required")
    crosswind = -wind_xy[0] * math.sin(course) + wind_xy[1] * math.cos(course)
    if abs(crosswind) >= airspeed:
        raise ValueError("requested course is infeasible in this crosswind")
    return wrap_angle(course - math.asin(crosswind / airspeed))


class SurveyRoute:
    """Counterclockwise capsule with physical semicircle transitions."""

    def __init__(self, center_y: float, half_length: float = 250.0, radius: float = 80.0):
        self.center_y, self.half_length, self.radius = center_y, half_length, radius
        self.leg = "bottom"
        self.laps = 0

    def target(self, position: np.ndarray, altitude: float, airspeed: float = 20.) -> FlightTarget:
        x, y = position[:2]
        if self.leg == "bottom" and x >= self.half_length:
            self.leg = "right"
        elif self.leg == "right" and x <= self.half_length + 5 and y > self.center_y + 20:
            self.leg = "top"
        elif self.leg == "top" and x <= -self.half_length:
            self.leg = "left"
        elif self.leg == "left" and x >= -self.half_length - 5 and y < self.center_y - 20:
            self.leg = "bottom"
            self.laps += 1
        if self.leg in ("right", "left"):
            center = np.array([self.half_length if self.leg == "right" else -self.half_length, self.center_y])
            return orbit_target(position, center, self.radius, altitude,airspeed)
        desired_y = self.center_y + (self.radius if self.leg == "top" else -self.radius)
        return FlightTarget(math.atan2(desired_y-y, -80 if self.leg == "top" else 80), altitude,airspeed)

    def cross_track_error(self, position: np.ndarray) -> float:
        x, y = position[:2]
        if abs(x) <= self.half_length:
            return min(abs(y-self.center_y-self.radius), abs(y-self.center_y+self.radius))
        return abs(math.hypot(abs(x)-self.half_length, y-self.center_y)-self.radius)
