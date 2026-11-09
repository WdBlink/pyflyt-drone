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
    max_bank: float = math.radians(30.0)
    max_pitch: float = math.radians(12.0)


class FixedwingAutopilot:
    """Nested PI/PD loops, shared by every prospective mission policy."""

    def __init__(self, gains: ControlGains | None = None):
        self.gains = gains or ControlGains()
        self.altitude_integral = 0.0
        self.speed_integral = 0.0

    def step(self, state: np.ndarray, target: FlightTarget, dt: float) -> np.ndarray:
        if state.shape != (4, 3) or not np.isfinite(state).all():
            raise ValueError("navigation state must be a finite (4, 3) array")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        rates, attitude, velocity, position = state
        roll, pitch, heading = attitude
        g = self.gains
        speed = float(np.linalg.norm(velocity))  # still-air development cases
        height_error = target.altitude - position[2]
        speed_error = target.airspeed - speed
        self.altitude_integral = float(np.clip(self.altitude_integral + height_error * dt, -20.0, 20.0))
        self.speed_integral = float(np.clip(self.speed_integral + speed_error * dt, -15.0, 15.0))
        bank = float(np.clip(
            -g.heading_p * wrap_angle(target.heading - heading) + target.bank_feedforward,
            -g.max_bank, g.max_bank,
        ))
        pitch_target = float(np.clip(
            g.pitch_trim - math.atan2(g.altitude_p * height_error + g.altitude_i * self.altitude_integral, max(speed, 5.0)),
            -g.max_pitch, g.max_pitch,
        ))
        aileron = float(np.clip(g.roll_p * (bank - roll) - g.roll_d * rates[0], -1.0, 1.0))
        elevator = float(np.clip(g.elevator_trim + g.pitch_p * (pitch_target - pitch) - g.pitch_d * rates[1], -1.0, 1.0))
        turn_rate = -9.81 * math.tan(bank) / max(speed, 5.0)
        rudder = float(np.clip(0.4 * (rates[2] - turn_rate * math.cos(roll)), -0.5, 0.5))
        throttle = float(np.clip(g.throttle_trim + g.speed_p * speed_error + g.speed_i * self.speed_integral, 0.0, 1.0))
        return np.array([aileron, -aileron, elevator, rudder, 0.0, throttle])


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
