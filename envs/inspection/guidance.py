"""Causal local tracking and translating-circle guidance, without mission truth."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from envs.inspection.control import FlightTarget


@dataclass
class LocalTrack:
    """Alpha-beta filter for one observed handle in one aircraft's knowledge scope.

    Callers must supply only received/local measurements, never ground truth.
    A prediction is a navigation estimate and cannot become report evidence.
    Duplicate or older measurement stamps do not rewind the state.
    """

    position: np.ndarray
    velocity: np.ndarray
    measured_at: float
    alpha: float = .5
    beta: float = .08

    @classmethod
    def first(cls, xy, measured_at):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(measured_at):
            raise ValueError("finite planar measurement and stamp required")
        return cls(xy.copy(), np.zeros(2), float(measured_at))

    def predict(self, now):
        if not math.isfinite(now) or now < self.measured_at:
            raise ValueError("prediction must be causal")
        return self.position + self.velocity*(now-self.measured_at)

    def update(self, xy, measured_at):
        xy = np.asarray(xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(measured_at):
            raise ValueError("finite planar measurement and stamp required")
        if measured_at <= self.measured_at:
            return False
        dt = measured_at-self.measured_at
        residual = xy-self.predict(measured_at)
        self.position += self.velocity*dt + self.alpha*residual
        self.velocity += self.beta/dt*residual
        speed = float(np.linalg.norm(self.velocity))
        if speed > 4.0:
            self.velocity *= 4.0/speed
        self.measured_at = float(measured_at)
        return True


def moving_orbit_target(position, center, center_velocity, wind_xy, *, radius=90.0,
                        altitude=96.0, airspeed=18.0):
    """Translate a CCW circle using an observed center estimate.

    Solve the airspeed constraint exactly for tangential speed, including
    estimated center drift and wind. Radial feedback captures the circle.
    Feedforward differentiates the constant-drift air-velocity direction;
    center acceleration and filter corrections are handled by feedback.
    The returned heading is a ground course, as required by FlightWorld.
    """
    vectors = [np.asarray(x, dtype=float) for x in (position, center, center_velocity, wind_xy)]
    position, center, center_velocity, wind_xy = vectors
    if (any(x.shape != (2,) or not np.isfinite(x).all() for x in vectors)
            or not all(math.isfinite(x) and x > 0 for x in (radius, altitude, airspeed))):
        raise ValueError("finite planar inputs and positive geometry required")
    offset = position-center
    distance = float(np.linalg.norm(offset))
    radial = offset/max(distance, 1e-6) if distance > 1e-6 else np.array([0., -1.])
    tangent = np.array([-radial[1], radial[0]])
    drift = center_velocity-wind_xy
    radial_speed = float(np.clip(-.7*(distance-radius), -5., 5.))
    corrected_drift = drift+radial_speed*radial
    a = float(corrected_drift@tangent)
    discriminant = a*a+airspeed*airspeed-float(corrected_drift@corrected_drift)
    if discriminant <= 0:
        raise ValueError("moving circle infeasible at requested airspeed")
    s = -a+math.sqrt(discriminant)
    ground_velocity = center_velocity+radial_speed*radial+s*tangent
    # Exact uniform-center bank. The sign is reversed for PyFlyt ENU roll.
    a0, b0 = float(drift@tangent), float(drift@radial)
    q = math.sqrt(a0*a0+airspeed*airspeed-float(drift@drift))
    s0 = -a0+q
    ds = b0*(1-a0/q)
    air_velocity = drift+s0*tangent
    acceleration = (ds*tangent-s0*radial)*(s0/radius)
    heading_rate = float(air_velocity[0]*acceleration[1]-air_velocity[1]*acceleration[0])/(airspeed*airspeed)
    bank = -math.atan(airspeed*heading_rate/9.81)
    return FlightTarget(math.atan2(ground_velocity[1], ground_velocity[0]), altitude, airspeed, bank)
