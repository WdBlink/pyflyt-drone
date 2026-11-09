"""Shared PyFlyt execution with a safety audit after every physics integration."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Callable

import numpy as np
import pybullet
from PyFlyt.core import Aviary

from envs.inspection.control import (
    FixedwingAutopilot, FlightTarget, SurveyRoute, air_heading_for_course,
    orbit_target, wrap_angle,
)


class AuditedAviary(Aviary):
    def __init__(self, **kwargs):
        self.before_physics: Callable[[float], None] | None = None
        self.after_physics: Callable[[float], None] | None = None
        super().__init__(**kwargs)

    def stepSimulation(self):
        t = (self.physics_steps + 1) / self.physics_hz
        if self.before_physics is not None:
            self.before_physics(t)
        pybullet.stepSimulation(physicsClientId=self._client)
        if self.after_physics is not None:
            self.after_physics(t)


@dataclass(frozen=True)
class FlightLimits:
    min_altitude: float = 60.0
    max_altitude: float = 100.0
    min_airspeed: float = 15.0
    max_airspeed: float = 25.0
    max_bank_degrees: float = 45.0
    min_separation: float = 15.0
    max_wind: float = 2.0
    exit_radius_error: float = 10.0
    exit_course_error_degrees: float = 25.0

    @classmethod
    def from_contract(cls, contract):
        return cls(*contract["altitude_m"], *contract["airspeed_m_s"], contract["bank_abs_degrees"],
                   contract["minimum_separation_m"], contract["wind_max_m_s"], contract["exit_radius_error_m"],
                   contract["exit_course_error_degrees"])


class FlightWorld:
    """Own-aircraft navigation/airdata only; no target information in control."""

    def __init__(self, seed: int, count: int = 4, wind_xy=(0.0, 0.0),
                 task="survey", half_length=250.0, radius=80.0, jitter=True, limits=None, bound=None):
        rng = np.random.default_rng(seed)
        self.count, self.radius, self.half_length = count, radius, half_length
        self.limits = limits or FlightLimits()
        self.centers = np.array([-300.0, -100.0, 100.0, 300.0])[:count] if count > 1 else np.array([0.0])
        self.altitudes = np.array([72.0, 80.0, 88.0, 96.0])[:count] if count > 1 else np.array([80.0])
        self.wind = np.array([*wind_xy, 0.0], dtype=float)
        if not np.isfinite(self.wind).all() or np.linalg.norm(self.wind) > self.limits.max_wind+1e-6:
            raise ValueError("wind must be within the prospective 2 m/s qualification envelope")
        yaw = rng.uniform(-.05, .05, count) if jitter else np.zeros(count)
        starts = np.array([[-half_length if task in ("survey", "sortie_exit") else 0.0,
                            center-radius, altitude]
                           for center, altitude in zip(self.centers, self.altitudes)])
        if jitter:
            starts[:, :2] += rng.uniform(-1.0, 1.0, (count, 2))
        options = [{"starting_velocity": np.array([20*math.cos(a), 20*math.sin(a), 0.0])+self.wind}
                   for a in yaw]
        self.world = AuditedAviary(start_pos=starts, start_orn=np.column_stack((np.zeros((count, 2)), yaw)),
                                  drone_type="fixedwing", drone_options=options,
                                  np_random=np.random.default_rng(seed))
        if np.any(self.wind):
            self.world.register_wind_field_function(lambda t, pos: np.broadcast_to(self.wind, pos.shape).copy())
            for drone in self.world.drones:
                drone.update_state()
        self.world.set_mode(-1)
        self.controllers = [FixedwingAutopilot() for _ in range(count)]
        self.routes = [SurveyRoute(center, half_length, radius) for center in self.centers]
        self.modes = ["survey" if task in ("survey", "sortie_exit") else task] * count
        self.headings = np.zeros(count)
        self.exit_dwell = np.zeros(count)
        self.physics_audits = 0
        self.violations: dict[str, dict] = {}
        self.min_separation = None
        self.min_altitude, self.max_altitude = float(starts[:, 2].min()), float(starts[:, 2].max())
        self.min_airspeed, self.max_airspeed = 20.0, 20.0
        self.max_bank = 0.0
        self.bound = 550.0 if task in ("survey", "sortie_exit", "loiter") else 5000.0
        if bound is not None:
            self.bound = bound
        self.last_targets = [FlightTarget(0, float(a)) for a in self.altitudes]
        self.world.after_physics = self._audit

    @property
    def t(self):
        return self.world.elapsed_time

    def _violation(self, reason, t, drone_ids):
        item = self.violations.setdefault(reason, {"first_at": t, "physics_steps": 0, "drone_ids": []})
        item["physics_steps"] += 1
        item["drone_ids"] = sorted(set(item["drone_ids"]) | set(drone_ids))

    def _audit(self, t):
        positions, velocities, rolls = [], [], []
        for drone in self.world.drones:
            position, quaternion = self.world.getBasePositionAndOrientation(drone.Id)
            velocity, _ = self.world.getBaseVelocity(drone.Id)
            positions.append(position)
            velocities.append(velocity)
            rolls.append(self.world.getEulerFromQuaternion(quaternion)[0])
        positions, velocities, rolls = np.array(positions), np.array(velocities), np.abs(rolls)
        if not all(np.isfinite(x).all() for x in (positions, velocities, rolls)):
            raise RuntimeError("nonfinite physics state")
        speed = np.linalg.norm(velocities-self.wind, axis=1)
        altitude = positions[:, 2] - 0.0  # explicit flat terrain elevation
        self.physics_audits += 1
        self.min_altitude = min(self.min_altitude, float(altitude.min()))
        self.max_altitude = max(self.max_altitude, float(altitude.max()))
        self.min_airspeed = min(self.min_airspeed, float(speed.min()))
        self.max_airspeed = max(self.max_airspeed, float(speed.max()))
        self.max_bank = max(self.max_bank, float(rolls.max()))
        for reason, mask in [
            ("altitude", (altitude < self.limits.min_altitude) | (altitude > self.limits.max_altitude)),
            ("airspeed", (speed < self.limits.min_airspeed) | (speed > self.limits.max_airspeed)),
            ("bank", rolls > math.radians(self.limits.max_bank_degrees)),
            ("flight_boundary", (np.abs(positions[:, :2]) > self.bound).any(axis=1)),
        ]:
            if mask.any():
                self._violation(reason, t, np.flatnonzero(mask).tolist())
        for i in range(self.count):
            for j in range(i+1, self.count):
                distance = float(np.linalg.norm(positions[i]-positions[j]))
                self.min_separation = distance if self.min_separation is None else min(self.min_separation, distance)
                if distance < self.limits.min_separation:
                    self._violation("separation", t, [i, j])
        aircraft_ids = {drone.Id for drone in self.world.drones}
        if any(contact[1] in aircraft_ids or contact[2] in aircraft_ids for contact in self.world.getContactPoints()):
            self._violation("collision", t, list(range(self.count)))
        for i in range(self.count):
            offset = positions[i, :2]-np.array([0.0, self.centers[i]])
            tangent = math.atan2(offset[1], offset[0])+math.pi/2
            course = math.atan2(velocities[i, 1], velocities[i, 0])
            inside = (self.modes[i] == "exit" and abs(np.linalg.norm(offset)-self.radius) <= self.limits.exit_radius_error
                      and abs(wrap_angle(course-tangent)) <= math.radians(self.limits.exit_course_error_degrees))
            self.exit_dwell[i] = self.exit_dwell[i]+1/240 if inside else 0.0

    def target(self, i):
        state = self.world.state(i)
        mode = self.modes[i]
        if mode in ("exit", "loiter"):
            return orbit_target(state[3], np.array([0.0, self.centers[i]]), self.radius, float(self.altitudes[i]))
        if mode == "survey":
            return self.routes[i].target(state[3], float(self.altitudes[i]))
        return FlightTarget(float(self.headings[i]), float(self.altitudes[i]))

    def step(self):
        for i, controller in enumerate(self.controllers):
            target = self.target(i)
            self.last_targets[i] = target
            air_heading = air_heading_for_course(target.heading, self.wind[:2], target.airspeed)
            velocity, _ = self.world.getBaseVelocity(self.world.drones[i].Id)
            airspeed = float(np.linalg.norm(np.array(velocity)-self.wind))
            command = controller.step(self.world.state(i), replace(target, heading=air_heading),
                                      self.world.step_period, measured_airspeed=airspeed)
            self.world.set_setpoint(i, command)
        self.world.step()

    def summary(self):
        return {"physics_audits": self.physics_audits, "physics_steps": self.world.physics_steps,
                "violations": self.violations, "min_separation_m": self.min_separation,
                "altitude_m": [self.min_altitude, self.max_altitude],
                "airspeed_m_s": [self.min_airspeed, self.max_airspeed],
                "max_bank_rad": self.max_bank, "exit_dwell_seconds": self.exit_dwell.tolist()}

    def close(self):
        self.world.disconnect()
