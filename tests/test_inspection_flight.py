"""Independent rejection checks for physical qualification, not task scores."""

import json
from pathlib import Path

import numpy as np
import pytest

from envs.inspection.control import FlightTarget, air_heading_for_course, wrap_angle
from envs.inspection.flight import FlightLimits, FlightWorld
from eval.qualify_inspection_g1 import run_case, write_json_atomic


CONTRACT = Path(__file__).resolve().parents[1]/"paper/research/20261104-inspection/planning/gate-contract.v3.json"


def test_safety_hook_is_after_both_physics_steps():
    flight = FlightWorld(0, count=1, jitter=False)
    callbacks = []
    original = flight.world.after_physics
    flight.world.after_physics = lambda t: (original(t), callbacks.append((t, flight.world.getBasePositionAndOrientation(flight.world.drones[0].Id)[0])))
    try:
        flight.step([FlightTarget(0,80)])
        assert [x[0] for x in callbacks] == [1/240, 2/240]
        assert callbacks[0][1] != callbacks[1][1]
        assert flight.physics_audits == flight.world.physics_steps == 2
    finally:
        flight.close()


def test_contract_safety_limit_is_actually_enforced():
    flight = FlightWorld(0,count=1,jitter=False)
    flight.limits=FlightLimits(min_altitude=81)  # inject an unsafe state after valid initialization
    try:
        flight.step([FlightTarget(0,80)])
        assert flight.violations["altitude"]["physics_steps"] == 2
    finally:
        flight.close()


def test_crab_cancels_crosswind_in_ground_velocity():
    for course in [0, np.pi/2, np.pi, -np.pi/2]:
        wind = np.array([1., -1.])
        heading = air_heading_for_course(course, wind, 20)
        velocity = np.array([20*np.cos(heading), 20*np.sin(heading)])+wind
        assert wrap_angle(np.arctan2(velocity[1], velocity[0])-course) == pytest.approx(0)


def test_wrong_heading_generator_fails_independent_schedule(monkeypatch):
    contract = json.loads(CONTRACT.read_text())["G1"]
    original = FlightWorld.step
    def wrong_heading(self,targets,**kwargs):
        original(self,[FlightTarget(0.0,float(a)) for a in self.altitudes],**kwargs)
    monkeypatch.setattr(FlightWorld,"step",wrong_heading)
    result = run_case("heading_changes", 0, 120, contract)
    assert result["operational_error"] is None
    assert not result["passed"]
    assert result["settled_max_course_error_rad"] > .7


def test_cleanup_error_preserves_physics_and_trace(monkeypatch):
    contract = json.loads(CONTRACT.read_text())["G1"]
    original = FlightWorld.close

    def bad_close(self):
        original(self)
        raise RuntimeError("injected cleanup")

    monkeypatch.setattr(FlightWorld, "close", bad_close)
    result = run_case("straight", 0, 1, contract)
    assert not result["passed"]
    assert "injected cleanup" in result["operational_error"]
    assert result["safety"]["physics_steps"] == 240
    assert len(result["trace"]) == 10


def test_interrupted_manifest_write_preserves_previous_grid(tmp_path, monkeypatch):
    path = tmp_path/"manifest.json"
    path.write_text('{"grid":"running"}')
    original = Path.write_text

    def interrupted(self, data, *args, **kwargs):
        original(self, data[:5], *args, **kwargs)
        raise OSError("injected interrupted write")

    monkeypatch.setattr(Path, "write_text", interrupted)
    with pytest.raises(OSError):
        write_json_atomic(path, {"grid": "completed"})
    assert json.loads(path.read_text()) == {"grid": "running"}
