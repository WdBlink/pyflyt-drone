"""Control-coordinate and reset checks; no claim of mission qualification."""

import json
import math

import numpy as np
from PyFlyt.core import Aviary
import pytest

from envs.inspection.control import FixedwingAutopilot, FlightTarget, orbit_target, wrap_angle


def navigation_state():
    return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                     [20.0, 0.0, 0.0], [0.0, 0.0, 80.0]])


def test_shortest_heading_wrap_and_counterclockwise_bank():
    assert wrap_angle(math.radians(2) - math.radians(359)) == pytest.approx(math.radians(3))
    state = navigation_state()
    left = FixedwingAutopilot().step(state, FlightTarget(math.pi / 2), 1 / 120)
    right = FixedwingAutopilot().step(state, FlightTarget(-math.pi / 2), 1 / 120)
    # Positive left-aileron deflection produces positive roll (right bank).
    assert left[0] < 0 < right[0]
    assert left[1] > 0 > right[1]


def test_climb_uses_nose_up_and_throttle_is_bounded():
    state = navigation_state()
    level = FixedwingAutopilot().step(state, FlightTarget(0), 1 / 120)
    climb = FixedwingAutopilot().step(state, FlightTarget(0, altitude=95), 1 / 120)
    assert climb[2] < level[2]
    slow = state.copy()
    slow[2, 0] = 8
    command = FixedwingAutopilot().step(slow, FlightTarget(0), 1 / 120)
    assert 0 <= level[5] < command[5] <= 1
    assert np.all(np.abs(command[:5]) <= 1)


def test_orbit_tangent_reverses_and_outside_guidance_points_inward():
    position = np.array([90.0, 0.0, 80.0])
    ccw = orbit_target(position, np.zeros(2), 90)
    cw = orbit_target(position, np.zeros(2), 90, direction=-1)
    assert ccw.heading == pytest.approx(math.pi / 2)
    assert cw.heading == pytest.approx(-math.pi / 2)
    assert ccw.bank_feedforward < 0 < cw.bank_feedforward
    position[0] = 120
    outside = orbit_target(position, np.zeros(2), 90)
    assert math.cos(outside.heading) < 0


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_target_rejected(bad):
    with pytest.raises(ValueError):
        FlightTarget(bad)


def test_invalid_navigation_and_time_rejected():
    for state, dt in [(np.zeros((3, 4)), 1), (navigation_state(), 0)]:
        with pytest.raises(ValueError):
            FixedwingAutopilot().step(state, FlightTarget(0), dt)


def physical_prefix(seed):
    world = Aviary(start_pos=np.array([[0.0, 0.0, 80.0]]),
                   start_orn=np.zeros((1, 3)), drone_type="fixedwing",
                   np_random=np.random.default_rng(seed))
    try:
        world.set_mode(-1)
        controller = FixedwingAutopilot()
        for _ in range(60):
            world.set_setpoint(0, controller.step(world.state(0), FlightTarget(0), world.step_period))
            world.step()
        assert world.physics_steps == 120
        assert world.elapsed_time == pytest.approx(0.5)
        return world.state(0).copy()
    finally:
        world.disconnect()


def test_seed_zero_replays_physics_and_distinct_seed_changes_motor_noise():
    # Supplying np_random avoids PyFlyt 0.29's truthiness test on seed=0.
    a, b, c = physical_prefix(0), physical_prefix(0), physical_prefix(1)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert 79 < a[3, 2] < 81


@pytest.mark.parametrize("count", [1, 4])
def test_operational_failure_preserves_failed_attempt(monkeypatch, count):
    from eval import qualify_inspection_control as runner

    def fail_step(*args, **kwargs):
        raise RuntimeError("injected controller failure")

    monkeypatch.setattr(runner.FixedwingAutopilot, "step", fail_step)
    result = runner.fly(1, 0, "straight", count)
    assert not result["envelope_pass"]
    assert result["simulated_seconds"] == 0
    assert "injected controller failure" in result["operational_error"]
    json.dumps(result, allow_nan=False)
    assert result["valid_completed_control_steps"] == 0
    assert result["terminal_state"]["t"] == 0


@pytest.mark.parametrize("count", [1, 4])
def test_first_nonfinite_state_retains_fault_location(monkeypatch, count):
    from eval import qualify_inspection_control as runner

    original = Aviary.all_states

    def bad_state(world):
        states = original.fget(world)
        states[0] = states[0].copy()
        states[0][2, 1] = np.nan
        return states

    monkeypatch.setattr(Aviary, "all_states", property(bad_state))
    result = runner.fly(1, 0, "straight", count)
    assert not result["envelope_pass"]
    assert "nonfinite_state" in result["violations"]
    assert result["terminal_state"]["nonfinite_indices"] == [[0, 2, 1]]
    assert result["terminal_state"]["states"][0][2][1] is None
    assert result["valid_completed_control_steps"] == 0
    json.dumps(result, allow_nan=False)
