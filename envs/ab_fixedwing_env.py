"""Factory for the fixed-wing A-to-B waypoint training environment."""

from __future__ import annotations

from typing import Any, Literal

import gymnasium as gym

import PyFlyt.gym_envs  # noqa: F401 - registers PyFlyt gym environments
from PyFlyt.gym_envs import FlattenWaypointEnv

from envs.utils import WindOnResetWrapper


def make_fixedwing_ab_env(
    *,
    render_mode: None | Literal["human", "rgb_array"] = None,
    num_targets: int = 1,
    goal_reach_distance: float = 2.0,
    flight_dome_size: float = 100.0,
    max_duration_seconds: float = 120.0,
    angle_representation: Literal["euler", "quaternion"] = "quaternion",
    agent_hz: int = 30,
    context_length: int = 1,
    wind_config: dict[str, Any] | None = None,
) -> gym.Env:
    """Create the flattened PyFlyt fixed-wing waypoint environment used by A-to-B training."""

    env = gym.make(
        "PyFlyt/Fixedwing-Waypoints-v3",
        render_mode=render_mode,
        num_targets=int(num_targets),
        goal_reach_distance=float(goal_reach_distance),
        flight_dome_size=float(flight_dome_size),
        max_duration_seconds=float(max_duration_seconds),
        angle_representation=angle_representation,
        agent_hz=int(agent_hz),
    )

    if bool((wind_config or {}).get("enabled", False)):
        env = WindOnResetWrapper(env, wind_config)

    return FlattenWaypointEnv(env, context_length=int(context_length))
