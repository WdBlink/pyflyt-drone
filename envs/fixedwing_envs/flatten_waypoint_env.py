"""Wrapper class for flattening the waypoint envs to use homogeneous observation spaces.
Locally patched version to handle variable num_targets robustly.
"""

from __future__ import annotations

import numpy as np
from gymnasium.core import Env, ObservationWrapper
from gymnasium.spaces import Box

from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class FlattenWaypointEnv(ObservationWrapper):
    """FlattenWaypontEnv."""

    def __init__(self, env: Env, context_length=2):
        """__init__.

        Args:
            env (Env): a PyFlyt Waypoints environment.
            context_length: how many waypoints should be included in the flattened observation space.

        """
        super().__init__(env=env)
        # Relaxed check: as long as env has waypoints attribute (even if mocked or different type), we proceed
        # But strictly speaking we should import WaypointHandler. 
        # Since we are in local project, we assume PyFlyt is installed.
        
        # Check if wrapped env has waypoints
        # Note: Depending on wrapper stack, 'env' might be a wrapper. 
        # We check env.unwrapped for 'waypoints' usually.
        if not hasattr(env.unwrapped, "waypoints"):
             # Try direct attribute (if not wrapped yet or exposed)
             if not hasattr(env, "waypoints"):
                raise AttributeError(
                    "Only a waypoints environment can be used with the `FlattenWaypointEnv` wrapper."
                )

        self.context_length = context_length
        self.attitude_shape = env.observation_space["attitude"].shape[0]  # type: ignore [reportGeneralTypeIssues]
        self.target_shape = env.observation_space["target_deltas"].feature_space.shape[  # type: ignore [reportGeneralTypeIssues]
            0
        ]  # type: ignore [reportGeneralTypeIssues]
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.attitude_shape + self.target_shape * self.context_length,),
            dtype=np.float64
        )

    def observation(self, observation) -> np.ndarray:
        """Flattens an observation from the super env.

        Args:
            observation: a dictionary observation with an "attitude" and "target_deltas" keys.

        """
        # Robust implementation
        targets = np.zeros((self.context_length, self.target_shape), dtype=observation["target_deltas"].dtype)
        
        source_targets = observation["target_deltas"]
        if source_targets.shape[0] > 0:
            num_targets = min(self.context_length, source_targets.shape[0])
            targets[:num_targets] = source_targets[:num_targets]

        # Use flatten() instead of * unpacking to ensure 1D array output
        new_obs = np.concatenate(
            [observation["attitude"], targets.flatten()]
        ) 

        return new_obs
