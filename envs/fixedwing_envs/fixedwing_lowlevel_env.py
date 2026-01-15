from __future__ import annotations
import math
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PyFlyt.core.aviary import Aviary

class FixedwingLowLevelEnv(gym.Env):
    """Fixed-wing low-level control training environment.
    
    This environment directly exposes the fixed-wing's 6 control surfaces and thrust (mode=-1):
    Action: np.ndarray([left_ail, right_ail, hstab, vstab, flap, thrust]), range [-1, 1].
    Observation: Attitude, velocity, position, previous action, and high-level targets [psi_ref, h_ref, V_ref].
    Reward: Based on tracking error of high-level targets, with safety and smoothness constraints.
    """
    
    metadata = {"render_modes": ["human"], "name": "fixedwing_lowlevel_env"}

    def __init__(
        self, render_mode: Optional[str] = None, wind_config: Optional[dict[str, Any]] = None
    ):
        super().__init__()
        self.render_mode = render_mode
        self._wind_config = wind_config or {}
        
        # Define simulation parameters
        self.start_height_m = 10.0
        self.start_speed_mps = 15.0
        self.min_speed_mps = 5.0
        self.target_speed_range = (10.0, 20.0)
        self.target_height_range = (5.0, 20.0)
        
        # Initialize Aviary
        start_pos = np.array([[0.0, 0.0, self.start_height_m]])
        start_orn = np.array([[0.0, 0.0, 0.0]])
        drone_options = dict(starting_velocity=np.array([self.start_speed_mps, 0.0, 0.0]))
        
        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="fixedwing",
            drone_options=drone_options,
            render=self.render_mode == "human",
            physics_hz=240,
            world_scale=1.0,
        )

        if bool(self._wind_config.get("enabled", False)) and not bool(
            self._wind_config.get("randomize_on_reset", False)
        ):
            from envs.utils import _register_wind_field

            _register_wind_field(self.env, self._wind_config, self.env.np_random)
        
        # Set to surface control mode
        self.env.set_mode(-1)

        self._episode_steps = 0
        self.prev_action = np.zeros(6, dtype=np.float64)
        self.target = np.array([0.0, self.start_height_m, self.start_speed_mps], dtype=np.float64)

        # Define Observation Space
        # 3 (ang_vel) + 3 (ang_pos) + 3 (lin_vel) + 3 (lin_pos) + 6 (prev_action) + 3 (target) = 21
        low = -np.inf * np.ones(21, dtype=np.float64)
        high = np.inf * np.ones_like(low)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        
        # Define Action Space: 6 surfaces + thrust
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset()

        if bool(self._wind_config.get("enabled", False)) and bool(
            self._wind_config.get("randomize_on_reset", False)
        ):
            from envs.utils import _register_wind_field

            _register_wind_field(self.env, self._wind_config, self.env.np_random)
        self.env.set_mode(-1)
        self._episode_steps = 0
        self.prev_action = np.zeros(6, dtype=np.float64)
        
        # Randomize target
        psi_ref = float(self.env.np_random.uniform(-math.pi, math.pi))
        h_ref = float(self.env.np_random.uniform(*self.target_height_range))
        V_ref = float(self.env.np_random.uniform(*self.target_speed_range))
        self.target = np.array([psi_ref, h_ref, V_ref], dtype=np.float64)
        
        obs = self._compute_obs()
        info = {"target": self.target.copy()}
        return obs, info

    def step(self, action):
        self._episode_steps += 1
        self.prev_action = action.copy()
        
        # Step physics
        self.env.set_all_setpoints(action.reshape(1, -1))
        self.env.step()
        
        obs = self._compute_obs()
        
        # Extract state for reward calculation
        # state: [ang_vel(3), ang_pos(3), lin_vel(3), lin_pos(3)]
        state = self.env.state(0)
        if isinstance(state, np.ndarray):
            state = state.flatten()
        elif isinstance(state, (list, tuple)):
            state = np.concatenate(state).flatten()

        current_psi = state[5] # Yaw
        current_vel = np.linalg.norm(state[6:9])
        current_alt = state[11]
        
        target_psi, target_h, target_v = self.target
        
        # Calculate errors
        psi_err = abs(self._wrap_pi(target_psi - current_psi))
        h_err = abs(target_h - current_alt)
        v_err = abs(target_v - current_vel)
        
        # Reward function
        reward = - (1.0 * psi_err + 1.0 * h_err + 0.5 * v_err)
        reward += 0.1 # Survival reward
        
        # Termination conditions
        terminated = False
        if current_alt < 1.0 or current_alt > 100.0:
            terminated = True
            reward -= 100.0
        
        truncated = False
        if self._episode_steps >= 2000: # Max steps
            truncated = True
            
        info = {"target": self.target.copy()}
        
        return obs, reward, terminated, truncated, info

    def _compute_obs(self):
        state = self.env.state(0)
        if isinstance(state, np.ndarray):
            state = state.flatten()
        elif isinstance(state, (list, tuple)):
            state = np.concatenate(state).flatten()
            
        # state: [ang_vel(3), ang_pos(3), lin_vel(3), lin_pos(3)]
        return np.concatenate([
            state,
            self.prev_action,
            self.target
        ])
        
    def _wrap_pi(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def close(self):
        if hasattr(self.env, "disconnect"):
            self.env.disconnect()
