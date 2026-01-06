import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.fixedwing_objlock_env import FixedwingObjLockEnv


class FlattenObjLockEnv(gym.Wrapper):
    """
    Wrapper for FixedwingObjLockEnv that flattens the observation space.
    
    New Observation:
    [attitude (23), target_vector (3), duck_vision (9)]
    Total size: 35 (typically)
    """

    def __init__(self, env: FixedwingObjLockEnv):
        super().__init__(env)
        self.env = env
        
        # Compute flattened size
        obs_space = env.observation_space
        self.attitude_shape = obs_space["attitude"].shape[0]
        self.target_shape = obs_space["target_vector"].shape[0]
        self.vision_shape = obs_space["duck_vision"].shape[0]
        
        total_size = self.attitude_shape + self.target_shape + self.vision_shape
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._flatten_obs(obs), reward, term, trunc, info

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        return np.concatenate([
            obs["attitude"],
            obs["target_vector"],
            obs["duck_vision"]
        ], axis=0).astype(np.float32)
