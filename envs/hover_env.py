import gymnasium as gym
import numpy as np
import PyFlyt.gym_envs
import time

class HoverEnv(gym.Env):
    """
    悬停环境，用于训练无人机在指定高度悬停
    """
    def __init__(self, render_mode="human"):
        super().__init__()
        self.env = gym.make("PyFlyt/QuadX-Hover-v3", render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
