import gymnasium as gym
import PyFlyt.gym_envs

# 创建环境
env = gym.make("PyFlyt/QuadX-Hover-v3", render_mode="human")

# 重置环境
obs, _ = env.reset()

# 运行几个步骤
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    
    if terminated or truncated:
        obs, _ = env.reset()

env.close()
