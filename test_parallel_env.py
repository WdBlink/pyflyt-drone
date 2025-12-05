import gymnasium as gym
import PyFlyt.gym_envs
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import time

def make_env():
    return gym.make("PyFlyt/QuadX-Hover-v3", render_mode=None)

def test_parallel():
    print("Testing parallel environments...")
    num_envs = 4
    
    # Create vectorized environment
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    
    obs = env.reset()
    print(f"Parallel observation shape: {obs.shape}")
    
    start_time = time.time()
    for _ in range(100):
        actions = [env.action_space.sample() for _ in range(num_envs)]
        obs, rewards, dones, infos = env.step(actions)
        
    end_time = time.time()
    print(f"Processed 100 steps in {num_envs} environments in {end_time - start_time:.2f} seconds")
    
    env.close()
    print("Parallel test passed!")

if __name__ == "__main__":
    test_parallel()
