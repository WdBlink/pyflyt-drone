import gymnasium as gym
import PyFlyt.gym_envs
import numpy as np

def test_reset():
    print("Testing environment reset...")
    env = gym.make("PyFlyt/QuadX-Hover-v3", render_mode=None)
    
    obs, _ = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Test multiple resets
    for i in range(5):
        obs, info = env.reset()
        print(f"Reset {i+1}: Obs mean={np.mean(obs):.4f}")
        
    env.close()
    print("Reset test passed!")

if __name__ == "__main__":
    test_reset()
