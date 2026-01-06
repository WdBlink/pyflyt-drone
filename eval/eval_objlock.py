"""项目文件：Fixedwing-ObjLock 模型评估脚本
说明:
    加载已训练的 PPO 模型和 VecNormalize 统计量，评估模型在 FixedwingObjLockEnv 环境中的表现。
"""

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure imports work
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.fixedwing_objlock_env import FixedwingObjLockEnv
from envs.flatten_objlock_env import FlattenObjLockEnv
import PyFlyt.gym_envs

# 默认评估配置
EVAL_CONFIG = {
    "model_path": "models/obj_lock_only_ppo_v1.0/best_model.zip",
    "vecnorm_path": None, # Auto-detect
    "num_episodes": 10,
    "flight_dome_size": 100.0,
    "max_duration_seconds": 60.0,
    
    # Duck Configs
    "duck_camera_capture_interval_steps": 5,
    "duck_lock_hold_steps": 10,
    "duck_strike_distance_m": 8.0,
    "duck_global_scaling": 30.0,
    
    # Obstacle Configs
    "num_obstacles": 10,
    "obstacle_radius": 2.0,
    "obstacle_height_range": (10.0, 30.0),
    "obstacle_safe_distance_m": 10.0,
}

def _infer_vecnorm_path(model_path: str, vecnorm_path: str | None) -> str | None:
    if vecnorm_path:
        return vecnorm_path
    model_dir = os.path.dirname(model_path)
    candidates = [
        os.path.join(model_dir, "vecnorm.pkl"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def make_eval_env(render_mode="human"):
    """
    创建评估环境
    """
    env = FixedwingObjLockEnv(
        sparse_reward=False, # Doesn't matter for eval, but keep consistent
        render_mode=render_mode,
        angle_representation="euler",
        flight_dome_size=float(EVAL_CONFIG["flight_dome_size"]),
        max_duration_seconds=float(EVAL_CONFIG["max_duration_seconds"]),
        agent_hz=30,
        
        # Obstacles
        num_obstacles=EVAL_CONFIG["num_obstacles"],
        obstacle_radius=EVAL_CONFIG["obstacle_radius"],
        obstacle_height_range=EVAL_CONFIG["obstacle_height_range"],
        obstacle_safe_distance_m=EVAL_CONFIG["obstacle_safe_distance_m"],
        
        # Duck Configs
        duck_camera_capture_interval_steps=EVAL_CONFIG["duck_camera_capture_interval_steps"],
        duck_lock_hold_steps=EVAL_CONFIG["duck_lock_hold_steps"],
        duck_strike_distance_m=EVAL_CONFIG["duck_strike_distance_m"],
        duck_global_scaling=EVAL_CONFIG["duck_global_scaling"],
    )
    
    # 扁平化观测空间
    env = FlattenObjLockEnv(env)
    return env

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate PPO model for Fixedwing ObjLock")
    parser.add_argument("--model", type=str, default=EVAL_CONFIG["model_path"], help="Path to the model file")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to the vecnorm file")
    parser.add_argument("--episodes", type=int, default=EVAL_CONFIG["num_episodes"], help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    model_path = args.model
    vecnorm_path = _infer_vecnorm_path(model_path, args.vecnorm)
    render_mode = None if args.no_render else "human"
    
    print(f"Loading model from: {model_path}")
    print(f"Loading vecnorm from: {vecnorm_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 创建 DummyVecEnv
    env = DummyVecEnv([lambda: make_eval_env(render_mode)])
    
    # 加载 VecNormalize
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: vecnorm.pkl not found, running without VecNormalize.")

    # 加载模型
    model = PPO.load(model_path, env=env)
    
    print(f"Starting evaluation for {args.episodes} episodes...")
    
    total_rewards = []
    success_count = 0
    
    for i in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1
            
            if render_mode:
                time.sleep(1/30.0) # Limit FPS
                
        # Info is a list for VecEnv
        info_dict = info[0]
        is_success = info_dict.get("duck_strike", False)
        if is_success:
            success_count += 1
            
        total_rewards.append(episode_reward)
        print(f"Episode {i+1}: Reward={episode_reward:.2f}, Steps={step_count}, Success={is_success}")
        
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = success_count / args.episodes
    
    print("-" * 30)
    print(f"Evaluation Complete")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    
    env.close()

if __name__ == "__main__":
    evaluate()
