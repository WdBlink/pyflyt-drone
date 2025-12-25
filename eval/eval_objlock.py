"""项目文件：Fixedwing-Waypoints-v3 模型评估脚本

说明:
    加载已训练的 PPO 模型和 VecNormalize 统计量，评估模型在 Fixedwing-Waypoints-v3 环境中的表现。
    支持可视化渲染和量化指标计算。
"""

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
import torch
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# 确保能导入 PyFlyt 环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
from envs.models_env import RandomDuckOnResetWrapper
# from train.train_Fixedwing_Waypoints_ObjLock import RandomDuckOnResetWrapper, TRAIN_CONFIG as TRAIN_CONFIG_OBJLOCK

# 评估配置（默认值，可通过命令行参数覆盖）
EVAL_CONFIG = {
    "model_path": "models/waypoints_ppo_v3.5/best_model.zip",
    "vecnorm_path": None,
    "num_episodes": 10,
    "flight_dome_size": 100,
    "num_targets": 8,
    "goal_reach_distance": 8,
    "max_duration_seconds": 120.0,
    "context_length": 2,
    "render": True,
}

def _infer_vecnorm_path(model_path: str, vecnorm_path: str | None) -> str | None:
    if vecnorm_path:
        return vecnorm_path
    model_dir = os.path.dirname(model_path)
    candidates = [
        os.path.join(model_dir, "vecnorm.pkl"),
        "models/waypoints_ppo_v3.3/vecnorm.pkl",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def make_eval_env(render_mode="human"):
    """
    创建评估环境
    """
    # 创建基础环境
    env = gym.make(
        "PyFlyt/Fixedwing-Waypoints-v3",
        render_mode=render_mode,
        num_targets=int(EVAL_CONFIG["num_targets"]),
        goal_reach_distance=float(EVAL_CONFIG["goal_reach_distance"]),
        angle_representation="euler",
        flight_dome_size=float(EVAL_CONFIG["flight_dome_size"]),
        max_duration_seconds=float(EVAL_CONFIG["max_duration_seconds"]),
        agent_hz=30,
    )

    duck_urdf = os.path.join(pybullet_data.getDataPath(), "duck_vhacd.urdf")
    duck_xy_radius = float(EVAL_CONFIG["flight_dome_size"]) * 0.6
    env = RandomDuckOnResetWrapper(
        env,
        urdf_path=duck_urdf,
        xy_radius=duck_xy_radius,
        min_origin_distance=8.0,
        base_z=0.03,
        global_scaling=20.0,
    )
    
    # 扁平化观测空间
    env = FlattenWaypointEnv(env, context_length=EVAL_CONFIG["context_length"])
    return env

def evaluate():
    """
    评估主函数
    """
    parser = argparse.ArgumentParser(description="Evaluate PPO model for Fixedwing Waypoints")
    parser.add_argument("--model", type=str, default=EVAL_CONFIG["model_path"], help="Path to the model file")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to the vecnorm file")
    parser.add_argument("--episodes", type=int, default=EVAL_CONFIG["num_episodes"], help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    # 更新配置
    model_path = args.model
    vecnorm_path = _infer_vecnorm_path(model_path, args.vecnorm)
    render_mode = None if args.no_render else "human"
    
    print(f"Loading model from: {model_path}")
    print(f"Loading vecnorm from: {vecnorm_path}")

    if not os.path.exists(model_path):
        # 尝试回退到 final_model
        fallback_path = model_path.replace("best_model", "final_model")
        if os.path.exists(fallback_path):
            print(f"Model not found at {model_path}, using {fallback_path} instead.")
            model_path = fallback_path
        else:
            print(f"Error: Model file not found at {model_path}")
            return

    # 创建 DummyVecEnv 用于评估（VecNormalize 需要 VecEnv 包装）
    # 注意：这里我们只创建一个环境
    env = DummyVecEnv([lambda: make_eval_env(render_mode)])
    
    # 加载 VecNormalize 统计量
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: vecnorm.pkl not found, running without VecNormalize.")
    env.seed(int(time.time()) % 2**31)
    # 加载模型
    model = PPO.load(model_path, env=env)
    
    print(f"Starting evaluation for {args.episodes} episodes...")
    
    # 记录详细指标
    episode_rewards = []
    episode_lengths = []
    success_count = 0 # 简单的成功计数（如果 info 中有相关字段）
    
    for i in range(args.episodes):
        obs = env.reset()
        # 解包到底层 FixedwingWaypointsEnv，读取真实航点
        raw = env.venv.envs[0]          # DummyVecEnv 里唯一一个 env
        while hasattr(raw, "env") and not hasattr(raw, "waypoints"):
            raw = raw.env               # VecNormalize -> FlattenWaypointEnv -> FixedwingWaypointsEnv
        if hasattr(raw, "waypoints"):
            print(f"Episode {i+1} targets:\n", raw.waypoints.targets)
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # VecEnv 返回的 done 是一个数组，我们这里只有一个环境
            done = done[0]
            total_reward += reward[0]
            steps += 1
            
            if render_mode == "human":
                # 稍微加点延时，避免画面太快（如果环境本身的 render 没有做时钟同步）
                # PyFlyt 通常不需要手动 sleep，这里仅作备用
                # time.sleep(0.01)
                pass

        # 记录本回合数据
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 尝试从 info 中获取成功标志（PyFlyt 某些环境可能有，没有则忽略）
        # info 是一个 list（因为是 VecEnv）
        episode_info = info[0]
        # 假设如果有 success key
        # if "is_success" in episode_info and episode_info["is_success"]:
        #     success_count += 1
            
        print(f"Episode {i+1}: Reward = {total_reward:.2f}, Steps = {steps}")

    env.close()

    # 计算统计指标
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    # print(f"Success Rate: {success_count}/{args.episodes} ({success_count/args.episodes*100:.1f}%)")

if __name__ == "__main__":
    evaluate()
