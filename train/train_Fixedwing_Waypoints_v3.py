"""项目文件：Fixedwing-Waypoints-v3 PPO 训练脚本

说明:
    使用 Stable-Baselines3 的 PPO 算法训练 PyFlyt/Fixedwing-Waypoints-v3 环境。
    该环境的目标是通过控制 roll, pitch, yaw, thrust 来追踪一系列航点。
"""

import os
import sys
import gymnasium as gym
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

# 确保能导入 PyFlyt
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv

# 训练配置
TRAIN_CONFIG = {
    "total_timesteps": 2_000_000,
    "num_envs": 16,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "log_dir": "logs/waypoints_ppo",
    "model_dir": "models/waypoints_ppo",
    "flight_dome_size": 200.0,
    "max_duration_seconds": 120.0,
    "context_length": 2,  # 观测中包含当前目标点和下一个目标点
}

def make_env(rank: int, seed: int = 0):
    """
    创建环境的工厂函数
    """
    def _init():
        # 创建基础环境
        # Actions: [roll, pitch, yaw, thrust]
        env = gym.make(
            "PyFlyt/Fixedwing-Waypoints-v3",
            render_mode=None,
            angle_representation="euler", # 使用欧拉角更直观
            flight_dome_size=TRAIN_CONFIG["flight_dome_size"],
            max_duration_seconds=TRAIN_CONFIG["max_duration_seconds"],
            agent_hz=30,
        )
        
        # 扁平化观测空间，以便 MLP 网络处理
        # FlattenWaypointEnv 会将环境的 Dict 观测转换为 Box 观测
        env = FlattenWaypointEnv(env, context_length=TRAIN_CONFIG["context_length"])
        
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    """
    训练主函数
    """
    # 设置随机种子
    set_random_seed(TRAIN_CONFIG["seed"])
    
    # 创建目录
    os.makedirs(TRAIN_CONFIG["log_dir"], exist_ok=True)
    os.makedirs(TRAIN_CONFIG["model_dir"], exist_ok=True)

    # 检测设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 创建并行训练环境
    print("Creating training environments...")
    env = SubprocVecEnv([make_env(i, TRAIN_CONFIG["seed"]) for i in range(TRAIN_CONFIG["num_envs"])])
    
    # 观测和奖励归一化
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 创建评估环境（独立于训练环境）
    print("Creating evaluation environments...")
    eval_env = SubprocVecEnv([make_env(i + TRAIN_CONFIG["num_envs"], TRAIN_CONFIG["seed"]) for i in range(4)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 设置回调函数
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=TRAIN_CONFIG["model_dir"],
        log_path=TRAIN_CONFIG["log_dir"],
        eval_freq=10000 // TRAIN_CONFIG["num_envs"], # 每 10000 步评估一次
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000 // TRAIN_CONFIG["num_envs"],
        save_path=TRAIN_CONFIG["model_dir"],
        name_prefix="waypoints_ppo"
    )

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=TRAIN_CONFIG["learning_rate"],
        n_steps=TRAIN_CONFIG["n_steps"],
        batch_size=TRAIN_CONFIG["batch_size"],
        n_epochs=TRAIN_CONFIG["n_epochs"],
        gamma=TRAIN_CONFIG["gamma"],
        gae_lambda=TRAIN_CONFIG["gae_lambda"],
        clip_range=TRAIN_CONFIG["clip_range"],
        ent_coef=TRAIN_CONFIG["ent_coef"],
        verbose=1,
        tensorboard_log=TRAIN_CONFIG["log_dir"],
        seed=TRAIN_CONFIG["seed"],
        device=device
    )

    print(f"Starting training for {TRAIN_CONFIG['total_timesteps']} timesteps...")
    
    try:
        model.learn(
            total_timesteps=TRAIN_CONFIG["total_timesteps"],
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted by user.")
    finally:
        # 保存最终模型
        print("Saving final model...")
        model.save(os.path.join(TRAIN_CONFIG["model_dir"], "final_model"))
        env.save(os.path.join(TRAIN_CONFIG["model_dir"], "vecnorm.pkl"))
        
        env.close()
        eval_env.close()
        print("Done.")

if __name__ == "__main__":
    train()
