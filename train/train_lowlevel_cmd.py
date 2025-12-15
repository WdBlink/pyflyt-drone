"""项目文件：固定翼底层指令控制训练脚本

作者: wdblink

说明:
    训练一个底层策略网络（Low-Level Policy），用于根据给定的高级指令（航向、高度、空速）
    输出飞机的控制量（副翼、升降舵、方向舵、油门）。
    该模型将作为分层控制架构中的底层控制器。
"""

import os
import sys

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from envs.fixedwing_lowlevel_env import FixedwingLowLevelEnv
import PyFlyt.gym_envs

# 训练参数配置
TRAIN_CONFIG = {
    "total_timesteps": 2_000_000,
    "num_envs": 8,
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
    "log_dir": "logs/lowlevel_ppo",
    "model_dir": "models/lowlevel_ppo",
}

def make_env(rank: int, seed: int = 0):
    """
    创建环境的工厂函数
    """
    def _init():
        # 创建基础环境，使用欧拉角表示姿态，便于底层控制理解
        env = gym.make(
            "PyFlyt/Fixedwing-Waypoints-v3",
            render_mode=None,
            angle_representation="euler",
            flight_dome_size=200.0,
            max_duration_seconds=120.0,
            agent_hz=30, # 底层控制通常需要较高的频率，30Hz对于高层指令跟踪可能足够，但对于姿态控制可能偏低，这里保持默认
        )
        # 包装为底层控制训练环境
        env = FixedwingLowLevelEnv(env)
        env.reset(seed=seed + rank)
        return env
    return _init

def train():
    """
    训练主函数
    """
    # 创建目录
    os.makedirs(TRAIN_CONFIG["log_dir"], exist_ok=True)
    os.makedirs(TRAIN_CONFIG["model_dir"], exist_ok=True)

    # 创建并行环境
    env = SubprocVecEnv([make_env(i) for i in range(TRAIN_CONFIG["num_envs"])])
    
    # 观测归一化
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 定义回调函数
    eval_env = SubprocVecEnv([make_env(i + TRAIN_CONFIG["num_envs"]) for i in range(2)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=TRAIN_CONFIG["model_dir"],
        log_path=TRAIN_CONFIG["log_dir"],
        eval_freq=10000,
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=TRAIN_CONFIG["model_dir"],
        name_prefix="lowlevel_ppo"
    )

    # 初始化模型
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
        vf_coef=TRAIN_CONFIG["vf_coef"],
        max_grad_norm=TRAIN_CONFIG["max_grad_norm"],
        verbose=1,
        tensorboard_log=TRAIN_CONFIG["log_dir"],
        seed=TRAIN_CONFIG["seed"],
        device="cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
    )

    print(f"开始训练底层控制器，目标步数: {TRAIN_CONFIG['total_timesteps']}")
    print(f"设备: {model.device}")

    # 开始训练
    model.learn(
        total_timesteps=TRAIN_CONFIG["total_timesteps"],
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # 保存最终模型和归一化统计量
    model.save(os.path.join(TRAIN_CONFIG["model_dir"], "final_model"))
    env.save(os.path.join(TRAIN_CONFIG["model_dir"], "vecnorm.pkl"))
    
    print("训练完成。")
    env.close()
    eval_env.close()

if __name__ == "__main__":
    train()
