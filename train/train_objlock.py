"""项目文件：Fixedwing-ObjLock PPO 训练脚本
版本：v1.0
说明:
    使用 Stable-Baselines3 的 PPO 算法训练 PyFlyt/FixedwingObjLockEnv 环境。
    该环境的目标是控制固定翼无人机通过摄像头锁定并撞击小黄鸭，同时避免与障碍物碰撞。
"""

import os
import sys
import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from envs.fixedwing_objlock_env import FixedwingObjLockEnv
from envs.flatten_objlock_env import FlattenObjLockEnv
import PyFlyt.gym_envs  # Ensure PyFlyt envs are registered if needed

# 训练配置
TRAIN_CONFIG = {
    "total_timesteps": 1_000_000,
    "num_envs": 16,
    "sparse_reward": False,
    "n_eval_episodes": 10,
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 42,
    "log_dir": "logs/obj_lock_only_ppo_v1.0",
    "model_dir": "models/obj_lock_only_ppo_v1.0",
    
    # 环境参数
    "flight_dome_size": 100.0,
    "max_duration_seconds": 60.0,
    
    # 鸭子参数
    "duck_camera_capture_interval_steps": 5,
    "duck_lock_hold_steps": 5,
    "duck_strike_distance_m": 10.0,
    "duck_strike_reward": 200.0,
    "duck_lock_step_reward": 0.2,
    "duck_approach_reward_scale": 0.1,
    "duck_global_scaling": 60.0,
    
    # 障碍物参数
    "num_obstacles": 0,
    "obstacle_radius": 2.0,
    "obstacle_height_range": (10.0, 30.0),
    "obstacle_safe_distance_m": 10.0,
    "obstacle_avoid_reward_scale": 1.0,
    "obstacle_avoid_max_penalty": 5.0,

    # 相机参数
    "camera_profile": "cockpit_fpv",
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model", type=str, default="models/obj_lock_only_ppo_v1.0/best_model.zip")
    parser.add_argument("--vecnorm_path", type=str, default=None)
    parser.add_argument("--total_timesteps", type=int, default=None)
    return parser.parse_args()

def _infer_vecnorm_path(pretrained_model: str | None, vecnorm_path: str | None) -> str | None:
    if vecnorm_path:
        return vecnorm_path
    if not pretrained_model:
        return None

    candidate_dirs = [
        os.path.dirname(pretrained_model),
        TRAIN_CONFIG["model_dir"],
    ]
    for d in candidate_dirs:
        if not d:
            continue
        p = os.path.join(d, "vecnorm.pkl")
        if os.path.exists(p):
            return p
    return None

def make_env(rank: int, seed: int = 0, use_egl: bool = False):
    """
    创建环境的工厂函数
    """
    def _init():
        env = FixedwingObjLockEnv(
            sparse_reward=TRAIN_CONFIG["sparse_reward"],
            render_mode="rgb_array",
            angle_representation="euler",
            flight_dome_size=TRAIN_CONFIG["flight_dome_size"],
            max_duration_seconds=TRAIN_CONFIG["max_duration_seconds"],
            agent_hz=30,
            use_egl=use_egl,
            
            # Obstacles
            num_obstacles=TRAIN_CONFIG["num_obstacles"],
            obstacle_radius=TRAIN_CONFIG["obstacle_radius"],
            obstacle_height_range=TRAIN_CONFIG["obstacle_height_range"],
            obstacle_safe_distance_m=TRAIN_CONFIG["obstacle_safe_distance_m"],
            obstacle_avoid_reward_scale=TRAIN_CONFIG["obstacle_avoid_reward_scale"],
            obstacle_avoid_max_penalty=TRAIN_CONFIG["obstacle_avoid_max_penalty"],
            
            # Duck Configs
            duck_camera_capture_interval_steps=TRAIN_CONFIG["duck_camera_capture_interval_steps"],
            duck_lock_hold_steps=TRAIN_CONFIG["duck_lock_hold_steps"],
            duck_strike_distance_m=TRAIN_CONFIG["duck_strike_distance_m"],
            duck_strike_reward=TRAIN_CONFIG["duck_strike_reward"],
            duck_lock_step_reward=TRAIN_CONFIG["duck_lock_step_reward"],
            duck_approach_reward_scale=TRAIN_CONFIG["duck_approach_reward_scale"],
            duck_global_scaling=TRAIN_CONFIG["duck_global_scaling"],
        )

        # 扁平化观测空间
        env = FlattenObjLockEnv(env)
        
        env.reset(seed=seed + rank)
        return env
    return _init

class ObjLockEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._duck_strike_buffer: list[int] = []

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        info = locals_.get("info")
        if isinstance(info, (list, tuple)) and len(info) > 0:
            info = info[0]

        if locals_.get("done") and isinstance(info, dict):
            self._duck_strike_buffer.append(int(bool(info.get("duck_strike", False))))

        super()._log_success_callback(locals_, globals_)

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._duck_strike_buffer = []
        return super()._on_step()

    def _on_event(self) -> None:
        super()._on_event()
        if len(self._duck_strike_buffer) > 0:
            success_rate = np.mean(self._duck_strike_buffer)
            self.logger.record("eval/duck_strike_rate", success_rate)

def main():
    args = parse_args()
    total_timesteps = args.total_timesteps or TRAIN_CONFIG["total_timesteps"]
    
    set_random_seed(TRAIN_CONFIG["seed"])
    
    # 检查是否有可用 GPU
    use_egl = False
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
            # EGL 渲染通常在有 GPU 的 Linux 环境下使用
            if sys.platform == "linux":
                use_egl = True
    except ImportError:
        pass

    # 创建训练环境
    env = SubprocVecEnv(
        [make_env(i, seed=TRAIN_CONFIG["seed"], use_egl=use_egl) for i in range(TRAIN_CONFIG["num_envs"])]
    )
    
    # 载入 VecNormalize
    vecnorm_path = _infer_vecnorm_path(args.pretrained_model, args.vecnorm_path)
    if vecnorm_path:
        print(f"Loading VecNormalize from {vecnorm_path}")
        env = VecNormalize.load(vecnorm_path, env)
        env.training = True
        env.norm_reward = True
    else:
        print("Creating new VecNormalize")
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # 创建评估环境
    eval_env = SubprocVecEnv(
        [make_env(i + TRAIN_CONFIG["num_envs"], seed=TRAIN_CONFIG["seed"] + 1000, use_egl=use_egl) 
         for i in range(1)]
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Callbacks
    eval_callback = ObjLockEvalCallback(
        eval_env,
        best_model_save_path=TRAIN_CONFIG["model_dir"],
        log_path=TRAIN_CONFIG["log_dir"],
        eval_freq=max(10000 // TRAIN_CONFIG["num_envs"], 1),
        n_eval_episodes=TRAIN_CONFIG["n_eval_episodes"],
        deterministic=True,
        render=False,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // TRAIN_CONFIG["num_envs"], 1),
        save_path=TRAIN_CONFIG["model_dir"],
        name_prefix="ppo_objlock",
    )

    # 模型初始化或加载
    if args.pretrained_model:
        print(f"Loading pretrained model from {args.pretrained_model}")
        model = PPO.load(
            args.pretrained_model,
            env=env,
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
            tensorboard_log=TRAIN_CONFIG["log_dir"],
            device="auto"
        )
    else:
        print("Creating new PPO model")
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
            tensorboard_log=TRAIN_CONFIG["log_dir"],
            device="auto",
            verbose=1
        )

    print(f"Starting training for {total_timesteps} steps...")
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[eval_callback, checkpoint_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("Training interrupted")
    finally:
        print("Saving final model and normalization stats...")
        model.save(os.path.join(TRAIN_CONFIG["model_dir"], "final_model"))
        env.save(os.path.join(TRAIN_CONFIG["model_dir"], "vecnorm.pkl"))
        env.close()
        eval_env.close()
        print("Done.")

if __name__ == "__main__":
    main()
