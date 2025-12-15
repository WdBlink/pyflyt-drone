"""项目文件：PPO 并行训练脚本（固定翼 A→B）

作者: wdblink

说明:
    使用 Stable-Baselines3 的 PPO 对基于 PyFlyt 的固定翼环境进行并行训练，
    环境观测通过 FlattenWaypointEnv 扁平化后以 Box 空间输入 MLP 策略网络。
    支持 VecNormalize、评估回调与模型保存。
"""

from __future__ import annotations

import os
from typing import Callable

import yaml
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback

from envs.ab_fixedwing_env import make_fixedwing_ab_env


def _make_env_from_cfg(cfg: dict) -> Callable[[], gym.Env]:
    """返回环境构造的闭包，便于向量化创建。

    Args:
        cfg: 环境配置字典。

    Returns:
        可调用对象，用于创建单个 Gymnasium 环境实例。
    """

    def _thunk() -> gym.Env:
        return make_fixedwing_ab_env(
            render_mode=None,
            num_targets=cfg["num_targets"],
            goal_reach_distance=cfg["goal_reach_distance"],
            flight_dome_size=cfg["flight_dome_size"],
            max_duration_seconds=cfg["max_duration_seconds"],
            angle_representation=cfg["angle_representation"],
            agent_hz=cfg["agent_hz"],
            context_length=cfg["context_length"],
        )

    return _thunk


def main() -> None:
    """PPO 并行训练入口。"""

    with open("configs/env.yaml", "r", encoding="utf-8") as f_env:
        env_cfg = yaml.safe_load(f_env)

    with open("configs/ppo.yaml", "r", encoding="utf-8") as f_ppo:
        ppo_cfg = yaml.safe_load(f_ppo)

    num_envs: int = int(ppo_cfg.get("num_envs", 8))

    thunks = [_make_env_from_cfg(env_cfg) for _ in range(num_envs)]
    vec_cls = SubprocVecEnv if num_envs > 1 else DummyVecEnv
    vec_env = vec_cls(thunks)

    vec_env = VecNormalize(
        vec_env,
        norm_obs=bool(ppo_cfg.get("normalize_obs", True)),
        norm_reward=bool(ppo_cfg.get("normalize_reward", True)),
        clip_obs=10.0,
    )

    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=float(ppo_cfg.get("learning_rate", 3e-4)),
        n_steps=int(ppo_cfg.get("n_steps", 1024)),
        batch_size=int(ppo_cfg.get("batch_size", 256)),
        gamma=float(ppo_cfg.get("gamma", 0.99)),
        gae_lambda=float(ppo_cfg.get("gae_lambda", 0.95)),
        clip_range=float(ppo_cfg.get("clip_range", 0.2)),
        ent_coef=float(ppo_cfg.get("ent_coef", 0.0)),
        vf_coef=float(ppo_cfg.get("vf_coef", 0.5)),
        seed=int(ppo_cfg.get("seed", 42)),
        verbose=1,
    )

    # 评估环境（不渲染），用于定期评估成功率与到达时间
    eval_env = DummyVecEnv([_make_env_from_cfg(env_cfg)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)

    os.makedirs("models", exist_ok=True)
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path="models",
        log_path="models",
        eval_freq=max(10000 // num_envs, 1000),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    total_timesteps = int(1_000_000_000)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    model.save("models/ppo_fixedwing_ab.zip")
    vec_env.save("models/vecnorm.pkl")


if __name__ == "__main__":
    main()

