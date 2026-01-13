"""项目文件：固定翼高层指令训练脚本

作者: wdblink

说明:
    使用 PyFlyt 的固定翼追航点环境作为基础任务，训练一个高层策略网络（High-Level Policy），
    高层策略输出高级指令（航向 Heading、目标高度 Altitude、目标空速 Airspeed）。
    高层指令通过已训练的底层控制器（Low-Level Policy, PPO）转换为环境可接受的控制命令
    （roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd），从而在环境中连续追航点。
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# 需要显式导入以确保注册 PyFlyt 环境
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv

from envs.fixedwing_lowlevel_env import FixedwingLowLevelEnv
from envs.utils import WindOnResetWrapper


class HighLevelCmdEnv(gym.Env):
    """固定翼高层指令控制训练环境。

    设计:
        - 基础任务: 继承 PyFlyt/Fixedwing-Waypoints-v3 的追航点任务（奖励/终止与基础环境一致）
        - 高层动作: [heading_cmd, altitude_cmd, airspeed_cmd]
        - 低层控制: 加载已训练的底层 PPO 模型，将高层指令转换为环境动作 [roll, pitch, yaw, thrust]
        - 观测: 使用 FlattenWaypointEnv 的扁平观测（包含目标点误差等），便于高层策略推理

    可扩展性:
        - 可替换底层控制器（不同算法/模型）
        - 可调整高层动作范围、奖励塑形与航点上下文长度
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        flight_dome_size: float = 200.0,
        max_duration_seconds: float = 120.0,
        agent_hz: int = 30,
        context_length: int = 2,
        low_model_path: str = "models/lowlevel_ppo/best_model.zip",
        low_vecnorm_path: str = "models/lowlevel_ppo/vecnorm.pkl",
        deterministic_low: bool = True,
        wind_config: dict | None = None,
    ):
        """初始化高层环境。

        Args:
            render_mode: 渲染模式（None 或 "human"）
            flight_dome_size: 飞行范围半径
            max_duration_seconds: 最大仿真时长
            agent_hz: 智能体交互频率
            context_length: FlattenWaypointEnv 的航点上下文长度
            low_model_path: 底层 PPO 模型路径
            low_vecnorm_path: 底层 VecNormalize 统计量路径
            deterministic_low: 底层模型是否使用确定性动作
        """
        super().__init__()

        # 基础环境 + 扁平化观测
        base_env = gym.make(
            "PyFlyt/Fixedwing-Waypoints-v3",
            render_mode=render_mode,
            angle_representation="euler",
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            agent_hz=agent_hz,
        )
        if wind_config and bool(wind_config.get("enabled", False)):
            base_env = WindOnResetWrapper(base_env, wind_config)
        self._base_env = FlattenWaypointEnv(base_env, context_length=context_length)

        # 保存范围参数用于动作空间定义
        self._flight_dome_size = flight_dome_size

        # 高层动作空间: [heading, altitude, airspeed]
        # - heading_cmd: [-pi, pi]
        # - altitude_cmd: [0, flight_dome_size]
        # - airspeed_cmd: [0.0, 30.0]（简化选择，后续可调整/标定）
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.pi, flight_dome_size, 30.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 高层观测空间沿用 FlattenWaypointEnv
        self.observation_space = self._base_env.observation_space

        # 构造一个底层包装器用于生成底层模型的观测（不用于 step）
        self._low_wrapper = FixedwingLowLevelEnv(self._base_env)

        # 加载底层 PPO 模型
        if not os.path.exists(low_model_path):
            raise FileNotFoundError(
                f"未找到底层模型文件: {low_model_path}，请先完成底层训练"
            )
        self._low_model = PPO.load(low_model_path)
        self._low_deterministic = deterministic_low

        # 加载 VecNormalize 统计量用于手动归一化底层观测
        if not os.path.exists(low_vecnorm_path):
            raise FileNotFoundError(
                f"未找到底层归一化统计量: {low_vecnorm_path}，请确保底层训练保存了 vecnorm.pkl"
            )
        from stable_baselines3.common.vec_env import VecNormalize

        self._low_vecnorm = VecNormalize.load(low_vecnorm_path, DummyVecEnv([lambda: self._low_wrapper]))
        # 评估模式：不更新统计，不归一化奖励
        self._low_vecnorm.training = False
        self._low_vecnorm.norm_reward = False
        self._clip_obs = getattr(self._low_vecnorm, "clip_obs", 10.0)

    def _normalize_angle(self, angle: float) -> float:
        """角度标准化到 [-pi, pi]。"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def _normalize_low_obs(self, obs: np.ndarray) -> np.ndarray:
        """根据 VecNormalize 的统计量对底层观测进行手动归一化。"""
        # VecNormalize 使用 RunningMeanStd（mean, var）
        obs_rms = self._low_vecnorm.obs_rms
        mean = obs_rms.mean
        var = obs_rms.var
        eps = 1e-8
        norm = (obs - mean) / np.sqrt(var + eps)
        # 裁剪
        return np.clip(norm, -self._clip_obs, self._clip_obs).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境并返回扁平化观测。"""
        obs, info = self._base_env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: np.ndarray):
        """执行一步：高层动作经底层控制器转换为环境动作，再前进一步。

        Args:
            action: 高层指令 [heading_cmd, altitude_cmd, airspeed_cmd]
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # 解析高层动作
        heading_cmd = float(action[0])
        altitude_cmd = float(action[1])
        airspeed_cmd = float(action[2])

        # 写入底层目标
        self._low_wrapper.target_heading = self._normalize_angle(heading_cmd)
        self._low_wrapper.target_altitude = float(np.clip(altitude_cmd, 0.0, self._flight_dome_size))
        self._low_wrapper.target_airspeed = float(np.clip(airspeed_cmd, 0.0, 100.0))

        # 生成底层模型观测（未归一化）
        low_obs_raw = self._low_wrapper._get_obs(None).astype(np.float32)
        # 归一化到底层模型训练分布
        low_obs_norm = self._normalize_low_obs(low_obs_raw)

        # 底层模型推理得到环境动作（roll, pitch, yaw, thrust）
        low_action, _ = self._low_model.predict(
            low_obs_norm, deterministic=self._low_deterministic
        )

        # 执行基础环境一步
        obs, reward, terminated, truncated, info = self._base_env.step(low_action)

        return obs, reward, terminated, truncated, info


# 训练参数
TRAIN_CFG = {
    "total_timesteps": 20_000_000,
    "num_envs": 16,
    "learning_rate": 3e-4,
    "n_steps": 1024,
    "batch_size": 256,
    "n_epochs": 10,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": 123,
    "log_dir": "logs/highlevel_ppo",
    "model_dir": "models/highlevel_ppo",
    "wind": {
        "enabled": False,
        "mode": "constant",
        "wind_enu_mps": [0.0, 0.0, 0.0],
    },
}


def make_env(rank: int, seed: int = 0):
    """高层环境工厂（并行环境用）。"""

    def _init():
        env = HighLevelCmdEnv(
            render_mode=None,
            flight_dome_size=200.0,
            max_duration_seconds=120.0,
            agent_hz=30,
            context_length=2,
            low_model_path=os.path.join("models", "lowlevel_ppo", "best_model.zip"),
            low_vecnorm_path=os.path.join("models", "lowlevel_ppo", "vecnorm.pkl"),
            deterministic_low=True,
            wind_config=TRAIN_CFG.get("wind", None),
        )
        env.reset(seed=seed + rank)
        return env

    return _init


def train():
    """训练高层策略。"""
    os.makedirs(TRAIN_CFG["log_dir"], exist_ok=True)
    os.makedirs(TRAIN_CFG["model_dir"], exist_ok=True)

    # 并行环境
    env = SubprocVecEnv([make_env(i, TRAIN_CFG["seed"]) for i in range(TRAIN_CFG["num_envs"])])

    # 初始化模型
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=TRAIN_CFG["learning_rate"],
        n_steps=TRAIN_CFG["n_steps"],
        batch_size=TRAIN_CFG["batch_size"],
        n_epochs=TRAIN_CFG["n_epochs"],
        gamma=TRAIN_CFG["gamma"],
        gae_lambda=TRAIN_CFG["gae_lambda"],
        clip_range=TRAIN_CFG["clip_range"],
        ent_coef=TRAIN_CFG["ent_coef"],
        vf_coef=TRAIN_CFG["vf_coef"],
        max_grad_norm=TRAIN_CFG["max_grad_norm"],
        verbose=1,
        tensorboard_log=TRAIN_CFG["log_dir"],
        seed=TRAIN_CFG["seed"],
        device="cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
    )

    print(f"开始训练高层控制器，目标步数: {TRAIN_CFG['total_timesteps']}")
    print(f"设备: {model.device}")

    model.learn(total_timesteps=TRAIN_CFG["total_timesteps"], progress_bar=True)

    # 保存模型
    model.save(os.path.join(TRAIN_CFG["model_dir"], "final_model"))
    print("训练完成。")
    env.close()


def smoke_test_once():
    """初始化并执行一步以验证无报错。"""
    env = HighLevelCmdEnv(
        render_mode=None,
        low_model_path=os.path.join("models", "lowlevel_ppo", "best_model.zip"),
        low_vecnorm_path=os.path.join("models", "lowlevel_ppo", "vecnorm.pkl"),
        wind_config=TRAIN_CFG.get("wind", None),
    )
    obs, _ = env.reset()
    # 随机高层动作
    high_act = env.action_space.sample()
    step_out = env.step(high_act)
    print("Sanity step OK:", isinstance(step_out, tuple) and len(step_out) == 5)
    env.close()


if __name__ == "__main__":
    # 默认先做一次快速验证，再启动训练
    # smoke_test_once()
    # 如需训练，取消下行注释
    train()
