"""项目文件：A→B 固定翼环境工厂

作者: wdblink

说明:
    提供一个基于 PyFlyt 的固定翼环境工厂方法，用于创建可渲染并兼容
    Gymnasium 的训练环境。初期实现直接封装官方 `PyFlyt/Fixedwing-Waypoints-v3`
    环境，并通过 `FlattenWaypointEnv` 扁平化观测空间，以便 PPO 训练。

    设计考虑：
    - 可扩展：后续可在此处引入 A/B 固定目标、奖励塑形、风场等配置。
    - 可维护：集中环境参数与构造逻辑，减少训练脚本耦合。
"""

from __future__ import annotations

import gymnasium as gym
from typing import Optional

try:
    # 官方提供的扁平化封装，兼容 Dict/Sequence 观测到 Box
    from PyFlyt.gym_envs import FlattenWaypointEnv
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "未找到 PyFlyt.gym_envs.FlattenWaypointEnv，请先安装并在 Python 环境中可用"
    ) from exc


def make_fixedwing_ab_env(
    render_mode: Optional[str] = "human",
    num_targets: int = 1,
    goal_reach_distance: float = 2.0,
    flight_dome_size: float = 100.0,
    max_duration_seconds: float = 120.0,
    angle_representation: str = "quaternion",
    agent_hz: int = 30,
    context_length: int = 1,
):
    """创建固定翼 A→B 训练环境（初版使用官方 Waypoints 环境）。

    此函数返回一个经过 Flatten 处理后的环境，以便与 Stable-Baselines3 的
    PPO 直接兼容。
    
    关键特性：
    - 随机航点：默认情况下（不传递 targets 参数），环境会在 `flight_dome_size`
      范围内随机生成 `num_targets` 个目标点。每次 `reset()` 都会重新生成。
    - 扁平化观测：使用 `FlattenWaypointEnv` 将复杂的 Dict 观测转换为 Box。

    Args:
        render_mode: 渲染模式，取值如 `"human"` 或 `None`。
        num_targets: 航点数量，先期设为 1。
        goal_reach_distance: 判定到达航点的距离阈值。
        flight_dome_size: 允许飞行范围半径。
        max_duration_seconds: 最大仿真时长。
        angle_representation: 姿态表示，`"euler"` 或 `"quaternion"`。
        agent_hz: 交互频率（智能体与环境步进的频率）。
        context_length: 扁平化观测中包含的“立即目标”数量。

    Returns:
        Gymnasium 环境实例，观测空间已扁平化为 Box。
    """

    base_env = gym.make(
        "PyFlyt/Fixedwing-Waypoints-v3",
        sparse_reward=False,
        num_targets=num_targets,
        goal_reach_distance=goal_reach_distance,
        flight_dome_size=flight_dome_size,
        max_duration_seconds=max_duration_seconds,
        angle_representation=angle_representation,
        agent_hz=agent_hz,
        render_mode=render_mode,
    )

    # 扁平化以适配主流 RL 算法
    env = FlattenWaypointEnv(base_env, context_length=context_length)
    return env

