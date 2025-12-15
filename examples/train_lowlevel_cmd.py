"""
固定翼底层控制训练脚本（PyFlyt + Gymnasium）：
- 训练目标：让智能体输出六维面控与油门，跟踪高层目标（航向/高度/空速）
- 环境实现：自定义Gym环境封装Aviary，动作为6维[-1,1]面控，观测包含姿态、速度、位置、上一动作与目标
- 可视化：可选第三人称追尾相机（render_mode='human'时启用）

作者: wdblink
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import gymnasium
import numpy as np
from gymnasium import spaces

from PyFlyt.core.aviary import Aviary


class FixedwingLowLevelEnv(gymnasium.Env):
    """固定翼底层控制训练环境。

    本环境直接暴露固定翼的六维面控与油门动作（mode=-1）：
    动作为np.ndarray([left_ail, right_ail, hstab, vstab, flap, thrust])，范围为[-1, 1]。
    观测包含姿态/速度/位置、上一动作以及要跟踪的高层目标[psi_ref, h_ref, V_ref]。
    奖励以高层目标跟踪误差为主，并包含安全与平滑约束。
    """

    metadata = {"render_modes": ["human"], "name": "fixedwing_lowlevel_env"}

    def __init__(
        self,
        flight_dome_size: float = 800.0,
        max_duration_seconds: float = 60.0,
        agent_hz: int = 30,
        render_mode: None | str = None,
        start_height_m: float = 120.0,
        start_speed_mps: float = 25.0,
        min_speed_mps: float = 8.0,
        target_speed_range: Tuple[float, float] = (20.0, 35.0),
        target_height_range: Tuple[float, float] = (100.0, 200.0),
        target_heading_random: bool = True,
    ):
        """初始化环境与Aviary。

        Args:
            flight_dome_size: 飞行半径限制，超过则截断
            max_duration_seconds: 单回合最大时长
            agent_hz: 智能体交互频率（Hz）
            render_mode: 可为None或'human'
            start_height_m: 初始高度（m）
            start_speed_mps: 初始速度（m/s）
            min_speed_mps: 失速阈值（m/s）用于安全约束
            target_speed_range: 目标空速范围（m/s）
            target_height_range: 目标高度范围（m）
            target_heading_random: 是否随机目标航向（弧度）
        """
        super().__init__()

        self.render_mode = render_mode
        self.flight_dome_size = flight_dome_size
        self.max_duration_seconds = max_duration_seconds
        self.agent_hz = agent_hz
        self.step_dt = 1.0 / float(agent_hz)

        self.start_height_m = start_height_m
        self.start_speed_mps = start_speed_mps
        self.min_speed_mps = min_speed_mps
        self.target_speed_range = target_speed_range
        self.target_height_range = target_height_range
        self.target_heading_random = target_heading_random

        start_pos = np.array([[0.0, 0.0, self.start_height_m]])
        start_orn = np.array([[0.0, 0.0, 0.0]])
        drone_options = dict(starting_velocity=np.array([self.start_speed_mps, 0.0, 0.0]))

        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="fixedwing",
            drone_options=drone_options,
            render=self.render_mode == "human",
            physics_hz=240,
            world_scale=1.0,
        )
        self.env.set_mode(-1)

        self._episode_steps = 0
        self._elapsed_time_s = 0.0
        self.prev_action = np.zeros(6, dtype=np.float64)
        self.target = np.array([0.0, self.start_height_m, self.start_speed_mps], dtype=np.float64)

        low = -np.inf * np.ones(3 + 3 + 3 + 3 + 6 + 3, dtype=np.float64)
        high = np.inf * np.ones_like(low)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float64)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict]:
        """重置环境并采样新的高层目标。"""
        super().reset(seed=seed)
        self.env.reset()
        self.env.set_mode(-1)
        self._episode_steps = 0
        self._elapsed_time_s = 0.0
        self.prev_action = np.zeros(6, dtype=np.float64)

        psi_ref = float(self.env.np_random.uniform(-math.pi, math.pi)) if self.target_heading_random else 0.0
        h_ref = float(self.env.np_random.uniform(*self.target_height_range))
        V_ref = float(self.env.np_random.uniform(*self.target_speed_range))
        self.target = np.array([psi_ref, h_ref, V_ref], dtype=np.float64)

        obs = self._compute_obs()
        info = {"target": self.target.copy()}
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """执行一步控制并计算奖励、终止与截断。"""
        act = np.asarray(action, dtype=np.float64)
        act = np.clip(act, -1.0, 1.0)
        self.env.set_setpoint(0, act)
        self.env.step()
        self._episode_steps += 1
        self._elapsed_time_s += self.step_dt

        obs = self._compute_obs()
        reward = float(self._compute_reward())
        terminated = bool(self._is_terminated())
        truncated = bool(self._is_truncated())
        info = {"target": self.target.copy()}

        self.prev_action = act
        if self.render_mode == "human":
            self._update_follow_camera()

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        """关闭环境资源。"""
        try:
            self.env.disconnect()
        except Exception:
            pass

    def _compute_obs(self) -> np.ndarray:
        """计算观测向量。"""
        s = self.env.state(0)
        ang_vel = s[0]
        ang_pos = s[1]
        lin_vel = s[2]
        lin_pos = s[3]
        return np.concatenate([ang_vel, ang_pos, lin_vel, lin_pos, self.prev_action, self.target], axis=-1).astype(
            np.float64
        )

    def _compute_reward(self) -> float:
        """根据目标跟踪与安全约束计算奖励。"""
        s = self.env.state(0)
        ang_pos = s[1]
        lin_vel = s[2]
        lin_pos = s[3]

        psi = float(ang_pos[2])
        speed = float(np.linalg.norm(lin_vel))
        alt = float(lin_pos[2])

        psi_ref, h_ref, V_ref = self.target.tolist()

        heading_err = self._wrap_pi(psi_ref - psi)
        alt_err = h_ref - alt
        speed_err = V_ref - speed

        r_track = -1.5 * abs(heading_err) - 1.2 * abs(alt_err) - 0.8 * abs(speed_err)

        roll = float(ang_pos[0])
        pitch = float(ang_pos[1])
        r_stability = -0.3 * max(0.0, abs(roll) - math.radians(35.0)) - 0.3 * max(0.0, abs(pitch) - math.radians(20.0))

        r_actions = -0.05 * float(np.linalg.norm(self.prev_action))

        r_bounds = -1.0 * max(0.0, float(np.linalg.norm(lin_pos[:2])) - self.flight_dome_size)
        r_stall = -2.0 if speed < self.min_speed_mps else 0.0

        return r_track + r_stability + r_actions + r_bounds + r_stall

    def _is_terminated(self) -> bool:
        """判断是否任务终止（失败或完成）。"""
        s = self.env.state(0)
        lin_vel = s[2]
        lin_pos = s[3]
        speed = float(np.linalg.norm(lin_vel))
        alt = float(lin_pos[2])
        if alt < 1.0:
            return True
        if speed < 5.0:
            return True
        return False

    def _is_truncated(self) -> bool:
        """判断是否回合截断（出界或时长限制）。"""
        s = self.env.state(0)
        lin_pos = s[3]
        if float(np.linalg.norm(lin_pos[:2])) > self.flight_dome_size * 1.2:
            return True
        if self._elapsed_time_s >= self.max_duration_seconds:
            return True
        return False

    def _update_follow_camera(self) -> None:
        """更新第三人称追尾相机。"""
        s = self.env.state(0)
        pos = s[3]
        yaw = s[1][2]
        speed = float(np.linalg.norm(s[2]))
        camera_distance = max(15.0, min(60.0, 20.0 + 1.0 * speed))
        camera_yaw = math.degrees(yaw) + 180.0
        camera_pitch = -20.0
        self.env.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=pos.tolist(),
        )

    @staticmethod
    def _wrap_pi(a: float) -> float:
        """将角度差规约到[-pi, pi]。"""
        while a > math.pi:
            a -= 2.0 * math.pi
        while a < -math.pi:
            a += 2.0 * math.pi
        return a


def train_sac(total_timesteps: int = 50_000, render_mode: None | str = None) -> None:
    """使用SAC训练固定翼底层控制。"""
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.vec_env import DummyVecEnv
    except Exception as e:
        print("未检测到stable-baselines3，无法进行SAC训练。请先安装：pip install stable-baselines3")
        raise e

    def _make_env():
        return FixedwingLowLevelEnv(render_mode=render_mode)

    vec_env = DummyVecEnv([_make_env])
    model = SAC(
        policy="MlpPolicy",
        env=vec_env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=200_000,
        batch_size=256,
        gamma=0.99,
        tau=0.02,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("sac_fixedwing_lowlevel")
    print("训练完成，模型已保存为 sac_fixedwing_lowlevel.zip")


def evaluate_model(model_path: str = "sac_fixedwing_lowlevel", steps: int = 2000, render_mode: str = "human") -> None:
    """加载训练好的模型并进行评估。"""
    try:
        from stable_baselines3 import SAC
    except Exception as e:
        print("未检测到stable-baselines3，无法加载模型。请先安装：pip install stable-baselines3")
        raise e

    env = FixedwingLowLevelEnv(render_mode=render_mode)
    
    try:
        model = SAC.load(model_path, env=env)
        print(f"成功加载模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    obs, _ = env.reset()
    term, trunc = False, False
    
    # 初始化统计指标
    total_rewards = []
    episode_reward = 0.0
    
    # 误差统计列表 (所有步)
    heading_errors = []
    altitude_errors = []
    speed_errors = []
    
    current_step = 0
    episodes = 0
    
    print(f"\n开始评估模型: {model_path}")
    print(f"计划运行步数: {steps}")
    print("-" * 50)
    
    while current_step < steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, rew, term, trunc, info = env.step(action)
        
        # 记录单步数据
        episode_reward += rew
        target = info["target"]
        
        # 从obs或env获取当前状态计算误差
        # obs结构: ang_vel(3), ang_pos(3), lin_vel(3), lin_pos(3), prev_action(6), target(3)
        # ang_pos在索引3-5, lin_vel在索引6-8, lin_pos在索引9-11
        current_psi = obs[5]  # ang_pos[2]
        current_vel = np.linalg.norm(obs[6:9])
        current_alt = obs[11] # lin_pos[2]
        
        target_psi, target_h, target_v = target
        
        # 计算误差
        h_err = abs(target_h - current_alt)
        v_err = abs(target_v - current_vel)
        psi_err = abs(FixedwingLowLevelEnv._wrap_pi(target_psi - current_psi))
        
        altitude_errors.append(h_err)
        speed_errors.append(v_err)
        heading_errors.append(math.degrees(psi_err))
        
        current_step += 1
        
        if term or trunc:
            total_rewards.append(episode_reward)
            episodes += 1
            print(f"Episode {episodes}: Reward = {episode_reward:.2f}, "
                  f"Avg Alt Err = {np.mean(altitude_errors[-int(env._episode_steps):]):.2f}m, "
                  f"Avg Spd Err = {np.mean(speed_errors[-int(env._episode_steps):]):.2f}m/s")
            
            episode_reward = 0.0
            obs, _ = env.reset()
            term, trunc = False, False
            
    env.close()
    
    # 输出最终统计结果
    print("-" * 50)
    print("评估完成 Summary:")
    print(f"Total Episodes: {episodes}")
    if total_rewards:
        print(f"Mean Reward per Episode: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    
    if heading_errors:
        print(f"Heading Error (deg): Mean={np.mean(heading_errors):.2f}, Max={np.max(heading_errors):.2f}, RMSE={np.sqrt(np.mean(np.square(heading_errors))):.2f}")
        print(f"Altitude Error (m) : Mean={np.mean(altitude_errors):.2f}, Max={np.max(altitude_errors):.2f}, RMSE={np.sqrt(np.mean(np.square(altitude_errors))):.2f}")
        print(f"Speed Error (m/s)  : Mean={np.mean(speed_errors):.2f}, Max={np.max(speed_errors):.2f}, RMSE={np.sqrt(np.mean(np.square(speed_errors))):.2f}")
    print("-" * 50)


if __name__ == "__main__":
    # 训练模型 (注释掉以跳过训练)
    train_sac(total_timesteps=100_000, render_mode=None)
    
    # 加载并评估模型
    evaluate_model(model_path="sac_fixedwing_lowlevel", steps=2000, render_mode="human")

