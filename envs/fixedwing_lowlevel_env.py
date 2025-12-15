"""项目文件：固定翼底层控制环境包装器

作者: wdblink

说明:
    将 PyFlyt 的固定翼环境包装为适合底层控制训练的环境。
    输入：航向 (Heading), 高度 (Altitude), 空速 (Airspeed), 航点 (Waypoint) 偏差
    输出：副翼 (Aileron), 升降舵 (Elevator), 方向舵 (Rudder), 油门 (Throttle)
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

class FixedwingLowLevelEnv(gym.Wrapper):
    """
    固定翼底层控制环境包装器。
    
    目标：根据给定的指令（航向、高度、空速等）控制飞机姿态。
    在训练阶段，我们随机生成这些指令作为观测的一部分，或者计算当前状态与目标状态的偏差。
    """
    def __init__(self, env):
        super().__init__(env)
        
        # 定义底层控制的目标空间（指令空间）
        # 假设指令包括：
        # 0: 目标滚转角 (Target Roll) - 对应航向控制的中间量，或者直接给航向误差
        # 1: 目标俯仰角 (Target Pitch) - 对应高度控制的中间量
        # 2: 目标空速 (Target Airspeed)
        # 或者更直接地，我们让上层输出：目标航向(Heading), 目标高度(Altitude), 目标速度(Speed)
        
        # 这里的实现方式是：随机采样一个目标状态（航向、高度、速度），
        # 并将其与当前状态的误差作为 Observation 的一部分。
        
        # 原始环境的 Observation Space (FlattenWaypointEnv):
        # [attitude(4/3), ang_vel(3), lin_vel(3), lin_pos(3), error_to_target(3*context)]
        # 我们需要修改 Observation Space，使其包含当前状态和目标指令的偏差。
        
        # 为了简化，我们假设底层策略的输入是：
        # [Roll, Pitch, Yaw, P, Q, R, U, V, W, Z, Target_Heading_Error, Target_Altitude_Error, Target_Speed_Error]
        # 或者直接使用原始观测 + 目标指令
        
        # 这里我们定义一个新的 Observation Space
        # 假设我们只关心相对误差
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64
        )
        
        # 动作空间保持不变：[Aileron, Elevator, Rudder, Throttle]
        self.action_space = env.action_space
        
        self.target_heading = 0.0
        self.target_altitude = 10.0
        self.target_airspeed = 10.0
        
        self._max_step = 500
        self._step_count = 0

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        
        # 随机生成新的控制指令
        self.target_heading = np.random.uniform(-np.pi, np.pi)
        self.target_altitude = np.random.uniform(5.0, 20.0)
        self.target_airspeed = np.random.uniform(10.0, 20.0)
        
        self._step_count = 0
        
        return self._get_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        
        # 计算自定义奖励
        # 获取当前状态
        # 注意：需要根据 FlattenWaypointEnv 的具体实现来解析 obs
        # 假设 obs 结构: [ang_vel(3), ang_pos(3), lin_vel(3), lin_pos(3), ...]
        # 具体取决于 FlattenWaypointEnv 的实现细节，通常 PyFlyt 的 state 顺序是:
        # ang_vel (3), ang_pos (3/4), lin_vel (3), lin_pos (3)
        
        # 解析状态 (假设 angle_representation="euler")
        # 如果是 quaternion，需要转换，这里假设 env 初始化时使用了 euler
        
        # 为了通用性，最好直接从 info 或 base_env 获取真实状态
        # 这里简化处理，假设我们能从 env 获取 drone 对象
        
        # 尝试从不同层级获取 aviary
        env = self.env.unwrapped
        if hasattr(env, 'aviary'):
            drone = env.aviary.drones[0]
        elif hasattr(env, 'env') and hasattr(env.env, 'aviary'):
             drone = env.env.aviary.drones[0]
        elif hasattr(env, 'drones'):
             drone = env.drones[0]
        else:
             # 如果是 PyFlyt 新版结构
             if hasattr(env, 'env'):
                 env = env.env
             if hasattr(env, 'drones'):
                 drone = env.drones[0]
             else:
                 raise AttributeError(f"Cannot find drones in env: {dir(env)}")
        # 展平 state 如果它是 (4, 3) 形状
        state = drone.state.flatten() if hasattr(drone.state, 'flatten') else drone.state
        
        # 解析状态
        # 打印状态维度以便调试 (只打印一次)
        if self._step_count <= 1:
            print(f"DEBUG: drone.state shape: {state.shape}, content: {state}")
            
        if len(state) == 12:
            # Euler representation
            # [ang_vel(3), ang_pos(3), lin_vel(3), lin_pos(3)]
            p, q, r = state[0:3]
            roll, pitch, yaw = state[3:6]
            u, v, w = state[6:9]
            x, y, z = state[9:12]
        elif len(state) == 13:
            # Quaternion representation
             pass
        else:
             # 如果是其他维度，可能 PyFlyt 版本不同
             # 尝试通用解析，假设最后三个是位置，倒数 3-6 是速度
             # 这里先抛出异常或者打印警告
             # 假设至少有位置和姿态
             # 如果 state 只有 4 维，可能是四元数？或者只有位置？
             # 根据错误信息 "expected 3, got 1"，说明 state[3:6] 返回了 1 个值？
             # 不，是 state[3:6] 返回了例如 array([val]) 无法解包成 3 个变量？
             # 或者 state 本身是个标量？
             # 根据 "got 1"，意味着 len(state[3:6]) == 1
             # 这说明 state 长度可能只有 4？
             pass
        
        # 如果 drone 对象有 attitude/position/velocity 属性，优先使用
        if hasattr(drone, 'attitude'):
            current_heading = drone.attitude[2]
        else:
             current_heading = state[5] # Yaw (Euler)
             
        if hasattr(drone, 'position'):
            current_altitude = drone.position[2]
        else:
            current_altitude = state[11] # Z
            
        if hasattr(drone, 'velocity'):
            current_airspeed = np.linalg.norm(drone.velocity)
        else:
            current_airspeed = np.linalg.norm(state[6:9])
        
        # 计算误差
        heading_error = self._normalize_angle(self.target_heading - current_heading)
        altitude_error = self.target_altitude - current_altitude
        airspeed_error = self.target_airspeed - current_airspeed
        
        # 计算奖励：鼓励减小误差
        reward = - (1.0 * abs(heading_error) + 
                    1.0 * abs(altitude_error) + 
                    0.5 * abs(airspeed_error))
        
        # 存活奖励
        reward += 0.1
        
        # 增加稳定性惩罚（角速度）
        ang_vel = drone.aux_state[0:3] # p, q, r
        reward -= 0.01 * np.linalg.norm(ang_vel)

        # 终止条件
        if self._step_count >= self._max_step:
            truncated = True
            
        # 如果坠毁或飞出边界
        if terminated:
            reward -= 100.0
            
        # 构造新的观测
        custom_obs = self._get_obs(obs)
        
        return custom_obs, reward, terminated, truncated, info

    def _get_obs(self, base_obs):
        """
        构造包含指令误差的观测向量
        """
        # 尝试从不同层级获取 aviary
        env = self.env.unwrapped
        if hasattr(env, 'aviary'):
            drone = env.aviary.drones[0]
        elif hasattr(env, 'env') and hasattr(env.env, 'aviary'):
             drone = env.env.aviary.drones[0]
        elif hasattr(env, 'drones'):
             drone = env.drones[0]
        else:
             # 如果是 PyFlyt 新版结构
             if hasattr(env, 'env'):
                 env = env.env
             if hasattr(env, 'drones'):
                 drone = env.drones[0]
             else:
                 raise AttributeError(f"Cannot find drones in env: {dir(env)}")
        
        # 展平 state 如果它是 (4, 3) 形状
        state = drone.state.flatten() if hasattr(drone.state, 'flatten') else drone.state
        
        # 状态
        
        # 尝试直接从属性获取，如果失败则从 state 解析
        # 优先使用属性，因为它们可能经过了处理（如四元数转欧拉角）
        if hasattr(drone, 'attitude'):
            roll = drone.attitude[0]
            pitch = drone.attitude[1]
            yaw = drone.attitude[2]
        elif len(state) >= 6:
            # 假设 Euler 模式: state[3:6]
            roll, pitch, yaw = state[3:6]
        else:
            # 无法解析，赋予默认值防止报错
            roll, pitch, yaw = 0.0, 0.0, 0.0
            
        if hasattr(drone, 'aux_state'):
             p, q, r = drone.aux_state[0:3]
        elif len(state) >= 3:
             p, q, r = state[0:3]
        else:
             p, q, r = 0.0, 0.0, 0.0
             
        if hasattr(drone, 'body_velocity'):
             u, v, w = state[6:9] if len(state) >= 9 else (0.0, 0.0, 0.0)
        elif len(state) >= 9:
             u, v, w = state[6:9]
        else:
             u, v, w = 0.0, 0.0, 0.0
             
        if hasattr(drone, 'position'):
             z = drone.position[2]
        elif len(state) >= 12:
             z = state[11]
        else:
             z = 0.0
        
        # 误差
        heading_error = self._normalize_angle(self.target_heading - yaw)
        altitude_error = self.target_altitude - z
        airspeed_error = self.target_airspeed - np.linalg.norm([u, v, w])
        
        obs = np.array([
            roll, pitch, p, q, r, u, v, w, z,
            heading_error, altitude_error, airspeed_error
        ], dtype=np.float32)
        
        return obs

    def _normalize_angle(self, angle):
        """将角度标准化到 [-pi, pi]"""
        return (angle + np.pi) % (2 * np.pi) - np.pi

