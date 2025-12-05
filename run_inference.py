"""项目文件：固定翼 A→B 环境模型加载与推理演示

作者: wdblink

说明:
    加载训练好的 PPO 模型，并在渲染环境下运行，同时记录并绘制 A→B 飞行轨迹。
"""

from __future__ import annotations

import argparse
import time
import yaml
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.ab_fixedwing_env import make_fixedwing_ab_env
from utils.vis import plot_xy_trajectory, plot_3d_trajectory


def main() -> None:
    """模型推理与可视化主函数。"""
    parser = argparse.ArgumentParser(description="运行固定翼无人机模型推理与可视化")
    parser.add_argument("--no-render", action="store_true", help="强制关闭 3D 渲染窗口（用于无头模式或调试）")
    args = parser.parse_args()

    # 1. 加载配置
    with open("configs/env.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2. 创建与训练时配置一致的环境（启用渲染）
    # 注意：因为训练时使用了 VecNormalize，推理时也必须包裹 VecNormalize 并加载统计数据
    # 警告：PyFlyt 的渲染可能在某些系统上导致 SegFault，如果遇到问题，尝试 render_mode=None
    # 或者不使用 render_mode，而是依靠 trajectory 绘图来观察
    render_mode = cfg["render_mode"]
    
    # 如果命令行指定了不渲染，则覆盖配置
    if args.no_render:
        print("命令行参数指示关闭渲染窗口。")
        render_mode = None
    
    # 用户反馈 WSL2 支持 3D 窗口，因此不再强制关闭
    # 如果在 WSL2 等环境下遇到 OpenGL/X11 问题，请手动将 configs/env.yaml 中的 render_mode 改为 null
    
    env = make_fixedwing_ab_env(
        render_mode=render_mode,
        num_targets=cfg["num_targets"],
        goal_reach_distance=cfg["goal_reach_distance"],
        flight_dome_size=cfg["flight_dome_size"],
        max_duration_seconds=cfg["max_duration_seconds"],
        angle_representation=cfg["angle_representation"],
        agent_hz=cfg["agent_hz"],
        context_length=cfg["context_length"],
    )
    
    # 包装为 DummyVecEnv 以匹配 SB3 接口
    env = DummyVecEnv([lambda: env])
    
    # 3. 加载归一化统计参数与模型
    # 必须加载训练时保存的 vecnorm.pkl，否则观测分布不一致会导致策略失效
    try:
        env = VecNormalize.load("models/vecnorm.pkl", env)
        # 推理时不仅不需要更新统计量，通常也不归一化奖励（便于直观观察原始回报），但需归一化观测
        env.training = False
        env.norm_reward = False
    except FileNotFoundError:
        print("警告：未找到 models/vecnorm.pkl，将使用未归一化的环境运行（可能导致效果极差）。")

    try:
        model = PPO.load("models/best_model.zip")
        print("成功加载最佳模型 models/best_model.zip")
    except FileNotFoundError:
        model = PPO.load("models/ppo_fixedwing_ab.zip")
        print("加载最终模型 models/ppo_fixedwing_ab.zip")

    # 4. 推理循环
    obs = env.reset()
    trajectory = []
    trajectory_3d = []
    
    # 获取起点 A 与终点 B 的大概坐标用于绘图（从配置读取）
    # 注意：实际环境中的随机航点可能与此不同，但若是 num_targets=1 且配置已生效，则大致相符
    # 如果环境返回了真实目标点（通过 info 或 wrapper），我们应优先使用
    point_a_3d = tuple(cfg.get("A", [0, 0, 10]))
    point_b_3d = tuple(cfg.get("B", [50, 0, 10]))
    
    # 尝试从环境中获取真实目标点（仅针对单目标）
    try:
        raw_env = env.envs[0]
        while hasattr(raw_env, 'env') and not hasattr(raw_env, 'waypoints'):
            raw_env = raw_env.env
        if hasattr(raw_env, 'waypoints') and hasattr(raw_env.waypoints, 'targets'):
             targets = raw_env.waypoints.targets
             if len(targets) > 0:
                 point_b_3d = tuple(targets[0])
                 print(f"从环境获取到随机目标点: {point_b_3d}")
    except Exception as e:
        print(f"无法获取真实目标点，使用默认配置: {e}")

    point_a = point_a_3d[:2]
    point_b = point_b_3d[:2]

    print("开始推理演示...")
    done = False
    total_reward = 0.0
    
    while not done:
        # 预测动作，deterministic=True 表示使用确定性策略（不采样）
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, done, info = env.step(action)
        
        # 添加帧率控制，避免 WSL2 下 PyBullet 渲染过快导致段错误
        if render_mode == "human":
            # 尝试降低到更低的帧率，或者不使用 time.sleep 而是在一定步数后暂停
            # 实验表明简单的 sleep 有时仍会导致问题，尝试更长的间隔
            time.sleep(1.0 / 20.0) 
            
            # 强制处理事件，防止窗口无响应（如果后端支持）
            # PyBullet 自带的 GUI 循环在 step 中，无需额外处理，但频繁调用可能导致竞争
        
        # 记录轨迹：从 info 或环境内部状态获取真实位置
        # VecEnv 的 info 是列表，取第一个
        # 尝试从 info 中提取位置（部分环境支持），若无则无法精确记录，这里暂留接口
        # PyFlyt 的 info 通常包含 'out_of_bounds', 'collision', 'env_complete' 等
        # 若需精确位置，可修改环境包装返回 pos，或者直接访问 env.envs[0].env.env.drones[0].state
        
        # 访问底层 PyFlyt 环境获取真实位置 (x, y, z)
        # 层级：DummyVecEnv -> VecNormalize -> FlattenWaypointEnv -> FixedwingWaypointsEnv -> Aviary
        try:
            # VecEnv 的 envs 列表可能被多层封装
            if hasattr(env, 'envs'):
                raw_env = env.envs[0]
            else:
                raw_env = env
            
            # 尝试解包直到获取 Aviary
            while hasattr(raw_env, 'env') or hasattr(raw_env, 'base_env'):
                if hasattr(raw_env, 'base_env'):
                    raw_env = raw_env.base_env
                elif hasattr(raw_env, 'env'):
                    raw_env = raw_env.env
            
            # 此时 raw_env 应为 FixedwingWaypointsEnv (或类似)
            # 根据 dir(raw_env) 输出，它直接具有 'drones' 属性，说明它本身就是 Aviary 或者其子类
            # PyFlyt 的环境结构中，FixedwingWaypointsEnv 继承自 AviaryWrapper 或包含 Aviary
            # 但这里的输出显示 raw_env 具有 'drones', 'all_states' 等属性，这表明 raw_env 很可能就是 Aviary 实例本身
            # 或者 FixedwingWaypointsEnv 继承自 Aviary
            
            # 尝试直接访问 raw_env.drones
            if hasattr(raw_env, 'drones'):
                 drone = raw_env.drones[0]
                 # 再次检查 drone 属性
                 if hasattr(drone, 'position'):
                     pos = drone.position
                 elif hasattr(drone, 'state'):
                     # state: [ang_vel, ang_pos, lin_vel, lin_pos]
                     pos = drone.state[-3:]
                 else:
                     # 无法解析
                     raise AttributeError(f"Drone object has no position or state: {dir(drone)}")
                 
                 trajectory.append((pos[0], pos[1]))
                 # 同时记录 3D 轨迹 (x, y, z)
                 if len(pos) >= 3:
                     # 确保保存为纯 float 列表或 tuple，而非 numpy array 导致混合
                     trajectory_3d.append((float(pos[0]), float(pos[1]), float(pos[2])))
            
            elif hasattr(raw_env, 'aviary'):
                # 尝试获取位置：PyFlyt Drone 基类通常不直接暴露 .pos
                # 而是通过 state 属性。对于 Fixedwing，state 是一个扁平数组
                # 我们需要查看 aviary.state(0) 或者直接访问 drones[0].state
                # 如果上述 drones[0].pos 失败，说明没有该属性
                # 根据 PyFlyt 源码，state 的最后 3 个元素通常是位置 (x,y,z)
                # 或者 drones[0].position (如果使用了该属性名)
                drone = raw_env.aviary.drones[0]
                if hasattr(drone, 'position'):
                    pos = drone.position
                else:
                    # 尝试从 state 解析，state通常为 [ang_vel, ang_pos, lin_vel, lin_pos]
                    # lin_pos 是最后三位
                    pos = drone.state[-3:]
                
                trajectory.append((pos[0], pos[1]))
                # 同时记录 3D 轨迹 (x, y, z)
                if len(pos) >= 3:
                    trajectory_3d.append((float(pos[0]), float(pos[1]), float(pos[2])))
            else:
                if len(trajectory) == 0:
                     print(f"DEBUG: raw_env 无 aviary 属性. dir(raw_env): {dir(raw_env)}")
        except Exception as e:
            # pass # 无法获取内部状态则跳过记录
            # 调试：打印错误以便排查
            if len(trajectory) == 0: # 只打印第一次错误
                 print(f"DEBUG: 获取位置失败: {e}, raw_env type: {type(raw_env)}")

        total_reward += reward[0]
        
        # VecEnv 会自动 reset，done 变为 True 时 obs 已经是新回合的了
        # 这里演示一回合即可
        if done[0]:
            print(f"回合结束，总回报: {total_reward:.2f}")
            break

    env.close()
    
    # 5. 绘制轨迹
    if trajectory:
        print(f"绘制轨迹，共 {len(trajectory)} 个点")
        plot_xy_trajectory(trajectory, point_a, point_b)
        
        if trajectory_3d:
             print(f"绘制 3D 轨迹，共 {len(trajectory_3d)} 个点")
             try:
                 plot_3d_trajectory(trajectory_3d, point_a_3d, point_b_3d)
             except Exception as e:
                 print(f"绘制 3D 轨迹失败: {e}")
    else:
        print("未能记录到轨迹点。")

if __name__ == "__main__":
    main()
