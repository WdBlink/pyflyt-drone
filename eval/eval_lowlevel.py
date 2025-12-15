"""项目文件：底层控制模型评估脚本

作者: wdblink

说明:
    加载训练好的底层控制模型（PPO），并在测试环境中运行，
    评估其对高级指令（航向、高度、空速）的跟踪性能。
    
    量化指标：
    1. 航向误差 (Heading Error): 平均绝对误差 (MAE) 和均方根误差 (RMSE)
    2. 高度误差 (Altitude Error): MAE, RMSE
    3. 空速误差 (Airspeed Error): MAE, RMSE
    4. 稳定性 (Stability): 角速度范数均值
    5. 存活率 (Survival Rate): 成功完成回合的比例（未坠毁/未出界）
"""

import os
import sys
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import PyFlyt.gym_envs

# 添加项目根目录到 sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.fixedwing_lowlevel_env import FixedwingLowLevelEnv

# 评估参数配置
EVAL_CONFIG = {
    "model_path": "models/lowlevel_ppo/best_model.zip", # 或 final_model.zip
    "vecnorm_path": "models/lowlevel_ppo/vecnorm.pkl", # 归一化统计量
    "num_episodes": 20, # 评估回合数
    "render": False,    # 是否渲染
    "plot_results": True, # 是否绘制跟踪曲线
}

def make_eval_env():
    """创建评估环境"""
    env = gym.make(
        "PyFlyt/Fixedwing-Waypoints-v3",
        render_mode="human" if EVAL_CONFIG["render"] else None,
        angle_representation="euler",
        flight_dome_size=200.0,
        max_duration_seconds=120.0,
        agent_hz=30,
    )
    env = FixedwingLowLevelEnv(env)
    return env

def evaluate():
    """执行评估"""
    
    # 检查模型文件是否存在
    if not os.path.exists(EVAL_CONFIG["model_path"]):
        print(f"错误: 找不到模型文件 {EVAL_CONFIG['model_path']}")
        return

    # 1. 创建环境
    # PPO 需要 VecEnv，这里使用 DummyVecEnv
    env = DummyVecEnv([make_eval_env])
    
    # 2. 加载归一化统计量 (如果训练时使用了 VecNormalize)
    if os.path.exists(EVAL_CONFIG["vecnorm_path"]):
        print(f"加载归一化统计量: {EVAL_CONFIG['vecnorm_path']}")
        env = VecNormalize.load(EVAL_CONFIG['vecnorm_path'], env)
        env.training = False # 评估模式，不更新统计量
        env.norm_reward = False # 评估时不需要归一化奖励
    else:
        print("警告: 未找到归一化统计量，直接使用原始环境。如果训练使用了 VecNormalize，结果可能不准确。")

    # 3. 加载模型
    print(f"加载模型: {EVAL_CONFIG['model_path']}")
    model = PPO.load(EVAL_CONFIG["model_path"], env=env)

    # 4. 运行评估循环
    all_heading_errors = []
    all_altitude_errors = []
    all_airspeed_errors = []
    all_ang_vels = []
    success_count = 0
    
    print(f"开始评估，共 {EVAL_CONFIG['num_episodes']} 个回合...")
    
    for ep in range(EVAL_CONFIG["num_episodes"]):
        obs = env.reset()
        done = False
        
        # 记录单个回合的数据
        ep_heading_errors = []
        ep_altitude_errors = []
        ep_airspeed_errors = []
        ep_ang_vels = []
        
        # 获取环境包装器以便访问真实状态和目标
        # 由于使用了 VecNormalize -> DummyVecEnv -> FixedwingLowLevelEnv
        # 我们需要解包访问内部属性
        base_venv = env.venv if hasattr(env, 'venv') else env
        raw_env = base_venv.envs[0].unwrapped
        
        step_cnt = 0
        done_flag = False
        while not done_flag:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            done_flag = bool(done[0])
            
            # 获取当前状态和目标（来自包装器的内部目标）
            wrapper = base_venv.envs[0]
            
            # 获取目标
            tgt_heading = wrapper.target_heading
            tgt_alt = wrapper.target_altitude
            tgt_spd = wrapper.target_airspeed
            
            # 直接使用包装器计算未归一化观测，并从中提取误差与角速度
            raw_obs = wrapper._get_obs(None)
            h_err = raw_obs[-3]
            a_err = raw_obs[-2]
            s_err = raw_obs[-1]
            cur_ang_vel = np.linalg.norm(raw_obs[2:5])
            
            ep_heading_errors.append(abs(h_err))
            ep_altitude_errors.append(abs(a_err))
            ep_airspeed_errors.append(abs(s_err))
            ep_ang_vels.append(cur_ang_vel)
            
            step_cnt += 1
            
            if done_flag:
                # 检查是否是因为 max_step 结束（视为存活/成功）还是坠毁
                # info 是个列表 (VecEnv)
                inf = info[0]
                # PyFlyt 的 info 包含 'collision', 'out_of_bounds'
                # 如果这些为 True，则视为失败
                collision = inf.get('collision', False)
                out_of_bounds = inf.get('out_of_bounds', False)
                
                if not collision and not out_of_bounds:
                    success_count += 1
                    
        # 回合结束，记录平均误差
        if ep_heading_errors:
            all_heading_errors.extend(ep_heading_errors)
            all_altitude_errors.extend(ep_altitude_errors)
            all_airspeed_errors.extend(ep_airspeed_errors)
            all_ang_vels.extend(ep_ang_vels)
            
        print(f"Episode {ep+1}/{EVAL_CONFIG['num_episodes']} | "
              f"Steps: {step_cnt} | "
              f"MAE Heading: {np.mean(ep_heading_errors):.4f} | "
              f"MAE Alt: {np.mean(ep_altitude_errors):.4f} | "
              f"MAE Spd: {np.mean(ep_airspeed_errors):.4f}")

    env.close()
    
    # 5. 计算并打印总体指标
    print("\n" + "="*40)
    print("评估结果汇总")
    print("="*40)
    
    def calc_metrics(errors, name):
        if not errors: return
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors)**2))
        print(f"{name}:")
        print(f"  MAE (平均绝对误差): {mae:.4f}")
        print(f"  RMSE (均方根误差): {rmse:.4f}")

    calc_metrics(all_heading_errors, "航向误差 (Heading Error) [rad]")
    calc_metrics(all_altitude_errors, "高度误差 (Altitude Error) [m]")
    calc_metrics(all_airspeed_errors, "空速误差 (Airspeed Error) [m/s]")
    
    print(f"稳定性 (平均角速度范数): {np.mean(all_ang_vels):.4f} [rad/s]")
    print(f"存活率 (Survival Rate): {success_count / EVAL_CONFIG['num_episodes'] * 100:.2f}%")
    print("="*40)
    
    # 6. (可选) 绘制最后一回合的跟踪曲线
    # 若需绘制，需在循环中保存具体时序数据
    pass

if __name__ == "__main__":
    evaluate()
