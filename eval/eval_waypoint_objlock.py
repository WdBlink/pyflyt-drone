"""项目文件：Fixedwing-Waypoints-v3 模型评估脚本

说明:
    加载已训练的 PPO 模型和 VecNormalize 统计量，评估模型在 Fixedwing-Waypoints-v3 环境中的表现。
    支持可视化渲染和量化指标计算。
"""

import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
import torch
import pybullet_data
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy

# 确保能导入 PyFlyt 环境
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import PyFlyt.gym_envs
# from PyFlyt.gym_envs import FlattenWaypointEnv # Use local patched version
from envs.flatten_waypoint_env import FlattenWaypointEnv
from envs.fixedwing_waypoint_objlock_env import FixedwingWaypointObjLockEnv
from envs.utils import PyBulletDebugOverlay

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from train.train_Fixedwing_Waypoints_ObjLock import TRAIN_CONFIG as TRAIN_CONFIG_OBJLOCK

# 评估配置（默认值，可通过命令行参数覆盖）
EVAL_CONFIG = {
    "model_path": "models/obj_strick_ppo_v2.1/best_model.zip",
    "vecnorm_path": None,
    "num_episodes": 10,
    "flight_dome_size": 200,
    "waypoint_spawn_size": 100,
    "num_targets": 10,
    "goal_reach_distance": 8,
    "max_duration_seconds": 300.0,
    "context_length": 2,
    "render": True,
    # 终局目标：完成所有航点后，进入“相机锁定并撞击小黄鸭”阶段
    "duck_place_at_last_waypoint": False,
    "duck_camera_capture_interval_steps": 6,
    "duck_lock_hold_steps": 10,
    "duck_strike_distance_m": 8,
    "duck_strike_reward": 200.0,
    "duck_lock_step_reward": 0.1,
    "duck_approach_reward_scale": 0.05,
    "duck_switch_min_consecutive_seen": 2,
    "duck_switch_min_area": 0.0005,
    "duck_global_scaling": 30.0,

    #障碍物参数设置
    "num_obstacles": 0,
    "obstacle_radius": 2.0,
    "obstacle_height_range": (10.0, 30.0),
}

def _infer_vecnorm_path(model_path: str, vecnorm_path: str | None) -> str | None:
    if vecnorm_path:
        return vecnorm_path
    model_dir = os.path.dirname(model_path)
    candidates = [
        os.path.join(model_dir, "vecnorm.pkl"),
        "models/waypoints_ppo_v3.5/vecnorm.pkl",
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None

def _resolve_single_env(env):
    """Resolve the underlying environment from a VecEnv or Wrapper."""
    if hasattr(env, "venv"): 
        env = env.venv
    if hasattr(env, "envs"): 
        env = env.envs[0]
    return env

def _unwrap_env(env, max_depth=32):
    """Unwrap wrappers to find the base environment."""
    cur = env
    for _ in range(max_depth):
        if hasattr(cur, "env"):
            cur = cur.env
        elif hasattr(cur, "base_env"):
             cur = cur.base_env
        else:
            break
    return cur

def _find_env_in_chain(env, predicate, max_depth=32):
    """Walk down the wrapper chain and return the first env satisfying predicate."""
    cur = env
    for _ in range(max_depth):
        if predicate(cur):
            return cur
        if hasattr(cur, "env"):
            cur = cur.env
        else:
            break
    return None

def _try_get_state12(env):
    """Try to extract 12D state (p,q,r, u,v,w, phi,theta,psi, x,y,z) from environment."""
    if hasattr(env, "state"):
        s = env.state
        if isinstance(s, dict) and "attitude" in s:
            return s["attitude"]
        if isinstance(s, np.ndarray):
             return s
    return None

def _try_get_targets(env):
    """Try to extract targets list from environment."""
    if hasattr(env, "waypoints") and hasattr(env.waypoints, "targets"):
        return env.waypoints.targets
    return None

def make_eval_env(render_mode="human"):
    """
    创建评估环境
    """
    # 创建 FixedwingWaypointObjLockEnv
    env = FixedwingWaypointObjLockEnv(
        num_targets=int(EVAL_CONFIG["num_targets"]),
        goal_reach_distance=float(EVAL_CONFIG["goal_reach_distance"]),
        flight_dome_size=float(EVAL_CONFIG["flight_dome_size"]),
        waypoint_spawn_size=float(EVAL_CONFIG["waypoint_spawn_size"]),
        max_duration_seconds=float(EVAL_CONFIG["max_duration_seconds"]),
        angle_representation="euler",
        agent_hz=30,
        render_mode=render_mode,
        # 使用默认的小黄鸭配置或根据需要添加
        # Duck Configs
        duck_strike_distance_m=EVAL_CONFIG["duck_strike_distance_m"],
        duck_approach_reward_scale=EVAL_CONFIG["duck_approach_reward_scale"],
        duck_global_scaling=EVAL_CONFIG["duck_global_scaling"],
        # 障碍物参数
        num_obstacles=EVAL_CONFIG["num_obstacles"],
        obstacle_radius=EVAL_CONFIG["obstacle_radius"],
        obstacle_height_range=EVAL_CONFIG["obstacle_height_range"],
    )
    
    # 扁平化观测空间
    env = FlattenWaypointEnv(env, context_length=EVAL_CONFIG["context_length"])
    return env

def evaluate():
    """
    评估主函数
    """
    parser = argparse.ArgumentParser(description="Evaluate PPO model for Fixedwing Waypoints")
    parser.add_argument("--model", type=str, default=EVAL_CONFIG["model_path"], help="Path to the model file")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to the vecnorm file")
    parser.add_argument("--episodes", type=int, default=EVAL_CONFIG["num_episodes"], help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    args = parser.parse_args()

    # 更新配置
    model_path = args.model
    vecnorm_path = _infer_vecnorm_path(model_path, args.vecnorm)
    render_mode = None if args.no_render else "human"
    
    print(f"Loading model from: {model_path}")
    print(f"Loading vecnorm from: {vecnorm_path}")

    if not os.path.exists(model_path):
        # 尝试回退到 final_model
        fallback_path = model_path.replace("best_model", "final_model")
        if os.path.exists(fallback_path):
            print(f"Model not found at {model_path}, using {fallback_path} instead.")
            model_path = fallback_path
        else:
            print(f"Error: Model file not found at {model_path}")
            return

    # 创建 DummyVecEnv 用于评估（VecNormalize 需要 VecEnv 包装）
    # 注意：这里我们只创建一个环境
    env = DummyVecEnv([lambda: make_eval_env(render_mode)])
    
    # 加载 VecNormalize 统计量
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: vecnorm.pkl not found, running without VecNormalize.")
    env.seed(int(time.time()) % 2**31)
    # 加载模型
    model = PPO.load(model_path, env=env)
    
    print(f"Starting evaluation for {args.episodes} episodes...")
    
    # 记录详细指标
    episode_rewards = []
    episode_lengths = []
    success_count = 0 # 简单的成功计数（如果 info 中有相关字段）
    
    overlay = PyBulletDebugOverlay()

    for i in range(args.episodes):
        obs = env.reset()
        # 解包到底层 FixedwingWaypointsEnv，读取真实航点
        raw = env.venv.envs[0]          # DummyVecEnv 里唯一一个 env
        while hasattr(raw, "env") and not hasattr(raw, "waypoints"):
            raw = raw.env               # VecNormalize -> FlattenWaypointEnv -> FixedwingWaypointsEnv
        if hasattr(raw, "waypoints"):
            print(f"Episode {i+1} targets:\n", raw.waypoints.targets)
        
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # VecEnv 返回的 done 是一个数组，我们这里只有一个环境
            done = done[0]
            
            total_reward += reward[0]
            steps += 1

            if render_mode == "human":
                try:
                    env0 = _resolve_single_env(env)
                    # 查找包含 state 的环境 (FixedwingWaypointsEnv)
                    state_env = _find_env_in_chain(env0, lambda e: hasattr(e, "state"))
                    # 查找包含 waypoints 的环境
                    waypoints_env = _find_env_in_chain(env0, lambda e: hasattr(e, "waypoints"))
                    raw_env = _unwrap_env(env0)

                    state12 = _try_get_state12(state_env if state_env is not None else raw_env)
                    
                    if i == 0 and steps == 0:
                        print(f"DEBUG: raw_env type: {type(raw_env)}")
                        if state_env:
                             print(f"DEBUG: Found state_env type: {type(state_env)}")
                        else:
                             print("DEBUG: state_env not found")
                        
                        env_to_use = state_env if state_env is not None else raw_env
                        if hasattr(env_to_use, "state"):
                             s = env_to_use.state
                             print(f"DEBUG: env.state type: {type(s)}")
                             if isinstance(s, dict):
                                 print(f"DEBUG: env.state keys: {s.keys()}")
                                 if "attitude" in s:
                                     print(f"DEBUG: attitude shape: {s['attitude'].shape}")
                                     print(f"DEBUG: attitude sample: {s['attitude'][:12]}")
                        else:
                             print("DEBUG: env_to_use has no state attribute")

                    targets = _try_get_targets(waypoints_env if waypoints_env is not None else raw_env)

                    altitude = 0.0
                    airspeed = 0.0
                    thrust = 0.0
                    targets_remaining = 0
                    target_dist = 0.0

                    if state12 is not None:
                        # state: [ang_vel(3), ang_pos(3), lin_vel(3), lin_pos(3), action(4), aux(4)]
                        # Indices: 0-2, 3-5, 6-8, 9-11
                        
                        vx, vy, vz = state12[6], state12[7], state12[8]
                        z = state12[11]
                        altitude = z
                        airspeed = np.linalg.norm([vx, vy, vz])
                        
                        if steps % 50 == 0:
                             print(f"DEBUG Step {steps}: Alt={altitude:.2f}, Speed={airspeed:.2f}")
                             # Check target deltas in raw env
                             if hasattr(raw_env, "state") and "target_deltas" in raw_env.state:
                                 td = raw_env.state["target_deltas"]
                                 print(f"DEBUG Target Deltas Shape: {td.shape}")
                                 if td.shape[0] > 0:
                                     print(f"DEBUG Last Delta: {td[-1]}")
                    
                    # 获取推力 (action is [roll, pitch, yaw, thrust])
                    # We don't have easy access to the last action unless we store it or get it from env
                    # PyFlyt env usually stores last_cmd or similar, but let's just use the action we just sent
                    # action is a numpy array from model.predict
                    if isinstance(action, np.ndarray) and action.size >= 4:
                        # action usually normalized? In PyFlyt it depends on the env.
                        # Assuming [roll, pitch, yaw, thrust]
                        thrust = float(action[0][3]) if action.ndim > 1 else float(action[3])

                    if targets is not None:
                        # 简单的剩余目标计数：需要在 env 中追踪
                        # 这里我们只做简单估计，或者从 waypoints_env 获取
                        if waypoints_env and hasattr(waypoints_env, "waypoints"):
                            # waypoints.targets is list of targets.
                            # waypoints.target_index is current index
                            idx = getattr(waypoints_env.waypoints, "target_index", 0)
                            targets_remaining = len(targets) - idx
                            
                            # Calculate distance to current target
                            if idx < len(targets) and state12 is not None:
                                curr_target = targets[idx]
                                curr_pos = state12[9:12]
                                target_dist = np.linalg.norm(curr_target - curr_pos)
                    
                    overlay.update(altitude, airspeed, thrust, targets_remaining, target_dist)
                    
                except Exception as e:
                    # print(f"HUD Update Error: {e}")
                    pass

                # 稍微加点延时，避免画面太快（如果环境本身的 render 没有做时钟同步）
                # PyFlyt 通常不需要手动 sleep，这里仅作备用
                # time.sleep(0.01)
                pass

        # 记录本回合数据
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # 尝试从 info 中获取成功标志（PyFlyt 某些环境可能有，没有则忽略）
        # info 是一个 list（因为是 VecEnv）
        episode_info = info[0]
        # 假设如果有 success key
        # if "is_success" in episode_info and episode_info["is_success"]:
        #     success_count += 1
            
        print(f"Episode {i+1}: Reward = {total_reward:.2f}, Steps = {steps}")

    env.close()

    # 计算统计指标
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print("\nEvaluation Results:")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    # print(f"Success Rate: {success_count}/{args.episodes} ({success_count/args.episodes*100:.1f}%)")

if __name__ == "__main__":
    evaluate()
