"""项目文件：Fixedwing-ObjLock 模型评估脚本
说明:
    加载已训练的 PPO 模型和 VecNormalize 统计量，评估模型在 FixedwingObjLockEnv 环境中的表现。
"""

import os
import sys
import time
import argparse
from typing import Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure imports work
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.fixedwing_objlock_env import FixedwingObjLockEnv
from envs.flatten_objlock_env import FlattenObjLockEnv
import PyFlyt.gym_envs

# 默认评估配置
EVAL_CONFIG = {
    "model_path": "models/obj_lock_only_ppo_v1.3_hist/best_model.zip",
    "vecnorm_path": None, # Auto-detect
    "num_episodes": 10,
    "flight_dome_size": 200.0,
    "max_duration_seconds": 120.0,
    
    # Duck Configs
    "duck_camera_capture_interval_steps": 24,
    "duck_lock_hold_steps": 5,
    "duck_strike_distance_m": 10.0,
    "duck_global_scaling": 40.0,
    
    # Obstacle Configs
    "num_obstacles": 0,
    "obstacle_radius": 2.0,
    "obstacle_height_range": (10.0, 30.0),
    "obstacle_safe_distance_m": 10.0,

    # Evaluation Configs
    "camera_profile": "cockpit_fpv",
    "wind": {
        "enabled": True,
        "mode": "gust_sine", # "gust_sine" or "constant"
        "wind_enu_mps": [5.0, 0.0, 0.0], #风速向量，坐标系是 ENU（East, North, Up），单位 m/s
        "gust_amp_enu_mps": [2.0, 0.0, 0.0],
        "gust_freq_hz": 0.2,
        "gust_phase_rad": 0.0,
        "randomize_on_reset": False,
        "randomize_gust_phase": False,
    },
}

def _infer_vecnorm_path(model_path: str, vecnorm_path: str | None) -> str | None:
    if vecnorm_path:
        return vecnorm_path
    model_dir = os.path.dirname(model_path)
    candidates = [
        os.path.join(model_dir, "vecnorm.pkl"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_single_env(env):
    if hasattr(env, "venv"):
        env = env.venv
    if hasattr(env, "envs"):
        env = env.envs[0]
    return env


def _unwrap_env(env, max_depth: int = 32):
    cur = env
    for _ in range(max_depth):
        if hasattr(cur, "env"):
            cur = cur.env
        elif hasattr(cur, "base_env"):
            cur = cur.base_env
        else:
            break
    return cur


def _try_get_drone(env):
    base = _unwrap_env(_resolve_single_env(env))
    
    # If base is Aviary
    if hasattr(base, "drones"):
        return base.drones[0]
        
    # If base is Env that wraps Aviary in .env (and _unwrap_env didn't catch it for some reason, 
    # or stopped at Env because Env didn't look like a wrapper?)
    # But _unwrap_env checks hasattr(cur, "env").
    
    aviary = getattr(base, "env", None)
    if aviary is not None and hasattr(aviary, "drones"):
        return aviary.drones[0]
        
    return None


def _reshape_if_flat(arr: np.ndarray, height: int, width: int) -> np.ndarray:
    if arr.ndim == 1:
        c = int(arr.size // (height * width))
        if c <= 0:
            return arr
        return arr.reshape(height, width, c)
    return arr


def _capture_rgb_depth_seg(env):
    drone = _try_get_drone(env)
    if drone is None:
        return None, None, None

    cam = getattr(drone, "camera", None)
    rgb = getattr(drone, "rgbImg", None)
    rgba = getattr(drone, "rgbaImg", None)
    depth = getattr(drone, "depthImg", None)
    seg = getattr(drone, "segImg", None)

    if rgb is None and rgba is None and cam is not None:
        rgb = getattr(cam, "rgbImg", None)
        rgba = getattr(cam, "rgbaImg", None)
        depth = getattr(cam, "depthImg", None) if depth is None else depth
        seg = getattr(cam, "segImg", None) if seg is None else seg

    if rgb is not None:
        rgb = np.asarray(rgb)
    elif rgba is not None:
        rgba = np.asarray(rgba)
        rgb = rgba[..., :3] if rgba.ndim >= 3 and rgba.shape[-1] >= 3 else None

    if depth is not None:
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

    if seg is not None:
        seg = np.asarray(seg)
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[..., 0]

    if rgb is not None:
        if cam is not None and hasattr(cam, "camera_resolution"):
            h = int(cam.camera_resolution[0])
            w = int(cam.camera_resolution[1])
            rgb = _reshape_if_flat(rgb, h, w)
            if rgb.ndim == 3 and rgb.shape[-1] > 3:
                rgb = rgb[..., :3]

    return rgb, depth, seg


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    if np.nanmax(x) <= 1.0:
        x = x * 255.0
    x = np.clip(x, 0.0, 255.0)
    return x.astype(np.uint8)


def _save_rgb(path: str, rgb: np.ndarray) -> bool:
    try:
        from PIL import Image

        Image.fromarray(_to_uint8_rgb(rgb)).save(path)
        return True
    except Exception:
        return False

def make_eval_env(render_mode="human"):
    """
    创建评估环境
    """
    env = FixedwingObjLockEnv(
        sparse_reward=False, # Doesn't matter for eval, but keep consistent
        render_mode=render_mode,
        angle_representation="euler",
        flight_dome_size=float(EVAL_CONFIG["flight_dome_size"]),
        max_duration_seconds=float(EVAL_CONFIG["max_duration_seconds"]),
        agent_hz=30,
        camera_profile=str(EVAL_CONFIG.get("camera_profile", "cockpit_fpv")),
        wind_config=EVAL_CONFIG.get("wind", None),
        
        # Obstacles
        num_obstacles=EVAL_CONFIG["num_obstacles"],
        obstacle_radius=EVAL_CONFIG["obstacle_radius"],
        obstacle_height_range=EVAL_CONFIG["obstacle_height_range"],
        obstacle_safe_distance_m=EVAL_CONFIG["obstacle_safe_distance_m"],
        
        # Duck Configs
        duck_camera_capture_interval_steps=EVAL_CONFIG["duck_camera_capture_interval_steps"],
        duck_lock_hold_steps=EVAL_CONFIG["duck_lock_hold_steps"],
        duck_strike_distance_m=EVAL_CONFIG["duck_strike_distance_m"],
        duck_global_scaling=EVAL_CONFIG["duck_global_scaling"],
    )
    
    # 扁平化观测空间
    env = FlattenObjLockEnv(env)
    return env

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate PPO model for Fixedwing ObjLock")
    parser.add_argument("--model", type=str, default=EVAL_CONFIG["model_path"], help="Path to the model file")
    parser.add_argument("--vecnorm", type=str, default=None, help="Path to the vecnorm file")
    parser.add_argument("--episodes", type=int, default=EVAL_CONFIG["num_episodes"], help="Number of episodes to evaluate")
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument("--save-frames", action="store_true", help="Save a few onboard-camera frames during evaluation")
    parser.add_argument("--frames-outdir", type=str, default="eval_frames/objlock", help="Output directory for saved frames")
    parser.add_argument("--frames-interval", type=int, default=30, help="Save every N steps")
    parser.add_argument("--frames-max-per-episode", type=int, default=20, help="Max frames saved per episode")
    parser.add_argument("--save-depth-seg", action="store_true", help="Also save depth/seg as .npy")
    args = parser.parse_args()

    model_path = args.model
    vecnorm_path = _infer_vecnorm_path(model_path, args.vecnorm)
    render_mode = None if args.no_render else "human"
    
    print(f"Loading model from: {model_path}")
    print(f"Loading vecnorm from: {vecnorm_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # 创建 DummyVecEnv
    env = DummyVecEnv([lambda: make_eval_env(render_mode)])
    
    # 加载 VecNormalize
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: vecnorm.pkl not found, running without VecNormalize.")

    # 加载模型
    model = PPO.load(model_path, env=env)
    
    print(f"Starting evaluation for {args.episodes} episodes...")
    
    total_rewards = []
    success_count = 0
    
    if args.save_frames:
        os.makedirs(args.frames_outdir, exist_ok=True)

    for i in range(args.episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        saved_frames = 0

        episode_dir = None
        if args.save_frames:
            episode_dir = os.path.join(args.frames_outdir, f"ep_{i:03d}")
            os.makedirs(episode_dir, exist_ok=True)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1

            if (
                args.save_frames
                and saved_frames < args.frames_max_per_episode
                and args.frames_interval > 0
                and (step_count % args.frames_interval) == 0
            ):
                rgb, depth, seg = _capture_rgb_depth_seg(env)
                if rgb is not None and rgb.size > 0:
                    rgb_path = os.path.join(episode_dir, f"step_{step_count:06d}.png")
                    ok = _save_rgb(rgb_path, rgb)
                    if not ok:
                        np.save(rgb_path.replace(".png", ".npy"), rgb)

                    if args.save_depth_seg:
                        if depth is not None:
                            np.save(
                                os.path.join(episode_dir, f"step_{step_count:06d}_depth.npy"),
                                depth,
                            )
                        if seg is not None:
                            np.save(
                                os.path.join(episode_dir, f"step_{step_count:06d}_seg.npy"),
                                seg,
                            )

                    saved_frames += 1
            # 稍微加点延时，避免画面太快（如果环境本身的 render 没有做时钟同步）
            # PyFlyt 通常不需要手动 sleep，这里仅作备用
            time.sleep(0.08)

        info_dict = info[0]
        is_success = info_dict.get("duck_strike", False)
        if is_success:
            success_count += 1

        total_rewards.append(episode_reward)
        print(
            f"Episode {i+1}: Reward={episode_reward:.2f}, Steps={step_count}, Success={is_success}"
        )
        
    mean_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    success_rate = success_count / args.episodes
    
    print("-" * 30)
    print(f"Evaluation Complete")
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"Success Rate: {success_rate*100:.1f}%")
    
    env.close()

if __name__ == "__main__":
    evaluate()
