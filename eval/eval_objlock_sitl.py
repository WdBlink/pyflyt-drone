"""项目文件：Ardupilot SITL ObjLock 模型评估脚本
说明:
    使用 ArdupilotGazeboObjLockEnv（ROS/MAVROS）获取飞行状态和相机图像，
    加载 PPO 模型进行评估，并通过 MAVROS 输出控制指令到 SITL。
"""

import os
import sys
import time
import argparse
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure imports work
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from envs.ardupilot_ros_objlock_env import ArdupilotGazeboObjLockEnv
from envs.flatten_objlock_env import FlattenObjLockEnv

# 默认评估配置（SITL）
EVAL_CONFIG = {
    "model_path": "models/obj_lock_only_ppo_v2.3_hist/best_model.zip",
    "vecnorm_path": None,  # Auto-detect
    "num_episodes": 3,
    "max_duration_seconds": 120.0,
    "agent_hz": 30,
    # ROS topics (defaults match run_vtail_with_ros_bridge.sh)
    "rgb_image_topic": "/pod_camera/image_raw",
    "depth_image_topic": "/pod_depth_camera/image_raw",
    # MAVROS topics
    "mavros_imu_topic": "/mavros/imu/data",
    "mavros_odom_topic": "/mavros/local_position/odom",
    "mavros_vel_topic": "/mavros/local_position/velocity_local",
    "mavros_rc_override_topic": "/mavros/rc/override",
    "mavros_rc_out_topic": "/mavros/rc/out",
    # Vision backend
    "vision_backend": "fastsam",
    "fastsam_text_prompt": "a photo of a yellow duck",
    # Use GPU for FastSAM if available (falls back to CPU inside the env if CUDA isn't usable).
    "fastsam_device": "cuda",
    # Optional pre-vision goal pursuit. Set x/y/z to enable.
    "goal_x": None,
    "goal_y": None,
    "goal_z": None,
    "goal_vector_in_body_frame": True,
    "goal_switch_on_vision": True,
    "goal_visible_hold_steps": 3,
    "goal_reacquire_steps": 15,
}


def _unwrap_env_for_debug(vec_env) -> tuple[Any, Any]:
    """
    Returns (flatten_env, base_env).
    flatten_env: FlattenObjLockEnv
    base_env: ArdupilotGazeboObjLockEnv
    """
    v = vec_env.venv if hasattr(vec_env, "venv") else vec_env
    env0 = v.envs[0]
    base_env = env0.env if hasattr(env0, "env") else env0
    return env0, base_env


def _fmt_arr(x, n: int = 3) -> str:
    if x is None:
        return "None"
    a = np.asarray(x, dtype=np.float32).reshape(-1)
    if a.size == 0:
        return "[]"
    vals = ", ".join(f"{v:.3f}" for v in a[:n])
    if a.size > n:
        vals += ", ..."
    return f"[{vals}]"


def _infer_vecnorm_path(model_path: str, vecnorm_path: str | None) -> str | None:
    if vecnorm_path:
        return vecnorm_path
    model_dir = os.path.dirname(model_path)
    candidate = os.path.join(model_dir, "vecnorm.pkl")
    return candidate if os.path.exists(candidate) else None


def make_eval_env():
    goal_pos = None
    if (
        EVAL_CONFIG.get("goal_x") is not None
        and EVAL_CONFIG.get("goal_y") is not None
        and EVAL_CONFIG.get("goal_z") is not None
    ):
        goal_pos = (
            float(EVAL_CONFIG["goal_x"]),
            float(EVAL_CONFIG["goal_y"]),
            float(EVAL_CONFIG["goal_z"]),
        )

    env = ArdupilotGazeboObjLockEnv(
        angle_representation="euler",
        agent_hz=int(EVAL_CONFIG["agent_hz"]),
        max_duration_seconds=float(EVAL_CONFIG["max_duration_seconds"]),
        rgb_image_topic=str(EVAL_CONFIG["rgb_image_topic"]),
        depth_image_topic=str(EVAL_CONFIG["depth_image_topic"]),
        mavros_imu_topic=str(EVAL_CONFIG["mavros_imu_topic"]),
        mavros_odom_topic=str(EVAL_CONFIG["mavros_odom_topic"]),
        mavros_vel_topic=str(EVAL_CONFIG["mavros_vel_topic"]),
        mavros_rc_override_topic=str(EVAL_CONFIG["mavros_rc_override_topic"]),
        mavros_rc_out_topic=str(EVAL_CONFIG["mavros_rc_out_topic"]),
        vision_backend=str(EVAL_CONFIG["vision_backend"]),
        fastsam_text_prompt=str(EVAL_CONFIG.get("fastsam_text_prompt", "")) or None,
        fastsam_device=str(EVAL_CONFIG.get("fastsam_device", "cuda")),
        enable_rc_override=True,
        goal_position_enu=goal_pos,
        goal_vector_in_body_frame=bool(EVAL_CONFIG.get("goal_vector_in_body_frame", True)),
        goal_switch_on_vision=bool(EVAL_CONFIG.get("goal_switch_on_vision", True)),
        goal_visible_hold_steps=int(EVAL_CONFIG.get("goal_visible_hold_steps", 3)),
        goal_reacquire_steps=int(EVAL_CONFIG.get("goal_reacquire_steps", 15)),
    )
    env = FlattenObjLockEnv(env)
    return env


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate PPO model for ObjLock (SITL)")
    parser.add_argument("--model", type=str, default=EVAL_CONFIG["model_path"])
    parser.add_argument("--vecnorm", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=EVAL_CONFIG["num_episodes"])
    parser.add_argument("--no-rc", action="store_true", help="Disable RC override publishing")
    parser.add_argument("--vision-backend", type=str, default=EVAL_CONFIG["vision_backend"])
    parser.add_argument(
        "--fastsam-prompt",
        type=str,
        default=EVAL_CONFIG["fastsam_text_prompt"],
        help="FastSAM text prompt, e.g. 'a photo of a yellow duck'",
    )
    parser.add_argument("--rgb-topic", type=str, default=EVAL_CONFIG["rgb_image_topic"])
    parser.add_argument("--depth-topic", type=str, default=EVAL_CONFIG["depth_image_topic"])
    parser.add_argument("--goal-x", type=float, default=EVAL_CONFIG["goal_x"])
    parser.add_argument("--goal-y", type=float, default=EVAL_CONFIG["goal_y"])
    parser.add_argument("--goal-z", type=float, default=EVAL_CONFIG["goal_z"])
    parser.add_argument(
        "--goal-world-frame",
        action="store_true",
        help="Use world-frame goal vector instead of body-frame",
    )
    parser.add_argument(
        "--no-goal-switch-on-vision",
        action="store_true",
        help="Keep goal vector even when vision target is visible",
    )
    parser.add_argument("--goal-visible-hold-steps", type=int, default=EVAL_CONFIG["goal_visible_hold_steps"])
    parser.add_argument("--goal-reacquire-steps", type=int, default=EVAL_CONFIG["goal_reacquire_steps"])
    parser.add_argument(
        "--log-every",
        type=int,
        default=30,
        help="Print runtime obs/action/debug every N steps (0 disables periodic logs)",
    )
    args = parser.parse_args()

    model_path = args.model
    vecnorm_path = _infer_vecnorm_path(model_path, args.vecnorm)

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Update config from CLI
    EVAL_CONFIG["vision_backend"] = args.vision_backend
    EVAL_CONFIG["fastsam_text_prompt"] = args.fastsam_prompt
    EVAL_CONFIG["rgb_image_topic"] = args.rgb_topic
    EVAL_CONFIG["depth_image_topic"] = args.depth_topic
    EVAL_CONFIG["goal_x"] = args.goal_x
    EVAL_CONFIG["goal_y"] = args.goal_y
    EVAL_CONFIG["goal_z"] = args.goal_z
    EVAL_CONFIG["goal_vector_in_body_frame"] = not bool(args.goal_world_frame)
    EVAL_CONFIG["goal_switch_on_vision"] = not bool(args.no_goal_switch_on_vision)
    EVAL_CONFIG["goal_visible_hold_steps"] = int(max(1, args.goal_visible_hold_steps))
    EVAL_CONFIG["goal_reacquire_steps"] = int(max(1, args.goal_reacquire_steps))

    env = DummyVecEnv([lambda: make_eval_env()])

    # Optional VecNormalize
    if vecnorm_path is not None and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("Warning: vecnorm.pkl not found, running without VecNormalize.")

    # Disable RC override if requested
    if args.no_rc:
        _, base_env_for_rc = _unwrap_env_for_debug(env)
        if hasattr(base_env_for_rc, "_enable_rc_override"):
            base_env_for_rc._enable_rc_override = False

    model = PPO.load(model_path, env=env)
    flat_env, base_env = _unwrap_env_for_debug(env)
    print(
        "Runtime config:",
        f"model={model_path}",
        f"vecnorm={'on' if vecnorm_path and os.path.exists(vecnorm_path) else 'off'}",
        f"vision={EVAL_CONFIG['vision_backend']}",
        f"prompt={EVAL_CONFIG['fastsam_text_prompt']!r}",
        f"rc_override={'off' if args.no_rc else 'on'}",
        f"goal=({EVAL_CONFIG['goal_x']}, {EVAL_CONFIG['goal_y']}, {EVAL_CONFIG['goal_z']})",
        f"goal_frame={'body' if EVAL_CONFIG['goal_vector_in_body_frame'] else 'world'}",
        f"switch_on_vision={'on' if EVAL_CONFIG['goal_switch_on_vision'] else 'off'}",
    )
    print(
        "Obs layout:",
        f"attitude={getattr(flat_env, 'attitude_shape', '?')}",
        f"target_vector={getattr(flat_env, 'target_shape', '?')}",
        f"duck_vision={getattr(flat_env, 'vision_shape', '?')}",
    )
    print(f"Starting SITL evaluation for {args.episodes} episodes...")

    total_rewards = []
    for i in range(args.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        step_count = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += float(reward[0])
            step_count += 1

            if args.log_every > 0 and (step_count == 1 or step_count % args.log_every == 0):
                info0 = info[0] if isinstance(info, (list, tuple)) and info else info
                snap = base_env.get_debug_snapshot() if hasattr(base_env, "get_debug_snapshot") else {}
                act0 = action[0] if isinstance(action, np.ndarray) and action.ndim > 1 else action
                obs0 = obs[0] if isinstance(obs, np.ndarray) and obs.ndim > 1 else obs

                tdim = int(getattr(flat_env, "attitude_shape", 0))
                vdim = int(getattr(flat_env, "vision_shape", 0))
                target_vec = None
                duck_vis = None
                if isinstance(obs0, np.ndarray) and obs0.ndim == 1 and tdim > 0:
                    target_vec = obs0[tdim : tdim + 3]
                    if vdim > 0:
                        duck_vis = obs0[-vdim:]

                print(
                    f"[ep {i+1:02d} step {step_count:05d}] "
                    f"connected={bool(snap.get('mavros_connected', False))} "
                    f"have_rgb={bool(snap.get('have_rgb', False))} "
                    f"have_depth={bool(snap.get('have_depth', False))} "
                    f"have_imu={bool(info0.get('have_imu', False) if isinstance(info0, dict) else False)} "
                    f"have_pos={bool(info0.get('have_pos', False) if isinstance(info0, dict) else False)} "
                    f"have_servo={bool(info0.get('have_servo', False) if isinstance(info0, dict) else False)} "
                    f"mode={str(snap.get('tracking_mode', 'n/a'))} "
                    f"goal_d={float(snap.get('goal_distance_m', 0.0)):.2f} "
                    f"action={_fmt_arr(act0, 4)} "
                    f"lin_pos={_fmt_arr(snap.get('lin_pos'), 3)} "
                    f"lin_vel={_fmt_arr(snap.get('lin_vel'), 3)} "
                    f"aux={_fmt_arr(snap.get('aux_state'), 6)} "
                    f"target_vec={_fmt_arr(target_vec, 3)} "
                    f"duck_vis={_fmt_arr(duck_vis, 9)}"
                )
            time.sleep(0.01)

        total_rewards.append(ep_reward)
        print(f"Episode {i+1}: Reward={ep_reward:.2f}, Steps={step_count}")

    if total_rewards:
        print(f"Mean Reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")

    env.close()


if __name__ == "__main__":
    evaluate()
