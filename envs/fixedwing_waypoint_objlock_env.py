from __future__ import annotations

import os
import pkgutil
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
import pybullet_data
from gymnasium import spaces

from PyFlyt.core.abstractions.camera import Camera
from PyFlyt.gym_envs.fixedwing_envs.fixedwing_base_env import FixedwingBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class FixedwingWaypointObjLockEnv(FixedwingBaseEnv):
    """
    Fixedwing Waypoints + Object Lock Environment.

    Actions are roll, pitch, yaw, thrust commands.
    The task consists of two phases:
    1. Waypoint Phase: Reach a sequence of aerial waypoints.
    2. Duck Phase: After the last waypoint, locate and strike a yellow duck on the ground using visual cues.

    Args:
        sparse_reward (bool): whether to use sparse rewards.
        num_targets (int): number of waypoints.
        goal_reach_distance (float): distance to waypoint to be considered reached.
        flight_mode (int): UAV flight mode.
        flight_dome_size (float): allowable flying area size.
        max_duration_seconds (float): max simulation time.
        angle_representation (Literal["euler", "quaternion"]): "euler" or "quaternion".
        agent_hz (int): looprate.
        render_mode (None | Literal["human", "rgb_array"]): render mode.
        render_resolution (tuple[int, int]): render resolution.
        duck_urdf_path (str | None): path to duck URDF. If None, uses default from pybullet_data.
        use_egl (bool): whether to attempt EGL hardware rendering (for headless GPU).
    """

    def __init__(
        self,
        sparse_reward: bool = False,
        num_targets: int = 4,
        goal_reach_distance: float = 2.0,
        flight_mode: int = 0,
        flight_dome_size: float = 100.0,
        max_duration_seconds: float = 120.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        duck_urdf_path: str | None = None,
        use_egl: bool = False,
        # Duck Phase Configs
        duck_camera_capture_interval_steps: int = 6,
        duck_lock_hold_steps: int = 10,
        duck_strike_distance_m: float = 2.0,
        duck_strike_reward: float = 200.0,
        duck_lock_step_reward: float = 0.1,
        duck_approach_reward_scale: float = 0.05,
        duck_switch_min_consecutive_seen: int = 2,
        duck_switch_min_area: float = 0.0005,
    ):
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 10.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # --- Waypoint Configs ---
        self.num_targets = num_targets
        self.sparse_reward = sparse_reward
        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=False,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=np.inf,
            flight_dome_size=flight_dome_size,
            min_height=0.5,
            np_random=self.np_random,
        )

        # --- Duck Phase Configs ---
        self.duck_urdf_path = duck_urdf_path or os.path.join(
            pybullet_data.getDataPath(), "duck_vhacd.urdf"
        )
        self.use_egl = use_egl
        self.duck_camera_capture_interval_steps = duck_camera_capture_interval_steps
        self.duck_lock_hold_steps = duck_lock_hold_steps
        self.duck_strike_distance_m = duck_strike_distance_m
        self.duck_strike_reward = duck_strike_reward
        self.duck_lock_step_reward = duck_lock_step_reward
        self.duck_approach_reward_scale = duck_approach_reward_scale
        self.duck_switch_min_consecutive_seen = duck_switch_min_consecutive_seen
        self.duck_switch_min_area = duck_switch_min_area

        self.duck_body_id: Optional[int] = None
        self._egl_plugin_id: Optional[int] = None
        self._camera_capture_patched = False

        # --- State Variables ---
        self._duck_phase = False
        self._seen_consecutive = 0
        self._lock_steps = 0
        self._prev_est_dist_m: Optional[float] = None
        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._steps_since_seen = 60  # Default max
        self._post_waypoints = False  # True when all waypoints are done

        # --- Observation Space ---
        # "attitude": [ang_vel, ang_pos, lin_vel, lin_pos, action, aux_state]
        # "target_deltas": Sequence of [dx, dy, dz]
        # "duck_vision": [visible, cx, cy, area, depth, steps_norm, d_left, d_center, d_right]
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_deltas": spaces.Sequence(
                    space=spaces.Box(
                        low=-2 * flight_dome_size,
                        high=2 * flight_dome_size,
                        shape=(3,),
                        dtype=np.float64,
                    ),
                    stack=True,
                ),
                "duck_vision": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
                ),
            }
        )

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict, dict]:
        """Reset the environment."""
        # 1. Base Reset
        super().begin_reset(seed, options)
        
        # 2. Waypoint Reset
        # 注意：Waypoints 需要 p (bullet client) 实例，而 FixedwingBaseEnv.env 就是 Aviary 实例 (即 bullet client)
        self.waypoints.reset(self.env, self.np_random)
        self.info["num_targets_reached"] = 0
        self.info["duck_strike"] = False

        # 3. Duck & EGL Reset
        self._reset_duck_phase_state()
        if self.use_egl:
            self._try_enable_egl()
            self._try_patch_camera_capture()
        self._spawn_duck()
        
        # 4. Disable Auto Camera (Save Performance)
        self._set_auto_camera_capture_enabled(False)

        super().end_reset()
        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep."""
        # --- Base Attitude ---
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        new_state: dict = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [ang_vel, ang_pos, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.concatenate(
                [ang_vel, quaternion, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )

        # --- Target Deltas ---
        # 修复广播错误：PyFlyt 的 WaypointHandler.distance_to_targets 在 targets 为空时可能返回异常形状
        # 或者 targets 未正确初始化。但在 reset 后 targets 应该是有值的。
        # 这里加上保护逻辑。
        # 注意：targets 有时可能是 list，需要转 numpy 检查形状
        has_targets = False
        if hasattr(self.waypoints, "targets"):
             targets = np.asarray(self.waypoints.targets)
             if targets.ndim == 2 and targets.shape[0] > 0:
                 has_targets = True
        
        if has_targets:
             new_state["target_deltas"] = self.waypoints.distance_to_targets(
                ang_pos, lin_pos, quaternion
            )
        else:
             # Fallback shape (N, 3)
             new_state["target_deltas"] = np.zeros((self.num_targets, 3), dtype=np.float64)

        # --- Duck Vision Features ---
        # Logic: 
        # If Waypoints not done -> Vision is zeros, camera disabled.
        # If Waypoints done -> Camera enabled, compute features.
        
        if self.waypoints.all_targets_reached:
            if not self._post_waypoints:
                self._post_waypoints = True
                self._set_auto_camera_capture_enabled(True)

            feature = self._compute_vision_features()
            
            # Check visibility for phase switching
            if not self._duck_phase:
                visible = bool(feature[0] > 0.5 and self._last_area >= self.duck_switch_min_area)
                if visible:
                    self._seen_consecutive += 1
                else:
                    self._seen_consecutive = 0
                
                if self._seen_consecutive >= self.duck_switch_min_consecutive_seen:
                    self._duck_phase = True
                    # print("DEBUG: Switched to Duck Phase!")

            new_state["duck_vision"] = feature
        else:
            self._post_waypoints = False
            self._duck_phase = False
            self._set_auto_camera_capture_enabled(False)
            new_state["duck_vision"] = np.zeros(9, dtype=np.float32)

        self.state = new_state

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward."""
        super().compute_base_term_trunc_reward()

        # --- Waypoint Phase Reward ---
        if not self.waypoints.all_targets_reached:
            if not self.sparse_reward:
                self.reward += max(3.0 * self.waypoints.progress_to_next_target, 0.0)
                self.reward += 1.0 / self.waypoints.distance_to_next_target

            if self.waypoints.target_reached:
                self.reward = 100.0
                self.waypoints.advance_targets()
                self.info["num_targets_reached"] = self.waypoints.num_targets_reached
        
        # --- Duck Phase Reward ---
        else:
            # Override truncation from waypoint completion
            self.truncation = False 
            
            if self._duck_phase:
                # 1. Lock Reward
                if self._last_cx > 0.0: # Visible
                    dist_to_center = np.sqrt((self._last_cx - 0.5)**2 + (self._last_cy - 0.5)**2)
                    if dist_to_center < 0.35: # Lock Center Radius
                        self._lock_steps += 1
                        self.reward += self.duck_lock_step_reward
                    else:
                        self._lock_steps = 0
                else:
                    self._lock_steps = 0
                
                # 2. Approach Reward
                est_dist = self._last_depth_m
                if self._prev_est_dist_m is not None and est_dist > 0:
                    diff = self._prev_est_dist_m - est_dist
                    if diff > 0:
                        self.reward += diff * self.duck_approach_reward_scale
                self._prev_est_dist_m = est_dist

                # 3. Strike Success
                if (self._lock_steps >= self.duck_lock_hold_steps and 
                    0 < est_dist <= self.duck_strike_distance_m):
                    self.termination = True
                    self.reward += self.duck_strike_reward
                    self.info["env_complete"] = True
                    self.info["duck_strike"] = True
                    # print("DEBUG: Duck Struck!")

    # --- Helper Methods ---

    def _reset_duck_phase_state(self):
        self._duck_phase = False
        self._seen_consecutive = 0
        self._lock_steps = 0
        self._prev_est_dist_m = None
        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._steps_since_seen = 60
        self._post_waypoints = False

    def _spawn_duck(self):
        """Spawns duck with manual contact array resize to avoid IndexError."""
        if self.duck_body_id is not None:
            try:
                self.env.removeBody(self.duck_body_id)
            except Exception:
                pass
            self.duck_body_id = None

        # Random Position Logic
        rng = self.np_random
        # Try to place near last waypoint if available, else random
        try:
            # self.waypoints.targets 是一个 (N, 3) 的 numpy array
            # 确保我们取的是最后一行，并且转换为 Python float
            if hasattr(self.waypoints, "targets") and len(self.waypoints.targets) > 0:
                last = self.waypoints.targets[-1]
                x, y = float(last[0]), float(last[1])
                z = float(last[2])
            else:
                x, y, z = 10.0, 0.0, 0.03
        except Exception:
             # Fallback
            x, y, z = 10.0, 0.0, 0.03

        # Add some noise
        # x += rng.uniform(-2.0, 2.0)
        # y += rng.uniform(-2.0, 2.0)
        
        quat = self.env.getQuaternionFromEuler([np.pi / 2, 0.0, rng.uniform(-np.pi, np.pi)])
        
        self.duck_body_id = self.env.loadURDF(
            self.duck_urdf_path,
            basePosition=[x, y, z],
            baseOrientation=quat,
            useFixedBase=True,
            globalScaling=10.0,
        )

        # Manual Contact Array Resize (Critical Fix)
        try:
            max_id = -1
            for i in range(self.env.getNumBodies()):
                uid = self.env.getBodyUniqueId(i)
                if uid > max_id:
                    max_id = uid
            required = max_id + 1
            if self.env.contact_array.shape[0] < required:
                new_size = required + 5
                self.env.contact_array = np.zeros((new_size, new_size), dtype=bool)
        except Exception:
            pass

    def _try_enable_egl(self):
        # EGL Loading Logic
        try:
            if self._egl_plugin_id is not None:
                return
            
            egl = pkgutil.get_loader("eglRenderer")
            plugin = egl.get_filename() if egl else "eglRendererPlugin"
            self._egl_plugin_id = self.env.loadPlugin(plugin, "_eglRendererPlugin")
        except Exception:
            self._egl_plugin_id = None

    def _try_patch_camera_capture(self):
        if self._camera_capture_patched:
            return
        
        try:
            import pybullet as p
            original_capture = Camera.capture_image
            
            def capture_image(cam_self):
                try:
                    _, _, rgbaImg, depthImg, segImg = cam_self.p.getCameraImage(
                        height=int(cam_self.camera_resolution[0]),
                        width=int(cam_self.camera_resolution[1]),
                        viewMatrix=cam_self.view_mat,
                        projectionMatrix=cam_self.proj_mat,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    )
                except Exception:
                    return original_capture(cam_self)
                
                # Reshape logic
                h, w = int(cam_self.camera_resolution[0]), int(cam_self.camera_resolution[1])
                return (
                    np.asarray(rgbaImg).reshape(h, w, -1),
                    np.asarray(depthImg).reshape(h, w, -1),
                    np.asarray(segImg).reshape(h, w, -1)
                )

            Camera.capture_image = capture_image
            self._camera_capture_patched = True
        except Exception:
            pass

    def _set_auto_camera_capture_enabled(self, enabled: bool):
        # Optimization: Disable camera when not needed
        try:
            drone = self.env.drones[0]
            if enabled:
                control_ratio = int(getattr(drone, "physics_control_ratio", 8))
                drone.physics_camera_ratio = control_ratio * self.duck_camera_capture_interval_steps
            else:
                drone.physics_camera_ratio = int(10**9)
        except Exception:
            pass

    def _compute_vision_features(self) -> np.ndarray:
        # Simplified feature extraction from drone's latest images
        try:
            drone = self.env.drones[0]
            seg = getattr(drone, "segImg", None)
            depth = getattr(drone, "depthImg", None)
            
            if seg is None or depth is None:
                return self._build_vision_vector(0.0)

            seg = np.asarray(seg)
            depth = np.asarray(depth)
            if seg.ndim == 3: seg = seg[..., 0]
            if depth.ndim == 3: depth = depth[..., 0]

            # Find Duck
            mask = (seg == self.duck_body_id)
            if not np.any(mask):
                 self._steps_since_seen = min(self._steps_since_seen + 1, 60)
                 return self._build_vision_vector(0.0, depth)
            
            # Found
            h, w = mask.shape
            ys, xs = np.nonzero(mask)
            self._last_cx = float(np.mean(xs)) / float(max(1, w - 1))
            self._last_cy = float(np.mean(ys)) / float(max(1, h - 1))
            self._last_area = float(np.count_nonzero(mask)) / float(max(1, h * w))
            
            depth_m = self._estimate_distance(depth, mask)
            if depth_m is not None:
                self._last_depth_m = float(depth_m)
                self._steps_since_seen = 0
                return self._build_vision_vector(1.0, depth)
            
        except Exception:
            pass
        
        return self._build_vision_vector(0.0)

    def _estimate_distance(self, depth, mask):
        # Depth buffer to meters conversion
        # PyBullet default: near=0.1, far=100.0 (or what Camera uses)
        # Using default values from typical config
        near, far = 0.1, 255.0 # Check Camera class defaults
        d_val = depth[mask]
        if len(d_val) == 0: return None
        d_min = np.min(d_val)
        
        denom = (far - (far - near) * d_min)
        if abs(denom) < 1e-9: return far
        z = (far * near) / denom
        return z

    def _build_vision_vector(self, visible: float, depth: Optional[np.ndarray] = None) -> np.ndarray:
        # [visible, cx, cy, area, depth, steps_norm, d_left, d_center, d_right]
        steps_norm = float(self._steps_since_seen) / 60.0
        
        d_left, d_center, d_right = 0.0, 0.0, 0.0
        if depth is not None:
             # Simple 3-zone depth
             h, w = depth.shape
             y_mid = h // 2
             x_1 = w // 3
             x_2 = 2 * w // 3
             # Use raw buffer for speed, rough proxy
             d_left = np.mean(depth[y_mid, :x_1])
             d_center = np.mean(depth[y_mid, x_1:x_2])
             d_right = np.mean(depth[y_mid, x_2:])

        return np.array([
            visible,
            self._last_cx,
            self._last_cy,
            self._last_area,
            self._last_depth_m,
            steps_norm,
            d_left, d_center, d_right
        ], dtype=np.float32)
