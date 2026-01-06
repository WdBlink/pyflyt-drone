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


class FixedwingObjLockEnv(FixedwingBaseEnv):
    """
    Fixedwing Object Lock Environment (No Waypoints).

    Actions are roll, pitch, yaw, thrust commands.
    The task is to locate and strike a yellow duck on the ground using visual cues.

    Args:
        sparse_reward (bool): whether to use sparse rewards.
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
        flight_mode: int = 0,
        flight_dome_size: float = 100.0,
        max_duration_seconds: float = 120.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        duck_urdf_path: str | None = None,
        use_egl: bool = False,
        # Obstacle Configs
        num_obstacles: int = 5,
        obstacle_radius: float = 2.0,
        obstacle_height_range: tuple[float, float] = (10.0, 30.0),
        obstacle_safe_distance_m: float = 20.0,
        obstacle_avoid_reward_scale: float = 1.0,
        obstacle_avoid_max_penalty: float = 2.0,
        # Duck Phase Configs
        duck_camera_capture_interval_steps: int = 6,
        duck_lock_hold_steps: int = 10,
        duck_strike_distance_m: float = 2.0,
        duck_strike_reward: float = 200.0,
        duck_lock_step_reward: float = 0.1,
        duck_approach_reward_scale: float = 0.05,
        duck_global_scaling: float = 20.0,
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

        self.sparse_reward = sparse_reward

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
        self.duck_global_scaling = duck_global_scaling

        self.duck_body_id: Optional[int] = None
        self._egl_plugin_id: Optional[int] = None
        self._camera_capture_patched = False
         
        # --- Obstacle Configs ---
        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.obstacle_height_range = obstacle_height_range
        self.obstacle_ids = []
        self.obstacle_safe_distance_m = obstacle_safe_distance_m
        self.obstacle_avoid_reward_scale = obstacle_avoid_reward_scale
        self.obstacle_avoid_max_penalty = obstacle_avoid_max_penalty
 
        # --- State Variables ---
        self._lock_steps = 0
        self._prev_est_dist_m: Optional[float] = None
        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._steps_since_seen = 60  # Default max
        self.duck_pos: Optional[np.ndarray] = None # Duck position in global frame

        # --- Observation Space ---
        # "attitude": [ang_vel, ang_pos, lin_vel, lin_pos, action, aux_state]
        # "target_vector": [dx, dy, dz] (Body frame vector to duck)
        # "duck_vision": [visible, cx, cy, area, depth, steps_norm, d_left, d_center, d_right]
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_vector": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
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
        
        self.info["duck_strike"] = False
        self.info["env_complete"] = False

        # 2. Duck & EGL Reset
        self._reset_duck_state()
        if self.use_egl:
            self._try_enable_egl()
            self._try_patch_camera_capture()
        self._spawn_duck()
        self._spawn_obstacles()
        
        # 3. Enable Camera Always
        self._set_auto_camera_capture_enabled(True)

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

        # --- Target Vector (Duck Relative Pos in Body Frame) ---
        if self.duck_body_id is not None and self.duck_pos is not None:
             # Global Delta
             diff = self.duck_pos - lin_pos
             # Rotate to Body Frame
             # Get Rotation Matrix (3x3) from Quaternion
             rot = np.array(self.env.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
             # Body = R.T * Global
             target_vector = rot.T @ diff
        else:
             target_vector = np.zeros(3, dtype=np.float64)
        
        new_state["target_vector"] = target_vector

        # --- Duck Vision Features ---
        feature = self._compute_vision_features()
        new_state["duck_vision"] = feature

        self.state = new_state

    def compute_term_trunc_reward(self) -> None:
        """Computes the termination, truncation, and reward."""
        super().compute_base_term_trunc_reward()

        if self.info.get("collision") or self.info.get("out_of_bounds"):
            return

        # --- Obstacle Avoidance ---
        self._apply_obstacle_avoidance_reward()
            
        # --- Dense Reward ---
        if not self.sparse_reward:
            # 1. Distance Reward (Guide towards duck)
            # Use real distance (from physics) or visual depth?
            # Physics distance is more reliable for "normal flight" guidance.
            dist_to_duck = np.linalg.norm(self.state["target_vector"])
            self.reward += 1.0 / max(dist_to_duck, 2.0)

            # 2. Lock Reward (Visual)
            if self._last_cx > 0.0: # Visible
                dist_to_center = np.sqrt((self._last_cx - 0.5)**2 + (self._last_cy - 0.5)**2)
                if dist_to_center < 0.35: # Lock Center Radius
                    self._lock_steps += 1
                    self.reward += self.duck_lock_step_reward
                else:
                    self._lock_steps = 0
            else:
                self._lock_steps = 0
            
            # 3. Approach Reward (Differential visual depth)
            est_dist = self._last_depth_m
            if self._prev_est_dist_m is not None and est_dist > 0:
                diff = self._prev_est_dist_m - est_dist
                if diff > 0:
                    self.reward += diff * self.duck_approach_reward_scale
            self._prev_est_dist_m = est_dist

        # --- Sparse Reward (Strike) ---
        # Check strike condition
        # We use physics distance for strike check usually, or visual depth if close enough
        dist_to_duck = np.linalg.norm(self.state["target_vector"])
        
        # Strike logic: Locked enough time AND close enough
        # Relaxed logic: Just close enough? User said "通过相机锁定目标并撞击".
        # Let's keep the lock requirement to encourage visual tracking.
        if (self._lock_steps >= self.duck_lock_hold_steps and 
            dist_to_duck <= self.duck_strike_distance_m):
            self.termination = True
            self.reward += self.duck_strike_reward
            self.info["env_complete"] = True
            self.info["duck_strike"] = True

    # --- Helper Methods ---

    def _apply_obstacle_avoidance_reward(self) -> None:
        if not isinstance(getattr(self, "state", None), dict):
            return

        vis = self.state.get("duck_vision")
        if vis is None:
            return

        vis = np.asarray(vis, dtype=np.float32).reshape(-1)
        if vis.shape[0] < 9:
            return

        d_left = float(vis[6])
        d_center = float(vis[7])
        d_right = float(vis[8])
        depths = [d for d in (d_left, d_center, d_right) if d > 0.0 and np.isfinite(d)]
        if not depths:
            return

        d_obs = min(depths)
        d_safe = float(self.obstacle_safe_distance_m)
        if d_safe <= 0.0:
            return

        if d_obs >= d_safe:
            return

        scale = float(self.obstacle_avoid_reward_scale) * 0.5 # Reduced scale for duck phase (always on)

        penalty = scale * (d_safe - d_obs) / d_safe
        penalty = min(penalty, float(self.obstacle_avoid_max_penalty))
        self.reward -= penalty

    def _reset_duck_state(self):
        self._lock_steps = 0
        self._prev_est_dist_m = None
        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._steps_since_seen = 60

    def _spawn_duck(self):
        """Spawns duck with manual contact array resize to avoid IndexError."""
        if self.duck_body_id is not None:
            try:
                existing = {self.env.getBodyUniqueId(i) for i in range(self.env.getNumBodies())}
                if self.duck_body_id in existing:
                    self.env.removeBody(self.duck_body_id)
            except Exception:
                pass
            self.duck_body_id = None

        # Random Position Logic
        rng = self.np_random
        
        # Spawn within flight dome, but on ground
        r = self.flight_dome_size / 2.0
        x = rng.uniform(-r, r)
        y = rng.uniform(-r, r)
        z = 0.05 # Ground

        self.duck_pos = np.array([x, y, z])

        quat = self.env.getQuaternionFromEuler([np.pi / 2, 0.0, rng.uniform(-np.pi, np.pi)])
        
        self.duck_body_id = self.env.loadURDF(
            self.duck_urdf_path,
            basePosition=[x, y, z],
            baseOrientation=quat,
            useFixedBase=True,
            globalScaling=self.duck_global_scaling,
        )

        # Manual Contact Array Resize
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

    def _spawn_obstacles(self):
        """Spawns cylindrical obstacles."""
        try:
            existing = {self.env.getBodyUniqueId(i) for i in range(self.env.getNumBodies())}
        except Exception:
            existing = set()

        for uid in self.obstacle_ids:
            try:
                if uid in existing:
                    self.env.removeBody(uid)
            except Exception:
                pass
        self.obstacle_ids = []

        rng = self.np_random
        min_h = float(self.obstacle_height_range[0])
        max_h = float(self.obstacle_height_range[1])
        if max_h < min_h:
            min_h, max_h = max_h, min_h

        for _ in range(int(self.num_obstacles)):
            h = float(rng.uniform(min_h, max_h))

            x = float(rng.uniform(-self.flight_dome_size / 2, self.flight_dome_size / 2))
            y = float(rng.uniform(-self.flight_dome_size / 2, self.flight_dome_size / 2))
            z = float(h / 2.0)

            # Avoid spawning near duck
            if self.duck_pos is not None:
                d_to_duck = np.linalg.norm(np.array([x, y]) - self.duck_pos[:2])
                if d_to_duck < 10.0:
                    continue

            # Avoid spawning near center (start pos)
            if x * x + y * y < 100.0:
                continue

            try:
                col_id = self.env.createCollisionShape(
                    self.env.GEOM_CYLINDER,
                    radius=float(self.obstacle_radius),
                    height=h,
                )
                vis_id = self.env.createVisualShape(
                    self.env.GEOM_CYLINDER,
                    radius=float(self.obstacle_radius),
                    length=h,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )
                body_id = self.env.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[x, y, z],
                )
                self.obstacle_ids.append(body_id)
            except Exception:
                pass

        try:
            max_id = -1
            for i in range(self.env.getNumBodies()):
                uid = self.env.getBodyUniqueId(i)
                if uid > max_id:
                    max_id = uid
            required = max_id + 1
            if self.env.contact_array.shape[0] < required:
                new_size = required + 50
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

            obs_d_left, obs_d_center, obs_d_right = self._estimate_obstacle_zone_distances_m(
                depth=depth, seg=seg
            )

            mask = (seg == self.duck_body_id)
            if not np.any(mask):
                self._steps_since_seen = min(self._steps_since_seen + 1, 60)
                return self._build_vision_vector(
                    0.0,
                    obstacle_depths=(obs_d_left, obs_d_center, obs_d_right),
                )
            
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
                return self._build_vision_vector(
                    1.0,
                    obstacle_depths=(obs_d_left, obs_d_center, obs_d_right),
                )
            
        except Exception:
            pass
        
        return self._build_vision_vector(0.0)

    def _depth_buffer_to_meters(self, depth_buffer: float) -> float:
        near, far = 0.1, 255.0
        denom = (far - (far - near) * depth_buffer)
        if abs(denom) < 1e-9:
            return float(far)
        return float((far * near) / denom)

    def _estimate_obstacle_zone_distances_m(
        self, depth: np.ndarray, seg: np.ndarray
    ) -> tuple[float, float, float]:
        duck_id = int(self.duck_body_id) if self.duck_body_id is not None else None
        seg_int = seg.astype(np.int64)
        if duck_id is None:
            mask = np.ones_like(seg_int, dtype=bool)
        else:
            mask = seg_int != duck_id

        h, w = mask.shape
        y_mid = int(h // 2)
        x_1 = int(w // 3)
        x_2 = int(2 * w // 3)

        def zone_mean_depth_buffer(x_start: int, x_end: int) -> float:
            zone_mask = mask[y_mid, x_start:x_end]
            if not np.any(zone_mask):
                return 0.0
            vals = depth[y_mid, x_start:x_end][zone_mask]
            if vals.size == 0:
                return 0.0
            return float(np.mean(vals))

        d_left_buf = zone_mean_depth_buffer(0, x_1)
        d_center_buf = zone_mean_depth_buffer(x_1, x_2)
        d_right_buf = zone_mean_depth_buffer(x_2, w)

        d_left = self._depth_buffer_to_meters(d_left_buf) if d_left_buf > 0.0 else 0.0
        d_center = self._depth_buffer_to_meters(d_center_buf) if d_center_buf > 0.0 else 0.0
        d_right = self._depth_buffer_to_meters(d_right_buf) if d_right_buf > 0.0 else 0.0
        return d_left, d_center, d_right

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

    def _build_vision_vector(
        self, visible: float, obstacle_depths: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ) -> np.ndarray:
        # [visible, cx, cy, area, depth, steps_norm, d_left, d_center, d_right]
        steps_norm = float(self._steps_since_seen) / 60.0
        
        d_left, d_center, d_right = obstacle_depths

        return np.array([
            visible,
            self._last_cx,
            self._last_cy,
            self._last_area,
            self._last_depth_m,
            steps_norm,
            d_left, d_center, d_right
        ], dtype=np.float32)
