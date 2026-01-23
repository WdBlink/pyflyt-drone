from __future__ import annotations

import os
import pkgutil
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
import pybullet_data
from gymnasium import spaces

from PyFlyt.core.abstractions.camera import Camera
# from PyFlyt.gym_envs.fixedwing_envs.fixedwing_base_env import FixedwingBaseEnv
from envs.fixedwing_envs.base.fixedwing_base_env import FixedwingBaseEnv
from envs.fixedwing_envs.base.fixedwing_vtail_base_env import FixedwingVtailBaseEnv


class FixedwingObjLockEnv(FixedwingVtailBaseEnv):
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
        camera_profile: Literal["cockpit_fpv", "chase", "default"] = "cockpit_fpv",
        camera_position_offset: tuple[float, float, float] | None = None,
        camera_angle_degrees: int | None = None,
        camera_FOV_degrees: int | None = None,
        camera_resolution: tuple[int, int] | None = None,
        # Obstacle Configs
        num_obstacles: int = 5,
        obstacle_radius: float = 2.0,
        obstacle_height_range: tuple[float, float] = (10.0, 30.0),
        obstacle_safe_distance_m: float = 20.0,
        obstacle_avoid_reward_scale: float = 1.0,
        obstacle_avoid_max_penalty: float = 2.0,
        # Duck Phase Configs
        duck_camera_capture_interval_steps: int = 12,
        duck_lock_hold_steps: int = 10,
        duck_strike_distance_m: float = 2.0,
        duck_strike_reward: float = 200.0,
        duck_lock_step_reward: float = 0.1,
        duck_approach_reward_scale: float = 0.05,
        duck_global_scaling: float = 20.0,
        duck_vision_history_len: int = 3,
        duck_vision_use_deltas: bool = True,
        # Visual Shaping Configs
        duck_distance_reward_scale: float = 1.0,
        duck_lock_center_radius: float = 0.55,
        duck_centering_reward_scale: float = 3,
        duck_visible_step_reward: float = 2,
        duck_area_reward_scale: float = 5.0,
        duck_lock_decay_steps: int = 1,
        duck_lock_lost_penalty: float = 0.5,
        duck_approach_reward_clip_m: float = 2.0,
        drone_model: str | None = None,
        drone_model_dir: str | None = None,
        wind_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 100.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
            wind_config=wind_config,
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
        self.duck_vision_history_len = int(max(1, duck_vision_history_len))
        self.duck_vision_use_deltas = bool(duck_vision_use_deltas)

        self.duck_distance_reward_scale = float(duck_distance_reward_scale)
        self.duck_lock_center_radius = float(duck_lock_center_radius)
        self.duck_centering_reward_scale = float(duck_centering_reward_scale)
        self.duck_visible_step_reward = float(duck_visible_step_reward)
        self.duck_area_reward_scale = float(duck_area_reward_scale)
        self.duck_lock_decay_steps = int(max(1, duck_lock_decay_steps))
        self.duck_lock_lost_penalty = float(duck_lock_lost_penalty)
        self.duck_approach_reward_clip_m = float(max(0.0, duck_approach_reward_clip_m))
        self._drone_model = None if drone_model is None else str(drone_model)
        self._drone_model_dir = None if drone_model_dir is None else str(drone_model_dir)

        self.duck_body_id: Optional[int] = None
        self._egl_plugin_id: Optional[int] = None
        self._camera_capture_patched = False

        self._camera_profile = camera_profile
        self._camera_position_offset = (
            np.asarray(camera_position_offset, dtype=np.float64)
            if camera_position_offset is not None
            else None
        )
        self._camera_angle_degrees = camera_angle_degrees
        self._camera_fov_degrees = camera_FOV_degrees
        self._camera_resolution = camera_resolution

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
        self._prev_area: Optional[float] = None
        self._vision_history = np.zeros(
            (self.duck_vision_history_len, 9), dtype=np.float32
        )
        self._vision_history_filled = 0
        self.duck_pos: Optional[np.ndarray] = None # Duck position in global frame

        # --- Observation Space ---
        # "attitude": [ang_vel, ang_pos, lin_vel, lin_pos, action, aux_state]
        # "target_vector": [dx, dy, dz] (Body frame vector to duck)
        # "duck_vision": [visible, cx, cy, area, depth, steps_norm, d_left, d_center, d_right]
        duck_vision_dim = 9 * self.duck_vision_history_len
        if self.duck_vision_use_deltas:
            duck_vision_dim += 4
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_vector": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "duck_vision": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(duck_vision_dim,), dtype=np.float32
                ),
            }
        )

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[dict, dict]:
        """Reset the environment."""
        # 1. Base Reset
        drone_options: dict[str, Any] = {"use_camera": True, "use_gimbal": False}
        if self._drone_model is not None:
            drone_options["drone_model"] = self._drone_model
        if self._drone_model_dir is not None:
            drone_options["model_dir"] = self._drone_model_dir

        if self._camera_profile == "cockpit_fpv":
            default_offset = np.array([0.8, 0.0, 0.12], dtype=np.float64)
            default_angle = -5
        elif self._camera_profile == "chase":
            default_offset = np.array([-3.0, 0.0, 1.0], dtype=np.float64)
            default_angle = 0
        else:
            default_offset = None
            default_angle = None

        offset = (
            self._camera_position_offset
            if self._camera_position_offset is not None
            else default_offset
        )
        if offset is not None:
            drone_options["camera_position_offset"] = offset

        angle = (
            int(self._camera_angle_degrees)
            if self._camera_angle_degrees is not None
            else default_angle
        )
        if angle is not None:
            drone_options["camera_angle_degrees"] = int(angle)

        fov = 90 if self._camera_fov_degrees is None else int(self._camera_fov_degrees)
        drone_options["camera_FOV_degrees"] = int(fov)

        if self._camera_resolution is not None:
            drone_options["camera_resolution"] = tuple(self._camera_resolution)
        else:
            drone_options["camera_resolution"] = (
                self.render_resolution if self.render_mode is not None else (128, 128)
            )

        super().begin_reset(seed, options, drone_options=drone_options)

        try:
            drone = self.env.drones[0]
            cam = getattr(drone, "camera", None)
            if cam is not None and hasattr(cam, "is_tracking_camera"):
                if self._camera_profile == "cockpit_fpv":
                    cam.is_tracking_camera = False
                elif self._camera_profile == "chase":
                    cam.is_tracking_camera = True
        except Exception:
            pass
        
        self.info["duck_strike"] = False
        self.info["env_complete"] = False
        self.info["is_success"] = False
        self.info["is_success"] = False
        self.info["is_success"] = False

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
        base_feature = self._compute_vision_features()
        new_state["duck_vision"] = self._build_duck_vision_observation(base_feature)

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
            # 1) Distance Reward (physics guidance)
            dist_to_duck = np.linalg.norm(self.state["target_vector"])
            self.reward += float(self.duck_distance_reward_scale) / max(float(dist_to_duck), 2.0)

            # 2) Visual Lock/Approach Reward (fast lock, short memory)
            vis = self.state.get("duck_vision") if isinstance(getattr(self, "state", None), dict) else None
            vis_arr: Optional[np.ndarray] = None
            if vis is not None:
                try:
                    vis_arr = np.asarray(vis, dtype=np.float32).reshape(-1)
                except Exception:
                    vis_arr = None

            visible = bool(vis_arr is not None and vis_arr.shape[0] >= 6 and float(vis_arr[0]) > 0.5)

            if visible:
                cx = float(vis_arr[1])
                cy = float(vis_arr[2])
                area = float(vis_arr[3])
                est_dist = float(vis_arr[4])

                self.reward += float(self.duck_visible_step_reward)
                self.reward += float(self.duck_area_reward_scale) * max(0.0, area)

                dist_to_center = float(np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2))
                r_lock = float(self.duck_lock_center_radius)
                r_lock = max(r_lock, 1e-6)

                center_score = max(0.0, (r_lock - dist_to_center) / r_lock)
                self.reward += float(self.duck_centering_reward_scale) * center_score

                if dist_to_center < r_lock:
                    self._lock_steps = min(self._lock_steps + 1, int(self.duck_lock_hold_steps))
                    self.reward += float(self.duck_lock_step_reward)
                else:
                    self._lock_steps = max(self._lock_steps - int(self.duck_lock_decay_steps), 0)

                if (
                    self._prev_est_dist_m is not None
                    and est_dist > 0.0
                    and np.isfinite(est_dist)
                ):
                    diff = float(self._prev_est_dist_m - est_dist)
                    clip_m = float(self.duck_approach_reward_clip_m)
                    if clip_m > 0.0:
                        diff = float(np.clip(diff, -clip_m, clip_m))
                    self.reward += diff * float(self.duck_approach_reward_scale)

                self._prev_est_dist_m = est_dist if est_dist > 0.0 and np.isfinite(est_dist) else None
                self._prev_area = area
            else:
                if self._lock_steps > 0:
                    self.reward -= float(self.duck_lock_lost_penalty)
                self._lock_steps = max(self._lock_steps - int(self.duck_lock_decay_steps), 0)
                self._prev_est_dist_m = None
                self._prev_area = None

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
            self.info["is_success"] = True

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
        self._prev_area = None
        self._vision_history[:] = 0.0
        self._vision_history_filled = 0

    def _build_duck_vision_observation(self, base_feature: np.ndarray) -> np.ndarray:
        base = np.asarray(base_feature, dtype=np.float32).reshape(-1)
        if base.shape[0] < 9:
            padded = np.zeros((9,), dtype=np.float32)
            n = min(int(base.shape[0]), 9)
            if n > 0:
                padded[:n] = base[:n]
            base = padded
        elif base.shape[0] > 9:
            base = base[:9]

        if self._vision_history.shape[0] >= 2:
            prev = self._vision_history[0].copy()
        else:
            prev = np.zeros((9,), dtype=np.float32)

        if self._vision_history.shape[0] > 1:
            self._vision_history[1:] = self._vision_history[:-1]
        self._vision_history[0] = base
        self._vision_history_filled = min(
            int(self._vision_history_filled) + 1, int(self._vision_history.shape[0])
        )

        history_flat = self._vision_history.reshape(-1)
        if not self.duck_vision_use_deltas:
            return history_flat.astype(np.float32)

        deltas = np.zeros((4,), dtype=np.float32)
        if (
            self._vision_history_filled >= 2
            and float(base[0]) > 0.5
            and float(prev[0]) > 0.5
        ):
            deltas[0] = float(base[1] - prev[1])
            deltas[1] = float(base[2] - prev[2])
            deltas[2] = float(base[3] - prev[3])
            deltas[3] = float(base[4] - prev[4])

        return np.concatenate([history_flat, deltas], axis=0).astype(np.float32)

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
                # print("DEBUG: Patched capture_image called")
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
                rgba = np.asarray(rgbaImg).reshape(h, w, -1)
                depth = np.asarray(depthImg).reshape(h, w, -1)
                seg = np.asarray(segImg).reshape(h, w, -1)
                
                # Update attributes so they are accessible via drone.rgbaImg etc.
                cam_self.rgbaImg = rgba
                cam_self.depthImg = depth
                cam_self.segImg = seg

                return rgba, depth, seg
            
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
