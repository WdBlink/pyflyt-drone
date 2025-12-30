"""项目文件：环境包装器集合

作者: wdblink

说明:
    提供用于 PyFlyt Gymnasium 环境的任务包装器：
    - `RandomDuckOnResetWrapper`: 在每次 reset 时生成小黄鸭目标物。
    - `WaypointThenDuckStrikeWrapper`: 先完成航点任务，完成后进入“基于相机锁定并撞击小黄鸭”的终局阶段。
"""

from __future__ import annotations

from typing import Any, Optional

import pkgutil

import gymnasium as gym
import numpy as np


class EnableEGLRenderOnResetWrapper(gym.Wrapper):
    _camera_capture_patched: bool = False

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._last_client_id: Optional[int] = None
        self._egl_plugin_id: Optional[int] = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._try_enable_egl()
        self._try_patch_camera_capture()
        return obs, info

    def _try_enable_egl(self) -> None:
        base = self.env.unwrapped
        aviary = getattr(base, "env", None)
        if aviary is None:
            return

        client_id = getattr(aviary, "_client", None)
        if client_id is None:
            return

        client_id = int(client_id)
        if self._last_client_id == client_id and self._egl_plugin_id is not None:
            return

        self._last_client_id = client_id
        self._egl_plugin_id = None

        try:
            # print(f"DEBUG: Loading EGL plugin for client {client_id}...")
            egl = pkgutil.get_loader("eglRenderer")
            if egl is not None:
                self._egl_plugin_id = int(
                    aviary.loadPlugin(egl.get_filename(), "_eglRendererPlugin")
                )
            else:
                self._egl_plugin_id = int(
                    aviary.loadPlugin("eglRendererPlugin", "_eglRendererPlugin")
                )
            # print(f"DEBUG: EGL plugin loaded, id={self._egl_plugin_id}")
        except Exception as e:
            # print(f"DEBUG: EGL load failed: {e}")
            self._egl_plugin_id = None

    @classmethod
    def _try_patch_camera_capture(cls) -> None:
        if cls._camera_capture_patched:
            return

        try:
            import pybullet as p
            from PyFlyt.core.abstractions.camera import Camera

            original_capture = Camera.capture_image

            def capture_image(self):
                try:
                    # print("DEBUG: calling getCameraImage with HARDWARE_OPENGL")
                    _, _, rgbaImg, depthImg, segImg = self.p.getCameraImage(
                        height=int(self.camera_resolution[0]),
                        width=int(self.camera_resolution[1]),
                        viewMatrix=self.view_mat,
                        projectionMatrix=self.proj_mat,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL,
                    )
                    # print("DEBUG: getCameraImage success")
                except Exception:
                    # print("DEBUG: getCameraImage failed, fallback")
                    return original_capture(self)

                rgbaImg = np.asarray(rgbaImg).reshape(
                    int(self.camera_resolution[0]), int(self.camera_resolution[1]), -1
                )
                depthImg = np.asarray(depthImg).reshape(
                    int(self.camera_resolution[0]), int(self.camera_resolution[1]), -1
                )
                segImg = np.asarray(segImg).reshape(
                    int(self.camera_resolution[0]), int(self.camera_resolution[1]), -1
                )
                return rgbaImg, depthImg, segImg

            Camera.capture_image = capture_image
            cls._camera_capture_patched = True
        except Exception:
            cls._camera_capture_patched = False


class RandomDuckOnResetWrapper(gym.Wrapper):
    """在每次 reset 时生成一个小黄鸭。"""

    def __init__(
        self,
        env: gym.Env,
        *,
        urdf_path: str,
        xy_radius: float,
        min_origin_distance: float = 5.0,
        base_z: float = 0.02,
        global_scaling: float = 0.7,
        place_at_last_waypoint: bool = False,
        use_waypoint_altitude: bool = True,
    ) -> None:
        """初始化包装器。

        Args:
            env: 被包装环境。
            urdf_path: URDF 路径。
            xy_radius: 随机生成的平面范围半径。
            min_origin_distance: 距离原点的最小半径，避免生成在起飞点附近。
            base_z: 生成高度（不使用航点高度时生效）。
            global_scaling: 模型缩放。
            place_at_last_waypoint: 若为 True，优先放置到最后一个航点附近。
            use_waypoint_altitude: place_at_last_waypoint=True 时，使用最后航点的 z 作为高度。
        """
        super().__init__(env)
        self.urdf_path = str(urdf_path)
        self.xy_radius = float(xy_radius)
        self.min_origin_distance = float(min_origin_distance)
        self.base_z = float(base_z)
        self.global_scaling = float(global_scaling)
        self.place_at_last_waypoint = bool(place_at_last_waypoint)
        self.use_waypoint_altitude = bool(use_waypoint_altitude)
        self.duck_body_id: Optional[int] = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置环境，并重新加载/移动小黄鸭。"""
        obs, info = self.env.reset(seed=seed, options=options)

        base = self.env.unwrapped
        aviary = getattr(base, "env", None)
        if aviary is None:
            return obs, info

        # 1. 尝试查找当前 aviary 里是否已经有我们的小黄鸭 (通过 body name 验证)
        # 因为 PyFlyt 每次 reset 都会重建 Aviary，所以这里通常是需要重新加载的
        # 但如果未来 PyFlyt 优化了复用逻辑，这里的 check 会很有用
        found_existing = False
        if self.duck_body_id is not None:
            try:
                # 检查 body id 是否有效且名字匹配
                body_info = aviary.getBodyInfo(int(self.duck_body_id))
                # body_info: (base_name, body_name) 都是 bytes
                if b"duck" in body_info[1].lower():
                    found_existing = True
            except Exception:
                self.duck_body_id = None

        rng = getattr(base, "_np_random", None)
        if rng is None:
            rng = np.random.default_rng()

        x, y, z = self._sample_duck_xyz(base, rng)
        yaw = float(rng.uniform(-np.pi, np.pi))
        quat = aviary.getQuaternionFromEuler([np.pi / 2, 0.0, yaw])
        target_pos = [float(x), float(y), float(z)]

        if found_existing and self.duck_body_id is not None:
            # 2. 如果已存在，直接移动位置 (更快)
            aviary.resetBasePositionAndOrientation(
                int(self.duck_body_id),
                target_pos,
                quat
            )
        else:
            # 3. 如果不存在，加载新的
            try:
                self.duck_body_id = int(
                    aviary.loadURDF(
                        self.urdf_path,
                        basePosition=target_pos,
                        baseOrientation=quat,
                        useFixedBase=True,
                        globalScaling=self.global_scaling,
                    )
                )
                # 必须通知 aviary 有新物体加入，否则 contact_array 不会扩容
                aviary.register_all_new_bodies()
            except Exception:
                self.duck_body_id = None

        return obs, info

        # 手动扩容 contact_array 避免调用 register_all_new_bodies 可能导致的死锁/性能问题
        try:
            # 找到当前最大的 body unique id
            max_id = -1
            for i in range(aviary.getNumBodies()):
                uid = aviary.getBodyUniqueId(i)
                if uid > max_id:
                    max_id = uid
            
            # 如果 contact_array 不够大，则扩容
            required_size = max_id + 1
            if aviary.contact_array.shape[0] < required_size:
                # print(f"DEBUG: Resizing contact_array from {aviary.contact_array.shape[0]} to {required_size}")
                new_size = required_size + 5 # 稍微多留点余量
                new_array = np.zeros((new_size, new_size), dtype=bool)
                # 不需要复制旧数据，因为 Aviary.step() 开头会重置它
                aviary.contact_array = new_array
        except Exception:
            pass

    def _sample_duck_xyz(self, base: Any, rng: Any) -> tuple[float, float, float]:
        """采样小黄鸭的位置。"""
        if self.place_at_last_waypoint and hasattr(base, "waypoints"):
            wp = getattr(base, "waypoints", None)
            if wp is not None and hasattr(wp, "targets"):
                try:
                    targets = np.asarray(wp.targets, dtype=np.float64)
                    last = targets[-1]
                    x = float(last[0])
                    y = float(last[1])
                    z = float(last[2]) if self.use_waypoint_altitude else float(self.base_z)
                    return x, y, z
                except Exception:
                    pass

        for _ in range(50):
            x = float(rng.uniform(-self.xy_radius, self.xy_radius))
            y = float(rng.uniform(-self.xy_radius, self.xy_radius))
            if float(np.hypot(x, y)) >= self.min_origin_distance:
                return x, y, float(self.base_z)
        return float(self.min_origin_distance), 0.0, float(self.base_z)


class WaypointThenDuckStrikeWrapper(gym.Wrapper):
    """完成所有航点后，进入“相机锁定并撞击小黄鸭”的终局阶段。

    约束:
        在撞鸭阶段不读取鸭子与飞机的真实位置，仅使用相机输出（分割图/深度图）估计距离并引导撞击。
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        num_targets_total: int,
        camera_capture_interval_steps: int = 6,
        lock_hold_steps: int = 10,
        strike_distance_m: float = 2.0,
        strike_reward: float = 200.0,
        lock_step_reward: float = 0.1,
        approach_reward_scale: float = 0.05,
        seg_min_fraction: float = 0.001,
        lock_center_radius: float = 0.35,
        depth_near: float = 0.1,
        depth_far: float = 255.0,
        disable_auto_camera_capture: bool = True,
    ) -> None:
        """初始化包装器。

        Args:
            env: 被包装环境。
            num_targets_total: 航点总数，达到后切换到撞鸭阶段。
            camera_capture_interval_steps: 撞鸭阶段相机捕获的步数间隔。
            lock_hold_steps: 连续锁定步数阈值。
            strike_distance_m: 撞击成功距离阈值（由深度图估计）。
            strike_reward: 撞击成功终局奖励。
            lock_step_reward: 锁定期间每步奖励。
            approach_reward_scale: 接近奖励系数（基于估计距离差分）。
            seg_min_fraction: 目标在分割图中的最小像素占比阈值。
            lock_center_radius: 目标质心距离画面中心的归一化半径阈值。
            depth_near: 相机近平面，用于深度 buffer 转距离。
            depth_far: 相机远平面，用于深度 buffer 转距离。
            disable_auto_camera_capture: 若为 True，尝试关闭 PyFlyt 的自动相机捕获，改为按需采样。
        """
        super().__init__(env)
        self._num_targets_total = int(num_targets_total)
        self._camera_capture_interval_steps = max(1, int(camera_capture_interval_steps))
        self._lock_hold_steps = int(lock_hold_steps)
        self._strike_distance_m = float(strike_distance_m)
        self._strike_reward = float(strike_reward)
        self._lock_step_reward = float(lock_step_reward)
        self._approach_reward_scale = float(approach_reward_scale)
        self._seg_min_fraction = float(seg_min_fraction)
        self._lock_center_radius = float(lock_center_radius)
        self._depth_near = float(depth_near)
        self._depth_far = float(depth_far)
        self._disable_auto_camera_capture = bool(disable_auto_camera_capture)

        self._duck_phase = False
        self._duck_phase_step = 0
        self._lock_steps = 0
        self._prev_est_dist_m: Optional[float] = None
        self._cached_locked = False
        self._cached_est_dist_m: Optional[float] = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置内部阶段状态。"""
        self._duck_phase = False
        self._duck_phase_step = 0
        self._lock_steps = 0
        self._prev_est_dist_m = None
        self._cached_locked = False
        self._cached_est_dist_m = None

        obs, info = self.env.reset(seed=seed, options=options)
        if self._disable_auto_camera_capture:
            self._set_auto_camera_capture_enabled(False)
        return obs, info

    def step(self, action: Any):
        """执行一步，并在撞鸭阶段注入基于相机的奖励与成功终止条件。"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not isinstance(info, dict):
            return obs, reward, terminated, truncated, info

        reached_all = self._is_waypoints_complete(info)
        if reached_all and not self._duck_phase:
            self._duck_phase = True
            self._duck_phase_step = 0
            self._lock_steps = 0
            self._prev_est_dist_m = None
            self._cached_locked = False
            self._cached_est_dist_m = None
            info["duck_phase"] = True

        if reached_all and (terminated or truncated) and not info.get("out_of_bounds", False) and not info.get("collision", False):
            terminated = False
            truncated = False

        if self._duck_phase:
            self._duck_phase_step += 1
            shaped_reward, strike_success = self._compute_duck_objective_reward()
            reward = float(reward) + float(shaped_reward)
            if strike_success:
                terminated = True
                truncated = False
                info["duck_strike"] = True
                info["is_success"] = True

        return obs, reward, terminated, truncated, info

    def _is_waypoints_complete(self, info: dict) -> bool:
        """判断是否已完成所有航点。"""
        num_targets_reached = info.get("num_targets_reached")
        if num_targets_reached is None:
            return bool(info.get("env_complete", False))
        return int(num_targets_reached) >= self._num_targets_total

    def _compute_duck_objective_reward(self) -> tuple[float, bool]:
        """计算撞鸭阶段 shaping 回报，并判断是否成功。

        说明:
            - 观测侧在撞鸭阶段可以是纯视觉输入，但奖励/终止允许使用仿真真值(特权信息)，以便更稳定训练。
            - 锁定判定仅作为辅助 shaping 信号，不要求持续严格锁定，从而兼容遮挡场景。
        """
        base = self.env.unwrapped
        aviary = getattr(base, "env", None)
        if aviary is None or not hasattr(aviary, "drones") or not aviary.drones:
            return 0.0, False

        duck_id = self._find_duck_body_id()
        if duck_id is None:
            return 0.0, False

        aircraft_id = int(getattr(aviary.drones[0], "Id"))

        gt_dist_m: Optional[float] = None
        try:
            duck_pos, _ = aviary.getBasePositionAndOrientation(int(duck_id))
            air_pos, _ = aviary.getBasePositionAndOrientation(int(aircraft_id))
            duck_pos = np.asarray(duck_pos, dtype=np.float64)
            air_pos = np.asarray(air_pos, dtype=np.float64)
            gt_dist_m = float(np.linalg.norm(duck_pos - air_pos))
        except Exception:
            gt_dist_m = None

        locked, _ = self._estimate_lock_and_distance_m(int(duck_id))
        if locked:
            self._lock_steps = min(self._lock_steps + 1, self._lock_hold_steps)
            lock_active = True
        else:
            self._lock_steps = max(self._lock_steps - 1, 0)
            lock_active = self._lock_steps > 0

        reward = 0.0
        if lock_active:
            reward += self._lock_step_reward

        if lock_active and gt_dist_m is not None and self._prev_est_dist_m is not None:
            reward += self._approach_reward_scale * (self._prev_est_dist_m - gt_dist_m)
        if gt_dist_m is not None:
            self._prev_est_dist_m = gt_dist_m

        contact_success = False
        try:
            contacts = aviary.getContactPoints(bodyA=int(aircraft_id), bodyB=int(duck_id))
            contact_success = bool(contacts)
        except Exception:
            contact_success = False

        strike_success = contact_success or (gt_dist_m is not None and gt_dist_m <= self._strike_distance_m)
        if strike_success:
            reward += self._strike_reward

        return reward, strike_success

    def _estimate_lock_and_distance_m(self, duck_id: int) -> tuple[bool, Optional[float]]:
        """基于相机分割图与深度图估计锁定状态与距离。"""
        if (self._duck_phase_step % self._camera_capture_interval_steps) != 0:
            return self._cached_locked, self._cached_est_dist_m

        seg, depth = self._capture_seg_and_depth()
        if seg is None or depth is None:
            self._cached_locked = False
            self._cached_est_dist_m = None
            return False, None

        mask = self._duck_mask_from_seg(seg, duck_id)
        if mask is None or not np.any(mask):
            self._cached_locked = False
            self._cached_est_dist_m = None
            return False, None

        h, w = int(mask.shape[0]), int(mask.shape[1])
        frac = float(np.count_nonzero(mask)) / float(max(1, h * w))
        if frac < self._seg_min_fraction:
            self._cached_locked = False
            self._cached_est_dist_m = None
            return False, None

        ys, xs = np.nonzero(mask)
        cy = float(np.mean(ys)) / float(max(1, h - 1))
        cx = float(np.mean(xs)) / float(max(1, w - 1))
        center_dist = float(np.hypot(cx - 0.5, cy - 0.5))
        locked = center_dist <= self._lock_center_radius

        est_dist_m = self._estimate_distance_from_depth(depth, mask)
        self._cached_locked = bool(locked and est_dist_m is not None)
        self._cached_est_dist_m = est_dist_m
        return self._cached_locked, self._cached_est_dist_m

    def _capture_seg_and_depth(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """从 PyFlyt 相机捕获分割图与深度图。"""
        base = self.env.unwrapped
        aviary = getattr(base, "env", None)
        if aviary is None or not hasattr(aviary, "drones") or not aviary.drones:
            return None, None

        drone = aviary.drones[0]
        cam = getattr(drone, "camera", None)
        if cam is None or not hasattr(cam, "capture_image"):
            return None, None

        try:
            _, depth, seg = cam.capture_image()
            seg = np.asarray(seg)
            depth = np.asarray(depth)
            if seg.ndim == 3 and seg.shape[-1] == 1:
                seg = seg[..., 0]
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            return seg, depth
        except Exception:
            return None, None

    def _duck_mask_from_seg(self, seg: np.ndarray, duck_id: int) -> Optional[np.ndarray]:
        """从 PyBullet segmentation buffer 解析出 duck 的像素 mask。"""
        try:
            seg64 = seg.astype(np.int64, copy=False)
        except Exception:
            return None

        direct = seg64 == int(duck_id)
        if np.any(direct):
            return direct

        high8 = (seg64 >> 24) == int(duck_id)
        if np.any(high8):
            return high8

        low24 = (seg64 & 0xFFFFFF) == int(duck_id)
        if np.any(low24):
            return low24

        return None

    def _estimate_distance_from_depth(self, depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """将深度 buffer 转换为距离估计（米），返回目标像素的中位数距离。"""
        depth_buf = depth.astype(np.float64, copy=False)
        vals = depth_buf[mask]
        if vals.size == 0:
            return None

        near = self._depth_near
        far = self._depth_far
        denom = (far - (far - near) * vals)
        denom = np.maximum(denom, 1e-9)
        z = (far * near) / denom
        z = z[np.isfinite(z)]
        if z.size == 0:
            return None
        return float(np.median(z))

    def _find_duck_body_id(self) -> Optional[int]:
        """在 wrapper 链中查找小黄鸭的 body id。"""
        cur: Any = self.env
        for _ in range(32):
            if hasattr(cur, "get_duck_body_id"):
                duck_id = cur.get_duck_body_id()
                return int(duck_id) if duck_id is not None else None
            if hasattr(cur, "duck_body_id"):
                duck_id = getattr(cur, "duck_body_id")
                return int(duck_id) if duck_id is not None else None
            if hasattr(cur, "env"):
                cur = cur.env
            else:
                break
        return None

    def _set_auto_camera_capture_enabled(self, enabled: bool) -> None:
        """启用/禁用 PyFlyt 固定翼默认的自动相机捕获。"""
        base = self.env.unwrapped
        aviary = getattr(base, "env", None)
        if aviary is None or not hasattr(aviary, "drones") or not aviary.drones:
            return
        drone = aviary.drones[0]
        if not hasattr(drone, "use_camera") or not getattr(drone, "use_camera"):
            return
        if not hasattr(drone, "physics_camera_ratio"):
            return

        try:
            if enabled:
                control_ratio = int(getattr(drone, "physics_control_ratio", 8))
                drone.physics_camera_ratio = control_ratio * self._camera_capture_interval_steps
            else:
                drone.physics_camera_ratio = int(10**9)
        except Exception:
            return


class WaypointThenDuckVisionObsWrapper(gym.Wrapper):
    """航点阶段保持原观测不变，撞鸭阶段切换为纯视觉特征向量观测。

    状态机:
        1) 航点阶段: 不改观测，关闭相机捕获以节省算力。
        2) 搜索/对准阶段: 已完成所有航点，但尚未在分割图中检测到小黄鸭；仍保留原观测，开启相机捕获。
        3) 撞鸭阶段: 在分割图中检测到小黄鸭(可配置连续帧数/最小面积)后，观测切换为纯视觉特征向量。

    说明:
        这样可以避免一完成航点就立刻切换为纯视觉观测，导致短时间内无法找到小黄鸭而任务失败。
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        num_targets_total: int,
        camera_capture_interval_steps: int = 6,
        depth_near: float = 0.1,
        depth_far: float = 255.0,
        max_steps_since_seen: int = 60,
        switch_min_consecutive_seen: int = 2,
        switch_min_area: float = 0.0005,
    ) -> None:
        """初始化包装器。

        Args:
            env: 被包装环境，要求观测为一维向量(Box)以保持航点阶段不变。
            num_targets_total: 航点总数。
            camera_capture_interval_steps: 搜索/撞鸭阶段相机捕获的步数间隔。
            depth_near: 相机近平面。
            depth_far: 相机远平面。
            max_steps_since_seen: 目标最近出现步数的归一化上限。
            switch_min_consecutive_seen: 从搜索阶段切换到撞鸭阶段所需的连续检测帧数。
            switch_min_area: 认为检测到目标的最小像素占比阈值。
        """
        super().__init__(env)
        self._num_targets_total = int(num_targets_total)
        self._camera_capture_interval_steps = max(1, int(camera_capture_interval_steps))
        self._depth_near = float(depth_near)
        self._depth_far = float(depth_far)
        self._max_steps_since_seen = int(max_steps_since_seen)
        self._switch_min_consecutive_seen = max(1, int(switch_min_consecutive_seen))
        self._switch_min_area = float(switch_min_area)

        self._post_waypoints = False
        self._seen_consecutive = 0

        self._duck_phase = False
        self._duck_phase_step = 0
        self._obs_dim: Optional[int] = None

        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._steps_since_seen = self._max_steps_since_seen

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置阶段状态。"""
        self._post_waypoints = False
        self._seen_consecutive = 0

        self._duck_phase = False
        self._duck_phase_step = 0

        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._steps_since_seen = self._max_steps_since_seen

        obs, info = self.env.reset(seed=seed, options=options)
        try:
            self._obs_dim = int(np.asarray(obs).shape[0])
        except Exception:
            self._obs_dim = None

        self._set_auto_camera_capture_enabled(False)
        return obs, info

    def step(self, action: Any):
        """执行一步，并按阶段切换观测。"""
        obs, reward, terminated, truncated, info = self.env.step(action)

        if not isinstance(info, dict):
            return obs, reward, terminated, truncated, info

        if self._is_waypoints_complete(info) and not self._post_waypoints:
            self._post_waypoints = True
            self._set_auto_camera_capture_enabled(True)

        if self._post_waypoints and not self._duck_phase:
            feature = self._compute_vision_features()
            visible = bool(feature[0] > 0.5 and self._last_area >= self._switch_min_area)
            if visible:
                self._seen_consecutive += 1
            else:
                self._seen_consecutive = 0

            if self._seen_consecutive >= self._switch_min_consecutive_seen:
                self._duck_phase = True
                self._duck_phase_step = 0

        if self._duck_phase:
            self._duck_phase_step += 1
            obs = self._build_duck_vision_obs(obs)

        return obs, reward, terminated, truncated, info

    def _is_waypoints_complete(self, info: dict) -> bool:
        """判断是否已完成所有航点。"""
        num_targets_reached = info.get("num_targets_reached")
        if num_targets_reached is None:
            return bool(info.get("env_complete", False))
        return int(num_targets_reached) >= self._num_targets_total

    def _build_duck_vision_obs(self, fallback_obs: Any) -> np.ndarray:
        """构造撞鸭阶段的纯视觉特征向量。"""
        obs_dim = self._obs_dim
        if obs_dim is None:
            try:
                obs_dim = int(np.asarray(fallback_obs).shape[0])
            except Exception:
                obs_dim = 0

        feature = self._compute_vision_features()
        out = np.zeros((obs_dim,), dtype=np.float32)
        n = min(out.shape[0], feature.shape[0])
        if n > 0:
            out[:n] = feature[:n]
        return out

    def _compute_vision_features(self) -> np.ndarray:
        """从相机分割/深度图中提取低维特征，支持遮挡时的短期记忆。"""
        seg, depth = self._read_latest_seg_and_depth()
        visible = 0.0

        if seg is not None and depth is not None:
            duck_id = self._find_duck_body_id()
            if duck_id is not None:
                mask = self._duck_mask_from_seg(seg, int(duck_id))
                if mask is not None and np.any(mask):
                    h, w = int(mask.shape[0]), int(mask.shape[1])
                    area = float(np.count_nonzero(mask)) / float(max(1, h * w))
                    ys, xs = np.nonzero(mask)
                    cy = float(np.mean(ys)) / float(max(1, h - 1))
                    cx = float(np.mean(xs)) / float(max(1, w - 1))
                    depth_m = self._estimate_distance_from_depth(depth, mask)
                    if depth_m is not None:
                        visible = 1.0
                        self._last_cx = cx
                        self._last_cy = cy
                        self._last_area = area
                        self._last_depth_m = float(depth_m)
                        self._steps_since_seen = 0

        if visible <= 0.0:
            self._steps_since_seen = min(self._steps_since_seen + 1, self._max_steps_since_seen)

        d_left, d_center, d_right = self._obstacle_depth_triplet(depth)
        steps_norm = float(self._steps_since_seen) / float(max(1, self._max_steps_since_seen))

        return np.asarray(
            [
                visible,
                float(self._last_cx),
                float(self._last_cy),
                float(self._last_area),
                float(self._last_depth_m),
                float(steps_norm),
                float(d_left),
                float(d_center),
                float(d_right),
            ],
            dtype=np.float32,
        )

    def _read_latest_seg_and_depth(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """读取最新的相机分割图与深度图。"""
        base = self.env.unwrapped
        aviary = getattr(base, "env", None)
        if aviary is None or not hasattr(aviary, "drones") or not aviary.drones:
            return None, None
        drone = aviary.drones[0]

        seg = getattr(drone, "segImg", None)
        depth = getattr(drone, "depthImg", None)
        if seg is None or depth is None:
            return None, None

        seg = np.asarray(seg)
        depth = np.asarray(depth)
        if seg.ndim == 3 and seg.shape[-1] == 1:
            seg = seg[..., 0]
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]
        return seg, depth

    def _obstacle_depth_triplet(self, depth: Optional[np.ndarray]) -> tuple[float, float, float]:
        """用深度图估计前方障碍物的粗略距离（左/中/右）。"""
        if depth is None:
            return 0.0, 0.0, 0.0

        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        depth_buf = depth.astype(np.float64, copy=False)
        near = self._depth_near
        far = self._depth_far
        denom = (far - (far - near) * depth_buf)
        denom = np.maximum(denom, 1e-9)
        z = (far * near) / denom
        z = np.where(np.isfinite(z), z, far)

        h, w = int(z.shape[0]), int(z.shape[1])
        y0 = int(0.35 * h)
        y1 = int(0.85 * h)
        x1 = int(w / 3)
        x2 = int(2 * w / 3)

        left = float(np.min(z[y0:y1, 0:x1]))
        center = float(np.min(z[y0:y1, x1:x2]))
        right = float(np.min(z[y0:y1, x2:w]))
        return left, center, right

    def _find_duck_body_id(self) -> Optional[int]:
        """在 wrapper 链中查找小黄鸭的 body id。"""
        cur: Any = self.env
        for _ in range(32):
            if hasattr(cur, "get_duck_body_id"):
                duck_id = cur.get_duck_body_id()
                return int(duck_id) if duck_id is not None else None
            if hasattr(cur, "duck_body_id"):
                duck_id = getattr(cur, "duck_body_id")
                return int(duck_id) if duck_id is not None else None
            if hasattr(cur, "env"):
                cur = cur.env
            else:
                break
        return None

    def _duck_mask_from_seg(self, seg: np.ndarray, duck_id: int) -> Optional[np.ndarray]:
        """从 PyBullet segmentation buffer 解析出 duck 的像素 mask。"""
        try:
            seg64 = seg.astype(np.int64, copy=False)
        except Exception:
            return None

        direct = seg64 == int(duck_id)
        if np.any(direct):
            return direct

        high8 = (seg64 >> 24) == int(duck_id)
        if np.any(high8):
            return high8

        low24 = (seg64 & 0xFFFFFF) == int(duck_id)
        if np.any(low24):
            return low24

        return None

    def _estimate_distance_from_depth(self, depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
        """将深度 buffer 转换为距离估计（米）。"""
        depth_buf = depth.astype(np.float64, copy=False)
        vals = depth_buf[mask]
        if vals.size == 0:
            return None

        near = self._depth_near
        far = self._depth_far
        denom = (far - (far - near) * vals)
        denom = np.maximum(denom, 1e-9)
        z = (far * near) / denom
        z = z[np.isfinite(z)]
        if z.size == 0:
            return None
        return float(np.median(z))

    def _set_auto_camera_capture_enabled(self, enabled: bool) -> None:
        """按阶段控制 PyFlyt 的自动相机捕获频率。"""
        base = self.env.unwrapped
        aviary = getattr(base, "env", None)
        if aviary is None or not hasattr(aviary, "drones") or not aviary.drones:
            return
        drone = aviary.drones[0]
        if not hasattr(drone, "use_camera") or not getattr(drone, "use_camera"):
            return
        if not hasattr(drone, "physics_camera_ratio"):
            return

        try:
            if enabled:
                control_ratio = int(getattr(drone, "physics_control_ratio", 8))
                drone.physics_camera_ratio = control_ratio * self._camera_capture_interval_steps
            else:
                drone.physics_camera_ratio = int(10**9)
        except Exception:
            return
