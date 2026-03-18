"""Base PyFlyt Environment for the Fixedwing model using the Gymnasim API."""

from __future__ import annotations

import time
from typing import Any, Callable, Literal

import gymnasium
import numpy as np
import pybullet as p
from gymnasium import spaces
from PyFlyt.core.utils.compile_helpers import check_numpy

class _AdapterCamera:
    def __init__(self) -> None:
        self.view_mat = np.eye(4, dtype=np.float64).reshape(-1).tolist()
        self.proj_mat = np.eye(4, dtype=np.float64).reshape(-1).tolist()
        self.is_tracking_camera = False


class _AdapterDrone:
    def __init__(self) -> None:
        self.camera = _AdapterCamera()
        self.physics_control_ratio = 8
        self.physics_camera_ratio = 8
        self.rgbaImg = None
        self.depthImg = None
        self.segImg = None


class ArdupilotAdapter:
    """Minimal Aviary-compatible adapter used during the ROS/ArduPilot migration."""

    GEOM_CYLINDER = 4

    @staticmethod
    def _quat_to_euler_xyz(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = float(np.arctan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1.0:
            pitch = float(np.sign(sinp) * (np.pi / 2.0))
        else:
            pitch = float(np.arcsin(sinp))

        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = float(np.arctan2(siny_cosp, cosy_cosp))
        return roll, pitch, yaw

    def __init__(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        drone_type: str,
        render: bool,
        drone_options: dict[str, Any],
        np_random: np.random.Generator,
    ) -> None:
        self.start_pos = np.asarray(start_pos, dtype=np.float64)
        self.start_orn = np.asarray(start_orn, dtype=np.float64)
        self.drone_type = str(drone_type)
        self.render = bool(render)
        self.drone_options = dict(drone_options)
        self.np_random = np_random

        self.drones = [_AdapterDrone()]
        self.contact_array = np.zeros((1, 1), dtype=bool)
        self._mode = 0
        self._wind_field_fn = None

        self._ang_vel = np.zeros((3,), dtype=np.float64)
        self._ang_pos = self.start_orn[0].copy()
        self._lin_vel = np.zeros((3,), dtype=np.float64)
        self._lin_pos = self.start_pos[0].copy()
        self._aux = np.zeros((6,), dtype=np.float64)
        self._last_setpoint = np.zeros((4,), dtype=np.float64)
        #rc_surface的接收变量名称或许可以简化
        self._rc_surface_channel_indices = np.asarray(
            self.drone_options.get("mavros_surface_channel_indices", [0, 1, 2, 3, 4]),
            dtype=np.int64,
        ).reshape(-1)
        self._rc_throttle_channel_index = int(
            self.drone_options.get("mavros_throttle_channel_index", 5)
        )
        self._rc_pwm_min = float(self.drone_options.get("mavros_rc_pwm_min", 1000.0))
        self._rc_pwm_max = float(self.drone_options.get("mavros_rc_pwm_max", 2000.0))
        self._rc_pwm_trim = float(self.drone_options.get("mavros_rc_pwm_trim", 1500.0))
        self._vision_rgb_topic = str(
            self.drone_options.get("vision_rgb_topic", "/camera/image_raw")
        )
        self._vision_depth_topic = str(
            self.drone_options.get("vision_depth_topic", "/camera/depth/image_raw")
        )
        self._vision_depth_is_meters = bool(
            self.drone_options.get("vision_depth_is_meters", True)
        )


        self._next_uid = 1
        self._body_ids: list[int] = []

        self._rclpy = None
        self._node = None
        self._executor = None
        self._owns_rclpy_init = False
        self._ros_state_ready = False
        self._state_update_seq = 0
        self._init_mavros_state_subscribers()

    def disconnect(self) -> None:
        try:
            if self._executor is not None and self._node is not None:
                self._executor.remove_node(self._node)
        except Exception:
            pass

        try:
            if self._node is not None:
                self._node.destroy_node()
        except Exception:
            pass

        if self._rclpy is not None and self._owns_rclpy_init:
            try:
                if self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass

        self._executor = None
        self._node = None
        self._rclpy = None
        self._owns_rclpy_init = False
        self._ros_state_ready = False

    def _init_mavros_state_subscribers(self) -> None:
        try:
            from copy import deepcopy
            import rclpy
            from geometry_msgs.msg import TwistStamped
            from nav_msgs.msg import Odometry
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image, Imu
            from mavros_msgs.msg import RCOut
        except Exception:
            return

        try:
            if not rclpy.ok():
                rclpy.init(args=None)
                self._owns_rclpy_init = True

            self._rclpy = rclpy
            self._node = Node(f"ardupilot_adapter_{int(time.time() * 1000) % 1000000}")
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            _mavros_imu_topic = str(
                self.drone_options.get("mavros_imu_topic", "/mavros/imu/data")
            )
            _mavros_odom_topic = str(
                self.drone_options.get("mavros_odom_topic", "/mavros/local_position/odom")
            )
            _mavros_vel_topic = str(
                self.drone_options.get("mavros_vel_topic", "/mavros/local_position/velocity_local")
            )
            _mavros_rc_out_topic = str(
                    self.drone_options.get("mavros_rc_out_topic", "/mavros/rc/out")
                )
            qos_latest = deepcopy(qos_profile_sensor_data)
            qos_latest.depth = 1
            self._node.create_subscription(
                Imu,
                _mavros_imu_topic,
                self._on_imu,
                qos_latest,
            )
            self._node.create_subscription(
                Odometry,
                _mavros_odom_topic,
                self._on_odom,
                qos_latest,
            )
            self._node.create_subscription(
                TwistStamped,
                _mavros_vel_topic,
                self._on_vel,
                qos_latest,
            )
            self._node.create_subscription(
                    RCOut,
                    _mavros_rc_out_topic,
                    self._on_rc_out,
                    qos_latest,
                )
            self._node.create_subscription(
                Image,
                self._vision_rgb_topic,
                self._on_rgb_image,
                qos_latest,
            )
            self._node.create_subscription(
                Image,
                self._vision_depth_topic,
                self._on_depth_image,
                qos_latest,
            )
            self._ros_state_ready = True
        except Exception:
            self._ros_state_ready = False

    def _spin_ros_once(self, timeout_sec: float = 0.0) -> None:
        if self._executor is None:
            return
        try:
            self._executor.spin_once(timeout_sec=timeout_sec)
        except TypeError:
            self._executor.spin_once(timeout_sec)
        except Exception:
            pass

    def _on_imu(self, msg: Any) -> None:
        try:
            av = msg.angular_velocity
            q = msg.orientation
            self._ang_vel = np.array([av.x, av.y, av.z], dtype=np.float64)
            roll, pitch, yaw = self._quat_to_euler_xyz(
                float(q.x), float(q.y), float(q.z), float(q.w)
            )
            self._ang_pos = np.array([roll, pitch, yaw], dtype=np.float64)
            self._state_update_seq += 1
        except Exception:
            return

    def _on_odom(self, msg: Any) -> None:
        try:
            pmsg = msg.pose.pose.position
            qmsg = msg.pose.pose.orientation
            vmsg = msg.twist.twist.linear

            self._lin_pos = np.array([pmsg.x, pmsg.y, pmsg.z], dtype=np.float64)
            self._lin_vel = np.array([vmsg.x, vmsg.y, vmsg.z], dtype=np.float64)

            roll, pitch, yaw = self._quat_to_euler_xyz(
                float(qmsg.x), float(qmsg.y), float(qmsg.z), float(qmsg.w)
            )
            self._ang_pos = np.array([roll, pitch, yaw], dtype=np.float64)
            self._state_update_seq += 1
        except Exception:
            return

    def _on_vel(self, msg: Any) -> None:
        try:
            vmsg = msg.twist.linear
            self._lin_vel = np.array([vmsg.x, vmsg.y, vmsg.z], dtype=np.float64)
            self._state_update_seq += 1
        except Exception:
            return

    def _norm_surface_pwm(self, pwm_value: float) -> float:
        span = max(1.0, 0.5 * (self._rc_pwm_max - self._rc_pwm_min))
        return float(np.clip((pwm_value - self._rc_pwm_trim) / span, -1.0, 1.0))

    def _norm_throttle_pwm(self, pwm_value: float) -> float:
        span = max(1.0, self._rc_pwm_max - self._rc_pwm_min)
        return float(np.clip((pwm_value - self._rc_pwm_min) / span, 0.0, 1.0))
    def _on_rc_out(self, msg: Any) -> None:
        try:
            channels = np.asarray(msg.channels, dtype=np.float64).reshape(-1)
            if channels.size == 0:
                return

            n_surfaces = min(5, self._aux.shape[0] - 1)
            for i in range(n_surfaces):
                if i >= self._rc_surface_channel_indices.size:
                    break
                ch_idx = int(self._rc_surface_channel_indices[i])
                if 0 <= ch_idx < channels.size:
                    self._aux[i] = self._norm_surface_pwm(channels[ch_idx])

            if self._aux.shape[0] >= 6:
                t_idx = self._rc_throttle_channel_index
                if 0 <= t_idx < channels.size:
                    self._aux[5] = self._norm_throttle_pwm(channels[t_idx])

            self._state_update_seq += 1
        except Exception:
            return

    @staticmethod
    def _ros_image_to_array(msg: Any) -> None | np.ndarray:
        h = int(getattr(msg, "height", 0))
        w = int(getattr(msg, "width", 0))
        step = int(getattr(msg, "step", 0))
        enc = str(getattr(msg, "encoding", "")).lower()
        data = getattr(msg, "data", None)
        if h <= 0 or w <= 0 or step <= 0 or data is None:
            return None

        enc_map: dict[str, tuple[Any, int]] = {
            "mono8": (np.uint8, 1),
            "8uc1": (np.uint8, 1),
            "8sc1": (np.int8, 1),
            "16uc1": (np.uint16, 1),
            "16sc1": (np.int16, 1),
            "32sc1": (np.int32, 1),
            "32fc1": (np.float32, 1),
            "rgb8": (np.uint8, 3),
            "bgr8": (np.uint8, 3),
            "rgba8": (np.uint8, 4),
            "bgra8": (np.uint8, 4),
        }
        if enc not in enc_map:
            return None

        dtype, channels = enc_map[enc]
        itemsize = np.dtype(dtype).itemsize
        row_elems = step // itemsize
        needed = w * channels
        if row_elems < needed:
            return None

        arr = np.frombuffer(data, dtype=dtype)
        expected = h * row_elems
        if arr.size < expected:
            return None
        arr = arr[:expected].reshape(h, row_elems)
        arr = arr[:, :needed]
        if channels == 1:
            arr = arr.reshape(h, w, 1)
        else:
            arr = arr.reshape(h, w, channels)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _depth_meters_to_buffer(depth_m: np.ndarray) -> np.ndarray:
        near = 0.1
        far = 255.0
        z = np.asarray(depth_m, dtype=np.float32)
        out = np.zeros_like(z, dtype=np.float32)
        valid = np.isfinite(z) & (z > 0.0)
        denom = np.maximum(z[valid] * (far - near), 1e-9)
        out[valid] = (far * (z[valid] - near)) / denom
        return np.clip(out, 0.0, 1.0)

    def _on_rgb_image(self, msg: Any) -> None:
        try:
            arr = self._ros_image_to_array(msg)
            if arr is None:
                return

            enc = str(getattr(msg, "encoding", "")).lower()
            if enc == "bgr8":
                arr = arr[..., ::-1]
            elif enc == "bgra8":
                arr = arr[..., [2, 1, 0, 3]]

            if arr.shape[-1] == 3:
                alpha = np.full((arr.shape[0], arr.shape[1], 1), 255, dtype=np.uint8)
                rgba = np.concatenate([arr.astype(np.uint8, copy=False), alpha], axis=-1)
            elif arr.shape[-1] == 4:
                rgba = arr.astype(np.uint8, copy=False)
            else:
                gray = arr[..., 0].astype(np.uint8, copy=False)
                rgba = np.repeat(gray[..., None], 4, axis=-1)
                rgba[..., 3] = 255

            drone = self.drones[0]
            drone.rgbaImg = np.ascontiguousarray(rgba)

            self._state_update_seq += 1
        except Exception:
            return

    def _on_depth_image(self, msg: Any) -> None:
        try:
            arr = self._ros_image_to_array(msg)
            if arr is None:
                return

            depth = arr[..., 0]
            enc = str(getattr(msg, "encoding", "")).lower()
            if enc == "16uc1":
                depth_m = depth.astype(np.float32) * 0.001
            else:
                depth_m = depth.astype(np.float32)

            if self._vision_depth_is_meters:
                depth_img = self._depth_meters_to_buffer(depth_m)
            else:
                depth_img = np.clip(depth_m, 0.0, 1.0)

            self.drones[0].depthImg = np.ascontiguousarray(depth_img[..., None])
            self._state_update_seq += 1
        except Exception:
            return

    def register_wind_field_function(self, fn: Callable[[float, np.ndarray], np.ndarray]) -> None:
        self._wind_field_fn = fn

    def getDebugVisualizerCamera(self):
        return ()

    def register_all_new_bodies(self) -> None:
        pass

    def set_mode(self, mode: int) -> None:
        self._mode = int(mode)

    def step(self) -> None:
        self._spin_ros_once(timeout_sec=0.0)

    def state(
        self, idx: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        _ = idx
        # Drain currently queued state callbacks before returning.
        # With depth=1 per topic, this effectively syncs to latest queued samples.
        for _ in range(16):
            prev_seq = self._state_update_seq
            self._spin_ros_once(timeout_sec=0.0)
            if self._state_update_seq == prev_seq:
                break
        return (
            self._ang_vel.copy(),
            self._ang_pos.copy(),
            self._lin_vel.copy(),
            self._lin_pos.copy(),
        )

    def aux_state(self, idx: int) -> np.ndarray:
        _ = idx
        return self._aux.copy()

    def set_setpoint(self, idx: int, action: np.ndarray) -> None:
        _ = idx
        self._last_setpoint = np.asarray(action, dtype=np.float64).copy()

    def getQuaternionFromEuler(self, euler_xyz: list[float]) -> tuple[float, float, float, float]:
        return p.getQuaternionFromEuler(euler_xyz)

    def getMatrixFromQuaternion(self, quat_xyzw: tuple[float, float, float, float]) -> list[float]:
        return p.getMatrixFromQuaternion(quat_xyzw)

    def _new_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def loadURDF(self, *args: Any, **kwargs: Any) -> int:
        _ = args, kwargs
        uid = self._new_uid()
        self._body_ids.append(uid)
        return uid

    def getNumBodies(self) -> int:
        return len(self._body_ids)

    def getBodyUniqueId(self, index: int) -> int:
        i = int(index)
        if i < 0 or i >= len(self._body_ids):
            return -1
        return int(self._body_ids[i])

    def removeBody(self, body_id: int) -> None:
        try:
            self._body_ids.remove(int(body_id))
        except ValueError:
            pass

    def createCollisionShape(self, *args: Any, **kwargs: Any) -> int:
        _ = args, kwargs
        return self._new_uid()

    def createVisualShape(self, *args: Any, **kwargs: Any) -> int:
        _ = args, kwargs
        return self._new_uid()

    def createMultiBody(self, *args: Any, **kwargs: Any) -> int:
        _ = args, kwargs
        uid = self._new_uid()
        self._body_ids.append(uid)
        return uid

    def loadPlugin(self, *args: Any, **kwargs: Any) -> int:
        _ = args, kwargs
        return 0

    def changeVisualShape(self, *args: Any, **kwargs: Any) -> None:
        _ = args, kwargs


class FixedwingBaseEnv(gymnasium.Env):
    """Base PyFlyt Environment for the Fixedwing model using the Gymnasim API."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        start_pos: np.ndarray = np.array([[0.0, 0.0, 1.0]]),
        start_orn: np.ndarray = np.array([[0.0, 0.0, 0.0]]),
        flight_mode: int = 0,
        flight_dome_size: float = np.inf,
        max_duration_seconds: float = 10.0,
        angle_representation: Literal["euler", "quaternion"] = "quaternion",
        agent_hz: int = 30,
        render_mode: None | Literal["human", "rgb_array"] = None,
        render_resolution: tuple[int, int] = (480, 480),
        wind_config: None | dict[str, Any] = None,
    ):
        """__init__.

        Args:
            start_pos (np.ndarray): start_pos
            start_orn (np.ndarray): start_orn
            flight_mode (int): flight_mode
            flight_dome_size (float): flight_dome_size
            max_duration_seconds (float): max_duration_seconds
            angle_representation (Literal["euler", "quaternion"]): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution

        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise ValueError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
            )
        self.render_mode = render_mode
        self.render_resolution = render_resolution
        self.wind_config = wind_config

        """GYMNASIUM STUFF"""
        # attitude size increases by 1 for quaternion
        if angle_representation == "euler":
            attitude_shape = 12
        elif angle_representation == "quaternion":
            attitude_shape = 13
        else:
            raise ValueError(
                f"angle_representation must be either `euler` or `quaternion`, not {angle_representation}"
            )

        self.attitude_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(attitude_shape,), dtype=np.float64
        )
        self.auxiliary_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64
        )
        high = np.ones((4,), dtype=np.float64)
        low = -high
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

        # the whole implicit state space = attitude + previous action + auxiliary information
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                attitude_shape
                + self.action_space.shape[0]
                + self.auxiliary_space.shape[0],
            ),
            dtype=np.float64,
        )

        """ ENVIRONMENT CONSTANTS """
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.flight_mode = flight_mode
        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1

    def _maybe_apply_wind_field(self) -> None:
        cfg = self.wind_config or {}
        if not bool(cfg.get("enabled", False)):
            return

        mode = str(cfg.get("mode", "constant")).lower()
        if mode not in ("constant", "gust_sine"):
            raise ValueError(f"Unsupported wind mode: {mode}")

        def _sample_vec3(
            base_key: str, range_key: str, default: tuple[float, float, float]
        ) -> np.ndarray:
            base = np.asarray(cfg.get(base_key, default), dtype=np.float64).reshape(3)
            if not bool(cfg.get("randomize_on_reset", False)):
                return base

            ranges = cfg.get(range_key, None)
            if ranges is None:
                return base

            if (
                not isinstance(ranges, (list, tuple))
                or len(ranges) != 3
                or not all(isinstance(r, (list, tuple)) and len(r) == 2 for r in ranges)
            ):
                raise ValueError(f"Invalid {range_key}: {ranges}")

            lows = np.asarray([r[0] for r in ranges], dtype=np.float64)
            highs = np.asarray([r[1] for r in ranges], dtype=np.float64)
            return self.np_random.uniform(lows, highs).astype(np.float64)

        base_wind = _sample_vec3(
            base_key="wind_enu_mps",
            range_key="wind_enu_mps_range",
            default=(0.0, 0.0, 0.0),
        )

        if mode == "constant":
            wind_enu = base_wind

            def wind_field(time_s: float, positions_m: np.ndarray) -> np.ndarray:
                n = int(positions_m.shape[0])
                return np.repeat(wind_enu.reshape(1, 3), repeats=n, axis=0)

            self.env.register_wind_field_function(wind_field)
            return

        gust_amp = _sample_vec3(
            base_key="gust_amp_enu_mps",
            range_key="gust_amp_enu_mps_range",
            default=(0.0, 0.0, 0.0),
        )
        gust_freq_hz = float(cfg.get("gust_freq_hz", 0.0))
        gust_phase = float(cfg.get("gust_phase_rad", 0.0))
        if bool(cfg.get("randomize_on_reset", False)) and bool(
            cfg.get("randomize_gust_phase", True)
        ):
            gust_phase = float(self.np_random.uniform(0.0, 2.0 * np.pi))

        def wind_field(time_s: float, positions_m: np.ndarray) -> np.ndarray:
            n = int(positions_m.shape[0])
            gust = gust_amp * np.sin(2.0 * np.pi * gust_freq_hz * time_s + gust_phase)
            wind_enu = base_wind + gust
            return np.repeat(wind_enu.reshape(1, 3), repeats=n, axis=0)

        self.env.register_wind_field_function(wind_field)

    def reset(
        self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> tuple[Any, dict]:
        """reset.

        Args:
            seed: seed to pass to the base environment.
            options: None

        """
        raise NotImplementedError

    def close(self) -> None:
        """Disconnects the internal Aviary."""
        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

    def begin_reset(
        self,
        seed: None | int = None,
        options: None | dict[str, Any] = dict(),
        drone_options: None | dict[str, Any] = dict(),
    ) -> None:
        """The first half of the reset function."""
        super().reset(seed=seed)

        # if we already have an env, disconnect from it
        if hasattr(self, "env"):
            self.env.disconnect()

        self.step_count = 0
        self.termination = False
        self.truncation = False
        self.state = None
        self.action = np.zeros((4,))
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        # need to handle Nones
        if options is None:
            options = dict()
        if drone_options is None:
            drone_options = dict()

        # camera handling
        drone_options["use_camera"] = drone_options.get("use_camera", False) or bool(
            self.render_mode
        )
        drone_options["camera_fps"] = int(120 / self.env_step_ratio)

        # init env
        self.env = ArdupilotAdapter(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="fixedwing",
            render=self.render_mode == "human",
            drone_options=drone_options,
            np_random=self.np_random,
        )
        self._maybe_apply_wind_field()

        if self.render_mode == "human":
            self.camera_parameters = self.env.getDebugVisualizerCamera()

    def end_reset(
        self, seed: None | int = None, options: None | dict[str, Any] = dict()
    ) -> None:
        """The tailing half of the reset function."""
        # register all new collision bodies
        self.env.register_all_new_bodies()

        # set flight mode
        self.env.set_mode(self.flight_mode)

        # wait for env to stabilize
        for _ in range(10):
            self.env.step()

        self.compute_state()

    def compute_state(self) -> None:
        """Computes the state of the Rocket."""
        raise NotImplementedError

    def compute_auxiliary(self) -> np.ndarray:
        """This returns the auxiliary state form the drone."""
        return self.env.aux_state(0)

    def compute_attitude(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """state.

        This returns the base attitude for the drone.
        - ang_vel (vector of 3 values)
        - ang_pos (vector of 3/4 values)
        - lin_vel (vector of 3 values)
        - lin_pos (vector of 3 values)
        - quaternion (vector of 4 values)
        """
        raw_state = self.env.state(0)

        # state breakdown
        ang_vel = raw_state[0]
        ang_pos = raw_state[1]
        lin_vel = raw_state[2]
        lin_pos = raw_state[3]

        # quaternion angles
        quaternion = p.getQuaternionFromEuler(ang_pos)

        return ang_vel, ang_pos, lin_vel, lin_pos, quaternion

    def compute_term_trunc_reward(self) -> None:
        """compute_term_trunc_reward."""
        raise NotImplementedError

    def compute_base_term_trunc_reward(self) -> None:
        """compute_base_term_trunc_reward."""
        # exceed step count
        if self.step_count > self.max_steps:
            self.truncation |= True

        # collision
        if np.any(self.env.contact_array):
            self.reward = -100.0
            self.info["collision"] = True
            self.termination |= True

        # exceed flight dome
        if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination |= True

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info

        """
        # reset the reward
        self.reward = -0.1

        # pass the action, but clip the throttle
        self.action = action.copy()
        aviary_action = action.copy()
        aviary_action[..., -1] = (aviary_action[..., -1] / 2.0) + 0.5
        self.env.set_setpoint(0, aviary_action)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break

            self.env.step()

            # compute state and done
            self.compute_state()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info
    def render(self) -> np.ndarray:
        """Render."""
        check_numpy()
        if self.render_mode is None:
            raise ValueError(
                "Please set `render_mode='human'` or `render_mode='rgb_array'` in init to use this function."
            )

        _, _, rgbaImg, _, _ = self.env.getCameraImage(
            width=self.render_resolution[1],
            height=self.render_resolution[0],
            viewMatrix=self.env.drones[0].camera.view_mat,
            projectionMatrix=self.env.drones[0].camera.proj_mat,
        )

        rgbaImg = np.asarray(rgbaImg, dtype=np.uint8).reshape(
            self.render_resolution[0], self.render_resolution[1], -1
        )

        return rgbaImg
