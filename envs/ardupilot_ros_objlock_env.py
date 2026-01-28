from __future__ import annotations

"""
ArduPilot + Gazebo (gz sim) ObjLock environment (ROS-backed).

This file intentionally only provides a *skeleton* Gymnasium environment that:
- Subscribes to vehicle state via MAVROS topics (IMU / odom / velocity).
- Subscribes to a pod camera RGB topic and (optionally) a depth topic.
- Publishes control via RC override (MAVROS), mapping PPO actions [-1, 1] -> PWM.

It is designed to replace the PyFlyt-based FixedwingObjLockEnv as a data source,
while keeping the same *observation structure* (Dict with attitude/target_vector/duck_vision)
so your existing FlattenObjLockEnv + SB3 PPO pipeline can be reused.

Important:
- This env assumes external processes are already running:
  1) gz sim world (ardupilot_gazebo) + ArduPilot SITL
  2) mavros node (ROS1/ROS2 depending on your setup)
  3) (optional) ros_gz_bridge to bridge gz camera topics to ROS Image messages
- This env does NOT start mavros or gz sim for you.
"""

import time
import threading
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np

# Gymnasium is required when you want to plug this env into SB3 (PPO).
# For simple "does ROS subscription update?" smoke tests, we keep a fallback
# path so the module can be imported without gymnasium installed.
try:
    import gymnasium as gym  # type: ignore
    from gymnasium import spaces  # type: ignore
except Exception:  # pragma: no cover
    gym = None
    spaces = None


def _clamp(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def _quat_to_euler_xyz(qx: float, qy: float, qz: float, qw: float) -> tuple[float, float, float]:
    """Quaternion -> Euler (roll, pitch, yaw), XYZ intrinsic (common aerospace convention)."""
    # Numerically-stable implementation; assumes unit quaternion.
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1.0:
        pitch = float(np.sign(sinp) * (np.pi / 2.0))
    else:
        pitch = float(np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


@dataclass
class _LatestState:
    # ROS time isn't required for the skeleton; we use monotonic arrival time.
    t_mono: float = 0.0

    # Angular velocity (rad/s), body frame.
    ang_vel: Optional[np.ndarray] = None  # (3,)

    # Orientation quaternion (x,y,z,w), world->body or body->world depending on msg;
    # MAVROS /imu/data orientation is body in ENU; we treat it consistently as "attitude quaternion".
    quat_xyzw: Optional[np.ndarray] = None  # (4,)

    # Linear velocity (m/s) in world frame (ENU) by default.
    lin_vel: Optional[np.ndarray] = None  # (3,)

    # Position (m) in world frame (ENU) by default.
    lin_pos: Optional[np.ndarray] = None  # (3,)

    # Servo outputs as PWM (microseconds), indexed 1..N (stored 0-based in array).
    servo_pwm: Optional[np.ndarray] = None  # (N,)


@dataclass
class _LatestMavrosState:
    t_mono: float = 0.0
    connected: bool = False
    armed: bool = False
    mode: str = ""


def _pwm_to_norm(pwm: float, *, pwm_min: int, pwm_trim: int, pwm_max: int) -> float:
    """
    Map PWM to normalized [-1, 1] around TRIM.

    - For control surfaces (trim ~1500): 1000 -> -1, 1500 -> 0, 2000 -> 1.
    - For throttle (trim == min == 1000): 1000 -> 0, 2000 -> 1 (i.e. non-negative).
    """
    p = float(pwm)
    lo = float(pwm_min)
    mid = float(pwm_trim)
    hi = float(pwm_max)

    # Guard degenerate configs.
    if not np.isfinite(p) or hi <= lo:
        return 0.0

    if p >= mid:
        denom = max(1.0, hi - mid)
        x = (p - mid) / denom
    else:
        denom = max(1.0, mid - lo)
        x = (p - mid) / denom
    return _clamp(x, -1.0, 1.0)


_BaseGymEnv = gym.Env if gym is not None else object


class ArdupilotGazeboObjLockEnv(_BaseGymEnv):
    """
    Gymnasium-compatible environment that reads observations from ROS/MAVROS.

    Observation (Dict):
      - attitude: [ang_vel(3), ang_pos_euler(3) or quaternion(4), lin_vel(3), lin_pos(3),
                   prev_action(4), aux_state(6)]
      - target_vector: placeholder (3,) default zeros (no privileged info)
      - duck_vision: placeholder features (history + deltas), default zeros until you plug vision in

    Action (Box):
      - [roll, pitch, yaw, thrust] in [-1, 1]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        angle_representation: Literal["euler", "quaternion"] = "euler",
        agent_hz: int = 30,
        max_duration_seconds: float = 120.0,
        duck_vision_history_len: int = 3,
        duck_vision_use_deltas: bool = True,
        # Topic names (ROS2 defaults are user-setup dependent; treat these as config knobs)
        mavros_imu_topic: str = "/mavros/imu/data",
        mavros_odom_topic: str = "/mavros/local_position/odom",
        mavros_vel_topic: str = "/mavros/local_position/velocity_local",
        mavros_rc_override_topic: str = "/mavros/rc/override",
        mavros_servo_output_topic: str = "/mavros/servo_output/raw",
        mavros_rc_out_topic: str = "/mavros/rc/out",
        rgb_image_topic: Optional[str] = None,
        depth_image_topic: Optional[str] = None,
        # RC mapping (ArduPilot default: 1 roll, 2 pitch, 3 throttle, 4 yaw)
        rc_chan_roll: int = 1,
        rc_chan_pitch: int = 2,
        rc_chan_throttle: int = 3,
        rc_chan_yaw: int = 4,
        pwm_center: int = 1500,
        pwm_span: int = 500,
        pwm_throttle_min: int = 1000,
        pwm_throttle_max: int = 2000,
        # Servo -> aux_state mapping (defaults match your mini_talon_vtail model.sdf comments)
        servo_ch_aileron: int = 1,
        # aux_state[2] always reads from this channel (default: SERVO2)
        servo_ch_vtail_left: int = 2,
        servo_ch_throttle: int = 3,
        # aux_state[3] always reads from this channel (default: SERVO4)
        servo_ch_vtail_right: int = 4,
        split_aileron_left_right: bool = True,
        vtail_mix_to_elev_rudder: bool = False,
        # SERVOx min/trim/max for normalization.
        # If None, use common defaults: 1000/1500/2000 for control surfaces and 1000/1000/2000 for throttle.
        servo_params: Optional[dict[int, tuple[int, int, int]]] = None,
        # Safety: allow disabling control publishing for passive observation tests.
        enable_rc_override: bool = True,
        # Blocking behavior
        reset_timeout_sec: float = 5.0,
        step_wait_timeout_sec: float = 0.5,
    ):
        super().__init__()

        if agent_hz <= 0:
            raise ValueError("agent_hz must be > 0")
        self.agent_hz = int(agent_hz)
        self.dt = 1.0 / float(self.agent_hz)
        self.max_steps = int(self.agent_hz * float(max_duration_seconds))

        self.angle_representation = angle_representation
        self.duck_vision_history_len = int(max(1, duck_vision_history_len))
        self.duck_vision_use_deltas = bool(duck_vision_use_deltas)

        # --- Spaces ---
        if self.angle_representation == "euler":
            attitude_dim = 3 + 3 + 3 + 3 + 4 + 6  # 22
        elif self.angle_representation == "quaternion":
            attitude_dim = 3 + 4 + 3 + 3 + 4 + 6  # 23
        else:
            raise ValueError("angle_representation must be 'euler' or 'quaternion'")

        duck_vision_dim = 9 * self.duck_vision_history_len
        if self.duck_vision_use_deltas:
            duck_vision_dim += 4

        if spaces is not None:
            self.observation_space = spaces.Dict(
                {
                    "attitude": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(attitude_dim,), dtype=np.float32
                    ),
                    "target_vector": spaces.Box(
                        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
                    ),
                    "duck_vision": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(duck_vision_dim,),
                        dtype=np.float32,
                    ),
                }
            )

            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(4,), dtype=np.float32
            )
        else:
            # Allows importing/running a basic ROS subscription test without gymnasium.
            self.observation_space = None
            self.action_space = None

        # --- ROS wiring (lazy imports so the repo can be read without ROS installed) ---
        self._ros_ok = False
        self._ros_err: Optional[Exception] = None
        self._rclpy = None
        self._node = None
        self._executor = None

        self._mavros_imu_topic = str(mavros_imu_topic)
        self._mavros_odom_topic = str(mavros_odom_topic)
        self._mavros_vel_topic = str(mavros_vel_topic)
        self._mavros_rc_override_topic = str(mavros_rc_override_topic)
        self._mavros_servo_output_topic = str(mavros_servo_output_topic)
        self._mavros_rc_out_topic = str(mavros_rc_out_topic)
        self._rgb_image_topic = str(rgb_image_topic) if rgb_image_topic else None
        self._depth_image_topic = str(depth_image_topic) if depth_image_topic else None

        self._rc_chan_roll = int(rc_chan_roll)
        self._rc_chan_pitch = int(rc_chan_pitch)
        self._rc_chan_throttle = int(rc_chan_throttle)
        self._rc_chan_yaw = int(rc_chan_yaw)
        self._pwm_center = int(pwm_center)
        self._pwm_span = int(pwm_span)
        self._pwm_throttle_min = int(pwm_throttle_min)
        self._pwm_throttle_max = int(pwm_throttle_max)

        self._servo_ch_aileron = int(servo_ch_aileron)
        # NOTE: Despite the name, these are simply "tail surface channel 1/2" in the aux_state wiring.
        # We keep the naming because your current Gazebo model is V-tail (ruddervators).
        self._servo_ch_vtail_left = int(servo_ch_vtail_left)   # -> aux_state[2]
        self._servo_ch_throttle = int(servo_ch_throttle)
        self._servo_ch_vtail_right = int(servo_ch_vtail_right)  # -> aux_state[3]
        self._split_aileron_left_right = bool(split_aileron_left_right)
        self._vtail_mix_to_elev_rudder = bool(vtail_mix_to_elev_rudder)

        self._servo_params = servo_params or {}
        self._enable_rc_override = bool(enable_rc_override)

        self._reset_timeout_sec = float(reset_timeout_sec)
        self._step_wait_timeout_sec = float(step_wait_timeout_sec)

        # --- Runtime state ---
        self._lock = threading.Lock()
        self._latest = _LatestState()
        self._mavros_state = _LatestMavrosState()
        self._last_action = np.zeros((4,), dtype=np.float32)
        self._step_count = 0

        # Vision feature history (same layout as FixedwingObjLockEnv)
        self._steps_since_seen = 60
        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._vision_history = np.zeros((self.duck_vision_history_len, 9), dtype=np.float32)
        self._vision_history_filled = 0

        # Latest raw image messages (optional; decode/process externally or extend this class)
        self._latest_rgb_msg = None
        self._latest_depth_msg = None

        self._last_aux_state = np.zeros((6,), dtype=np.float32)
        self._last_servo_pwm_used_for_aux: Optional[np.ndarray] = None

        self._try_init_ros()

    # ----------------------------
    # ROS setup + callbacks
    # ----------------------------

    def _try_init_ros(self) -> None:
        try:
            import rclpy  # type: ignore
            from rclpy.executors import SingleThreadedExecutor  # type: ignore
            from rclpy.node import Node  # type: ignore
            from rclpy.qos import (  # type: ignore
                QoSProfile,
                QoSReliabilityPolicy,
                QoSHistoryPolicy,
                qos_profile_sensor_data,
            )

            # Messages (optional)
            from sensor_msgs.msg import Imu, Image  # type: ignore
            from nav_msgs.msg import Odometry  # type: ignore
            from geometry_msgs.msg import TwistStamped  # type: ignore

            # MAVROS messages/services: import individually so one missing symbol doesn't disable others.
            try:
                from mavros_msgs.msg import OverrideRCIn  # type: ignore
            except Exception:
                OverrideRCIn = None

            try:
                from mavros_msgs.msg import ServoOutputRaw  # type: ignore
            except Exception:
                ServoOutputRaw = None

            try:
                from mavros_msgs.msg import RCOut  # type: ignore
            except Exception:
                RCOut = None

            try:
                from mavros_msgs.msg import State  # type: ignore
            except Exception:
                State = None

            try:
                from mavros_msgs.srv import StreamRate  # type: ignore
            except Exception:
                StreamRate = None

            self._rclpy = rclpy
            if not rclpy.ok():
                rclpy.init(args=None)

            node_name = f"objlock_env_{int(time.time() * 1000) % 1000000}"
            self._node = Node(node_name)
            self._executor = SingleThreadedExecutor()
            self._executor.add_node(self._node)

            # Keep subscription handles alive (rclpy will stop receiving callbacks if they are GC'd).
            self._subs = []

            # MAVROS in ROS2 often publishes sensor-like topics as BEST_EFFORT.
            # Use sensor-data QoS to avoid RELIABILITY incompatibility.
            try:
                qos_sensor = qos_profile_sensor_data
            except Exception:
                qos_sensor = QoSProfile(
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=10,
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                )

            qos_reliable = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=QoSReliabilityPolicy.RELIABLE,
            )

            # Store message classes for publisher
            self._msg_OverrideRCIn = OverrideRCIn

            # Subscriptions
            if State is not None:
                self._subs.append(
                    self._node.create_subscription(
                        State, "/mavros/state", self._on_mavros_state, qos_reliable
                    )
                )
            self._subs.append(
                self._node.create_subscription(
                    Imu, self._mavros_imu_topic, self._on_imu, qos_sensor
                )
            )
            self._subs.append(
                self._node.create_subscription(
                    Odometry, self._mavros_odom_topic, self._on_odom, qos_sensor
                )
            )
            self._subs.append(
                self._node.create_subscription(
                    TwistStamped, self._mavros_vel_topic, self._on_vel, qos_sensor
                )
            )
            if ServoOutputRaw is not None:
                self._subs.append(
                    self._node.create_subscription(
                        ServoOutputRaw,
                        self._mavros_servo_output_topic,
                        self._on_servo_output_raw,
                        qos_sensor,
                    )
                )
            if RCOut is not None:
                self._subs.append(
                    self._node.create_subscription(
                        RCOut,
                        self._mavros_rc_out_topic,
                        self._on_rc_out,
                        qos_reliable,
                    )
                )
            if self._rgb_image_topic:
                self._subs.append(
                    self._node.create_subscription(
                        Image, self._rgb_image_topic, self._on_rgb, qos_sensor
                    )
                )
            if self._depth_image_topic:
                self._subs.append(
                    self._node.create_subscription(
                        Image, self._depth_image_topic, self._on_depth, qos_sensor
                    )
                )

            # Publisher (RC override)
            self._rc_pub = None
            if OverrideRCIn is not None:
                self._rc_pub = self._node.create_publisher(
                    OverrideRCIn, self._mavros_rc_override_topic, 10
                )

            # Service clients (optional; used to request streams if topics are silent)
            self._srv_stream_rate = None
            if StreamRate is not None:
                self._srv_stream_rate = self._node.create_client(
                    StreamRate, "/mavros/set_stream_rate"
                )

            self._ros_ok = True
        except Exception as e:
            self._ros_ok = False
            self._ros_err = e

    def _spin_once(self, timeout_sec: float = 0.0) -> None:
        if not self._ros_ok or self._executor is None:
            return
        # SingleThreadedExecutor doesn't have spin_once in older distros; handle both.
        try:
            self._executor.spin_once(timeout_sec=timeout_sec)
        except TypeError:
            # Some rclpy versions use "timeout_sec" as positional.
            self._executor.spin_once(timeout_sec)

    def _on_imu(self, msg) -> None:
        t = time.monotonic()
        try:
            av = msg.angular_velocity
            ang_vel = np.array([av.x, av.y, av.z], dtype=np.float32)
            q = msg.orientation
            quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float32)
        except Exception:
            return
        with self._lock:
            self._latest.t_mono = t
            self._latest.ang_vel = ang_vel
            self._latest.quat_xyzw = quat

    def _on_odom(self, msg) -> None:
        t = time.monotonic()
        try:
            p = msg.pose.pose.position
            lin_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
            v = msg.twist.twist.linear
            lin_vel = np.array([v.x, v.y, v.z], dtype=np.float32)
        except Exception:
            return
        with self._lock:
            self._latest.t_mono = t
            self._latest.lin_pos = lin_pos
            self._latest.lin_vel = lin_vel

    def _on_vel(self, msg) -> None:
        t = time.monotonic()
        try:
            v = msg.twist.linear
            lin_vel = np.array([v.x, v.y, v.z], dtype=np.float32)
        except Exception:
            return
        with self._lock:
            self._latest.t_mono = t
            self._latest.lin_vel = lin_vel

    def _on_mavros_state(self, msg) -> None:
        t = time.monotonic()
        try:
            connected = bool(getattr(msg, "connected", False))
            armed = bool(getattr(msg, "armed", False))
            mode = str(getattr(msg, "mode", "")) or ""
        except Exception:
            return
        with self._lock:
            self._mavros_state.t_mono = t
            self._mavros_state.connected = connected
            self._mavros_state.armed = armed
            self._mavros_state.mode = mode

    def _on_rgb(self, msg) -> None:
        self._latest_rgb_msg = msg

    def _on_depth(self, msg) -> None:
        self._latest_depth_msg = msg

    def _on_servo_output_raw(self, msg) -> None:
        """
        MAVROS ServoOutputRaw provides servo1_raw..servo8_raw (and possibly more depending on build).
        We store them 1-based into a 0-based numpy array.
        """
        t = time.monotonic()
        vals: list[int] = []
        for i in range(1, 17):
            k = f"servo{i}_raw"
            if hasattr(msg, k):
                try:
                    vals.append(int(getattr(msg, k)))
                except Exception:
                    vals.append(0)
        if vals:
            with self._lock:
                self._latest.t_mono = t
                self._latest.servo_pwm = np.asarray(vals, dtype=np.int32)

    def _on_rc_out(self, msg) -> None:
        """
        MAVROS RCOut provides an array of PWM outputs (channels), typically length 8 or 16.
        We store it as servo_pwm as a fallback if ServoOutputRaw isn't available.
        """
        t = time.monotonic()
        try:
            ch = list(getattr(msg, "channels"))
            if ch:
                with self._lock:
                    self._latest.t_mono = t
                    self._latest.servo_pwm = np.asarray([int(x) for x in ch], dtype=np.int32)
        except Exception:
            pass

    # ----------------------------
    # Gymnasium API
    # ----------------------------

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        if gym is not None:
            super().reset(seed=seed)
        self._step_count = 0
        self._last_action[:] = 0.0

        # Reset vision buffers
        self._steps_since_seen = 60
        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._vision_history.fill(0.0)
        self._vision_history_filled = 0
        self._last_aux_state[:] = 0.0

        if not self._ros_ok:
            raise RuntimeError(
                "ROS2 (rclpy) is not available or failed to initialize. "
                f"Import/init error: {self._ros_err!r}"
            )

        # Wait for initial state messages so obs isn't all zeros.
        t0 = time.monotonic()
        while (time.monotonic() - t0) < self._reset_timeout_sec:
            self._spin_once(timeout_sec=0.05)
            if self._latest.ang_vel is not None and self._latest.lin_pos is not None:
                break

            # If MAVROS is connected but key topics remain silent, request stream rates once.
            if (
                (time.monotonic() - t0) > 1.0
                and self._mavros_state.connected
                and not getattr(self, "_stream_rate_requested", False)
            ):
                self._request_stream_rate_all(message_rate_hz=self.agent_hz)
                self._stream_rate_requested = True

        obs = self._build_obs()
        info = {
            "ros_ok": self._ros_ok,
            "mavros_connected": bool(self._mavros_state.connected),
            "have_imu": self._latest.ang_vel is not None,
            "have_pos": self._latest.lin_pos is not None,
            "have_vel": self._latest.lin_vel is not None,
            "have_rgb": self._latest_rgb_msg is not None,
            "have_depth": self._latest_depth_msg is not None,
            "have_servo": self._latest.servo_pwm is not None,
            "duck_strike": False,
            "env_complete": False,
            "is_success": False,
        }
        return obs, info

    def step(self, action: np.ndarray):
        if not self._ros_ok:
            raise RuntimeError(
                "ROS2 (rclpy) is not available or failed to initialize. "
                f"Import/init error: {self._ros_err!r}"
            )

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError(f"Expected action shape (4,), got {action.shape}")

        # Clip to action space
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        if self._enable_rc_override:
            self._publish_rc_override(action)
        self._last_action[:] = action

        # Wait for fresh state after sending command (basic sync).
        last_t = float(self._latest.t_mono)
        t0 = time.monotonic()
        while (time.monotonic() - t0) < self._step_wait_timeout_sec:
            self._spin_once(timeout_sec=0.0)
            if float(self._latest.t_mono) > last_t:
                break
            # Small sleep to reduce busy loop
            time.sleep(0.001)

        obs = self._build_obs()
        reward = 0.0
        terminated = False
        truncated = False

        self._step_count += 1
        if self._step_count >= self.max_steps:
            truncated = True

        info = {
            "ros_ok": self._ros_ok,
            "mavros_connected": bool(self._mavros_state.connected),
            "have_imu": self._latest.ang_vel is not None,
            "have_pos": self._latest.lin_pos is not None,
            "have_vel": self._latest.lin_vel is not None,
            "have_rgb": self._latest_rgb_msg is not None,
            "have_depth": self._latest_depth_msg is not None,
            "have_servo": self._latest.servo_pwm is not None,
            "duck_strike": False,
            "env_complete": False,
            "is_success": False,
        }
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        try:
            if self._executor is not None and self._node is not None:
                self._executor.remove_node(self._node)
            if self._node is not None:
                self._node.destroy_node()
            if self._rclpy is not None and self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass

    def get_debug_snapshot(self) -> dict[str, Any]:
        """
        Return a small, non-ROS message snapshot for quick debugging/printing.
        This is intended for local tests (e.g. scripts/test_ardupilot_ros_objlock_env.py).
        """
        with self._lock:
            servo = None
            if self._latest.servo_pwm is not None:
                servo = self._latest.servo_pwm.copy()
            return {
                "t_mono": float(self._latest.t_mono),
                "ang_vel": None
                if self._latest.ang_vel is None
                else self._latest.ang_vel.copy(),
                "quat_xyzw": None
                if self._latest.quat_xyzw is None
                else self._latest.quat_xyzw.copy(),
                "lin_vel": None
                if self._latest.lin_vel is None
                else self._latest.lin_vel.copy(),
                "lin_pos": None
                if self._latest.lin_pos is None
                else self._latest.lin_pos.copy(),
                "servo_pwm": servo,
                "servo_pwm_used_for_aux": None
                if self._last_servo_pwm_used_for_aux is None
                else self._last_servo_pwm_used_for_aux.copy(),
                "aux_state": self._last_aux_state.copy(),
                "mavros_connected": bool(self._mavros_state.connected),
                "mavros_mode": str(self._mavros_state.mode),
            }

    # ----------------------------
    # Observation building
    # ----------------------------

    def _build_obs(self) -> dict[str, np.ndarray]:
        attitude = self._build_attitude()
        target_vector = np.zeros((3,), dtype=np.float32)  # No privileged info by default.
        duck_vision = self._build_duck_vision_observation(self._compute_vision_features_base())
        return {
            "attitude": attitude,
            "target_vector": target_vector,
            "duck_vision": duck_vision,
        }

    def _build_attitude(self) -> np.ndarray:
        ang_vel = self._latest.ang_vel
        quat = self._latest.quat_xyzw
        lin_vel = self._latest.lin_vel
        lin_pos = self._latest.lin_pos

        if ang_vel is None:
            ang_vel = np.zeros((3,), dtype=np.float32)
        if quat is None:
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        if lin_vel is None:
            lin_vel = np.zeros((3,), dtype=np.float32)
        if lin_pos is None:
            lin_pos = np.zeros((3,), dtype=np.float32)

        if self.angle_representation == "euler":
            roll, pitch, yaw = _quat_to_euler_xyz(
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            )
            ang_pos = np.array([roll, pitch, yaw], dtype=np.float32)
            base = np.concatenate([ang_vel, ang_pos, lin_vel, lin_pos], axis=0)
        else:
            base = np.concatenate([ang_vel, quat.astype(np.float32), lin_vel, lin_pos], axis=0)

        aux_state = self._build_aux_state()
        out = np.concatenate([base, self._last_action, aux_state], axis=0).astype(np.float32)
        return out

    def _get_servo_params(self, ch_1based: int) -> tuple[int, int, int]:
        """
        Return (min, trim, max) for a given SERVO output channel.
        """
        if ch_1based in self._servo_params:
            a, b, c = self._servo_params[ch_1based]
            return int(a), int(b), int(c)
        # Common defaults: control surfaces 1000/1500/2000, throttle 1000/1000/2000.
        if ch_1based == self._servo_ch_throttle:
            return 1000, 1000, 2000
        return 1000, 1500, 2000

    def _servo_pwm(self, ch_1based: int) -> Optional[int]:
        """
        Read a single servo PWM from the latest buffer.
        Prefer using a snapshot for consistent multi-channel reads.
        """
        with self._lock:
            data = None if self._latest.servo_pwm is None else self._latest.servo_pwm.copy()
        return self._servo_pwm_from_snapshot(data, ch_1based)

    def _servo_pwm_from_snapshot(
        self, servo_pwm: Optional[np.ndarray], ch_1based: int
    ) -> Optional[int]:
        if servo_pwm is None:
            return None
        idx = int(ch_1based) - 1
        if idx < 0 or idx >= int(servo_pwm.shape[0]):
            return None
        v = int(servo_pwm[idx])
        return v if v > 0 else None

    def _servo_norm(self, ch_1based: int) -> Optional[float]:
        pwm = self._servo_pwm(ch_1based)
        if pwm is None:
            return None
        mn, tr, mx = self._get_servo_params(ch_1based)
        return float(_pwm_to_norm(pwm, pwm_min=mn, pwm_trim=tr, pwm_max=mx))

    def _servo_norm_from_snapshot(
        self, servo_pwm: Optional[np.ndarray], ch_1based: int
    ) -> Optional[float]:
        pwm = self._servo_pwm_from_snapshot(servo_pwm, ch_1based)
        if pwm is None:
            return None
        mn, tr, mx = self._get_servo_params(ch_1based)
        return float(_pwm_to_norm(pwm, pwm_min=mn, pwm_trim=tr, pwm_max=mx))

    def _build_aux_state(self) -> np.ndarray:
        """
        Build aux_state(6) per the PDF definition:
          [left_aileron, right_aileron, elevator, rudder, main_wing, throttle]

        Notes:
        - mini_talon_vtail.yaml sets main_wing deflection_limit=0, so main_wing is always 0.
        - If vtail_mix_to_elev_rudder=False (default), we keep a strict wiring:
            aux_state[2] <- normalize(SERVO{servo_ch_vtail_left})
            aux_state[3] <- normalize(SERVO{servo_ch_vtail_right})
          This matches your requirement: servo->aux_state wiring stays the same regardless
          of whether the tail is V-tail or conventional; you handle interpretation later.
        - If vtail_mix_to_elev_rudder=True, we derive an "equivalent" (elevator, rudder)
          from the two tail surface channels (useful only if you explicitly want that).
        """
        # Default: keep the last aux_state if no new servo data yet (helps continuity).
        with self._lock:
            servo_pwm = None if self._latest.servo_pwm is None else self._latest.servo_pwm.copy()
        if servo_pwm is None:
            return self._last_aux_state.copy()

        # Keep for debugging: show exactly which PWM values were used to compute aux_state.
        self._last_servo_pwm_used_for_aux = servo_pwm

        u_ail = self._servo_norm_from_snapshot(servo_pwm, self._servo_ch_aileron)
        u_thr = self._servo_norm_from_snapshot(servo_pwm, self._servo_ch_throttle)
        u_vl = self._servo_norm_from_snapshot(servo_pwm, self._servo_ch_vtail_left)
        u_vr = self._servo_norm_from_snapshot(servo_pwm, self._servo_ch_vtail_right)

        # Aileron: if only one channel, you can either duplicate or split +/-.
        if u_ail is None:
            left_ail = 0.0
            right_ail = 0.0
        elif self._split_aileron_left_right:
            # Matches the common "left = -right" aileron deflection convention.
            left_ail = float(-u_ail)
            right_ail = float(u_ail)
        else:
            left_ail = float(u_ail)
            right_ail = float(u_ail)

        # Tail: V-tail mixing -> equivalent elevator + rudder.
        if self._vtail_mix_to_elev_rudder and (u_vl is not None) and (u_vr is not None):
            # Convention: elevator is symmetric, rudder is differential.
            elev = float((u_vl + u_vr) * 0.5)
            rud = float((u_vr - u_vl) * 0.5)
        else:
            elev = float(u_vl) if u_vl is not None else 0.0
            rud = float(u_vr) if u_vr is not None else 0.0

        thr = float(u_thr) if u_thr is not None else 0.0

        # main_wing is not actuated for this model.
        main_wing = 0.0

        aux = np.array([left_ail, right_ail, elev, rud, main_wing, thr], dtype=np.float32)
        self._last_aux_state[:] = aux
        return aux

    def _compute_vision_features_base(self) -> np.ndarray:
        """
        Returns the base 9D vision feature:
          [visible, cx, cy, area, depth, steps_norm, d_left, d_center, d_right]

        Skeleton behavior:
        - Always returns "not visible" until you extend this class to run a detector/segmenter.
        - Keeps a steps_since_seen counter (like the PyFlyt env) to support history.
        """
        self._steps_since_seen = min(self._steps_since_seen + 1, 60)
        steps_norm = float(self._steps_since_seen) / 60.0
        return np.array(
            [
                0.0,  # visible
                float(self._last_cx),
                float(self._last_cy),
                float(self._last_area),
                float(self._last_depth_m),
                steps_norm,
                0.0,  # d_left
                0.0,  # d_center
                0.0,  # d_right
            ],
            dtype=np.float32,
        )

    def _build_duck_vision_observation(self, base_feature: np.ndarray) -> np.ndarray:
        base_feature = np.asarray(base_feature, dtype=np.float32).reshape(9)

        # Push into history (newest at index 0)
        if self.duck_vision_history_len <= 1:
            hist = base_feature.reshape(1, 9)
        else:
            self._vision_history[1:] = self._vision_history[:-1]
            self._vision_history[0] = base_feature
            self._vision_history_filled = min(
                self._vision_history_filled + 1, self.duck_vision_history_len
            )
            hist = self._vision_history

        hist_flat = hist.reshape(-1)

        if not self.duck_vision_use_deltas:
            return hist_flat.astype(np.float32)

        # Deltas: only meaningful if current and previous are visible; skeleton keeps 0.
        deltas = np.zeros((4,), dtype=np.float32)
        return np.concatenate([hist_flat, deltas], axis=0).astype(np.float32)

    # ----------------------------
    # Control publishing
    # ----------------------------

    def _publish_rc_override(self, action: np.ndarray) -> None:
        """
        Publish RC override via MAVROS.

        Mapping:
          roll/pitch/yaw in [-1,1] -> pwm_center +/- pwm_span
          throttle in [-1,1] -> [pwm_throttle_min, pwm_throttle_max]
        """
        if getattr(self, "_rc_pub", None) is None or self._msg_OverrideRCIn is None:
            # MAVROS message type not available; still allow stepping to build observations.
            return

        roll, pitch, yaw, thr = [float(x) for x in action.tolist()]

        pwm_roll = int(round(self._pwm_center + _clamp(roll, -1.0, 1.0) * self._pwm_span))
        pwm_pitch = int(round(self._pwm_center + _clamp(pitch, -1.0, 1.0) * self._pwm_span))
        pwm_yaw = int(round(self._pwm_center + _clamp(yaw, -1.0, 1.0) * self._pwm_span))

        # Throttle uses explicit min/max mapping.
        t01 = (_clamp(thr, -1.0, 1.0) * 0.5) + 0.5
        pwm_thr = int(round(self._pwm_throttle_min + t01 * (self._pwm_throttle_max - self._pwm_throttle_min)))

        msg = self._msg_OverrideRCIn()
        # MAVROS expects a list of 18 channels in ROS2 (often), but some builds use 8.
        # Fill with 0 (no override) and set the channels we care about.
        n = getattr(msg, "channels", None)
        if n is None:
            return

        # Determine length and create a list.
        try:
            ch_len = len(msg.channels)
        except Exception:
            ch_len = 8

        channels = [0] * ch_len

        def _set(ch_idx_1based: int, pwm: int) -> None:
            idx = int(ch_idx_1based) - 1
            if 0 <= idx < len(channels):
                channels[idx] = int(pwm)

        _set(self._rc_chan_roll, pwm_roll)
        _set(self._rc_chan_pitch, pwm_pitch)
        _set(self._rc_chan_throttle, pwm_thr)
        _set(self._rc_chan_yaw, pwm_yaw)

        msg.channels = channels
        try:
            self._rc_pub.publish(msg)
        except Exception:
            pass

    def _request_stream_rate_all(self, message_rate_hz: int = 30) -> bool:
        """
        Ask MAVROS to request MAVLink data streams.

        This is a best-effort helper: some FCU setups won't honor it, but when it works
        it fixes the common situation where /mavros/state updates but IMU/position topics
        remain silent.
        """
        client = getattr(self, "_srv_stream_rate", None)
        if client is None:
            return False
        try:
            # Wait briefly for service discovery.
            if not client.wait_for_service(timeout_sec=0.2):
                return False

            srv_type = client.srv_type
            req = srv_type.Request()
            # STREAM_ALL = 0
            req.stream_id = 0
            req.message_rate = int(max(1, message_rate_hz))
            req.on_off = True

            fut = client.call_async(req)
            # Spin until complete (bounded)
            t0 = time.monotonic()
            while time.monotonic() - t0 < 0.5:
                self._spin_once(timeout_sec=0.05)
                if fut.done():
                    break
            return bool(fut.done())
        except Exception:
            return False
