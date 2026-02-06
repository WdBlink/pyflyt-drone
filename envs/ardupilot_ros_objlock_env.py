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


def _quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion -> 3x3 rotation matrix (body->world)."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


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

    # Prefer SERVO_OUTPUT_RAW (true actuator outputs). Keep RCOut as a fallback.
    # We also keep a "selected" view (servo_pwm) which prefers servo_output_raw when available.
    servo_pwm_servo_raw: Optional[np.ndarray] = None  # (N,)
    servo_pwm_rc_out: Optional[np.ndarray] = None  # (N,)
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


def _select_cuda_if_available(requested: str) -> str:
    """
    Best-effort device selection for Ultralytics models.
    If user requests CUDA but torch/CUDA isn't available, fall back to CPU.
    """
    req = str(requested or "").strip()
    if not req:
        return "cpu"
    if req.lower().startswith("cuda"):
        try:
            import torch  # type: ignore

            if bool(torch.cuda.is_available()):
                return req
        except Exception:
            pass
        return "cpu"
    return req


_BaseGymEnv = gym.Env if gym is not None else object


class ArdupilotGazeboObjLockEnv(_BaseGymEnv):
    """
    Gymnasium-compatible environment that reads observations from ROS/MAVROS.

    Observation (Dict):
      - attitude: [ang_vel(3), ang_pos_euler(3) or quaternion(4), lin_vel(3), lin_pos(3),
                   prev_action(4), aux_state(6)]
      - target_vector: optional goal vector (or zeros) depending on goal settings
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
        # Vision backend (segment RGB -> mask -> duck_vision).
        vision_backend: Literal["fastsam", "none"] = "fastsam",
        fastsam_weights_path: Optional[str] = None,
        fastsam_device: str = "cuda",
        fastsam_retina_masks: bool = True,
        fastsam_imgsz: int = 1024,
        fastsam_conf: float = 0.9,
        fastsam_iou: float = 0.9,
        fastsam_text_prompt: Optional[str] = "a photo of a yellow duck",
        # How often to run segmentation (in env steps). 1 = every step.
        seg_interval_steps: int = 2,
        # Depth scaling for 16UC1 images (assume millimeters by default).
        depth_scale_16u: float = 0.001,
        # Topic names (ROS2 defaults are user-setup dependent; treat these as config knobs)
        mavros_imu_topic: str = "/mavros/imu/data",
        mavros_odom_topic: str = "/mavros/local_position/odom",
        mavros_vel_topic: str = "/mavros/local_position/velocity_local",
        mavros_rc_override_topic: str = "/mavros/rc/override",
        mavros_servo_output_topic: str = "/mavros/servo_output/raw",
        mavros_rc_out_topic: str = "/mavros/rc/out",
        # If you use ardupilot_gazebo's run_vtail_with_ros_bridge.sh, these are the defaults.
        rgb_image_topic: Optional[str] = "/pod_camera/image_raw",
        depth_image_topic: Optional[str] = "/pod_depth_camera/image_raw",
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
        # Optional pre-vision goal pursuit settings.
        goal_position_enu: Optional[tuple[float, float, float]] = None,
        goal_vector_in_body_frame: bool = True,
        goal_switch_on_vision: bool = True,
        goal_visible_hold_steps: int = 3,
        goal_reacquire_steps: int = 15,
        goal_reach_distance_m: float = 0.0,
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
        self.vision_backend = str(vision_backend).lower()
        self.fastsam_weights_path = fastsam_weights_path
        self.fastsam_device = _select_cuda_if_available(str(fastsam_device))
        self.fastsam_retina_masks = bool(fastsam_retina_masks)
        self.fastsam_imgsz = int(fastsam_imgsz)
        self.fastsam_conf = float(fastsam_conf)
        self.fastsam_iou = float(fastsam_iou)
        self.fastsam_text_prompt = fastsam_text_prompt
        self.seg_interval_steps = int(max(1, seg_interval_steps))
        self.depth_scale_16u = float(depth_scale_16u)

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
        self._goal_pos_enu = (
            np.asarray(goal_position_enu, dtype=np.float64).reshape(3)
            if goal_position_enu is not None
            else None
        )
        self._goal_vector_in_body_frame = bool(goal_vector_in_body_frame)
        self._goal_switch_on_vision = bool(goal_switch_on_vision)
        self._goal_visible_hold_steps = int(max(1, goal_visible_hold_steps))
        self._goal_reacquire_steps = int(max(1, goal_reacquire_steps))
        self._goal_reach_distance_m = float(max(0.0, goal_reach_distance_m))

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
        self._latest_rgb = None
        self._latest_depth = None
        self._latest_rgb_t = 0.0
        self._latest_depth_t = 0.0

        self._seg_last_mask = None
        self._seg_last_step = -1
        self._seg_last_t = 0.0

        # FastSAM lazy-load
        self._fastsam_model = None
        self._fastsam_model_failed = False

        self._last_aux_state = np.zeros((6,), dtype=np.float32)
        self._last_servo_pwm_used_for_aux: Optional[np.ndarray] = None
        self._last_target_vector = np.zeros((3,), dtype=np.float32)
        self._last_goal_distance_m = 0.0
        self._goal_suppressed_by_vision = False
        self._goal_visible_steps = 0
        self._goal_lost_steps = 0
        self._tracking_mode = "goal_vector" if self._goal_pos_enu is not None else "vision_only"

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
        rgb = self._ros_image_to_rgb(msg)
        if rgb is None:
            return
        with self._lock:
            self._latest_rgb = rgb
            self._latest_rgb_t = time.monotonic()

    def _on_depth(self, msg) -> None:
        self._latest_depth_msg = msg
        depth = self._ros_image_to_depth(msg)
        if depth is None:
            return
        with self._lock:
            self._latest_depth = depth
            self._latest_depth_t = time.monotonic()

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
                arr = np.asarray(vals, dtype=np.int32)
                # SERVO_OUTPUT_RAW is the preferred source for aux_state (actual actuator outputs).
                self._latest.servo_pwm_servo_raw = arr
                self._latest.servo_pwm = arr

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
                    arr = np.asarray([int(x) for x in ch], dtype=np.int32)
                    self._latest.servo_pwm_rc_out = arr
                    # Only use RCOut if we don't have SERVO_OUTPUT_RAW.
                    if self._latest.servo_pwm_servo_raw is None:
                        self._latest.servo_pwm = arr
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
        self._last_target_vector[:] = 0.0
        self._last_goal_distance_m = 0.0
        self._goal_suppressed_by_vision = False
        self._goal_visible_steps = 0
        self._goal_lost_steps = 0
        self._tracking_mode = "goal_vector" if self._goal_pos_enu is not None else "vision_only"

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
            "tracking_mode": str(self._tracking_mode),
            "goal_distance_m": float(self._last_goal_distance_m),
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
            "tracking_mode": str(self._tracking_mode),
            "goal_distance_m": float(self._last_goal_distance_m),
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
            servo_servo_raw = None
            if self._latest.servo_pwm_servo_raw is not None:
                servo_servo_raw = self._latest.servo_pwm_servo_raw.copy()
            servo_rc_out = None
            if self._latest.servo_pwm_rc_out is not None:
                servo_rc_out = self._latest.servo_pwm_rc_out.copy()
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
                "servo_pwm_servo_raw": servo_servo_raw,
                "servo_pwm_rc_out": servo_rc_out,
                "servo_pwm_used_for_aux": None
                if self._last_servo_pwm_used_for_aux is None
                else self._last_servo_pwm_used_for_aux.copy(),
                "aux_state": self._last_aux_state.copy(),
                "mavros_connected": bool(self._mavros_state.connected),
                "mavros_mode": str(self._mavros_state.mode),
                "have_rgb": self._latest_rgb is not None,
                "have_depth": self._latest_depth is not None,
                "target_vector": self._last_target_vector.copy(),
                "goal_distance_m": float(self._last_goal_distance_m),
                "tracking_mode": str(self._tracking_mode),
            }

    # ----------------------------
    # Observation building
    # ----------------------------

    def _build_obs(self) -> dict[str, np.ndarray]:
        attitude = self._build_attitude()
        duck_vision = self._build_duck_vision_observation(self._compute_vision_features())
        visible = bool(duck_vision.shape[0] > 0 and float(duck_vision[0]) > 0.5)
        target_vector = self._compute_target_vector(vision_visible=visible)
        return {
            "attitude": attitude,
            "target_vector": target_vector,
            "duck_vision": duck_vision,
        }

    def _compute_target_vector(self, vision_visible: bool) -> np.ndarray:
        """
        Optional privileged guidance:
        - Before stable visual lock: vector to fixed ENU goal.
        - After stable visual lock: suppress goal vector so policy can rely on vision.
        """
        if self._goal_pos_enu is None:
            self._tracking_mode = "vision_only"
            self._last_goal_distance_m = 0.0
            self._last_target_vector[:] = 0.0
            return self._last_target_vector.copy()

        # Hysteresis for vision switch.
        if self._goal_switch_on_vision:
            if vision_visible:
                self._goal_visible_steps += 1
                self._goal_lost_steps = 0
                if self._goal_visible_steps >= self._goal_visible_hold_steps:
                    self._goal_suppressed_by_vision = True
            else:
                self._goal_lost_steps += 1
                self._goal_visible_steps = 0
                if self._goal_suppressed_by_vision and self._goal_lost_steps >= self._goal_reacquire_steps:
                    self._goal_suppressed_by_vision = False
        else:
            self._goal_suppressed_by_vision = False

        with self._lock:
            lin_pos = None if self._latest.lin_pos is None else self._latest.lin_pos.copy()
            quat = None if self._latest.quat_xyzw is None else self._latest.quat_xyzw.copy()

        if lin_pos is None:
            self._tracking_mode = "goal_wait_pose"
            self._last_goal_distance_m = 0.0
            self._last_target_vector[:] = 0.0
            return self._last_target_vector.copy()

        diff_world = self._goal_pos_enu.astype(np.float64) - lin_pos.astype(np.float64)
        dist = float(np.linalg.norm(diff_world))
        self._last_goal_distance_m = dist

        if self._goal_reach_distance_m > 0.0 and dist <= self._goal_reach_distance_m:
            self._tracking_mode = "goal_reached"
            self._last_target_vector[:] = 0.0
            return self._last_target_vector.copy()

        if self._goal_suppressed_by_vision:
            self._tracking_mode = "vision_target"
            self._last_target_vector[:] = 0.0
            return self._last_target_vector.copy()

        if self._goal_vector_in_body_frame and quat is not None and quat.shape[0] == 4:
            rot_bw = _quat_to_rotmat(
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            )
            out = (rot_bw.T @ diff_world).astype(np.float32)
            self._tracking_mode = "goal_vector_body"
        else:
            out = diff_world.astype(np.float32)
            self._tracking_mode = "goal_vector_world"

        self._last_target_vector[:] = out
        return self._last_target_vector.copy()

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

    def _compute_vision_features(self) -> np.ndarray:
        """
        Returns the base 9D vision feature:
          [visible, cx, cy, area, depth, steps_norm, d_left, d_center, d_right]

        Vision behavior:
        - If a segmentation backend is available and RGB is received, compute mask.
        - If depth is available, estimate distance and obstacle ranges.
        - Otherwise, return "not visible" with a decaying steps_norm.
        """
        rgb, depth = self._get_latest_images()

        if rgb is None:
            return self._build_vision_vector(visible=False, obstacle_depths=(0.0, 0.0, 0.0))

        mask = self._segment_duck_mask(rgb)
        if mask is None or mask.size == 0:
            obs_d = self._estimate_obstacle_zone_distances_m(depth, None)
            return self._build_vision_vector(visible=False, obstacle_depths=obs_d)

        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(bool)

        mask_for_depth = mask
        # Bridge outputs RGB (1280x720) and depth (often 641x480) with different sizes.
        # Align mask to depth shape for depth-based features to avoid boolean index mismatch.
        if depth is not None:
            d2 = depth[..., 0] if depth.ndim == 3 else depth
            if d2 is not None and d2.shape != mask.shape:
                try:
                    from PIL import Image

                    mask_for_depth = (
                        np.asarray(
                            Image.fromarray(mask.astype(np.uint8)).resize(
                                (int(d2.shape[1]), int(d2.shape[0])), resample=Image.NEAREST
                            )
                        )
                        > 0
                    )
                except Exception:
                    mask_for_depth = None

        obs_d = self._estimate_obstacle_zone_distances_m(depth, mask_for_depth)

        if not np.any(mask):
            return self._build_vision_vector(visible=False, obstacle_depths=obs_d)

        # Found target
        h, w = mask.shape
        ys, xs = np.nonzero(mask)
        self._last_cx = float(np.mean(xs)) / float(max(1, w - 1))
        self._last_cy = float(np.mean(ys)) / float(max(1, h - 1))
        self._last_area = float(np.count_nonzero(mask)) / float(max(1, h * w))

        if depth is not None and mask_for_depth is not None:
            depth_m = self._estimate_distance(depth, mask_for_depth)
            if depth_m is not None and np.isfinite(depth_m):
                self._last_depth_m = float(depth_m)
                self._steps_since_seen = 0
                return self._build_vision_vector(visible=True, obstacle_depths=obs_d)

        # Visible but no depth
        self._steps_since_seen = 0
        return self._build_vision_vector(visible=True, obstacle_depths=obs_d)

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

        # Deltas: only meaningful if current and previous are visible.
        deltas = np.zeros((4,), dtype=np.float32)
        if (
            self._vision_history_filled >= 2
            and float(base_feature[0]) > 0.5
            and float(self._vision_history[1][0]) > 0.5
        ):
            prev = self._vision_history[1]
            deltas[0] = float(base_feature[1] - prev[1])
            deltas[1] = float(base_feature[2] - prev[2])
            deltas[2] = float(base_feature[3] - prev[3])
            deltas[3] = float(base_feature[4] - prev[4])
        return np.concatenate([hist_flat, deltas], axis=0).astype(np.float32)

    # ----------------------------
    # Vision helpers (ROS Image -> numpy, segmentation, feature extraction)
    # ----------------------------

    def _get_latest_images(self) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        with self._lock:
            rgb = None if self._latest_rgb is None else self._latest_rgb.copy()
            depth = None if self._latest_depth is None else self._latest_depth.copy()
        return rgb, depth

    def _ros_image_to_rgb(self, msg) -> Optional[np.ndarray]:
        try:
            h = int(msg.height)
            w = int(msg.width)
            encoding = str(msg.encoding).lower()
            data = msg.data
        except Exception:
            return None

        if h <= 0 or w <= 0:
            return None

        # Try cv_bridge if available
        try:
            from cv_bridge import CvBridge  # type: ignore

            bridge = CvBridge()
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            return np.asarray(cv_img)
        except Exception:
            pass

        # Manual decode
        if encoding in ("rgb8", "bgr8"):
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
            if encoding == "bgr8":
                img = img[..., ::-1]
            return img
        if encoding in ("rgba8", "bgra8"):
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 4)
            if encoding == "bgra8":
                img = img[..., [2, 1, 0, 3]]
            return img[..., :3]
        if encoding in ("mono8",):
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            return np.stack([img, img, img], axis=-1)
        if encoding in ("mono16", "16uc1"):
            img = np.frombuffer(data, dtype=np.uint16).reshape(h, w)
            # Scale to 8-bit for segmentation if needed
            img8 = np.clip(img / 256.0, 0, 255).astype(np.uint8)
            return np.stack([img8, img8, img8], axis=-1)
        return None

    def _ros_image_to_depth(self, msg) -> Optional[np.ndarray]:
        try:
            h = int(msg.height)
            w = int(msg.width)
            encoding = str(msg.encoding).lower()
            data = msg.data
            is_be = bool(getattr(msg, "is_bigendian", False))
        except Exception:
            return None

        if h <= 0 or w <= 0:
            return None

        if encoding in ("32fc1", "32f", "float32"):
            arr = np.frombuffer(data, dtype=np.float32).reshape(h, w)
        elif encoding in ("16uc1", "mono16"):
            arr = np.frombuffer(data, dtype=np.uint16).reshape(h, w).astype(np.float32)
            arr = arr * float(self.depth_scale_16u)
        else:
            # Best-effort: try float32
            try:
                arr = np.frombuffer(data, dtype=np.float32).reshape(h, w)
            except Exception:
                return None

        if is_be:
            arr = arr.byteswap().newbyteorder()
        return arr

    def _resolve_default_fastsam_weights_path(self) -> Optional[str]:
        # Try repo root FastSAM-s.pt if present.
        try:
            import os

            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            candidate = os.path.join(repo_root, "FastSAM-s.pt")
            if os.path.exists(candidate):
                return candidate
        except Exception:
            pass
        return None

    def _ensure_fastsam_loaded(self) -> None:
        if self._fastsam_model is not None or self._fastsam_model_failed:
            return
        weights = self.fastsam_weights_path or self._resolve_default_fastsam_weights_path()
        if not weights:
            self._fastsam_model_failed = True
            return
        try:
            from ultralytics import FastSAM  # type: ignore

            self._fastsam_model = FastSAM(weights)
        except Exception:
            self._fastsam_model = None
            self._fastsam_model_failed = True

    def _fastsam_results_to_binary_mask(self, results_obj) -> Optional[np.ndarray]:
        try:
            masks = getattr(results_obj, "masks", None)
            if masks is None or masks.data is None:
                return None
            data = masks.data
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            if data.ndim == 3:
                return (data > 0.5).any(axis=0).astype(np.uint8)
            if data.ndim == 2:
                return (data > 0.5).astype(np.uint8)
            return None
        except Exception:
            return None

    def _segment_duck_mask(self, rgb: np.ndarray) -> Optional[np.ndarray]:
        if self.vision_backend == "none":
            return None
        if self.vision_backend != "fastsam":
            return None

        # Cache by step to reduce compute.
        if self._seg_last_mask is not None and (self._step_count - self._seg_last_step) < self.seg_interval_steps:
            return self._seg_last_mask

        self._ensure_fastsam_loaded()
        if self._fastsam_model is None:
            return None

        try:
            inference_kwargs = {
                "device": self.fastsam_device,
                "retina_masks": bool(self.fastsam_retina_masks),
                "imgsz": int(self.fastsam_imgsz),
                "conf": float(self.fastsam_conf),
                "iou": float(self.fastsam_iou),
            }
            if self.fastsam_text_prompt:
                inference_kwargs["texts"] = str(self.fastsam_text_prompt)

            results = self._fastsam_model(rgb, **inference_kwargs)
            mask_np = self._fastsam_results_to_binary_mask(results[0]) if results else None
            if mask_np is None:
                return None
            if mask_np.shape[:2] != rgb.shape[:2]:
                try:
                    from PIL import Image

                    mask_np = np.asarray(
                        Image.fromarray(mask_np).resize(
                            (rgb.shape[1], rgb.shape[0]), resample=Image.NEAREST
                        )
                    )
                except Exception:
                    return None
            self._seg_last_mask = mask_np.astype(np.uint8)
            self._seg_last_step = self._step_count
            self._seg_last_t = time.monotonic()
            return self._seg_last_mask
        except Exception:
            return None

    def _build_vision_vector(
        self, visible: bool, obstacle_depths: tuple[float, float, float]
    ) -> np.ndarray:
        if not visible:
            self._steps_since_seen = min(self._steps_since_seen + 1, 60)
        steps_norm = float(self._steps_since_seen) / 60.0
        d_left, d_center, d_right = obstacle_depths
        return np.array(
            [
                1.0 if visible else 0.0,
                float(self._last_cx),
                float(self._last_cy),
                float(self._last_area),
                float(self._last_depth_m),
                steps_norm,
                float(d_left),
                float(d_center),
                float(d_right),
            ],
            dtype=np.float32,
        )

    def _estimate_obstacle_zone_distances_m(
        self, depth: Optional[np.ndarray], target_mask: Optional[np.ndarray]
    ) -> tuple[float, float, float]:
        if depth is None:
            return 0.0, 0.0, 0.0
        if depth.ndim == 3:
            depth = depth[..., 0]

        h, w = depth.shape
        y_mid = int(h // 2)
        x_1 = int(w // 3)
        x_2 = int(2 * w // 3)

        if target_mask is None:
            mask = np.ones_like(depth, dtype=bool)
        else:
            if target_mask.ndim == 3:
                target_mask = target_mask[..., 0]
            mask = ~target_mask.astype(bool)

        def zone_mean_depth(x_start: int, x_end: int) -> float:
            zone_mask = mask[y_mid, x_start:x_end]
            if not np.any(zone_mask):
                return 0.0
            vals = depth[y_mid, x_start:x_end][zone_mask]
            vals = vals[np.isfinite(vals) & (vals > 0.0)]
            if vals.size == 0:
                return 0.0
            return float(np.mean(vals))

        d_left = zone_mean_depth(0, x_1)
        d_center = zone_mean_depth(x_1, x_2)
        d_right = zone_mean_depth(x_2, w)
        return d_left, d_center, d_right

    def _estimate_distance(self, depth: np.ndarray, mask: np.ndarray) -> Optional[float]:
        if depth.ndim == 3:
            depth = depth[..., 0]
        vals = depth[mask]
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size == 0:
            return None
        # Use median for stability
        return float(np.median(vals))

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
