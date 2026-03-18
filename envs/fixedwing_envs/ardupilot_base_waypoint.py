"""Waypoint handler decoupled from PyFlyt rendering/Bullet visuals."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class WaypointHandler:
    """Lightweight waypoint handler for ArduPilot+Gazebo pipelines.

    Keeps the same public interface used by existing environments, but:
    - only samples/manages waypoint coordinates,
    - does not spawn visual target bodies in Bullet.
    """

    def __init__(
        self,
        enable_render: bool,
        num_targets: int,
        use_yaw_targets: bool,
        goal_reach_distance: float,
        goal_reach_angle: float,
        flight_dome_size: float,
        min_height: float,
        np_random: np.random.Generator,
    ):
        self.enable_render = enable_render
        self.num_targets = num_targets
        self.use_yaw_targets = use_yaw_targets
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.flight_dome_size = flight_dome_size
        self.min_height = min_height
        self.np_random = np_random

        self.p: Any = None
        self.targets: np.ndarray = np.zeros((0, 3), dtype=np.float64)
        self.yaw_targets: np.ndarray = np.zeros((0,), dtype=np.float64)
        self.new_distance = np.inf
        self.old_distance = np.inf
        self.yaw_error_scalar = np.inf

    def reset(
        self,
        p: Any,
        np_random: None | np.random.Generator = None,
    ) -> None:
        """Reset waypoints.

        Args:
            p: kept for interface compatibility; may provide quaternion->matrix helper.
            np_random: optional RNG override.
        """
        self.p = p
        if np_random is not None:
            self.np_random = np_random

        self.new_distance = np.inf
        self.old_distance = np.inf
        self.yaw_error_scalar = np.inf

        self.targets = np.zeros(shape=(self.num_targets, 3), dtype=np.float64)
        thetas = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        phis = self.np_random.uniform(0.0, 2.0 * math.pi, size=(self.num_targets,))
        for i, theta, phi in zip(range(self.num_targets), thetas, phis):
            dist = self.np_random.uniform(low=1.0, high=self.flight_dome_size * 0.9)
            x = dist * math.sin(phi) * math.cos(theta)
            y = dist * math.sin(phi) * math.sin(theta)
            z = abs(dist * math.cos(phi))
            self.targets[i] = np.array(
                [x, y, z if z > self.min_height else self.min_height],
                dtype=np.float64,
            )

        if self.use_yaw_targets:
            self.yaw_targets = self.np_random.uniform(
                low=-math.pi, high=math.pi, size=(self.num_targets,)
            ).astype(np.float64)
        else:
            self.yaw_targets = np.zeros((0,), dtype=np.float64)

    @staticmethod
    def _quat_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
        """Fallback quaternion(x, y, z, w) -> 3x3 rotation matrix."""
        q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
        if q.shape[0] != 4:
            return np.eye(3, dtype=np.float64)

        x, y, z, w = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z

        return np.array(
            [
                [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
                [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
                [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
            ],
            dtype=np.float64,
        )

    def _rotation_from_quaternion(self, quaternion: np.ndarray) -> np.ndarray:
        if self.p is not None and hasattr(self.p, "getMatrixFromQuaternion"):
            try:
                mat = np.asarray(self.p.getMatrixFromQuaternion(quaternion), dtype=np.float64)
                return mat.reshape(3, 3)
            except Exception:
                pass
        return self._quat_xyzw_to_matrix(quaternion)

    @property
    def distance_to_next_target(self) -> float:
        return self.new_distance

    def distance_to_targets(
        self,
        ang_pos: np.ndarray,
        lin_pos: np.ndarray,
        quaternion: np.ndarray,
    ) -> np.ndarray:
        rotation = self._rotation_from_quaternion(quaternion)

        targets_arr = np.asarray(self.targets, dtype=np.float64)
        if targets_arr.ndim != 2 or targets_arr.shape[0] == 0:
            self.old_distance = self.new_distance
            self.new_distance = np.inf
            if self.use_yaw_targets:
                return np.zeros((0, 4), dtype=np.float64)
            return np.zeros((0, 3), dtype=np.float64)

        target_deltas = np.matmul((targets_arr - lin_pos), rotation)

        self.old_distance = self.new_distance
        self.new_distance = float(np.linalg.norm(target_deltas[0]))

        if self.use_yaw_targets:
            yaw_targets_arr = np.asarray(self.yaw_targets, dtype=np.float64).reshape(-1)
            yaw_errors = yaw_targets_arr - float(ang_pos[-1])
            yaw_errors[yaw_errors > math.pi] -= 2.0 * math.pi
            yaw_errors[yaw_errors < -math.pi] += 2.0 * math.pi
            yaw_errors = yaw_errors[..., None]
            target_deltas = np.concatenate([target_deltas, yaw_errors], axis=-1)
            self.yaw_error_scalar = float(np.abs(yaw_errors[0])) if yaw_errors.size else np.inf

        return target_deltas

    @property
    def progress_to_next_target(self) -> float:
        if np.any(np.isinf(self.old_distance + self.new_distance)):
            return 0.0
        return float(self.old_distance - self.new_distance)

    @property
    def target_reached(self) -> bool:
        if not self.new_distance < self.goal_reach_distance:
            return False
        if not self.use_yaw_targets:
            return True
        return bool(self.yaw_error_scalar < self.goal_reach_angle)

    def advance_targets(self) -> None:
        targets_arr = np.asarray(self.targets, dtype=np.float64)
        if targets_arr.ndim == 2 and targets_arr.shape[0] > 1:
            self.targets = targets_arr[1:]
            if self.use_yaw_targets:
                self.yaw_targets = np.asarray(self.yaw_targets, dtype=np.float64)[1:]
        else:
            self.targets = np.zeros((0, 3), dtype=np.float64)
            self.yaw_targets = np.zeros((0,), dtype=np.float64)

    @property
    def num_targets_reached(self) -> int:
        return int(self.num_targets - len(self.targets))

    @property
    def all_targets_reached(self) -> bool:
        return len(self.targets) == 0
