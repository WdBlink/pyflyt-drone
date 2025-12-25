import gymnasium as gym
import gymnasium as gym
import numpy as np

class RandomDuckOnResetWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        *,
        urdf_path: str,
        xy_radius: float,
        min_origin_distance: float = 5.0,
        base_z: float = 0.02,
        global_scaling: float = 0.7,
    ):
        super().__init__(env)
        self.urdf_path = str(urdf_path)
        self.xy_radius = float(xy_radius)
        self.min_origin_distance = float(min_origin_distance)
        self.base_z = float(base_z)
        self.global_scaling = float(global_scaling)
        self.duck_body_id: int | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._spawn_duck()
        return obs, info

    def _spawn_duck(self) -> None:
        base = self.env.unwrapped
        aviary = getattr(base, "env", None)
        if aviary is None:
            return

        if self.duck_body_id is not None:
            try:
                existing = {int(aviary.getBodyUniqueId(i)) for i in range(int(aviary.getNumBodies()))}
                if int(self.duck_body_id) in existing:
                    aviary.removeBody(int(self.duck_body_id))
            except Exception:
                pass
            self.duck_body_id = None

        rng = getattr(base, "_np_random", None)
        if rng is None:
            rng = np.random.default_rng()

        for _ in range(50):
            x = float(rng.uniform(-self.xy_radius, self.xy_radius))
            y = float(rng.uniform(-self.xy_radius, self.xy_radius))
            if (x * x + y * y) ** 0.5 >= self.min_origin_distance:
                break
        else:
            x, y = float(self.min_origin_distance), 0.0

        yaw = float(rng.uniform(-np.pi, np.pi))
        # 修正姿态：绕 X 轴旋转 90 度使其直立
        quat = aviary.getQuaternionFromEuler([np.pi / 2, 0.0, yaw])

        self.duck_body_id = int(
            aviary.loadURDF(
                self.urdf_path,
                basePosition=[x, y, self.base_z],
                baseOrientation=quat,
                useFixedBase=True,
                globalScaling=self.global_scaling,
            )
        )