
import gymnasium as gym
import numpy as np
import yaml
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from envs.ab_fixedwing_env import make_fixedwing_ab_env

def _make_env_from_cfg(cfg: dict):
    def _thunk() -> gym.Env:
        return make_fixedwing_ab_env(
            render_mode=None,
            num_targets=cfg["num_targets"],
            goal_reach_distance=cfg["goal_reach_distance"],
            flight_dome_size=cfg["flight_dome_size"],
            max_duration_seconds=cfg["max_duration_seconds"],
            angle_representation=cfg["angle_representation"],
            agent_hz=cfg["agent_hz"],
            context_length=cfg["context_length"],
        )
    return _thunk

def get_target_pos(env_instance):
    # 尝试从不同层级获取目标位置
    raw = env_instance
    # 解包 wrapper
    while hasattr(raw, 'env') and not hasattr(raw, 'waypoints'):
        raw = raw.env
    
    if hasattr(raw, 'waypoints') and hasattr(raw.waypoints, 'targets'):
        return np.array(raw.waypoints.targets).copy()
    return None

def main():
    with open("configs/env.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    num_envs = 4
    thunks = [_make_env_from_cfg(cfg) for _ in range(num_envs)]
    
    # 使用 DummyVecEnv 测试 (串行)
    print("Testing DummyVecEnv (Serial)...")
    vec_env = DummyVecEnv(thunks)
    vec_env.reset()
    
    targets = []
    for i in range(num_envs):
        # DummyVecEnv可以直接访问 envs
        t = get_target_pos(vec_env.envs[i])
        targets.append(t)
        print(f"Env {i} Target: {t}")
    
    # Check if they are different
    all_diff = True
    for i in range(num_envs):
        for j in range(i+1, num_envs):
            if np.allclose(targets[i], targets[j]):
                print(f"Env {i} and Env {j} have SAME target!")
                all_diff = False
    
    if all_diff:
        print("DummyVecEnv: All targets different (Good).")
    
    vec_env.close()
    
    # 使用 SubprocVecEnv 测试 (并行)
    # 注意：SubprocVecEnv 无法直接访问 envs[i].waypoints，因为它们在不同进程
    # 我们只能通过 info 或者 wrapper 来获取，或者假设如果 Dummy 是对的，Subproc 只要 seed 不同也是对的
    # 但这里我们想验证是否发生了 fork 导致的 RNG 重复
    
    # 为了验证 SubprocVecEnv，我们可以在 step 的 info 中返回 target
    # 但不想修改环境代码。
    # 我们可以看 reset 返回的 observation。如果是 random target，obs 应该包含 target_deltas
    # FlattenEnv 把 target_deltas 放在 obs 的后半部分。
    
    print("\nTesting SubprocVecEnv (Parallel)...")
    vec_env = SubprocVecEnv(thunks)
    obs = vec_env.reset()
    
    # obs shape: (num_envs, obs_dim)
    # 比较 obs 是否相同
    print(f"Obs shape: {obs.shape}")
    
    all_diff = True
    for i in range(num_envs):
        print(f"Env {i} Obs Sample: {obs[i, -5:]}") # 看最后几个值，通常是 target 相关
        for j in range(i+1, num_envs):
            if np.allclose(obs[i], obs[j]):
                print(f"Env {i} and Env {j} have SAME observation (Potential RNG duplication)!")
                all_diff = False
                
    if all_diff:
        print("SubprocVecEnv: All observations different (Good).")
    else:
        print("SubprocVecEnv: Some observations identical (Bad).")

    vec_env.close()

if __name__ == "__main__":
    main()
