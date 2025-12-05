
import gymnasium as gym
import numpy as np
import yaml
from envs.ab_fixedwing_env import make_fixedwing_ab_env
from PyFlyt.gym_envs.utils.flatten_waypoint_env import FlattenWaypointEnv

def test_reset_randomness():
    # 加载配置
    with open("configs/env.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 创建环境
    env = make_fixedwing_ab_env(
        render_mode=None,
        num_targets=cfg["num_targets"],
        goal_reach_distance=cfg["goal_reach_distance"],
        flight_dome_size=cfg["flight_dome_size"],
        max_duration_seconds=cfg["max_duration_seconds"],
        angle_representation=cfg["angle_representation"],
        agent_hz=cfg["agent_hz"],
        context_length=cfg["context_length"],
    )
    
    print("Testing reset randomness (Target Positions)...")
    
    def get_target_pos(env_instance):
        # 尝试从不同层级获取目标位置
        raw = env_instance
        
        # 解包 wrapper 直到找到 base env 或者含有 waypoints 属性的 env
        while hasattr(raw, 'env') and not hasattr(raw, 'waypoints'):
            raw = raw.env
        
        if hasattr(raw, 'waypoints'):
            if hasattr(raw.waypoints, 'targets'):
                # PyFlyt 的 targets 是一个 numpy array，我们拷贝一份以免引用改变
                return np.array(raw.waypoints.targets).copy()
        return None

    # Reset 1
    env.reset(seed=42)
    target1 = get_target_pos(env)
    print(f"Reset 1 (seed=42) Target: {target1}")
    
    # Reset 2
    env.reset(seed=42)
    target2 = get_target_pos(env)
    print(f"Reset 2 (seed=42) Target: {target2}")
    
    # Reset 3
    env.reset(seed=43)
    target3 = get_target_pos(env)
    print(f"Reset 3 (seed=43) Target: {target3}")
    
    # Reset 4
    env.reset()
    target4 = get_target_pos(env)
    print(f"Reset 4 (no seed) Target: {target4}")
    
    # Reset 5
    env.reset()
    target5 = get_target_pos(env)
    print(f"Reset 5 (no seed) Target: {target5}")

    if target1 is not None:
        if np.allclose(target1, target2):
             print("=> Seed 42 produces consistent targets.")
        else:
             print("=> Seed 42 produces DIFFERENT targets (Unexpected).")

        if not np.allclose(target1, target3):
             print("=> Different seeds produce different targets.")
        else:
             print("=> Different seeds produce SAME targets (Problematic).")
             
        if not np.allclose(target4, target5):
             print("=> No seed resets produce different targets (Good).")
        else:
             print("=> No seed resets produce SAME targets (Problematic).")
    else:
        print("Error: Could not extract target position.")

    env.close()

if __name__ == "__main__":
    test_reset_randomness()
