"""项目文件：固定翼 A→B 环境渲染演示

作者: wdblink

说明:
    启动并渲染固定翼环境，演示从 reset 到 step 的基本循环。
    该脚本不做训练，仅用于可视化检查环境与扁平化适配是否正常。
"""

from __future__ import annotations

import yaml

from envs.ab_fixedwing_env import make_fixedwing_ab_env


def main() -> None:
    """运行渲染演示主函数。"""

    with open("configs/env.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    env = make_fixedwing_ab_env(
        render_mode=cfg["render_mode"],
        num_targets=cfg["num_targets"],
        goal_reach_distance=cfg["goal_reach_distance"],
        flight_dome_size=cfg["flight_dome_size"],
        max_duration_seconds=cfg["max_duration_seconds"],
        angle_representation=cfg["angle_representation"],
        agent_hz=cfg["agent_hz"],
        context_length=cfg["context_length"],
    )

    term, trunc = False, False
    obs, _ = env.reset()

    steps = 0
    while not (term or trunc) and steps < 2000:
        # 使用随机动作进行演示
        action = env.action_space.sample()
        obs, rew, term, trunc, _info = env.step(action)
        steps += 1

    env.close()


if __name__ == "__main__":
    main()

