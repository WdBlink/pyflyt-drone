"""项目文件：训练回调与评估工具

作者: wdblink

说明:
    提供自定义回调，用于在训练过程中记录评估指标并保存最佳模型。
    目前实现对成功率与平均回报的简单统计包装，复用 SB3 的 EvalCallback。
"""

from __future__ import annotations

from typing import Any, Dict

from stable_baselines3.common.callbacks import BaseCallback


class InfoLoggerCallback(BaseCallback):
    """示例信息回调，用于扩展记录训练过程中的自定义信息。"""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 这里可以添加自定义日志，如从 env.get_attr 拉取统计信息
        return True

