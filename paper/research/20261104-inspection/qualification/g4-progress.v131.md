# 正式主实验：已验收 91 条回调

当前 91/1296 条回调、5/72 个完整场景。新增 [第 91 条](formal-main-checkpoints.v1/after-0091.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L2-08/B1/r0`：1 次尝试，末次峰值滞后 11.263 ms，H/E=1/1。新增通过的 LLM 尝试用量 0 次、0 tokens；所有尝试记录均保留。

[原始现场记录](formal-main-checkpoints.v1/after-0091-main-rolling-observation.v1.json)保留刚完成的非 LLM 回调和放行回执。旧守卫因实测留存 29,800,519 B 超过 29,546,687 B 在 order91 停止。复查最近 61 次尝试，仅该分项超出。此回调继续计入主矩阵，下一槽 `main-ID-L2-08/B3-M1/r0` 尚未派发。

[新主阶段投影](formal-main-checkpoints.v1/after-0091-main-phase-wire-projection.v1.json)将非 LLM 单次 raw 上界设为 32 MiB，其余尝试分项不变。按剩余 200 classic、1,005 LLM 的两次尝试、原生文件和所有命令记录、暂存与固定 1 GiB/10 GiB，起始门槛为 125,090,872,760 B。独立同卷双读 396550246400 → 396549570560 B。

[准入决定](formal-main-checkpoints.v1/after-0091-classic-buffered-rolling-capacity-admission.v1.json)只覆盖原范围内 orders 91–125。完成原生登记后，由[新守卫](guard-formal-main-rolling-classic-v3.py)先只读核验，之后仍逐槽双读、真实派发后重挂。留出集仍封存，正式统计、图表及正文尚未开始。
