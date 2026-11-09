# 正式主实验：第 67 条与 ID L2 的逐槽容量复核

主实验已验收 67/1296 条、完整场景 3/72，累计 68 次实际 600 s 尝试。ID L2 已收 13/18 条。新增 `B4-M2/r1`、`B4-M1/r2` 均首次跑满 600 s / 144,000 步，H/E=1，最大时钟滞后 8.457、8.998 ms；新增实际本地模型用量 43 次、293,116 tokens。原一次时钟超限保留。

[第 66 条](formal-main-checkpoints.v1/after-0066.json)、[第 67 条](formal-main-checkpoints.v1/after-0067.json)和[逐槽放行记录](formal-main-checkpoints.v1/after-0067-buffered-rolling-observation.v1.json)已封存。两次双读数均超过各自门槛，实际派发后均重挂检查点。最近 35 次实际尝试都在分项上界内；下一槽门槛 137,135,236,780 B。

下一槽 order67，`main-ID-L2-13/B3-M3/r2`，本检查点未派发。round 126 仍限本场景 orders 59–71，任一异常或 order72 即停止。全流程容量未获准，留出集、正式统计、图表及正文尚未开始。
