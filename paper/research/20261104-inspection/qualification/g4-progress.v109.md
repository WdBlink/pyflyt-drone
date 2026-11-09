# 正式主实验：第 58 条与 ID L2 的四条回执

主实验已验收 58/1296 条、完整场景 3/72，累计 59 次实际 600 s 尝试。ID L2 场景 `main-ID-L2-13` 已收 4/18 条。新增 `B3-M3/r0`、`B3-M2/r1` 均首次跑满 600 s / 144,000 步，H/E=1，最大时钟滞后 14.329、9.492 ms；两条新增实际本地模型用量 44 次、268,311 tokens。原一次时钟超限保留。

[第 57 条](formal-main-checkpoints.v1/after-0057.json)、[第 58 条](formal-main-checkpoints.v1/after-0058.json)和[逐槽放行复核](formal-main-checkpoints.v1/after-0058-rolling-guard-observation.v1.json)已封存。两次双读数均超过各自的容量门槛，实际派发后均重新挂回检查点。orders 32–57 的 26 次实际尝试仍在分项上界内。剩余经典 207 条、模型 1031 条，下一槽门槛 137,629,328,902 B。

下一槽 order58，`main-ID-L2-13/B2/r0`，本检查点未派发。round 123 仍限本场景 orders 54–71，任一守卫异常或 order72 即停止。全流程空间未获准，留出集、正式统计、图表及正文尚未开始。
