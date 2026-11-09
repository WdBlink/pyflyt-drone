# 正式主实验：第 61 条与双侧容量余量

主实验已验收 61/1296 条、完整场景 3/72，共 62 次实际 600 s 尝试。ID L2 场景已收 7/18 条。新增 `B3-M1/r2`、`B4-M1/r0` 均首次跑满 600 s / 144,000 步，H/E=1，最大时钟滞后 9.711、9.043 ms；两条新增实际本地模型用量 39 次、266,327 tokens。原一次时钟超限保留。

[第 60 条](formal-main-checkpoints.v1/after-0060.json)、[第 61 条](formal-main-checkpoints.v1/after-0061.json)与[逐槽放行复核](formal-main-checkpoints.v1/after-0061-buffered-rolling-observation.v1.json)已封存。两次双读数均超过各自滚动门槛；实际派发后均重新挂回检查点。orders 32–60 的 29 次实际尝试仍在现有分项上界内。剩余经典 206 条、模型 1029 条，下一槽门槛 137,609,335,016 B。

下一槽 order61，`main-ID-L2-13/B4-M2/r0`，本检查点未派发。round 126 仍限本场景 orders 59–71，任一异常或 order72 即停止。全流程空间未获准，留出集、正式统计、图表和正文尚未开始。
