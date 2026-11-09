# 正式主实验：ID L2 首两条与逐槽容量复算

主实验已验收 56/1296 条、完整场景 3/72，累计 57 次实际 600 s 尝试。首个 ID L2 场景 `main-ID-L2-13` 新增 `B4-M1/r1`、`B3-M2/r0` 均首次跑满 600 s / 144,000 步，H/E=1，最大时钟滞后为 8.621、10.058 ms；两条新增实际本地模型用量 45 次、271,525 tokens。原一次时钟超限保留。

[第 55 条](formal-main-checkpoints.v1/after-0055.json)、[第 56 条](formal-main-checkpoints.v1/after-0056.json)和[逐槽放行复核](formal-main-checkpoints.v1/after-0056-rolling-guard-observation.v1.json)已封存。两次双读数均符合当时主阶段门槛；实际派发后均重新挂回检查点。orders 32–55 的 24 次实际尝试均在更新分项上界内。剩余经典 207 条、模型 1033 条，下一槽门槛 137,807,284,646 B，只按已验收回执滚动。

下一槽 order56，`main-ID-L2-13/B3-M3/r0`，本检查点未派发。round 123 仍只覆盖本场景 orders 54–71，任一守卫异常或 order72 即暂停。全流程空间未获准，留出集、正式统计、图表及正文尚未开始。
