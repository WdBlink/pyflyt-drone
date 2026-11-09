# 正式主实验：第 46 条与逐槽空间记录

主实验已验收 46/1296 条、完整场景 2/72，累计 47 次实际 600 s 尝试。首个 OOD L2 场景已收 10/18 条。新增 `B3-M1/r1` 与 `B3-M3/r1` 均首次完成 600 s / 144,000 步，H/E=1，最大时钟滞后 12.359 和 8.832 ms；新增本地模型用量 35 次、245,844 tokens。原一次时钟超限保留。

[第 45 条](formal-main-checkpoints.v1/after-0045.json)、[第 46 条](formal-main-checkpoints.v1/after-0046.json)和[逐槽空间记录](formal-main-checkpoints.v1/after-0046-buffered-guard-observation.v1.json)已封存。两次放行实测 144,300,945,408 B、144,248,188,928 B，均超过 139,100,000,000 B；实际派发后均重挂检查点。orders 32–45 的 14 次尝试复查后，各分项均在更新上界内。round 116 决定范围维持为本场景 orders 42–53；须逐槽核对，并在任一异常或 order54 停下。

下一槽 order46，`main-OOD-L2-06/B0/r0`，在本检查点未派发。全流程空间尚未获准；留出集、正式统计、图表及正文尚未开始。
