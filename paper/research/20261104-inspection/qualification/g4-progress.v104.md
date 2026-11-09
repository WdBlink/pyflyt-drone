# 正式主实验：第 50 条与 OOD L2 的 14 条回执

主实验已验收 50/1296 条、完整场景 2/72，累计 51 次实际 600 s 尝试。首个 OOD L2 场景已收 14/18 条。新增 `B3-M3/r2` 与 `B4-M1/r1` 均首次完成 600 s / 144,000 步，H/E=1，最大时钟滞后 9.118 和 26.289 ms；两条新增实际本地模型用量 34 次、242,657 tokens。原一次时钟超限保留。

[第 49 条](formal-main-checkpoints.v1/after-0049.json)、[第 50 条](formal-main-checkpoints.v1/after-0050.json)与[逐槽空间记录](formal-main-checkpoints.v1/after-0050-buffered-guard-observation.v1.json)已封存。两次放行实测 144,180,928,512 B 和 144,139,333,632 B，均超过 139,100,000,000 B；实际派发后均重挂检查点。orders 32–49 共 18 次尝试复查后，各分项仍在更新上界内。round 116 的范围保持为 orders 42–53，每次只放行一槽；任何守卫异常或到达 order54 即暂停。

下一槽 order50，`main-OOD-L2-06/B4-M1/r0`，本检查点未派发。全流程空间未获准；留出集、正式统计、图表及正文尚未开始。
