# 正式主实验：第 48 条与 OOD L2 的 12 条回执

主实验已验收 48/1296 条、完整场景 2/72，累计 49 次实际 600 s 尝试。首个 OOD L2 场景已收 12/18 条。新增 `B0/r0` 与 `B4-M2/r0` 均首次完成 600 s / 144,000 步，H/E=1，最大时钟滞后 10.741 和 8.923 ms；后一条新增本地用量 28 次、176,104 tokens。原一次时钟超限保留。

[第 47 条](formal-main-checkpoints.v1/after-0047.json)、[第 48 条](formal-main-checkpoints.v1/after-0048.json)与[逐槽空间记录](formal-main-checkpoints.v1/after-0048-buffered-guard-observation.v1.json)已封存。两次放行实测 144,199,057,408 B 和 144,228,237,312 B，均超过 139,100,000,000 B；实际派发后均重挂检查点。orders 32–47 共 16 次尝试复查后，各分项在更新上界内。round 116 的范围保持为 orders 42–53，每次只放行一槽；任何守卫异常或到达 order54 时先暂停。

下一槽 order48，`main-OOD-L2-06/B3-M3/r2`，本检查点未派发。全流程空间未获准；留出集、正式统计、图表及正文尚未开始。
