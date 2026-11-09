# 正式主实验：首个 OOD 场景收齐

主实验已验收 36/1296 个正式回调，完整场景 2/72，累计 37 次实际 600 s 尝试；此前一条主实验时钟超限和其预注册重试保留。首个 OOD 场景 `main-OOD-L3-06` 的 18 条记录覆盖全部八个配置，均首次完成，场景内最大时钟滞后 74.282 ms，实际 321 次本地模型调用、2,068,768 tokens。该记录只验证完整性，不估计效应。

本段新增 `B3-M1/r0`、`B0/r0`、`B1/r0` 各跑满 600 s / 144,000 步，均 H/E=1，最大时钟滞后依次为 9.276、10.564、10.413 ms。对应 [第 34 条](formal-main-checkpoints.v1/after-0034.json)、[第 35 条](formal-main-checkpoints.v1/after-0035.json)、[第 36 条](formal-main-checkpoints.v1/after-0036.json)和[场景完整性](formal-main-checkpoints.v1/first-ood-root-closure.v1.json)已封存。

第 36 条后的同卷可用量为 156,080,394,240 B；仍使用 [round111 核验](formal-main-checkpoints.v1/after-0033-export-capacity-acceptance.v1.json)的 152,971,003,172 B 门槛，每条派发前复测。下一槽为 order36，`main-OOD-L2-06/B2/r0`，检查点确认时未派发。留出集未开启，正式统计、图表与正文尚未开始。
