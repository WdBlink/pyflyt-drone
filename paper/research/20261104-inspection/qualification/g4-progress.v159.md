# 正式主实验：已验收 144 条回调

`main-ID-L3-07` 的八种配置已全部闭合，18 条回调来自 20 次实际尝试；验收回调中的最高时钟滞后为 94.655 ms。该场景保留的所有尝试合计 353 次模型调用、2,256,386 tokens。

当前 144/1296 条回调、8/72 个完整场景。新增 [第 143 条](formal-main-checkpoints.v1/after-0143.json)、[第 144 条](formal-main-checkpoints.v1/after-0144.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L3-07/B3-M2/r0`：1 次尝试，末次峰值滞后 37.880 ms，H/E=1/1；`main-ID-L3-07/B2/r0`：1 次尝试，末次峰值滞后 12.595 ms，H/E=1/1。新增通过的 LLM 尝试用量 30 次、163,979 tokens；所有尝试记录均保留。

补充证据：[回执](../qualification/formal-main-checkpoints.v1/scoped-main-ID-L3-07-closure.v1.json)。

[现场守卫记录](formal-main-checkpoints.v1/after-0144-main-rolling-observation.v1.json)保留实际双读与容量门槛。此检查点在 order144 `main-OOD-L3-08/B4-M1/r1`，下一槽未启动。后续继续受原生 round 166 准入和逐槽守卫约束。

留出集仍封存，尚未执行正式统计、图表及正文。
