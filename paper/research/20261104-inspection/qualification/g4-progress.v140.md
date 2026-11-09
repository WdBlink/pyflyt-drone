# 正式主实验：已验收 108 条回调

`main-ID-L2-08` 的八种配置已全部闭合，18 条回调来自 18 次实际尝试；验收回调中的最高时钟滞后为 48.783 ms。该场景保留的所有尝试合计 343 次模型调用、2,203,709 tokens。

当前 108/1296 条回调、6/72 个完整场景。新增 [第 107 条](formal-main-checkpoints.v1/after-0107.json)、[第 108 条](formal-main-checkpoints.v1/after-0108.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L2-08/B4-M2/r1`：1 次尝试，末次峰值滞后 10.031 ms，H/E=1/1；`main-ID-L2-08/B3-M2/r0`：1 次尝试，末次峰值滞后 9.933 ms，H/E=1/1。新增通过的 LLM 尝试用量 59 次、343,874 tokens；所有尝试记录均保留。

补充证据：[回执](../qualification/formal-main-checkpoints.v1/scoped-main-ID-L2-08-closure.v1.json)。

[现场守卫记录](formal-main-checkpoints.v1/after-0108-main-rolling-observation.v1.json)保留实际双读与容量门槛。此检查点在 order108 `main-ID-L1-03/B4-M1/r1`，下一槽未启动。后续继续受原生 round 147 准入和逐槽守卫约束。

留出集仍封存，尚未执行正式统计、图表及正文。
