# 正式主实验：已验收 126 条回调

`main-ID-L1-03` 的八种配置已全部闭合，18 条回调来自 18 次实际尝试；验收回调中的最高时钟滞后为 11.359 ms。该场景保留的所有尝试合计 320 次模型调用、2,085,694 tokens。

当前 126/1296 条回调、7/72 个完整场景。新增 [第 125 条](formal-main-checkpoints.v1/after-0125.json)、[第 126 条](formal-main-checkpoints.v1/after-0126.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L1-03/B3-M3/r2`：1 次尝试，末次峰值滞后 9.355 ms，H/E=1/1；`main-ID-L1-03/B3-M1/r2`：1 次尝试，末次峰值滞后 9.149 ms，H/E=1/1。新增通过的 LLM 尝试用量 31 次、213,000 tokens；所有尝试记录均保留。

补充证据：[回执](../qualification/formal-main-checkpoints.v1/scoped-main-ID-L1-03-closure.v1.json)。

[现场守卫记录](formal-main-checkpoints.v1/after-0126-main-rolling-observation.v1.json)保留实际双读与容量门槛。此检查点在 order126 `main-ID-L3-07/B3-M3/r2`，下一槽未启动。原准入到此失效；下一场景另行核定。

留出集仍封存，尚未执行正式统计、图表及正文。
