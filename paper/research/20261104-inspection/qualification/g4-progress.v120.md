# 正式主实验：已验收 75 条回调

当前 75/1296 条回调、4/72 个完整场景。新增 [第 75 条](formal-main-checkpoints.v1/after-0075.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L3-15/B4-M2/r1`：2 次尝试，末次峰值滞后 48.011 ms，H/E=1/1。新增通过的 LLM 尝试用量 16 次、100,512 tokens；所有尝试记录均保留。

[首试与重试证据](formal-main-checkpoints.v1/second-clock-retry-order0074.v1.json)：首试峰值滞后 146.196 ms，21 步超过 100 ms，作运行失败保留；同一冻结输入的唯一重试通过。两次实际用量合计 34 次调用、215,206 tokens。原因未归因。

[现场守卫记录](formal-main-checkpoints.v1/after-0075-main-rolling-observation.v1.json)保留实际双读与容量门槛。此检查点在 order75 `main-ID-L3-15/B3-M1/r2`，下一槽未启动。后续继续受原生 round 134 准入和逐槽守卫约束。

留出集仍封存，尚未执行正式统计、图表及正文。
