# 正式主实验：已验收 90 条回调

当前 90/1296 条回调、5/72 个完整场景。新增 [第 89 条](formal-main-checkpoints.v1/after-0089.json)、[第 90 条](formal-main-checkpoints.v1/after-0090.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L3-15/B0/r0`：1 次尝试，末次峰值滞后 10.818 ms，H/E=1/1；`main-ID-L3-15/B3-M2/r0`：1 次尝试，末次峰值滞后 9.165 ms，H/E=1/1。新增通过的 LLM 尝试用量 30 次、163,578 tokens；所有尝试记录均保留。

[封存 `main-ID-L3-15` 场景](formal-main-checkpoints.v1/scoped-main-ID-L3-15-closure.v1.json)：八种配置的 18 条回调齐备；16 条首试通过，两次首试时钟超限及对应同输入重试均保留。已验收回调峰值滞后最大 97.081 ms，所有尝试共 335 次本地调用、2,111,911 tokens。这里只核验回执完整性。

[现场守卫记录](formal-main-checkpoints.v1/after-0090-main-rolling-observation.v1.json)保留实际双读与容量门槛。此检查点在 order90 `main-ID-L2-08/B1/r0`，下一槽未启动。后续继续受原生 round 144 准入和逐槽守卫约束。

留出集仍封存，尚未执行正式统计、图表及正文。
