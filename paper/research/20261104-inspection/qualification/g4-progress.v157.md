# 正式主实验：已验收 140 条回调

第 139 号槽 `main-ID-L3-07/B4-M2/r0` 首试时钟峰值 183.100 ms，有 19 步超过 100 ms。唯一一次同计划重试峰值 8.624 ms、0 步超限，评分已核对后计入主矩阵。两次均完成 600 s／144,000 步，两份原始尝试及用量均保留，共 49 次调用、315,473 tokens。超限机制尚未归因。

当前 140/1296 条回调、7/72 个完整场景。新增 [第 139 条](formal-main-checkpoints.v1/after-0139.json)、[第 140 条](formal-main-checkpoints.v1/after-0140.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L3-07/B3-M2/r2`：1 次尝试，末次峰值滞后 9.237 ms，H/E=1/1；`main-ID-L3-07/B4-M2/r0`：2 次尝试，末次峰值滞后 8.624 ms，H/E=1/1。新增通过的 LLM 尝试用量 51 次、305,548 tokens；所有尝试记录均保留。

补充证据：[回执](../qualification/formal-main-checkpoints.v1/fifth-clock-retry-order0139.v1.json)、[回执](../qualification/formal-main-checkpoints.v1/fifth-clock-retry-order0139-review.v1.json)。

[现场守卫记录](formal-main-checkpoints.v1/after-0140-main-rolling-observation.v1.json)保留实际双读与容量门槛。此检查点在 order140 `main-ID-L3-07/B3-M1/r1`，下一槽未启动。后续继续受原生 round 166 准入和逐槽守卫约束。

留出集仍封存，尚未执行正式统计、图表及正文。
