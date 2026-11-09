# 正式主实验：已验收 130 条回调

第 129 号槽 `main-ID-L3-07/B3-M2/r1` 首试时钟峰值 139.382 ms，有 8 步超过 100 ms。唯一一次同计划重试峰值 85.312 ms、0 步超限，评分已核对后计入主矩阵。两次均完成 600 s／144,000 步，两份原始尝试及用量均保留，共 56 次调用、322,377 tokens。超限机制尚未归因。

当前 130/1296 条回调、7/72 个完整场景。新增 [第 129 条](formal-main-checkpoints.v1/after-0129.json)、[第 130 条](formal-main-checkpoints.v1/after-0130.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L3-07/B4-M1/r2`：1 次尝试，末次峰值滞后 19.400 ms，H/E=1/1；`main-ID-L3-07/B3-M2/r1`：2 次尝试，末次峰值滞后 85.312 ms，H/E=1/1。新增通过的 LLM 尝试用量 41 次、247,487 tokens；所有尝试记录均保留。

补充证据：[回执](../qualification/formal-main-checkpoints.v1/fourth-clock-retry-order0129.v1.json)、[回执](../qualification/formal-main-checkpoints.v1/fourth-clock-retry-order0129-review.v1.json)。

[现场守卫记录](formal-main-checkpoints.v1/after-0130-main-rolling-observation.v1.json)保留实际双读与容量门槛。此检查点在 order130 `main-ID-L3-07/B3-M3/r1`，下一槽未启动。后续继续受原生 round 166 准入和逐槽守卫约束。

留出集仍封存，尚未执行正式统计、图表及正文。
