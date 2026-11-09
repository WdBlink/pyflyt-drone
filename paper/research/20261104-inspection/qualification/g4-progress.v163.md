# 正式主实验：已验收 152 条回调

第 151 号槽 `main-OOD-L3-08/B3-M1/r0` 首试时钟峰值 101.866 ms，有 1 步超过 100 ms。唯一一次同计划重试峰值 12.932 ms、0 步超限，评分已核对后计入主矩阵。两次均完成 600 s／144,000 步，两份原始尝试及用量均保留，共 27 次调用、177,074 tokens。超限机制尚未归因。

当前 152/1296 条回调、8/72 个完整场景。新增 [第 151 条](formal-main-checkpoints.v1/after-0151.json)、[第 152 条](formal-main-checkpoints.v1/after-0152.json) 已核 600 s / 144,000 步与实际回执：`main-OOD-L3-08/B3-M2/r0`：1 次尝试，末次峰值滞后 21.920 ms，H/E=1/1；`main-OOD-L3-08/B3-M1/r0`：2 次尝试，末次峰值滞后 12.932 ms，H/E=1/1。新增通过的 LLM 尝试用量 44 次、253,430 tokens；所有尝试记录均保留。

补充证据：[回执](../qualification/formal-main-checkpoints.v1/sixth-clock-retry-order0151.v1.json)、[回执](../qualification/formal-main-checkpoints.v1/sixth-clock-retry-order0151-review.v1.json)。

[现场守卫记录](formal-main-checkpoints.v1/after-0152-main-rolling-observation.v1.json)保留实际双读与容量门槛。此检查点在 order152 `main-OOD-L3-08/B4-M2/r0`，下一槽未启动。后续继续受原生 round 166 准入和逐槽守卫约束。

留出集仍封存，尚未执行正式统计、图表及正文。
