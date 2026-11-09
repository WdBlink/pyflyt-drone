# 正式主实验：已验收 87 条回调

当前 87/1296 条回调、4/72 个完整场景。新增 [第 87 条](formal-main-checkpoints.v1/after-0087.json) 已核 600 s / 144,000 步与实际回执：`main-ID-L3-15/B3-M2/r2`：1 次尝试，末次峰值滞后 9.247 ms，H/E=1/1。新增通过的 LLM 尝试用量 30 次、173,460 tokens；所有尝试记录均保留。

[原始容量观察](formal-main-checkpoints.v1/after-0087-main-rolling-observation.v1.json)保留实际放行记录。下一槽 `main-ID-L3-15/B3-M1/r0` 尚未派发。原守卫只读核验在用量回执分项拒绝：实测 13,065 B，高于旧上界 13,063 B。最近 57 次尝试只有该分项超界；已通过的回调仍完整保留。

[更新的主阶段投影](formal-main-checkpoints.v1/after-0087-main-phase-wire-projection.v1.json)将该分项预留设为 16,384 B，其他尝试分项不变。按剩余 202 classic、1,007 LLM 的两次尝试、更新的暂存、原生文件与命令记录及固定 1 GiB/10 GiB，门槛为 123,790,245,542 B。独立同卷双读 397720354816 → 397720223744 B，现场仍要重新双读。

[准入决定](formal-main-checkpoints.v1/after-0087-receipt-buffered-rolling-capacity-admission.v1.json)只覆盖原范围尾部 orders 87–125。完成原生登记后，由[修订守卫](guard-formal-main-rolling-receipt-v2.py)核对原记录与新上界；继续一次仅放行一槽，实际派发即重挂。留出集仍封存，正式统计、图表及正文尚未开始。
