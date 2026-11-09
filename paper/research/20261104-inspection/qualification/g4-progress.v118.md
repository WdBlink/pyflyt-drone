# 正式主实验：第四个完整场景与主阶段容量复核

主实验已验收 72/1296 条回调、4/72 个完整场景，共 73 次实际 600 s 尝试。[第 72 条](formal-main-checkpoints.v1/after-0072.json) `B3-M3/r1` 首次跑满 600 s / 144,000 步，H/E=1，最大时钟滞后 12.212 ms，实际 12 次本地模型调用、87,748 tokens。已[封存 ID L2-13 场景](formal-main-checkpoints.v1/first-id-l2-root-closure.v1.json)：八配置 18/18 条均首次完成，峰值滞后 52.991 ms，实际用量 316 次、2,035,004 tokens。历史一次时钟超限和对应重试均保留。此处只核数据完整性与运行条件。

最近 40 次实际尝试均未超过既有分项上界。旧准入于 order72 终止。[真实主阶段原生状态重测](formal-main-checkpoints.v1/after-0072-main-phase-wire-projection.v1.json)以 204 条 classic、1,020 条 LLM 的两次尝试计，raw 102,812,477,736 B、暂存 88,977,872 B，保留 1 GiB 杂项和 10 GiB 余量、原生文件、命令记录及例外字节在内的固定开销 22,199,320,246 B；合计 125,100,775,854 B。独立同卷两次读数 131,055,321,088 → 131,055,120,384 B，现场放行仍须重新双读。

[容量决定](formal-main-checkpoints.v1/after-0072-three-id-roots-rolling-capacity-admission.v1.json)只覆盖 orders 72–125，即 `main-ID-L3-15`、`main-ID-L2-08`、`main-ID-L1-03`。先通过原生登记，再由[逐槽守卫](guard-formal-main-rolling.py)核冻结前缀、所有上一槽尝试与结算、原生形状、容量双读、实际派发后重挂。每两条实际回调登记，30 分钟到期后在第一个安全边界登记；超界或 order126 即停。当前 order72 `main-ID-L3-15/B4-M2/r2` 尚未派发。留出集仍封存，正式统计、图表及正文尚未开始。
