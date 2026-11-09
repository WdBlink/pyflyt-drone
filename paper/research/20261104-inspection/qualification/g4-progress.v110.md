# 正式主实验：第 59 条与经典策略容量更新

主实验已验收 59/1296 条、完整场景 3/72，共 60 次实际 600 s 尝试。ID L2 场景新增 `B2/r0` 首次完成 600 s / 144,000 步，H/E=1，最大时钟滞后 10.863 ms，无模型调用。原另一次时钟失败保留；本场景已收 5/18 条。

[第 59 条回执](formal-main-checkpoints.v1/after-0059.json)及[容量超额记录](formal-main-checkpoints.v1/after-0059-classic-size-observation.v1.json)已封存。经典非 wire 文件从样本上界 29,022,399 B 增至 29,219,022 B，超 196,623 B；orders 32–58 的 27 次实际尝试，其余分项均未超。round 123 守卫因此停止，下一 `B3-M1/r2` 未派发。

[新的独立复核决定](formal-main-checkpoints.v1/after-0059-id-l2-buffered-rolling-capacity-admission.v1.json)把剩余每次经典尝试的 raw/packed 上界各加 512 KiB，保留模型归档缓冲、两次尝试上限、其它预留和 10 GiB。按 206 条经典、1031 条模型剩余槽复算，起始门槛 137,787,290,760 B，独立同卷双读约 142.409 GB。原生登记后，仅 orders 59–71 可按实际已验收回执滚动、逐槽双读放行，实际派发后挂回检查点。任何新异常或 order72 即停。

全流程容量仍未满足，留出集继续封闭，正式统计、图表和论文正文尚未开始。
