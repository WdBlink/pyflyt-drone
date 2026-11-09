# 正式主实验：首个 OOD L2 场景收齐

主实验已验收 54/1296 条、完整场景 3/72，累计 55 次实际 600 s 尝试。首个 OOD L2 场景 `main-OOD-L2-06` 的 18 条覆盖全部八配置，均首次完成，场景内最大时钟滞后 92.947 ms，实际用量 322 次本地模型调用、2,071,453 tokens。最后 `B3-M1/r0` 跑满 600 s / 144,000 步，H/E=1，最大时钟滞后 7.877 ms，实际 18 次、122,590 tokens。原另一个场景的一次时钟失败保留。

[第 54 条](formal-main-checkpoints.v1/after-0054.json)、[场景完整性](formal-main-checkpoints.v1/first-ood-l2-root-closure.v1.json)和[空间及停止记录](formal-main-checkpoints.v1/after-0054-closed-ood-l2-observation.v1.json)已封存。orders 32–53 的 22 次实际尝试容量分项仍在现有上界内，原守卫按范围停在 order54；下一槽 `main-ID-L2-13/B4-M1/r1` 未派发。

独立复核支持 [下一场景限域决定](formal-main-checkpoints.v1/after-0054-id-l2-rolling-capacity-admission.v1.json)，范围仅 orders 54–71。保留每槽两次尝试、每次模型归档 512 KiB 余量、全部其它预留和原 10 GiB。第一槽主阶段含暂存门槛 137,985,240,390 B；独立两次同卷读数约 140.406 GB。每槽以冻结顺序和实际验收回执扣除已完成槽，重新计算剩余门槛，复测两次并在实际派发后挂回暂停。原生登记后才可逐槽继续；出现异常或到达 order72 即停止。全流程预测空间仍不足，后续阶段及导出未获准；留出集、正式统计、图表和正文未开始。
