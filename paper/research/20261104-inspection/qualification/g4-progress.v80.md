# 正式主实验七个回调

固定主回调累计7/1296、完整独立根0/72、8次实际600s飞行（含保留的一个100ms超限）。新增 order5 B3-M1/r1 和 order6 B3-M2/r1 均首尝试完成600s/144000步，峰值8.267/9.746ms、H/E均1，原评分重算、SDK和用量回执闭合。实际19calls/127944tokens和30calls/166371tokens。M1/M2/M3均已至少一条实际正式回调通过；不是各条件均合格或方法结果。

当前安全检查点 `qualification/formal-main-checkpoints.v1/after-0007.json` SHA `d45a91680ff27345c1d957f5acfee8ce3df96e0a78c59672c20f5f6ec28a565e`，可用 162443894784 B，下一 order7 B3-M1/r2 无预约或派发，下一段按固定 order7/8。round95 已登记实际时钟故障/同输入重试（request `28d17224d8d16c2af44bcc2be903887e8b293cb2b3bfb4cfbad463a8e55f7bd3`），保留所有分母。holdout 未开启，正式统计/图表/正文未开始。
