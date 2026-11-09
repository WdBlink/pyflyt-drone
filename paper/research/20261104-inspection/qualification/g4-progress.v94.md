# 正式主实验三十三回调与同卷空间波动

主回调累计33/1296，完整场景1/72，34次实际600s尝试（早期一个超限保留）。新增 main-OOD-L3-06 order32 B3-M2/r2 首尝试600s/144000步、H/E1、峰值31.147ms、动态评分和SDK/终态闭合，known 30calls/160198tokens。第33槽的文件大小没有增加此前观测到的各计入容量分量最大值。

检查点 `qualification/formal-main-checkpoints.v1/after-0033.json` SHA `7de87e54ec96846cd75ac0a0b47851a10e6df59cae2c768ce7a799ea0cdf96bd`；下一 order33 B3-M1/r0 无派发或预约。原生 round109 已登记首轮真实容量复验，request `e86f4595dce9e4e8f4d4484b6daf89ed9dc6049623411f6923157fb4a56447a1`；放行第33槽前实测156789006336B，闭合时同卷可用 155383451648B，再低于已审156017544308B，保持 checkpoint.request。正式新增文件不能解释所有同卷波动，未归因。

用同一已验证 APFS 文件级无损压缩核对八个完整已封存G1/executor开发manifest的560个原轨迹SHA，其中520个≥256KiB压缩；全部560原字节复读精确，实际可用+1645043712B、终测156990947328B。证据 `qualification/formal-main-checkpoints.v1/closed-qualification-trace-pack.v1.json` SHA `1e38c8b7eaecdfa6f2aca80abc7fa549ffb3561ae44314b1f3bea29407146c07`。未变更任何正式数据、历史失败状态或资格SHA。原生导出形状正在只读实测，尚无新容量门槛或放行。holdout未开启，统计、图表与论文正文未开始。
