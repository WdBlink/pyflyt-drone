# 正式主实验二十二个回调

主实验累计22/1296有效回调、完整场景1/72、23次实际600s尝试（含原时钟超限）。首个 OOD L3 根 main-OOD-L3-06 的 order20 B4-M2/r1 与 order21 B3-M3/r2 均首尝试600s/144000步，峰值8.960/10.300ms、H/E1，动态评分和 SDK/终态/用量闭合。实测27calls/174588tokens和19calls/139412tokens。

检查点 `qualification/formal-main-checkpoints.v1/after-0022.json` SHA `4939ed6c35e8bf33867599e98c48cd8dc4b445cbf38f4ac5c672ab25be676b42`，可用 162435633152 B，下一 order22 B4-M1/r2 未派发、未预约、未飞行，之后 order23 B2/r0。原生 round103 已登记前两条 OOD 回调（request `b6ada981bdf7c404aa224daf1ecac5110c18f3fa13137965a53d3730e1584fd3`）。holdout未开启，统计、图表与正文未开始。
