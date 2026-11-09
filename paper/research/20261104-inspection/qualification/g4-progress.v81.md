# 正式主实验九个回调

固定主回调累计9/1296、完整独立根0/72、10次实际600s飞行（含保留的一个超限）。新增 order7 B3-M1/r2 和 order8 B3-M2/r2 均首尝试600s/144000步，峰值9.491/10.212ms、H/E均1、SDK/终态/评分重算和用量均闭合。新增实测19calls/127989tokens和30calls/164858tokens。完整第一根需要18个回调；当前无根级正式结果。

检查点 `qualification/formal-main-checkpoints.v1/after-0009.json` SHA `4afc83f0fa52f1d4ac7f517dc01da4d67497897ea19f99de2b3d71a88cdc7f94`，可用 162476650496 B。下一 order9 B4-M1/r2 无 dispatch、预约或飞行，之后 order10 B3-M3/r0。round96 已记录通过 order6 的实际证据（request `aa230986e1a24f258e283fdc74b1b70b1bbc01b32bd6aa4b9a6bf7dc93590918`）。holdout仍未开启，科学统计/图表/正文未开始。
