# 正式主实验十七个回调

固定主回调累计17/1296、完整独立根0/72、18次实际600s飞行（含保留的一个超限）。新增 order15 冻结经典比较对象 B1/r0 和 order16 B3-M1/r0 均首尝试600s/144000步，峰值11.010/8.992ms、H/E均1，SDK/终态/评分重算闭合。M1实测20calls/133129tokens并known结算。三个经典槽均已闭合，首根仅余最后B4-M1。

检查点 `qualification/formal-main-checkpoints.v1/after-0017.json` SHA `bcd1b6b7e60af61b633c39d0b141c4928b8c446d10accba2d51bf4d9faa7b01b`，可用 161650552832 B。下一 order17 B4-M1/r0 无派发、预约或飞行；计划在首根闭合后、首个OOD L3派发前做检查点。round100 已记录通过 order14 的实际证据（request `8e8c2089a13e53b5c578616ce2961924a5818df0b4e5d12bf1d5274865775118`）。holdout未开启，科学统计、图表与正文未开始。
