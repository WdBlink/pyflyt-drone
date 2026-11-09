# 正式主实验三十个回调

主实验累计30/1296有效回调、完整场景1/72、31次实际600s尝试（含原超限）。首个 OOD L3 根 main-OOD-L3-06 的 order28 B3-M1/r1 与 order29 B3-M2/r0 均首尝试600s/144000步，峰值8.767/9.196ms、H/E1，动态评分和 SDK/终态/用量闭合。实测19calls/126162tokens与30calls/165800tokens。

检查点 `qualification/formal-main-checkpoints.v1/after-0030.json` SHA `f7fad1bdf701e44031b5a50faf669f0666f1e5b379a783d47e3ed1819ba4682d`，可用 161048367104 B，下一 order30 B3-M2/r1 未派发、未预约、未飞行，之后 order31 B3-M3/r0。原生 round107 已登记通过 order27 的证据（request `a2ad15dc53659aa79bef879361ed6756f5e523db58ba4e67da1e1a084127897e`）。holdout未开启，科学统计、图表和论文正文未开始。
