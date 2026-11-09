# 首个 OOD L3 正式回调

主实验累计20/1296有效回调、完整场景1/72、21次实际600s尝试（含原时钟超限）。首个 OOD L3 根 main-OOD-L3-06 的 order18 B3-M3/r1 与 order19 B4-M1/r0 均首尝试600s/144000步，峰值7.937/9.984ms、H/E1，动态评分重算和原 SDK/终态/用量闭合。实测16calls/114999tokens和16calls/116181tokens。当前无新的完整根或正式统计结果。

检查点 `qualification/formal-main-checkpoints.v1/after-0020.json` SHA `465b9d9b285820ef99fd80061af3240ab38526078fcd8e452c2b319f484518b0`，可用 161380663296 B，下一 order20 B4-M2/r1 未派发、未预约、未飞行，之后 order21 B3-M3/r2。原生 round102 已记录首根全18槽闭合（request `02240dc92ee56305472a4fae543e3949e5145afba166d8c9c1f821a3be649d59`）。holdout未开启，统计、图表和论文正文未开始。
