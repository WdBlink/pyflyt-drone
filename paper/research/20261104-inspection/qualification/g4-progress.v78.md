# 正式主实验首个本地模型槽

主回调累计3/1296，完整独立根0/72；同首根 main-ID-L1-00 的 order0 B2、order1 B0 和 order2 B3-M3/r2 均真实首尝试600s/144000步。新增B0/B3-M3峰值lag 11.274/9.449ms，H/E均1、SDK raw、world、terminal和评分重算闭合。原生首槽检查点 round93 request `1c5e77b7c47256822f4c9bcee3a678a33fb57c1ebb5b08a8af8814c962f8a38f`。

检查点 `qualification/formal-main-checkpoints.v1/after-0003.json` SHA `c922e929291b8ed357d60976da1852f38194906b30e6573c2cf347d922b00c0d`，实际只读准入与下一槽无派发；本次可用 163679563776 B。M3实际18calls/129930tokens，全部正常返回，进程清理closed、native settlement known；18段等待中物理均继续推进、0个飞完后响应。客户端18.178–58.003s，服务12.179–52.082s；两者不作因果分解。原始actor和计时摘要 `qualification/formal-main-checkpoints.v1/first-llm-clock.v1.json`。这不是正式时延干预轨或方法比较；不改变后续排序或分母。

主 runner 等待在 order3 B4-M2/r1 派发前，之后是 order4 B3-M2/r0。holdout 未开启，正式统计、图表和论文正文尚未开始。批准的用户范围为原本地串行计划；无付费模型调用或额外计算购买。
