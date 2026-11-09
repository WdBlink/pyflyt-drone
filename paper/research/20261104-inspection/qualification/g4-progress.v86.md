# 首个正式主场景闭合

固定主回调累计18/1296，完整独立根1/72。首根 main-ID-L1-00 的三个经典各1、五个LLM各3，一共18个实际回调逐条与顺序/SDK/终态/world SHA 重核。19次实际600s飞行，保留一个已知时钟超限及同输入重试。18个有效回调中峰值最大13.4728ms。第一根LLM实际367calls/2365065tokens（含失败尝试）。首根闭合是记录完整性，不是统计或正式 root-means SDK 输出。

新增 order17 B4-M1/r0 首尝试600s/144000步，峰值8.568ms、H/E1、评分和用量闭合（19calls/138186tokens）。检查点 `qualification/formal-main-checkpoints.v1/after-0018.json` SHA `a45f7a4d9fa696a5d8e0b235610d5197e0e88979145796ca88430300c67b1a80`，首根整体证据 `qualification/formal-main-checkpoints.v1/first-root-closure.v1.json` SHA `f35e7015d23ad27bd1f87095c2672d9dfcf8eb876e1ab4d876b94ad78efd4d4e`。可用 161597964288 B。下一 order18 main-OOD-L3-06/B3-M3/r1 无派发、预约或飞行，之后 order19 B4-M1/r0。round101 已记录通过 order16 的实际证据（request `832cfdce1ba9498e6ab47fd4324cba55647ef8b437de19fbfe7760f89c8ac43b`）。holdout未开启，科学统计、图表与论文正文未开始。
