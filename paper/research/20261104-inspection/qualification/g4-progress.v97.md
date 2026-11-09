# 正式主实验：OOD L2 的前三条记录

主实验已验收 39/1296 条，完整场景 2/72；累计 40 次实际 600 s 尝试，包含此前保留的一次时钟超限。首个 OOD L2 场景 `main-OOD-L2-06` 新增 `B2/r0`、`B4-M1/r2`、`B3-M2/r0`。三条均首次跑满 600 s / 144,000 步，H/E=1，最大时钟滞后分别为 10.872、9.139、9.721 ms；新增本地模型用量 51 次、310,279 tokens。

封存的检查点：[第 37 条](formal-main-checkpoints.v1/after-0037.json)、[第 38 条](formal-main-checkpoints.v1/after-0038.json)、[第 39 条](formal-main-checkpoints.v1/after-0039.json)。以同一已验收盘点程序复核 order32–38 共七次尝试，所有计入容量的逻辑字节分量均未超过先前样本上界；[盘点记录](formal-main-checkpoints.v1/after-0039-observed-size-window.v1.json)。同卷可用 156,788,985,856 B，维持 152,971,003,172 B 门槛，每槽派发前再测。

下一槽 order39，`main-OOD-L2-06/B1/r0`，在检查点确认时未派发。留出集未开启，正式统计、图表与正文尚未开始。
