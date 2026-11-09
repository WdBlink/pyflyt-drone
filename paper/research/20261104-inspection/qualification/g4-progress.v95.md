# 主实验检查点：原生导出空间复验

正式主回调仍为 33/1296，完整场景 1/72，34 次实际 600 s 尝试。回调、分母和失败记录沿用 `formal-main-checkpoints.v1/after-0033.json`。第 34 个固定槽（order33，main-OOD-L3-06/B3-M1/r0）尚未派发；留出集、正式统计、图表与正文未开始。

独立复验使用原生导出的同一默认 ZIP_DEFLATED。83 份已有原生文档从 424,671,485 B 压至 96,948,564 B；两组高熵压力样本最高约 70.06%，解压后 SHA-256 均吻合。封存的 [结果](formal-main-checkpoints.v1/native-export-capacity-shape.v1/results.json)、[探针](formal-main-checkpoints.v1/native-export-capacity-shape.v1/probe.py)和[实际对象索引](formal-main-checkpoints.v1/native-export-capacity-shape.v1/actual-object-index.json)可复核。

[容量接受记录](formal-main-checkpoints.v1/after-0033-export-capacity-acceptance.v1.json)（SHA-256 `2d90b212c977e6e5e5d00b521ada3fb2ddc4c10f78a5c6114fb111e042ead6ed`）仅把未来原生 round-doc JSON 的 ZIP 部分改按 80% 计，另保留 64 MiB ZIP 开销。完整目录副本、其它 ZIP 分量和原有 10 GiB 余量均保留。新门槛 152,971,003,172 B；封存时同卷可用 156,982,423,552 B。该数是基于实测样本的规划值；下一槽放行前仍须重新测同卷可用量，阶段完成及真正导出前也须复查。未改变任何正式输入、输出或评分规则。
