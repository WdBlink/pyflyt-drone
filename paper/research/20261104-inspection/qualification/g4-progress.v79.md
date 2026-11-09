# 正式主实验已闭合五槽

已按原定序完成5/1296有效主回调，完整独立根仍0/72，6次实际飞行。新增 order3 B4-M2/r1 一次通过，600s/144000步、峰值8.398ms、H/E1，实际27calls/173373tokens。新增 order4 B3-M2/r0 先发生唯一已知物理时钟故障：同根/同配置/同采样计划，第一次600s/144000步，峰值105.028ms、1个>100ms tick（sim 379.525、非sensor tick，不归因）；仅原冻结cap2允许同输入重试。第二次600s/144000步，峰值10.514ms、0超限、H/E1，通过终态和SDK。两次M2各真实30calls/169959tokens并known结算，原失败状态、使用和原字节均保留。

实际检查点 `qualification/formal-main-checkpoints.v1/after-0005.json` SHA `6b58240b38268bc583dda54c1c244b5106245e2d891fe812fac45380bc32b5af`。逐尝试时钟/用量及原工件绑定 `qualification/formal-main-checkpoints.v1/first-clock-retry.v1.json` SHA `44b4639c5edcf91b158bc59ebbc3ff598508727a7c830ec5696227c020d0720d`。关闭同槽重试后才回到安全检查点（跨30分钟提示期间无外部重负载）；下一个 order5 B3-M1/r1 尚无 dispatch、预约或飞行。当前可用 161750552576 B，资格样本压力投影157206686420 B；需要继续复测。holdout 未开启，模型/协议/评分或分母未调整，未开始科学统计/图表/正文。
