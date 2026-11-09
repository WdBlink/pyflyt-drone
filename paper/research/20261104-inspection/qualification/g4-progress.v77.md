# 正式主实验首槽

原生 round92 已登记 exact primary manifest/order/sampler 与原 B1 选择（request `52ec34597f8501c8ba6537ef1f83c6e8ebbd37459dfbe1c8d3119b043f9dfba9`）。生产只读准入通过，首次派发前可用 164,808,445,952 B，cap2 样本压力形状 157,206,686,420 B。正式 SDK 记录器：`logs/inspection-main-primary.v1`，全 1296 分母仍在，唯一串行进程。

order0 main-ID-L1-00/B2/r0 实际首尝试已闭合为 completed，600 秒/144000 步，max lag 0.010533708002185449 秒，无 >100ms 控制 tick；H/E 各1，评分重算和回执一致。检查点 `qualification/formal-main-checkpoints.v1/after-0001.json` SHA `fe8328a1e5022847f62a6244b40a7be27c223fed542f65903b7a01f16826c93e` 绑定实际 SDK order/raw、terminal、world/attempts 和下一槽未派发。回调1/1296，根级完整0；无正式模型调用。当前 runner 在 order1 同根 B0 预约/派发前等待 checkpoint.request 释放，order2 为 B3-M3/r2。holdout 未开启。本单槽只证明生产回调闭合，不能比较策略或作论文结果。
