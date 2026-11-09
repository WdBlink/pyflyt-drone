# 正式主实验：第七个场景闭合后的容量核定

已验收 126/1296 条，7/72 个完整场景。上一场景的 18 条回调均为首试通过，详见[场景记录](formal-main-checkpoints.v1/scoped-main-ID-L1-03-closure.v1.json)。

在预约前暂停点核对余下 1,170 条（195 条经典基线、975 条模型方法）。[实测容量报告](formal-main-checkpoints.v1/after-0126-main-phase-capacity-sizing.v1.json)按最多两次尝试保留 131,532,871,942 B；同卷读数 678,840,692,736、678,840,643,584 B。报告保留最近 36 次实际尝试的文件尺寸、原生账本余量和持久请求文件预算。

[独立审阅](formal-main-checkpoints.v1/after-0126-main-phase-capacity-review.v1.json)绑定最终守卫和容量报告。[本次准入文件](formal-main-checkpoints.v1/after-0126-main-phase-rolling-capacity-admission.v1.json)限定为主矩阵 `order126..1295`。每条仍须先过真实暂停、回执、原生账本和两次同卷容量核验。

本检查点记录时，下一条 `main-ID-L3-07/B3-M3/r2` 未派发；留出集仍封存。
