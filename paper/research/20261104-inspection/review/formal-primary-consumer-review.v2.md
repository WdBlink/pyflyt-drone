# Primary consumer 修复：immutable v2 独立有限复验

结论：**request changes；PC5 仍有 1 个同类嵌套 bool/int 残余。** 原 PC1–PC4、PC5 外层 repeat 反例及 C1 均已按预期拒绝/无副作用；完整末次缓存、readonly、source/profile 与 main/holdout 对照继续通过。本轮不批准修复全集；working 的后续 canonical 修订未参与本结果，也未获通过。

## 对象、隔离与实际运行

reviewer `/root/statistics_design_review`，独立上下文、同模型家族，非人工或跨模型审查；按 autoresearch 保留原反例、失败与受限证据。

immutable subject：`planning/formal-admission-candidate.v3/primary-consumer-repair-freeze.v2/binding.json` SHA256 `4ea0511c325f47a15c5463a7ad2b137d48864ddf60b46a443b2f7442f5616ef0`。36 source + 17 tests + 2 validation logs 共55个绑定文件均与隔离副本核对一致。只用新 overlay `/tmp/formal-primary-consumer-repair-review-v2/project`：复制全部 subject code，保持 HERE 层数，imports 优先 copied eval；生产 envs/已有 experiment/纯 SDK 只读借用。`test_formal_cache` 所需旧 synthetic child fixture 单独复制并记 SHA，属于测试适配而非新增 production subject。未以 working 字节作任何 gate。

关键 source：model service SHA `484d422b50f76716a0dd3222884b6f2d41ba123ec33821e7995f101df80969e6`；LLM coordinator SHA `4261be147a9f67a64b0a17ef78adecbdeb32e6193d7f19a8ddd1e996e52e36cc`。其余 exact SHA、借用说明及全部探针代码见 evidence。

实际结果：

| 检查 | 结果 | 性质 |
|---|---|---|
| subject 的 dispatch/service/settlement/runner/cache 5 个相关文件 | **73 passed in20.99s** | primary pure/Fake、actual provided SDK、5个实际 synthetic child；无模型/物理 |
| 独立旧 cache/phase 反例改写和缓存正控 | **5 passed in0.24s** | literal cache；首个 physical/meter validator 明确 Fake |
| 独立 service/C1 改写及负控 | **5 passed, 1 failed in0.17s** | 唯一实质失败为下述 PC5 嵌套残余 |
| owner_snapshot field / 实际 worker-input pickle | exit0 | 纯 schema/payload，无 spawn |

我的第一次 cache 探针移植漏了 `write` 辅助函数，5项均因 NameError 失败；这是本人 fixture 错误，日志及初始脚本保留。仅修正 `/tmp` 探针后重跑5项通过，不列为产品缺陷，也未改 subject。

作者日志为旧 full362（早于5项追加）及新 targeted45；本人未声称或推算 full367，也未重跑完整 secondary/stage graph。

## 原反例逐项复验

- **PC1 关闭原 trailing-attempt 漏验**：原“00 completed + 01仅completed marker/world、缺intent”现拒绝，文件 hash/集合不变。coordinator 先检查 canonical attempt 序列/cap2，并要求 cached completed 是最后一个；measurement 用 readonly 分支。独立正控分别接受唯一成功00，以及已知 clock-failed00 + 有效末次01；前者 Y/score.Q=.4，后者=.9，terminal receipt 与 world SHA 均指向同一被接纳的末次，未混用或重跑。
- **PC2 关闭**：缺 `.lock` 的旧反例拒绝且不新建文件；完整 cache 开现有 lock 的 readonly 分支，episode全文件集合/hash不变，未 reserve/settle/spawn。所附 synthetic child 正控还通过真实 cache validator 校验，counter仍1次，旧 owner effects 不增加。
- **PC3 关闭**：stored formal plan 被 expected 删除 formal_dispatch 并标 development 的旧反例，现在读 plan 后即拒绝；不再返回 development 证据。所附原正常开发及 main/holdout 路径继续通过。
- **PC4 关闭原 authority split**：Fake validated snapshot 无该 reservation，即使 raw control/checkpoint **相互自洽**且含 reserved entry，也拒绝。反向正控：verified snapshot 有正确 reservation/profile，raw control 是不可解析文本，formal service仍接受该正确 snapshot，表明它没有回退 raw read。settlement 同样取 admission 的 owner_snapshot；未把 Fake interleaving 冒充真实 native transition。
- **PC5 原外层反例修复**：purpose.repeat_ordinal=false 对 slot整数0现在拒绝；合法 source/profile 正控和 M2 receipt 对 M1 slot 的拒绝负控仍正确。**但下述嵌套同字段仍宽松，PC5不能整体关闭。**
- **C1 闭合所测 canonical 边界**：实际 main()/argparse 接收同一 canonical formal-role 路径且遗漏 refs 时，现在在 out 创建前拒绝。未扩大到任意改名文件/hostile copies，也未声称旧代码已有 physics bypass。

以上独立旧7情形包含一个模型负控，不把它计为缺陷。Fake cache 正控仅证明控制流/链绑定组合；不是正式 flight、真实模型计量或 actual G4 admission。

## PC5-b · P2：嵌套 dispatch binding 仍用普通 dict equality

位置：subject `inspection_model_service.py::admitted_limits`。新增 type guard 只覆盖 purpose 的**外层** repeat；嵌套 `purpose.formal_dispatch_binding != formal.binding()` 仍为 Python普通字典比较。

精确 literal/Fake 反例（`test_PC5_nested_binding_bool_repeat_should_reject`）：

1. admitted binding 的 `repeat_ordinal` 是 native int0，purpose 外层 repeat 保持 native int0；其他 profile、reservation、phase/model字段均保持合法。
2. 只将 purpose 的嵌套 `formal_dispatch_binding.repeat_ordinal` 改为 JSON false。
3. 当前 service 返回合法 profile，没有抛错；`False == 0` 使嵌套字典仍相等。红测试保存 accepted_profile=true，并以“应拒绝但实际接受”失败。

这是实际 service guard 的 unit 复现，formal dispatch admission 为明确 Fake，无 MLX/model生成。新的 settlement 已对整个 purpose 做 canonical digest 比较，能在后续阶段拒绝此类型差异；后置拒绝不能替代 load/per-call 前门。建议 service 对嵌套 binding 作 canonical/严格类型比较，覆盖与 primary binding 相关的 native数值字段；不可仅保留外层 type 检查。root 的 working 修补另版复验，本报告不回填其结果。

## Snapshot、primary 正控和时序边界

`ExecutionPlan.owner_snapshot` 实测为 kwonly、compare=False、repr=False；换 snapshot 不改变计划 equality。实际 `FormalEpisode.worker_input` 的 kwargs pickle 不含 owner sentinel。这里验证的是当前真正传给 worker 的 ref字典，不宣称通用 pickle整个 ExecutionPlan 会自动剔除 owner。

新增 secondary 类型的 primary 回归：73项中 main1296 / holdout216 的 actual SDK+Fake完整顺序与 root means 对照继续通过，原三矩形完整分母/故障保留仍在。只认证这些 primary 路径；secondary content/clean/ablation/delay、全graph、最终source inventory未审。

**所有集中测试已在 telemetry 前结束。** 日志元数据 UTC窗口：

- provided73：12:46:25.395791–12:46:46.544290；
- 初始 fixture错误：12:49:20.413527–12:49:20.992356；
- service独立6项：12:49:21.035557–12:49:21.318966；
- 修正cache5项：12:50:06.020384–12:50:06.350647；
- 最后 schema/进程清理检查完成：**2026-11-06T12:51:01.525454Z**。

上述 log birthtime为shell-open、mtime为最后输出，非内核精确采样时间。结束时无匹配overlay命令；5份 synthetic child process-tree均closed、remaining_owned=[]，并已告知root全部测试停止。之后只做报告归档，不再集中测试/新probe；不对后续 telemetry clock作原因推断。

原 v1 报告和证据保持不变。未执行实际 formal RNG、MLX、flight或真实 native owner写，actual研究数据未生成。本结论保留 PC5-b，不能批准 primary consumer全部修复，不授 G4/main或secondary完整graph。
