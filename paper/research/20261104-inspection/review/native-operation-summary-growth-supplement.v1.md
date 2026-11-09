# Native operation summary 增长补充审计 v1

结论：**若只把未来 documents map 改成 ref，而 `record()` 仍内嵌完整 budget result，128MiB 明确不足以覆盖所给剩余规模。** `unreported_fields` 的复制形成平方增长。这个结论针对旧内嵌表示，不是并发新增的 v2 operation-result-ref 候选的容量结论；该新候选未在本补充中审查或批准。

范围：2026-11-06，同模型家族、独立上下文的限定核算。只执行纯内存 synthetic fixture、JSON serialization 与整数公式；未实例化 Controller，未调用 record/save/任何 CLI，未读取真实 ledger，未迁移、实现或生成 flight/model/formal 实例。暂停期间成果先写 `/tmp`，获恢复通知后再归档。原来源/owner/live 文件未修改。

## 源码链与精确公式

v1 `budget._summary()` 对每个非 reserved/not_started 的 receipt 检查 calls/tokens/cost，缺失字段留在 `unreported_fields[reservation]`。`inspection_model_receipts.py` 对 native meter 的 usage 严格只认 calls/tokens；`run_inspection_llm_episode.py` 原样传入 settle 并明确 energy/capital cost unmetered。因此已知 calls/tokens 的正常 known receipt 仍有 `['cost']`，不能假定、补写 cost0 或删除这种未计量信息。

`reserve_budget()` 和 `settle_budget()` 都把完整 `summary(state)` 作为 result，`record()` 同时在 operations 与 history 存它；JSON 不因为 Python 对象被共享就自动去重。

设未来有 N 个新 known、未计量 cost 的 LLM **resource attempts**，每次恰好 reserve+settle。先取历史 missing-cost 条目 K0=0，保守忽略现存条目：第 i 次 reserve 的 map 有 i−1 项，settle 有 i 项。故所有新 result 共包含 N² 个条目，operations/history 双存为 **2N²**。若实际已有 K0 项，则另增4K0N个副本。重试按新 resource attempt 计；cache 若完全不再调用 owner，则不增加这些 operation，不能一概当成新 attempt。

对当前 atomic 使用的 indent2 JSON，在原 result 嵌套深度，令每个 reservation ID 至少 L 个 ASCII 字节，仅 `unreported_fields` 相对于空 map 的新增字节下界为（N>0、K0=0）：

```text
control:    (2L + 108) N² + 20(2N − 1)
checkpoint: (2L + 120) N² + 24(2N − 1)
compact:    (2L +  24) N² −  2(2N − 1)
```

这是实际 JSON serialization 的增量，不含新 operation/status/reservation/request hash、真实 ledger、其他 budget 字段、future rounds 或 snapshot refs。L=1 是“每个真实 key 至少一字节”的下界，不是声称存在1800个不同单字符 ID；真实 key 更长只会增加字节。

独立小 fixture 使用实际 v1 `budget.summary()`，N=1/2/5/10/30，先 reserved 再 known，只有 calls/tokens，无 cost。五组 missing-cost 数量均按 i−1/i 增长，三种 serialization 的新增字节与公式逐字节一致。最大实体化仅N30：control 增105,580 bytes，checkpoint 增116,616 bytes；没有创建大 N 的平方状态。大规模数字全由整数公式计算。

## 给定规模的保守下界

基线复用上一份只读审计的点时观察：control 71.5701MiB、checkpoint 73.9618MiB（来源与时间/SHA 见 binding），不是这次解析真实状态。下表一律用最小 L=1、K0=0、无重试，仍保留原基线：

| 新 known LLM attempts N | 规模解释（非实际完成数） | 仅新增 map MiB | control 基线+新增 MiB | checkpoint 基线+新增 MiB |
|---:|---|---:|---:|---:|
| 90 | 完整108-slot pilot的名义 LLM 槽 | 0.853 | 72.423 | 74.908 |
| 1260 | 1512 main/holdout槽的名义 LLM 槽 | 166.594 | 238.164 | 258.734 |
| 1350 | 上述名义两部分合计 | 191.239 | 262.809 | 286.068 |
| 1800 | 主 Agent 给出的附加实验后上限尺度 | 339.958 | 411.528 | 451.013 |

这里未扣已完成/缓存的具体槽位，不能把名义规模写成精确剩余 dispatch 数；也未读取现存 K0。结论由 main/holdout 单部分的保守量级已经成立，不依赖精确 pilot 余量。历史未计量项、实际长 ID、已知失败重试及其他数据均被排除。固定旧基线下，仅这一下界到第734个新 known attempt 时 control 已越128MiB；实际可更早。这是条件性容量阈值，不是任务/预算上限。

即改成 compact JSON，N1800 的 L1 下界也会新增80.331MiB，总计151.901MiB；仅去缩进不足以解决给定上限尺度。N1800 若均发生两次已知 attempt，则平方项变为四倍，证据中另列N3600公式，但没有假定实际会如此。

## 最小建议与未验证边界

仅建议同一 owner 为**未来** operation results 使用 create-once、content-addressed 完整 result 对象，并在 operations/history 存小 ref；不更改 usage_ledger、未计量字段或预算推导，不做新 ledger/scheduler。`operation()` 经验证 resolver 返回完整原 JSON result value以维持重复 op 的行为；若合同要求原 wire bytes，也应明确保留 bytes，而不是事后从部分字段重建。旧 inline result 与历史 key/value 保留；不能通过只保留 totals、删 unreported_fields 或填 cost0 来“通过容量”。

仍需另审 ref 在 record/operation/history 读取、durable write→checkpoint→control 的故障顺序、missing/tampered object、迁移值保持、export/ZIP 闭包、旧 CLI 兼容及 planned-horizon 容量。完整 result archive、CLI transport 和 owner receipt logs 的总磁盘量仍可能 O(N²)；ref 主要移除单个 controller/checkpoint JSON 的重复，并非使全部历史信息常量大小。未运行新候选的这些检查。

本次首个 probe 试图先断言 v1/v2 相关 AST 相等，遇到并发新增 result-ref 的候选差异而停止，尚未执行 fixtures。随后成功核算只使用冻结 v1 源；不回退或覆盖并发代码，也不把旧表示的结果套到新候选上。

精简证据为 `review/evidence/native-operation-summary-growth-supplement-v1/bindings.json`：实际小 fixture、精确系数、规模公式、来源SHA、旧基线来源及排除项。完成本补充后暂停R写入。此报告无 migration/capacity/G4/main GO。
