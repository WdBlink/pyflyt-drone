# G4开发进度v13

2026-11-06 UTC：同一个native owner表示/transport迁移已实际完成并逐字段核验。见recovery/owner-document-snapshots-v2/actual-migration-binding.v1.json与ACTIVE_CONTROLLER.md。先在order9 gate确认driver无descendants/活model-flight、最后callback cleanup closed、63源61依赖10输入不变、14资源操作allknown，无held/unknown；原owner关闭round27，完整before pair保留。新原生migration API从a112e7…迁入d809362cdb06563d520b197f9974af39c32c435332f6c13b31421b6f02753847。

实际before/after除了implementation、追加history event+operation外所有top-level研究字段精确相同；全部old history精确前缀、old operations每key/value、rounds/ledger/receipts/Brief/frozen plan/budget精确保持。oldcontrol82031061B→compact62013621B、checkpoint62013707B；只改变表示，不删历史。完整after snapshot、state/file SHA分别存档。通过已安装旧稳定CLI exec入口再次执行同packet/opid，返回值与first完全相同、control/checkpoint bytes未变；旧直接Controller因pin拒绝。当前source38每次native action实际校验、原SDK13libs和pilot63不改。future baseline/result refs由同一owner管理，历史lazy/16MiB对象/128MiB owner及pre-save容量保护仍适用。60项新测试与43原生软件集成、R1独立闭合和有限容量投影的范围/限制沿v12，不外推无限容量。

实际模型usage仍286calls/1813213tokens、14operations全known，无新模型调用/费用记录、无未结清预留。cost/host/资本/能耗未计量。固定pilot仍9/108 validcallbacks/0complete roots；原clock failure/唯一infra retry/全部旧协议代分母保留。新begin后才在source/input/dependency完整复核后释放order9；没有重跑或重标过去episode。

G1–G3和同级非LLM event/clean600s参考对/评分/执行回放保持；G4仍needs_work。正式入口G4Proof v3有限复核42pass及grade链修复关闭、pure phase identity17pass仅为已完成组件；完整namespace/dataset/generator/worker/runner/service/receipt接线仍待完成。还需完整108固定pilot、前瞻B4-M1同级clean+20s实际工程集成、最终source/model/prompt/split/retry/hardware冻结及独立G4接受。未抽actual main/holdout seed/order/instance，未开始结果正文。预算{}、全部19项义务与原实验分母/附加协议不变。
