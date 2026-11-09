# 当前唯一控制器入口

当前owner为本run专用 `runtime/autoresearch-document-snapshots-v2/scripts/research.py`，implementation `d809362cdb06563d520b197f9974af39c32c435332f6c13b31421b6f02753847`。继续拥有原 `.paper/control.json` 与最新durable checkpoint；不是新run或第二ledger/scheduler。全局Skill未修改。

原 `runtime/autoresearch-state-reader-v1/scripts/research.py` 是经实际独立审查、safe gate安装的稳定exec兼容入口（SHA `bc7e2c0496b215d422b3913da21dcf7283cf8b647c1d4a059350d55cdd6324f5`）。它先exec新owner，不导入旧Controller。已终止的v2 pilot曾从旧路径调用，但38项owner/compat source每次native action实际校验；旧13个pure recording SDK lib字节不变。旧直接Controller现因implementation pin拒绝。不要通过全局安装入口或旧直接Controller写此run。

原 reader-v1修复与a112e7…历史见 `recovery/state-reader-v1/`；此前默认16MiB误拒自身17MiB checkpoint的来源、完整pair与已完成迁移记录保持。新修复解决每轮约6MiB document map内嵌重复和budget missing-cost summary平方重复：仅未来round baseline及完整operation wire result用 `.paper/round-documents/`、`.paper/operation-results/` 内容地址不可变对象，仍属同owner。旧inline/request/decision/operation/history/receipts/ledger保留原值。owner pair/export control用compact whitespace，外部packet与每object仍16MiB、owner仍128MiB；保存前容量检查拒绝超界。普通load验证live refs/transport，历史refs在实际使用/recover/export检查全闭包。

独立source review v1发现R1去重漏节点格式，v2已修复并独立60pass、原recover/export/ZIP反例全部拒绝；43原生软件集成pass。见 `review/native-owner-representation-v2-review.v1.md`、`.v2.md` 和 `recovery/owner-document-snapshots-v2/{freeze.v2,independent-review.v2.json,transport.v2.json}`。实际order9 gate、closed round27、14资源操作全known/无在途后，使用新原生 `migrate-round-documents` API迁移。完整before/after、gate/packet、terminal/idempotent replay及逐字段校验见该recovery目录。新gate只批准此源与表示/transport修复，不授予G4、formal数据、付费或投稿许可。

实际迁移前后旧非implementation研究字段精确相同、history精确前缀、全部旧operations/rounds/usage ledger/Brief/frozen plan/model receipts原样，仅追加迁移event和一个operation；286calls/1813213tokens不变，cost/host未计量。不手工编辑 `.paper` 或回执。G4仍needs_work。
