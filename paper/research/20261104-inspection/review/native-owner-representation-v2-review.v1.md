# Native owner representation v2 实现/迁移/transport 独立审查 v1

结论：**request changes，不能批准此冻结候选迁移。** 独立重跑45项新测试与43项原生回归均通过，但新增反例证明 full closure 的逐记录格式验证被去重逻辑绕过，recover 和 portable export/ZIP 因此接受直接 resolver 会拒绝的坏记录。没有写 `independent-review.v1.json` 批准记录，也没有执行真实迁移。

范围：2026-11-06，`/root/statistics_design_review`，与主 Agent 分离上下文、同模型家族。只读生产/候选及冻结源；新增本报告与小 binding。所有状态改写、故障注入和导出仅发生于隔离系统临时目录，脚本和运行 log 在 `/tmp`；无真实 owner CLI/state 操作，无 physics/model、formal seed/实例或 paid API。实际 fresh gate、预算 closeout 与迁移结果不由本报告认证。

## 冻结身份

- freeze binding SHA256：`150f6fc23047430114d5cd2f5133989814269e16d40f0c4cc641607f5ec9a4aa`，46项文件核对通过。
- subject SHA256：`045407dc2437a8deb919a4ff2c78c1aff87fb471c11c8878cad9cfdfe73a9c0e`，38项 subject 文件（37 target + 1 compatibility proposal）核对通过。
- from implementation：`a112e7aaff3ab6c99f983a2030fc190d7ccd1af940a943604d94ad318e8fef20`。
- target implementation：`646a9bbce01bd8bdd1e13b7e846a1b328121bfbfe31503e2fbfc5023b475284b`，独立按 implementation_paths/digest 公式复算一致。
- transport SHA256：`243a39bc0571860eca229d3088be40290d73b75d9f20ff6f827046376f783591`。
- `lib/round_documents.py` SHA256：`a8b6881fdb8b96ef56d130d81db347dfc35ed2e1ea517999575bad24e4d6bcfe`；`lib/runtime.py`：`faed639fd21264b794afab18b83aa8a7e06b4533934481980dac34bc9a847db8`。

当前实际旧 CLI 仍是 SHA256 `5ca105c6a26cd631ca6c5cd70a1698677b23b924a0f26a0579a349607121d7a0`；candidate compat proposal 为 `bc7e2c0496b215d422b3913da21dcf7283cf8b647c1d4a059350d55cdd6324f5`，尚未安装。核对旧 pilot63 中13份 v1 runtime lib 均未改变。

## P1 / R1：closure 去重跳过 record 格式，恢复与导出误接纳

位置：`lib/round_documents.py:172` 的 `closure()`，尤其176–196。baseline 与 operation-result 都先从节点取 ref，用 `encoded(ref)` 查 `seen`；命中后不再调用 `resolve`/`resolve_result`，于是跳过了该节点自身“恰好一个 inline/ref”规则。operation record 若既没有 result 也没有 result_ref，则整条被 `if 'result_ref'` 忽略。

去重可以复用一个 object 的 bytes/schema/hash 校验，但不能复用另一个 **record 节点** 的 shape 验证。两层的对象不同。

独立建立三个合成坏状态，所有 ref 指向真实、完整、hash匹配的合成对象；没有攻击实际 owner。每例先验证直接 resolver 拒绝，再使 control 与 latest checkpoint 不一致，调用实际 `recover()`；另打包同坏状态的 relocated-compatible export 与完整 ZIP，调用实际 `verify_export()`：

| 合成反例 | 直接 resolver | 实际 recover | 实际 export + ZIP |
|---|---|---|---|
| 先 valid documents_ref，后历史 baseline 同 ref 且同时有 `documents` | Refusal：exactly one baseline | `recovered` | `verified`，4文件 hash/ZIP 检查通过 |
| operations valid result_ref，后 history 同 ref 且同时有冲突 inline `result` | Refusal：exactly one result | `recovered` | `verified`，4文件 hash/ZIP 检查通过 |
| operation 只有 request_sha256，没有 result/ref | Refusal：exactly one result | `recovered` | `verified`，3文件 hash/ZIP 检查通过 |

三个 pre-recover `load()` 都先正确拒绝 pair mismatch，说明实际走了恢复入口；恢复之后却接纳了违反自身格式契约的 latest state。ZIP 字节完整性通过不能弥补 controller 记录语义缺失/歧义。迁移同样调用 full closure，因此这个问题也不能以“普通 load 故意 lazy”解释。

原始证据 `/tmp/native-owner-v2-closure-entry-probes.v1.json`，执行脚本 `/tmp/native-owner-v2-closure-entry-probes.py`；返回值、确切 fixture 路径、source SHA 与脚本/output SHA 已收录小 binding。最早五项纯 resolver/closure 反例另留 `/tmp/native-owner-v2-dedup-probe.v1.json`。

最小修复要求：每个 baseline 和每个 operations/history operation 节点都先做 exact-one 与 ref/inline shape 验证，包括 missing case；仅 object 文件的读取/strict JSON/schema/hash/count 校验可以按身份缓存。同 ref 的 valid-first/invalid-second、反向顺序、跨 operations/history 及跨 live/history 都要覆盖。修复后独立重放这三个 recover/export/ZIP 入口反例；不得通过忽略坏节点或重建其值来修复。

## 实际通过的检查及边界

执行于 candidate cwd，均设 `PYTHONDONTWRITEBYTECODE=1`/`-B`，pytest 禁用 cache：

```text
<repo>/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_round_documents.py --basetemp=/tmp/native-owner-v2-review-tests45 -q
PYTHONPATH=tests python3 -B -m unittest tests.test_runtime tests.test_budget tests.test_integration -q
```

第一项 Python3.11.15：**45 passed in3.59s**；第二项系统 Python3.14.7：**43 passed in17.696s**。43项在具备已有 matplotlib/TeX 依赖的系统解释器运行，未在飞行 `.venv` 运行绘图集。日志 SHA 见 binding。测试使用合成 temporary owner、实际 staged compat bytes，以及合成 reserve/settle/replay；不是实际 ledger migration 或研究数据。

结合代码与通过的测试，已观察到以下候选行为，但不因局部通过而覆盖 R1：

- future begin/rewind 使用 documents_ref，prepare/apply 不回填 map；future `record` 将完整 result 写 owner content-addressed object，operation replay 返回原 JSON value，未把 unmetered cost 改0。旧 inline、历史前缀与旧 op key/value 在合成迁移检查中保留。
- object 有16MiB bound、strict JSON、路径限定、无链接检查、no-clobber install/file fsync/parent fsync；reuse 后同步父目录。save 对 compact checkpoint/control 都在任一写入之前检查128MiB，正常 live 第二写中断后使用唯一 latest checkpoint。R1 是 full closure 的逐节点验证错误，不是这些 object hash 校验本身失效。
- ordinary load 只查 active/pending refs；historical refs 在 replay/recover/export 时实际消费。此 lazy 策略有明确声明；R1 恰好发生于声称完整检查的入口，必须修复。
- 新 migration kind 检查 closed active/pending、ready、unresolved usage、from/to、source-bound review/transport 与 exact state/usage gate；旧 migrate-state-reader 只允许已存在 op 的原样 replay。源候选不能自行证明 `no_active_request_model_flight=true` 对应真实进程，实际 gate 仍须主 Agent 提供/核验。
- 顶层 CLI 进入37-file target implementation；迁移事件绑定的 transport 要求 target37 + compat entry 的精确集合，load 在每次 native action 前核源。wrapper 仅 exec 新入口，没有先 import 旧 Controller。该补绑定是前瞻 amendment，不追认旧63 envelope 已覆盖 CLI。
- export 将 owner objects 一起复制，manifest/ZIP 集合与 bytes 检查、可移植路径在合成测试通过；但不能在 R1 未闭合时称 full historical closure 已验证。

## 容量限定

本次读了冻结的 capacity-projection.py 与结果，未重跑其对真实 old owner 的只读投影，也未解析真实大状态。其模板是13,000 future operations、5,000 known LLM attempts、2,000 recent-sized round envelopes；projected checkpoint **131,054,594 bytes**，128MiB bound 为134,217,728 bytes，余量 **3,163,134 bytes（约3.02MiB）**。

这是明确模板的表示投影，不是任意未来 workload 保证。迁移 gate 的实际 base/封闭轮次若与投影绑定的 base_state_sha 不同，应重新对齐/评估余量；更多/更大 history/result/round envelope 和16MiB单对象限制仍会 fail closed。immutable full result archives/CLI logs 总磁盘增长也未消失。不能把此前平方增长公式或本次模板投影直接变成新实现容量批准。

## 本轮处置

小 binding：`review/evidence/native-owner-representation-v2-review-v1/bindings.json`。它记录冻结身份、source验证、两组测试、完整最小反例返回值及容量余量。共享 candidate/frozen source 与真实 owner 均未改动。

本轮在确认 recover/export 最小影响后收束，不扩展成泛安全审计。**不批准 freeze.v1，不出具 migration approval JSON。** 后续应保留本版证据，在修复后的新 freeze 上另做限定复验；真实迁移、fresh gate 放行与 G4/main 仍未执行或认证。
