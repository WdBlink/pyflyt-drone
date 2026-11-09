# Native owner representation v2：R1 限定修复复验

结论：**R1 已关闭，批准此次冻结的 source candidate。** 每个节点现在先检查 exact-one inline/ref，再仅复用不可变对象的字节验证；原误接纳反例全部转为拒绝，正常重复引用与旧 inline 仍可用。本批准不表示真实 closed-round/fresh gate 已核验、实际已迁移或 G4/main GO。

审查者：`/root/statistics_design_review`，同模型家族、分离上下文。日期2026-11-06。本次只审 freeze.v1→freeze.v2 的 R1 修复，不扩大 v1 审查边界；原 v1 报告及证据保留。只读候选/生产/实际 owner；合成操作仅在隔离 temporary fixtures，未调用真实 owner CLI、运行模型/物理或生成 formal 实例。

## 冻结绑定与唯一差异

freeze.v2 binding SHA256 `6284c47693968aa474dbbb6aa19a084196370df5d349054a51d7adc5349b40ef`，46项绑定文件通过。subject SHA256 `f97b9c72b7a581009871b242c38e2b036841f529f8b56af3028b28071ccad783`，37 target + 1 compatibility proposal 共38项通过。独立复算 target implementation 为 `d809362cdb06563d520b197f9974af39c32c435332f6c13b31421b6f02753847`；from 仍为 `a112e7aaff3ab6c99f983a2030fc190d7ccd1af940a943604d94ad318e8fef20`。

与 freeze.v1 相比，唯一实现差异是 `lib/round_documents.py`，新 SHA256 `83f4fd49742ebe7e895400d0ce3e28fc737d054bf006e9d6cb9e176c382f6215`。新增 `require_one()` 被 resolver 与 closure 共用；每个 baseline、operations record、含 operation 的 history record 都在对象去重之前检查自身 exact-one，缺两种 result 也拒绝。其他实现文件逐项 hash 不变；新增测试是相应 regression。

transport ref 精确为 `recovery/owner-document-snapshots-v2/transport.v2.json`，SHA256 `2f698501f2d0504d3a7cb24dcff34cecb94aca2bce69b4db17b240ad1c609832`。检查其集合为 target37 + 稳定旧路径兼容入口，target bytes 与 subject 一致；proposal wrapper 仍为 `bc7e2c0496b215d422b3913da21dcf7283cf8b647c1d4a059350d55cdd6324f5`。这里验证的是未安装 proposal 及隔离 staged wrapper，不追认旧63 envelope 或实际当前旧 CLI 已是新入口。

## 实际复验

独立执行（candidate cwd，Python3.11.15，禁 bytecode/cache）：

```text
PYTHONDONTWRITEBYTECODE=1 <repo>/.venv/bin/python -B -m pytest -p no:cacheprovider tests/test_round_documents.py --basetemp=/tmp/native-owner-v2-review-tests60 -q
```

结果 **60 passed in3.86s**。包括原45项、五种 malformed 节点在 closure/recover/portable export+ZIP 的15项回归。另独立复用原反例：

| 复验 | 修复后观察 |
|---|---|
| 原5项 direct/closure | 全部拒绝：dual baseline、dual result 和 missing result 不再被 seen 去重跳过。 |
| 原3项实际 recover | 均 Refusal：exactly one；不再 recovered。 |
| 原3项 portable export + 完整 ZIP | 均 Refusal：exactly one；不再 verified。 |
| 正常同 ref 多次出现、legacy inline result | 正常接受，closure 得到预期2个对象。 |

原 recover/export 脚本保留不改；本次复制为 `/tmp/native-owner-v2-closure-entry-probes.repair-v2.py`，只将构造坏 export 的 object 集合预先固定，避免新的 closure 正确拒绝后无法继续到实际 verify_export 入口。结果见 `/tmp/native-owner-v2-closure-entry-probes.v2.json`；五项 direct/closure 及 positive controls 见 `/tmp/native-owner-v2-dedup-probe.v2.json`。均为原 R1 的隔离重放，没有实际账本损坏或研究数据。

核对 owner 提供的冻结43项原生 runtime/budget/integration 最终 log：**Python3.14，43 tests，18.771s，OK**，SHA256 `72aa303181c04c94fb002ed7b7b2cdf12b1d24abc4d0731ab1648b3b4909852e`。本轮没有再次独立执行43项；上一版独立43项通过，且本轮除上述单模块修复外实现字节不变。不能把“日志核对”混称本轮独立重跑。

## 批准范围与实际迁移条件

本次 source-candidate 审查范围无 open 实质 finding。`recovery/owner-document-snapshots-v2/independent-review.v2.json` 明确绑定 from/to/transport、subject 与本报告，`preserves_contract=true`、`open_findings=[]`。完整旧 inline/history/operations/ledger 保留、future refs、cost 未补0、compact owner pair 和 transport 的原 v1 审查正面证据及限制继续适用。

本次不重新认证容量模板：此前131,054,594-byte checkpoint 投影仍是有约3.02MiB余量的特定模板，不是任意工作量保证；实际 base/state 若变化需按 fresh subject 核验。普通 load 的历史 lazy 策略及对象/owner大小 bound 仍保留。真实 no-inflight、closed round/pending、最新 usage/state/gate、迁移前后历史/ledger/源绑定及恢复结果仍由主 Agent 在实际 gate 核查，未因此复验自动放行下一 episode。

小 binding：`review/evidence/native-owner-representation-v2-review-v2/bindings.json`。候选源未由审查者修改，真实 owner 未迁移，旧 v1 报告/证据未覆盖。本轮在 R1 闭合后收束。
