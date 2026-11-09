# LLM episode 入账适配修复候选：独立复验 v1

日期：2026-11-06。**固定 506f16 候选修复了原五组反例的首次执行/直接拒绝路径，但缓存完整性仍有两处可执行残余，暂不能按该范围验收。** 原时钟+seed 组合不再重试，started 写故障会清理已拥有 child，normal 独立 PG 按“已追踪则清理；未登记则 unknown 阻断”处理；直接删除 archive、601 enqueue、错误 embedded source 均被拒。新增 K1/K2 需要修改 terminal 元数据，不能说成程序会自发升级 unknown。

仅 CPU/subprocess 合成复验：假 flight/world、假 model/token records、假 owner effects、假 evaluator；没有 MLX、GPU、真实 physics/LLM、paid API 或 live owner mutation，不授予 G4。新增 injected-delay 语义明确不在本版覆盖；主 Agent 后续的 37a40…/延迟 guard 不回填本报告。

## 固定对象与证据

证据根 `E/`：

`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/evidence/llm-episode-native-accounting-repair-v1/`

全部检查执行 `E/source/` 的快照，之后 current 源被主 Agent 修订不影响本报告对象。只新增本报告及该证据目录，原报告/证据未覆盖；原报告 SHA 仍为 `90a2415883e924a1a070e4e9326ced2b7247ce89c0b148ddf9b8da11a7aafd16`。

| 候选源 | SHA-256 |
|---|---|
| `inspection_llm_episode_receipts.py` | `a0e4193b78769e3b8e8fd7383bc55de80d1ee00151539ab3337c1301a25adff9` |
| `run_inspection_llm_episode.py` | `506f16abbbc4a71d9e91a680f86de2cfa1ae0dad00ccad48576cbe3f9a0d122b` |
| `inspection_process_cleanup.py` | `361ba11b8ea088627e4c0bd832168bb26821297e5d8f54bf6d43c2ba327a9c70` |
| episode tests snapshot | `907ba637358ba02483e784619b5ea98988001e6422d1af4df72af197505cde2c` |

8 项快照源及共同依赖见 `source-binding.json`，SHA `9a93c5479c02c292b7a8ac7b5712bb7c982a486720379130bb4a9c924092d24d`。

| 实际执行证据 | SHA-256 |
|---|---|
| `pytest.log` | `ff0289d474e6b14ae1706550c6319e5b6ef0f7225697bbc848f7d52ac5ebd98b` |
| `replay-results.json` | `b4ea5946cffe7e6c924ac7997898331f2a64eaf4b21496dc98e2c0c506fe2e6e` |
| `terminal-cache-scope-results.json` | `39dbe3b722d7e91551ad8174da80a78e6db93f44c3815d8ecd9fdcb22c864ea8` |
| `evidence-index.json`：745 个 source/log/script/fixture 文件 | `9c66d80650e8cf91a3be9c8c8f8e944867f0d8d82f1eca4a5ad1959118615250` |

独立实际运行是 **episode 文件 18 passed in 10.58s**；保留的 owner log 为 **19 passed in 11.40s**，后者包含 episode 文件与另一个 cleanup 测试文件。没有为凑计数把 owner log 当独立重现。下面的实际进程追踪/unknown/清理反例由本审查另外执行。

## 原 R1–R5 逐项复验

| 原问题 | 固定候选上的实测结果 | 判定范围 |
|---|---|---|
| R1 clock+wrong-seed | clock=.2；public seed42、actual seed43；只 dispatch1；retry=false；准确错误为 `actual generation request differs from public request`；再调用阻断 | 原组合路径关闭 |
| R2 started 写失败 | child 83063 返回 -15，不再存活；process-tree=closed；无 terminal；仅 reserve，再调用因 missing terminal 阻断 | 原 spawn/start 窗口关闭 |
| R2 normal 快速独立 PG | 子代 83071 未登记，仍活时 process-tree=unknown、terminal=unknown_process_cleanup/Y=None、retry=false；不擅杀不明归属，由审查脚本只清理自己创建的子代 | 初次执行符合新明确范围；缓存见 K2 |
| R2 normal 可追踪独立 PG | root 保持 .25s，子代 83087 记录了 birth/parent/group，normal close 后已无进程；此时允许 completed/Y0 | 已登记子代 normal 清理通过 |
| R3 单独删除 archive | `settled archive/usage/packet changed or missing`；dispatch仍1、owner effects仍原 reserve/settle | 原单文件删除反例关闭；集合遗漏见 K1 |
| R4 原一条合法命令+enqueue601 | 准确错误 `dispatch does not follow original decoded commands/deadline`；retry=false；再调用阻断 | 原 witness 矛盾关闭 |
| R5 embedded sources 全错 | 准确错误 `embedded actual source bytes differ from the pre-dispatch hashes`；retry=false；再调用阻断 | 原 source-text 矛盾关闭 |

原 R1 的反例保留“仅第一 attempt 错 seed，若允许重试第二 attempt 本可正常”的原设定，不是改成永久错误后偶然穷尽。新 `audit_operational_failure=True` 让已知 meter 的 clock 失败也经过消息/seed/解码/source/frozen-score 路径；第一 attempt 的 admission_error 明确标出 seed join 失配而非只看 clock，第二 attempt 从未建立。原两个 owner effects 请求的是如实 known usage；这与是否允许 retry 分开。

原 R4 重放了包含 `enter_safe_exit(drone_id=0, expected_version=0)` 的命令，response sample=1、enqueue witness=601、mutating_calls=1；不是仅用空命令测试。日志中的 enqueue 是“入队未接受”，本审查没有将它称作实际工具执行成功。

所有重放脚本及逐项 transformed child、输入 plan、fake 原请求/响应/token/actor/flight/settle 文件都留在 `E/probes/`。原五条简单反例的改进不消除下面两个缓存变体。

## K1 [P2] settlement_file_hashes 可省略 archive，精确必需集合尚未独立导出

候选 cache 只要求 `terminal.settlement_file_hashes` 非空，然后对**其中列出的**路径检查在 run 下及 digest；没有要求键集合等于该 reservation 的三个必需文件。terminal 本身不在 `_bindings` 的 payload 中。

`E/probes/R3-omitted-archive-binding/` 实测：

1. 正常完成合成 episode，写齐 original archive/usage/settle 文件。
2. 删除 archive，并只从 terminal 的 settlement_file_hashes 移除 `.zip` 键，保留原 usage-receipt 与 settle-packet 两键及值；其他 payload/evidence/model 文件不改。
3. 再调用返回 **Y=0、cache_reused=true**，archive 不存在；dispatch=1，owner effects 仍为原 reserve/settle。

因此“删除 archive 会被拒”成立于 terminal map 未被缩减的测试条件，不能泛化为精确整集结算证明。没有重新生成或新增收费；缺口是缓存接收了不完整的结算证据集合。

应从可信 `run + reservation` 独立导出三个固定绝对路径（archive、usage receipt、reservation settle packet），要求键集合精确相等，再核路径/哈希及适用的归档内容关系。不能从被审 terminal 自己提供的集合决定需要哪些证据。

此候选 source 的 cache 检查是文件 SHA，producer ZIP 使用 `testzip()`/CRC；没有独立重算 ZIP manifest 的语义检查。若报告只声称“同一已固定归档字节的 SHA 身份”，不必将它额外描述为已经实现了 manifest 逐项重放。

## K2 [P1] unknown process proof 可被 terminal status/Y 两字段升级成 cached completed，仍带活孤儿

候选首次执行已经正确产生 unknown，但 cache completed 分支只重跑 `validate_llm_episode`，不独立要求 process-tree proof 为 closed，也不核不可删的 intent/start/process-return/tree 等 receipt 语义。

`E/probes/R2-terminal-status-rewrite/` 使用真实 CPU 快速独立 PG 子代、原真实 inventory 代码，没有 mock process monitor：

- 首次执行正确阻断：terminal=`unknown_process_cleanup`、Y=None，process-tree.status=`unknown`；
- 子代 PID **84829** 当时 `ps` 为 `84829 1 Ss Fri Nov 6 06:14:54 2026`，非 zombie 且仍活；
- 只把 terminal 的 `status` 改为 `completed`、`Y` 改为 0；**没有改 payload hash、evidence、process-tree、source、flight、model-operation、archive 或 settle 文件**；
- 第二次 cache 返回 **Y=0、cache_reused=true**，同一子代仍为 PPID1/Ss 活进程；child dispatch=1，owner effects仍只有原 reserve/settle。

这不是程序在没有数据改动时自行升级，也不是新 flight；它是已明确记录的 unknown 事实被缓存入口信任的两个 terminal 元数据覆盖。事后审查脚本才 kill 自己创建的 84829 并确认 ps 为空。证据包含 `terminal-cache-scope-results.json`、脚本、原 process-tree 及全部未改的 payload。

应独立核验必需 dispatch/progress/cleanup receipt 的 plan/reservation/ordinal/return/stdout 等绑定和 process-tree closed 语义，再判断可 cached completed。缺 receipt 或仍 unknown 的 proof 不能靠 leaf terminal.status/Y 重新标注完成；应保持阻断，等待有证据的 reconciliation。若另有外部可信、预先绑定的 terminal SHA 且在此 API 前验证，两例的 terminal 修改会被外层拦截；本 API 没有该 expected terminal SHA 参数，不能在此报告假定这种锚已经存在。

## 清理保证的准确范围

本候选采用 `ps` inventory 的 parent/birth/group/uid/comm 识别；每次 wait 周期观察已拥有树，normal/exception finally 都 close。实测两条分支分别覆盖：

- 已登记的独立 PG 子代可在原 parent 退出后按记录 birth 清理，保留 owned_births，完成时 remaining_owned 为空。
- 漏过登记的新同 UID Python orphan 不被擅杀，而是 unknown + 不 completed、不 retry；这可能保守阻断同一时段启动的无关 Python daemon。审查没有给其他进程发信号。

只证明本报告这些真实 CPU 子进程时序。它不是 kernel subreaper，不能证明任意瞬时 fork/setsid/parent-exit、非 Python 未登记后代、恶意 host、PID 身份的任意竞态或断电恢复。新范围下，fast orphan 留活但结果 unknown 是预期保守行为，不将其等同于原“留活且 completed”的 P1；K2 是另一个明确带 terminal 修改的缓存路径。

## 未退化项及未覆盖边界

18 项 episode 测试实际保留 policy Y=0/cache 不重飞、not_started/policy_timeout/context-gap、known clock 最多一次 retry、穷尽不第三次 dispatch、missing completion→unknown、单独 source/seed fault 不 retry、payload/plan 变更及并发 lock 阻断。测试 owner 是假 effect callback，ledger 未发生真实 reserve/settle；已知/未知 packet 的生成不等于 live native owner 集成已实测。

本候选仍先 settle 再写 terminal；missing terminal 若发生于 known settle 之后，只阻断 episode，不能称该 usage 仍 held。这个窗口已按任务明确范围处理，不把已知 calls/token 伪写为 unknown。未知 meter 的 held/native budget 语义沿用原审查的只读源码/内存合成验证，没有访问 live `.paper`。

fake evaluator 只打印 score_equal/Y0，故这些测试不能证明实际完整 scorer、真实 case、实际 MLX、真实 physics/trace、权重/费用或正式八条件准入。完整 scorer 的生产保证还要求外围预冻结并绑定真实 evaluator/import closure、Python 与输入来源。本报告不审新增外围 driver、108 槽结果或新 injected-delay 模式；当前 a0e419 版本没有在本复验中获准支持注入延迟绑定。

## 实际执行与冻结判定

工作目录 `E/source/`，使用仓库 `.venv/bin/python`（3.11.15），环境 `PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`：

```text
python -B -m pytest tests/test_inspection_llm_episode_receipts.py -q -p no:cacheprovider --basetemp=../pytest-tmp
python -B ../replay_original_faults.py
python -B ../probe_terminal_cache_scope.py
```

分别生成 pytest/replay/terminal-cache-scope 的 log 与 JSON。无新 MLX/physics/真实 owner 操作。所有自建直接 child 已回收；需要审查脚本处理的自建孤儿已 SIGKILL 并 ps 确认消失，PID 83063/83071/83087/84829 的最终空 inventory 见 `process-cleanup-check.json`。没有 `.paper` 目录生成，也未改旧审查报告。

**固定 a0e419/506f16/361ba 候选：原五条直接路径有实测改进，但 K1、K2 使缓存完整性复验仍不通过。** 主 Agent 已获具体反例并另行修补，此结论只冻结本版，不预先批准 37a40… 或任何后续差异。后续需在新证据目录重放省略 archive key、unknown tree+terminal 两字段以及对应正常/cache 正例，再作小范围关闭判定。

Attention / reviewed by inherited-parent model：同模型家族独立上下文的源代码与真实 CPU subprocess 复验，非跨模型盲审或人类专家认证；全部模型/飞行/owner 数据合成，不认证 G4、main/holdout、真实 generator、实际模型或新增注入延迟边界。
