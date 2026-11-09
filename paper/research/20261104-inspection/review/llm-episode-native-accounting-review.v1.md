# LLM episode 原生入账适配：原版独立审查 v1

日期：2026-11-06。**原版不通过本次有界审查：复现 2 个 P1 和 3 个 P2。** 其中时钟失效会遮住同次请求的 seed/causal 错误并允许重试；spawn 后写 started 失败以及正常返回路径可能留下活子进程。原 12 项测试确实通过，但没有覆盖这些组合与生命周期窗口。

本报告绑定下述原版快照，主 Agent 随后开始的修复不回填本报告，也未获本报告预先批准。全部测试与反例均为**明确 synthetic 的 CPU 子进程、假 world、假模型 service 记录、假 evaluator/owner effect**；没有 MLX、GPU、真实物理、LLM 生成、付费 API、实际 owner reservation 或 `.paper` 写入。不是实际模型/飞行/G4 admission 证明，也不是统计协议重设计。

## 版本、权限及证据冻结

只新增本报告和独占证据目录：

`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/llm-episode-native-accounting-v1/`

下文以 `E/` 指该证据根，`E/source/` 为开始审查时逐字节复制的源。所有反例 import **该快照**，没有在主 Agent 修补后的模块上继续跑原版探针。未修改实现、控制器、输入、live logs 或他人文件。

| 原对象 | SHA-256 |
|---|---|
| `eval/inspection_llm_episode_receipts.py` | `f7e2d611df1581bf7363989774a5a5f7569fcb834f37891bc65d307faa64863e` |
| `eval/run_inspection_llm_episode.py` | `0ea9f8e7664cceea554f0ecdfa44fde37973b86588074195566af71dfa4222c2` |
| `tests/test_inspection_llm_episode_receipts.py` | `83d21e5889ed2290683f5f24194fd0e367bed529cc04a77b44bb5e2548a7e3e3` |
| shared classic receipt/physical validator | `59f88d5a3ec90497e19ec91ac9b914f09fe6a2303d029e6aecbb6fb2f958387b` |
| strict model-operation meter | `922372548ceb972e9f95d1c52923cfe7d9dd6ce12dba2feec3b2d44a99167683` |
| parser/public protocol | `11a4e86348dad6fa94849f20a1086c21c24a09307d3e9f9c4fd8d642c14bcd91` |
| existing process cleanup helper | `9b5e3931a1d987066bd190a5d7df935f1edd239a727866cdc7b76f6f6f420a27` |

原 service 源也已复制，完整 8 项源哈希见 `E/source-binding.json`，SHA `5b8c4bcf4d7bb0d01c6ddae112b5b280a405a8c9591353ea6159e8bd0495d58f`。测试用 Python 为仓库 `.venv/bin/python`（3.11.15），禁写 bytecode、禁用 pytest cache。

| 核心执行证据 | SHA-256 |
|---|---|
| `E/pytest.log` | `f5bbe17480d4c1e5f755c22e010a6c83812c7ccb7b27ea35483a1ba2ab9e34c1` |
| `E/probe-results.json` | `f96f7439eafadb0c6ddc5f8cd994509269169f3e6c8e98d1ac756286a988f2c3` |
| `E/boundary-results.json` | `2bd340cd49eb5e8ac7141acb44868c68e9e00d4e77a0c270c717f8fd91bda5df` |
| `E/evidence-index.json`：616 个源/脚本/日志/fixture 文件的完整哈希 | `f1f3398c822953ade1d9c162804052ca428f6f14c3d02ee9986200593d47aa2d` |

## R1 [P1] clock/process 失败先分类，跳过非物理 admission，导致 causal 错误被重试

原 supervisor `run_inspection_llm_episode.py:179–187` 仅当 `value is not None` 时调用完整 `validate_llm_episode`；`retry` 却先由 meter 已知及 physical failure allowlist 得出。缓存失败重试路径 `110–115` 也只复查 meter/physical。因而只要同时存在 clock 错误，seed、消息/解码、其他非物理 admission 缺陷可能完全不被核查。

实测 `E/probes/clock-plus-wrong-seed/`：在原 `first-clock` fixture 中只增加“第一 attempt 实际生成 seed 错 1”的组合故障，第二 attempt 正常；plan/profile/实际脚本源 SHA 均按 fixture 预先绑定。第一 attempt：

- max lag=.2，physical reasons 只有 `clock_lag_exceeded`；
- actor public seed=**42**，原 operation `call-0001/request.json` seed=**43**；
- 原 terminal 为 `known_operational_failure`，`retry_allowed=true`、`admission_error=null`；
- 实际触发 **2** 次 synthetic child dispatch、两轮 `budget-reserve/budget-settle`，第二次返回 Y=0。

这是已知 seed/causal 错误被 clock 分类遮住后的自动重试，不是单纯的未知使用量。单独 wrong-seed 原测试会阻断，组合后却允许，不能用“单故障测试通过”关闭此问题。两个 attempt 的请求资源都有计数与 settle，并不是零费用调用；问题是违背“causal/source/score fault 不进入基础设施 retry”的方法对称性与准入条件。

建议将可独立完成的 source/model/causal/frozen-score 审计与 physical qualification 拆开；meter 已知的失败 attempt 也必须先排除这些非物理故障。仅在证据完整、非物理链全部合格且剩余原因纯属预声明基础设施集合时，才可 retry；日志缺失/pending/矛盾保守阻断。修复后需重放该组合，而不只重跑独立 seed/clock 两例。

## R2 [P1] spawn/start/wait 未统一清理；normal 返回不保证已拥有子代退出

原 supervisor `135–143` 在 `Popen` 成功后先写 `started.json`，之后才进入包住 `wait` 的 try/except；正常 `wait` 返回没有 `_stop`。原 `_stop` 即使在事后调用，也只能从当时存活 root 的 parent 关系盘点，不能找回已被 reparent 的任意独立进程组。

**反例 A：started 写入故障。** `E/probes/started-write-failure/` 的真实 CPU child 只先 sleep 30s。注入 `durable_once(started.json)` 的 OSError 后，execute 抛错，但 child PID **66274** 仍活，terminal 不存在，仅发生 reserve。审查脚本随后对该自己创建的 PG 发 SIGKILL 并 wait/reap。没有模型或物理工作。unknown/missing-terminal 的 replay gate 本身阻断了再 dispatch，但没有停止已经拥有的子进程。

**反例 B：normal 返回留下独立 PG 子代。** synthetic child 首先执行：

```python
leak = subprocess.Popen(
    [sys.executable, '-c', 'import time; time.sleep(30)'],
    start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
```

随后写齐原 fixture 的 model/flight/actor 记录并正常 exit 0。原 supervisor 返回 terminal `completed` / Y=0 / known settle 时，这个独立 PG 的 CPU 子代仍活。

首次证据 PID **66276** 见 `E/probes/normal-orphan/dispatch-count.orphan`。追加的可核时序反例 `E/probes/normal-orphan-timing/` 令 root 在 fork 后保持至少 100ms：

- root PID **70388**，子代 PID **70389**；
- fork wall_ns=`1793915778891096000`；root-before-exit wall_ns=`1793915779005761000`，跨度约 **114.665ms**；
- adapter 返回 wall_ns=`1793915779038503000`；随后 `ps` 为 `70389 1 Ss Fri Nov 6 05:56:18 2026`，确认是 reparent 后仍存活的非 zombie；
- 审查脚本最终 kill 自己创建的 PG，后续 `ps` 为空，记录在 `boundary-results.json` 和 `review-process-cleanup.json`。

建议从 Popen 成功起就统一 finally 覆盖 started 写入、wait、正常退出及异常路径。要处理会 setsid 的已拥有 service，需在 root 活着时持续记录 birth/parent 身份及显式 service intent PID，正常完成也要核验/停止已记录的拥有子代；无法证明归属或清理成功应保持 unknown，不能 known/completed。仅事后向已经退出 root 的普通 PG 发信号不足以解决独立 PG 的情况。该建议不声称用户态轮询能捕获任意瞬时 fork→setsid→parent-exit，也不授权杀不明归属的同 UID Python 进程。

## R3 [P2] cache 不复验已 settle 的外部 archive/usage/packet

原 `_bindings`（supervisor `61–64`）只覆盖 attempt 内文件及 model-operation；`resource/complete-attempt.zip`、`usage-receipt.json` 和 `resource-requests/*.settle.json` 在其外。completed cache 路径 `101–107` 不读取这些外部结算证据。

实测 `E/probes/missing-settled-archive/`：原正常 fixture 完成后，删除
`owner/qualification/local-model-episodes/fixture-M1-a0/complete-attempt.zip`；再次 execute，仍返回 **Y=0、cache_reused=true**。child counter 保持 **1**，owner effects 仍只有原 reserve/settle，没有新调用。

这不是“cache 免费重新生成”——没有重飞/生成；它是已宣称整集结算证据链的复核缺口。原 operation 仍能重算 calls/token，不等于 native owner 所引用的归档仍存在且完整。建议 terminal 绑定 external usage receipt、settle packet、archive SHA，并在 cache 校验其存在/哈希/ZIP manifest 与归档 entry 内容一致。ZIP 的 `testzip()` 只检查 CRC，不代替 manifest 语义一致性。此审查没有实际篡改真实 owner ledger，也不推断其全局 stale 检查一定会漏报。

## R4 [P2] 忽略 enqueue witness，接受 601s 与 response sample 矛盾的派发日志

原 admission `inspection_llm_episode_receipts.py:137–140` 用 response 的 `worker_after_response` 和 final 标记计算期望队列，没有读取 decoded 行的 `worker_progress_at_enqueue`。

实测 `E/probes/enqueue-after-deadline/`：合法解析的一条 `enter_safe_exit(drone_id=0, expected_version=0)`，response sample 时钟为 **1.0**，decoded 行的 enqueue witness 为 **601.0**，run mutating_calls=1；所有 model request/response/token 字段自洽。原 adapter 仍 **completed/Y=0**，接受这一包含派发的日志。没有真实 worker tool ACK，也不将“enqueued”说成已经执行成功。

即使已冻结的正常 runner 通常把同一次 sample 写两处，admission 也不应接受两处矛盾。建议明确该 witness 是“对应 response 的同一 sampled progress”并检查四维有限值及精确一致；final/600 截止后期望队列为空。不要将同一旧 sample 的相等性扩大为每条实际 enqueue 时刻或物理执行截止的独立测量保证。

## R5 [P2] 未核 embedded source 文本，与声明 hash 矛盾仍 completed

shared `validate_flight` 只比较 run 的 `source_hashes`/dependency maps 与 unchanged flags；本版 LLM admission 没有 classic measurement loader 所做的 embedded `run.sources` 字节哈希重算。原 12 个 fixture 的 run 甚至没有 sources 字段，不能覆盖真实 source-text 证明。

实测 `E/probes/inconsistent-embedded-source/`：child 写出

```python
run['sources'] = {name: 'mismatched embedded source' for name in plan['source_hashes']}
```

同时保持 `run.source_hashes=plan.source_hashes`、unchanged=true。实际文本 SHA 与每个声明 hash 均不符，原 admission 仍 **completed/Y=0**，详见 `boundary-results.json`。初始 expected map 正确并不能消除保存的原代码文本与该 map 的矛盾。

建议真实运行必须保存完整非空 string-valued sources，键集合精确等于预声明 source_hashes，并逐项求 UTF-8 SHA 后比较。外层还必须确保 expected 是 dispatch 前的完整实际 import/source closure，而非从事后 run/plan 自我复制；本报告不审新增外围 driver 的完整性，也不把 hash 声明本身当实际执行来源的独立证明。

## 原版已支持且实测到的边界

原 12 项 pytest 在快照上全部通过（`12 passed in 0.52s`）：

- 合成 normal、not_started、policy_timeout、context-gap 的完整 policy Y=0 被保留；再次 cache 不产生新的 child dispatch 或 reserve/settle。cache 仍会调用离线 evaluator，不是完全不启动任何 subprocess。
- 单一已知 clock 故障可一次重试，连续 clock 故障到 2 attempt 后穷尽且不再启动第三次；这些不能替代 R1 的组合验证。
- 单独 wrong-seed/source-flag fault 阻断 retry；missing service completion 按 unknown usage settle，usage={}，没有补成零调用；missing terminal、不同 plan、payload 篡改、并发 lock 都阻断。
- strict model-operation validator 会重算原 call-* 连续序号、完整文件树、request profile/headroom、response 与 token journal、deadline、completion usage、reservation 上限。其 token meter 是 submitted input + runtime-observed generated，不是 API billing、FLOPs、资本/能源或隐藏 work 的独立计量。

独立附加拒绝反例：直接调用只读 admission，绕开 cache hash 的更早拒绝层；删除 actor 中所有 call joins 被拒为 `unreconciled requests, call/decision cap or command counts`；删除原 call 的 response 被拒为 `incomplete or unexpected call record tree`。没有发现这两种“静默漏 call 仍以已知零 usage 入账”的情况。

## native CLI / unknown / held 的准确语义

静态核对 `owner_action`：只构造当前 run 下唯一原生 `scripts/research.py --run ... budget-reserve/budget-settle --packet ... --operation-id ... [--reservation ...]`，自身未写 `.paper`。独立 transport 探针真的启动一个**隔离假 CLI**，核对 argv、保留 stdout transport；非零 exit=2 被拒，log 留存。它不等于在真实接受计划上执行了一次 native reserve/settle。

另用原 SDK `lib.budget`（SHA `89347301003d265d30d30cc4d6f1a0952ff29f4156eb6f8463825601346f687e`）对**内存合成 state**核对 unknown 语义：known calls/token=0，reserved held=**32 calls / 180000 tokens**，unknown operation 存在，下一资源 admit 被拒为 `Reconcile unknown usage before another resource operation; local work remains available`。这支持“native owner 若按该 unknown packet 结算，则 held/资源阻断”的源码语义；未验证任何实际 live ledger entry。

必须区分 **usage unknown** 与 **episode terminal missing**：原流程先 known settle，再写 terminal。`E/probes/terminal-write-after-known-settle/` 注入末尾 terminal 写故障后，下一次 execute 因 missing terminal 阻断，child dispatch 仍只有 1；但请求已是 **known settle，calls=1/tokens=4**。因此“所有 missing terminal 都仍 held”不是本版保证。若 meter 已完整证明，不能只因评分 terminal 缺失便伪称 usage unknown；需要单列 episode 未 reconciled 与 usage 已 known 的恢复规则。未补写 terminal，也未通过重跑恢复。

同样，原版 `known_operational_failure` 状态名混合了 meter 已知和未获 scored admission；不能只看这个名字就推断可 retry，必须看完整原因与 retry_allowed。R1 正是这两个层面的验证未完全分开。

## 实际命令与复验边界

以下均在 `E/source/` 下以原快照运行，Python 绝对路径为仓库 `.venv/bin/python`：

```text
python -B -m pytest tests/test_inspection_llm_episode_receipts.py -q -p no:cacheprovider --basetemp=../pytest-tmp
python -B ../probe_native_accounting.py
python -B ../probe_boundary_details.py
```

输出分别保存为 `pytest.log / probe.log / boundary.log`；脚本、原 transformed child、plan、全部 fake flight/actor/model journal、owner-effect packet/transport、归档及 PID 时序留在证据目录。探针使用 exist_ok=False 保护已生成记录；重放须使用新的同层证据副本，不覆盖本次已冻结 artifact。

所有自建 sleep 进程均已清理：直接 child kill 后 wait/reap；自己创建的独立 PG 子代 kill 后 ps 确认消失。没有因 orphan 测试给任意同 UID Python 进程发信号；`review-process-cleanup.json` 保存 PID 复查。证据目录没有 `.paper`。未对掉电持久性、任意 host loss、未知父子身份或硬 GPU 取消作保证。

**最终判定：原版 R1–R5 保持开放；修补后必须另版重放同一组反例，不以 12 个独立 fixture 通过宣布 G4 或真实整集运行已接通。** 本报告不认证真实模型权重、真实 service/物理并发、实际完整 scorer/源闭包接线、native owner live state、费用、正式 split/trace 或后续源码版本；fixture 的假 evaluator 只输出 Y=0，其通过不能替代真实冻结 evaluator 对完整 score 的重算。

Attention / reviewed by inherited-parent model：同模型家族独立上下文的有界源代码与实际 CPU subprocess 审查，非跨模型盲审或人类专家认证。只覆盖本报告固定原版及保留的 synthetic evidence；主 Agent 正在进行的修复须独立另版审查。
