# Local model runner integration：独立定向修复复审 v1

结论：本轮对上一版六项发现的定向反例已闭合。修复中途还复现了 world 为 `[]`/`null` 时残留 running 的同类缺陷；主 Agent 补入 dict guard 后，两例均转为已落盘 operational_failure。现有相关 **48 项测试通过**。这只证明已检查的 prototype 集成边界，不是 G4、真实模型硬件、真实 LLM episode 或未知故障恢复资格。

审查为同模型家族独立上下文的工程复审。生产源码、owner/live、真实 `.paper`、经典 cohort、旧报告和旧证据均未修改。所有模型/worker 均为显式 fake 或纯 OS 子进程；**真实 generation=0、GPU forward=0、PyFlyt 物理=0**。

## 实际版本

工作树 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`，HEAD `270c6f0946b3be59527b2b77982f75a70da7ef77`。新增文件以实际内容 SHA-256 为准。

| 文件 | 本轮最终审查 SHA-256 |
|---|---|
| `eval/inspection_llm_actor.py` | `229e78538b8acd0d7cb3b3cdaf972cd5311e7da022f4a267f4eac5862e4abf48` |
| `eval/inspection_model_client.py` | `40ccd5f5f4100e4ed42a6ac22ecf35f3bc24a9db4e2b57f1826625b78afcb158` |
| `eval/run_inspection_llm_development.py` | `471ee3299aa5b1868bc1df2c80448bfebe9edf986661aefb09c007490bc1b075` |
| `eval/inspection_process_cleanup.py` | `9b5e3931a1d987066bd190a5d7df935f1edd239a727866cdc7b76f6f6f420a27` |
| `eval/inspection_llm_protocol.py`（未变） | `c323f708fd2c2eb54a440758bd162785f3ee2fcebd95b9bca21eaf7ed6ef44d4` |
| `eval/inspection_model_service.py`（未变） | `cfbb8e908677d439c56c31ebcc3ca196a72ccecd3d7d6829b07d6aead6c11810` |

本轮最初读取/运行的 runner SHA 为 `73f4d09d1816ac0bc61f331b81afe4280762634a7257f7cf662baceb4793e890`，尚无 world dict guard。该阶段 `[]`/`null` 的 JSON/log 独立保留，未覆盖成通过结果。最终源码快照在 `review/evidence/local-model-runner-integration-repair-v1/source-repair-final/`。初读 runner 副本由最终源码去掉唯一的 dict guard 后重建，其 SHA 与首次实际读取 SHA **逐字节一致**；明确保存在 `source-before-world-type-guard/`，不冒称为修补后源码。

## 六项定向结果

| 原发现 | 实际定向检查 | 最终结论 |
|---|---|---|
| F1：绑定失败仍 completed，缺 world 仍 completed | 原 private 副本删除、冻结 source 变化、world 缺失，均 exit1 / operational_failure。再检查 source 删除、dependency 删除/变化、world599s/143760steps，均安全失败。合法 fake world 为 completed/600/144000，正常分支仍 exit0。runner206–224 将 unchanged 失败与 world 检查并入 error。 | 已关闭已列反例；不是完整分数/回执真实性验收。 |
| F2：坏 completion 在 finally 重复抛错，run 留 running | `{`、缺 completion、usage calls=True 三分支均 exit1 / operational_failure，保存 model_operation_unknown，不再抛出离开 finally。runner195–205 仅在统一 finally 分支解析 completion，并校验 calls/tokens 为非负 native int、未超 profile。 | 原反例已关闭；运行中发现的 world TypeError 残留亦已关闭，见下表。 |
| F3：70k unread stdin 阻塞协调器 | 与原反例相同的 ready 后不读 stdin 的假服务、70,000字符消息。新 submit 约0.00019s返回；80ms transport watchdog 约0.088s后停止 child/writer并抛 ModelOperationUnknown。submitted=1、inflight 保留。另在 blocked writer 上 finish(timeout=.04)，约0.049s返回 unknown并停止 child/writer，没有在 stdin.close 锁上挂住。 | writer thread 与 poll watchdog 关闭原同步阻塞路径；不是 GPU/敌对 OS 硬取消认证。 |
| F4：父 worker 被终止后 daemon planner 存活 | 原隔离 OS 结构保留；本轮调用生产 stop_worker_tree，对 coordinator 自己的 direct child 及其 daemon 孙进程快照后清理，实际 descendants 包含目标孙进程、remaining_live=[]，父与孙均停止。现有 spawn worker/daemon 测试也通过。 | 已覆盖受控本机 owned 进程树清理。 |
| F5：runner 无消融参数、未传 actor/写 manifest | 三个 CLI 值逐一运行 fake main：periodic、no-cross-aircraft-memory、no-progressive-confirmation-guidance。每例截获实际 actor 构造参数与落盘 manifest.ablation，二者与 CLI 完全一致，正常 completed。runner47/51–53/83/95–96 已接通。 | CLI 接线已关闭；不扩大开关实际语义。 |
| F6：job succeeded/accepted report 不触发 B4 | 原 t=0 请求完成、t=10仅新增 succeeded job，现 due=True、public_event_or_job_change。补查 report_slots_used 改变：due=True；accepted report receipt：due=True且report_receipt。actor83–97 现监控全部 job 状态、slots和接受的报告。 | 已关闭原成功/报告监控反例；仍受10s最短事件请求间隔、20s周期与单inflight约束。 |

world 类型残留的两阶段结果单独保留：

| world 值 | 初读 runner `73f4d…` | 最终 runner `471ee3…` |
|---|---|---|
| `[]` | TypeError 离开 finally，磁盘 running | exit1 / operational_failure，run 落盘 |
| `null` | TypeError 离开 finally，磁盘 running | exit1 / operational_failure，run 落盘 |

最终 runner216 的 `not isinstance(world, dict)` 在索引前短路，两例由 ValueError 进入原错误处理。这里没有无限扩展类型审查；只复验了已经发现的共享根因。

## 没有通过更换 fixture 掩盖原问题

主循环探针沿用上一版 fake context/client/process，只补上新正常分支所需的 `simulated_seconds=600`、`physics_steps_executed=144000` 和 `usage={calls:int,tokens:int}`，并为新增冻结 helper 提供 fake 源文件。它不执行 real worker/service，也不伪造一份可被科学分析接纳的 actual world。

为排除“补 fixture 导致旧反例消失”，本轮把**同一份补齐字段的 fixture**再用于上一版冻结 runner `5a8c420e601a3576fdef91fd5c46634f410b40bfa3d9dca4de8cf4a69c6a3ee9`：

| 同一 fixture | 冻结旧版 | 最终修补版 |
|---|---|---|
| 正常 fake completion/world | completed / exit0 | completed / exit0 |
| private 删除 | completed / exit0 / unchanged=False | operational_failure / exit1 |
| source 改变 | completed / exit0 / unchanged=False | operational_failure / exit1 |
| world 缺失 | completed / exit0 | operational_failure / exit1 |
| completion=`{` | JSONDecodeError，run=running | operational_failure / exit1 / model_operation_unknown |

冻结旧 runner 的重跑使用其原源码副本，仅在隔离 fixture 中加载；没有修改旧 report/result/source。结果见 `original-v1-same-fixture.json`。

## 主循环行为回归及保留边界

正常模型 reply：实际 fake main 入队1条命令；end-to-end 耗时超过 fixture cap：0条；shared physical time=600：0条。known runtime failure 与 unknown transport 均 operational_failure/exit1，0条命令；not_started 为 completed/0命令，不能冒称一次生成。两类失败没有自动重跑，也没有转换为 Y0。没有 completion/usage 时保存 unknown，而非推断免费使用。

这里的 progress0→5/600 由 fake worker 提供，证明 runner 的分支和记录接线；它不是新的物理推进、真实1×、完整600s或硬件时延样本。未变的公共 wire/service admission 可沿原精确 SHA 复用其已有纯边界结论；本轮没有重跑真实 admission，也没有触碰任何 live reservation、权重、model service operation 或 owner结算。

client 的默认 watchdog 为120s，service 的结果截止仍是已声明的软 deadline。watchdog 触发后依赖本机进程终止/回收，耗时可超过触发阈值；本轮测的是 .08s fake 阈值与受控进程，不能宣称真实 GPU 必定在120s内停止。stop_worker_tree 以一次 inventory 的后代集合及 birth 字段确认身份，受控测试已通过；并不保证快照后新生/已提前孤立进程、掉电、敌对 OS 或不可终止硬件的完整恢复。

三消融的**接线**现已成立。periodic 确实改变事件请求时机；no-cross-aircraft-memory 仍只是移除持久 plan/note 与派生 alias hints，公共已送达跨机观测缓存保留；no-progressive-confirmation-guidance 仍只是删除对应提示语。原报告对这些语义的限制继续适用，不能改称完全去掉所有跨机推理或确定性递进确认模块。未产生任何真实方法性能证据。

完整分数/时钟/实际模型 request/token receipt 的因果一致性、LLM episode adapter 与 SDK 入账、操作故障的外部 unknown 恢复、实际软件/硬件依赖冻结、动态 full context 和真实生成/时延仍未验收。当前 prototype 的600/144000与 usage shape检查不替代它们；本轮没有新增 G4 或 formal GO。

## 实际命令与证据

全部从仓库运行，主探针、管道探针、OS清理探针及旧版对比都使用 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B`。精确探针源和 JSON/log 保存在 `/tmp/inspection-llm-integration-repair-review-v1/`，交付副本在 `review/evidence/local-model-runner-integration-repair-v1/`。

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-integration-repair-review-v1/probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-integration-repair-review-v1/pipe_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-integration-repair-review-v1/orphan_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-integration-repair-review-v1/compare_original.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_llm_actor.py tests/test_inspection_model_client.py tests/test_inspection_process_cleanup.py tests/test_inspection_llm_protocol.py tests/test_inspection_model_admission.py -q -p no:cacheprovider --basetemp /tmp/inspection-llm-integration-repair-review-v1/pytest
```

实际结果为 **48 passed in0.64s**；主探针最终19个 main分支与3个 CLI消融分支完成，另有 actor监控/截止分支、两种管道关闭路径、实际 OS 后代清理和同fixture旧版对比。report slots/accepted receipt 还用同 frame/actor 的隔离 inline纯函数检查，结果在 report-monitor.json。主探针首次与 dict guard 后分别执行，前后 JSON/log 全部保留。所有本轮 fake 子进程已清理。证据/source SHA 由新 binding.json 固定，旧版证据与报告未回填。
