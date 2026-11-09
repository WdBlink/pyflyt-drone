# 统一执行器独立工程复核 v1

审查日期为 2026-11-05。审查者为独立上下文 `/root/mission_review`。依据为 Architect Verify、show-me-your-work 的独立审查要求及 Codex 适配规则。本审查属于同模型家族的独立上下文，不冒称跨模型家族或人工领域评审。Git HEAD 为 `270c6f0946b3be59527b2b77982f75a70da7ef77`，inspection 变更尚未由该 HEAD 跟踪，以下 SHA-256 才是实现版本依据。

只写本文件。没有改生产代码、回滚他人改动、运行大型 grid、付费 API、600 s 任务或完整 replay。源码在审查期间持续推进，因此保留反例版本及最后复验版本，结论不自动扩展到后续编辑。

## 当前结论

最后复验的 executor=`30db1cc7…`、mission=`0cf99e1b…`、reference=`85f97b6c…` 下，本次复现的四项具体问题已修复。9 个现有执行器检查函数直接调用通过，补充的 malformed/late planner 续飞和逐物理步 tracking 导出检查也通过。没有发现仍需阻止继续开发的具体代码缺陷。

**当前仍是 development-unqualified 的新执行器。新执行器内部计划事件重放、新 G1 与完整新 G3 尚未完成。** 这三项是资格结论的实际缺口，不能用本次有界检查、旧 G1、20 s smoke 或旧加速 B0 分数替代。任意换区、持续观察、交接和恢复的共同物理包线还需新资格取证，G4、主模型、统计适配和 human_review_pending 均未因此改变。

## 本次发现、修复与反例

### F1 [P1，已修复] 存活但无回复的 planner 长期占住唯一 pending

反例版本为 execution=`cfb09d95e7374227aaa0fee2f3519f1031b68f475f2f86259c7cd1f0c20bdcc1`。当时 `advance:207–211` 只检测进程退出；`pending` 只有拿到回应才释放，而 `advance:335` 附近要求 pending 为空才能提交后续请求。

本次用真实 executor 状态机、一个明确模拟“存活且不回包”的 process 替身和合法 pending request 做内存探针。t=91 时旧 survey 已 failed，pending 仍存在；t=92 新 exit 仍 queued，infrastructure_errors 为空。实际已提交程序继续推进，不是物理停摆，但全机后续规划被锁住，且 runner 只据 infrastructure_errors 区分基础设施失败，可能将 planner 故障混入正常策略结果。

最后版本在 `execution.py:187–188,272–288` 记录 2 s simulation capture 过期，并使用独立 5 s 墙钟回复预算。超时记 planner_no_reply_timeout、清 pending、停止继续请求，必要时终止 planner，不在物理 tick 中 join。runner 根据 infrastructure_errors 将 attempt 标为 operational_failure。飞机仍取此前已提交的完整续段。

补充复验把 deadline 设为已过期，不真实等待 5 s。结果 pending 被清除，基础设施错误被记录，替身 terminate 被调用，四机续段保持。新增回归函数也通过。此处的 5 s 是开发故障分类阈值，不是已资格的实时 SLA。

### F2 [P2，已修复并保留资格边界] 捕获门只检查 XY

同一反例版本在 `execution.py:218–228` 只检查 XY 距离 ≤6 m。本次给出 XY 精确落在原计划上、0 号机实际 Z=92 m 而原期望 Z=72 m 的合法有限导航数组，proposal 被 program_committed，revision 增为 1，job 进入 running。这个内存状态机探针没有改变真实飞机或证明新轨迹可飞。

最后版本的 `_capture` 位于 `execution.py:212–227`，检查 XY、高度区间、航向、速度和 bank 差。原 Z92/72 反例现在产生 capture_envelope_rejected，revision 不变。新增 `audit_tracking` 在逐物理步记录水平误差和高度区间越界；`mission.py:60–66` 接入实际物理 callback，`mission.py:416–418` 导出 execution_tracking_240hz。3 m 稳态高度偏差的内存计数反例，以及真实 2 个物理步后严格 JSON 导出，均通过。

高度模型需要准确描述。变更高度后的前 20 s 使用旧/新高度之间加两侧 2 m 的 transition interval；之后才是新目标 ±2 m。它不是全时刻均处于新高度 2 m 管内。水平交通准入使用 27 m 水平下界，不依赖高度错层作为防撞依据。6 m 水平误差界、20 s 过渡及稳态 ±2 m 都仍需实际资格，增加监控本身不等于误差管成立。

### F3 [P2，已修复] queued source 被错误标为 stale，未在 receiver 获取后释放

反例版本的 `execution.py:143–152` 只取消 running source，但 mission 的 handoff 入口接受 active queued source。本次建立 source/destination 两个 observe job，精确 source token 仍有效。destination 获得一条新的合格本机样本后 running，source 仍 queued，destination 却被设置 handoff_source_stale_at。

最后版本 `execution.py:149–158` 对精确旧 token 的 queued/running source 都能取消。原反例现在将 source 置 cancelled，并写 handoff_effective_at。原有“source successor 不得被旧 handoff 取消”的反例仍通过。receiver 的样本与服务记录来自本机新测量，不是 ACK、旧包或预测。

这里只验证状态事务；并未据此证明接收机物理获取、双机观察和完整恢复的场景级能力均已通过新包线。那些能力还需真实联合 fixture 和新资格。

### F4 [P2，已修复] B0 在同一次 decide 中仍会重复创建静态报告

反例版本 reference=`e00d38335aeddeb1201a33afc592de1c75103e5d0838f0e008ec34d699ad96b1`。`reference.py:68–78` 只查询 ACK 后的 report_points，不包含本批次尚未 ACK 的新报告。

两架机分别有同一公开位置 `[100,100]`、同类、两份相隔 25 m 的合格 inspect 证据时，一次 decide 产出 report-candidate-0 和 report-candidate-1 两个新 ID。该纯策略反例说明先前空间去重尚不能消除同批重复名额消耗，与 executor 动态物理能力无关。

最后 reference=`85f97b6c…` 在 `reference.py:15–16,29–35,77–94` 纳入 pending_reports 和 alias。复验中同批只发一份报告，ACK 前下一次 decide 不再重复，accepted 时两个句柄指向同 ID，rejected 时临时保留回滚并允许下一轮重试。新增回归函数通过。这是策略逻辑修复，没有重跑新 600 s 结果，不能把未来成绩写回旧 diagnostic。

## 已核对的模块契约

| 检查面 | 实际实现与本次结论 | 尚需资格或明确边界 |
| --- | --- | --- |
| 飞行命令所有权 | `flight.py:149–165` 只接显式 FlightTarget 和 exit_flags；mission 不再写 flight modes/routes/headings。 | 旧 FlightWorld 行为与当前不同，旧 G1 不能背书新模式。 |
| 作业单写者 | `mission.py:31–32` 暴露 read-view alias；检索运行代码中 JobBook create/cancel/transition 仅由 execution.py 调用。planner 不写任务状态。 | Python alias 不是访问控制沙箱；此结论针对实际运行调用链。 |
| 因果输入 | 每机 local/support 独立；跨机种子经 mission 已到达 observations 选择；ObservedTrack 带 consumer、measured/available 和证据 ID。 | 没有用 entity_key/隐藏未来救轨迹。相同可见历史不同未来的完整成对物理 fixture 本次未运行。 |
| 本地滤波与服务 | `execution.py:119–146` 拒绝未来时间、重复 ID、旧/同刻回退；只有 job 创建后的新合格本机样本可启动服务。 | LocalTrack 估计用于导航，不能生成报告证据；服务开始、采集与 ground 到达是不同事实。 |
| 规划边界 | `planning.py` 只导入 control/guidance/paths；计划在独立进程构建，commit=now+2，携带 revision 与四机 token。 | PlanRequest/Proposal 尚无已实测外部 QualifiedEnvelope，当前 envelope ID 明确 unqualified。 |
| 物理 tick | 每 tick 最多 poll 一份 bounded reply，reply 限制约 1.1 MB；队列 maxsize=1，put_nowait，未做 join 等待。request audit 已改为排除 programs 后取 fields，避免先 deepcopy 整个航迹。 | 反序列化、传感器 ray、状态复制、服务结算与物理仍占 worker 预算；“有界”不是 8.333 ms 或 1× 已通过。 |
| 计划失效 | poll/adopt 都检查 revision、tokens 和 job 非终态；cancel 不清物理占用；过期/错误 reply 记故障或丢弃而不抛出停飞。 | 本次 malformed/late reply 内存检查证明四机原续段不变；未模拟长期 OS/CPU/网络资源争用。 |
| 取消/退休 | 取消逻辑作业，保留 committed program；mission 的外生 retirement 通过 executor preempt exit，旧 token proposal 不能复活。 | 退休、取消、deadline 和捕获同 tick 的完整联合物理事务仍需新增边界取证。 |
| 恢复与尾环 | `planning.py:28–47,174–277` 每个候选含有限工作与目标无关 recovery；`paths.py:253–300` 检查连续前缀、闭合等周期尾环与完整名义交通占用。 | 等周期尾环和恒定 wind 是该几何证明前提；解析几何不证明 roll、垂直捕获、风下跟踪管或实际恢复可飞。 |
| 几何交通门 | fleet_clearance 验证样本插值速度 ≤27 m/s，扣除 27×sample_dt 的连续时间界，水平下界至少 27 m。 | 两机各 6 m 管成立时才推得实际水平距 ≥15 m。新 tracking 数据与门槛需要冻结、执行和独立审查。 |
| 服务/上传截止 | survey 从实测 route capture 起计且需 lap；observe 从首份新样本起计，逐 3 s 格及末尾 partial bin 检查，缺测不补时。 | 6 s 是 transport/scorer grace，未被拿来放宽本地缺测。末尾 partial bin 的公开工具说明还应在资格契约中写清端点。 |

FlightExecutor 的现有工作路线接受任意公开 stripe 并由有限连接器规划，已经不是旧固定 stripe 实现。**接口能力和纯几何候选存在不等于真实四机跨区 pass、持续 observe、handoff、恢复的联合成功已通过。** 本次不把旧架构的固定 stripe 限制误写成新代码现状，也不把新增接口当作物理资格。

Mission 在 `mission.py:30` 另外覆盖控制器为 pitch_p=4.5、elevator_trim=-.1。新 G1/G3 必须绑定这个实际控制配置、18 m/s、新爬升/减速/恢复模式和实际 source，而不能只运行使用 FlightWorld 默认 gains 的旧资格脚本后套用结论。

## 现有运行产物与归因

本次核验 run/world 的绑定，没有重跑这些任务。

| 记录 | 实际结果 | 本次允许的解释 |
| --- | --- | --- |
| executor-smoke.v1 | 20 s wallclock_1x；wall=20.006502875010483 s；max lag=.09067995871106783 s；4800/4800 审计；H=1,E=0。 | 仅是其冻结旧版本的一次短 smoke。execution/paths/planning/mission/reference/runner 均与最后复验版有差异，不能作为当前完整 1× 资格。 |
| executor-diagnostic.v1 | 600 s accelerated；wall=46.36888900003396 s；144000/144000 审计；H=E=1，C=.9839203675344563，S=.9，D=0，Y=.6279734558448188，SR=0。 | 验证其旧冻结版本的长程诊断与退出。不是 1×、新策略结果或正式任务比较。 |

diagnostic 的工具日记有 44 survey、41 cancel、33 inspect、10 report 和 3 exit，**没有 observe 调用**。10 个 accepted 新报告全部是 static，148.50833333333333 s 前用满；报告引用证据对应的隐藏 source entity 计数为 truth-004 三份、truth-001/000/002 各两份、truth-005 一份，即五个实体占十个名额。这个 evaluator-only 审查用于定位重复消耗，不给 Actor 真值反馈。

因此旧 D=0 不能解释为“executor 持续动态能力低”，因为该策略没有请求 observe。也不能反向解释为“动态能力已好，只是策略差”；当前缺少该能力的合格正例。B0 多数类别/空间及 pending 去重修复后的新结果尚未运行，旧运行保持原 hash 与原结论。

## 必须保留的未通过项

1. **执行器内部计划事件重放尚未实现。** 当前 `eval/replay_inspection_reference.py:47,65–76` 重放外部工具日记，会构造正常 Mission 并启动异步 planner；比较字段不含 execution_audit、tracking、基础设施错误。新执行器的 request/response 到达 tick、接受/拒绝、revision、capture metrics、committed program 和 planner 故障都是行为输入。只重复外部 RPC 或比较旧 13 组字段不能确定性验证这些内部事件。需要另建版本显式记录/重放它们，保留实际输入版本和失败；不要改变旧 v5 的 replay 结果来声称新系统通过。
2. **新 G1 未冻结/执行。** 需覆盖实际控制 gains、18 m/s、转场与 bank reversal、爬升/减速捕获、有限观察、cancel/retire 后的 recovery 和尾环，并验证水平/高度/相位状态界。旧 100-slot G1 仍只属于旧资格版本。
3. **新 G3 未通过。** 需要冻结上述 envelope 与错误规则后，完成含实际 planner/sensor/transport 的完整当前源码 1×，记录 CPU 争用、deadline、expired/no-reply、tracking breaches 和新观察/跨区非空成功链。20 s 旧 smoke 与加速 600 s 不能替代。
4. **联合场景证据不足。** 本次现有 9 个测试主要是状态/因果/几何及 1 s 物理推进。它们不替代实际进入别区完成 pass、持续 observe、receiver 实际 acquisition、双机交通冲突拒绝与安全恢复。所有故障 attempt 应保留，不能只收集成功片段。

这些缺口阻止资格完成结论，不要求无界扩建规划器，也不授权正式 benchmark、主模型或付费计算。human_review_pending。

## 实际执行的最小检查

均使用 `.venv/bin/python -B` 和 `PYTHONDONTWRITEBYTECODE=1`，没有写临时证据或 pytest cache。

- 阅读 architecture judgment/rationale、execution/planning/paths/flight/mission/reference、相关 guidance/scenario/control、现有 tests 和 replay/runner 关联调用。
- 在旧 cfb09d95 executor 复现 queued handoff、Z92/72 捕获、存活无回复 pending 三项状态机反例；在旧 e00d3833 reference 复现同批重复报告。
- 对最后版本直接调用 9 个现有 test 函数，全部通过。没有把直接函数调用说成完整 pytest suite；其中一个检查实际积分四机 1 s、240 次物理审计并持续位移，无 planner 回包也不暂停。
- 复验 queued handoff 正例、Z capture 拒绝、过期 wall deadline 的基础设施分类、3 m 稳态高度 breach 计数、B0 accepted/rejected ACK 两分支。
- 补充 malformed pickle 及故障后 late reply 检查，advance 正常返回四机 targets，原 program hash 不变。
- 实际 Mission 前进一个 control tick，execution_tracking_240hz.physics_samples=2，business_audit 严格 JSON 可序列化。
- 核验旧 smoke/diagnostic world hash、source 变化和报告/工具日记。最后有界复验期间核心 hash 未变。

## 版本绑定

| 最后复验文件 | SHA-256 |
| --- | --- |
| envs/inspection/execution.py | `30db1cc7701f39caea8ccf72bd212d3e1ff28098e9dbeaa9f3748a61c901fdf8` |
| envs/inspection/planning.py | `1e2326ad3defa40fd3327ad5ed84cfc35f2d83b3c242fccd9f1b87a6e090bdec` |
| envs/inspection/paths.py | `5de746f593206d618199da698b64a053115ca35180b4459e64b428f83ece0d93` |
| envs/inspection/flight.py | `04d8a418945e05e4a0d38d3e4be83171ec86bad581a9ebbaface382372548819` |
| envs/inspection/mission.py | `0cf99e1b7ea79950d208187d5c2b5d1f9475220e9ecf3b0374d7f883f3be6de7` |
| envs/inspection/reference.py | `85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3` |
| envs/inspection/guidance.py | `9ab9c5290cbce60ad7f9fead635318f41561dedd1cfc2f0984b1cd45385bc0b9` |
| envs/inspection/scenario.py | `9b4a80e355eda36057fe28eff603afe9ef0145646f5349ba59a2a01e25900a50` |
| envs/inspection/protocol.py | `c32bdabb67bf8493ff5125d0af4fc289d1d0210db762016a9c56a96dd2f174a4` |
| envs/inspection/control.py | `611c0e8cf51bef2623c45d03047725e67f4a623b600e7e34b3dd18b04b7f721c` |
| tests/test_inspection_execution.py | `b8e19b9401fccc52b1f12be09631e7e1d36cb9522585cc51c0bf355f32fdf7d9` |
| eval/run_inspection_reference.py | `7ac2778d0dab50f098272f5f4a60b6da2d209dc0d2eea7a44c935cc5a6849527` |
| eval/replay_inspection_reference.py | `699c2dc4a6f9bac7b06dadeba5064c6a49a310059686ab05f24c55875b5ca59b` |

| 被读取的历史证据 | SHA-256 |
| --- | --- |
| reference/executor-smoke.v1/run.json | `8c2e45fe983588a4772b4c5d5be6dae9adb47f51dadb2302247cc1e07e3f6be4` |
| reference/executor-smoke.v1/world.json | `b9f20b833006ad84e146b957c6d4289fb0e360e6f28da23271e600930bf2a6d9` |
| reference/executor-diagnostic.v1/run.json | `972a86969ff8875aaa3e5cace451c4e61f78708f6b7acae2b5a8891e14232a83` |
| reference/executor-diagnostic.v1/world.json | `7a44b8e2b686af209d87adc14b1a2419856de4973bc766b771d8211ae09162f0` |

历史证据路径相对 `paper/research/20261104-inspection/qualification/`。它们的冻结 execution hash 为 `992bc2ce…`，reference hash 为 `d136d825…`；并非本次最后复验源码，不回填旧结果。
