# Mission 独立工程审查 v1

审查日期为 2026-11-05。审查者为独立子 Agent `/root/mission_review`，与实现上下文分开，属于同模型家族的独立上下文审查，不构成跨模型家族审查。工作树为 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`，Git HEAD 为 `270c6f0946b3be59527b2b77982f75a70da7ef77`。被审查的 inspection 代码尚未被该 HEAD 跟踪，因此以下 SHA-256 才是实现版本依据。

## 结论

当前版本不宜开始作为 G2/G3 资格取证的 600 s 真实 1× 参考运行。先修动态测量封存绕过、相机从机体内部起射、SR 判据、优先级发布边界和失败审计保留，再执行有界回归。若先运行，只能保留为带已知缺陷的诊断 attempt，不能据此宣布 G2/G3 通过。

`diagnostic.v4` 是已经完成的加速 dev 诊断。核验其 `world.json` SHA-256 与 `run.json` 绑定一致，且 run 内全部源码 hash 与本次读取的当前源码一致。其 `C=0.9264931087289433, S=1, D=0.011764705882352941, Y=0.6460859382037655, H=E=1`。其 `SR=1` 来自下述错误实现，按原计划应为 0。它没有证明 1× 时钟，也没有证明动态观察平台已具备完整能力。

没有改动任何代码、控制 grid 或运行证据。本次只新增本报告。没有重跑 600 s，没有运行 pytest，没有重跑仍在进行的 G1-full.v1，也未把新的 handoff fixture 当作正式结果。

## 进入资格参考运行前的具体问题

### F1 [P1] 重排 evidence_ids 可以改写已经封存的动态测量

位置为 `envs/inspection/protocol.py:112–123`，关联评分位置为 `envs/inspection/scoring.py:100–104`。`measurement_id` 直接取请求数组的第一个 evidence ID，只有这个 ID 相同才检查是否修改已有测量。数组顺序因此变成一个未校验的封存别名。

本次用真实 `ReportLedger` 与真实动态 scorer 做了纯内存探针。注册同一实体、同一 candidate 的 `e1(t=1)` 和 `e2(t=2)`，两份质量均为 .95 且位置均为 `(0,0)`。首次提交 `measured_at=2, evidence_ids=[e2,e1], claimed_class=wrong` 获受理。保持顺序改为 `risk` 会被拒绝，但改用 `[e1,e2]` 后提交 `risk` 获受理。账本同时出现 `(e2,2,wrong)` 与 `(e1,2,risk)`。同刻按 evidence ID 升序决胜，第一格选择 `e1` 并得 1 分，整体 D 从该格零分变为 `1/60`。

这违反原计划第 142 行的动态点封存，也绕过第 144 行“最新错误样本零分，不回退正确样本”的意义。现有 `test_dynamic_measurement_is_sealed_and_later_new_points_keep_slot` 只使用单 evidence ID，未覆盖该路径。

修复应为每个新测量确定不可被请求顺序改变的原始采样身份，要求该身份的实际采集时间与声明时间一致，并禁止同一原始新测量借助别名重提交不同估计/类别。多份历史证据可以作为辅助 provenance，但不应重新定义“哪个新测量”。补充数组重排、添加历史证据和同刻不同证据的正反例。

### F2 [P1] 射线从机体中心起射，无遮挡目标会被本机部件以 fraction=0 拦截

位置为 `envs/inspection/mission.py:123–135,151–157`。传感器原点直接使用飞机位置，公开卡片则声明理想稳定云台。代码没有定义相机安装外参，也没有排除起射机的碰撞几何。

本次创建实际 `InspectionMission(0, reference=True)`，未推进物理，未更改飞机状态。把一个 ground proxy 移到 0 号机正下方，仅作为受控传感器 fixture。飞机位置为 `[-249.37345952159944,-379.17448884544456,72]`，proxy 高 .6 m。`rayTest` 返回 `(body=1, link=6, hitFraction=0.0)`，其中 body 1 是本机。`_visible` 因而返回 False。

诊断对照只把探针射线起点临时放到飞机下方 2 m，不改变实际实现或飞机状态。此时正下方无遮挡目标返回 True，300 m 离轴目标返回 False，在射线路径中加入 box 后返回 False。该探针说明外部遮挡逻辑存在，但当前安装原点把内部起射造成的自遮挡混进了可见性。

需要冻结一个合理的相机安装模型，或者明确并正确处理自身几何，再用实际 ray 的无遮挡、机体姿态、离轴和外部遮挡正反例校验。不能把临时下移 2 m 的探针本身当作已校准的安装方案。这一问题影响 C、候选发现和后续传感器可达性校准，不由当前较高 C 自动消除。

### F3 [P2] SR 使用聚合 Q 阈值，和冻结前候选协议的各项门槛不一致

位置为 `envs/inspection/mission.py:395–396`。实现是 `SR=int(H*E*Q>=.6)`，原计划第 156 行是 `I(H=1,E=1,C>=.8,S>=.6,D>=.5)`。`diagnostic.v4` 正好展示影响。D 只有约 .01176，却因静态和覆盖较高而输出 SR=1。

需要使用完整、公开且有版本的成功条件，补充“Q 高但 D 未达门槛”的反例。旧 diagnostic 的原始 JSON 应保留；修正后的重评分另建版本，并标注它是旧轨迹的 scorer 重评分，不能覆盖原 attempt。

### F4 [P2] 180 s 公开快照仍发布旧任务卡，评分已经使用新权重

位置为 `envs/inspection/mission.py:79–87,274–281`。事件在物理推进之前处理。某个控制 tick 从 `179.991666…` 推进到 `180.0` 后，传感和公开快照仍使用任务卡 v1。下一次控制 tick 才更新任务卡，下一次定期 state 发布要到 180.5 s。评分的 `weight` 则直接从 `stamp>=180` 起生效。

本次用真实 `InspectionMission.step` 的纯内存 tick fixture 验证，没有模拟飞机物理。将 flight tick 从 `180-1/120` 推到 `180.0`，该次 step 后 `task_card_version=1` 且 events 为空；随后执行 `_events()` 才变成 v2，事件声称 `t=180.0`。首次探针遗漏 fixture 的 `covered` 字段，补齐该内存字段后复验成功，未修改源码。

需要使跨入事件时刻的任务状态在当刻采样/发布前生效，或显式采用并冻结其他一致时序。原计划第 72、114、146 行要求公开生效时间和评分格起点一致，不能仅在日志中写一个较早的 `effective_from`。

### F5 [P2] worker 出错时仅保留摘要，丢失失败 attempt 的已有业务审计

位置为 `eval/run_inspection_reference.py:64–68`。正常完成时调用 `mission.audit()`，异常路径只保留异常、simulated_seconds 与 safety。已经发生的 job、命令、报告、传感、trace 和 coverage evidence 不进入 `world.json`，progress 也只有每秒聚合摘要。

本次通过替换成明确标记的内存 FakeMission，注入一次 worker.step 异常，并将所有 Path 写入截获到内存。结果 JSON keys 只有 `operational_error,safety,simulated_seconds,status`。FakeMission 已有的 `trace/tool_calls` 审计没有保留。这是 runner 的异常路径检查，不是物理运行或真实基础设施故障。

需要在不依赖失败 scorer 的情况下，尽力序列化已有原始审计，分别记录 primary error、audit error 和 cleanup error。主错误不能被二次导出失败覆盖。修复后应有一个短程注入失败 fixture，证明失败仍保留实际版本、已发生的工具/任务和观测证据。

## 各检查面的证据与边界

| 检查面 | 本次结论 | 依据与边界 |
| --- | --- | --- |
| hidden truth/seed/future/grader | actor 输入未发现直接泄露 | `public_state` 深拷贝仅公开状态，`tool` 使用 allowlist，Observation 不含 entity_key。ReferencePolicy.decide 只接收序列化公开字典。truth、pending、sensor_audit 和 score 留在 worker。属于 API 边界，未证明对同用户任意恶意 Python 代码的 OS 沙箱隔离。 |
| 真实位姿、视锥、ray | 实现使用真实位置，存在 F2 | `state(i)[3]`、off-nadir、gimbal 视锥与 `rayTest` 均被使用。理想稳定云台使姿态不直接旋转光轴，是公开建模假设。实际相机安装缺失仍需修。 |
| 噪声独立于调用次数 | 源码符合该项基本机制 | RNG 由 `[seed,tick,drone_id,entity_index,stream]` 构造，未使用工具调用计数。sensor mode/质量改变误差分布属于公开物理/传感条件。未另跑跨策略噪声统计校准。 |
| 报告名额与首次关联 | 源码及现有 fixture 覆盖主要逻辑 | schema 通过的新 ID 消耗不可退名额，撤回不退款，occupied 含已撤回关联；首次按观测时刻与估计位置匹配，类别不参与，20 m 包含边界，距离/隐藏 ID 稳定决胜。association_ambiguities 被记录。 |
| 动态封存 | 未通过 | F1 的实际纯函数探针已复现。Observation frozen 不等于动态报告版本无法别名改写。 |
| ACK、完成、版本、断链 | 主要机制可追踪，未宣布全部 fixture 通过 | ACK 创建 queued，执行才 running，survey 需完整 lap，inspect 需两份相隔 20 m 的合格视角，exit 需实际 dwell；cancel 校验 version 和 link。断链不阻止本机 job，pending 保留原测量时间，恢复时重写真实 arrived_at。已有 tests 的 synthetic clock 与物理 handoff fixture 只被阅读，本次未重跑。 |
| handoff | 源码有 acquire-before-release | destination 获新合格测量后才取消 source，并保留 handoff_effective_at。当前受控 ground proxy fixture 仍属于协议资格检查，未被本报告当作完整场景或正式结果。 |
| 覆盖的因果性 | 到达前不公开，未使用 hidden target 代写 | cells 来自公开 ROI；实际视锥、image quality 和逐 cell ray 生成 packet；离线 packet 保留，交付后更新 covered。F2 会影响其可达性。当前 C 是固定 20 m cell 中心采样的面积近似。 |
| 完整分母和同刻真值 | 源码符合主要公式 | 遍历全部有 rule_id 实体，未发现实体仍留分母；6 static、3×60 dynamic；动态调用 `entity.position(selected.measured_at)`。预筛合法且及时，再按时间与 ID 排序，最后检查身份/类别/质量。v4 静态权重分母 10，动态权重分母 340。 |
| worker 队列是否等待 actor | 没有发现正常路径上的满队列等待 | `mp.Queue()` 不设置 maxsize，worker 不等待 actor RPC 返回；每 tick 最多处理 16 命令。JSON/传感/命令执行仍在物理 worker 同一线程，可能产生 lag；未用慢消费者或资源竞争跑长期压力测试。不能据此声称已满足 .1 s lag。 |
| 运行失败、版本绑定 | 绑定已有，失败审计有 F5 | runner 保存源码全文及 hash、world hash、结束时 sources_unchanged；v4 hash 已实际核验。run manifest 尚未绑定 gate-contract、测试、依赖与飞机资产 hash，也未把它们称为 G4 完整冻结。 |

## 仍需明确保留的资格边界

1. 当前 `assign_survey` 只接受本机 `stripe-i`，inspect/observe 只控制理想云台，飞机继续固定路线。任意跨 region 重分配没有实现。这是已知能力边界，不能因此宣布完整平台或 G4 就绪，本报告没有要求顺手扩建调度系统。
2. `grade` 的 E 仅检查四机当前退出 dwell。原计划第 154 行要求的显式航程/时间预算与退出余量尚未实现，也没有在任务卡公开。不能把当前 E=1 解释为已通过完整资源终态资格。应补齐预算检查，或在新的 dev 资格说明中准确收窄 E 的含义并保持原要求未通过，不能悄然改写原计划。
3. runner 记录 lag，却不会因超过 G3 的 .1 s 容限自动改变 `completed`；`sources_unchanged=false` 也仅影响进程返回码。下一次 1× 结果必须另行按 gate-contract 验证，`completed` 只是执行结束。若保留这种 runner 结构，外层 gate verdict 必须明确检查 lag、source binding、600 s 完整时长及退出码。
4. 现有 v4 的 D 很低，且没有 handoff tool 调用。它能证明公开队列下有静态交付与物理安全退出，不能证明连续动态观察或场景级交接质量。G3 的参考门槛 Y>=.2/static recall>=.5 本身较弱，不应扩大解释成动态任务通过。
5. 主模型运行、统一模型/工具预算、统计适配、正式资源授权、G4 freeze 与 human review 均未完成。当前仍为 `human_review_pending`。
6. 本次没有核验或修改控制器的六项已通过审查，也没有重新审核 G1 grid。该证据仍由父任务独立汇总。

## 实际执行的有界检查

全部使用 `.venv/bin/python -B` 并设置 `PYTHONDONTWRITEBYTECODE=1`，未生成 pycache 或临时证据文件。

- 读取指定代码、现有协议 tests、gate-contract G2/G3 和原计划第 4–6 节，补读 flight 的原点/每物理步审计集成。
- 对 v4 的 run/world 做 SHA-256 绑定与当前源码等值核验。
- 真实 ReportLedger + scorer 的 evidence_ids 重排反例，结果如 F1。
- 真实 PyBullet 零物理步、受控 ground proxy 的 camera ray 检查，结果如 F2；未移动飞机。
- 真实 Mission.step 的内存 tick 替身，检查 180 s 任务卡边界，结果如 F4。
- worker 的内存 FakeMission 注入失败，文件写入全部截获到内存，结果如 F5。

## 被审查版本

以下 hash 在本报告写入前复核，与首轮读取一致。后续修复必须另做新版本审查，不能沿用本报告的问题结论当作新源码已通过。

| 文件 | SHA-256 |
| --- | --- |
| `envs/inspection/protocol.py` | `80aae91f7f8ccbbcf69ee04c6d88ffede7b9408942d47aa4321c3a9a880e4de3` |
| `envs/inspection/scenario.py` | `c68c9e95d518e0d03eab17092ef9b63c81206f6869a92436e440cdbc7717cad3` |
| `envs/inspection/mission.py` | `edecff4bddb310e4b32acf701255250e33d464364fbc7f22f90769848052b49a` |
| `envs/inspection/reference.py` | `c1535f4d10ca48018dde6a6005bc3ca7b45cd6bc4694063504336e21f37cc8fe` |
| `envs/inspection/scoring.py` | `bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a` |
| `envs/inspection/flight.py` | `272efaf8ad0d4aadbfefaeec25e8be64a75c601e5fd6d85407f4ccd9e1fd9686` |
| `eval/run_inspection_reference.py` | `8d6e079bf4c4ce732358e45ab526079d1ced9345ee11cb27a26277a35ff8673d` |
| `tests/test_inspection_protocol.py` | `33d496d39d2337536508a22bccec25f3c0cde54b5550740358d18433ff6e4cf4` |
| `planning/gate-contract.v3.json` | `6723955a0efe68801af57fd3adf1a1b94c5c8cdaea0e7d0437fc84862a7b37de` |
| `inputs/BENCHMARK_RESEARCH_PLAN.original.md` | `dd01e21442950abccc543bb2ba4d53d1901fb2669dba21cbfe9aad7f13e591b0` |
| `qualification/reference/diagnostic.v4/world.json` | `04bd5990234ec5b81405980eaf0274f5a0f54cba10d2525b2fd0694b41c8da28` |

表中 planning、inputs、qualification 三项的路径均相对 `paper/research/20261104-inspection/`。

## 追加复验：修复后的 B0 参考运行启动条件

同日收到父任务的修复通知后，读取了实际源码差异并做最小复验。以下结论仅对本节的新 hash 生效，替代前文“当前版本不宜开始”的启动判断；前文保留为旧版本审查记录，不删除或改写旧 diagnostic。

**未发现阻止启动这条 600 s 真实 1× B0 dev 参考运行的剩余工程阻断。** 此结论允许开始取证，不表示尚未执行的 1× 运行、G2/G3 全部检查、动态高质量追踪、完整平台或 G4 已通过。父任务仍需汇总完整 tests 的实际结果，并在运行结束后检查 lag、完整时长、source binding、各分数和退出状态。`completed` 仍仅表示执行结束。

本次直接调用了新增的六个回归检查函数，复用 pytest.MonkeyPatch 做内存隔离，但没有运行 pytest 命令。worker fixture 的文件 IO 被 MemoryPath 截获到内存，因此没有在允许的报告路径之外写入文件。此外补做实际 camera ray 三分支和 1 s 逐物理步航程检查，共八项有界检查通过。

| 原问题或补充项 | 复验结果 | 实际检查 |
| --- | --- | --- |
| F1 动态封存 | 原重排反例被拒绝 | 新 ID 从声明 measured_at 对应的新证据中确定，历史 e1 不再替换 t2 的 e2。新增重排检查通过，账本仅保留一个版本。 |
| F2 相机自遮挡 | 原正下方 fixture 通过 | 新 optical origin 使用实际姿态旋转 body `(0,0,-.3)`，Observation 增加 camera_xyz，任务卡公开外参。使用实际新 origin 的正下方无遮挡返回 True，300 m 离轴返回 False，外部 box 遮挡返回 False。 |
| F3 SR | 分项门槛反例通过 | Q>.6 但 D=.01 时 SR=0，不再把静态/覆盖对动态门槛的补偿当作成功。 |
| F4 事件发布 | 180.0 s 边界检查通过 | 积分后、采样/发布前再执行当刻事件，180.0 的 public state 已为任务卡 v2。该检查仍是 tick fixture，不是 180 s 真实飞行。 |
| F5 失败审计 | 实际短程注入失败保留审计 | 新 worker 异常路径调用不依赖 scorer 的 business_audit。新增 fixture 在实际 warmup 后、任务约 .1 s 注入 step 异常，partial_business_audit 保留 tool_calls 和 job_transitions，primary error 仍在。 |
| 非有限输入日志 | 严格 JSON 序列化通过 | ledger 与 mission 对 NaN 请求保留拒绝日志，allow_nan=False 可以序列化；malformed RPC 改为 rejection，不再直接终止世界。 |
| E 的资源条件 | 明确航程预算和余量已落地 | 任务卡公开每机 18000 m 航程预算和 1000 m 退出余量；E 同时检查实际 dwell 与余量。将累计里程设为 17500 m 的反例使 E=0。这是时间/航程代理，未称真实电池。 |
| 实际里程累计 | 1 s 小运行通过 | 四机累计距离分别为 `20.095705820986673,20.095286910442624,20.08834024675149,20.099870600669984` m；`physics_audits=physics_steps=240`，确认里程在逐物理步 callback 中累计。 |

固定 stripe、inspect/observe 下固定飞行路线、动态跟踪质量、主模型与统计适配、G4 freeze 和 `human_review_pending` 的边界保持不变。新增相机外参与范围预算改变了 dev 平台版本，旧 diagnostic.v4 不能替代新版本的参考运行或传感校准。没有重跑 600 s、物理 handoff fixture 或控制 grid，没有新增代码改动。

本次复验源码读取前后 hash 一致。

| 文件 | 修复后 SHA-256 |
| --- | --- |
| `envs/inspection/protocol.py` | `c32bdabb67bf8493ff5125d0af4fc289d1d0210db762016a9c56a96dd2f174a4` |
| `envs/inspection/scenario.py` | `a402003a8c20d464f6a80a9d5b6fbc925895403c0887c7afeded0cebae691c58` |
| `envs/inspection/mission.py` | `75031106086a467326e6851a932809c61609952d09bb8171c8bae2bf74b528f5` |
| `eval/run_inspection_reference.py` | `aab0172251aa8aa3cf8cbcb5ad49aa1572ef33d5fe8bebbc11d5e909a58209bf` |
| `tests/test_inspection_protocol.py` | `f986274171e43257b280cf2de875ba07634a5869e3c38c8fe23fe5e34f3ef832` |

`reference.py`、`scoring.py`、`flight.py` 与前文 hash 相同。gate-contract 和原计划未因本次复验改写。

## 追加审查：realtime.v1 的真实 1× 参考证据

本节审查已完成的 `qualification/reference/realtime.v1`，以及同目录下的 `realtime.v1.replay.json`、`realtime.v1.analysis.json`。本次没有重新运行 600 s、replay 或物理仿真，只读取产物、核验 hash、审阅当前 runner/replay/analyzer 实现，并独立重算记录中的聚合分数、传感器误差和阻塞 probe 数值。

### 可以支持的结论

这条 seed=0、reference layout、events=true 的 B0 dev attempt 完成了公开观测到确认/报告再到安全退出的真实 600 s 连续 1× 闭环。它达到 gate-contract.v5 的参考收益、静态 recall、时钟和正常 inspect 测量点阈值。**它没有达到安全任务成功，SR=0；动态交付极弱，不能据此宣称持续目标追踪、完整动态场景或完整 G2/G3/G4 通过。**

`run.json` 和 `world.json` 都是 completed，run 记录 sources_unchanged=true。实际 world SHA-256 与 run 绑定一致。run 内 9 份源码全文均与对应 source_hashes 一致，并且与本次审查时的当前源码以及 `qualification/source/reference-v5` 副本逐份一致。analysis 的 contract hash 对应 gate-contract.v5，v5 的记录创建时间早于 run 的 started_at。此前各节的 hash 保持原样，本节记录本次 attempt 的新版本。

| 项目 | 独立核对结果 | 可支持范围 |
| --- | --- | --- |
| 1× 完整时长 | simulated_seconds=600；wall_seconds=600.0064224579837 | 本机、单个 B0 dev episode 完整连续运行。 |
| 时钟 lag | max=0.03386074965044372 s；p95=0.004585541947747629 s，max 小于 .1 s | 达到 v5 本次时钟阈值，不等于其他负载/模型环境的普遍可靠性。 |
| progress | 600 条，t 从 1 到 600，sim_time 与 wall_elapsed 均严格递增；记录行最大 wall_elapsed−t 为 .013614624971523881 s | 是与 world 时间尺度一致的进度记录；本次未把稀疏 progress 的最大值替代逐 tick max lag。 |
| 协调器 5 s 阻塞 | actual_wait=5.000447083031759 s；sim_advance=4.99166666666666 s；差 .008780416365098631 s；0 号机三维位移 95.26729898113287 m | shared_progress 在 worker 每 tick 更新，协调器读取前后均短暂持锁，sleep 时不持锁；可证明该等待期间 world 持续推进。它是 sleep surrogate，不是实际模型/API latency。 |
| 分数 | C=.9280245022970903，S=1，D=.0029411764705882353，Q=Y=.6436552262558929，H=E=1，SR=0，static_recall=1 | 独立从覆盖 evidence 并集、静态权重和全部动态 bins 重算，所有聚合值与 world 一致；v5 的 Y≥.2 与 recall≥.5 参考门槛通过。 |
| 物理安全与资源终态 | 144000 次物理积分对应 144000 次安全审计，无记录的硬违规；最小机间距 49.22811236593713 m；四机剩余航程约 5999.1 m，超过 1000 m 余量 | 支持本次 H/E。3 号机于 360 s 外生退出，其余三机于 470.0083333333333 s 申请退出，四个 exit job 均 succeeded。 |
| 正常 inspect 测量 | n=101，平面向量 RMSE=1.2931653858508028 m，类别正确 100/101=.9900990099009901 | 独立按 sensor_audit 的同刻 truth 重算，达到 v5 的 ≤3 m 与 ≥.9 阈值。“正常”指相机未退化，样本也包含正常干扰物；这里是 inspect 测量统计，不冒充独立确认对象成功率。 |
| 正常 survey 测量 | n=416，RMSE=5.711300619281534 m，类别正确率=.7788461538461539 | 与 analysis 完全一致，是探测后测量的描述统计。 |
| 退化 survey 测量 | n=92，RMSE=17.389005765611383 m，类别正确率=.6521739130434783 | 与 analysis 完全一致，可见退化条件下误差增加；并非跨策略因果效应估计。 |
| 背景假候选 | n=18，独立计数与 analysis 一致 | 满足出现背景假候选的点检查，不估计真实视觉虚警率。 |
| 执行日记重放 | 产物记录 110/110 个调用、response_discrepancies 为空，13 个字段组 exact_matches 全为 true | 本次核验 replay 输入 run/world hash 及比较实现，未重跑物理重放。它重放已执行工具日记，不重跑策略，不增加 root、episode 或策略样本数。 |

重放的实际比较字段组为 **13 个**，不是此前消息中的 14 个。它们是 score、safety、trace、coverage_evidence、required_cells、sensor_audit、delivered_observations、jobs、job_transitions、tool_calls、reports、hidden_entities、public_events。replay 代码逐字段比较完整 JSON 结构，并检查调用响应与总调用数，因此强于仅比较汇总 Y，但仍只验证这一份执行日记。

### G2/G3 的集成证据与缺口

本次 110 个工具调用为 inspect_candidate 39 次、assign_survey 31 次、cancel_job 27 次、submit_or_revise_report 10 次、enter_safe_exit 3 次，**没有 observe_track 或 handoff_observation 调用**。三个动态实体的 180 格中只有 truth-006 的一格非零，其他 179 格为零；动态总权重分母仍为 340，没有删除困难或缺失格。高 Y 主要来自覆盖和静态交付，不能被改述为动态任务可解性或持续跟踪质量已合格。

本次任务在 180 s 记录优先级 v2，在 360 s 记录外生能力退出并最终安全盘旋，这支持相应 G2 集成链。断链 210–240 s 内没有目标 Observation，所以这条 episode **没有**实际目标观测补传的端到端证据。期间有 60 份覆盖 packet，其中 7 份非空，保留从 210 s 起的原始 measured_at 并在 240 s 或之后到达，支持覆盖账的 store-forward 链。目标观测补传与交接仍需引用相应正反 fixtures，不能从没有相关观测的 episode 推断其已实证。

analyzer 的 admitted、reference_and_clock_evidence_pass 和 sensor_point_thresholds_pass 对本次数据为 true，源码中的条件与本次阈值判断一致。它没有实现整个 G2/G3 的一揽子验收。尤其是检测机会/漏检分母、资源竞争或实际主模型等待、跨布局与 OOD 可靠性、同条件跨方法噪声对照，以及任意跨 region 执行和持续目标跟踪都未由这些布尔值解决。normal_sensor_confirmed_class_accuracy_min 当前被 analyzer 用于 inspect 测量类别准确率，报告应继续使用实际测量统计口径，不能扩大成候选级确认成功率。

`replay_inspection_reference.py` 明确拒绝含 nested handoff 的工具日记，因为现有日志尚未记录嵌套调用的顶层身份。这条 B0 记录不含 handoff，所以该限制没有使本次 replay 失效；后续增加交接后需单独修复和资格检查。replay 与 analyzer 都依赖“当前源码仍等于冻结源码”的检查，后续在另一版本修改 observer/跨区执行后应保留本次 snapshot 与产物，不用新源码重写旧资格。

当前 `qualification/RESULTS.md` 与 `qualification/PROGRESS.md` 仍包含早期“参考未完成/修复中”的状态。它们需要由父任务在汇总时更新，或明确标记为历史快照；本审查只追加本文件，没有越权编辑其他报告。`human_review_pending`、主模型与统计适配未完成、G4 未完成的边界保持不变。

### 本次证据与源码绑定

| 文件 | 本次 SHA-256 |
| --- | --- |
| `qualification/reference/realtime.v1/run.json` | `e2549a999a6cb470c453317ba710c92db60ba214a8fc6f746bf182257c73da7d` |
| `qualification/reference/realtime.v1/world.json` | `a4804aeab09bd998d0b5dc1b5be18e4bbfd7c6cdb0eebe9d7a24eabedaa1684d` |
| `qualification/reference/realtime.v1/progress.jsonl` | `4e34b6d0bd9d9ade83e5cb3db89f9def1c86089db5daa73bf339f6bfefcaf9b0` |
| `qualification/reference/realtime.v1.replay.json` | `545beae8fdb60635cdef8a500f2c1af2eac00fa38eabc03a7c77f57acc17e4ab` |
| `qualification/reference/realtime.v1.analysis.json` | `022976fc27b9340124e815254395756f9f3636082baa62a961c400ec37ebbe01` |
| `planning/gate-contract.v5.json` | `2eecb39d2300fd2f6f3627ba67861e56d923bbdbb0937098a44f905f733d54e3` |
| `envs/inspection/mission.py` | `d2939ab883c83e26a3bf7aae7548947c1e314b17247113d173a744ab10bf8e2e` |
| `eval/run_inspection_reference.py` | `1d95f195f556c79ed1aca79e8fb3411454a529618324adc314dfbeb1e9e6b4a7` |
| `eval/replay_inspection_reference.py` | `afb3060f62ad6ac4b58bb2533850f97eef4c1505ecfb78b3a2a773c5ca498718` |
| `eval/analyze_inspection_qualification.py` | `7db62ac4ec297b770d42e42bbd78a1b66d2a26f00179862e2ce7629f4daeac83` |

本节中 qualification/planning 路径相对研究目录，其余源码路径相对仓库。run 的其余冻结源文件与前文一致：protocol=`c32bdabb…`，scenario=`a402003a…`，reference=`c1535f4d…`，scoring=`bc8a2194…`，flight=`272efaf8…`。另外 control 为 `611c0e8cf51bef2623c45d03047725e67f4a623b600e7e34b3dd18b04b7f721c`，inspection/__init__ 为 `cb0d38bbca6b531b261ca5a5f2f79664e352c9ddb6c19acc1777784c3115107a`。协议 tests 的 hash 仍为 `f986274171e43257b280cf2de875ba07634a5869e3c38c8fe23fe5e34f3ef832`，本次未再运行测试。

## 追加审查：目标补传物理 fixture 与归档重放 v2

本次只读核对了 `eval/probe_inspection_store_forward.py`、`qualification/reference/store-forward.v1.json`、归档绑定 helper、replay/analysis v2 及当前 B0 的延迟确认回归。没有重跑 600 s、grid、replay 或 33 s 物理 probe。只运行了源码 bundle 的只读校验、记录的数学核对，以及新增纯 ReferencePolicy 回归函数。

### 目标补传的具体证据

**目标观测 store-forward 的物理实证缺口已由这份独立受控 fixture 补充。** 这不改变 realtime.v1 本身“断链时没有目标观测”的事实，也不增加完整 episode 或策略样本。

probe 使用四机真实 PyBullet 物理，只保留一个受控动态代理。它从空中初始状态把绝对 simulation epoch 设为 208 s，随后实际积分到 241 s，共 33 s。前 208 s 没有积分，也没有在此构造出 208 s 的历史安全、里程或任务记录。`safety.physics_steps=57840` 是带 epoch 偏移的绝对计数，扣除 `208×240=49920` 后实际执行 7920 步，恰与 `physics_audits=7920` 一致。记录中没有硬安全违规，trace 共有 330 行，从 208.1 到 241.0 s。

7 条目标观测在 210.5、211、211.5、212、212.5、213.5、214 s 采集，均于链路恢复的 240 s 到达，年龄为 26–29.5 s。独立逐条核对 late_target_observations 与 report_versions，其 evidence_id、measured_at、arrived_at 全部一致。观测由实际 camera origin、视锥、ray 和物理状态产生，observe_track 通过实际工具受理后自动保存和上传，不把恢复时刻冒充采集时刻。

单目标诊断的 `D_delivered=0`、`D_observed=2/60` 与数据一致。按 208 s 开始的固定 3 s 格独立选最新样本，前两格的 observed 候选为 `obs-1-422-0(t=211)` 和 `obs-1-428-0(t=214)`，与源代码中明确的同刻轨迹相比，位置误差分别为 .3714954176120108 m、1.6162186034560693 m，均得质量 1。它们在 240 s 到达，已经晚于对应 217/220 s 的 delivered 截止。最初的及时报告类别错误，不能补得 delivered 分数。

该 D 值使用完整 60 格诊断分母，而实际物理只执行 33 s；它不表示完整 180 s 要求窗口已经飞完，也不表明整个 30 s 断链或整个 33 s interval 都持续获得目标测量。脚本直接发出受控报告和 observe 工具调用，没有运行 ReferencePolicy，因此不构成修复后 B0 的策略质量样本。probe 的 source_snapshots、source_hashes 与本次当前源逐份一致，sources_unchanged=true。

### 归档绑定与当前策略版本

从 `/private/tmp` 只读调用 `verify_source_bundle`，使用绝对归档 source_root，9 份旧 run 源码的 embedded hash 与归档文件全部通过，实际解析路径均在归档根内。helper 额外检查归档的 envs/eval package scaffolding 为空，并拒绝相对路径或 symlink 逃逸。相反，不指定归档、尝试验证当前工作树时，明确拒绝 `envs/inspection/reference.py` 与旧 run 不一致，避免把修复后策略当作旧成绩的源码。

replay v2 的 `input_run_sha256` 和 `input_world_sha256` 仍对应原 realtime.v1，replayer_sha256 与实际 replayer 文件一致；其 source_root 指向 reference-v5。产物记录 110/110 调用、13 个完整字段组精确相等、无 response 差异。当前 replayer 在导入 Mission 前选择归档路径，运行时再核对 Mission 实际来源路径，防止已经导入的当前模块混入。以上是本次对既有产物和实现的核对，没有第二次独立物理重放。

analysis v2 绑定同一归档根，其 analysis_sha256 与实际 analyzer 一致，contract_sha256 仍为 v5。attempts、sensor_groups、背景候选数量和两个通过布尔值与原 analysis 完全相同。归档根解析方式补齐了此前“旧实现修改后仍可核验旧产物”的工程边界，未改变原分数或增加样本。

当前 reference.py 已另有 `confirmed` 状态，完成遥测先于最终确认包时等待证据，不重复发 inspect。本次直接调用新增纯回归 `test_reference_waits_for_delayed_confirmation_proof_without_reinspection`，结果通过；没有重跑父任务报告的 31 protocol/20 scoring 全套测试。当前策略 hash 与旧归档不同，其后续成绩应另开版本；不能把新策略推断为已具有旧 run 未展示的持续动态追踪能力。

本次没有发现使上述受控补传证据或旧 run 的归档绑定失效的问题。固定 stripe、持续目标追踪、跨 region 执行、完整 G2/G3/G4、主模型/统计适配和 `human_review_pending` 的边界保持不变。两个架构候选的比较尚不是已实现、已资格通过的运行能力。

### 本次追加版本绑定

| 文件 | SHA-256 |
| --- | --- |
| `qualification/reference/store-forward.v1.json` | `6a06547dc6cde2af89a875b647f53f7ce4282628b955c573293dcb402a279630` |
| `eval/probe_inspection_store_forward.py` | `5326c44aa1cb2a5a32861666422a5db7b8b539f2581a4b382799e73c124c405a` |
| `qualification/reference/realtime.v1.replay.v2.json` | `046cb703021bdbb549b8c0710d3a1e933d36ba5e1fb80c60f50aa3f1da4ffebc` |
| `eval/replay_inspection_reference.py` | `699c2dc4a6f9bac7b06dadeba5064c6a49a310059686ab05f24c55875b5ca59b` |
| `qualification/reference/realtime.v1.analysis.v2.json` | `7edea6329f39a4c069c926a1d76c4a48261ef0c2233c49112c3fb6b152601606` |
| `eval/analyze_inspection_qualification.py` | `04b1a63d3fb7f4f90451e198be146860b5f67f23e2a1d69afb0e5602062e2344` |
| `eval/inspection_artifacts.py` | `1fd883eea301136fa807bba614e35376c169202dcced7f4e5ec6003ad6df3e4f` |
| 当前 `envs/inspection/reference.py` | `d136d825b6bc43347656aa0178584cadfab01f7c4bf85472f6d1c8f5b3b6c94b` |
| 当前 `tests/test_inspection_protocol.py` | `1b15268bcd4318ae7725ee7481e052e6e8bf8dc2051fc7ffe7790ad04829ffc6` |

原 realtime.v1 的 run/world hash、归档 reference.py 的 `c1535f4d…` 以及上一节的实际结果均保留不变。本节 qualification 路径相对研究目录，源码路径相对仓库。
