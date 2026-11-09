# G2 v16 独立工程复核 v1

审查日期为 2026-11-05。审查者为 `/root/mission_review`，属于同模型家族的独立上下文，不是生产实现者，也不构成人类领域或跨模型审查。工作树为 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`，HEAD=`270c6f0946b3be59527b2b77982f75a70da7ef77`；当前新增文件未由该 HEAD 跟踪，结论绑定下列实际 SHA-256。

本次按 good-benchmarks 的纯审计方式核对给定契约、评分调用链和既有证据，不重设任务/门槛、不另建评测。只新增本报告；临时测试日志在 `/tmp/g2-v16-review.PDRdIR/`。没有修改生产源码或旧结果，没有运行 600 s、新长物理、API 或完整 grid，没有读取进行中 G1 的 partial 结果，也没有重审旧 140-slot 或论文。

## 结论与确定阻断

`qualification/protocol/store-forward.v2.json` 的新补传证据有效。它实际从 0 飞至 241 s，57840 次物理积分与审计一致；不是旧 epoch208 跳钟、只积分 33 s 的 v1。34 条目标包在断链期间采集并全部于 240 s 到达，239 s 时的 32 条待传包保留冻结的测量值。单目标完整 60 格诊断独立复算为 delivered=4/60、observed=8/60。没有发现使这份 v2 产物失效的源码或计数不一致。

**完整 G2 仍不能标通过。确定缺口是当前 v16 的外生退休实际场景尚未被本次证据覆盖。** 新产物止于 241 s，只有 180 s 优先级事件；没有到达 360 s 退休，没有记录退休后释放任务并实际进入安全退出盘旋。新增 mock 语义反例通过，不能代替这个真实执行链。新物理产物的四机 exit_dwell 也均为 0，不能凭无硬违规推得 E 或退出验收通过。

本次 51 项协议/评分测试及 2 项单步物理审计正反例通过，支持下表限定的实现与协议结论。新完整 600 s 真实 1× 未运行属于 G3 待证，进行中的新 G1 仍由其自身完整产物判定；它们没有被本报告提前放行。17 m/s、pitch_trim=-.07、70/80 m 观察候选和 6 m/2 m 条件仍是 v16 开发资格范围，原 v11 完整失败保留不改，human_review_pending。

## G2 十项覆盖表

“已覆盖”只表示表内具体工程证据通过，不表示完整 benchmark、科学可靠性或所有实际事件组合通过。要求来自 `planning/gate-contract.v16.json` 的 G2 十项及 positive_negative_fixtures_required。

| G2 项 | 正例及实际证据 | 反例或边界检查 | 当前判定与限制 |
| --- | --- | --- | --- |
| 1. 坐标与平地 AGL | 任务卡公开 terrain=0；`flight.py:110–145` 用物理位置 Z−0 审计。v2 实际高度 94.82024745957219..97.09721481199054 m，无硬违规。 | `test_safety_hook_is_after_both_physics_steps` 与 `test_contract_safety_limit_is_actually_enforced` 本次单独通过；后者把有效初态的最低允许高度改成 81 m 后，实际两步均报高度违规。 | 平地坐标/审计实现范围已覆盖。两个单步检查不是新控制族资格或真实地形验证。 |
| 2. 多机与移动代理同步 | v2 四机从 0 连续实际飞行，代理于 206 s 加入；`mission.py:88–94` 在每次物理积分前按当刻轨迹移动。原始 240 Hz 行覆盖 1/240..241，计数与物理一致，trace 2410 行。 | 两步 callback 检查确认审计在两次物理积分后发生且位置改变；本次独立确认所有目标 sensor_audit truth 与声明的同刻轨迹一致。 | 有当前受控物理正例。没有另做故意让移动代理与飞机错拍的完整故障注入，不把计数相等称为全物理状态重放。 |
| 3. Actor RPC 排除 truth/seed/future/grader | public snapshot 只含公开任务、健康、已到达观测/覆盖；任务卡/观测 deep copy，实体 key 和 sensor_audit 留在裁判侧。 | `test_public_snapshot_excludes_truth_and_is_a_copy` 递归检查字段；get_truth、__dict__、set_seed、set_retired、grade 等请求被拒；未知候选与旧任务卡被拒。 | API 信息边界已覆盖。没有将同用户任意 Python 代码的 OS 文件权限当作已隔离沙箱。 |
| 4. 不可变证据与因果到达 | Evidence/Observation frozen，ledger 拒绝改写 minted ID；v2 同 ID 的测量值、时刻和实际 arrival 在待传→已达→报告链一致。 | frozen mutation/re-register 反例；未来 measurement/arrival、错误 candidate/entity、低质量、无支持证据和 NaN；动态 ID 重排不能改写 seal。 | 纯协议及本次真实补传范围已覆盖。pending 包内名为 arrived_at 的值是待传时原始运输计划值，实际到达由 `_deliver` 的 replace 后记录，不能混读。 |
| 5. queued/running/terminal、旧版本与取消 | v2 inspect 在 207 queued、207.5 running、209 succeeded；observe 在 209.5 queued、210 running，至 241 仍 running。真实短 handoff fixture 本次随 protocol suite 通过，source 在 receiver 新合格观测前保持 running，之后 cancelled。 | JobBook ACK 不等于完成、active job 拒绝、stale cancel 拒绝；断链取消拒绝是 mock 时钟 fixture。 | 生命周期核心及短真实获取/交接覆盖。observe 60 s 服务并未完成，不能称成功观察；短 handoff 不等于 180 s 联合场景。 |
| 6. 不退名额与首次固定关联 | ReportLedger 的第一新 ID 占名额，固定按声明观测时刻与估计匹配；v2 一份 ID 的 37 个版本只占 1 格。 | withdrawal/duplicate 不退款；新第三 ID 超容量拒绝；首次未匹配以后不能救回、不能换实体；20 m 包含端点、等距稳定、类别不参与匹配。 | 已覆盖纯 ledger 正反例及本次单实体报告链。没有外推到多目标交叉的全部关联可辨识性。 |
| 7. 静态最后合法版本与全隐藏分母 | 实际 Mission 聚合初始无观测 fixture 仍有 6 静态实体、180 动态格，S/D/Y=0；静态最后合法接收版本计分。 | 5 m 合法替代估计、最后错误修订降低成绩、同 tick 接收顺序决胜、无证据/错误身份不得分。 | 当前测试与不变 core 范围已覆盖；v2 是单动态代理 fixture，没有在它身上伪造静态任务总成绩。 |
| 8. 动态同测量时刻 truth、先筛后选再判正确 | v2 独立从原报告/观测按 60 格选择并重算，delivered/observed 及全部 delivered evidence IDs 与产物一致。 | 定位边界、内部分格端点只计一次、迟到/非法先排除、最新错误不退回旧正确、同刻 ID 决胜、observed−delivered 可为负、截断协议直接拒绝。 | 评分语义及该受控 trace 诊断已覆盖。保留 60 分母不等于已经飞完整 180 s 需求窗口。 |
| 9. 迟报/补传与格起点优先级 | v2 实际 210–240 s 断链、34 条目标数据在 240 到达；180 s 公开优先级事件确实出现，后续工具使用卡 v2。 | mock `_deliver` 证明 job 结束后旧包仍按采集时路由上传；覆盖包到达后才入账；priority crossing 的格起点权重为纯 scorer 反例，180 s 发布边界为 mock integration。 | 补传的当前真实正例通过；优先级排序/权重语义通过。此目标 206 s 才出现，没有实际 straddle 180 s 的目标窗口，不能由本 run 称全事件效果已验收。 |
| 10. 安全退出/外生退休及自致失败不豁免 | 当前源码要求四机全部实际 exit dwell 与余量；协议测试证明 exit ACK 不足、资源不足则 E=0。 | 本次额外 mock 360 s 事件取消旧 3 号任务、创建 exit、拒绝退休机新工作且保留物理 program；synthetic dwell 反例确认退休/非退休机缺退出均 E=0，自致硬违规 H/Y=0。 | **实际场景待证，阻断完整 G2 通过。** mock 执行了 0 物理步；v2 没有退休事件或真实退出 dwell。 |

## 新补传产物的逐层核验

### 1. 源码与执行范围

脚本 `eval/probe_inspection_store_forward.py:22–40` 在正常 Mission 构造后先把地面代理移出场景，飞机和时钟没有重置；真实飞到 206 s，再在 1 号机当时地面投影处加入一个受控动态代理。隐藏轨迹 origin 为 `(246.69200641888253,-180.1690207345306)`，appearance=206，speed=1.5。目标布置是 fixture 操纵，不是策略发现能力或科学场景样本。

脚本使用 `ControlledPlannerExecutor`，其 `eval/qualify_inspection_executor.py:45–71` 在物理步之间计算纯 plan、下一 control tick 交付，且不模拟 planner 墙钟等待。这对协议/物理 fixture 合法，但不证明独立 planner 真实时延、资源争用或 1×。

产物 sources_unchanged=true。16 份 source_snapshots 的 SHA-256 全部匹配 source_hashes，且逐字节等于本次当前文件。实际 `mission.py:33–34` 使用四机 96 m 初态及 inspection_gains；`control.py:103–109` 为 pitch_trim=-.07，planning AIRSPEED=17，任务卡公开 70/80 m 有限观察和 dev-v6 语义，与 v16 候选一致。产物没有直接嵌入 gate-contract hash，本报告独立绑定 v16 并核对配置；不能声称所有一般环境/资产已因此完成 G4 冻结。

### 2. 实际步数与安全审计

initial_sim_epoch=0，final_sim_epoch=241，physically_integrated_seconds=241。physics_steps_executed=safety.physics_steps=safety.physics_audits=57840=241×240。不是带 208 s 偏移的计数。

本次用 stdlib 解码 execution_tracking_trace，校验 raw SHA、字节数 `57840×17×8`，逐行验证 `t=i/240`，独立重计最大 XY 误差 4.090274066984552 m，水平/高度区间违规均 0，与摘要一致。记录不是完整原始仿真状态 replay；这份受控单例也不能覆盖正在运行的 G1 matrix。

四机最小间距为 98.0854965852335 m，无记录的硬安全违规；全部 exit_dwell=0。没有将 H 的安全事实扩展成 E 或终态成功。

### 3. 239 s 待传、240 s 恢复及曝光路由

inspect 的 ACK 在 207 s，实际两视图完成在 209 s；209.5 s 提交测于 209 的 report，并批准 observe 60 s、route=`fixture-report`。后续没有新的 Actor 工具改派或改 report ID。

239 s checkpoint 留下 32 条 route 已绑定 fixture-report 的 pending 目标包，measured_at 从 210.5 到 238.5。34 条最终 late 记录增加的恰为 239、239.5 两条，全部实际 arrived_at=240，等待 0.5..29.5 s。32 条交集的 evidence_id、原始测量时间、估计和类别逐条保持不变。

注意 pending 对象的 arrived_at 字段仍可能是最初 `measured_at+.5`，例如 `(210.5,211.0)`；它尚未上传，不是到达证明。`mission.py:265–276` 等恢复时才把真实 arrived_at 替换为当前 t，并注册到 ledger。实际 delivered observation 与 report audit 均为 240，评分没有用待传字段冒充早到。

`mission.py:250–259` 在曝光时把 job 已批准的 route 存入 pending，并在四机该帧均曝光后才处理 local postcondition；上传时只使用包中 route。checkpoint 的过滤也显式要求原 route。独立对全部 37 个 report audit 核对，其 ID、measured_at、estimate、class、candidate、实际接收 t 与 delivered observation 完全一致，sensor provenance 均对应 fixture-moving，37 个 evidence ID 各不重复。新 ID 名额只占 1 格。

本 run 没有把不同 job/report 的旧包在 cancel/reassign 后补传的完整物理情形；相关晚包不依赖当前 active job 的行为由现有 mock fixture 覆盖，不冒充这一实际场景已经执行。

### 4. 不调用 scorer 的独立诊断重算

按原始 entity 的轨迹函数，在每个被选测量的 measured_at 计算 truth；所有 sensor_audit 中目标 truth 也与同刻轨迹一致。独立从 report audit 构建每格候选，先检查时间/到达，再按 `(-measured_at,evidence_id)` 选择，最后检查类别、质量和 3/5/10 m 定位档位，没有调用生产 scorer 重算。

| 轨道 | 非零格及对应 evidence | 结果 |
| --- | --- | --- |
| delivered | 格 1=`obs-1-418-0`，10=`obs-1-472-0`，11=`obs-1-478-0`，12=`obs-1-481-0` | 4 个质量 1 格，56 个无合法及时样本；4/60=.06666666666666667。全部 60 个 selected ID 与产物一致。 |
| observed | 格 1/2/3/7/8/10/11/12，对应 tick 418/424/428/454/460/472/478/481 | 8 个质量 1 格，51 个无样本格，另 1 格最新类别错误为零；8/60=.13333333333333333，没有退回旧正确样本。 |

第一格覆盖 [206,209]，其余格为左开右闭；完整要求窗口结束于 386 s，最后上传宽限至 392 s。当前仅飞至 241 s，observe 服务也仍 running。60 格全部保留，只是一次有限 trace 的单目标诊断，不能称 180 s 成功或完整 episode 主分数。6 s 为交付宽限，不是本地缺测容忍。

## 已执行检查与证据类型

1. `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_protocol.py tests/test_inspection_scoring.py -q -p no:cacheprovider --basetemp /tmp/g2-v16-review.PDRdIR/pytest`，退出 0，**51 passed in 9.23s**。其中包含短真实 handoff、短 worker 失败注入，也包含明确 mock 时钟的补传/覆盖/事件边界；没有把所有测试都称作完整物理事件。
2. 仅选择 `test_safety_hook_is_after_both_physics_steps` 和 `test_contract_safety_limit_is_actually_enforced`，退出 0，**2 passed in 1.30s**；没有运行该文件的 120 s heading fixture 或 grid。
3. 额外 retirement/grade 内存语义检查，**0 物理步**。360 s mock 事件将旧 queued job cancelled、新 exit queued，退休机新 survey 被拒，program hash 不变；退休机缺 dwell、非退休机缺 dwell 均 E=0，synthetic 自致硬违规 H/Y=0。它是反例覆盖，不是 360 s 真实退休运行。
4. 产物/源码 hash、34 个补传 observation、32 个 pending、37 个报告版本、60 格选择、57840 行 tracking 的独立只读重计通过。

临时证据为 `/tmp/g2-v16-review.PDRdIR/tests.log`、`flight-tests.log`、`retirement-semantic.json`。主测试日志 SHA-256=`e3d2dc98944027971af0793d8efe90b86a837c684f05574513aa7982655e38b0`。

## 复用范围与状态文件

`envs/inspection/protocol.py` 与 `scoring.py` 均逐字节等于 `qualification/source/executor-v11/envs/inspection/` 对应文件。可复用此前对账本、动态选择/静态最终版本等**不变纯 core** 的审查；本次也重新运行当前协议/评分测试。不能把这份字节等价升级成当前 sampler、曝光路由、planner、控制、终态实际场景或新 G1/G3 已通过，它们的调用/世界状态已经变化。

本次读取了 PROGRESS.md 末三节。其末节仍写“新补传尚未实际运行”，与随后已经完成的 store-forward.v2 不同；父任务应在阶段汇总时更新状态或标明历史时间。本报告没有改 PROGRESS，也没有以其中的阶段计划代替运行证据。原 v11 的失败、其他开发失败和当前完整 G1 的进行中状态均保留，本审查没有读取或重标它们。

## 当前版本绑定

下列生产/评测文件均与本次补传产物的 source snapshots 一致。source 当前值在有界检查后再次核对，没有改变。

| 文件 | SHA-256 |
| --- | --- |
| planning/gate-contract.v16.json | `42ec6e4886463011ab3c00915aa7d91656e2ea043fb7989a6ff6d53f85d587af` |
| qualification/protocol/store-forward.v2.json | `e9b7a595442eab871a2e0dfbd657352bbbe7f546ecd42618563c6bc9440e95bc` |
| envs/inspection/execution.py | `df11567e1af3699f3bd9a99d3580ebba8062f806bbad8bd7e46c1b86f25bfdef` |
| envs/inspection/paths.py | `2f9dd9917b2e92555ccae190e3e4ece553404f11b51d48562cc9aab6ebad8989` |
| envs/inspection/flight.py | `ef9be915c92c207a3e8f43b9ddc1907010beea59be6d96136330e82a5d67cbdb` |
| envs/inspection/protocol.py | `c32bdabb67bf8493ff5125d0af4fc289d1d0210db762016a9c56a96dd2f174a4` |
| envs/inspection/scoring.py | `bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a` |
| envs/inspection/control.py | `c676d237efe1b6227d238b4e5948c54a5725626e9d3f69e74ed9c1629a235270` |
| envs/inspection/__init__.py | `cb0d38bbca6b531b261ca5a5f2f79664e352c9ddb6c19acc1777784c3115107a` |
| envs/inspection/scenario.py | `c9c1e8fc09f244c32307d29cfb11897a7bb348def3745cd5e2b564f3a1397a49` |
| envs/inspection/planning.py | `69f2553852471bf0ec5c365190f2123633d6d0b42d8e9379fe15febd62af3a5d` |
| envs/inspection/mission.py | `491fe4c8123241e61a3d6392bfeea618604f9bb2717df035d095e0506ca550da` |
| envs/inspection/reference.py | `85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3` |
| envs/inspection/guidance.py | `9ab9c5290cbce60ad7f9fead635318f41561dedd1cfc2f0984b1cd45385bc0b9` |
| eval/probe_inspection_store_forward.py | `31c8d92fe82673839f6831fc78d2bc1f5ecfd4cfa1028b211885c2780509eb1e` |
| eval/qualify_inspection_executor.py | `f5c58c340018933bfdd81a69d2a581846e9ba3046da9b567d79a34c0deaf7586` |
| eval/qualify_inspection_g1.py | `852ea8c02cade269b48ea1bf99158913ddaef2038f6a368bd0987c31cc794522` |
| eval/inspection_tracking.py | `0720d79beee9fdbaf873cafa477cfa04d5ecf76335dd01f30f9a86d0bcdb1e5c` |
| tests/test_inspection_protocol.py | `9ef62c113a607120128359eb6aeb2c21f15067c012945ddf351ce0bc89e9cb7d` |
| tests/test_inspection_scoring.py | `0f9c4f34c5c0f6a92dcaca6fe51b1c6831b826e221eae7ca710499bc41049403` |
| 读取时 qualification/PROGRESS.md | `4196ff09bb913cc55693a5a460e59b2a798a636339a27c84004d9b7fd52d1162` |

表中 planning/qualification 路径相对 `paper/research/20261104-inspection/`，其余相对仓库；PROGRESS 的 hash 只是读取时状态，不是禁止父任务正常更新进展。完整 G2 的下一份验收证据应补上 v16 实际外生退休与退出链，并把其他 mock/短 fixture 的结论保持在相应范围，不能凭代码存在或测试数量盖章全部通过。
