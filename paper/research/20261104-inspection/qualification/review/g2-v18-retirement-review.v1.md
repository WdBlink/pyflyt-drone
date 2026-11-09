# G2 v18 实际退休链独立复核 v1

审查日期为 2026-11-05。审查者为 `/root/mission_review`，属于同模型家族的独立上下文，不是人工领域或跨模型审查。工作树为 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。只新增本报告，必要小结果存于 `/tmp/g2-v18-review.qtPQtZ/`，没有改源码、旧结果或他人改动，没有重跑长 physics、600 s、API 或 grid，也没有使用其他组的进行中计数宣布资格通过。

## 结论

`qualification/protocol/retirement.v2.json` 是有效的当前 v18 受控物理退出证据。17 份源码及 61 项指定依赖的记录 hash 与当前文件一致。独立解码重计确认 0→500 s、120000 个实际物理步的完整 tracking 行；360 s 生产事件取消旧作业并升版本，原物理 program 当刻保留，360.5 s 才提交新退出程序，441.8833333333333 s 达到实际退出终态。四机最终均有足够 dwell 和资源余量。

**完整 G2 仍未通过。发现一个实际未闭合缺陷：退役机在 360 s 之后仍采集、公开新的 mission 数据。** 当前 16 项 checks 全为 true，只覆盖命令/作业/返航链，没有检查任务容量是否在事件后退出。主任务已明确采用保守语义，从 360 s 起停止该机的新目标、覆盖和 background mission 采样，同时保留 360 s 前已采集数据的合法迟到交付。当前实现不满足这个边界。

因此可以关闭前一份 v16 报告中“没有实际退休→释放作业→物理安全退出链”的证据缺口，不能把它升级成 G2 全部通过。v18 全 40 项新 G1 及新的 600 s 真实 1× G3 均由各自完整执行与审查决定，本报告没有提前放行。该 500 s probe 使用 controlled next-tick planner，既不是 1×，也不是新策略/科学样本。human_review_pending。

## F1 [P1，未关闭] 任务容量退出后仍产生新的任务观测与覆盖

相关位置为 `envs/inspection/mission.py:113–118,163–187,247–259,265–285`。退休事件将 3 号机加入 retired，preempt 旧任务并创建 exit。工具入口会拒绝退休机新作业，但 `_sample` 只跳过 `flight.exit_active[drone_id]`，没有跳过 retired。exit_active 到实际 home tail 才生效，所以整个安全返航期间相机仍向 mission 提供新数据。

本次直接从已完成的 retirement.v2 原始业务审计中独立计数：

- 3 号机在 measured_at≥360 s 仍生成 **124 份非空 coverage packet**，采集范围为 360.0..421.5 s。
- 同期产生并公开交付 **101 条新的 survey Observation**，原始对象类别为 normal 38、road_obstruction 25、bank_anomaly 24、infrastructure_damage 14；不是旧包改了时间戳。
- 首条为 `obs-3-720-0`，measured_at=360.0，arrived_at=360.5。最后一条 ID 为 `obs-3-843-0`，measured_at=421.5。
- 本 fixture 的先前覆盖已经包含这些 cell，按到达顺序计增量，新增加的 coverage cell 为 0。这个事实不能消除新的公开观测，也不能证明其他场景的 C 不受影响；本报告没有声称本次 C 已被实际抬高。

这个缺陷削弱了“容量损失”事件，不能只凭拒绝新命令称任务执行已完全退出。不要把当前 v2 的 passed=true 改成全 G2 成功，也不要修改正在执行的冻结 v18 或回填旧 manifest。

新版本修复与验收需要同时保留两条边界。退役时刻及之后不得新增该机 mission target/background/coverage 样本；退役前已采集的包仍按原 measured_at、原估计、原曝光路由和实际 arrival 上传。当前记录中恰有 `obs-3-719-0` 在 359.5 s 采集、360.0 s 到达，是应保留的合法迟到包。只禁止上传全部退休机数据会错误地删掉它。后续新 actual retirement probe 应增加这两类正反断言，并保留物理返航/原占用/终态检查。

## 对实际退休链的独立核验

### 源码、依赖与 profile

记录中的 source_snapshots/source_hashes 共 17 份，逐份验证 embedded hash、当前文件内容均相同；sources_unchanged=true。dependency_hashes 共 61 项，当前逐项 hash 全部一致，dependencies_unchanged=true。依赖包括 PyFlyt 的 41 个 .py、12 个 .urdf、6 个 .yaml、1 个 .dae 及 PyBullet .so；该闭包不等于整个虚拟环境或正式 G4 lock。

profile 对应 v18 的 nominal17 m/s、共享 pitch_trim=-.07、10 m 同时刻水平误差、稳态高度±2 m。所有模式的名义连续水平间距收紧为 35 m，围栏为 540 m。两次实际计划的 geometry 审计均给出 37.30381781286796 m 连续名义下界，低于等于 421.3809235431322 m 的名义绝对 XY 上界，满足候选准入门槛。这些名义值没有被冒称为整体新 G1 物理资格。

### 完整积分与 tracking 重计

本次用 stdlib base64/zlib/struct 解码原始 trace，独立验证 raw SHA-256、`120000×17×8` 字节、所有有限数值及每行 `t=i/240`，没有调用生产 recount 函数重算。

| 指标 | 独立结果 |
| --- | --- |
| 实际时间与步数 | initial=0，final=500；120000=500×240=physics_steps=physics_audits=tracking rows。 |
| 原始 tracking hash | `f91bf44aa607cdecb7598cc60e29341dabd3cbb52ae215db5978c669f0e8b6a3`。 |
| 明确门槛 | trace.xy_tube_m=10，按 10 重计，没有用新默认解释旧 6 m 结果。 |
| 最大 XY 误差 | 4.090187918431436 m，与产物摘要完全一致。 |
| 越界计数 | 水平10 m和记录的高度区间越界均 0。 |
| 原始实际 Z 范围 | 95.0640173743258..97.09720555874213 m。 |
| 硬安全摘要 | violations 为空；记录的最小机间距 38.81609667589731 m，空速 15.953022113586357..20 m/s。 |

120000 行原始量是逐步误差、实际 Z 和高度区间，不是逐步完整物理状态。完整 OwnState/PI 保存于请求、提交和 seam 等离散事件，本次核对了 18 份；不能称 120000 份完整 OwnState 或物理状态 replay。硬安全全步摘要由实际 FlightWorld hook 产生，本次没有从仅含误差/Z的行伪造重算全部速度、bank 或碰撞。

### 事件、作业与占用链

280 s 的 survey3 请求使用 version0、task_card_version2、duration160，受理为 job-0001/version1。280.5 s 实际程序提交后 running，实际服务起点为 336.7916666666667 s。359.9916666666667 s 的快照仍为 running，证明事件确实打断一个未完成的合法任务。

360 s 仅出现一次 production exogenous_capacity_exit。旧任务 cancelled/finished_at=360，创建 job-0002/exit/version2/origin=exogenous_capacity_exit。之后对该机新 survey 的请求被拒为 aircraft externally retired，旧 version1 的 cancel 被拒为 stale version。

退休前后 program hash 均为 `ff51a32d…`，且等于先前 survey 的 program_committed hash。四机这个 control tick 的实际位移分别为 .14239962329959036、.14244789536594593、.14240151030096307、.14167792848743113 m，均大于 .01 且小于 27/120=.225 m，记录值与独立距离计算一致。这支持没有在取消时释放或瞬移原物理占用。

exit 的新 request 为 now360/commit_at360.5，token 绑定 job-0002/version2；360.0083333333333 s 收到 proposal，360.5 s 才提交并增 reservation revision 至2，新 program hash 为 `accf45d5…`。该 request 的四机 OwnState 均 at360，位置逐份等于 360 s 的实际 trace。两次请求的 8 份 OwnState、两次 commit 的 2 份及 8 个 seam 的 8 份均有相应时刻及完整 OwnState/PI 字段。

3 号机的 recovery_start 在 362.5 s，tail_start 在 421.8833333333333 s；exit job 在 441.8833333333333 s succeeded，而非 ACK 时或抵达 tail 当刻。所有 seam_capture_failures=0，没有基础设施错误。其他三机在 458.6666666666667 s 进入各自 home tail。

### 独立核对实际 home 轨迹与终态

trace 恰有 5000 条 10Hz 实际位置记录，逐条 `t=(i+1)/10`。独立检查最后 20 s，每机 201 个位置点对其本机 home center/radius80 均满足 radius±10；位置差分估计的切线 course 偏差均小于 .612°。这是 10Hz 位置差分核对，不冒称精确重算每个240Hz的course。

| 机号 | 最后20 s实际半径范围(m) | 差分course最大误差(°) | 最终全步exit dwell(s) |
| --- | --- | --- | --- |
| 0 | 79.84222..79.92438 | .60617 | 41.33333333333926 |
| 1 | 79.84614..79.91583 | .61103 | 41.33333333333926 |
| 2 | 79.84770..79.91883 | .61101 | 41.33333333333926 |
| 3 | 79.89656..79.90077 | .61061 | 78.11666666667779 |

四机剩余航程分别为 9494.489851449427、9494.09069810003、9493.93996001635、9498.04499306157 m，均超过 1000 m。支持这一受控 fixture 的实际安全退出及资源终态条件，不表示完成600 s任务、主 Y 或真实电池验收。

## v1 失败的保留与 v2 的边界

retirement.v1.failure.json 记录首次输出时 numpy.bool_ 不能被 JSON 序列化，导致没有最终 physical trace。failure receipt 中的 log hash 与实际 `runtime/retirement.v1.log` 一致，script_snapshot 与其 script_sha256 一致，未发现 retirement.v1.json 最终 trace 文件。它是操作失败，不能计为一次资格通过，也不能从失败 receipt 推断未持久化轨迹安全。

v2 在 checks 输出处转为 native bool，再从 0 重跑，形成当前独立的 120000 行产物。本次核对 16 个 JSON checks 均为 native bool 且 true；这不是修复后回填 v1 的结果。16 项 checks 本身缺少 F1 的采样终止谓词，不能因布尔全真忽略已记录的新观测。

## G2 十项映射与当前状态

v18 与 v16 的 G2 十项契约字节结构相同。protocol.py、scoring.py 仍与 executor-v11 及 v16 审查版本逐字节相同；FlightWorld 与 v16 相同，Mission 与 v16 的差异仅为导出 trace.xy_tube_m 元数据，相关采样、传输、任务 API 与评分聚合没有改写。本次重新运行当前协议/评分及两项单步检查，共 **53 passed in 10.14s**，日志位于 `/tmp/g2-v18-review.qtPQtZ/tests.log`。

| G2 项 | 当前工程结论 | 证据范围与仍待项 |
| --- | --- | --- |
| 1 坐标/平地 AGL | pass，限定平地 | 53项中的两步实际审计/越界反例；当前完整Z行与安全hook。不外推真实地形。 |
| 2 同步机群与对象 | pass，当前受控场景/接口范围 | 0→500全步时间、实际多机trace、正常production对象callback，结合v16同步核对。不称完整物理state replay。 |
| 3 Actor truth/seed/future/grader隔离 | pass，API范围 | 当前recursive snapshot/allowlist/未知candidate与过期卡反例；未声称OS沙箱。 |
| 4 不可变证据与因果到达 | pass，协议与已审补传范围 | 不变core正反例及v16真实补传、当前回归；退休后采样属于F1的授权/容量边界，不能用因果合法掩盖。 |
| 5 生命周期/旧版本/取消 | pass，当前实际退休子链 | 本次running→cancel→exit queued/running/succeeded、stale cancel实际拒绝；短真实handoff随当前protocol回归通过。 |
| 6 不退名额/首次关联 | pass，纯core范围 | 不变ledger、同ID/新ID/withdraw/20m匹配及重排反例；没有新科学方法样本。 |
| 7 静态最后版本/全分母 | pass，协议/聚合范围 | 当前静态修订与6静态/180动态零格回归，保持原分母。 |
| 8 动态同刻truth与先筛后选 | pass，纯core及已审诊断 | 不变scorer、边界/最新错误/迟到/同刻ID测试；v16补传4/60与8/60仍只是历史有限trace诊断，未重标为v18物理表现。 |
| 9 迟报补传/格起点优先级 | pass，现有机制覆盖范围 | v16实际34目标包/32pending正例与当前mock/纯scorer反例可复用；本次仍有180优先级事件。没有将旧241s作为v18同profile完整复跑。 |
| 10 安全退出、退休、自致失败不豁免 | **partial/fail，F1未关闭** | 原缺实际取消/返航/安全盘旋链已补足；不豁免负例沿用当前相同grade逻辑与v16 mock。退役后仍采集新mission数据，任务容量退出未闭合。 |

这些 pass 是具体工程检查的范围判断，不是G2全平台盖章。当前明确阻断是F1，不把未完成的G1/G3混称这个代码缺陷，也不让其他组完成情况替代它。后续 guard 修复必须新版本、新实际退休 probe；events=False 的 v18 G1 是否可组合复用，需在其完整结束后按实际 diff 与独立审查判断，本报告没有预先授予复用。

## 当前证据绑定

| 产物/文件 | SHA-256 |
| --- | --- |
| planning/gate-contract.v18.json | `f49a4439ed3fed4392d34ac69da5d087dea5c825ace49da39c1d2187163c8883` |
| qualification/protocol/retirement.v2.json | `99f1f0408dd48f950710d053c3f1d5cd99575dfc62400fd8f26164a90f393086` |
| qualification/protocol/retirement.v1.failure.json | `403260f7f23bbe4616048da20021c8967b706dca4e8d8a35815276a37fa9d674` |
| qualification/runtime/retirement.v1.log | `7f4eb429de2e21067e9139c169945bf674da6e00690023542505c914e4b24f39` |
| qualification/review/g2-v16-review.v1.md | `4f64037a1cf6e10b521c2baf72fbdcd8f3f5e4f8936d6146ab7b3fbdda0485de` |
| envs/inspection/control.py | `c676d237efe1b6227d238b4e5948c54a5725626e9d3f69e74ed9c1629a235270` |
| envs/inspection/execution.py | `dc301939ea1d55f08f01c69af5561c75f762ac45bdd0d144d677514387425ea4` |
| envs/inspection/flight.py | `ef9be915c92c207a3e8f43b9ddc1907010beea59be6d96136330e82a5d67cbdb` |
| envs/inspection/guidance.py | `9ab9c5290cbce60ad7f9fead635318f41561dedd1cfc2f0984b1cd45385bc0b9` |
| envs/inspection/mission.py | `24bc2a332508640f4043398f649e180b9cae167b930d22e2108135651c855b75` |
| envs/inspection/paths.py | `dbd31dc94a7cdcbcb17b8d9f1847fa88cf4cd12e4609ec8ba57e2233a367855b` |
| envs/inspection/planning.py | `b87fcdedf3aa26a46560056376fb69f9ff969270149c9cae429b4a9e294df6ab` |
| envs/inspection/protocol.py | `c32bdabb67bf8493ff5125d0af4fc289d1d0210db762016a9c56a96dd2f174a4` |
| envs/inspection/reference.py | `85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3` |
| envs/inspection/scenario.py | `c4f56e2783bc21c087c3abdcbfae9a7472feae2bbbe1e007d86bb39b28c40af2` |
| envs/inspection/scoring.py | `bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a` |
| envs/inspection/__init__.py | `cb0d38bbca6b531b261ca5a5f2f79664e352c9ddb6c19acc1777784c3115107a` |
| envs/__init__.py | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| eval/probe_inspection_retirement.py | `a4d79a7eb9a366ecea375710e5d7a0dbad111c41f00f8d8f849bfc2247e1e358` |
| eval/qualify_inspection_executor.py | `b76f6745e4f1cfed8c0b4055b683a93811f23685378ef2c84c3e723d8010a2cd` |
| eval/qualify_inspection_g1.py | `852ea8c02cade269b48ea1bf99158913ddaef2038f6a368bd0987c31cc794522` |
| eval/inspection_tracking.py | `19151659b4a969c51a6cbcb94b4f10aa1cb4489bebec6bd50f28b639a07884ba` |
| tests/test_inspection_protocol.py | `9ef62c113a607120128359eb6aeb2c21f15067c012945ddf351ce0bc89e9cb7d` |
| tests/test_inspection_scoring.py | `0f9c4f34c5c0f6a92dcaca6fe51b1c6831b826e221eae7ca710499bc41049403` |
| tests/test_inspection_flight.py | `c0d671fa7dd69539662dac3a5361bd8797837ad9270c575b764f1d1b5e11812e` |

依赖索引采用 `json.dumps(dependency_hashes,sort_keys=True,separators=(',',':'))` 的 SHA-256 为 `e341ce56ff077476e9752cf57372189de38091462697e24d048c5d9b3a823dbd`，完整61项值保存在原产物。本次结束前17源与61依赖再次核对无变化。

临时小结果为 `/tmp/g2-v18-review.qtPQtZ/recount.json`（`53286efc73b6c2ab344ca17d1c4a9d04fa5558ad7b2e6c93559888498a4cc768`）、`capacity-gap.json`（`7006c4fd403da3b76214c11ebcb7ecc3a084d7435b9eb5c6a26d8e7c2be414a5`）与 `tests.log`（`d46043dd252d31e1edf27349ef08447d7f7b0272f476539ba90a9466358d23de`）。表中 planning/qualification 路径相对研究根，其余源码路径相对仓库。旧结果均未回填新SHA或新门槛。
