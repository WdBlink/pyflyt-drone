# 新G1执行器资格设计独立审查 v1

日期：2026-11-05。独立上下文工程审查，继承主 Agent 模型配置；不是跨模型或领域人工复核。仅写本报告，未修改生产代码、契约或已有结果，未运行完整矩阵或付费服务。

## 结论

新设计的claim边界正确：ControlledPlannerExecutor复用共享job/planner/guidance，但把纯规划放在物理步之间同步计算，并在逻辑下一控制tick交回结果。它只能提供G1控制/因果可解性证据，不能称1×、实际规划时延或G3通过。

初审发现的契约未绑定、tracking审计覆盖计数未纳入通过判据两项P2，已在本轮v9版本补充并经有界反例核对。**仍有一项P2：mission构造后的fixture/precondition设置在try/finally外，设置异常漏close；直接run_case入口也未validate_contract。应在最终资格矩阵前修复。** 主Agent已确认该问题，计划保留运行中pilot版本、之后修复，不改写已跑结果。

已完成pilot.v1–v4全是诊断子集，qualified=false；v4只有observe/root0一条passed=true，不能升级为完整G1。旧100/100仅对应旧源码/旧控制profile；当前新profile的100-slot基线与20 roots×2个执行器family完整运行仍待提供证据。运行中的pilot.v5不属于本报告结果审查。

## 审阅范围与版本

已读eval/qualify_inspection_executor.py，planning.py的C2 bridge与计划输入/输出链，control.py的bounded vertical/anti-windup，flight.py显式altitudes，Mission96m初态/public_state/tool/逐物理步audit，以及执行器完成/捕获判据。读取gate-contract.v6/v7/v8，另按主Agent修复说明核对当前v9。pilot.v4和altitude-energy.v6均在确认完整结果已写入后读取；没有读取未完成slot冒充终态。

| 当前复核文件 | SHA256 |
| --- | --- |
| qualify_inspection_executor.py | a9121fa44485ccf0ed5f848fe416a732143ee1f777983d95a32bb87b8a401b82 |
| planning.py | 6f34edf4c112012fb9c76ae72be8d182296cf2b7e45d9be3d58b9a69cc0d978d |
| control.py | e6cdb38516dc172bd812dc7334862201f20912a1787ac800004953cca92dd4b5 |
| flight.py | ef9be915c92c207a3e8f43b9ddc1907010beea59be6d96136330e82a5d67cbdb |
| mission.py | 23b2f832fb08d0449a039307cf3a0147022f512d8cd34265065d1da53efadd45 |
| execution.py | 6dfefadaf1a2de93464e6feaf47fd831166d7af760bc475dbee280e916c5d595 |
| gate-contract.v9.json | 3122989ce399774878435e6852494a85f4a7cea14441bacf25b8b16ceeb8a819 |

初审runner为bef283fabdd97a6a2753c9c69db1652ee35ef6acdde72cbeaf0ccbbcde2c2ce2；历史pilot内嵌其各自执行版本。本报告不把当前SHA回填历史。

## 实质发现及最小修复

### P2-A：契约声明可被忽略；v9主入口已补，直接run_case仍需校验

初审runner:73–75、128、136、139、154–157中，family时长和minimum_delivered_D来自contract，其余风、180秒任务、240秒退出、20秒dwell为常量；execution.py的6m、±2m/20秒为共享常量，lead/profile/初态也无一致性验证。

独立从pilot.v4内嵌旧run_case抽取实际passed表达式，用该条真实通过统计作fixture：

- 原passed=true；
- 声明xy_tracking_tube_m=1、steady_altitude_error_m=0.1，而实际统计为3.432m/1.389m，旧表达式仍true；
- 当时实际inspection_gains与v8字典恰好一致，因此这是验收接口漏洞，不是已观测的v4参数漂移。

当前validate_contract:74–93核对当前inspection_gains、96m初态和所支持的固定物理/执行器阈值；main:222调用它。run_case:107–111另核对实际Hz/step period/lead，165–168读取observe时长和退出触发；205记录actual_control_profile。保持共享生产实现的固定阈值并拒绝不匹配contract，是可接受的最小方案，不必为这次审查重构成任意参数平台。

本审查验证v9被接受，过时v8、篡改gain、1m tube、0.9s lead、80m初态、重复roots均被拒绝。当前仍需在run_case入口调用同一validator：直接调用此公共检查函数目前可绕过main的前置校验，并可能在world创建后才发现缺字段（见P2-C）。

### P2-B：缺失tracking audit可因零breach误过；v9判据已修

初审passed只比较FlightWorld.physics_audits与world.physics_steps，未比较executor.tracking.physics_samples，也未要求实际完成全部请求物理步。把pilot.v4 fixture的tracking.physics_samples改为0，旧表达式仍true。

当前runner:183–189要求：

- 实际t等于请求duration；
- expected_steps>0；
- world.physics_steps = flight.physics_audits = tracking.physics_samples = duration×240；
- 全4机exit dwell满足20秒，并保留退出range reserve；
- 硬安全、基础设施错误、水平管与高度envelope breach均为零。

本审查直接执行当前表达式的有界fixture：完整fixture为true；tracking计数0、模拟时长359代替360、单机剩余range不足1000m分别为false。该判据缺口在当前main路径已关闭，不把这项单元检查当成新360秒物理运行。

### P2-C：fixture/precondition异常发生在finally之前，漏资源清理（仍未修）

当前runner在103行创建mission，106–120行计算selected并设置观测fixture，131行才进入运行try，195–200行才负责mission.close。107–111虽手动关闭一种Hz/lead不匹配情况，但无法覆盖其他lookup、fixture reset或Entity构造异常。

独立实测使用真实初始化的InspectionMission（四机altitudes=[96,96,96,96]、physics_hz=240、control_hz=120、lead=.5），仅在构造完成后将world.resetBasePositionAndOrientation替换为抛RuntimeError("injected fixture reset failure")，并用spy记录close。调用run_case("observe",0,v9)得到：

- 异常直接上抛，未返回case业务记录；
- close spy为false；
- 本审查finally手动调用close，未遗留world。

另一次并行版本检查用旧v8直接调用新run_case，aircraft_pairing_block缺失在mission已创建后抛KeyError，close也未调用；这不是reset故障本身，但独立显示直接入口前置校验与生命周期范围有同一缺口。主grid外层仍能写一个operational_failure文件、保留slot分母；问题是已有world/queue资源泄漏和setup上下文丢失，连续故障可能污染长矩阵资源状态。

最小修复：

1. run_case第一步validate_contract，完成无资源的参数校验；
2. mission成功构造后立即进入try/finally，把selected/runtime断言/fixture设置/actor运行/评分全部放入保护范围；
3. setup失败也返回或留存失败阶段与可取得的部分audit；关闭失败继续追加operational_error；
4. 增加上述reset故障和无效contract直接入口的有界检查，确认close恰一次且失败仍落入slot。

不用重跑全部矩阵来验证资源修复；但最终矩阵需绑定修复后的新源码，不回写历史pilot。

## 分母、异常与来源闭包

当前main的默认executor grid为root_ids0…19×observe/transfer，共40slot；它是新G1的两个扩展family，不是待重跑的5family×20基线100slot。每family分母固定len(roots)=20，失败completed和operational_failure不被删除；任一operational_failure阻止qualified，即使数值成功率仍≥.95。子集运行full_grid=false；--roots正且在可用范围、families唯一且受限。至少20个唯一非负int roots由validator检查，当前0…19的选择使每架selected机都有冷静/最大风根；这并不提供每个drone×wind子组的独立95%统计保证。

完整grid先写unstarted，再原子保存running；case JSON原子写入后保存hash与终态。异常外层写operational_failure文件，源码/依赖/contract结束不变检查进入qualified。这个留档设计保留分母，P2-C是其中仍需补齐的资源/部分记录缺口。

来源快照在本地模块导入前捕获，并在main运行前/结束时检查；包含inspection全部.py与qualify_inspection_g1.py原子写helper。pilot.v1/v2只保存13个文件，遗漏该helper，不能声称它们完整冻结了这个依赖；v3/v4已补为14个文件。61项PyFlyt源码/机型资产及PyBullet二进制哈希、PyFlyt/pybullet/numpy/numba版本也保留。历史pilot的slot文件SHA均匹配其manifest，未发现用当前源码替换历史内容；旧缺口与修复分开，不因失败记录保留就升级它们的可复现性。

## 因果边界、实际完成与claim

ControlledPlannerExecutor._request调用同一个super请求构造、同一个plan与PlanProposal，_poll复用共享接收/新鲜性/提交逻辑。差异是同步计算可暂停墙钟期间的物理推进，完成结果指定在now+1/120的逻辑时刻回包，并清空真实wall deadline。这是明确受控的G1执行方式，不能从它推出实际规划服务满足0.5秒capture lead。真实1×worker仍需独立异步G3证据。

fixture设置与独立grader保留Entity/truth；actor循环的候选、证据、版本和job状态来自public_state，并经tool发命令，没有把真实位置、未来轨迹或隐藏class拿来选候选。root/selected是资格fixture预先配置，物理tick仅用于轮询节拍；此进程内测试不是Actor RPC隔离的完整证明。public_state用deepcopy和public observation，不返回truth；本地执行器使用同机已minted观测及明确到达的跨机seed，不把预测变成报告测量。相关因果边界负例在局部测试中通过。

observe的实际资格指标是fixed appearance=0、60×3秒格、测量时刻真值与6秒交付宽限的D≥.8，不从首次报告重新起窗，也不只对发现后的格评分。observe ACK不单独算通过。transfer要求4个实际survey job succeeded；这些job由完整物理pass与service interval完成，不用ACK替代。所有机体最终还要实际exit dwell、硬安全、tube逐240Hz和退出range reserve。inspect的两次真实合格视图、observe job内部连续服务格与D是不同判据，不能把其中一个摘要当作全部语义的替代。

工程case延长330→360未改变主任务600秒。grade_dynamic_bins使用case deadline，但delivered格仍是0…180且最晚到186秒，因此330/360不会延长这60格或放宽其6秒宽限。minimum_delivered_D在v6–v9均为.8。

## C2 bridge、初态与vertical控制

C2 bridge只使用已提交own program、consumer-scoped ObservedTrack及其因果常速度外推。reference_acceleration对own committed future program求值，不是读取目标未来真值；preview也是本地导航预测，不是新的传感证据。接入后保留target-independent home tail，物理前缀不会因取消逻辑立即释放。

独立使用normalized quintic Hermite基函数核对一段4秒moving-circle capture：生成样本位置最大差1.08e−13m，首末位置误差分别0和约7.1e−15m。其底层解析quintic满足端点位置/速度/加速度约束；实际TimedPath是有限采样插值，不能把它进一步称为实机连续C2保证。最终跟踪能力仍由240Hz物理tube判定。

bounded vertical对desired vertical command作限幅，anti-windup在同向饱和时冻结高度积分。初审.8profile的120个饱和控制步后积分为0，反向误差可以解积分；这不是实测爬升率≤.8/1.2的证明，当前docstring及v9契约明确这一点。

v7采用所有机体96m初态，是新的dev可观测几何校准；FlightWorld校验每机altitude有限且在60–100m，复制输入避免alias。相对旧72/80/88/96初态，它改变传感可见性和控制条件，需要新资格，不能把旧100/100继承过来。

v8相对v7只增加.5s capture lead、两个工程case330→360；主600及D.8未变。v9又明确从desiredVZ=.8变为1.2，并把transition窗口从abs(Δh)/.8+20改成abs(Δh)/1.2+20；该改变不是v8的原结果，应单独冻结。当前actual inspection_gains与v9字典一致，v8被validator拒绝。

## 已有运行证据与失败保留

### Executor pilots

所有下列slot hash匹配manifest；物理/flight/tracking计数与330×240或360×240一致。各manifest均full_grid=false、qualified=false。

| pilot | family/root | D | passed | 关键限制 |
| --- | --- | --- | --- | --- |
| v1 / v6 | observe/0 | .411111 | false | D不足；失败保留 |
| v1 / v6 | transfer/0 | — | true | 仅该控制slot通过 |
| v2 / v6 | observe/0 | .627778 | false | D不足 |
| v3 / v7 | observe/0 | .461111 | false | D不足，最终一机exit dwell19.483s<20 |
| v3 / v7 | observe/1 | .766667 | false | D不足；altitude envelope breach783步，steady峰值2.489m |
| v4 / v8 | observe/0 | .977778 | true | 360s，四机exit均≥89.9s；tube breach为0；仅一个旧.8profile子集 |

本审查从accepted report journal、实际delivered observation与sensor audit的同测量时刻truth独立重算5份observe记录：每份都是完整60格，D与记录精确一致，未发现未来/未到达证据进入版本。v4的59个非零格与.977778不意味着全部格满分；没有删掉初期零格或错误证据。所有原失败均保留，没有因新lead/初态修复覆盖旧结果。

### Altitude-energy probes

v6完整结果已可读取。六份snapshot SHA与声明匹配，tracking/physics审计计数相等。下表只记录这些局部calibration probe，不能当20 roots×family的完整资格：

| 记录 | desiredVZ | 风 | 最大XY误差m | 最大settled Z误差m | passed |
| --- | --- | --- | --- | --- | --- |
| v1 | 未限幅旧profile | 0 | 16.640 | 1.218 | false |
| v2 | 未限幅、加高度油门项 | 0 | 12.403 | 1.526 | false |
| v3 | .8 | 0 | 2.155 | 1.054 | true |
| v4 | .8 | (-2,0) | 3.685 | 1.245 | true |
| v5 | .8、anti-windup/transition envelope | (-2,0) | 3.630 | .909 | true |
| v6 | 1.2、40s最大高度步settle窗口 | (-2,0) | 3.479 | 1.252 | true |

v1/v2各160秒，v3–v6各180秒；旧home72/80/88/96和30秒climb/90秒descend等具体schedule由各记录冻结，不能混为相同初态/时序比较。v6证明该单次风下高度切换probe满足其记录判据，不是当前新G1已通过。

## 实际检查与当前停止点

执行：

```sh
./.venv/bin/python -m pytest -q tests/test_inspection_execution.py tests/test_inspection_guidance_paths.py
```

结果21 passed in4.29s。该检查运行于本轮初审/修复推进期间；没有用它冒充新v9完整物理资格。另执行旧/新通过谓词反例、validator坏契约反例、C2 Hermite独立重算、anti-windup短检查、真实初始化fixture故障、历史D独立60格重算与hash/counter核对。未重跑任何完整G1或付费实验。

当前应先修P2-C并做有界复验，再冻结新源码进行待完成资格。其余未完成项包括当前profile的基线100slot、完整executor40slot、实际异步1×/G3、主研究场景与方法比较，以及领域人工复核。pilot或局部测试通过不能替代它们。

## Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- P2-C仍开放：初始化后的fixture错误漏finally/close；run_case入口也需validate。
- P2-A/B的main校验与完整步数判据已做反例复核；v9不同于v4旧profile，不回填历史。
- ControlledPlannerExecutor不是1×/G3证据，旧100/100不是当前源码资格，human_review_pending。


## v11剩余P2有界复验（追加）

本节更新初审的开放状态：**P2-C已关闭，本轮列出的P2-A/B/C均无剩余开放项。** 仅复核生命周期修复、集中验收判据与3项guard；未重新审查全系统，未运行完整G1矩阵，也未把pilot的新旧失败重新分类。

本次绑定在测试前后未变化：

| 文件 | SHA256 |
| --- | --- |
| qualify_inspection_executor.py | a8064b2eef9d68557006d8102b31b2c1ce9d823762b1be3b6cff67e02c32cbc8 |
| test_inspection_executor_qualification.py | 8b1d9b5efff6af3e1924f1617f238f4efe006bc93d7dafad8373c77a61e7c398 |
| gate-contract.v11.json | 3af795fa1b00cce49d29fb1732ff3758fbc44476ed2d185ec7d681ed715a3752 |
| control.py | e6cdb38516dc172bd812dc7334862201f20912a1787ac800004953cca92dd4b5 |
| execution.py | 4ef30355d997cc0f5c8e10b44dc062e003c356def1d6a5265ee7afb315a3db68 |

### 生命周期修复

run_case:122现在在创建mission前validate_contract，并验证root登记。runtime/初态检查与fixture reset已移入146–160的try；211–215的finally负责close，setup失败也走207–210保存partial business audit与operational_error。

实际执行tests/test_inspection_executor_qualification.py:44–63的真实world故障注入：InspectionMission成功初始化后，resetBasePositionAndOrientation抛"injected fixture setup failure"。run_case正常返回passed=false的失败记录，原错误保留，close spy恰为[True]，记录physics_steps=0。这直接覆盖初审复现的漏close路径，不是仅对mock资源测试。

### 单一判据与审计完整性

coverage_valid:96–103集中要求完整duration、四机exit/range数组、物理步/flight审计/tracking审计等于请求duration×240，以及actual_control_profile匹配。case_passed:106–118先拒绝operational_error，再结合coverage、任务D或四个实际survey succeeded、硬安全/管违规、退出dwell和range reserve。

run_case:224–225把缺失/不足coverage标成operational_error，再经226的唯一case_passed判定；main按operational_error保存operational_failure。这类平台覆盖失败因此不能作为普通不通过slot被每族95%门槛平均掉。pipeline没有另留旧passed表达式作第二套通过口径。

新增guard中，完整fixture可通过；tracking0、物理步86398代替86400、空exit数组分别不能通过。更严1m管和与实际不符的gain也被validator拒绝。本次没有增加分数门槛或弱化既有门槛。

### 契约与结论范围

v11仍为原root_ids0…19×observe/transfer、每族门槛.95、fixed appearance=0的180秒D≥.8，XY6m、steady Z2m、四机exit20秒、360秒工程case。actual profile desiredVZ=1.2、96m初态、capture lead=.5与当前实现匹配。新增inspect_camera_trial_seconds=2进入声明/validator及构造后runtime检查；本次仅核对其绑定，不重新评估该策略性能。

执行：

```sh
./.venv/bin/python -m pytest -q tests/test_inspection_executor_qualification.py
```

结果：**3 passed in0.51s**。未等待或重跑全矩阵，未重新审阅pilot.v5/v6结果；既有失败保持原版本记录。下一步可以冻结该修复版本执行原20 roots×2 family矩阵，以实际分布判定，不按单root改变D或安全阈值。这个设计审查结论不是新矩阵已通过，更不替代当前profile基线100slot、异步1×/G3或领域人工复核。

### 追加 Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- P2-C真实setup异常关闭路径已修复；初审P2-A/B/C均已在限定范围内关闭。
- 没有把guard fixture或旧pilot升级为完整G1结果；现有全部实验/失败记录保持各自源码绑定，human_review_pending。


