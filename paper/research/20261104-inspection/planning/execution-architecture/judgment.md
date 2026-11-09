# 独立交叉评判

建议以 **A 的单写者 executor、独立规划进程和原子四机 continuation** 为实现基线，移植 B 的两个局部构造。第一项是 18 m/s、R90、AGL96 的因果移动观察环及其恒空速公式。第二项是 R100 的短 S 形 stripe 连接，作为有限连接候选而非强制唯一航路。保留 A 的四机正反例作为交通验收素材，重新计算移植后的交通安排。

这是同一模型家族的独立上下文评判。两份候选的 sketch、rationale、probe 已完整阅读，也核对了当前 control、flight、mission、protocol 及动态评分语义。评判只写本文件，没有改候选或生产代码，没有运行 600 s、grid、模型 API 或正式数据。

## 证据等级与评分

这里的 0 表示缺失或与要求冲突，1 表示有方向但关键机制未闭合，2 表示形状可实施但需要指定修订或否证，3 表示该项架构契约处理完整。分数用于设计选择，**不是物理资格分数**。两个 sketch 的主要方法仍是 NotImplementedError；不存在新模式的物理通过证据。

| Rubric，0..3 | A | B | 证据与扣分原因 |
| --- | ---: | ---: | --- |
| 围栏、最小转弯与间距的物理可行性 | 2 | 2 | A 给出完整四机循环换区的理想正反例，但 20 m/s 逆风观察的视锥余量极窄。B 给出更短的 12 个单机连接与更有余量的 18 m/s 环，但没有四机连接后的间距正例。两者都未证明 roll、捕获、加减速、相位误差及完整恢复段可飞。 |
| 因果信息边界 | 2 | 3 | 两者都禁止隐藏未来，区分本地可用与上传到达。B 明确逐机 estimator 和已送达 ObservedSeed。A 的文字说明正确，但 TrackEstimate 没有知识域，PlanRequest.tracks 是扁平全集，容易把另一机未传输的私有估计用于本机跟踪。B 的 3 分只评价所写信息契约，仍需结构及负例测试。 |
| 任务能力与公平的共享执行 | 2 | 2 | 都允许任意公开 region、真实转场、先获取再 handoff，交通不代做任务分配，也不靠全拒绝交差。A 有非空交通 witness。B 的能力计划完整，但成功吞吐未测，而且把公开上传 grace 复用成传感缺测规则，需要纠正。 |
| 单一所有权、取消与失败语义 | 3 | 2 | A 把 JobBook、版本、任务进度和航迹提交放在一个物理 worker owner，迟到 proposal 无权复活终态。B 的 JobBook 与 executor 分工原则上可行，并非天然双写，但 service 状态与 JobBook status 之间多了一次外部对账；取消、handoff、retirement、deadline 同 tick 的事务顺序没有闭合。 |
| 有界资格与复用成本 | 2 | 1 | A 将耗时几何与搜索放独立进程，较容易隔离 120 Hz 风险，但泛化路径搜索和迁移全部关联仍偏大。B 在 tick 中增量 Hybrid A* 及全 program 认证；max_expansions 不限制单次扩展、全前缀/尾环认证和 Python 数据处理耗时，不能据此声称真实 120 Hz。 |
| 小而可维护的接口 | 2 | 2 | 两者都保留任务级 API，隐藏导航细节。A 的三方法面较深，但重复 AircraftCommand、未带资格标识的证书和可变长四机 tuple 需收紧。B 的单 tick 很清楚，但调用者仍协调 JobBook facts，私有通用图和搜索器的维护成本不能因 public API 短就忽略。 |
| 合计 | **13/18** | **12/18** | 选择依据是控制时限和状态事务边界，不是合计的微小差距。 |

A 的所有权优势不能补偿物理失败。若移植的观察模式不能在现有控制器上获得非空可行域，应停止调度器扩写，回到控制与观察几何，而不是继续按这张分数表推进。

## 已复核的解析事实

两个 stdlib probe 均在本次评判中复跑，所有现有断言通过。两个 sketch 和两个 probe 在内存 compile 通过。B 输出与保存的 geometry_probe.json 逐字段相同。没有导入 sketch 执行 executor，没有由这些结果推断物理或实时资格。

- A 的 12 个有向连接最大端点误差约 1.71e-13 m，理想包络 x=[-340,-160]、y=[-380,400]。最长转场约 58.27 s，不含等门、roll 与捕获。
- A 四机 0→1→2→3→0 同时出发的采样最小距约 16 m，等一整圈的方案约 19.713 m，两者不满足暂拟双 6 m 管。第 4 机继续实际飞 8 s 的方案，采样最小距约 31.657 m，扣除 1 m 连续时间界后约 30.657 m。该周期尾部检查成立于理想恒速点质点，不能把 8 s 原样用于新 R100 连接或有风真实飞机。
- B 的 R100 连接端点为 (-250,c-80,E)→(-50,c′-80,E)，理想包络 x=[-250,-50]、y=[-380,220]。200/400/600 m 跨度在 20 m/s 下分别约 15.71/25.71/35.71 s。相邻区连接没有中间直线，需要从左 bank 过渡到右 bank；解析切线连续不代表 roll-rate 可行。平滑后必须重算端点、围栏和时序。
- 同一个合成匀速中心 1.5 m/s、相反风 2 m/s 工况，B 在 20 m/s、R90 时最大解析 bank 约 32.03°，18 m/s、R90 时约 27.63°。后者在 30°控制目标下仍只有约 2.37°反馈余量。控制器实际硬审计是 45°，不能把它当作可以绕过 30°控制目标的许可。18 在公开 15..25 包线内，20 是标称值；选择 18 合法，但加减速和观察均须新资格。
- B 的原视锥余量约 10.667 m，还没有减水平光学偏移和被测车辆高度。当前 mission.camera_origin 的 body offset 是 0.3 m，_visible 对有 body 的目标使用 z=0.6 m。用同样保守的分量界补这两项后，余量约 9.652 m。A 的 20 m/s、R106、AGL98 角点原余量约 0.943 m，补 0.6 m 目标高度后约 0.228 m。这些值仍假设所写高度管、水平管、估计半径和恒定中心速度成立，未覆盖目标加速度、观测中断、roll transient 与速度误差。

围栏、转弯、间距、视锥必须在同一已准入 program 上同时成立，不能把四项各自的单机好例拼成全机证书。6 m 水平与 2 m 竖直管若按各向异性圆柱表示，不能把 6 当作三维欧氏误差半径。保守球界需要每机 sqrt(6²+2²)≈6.325 m；也可以直接计算两圆柱的最小 3D 距离。B 直接要求名义水平距至少 27 m 对这个管是充分的，虽然较保守。A 的 30.657 m witness 在更保守的球界下仍有理想余量，但没有资格意义。

## 采用的最小基线

保留现有 FlightTarget、FixedwingAutopilot、共享 FlightWorld 与每 240 Hz 积分审计。只新增一个 executor 公共边界和一个私有纯路径模块。独立规划进程只读不可变快照，返回 proposal；它不是第二个业务 owner，也不能导入 Mission、Aviary、生成器或 grader。mission 继续拥有传感模拟、隐藏 provenance、不可变 Evidence、上传、报告和评分。

executor 私有持有现有 JobBook，负责工具提交后的版本校验、生命周期、逐机 causal filter、服务窗口、四机 reservation 及原子提交。mission 只解析/校验外部请求并调用 submit，不能再直接 jobs.transition 或写 flight.modes/routes/headings。传感任务 postcondition 从 _sample 迁出。若先复用现有观测关联，应只留一套观测空间规则，并送 stripped、不可变的本地记录；无需为了满足接口形式立刻搬迁整个 Evidence 构造，也不能平行保留两套 candidate 关联或两套 job transition。

每架飞机的已提交 program 由不可中断前缀、有限作业段、**已经占用相应时空的目标无关恢复段**和原始退出圆组成。默认等待继续飞已有程序。取消只结束逻辑作业与新曝光绑定，物理占用不释放；飞到既定可行恢复门后再恢复。deadline、退役、planner 停止、测量失效都由同一 owner 选择现有程序的续段，不能临时向已故障 planner 请求逃生路线。固定退出站与任意工作 stripe 并不冲突。

初版路径只实现少量解析前向 primitive、预先声明的 survey/exit ports、有限 tangent/capture 候选以及两种 stripe 连接族。移动环和通向退出的有限后缀也必须在准入前真实可连接。按固定 FIFO/token 顺序试有限 departure phases，找不到则给公开排队原因和有限截止。**先不实现通用 Hybrid A*、全地图 heading/bank lattice 或 NMPC**。若小范围正例无法覆盖必要 inspect/observe/exit 捕获，再用失败证据决定是否扩展连接器，不能用“框架以后能解决”代替这轮任务。

## 两项 graft

1. 移植 B 的 18 m/s、R90、AGL96 移动观察模式与恒空速 air-relative heading-rate 推导。所有中心/速度来自本机已可用历史。每段有有限有效期，下一段需连续捕获与全机准入。18 是待资格参数，不是新固定假设，也不能把合成的 1.5 m/s 当作场景公开上界。需在现有正 roll 右转、ENU course 经 flight 做一次 crab correction 的约定下适配 feedforward 符号，避免重复 crab。
2. 移植 B 的 R100 短 S 连接作为更短、更低转场曲率的候选，同时保留 A 的连接族用于有限选择。换用短 S 不继承 A 的 8 s 交通安排。先加入可飞的 roll-in/out 和 bank reversal，再用相同围栏/间距认证入口重算，不新增 actor 航点参数。

B 的逐机知识域形状与最终退出尾环也值得吸收，但在本裁决中它们属于修复共同硬契约的必需接口收紧，不额外扩为第三套架构。

## 实现前必须收紧的接口

1. **逐机知识域。** TrackEstimate 增加 consumer_drone 或等价 knowledge_scope，及 supporting detection IDs、这些来源在该域的 available_at。PlanRequest 按四机任务分组提供估计。handoff 只转交已按声明通道到达的 seed；看见 handle 不意味着接收机能读源机的最新私有滤波。全机自身导航和已提交航迹可共享用于交通安全。相同可见历史、不同隐藏未来在分叉前必须得到相同 guidance；下一真实测量前不得因实体未来变化而改变计划。
2. **资格参数与固定四机形状。** QualifiedEnvelope 由外部实测产物冻结，带 ID、风/速度/半径/roll-rate/捕获适用域、水平/竖直及相位误差界。planner 不能自选更小 tube 给自己发证。FleetProposal 绑定该 ID、当前 reservation revision、四个 job token、输入 tick、capture/commit 截止以及有界 primitive 列表。四机命令和 reservation 在构造边界验证恰好四个唯一 aircraft ID；内部复用 FlightTarget，不维护第二套含义相同的 heading/airspeed DTO。
3. **真实 120 Hz 的预算。** 搜索、全段几何、无限尾环检查在独立进程。控制 tick 只处理有上限的命令/本地包、检查有限大小的新鲜 proposal、检查捕获状态并取四个 setpoint。限制消息字节数与 program 长度；不能仅 poll 后反序列化无限长历史。未处理传感包保留原测量/可用时刻，不可伪造新鲜样本，Evidence transport 也不能因控制队列合并丢失。慢、永不返回、积压和过期 proposal 都继续同一个 committed program。实际资格需测含物理与传感的 1×运行，不能用“无 wait”证明整个 8.333 ms 周期足够。
4. **完整回退与单事务。** constructor 安装的初始 program 也必须在有限 capsule 等待后有完整物理退出及 20 s dwell，不能只有无限 home hold。所有 proposal 都包含恢复占用和到达退出尾环的距离/时间。固定同 tick 的处理顺序，先失效事件/版本，再接纳本地历史和 proposal，最后服务完成；同 tick 取消/退役优先于旧任务成功，具体顺序写入协议。handoff 只有接收机新合格样本及双机续段已提交才取消精确旧 source token。迟到包仍上传冻结旧绑定，终态不可复活。断 ground link 不改变机载执行时钟。
5. **几何认证。** 处理整个连续 primitive 及实测误差管，检查围栏、AGL、空速、bank/roll-rate、有限前缀与全相位尾环、deadline 和实际已飞距离。三维间距采用正确管运算，或保守要求水平距超过 15 加两水平管与数值余量。采样证书另扣速度 Lipschitz 界，27 m/s 两机与 50 ms 区间的补偿为 1.35 m。资格管不成立时记 breach/uncertified 与失败，已有回退不能把已发生违规抹掉。
6. **服务与上传两种截止。** Survey 从实测目标路线捕获开始计 duration，并完成真实 pass/lap。Observe 从第一条新的合格本地样本起计墙钟服务区间，有有限 acquisition timeout；丢测不补时、预测不填 bin，末尾 partial bin 规则要写清。公开 dynamic_requirement.grace_seconds=6 在现有 scoring.py 中是每个 bin 的上传到达宽限，**不是传感丢失 grace**。B 的 observation_grace 应改为 delivery_grace_seconds，仅由 transport/scorer 使用；本地服务只统计自己的测量 bin/gap，若另设失测阈值，作为清楚命名的执行规则单独声明。不可改变独立 grader 从隐藏 appeared_at 开始的 180 s 窗口或分母。服务成功、证据已产生、ground 已收到必须保持三个事实。duration 起点变更须同步工具语义版本与调用方测试，不能回写旧结果。

这些是窄接口约束，不要求建立新插件框架、地图服务、消息总线或独立导航产品。

## 拒绝项

- 拒绝把 B 的固定搜索展开数或 Python 方法标为 bounded 当作真实 120 Hz 证据，尤其不把全航迹认证藏在该 tick 的一次调用中。
- 拒绝 A 的无知识域扁平 tracks 原样进入实现；中央 owner 不能变成未声明的传感融合通道。
- 拒绝在 20 m/s 角点只剩亚米视锥余量时继续扩大调度器，再期待 controller 自行实现理想管。也拒绝复用 R80 的静态 orbit feedforward 来跟踪移动中心。
- 拒绝用 B 的 6 s 上传 grace 放宽传感后置条件；不以 route lap、预测 camera cone 或 ACK 代替真实样本/退出 dwell。
- 拒绝仅做全拒绝安全样例、只比较 waypoint 距离、依赖 8 m 高度台阶、取消即释放空域，或先规划作业再在失败时补算恢复。
- 拒绝在第一轮引入通用 Hybrid A*、lattice、NMPC 和第三方大型导航框架。两份草图的通用搜索扩展都应以已复现的缺口作为启动条件。

## 最先物理否证的风险与后续验收

**先否证观察环的真实误差管与视锥，不先写复杂交通搜索。** 用 upstream PyFlyt 短 fixture 从合法初态实际减速、爬升并捕获 18 m/s、R90、AGL96 环。目标仅向 sensor/auditor 提供合成真值，executor 只收 2 Hz 因果带噪测量。采用 2 m/s 相反风与合成 1.5 m/s 中心移动，再加短失测和一次转向。fixture 长度以捕获后至少一圈为准，不以预设短时长掩盖稳态不足。每 240 Hz 测量实际 bank/空速/AGL/围栏，并独立记录空间与相位误差、真实视锥、qualified views 和缺失 bins。若暂拟 6 m/2 m 管或观察所需余量不能成立，就否证该组参数；不自动放宽 hard limits。第一阶段无须运行 600 s。

随后按小步顺序验证。

1. 单机最短/最长两类 stripe 连接，覆盖两个转向、R100 bank reversal、观察环任意选定捕获相位到原始退出圆、以及恢复段。确认可飞连接实际存在，不能只测圆本身。
2. 同一四机 world 的交错换区，以及两机同 stripe/同观察区并行。必须有实际进入别区并完成 pass 的非空正例和明确冲突负例，记录全部实际间距。A 的 witness 可作为输入灵感，移植后不得沿用其解析结果。
3. 取消、handoff 接收失败、source 换版本、退役与截止同 tick、晚包、ground 断链，以及 planner 慢/停止。逻辑终态与物理回退均只有一个 owner；有完整物理步数和真实持续前进。handoff 至少有一次接收机实际获取成功，不能只覆盖拒绝路径。
4. 同历史不同隐藏未来、未送达跨机估计、重复/乱序/未来 available_at、近邻歧义、range/deadline 不足等负例。不准用私有实体键救关联，也不准用预测填证据。
5. 上述小 fixture 成立后才为新增控制族冻结并执行新 G1 资格，保留原控制族回归，再测真实 1×与 executed-journal replay。旧 100/100 和旧 600 s reference 不能替新模式背书；G2/G3/G4、正式比较及人工领域复核仍待完成。

本次独立评判已经完成。结论支持进入最小可执行否证与接口实现，未授予任何新物理资格。
