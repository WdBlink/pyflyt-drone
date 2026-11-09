# 候选 B：带时间的曲率图与因果移动观察环

## Problem

现状是 `mission._advance_jobs()` 直接写 `flight.modes`，所有非退出工作都继续原 capsule；`_sample()` 更新 `aim_xy`，但不会改变飞机航迹。`assign_survey` 以机号限制 stripe，`observe` 只需“计时结束前有一次测量”。需要同时替换这两项能力，而保留共享 PyFlyt 世界、120 Hz 控制/240 Hz 审计、版本化 JobBook、不可变证据、延迟上传及原评分器。本候选来自同一家族模型的独立上下文；未读取另一候选。生产代码、研究 state 与已有证据均未改动。

## Usage (caller's view)

Actor 继续调用现有六项任务工具。例如机 0 可提交 `assign_survey(region_id="stripe-3", duration=90, expected_version=...)`；返回 accepted 时工作仍 queued。执行进度公开 `waiting_traffic → transiting → surveying`，进入目标 capsule 后的真实完整一圈才满足 survey postcondition。交通延迟有公开原因和截止时间，不暗中改配到其他区域。

Actor 从已到达的 observation 得到 candidate handle、先建立 report，再调用 `observe_track(candidate_id=..., duration=30, report_id=...)`。executor 实际转场至预测位置周围的移动观察环，而相机用因果预测指向候选。`handoff_observation` 启动接收机获取；源机直到接收机取得新的合格样本才交接，目标身份只由观测空间关联支持。取消立即终止任务采样/新上传绑定，飞机仍飞已经预约的安全延续；已采样的证据可照常延迟到达。

Worker 每控制 tick 只调用 `decision = executor.tick(FleetFrame(...))`，把版本匹配的 facts 交给 JobBook，把四个 targets 一起交给 `FlightWorld.step(targets=...)`。传感器在原有 2 Hz 时点采样；本地测量进入下一 tick，公网 uplink 仍独立。完整示例先于 types，见 `sketch.py` 顶部。

## Shape

**数据和所有权。** `Four[OrderSlot]` 是 JobBook 的不可变活动工作投影；每项带精确 `(job_id, aircraft, version)`。JobBook 是状态/版本唯一写者，对执行 facts 的序号去重并忽略旧 token。`FleetExecutor` 是估计器、服务窗口、几何图、预约、计划与导航目标的唯一写者。`FlightWorld` 保留 controller/物理/audit；删除可被 mission 随意写的 modes/routes/headings 作为任务控制入口；资格脚本同次迁移到同一 targets 路径，不能保留两套工作执行器。mission 仍独占传感、transport、ledger 和隐藏世界。这样遵循 `single source of truth` 与 `boundary-discipline`；外部没有可变 `aim_xy` 与执行器内部 tracker 的同步问题。

每机 `_ActorState` 只消费它自己的本地相机测量，或明确已送达的请求 seed；不共享未送达 tracker。`LocalView` 区分 `measured_at` 与本地 `available_at`，actor 的 `Observation.arrived_at` 继续由 transport 决定。每条测量保留 evidence ID 与采样时 job token；本地记录中无 `entity_key`、seed、未来轨迹、appearance time、真类别或 grading verdict。安全层可使用全部四架本方飞机的导航状态，这一权限对各 policy 一致，不能把它转化成目标真值。滤波按测量时间更新，重包幂等，晚包可上传但不倒带飞行。候选的重新关联靠空间门控/协方差，歧义时记 gap，不借隐藏 ID 消歧。

**一张前向曲率图。** 所有工作编译成 `_Program(prefix, periodic_tail, tube)`，所有新动作通过同一时间预约与认证入口。等待/服务 capsule 按截止时间有限展开到 prefix，最终 periodic tail 统一为本机原始退出圆，确保不是永远巡航而忘记退出。图的基础边是精确积分的直线/左右圆弧，survey capsule 是其已知闭环；对任意公开 region 使用同一个几何构造。以半径 100 m，从 `(-250,c-80,E)` 先向目标方向转 90°，走竖直直线 `|c'-c|-200`，反向转 90°，抵达 `(-50,c'-80,E)`。12 个有向 stripe 对均可构造，完全位于 `x∈[-250,-50], y∈[-380,220]`。跨 200/400/600 m 分别长 314.16/514.16/714.16 m，20 m/s 下约 15.71/25.71/35.71 s；不含等端口、捕获与交通延迟。现有 capsule 一圈 1502.65 m / 75.13 s，因此“排队”是继续飞一圈到下一 departure phase，不是停住。

任意当前姿态、观察环和安全退出之间用同一图的增量 hybrid search 连接；保留连续的真实端点，位置/heading 离散只用于去重。初步 20 m/s 转场新边 R≥100 m（18 m/s 观察环 R=90 m），不能把量化后的节点当作飞机位置，也不能把“接近 goal”当作已捕获；必须进入经过资格验证的捕获走廊。标准 survey/exit 的 R=80 m 保留原定义。圆弧与直线是本候选自己的解析表达，没有复制第三方实现；若实施阶段引入 Dubins 或其他路径库，先核验其原始来源和许可证，再替换这一内部部分。

**持续移动观察。** 禁止以车辆速度追逐航点（固定翼无法悬停）。本机因果 constant-velocity 估计产生短期中心、速度与不确定区，构造随中心平移的环，最佳候选用半径 90 m、AGL 96 m、空速 18 m/s，巡航/转场仍为 20 m/s。18 m/s 位于已声明 15..25 m/s 包线内，本轮父级明确允许评估该模式；减速过渡也必须实际飞行，并单独取得资格。圆周切向 `t`、`d=estimated_center_velocity-wind` 时，令 `s=-d·t+sqrt((d·t)^2+Va²-|d|²)`，命令地速方向为 `center_velocity+s*t`，用该运动的空气航向导数生成 bank feedforward；恒定移动中心情形下空速精确为 Va。这个公式避免将静止 `orbit_target` 的 feedforward 原样用于移动圆。径向纠偏、有限 roll 响应仍需实测。

中心更新只产生新计划后缀，不能直接平移已提交航迹。短期匀速中心只是当前 belief；交通预约依据自己已提交的飞行程序，不依据未知目标未来真位置，segment 到期后的延续须与目标无关。新后缀要有连续捕获及独立于目标的安全延续，经四机认证后整批提交。几何条件不满足、测量过时或 bank 无余量时，继续既有安全程序，公开 coasting/reacquiring/gap 或失败；不得缩半径、增大围栏或用隐藏速度救场。

**几何的可行条件并不宽。** `probe_geometry.py` 合成一项 1.5 m/s 匀速中心、相反 2 m/s 风，仅用于否证公式：R=80/90/100/105 m 的最大需求 bank 约 35.13/32.03/29.38/28.20°。因此 20 m/s 的原 80 m 观察圆不能直接承接该工况，100 m 也仅剩约 0.62° 反馈余量。父级补充允许在已声明速度包线内使用 18 m/s 后，选择 R=90 m；同一合成工况的最大需求降至 27.63°，留下约 2.37° 反馈余量。不能把 1.5 m/s 当作已公开的目标速度上界；实施时必须从当前估计检查需求。无误差时 96 m 高度的 50° 离轴距离限约 114.41 m；若考虑 0.3 m 相机偏移与暂拟 ±2 m 高度误差，下界变为 111.67 m。再扣 90 m 半径、6 m 水平跟踪管与 5 m 中心不确定半径，余约 10.67 m。相比维持 20 m/s 的 100 m 环，增加的 10 m 视线余量值得新增速度模式资格成本。这里 **6/2/5 m 都是待否证的设计假设，不能当作现有资格结果**。目标估计不确定区是预测风险量，也不是真值包络；作业成功只认实际样本。一般 survey 样本符合原质量门槛时仍可合法进入 report，不把 inspect 变成唯一评分路径；本候选不改原 ledger/scoring 规则。

**四机预约与持续物理。** 任何提交都是完整 `Four[_Program]` 的原子替换；任务前缀后面必须有不依赖目标的退出/安全闭环。四个原始退出圆保持稳定位置，与工作 stripe 独立，因此退出 station 固定不等于工作 stripe 固定。相邻原 survey capsule 的名义间隙 40 m；按暂拟 6 m 水平 tube，所需名义水平间隙是 `15+6+6=27 m`。现有 8 m 高度间隔不能独自满足 15 m。共享 capsule 必须预约足够的相位差；同中心观察环可有多个不同相位，但不能因此免除转场预约。

认证覆盖前缀与无限重复尾环，不能截取 30 s 后就忘记冲突。四个终态退出圆之间直接采用相位无关的 40 m 几何间隙；finite prefix 与其他飞机退出尾圆之间仍检验整个交叠区间的相位和时间。观察环预测仅存在于有限前缀，未使用未知目标未来来证明无限安全。采样距离检测须有连续时间余量，例如最大地速按包络 25+2=27 m/s、50 ms 区间，用中点距离需再扣两机相对速度的 `54*0.05/2=1.35 m`。围栏用水平 tube 侵蚀，高度单独使用竖直 tube，所有飞机每次物理积分继续独立硬审计。

计划采用 FIFO/精确 token 的确定性流量优先级，不阅读任务权重去代做 mission allocation。search 每 tick 只做固定数量展开，未完成则沿原已认证程序继续飞；不等待求解器、不暂停 physics。新方案必须留出从真实当前管到退出圆的完整路径、20 s dwell、deadline 和路径消耗后 1000 m reserve。取消/退役只切换任务 ownership，不能释放仍被飞机占用的空域；退出可正常预占任务但也必须飞完物理转场。测量脱管时公开 uncertified 并记录失败，不能以安全滤波器的名字掩盖实际违规。

**作业时序。** queued 表示尚未取得可执行路径；running 开始于计划实际执行。Survey 的 duration 从实测捕获目标航线开始，须完成真实整圈；Observe 的 duration 从第一条新合格样本开始，获取有明确超时，并预留退出时间。该起算点澄清改变了旧实现，必须更新工具语义版本和测试，不能悄悄沿用旧解释。Observe 的每个公开 `bin_seconds` 服务小段至少有一条合格本地样本、缺测串不超过公开 grace，预测不填 bin；bins 以本次获取起点锚定，不读取隐藏 appeared_at。全部预期 2 Hz 缺测区间逐段记录，样本间真实可见性保持未知。服务 facts 只证明本地执行，不能等同报告已到达或 D 得分。交接获取事实绑定旧源 token，绝不取消后来取代它的新工作；失联不停止机上执行，退休/取消不使旧证据变成新工作证据。

**接口深度。** 对 mission 只新增一个 `tick(frame) -> decision`，隐藏因果跟踪、观察模式、路径连接、交通排队、安全延续与完成监测。Actor 不接触 waypoint、planner 阶段、半径、置信阈值或目标真值。不可变 frame 再加 executor 单写者，比让 mission 逐项协调 planner/tracker/route manager 更符合 `encode-lessons-in-structure`。Python 中私有命名不是安全边界；实施应加窄范围依赖检查，禁止 mission 直接写 executor 内部或其他模块导入 `_...` helpers，真实信任边界仍由 worker/RPC allowlist 保证。

## Synthesis decision

留给父级 arena 综合。此候选未读取另一上下文，也未声称已有独立裁判结果。

## Tradeoffs accepted

- 接受等待完整 capsule 相位与显式 traffic failure，换取真实前向固定翼转场及可审计准入；任意 stripe 不等于任意时刻保证准入。
- 接受 executor 内部集中较多几何/时序复杂度，换取 actor/mission 的小接口和单写者；不再建立 tracker → planner → traffic → route 多个可变服务链。
- 接受观察高度约 96 m、空速降至 18 m/s 并新增资格工作，换取无需悬停且留有视线余量的标准固定翼观察；不把高度差当作四机避碰方案。
- 接受保守的尾环/交通拒绝，换取不靠无限预测或暂停世界。拒绝率和 cross-stripe 实际成功率必须实测，不能只通过负例。

## Alternatives considered

**可行替代：直接联合 receding-horizon NMPC。** 每次以四机测得姿态和本地目标 belief 为初值，联合优化未来 bank/airspeed/altitude 与相机方向，围栏、空速、转弯率、3D 间距为硬约束，目标代价来自当前 actor 意图；控制步只执行首段，超时沿已备份闭环飞行。用共享 tracking/exit terminal set 和短时离散 shooting 即可实现，不需要学习器。它把几何规划和相位预约完全藏在一个优化器中，接口同样深，并更善于处理密集移动目标；但多机非凸解的确定性、运行时、可行解保持与物理偏差同时耦合，所需新资格明显大于解析图。本候选选择持久化可审计路径及保守 admission，优先降低否证成本。NMPC 也不能省略持续备份；没有可行 terminal set 不是靠缩短 horizon 就能解决。

**可行但能力较保守的替代：单向航路网与路口 token。** 所有机沿固定 forward-only 网络巡航；跨区经带 token 的公共交叉口，观察任务预订独占 moving bubble。它可实现任意 stripe 且不会停机，但一次观察将排斥其他机的大块空域，交接和同目标多机难充分使用相位自由度；token 与 bubble 的等待语义也容易泄露给调用方。只有观察范围较少且串行流量可接受时才优于候选 B。

## Open questions and risks

1. 90 m / 96 m / 18 m/s 的真实 moving-ring 在 2 m/s 风与测量噪声下，能否同时保持 30° 命令目标、±2 m 高度及 ≤6 m 水平管？解析 bank 尚有约 2.37° 反馈余量，实际 roll 与风响应未知，应先否证它；失败时重新权衡环形状/高度/控制方法，不能宣布任务不再需要跟踪。
2. 有限 primitive search 能否从观察环任意相位接到 survey/exit，且不因受限端口高频拒绝？几何上存在 stripe 对之间的路径不等于任意当前姿态可在 deadline 前到达。
3. 四个交错转场、两机同区域及 moving-ring 与 survey 同时运行时，等待相位是否存在足够机会？必须包含成功用例，不能通过“全部 queue/reject”满足安全。
4. 新的服务 duration 起点与获取超时是否与最终研究任务措辞一致？实施前在工具文档/semantic version 明示并由父级统一，不需要 actor 提交路径参数。领域人工复核仍 pending。

## 最小否证 probes 与检查

已运行标准库解析 probe，输出 `geometry_probe.json`；12/12 有向连接端点和边界成立，R80 moving-ring 的超 30° 负例被捕获。`sketch.py` 与 probe 已做 `py_compile`；它们不是生产能力或 G1–G4 资格。

实施前依次做下列独立小 fixtures；均真实共享 PyFlyt、只在初态设置飞机位置、每 240 Hz 积分审计，不运行 600 s/grid：

1. **先否证环，约 35–45 s。** 单机从合法初态以 20 m/s 巡航，真实减速后进入 R90/96 m/18 m/s 环；合成 1.5 m/s 目标，2 Hz 带噪样本、独立 0.5 s actor transport、2 m/s 相反风。控制输入只读本地样本；真代理仅供光线/独立误差量测。一次稳态圈即检查水平/竖直 tube、view gap、bank/airspeed/围栏。再给突然缺测/转向，须有 coasting/gap/安全延续，不读取未来。任一管失败即否证观察几何假设。
2. **单机端口转场，约 45–70 s。** 先做最短 200 m 与最长 600 m 转场，覆盖两种 turn sense；检查真实轨迹连接、捕获误差、步数审计和目标 stripe lap 计数。纯解析路径通过不足以放行 capture。
3. **四机交错，约 90 s。** 同一 world 请求 0↔3、1↔2 对向换区，既有路径持续飞行。至少一条跨区任务实际捕获目标，四机均不违规，阻塞有因果理由；再测试两机同 stripe 的相位准入。若所有跨区请求永远拒绝，此设计失败。
4. **密集观察/交接，约 45–90 s。** 两机在同 moving ring 观察并交接，另外两机继续 capsule。若冲突排队也须至少完成一次 receiver 真实获取，source 在此之前继续；改变 source version 后旧获取不得取消新工作。
5. **负例可用合成 frame，不需长物理。** stale version/cancel、重复 view、未来 local availability、假候选、同类近邻、脱管、目的区围栏冲突、不可行 bank、deadline/range 不足、退休和延迟上传。改变隐藏 seed/entity key 或未来运动但保持历史 frame 相同，下一 guidance 必须相同。把 planner 固定慢/永不返回，仍有完整物理步数与有限目标，不出现 freeze。

这些 probes 通过后才按新增 control families 重新做冻结 G1 资格，并检验真实 1× wall clock 与 executed-journal replay；不能继承旧的 100/100 名额，也不以解析几何冒称物理合格。

## Next implementation step

先实现上述 moving-ring 单机因果 fixture 和其独立审计，再实现 `FleetExecutor.tick` 的不可变输入、版本对账与四机安全续飞骨架；观察环的新速度与捕获可行域未通过前不扩写复杂调度器。
