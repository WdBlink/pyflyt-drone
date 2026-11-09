# Candidate A — 单写者 executor 与原子全机航迹预留

## Problem

当前 `mission._advance_jobs` 写 `flight.modes`，`FlightWorld.target` 把非退出任务继续放在本机原 capsule，`_sample` 仅更新相机 `aim_xy`。所以 observe 没有因果跟飞，`assign_survey` 被 `region_id == stripe-{aircraft}` 锁死。这里建议由一个共享 executor 同时拥有任务生命周期、观测空间 track 和实际飞行意图；交通只决定已选任务何时安全执行，不代替 actor 选目标。保留一个 240 Hz 物理 / 120 Hz 控制的四机 world、每次积分的硬审计、公开任务 API 与独立 grader。此候选没有改生产代码、现有证据或真实研究 state，未读取另一候选；比较身份是同一继承模型的独立上下文。

## Usage (caller's view)

先写调用者用法，再写出的 `sketch.py` 包含三个实际调用点：

1. `assign_survey(drone=0, region="stripe-3", duration=80, expected_version=7)` 经现有工具边界解析为 `SurveyTask`，调用 `executor.submit`。返回 queued ACK；四个公开 region 一律按地图 ID 查询，跟飞机编号无关。
2. 物理 worker 每 tick 把四机自身导航、已在本机可用的传感包、已经发生的公开事件交给 `executor.advance`；将其四个控制目标一次性交给 `flight.step`。planner 无回复时仍然飞已预留的路径。
3. `observe_track` 提交已观察 handle；`handoff_observation` 带 source/destination 两个版本原子提交。actor 只读任务进度、真实到达的观测与已有报告，不操作航点、bank、交通槽或内部 filter。

Actor envelopes / 工具名不变；新增公开进度字段如 phase、work_started_at、last_measured_at、gaps 需要在开发协议冻结时登记。建议 duration 表示到达并开始有效作业后的连续墙钟区间：survey 在实际捕获目标 capsule 后开始，observe 在首个新的 qualified 本地样本后开始；transit/acquire 单独计时，受 90 s acquisition 上限、任务 deadline 和退出航程约束。这是相对当前“切换模式即开始 duration”的明确语义变化，不能不说明就沿用旧结果。inspect 的 90 s 失败期限可保留从实际 running 开始计；无法在期限内到达的任务应公开失败。

## Shape

### 最小深接口与状态所有权

生产调用路径为 `mission -> SharedFlightExecutor -> flight/control`，路径生成是 executor 私有 worker。公开接口只有 `submit`、每物理控制 tick 的 `advance`、`public_jobs`，隐藏路线生成、相位、观测滤波、交通准入与失败处理，符合 interface-depth；不把 actor wire dict 或 PyFlyt 对象穿过接口，符合 boundary-discipline。

| 状态 | 唯一写者 | 其他模块如何使用 |
|---|---|---|
| 隐藏实体、RNG、可见性射线、私有 provenance | mission 的 sensor/evaluator 边界 | executor 只收噪声、丢包后且本地已经可用的观测域数据 |
| 原始导航、controller 积分器、真实距离、240 Hz 硬审计/exit dwell | FlightWorld/control | executor 收不可变四机导航 frame，发期望 course/height/speed |
| JobBook、versions、active job、退休、取消和 phase | SharedFlightExecutor | mission 解析 RPC 后仅提交 typed task，不再直接 transition 或写 mode |
| candidate handle、因果滤波历史、关联歧义、观测 gap | executor 私有 per-aircraft/candidate state | sensor 接收相机 directive；mission 接收一次性的 evidence binding |
| 已提交四机航迹、工作段/回退段、交通 revision | executor 的单线程 reservation owner | 私有 planner 只有不可变快照和 proposal，不可写 world 或提交自己 |
| 最终 immutable Observation/Evidence、报告、上传到达时间、评分 | mission/protocol/scoring | executor 不收 identity_correct、真 class、实体 ID、future、grader 回调 |

`LocalDetection` 在 sensor 边界只带 opaque detection ID、噪声后的坐标/协方差、测得类别/类型、质量、测量时刻、本地可用时刻、相机/机体位置、曝光 job/version。executor 做观测空间关联后输出 `EvidenceBinding`；mission 把 binding 与 evaluator 私有的 detection→entity provenance 合并，**一次**封装不可变 Evidence。取消后已曝光的包仍可按原冻结绑定正常到达，不能重新绑定到新 job 或触发取消任务完成。若不希望本轮迁移 evidence 构造，可暂保留 mission 的观测空间 associator，但必须只把同样 stripped Observation 交给 executor，不能同时保留第二套 handle 关联规则。

本地可用与上传到达是两条时钟。0.5 s ground transport 不阻止机载跟踪；本机 packet 真正产生且已可用后，最早下一个控制 tick 才参与导航。远端机的历史仅能按已实际可用的交接/通信通道转交；不能访问尚在另一飞机传输队列的包。工具仍只允许 actor 已经收到过的 handle。`FlightWorld` 不再公开可写的 `modes/routes/headings`；每帧接收四个目标和独立的 exit audit directive。用 AST 边界检查禁止执行器导入 scenario/mission/scoring 或私有 Evidence；Python 的下划线本身不构成保护。

### 任意 stripe 的真实有界转移

先保留已验证 capsule：半长 250 m、半径 80 m。每个公开 stripe 定义 capsule 及其底边所有向东切点，任何 aircraft 都可以选择它。飞机继续绕当前已预留航线飞到一个捕获门，然后接圆弧—直线—圆弧；不从任意 heading 硬切新目标。对底边相同 x 的两个门，下面已有一个构造性 witness，无需先引入依赖库：

- 目标 y 更高：radius90 的左转 90°，向北直飞 |Δy|，再左转 270°。
- 目标 y 更低：左转 270°，向南直飞 |Δy|，再左转 90°。
- 全部 12 个有向 stripe 对的长度为 `2π×90+|Δy|`，最远 1165.49 m / 58.27 s，起止均向东。若用 x=-250 门，解析包络是 x∈[-340,-160]、y∈[-380,400]，距围栏至少 150 m。任一底边门可把 x 包络平移，需重新查 fence。

当前 capsule 理想一圈 75.13 s，等门是持续真实飞行。选择不同底边 x/到达相位比“总等下一整圈”更有调度能力。`probe_geometry.py` 用实际连续理想圆弧构造了四机 `0→1→2→3→0`：前三机 t=0 起飞入转移段，第四机真实沿底边飞 8 s 再入转移段；理想连续 3D 间距保守下界 30.657 m，大于 `15+6+6=27 m`。同期同步转移只有 16 m；单等整圈也只有 19.713 m，不能接纳。这个正例只证明交通方案不等同于禁止跨 stripe；未证明真实控制后的 6 m 管成立。

任意 observe/inspect 位置的到达、取消后恢复、退出恢复，用从当前预测捕获姿态到目标 orbit/capsule 切点的有界曲率路径。实现先枚举同向圆的外切连接和必要的异向/三弧连接，裁掉触 fence 的候选；再加 bank-rate 连续的 roll-in/out 与 climb 段并重新求时间和包络。若解析候选无解，有限 heading/bank lattice 可作为同一私有 planner 的补充，不新增 actor 接口。不能把未平滑的 Dubins 曲率跳变直接当可飞轨迹。本候选 probe 是自行写的圆弧几何，无复制路径库代码；后续若复用库，实施前另核验官方来源及 license。

### 跟随因果估计，而不是预订未知目标未来

观测 filter 用测量时刻与测量协方差维护 `[x,y,vx,vy]`，对乱序/重复 packet 幂等处理。只用 past samples，预测显式标为 prediction，绝不生成新 evidence。关联用 innovation gate；多个候选同时通过且不可判别时记 ambiguous gap，不选真实 entity ID 解歧义。检测更新可以立刻影响 gimbal aim，但新的飞机中心/路径必须经过交通准入，不能每次估计更新瞬间平移 orbit。

观察期飞机沿围绕短时预测中心的 circle/racetrack 飞行，预测只决定飞机**自己要执行的已知路径**。外部目标不执行预约，不假定它未来按模型运动。每 0.5 s 新本地样本或必要导航事件产生 proposal；旧路径保留到下一捕获门才切换。实际目标改变时，先在当前已预留路径上瞄准新因果位置，无法看见则如实形成 gap，再请求新的可飞工作段。姿态/中心移动/airspeed 的期望变化都必须受 roll-rate 与加速度约束。

camera 最大地面离轴角 50° 是持续观察的真实约束，而不是 20° inspect FOV 单独能解决的问题。稳态需要存在半径区间：

`max(80, R_bank, R_roll/accel) ≤ R ≤ (h_min-0.3)tan(50°) - track_uncertainty - flight_tube - 0.3`

右侧第二个 0.3 m 是旋转后的 body optical offset 水平范数保守界。uniform wind 与匀速目标下，`U=|wind-v_center|`，固定 airspeed20 的最坏向心加速度是 `(20+U)^2/R`；只写 `20²/R` 或 `20(20+U)/R` 会漏掉 crab angle 随圆周变化的项。R_bank 应按此、实际速度误差和可用 bank 余量计算，不能只给理想 80 m orbit。

小 probe 给出两种**待验证条件**：U≤1.5 m/s 时 R95、command AGL96、垂直误差≤2 m、飞机水平管≤6 m、track uncertainty≤6 m，理想 bank26.38°、可见性剩余约4.37 m。U=3.5 m/s（2 m/s 逆风 + 1.5 m/s 目标速度的资格角点）时可用 R106、AGL98，但需要垂直误差≤1 m、水平管≤4 m、track uncertainty≤4 m；理想 bank27.97°，可见性只剩 **0.943 m**。两者都未计目标加速度、实际速度误差、转弯过渡，所以后一个尤其容易被物理 probe 否证。不是宣称当前 controller 已满足。

选择 R/高度/相位由这个可行区间和交通共同决定。区间空时公开 `observation_geometry_unavailable`，不能硬开 camera 然后声称 continuous tracking。目标 uncertainty 超阈或最后测量年龄超 3 s 时，继续按已预留轨迹 coast，并将相机切到公开记录的广角 reacquire；只在新真实测量满足关联门后回到窄角观察。广角的实际质量若合格仍可正常形成报告，不把 inspect 设为唯一可计分路径。把观察速度降为已声明15..25包线内的18 m/s是可单独比较的补救方案，但本候选主设计仍按20 nominal；18也必须新资格，不能用它回写既有20 m/s证据。

### 交通准入、回退与时钟

每个 continuation 是“不可中断前缀 + 有限工作段 + **已经预留**的恢复段 + 安全 hold”。恢复优先到原始独立 home capsule，home 与 exit 的地理身份可以固定，**工作 stripe 不固定**。四个 home capsule 之间至少40 m水平间隔。拒绝/排队时飞机飞已有 continuation；取消时逻辑 job 立即终止，飞机沿不可中断前缀接已预留恢复段。新的版本只可替换后缀。退出同样先飞到本机声明的 exit circle，再由240 Hz实际位置/course获得20 s dwell，不能因 job kind=exit 就算完成。

planner 在独立进程中耗时计算，physics worker 永不 wait。proposal 绑定 job versions、track generation、reservation revision、future capture tick 与最晚 commit tick；owner 只接纳仍新鲜的完整 fleet revision，不能先改一架再改另一架。固定 FIFO + 安全退出优先只是交通优先级；不改 actor 的 region/handle，也不替 actor 续 observe/改任务价值。过期、冲突或数值失败保留旧路径并记公开原因，超过可作业截止线则 terminal failed。不能让 queued 无限静默等到600 s。

每个候选同时查整个四机 swept tube，而非 waypoint 距离：围栏要包含管的全部位置、AGL管必须落在60..100、最小3D间距要大于15+两管+数值界。可用解析极值/段间最近距离，或者采样加速度/速度 Lipschitz 保守界；一般27 m/s ground speed界时50 ms采样需另加至少1.35 m的 pair 距离插值余量。本 probe 是20 m/s理想点质点，补的是1 m；不能复制到真实25+2上界。真实输出保留240 Hz硬审计作最终否证。

时间占位还需要沿航迹相位误差控制或显式扩大的时间管；当前只有 heading/altitude/speed 的 controller 不自动给出6 m长期位置界。实施应给 `FlightTarget.airspeed` 加**经过资格的**小幅相位纠偏并同时重查转弯 bank，或在准入中传播真实相位不确定性。不能拿恒定 nominal20 的开环 ETA 假装永远精确。尾部 home 的空间互斥减少长期相位依赖；新工作段只在下一次重规划前的有限窗口接受更紧的管。

恢复段预留的是飞机的可执行计划，不要求未知目标未来。每次新测量只替换允许变更的后缀。观察更新被交通拒绝时相机可以继续因果指向，飞机保持旧路径；见不到的部分就是丢失，而不是偷偷插入无预约追踪。若真实飞机离开资格 tube，必须记录 envelope_breach 并执行已经安排的恢复；这仍可能导致 H=0，不能声称回退能神奇修复已经发生的危险。当前目标是实测风包线下可否证的 executor，不是未知扰动的全未来形式安全证明。

准入还查实际已飞距离 + 工作/恢复预测距离 +1000m exit reserve，以及 deadline 前实际可达的退出时间。接受后更差估计或障碍使任务失败也保留原承诺的退出路径。任务完成不允许消耗最后保底航程后再临时想办法退出。

### 作业后置条件、交接和 race

- survey：真实捕获所请求 capsule，duration作业时钟结束且完成至少一个真实 pass/lap；既有 coverage 仍由实际camera footprint与射线计，不因路线圈数直接加分。
- inspect：来自同一 observed handle 的两次实际合格新观测，飞机视点相距≥20 m；隐藏 class/identity/localization correctness 不参与 job 成功。
- observe：首次合格新观测开始 monitor interval，时间不会因丢失/断联暂停。保留所有3 s monitor bins和每次gap，duration结束时若有空bin则failed并返回partial progress；可继续 reacquire直到区间结束，不能补时抹掉漏看的部分。这个 job window 不改变 grader 从隐藏出现时刻起算的固定180 s窗口和分母。6 s grace 是报告上传宽限，不能用来替换缺失测量。
- handoff：destination在真实飞到可看位置后获得新的qualified本地观测，并且两机后续路径原子接纳，才结束source的观察；source不因ACK而放弃。destination的首帧可以共享冻结的过去source估计，但不能共享未来或未实际可用包。失败/取消destination时source继续自己的既有有效任务，直到其原区间/退出规则结束。source退休、超时或失测导致的缺口必须记录。
- cancel/stale/event：cancel提升version，未提交proposal失效；终态任务不可被迟到planner/packet复活。360 s退休进入同一个ExitTask路径，不开旁路写flight mode。180 s priority只更新公开authority/context，既有有效任务不因priority换值被隐式重分配。

job succeed 与 ground 收齐 evidence 是两件事；公开 progress 可带 `evidence_pending` / acquired evidence IDs已产生但未到达的状态，不能通过重复inspect补偿0.5 s传输延迟。root已经在reference策略中处理confirmed等待；本候选不修改该策略或旧600 s结果。末尾成功判据仍由实际上传到达与独立grading决定。

## Synthesis decision

这是独立候选，未做跨候选裁决；建议以本候选的“单写者任务/飞行意图 + 原子四机预留 + 已预留回退 + 局部因果tracking”为一个完整可评判单元，不能只摘走轨迹目标函数而丢掉准入和回退。

## Tradeoffs accepted

- 接受显式 transit/acquire 与可能排队/失败，换取真实跨区与如实暴露的观察缺口；不承诺任意时刻任意任务都能被四机无条件接纳。
- 接受较复杂的私有 planner，换取 actor 保持六种任务工具且不知道路径/traffic细节。几何正例不等于控制资格；必须先测出真实tube，不能用愿望定tube。
- 接受每次任务预留回退和可能回home的航程开销，换取没有planner回复/取消时不依赖临时生成安全路。空闲home固定只为交通兜底，所有public stripe仍可工作。
- 接受一个物理worker为所有状态单写者，换取不需要锁跨两个job版本、两个handoff参与者和四条路径；planner只读快照不会争写。

## Alternatives considered

**结构不同的可行替代：公开地图上的有向交通网络 + block token。** 固定搭建宽裕的转弯节点、单向进出走廊、capsule切向站位及若干互斥观察区；aircraft独立执行不可逆的 arc/straight edge，中央交通只授予进入下一block的token。拿不到token就真实绕预留holding loop，不能停在节点；有足够空余空间时观察区允许source/destination两架处于相隔半圈的状态完成handoff。所有12个region转移仍通过网络可达，观察区根据因果估计移动时必须重新占用相邻block，不把目标固定到stripe。它把全机连续时间预约换成空间互斥，安全边界与失联恢复更易做，但频繁移动的目标可能跨block，dense情况下吞吐与观察可达性更差，且维护合法拓扑/大半径holding空间的地图成本高。actor表面仍简单，但controller/caller若要知道token和block会变浅，因此必须同样藏在executor里。它在保守接纳范围内可行，不是给无holding余地的飞机“在路口等着”。

**不选分散的逐机追踪 + 最近邻避碰。** 四机独立改orbit中心、接近时切“avoid”虽代码少，却没有共同剩余路径/回退不变量，无法把15m与80m转弯同时保证；两机同时避向边界时没有可实现完成条件。这不是本候选的备用路径。

## Open questions and risks

- 20 nominal、0/2 m/s风、上下限60..100下，旧controller加capture/相位纠偏能否真正在移动orbit中保持4~6m水平管及1~2m高度管？尤其full-corner可见性只剩0.943m，是否会被速度/roll transient直接否证？这应由下面最先的物理probe回答，而不是人确认一个常量。
- 原恒速capsule资格是否支持新的时间槽控制，还是必须显式传播phase uncertainty？若不能满足管，就扩大管重新准入；若观察几何区间消失，应降低承诺或另测18m/s，不能直接放宽hard约束。
- 观察目标近障碍、相邻candidate交叉或 camera degraded 时，现有射线/关联是否会暴露真实长期空bin？可以失败，不能借隐藏身份保持看似连续的track。
- 在有限planner预算和恢复预留下，非空正例是否能扩展到四机同stripe、相邻动态轨迹、连续handoff/取消？全拒绝不是工程完成；需同时有接纳成功案例和真正不可行的明确拒绝案例。

## Next implementation step

先实现只有可观测输入的单机路径捕获/移动orbit小probe，并测出实际空间/相位误差和可见率，再实现reservation owner；不要先改actor策略掩盖executor不可飞问题。

## 已执行检查与最小否证计划

已执行：`python3 paper/research/20261104-inspection/planning/execution-architecture/candidate_a/probe_geometry.py`（只用stdlib，无PyFlyt/模型/真实state），以及两个Python文件的内存compile和三个候选文件的尾随空白检查。12个stripe endpoint误差≤1.71e-13m，解析包络通过；四机几何反例与上述8s正例均复现。`python`在当前shell不存在，实际成功命令是`python3`。下列是**待实施/未跑**检查，不能引用为新qualification：

1. 最小P0：单机真实PyFlyt，90~120 s，每240Hz审计；注入预先声明的无身份因果测量流（不传未来），先测20m/s、R95/96m、再测R106/98m的逆风角点；保留2Hz/10%drop、0.5s地面延迟。检查实际bank/alt/speed/fence、相位误差、光学origin后50°可见率；任一tube失守或无非空R区间即否证当前参数，不直接跑600s。
2. 最小P1：四机实际world复现0→1→2→3→0几何正例，roll-ramp后重新排程；对比同步反例应被准入拒绝。真实最小距必须≥15m、所有240Hz审计无违规，每机必须进入另一个stripe并完成真实pass，不能只看ACK。
3. 最小P2：双机handoff，故意遮挡/断联3~8s、planner迟到、candidate交叉；source只有在destination新观测和双机预留成立后退出。日志应含缺失bin、旧version拒绝、取消后包正常到达但不复活任务；测量deadline与arrival deadline分开核对。
4. 边界fixture：路径入口姿态错误/大roll、最外stripe、重复包/乱序包、取消与退休同tick、range不足、任务来得过晚、planner完全停止、四机同时求同一观察区。断planner时physics仍推进，既定回退仍在审计；不可行必须public fail/queue到可见截止，不能固定回本机stripe假装执行成功。
5. 因果负例：给planning输入加隐藏entity/seed/future接口应结构检查失败；相同observable history、不同隐藏未来在分叉前得到相同控制/预留/状态；预测轨迹不得生成evidence。改动后重跑原控制家族回归与新模式资格，再考虑真实端到端600s和后续G2/G3/G4，不复用旧100/100给新模式背书。
