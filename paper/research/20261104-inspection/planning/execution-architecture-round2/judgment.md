# 第二轮独立 cross-judge

建议选 **B 的显式有限固定弧、带长度的曲率 ramp、相位门切向移心**作为最小实现 base。保留现有 TimedPath/RouteProgram、共享控制器、单 owner、异步规划与全程序预约。吸收 A 的两项契约：获取途中也检查并保留真实已见机会、允许必要续约；工作到恢复的资格入口覆盖完整自身状态及控制器记忆。**暂不引入 A 的低阶闭环 rollout 作为安全证书或第二套导航执行器。**

这不是 B 已物理可行的裁决。两者都有未闭合的物理接缝；首个放行条件是含真实 ramp 的初次获取、短 work、规划停止后自动进入已预约恢复的短物理否证。若失败，保留失败，不先铺开调度器、600 s 或新大矩阵。本评判是同一模型家族的独立上下文，不是跨模型一致性。

## 已核对的事实与证据边界

完整阅读了本目录 grounding、两份 rationale/sketch、两份 probe 源码及结果。A 的 JSON trace 共 550 行，已全量解析检查时刻、有限值及所有记录；没有复跑其物理 probe。B 的解析 probe 在禁止 write_text 的只读运行环境下复算，除 wall_seconds 外与保存结果完全一致。四个候选 Python 文件内存 compile 通过。未修改候选、生产源码或旧数据。

当前 execution/planning/paths/guidance/control/flight/mission/protocol/scoring 边界及最终 v11 manifests 已核对。两份 manifest 绑定相同契约，源码/依赖/契约未变标记均为真；全部 140 个 slot 文件的哈希与 manifest 一致。最终结果如下，不能继续引用 grounding 的中间计数。

| 已执行 v11 矩阵 | 完成数 | 合格数 | 含义 |
| --- | ---: | ---: | --- |
| g1-full.v2 五族基础控制 | 100/100 | 100/100 | 只覆盖所冻结基础控制族。 |
| executor-full.v1 transfer | 20/20 | 20/20 | 覆盖原冻结转场模式；不替新 ramp 或新观察背书。 |
| executor-full.v1 observe | 20/20 | **1/20** | 只有 root 1 通过，新执行器 G1 未通过。 |

observe 的 D 范围为 .033333..85。root 7、19 虽有 D=.833333/.827778，XY 最大误差为 6.331/7.027 m，仍失败。其余一些 roots 的 D 很低却没有 hard 或管违规；root 10 的 D=.033333、XY3.584 m，即不能把整个感知失败归因于飞控危险。全 40 个 executor slots 均无 hard 违规、每个物理及 tracking 审计数均为 86400。observe 最大 XY8.231 m、稳定 Z2.464 m，不能把安全飞行等同持续感知合格，也不能将这些原 6m/2m 失败重标为新 10m/3m 通过。

controlled next-tick G1 在物理步之间计算，不是 1×。旧 executor-realtime.v1 的 D=.052941、最大 lag=.101875>.1、XY13.895 m 仍是失败。精确 replay 证明可重建原执行，不证明时钟、控制或动态任务已通过。两候选均没有新的 180 s 原窗 D、四机新模式资格或真实 1×结果。

## 六项评分

0 表示缺失/冲突，1 表示有方向但核心机制待闭合，2 表示可实施且有明确修订或验证缺口，3 表示该项架构契约完整。分数是设计比较，不是 qualification。

| Rubric，0..3 | A | B | 独立判断 |
| --- | ---: | ---: | --- |
| 物理几何与感知可行性 | 2 | 2 | A 的同律从真实状态转向有一个有利单例，但长前缀预测、blend、恢复未验证，预测误差还有坐标缺陷。B 的必要几何保留非空候选，但 ramp、初捕获、窄轴误差、目标不确定及恢复均未验证。均不足以升级为物理通过。 |
| 因果隔离 | 3 | 3 | 两者均逐机、只读已可用历史，冻结导航段，不把预测变证据，也不以私有目标未来做恢复。实施仍需消费者、时刻、seed 和同历史负例。 |
| 实际任务能力与公平共享 | 2 | 2 | 都正面移除反复获取精确目标中心圆及获取中不更新的障碍，保留原任务/评分与共享执行。实际成功率、复杂交接及四机拥挤尚缺证据，B 的全 work 光学 screen 还与允许获取盲段冲突。 |
| 单 owner、取消与失败 | 3 | 3 | 两者保留唯一 JobBook/program owner、版本事务、旧占用、已预约恢复、实际样本 handoff。B program+cell 需同事务；A controller/predictor 快照不能成为新写者。 |
| 有界 1×实现及资格成本 | 1 | 2 | A 新增自身动力响应、同律积分器、入口集与时变管，且现有 1s误差无效；最多五候选并不能界定完整恢复成本。B 可复用现有有限路径和 follower，新增资格集中在少量 ramp/移心及接缝，但两次全 fleet 检查和 .5s lead 仍需真实时限验证。 |
| 小而可维护的接口 | 2 | 2 | A 表面小但引入第二种 program law 与响应模型。B 复用现有表示较小，但请求缺资源和完整自身状态，cell 的 handle 也应去除任务权限含义；不能因改动模块数少直接给满分。 |
| 总计 | **13/18** | **14/18** | B 胜在最小增量及更小的新增模型承诺，不因解析图好看、候选自评或文案长度。 |

两个方案共享的单 owner/异步骨架已经存在，不应为此再实现一套。若 B 的真实初捕获与有限移心反复被否证，再决定是否值得独立校准 A 的闭环 predictor；不能把两个尚未资格的执行分支同时放进 production。

## 三项不能忽略的否证

**A 的 55 s 结果不能给 rollout 定管。** 原单机实际 13200 次物理审计、无 hard 违规、简化光学 visible_fraction=1、最大新测量间隔1s，可以保留为一个有利初态下的真实场律开发结果。其预测器每1s用实际位置重新初始化，且把机体系 state[2] 减世界系风；原 3.139128 m 既不是校准模型误差上限，也不是6s work或完整恢复的误差界。即使修复坐标，也不能从每秒重新归零推导连续6s、回家或600s的管。该运行没有 Z predictor、fleet admission、恢复、ray、完整任务证据或异步时延；真实 FlightWorld 控制/audit 使用独立世界速度，不应把预测器缺陷扩大成“所有物理数据均无效”，也不能反过来用这些物理摘要替预测器免责。

**B 的解析余量不覆盖 ramp 与初始 seed。** 原理想输入得到 phase-gated 最大距离91.890 m，叠10m自身管、4m目标误差、.3m水平光心和.1075m离散余量后106.298 m，对应 reach109.761 m，仅剩约3.463 m。完整固定圆同输入132.819 m失败；1.5s gate 的106.841 m通过必要 screen，3.5s gate 的134.691 m失败。这些都没有 ramp。2s ramp 有真实长度、改变终点及下一圆心；以恒定地速 g 作量级估计，曲率0→1/R的平滑 ramp 会把切圆中心向前推约 g×2/2，在16..20m/s时约16..20m，已经大于3.463m余量。实际有风积分必须重新计算，不能机械把这个量加到最终目标距离，也不能沿用无 ramp 的 pass。

更关键的是当前 LocalTrack.first 从 survey seed 初始化，单轴 sigma=4m，后续正常 inspect 才是1m。单个二维独立各向同性 sigma4 高斯测量落在4m半径内的概率约39.35%，95%半径约9.791m；sigma1 的95%半径约2.448m。95%只是这里展示量级的数学例子，不是新增 benchmark 门槛。把 B 原4m替换为9.791m，原无 ramp 理想 screen 的剩余量已变成约−2.328m。不能用正常 inspect 的误差给首个 survey seed、stale wide reacquisition 或 degraded样本统一设4m。滤波后的误差还取决于样本模式/协方差、时间间隔、创新和运动失配，Gaussian没有确定有限支持。当前 LocalTrack/ObservedTrack 没保存这些不确定来源，需窄补接口与校准；不能默默把4m当成已知真值包络。

**共同周期只证明名义周期，不证明物理相位自由。** 当前 fleet_clearance 对各条名义尾环同周期、所有有限前缀结束后再查一共同周期，数学上成立于固定时序参考。它不自动证明真实飞机长期保持相位管，更不允许新参考每次重锚到实际位置后清零误差。若改用 phase-free 的尾环安全论证，必须按每机完整 R80 home 环带和实测径向误差检查，且检查所有前缀/恢复与别人可能占据的整个尾环。四个 home 中心相距200m，理想环间40m，双10m径向管仍留20m，足以满足15m，但这只覆盖已经合法捕获各自 home 的 tail-tail，不能覆盖进入/穿过该环的有限恢复。不能借 phase-free 几何重定义旧失败的 tracking 结果。

## 最小 base、两项 graft 与必要接口修订

B 的实现先限制为 KEEP、左右切弧、有限相位门、有限切向 runout、R80/R90，以及有长度的曲率 ramp。重用现有 follower、RouteProgram、完整恢复和同一个 fleet clearance 入口。保留当前 actor 工具、camera/transport/protocol/scoring 的职责；无新公开航点、半径、预测器或 traffic API。保留每机一个纯物理 cell，在 inspect→observe 后复用它；新任务不能因为目标不在圆心而强制 CSC/C2 获取。CSC 仍可服务于既有转场/回家，不作为每次 observe 的必经接近。

从 A 移植以下两项，不引入其整套动力模型。

1. **真实已见机会与获取中续约。** 不等待 planned_service_from 或固定2s试相机结束才请求必要转弯。已有真实 qualified frame 立即按原 postcondition 记获取/服务；相机继续用合法本机新历史，导航只在已准入 future gate 改变。优先保留仍有用的已承诺弧，排序包括首个可用视图、短窗盲区与切换成本。取消 mandatory 2s delay 不等于无条件即时大转弯。
2. **完整自身入口域。** 未来 gate 与 work→fallback 的可接收域覆盖位置、世界地速/独立空速、course/air-heading、roll与角速率、Z/VZ/pitch及控制器积分器。几何切线连续、bank名义连续或“实际距名义≤10m”都不能单独证明下一段能捕获。保留 read-only snapshot 和适用域检查，不复位真实状态/PID，不让 predictor 修改 autopilot。

实施接口需要明确补齐以下数据与规则。

- `_EditRequest.base` 当前复用的 PlanRequest 没有 actual spent/remaining range；承诺“实际已飞+未来+1000m”不能凭空计算。增加四机实际剩余资源及可信导航时刻。`_OwnNavigation` 增加必要完整状态/积分器快照，明确世界/机体坐标，避免 A 已暴露的混系错误。保持一个不可变快照，不新增状态写者。
- 目标 forecast 保留 consumer、来源 IDs、local/delivered availability、sensor mode/covariance、样本历史或足够统计、age/innovation及误差增长说明。A 的 anchor 和 B 的 ObservedTrack 都不能只凭位置/速度及4m常量替代该信息。navigation forecast 可指导候选，不是观测证据、安全管或目标未来真值。
- 所有 accepted work 包含**在接受时就已预约**的目标无关恢复和原home尾环；包括首个程序、KEEP、初捕获与所有 job kinds。已承诺有限 work 到期即可执行恢复，无需 planner 再回复。新增 ramp 必须用于所有实际发生的 bank/速度/高度接缝，包括 work→recovery→tail；复用 recovery 函数不自动继承新入口状态资格。
- 若采用10m水平候选管，则所有 survey/inspect/observe/exit/transfer/恢复/旧占用一起使用连续名义间距至少35m、连续 nominal fence≤540m，不能只给 observe 换数字。当前 fleet_clearance 已扣27×dt的间距补偿、加13.5×dt的围栏补偿；dt=.1时，要检查输出 lower≥35、upper≤540，不能只把采样点阈值设35/540。若管随年龄变化，逐段使用对应双方界；任何混用都需显式更保守界。
- 10m/3m、2s ramp、4..12s work、≤3s gate、.5s lead均未资格。96±3在数值上落在硬AGL内，不代表起始过渡/接缝都满足它。扩大管必须前瞻重测并统一保守准入，不能以旧峰值小于10/3就重标旧矩阵。若扩大后所有需要任务均被拒绝，平台能力仍失败。
- `_Cell` 只是飞行几何/相位状态，不是任务、候选身份或报告权限。建议移除 navigation_handle 的授权含义，绑定仍在当前 JobBook/token/本机历史。cell随program/revision原子替换；旧报告 route 在曝光时冻结，续用物理圆不允许旧任务自动向新 report 发布。
- 物理 hard screen 是所有段的硬准入。B step4“EVERY point有限work都须光学可见”与step3允许真实获取盲段冲突，应先统一语义。获取阶段可对明确有限盲段排序、报告 gap和90s超时，仍须完整物理准入；不能因为初始未见就拒绝一切可能重获的路径，也不能把预测可见当作服务成功。observe的实际缺测原样进入job与grader。

## 需要先否证的物理与因果接缝

首个短 fixture 必须从合法、抖动的真实20m/s初态及真实首个survey seed开始，用现有控制器实际减速/切弧。不要从理想圆上的18m/s有利状态起跑来验证初捕获。测试只需要小而有针对性的开发配置，不等新大矩阵，也不需要主研究test roots。

1. **含 ramp 的初捕获。** 在2m/s不利风下构造合法历史，让已经看到的候选处于当前航向侧方。用真实2Hz窄/宽相机、既有噪声与miss/ray规则喂本机filter；规划器不能读sensor/auditor的真值。包含首个sigma4 seed、后续sigma1视图和真实.5s seed上传。实际记录ramp结束位姿/新圆心、首个qualified视图、盲区与原3s bins；无ramp的1.5/3.5s图不作通过证据。若有限族不能挽回实际视野，拒绝这组primitive参数，不能延后隐藏180s开始或删前几格。
2. **短 work→完整恢复。** 至少在一个非零roll/roll-rate的弧/移心相位永久停止planner，使已预约work自然到期，真实进入return和home并完成20s dwell。逐240Hz审计hard、管、phase、入口状态与PID。短 fixture若不足以飞完返回/dwell，明确记接缝尚未覆盖，不能用截断前的安全认定完整fallback；下一小fixture可专门验证该段，不先跑600s。
3. **取消、退休与handoff事务。** 在未来gate前后、ramp中和接收机新样本同tick发取消/换版本/退役，确认旧proposal失效但旧飞行占用保留。取消可结束采样权限，不能修改冻结导航中心；提前退出需要新的完整证书。接收失败或planner死亡时source只按自己的原期限继续；只有receiver实际新的合格样本及精确旧source token才能完成handoff。保留现有四机同帧全部曝光后才执行后置条件的顺序与曝光时报告绑定。
4. **未知目标转弯与失测。** 两个fixture在某时刻前拥有完全相同的本机历史和导航，之后目标转向/身份邻近不同；分叉前proposal/guidance必须一致。在下一真实帧前不许利用真未来修正。转弯后的forecast创新应增大风险/缩短work或明确gap，不能把LocalTrack的4m/s clamp当真速度界，也不能把6s上传宽限当失测补时。宽帧可见不代表quality≥.8，健康survey需约43.30°以内；低质帧可更新导航但不计service。
5. **四机与真实异步调度。** 一机首捕获成功后才做同一world的双观察/交接与另外两机survey/退出。既有占用均保留、至少有非空成功任务；尾环用明确的相位或保守环带证明。新候选改为35m后，旧transfer20/20不能自动沿用。最后做真实1×慢/晚/停止planner与轮转续约，证明四机更新不饥饿、无blocking/unbounded解析、clock maxlag仍≤.1；controlled next-tick通过不代替此项。

所有新profile、源、队列预算与失败规则应先冻结，再运行前瞻资格。A最多五候选、B≤16且只对最优两项做fleet检查都不构成时限证明；B若两项均因交通失败，可以公开bounded-library failure，不能称全局不可行，也不能隐藏全拒绝的能力失败。优先级只解决交通/续约，不替actor改派目标。

## 不变的验收与拒绝项

保留四机、240Hz物理/120Hz控制、AGL60..100、空速15..25、45°hard bank及≤30°控制目标、15m实际间距、XY±550、600s主截止、真实退出20s和1000m余量。保留原隐藏appearance起算180s、全部60个3s bins、6s实际到达宽限、全部实体/格分母、定位/身份/类别及原D/G1门槛。局部job服务起点和导航预测不能替换这些grader定义。

不采纳A的错误坐标预测误差作为管、不采纳B的无ramp解析成功作为物理资格、不引入通用导航框架、不双轨保留旧目标CSC获取和新观察器、不借新误差管回写旧失败。小fixture通过后仍须新G1完整prospective矩阵与独立复核、真实1×及G2–G4；原八配置/19项科学义务与人工领域复核状态不变。本裁决只支持最小实现与否证顺序，未放行正式实验。
