# Secondary analysis consumer：独立有限设计审查 v1

结论：固定分母、parent 配对与纯描述性 bootstrap 的主结构可实施，未见需改变原19义务或 global6 的理由；但草案尚不宜直接标成已冻结接受。实施前须明确 **D1 采用见证的双路径与同刻规则、D2 NA/删失的汇总定义、D3 随机流参数**。以下是前瞻的具体建议，不是实际 secondary 结果、实现批准或新的统计检验。审查者 `/root/statistics_design_review` 为独立上下文、同模型家族，无实现职责；不授 literature/experiment/G4/main/human 批准。

Subject 为 `planning/secondary-analysis-consumer.v1.md`，SHA256 `aa1831399dac53c18d54a22aa9b8f7a2fac42827a914e5d8b937e96f852778d0`。证据 binding 保存12个明确输入的实际 SHA、阅读范围及微小标准库检查；没有全树扫描、pytest、NumPy draws、生产 import、模型/物理、native CLI 或 owner 操作。既有 accepted-design 83f…、主分析40source/22tests/408+15和 read-only P2 的状态仅作父 Agent 给定背景，本轮未重审或复跑，不据此扩大批准。

## 1. 固定单位、对照与 bootstrap

12个预定 main L3 parents（ID/OOD各00–05）是全部独立单位，不是新增12 roots，更不是504独立样本。clean 为12×18=216额外 flight，ablation 为12×3arm×3repeat=108，latency 为12×5arm×3repeat=180。clean join 原primary216槽；另两轨各 join 同一组原primary B4-M1 36槽，不把重复引用同一基线计成新飞行或额外重复。fresh-zero36次属于latency180，不以原main零注入数据替换或择优。delay-minus-fresh-zero、primary参照、ablation-minus-primary 的方向和命名须在输出中固定；event-minus-clean 按草案保持正负含义，不自动称“损失必为负”。

建议先按 parent/condition/arm 聚合预定 repeats（经典1、LLM3），然后构造完整 parent 向量：同 parent 的所有条件、arms、两轨原/重评分量、基线及辅助计数同时携带。按域从6 parents有放回取6个索引；一套索引用于同轨所有差值与组件。先各域 root 均值；合并12只计算两域各1/2的宏均值。不可按飞机、target、时间格、repeat或arm分别重采样；也不可为各对比独立抽基线而破坏共同向量。策略失败的冻结Y=0保留；infra重试不是额外repeat；缺槽、unknown、非terminal或未锁定基线应阻断该必需分析，不删失为“低质量数据”或随意补0。

99999 draws、percentile nominal95、linear quantile可沿用原计划的描述角色，但6 root/域很小，不能以区间排除0声称显著、等效或新的确认结论；常量差导致零宽区间也不能升级为确定性保证。SR先repeat均值再parent聚类；成本同时保留已接纳flight和全部known attempts的资源，未计量monetary/energy不填0。

**D3（冻结前必补）：**可采用明确提案 `SeedSequence([domain_seed,3,track_salt])` + PCG64，ID/OOD沿用2026110401/2026110402，salt提案clean=1、ablation=2、latency=3。每域/轨固定parent canonical顺序及(99999,6)索引矩阵的生成布局、整数dtype和NumPy版本；多指标复用同一矩阵，不能看到结果再选salt或按condition改流。这些常数是本审查建议，尚未登记为接受值；本轮没有实际生成任何随机数。它们区分计算随机流，不证明真实跨root输出独立。

## 2. D1：job采用并不只有 program_committed 一条路径

原计划§9要求合法新/修改job为running，航迹/观察配置已实际采用；不要求succeeded。旧 `packet_response` 取合格观测的 **measured_at**，还要求到达deadline、匹配job/规则等，并可能结合静态报告改进；它是信息取得/交付约束下的响应诊断，绝不能改名成首次job采用，也不能改用ACK或plan_response替代。

现有源支持两种不同的实际执行见证：

1. **新program的控制采用。** `execution.advance` 将program写入该drone的 `self.programs`、必要时把job转running并记 `program_committed`；同一次advance末尾对当前program调用target，`mission._advance_jobs→mission.step→FlightWorld.step` 把返回目标送入controller/set_setpoint，随后真正world.step。源调用闭合，故在完整合格receipt与冻结单写者源下，可以有条件地推断采用发生于该控制tick。
2. **旧安全航迹上的实际观察配置/采集。** `execution.local_measurement` 明确允许inspect/observe在已承诺continuation上通过本地实际acquisition由queued→running，理由为 `actual local acquisition on committed flight`。该分支不要求出现新的program_committed。`mission._sample` 先按当前active job构造全部相机帧，再逐条调用local_measurement；job transition、service_started_at及相应采样证据应作为第二路径。只搜索commit会把这条合法路径误判为未响应。此处依据真实可达源码分支，没有运行或声称观察到新flight反例。

240Hz tracking仅保存t、每机横向误差、z及上下界，**不含逐行job_id、program_sha或实际setpoint**。它能证明实际物理步与当时program跟踪约束，不能单独证明哪项job首次采用，更不能独立重放控制量。宜明确称“冻结调用链+关联审计支持的仿真时刻采用推断”，不能包装成逐控制输入的直接记录或硬件attestation。

最小可实现选择器须先定义每event的公开受影响规则/责任workload，并核新/修改job的接受tool、job_id/version/drone、合法rule_id/task_card_version、参数及创建时已有的因果候选证据；自动refresh旧job不自动等于事件后的新改派。commit分支关联plan request/response/commit的serial/hash和job，并按同次advance的真实顺序核job在目标输出时仍running，再要求commit后实际完成物理步。一个commit后同次advance立即因90s条件转failed的路径，在浮点时间上可有started_at==finished_at==commit_at；仅用 `started_at<=commit<=finished_at` 会误判，必须读transition顺序与冻结源阶段。若没有后续实际步（例如截止处marker），不计控制采用。

local-acquisition分支应核running transition的明确原因与同帧实际采样，关联job候选/机号/规则及本地服务起点，不能用最终status必须running来排除后来正常succeeded的job。优先使用原始本地采样及transition，不能强迫它等到地面上传才把采用计时；断链只改变后续信息可见时刻。若某receipt现存字段不足以唯一关联，返回 `adoption_not_identifiable_from_trace`，不得伪造时刻，亦不得将证据缺口当策略未响应右删失。必要时未来正式源只需增加有source绑定的首次“job/config实际应用”紧凑marker（job/version/drone/program或config、phase、physics index）；不必记全setpoint，也不修改当前在途104源。本轮不声称现有任何具体flight已满足该关联。

同刻规则须在实现前冻结：以既定事件/公开更新生效时刻τ为原点，同时另保留actor实际获知时间（如有），两者不混称；同刻合法响应可为0，但须由phase/规则版本证明在生效后，旧version或生效前动作不算。当前120Hz控制/240Hz物理的频率须逐receipt核验；可把commit控制tick作为采用时间t，并保存首个实际physics hook t+1/240及完整控制步末端t+1/120作验证。不得把hook时间、10Hz pose、墙钟或首次采集/报告时刻混成同一个精确时刻，也不声称亚tick首物理效果。local-acquisition按实际采样/transition的仿真tick计，单列其通常2Hz采样分辨率。跨journal同刻若无法按冻结调用顺序唯一消歧，保留不可识别，不按JSON排序猜先后。

## 3. D2：NA、删失与恢复的最小诚实摘要

现有helper的 `already_satisfied` 只表示pre-event质量达到静态.6或动态.5的操作性阈值。动态仍有未来原3s格和交接要求；静态达到.6也未必代表全部更高质量/优先级需求完成。应命名 `pre_event_threshold_met`，与原计划严格“无需改派”区分。后者只有冻结的完整公开后验条件和裁判证据能证明，不能由这一个阈值或“没有发命令”推得。软优先级允许合理取舍；未见新job响应仅是该诊断事实，不自动判策略失败。不要用事后首次成功或隐藏未来真值决定eligible。

逐event×target原表必须保留：parent/domain/condition/arm/repeat，τ及类型/公开规则、pre-event证据与质量、eligible和各NA原因、deadline、followup H、响应/恢复见证类型、观测时间或删失、不可识别原因。not_yet_active、followup_ended、rule_unchanged、no_affected_active_workload、pre-threshold-met分开；NA不设为0或deadline。对eligible未达成记录U=H、censored=true，但U只表示观察截止，不是真实响应时间。trace缺失与policy未响应分开。

**建议最低摘要先固定为：**每域/方法/事件类别公开全分母、eligible、每类NA、有效见证、未响应/未恢复数及比例、H的范围/分布，并保留所有U/H/indicator行；若列成功者时间分布，只能标“已观测者的条件分布”，不得以其均值/中位数概括总体性能，也不能把mean(U)叫真实平均响应时间。不同方法的eligible可能受自身pre-event成绩/已有workload影响，条件分布不是天然相同的配对总体；须并列分母，不能只保留双方成功/双方eligible而不披露选择。

若还需要一个可比较的时间摘要，应在正式数据前另定少量固定τ*，对所有eligible报告“已知在τ*内响应”的比例下界及对提前删失者的上界：下界为已观测T<=τ*的数量/N，上界再加censored且H<τ*的数量/N；不是KM、不要求删失独立，也不将未知者硬算失败。可按完整root计数向量聚类重采样，明确其分母与目标是eligible行比例还是等root均值，两者不可互换。该摘要是可选的描述提案，未自动加入义务或冻结；无需为了交付本接口新增KM、log-rank、p或所谓显著性分析。

恢复保持原定义：静态候选起点s须有后续完整30s稳定且s+30<=deadline；同时保留recovery_start=s和confirmed_at=s+30，不能称在线在s已经知道恢复。动态使用10个连续原3s格、原格边界和6s到达宽限，保留按最新合法测量先筛选的错误格；不重置grid，不让同刻稍后的好版本擦去此前坏版本。短followup不足以确认稳定应右删失/说明原因，不能缩短30s或10格以增加成功。deadline由原target日历/原窗口确定，不能由方法完成速度调整。

## 4. 正式接线边界与验收清单

common-calendar先对两个实际world各自exact重现原C/S/D/Q/H/E/Y/SR，再在原event parent日历重算，保留原主Y。四类联合事件差只说明联合干预，不分解为纯通信、纯优先级或纯物理因果效应。clean actor不接收离线未来卡片；真值关联只在受控离线评分中使用。

正式wrapper必须使用admitted typed parent/DatasetProof/execution/dispatch及完整原SDK bytes，不把dev-only `validate_development_case` 用于formal，也不删除校验来“兼容”。baseline引用须绑定实际scores/components/SR及本次读取的parent receipt；secondary arm、canonical parent、repeat、sampler、native known资源、formal track派生的reservation身份和只读树快照均须从已验证对象取得，不信任同名JSON或任意prefix。主/secondary tree在验证后和发布前核同一bytes；这些是未来接线验收要求，本轮没有执行真实formal consumer或认证其全graph。

建议最小软件负/正例（尚未运行）：两条采用路径；仅plan_response/ACK；commit无后续step；commit同advance terminal；同刻旧/新卡片；新job合法采用后再succeeded；断链本地采用晚上传；pre-threshold但仍有未来动态格；无eligible parent；NA与右删失与trace缺口区分；30s/10格边界、同刻坏→好；失配parent/arm/sampler、缺重复或unknown；只读无副作用；所有track资源前缀和共同字节快照。对这些义务写明规则后再审实现，不凭本草案或本报告宣称已通过。

全部实际检查仅为定向读取与9项标准库literal算术/字段比较，最后检查 `2026-11-06T17:53:25.992454Z`–`17:53:25.994483Z`，内部耗时0.0016秒；没有执行上述待实现的运行时测试。归档后停止全部检查和R写入。
