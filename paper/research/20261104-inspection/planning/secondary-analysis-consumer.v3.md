# Secondary analysis consumer：实施契约 v3（同刻phase更正，待限定复核）

这是既定F4/F5/T4与G4分析入口的开发设计，尚未独立接受，不是正文、结果或新实验。实际formal数据未生成；原19义务、72/12 roots、8条件、评分、global6及原B1选择不变。现有primary分析候选40source/22tests、最终15/408软件检查和有限修复审查作为明确基线，不能代替本接口。

## 输入、独立单位与比较

只读同一accepted main数据/primary execution派生的固定12个L3 parents：ID/OOD各canonical 00–05。clean216额外flight含8条件原repeat；ablation108仅B4-M1三arm各3；latency180仅B4-M1五arm(0/2/5/10/20s)各3，zero也fresh，不选用可选的primary-trace reuse。它们均不增加独立root。每方法/arm先在同parent内平均原repeat，再按domain各6 parents描述；合并12只作等ID/OOD权的描述，单列两域结果。

clean需join这12 parents的原primary所有18槽；ablation/latency仅join对应12×3的原primary B4-M1槽。baseline必须来自完整、已独立锁定的main分析及其具体artifact引用，并重新核验本次所需真实parent receipts。不能用同名case或任意未绑定score JSON充当主结果。primary subject需显式绑定admitted-episode scores、components、SR等下游实际读取文件；这些字段增加不改变主统计或主检验。

事件/clean以原event parent为共同需求日历，两个world均先exact重现各自原score，再离线重计common-calendar Y及C/S/D/Q/H/E。原始主Y不改，clean actor不见未来event卡片。报告signed event-minus-clean共同日历差，同时保留原始两轨score；本轨是四类联合干预，不能将其全部解释为单一优先级或纯物理损失。

三消融以原primary B4-M1为对照，保留同parent/condition/repeat/sampler身份。五时延arm主要比较各delay与本轨fresh-zero，原primary B4-M1另列稳定性参照，不事后择优选zero。每arm完整保留C/S/D/Q/H/E/Y/SR、有效/迟到/错误动态格及actual token/call/response-delay记录；没有response时用NA与明确计数，不用0伪造时延。实际推理时间和注入wait分列，注入值不直接当作实际链路测量。

配对差点估计与root-cluster bootstrap仍是描述性，兄弟/arm/repeats同组携带；不加p、Holm family或显著性发现。拟固定99999 draws、linear quantile，使用与主分析分离的预先指定PCG64 stream；具体salt在实现/独立审查前固定并登记，不能由正式数据调整。主Hoeffding/六项Holm不改，有限样本/共享主机独立性限制继续存在。成本字段区分valid-flight与全部known retry资源；monetary/energy仍未计量，不称费用0。

## 事件执行与信息交付的两个时刻

原计划§9的改派响应指合法新/修改job处于running且程序已被持续执行器采用，ACK或仅生成计划不算，也不需等job succeeded。现有dev `packet_response` 返回一次合格观测的measured_at，包含航行/采集等待，不能直接把它改名为上述job响应。现有per-target helper/反例仍保留，正式适配需明确区分actual-job adoption、首次合格信息响应与质量恢复。

拟用既有原始`execution_audit.program_committed`与job `created_at/started_at/finished_at`、rule_id/task_card_version作为job采用的候选witness，结合冻结源中executor.advance返回该program target→FlightWorld controller.setpoint→真实world.step，以及完整240Hz audit，判断是否已真实执行。禁止将program plan_response或无物理后续的marker当采用。需要独立确认这一已有trace足以支持的精度/同刻规则；若不足，保留缺口，不从10Hz pose或ACK臆造首控制时刻。当前stage不修改在途104生产来源。

是否需要改派、受影响workload、pre-event公开证据/规则、已达到阈值与后续恢复分开。现有helper的`already_satisfied`只按pre-event质量阈值(.6/.5)判定，不能自动解释为全部未来连续需求已完成；合法的软优先级取舍也不因无新命令就判任务失败。拟将这种操作性NA如实命名/保留，另报eligible、NA分项、未响应/未恢复右删失及followup horizon；不只平均成功记录，不把deadline删失值当真实响应时间。

恢复维持原静态后续30s稳定质量、动态10个原3s格与原窗口/6s宽限；不重置grid，不让后来好版本掩盖同钟早先坏版本。事件×target行和多机/时间格不当独立样本；描述时保留root聚类与分母。如何呈现变长followup的时间摘要须在实现前明确，不能默认为未登记KM/显著性分析。

## 接线与发布边界

secondary request→fresh G4/namespace/dataset/primary+secondary execution→完整原SDK envelope/order/原repeat root mean→实际read-only dispatch/receipt/native settlement。clean复用原三envelope，ablation/latency复用现有两envelope；拒错误arm、漏槽、非terminal、unknown及重新标签。所有SDK验证针对同一捕获bytes，复制保留原bytes，仅投影所需relative raw路径。

沿用已修的验证前全episode/model-operation/resource/reserve/settle树快照，包含全部递归目录与entry类型；验证后与发布前只比较原集与原SHA。secondary真实reservation前缀是`formal-{track}-{g4sha12}-{order_index:04d}-aN`，与primary phase前缀不同，复用snapshot时必须由admitted typed track派生，不能放开任意prefix参数。parent原primary的快照同样保留。结果subject绑定全部实际语义输出和metadata ZIP；不自造G4、main lock或holdout opening，不发布论文或增加原生账本。

当前待独立设计判断：job采用的已有trace是否充分；pre-threshold NA的准确语义；不等followup/censor的描述摘要；cluster stream与对照口径。其余是按既定设计接线和正反例，不能在正式结果后调整分析定义。

## v2 前瞻固定决定（依独立设计审查补齐D1–D3）

D1：正式输出把job/config采用与首次合格信息响应、质量恢复分成不同字段，保留原dev helper原名/结果。采用选择器须同时支持program-commit和local-acquisition两路径。commit路径按原ordered plan-request/response/commit及job transitions关联serial/hash/job/version/drone，核创建/合法修改在event生效后、公开rule/card版本一致；同advance即terminal不算。采用时间标作冻结单写者调用链支持的control-tick推断，另存第一后续physics hook t+1/240、控制步末端t+1/120与完整步覆盖，不能声称记录了逐行job/setpoint。local-acquisition路径按明确running transition reason、service_started_at和同一实际本地sensor frame关联candidate/drone/object-type及采样；不等地面上传、不要求新commit、不排除后来正常succeeded。

每次worker控制迭代起点t的冻结顺序是：commands/job(t) → mission.step内head_events(t) → executor.advance/targets(t) → controller/setpoint(t) → world.step(t到t+Δcontrol，含实际physics hooks) → tail_events(t+Δcontrol) → 条件camera-frame/local_measurement/deliver/refresh(t+Δcontrol)，之后worker才可能发布状态。若上一迭代刚跨事件边界，尾部事件及该边界相机帧可早于下一迭代标为同一t的commands；head_events通常只是幂等重查。因此边界τ才处理的新job不得倒算为已作用于上一迭代末、同为τ的相机帧；不能由同一浮点t猜测phase先后。

保留三种独立时刻与来源：τ_nominal是原calendar/规则登记的生效时刻，也是原endpoint起点；t_apply_tick是冻结单写者源和完整步覆盖能证明或推断的实际应用tick/phase；t_actor_known只据实际因果消息或actor输入见证，缺失则unknown。ScenarioMission priority public_events.t及effective_from存的是c.priority_at登记时间，并未另测self.t应用tick；τ不在control边界时不能等同t_apply_tick。link/camera按时间计算availability/quality，不假称存在priority式mutation记录。producer public_state或updates.put不等于actor已收到或模型请求已包含。不能将τ_nominal悄悄改为actor获知时刻，也不能从新card编号自动补齐t_actor_known。

旧card或生效前行为不算；同刻零响应只在新card、合法参数和完整phase可证明时允许。缺少唯一可用见证则输出adoption_not_identifiable_from_trace与原因，不能冒充未响应删失；该类缺口阻断强job采用时间结论，先查dev trace/源，不改在途104。无commit后真实step、仅ACK/plan_response或同advance已terminal均不算新program采用。control tick、第一后续physics hook和local-acquisition的相机采样时刻分别保存；不从10Hz pose或墙钟替换。

D2：原already_satisfied重命名为pre_event_threshold_met，仅表示pre-event .6/.5操作性质量门，不称全部未来需求已满足或严格无需改派。仍保留not_yet_active/followup_ended/rule_unchanged/no_affected_active_workload各NA原因；NA、policy右删失、trace不可识别三者分列。按完整root/方法/domain/event类型保存全分母、eligible、各NA、identified witness、未响应/未恢复数及比例、followup H范围/分布和全部U/H/indicator；不得把censored U叫真实响应时间。已观测时间摘要只名为observed-only conditional distribution，不拿成功者均值推总体，不挑双方成功/双方eligible；本版不新增KM/log-rank/τ*或p。静态recovery_start与confirmed_at=start+30同时存，动态保存10原格window/最终到达确认witness，后见稳定不能声称在线在start已知道。不同方法eligible受自身pre-event状态影响，分母必须并列。

D3：固定NumPy1.26.4，99999draws，quantile=linear；每域/track用PCG64(SeedSequence([domain_seed,3,track_salt]))，domain_seed ID=2026110401/OOD=2026110402，salt clean=101、ablation=102、latency=103，避开既有SR salt1。每域6 parent按canonical case_id排序，先一次生成rng.integers(0,6,size=(99999,6),dtype=np.int64,endpoint=False)；全方法/arms/原与重评分量/差值同用这一个索引矩阵，分batch只切矩阵不改流。两域各自先求均值，combined仅等域1/2宏平均；原始兄弟/repeats从不拆作root。记录seed/shape/dtype/index-byte-SHA（little-endian int64），不输出巨型索引JSON。所有区间仅nominal descriptive，不加确认家庭。

下游scalar score输出只投影C/S/D/Q/H/E/Y/SR并绑定full-score SHA，完整60-bin等细节仍留原SDK raw/world，不拼成超16MiB的单JSON。大event表、闭合文件树或逐格明细按episode分文件/ZIP成员并绑定索引，既不删除分母，也不将metadata当raw。

以上常数与解释均在实际main/holdout生成前登记，不来自目标结果；v1与独立review保留。该v2仍须对应实现/正反例和有限复核，不自动放行G4。


v3只更正独立v2审查定位的D1-phase文字，D2/D3及既定分母/数值常数不改；v1/v2与两份旧review保留。尚未运行本协议的正式数据分析，需实现与正反例后再独立验收。
