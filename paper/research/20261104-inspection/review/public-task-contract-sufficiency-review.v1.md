# G4 公共任务与工具契约说明充分性复核

审查者 `/root/control_review`，同模型家族、独立工程上下文。本轮是**source-only**审查：未读取当前pilot分数、表现或actor输出，未运行模型、物理、测试或反例fork，未修改源、原型、live owner与旧结果。只对公开任务/工具说明和真实语义的对应关系作判断，不选择更高分的策略。

## 结论

**当前接口足以形成合法JSON和启动工具，但不足以支持“模型已被完整告知必要公开交付契约”的G4主张。** 三项P2说明缺口见下文：首次report ID/修订生命周期、有效证据对应条件、dynamic自动交付/封存与时间语义。它们可以用不包含truth的共享契约补足；不需要改变grader、提供正确性oracle、公开隐藏实体或加入经典策略算法。

这不是证明任务不可解，也没有把任何pilot表现归因于说明缺口。`quality>=.8`是当前grader的有效质量前提，**不是所有低质量调用都会被拒绝的schema门**；两视图/20m是inspect job完成规则，**不是ReportLedger的报告硬准入规则**。这两个边界不能在补说明时混写。

审查版本：actor `056bdbce49bd9096a73df1d580d25b5f48987e228e742354805edaa5c863a497`；scenario `c4f56e27…`；protocol `c32bdabb…`；scoring `bc8a2194…`；B0 reference `85f97b6c…`；classic `763b1432…`；accepted design `83f8f1a7…`。完整SHA及归档字节见 [source-binding.json](public-task-contract-sufficiency-v1-evidence/source-binding.json)。下述文件/行号均指该归档，不随之后的prompt改动重写。

## 已充分给出的部分

actor SYSTEM（`eval/inspection_llm_actor.py:28–56`）已公开600s连续飞行、推理期间物理继续、只能使用收到的公共数据、local candidate不是真实全局身份、ACK不等于job成功或质量交付、旧证据不是新动态测量、正常候选不要报告、有限槽/优先修订、版本复制、退休后不能新采样、安全退出和严格命令字段/JSON边界。工具字段列表由`inspection_llm_protocol.py:9–18`直接进入SYSTEM。

`scenario.py:72–106`已给任务deadline、几何/安全/航程/报告预算、类别/规则/优先级/区域、dynamic窗口180s/格3s/宽限6s、退出连续20s及采样Hz等。`_public_projection`（actor134–156）保留整个card、所有active jobs及公共telemetry，仅按统一预算丢较旧observation/terminal job；它没有悄悄删掉以下缺失规则——这些规则原本就不在card或SYSTEM里。Observation公开字段含quality、measured_at、arrived_at、candidate_id、sensor_mode等（protocol22–39），所以公开其合法使用关系不要求新增感知或truth。

## P2-1：首次report ID与修订生命周期未说明，且现有一句话有冲突

SYSTEM55行说report IDs必须“copied exactly”，但首次report的ID必须由调用方创建（protocol73–79），公共observation和public_state又没有预分配report ID（protocol22–39；mission322–328）。现有接口没有说明“首次自建唯一ID，之后照抄已接受ID”；照字面执行会形成无来源可抄的字段。

此外，同一ID的object_type不可变（protocol92–94）；新ID即占有限槽，包括没有形成有效关联/有效证据的提交（105–126）；首个空间关联保存在Report上，后续版本不重新关联。SYSTEM37行泛称优先修订，不能代替“修订不重置首关联/不把旧ID改投另一目标”的说明。ACK只回report ID/槽数（138–140），不会告诉模型初始关联对错，因此不能依靠试错ACK自动补齐这个契约。

**最小必要说明**：首次ID由调用方选择1..64字符，之后对同一报告复用它；新ID占一个持久槽；ID的类型与首次身份关联固定，修订不是换目标或重置关联；ACK只确认接收/存储，不证明关联或质量正确。应公开固定关联的**语义**，不公开实际associated_key、是否关联正确或隐藏truth距离。

## P2-2：可用证据的精确对应条件只给了部分

SYSTEM已说证据到达后才能用、报告使用真实证据的测量时刻/位置、不能把旧证据当新测量；但protocol101–126还有这些未明确给出的通用前提：

- evidence_ids为非空、无重复、全部已登记的ID；所有引用证据必须在提交前到达。
- 所有引用证据的candidate_id须与request中该handle一致；“可能是同一实体的多个local handle”不等于可把不同handle的证据混在一个候选引用里。真正entity对应仍由隐藏审计判定，不能提供oracle。
- `0<=measured_at<=now`；全部引用的测量时刻不晚于该stamp，至少一项恰等于stamp。
- 当前有效质量要求**所有**引用证据quality>=.8（120行）。card/SYSTEM仅写“high-quality”，没有定义该数值；低质量/不合因果的内容仍可能ACK并占槽，却不进入有效评分。

B0在reference68–74行、classic在72–84行已经将`.8`作为静态已知规则筛选；两者构造报告时使用同一候选的一项实际观测（reference45–50）。这里应补的是公开quality字段与证据引用的使用契约，而不是把这些策略的majority vote、历史长度、15m去重或任务优先队列教给模型。

必须区分：quality cut-off是有效交付前提，未公开它不能声称“有效证据规则已充分说明”；它不是协议逐项拒绝规则。Ledger也**没有**要求每份报告有两条证据、sensor_mode必须为inspect、或estimate_xy数值必须与某一条观测逐字节相等。本审查不新增这些限制，也不禁止合法测量时刻下的估计处理。

## P2-3：dynamic参数已给，但工具交付与封存语义未给全

card84行的180/3/6不是缺失项。缺失的是这些数值如何作用于工具/报告：

- observe以accepted report_id建立上传路径；在实际采样时绑定report路由，证据到达后自动向该ID追加报告测量（mission247–276、289–292）。这不是每格都要求模型另发一次手工submit；路径也不是靠稍后改一个旧报告重新指定。
- dynamic同一measurement/evidence首次写入后封存：同stamp/estimate/class重复是幂等，不同内容会拒绝，不能拿旧measurement修订成新定位或新类别（protocol123–135）。SYSTEM泛称“prefer revising”不能代替static与dynamic的区别。
- 全局需求窗口和3s格不随observe job的实际获得目标/首个本地测量重开；card94行的job duration起点是另一时钟。delivered按原需求格的测量时刻分格，完整报告和必需证据需在格尾+6s前到地面；observed仍须截止前最终上传（scoring74–77、94–103）。job成功/失败与全局及时交付已被SYSTEM部分区分，但未讲清自动上传和两个窗口。

**最小必要说明**是一次性公开上述工具效果与时间契约。可以说明需求窗口锚在目标出现这一概念，但不得提供每个隐藏目标的真实appearance时刻、未被感知的目标列表或未来事件。精确bin tie-break、误差量化分段、权重公式等仍可作为冻结离线rubric，不是合法调用所需的新增工具输入。

## 应隐藏与不应被变成策略答案的内容

应继续隐藏Evidence.entity_key/Report.associated_key、真实类别/位置、测量时刻的定位误差、正确性布尔/grade、隐藏出现时刻与未来事件、private root/seed及全target分母明细。protocol的ACK没有返回这些判定，LLM serializer也显式禁止truth/score/grader_output等字段（`inspection_llm_protocol.py:19–22,85–101`）；不应以补任务说明为由把零分原因变成即时correctness反馈。

可保留为冻结评价rubric的有localization误差3/5/10m分段、隐藏真值关联的具体最近/占用判断与平局、评分最新证据tie-break等。与之不同，调用方如何创建/复用ID、如何引用收到的证据、observe会产生什么输出、同measurement是否可重写，是公共输入/输出契约。

reference/classic的30s候选age、470s退出、majority窗口/算法、15m合并、竞价cost、cooldown等是方法实现选择，不是必须复制到共享任务卡的答案。inspect job本身以两个实际qualified views且aircraft位置差>=20m完成（execution152–165）；这可以作为job含义说明，却不能写成“所有报告必须先inspect成功”。accepted design625–640仍要求confirmation ablation保留相同工具/证据与grader：公开协议基础语义不应顺手给所有条件增加基线确认规划，或改掉消融的单因素边界。

## qualification_status与要求来源的边界

`scenario.py:107`的 `development_unqualified_new_modes` 会原样留在actor card中，没有具体mode/source/资格scope；它不是做合法工具决定所需的物理或业务约束。accepted design546–560同时说G1–G3在声明包线内qualified、G4 unresolved。仅这些输入不足以证明每个当前mode均qualified，也不足以把card笼统标签解释成所有飞行能力都未过资格。

因此这是一项**无scope的过时/易误读元数据风险**，不是已证的模型行为影响。本轮未看输出，不能声称它导致模型犹豫或某个分数。最小处理是将资格 provenance留在run/审计，actor只保留可执行合同或明确限定scope的事实；不能简单替换成“G4通过”或借此放宽权限。此项不另列为已证明的功能阻塞P2。

accepted design8–15要求same causal sensor/public tools、固定分母；546–560允许在development完成公共边界和prompt资格后冻结，之后不得用主实验结果调整。提供的`inputs/plan-excerpts.md`实际只有24行、五段摘录，**没有详细的共享任务卡/协议条款**；本报告不冒称读过完整原计划，也不把摘录没有写出的“全部rubric必须公开”当既定要求。

本审查建议只适用于前瞻、版本化公共合同。不能修补已发prompt、改旧Y、重标旧attempt或据当前pilot表现挑取规则；任何新source组合仍须按原冻结/资格/资源流程登记。本报告不决定暂停pilot的延续/替代或正式GO。

## 证据范围

必要时额外只读了mission的public_state、tool与投递段，以核实observe的实际I/O，并读execution的inspect完成条件；没有进行新的mission架构审查或运行。完整source-only对应矩阵见 [contract-matrix.v1.json](public-task-contract-sufficiency-v1-evidence/contract-matrix.v1.json)。归档11份输入源码/文本及其SHA、矩阵的总binding见 [binding.v1.json](public-task-contract-sufficiency-v1-evidence/binding.v1.json)，SHA `eacc414b1675ddf15384635017ac5249ccbb1efc4358561752a56f7acbc89e49`。这是独立工程说明审查，不是跨模型、人类领域review或G4结果验收。
