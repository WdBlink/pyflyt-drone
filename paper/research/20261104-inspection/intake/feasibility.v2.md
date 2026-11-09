# 可行性重审：本机开发路线与正式评测门分开

本版本支持的判断是：在已授权的既有本机资源内，可以继续实施这项实证benchmark研究及开发校准。它不表示G4通过、正式矩阵已可运行、LLM有效或论文已完成。原`feasibility.v1.md`和`packets/intake.v1.json`的conditional判断、四项prerequisites、旧失败与原receipt必须完整保留；通过唯一controller的正常rewind/recommit更新，不改写旧材料或预算门。

原状态基于尚无新平台运行证据。现在的增量证据为：G1–G3在声明的平地/结构化传感器/控制包线内独立复验；完成24根dev输入、实际600s L1/B0与L3/B0、L3/B1；L3/B0的105RPC和25组完整产物实际精确重放；本地回执、输入隔离、有限重试及SR适配原缺陷均保留并修复。详见`qualification/GATE_STATUS.md`、`qualification/g4-round16-evidence.v1.json`和所绑定独立报告。一次L3/B1不支持选出强基线或总体可靠性。

现有Apple M5 Max/64GiB内存可以分别加载固定版本的Qwen、GLM和Mistral三个家族，实测峰值约15.1/16.9/13.3GB。只做结构加载和分词，没有forward或generation；Mistral显式regex配置已按其官方固定V7 Tekken核对并复验，原权重字节不变。证据为`qualification/local-model-preflight/`及`planning/model-sources/`。这给出了可审计的本地模型路线，不能替代尚需实测的生成质量、时延、token预算和GPU/physics争用检查。

用户已选择RA-L并确认默认budget={}；最新明确要求先完成非LLM参考和G1–G4再进入正式实验/写作，授权的本地实现、修复和验证继续有效。没有付费API、采购算力、外发消息或投稿授权，也没有发生模型生成调用。初始完整科研计划接受后，任何dev模型生成仍须通过同一controller先预约、取得实际usage receipt再结算；不使用unreserved settle来绕过预约。

## 原prerequisites逐项处理

| 原条件 | 当前证据和适用阶段 | 重审后的边界 |
| --- | --- | --- |
| 实际venue选择与8–15篇校准 | RA-L实际答复、8篇已接受样本/图板测量、官方模板与页数；原venue packet/artifact/回答字节保持 | 本地初始研究路线的条件已有记录；rewind后须正常重交venue，不把旧receipt当新接受 |
| 具体plan-budget答复 | 用户对同一`figure-table-plan.ral.v1.json`、同一Brief和budget={}已有实际答复，绑定未变 | 复用原真实答复，不增加/移除cap或改变币种；模型操作仍先预约，任何新付费另需授权 |
| G1–G4 | G1–G3只在已登记开发范围有独立证据；G4仍needs_work，B2/六root比较、LLM生成、八条件接线、OOD规则和事件删失等未完成 | G4是正式main/holdout建立、开封和运行之前的硬门。可先有完整初始开发/研究计划，按前瞻程序解决G4；不能把未完成项改成“已过” |
| prospective root证据和图表先于结果写作 | 主形状72根/8条件及holdout12仍为候选，真实main/holdout未生成；现有均为开发或显式合成fixture | 保留完整分母和负结果；G4后正式跑完、分析/图表/独立审查后才写论文结果，当前不写结果prose |

因此重审为feasible的对象是“依既有授权开展完整研究及dev校准的执行路线”。尚未完成的实现、预算校准、最终模型/参数/源码/分割冻结、真实主结果、图表和论文是这条路线后续必须执行并验收的工作，不是对结果的预先保证。若后续证据显示资源或方法不能满足原完整义务，保持阻断并返工，不能借本状态放行正式实验或降级论文类型。

## 可执行路线与停止/失败处理

1. 本机非LLM开发：固定root与源版本，对预选六root比较B0/B1/B2；记录所有合法低分/失败和至多一次对称基础设施重试，仅在dev选择B_classic*。
2. 完整初始计划：原19义务、8配置、root/重复/分母与主要统计保持；先完成实际design、可运行recording/evaluator、WorkOrders及前瞻dev选择程序，不能用placeholder hash解锁资源。
3. 预约后的dev生成/pilot：固定开放权重版本与统一条件范围，实测token/时延/费用scope和1×资源竞争，校准prompt/预算；不能生成/查看main或holdout来调参。unknown保留并对账，不自动补零或重跑。
4. G4硬门：验收全部实际版本、模型/prompt/预算、场景家族/切分、重试/恢复、八条件与事件/SR/删失接线；关闭缺陷后明确冻结。只有此后才按已授权范围执行正式main/holdout。
5. 正式证据、图表、结果写作和独立论文审查按原依赖继续。没有可信结果就保留缺项与负结果，不预设LLM优胜，也不转成protocol-only论文。

## 资源与成本限度

沿用同一budget={}，没有隐含任务总量上限，但本状态不授权购买资源。正式主评测+holdout候选仍为1512个600s episode、252串行飞行小时，资格/dev/消融/兄弟/受控延迟/重试和分析另计。当前1×物理已实测；尚未把GPU推理并发或多world公平性当成已通过。使用现有Mac按顺序运行，进一步并行须先做资源/时钟资格。

本地API账单为零仅因没有API推理；本机资本、能耗及Codex原生用量未计量，不能写成计算免费。默认预算不是统计功效保证。主Y新有界root检验很保守，须报告条件和低功效；不按test显著性追加root。长期正式运行的实际资源依据仍由dev pilot给出，并遵守当前用户范围及后续实际决定。

替代的protocol-only写作不匹配用户当前实证目标，仍未采用；如必须改变原义务或目标，需要用户明确改范围。本次重审仅解决“conditional intake阻止初始研究计划”与“G4必须在正式运行前通过”的阶段混淆，不降低任何科学门。
