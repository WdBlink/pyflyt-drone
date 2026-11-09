# Experiment：假说、真实记录与结果身份

Use the main `lib.experiments.ExperimentRecorder` from [production.md](../production.md). It derives the planned grid, records attempted operational failures and unstarted slots, and propagates errors. Only an explicitly predeclared `MeasurementFailure` is a scientific measurement failure. Inspect what an API receives versus actually uses; passing labels through a generic fit interface alone neither proves nor rules out target leakage.

本阶段综合 Deli 实验方法、用户本地图表/证据适配和 PaperFlow 的 typed execution/效果账本，加入 Wiki 的科研诚信与测量纠偏审计问题。这是本分支方法说明，不是已集成 RIDAR 检测器、SciΠvot 评测器或远程集群。

## 输入和实验前检查

读取冻结 Brief/计划、与 requirement IDs 对应的假说/claim、数据与代码版本、评价器、统计方案、资源及已尝试方向。记录独立/依赖/控制变量、比较基线、指标及单位/方向、独立统计单位、重复和聚类结构、数据划分/标签暴露、纳入排除和失败分母、停止规则与失败条件。

评价器、基线实现和比较预算在看到目标结果前固定。先运行可辨认的正负对照/小实例检查测量是否真的测到了目标，再归因模型能力。全零、天花板、地板或异常增益先检查提取/评分/单位/数据流，不直接换更大模型或加难度掩盖测量错误。把继承的历史解释当成可检验假设，不能未经核对变成硬约束。

## 执行与结构化记录

使用明确授权且实际可用的执行器；开发 fixture 不调用真实模型或远程计算。保留真实任务/attempt ID、配置、模型/代码/数据版本、seed、开始/结束、退出/状态、逐次原始输出、样本 ID、测量值及失败原因。没有运行的配置标 not_started/not_run，不填预测数值。

结果至少能串起 config、原始 results、statistics 和 findings，另有目的/结果/局限的实验摘要。统计量由实际记录计算；原始数据与统计表达分工，实验阶段不靠手写 LaTeX 表格替代数据。早停、超时、失败、未知费用和未知效果均保留，失败不得静默从分母消失。

同名 seed 不证明配对，内部 query 不证明独立地点数；选择适用的原始点、差值、SD/SE/CI，预先定义其含义。多重比较、可选停止和重复调参影响推断时披露。新增假说或事后分析明确 exploratory，不改写原假说为“预注册”。不反复加样本直到显著，不只保留最好的单次结果。

## 科研结果与操作效果分开

- 已知工具失败/无效指标属于工作失败；保留终态回执，不能包装为科学负结果。
- 实验完成但未支持假说，是负结果或不确定科学结论；仍需数据、统计和适当范围，不自动消除必交比较。
- 已派发但终态/费用回执缺失属于效果不明，先检查现有原始输出和执行端，不盲目重放。由控制器 reconcile 保留原 attempt 和处置证据；检查完成本身不授予再次执行的许可。
- 审计任务 completed 不代表科学 GO。读取实际结论、资源和证据版本；STOP/WAIT 或改变的证据不能被另一条 GO 覆盖。

## 完整性审计与交接

逐项问：协议是否事后改变？评测数据是否塑造了方法？评价器/真值是否被篡改或针对性迎合？基线是否按自身定义且预算公平？被替代的版本是否已有未披露结果？每个数字是否来自相同指标、划分、单位和运行？

交给 figures 的是完整原始记录、纳入集合、失败/未知账、统计方案及摘要；交给 write 的是有证据范围的 finding。不能从少数配置推“穷尽”、从相关推因果、从平均改善推每个样本改善。无记录支持的解释保留为假说。出现测量缺陷应修测量并明确旧结果失效，再验证，不能维持旧结论继续精修。

## 有界迭代与失败分支

实验设计没有隐藏的五次迭代上限；可用资源由计划阶段用户回答的唯一 `budget` 和实际授权约束。新科研资源操作先预留可验证最大用量，执行后凭实际回执结算；未知用量不当作零，失败和超额消耗仍入账。每次变更保留改前设计、改变原因、探索性身份、所有结果和停止依据，不能只挑最高数值。增额先按 [user-input.md](../user-input.md) 展示当前消耗与具体提议并等待用户回答；尚可执行的独立本地工作继续。

失败后明确选择修复重跑、以实际数据呈现负结果、或修订研究范围。重跑要有新的版本/任务身份并指向原失败；负结果要经针对原 requirement 的语义审阅，不自动通过；删掉必须比较、换分母或降低证据门槛属于 Brief 范围修订。天花板、地板、未显著、意外结果都先检查测量/协议和既定统计停止规则，再设计可证伪的下一步，不照旧来源建议无限增试次。
