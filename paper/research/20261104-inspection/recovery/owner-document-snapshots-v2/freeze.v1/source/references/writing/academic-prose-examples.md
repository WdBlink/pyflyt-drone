# 论文表达案例：从问题到改法

配合 [表达约束](academic-prose.md) 使用。E01–E10 中的改写片段是教学改编，不是原论文逐字摘录或完整段落；模型名已泛化，E07 的数字专为演示构造。E11–E14 使用公开论文短摘录，逐例标出来源和位置，并区分原文、中文概述与本地分析。任何示例中的事实、数值、研究设计和引用占位符都不能直接搬进当前论文。

## 怎样使用

先从当前稿件和证据判断问题，再选最贴近的案例。完整起草或多段修订先浏览索引；局部修改只读相关案例，通常一至两个足够。读取案例时同时看“改法”和“反例与核对”，再用当前研究的真实材料改写。无法匹配时按表达约束处理，不强行套句式；不要求最终稿附上案例编号。

| 当前问题 | 案例 |
|---|---|
| 摘要像功能介绍，没有研究问题 | [E01](#e01) |
| 删第一人称后更难读，探索性身份丢失 | [E02](#e02) |
| 真实动机埋在讨论，想提前呈现 | [E03](#e03) |
| 方法先堆术语，读者不知道选择理由 | [E04](#e04) |
| 结果只带看图表，或把流程比较写成组件效果 | [E05](#e05) |
| 局限反复强调，结论过度弱化 | [E06](#e06) |
| 数字没变，但对应关系或结论变了 | [E07](#e07) |
| 文献真实，句子却归属过头 | [E08](#e08) |
| 想删除所有被动句、强结论或重复术语 | [E09](#e09) |
| 想把 AI 文风研究变成禁词或固定模板 | [E10](#e10) |
| 参考真实论文怎样简洁交代研究范围 | [E11](#e11) |
| 参考真实论文怎样把保留措辞绑定到估计量 | [E12](#e12) |
| 区分有证据的转折与机械的正反平衡 | [E13](#e13) |
| 从研究收录的 AI 原句辨认空泛导览 | [E14](#e14) |

## E01

**把摘要中的项目说明改成研究动作。**

前提：研究保持位置估计不变，只检验不确定性校准是否改善加权；当前句子介绍研究目的，结果另有证据段。

改前：
> A case study evaluates an uncertainty-calibration framework for map-assisted localization.

改后：
> We use map-assisted localization to test whether uncertainty calibration improves the weighting of fixed position estimates.

改法：让读者看到检验对象、动作和保持不变的量；“框架被评估”没有替读者说明研究问题。这是目的句，不宣称实验已经成功。[P1](#公开资料与适用范围) 支持按读者需要组织贡献。

反例与核对：若论文确实更新位置，就不能照搬 fixed；若原句已经准确交代问题，无需为了加 We 改写。摘要其他部分仍应报告实际主要发现。

## E02

**识别一次失败的润色：删除 We 并没有改善表达。**

前提：作者在检查早期结果后设计两个探索性对照，其中一个避免额外求解运行；所有位置和失败状态保持不变。

失败改法：
> The base-only and conditional-addition comparisons assess the incremental information from additional configurations.

可用改法：
> We compare the full model with a base-only model that requires no additional solver runs. We also test the added feature within the base-only model family. Both comparisons keep positions and failure states unchanged. We designed these exploratory comparisons after inspecting the earlier results.

改法：让执行者、两个不同的比较和设计时序可见。长度增加，但信息关系更清楚；修订目标不是最少词数。

反例与核对：如果实际是预先设计的对照，不得套用最后一句；如果额外运行成本未知，不得补造“无需额外运行”。主语清楚的被动句可以保留，见 E09。

## E03

**前移动机，同时保留发现的实验身份。**

前提：已有开发结果显示模型 A 的预测似然改善，而在人工独立先验下的受控更新风险上升；原稿直到讨论才解释这一矛盾。

改前的组织：引言泛论不确定性的重要性 → 方法和实验 → 讨论才出现上述发现。

可前移到引言的句子：
> In development experiments, model A improved predictive likelihood but increased controlled update risk under an artificial independent prior. This mismatch motivated evaluating uncertainty through the updates it induces.

改法：用已有观察建立具体研究问题，后文保留完整实验结果；参照 [P1](#公开资料与适用范围) 的读者顺序原则。

反例与核对：不能改成“Independent validation confirms that better likelihood causes worse navigation”。原证据既不是独立确认，也没有建立因果或导航效果。若记录表明这项观察出现于设计完成之后，就不能写 motivated，应写它如何帮助解释结果。

## E04

**先解释估计目标，再引入模型缩写。**

前提：模型围绕固定报告位置表示误差，误差可能有非零偏差，目标是条件二阶矩；相关数学条件已由推导支持。

改前：
> G denotes geometry, S denotes centered scatter, and A denotes the anchored moment. We combine G and A in the model.

改后先解释：
> Because the reported position is fixed, its uncertainty must account for dispersion and persistent bias. We therefore model the error second moment, which includes both components.

随后给出已有公式及条件，再定义 G、S、A 并说明组合方式。定义可以搬移，数学假设和模型限制不能因重排丢失。

反例与核对：二阶矩目标不能被写成“这个受限模型一定达到无约束最优”。若当前段落只是已定义术语的索引，原句可以保留；不要每次出现缩写都重讲动机。[P1](#公开资料与适用范围) 支持控制读者需要暂存的未解释概念。

## E05

**结果先给具体判断，同时分清比较回答的问题。**

前提：模型 A 的验证集 NLL 低于 B，区间覆盖仍偏离目标；A、B 使用的校准样本数不同，比较对象是完整校准流程。

改前：
> Figure 3 presents the performance of the models and demonstrates the effectiveness of the proposed component.

改后：
> Model A has lower validation NLL than B, while its region coverage remains below the target level (Figure 3). Because the calibration sample counts differ, this comparison evaluates the complete procedures; it does not isolate the added component's effect.

改法：把“有效”拆成实际指标与比较单位。两句均由具体事实触发，不是规定每个正面结果后都要补一个 but。

反例与核对：若样本、训练和预算均受控，且只改变目标组件，就不应继续套用“无法隔离组件效果”。复杂图的读法说明仍可保留。样本差异不能只靠一句“请谨慎解释”交代；后文应给出真实样本数和选择方式。

## E06

**合并局限、恢复结论，但不删除实际影响。**

前提：评价使用已见地点的留出查询，报告位置固定，更新采用人工独立先验；跨地图迁移没有测量。

改前：
> These findings should be interpreted cautiously. The limited evaluation may only suggest potential benefits. These limitations restrict generalization, and further validation is needed.

改后：
> The measured gains concern the weighting of unchanged positions under an artificial independent prior. Evaluation uses held-out queries at calibration locations. Transfer to independently produced maps remains untested.

改法：用已知效果和具体范围替换反复警告。共同条件在合适位置讲清楚后，不再每个结果段重复；摘要和独立图注仍保留必要信息。

必须保留或升级处理的反例：若所谓测试集实际参与调参，就不能只写“迁移未验证”。应纠正独立评价身份，并撤回依赖该身份的泛化主张。若统计证据只支持样本均值较低而不支持稳定收益，第一句也需按 E07 改成描述性结果。期刊要求独立局限章节时仍按其格式组织。

评审意见对照：“请增加三条局限”不可直接执行；“摘要声称跨域有效，但只评价了同域查询，请限定该主张或提供跨域证据”能定位真实问题。已准确限定范围且符合任务验收时，不因没有三条缺陷而继续返工。

## E07

**数字一致不等于语义一致。以下数值全部为教学虚构。**

前提：20 个地点中 A 在 4 个改善、16 个退化；汇总误差差值 A−B 为 −0.03，95% 区间为 [−0.08, 0.02]。另一张表中 120、300、80 分别对应开发集、验证集和补充集样本数。

两种错误改法：
> A consistently improves performance across 20 locations.
>
> We evaluated 300 development, 120 validation, and 80 supplementary samples.

第一句把汇总均值改成普遍改善；第二句保留了所有数字，却交换了数据集归属。

可用表述：
> The aggregate mean error difference (A−B) was −0.03, with a 95% interval of [−0.08, 0.02]. Error decreased at 4 of 20 locations and increased at 16. The comparison used 120 development, 300 validation, and 80 supplementary samples.

反例与核对：区间跨零不能改写成“A 与 B 等价”或“A 无效”；若要声称等价，需要相应设计和证据。均值降低也不能单独证明稳定优势。核对方向、地点计数、区间含义及样本对应关系；不能把这组虚构数字用于真实稿件。

## E08

**引用贴对了领域，仍可能贴错具体方法。**

前提：已读文献 A 支持多变量协方差重标定，但没有给出本研究采用的非负加性模型。以下 [A] 仅是引用占位符。

改前：
> Following A, we fit the nonnegative additive covariance model [A].

改后：
> Multivariate covariance recalibration provides relevant background [A]. In this study, we use a restricted nonnegative additive model.

改法：分开外部文献支持的背景与本研究的方法选择，不让同一个引用替两种不同主张背书。

反例与核对：没有在 A 中找到该模型，不意味着它首次提出；不能顺手加 novel。若其他文献确实提出了该模型，应引用实际来源。拆句后还须检查相邻句和方法介绍是否继续错误归属，不能仅检查 citation key 未变。

## E09

**正确动作也可能是保持原文。以下为构造的反例。**

情形一：方法段连续讨论图像的准备过程，实际流程确为先缩放、后校准。
> The images were resized before calibration.

保留：图像是当前话题，被动句清楚表达操作顺序。改成作者作主语并不会自动提高可读性。[P2](#公开资料与适用范围) 建议按话题选择语态。

情形二：存在完整证明，且所述条件正是定理假设。
> Under assumptions A1–A3, the estimator is unbiased.

保留：不得为“更谨慎”改成 may be unbiased。若证据其实只是有限次模拟，则不能照搬此判断，需要按真实证据改写。

情形三：前后句都指同一个 statistical calibration 概念。
保留 calibration 的术语一致性，不为词汇多样性轮换成 correction、adjustment、normalization；这些词可能指不同操作。若重复的是整段已知结论而非必要术语，仍应合并。

## E10

**公开研究如何形成写作例子，而不是身份判断器。**

材料一：[P3](#公开资料与适用范围) 在所研究的学生议论文任务中发现，默认提示下 GPT 的缓和表达比人类少，且提示变化会改变修辞特征分布。

错误用法：“AI 总会过度强调局限，所以全文删除 may 和 limitation。”

可用判断：“这份稿件是否存在没有具体作用的重复保留？”若段落仅反复说需谨慎，就按 E06 改；若保留意见限定了真实未知，就保留。研究中的缓和词频不能直接当作论文局限段长度的证据。

材料二：[P4](#公开资料与适用范围) 分析大规模生物医学摘要，观察到 LLM 普及后部分风格词频率上升。

错误用法：“出现 delve 就证明是 AI；把它换掉即可通过写作验收。”

可用改法（构造例）：将空泛的 “We delve into a comprehensive exploration of calibration” 改为实际研究动作，如 “We compare calibration error before and after rescaling”，前提是确实做了该比较。若实际只做理论推导，就不能借此编造实验。群体词频不能用于认定单篇作者身份，换词也不能补上缺失证据。

## E11

**真实论文的范围说明：限定讨论对象，不贬低已完成的贡献。**

来源：[Mensh & Kording，Discussion 首段](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619)。原文短摘录：
> This paper focused on the structure, or “anatomy,” of manuscripts.

相邻内容的中文概述：作者随后说明没有细谈选词、语法、创作与协作，并指向其他写作资料。

本地分析：先明确这篇文章解决哪一层问题，再解释未覆盖内容。没有把“未覆盖全部写作问题”写成“所给结构建议没有价值”。可迁移到综述的范围交代，但不能用范围声明掩盖支撑核心结论所必需的遗漏。

教学改编（需有真实范围依据）：
> This review examines uncertainty calibration for fixed predictions. It does not evaluate methods that jointly change predictions and uncertainty.

核对：读者是否知道覆盖和排除的对象，以及为何仍能理解当前贡献；不要把本例变成每段必须补一句排除项。

## E12

**真实研究的定量主张：保留词承担具体推断功能。**

来源：[Kobak et al.，arXiv v5 摘要](https://arxiv.org/abs/2406.07016v5)。原文短摘录：
> This excess word analysis suggests that at least 13.5% of 2024 abstracts were processed with LLMs.

上下文的中文概述：研究分析 PubMed 生物医学摘要的词频变化，以此估计 LLM 处理比例；下一句明确这是下界，且不同子语料存在差异。

本地分析：suggests 对应从词频推断处理比例，at least 对应下界，年份和摘要对象限定范围。这些限定各有作用，不能为了“更自信”删除，也不需要再叠加“perhaps possibly”。

错误改写：“13.5% of all scientific papers were written by AI.” 同时丢掉下界、对象与年份范围，并把“经 LLM 处理”换成“由 AI 写成”。

迁移与核对：先识别当前研究报告的是点估计、区间、下界还是观察值，再保留相应限定。学的是估计量与措辞的对应，不能把这句话当作本研究使用 LLM 的证据，也不把所有结果统一降为 suggests。

## E13

**真实研究的混合结果：有信息的 however 应当保留。**

来源：[Doshi & Hauser，arXiv v3 摘要](https://arxiv.org/abs/2312.00506v3)。前文中文概述：在线故事写作实验中，获得生成式 AI 创意的作品在写作质量等评价上受益，较低创造力的写作者收益尤其明显。随后原文写道：
> However, GenAI-enabled stories are more similar to each other than stories by humans alone.

本地分析：转折连接“单篇作品的评价”和“作品之间的相似性”两个实际测量维度，并非为客观感追加通用缺点。只保留受益或只保留同质化都会改变结果画像。

教学改编（仅当两个指标均有证据）：
> Calibration reduced NLL, while update risk increased under the specified prior.

反例与核对：若当前研究没有测量第二个维度，就不能为平衡句式编造一个代价。故事实验也不能直接证明科研论文采用 AI 后必然同质化；引用研究结论时保留任务情境。

## E14

**研究中收录的 AI 句子：导览明确，不等于研究问题明确。**

来源：[Crosthwaite，§4.1.3，Excerpt 22](https://onlinelibrary.wiley.com/doi/full/10.1111/modl.70089)。这是论文收录并标为 GPT 生成的样本，不是研究作者自己的论断：
> This essay will explore the history and development of boxing equipment.

原研究分析的中文概述：所研究 GPT 议论文频繁使用文章目的宣告，其使用方式与人类样本不同。

本地分析：这句话交代了范围，但没有提出需要回答的问题或形成的判断。在教学导读中可能够用；若它承担科研摘要的研究目的，就应检查是否过泛。不能仅凭 This essay will explore 判错。

科研语境的本地改编前提：确实控制数据与训练流程，只比较缩放前后校准误差。
> We test whether rescaling reduces calibration error when the training procedure and evaluation data are held fixed.

核对：新句是否对应真实研究动作，还是只是把空泛动词换成看似具体的 test。此改编并非该论文提供的修订版本，也未由该研究验证效果。

## 公开资料与适用范围

- **P1 — Mensh & Kording, Ten simple rules for structuring papers（2017）**：[PLOS 原文](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619)。支持面向陌生读者组织贡献、降低术语负担；属于写作指导，不是上述改编案例效果的实验验证，也不要求所有段落使用同一模板。
- **P2 — Effective Writing**：[Nature Scitable](https://www.nature.com/scitable/topicpage/effective-writing-13815989/)。支持动作使用动词、语态服务话题、时态影响主张范围；是写作教学资料，不作为当前投稿期刊的格式政策。
- **P3 — Prompt Sensitivity in AI Writing: A Methodological Caution for Corpus-Based Analysis of AI-Generated Language**：[The Modern Language Journal 原文](https://onlinelibrary.wiley.com/doi/full/10.1111/modl.70089)。比较学生议论文与 GPT 文本及不同提示条件；用于提醒修辞模式受任务和提示影响，不能外推为所有科研论文的统一 AI 风格。
- **P4 — Kobak et al., Delving into LLM-assisted writing in biomedical publications through excess vocabulary（2025）**：[作者公开版本](https://arxiv.org/abs/2406.07016)。研究对象是大规模摘要语料中的词频变化；不提供按单个词判定作者身份或写作质量的规则。
- **P5 — Doshi & Hauser, Generative AI enhances individual creativity but reduces the collective diversity of novel content（2024）**：[作者公开版本](https://arxiv.org/abs/2312.00506v3)、[发表版 DOI](https://doi.org/10.1126/sciadv.adn5290)。E13 摘录来自作者公开版本；其故事写作实验用于展示混合结果如何完整表达，不作为科研论文文风的直接因果证据。

这些来源支持相关原则或反例边界；标明“本地分析”“教学改编”的内容由本地编写，不冒充来源作者的建议或经验证的改写效果。常规写作直接使用本文件；只有需要核验或展开公开研究结论时才重读来源。
