# Literature：相关性、证据和科研启发分开核对

Use the `bibliography` operation in [production.md](../production.md) for metadata assembly. Preserve publisher given/family components instead of guessing compound surnames; keep canonical author identities aligned in the plan and verification. Year-only publication metadata can stay year-only. Keep temporal/checking notes in metadata, outside printed references; include volume/pages only when actually sourced.

本阶段综合 Deli 公开检索工作流的方法及其用户本地适配，加入 LLM-Wiki 对 ScholarCatalyst 回顾偏差的本地迁移推断。未复制 Deli 第三方原文；数值配额只是历史来源经验，不自动成为当前论文的验收阈值。

## 输入与检索

读取问题形成时间、taxonomy/争议、已有材料、当前候选主张、范围及外部来源授权。按概念、同义词、方法、反例、已知失败检索，并从种子文献向前/向后追踪。官方论文、作者代码、正式卷期或决定页优先作为身份和实现依据；摘要聚合站只作召回线索。

保留检索词、检索日期、范围、失败来源及原始候选，不把服务失败当成“没有论文”。真实空白 taxonomy 单元保留检索证据及 gap，不能为每格凑文献。

## 候选与引用计划

每项记录稳定 citation key、标题/作者/年份/版本、来源、核验状态和日期、读过的位置、支持的具体主张、使用深度与排除理由。A/B/C/D 表示深入论证、相关分析、简短支撑、排除，而不是文献质量的绝对等级。

可以保留 Deli 的 LQS 多维筛选（新近度、引用、venue、机构、录用），同时记录各维真实输入、计分约定和 unknown。来源只定义了部分档位，缺引用量或录用状态时不能编造分数。声望或总分不能覆盖直接相关性、严谨度、反证和全文支持。经典文献不因年份被丢弃。

正式录用升级需查真实收录/决定页并匹配版本，不能仅因 arXiv 摘要写 accepted 就改 BibTeX 身份。最终身份不明或不能支持主张的引用必须标缺口；核验率达到历史比例不等于剩余不明引用可以成为可靠证据。

## 科研启发性与回顾偏差（K13）

对 idea/方法用途额外记录：`question_formed_at`、候选首次可见时间/版本、为何可能有用、实际用途、困难负例和一个朴素检索基线。区分背景相关、提供对照、指出失效、支持具体方法和启发新假说。

问题形成后才看到的论文可以指导后续探索，但不能在回顾评测中写成当时可用。作者回忆“这篇启发了我”不是客观真值；历史截断也不能消除回顾标注偏差。保留相似但无帮助、热门但不适用、反驳当前方案的候选，评价检索是否帮助形成可检验问题，不仅统计命中数。

## 输出与失败

形成 bibliography、citation plan、核验记录、taxonomy/争议图谱及带时间边界的用途记录。每个交付文件由 packet.artifacts 哈希绑定，供写作逐 claim 读取。

缺少全文、身份冲突、引用只支持邻近话题、问题时间或候选可见性未知时明确限制。不得称已穷尽文献、已证明新颖或已独立复现；外部研究检索论文的作者数字不成为本项目实验结果。

## 引用分批核验与来源启发式

每累积二十个新引用前完成该批核验并保存批次范围、具体 key、原论文/正式收录/DBLP/OpenReview 依据、核验时间、结果和未决项。最后不足二十个的尾批也须在成稿前核验。标题、作者、年份、venue、DOI/arXiv 身份和句子支持关系分别核对，不能只在最后拿一个“verification rate”代替过程。更改引文版本或支持主张后重新检查相关批次；候选登记、已读和已核验三种身份分开。

若采用历史 Deli LQS，记录原方法的五维权重 30% 新近度、25% 引用影响、20% venue、10% 机构、15% 录用，以及 7/5 两个筛选参考界线；未知维度不得编造，新增映射要事先声明。它只是检索优先级启发式，不能把机构声望变成主张真值或用低分删掉关键反例。

历史来源里的二十至三十检索词、二百至五百候选、八十引用/每页三引用、新近度和录用比例等只对相应长综述给出经验参照。根据当前 scope 与 venue 解释适用性，不为数量灌水。taxonomy 每格多种检索措辞、前后向滚雪球和 A/B 深度引用有实际用途，但真实空白单元保留检索证据，不虚构论文填格。综述需比较既有综述的结构、覆盖和证据，不能只称“我们更新”。
