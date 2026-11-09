# Review：逐项交付、科学质量与计划合规

Use `bind-all`, `prepare-review` and `assemble-review` from [production.md](../production.md). The program derives the current obligations, exact hash maps, source closure, R6 checks and eight-role snapshot. An actual independent reviewer supplies identity, request-bound semantic judgments and findings. No reviewer count, paper count, named report file or revision label is universal.

本阶段结合 Deli 的专业门及用户新增 R6/逐项合同/保真、PaperFlow 的独立 requirement_checks 和 Wiki 的完整性审计。以下是当前分支的审阅方法，不承诺复现外部论文的检测率或真实录用概率。

## 输入与独立性

读取同一冻结版本的 Brief/逐项要求、轮前计划快照、当前计划、稿件 TeX/PDF、图表本体、实验/引用证据、数值与编译检查、历史缺陷及修订记录。缺计划快照无法判断是否事后改计划，缺真实 PDF 无法称成稿审阅。

专业 reviewer 使用新上下文，仅给本次合同和必要证据，不先给作者期望结论或其他人的分数。写作者不能自批自己产物。记录实际 reviewer/模型身份；若只有同模型独立上下文，明确独立性范围，不能冒称多模型。离线 fixture 的 reviewer 值仅测试接口，不是真实独立科学评审。

## G1–G5 的逐项依据

- G1 文献：身份/录用/版本核验、引用支持范围、taxonomy 覆盖与真实 gap、近期与经典来源、未核验比例和检索失败；不能仅看引用数。
- G2 实验：预先假说和统计设计、评价器/基线公平性、原始记录/失败/未知分母、独立单位、可复算统计与 claim 关系；不适用须由当前范围证据说明，不能因缺实验而临时豁免。
- G3 结构与保真：问题—选择—证据关系、术语/公式/定义、摘要结论一致、数字对应、引用归属和改写后的条件/强度；不能只查字符串或编译。
- G4 图表：venue 与逐项计划、输入/变换/数值/图注/claim 链、可比条件、真实单位/区间/失败、成稿可读性和所有正文引用。
- G5 综合：前述适用门、当前 PDF、独立逐项评审、缺陷回归、R6 和当前版本完整性。没有 blocker 才能进入确定性 finalizer；模拟总分不能抵消任何未交项。

## R6：单独核对承诺

R6 同时读轮前快照、现计划、成稿和真实交付。先运行 `plan_audit` 的机械比对，再逐条判断视觉/实验/内容/数值承诺。每项给计划 ID、稿件位置和证据，区分 fulfilled、disclosed_deviation、undisclosed_deviation、scope_expansion 和 plan_rewritten。

披露必须在稿件中具体解释缺了什么、为何缺、结论如何变化；只在计划、代码注释或 review 中说明不够。有解释的偏离仍是偏离，不升级 fulfilled。计划改写单独报告，不能通过新计划掩盖旧承诺。未披露偏离或无法核验均不能给 G5 pass。兑现率同时报分母和待评审项，不能用已判部分的 100% 声称全部完成。R6 不给论文总体分。

## 独立 requirement_checks

每项批准 ID 必须有真实产物、稿件插入位置、当前资产/证据 hash 和针对验收标准的语义判断。使用 packet contract 的 verdict 字段，不能只说“存在图F1”。漏项、多项、错绑、陈旧 hash、重复占用一个无关图表、仅写 limitation 代替必交比较或任何 unmet/blocked，均阻止最终验收。

审阅改变过的 artifact、binding、源码或证据后，旧 build/review 失效；未变化且绑定仍有效的证据才可复用。worker completed 是执行事实，不是承诺满足。

## 研究诚信与返工

检查每个报告数字能否回到产生它的记录，论文/代码/协议是否一致；已完成但不利/提前结束的运行是否保留，评测是否被结果驱动修改，标签是否参与方法塑形。对全零/全好结果先挑战测量； inherited 假设不能自动成为限制搜索的事实。同模型再想一次不是可靠纠偏的证据。

专业意见路由回 literature、experiment、figures 或 write，修后重编译并检查旧问题是否回归。来源中的固定评分档和强制优缺点数量只是历史作者标尺，不要凑缺陷、造优点或把模拟分数映射为外部接收。将实际缺项和失效范围写清，保留负例，不以审阅完成包装为 GO。

## 角色、历史评分与返工回归

Deli 的 R1 实验、R2 理论/定义、R3 表达/图表、R4 综合/新颖、R5 新读者提供审查视角，R6 单独查计划。按任务风险分配真正独立的执行者，普通措辞修订不机械召集六人。每位实际审阅记录模型身份、稿件/PDF/证据版本、问题位置、Major/Minor、建议和修复/回归证据；不得用角色标签捏造独立性或异模型。

若采用模拟评分，保留逐维理由和各实际 reviewer 总分，再取中位数；来源未给完整权重矩阵，不另造“官方算式”。首轮不超过七、每轮最多加 1.5 等是历史协议内防膨胀参照，采用时需标清；固定三至五个优缺点和至少一个 unresolved 不能迫使评审捏造问题。已全部修复则据实报告，未解决的重要缺陷不能因分数达到目标关闭。workshop/main/oral 等映射不是外部评价或录用保证。

修订后逐条回看历史 Major 和先前已修缺陷：原问题是否真正修复、关键证据是否改变、是否引入其他回归。只有报告“重新review过”不足；当前稿件和原问题的定位/版本必须能核对。失败门、未检查门和有范围依据的不适用门分别记录，G5 永远不能不适用。
