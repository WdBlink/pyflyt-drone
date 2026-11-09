# Venue：官方规则、已录用样本与模板

Use [production.md](../production.md): `source-fetch` retains real source identities. Declare `reading_index` and include it in artifacts; `venue-calibrate --publish` generates and acceptance recomputes the index from actual per-visual records. Never prefill a reading verdict. Resolve apparent numeric/emphasis conflicts using rendered pages and actual font spans; preserve extraction and reader corrections separately.

本阶段采用用户 Deli 本地新增的投稿校准方法和 PaperFlow 的官方资源/模板绑定方法。本文件为综合改写；不复制第三方 Deli 原文，不将作者经验阈值当成 venue 规则。详细归属及源文件哈希见 `provenance.json`。

## 输入

当前批准目标中的 venue、年份/track、方向、论文类型、页数口径，以及实际来源快照。用户尚未定目标时先比较有来源支持的候选及其代价，由用户已有选择落定；不能擅自把候选变成已批准目标。

选刊/会议是“请用户回答”节点。比较完成后运行 `ask-user --kind venue`，向用户展示来源、适配性、约束、未知和推荐，通过真实问答工具等待选择后 `confirm-venue --response`。推荐、预选和泛化执行授权不构成回答。已有明确目标选择可复用实际原文与来源，不重复索取同一决定；不能伪造对未展示候选的用户批准。详见 [user-input.md](../user-input.md)。

## 校准顺序

1. 查本届官方作者/审稿指南、CFP、author kit。分别记录硬性要求、建议、未知和检索日期。相邻年份规则或样本惯例不得填成官方要求。
2. 取得真实模板及资产，记录官方下载地址、原始字节哈希、选用成员/入口、引擎和 bibliography style。解包或槽位替换要保留模板内容和依赖；路径穿越、不可获取模板、缺 class/bst 不能用占位模板伪装解决。
3. 选择同 venue 同 track、同方向且论文类型可比的已录用/见刊论文，保存每篇 PDF、年份和 acceptance evidence。预印本只能标实际身份；arXiv 目标不能捏造录用状态。PaperFlow 的三篇代表样本是 intake 起点，Deli 本地建议 8–15 篇校准预算；样本不足披露不稳定性，不能为凑数用无关论文。
4. 运行包内 `venue_profile` 测量页数、caption 数、面板及结果表线索。PDF 总页数来自 `pdfinfo`；正文页数到 References 所在页。无 References 时是总页数回退，混合 `pages_basis` 的页数统计不可当同一量解释。caption/面板/结果表均含文本启发式，编号断裂及遗漏警告必须实际核对 PDF。
5. 宿主阅读样本 PDF 的每张图表，记录类型、位置、支持的主张、版式和可比条件。只有 caption 检索没有阅读不应填“该图证明了什么”。方法总图、主结果、消融、定性/失败案例有不同论证用途。
6. 聚合预算带与设计惯例，分别报告样本数和未核验样本。目标用中位数，范围用向外取整的 p25/p75；这描述常态，不是科学质量或投稿录用保证。会议硬页数和期刊篇幅习惯分开。

## 进入计划及成稿的约束

形成 `venue_profile`、样本 measurements/design records 和本篇图表计划。每项说明 question、claim、data、section、priority；问题不是答案，question 与 claim 不能相同。超预算带必须写出有研究依据的偏离理由；常见设计形态的缺席也须解释，不为满足数量追加虚构实验。

写作前校验计划，成稿后核对实际图表与计划并读所在 PDF 页面。不得沿用 Deli 旧长篇综述“固定图表数量”覆盖真实 venue。未跑模板编译、未核对页面或只用合成样本时，报告准确验证范围。

交给 plan 的是官方规则与模板来源、可比样本、量化预算、语义设计记录和未解决警告；模板/来源/计划变化会使下游绑定失效，须重新检查受影响阶段。

## 样本与每张图表的完整记录

校准采用用户 Deli 增量的默认 8–15 篇、同方向/同 track/同论文类型的已录用样本；会议记录最近两届及其年份，期刊以明确截至日期的近两年窗口作本地映射并保留偏离说明。8–15 是测量工作量与稳健性的经验政策，不是期刊强制要求。无法取得足量样本时列明已查范围、未取得的原因、剩余不确定性及本次采用的样本政策；不凑无关论文。少于八篇的合成 fixture 只验证程序，不替真实校准。

不是“每篇填一个图类型”即可。先用实际 PDF 测得全部 figure/table caption 编号，再逐个对应设计记录：编号、PDF 页/可见位置、图/表类型、body/appendix/supplement、回答的问题、支持的具体主张、数据和比较单位、视觉编码/面板布局、图注与结论关系、值得借鉴或避免之处。记录必须覆盖实测图表；遗漏、重复、编号错配或只泛写“architecture”不能冒充完成每图分析。面板计数窗口及 caption 识别有局限时返回 PDF 核对并留说明，不能通过删记录消除警告。

官方页数必须注明含不含 references/appendix/supplement、投稿/最终版本和年份/track；规则上限与样本中位数是两件事。样本缺 References 时正文页数是回退值，混合口径不能用中位数支持“该 venue 正文通常几页”。计划校验和最终稿页数应使用当前正式规则的同一口径。
