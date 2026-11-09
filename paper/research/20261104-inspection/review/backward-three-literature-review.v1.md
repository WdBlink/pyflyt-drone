# 新增三近邻与十四条引用的有限独立核对 v1

结论：在本次指定身份与机制范围内，未发现需阻断记录采用的错误。此结论依据保存的 primary metadata、指定正文段落和独立轻量机械比对；不认证全文复现、accepted venue、文献穷尽性、新颖性、human review、实验或 G4/main 准入。审查者为 `/root/statistics_design_review`，独立上下文、同模型家族；没有实现职责。

最终 subject 为 `literature/packet.core-and-backward.v3.json`，SHA256 `b3848d39908763a62594ff8d00568c6ef487e95d6af358b7b92c2e66f7897dfe`；catalog v3 为 `1a18a48bbab839d438cb25dd3f979d7873769d4ba1da212073cc8b69e3bbfb6b`。完整输入、八份本次获取源的实际字节、十四份 metadata/verification、bib/plan、原审查报告及搜索记录的精确 SHA/大小见 `review/evidence/backward-three-literature-v1/binding.json`。这些哈希于恢复 pilot 前的只读核验阶段采集，归档时未再批量读取或哈希源码。

| 新记录 | 身份与日期 | 实际阅读与支持边界 |
|---|---|---|
| TypeFly2023 | `2312.14950v2`，4 位作者的原始 `family, given` 组件与 bib 完全一致；首版 2023-12-08，v2 更新 2024-09-26 | 定向阅读 §5.1.3、§5.2、§5.3，并核 §7.1 响应时间定义。正文确实说明完整语句收到后即可执行、生成与执行重叠、技能调用待动作结束返回，以及以首个动作定义响应时间。 |
| CognitiveDrone2025 | `2503.01378v1`，8 位作者组件一致；首版/此版本日期 2025-03-03 | 定向阅读 §III-A/B、§V-A。支持较慢推理 VLM 与约 10 Hz VLA 控制结合、推理约 2 Hz，以及 Gazebo/ArduPilot 中带认知任务的选门 benchmark；这些频率是原文系统描述。 |
| RaceVLA2025 | `2503.02572v1`，7 位作者组件一致；首版/此版本日期 2025-03-04 | 只读摘要与引言的任务/接口描述：FPV 与语言输入，三个线速度分量与 yaw angular speed 输出。其原任务成绩没有转作本研究结果。 |

TypeFly metadata 的摘要确实用 ChatFly，而同一编号/作者/版本的标题与 HTML 正文用 TypeFly。该差异已经如实记录；当前机制支持来自 versioned 正文，故不因摘要名称差异否定这项有限支持，也不把差异解释为另一独立系统或未经核实的更名史。三篇已读段落均不足以确定模型等待期间的物理时钟推进方式；也未建立本研究的报告 deadline、固定分母或持续巡检交付契约。没有据此宣称其他系统统一暂停物理，亦没有把频率描述升级为独立延迟/调度保证。RaceVLA 的 submitted-to-IROS 注释不构成 accepted venue。

十四条 bib 的 key、标题、comma 作者组件、首发表年、versioned identifier、URL 和 arXiv-preprint 类型，均与 plan、verification 和 primary citation metadata 对上。沿用已登记的 Alpha3 LaTeX→Unicode 等价标题归一化；前十一条 bib 字段与 `references.core-and-mechanisms.v2.bib` 逐项相同，原十一篇机制阅读仍仅沿用三份旧有限报告的范围，未重新扩审正文。`references.core-and-backward.v2.bib` SHA256 为 `16ce5cb15ba8ad5ab9877047784ab3b075db20f0c84f3979dd09fb64198d233c`。从 native formatter 保存的 v1 到 v2 的实际差异为新增十四条 `eprint`，以及新三条 `howpublished` 去除重复的 arXiv 编号；逆变换精确恢复原字段。原 v1 输出哈希和 repair record 相符，没有暗中升级会议/期刊身份。

机械状态须区分：`citation-check.core-and-backward.v1.json` 为 **fail**（source evidence spans must be a list），CLI exit 0 不能当作 pass；原失败保留。catalog v3 相对 v2 仅补新三条 candidate 的 `evidence_spans` 列表，内容主张不变。`citation-check.core-and-backward.v2.json` 实际为 **pass/issues=[]**，SHA256 `92e141ed7b508ee6159edbca630e0338b6ac1faddd3eeb75c269e2df4a9b7740`，其契约仍只是 record integrity，不能代替外部真实性核对。

独立查看 GRaD-Nav++ v2 参考文献 [11]/[16]/[17]，确有三条相应工作级 arXiv 引文；这些条目没有指定版本，故不能声称来源当时引用的精确版本就是本次阅读版本。保存的关键词/primary-reference 观察及其 SHA 对应 trace；只支持已发生的有限搜索与已知 citing-paper 边，不是完整 cited-by graph、系统性穷尽检索或“未检出即不存在”。SMART 两个尚未取原文的候选仍属 registered/unverified。本轮不审统计原文。所有本轮阅读保持 retrospective；`question_formed_at` 是已知 run adoption 时刻，用户最初形成问题的日期未知。

独立检查为短时 Python 标准库只读解析/字节比较，没有运行模型、物理、RNG、native CLI 写入或集中测试。最终机械脚本运行于 `2026-11-06T15:50:18.977942Z`–`15:50:19.015202Z`，随后最后一次定向参考文献/方法片段读取约 0.1 s，并在父 Agent 恢复 pilot 前已报告结束；没有遗留子进程。归档后停止写 R。
