# G4开发进度v5

记录日2026-11-05 UTC。G1–G3仍仅在声明的开发包线内通过，G4为needs_work，未生成main/holdout。v4的原经典18slot、三次smoke及首次M1诊断结果不回改；原生已知用量仍为29calls/162210tokens，没有新增模型预约/调用。修复中的整集适配尚未用于真实模型pilot。

首次M1诊断和framing安装已完成独立复核：`review/first-model-flight-and-framing-install-review.v1.md`。26次请求、153503输入+8540生成/持久tokens、原始strict解析与入队、原score和144000时格均核对一致；原Y=0.40539069752552903、SR=0保持。安装新framing没有给旧飞行换分。

新开发namespace已按冻结总体生成五个strata：ID L1/L2/L3和OOD L2/L3各一个；master/五个root种子先落盘，joint候选次数4/12/4/59/68（共147），没有发生exhaustion或自动更换。`qualification/dev-population.v1/`保存全部输入、intent和逐候选audit；独立报告`review/development-population-worker-integration-review.v1.md`精确重放全部种子/输入/audit并查旧24根碰撞、schema roundtrip及接线。原24dev与六个pilot身份保持，新五根不是主试或旧根替代。新reader/worker接线相关73项测试通过；纯几何和fake接线不替代物理证据。

新OOD-L3场景B0实际600秒1×、144000步已完成，只有一次尝试。最大时钟滞后0.02099941667790972秒、无硬违规，H=E=1；C=0.9936936936936936、S=1、D=0.3492822966507177、Y=0.7809919967814705、SR=0。冻结evaluator全score重算一致，17源/61依赖/输入回执绑定，完整zip见`qualification/population-realtime-OOD-L3-B0.v1/`。SR0作为实际开发结果保留，不以换根或best-of-repeat处理；这只证明这一次物理链路与评分可运行，不证明总体可解性或SR可靠性。

该根的clean干预兄弟已显式登记在`qualification/dev-clean-sibling.v1/`，仅移除L3事件、采用L2公共任务卡；保持case/lineage身份、实体/ROI/wind和sensor seed，不增加独立root。实际600秒clean飞行正在进行，尚无完成或共同日历分析结论。

全八条件driver的首个prepare-only已保存原108内层槽（18classic+90LLM）、48outer、预先固定的90个sampler seeds；没有dispatch。独立纯fake复核已验证完整SDK形状和失败时完整分母，同时发现evaluator逐槽重新取hash、classic timeout只收直接子进程两项缺口，原报告`review/eight-condition-development-driver-review.v1.md`保留。修复候选改为按accepted design预冻结evaluator entry/closure全集、每槽查集合与字节，并使用出生身份进程树清理；修复复验未完成，旧prepare不能作为当前源码冻结。

整集模型适配初审发现clock故障可绕过causal seed核验、spawn后start回执写故障漏清理，以及settled archive缓存未复查、enqueue witness未查、embedded source未重算等反例。修复候选已要求每个meter已知attempt都核源/因果/冻结score后才允许基础设施retry，terminal绑定archive/usage/settlepacket，入队见证与source字节逐项复算。spawn/start/wait统一finally，期间追踪出生身份，normal也清理已登记子代；未登记新同UID Python孤儿作为unknown阻断而不擅自杀归属未知进程。这不是kernel subreaper或任意fork/setsid/断电保证，也可能保守阻断同期间的无关Python daemon。19项episode/cleanup及合计66项episode/cleanup/wire/actor测试通过，独立修复复验仍待。资源用量已知与episode终态未reconciled分开：缺失model完成回执保持unknown/held；已正常结算的known用量不会因稍后terminal写失败变成未知或归零，但episode仍阻断重跑。

仍待完成：上述修复的独立复验、实际native完整LLM整集/SDK链、剩余新总体strata物理资格、clean/event同日历与响应/恢复/右删失，固定原六root全八条件pilot，最终G4源/prompt/model/generator/split/retry清单。G4通过之前不开展正式集/holdout或论文结果正文。保持八条件、19义务、固定评分与总体范围；无付费API、新算力购买或投稿。
