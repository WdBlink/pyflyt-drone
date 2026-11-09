# Provisional seven-source citation records 独立限定审查 v1

结论：当前七条记录的标题、作者顺序、2026 年及固定 arXiv 版本与所存 primary metadata、官方版本页面一致；定位的方法段支持当前有限机制主张。未发现应阻断这七条记录采用的身份错误或实质性 unsupported extrapolation。此结论不授予文献覆盖完整、论文正文、novelty、G4 或 main admission，也不是实验复现。

审查日期：2026-11-06（Asia/Shanghai）。审查者为与主 Agent 分离上下文的同模型家族 Agent；没有跨模型或人类专家独立性的额外保证。只读原始记录、来源和 helper；新增本报告与独立证据，不修改 owner/controller、既有 source、records 或论文。未调用模型、物理环境、付费 API 或 native stage/commit 命令。

## 绑定与实际执行

入口为 `literature/packet.first-neighbors.v7.json`，SHA256 `4ffce040db78f0f4ee2d0822e2d9f6147451d5fa9a8b36597187c23f6bd09c2f`。catalog v5 SHA256 `243e2f6b9ba827df388f0b42ea064142de9812f0ddef8f1ba30d03178c653efa`；最终 bibliography v5 SHA256 `01850de0abd00d693d14e8df116e2a0d62032fd914e6e1c55802dc3bb69c1fbc`。34 个输入文件的完整 SHA256 见 `evidence/first-neighbor-citation-records-v1/input-bindings.json`。

实际读取 packet、mechanical check、catalog v5、citation-plan v2、citation-verification v2、references v2–v5、七份单独 verification、七份 primary HTML metadata、七份 text 的被定位方法及必要上下文；独立浏览对应的七个官方 arXiv abs 和 HTML 方法页面。机械 `pass` 只表示 record integrity，本审查另外核实来源身份和主张支持。目标阅读不称全文复现，未重做 MultiUAV 记录中继承的六文件 code audit，也未核验论文定量成绩。

执行 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B <evidence>/audit.py`，解析 metadata、核对标题/作者/版本/哈希、验证七条 quote 真实存在，检查两组七行 eprint repair 的字节逆变换，并在临时目录调用唯一纯输出函数 `lib.production.bibliography`。独立输出全部成功，临时目录已删除。首次尝试以顶层模块导入 helper 因相对 import 失败，改用 package import 后成功；该失败未运行 controller 或改写输入。

## 七项支持与必要边界

| 记录与身份 | 读到的有限支持 | 不可升级的解释 |
|---|---|---|
| [MultiUAV-Plat, 2606.31073v1](https://arxiv.org/abs/2606.31073v1)，6 作者，2026-06-30 | role-scoped 工具/信息权限、hidden validation 有方法上下文支持。 | 不从这些接口性质推导推理等待期间物理时钟是否继续。 |
| [MissionBench, 2607.22014v1](https://arxiv.org/abs/2607.22014v1)，8 作者，2026-07-24 | 任务级连续 4-DoF 航空控制及 explicit reporting 有直接支持。 | real-time 措辞不足以确定推理等待时环境进度。 |
| [ESARBench, 2605.01371v1](https://arxiv.org/abs/2605.01371v1)，4 作者，2026-05-02 | Event→Snapshot→Task，snapshot 内 victims/clues 空间布局固定。 | 不等于 AirSim 全部物理暂停；天气/初始化等仍是另外的变量。 |
| [Retriever, 2607.17213v2](https://arxiv.org/abs/2607.17213v2)，6 作者 | 明确 run clocks、sync policies、计算期间环境继续，以及 event time/input/clock/seed 等回放日志。 | 通用异步因果 runtime 已有先行工作；本记录不能支持把它本身当作新颖性。2026-07-19 是首版，2026-09-11 是 v2 修订，不能混称首次发表日。 |
| [α³-Bench, 2601.03281v1](https://arxiv.org/abs/2601.03281v1)，3 作者，2026-01-01 | A2A 在 dialogue level 异步且 logically instantaneous；网络影响可用性/可靠性，工具影响后续协调上下文。 | 不能单凭该表述认证 wall-clock 连续飞行。 |
| [PhysAI-Bench, 2609.23695v1](https://arxiv.org/abs/2609.23695v1)，5 作者，2026-09-20 | 从 mission traces 构造 multiple-choice action-selection；适合作为决策评测形式的边界 comparator。 | 方法中的 trace 来自 Alpha3 conversational episodes，不能改称独立核验的真实物理飞行遥测；也不是这里运行新的 closed-loop policy rollout。 |
| [AERIAL, 2610.31029v1](https://arxiv.org/abs/2610.31029v1)，5 作者，2026-10-22 | HTTP 请求至完整响应的 latency 排除 parsing/validation/SITL mission execution；场景与模型的经验性能映射用于模型选择。 | 不与固定模型下的 orchestration 直接等同，不进行跨 benchmark 分数比较。现有短 quote 主要支持 latency；模型选择子句还依赖已读 selection 方法上下文。 |

作者列表按 primary `citation_author` 原顺序逐项比对，未推测复姓。所有作者完整列表、原始 metadata dates、版本出现及断言见 `identity-checks.json`。读取锚点见 `reading-log.json`；其行号按 LF 切分，避免 Unicode separator 被 `splitlines()` 误计为新行。Retriever verification 文件名的 v1 是记录版本，记录本身明确绑定论文 v2。

“未建立/未解决时钟语义”在本结论中表示定位证据不足以确认，不能变成针对整篇文章或整个实现的否定证明。当前七项 claims 已保守表达此边界，无需据此宣称其他系统都暂停物理。

## 输出 repair 与标题归一化

提供 helper 的确存在接口不一致：`production.py` 的 bibliography 函数（418–462）允许字段不含 `eprint`，而 `science.py`（674–675）对 arXiv identity 强制核对 bibliography `eprint`。审查绑定 helper SHA256 分别为 `8e8b41849893980267378fa2843af69b0f65555cac5523bfc33f7a3e37275770`、`9f7a8eecb2741f18afaccfb3da1ba33951d88c2bd4fd1983dfc1b5953649cfaf`。

独立实测两组 repair（v2→v3、v4→v5）各仅增加 7 条与 primary 固定版本一致的 eprint 行，无删除；去掉这些行逐字节恢复 native 原件。另以七条身份调用原 helper，输出与 native v4 字节完全一致，SHA256 `59ac26fe3fa99c650a86415ebbd1cc0ee6d7fa05bed0b3fec7cb54d2a1a77384`。因此这是有出处、可逆的 literal output repair，没有捏造身份或修改 controller/helper。详细结果见 `output-repair-checks.json`。

Alpha3 primary metadata 的 `$\alpha^3$-Bench` 与 plain bibliography 的 `α³-Bench` 数学标题等价；独立按该唯一明确替换后标题完全匹配，其余题名未改。verification v2 保存 raw title 与归一化理由。此变换不是将 v2→v3 的 eprint-only 说明扩大为所有版本之间没有标题变化：native v4 已采用 Unicode 标题，再由 v4→v5 仅补 eprint。

## 时间与未纳入的 comparator

catalog 将 `question_formed_at` 明确解释为 run adoption timestamp；用户最初形成问题的日期未知。七条阅读全部保守记为 retrospective，没有把这次阅读追认成问题形成之前的因果灵感。此时间解释与记录一致，不审定用户原始形成日期。

SMART（MAPF testbed）和 GRaD-Nav++ 仅注册，尚待机制阅读；InferenceBench 是 serving optimization 硬负例。它们没有进入当前七条 plan/verification/bibliography，不能被当成已核的正文引用。此处 SMART 不与 MultiUAV 文本中另一个 SMART-LLM 混同。catalog 的 literature taxonomy/naive-baseline 标签也不构成实验 baseline 已实现或运行的证据。

最终限定判断：七条 provisional citation records 当前可追溯、身份一致且有限机制主张受支持；保留上述未知和适用范围。覆盖完整性、真实模型/飞行、定量复现、正式对比、主张新颖性及 G4/main 放行均不在本次审查范围。
