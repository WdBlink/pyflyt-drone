# 初始科研计划包独立审查 v1

审查者 `/root/control_review`；同模型家族的独立工程上下文。只读审查具体 v1 计划、设计、冻结 evaluator 及实际离线重计，不代写生产实现，不授 G4、正式 main/holdout、付费或论文完成资格。

## 结论

**当前 v1 尚不宜接受为 frozen initial plan。** 19 项义务、八配置、完整主/holdout 分母未减少，评分入口也不是占位 grader；但还存在以下提交前应解决的具体问题：冻结总体/次级端点的定义不足、当前 SDK 外层设计与拟冻结设计不相容，以及开发资格工作单的实际交接不完整。它们不要求先完成 G4 或先做模型生成；需要把本次初始计划的科学约束和可执行记录交接先写具体、验证一致。

合法阶段仍是完整初始计划→正常预约 scoped dev 生成→dev 校准/资格→G4 冻结→建立/开封正式 main/holdout。未来 actor、runner、prompt、兼容生成器及其 source/model manifests 可以是开发输出；本次已经承诺不可变的总体、端点、失败分母、主要比较与 grader 不能只留作“G4 前再定”。本报告不是否定该阶段顺序。

## 必须修正的发现

### P2-1：总体与部分次级端点仍只有“以后冻结”，未成为具体前瞻设计

证据：[design.ral.v1.json:599](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/experiment/design.ral.v1.json:599) 仅规定 G4 前再冻结 distributions/ranges、physical feasibility 与 identity guards，OOD 只说改变 specified parameters，但本文件没有给出这些 parameters、ID/OOD 分布/范围或确定的 dev-only 选择程序。第 560 行同时规定 scientific design、populations 不能在此计划下改变。现有 72 个 main IDs/12 个 holdout IDs 及数量只定义槽位，不定义从什么总体抽样。

次级合同在第 604–607 行也留有实质选择空间：event-clean 兄弟比较未列方法/重复集合；“removal of event trigger”没有固定周期值、与主策略相同的预算口径；0/2/5/10/20s 延迟没有指明额外加在何种实际等待之上、何时开始/结束及迟到结果处理；响应/恢复只说 G4 前定义，没有可判定的实际服务见证、恢复阈值、起止/删失/无需动作规则。原计划已有这些义务，例如 [原研究计划:240](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/inputs/BENCHMARK_RESEARCH_PLAN.original.md:240) 区分实际 running/配置采用与 ACK，并要求公开规则服务、NA 与右删失。

影响：若先冻结当前 exact packet，后续补总体和端点会改变本次所谓不可变设计；若把所有选择都称实现输出，则仍可在看过 dev 结果后自由挑目标总体、干预或恢复判据。本问题不是要求预先生成 test draws，而是要求预先规定其抽样族/约束和分析语义。原 `scenario-adapter.ral.v1.md` 也只是候选描述，且未被本 packet 绑定，不能补足这个缺口。

最小修复：在接受前具体冻结 ID/OOD 不同机制、范围/权重、非成绩抽样拒绝及 lineage/RNG 规则；可以选择有明确边界、输入与决策规则的 dev 校准程序，但不能只写“以后选”。明确十二个 L3 roots 的 event 方法/重复、消融固定周期/预算、延迟干预及响应/恢复/NA/删失端点。未来 G4 只验证这些规则及兼容实现，并产出冻结版本；不能自行改变 population、grader、主要比较或端点。无需先生成 main/holdout。

### P2-2：当前 SDK 生成的外层设计与拟冻结设计 SHA 不相容

证据：[plan.ral.v1.json:34](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/packets/plan.ral.v1.json:34) 冻结 `design.ral.v1.json` SHA `2195c892…`；该设计使用四个 descriptive option contrasts，包括 B1/B2 两个备选，区分于后续 selected-classic 的主六家庭，这个科学区分本身合理。

但实际 [inspection_recording.py:52](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_recording.py:52) 的 `designs(cases,context,classic_best)` 对 root-means 始终重写 experiment_id、metric、statistics 并新增 adapter_semantics，第 64–66 行只生成三个 contrasts（其中一个已经选定 classic）。本轮纯 CPU 调用 `designs(design['cases'],design,'B1'|'B2')['root-means']`，两种选择都与拟冻结设计不等，差异字段恰为 `adapter_semantics,experiment_id,metric,statistics`。这不是 case 顺序或零星 JSON 格式差异。

影响：现有记录适配器直接产出的外层设计不能满足 [science.py:191](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/science.py:191) 对 actual design 与 frozen hash 的要求。旧八条件形状预检证明旧 adapter 的完整槽位，不证明这份新的 canonical frozen design 已接线。不能冻结后再改 design hash 来迁就结果。`evaluator.v1.py` 是合法的离线分数重计器，不是 `measure(root,repeat,condition)` 或 SDK 汇总桥，不能单独填补这个交接。

最小修复：接受前提供并纯 CPU 预检到 **exact canonical outer design** 的记录/汇总桥：内层完整 1/3 重复的实际 records→原证据 SHA/完整分母→本设计的 root means，且正确保留未启动/unknown/操作失败。可以保留既有经典/核心代码和 SDK，不必改它们或先跑 LLM/physics；桥必须明确两份 descriptive classic-option 与单个 dev-selected primary classic 的关系。新的完整 envelope/adapter 需要绑定相应源码/预检，而不是仅引用旧预检已通过。

### P2-3：WorkOrders 只做 ID 覆盖，缺少开发资格的独立交接

证据：[plan.ral.v1.json:35](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/packets/plan.ral.v1.json:35) 的四张订单覆盖全部 19 IDs，但没有明确的 dependencies、实际 inputs/outputs、开发资格 acceptance；`W-qualified-evidence` 仅 validate 未来完整 `packets/experiment.v1.json`。设计第 549–555 行列出开发路线，但没有把六根校准、资源/实际八条件接线与 G4 结果交到一个可独立验收的 dev 订单。这样“initial plan 含 dev 校准工作单”仍主要是声明。

规则不是要求一项义务一个 Agent，而是要求每张工作单有依赖、输入/输出及作用域，见 [plan.md:18](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/knowledge/plan.md:18)。`runtime.check_orders()` 只查已注册动作、覆盖与路径，不会自动从设计文字执行 G4 科学门。

最小修复：为现单补充具体的 scope/输入/输出/依赖/验收，或增加承接 T2/F1/F2 等相关义务的 dev qualification 订单；区分 dev 证据、G4 冻结、正式 experiment、统计/图表与写作。使用已有合法 validate/copy/render/TeX action 收集 host 实际证据即可，不添加虚构 `run_llm` 或任意脚本动作，不把 completed 订单当 GO。

## 已验证的正面范围

- 当前 Brief 的 `F1–F6,T1–T4,S0–S7,M1` 共 19 项均被现 WorkOrders 覆盖，十项图表 ID/类型及 snapshot 原字节 SHA 与 `figure-table-plan.ral.v1.json` 一致；没有删减既有纸稿义务。
- 条件仍为 B0/B1/B2、B3-M1/M2/M3、B4-M1/M2；classic 各 1、五个 LLM 各 3 重复，每 root 18 episode。72 main roots 为 ID 三等级各 16、OOD L2/L3 各 12；独立复算主 1296、holdout 12×18=216、合计 1512，252 串行飞行小时只计主+holdout，不含开发/重试/兄弟/消融/延迟。
- dev 六根明确预选；classic 必须完成 18 个 B0/B1/B2 dev 槽，仅从 B1/B2 按等等级宏 Y 选，1e-12 tie 取 B1，结果在 main 前冻结。四个 SDK descriptive contrasts 不被冒充四/八项主要假设；主比较仍为两个同模型架构差与 selected-classic 差，ID/OOD 共六项 global Holm。
- [statistical-protocol.ral.v1.md:15](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/statistical-protocol.ral.v1.md:15) 明确主 p 是独立有界 root 的 Hoeffding 界，等级固定宏权重，Holm6；pointwise/Bonferroni 区间与 nominal bootstrap/参考 p 分工清楚，未承诺 .05 效应功效。完整 root 独立性、运行漂移/学习状态和 actual loader 仍要真实证据，不由数学模块认证。未来主统计源码/真实输入与其 protocol 一致性仍须绑定验收；本轮不重新跑 99999 bootstrap 或压力矩阵。
- policy timeout、malformed reply、工具拒绝、低分和 unsafe 仍保留实际 Y；基础设施仅有限、对称、原 case/config 的一次重试；unknown 与 source/input/grader mismatch 阻断，不补零或挑最佳 repeat。晚回包忽略与真实等待消耗飞行时间保留；次级延迟干预仍须按 P2-1 写精确。
- 三家族 registry 固定具体 repo/revision、候选预算与 tokenizer 配置，明确 prospective/not-G4，creation generation_calls=0。开发可以产生受初始设计约束的最终 prompt/预算/实现及资格输出；无付费 API、购买算力或外发/投稿授权。每次 dev 研究资源先走唯一 owner reservation，失败/unknown 结算保留；`budget={}` 不证明免费或资源已够。

## 冻结 evaluator 与实际记录复验

[evaluator.v1.py](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/experiment/evaluator.v1.py:19) 在导入前核 manifest 固定 SHA 和 17 份本地 Python 源 SHA，再核 world 的 caller-supplied expected SHA。17 项全部匹配；对其 AST 的本地 envs/eval imports 检查未发现漏列模块。实际运行使用 `.venv` Python 3.11、NumPy 1.26.4；manifest 记录 Python 与 NumPy 版本，NumPy 在入口实际核对。该 manifest 不是完整 PyFlyt/PyBullet/NumPy 二进制字节锁；原物理 source/61deps 资格及真实新 episode admission 仍是另外的范围。

它实际重建 delivered observations→sealed report ledger，使用测量时刻几何/类别与完整静态/60 个动态格 grader、覆盖分母、原硬安全/退出/剩余航程合成 C/S/D/Q/H/E/Y/SR；L1 使用 D=null 和 (C+S)/2。未实例化 world/mission/aircraft，未推进物理。

本轮在新进程独立执行了 `evaluator-actual-recount.v1.json` 中 **全部 7 份保存 world**：7/7 terminal SHA 匹配，world SHA 经实际入口验证，全部完整 score 与保存 recount 逐字段相等；不是只相信 `score_equal:true`。这些原记录包含 L1、L2、L3 与 B0/B1/B2 开发结果，不是正式主矩阵或新 rollout。

隔离临时副本的三项反例均拒绝：错误 expected-world SHA；manifest 仅追加换行；scoring.py 仅追加注释。三者分别命中 world、manifest、source guard，不由同一失败掩盖。原 evaluator、manifest 和 closure 均未修改。

**信任边界保留：** [inspection_common_calendar.py:48](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/experiment/evaluator-source.v1/eval/inspection_common_calendar.py:48) 使用保存 coverage/required_cells；第 54 行使用 `world.score.remaining_range_m`，第 58 行使用保存 violations/exit_dwell。它不是从 240Hz 全状态独立重建里程/安全，也不验证全部原 episode 的 source/clock/opportunity/admission；入口文档及输出明确“offline recount, not provenance admission”。caller 自报 expected hash 不能代替外部受信 receipt。上述功能是实际 score 重计而不是占位 grader，仍不能拿它替代完整 episode admission/G3/G4；本报告没有把相等分数叫作全物理重放。

## 版本绑定、重现与限制

R 为 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection`。除下表外，17 份 closure 的 exact SHA 由固定 `evaluator-source-manifest.v1.json` 绑定并逐一复核。

| 来源 | SHA-256 |
| --- | --- |
| `packets/plan.ral.v1.json` | `12a538c127ed25c0d12b076cc9d77482a18924702705756625669d112afb67f1` |
| `experiment/design.ral.v1.json` | `2195c892d83302a2908f0a8041fd6b5b54e757ecff494b701411c324f58dbf51` |
| `experiment/evaluator.v1.py` | `1716592abd6a7ea317aebb90e41367f338e7f32ac7ab3f0a69a9d2bd4ff02a3d` |
| `experiment/evaluator-source-manifest.v1.json` | `ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80` |
| `qualification/evaluator-actual-recount.v1.json` | `b53fbd3b4459200788ea87b486e8d29d4e18a31ae2227d18981db216e43c2d3b` |
| `planning/local-model-registry.v1.json` | `6d8547cb0597113d5411a771e1c75f807e43d769ecdfe4d92d32d41633e0fda6` |
| `planning/statistical-protocol.ral.v1.md` | `b18123390584f3f260d9bcc80b0455402ccc68f606ad428fe06804e04d2e795f` |
| `planning/plan-snapshot.ral.v1.json` | `720458795df000ca4c9cc08ae5022c80f2e6df6cc81e8aa3b68aa2b608edfd17` |
| `planning/figure-table-plan.ral.v1.json` | `553c47c76cd9d8a9c1229db7f63a582295ab1e261b5a0f9572af33fc57aa0930` |
| repo `eval/inspection_recording.py` | `9613a8e800d470110dbf86890837ba853c3f50834ae29f0722994927d27cb352` |

SDK mismatch 可直接纯 CPU 重现：读取冻结 design，分别以 B1/B2 调用现 `inspection_recording.designs(d['cases'],d,best)['root-means']`，比较 exact dict/hash；无需运行 callback 或物理。实际重计可用 `.venv/bin/python R/experiment/evaluator.v1.py --world WORLD --expected-world-sha256 SHA`，7 个 WORLD 由 recount 各行 **repo-relative** terminal 的同目录 `flight/world.json` 确定，不是 R-relative logs。临时反例与首批结果摘要在 `/var/folders/n_/2cw9gf5n52z_w6kllyb44ttw0000gn/T/inspection-initial-plan-review-v1-_mdw4jzb/summary.json`；其记录前两份实际重计与三个 guard，另外五份实际重计也均完成通过，未将该摘要冒称全七份执行日志。

观察时 state 为 `round-18`，accepted stages 仍 intake/venue，frozen_plan=null、ledger0；本 Agent 没有调用修改状态的 CLI 或 lock/load/save。最后只读 control/checkpoint SHA 为 `0cb41c7264ee502fce88ee77cb2bfafbfa3d69ab53b964f089f7f7c562182371` / `40f2338f63822cfc983fc1fed32f18ae439d830d906c44c9a494b243622dfe7c`。

只新增本报告及自有隔离临时证据，保留 v1 原件、旧失败与他人改动；未改生产/冻结经典或核心、`.paper`、live logs，未生成模型调用、新物理或正式数据。主 Agent 已表示将保留 v1 并准备 v2；未来版本须另做有界复验，不能把本报告的实际 grader/分母验证回填成 v1 plan acceptance 或 G4 通过。
