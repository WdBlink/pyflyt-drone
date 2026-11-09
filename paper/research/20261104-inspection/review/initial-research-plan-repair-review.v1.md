# 初始计划 v3 定向修复复验

审查者 `/root/control_review`；同模型家族、独立工程上下文。仅复验 [v1 报告](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/initial-research-plan-review.v1.md:1) 的三项 P2，不改旧报告/源/结果/owner，不重跑模型或物理。

## 结论

**原三项 P2 在 v3 的计划约束与记录兼容范围均已关闭。** 本轮未发现仍阻止完整初始科研计划接受的这三项残余缺陷。v3 可以进入唯一 owner 的正常初始 plan 接受流程；本结论不等于已经 commit，也不授 G4、实际模型调用、正式 main/holdout、写作或付费 GO。

本轮绑定的 evaluator.v2 正由另一独立审查复验导入安全修复。最终接受还应合并其对 **同一 SHA** 的结论；本报告不代其关闭 v1 的 package/bytecode shadow P1。若该独立复验仍有开放阻断，不能只凭本报告接受整体包。v1 原结论与失败证据保留，不回填为 v1 合格。

## P2-1：总体及次级实验定义——关闭

[population.ral.v1.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/population.ral.v1.json:11) 现给出 ID 的九顶点仿射/jitter 族、目标/normal 数量与布局、动态出现/速度/heading/radius/bend、事件日历；第 20 行给出不同的九角径向 OOD 图族、非条带静态布局、弯曲闭环轨迹及重叠链路/相机事件。没有只把 OOD 写成新 seed，也没有声称是真实道路/地形 OOD。

第 5–9、28–38 行规定独立命名空间、PRNG 子流、lineage/semantic/seed 重复 guard、固定风与传感器/物理范围、完整时间的几何/身份 45m/容量必要条件、有限候选/placement 上限及失败保存；抽样接受只能依这些前瞻条件，不依策略成绩。当前旧 dev inputs 不被倒称为满足新 identity45m。实际生成器及其新 ID/OOD 开发命名空间仍待 G4 实现/资格，不能把文本合同叫作已完成场景可解性证明。

[design.ral.v3.json:599](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/experiment/design.ral.v3.json:599) 通过 exact SHA 引用该总体；第 560 行明确冻结 grader、endpoints、population、counts 与主要家庭。G4 只能证明兼容实现并冻结其源/model/prompt manifest；不能选择未定义的新总体或用 manifest 替换这些规则。此修复消除了 v1 “scientific population 不可变，却 G4 前再选分布”的冲突。

次级规则也已具体化，见设计第 608–654 行：

- 同一预选十二个 L3 roots 上，八条件按各 1/3 重复追加 clean siblings，共 216；噪声/几何/seed 与 lineage 保持配对，原 event score 仍为 primary，共日历只作离线配对诊断。
- 三个单因素消融为 B4-M1×3×12=108；固定周期为 20 sim seconds，调用/token/时间限制与其他机制保持，单因素若在 baseline 中实际不存在则 G4 阻断，不能伪称有效消融。
- 延迟为 0/2/5/10/20s×3×12=180 个 arm slots；实际模型回复后额外等待固定 wall time，同一真实 1× worker 继续推进，不缓存回复或加速逻辑时钟。零延迟最多 36 条只能复用 exact-compatible primary trace，其余是额外真实飞行；实际服务、端到端、额外注入及派发年龄分别保存。
- 响应以事件后正确 rule/card 的真实合格新测量业务证据或合格静态报告修订为准；动态恢复使用原完整格中连续十格的质量判据，静态恢复须保持三十秒；逐 target deadline、右删失、NA/eligible 分母、不同 follow-up horizon 和 root 聚类均预先声明。剩余窗口不足三十秒不删除 eligible target。

这些是可实现的前瞻定义，尚未成为 actual loader/事件分析通过证据。G4 handfixtures 必须按已写的“完成”与固定 deadline 核验完整格的及时性裁决/三十秒 follow-up；截止前未完成就删失，不能以截短窗口补称恢复。新的响应端点是“合格业务证据响应”，不能在论文中误标为 ACK、首次 running 或更早的配置采用时刻。这里保留其定义边界，不额外降低规则或把 G4 未实现本身重新设为初始 plan 前提。

## P2-2：Exact canonical SDK 交接——关闭

[inspection_recording.py:78](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_recording.py:78) 新增 `frozen_outer_design`。指定该路径时核对完整八条件/1或3重复/cases/seed0；第 113 行不重写外层设计字节，第 121 行让原生 ExperimentRecorder 直接读取该原路径。旧生成式 route 保留；v3 的四个 descriptive option contrasts 因而不再被旧三-contrast 模板覆盖。

对 `canonical-sdk-preflight.v2/complete-sdk-records.zip` 的独立只读全量核验：

| 项目 | 独立复算 |
| --- | --- |
| ZIP SHA、两份 source snapshots | 与报告绑定一致；snapshot recording.py 等于当前实现 |
| canonical design | ZIP 内与实际 v3 文件逐字节相同，SHA `83f8f1a7…` |
| classic inner | 216 条，case/repeat/condition 唯一且恰好覆盖预期集合，全部 completed |
| LLM inner | 1080 条，恰好完整集合；fixture repeat0/1/2 为 0/.25/.5 |
| outer | 576 条完整集合；classic 为 0、LLM 为 .25，不复制 classic repeat |
| 逐 outer 聚合 | 每条恰好引用 1/3 条对应 inner，inner raw/results SHA、身份、均值及 schedule SHA 全部一致 |
| 原生科学校验 | 在自有临时目录对原 ZIP 资料重新调用 `science.validate('experiment', packet, tmp)`，返回 `[]`，实际绑定同一 v3 frozen hash |

未只抄 report 的 `passed:true`。独立检查覆盖 classic/LLM/root-means 的全部设计 SHA、实际集合、raw 值与聚合证据；没有执行任何 `measure` 物理或模型调用。合成 1296 callbacks/576 均值不是 1296 次已飞数据，fixture B1 不是 dev-selected B_classic*，也没有代替 primary Holm6。

另做一项实际异常注入：仅在自有 **2-root 软件 fixture** 使用 canonical route，让首个 callback 抛 `RuntimeError`。异常传播，classic 保留 6 条 stopped、LLM 保留 1 failed+29 stopped、outer 保留 16 条 stopped；三层 execution_status 均 failed，未产生合法 root means 或补零成功。原 72-root/1296 的官方合成归档未改写。

临时证据目录：`/var/folders/n_/2cw9gf5n52z_w6kllyb44ttw0000gn/T/inspection-plan-v3-review-m3yi3b56/` 为 ZIP 原记录的隔离展开及 native 校验输入；`/var/folders/n_/2cw9gf5n52z_w6kllyb44ttw0000gn/T/inspection-canonical-failure-review-snwijmtd/` 为实际异常分母记录。均非 owner 或真实飞行日志。

## P2-3：开发/G4 WorkOrder 交接——关闭

[plan.ral.v3.json:43](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/packets/plan.ral.v3.json:43) 新增 `W-development-G4`，映射 F1/F2/T2，声明 G1–G3/六根 dev pilot 依赖、设计/Brief/原计划输入、qualification packet 输出与源绑定独立验收。共五单均补 scope、inputs、outputs、dependencies、acceptance。后续 experiment 明确依赖 G4 manifest 及完整主/holdout/次级证据；图表/写作依赖完整已接受证据。

实际覆盖仍恰好全部 `F1–F6,T1–T4,S0–S7,M1` 十九项，没有新增未知 ID 或遗漏；使用既有 validate_packet/compile_tex 注册动作，无虚构 `run_llm`。这解决计划中的任务交接，不证明将来的 qualification packet 已存在或通过，也不证明 runtime 自动解释这些自然语言 dependencies；host 和实际入口仍须执行 G4、资源与正式数据访问门。

## 不变分母、资源与 evaluator 边界

八条件/重复、72 main 分层与 12 holdout 保持不变，独立重算为主 1296、holdout 216；clean 216、消融 108、delay 180（零 arm 可兼容复用最多 36）明确另计。主Y、等级宏平均、两个同模型架构差及 dev-selected classic 的 globalHolm6、失败/unknown/有限对称重试均未被此修复降低。默认 budget={}、现有本机与本地权重范围不变，不授权付费 API、采购硬件/算力或外发/投稿；实际 dev 操作仍须 owner 接受计划后、操作前正常 reserve 与实际结算。

v3 actual evaluator 绑定为 **v2**，SHA `de826d24…`，不再指向 v1。核对了当前 manifest 列出的 17 项本地源 SHA，目录实际文件集合恰好这 17 项；v2 recount 中七行全部绑定该新入口 SHA。没有重新执行这七份 v2 score，因为另一独立审查正在做此工作。

另一个审查的 [frozen-evaluator-review.v1.md](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/frozen-evaluator-review.v1.md:22) 证明 v1 package/pyc shadow P1；此前本 Agent 的 AST 正向依赖检查并没有覆盖这种实际 import 替换，因此不能把它当闭包安全认证。v2 严格文件集合/no-symlink/dont-write 修复的完整反例结论由该审查负责，本报告不抢先关闭。

原 evaluator 的业务重计仍依赖上游保存的硬安全/退出/240Hz-hook remaining range 等实际审计，不能从 10Hz位置补造精确里程，也不是完整物理状态重放。这里没有重新授予 G1/G2/G3 或实际场景总体资格。

## 冻结绑定与限制

R=`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection`。

| R 相对文件 | SHA-256 |
| --- | --- |
| `packets/plan.ral.v3.json` | `5c1a0278332c4ee6e4f0546530a56842c30c18c5fcdb015934debb5b6a864b78` |
| `experiment/design.ral.v3.json` | `83f8f1a72f5261aaa5783c0d190f3788fafcd05bca1e7d16a6901c69c9abf0de` |
| `planning/population.ral.v1.json` | `307fbd5e77f841bf4a1b431a478527fa3aa85905694fd0242068d26c729a434c` |
| `experiment/evaluator.v2.py` | `de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b` |
| `experiment/evaluator-source-manifest.v2.json` | `ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80` |
| `qualification/evaluator-actual-recount.v2.json` | `b822934f04fad9371e1f9ab23a8337215fa78237da9516b863c91c470d71ffe4` |
| `qualification/canonical-sdk-preflight.v2/report.json` | `70755750f9f5dde313eeb965bf2e1e1707291dd91f5e96865efe716df5b447a1` |
| `qualification/canonical-sdk-preflight.v2/complete-sdk-records.zip` | `acf45edb77811940b71c54b030929ff3b7a3f7755722e287cb4eb3045bafa174` |
| `qualification/canonical-sdk-preflight.v2/inspection_recording.py` | `a2fbdddcf7546b3f69f1cc8d2fc1d565620be3ce6582c6f61f8e93923decffcb` |
| `qualification/canonical-sdk-preflight.v2/preflight_inspection_canonical_recording.py` | `690f2578d3fc3977f7a8155887d134b6700f52654e2e48ff8c78de998e944fe2` |
| `review/frozen-evaluator-review.v1.md` | `b99c37e742086fc75acd2e85d9466e1b93da754dc50b1c66062a313c6af43102` |

观察时 owner 仍为 round-18、receipts=intake/venue、frozen_plan=null、ledger0；本 Agent 未作 commit/reserve 或 owner 写操作。只新增本报告及自有 CPU 软件 fixture/归档检查临时文件，没有修改旧报告、production、冻结经典/核心、live logs 或真实状态，也未运行新物理/模型。该结论限完整初始研究计划的定向修复，不是 G4 完成、正式性能、领域人工审查或跨模型认证。
