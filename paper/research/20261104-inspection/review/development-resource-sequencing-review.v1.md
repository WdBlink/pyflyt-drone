# 开发资源与科研阶段顺序独立复核 v1

审查者：`/root/control_review`，同模型家族的独立工程上下文；不是人类领域评审、跨模型认证或用户批准。审查对象为当前 run 的流程与资源边界，未重新授予 G1–G4、正式实验、付费或论文完成资格。

## 结论与当前阻塞

“接受完整初始科研执行计划（含 dev 校准工作单及 G4 硬门）→预约已授权的本地 dev 模型资源→完成 G4→建立/开封正式 main/holdout 并运行”的阶段顺序，**不违反所读 Skill 的通用顺序**。初始计划接受、开发资源预约、G4 科学验收和正式运行是四件不同的事。规则没有要求所有待研究结果或 G4 先于一切初始计划；规则要求先有有据的可行路线、完整计划及资源登记，再做相应操作。

但**当前原 intake 阻断 plan，当前不能预约模型生成资源**。已接受 intake 的 `feasibility.status` 是 `conditional`，原 prerequisites 把 G1–G4 列在其中；`planning.plan()` 对该字段明确要求 `feasible`，否则报 `Conditional or blocked feasibility cannot become a research plan without resolving intake`。这不是 budget 为 `{}` 引起的障碍，也不能通过先预约、unreserved settle、只加 WorkOrder 或修改 `.paper` 消除。

合法处理方式是正常、保留历史的 intake 重审：区分“现有本地资源内可以推进研究/开发资格”与“G4 后正式 main 才可运行”，用新授权与实际开发证据重新评估前者，完整保留原 conditional、原 prerequisites 和旧失败，继续把未完成的 G4 作为后者的硬门。**本报告不声称 G4 已过，也不接受尚不存在的 plan packet，不授 GO。**

## 实际读取范围与状态

工作根 `R=/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection`。当前唯一入口由 [ACTIVE_CONTROLLER.md](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/ACTIVE_CONTROLLER.md:3) 指向 `R/runtime/autoresearch-state-reader-v1/scripts/research.py`；全局安装不是当前 writer。所读规则为 autoresearch Skill 及该 pinned runtime 的 plan、packets、budget、user-input、production、runtime contract；流程按既有独立 Verify/show-me-your-work 边界执行。

在 `2026-11-05T16:35Z` 附近只读实际 control：schema 2，implementation `a112e7aaff3ab6c99f983a2030fc190d7ccd1af940a943604d94ad318e8fef20`，`active_round.id=round-16`；只有 intake/venue 两份 stage receipt，`frozen_plan=null`，usage ledger 空，无 running/uncertain attempt。批准 Brief SHA 为 `a4cf4597fe0bc584c95f1277717c34e026c82bfa6cca8daa566584c5b48c9bd6`，现 approval 绑定该 SHA；预算为 `{}`。

19 项义务实际为 `F1–F6,T1–T4,S0–S7,M1`，不能只用十张图表替代。当前图表文件是 `proposed_not_frozen_no_results`，包含十项原视觉承诺及待采集数据；[context.ral.v3.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/context.ral.v3.json:3719) 的 WorkOrders 仍为空。对 `planning/`、`inputs/` 的定向查找及当前 receipts 未发现已提交的完整初始 plan packet；此结论不覆盖未来新增文件。

最新范围记录 [qualification-authorization.v1.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/inputs/qualification-authorization.v1.json:3) 明确要求先完成非 LLM 参考任务和 G1–G4，再进入正式实验及论文写作；其 scope 允许本地实施与资格修复，未改变 19 项义务，也不授权付费 API、购买算力、外发、投稿。读取了该实际指令记录及既有 scope/回答来源，没有扫描无关聊天或敏感凭据；未取得完整聊天 transcript，不能独立证明文本作者身份或补全未提供的授权。

G1–G3 的既有独立资格报告继续只在其冻结包线与组合条件内成立，本轮不重计物理记录。当前 [g4-progress.v1.md](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/qualification/g4-progress.v1.md:3) 明确 G4 `needs_work`。24 个 dev roots、实际 L1/L3 和 replay 是开发路线证据，不是完整六根 pilot 或八条件主结果。三份本机 load 记录均为 `generation_calls=0`、`generated_tokens=0`，只证明结构 load/tokenize；不能证明推理延迟、时钟竞争、策略能力或正式 M1/M2/M3 已选定。Mistral tokenizer 配置仍须修复/复验，参见该进度文件第 36 行。

## 精确规则与实现条件

| 检查点 | 来源与实际条件 | 对本研究的影响 |
| --- | --- | --- |
| intake 可行性 | [packets.md:78](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/packets.md:78) 要求真实 resources、experiments、未解决条件；[planning.py:291](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/planning.py:291) 拒绝 accepted conditional/blocked intake 的 plan | 目前的直接阻塞。原 [intake.v1.json:41](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/packets/intake.v1.json:41) 与 control 内 accepted packet 完全相同，prerequisites 在第 69 行；不能当作未被接受的草稿忽略 |
| 有据的 feasible 路线 | [planning.py:162](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/planning.py:162) 要求选中的 feasible option 有实际读过的证据，不能只标 assumption；[runtime.py:839](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:839) 要求 readable 材料具备完整 controller `read-material` receipts | 只改 status 字符串不够；新增授权、资格/开发路线材料须按真实读取范围登记，未读材料继续标明限制 |
| 完整初始计划 | [plan.md:9](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/knowledge/plan.md:9)、[packets.md:129](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/packets.md:129) 要求 snapshot、WorkOrders、经验研究设计/evaluator；[runtime.py:859](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:859) 拒绝空订单、未知/遗漏 requirement ID、无效动作 | dev 订单可属于完整科研计划，但不能只冻结一个 dev 小计划而遗漏后续正式研究和稿件义务 |
| 计划冻结 | [runtime.py:851](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:851) 在已有 frozen plan 时拒绝不同 packet；第 922 行冻结整个 packet digest；[science.py:1006](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/science.py:1006) 要求后续真实设计字节 SHA 与 frozen plan 一致 | 不能先用 placeholder hash 解锁预约，再 G4 后静默更换实际设计或测量函数 |
| 资源预约 | [runtime.py:453](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:453) 依次检查 ready、budget answer、usage evidence；第 458 行要求 frozen plan 和 plan receipt，第 460 行要求 active round | 当前虽然 round-16 active，仍因无 plan receipt/frozen plan 不可预约；G4 本身不是该方法的专门字段检查 |
| 默认预算含义 | [budget.md:3](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/budget.md:3)、第 50–60 行要求操作前预约、核验各有上限 meter 的最大值；预约本身不授服务权限 | `{}` 没有隐含研究总量上限；也不意味着免费、无限算力或可支付。实际本地 dev 操作仍需符合最新 scope、资源解释和真实结算 |
| settle 的边界 | [budget.md:68](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/budget.md:68) 保留失败及 actual usage；第 75–86 行说明 unreserved 仅审计已经发生用量、未知用量需对账 | 不得以无 reservation 的 settle 作为“先生成再补账”路径；未测量的能耗/本机总成本应保持 unreported，不填零 |
| 执行与科学门 | [plan.md:19](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/knowledge/plan.md:19) 要求在科学/资源前置结果出来后重判断；[packets.md:156](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/packets.md:156) 允许 host 运行已授权研究，completed WorkOrder 不等于 fulfillment | typed actions 只有 validate/copy/render/TeX/image 等六类，没有任意 shell/Python 或模型生成动作；可用合法订单收集/校验 host dev 证据，不能伪造注册能力 |

## 正常重审与既有回答复用

以下是 owner 可执行前须准备的路径说明，本轮没有执行这些命令或改写状态。

1. **准备新版本重审材料。** 保存旧 `packets/intake.v1.json`、`intake/feasibility.v1.md` 和原 prerequisites，不在旧文件上改成“全部满足”。新 intake 明确：被评估为可推进的是在现有本地资源内实施 dev 资格的研究路线；G4、正式 split/model/prompt/预算冻结、正式实验和结果写作仍有后续硬门。对原四项 prerequisites 逐条写成已解决证据、仍未满足条件、适用阶段及变更理由。最新授权、实际 G1–G3、真实 dev L1/L3 和本机固定权重 load 可以支持这个重审；它们不能支持“正式八条件已可运行”或“G4 已完成”。若仍无可信、可执行的 dev 解决路线，就应继续 conditional，不能仅换措辞绕过。
2. **通过唯一 owner `rewind intake`，使用新的 operation ID 和明确 reason。** [runtime.py:944](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:944) 会移除该 stage 及后续 receipts；第 961 行把完整旧 receipts 和 reason 追加到 invalidated history。若要修改已接受的 artifact，`--preserve` 只能在其仍与 accepted hash 相同时先做版本化保留；第 951–957 行拒绝对已改字节补称旧 baseline。当前只有 intake/venue，所以两者都须重新提交。当前 active round 保留；当前 frozen plan 为 null，此操作没有既有初始计划可解除。不得初始化新账本、清 ledger 或手写 `.paper`。
3. **真实读取新材料，提交新版 intake。** controller read-material receipts 与 quoted spans 必须绑定新证据字节；`feasible` 的依据是可执行研究路线，未来正式门保持显式，原 conditional/prerequisites 在 history 和版本化文件中可追溯。
4. **重新提交 venue acceptance。** 旧 venue receipt 随 rewind 失效，并不意味着实际用户的 RA-L 选择被撤回。`rewind` 第 958–962 行未清除 `venue_confirmation`、`budget_confirmation`、approval 或 usage ledger。可复用原材料和实际回答，但新的 venue commit 仍重跑正常验证，不把失效 receipt 当当前接受状态。若比较、候选、来源字节或实际选择改变，须按真实新选择重新核验/问答。
5. **完整初始 plan 准备好后，走正常 validate/commit，再预约 scoped dev 生成资源。** `runtime.commit()` 第 914–920 行仍检查阶段顺序、venue 或 budget 绑定与全部验证。不得因为本报告认为“顺序原则可行”而跳过这些条件。

对回答复用进行了实际、只读 helper 检查：以真实 state 和当前 accepted venue packet 调用 `Controller.confirmed_venue`、`Controller.confirmed_budget`，两者通过；未调用 lock/load/commit/save 或发起问答。原 venue comparison digest 仍为 `95ac6c58fad6812bcf55e640a55ba0cfc816e2f9df0a0d4e20e2b4a6761ad82a`。venue/budget 实际回答文件 SHA 和 quote 均匹配现存字节；同 budget digest、Brief 除 budget 的 contract digest、图表计划路径及字节也匹配。纯 helper 前后 control 文件 SHA 相同。

复用条件具体如下：

- [user-input.md:12](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/user-input.md:12) 允许复用同一具体选择的真实原文，不为创建新记录重复提问。`confirmed_venue` [runtime.py:400](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:400) 仍核实原回答和 exact comparison digest。
- 现 budget 回答绑定 `planning/figure-table-plan.ral.v1.json`，SHA `553c47c76cd9d8a9c1229db7f63a582295ab1e261b5a0f9572af33fc57aa0930`；`{}` digest `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`；Brief 除 budget 的 digest `3a184d38995a78295e3812f93031a8a77c5fbebb8948d1b23d7439224f487954`。仅用新版本 intake/单独完整 stage packet补齐同一计划的执行细节，并不自动使该 budget answer 失效。
- 若实际改变该计划版本/路径/字节、Brief 契约、预算或具体资源选择，则不能以“还是 `{}`”复用不再相同的回答。[user-input.md:74](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/user-input.md:74) 与 [runtime.py:430](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:430) 要求当前绑定。新增付费或扩大资源权限还需其真实授权；budget 确认不代替它。
- 原 venue 的读图/模板/calibration 文件本轮未重新做科学审查；重新 commit 能否通过仍取决于其当前 exact artifacts 和正常 validator。复用选择回答不等于免验 venue packet。

## 初始计划仍需补齐的 artifact

当前十项图表列表不是完整初始科研执行计划。接受前至少需要以下具体交接；本报告不创建这些计划、替其验收或判定尚未提供的 packet 合格。

1. **原字节 plan snapshot 与完整工作分解。** 绑定现 Brief SHA、图表原承诺集和 snapshot；WorkOrders 覆盖全部 19 IDs，明确 producer、实际输入/输出、dev→G4→formal→figures/write 的依赖及失败路由。dev 校准可服务 F1/F2/T2/S3/S4 等义务，后续正式研究、统计和稿件也必须保留。用 registered actions 校验/收集实际 host 证据；不要添加虚构 `run_llm` action 或把完成 audit 作为 GO。
2. **经验研究 design、实际 evaluator 与 recording envelope。** [packets.md:135](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/packets.md:135) 明确在 plan 接受前创建，plan 必须带 `experiment_design_sha256`。[production.md:78](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/production.md:78) 要求先设计、生成 envelope、填写真实 measure，再冻结 evaluator；第 110 行要求完整条件矩阵预检。八配置不可删减成两臂来迎合旧 adapter。合成矩阵只能证明 SDK/分母/记录接线，不能冒充物理或模型运行，也不要求在 G4 前先运行正式八条件。
3. **前瞻性的 dev 选择与正式隔离。** 可在初始设计冻结 dev-only 选择程序、预选六根、候选范围、prompt/token/超时/重试校准规则、G4 验收和正式命名空间/生成规则；尚未产生的 test 实例不能冒充现成数据。正式模型/配置 manifest 的后续产生必须是该前瞻程序的可审计输出，不能任意追改已冻结 design/evaluator。当前尚不存在的具体 late-binding 方案是否充分，须接受前实际检查；本报告不以“可将来补”判其通过。
4. **资源解释与可核验操作记录。** 在现有本机 scope 下明确哪些是 load、哪些是 dev generation/episode；每次实际研究资源操作前正常 reserve，说明 purpose/dev-only、最大调用/token上界、模型/revision/配置与 receipt 路径；失败与 unknown 一并保留，实际 settle。无 capped meter 不等于无需保留真实记录。本机 API 账单为零不能写成计算无成本，能耗/资本成本未计量应披露。
5. **可执行的后续门。** G4 前禁止生成/开封实际 main/holdout，禁止正式成绩/论文结果写作。G4 后也须重新核验正式 source/model/prompt/预算/split/evaluator/同信息与重试、失败完整分母等实际条件，才可启动其已授权部分。现有 `check_orders()` 检查覆盖与动作字段，`reserve_budget()` 检查资源前提；它们并不理解任意额外 WorkOrder 依赖文本或自动验证领域 G4。host/正式入口必须落实该门，不能把预算预约成功当领域放行。本轮未扩大到审计尚未完成的正式模型执行入口。

这里有一项**规则与机械验证的差距**需要保留：`planning.plan()` 目前检查 feasible intake、snapshot、十项视觉义务及 venue；`science.validate()` [science.py:1124](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/science.py:1124) 没有 plan handler。因此少 `experiment_design_sha256` 的 packet 未必在初始 plan 阶段机械报错，但 experiment/figures 阶段必查 frozen hash。这个漏检不能覆盖 packets/production 的明确前置要求，也不能用来冻结无设计计划后解锁模型资源。初始冻结的 exact packet 之后不能靠 rewind 任意替换；同契约 `revise-orders` 只追加新版本订单及失败历史，不解除原承诺/设计约束，见 [packets.md:161](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/packets.md:161)。

## 证据绑定与未证范围

只进行文件读取、定向文本搜索、标准库摘要/hash及两个只读确认 helper；没有生成、forward、physics、paid call，没有 runtime/Brief/计划/规则/旧结果或真实 state 变更。control 读得 SHA 为 `2734edb3b148fefefe5dca92aa34f729b7c1cfe855f746080cd7e18ab9635ef7`，checkpoint 文件 SHA 为 `95480e64cac0fe53f0097a7cfa7c86ee199cc48416e581f9848d2ce8b14c4834`。这些是观察时字节绑定，后续正常 checkpoint 更新不自动使本报告陈述失效或证明新状态已复验。

| R 相对来源 | SHA-256 |
| --- | --- |
| `ACTIVE_CONTROLLER.md` | `89403b9440b4339d855ff8f389fedb2acc2de05655b6502e0493a6f9ea3925c2` |
| `packets/intake.v1.json` | `4878f6dd7cd6bdc75ca7cd02f74ce63d26ef8ab3bd603ab3a6d2aa7ed280898e` |
| `inputs/qualification-authorization.v1.json` | `dfa0dec75486794b0c45cee0d8526e02f68f125d763048a158708ff7a8754df2` |
| `planning/figure-table-plan.ral.v1.json` | `553c47c76cd9d8a9c1229db7f63a582295ab1e261b5a0f9572af33fc57aa0930` |
| `qualification/g4-progress.v1.md` | `128b804b873aee9619efe951d2d3fb824ee579c50b52a1ad6922df1822980861` |
| `planning/scenario-adapter.ral.v1.md` | `235ffc59830484392f990929acd0cd9771190fa39dad27267cd470d090d45f57` |
| `runtime/autoresearch-state-reader-v1/lib/runtime.py` | `5f887d7b188203a3beda96127b055d4e5009017eced0533f55afacc2a3c19f52` |
| `runtime/autoresearch-state-reader-v1/lib/planning.py` | `f294697792d60744808ed90fe989b91a04eba49403a41499aca082474e7e7b0b` |
| `runtime/autoresearch-state-reader-v1/lib/science.py` | `9f7a8eecb2741f18afaccfb3da1ba33951d88c2bd4fd1983dfc1b5953649cfaf` |
| `runtime/autoresearch-state-reader-v1/references/packets.md` | `bf2111d960d50a2cddc7915d5a11faada12004f688e037094ebd0c3bb0be7637` |
| `runtime/autoresearch-state-reader-v1/references/budget.md` | `12a26683292e119e9fad9fc4c79896ab2779a5bb16a86017e644e930513a5fd3` |
| `runtime/autoresearch-state-reader-v1/references/user-input.md` | `54df703670b24a9d586adf3efd19b6f372b14121be2a2324750d5c53a17f2639` |
| `runtime/autoresearch-state-reader-v1/references/knowledge/plan.md` | `c28278c06a48e68b76562e132cd41c5be647f1f59f4be7a52c7b75d490fe6173` |
| `runtime/autoresearch-state-reader-v1/references/production.md` | `fc2310f7f991b2e473bf3886bb4276ee10741a11e04d0dda81568a87351229fd` |

该报告完成的是阶段/资源治理判断，不是对未提供的初始计划、完整 G4、正式科学性能或人类领域审查的通过判断。G4 开发缺口的准确当前状态以其版本化实际记录和独立复验为准；本轮只核对其与流程边界的关系，不重新实现或复验全部开发模块。
