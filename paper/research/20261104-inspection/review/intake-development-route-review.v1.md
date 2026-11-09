# Intake 本地开发路线重审 v1

审查者：`/root/control_review`，同模型家族、独立上下文的工程 Verify。仅审查具体新版 intake 的证据与阶段边界；不是用户批准、G4 验收、plan acceptance 或正式运行 GO。

## 结论

修复材料清单遗漏后的 [intake.v2.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/packets/intake.v2.json:1)（SHA `ce3b5574c63324f178b3bd6817f9c5f3601c6c30581f747291a9427c3b30b591`）与 [feasibility.v2.md](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/intake/feasibility.v2.md:3) 在本轮有界审查内，真实区分了“既有授权/本机资源内可以开展完整研究与 dev 校准”及“G4 尚未通过，正式 main/holdout 仍不得建立、开封或运行”。当前未发现必须先改正的实质开放缺陷；可以作为正常、保留历史的 controller intake 重审输入。

这不等于接受后续完整计划或允许立即生成。当前 state 仍接受旧 conditional intake/venue，`frozen_plan=null`、usage ledger 空；必须由唯一 controller 正常 rewind/recommit 后重新验证。完整初始 plan、设计/evaluator、19 项 WorkOrders 覆盖、资源预约和 G4 后续硬门仍未被本报告替代。

## 已发现并修复的具体问题

**P2，已关闭：两个 readable material 最初未列入 artifacts。** 初读 packet SHA 为 `b75808d59e0baaaf7a00a24fd2464a51b4967d8a7d81893fb9cef73ce42560aa`；`qualification/g4-progress.v1.md` 和 `inputs/qualification-authorization.v1.json` 当时出现在 materials，却不在 artifacts。实际 [planning.py:66](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/planning.py:66) 会拒绝该材料，且关键进度/范围文件不能进入 packet artifact closure。

已向主 Agent 提出最小修复；主 Agent 只补齐这两个路径。现 [intake.v2.json:12](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/packets/intake.v2.json:12) 包含两者，全部 material 路径都在 artifacts 内；材料字节、SHA、span 与已有 read-material receipts 未变。当前 packet 的纯 `planning.intake()` 校验通过。该机械检查只是补充，以下结论还基于内容与实际证据读取。

## Span、quote、hash 与实际读取

工作根 `R=/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection`。只读实际 control 的 material_reads；下列四份 readable 材料均有同路径、同 SHA、从 0 到文件末尾的实际 controller receipt，packet read_spans 与字符长度一致。read-material 的区间是 Python 文本字符位置，不是 UTF-8 字节数。

| material | 完整字符区间 | SHA-256 |
| --- | --- | --- |
| `inputs/plan-excerpts.md` | `[0,472)` | `d428747fcc1e26af7eae37d036ee10453333fb95c46e7c1e718ad1f3275ebc01` |
| `intake/feasibility.v2.md` | `[0,2760)` | `57d8ccb71928c240cfe8747cb961460616169fc30ea6ebefc843e79996e4dc0d` |
| `qualification/g4-progress.v1.md` | `[0,3710)` | `2341ee9ba3ed9efa7c3d2bae81848b6a333eaa84d971cd75efe9950def420a18` |
| `inputs/qualification-authorization.v1.json` | `[0,787)` | `dfa0dec75486794b0c45cee0d8526e02f68f125d763048a158708ff7a8754df2` |

packet.evidence 与 selected option 各使用的两组 quote 均精确等于材料切片，并落在已读区间内：`plan-extract[101:124]` 为“尚不是已运行 benchmark 或实验结果。”；`feasibility-update[34:74]` 为“在已授权的既有本机资源内，可以继续实施这项实证benchmark研究及开发校准。”。后一条是作者的总结判断，不能单独证明可行性；本轮另读并核对了下述实际记录，没有只凭该总结或 validator 通过背书。

完整原计划仍明确 `metadata_only`、read_spans 为空、保留未逐字复核历史附录的限制，没有把旧 controller receipts 冒充完整计划阅读。原文件 SHA `dd01e21442950abccc543bb2ba4d53d1901fb2669dba21cbfe9aad7f13e591b0` 匹配当前字节。

## 内容与证据核对

- **原 conditional 与 prerequisites 保留。** 当前 accepted intake packet 仍精确等于 `packets/intake.v1.json`；旧 intake 的四项 accepted artifact 字节均与 receipt hash 相同，包含 `intake/feasibility.v1.md`。新版 `prior_conditional_prerequisites` 数组与旧四项逐字、顺序一致；原实验问题、metric、comparison、protocol、完整分母/失败路由也原样保留。新版可行性第 15–18 行逐条解释阶段归属，没有把 G4 或正式证据写成已满足。
- **原目标未被缩减。** [feasibility.v2.md:25](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/intake/feasibility.v2.md:25) 继续要求完整 19 义务、八配置、root/重复/失败分母、主要统计及真实 design/evaluator/WorkOrders；当前 Brief 实际 IDs 仍为 `F1–F6,T1–T4,S0–S7,M1`。1512 个候选 episode/252 串行小时保留，资格/dev/消融/兄弟/重试另计；没有因本机资源而删条件、变 protocol-only 或预设 LLM 胜出。
- **G4 仍是后续硬门。** 新版第 17、26–28 行和 packet.formal_evaluation_gates 明确：开发 pilot/校准仅用 dev，G4 冻结之前禁止实际 main/holdout 建立或暴露；正式结果、图表、结果写作依赖真实后续证据。`feasible` 对象是研究路线，不是 G4 结果。完整六根经典比较、LLM 生成/prompt/token/时钟竞争、八条件实际接线、OOD/holdout、事件/删失等仍列 remaining。
- **开发成绩未升级。** [g4-round16-evidence.v1.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/qualification/g4-round16-evidence.v1.json:2) 标识 G4 `needs_work`、三次完整 dev episode，而非三个独立场景或六根比较；replay/cache/offline reconstruction 明确不增加策略样本。L3/B0 和 L3/B1 terminal 均 completed、returncode 0、同一 copied-case SHA；Y 分别为 `.7541293810` 与 `.6739319673`，没有选出 B_classic*。B2 当时正在运行，本轮不读其未完成成绩，也不计作第四个已完成 episode。
- **证据指针确实绑定当前字节。** 复核 round16 evidence 中全部 16 个 `evidence_sha256` 文件与 10 个 repo-relative adapter source SHA，共 26 项、181,912 字节，全部匹配。dev manifest 声明 split=dev、24 records；已保存 replay 为 600s、105/105 RPC、25 组 exact_matches 全 true、response_discrepancies 为空，且 input run/world SHA 与 L3/B0 terminal 一致。本轮未重新运行 replay 或独立重算长 flight 分数；这些记录只支持开发路线，不证明模型有效或总体可靠性。
- **本地模型仍只有结构证据。** Qwen/GLM 原 load 与 Mistral 修复后 load.v2 均为 `generated_tokens=0`、`generation_calls=0`、`paid_inference_calls=0`。Mistral regex receipt 明确 tokenizer-only、固定官方 revision、四个保存示例 decoded 与原文本相同，修复后 load.v2 记录 `fix_mistral_regex=true`，权重 revision 未变。没有读取/重哈希数十 GB 权重、执行 tokenizer/forward/generation，或把该配置记录升级为模型策略资格。新 packet 将推理质量、时延、token/physics 争用列待测，范围准确。
- **资源授权未扩大。** 新版 authorization 与 [qualification-authorization.v1.json:3](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/inputs/qualification-authorization.v1.json:3) 的最新指令一致：在既有本机范围先做参考任务和 G1–G4；无 paid API、新硬件/算力采购、外发或投稿许可。`budget={}` 不表示计算免费；原生 host、资本/能耗未计量仍披露。任何 dev 模型生成继续要求完整初始 plan 接受后、实际操作前正常预约，失败/unknown 保留并对账，明确禁止 unreserved settle 绕路。

## 正常重交与尚未通过的门

拟采用唯一 controller 的 `rewind intake --preserve intake/feasibility.v1.md` 再重交 v2、同字节 venue，是与现状相容的保留历史路径。旧可行性字节此刻仍与 accepted hash 一致，满足 preserve 的必要字节前提；正常 rewind 仍须记录明确 reason、新 operation ID 并保留完整旧 conditional receipt/history。若改其他旧 accepted 文件，也必须在修改前按原绑定保存，不能事后补称 baseline。

venue/budget 原回答能否复用，继续受先前 [development-resource-sequencing-review.v1.md](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/review/development-resource-sequencing-review.v1.md:38) 所列 exact comparison/计划字节/Brief 契约/预算绑定限制；本轮没有重新接受 venue、改 Brief 或批准任何新资源。此前正常回答复用的判断不意味着旧 venue receipt 在 rewind 后仍有效，仍须重新提交并正常验证。

此新版 intake 解决的是研究路线与后续资格门的阶段混淆，不能替代尚缺的完整初始 plan packet/snapshot、经验设计与实际 evaluator、八条件 recording 预检、全部 WorkOrders 或真实 G4。不能靠 plan validator 暂未检查某字段的漏洞先冻结占位设计，再回填 hash。现 packet 的明确文字保留了这些要求，本轮未发现其降低科学门。

## 冻结绑定与执行限制

| 来源 | SHA-256 |
| --- | --- |
| `packets/intake.v2.json`（修复后） | `ce3b5574c63324f178b3bd6817f9c5f3601c6c30581f747291a9427c3b30b591` |
| `qualification/g4-round16-evidence.v1.json` | `fff53b51d4d4dee195990c5756b82dd4321a790a08cdcf22997fd966ece3363b` |
| `review/development-resource-sequencing-review.v1.md` | `14595ccb00fbb9eb989ff1f93e7c9bdf50a218827cd0ebd8519bc5d41c13581d` |
| `packets/intake.v1.json` | `4878f6dd7cd6bdc75ca7cd02f74ce63d26ef8ab3bd603ab3a6d2aa7ed280898e` |
| `intake/feasibility.v1.md` | `2f0272334f37f18600a8e33c98956b1f4058fb8101379472d8dae26ed802cdee` |

观察时真实 state 为 schema 2、implementation `a112e7aaff3ab6c99f983a2030fc190d7ccd1af940a943604d94ad318e8fef20`、`round-17`，Brief SHA `a4cf4597fe0bc584c95f1277717c34e026c82bfa6cca8daa566584c5b48c9bd6`；receipts 仍是旧 intake/venue。control/checkpoint 的读前后 SHA 分别保持 `5d5aaa7eac3fd90193cd5bc6ed6f0dcb48ad89c92204ebf6e8a1c5863fbd5e6c` / `dfeb8eb39f69ec2e369533c067abdc066af67654de72232f47dd551b5a580cd3`。`round16` 是证据文件命名，不被误认为当前 active round。

本轮只读取文件、核对小型 JSON/text/hash/span、执行无状态纯 intake 校验，并写本报告。未运行测试套件、physics、模型、replay 或长 trace 重计；未改 production、intake、计划、旧报告、state/Brief；材料清单修复由主 Agent 完成。既有 G1–G3 只按其冻结范围引用，未新授 G1/G2/G3/G4、正式性能、人类领域认证或论文完成。没有完整聊天 transcript；本轮授权范围判断依据提供的实际指令记录，不扩展到未提供的授权。
