# Delivery：验证当前版本并保留可恢复证据

Use [production.md](../production.md): `prepare-round` persists the exact request in the reserved control directory. After the host actually judges it, `deliver --decision ...` closes that checkpoint, rebuilds exported assets and verifies the complete ZIP as well as the manifest. The runtime export includes its upstream protocols; historical implementations must remain pinned rather than silently upgrading old state.

本阶段融合 Deli 本地证据绑定/交付合同、PaperFlow finalizer/效果恢复和上游 Autoresearch 的校验习惯。状态只由当前 controller 提交，不能另启 Deli checkpoint 账本，也不能将 Git metric keep 当成科研验收。

## 最终验证

读取所有阶段回执、当前 Brief/冻结计划、逐项绑定和独立 review。重新计算产物、原始证据、模板、稿件与 review 的版本关系；不是只接受前一阶段写下的 pass 字符串。缺项、变更、错绑、未插入稿件、未披露偏离、未解决效果不明或过期 review 都阻止交付。

完整论文应保留可编辑源稿、实际 PDF、bibliography、引用计划/核验、实验与图表来源及 sidecars、评审/修订、计划历史和快照索引。按当前 task/Brief 约定判适用范围，不能因材料缺失临时删掉必交角色。源包从干净解包目录构建；PDF 实际页面审阅另记录，文件头正确或编译成功不等于视觉可用。

最终由 `finalize` 重新验证并冻结可携带的 artifacts、控制记录及 manifest；交付后用 `verify` 检查导出本体。报告分别说明软件执行状态、科学发现、审计状态、未解决项、实测用量与未知成本。离线 fixture 使用 `software_fixture_complete`；真实内部审阅也不代表外部科学接受或授权投稿。

## 轮次和恢复

`prepare` 绑定当前任务、状态、规则和证据，宿主读取真实材料后给判断，`apply` 只记录通过结构/版本检查的结果。资料、规则或状态变更须重新准备，旧判断不可套新版本。重复同 operation ID 与同参数只应用一次，参数不同拒绝；单纯心跳不产生研究进展。

中断后先读取当前状态、已完成/失败/不明 attempt、最近回执和实际产物。已完成动作不得因没有模型总结就重放；已派发但无终态的动作先对账，reconcile 提供对应 attempt、实际 hash 与处置原因。not_started、failed 和 completed 是不同事实，检查完成不代表允许重跑。保留旧历史，不删 pending、归零计数或覆盖新文件制造恢复成功。

对证据改变使用 rewind 及新版本产物，原计划承诺与历史回执保留。遇到未知 schema 或执行实现变更，固定旧版读取或显式迁移；不默默解释，不将历史授权复制成新范围。

## 交付边界

用户收到当前可验证成果、证据索引和实质限制。提交期刊、公开发布、购买算力或跨任务发送是另外的授权边界，不能由完成状态推导。仅调用小入口或安装上游插件不会启动真实研究和后台任务。完整上游子命令/hook/rollback 仍可使用，但不得触碰当前论文 ledger、冻结评价器和历史记录。

## 完整论文的八类角色与未完成稿

完整研究到论文交付采用用户本地 Deli 的八角色证据合同：source（实际 TeX）、pdf（实际编译 PDF）、bibliography（BibTeX）、citation_plan（逐条 JSONL）、citation_verification（核验记录）、review（专业及 R6/逐项评审）、revision（修复及保真/回归记录）、snapshot（当前版本与证据索引）。适用实验/统计/图表本体与源数据另外通过相关门和绑定进入包；八个空文件或统一 passed JSON 不算交付。

纯研究、显式局部修改和仅交实验记录按真实批准范围决定必要角色，不能为了避免未完成而在最后缩减。默认研究入口不强迫尚未准备写论文的课题伪造 PDF；明确请求完整论文时也不能把研究摘要当论文完成。

未完成仍可出供审阅的 PDF，应在首页清楚标 draft，呈现完成情况和具体阻塞项；draft 输出不得位于最终交付/接受路径或挂 delivered。缺项、错误、停止、不明效果都保留真实状态，不能靠 banner 消除义务。最终当前版本满足全部适用条件后才移除草稿状态并冻结包。
