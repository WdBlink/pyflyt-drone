# Brief 单元素数组表示修复独立复核

审查者 `/root/control_review`；同模型家族、独立工程上下文。只审查这次表示修复及正常 owner 重绑定路径，不重做 v3 科学设计或 G4 审查。

## 结论

**本次是可逆、无损的表示修复，在既有任务授权内可以通过正常 owner API 处理。** v2 相对实际 accepted Brief v1 仅将 19 项义务的 `acceptance_criteria`、`evidence_requirements` 共 38 个完整原文 string，各包装为单元素 array。没有拆句、删字、改标点、替换 TODO 或降低要求；十九项文本、顺序、ID/种类以及其他字段精确不变。

可走正常 `set-contract → approve（保留真实既有授权来源）→ native ask-user budget 重发当前绑定 → 将同一具体默认选择的真实原答复绑定到新 question → confirm-budget → 重试不变 v3 plan`。这不是放松 validator、直接写 `.paper`、伪造新答复或获取新资源许可。本报告本身不是用户批准，也不保证之后 native commit 的所有独立前提已经满足；它只确认本次修复与路径合法，不授 G4/正式运行/付费/完成 GO。

## 变化和拒绝原因的独立核对

读取并比较 [brief.ral.v1.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/inputs/brief.ral.v1.json:1)、[brief.ral.v2.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/inputs/brief.ral.v2.json:1)、实际 control 中的 Brief 及 [brief-schema-repair.v1.json](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/brief-schema-repair.v1.json:1)：

- 实际 accepted Brief 精确等于 v1 JSON；新旧顶层仅 `deliverable_plan` 不同。
- 两份均恰好十九项，ID/顺序/除两字段外的条目字典完全相同。每项两个旧值都为 string，新值都精确等于 `[旧值]`；全部 38 项与 repair artifact 的 changed_paths/original_text 逐项相同。
- 对新值取唯一元素、其余字段不变的逆映射，完整 JSON 精确恢复 accepted v1。不是只相信 `inverse_mapping_exact:true`。
- question、direction、paper_type、scope、angle、audience、reviewers、resources、venue、contract_status、budget={} 及其所有嵌套内容均不变；原 19 项义务与八条件/完整实验分母没有被该格式修复改变。v3 packet 仍为 `5c1a0278…`，没有为消除错误改计划或 runtime。

实际失败记录 [/tmp/inspection-initial-plan-commit.v1.json](/tmp/inspection-initial-plan-commit.v1.json) 是 `status=refused`、`plan: item[0].acceptance_criteria: expected an array`。纯 CPU、无 owner 写入的本轮复验，以同一 v3 packet 和 accepted history 仅替换内存 `_brief`：旧 Brief 的 `planning.validate('plan',...)` 复现同一错误；新 Brief 返回 `[]`，`check_brief(new)` 返回十九个 IDs。

根因确为接口表示不一致：[planning.py:305](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/planning.py:305) 使用原 deliverable item validator；后者在 [deliverable_contract.py:175](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/vendor/deli-local/scripts/deliverable_contract.py:175) 要求 string array。此前顶层 Brief 接受不等于该 item schema 已通过。保留原拒绝并修正输入格式是正确处理，不需要改 runtime 或降低验收。

## 正常 owner 路径及精确限制

1. **`set-contract` 只在当前阶段可用。** [runtime.py:314](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:314) 要求 receipts 恰为 intake/venue；当前实际满足，且尚无 plan receipt/frozen plan。第 317–322 行继续禁止悄改研究问题、类型、资源与 calibrated venue。第 323–325 行把旧 Brief/approval 追加为 `brief_superseded` 历史，安装新 exact digest 并清 approval。应使用新的 operation ID，保持旧文件/失败/历史，不初始化或手改账本。
2. **重新 approve 绑定新 effective Brief digest。** 新文件 raw SHA 与 canonical effective digest 不是同一概念，见下表。`approve` [runtime.py:266](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:266) 核实际授权来源及新 exact Brief hash；不能仍传旧 hash 或假称已有旧 approval 自动覆盖新 bytes。无损表示修复可以由当前执行 Agent 以真实 delegated actual-user 来源重绑定，不能把 actor/时间写成“用户刚新批准了改过的研究范围”。原真实请求及最新“先非LLM/G1–G4再正式实验写作”的范围继续有效；表示修复没有扩大授权。
3. **旧 budget confirmation 必须重绑。** 即使 budget 仍为 `{}`、同一 figure/table plan 未变，Brief 的结构性 digest 已变；[runtime.py:437](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:437) 会拒绝旧 contract_sha。应通过 native `ask-user --kind budget` 生成当前 question，保留原问题/回答历史，不能改旧确认哈希来伪装一致。
4. **同一具体预算选择可复用真实原文。** [user-input.md:12](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/references/user-input.md:12) 明确允许复用已有同一具体选择的真实指令，无需为造新记录重复问人。此处科学/资源/预算原文全部可逆未变，原 plan 路径与字节仍一致，因而可将 `answer={choice:default}` 与原 human_response path/SHA/quote 放入**新版本 response**，绑定新 native question_sha256。保留原答复来源/时间事实，明确“复用既有实际答复”；不得编辑原答复、沿用旧 question SHA、伪造新聊天/新回答或拿此审查报告代替人类答复。
5. **正常 confirm 后再重试不变计划。** [runtime.py:423](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/runtime.py:423) 从当前 issued question 和真实来源记录新的 contract/budget/plan 绑定。之后正常 native commit 仍检查当前 approval、预算、阶段、全部实际 artifact/validator。v3 WorkOrder 的旧 Brief 文件路径可继续作为保留的原文来源；当前有效契约必须读 owner 的新批准 Brief，连同此无损映射及 superseded history 解释其表示差异，不能把旧 string 版本当成有效的新 schema。

本轮只在内存副本替换 Brief、调用 state-free `user_input.prepare/validate_response`。对同一 `budget-proposal.ral.v1.json`，原默认答案的 source SHA/quote 仍匹配，纯验证返回 `choice=default,budget={},proceed=true`。这说明 native 的新 question/原真实答复来源可以正确绑定，**不是已经 issue/answer/confirm 到真实 owner**，也不证明工具能认证文本作者身份。若实际计划、资源权限、预算或科学文本改变，此无损复用结论不适用；仍须按对应实际选择/授权规则处理。

## Digest 与证据绑定

R=`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection`。

| 对象 | SHA-256 |
| --- | --- |
| `inputs/brief.ral.v1.json` raw file | `8c616e5c2a926dfbdda6ce58596e591f5938864145e14f83c0909bcdd7611dc5` |
| `inputs/brief.ral.v2.json` raw file | `e84396c4b96f9a1681bce948e44949802b02422e7ac90b89759fa2e48a45b37b` |
| `planning/brief-schema-repair.v1.json` raw file | `0ffd9303f9f63b355904223a0cf6785e713ea4e299cfea4e28cc301f82ceade2` |
| 旧 effective Brief canonical digest | `a4cf4597fe0bc584c95f1277717c34e026c82bfa6cca8daa566584c5b48c9bd6` |
| 新 effective Brief canonical digest | `9169308d3db5f53716b0784d0cda30e94faec0c16d0b6051a7fa47607bc0ff9a` |
| 旧 contract-without-budget digest | `3a184d38995a78295e3812f93031a8a77c5fbebb8948d1b23d7439224f487954` |
| 新 contract-without-budget digest | `d01c98d6f72b89580b84b47219f42e088476fd8f1b387b72b53fb6fdb71ff79a` |
| 不变 `packets/plan.ral.v3.json` raw file | `5c1a0278332c4ee6e4f0546530a56842c30c18c5fcdb015934debb5b6a864b78` |
| 真实默认预算答案 `inputs/user-budget-answer-20261105.json` | `7afb340c5f18e7cbad4eda93d58d87cce1d0185e27c8adc1c7ead1d01f418cac` |
| 原实际请求 `inputs/user-request.txt`（原 approval 来源） | `cf0a0d75d4e81786bf14f22fab0fe6137e891b5283808691042d04012c812960` |

纯内存重发得到的示例 current question digest 为 `d03d73b9f7b2485c3fbe2dd4d72cc094611d01ebf2e3e440b2e08e2a7da43d7a`；真正执行时必须使用 native 实际返回值，不能从本报告复制为已签发 question。默认预算 digest 仍为 `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a`。

观察时 owner round-18、stages=intake/venue、frozen_plan=null、ledger0，Brief 尚为 v1。只读 control/checkpoint SHA 为 `0cb41c7264ee502fce88ee77cb2bfafbfa3d69ab53b964f089f7f7c562182371` / `40f2338f63822cfc983fc1fed32f18ae439d830d906c44c9a494b243622dfe7c`。本 Agent 未调用 set-contract/approve/ask-user/confirm-budget/commit/reserve，也未改 owner、旧文件、runtime、live logs 或 v3 科学设计；仅新增本报告和执行无副作用的内存校验，无模型或新物理。G4、正式运行及已有独立 evaluator 复验仍各自按其真实证据判断。
