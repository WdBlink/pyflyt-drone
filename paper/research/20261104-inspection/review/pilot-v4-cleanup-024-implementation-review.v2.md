# v4 slot024 late-cleanup 恢复实现限定复验 v2

结论：**accept_late_cleanup_only**。v1 报告中的返回窗口 P2 已在本次候选 v3 关闭；未发现本轮剩余阻断。本结论只接受 proposal.v3 描述的后续 cleanup disposition，不把原 terminal/tree 改成 closed，不是 G4、正式运行、模型质量接受或人工领域认证。结构化独立 review 在同名 evidence 目录的 `review.json`。

## 对象和版本边界

审查人为 `/root/control_review`，同模型家族、独立上下文。候选源 SHA256 `bb196f6c6939d9c050eb56f1b435eee68264347093b1cf85f0556cae12101c54`；proposal.v3 `c59a4e15439da55708acb84127ed02b9c7c2a0a7d962eb837a4f3c67cd742aac`；测试 `988afab06e2381277f8fee8c218cb52281e35e007d51cff19a20b2f733595b10`。均复制到本轮 evidence，并保存相对72cb源的精确 diff。原 v1 报告 SHA `0be5166605d9c20153b1109ad7bfb745470f550ac3406f811dfe61eef32bed73`、四反例及所有原记录未改。后续版本不在本结论内。

## 四原反例实际重放

仅作必要的 source/proposal 与新增 closed_inputs 字段迁移；原攻击点仍是 owner() 可观察窗口，实际文件 I/O 在 `/tmp`，不是仅伪造 snapshot 返回值。

| 原反例 | v3 实际结果 | 对应修复 |
| --- | --- | --- |
| 已绑定 archive 追加字节 | 拒绝：recovery input changed | 255–256行重新核全部原 proposal.files |
| acceptance.finding_id 换成未接受值 | 拒绝：recovery input changed | 201行保留初读 ref，260行终验该SHA，264行返回原ref |
| 新增空 call-9999 目录 | 拒绝：accepted input changed after validation | 117–135行收集递归目录，257行对照原快照 |
| 新增 FIFO | 拒绝：closed input must be a regular file or directory | 非普通 entry 直接拒绝 |

初始正常、清除每个变异后的正常、负例结束后的正常输入均通过；返回 acceptance SHA 等于初读 SHA。错误 repeat 身份仍被精确024 canonical guard 拒绝；缺少实际 applied finding 的合成 owner 仍被真实 `require_native_binding` 拒绝。目录/FIFO反例同时确认旧 `_bindings` 仍看不见这些 entry，因此拒绝来自新增闭包检查，没有靠改变原生产 reader。

边界：这是合成 native-shaped owner/独立 review 的隔离正反检查，真实 `require_native_binding` 执行；复制了原 plan/terminal/tree/receipt 字节。payload `_bindings` 在窗口测试中返回可信原集合以隔离本次修改，另直接使用原函数验证其目录行为。没有伪称合成 finding 是实际 native applied finding。执行脚本、完整结果和日志已封存。作者20 passed/13.63s日志另绑定，未代替上述独立实证，未重复整套。

## 实际记录与 proposal 绑定

独立再次执行实际 `inspect_original()`，用时 1.3196s：原冻结源/61依赖/输入、clock/actor/grader/meter及native sole-owner known重核通过；Y仍为 `0.368353660488492`，25 calls/166000 tokens，25 causal joins，证据 canonical SHA仍为 `c890cc121583e2767c26ddc556d1eb5c1510f38fc573b2635e23e117a0574e53`。原 tree 为 unknown，terminal仍 `unknown_process_cleanup`。未调用 prepare 或 continuation。

独立对 proposal.v3 的全部 103 个文件引用逐SHA核对；当前实际闭包 31 个目录、96 个文件与已归档 `closed_inputs` 精确相同。零新resource和已归档后续quiet观察字段符合限定处置；quiet事实不证明原26337归属。详见 `actual-inspect-original.json`、`proposal-inputs.json`。

源码 delta 也符合修复边界：prepare 在验证前持有闭包，末尾复查 live source/闭包/既有ref后才写proposal（176–194行）；verify 在 owner 前验原闭包，owner/native及原payload/usage检查后复查所有原files、目录、proposal/review refs和固定acceptance（230–260行）。未把验证后的新字节采成可信基准。它封闭本次可观察窗口，不承诺返回后永久不变或文件系统原子事务。

## 接受范围及后续门

可把本轮独立 JSON作为该精确 proposal 的 late-cleanup review 证据。后续仍须由正常 native finding 实际应用、原104source/61deps/10inputs的运行前检查及原wrapper身份/队列约束共同成立；本轮没有代替这些检查。239行以后旧 wrapper主体未变（现269行起），本轮复用前次有限源码审查；没有实际重放25-cache prefix、产生新SDK或运行25后effect。

原 unknown、原 abort108/SDK分母和既有raw分数保留；不新episode、不第三次retry，不把reconciliation当新独立重复。未运行模型、PyFlyt、正式RNG、reserve/settle、continuation或 native 写动作。最终G4与正式任务资格不由本审查授予。本轮检查结束后停止 CPU 检查及R写入。
