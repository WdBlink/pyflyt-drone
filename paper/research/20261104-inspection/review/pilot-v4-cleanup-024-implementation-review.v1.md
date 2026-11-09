# v4 slot024 late-cleanup 恢复实现独立审查 v1

结论：**request_changes**。原记录的只读重核通过，但当前候选在最终 native 读取至返回之间仍能接受已改变的绑定文件或目录。尚不签发 `accept_late_cleanup_only`，不得以本报告继续运行。仅审 development late-cleanup 恢复；不是 G4、正式实验或人工领域认证。

## 对象与实际检查

审查人为 `/root/control_review`，同模型家族、独立上下文。被审源 `continue_pilot.py` SHA256 为 `72cb11b3cde95756fcf661a103e6042a3fcdd5b941c963a88a9438c24f5ce023`；proposal.v2 为 `f81f4500466d5793636626afbb335564ecf5460fcc82305a9168f89af11ef52b`；测试为 `1073a1baaec2631f04cfd91a51949ac094fa8d2ad6cc5848bf4745609d9e8da3`。本目录保存被审源副本，行号均指该副本。Root 后续修改不属于本结论；原 72cb 源另保存在 recovery bundle 的 `continue_pilot.before-final-snapshot.v2.py`。

独立调用 `inspect_original()`（源117–155行），未调用 prepare/run_continuation，实际只读核对冻结源/依赖/输入、原 clock/actor/冻结 grader、模型 meter、sole-owner known 状态。结果：Y=`0.368353660488492`，25 calls / 166000 tokens，25 causal joins，证据 canonical SHA=`c890cc121583e2767c26ddc556d1eb5c1510f38fc573b2635e23e117a0574e53`。原 tree 仍 unknown、terminal 仍 `unknown_process_cleanup`，未改分或改写历史。一次检查用时1.3813s。细节在 `actual-inspect-original.json`。作者日志为16 passed/10.58s；本轮没有重复整套，也未把它当作窗口安全证明。

## 必要修复：P2 返回前未维持已验证闭包

以下四个反例均在独立 `/tmp` fixture 实际复现，正常输入及恢复后的正常输入都通过。fixture 使用**真实** `require_native_binding`，合成 native-shaped owner/独立 review，并复制原 plan/terminal/tree/receipt 字节；不是实际 native finding 或真实独立接受。为只隔离返回窗口，payload `_bindings` 的比较返回可信原集合；另直接用原 `_bindings` 实证目录/FIFO不进入其集合。没有改真实 owner 或原记录。

1. **归档字节改变仍接受**：207–208行先验 proposal.files，210行读取 owner 后，只重查部分原文件和 usage；231行前未重核整个既有闭包。owner hook 给已绑定 archive 追加字节，SHA从 `40e14f9534d75290d831d33ea09d1096943c62324fbd7d13bce2e1ea4289c73f` 变成 `9f3c22f4fccddb2463834550deb17a89f557f348c210d0581d63537ffa5bd4d6`，仍返回相同Y。这是合成有效 ZIP 的测试，不是原归档被改动。
2. **acceptance 换字节被后绑定**：178–182行检查的 acceptance SHA 未保留；234行重新 `checked(acceptance)`，可把新字节当作返回证据。owner hook 把 finding_id 改为 `unaccepted-other-finding`，函数仍返回Y，并返回新 acceptance SHA `9860b4215f09c203d4b29720bc8cd041f36c5cf63e04cc5bfc1693284e37e518`，与初读 `24380fc562037bbe023af1acd465069cd222db8cd677425de1af5ed1127a548c` 不同。该新 finding_id 从未通过180–182行。
3. **递归目录及 entry type 漏检**：217–219行复用仅文件的 `_bindings`。分别在 owner hook 中加入空 `call-9999/` 和 FIFO，两次均返回Y；实际原 `_bindings` 前后完全相同。故“原文件集合未变”不能证明具有 call/操作语义的完整目录闭合。

反例脚本、完整输出和实际输入 SHA 已封存。执行过的 `probe_windows.py` 原样保留；`replay_windows_retained_source.py` 仅将源路径改为已保留72cb副本，避免后续修补导致反例误绑新版本。重现：四线程限1，以项目 `.venv/bin/python` 执行该 replay；它只创建/删除 `/tmp` fixture，不运行 continuation、模型、物理、resource 或 native 写动作。脚本保留强源/proposal SHA guard。

最小修复：初读即保留 acceptance/proposal/review/source 等固定 ref 与解析所用字节；建立已接受 episode、对应 operation、明确 reserve/settle 文件的完整文件与递归目录快照，拒绝 symlink/非普通 entry。最后 owner/native 与 payload/usage 检查之后、返回之前，对照**最初**快照重核全部原文件/目录/固定 ref；不得把验证后新字节采成可信基准。返回 acceptance 应引用初读 ref。需要包含原 archive 与已绑定元证据；无需把未来无关 reservation 或无关进程纳入历史事实失效条件。这只封闭函数可观察窗口，不承诺返回后永久不变或提供文件系统原子事务。

## 已支持和未验证的范围

239–290行 source tracing 显示：恢复包装仅拦精确原episode、canonical plan及024前缀，其他调用委托原API；新SDK输出要求新目录、原order/sampler，精确缓存024返回前设置 checkpoint，拟在25新effect前暂停。原104生产源、旧 unknown、原 abort108与原SDK分母未被本轮修改。实际run前仍须核原104/61/10；历史cache验收不应依赖未来无关进程。上述路径本轮未实际运行；作者的 Fake 包装测试不是继续运行授权。

本轮未运行 MLX、PyFlyt、continuation、正式 RNG、reserve/settle 或 native 写动作；没有第三次 retry 或新episode。quiet observation 仅证明后续观察时安静，不能追认原未知PID的归属或把原清理改成closed。原始只读重核通过与恢复实现可接受是两个判断。新的 v3 修复须另作有界复验。本报告完成后停止本轮 CPU 检查及R写入。
