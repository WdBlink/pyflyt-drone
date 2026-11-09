# Process inventory repair v1：独立源码与兼容审查

结论：发现一项P2边界与既定保守fallback承诺不符，建议候选保持未安装，先闭合foreign UID活节点的处理。其余本轮已读 ownership/清理调用链和实际睡眠子进程兼容检查未发现新增阻断。这个P2不是已证误杀或丢失孤儿；原全表shutdown/unknown-orphan逻辑确实保留。性能微探针只支持降低idle inventory开销，不支持载荷clock已通过、1734次full PS是唯一cause或G4通过。

审查者 `/root/control_review`，同模型家族独立工程上下文。R=`paper/research/20261104-inspection`。只读未安装候选、13新tests、验证脚本、ABI/性能/contract及必要当前调用点；只运行所允许的Fake和测试创建的睡眠子进程。未修改源码、原始证据、owner或旧报告，未安装候选，没有模型、物理、正式RNG/data或载荷诊断。

输入 `planning/inspection-process-inventory-candidate.v1/freeze.v1/binding.json` SHA256 `d7ba805cba454170cab1ed653277ed52544b0e2e0f0ed5e076cad66b1b3aefd2`，12项冻结文件逐SHA核对。候选native源 `275efffae9fd7889daaba1c81457bc3e1dca49d1563ec30a4fa994c1eaa31198`，cleanup源 `ce210650067d2b4e6f49c47845dea36e034d2c7c2e1187899023ddf22f792a9d`；实际production cleanup仍为 `361ba11b8ea088627e4c0bd832168bb26821297e5d8f54bf6d43c2ba327a9c70`。独立binding `review/process-inventory-repair-review.v1.binding.json` SHA `cc1ec895f5c8575256f92255c3ad7699559d16185c0b98a3a2746b02ffd346a7`。

## P2：foreign活节点没有按声明进入完整PS fallback

冻结 `source/eval/inspection_darwin_inventory.py:109–112` 仅当 `info.uid==own_uid and status!='Z'` 才取实际executable path。foreign UID活节点会跳过path，保留name/comm作为command，并在`:120–132`继续作为native forest返回；没有触发 `RecordUnavailable`。因此“若owned foreign/unreadable，同样fallback完整PS”的明确边界当前并不成立。BSD info可读与完整path可读是两件事，跳过后不能把该节点说成已经完成同样的native读取验证。

独立Fake反例 `/tmp/inspection-process-inventory-review-v1/test_foreign_fallback.py`：已owned root10及child11均为foreign UID、稳定PID/birth/parent；path如调用会抛RecordUnavailable。实际结果却为hot_native1、全表读取0、path读取0，两个节点均返回，command只有`Python`短name。反例1 passed/0.00s，证明上述分支，无需创建真实跨UID进程或发送信号。

最小修复是对非Z foreign节点直接视作本backend不可完整读的情况并raise `RecordUnavailable`，让原完整PS承担它；增加该精确反例，而不重写ownership或信号算法。若另行选择允许foreign BSD-only记录，应先明确改其承诺/metadata范围并论证与原caller的兼容，不能仍声称foreign同样fallback。本轮建议沿既定保守边界修复。

本反例没有证明当前进程被错误收养/发送信号，command也不是现有信号授权字段；UID/PID/parent本身仍在记录。shutdown会重新完整PS，因此不能夸大成全局未知孤儿检测被删除。阻断点是本次承诺的fallback和观测完整性边界未实现。

## 原算法、调用方和fallback正向核验

独立解析production与candidate AST，核 `_inventory`、`stop_worker_tree`、`SubprocessTree.__init__`、`attach`、`wait`、`close` 六个函数/方法相同，不只抄freeze中的unchanged bool。实际改动在热 `observe(current=None)` 的数据源选择及新backend/统计。

- init与attach仍全表；只有process存在且root PID已在owned中，热observe才调用scoped reader（candidate cleanup`:120–125`）。传入显式current仍沿原算法。attach失败后root未绑定时仍全表，close能恢复直接子进程。
- wait仍0.1s cadence；close首轮读取可为forest，但随后`:168–175`原完整PS和全表same-UID新Python orphan检查不变，unknown仍阻断closed；`stop_worker_tree`全程未换backend。foreign记录问题不能掩盖这些已保留边界。
- `descendant_inventory`从所有已owned PID出发，已记录reparented child及其子代仍会访问；已退出root不让已owned child消失。重用且birth秒值不符的owned PID只返回该行、不把它当parent authority（native`:122`）。新child校验其ppid、父的microsecond birth/ppid/uid身份，path与children枚举后重查父身份；矛盾parent、循环、未知status、权限和不完整记录走异常，wrapper对已声明native失败回完整PS。
- 查阅当前 `eval/` 调用点：LLM episode、八条件driver只用wait/close，adaptive诊断的observe wrapper只记耗时并返回；没有生产调用方消费热observe的无关全表行。原close是需要全表的地方，实际保留。该返回值已明确只是observed forest，未来caller不能当global inventory。

API仍非原子进程快照或kernel subreaper。fork后在下一观察前退出并reparent、任意host loss均有原边界；最终全表可发现新same-UID Python孤儿，但不是任意非Python/跨UID/任意生存期的保证。owned持久birth比较仍为原秒分辨率字符串；native microseconds加强的是**本次读取期间**身份重查，没有升级跨tick的same-second PID reuse保护。不能将它写成全新的严格进程身份能力。

## 单元、实际子进程和ABI

working candidate源、tests、verifier先核与freeze相同，再运行working路径 `verify_staged_cleanup.py`（该脚本按working相对层级定位repo，不把快照副本当任意位置独立entry）。独立 **39 passed/9.96s**，包括原cleanup/receipt-fault tests及全部13新tests。特别包括真实attach失败后的全表恢复/终止，及在attach之后才创建新子代、由native发现并实际cleanup的测试；未漏这两个现场边界。该测试不是实际MLX/PyFlyt工作负载。

ctypes布局实核size136及comm/name/nice/start_seconds/start_microseconds offset=48/64/116/120/128，与冻结C probe输出一致；本机两个SDK header当前SHA也与冻结probe metadata相同。本轮没有重编C probe，实际native子进程测试又对parent/group/uid/born与完整PS做了比较。只据sizeof不能保证所有未来OS ABI，目标是这台声明的Darwin环境。

children API的返回值为PID **count**，buffer参数仍为**bytes**，已对照[Apple libproc wrapper实现](https://raw.githubusercontent.com/apple-oss-distributions/xnu/main/libsyscall/wrappers/libproc/libproc.c)。候选按int容量配置buffer，count饱和则扩容、最多4次/131072上限，超限或errno等进入fallback；所附100 PID反例确实防止除4造成截断。实际path长度/空读、BSD返回size和PID等检查也存在。该在线main源是API依据，不冒称其commit等于本机kernel；本机ABI证据另有绑定。

## 原性能结果的独立离线重算

未重跑新性能probe。冻结v5有40条交替legacy/native观测、20个pair；40/20分母、全部expected forest/identity/Python detection检查、源SHA及backend计数均实核。独立重算median：完整PS wall86.175209ms，self+child CPU85.638ms；native forest wall0.436084ms、self CPU0.4355ms，child CPU约0。原backend hot_native22、hot_legacy0、无fallback。这里的self+child是逐条CPU和的median，未把wall当作CPU。

这是idle四owned进程、每pair固定legacy先native后的微探针，完整PS约2200行与forest4行不是同一全表功能。它能支持“对现有热caller所需forest，减少实测inventory成本”；不能外推deadline、真实载荷clock通过、因果唯一性或候选已安装。前次1734次fullPS合计约456s是既有诊断的monitor wall观测，本报告没有重做该实测归因，也不把它和微探针CPU混为一谈。

原P2需另版修复并有界复验后才能关闭。即便关闭，仍需按contract正常source-amendment安装/provenance closure，再做新预登记载荷clock资格；原失败pilot不续跑或改分，G4不因此通过。本报告/绑定冻结后停止写R，不扩大审查。
