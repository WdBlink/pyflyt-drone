# v4 passive close observer 独立实现审查 v1

结论：**request_changes**。被动扫描/原判定保留的核心路径在本轮有限检查通过，但有1项P2：`check()` 接受已经丢失、改字节或变成符号链接的sidecar。当前不能签发 `accept_passive_metadata_scope`。本审查不接受030 late-cleanup、不运行continuation，不授G4或正式运行资格。

## 精确对象和范围

审查人为 `/root/control_review`，同模型家族独立工程上下文。冻结binding SHA=`1cc05ab4e33989bea22109dd56318f2007668cd46f64905059b258edaeb3201e`；observer源SHA=`0a08e7e33d35dd504c6b4bbfc1d7bb5111bdafaeb2e8e0fa1c45d0722f8909df`；测试SHA=`0525afdba3390a9dc66eb830de9d30eff0a93b6ae5d3ce4d04cf2da9fefed259`。冻结baseline/current import cleanup均为 `ce210650067d2b4e6f49c47845dea36e034d2c7c2e1187899023ddf22f792a9d`。源码、baseline、作者18 passed/0.03s日志、独立探针/完整结果均复制到本轮evidence，原件未改。

## P2：check未维护已写sidecar的身份与字节绑定

源163–169行只检查内存errors、线程/active/hook和observer源码SHA，随后直接返回`self.records`；没有校验每个记录指向的文件仍存在、为无alias的普通文件或SHA仍匹配159–161行保存的原ref。

独立探针在 `/tmp` 使用**真实原 `SubprocessTree.close`**、合成库存和fake wait对象，真实创建sidecar后，在同一observer上下文内执行以下I/O：

| 变异 | 实际结果 |
| --- | --- |
| unlink已保存sidecar | check成功返回旧ref，文件不存在 |
| 保持合法JSON，把保存的original_result.status从unknown改closed | check成功，旧ref SHA与当前SHA不同 |
| 替换为指向project外同字节文件的symlink | check成功，即使SHA相同仍违反无符号alias边界 |

该次原sidecar SHA为 `33928643f74f52490dd4212c745e597ad711fd80ecaf37c0ed956e32aa74c048`；改写后的SHA为 `1bde431b3ab8e71a7413a77c8f74ab043301d3eb6df2b1a8613f3a39e2cdbee5`。文件名/时间含本次实际运行值，完整结果见 `results.json`。这些变异没有改变原close的unknown返回，却能让未来caller错误地将缺失/失绑sidecar视为完整观察。

最小修复：`check()` 对每个**已持有的原ref**核对项目内的固定路径、各级无symlink、普通文件/所声明链接约束以及原SHA；失败应明确保持观察incomplete，不能重新把新字节的SHA采作可信基准。若需要caller证明某次close确有侧录，应按实际close记录匹配，不能把空records等同已观察一集。无close/cache路径是否允许空记录可由caller显式区分，本轮没有把这个未接线语义另列为实现缺陷。不得通过改原unknown判定修复观察器。

## 独立通过检查与source-only判断

`probe.py` 没有启动进程、worker、模型或资源。它实际调用原close并以库存返回值/fake wait替代系统进程，信号函数被设为一旦调用就失败；没有发送任何信号。未机械重复作者整套测试。

- 正常原close和被观察的原close都返回unknown、unknown PID=[99]；原库存调用3次，观察后仍3次，表内容不变。该3次对应本探针采用的fallback式observe，不声称所有真实运行恒为3次。
- 实际sidecar保存同uid Python baseline、最终未知row和原birth字符串；源按ordinal记录扫描，没有臆造SIG阶段或更高出生精度。恢复文件后check正控再次通过。
- 在真实写sidecar之前断言 `_inventory` 已恢复为原函数；最终退出也恢复原close。
- 真实I/O负例将OUTPUT置为普通文件，mkdir失败；原unknown决定仍逐值返回，check因记录的sidecar错误拒绝。
- 原库存抛出的同一个OSError对象穿过observer后仍原对象抛出；hook恢复，sidecar记录原异常类型。

源码81–121行在调用原库存时只委托一次，元数据副本失败走incomplete；外线程/重入/改变hook时标记错误并继续调用原close（81–85、91–110行）。作者测试覆盖线程/重入，此部分本轮为源码审阅和既有测试证据，未声称独立复跑。finally在metadata I/O前恢复库存（119–138）；原close返回对象直返、异常bare raise。源limits正确注明timeout路径可能仍含活worker，不把额外I/O一概称为物理已结束。OUTPUT在enter与write前作alias检查，但这不能代替最终check对已写记录的校验。

观察器仅接收close原有全局扫描，不捕获全部native hot森林历史或扫描间祖先。即使修复上述P2，也只改善命中row的留证，不能补回86715历史归属、改变原030状态，或证明caller已正确接线。原104磁盘源未改；新增adapter仍需独立source/provenance和后续caller审查。此候选尚未安装；本轮无模型/物理/正式RNG/native owner写入或continuation。报告与request_changes JSON冻结后停止本轮CPU检查及R写入。
