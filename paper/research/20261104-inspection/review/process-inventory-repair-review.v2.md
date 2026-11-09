# Process inventory v2：foreign fallback 修复独立复验

结论：原v1 P2已关闭；本次限定两行修复及相邻兼容边界未发现新增阻断项。候选仍未安装，这只闭合源码修复，不代表真实载荷clock、唯一根因、G4或新pilot资格通过。旧v1报告、freeze及反例保持不变。

审查者 `/root/control_review`，同模型家族独立工程上下文。R=`paper/research/20261104-inspection`。冻结 `planning/inspection-process-inventory-candidate.v1/freeze.v2/binding.json` SHA256 `edefe19a4b4c77faa92052b6fa86ebbcc60c0eaa6f43c69a5c0c762f1f60435e`，全部10项文件逐SHA核对，working源、tests、verifier先核相同后执行测试。独立binding为 [process-inventory-repair-review.v2.binding.json](process-inventory-repair-review.v2.binding.json)，SHA `a8d179bfe18ef61196f92483a6acea65b1f92fced0d3406b2943c23b9e4af3bf`。

**原foreign反例现在正确回退。** 冻结native源 `f1870d92317d5f482d5d5cc19f791fffecb23ac0fb7c777028be5e1060c59ab9` 仅在读取BSD记录后新增 `info.uid!=own_uid` 时raise `RecordUnavailable`（`inspection_darwin_inventory.py:107–108`）。该分支早于name/path、rows插入与子代扩展，因此foreign根或child都不会先贡献一个BSD-only短command记录；甚至foreign zombie也保守回退，未扩展信号权限或改变正常本机UID流程。

独立复用v1场景并参数化root10/child11：稳定PID/birth/parent，foreign path若读取会不可用。两项均恰好调用一次原完整PS fallback，返回完整sentinel（含非forest/orphan行），hot_native不增、hot_legacy及RecordUnavailable原因各增1，无partial native行被采用。2 passed/0.01s。脚本 `/tmp/inspection-process-inventory-review-v2/test_original_foreign_counterexample.py` 与日志SHA见binding，未创建真实跨UID进程或发送信号。

**正常分支和原清理逻辑未变。** candidate cleanup源仍 `ce210650067d2b4e6f49c47845dea36e034d2c7c2e1187899023ddf22f792a9d`，verifier仍 `2a8a8fb51de56b51bb19de484f02a7a4e0401920f8320466955f76ecfce0474a`，与已独立审过v1逐字节相同。原production仍为 `361ba11b8ea088627e4c0bd832168bb26821297e5d8f54bf6d43c2ba327a9c70`。因此v1已实核的六个保留算法AST、baseline/attach、unbound-root恢复、0.1s wait、完整shutdown/unknown-Python-orphan扫描和信号逻辑条件复用，没有安装或悄悄改变调用方。

实际从working相同字节运行 `verify_staged_cleanup.py`，**41 passed/10.03s**，包括原26兼容检查和15新tests。真实attach之后的dynamic newborn发现/清理，以及attach失败后的全表恢复/停止原未绑定child均执行通过，不是只跑新增foreign Fake。正常本机owned forest也在真实睡眠子进程测试中与完整PS比较并完成cleanup；范围仍不是MLX/physics工作负载或任意race保证。

**新微探针只作离线重算。** v6 source指针匹配新native/candidate cleanup源；40条观测/20个pair，全部expected forest、identity fields及Python detection检查为true，hot_native22、hot_legacy0、无fallback。独立重算median：完整PS wall75.393771ms、self+child CPU74.836ms；native wall0.315250ms、self CPU0.3155ms、child CPU约0。未重跑新的性能probe；不把与v1的数值差解释为UID guard本身造成的进一步加速，也不把idle四进程测试外推为载荷deadline通过。

ABI/计数单位和当前SDK范围没有新改动，沿用v1已核本机证据。本次不扩新平台、任意foreign读取能力或外部副作用。持久birth仍秒分辨率，native微秒重查只约束一次读取期间；周期观察、fork/reparent及非kernel-subreaper边界不变，不能升级为永久PID身份或host-loss清理保证。

后续仍须正常source-amendment安装及provenance闭包，再按contract进行新的预登记载荷clock资格；旧失败pilot不续跑、不重分类或改分。没有模型/物理、正式RNG/data或实际owner写入。本报告/绑定冻结后停止写R，不扩大审查。
