# v4 前瞻外部归属处置 v1 独立审查

结论为 **request_changes**。处置生成器终验proof后，caller仍会做观察复核与元数据检查，但返回前没有重新绑定刚生成的proof。实际隔离I/O已复现 proof被删除后callback仍返回旧引用。该问题阻断v1前瞻策略接受，037既有独立/native处置按已封存结果处理。

本报告由 `/root/control_review` 独立审阅，绑定policy SHA `172c847eeed52bf6a0018c837e8aa24801eeaedde522cff7e72063a64833764d`。五个源码和九个证据引用均按policy原SHA核对，五源码副本保存在evidence。作者26项通过日志与本轮独立探针分开保留。首轮探针因隔离fixture未建run目录在准备阶段停止，补目录后重放成功，候选字节与输入版本未变。

## P2 处置proof在caller的可观察窗口失绑

`reconcile_new` 确认原完整attempt、known用途和归属后写 `proof.json`，终验闭合树、原refs、policy、sidecar、query、raw和proof，随后返回Y与proof ref。`run_observed` 接收后再次 `observation.check()`，仅对侧录数作约束，直接返回证据。外层 `measured` 只再验policy、observer目标及companion，也没有核刚返回的proof链。此处存在明确的调用边界，proof的终验早于callback返回。 [生成器](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/recovery/pilot-v4-cleanup-037.v1/source/admit_external.py:169) [caller](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/recovery/pilot-v4-cleanup-037.v1/source/continue_pilot.py:30) [外层元数据](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/recovery/pilot-v4-cleanup-037.v1/source/continue_pilot.py:125)

独立探针运行真实 `reconcile_new`、`_events`、`match_sidecar` 和 `run_observed`，用归档037事件原字节模拟OS transport，以合成038上下文、原执行/完整validator/native policy预条件替身隔离边界。所有文件写入仅在/tmp。正控按原执行完成及known、完整validator、OS transport的顺序生成实际query/proof，返回值为0.3314065510597302，原unknown字段仍在。负例在第二次observer.check时删除实际 `proof.json`，sidecar自身SHA保持一致，callback却仍返回原值和SHA `124d99243880686fefe5e66216b8ecaf8bd0eb9ececeba5b054f9ba5648bd7de` 的已失效引用。该结果只证明此返回窗口，不是实际飞行或新的native处置。先已存在的unknown在同一fixture中原异常对象保留，也未发起查询或写proof。完整输出在 `results.json`。

最小修复是把caller的观察复核及固定元数据检查放在最终处置引用验证之前，最后按生成时保留的固定ref核原terminal/tree/sidecar/query/raw/proof/source和原闭合树，再直接返回。不得重采当前新SHA，也不能只确认proof文件存在。此修复只覆盖可观察的调用窗口，后续SDK视图重用仍需按manifest所述核验proof。

## 其余路径与边界

源码中原 `execute` 在任何前瞻reconcile前完成原清理、归档、结算及terminal写入。只有首次attempt的特定UnresolvedEpisode及恰一条侧录才进入新路径，重验完整原validator、原unknown、已登记且无剩余owned、精确单个未认领PID、实际policy所需的独立review与native finding。归属使用原最终扫描、同boot PID1的精确launchd事件、同一PID的时窗和后续无live记录。非匹配、缺证、原异常或已有unknown均有拒绝分支。该结论来自源码和已封存作者测试，未声称已实际运行新的live归属分支。 [处置入口](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/recovery/pilot-v4-cleanup-037.v1/source/admit_external.py:112) [独立及native policy入口](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/recovery/pilot-v4-cleanup-037.v1/source/admit_external.py:24)

续跑源码限定同v4 argv、选择、顺序、采样和新SDK目录。调用既有 `live_frozen_sources` 约束同104/61/10，历史024/030/037独立reader逐槽分支，037分支写下一槽前marker。原pilot入口在预约/启动前处理marker。observer仅重绑本进程输出目标，其已接受源码及review引用按旧reader固定。实际38-cache重建和order38前无effects停驻仍待运行后核对，本轮未执行外层runner或SDK。 [续跑入口](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/recovery/pilot-v4-cleanup-037.v1/source/continue_pilot.py:61) [原pilot暂停入口](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/run_inspection_eight_condition_pilot.py:153)

本轮没有重新核验037接受，没有运行实际OS查询、服务操作、飞行、模型或native写动作。无前瞻policy接受、G4或正式运行授权。证据封存后停止R写入。
