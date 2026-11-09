# Primary consumer 接线：immutable v1 独立有限审查

结论：**request changes；保留 5 项具体问题。** 所附 63 项 consumer 检查通过，但新增最小反例揭示 LLM 缓存的 trailing-attempt 漏验与 Y/world 混用、readonly measurement 的文件副作用、classic measurement 的角色重标，以及 service 的 raw-owner authority split 和 bool ordinal 类型宽松。没有据此声称实际 formal flight 或 G4 已被绕过；当前不存在真实 formal receipts。本报告只绑定 immutable v1，working 后续修复未参与、未获批准。

## 对象、隔离与执行事实

reviewer `/root/statistics_design_review`：独立上下文、同模型家族，不是人工/跨模型复核，按 autoresearch 区分软件 fixture、物理证据和实际资格。输入 `planning/formal-admission-candidate.v3/primary-consumer-freeze.v1/binding.json` SHA256 `5db76f08d3d14e4369fc169e333607a0727df57b23e78b9703d13a74748faeac`，32 source + 15 tests 的全部 47 个文件和 7 个 validation log SHA 均核对一致。

先复制 subject bytes 到 `/tmp/formal-primary-consumer-review-v1/project`。保持 tests 的 `HERE.parents[2]/[5]` 层数；stage `source/eval` 指向本 overlay 的顶层 copied eval，添加仅供测试的 eval package marker。生产 envs、已有 evaluator、旧纯 SDK 包以只读路径借用；未以 working candidate 的后续字节作输入或结束 gate。overlay 的 47 份副本在归档时仍与 subject 相同；具体适配与来源在 `overlay-binding.json`。

实际执行：

- overlay 中 `test_formal_dispatch.py` / `test_formal_model_service.py` / `test_formal_settlement.py` / `test_formal_runner.py`：**63 passed in 22.60s**。包括 actual provided SDK 文件写入、1296/216 个 **Fake callbacks**、in-process CLI/Fake spawn 边界；无模型、物理 worker、MLX load、正式 RNG 或真实 owner action。root 的全 346 只核绑定/声明，未独立重跑全套。
- `test_independent_cache_phase.py`：**3 passed in 0.31s**；断言旧缺陷实际存在。cache 测试假设首个 receipt/settlement 由 Fake validators 判为有效，运行原 coordinator、真实 IO/hash/control-flow；不冒称物理/计量验证通过。
- `test_independent_service_cli.py`：**4 passed in 0.18s**；三个 counterexamples 与一个正确拒绝的负控。service 使用 Fake dispatch；CLI 为实际 main()/argparse，在第一个输出目录创建后截停，无 child process。
- `check_evaluator_copy.py`：仅执行冻结 runner 的 evaluator-copy AST，读取实际已有冻结 evaluator / manifest / 17 个 source members，写 `/tmp`；文件集合与原 member SHA 全同。未执行 grader、formal generation 或方法评分。这里的 source-map 为 literal fixture，不自称真实 G4 source closure。

所有命令/脚本、原输出和精简 probes 已保存；`pytest passed` 在 counterexample 测试中表示缺陷断言成功，不是 guard 修复通过。

测试并发边界：63 项日志 shell-open birthtime **2026-11-06T12:04:48.869440Z**，最后输出 mtime **12:05:11.772747Z**（22.903307s 元数据窗口；内部 pytest 自报 22.60s）。它可能与主 Agent 的开发 pilot 运行重叠；不能用 run.started_at 充当 worker epoch，也不归因任何 clock peak。得知 flight 在运行后暂停集中测试，仅只读/少量 metadata/artifact 检查及 `/tmp` 脚本准备。新增 7 项和 copy 检查在主 Agent 明确通知无 flight/model 后才运行，精确元数据见 `test-time-windows.json`。

## PC1 · P1：完成缓存忽略后续 attempt，measurement 混用 Y 与另一 world

位置：`run_inspection_llm_episode.py` 的 cached completed 立即 return；`inspection_measurement_loader.py::_measure_completed_llm` 只预查最后 terminal.completed，然后取 coordinator 返回的 Y，却固定读取 `attempts[-1]/flight/world.json`。

最小 `/tmp` 反例：attempt-00 构造成通过缓存 hash/intent/start/return/tree 检查的 literal receipt，首个 flight/settlement 由 Fake validators 接受，Y=.4；追加 attempt-01，仅有 `terminal.status=completed` 和 Q=.9 的 world，故意没有 intent/start/native settlement。原代码成功返回：

- Y=.4、episode_receipt 指 attempt-00；
- score.Q=.9、world SHA 指 attempt-01；
- retained_attempts=2，但仅验证 attempt-00、仅查 `literal-a0` settlement，额外 01 从未检查。

这证明了**缓存/measurement 控制流**漏验，未声称 literal first receipt 是真实飞行。即使首个 receipt 本来有效，也不能把额外不完整 attempt 丢掉或混入其 world。应在 cached completed 返回前要求它是唯一末端成功/不存在更晚 attempt；独立验证 canonical/cap2 整体序列，measurement 从同一被认可 attempt 取 Y/world/score，未知后续 attempt 必须阻断。

## PC2 · P2：readonly LLM measurement 实际创建 .lock

同一 literal completed cache，无 `.lock`。调用 `_measure_completed_llm` 后成功返回并新建 `.lock`；没有 reserve/settle/spawn hook 被调用。原因是它复用 effectful coordinator，后者始终 `mkdir(exist_ok=True)` / `.lock.open('a')`，禁止 hooks 不能阻止这些文件效果。readonly 模式应拒绝缺失的必要文件并只读验证，不创建目录/lock/plan/attempt。该反例也不需要真实模型/物理。

## PC3 · P2：stored formal classic plan 可被返回为 development

`measure_completed_episode` 的 development 分支只拒 `expected` 带 formal_dispatch；它没检查读到的 `plan.json` 是否 formal。移除 expected.formal_dispatch、设 phase=development 后，stored plan 仍有 formal_dispatch、case_id=`main-literal`，函数在 Fake physical validator 接受该 plan 的前提下返回 `phase=development`。实际 plan/hash/attempt 分支均运行；物理/score validator 为明确 Fake。

它没有证明 formal native gate 已被绕过——真实 validate_flight 仍可能重核 formal plan——但返回 data-role 可错误降格，违反角色来源绑定。应从 stored plan/已认可 namespace 确定角色，并拒绝 main/holdout/formal plan 与 development expected 的重标。

## PC4 · P2：formal service 在 gate 后另读未验证 raw control

`inspection_model_service.admitted_limits` 调 admit_dispatch 后，仍直接读 `.paper/control.json` 取 reservation/purpose，而不是使用该次最终 native 验证的 owner snapshot。

Fake interleaving probe：gate 提供的旧 snapshot/checkpoint 没有该 reservation；在 gate 返回时只替换 raw control，加入满足 service 字段的 reserved entry。service 接受 profile，checkpoint 仍无此 entry。这是对**authority split / 二次 raw-read 行为**的 unit 反例，不是一次真实 native 迁移、预算 API 或正常并发故障的复现，也未执行生成。formal reservation 验证应取同一 sole-native 验证 snapshot；不能让另一未证明 durable/native 的 raw 文件取代它。

## PC5 · P2：purpose bool repeat 被当 native integer slot0

service 正常 Fake slot repeat_ordinal=0，把 reservation purpose.repeat_ordinal 改为 JSON false，仍获接受。Python `False == 0` 令普通字段/dict equality 混同类型；应以严格 typed/canonical 比较匹配 purpose/slot。没有执行实际 model generation。负控：自洽更新 registry hash/purpose 为另一个 M2 receipt，但 slot 仍 M1，原代码正确拒绝，未把所有模型判断说成失效。

## C1：遗漏 formal flags 的 canonical 路径启动副作用（有限 note）

给 reference CLI 传 `/tmp/.../datasets/main.v1/cases/main-ID-L1-00.json`（只含 formal-role literal header），不传两项 formal-dispatch flags。实际 cli_dispatch 返回 None，main() 已创建 out；探针在创建后立即截停，未触 metadata/Queue/Process/model。没有证明 main case 成功 decode、physics 或方法调用可绕过。该观察只覆盖 canonical formal-role 路径与启动副作用；任意改名输入/hostile copies 不扩大声明。缺失 formal refs 应在此类已知 formal 路径的 startup effect 前拒绝，不能把本 note 提升成已证实的正式 flight 绕过。

## 正确行为与保留范围

- 所附 dispatch/plan 测试重跑：八条件精确 slot、case/condition/repeat/sampler/phase；packet 与 parent revoke；LLM model profile / native known settlement 内容拒绝。actual classic argv 保持 600s/1x/generated/private worker payload，坏显式 formal gate 在 out/client/process 前拒绝。
- provided SDK 使用冻结顺序完成 Fake main1296 / holdout216 callbacks，outer576/96；三套 envelope 在首个 effect 前打开，故障对照保留 classic216 / llm1080 / rootmeans576 的完整主矩阵，未写成功 completion marker。Fake 值按 root 内 repeat mean，不额外当 flight。
- 读取既有 SDK artifacts 独立确认 main outer design 与其 accepted fixture **逐字相同**；evaluator fixture bytes 也相同。holdout hypothesis 明示 once-only descriptive direction/size、无第七确认性检验，SDK statistics interval=none；未跑正式统计。
- 实际 frozen evaluator-copy AST 的独立 IO 检查复制完整 17-source closure、manifest 和 evaluator 原字节。它证明拷贝机制，不授真实 G4/source inventory 完整资格；实际正式 SDK/flight 输出尚无。

本轮定向检查新增 dispatch/runner/service/receipt/measurement hooks，沿用既有内层 physical/causal/score validator 的有界来源审查，没有逐条重新认证所有 legacy helper 或全部 32 sources。sourceclosure 最终库存未收敛，primary 之外的 secondary/clean/ablation/delay 不在本报告范围。实际 R/datasets 不存在，所有数据均为临时软件 fixture；未运行真实 formal RNG/model/physics/owner动作。working 的修补另版再审，旧 immutable v1 与本负证据必须保留。当前不批准 primary consumer，更不授完整 G4/main。
