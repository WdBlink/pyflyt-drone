# Formal population writer：W1 修复有限复验 v2

结论：**关闭前版 W1（P2）。本次限定修复范围内未发现新的实质缺陷。** 当前/已 revoked 的固定 native finding 历史，都在原反例的新 entropy / canonical 路径重建前阻断。empty intent fd 先 fsync、再目录 fsync、再 entropy 的实际调用顺序已核对。该结论是未安装 source 组件复验，不是 G4 或正式数据生成批准。

## 绑定与差异范围

reviewer `/root/statistics_design_review`：独立上下文、同模型家族，不是人工或跨模型复核。按 autoresearch 保留旧失败与实际证据，不将 synthetic 测试当正式数据。

- 新 freeze：`planning/formal-admission-candidate.v3/writer-component-freeze.v3/binding.json`，SHA256 `1b41a242148df214626f21d5c3bcffd42fec0961c96bd9aaa9b2f7ee67a6ce11`；所有 files/working_sources SHA 独立核对一致。
- writer SHA256：`4bab05728bfdfead0252c00da9eb61e69e1d20bee32b3629ab2c2b90ea967451`。
- writer test SHA256：`eef14dcfd9e73ceb8d5ff388b3d07e851135350675e0a1f8f619fa4f809fdf99`。
- v2→v3 只改变 writer 和对应测试；其余 14 个 source/test Python 文件与前 freeze 逐字节相同。旧报告 `review/formal-population-writer-component-review.v1.md`（SHA `db0c3eb3ebb8b5ccf6e7ac71c8ff39db5f0146a7d17b6a6ae7ea1c2dfd27f327`）与反例全部保留。

已阅读完整 source/test diff，核对 `_require_unissued` 的 sole native owner reader、同一 snapshot 的 current-G4 验证和全历史 finding 扫描。namespace、holdout-opening、geometry 分别在对应 canonical claim / entropy / Fake generation 前调用该守卫；没有增加第二 ledger 或 scheduler。revocation 不删除旧 decision，扫描不按最新 active finding 过滤。

## 实际执行与原反例

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests/test_formal_population_writer.py --basetemp=/tmp/formal-writer-repair-provided-v1 -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider /tmp/test_formal_writer_repair_independent_v1.py --basetemp=/tmp/formal-writer-repair-independent-v1 -q
```

结果：**24 passed in 1.60s；独立 4 passed in 0.24s**，Python 3.11.15 / NumPy 1.26.4。全部为 `/tmp` literal entropy/Fake geometry；未执行实际 formal RNG、模型、物理或真实 owner 操作。

独立重复原 W1 的同一操作：literal main namespace → native-seal → 将整目录保留于 `retained-main-namespace-evidence` → 尝试登记另一组 literal seeds。对 current 与 revoked 两种历史，现均抛 `native finding already issued; recover its original evidence`；第二次 entropy trace 为空、canonical namespace 未重建、原 bytes 不变。旧版成功重抽的证据保持不变，未回填成通过。

所附 24 项还独立重跑了 namespace/opening/dataset × current/revoked 的全部 6 个新增回归：在各自旧 canonical 输出移走后，未触新 entropy / opening claim / Fake geometry，要求恢复旧证据。opening/dataset 回归中的 native marker 是临时软件 fixture，不是实际科学批准。旧 namespace/holdout/partial/CLI 函数边界未扩大复审；24 项中正常 main、typed locked-main / holdout opening、缺失/坏前门、失败保留、collision、exhaustion、末尾 source drift、固定 main 分母及禁止 self-review 的原对照继续通过。

## fsync 顺序及错误保留

新增独立观测在第一次 literal entropy 入口断言实际最后两个 sync 是 **empty regular fd → directory**；完整 JSON 随后仍 fsync regular fd 和目录。没有把函数名称/静态位置当成实际调用结果。

另在 empty regular fd 的 fsync 处注入一次 OSError：未触 entropy，empty `namespace-intent.json`、`namespace-failure.json`（`complete_intent_available=false`）保留，下一次调用在 draw 前拒绝。此结果证明列出的调用顺序和 process/error 恢复边界；没有进行物理断电实验，不提供任意 host-loss 或已丢未 native 登记 raw claim 的恢复保证。

## 依赖与保留限制

DatasetProof v2 source SHA `890ae2432ac6453c3f8c13ff95112703f858c43e35a50d24a6ab6abe65de1f93` 未改变。复用 `/root/control_review` 已冻结的原 P2 有界关闭结论：`review/dataset-admission-component-review.v2.md` SHA `dc1289e16ccc48e28a2fa204859520e9189b6d6e62fe39364323d29a1c06a8a5`；binding SHA `6b1106e0149f43608c660b0b2eeab847bcdcb9d4cb0db38d2104fd2483ec85c6`，本轮核对实际文件 SHA，未重新审查其完整逻辑。

本次仅复验 W1 守卫及 empty-fd fsync 修补，沿用前版其他有限组件边界；未重放已审 dev 候选、未重跑全 stage、未扩展 writer/consumer/worker 全 graph。production 未安装，实际 R/datasets 开始/结束均不存在，旧 production population 字节仍为 `bb709855…`。未来实际 fresh gate、唯一 owner 应用、完整 source closure、正式 namespace/dataset collision/内容/consumer/dispatch 准入仍须分别完成。本报告不授 G4/main，不批准实际 formal draw，也不声称任意宿主恢复或恶意状态篡改保证。
