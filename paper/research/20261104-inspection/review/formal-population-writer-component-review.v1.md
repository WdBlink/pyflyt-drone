# Formal population writer：独立有限组件审查 v1

结论：**request changes，存在 1 项 P2（W1）one-shot 前门残余。** 正常 canonical 文件保留时，namespace claim、partial/error 拒绝重跑、完整分母和 pending-native 边界符合本次检查；但已 native 登记的 main namespace 证据目录移走/丢失后，writer 会再次触发 entropy 并写出另一套 intent。consumer 随后拒绝新 SHA，不能撤销第二次 seed 注册。当前 v2 不批准。

## 输入与实际验证

reviewer `/root/statistics_design_review`，独立上下文、同模型家族，不是人工或跨模型审查；按 autoresearch 只核既有工程证据。未安装候选 freeze：`planning/formal-admission-candidate.v3/writer-component-freeze.v2/binding.json`，SHA256 `76220f6825226ea2129c64e296b17da85ef29150ecf29412914065cec4c7912d`。全部 frozen files / working_sources SHA 在归档前重新核对一致。

核心 writer `prepare_inspection_formal_population.py` SHA256 `fb9086334c64fdd7d85acdc4e007161c3b7940b7fa930ce9e0edd26fc9766069`。已读全部 276 行、17 项原测试、相关 G4/native context、namespace、固定 phase layout 与 typed-main 调用边界。shared-law / formal wrapper 与已审组件的 `afa5f706…` / `5e231f18…` 相同；本轮未再重放已审 147 个 dev candidates。

实际执行，均为 `/tmp` literal/Fake：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests/test_formal_population_writer.py --basetemp=/tmp/formal-writer-review-provided-v1 -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider /tmp/test_formal_writer_independent_v1.py --basetemp=/tmp/formal-writer-independent-fixtures-v1 -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider /tmp/test_formal_writer_holdout_independent_v1.py --basetemp=/tmp/formal-writer-holdout-independent-v1 -q
```

结果：**17 passed in 0.98s；7 passed in 0.40s；5 passed in 1.06s。** 7 项中的第一项是成功复现 W1，不是修复通过。5 项 holdout 补查在主 Agent 要求收束前已完成；其后未再扩探针。作者 combined208 日志已核 SHA，未独立重跑其余全 stage。Python 3.11.15 / NumPy 1.26.4；未调用任何真实 formal entropy/SeedSequence/default_rng。仅预设 literal 模拟 entropy，geometry 全为 Fake，不生成真实 main/holdout case。

## W1 · P2：已登记 namespace 的 canonical 证据缺失仍重新抽取

位置：writer `write_namespace_intent`，line 97 只取得 G4；line 119–128 只用 canonical folder/file 存在性作 one-shot claim，line 133–142 触发 entropy/SeedSequence。没有在这些副作用前独立检查同一固定 `NAMESPACE_FINDINGS[phase]` 是否已在 sole native owner 中 issued/applied。

精确反例（`independent7-tests.py::test_native_applied_namespace_moved_as_evidence_does_not_block_new_entropy`）：

1. 在 `/tmp` synthetic owner/source fixture 中，用 literal seeds 写 main intent，随后按同一已审 namespace predicate 的测试机制 native-seal 该 intent；owner 中已有 1 条固定 namespace finding。
2. 将整个 `datasets/main.v1` rename 为 `retained-main-namespace-evidence`；所有原 bytes 保留，只使 canonical 路径缺失。这模拟证据路径移走/丢失的恢复状态，不是假装正常并发能触发。
3. 再调用同一 writer / G4，改用另一组 literal seeds。当前 writer 成功返回新 canonical intent，第二次 entropy trace 为 `randbits(128) → SeedSequence → randbits(32) → randbits(32)`。

| 证据 | 旧 native 登记 | 新 writer 返回 |
|---|---|---|
| root seeds（literal） | 10000…10071 | 20000…20071 |
| intent SHA256 | `ba4a768c139ca3cc8197a36900045083caeb1782f89831ffa7f2527ccbf218b9` | `d75f635d577c586cf522fd61e7e428b5bc07c4d4cd10e6ee10703a64cee7922a` |

新 `NamespaceProof` 随后实际拒绝旧 native evidence SHA，因此**没有证明错误 dataset/flight 已获准入**。缺陷范围是 writer 已在已知 one-shot 身份之后再次注册 seeds；消费者的后置拒绝无法代替 entropy 前门。此复现依赖路径移走/缺失，不是断电或正常 concurrent writer 实验。

最小修复方向：在 namespace 副作用之前，经 sole native reader + current-G4 验证固定 finding ID 的所有历史，包含已 revoked 的 issued 记录；发现已登记则要求恢复原 bytes/证据，不再抽取。opening / dataset writer 宜同样使用各自固定 one-shot finding 的历史检查。后两者本报告不声称已各自复现；这是同类最小守卫的建议。未知、未 native 登记且又遗失全部持久 claim 的宿主恢复问题仍不能由本测试证明已解决。

## 其余已核边界

- namespace：G4、固定 phase layout、typed 非空 dev seed/semantic/config catalog 在 claim/entropy 前检查。malformed catalog、missing G4、source drift、缺 holdout opening 的测试均未建目录/触 entropy。canonical main folder 的 exclusive mkdir 和 intent `O_EXCL` 在正常重复调用中阻止二次 draw。
- entropy 或 root/master collision：保留 empty/partial 或完整 intent 及 `namespace-failure.json`；不换 seed。collision 必须在得到新 seed 后才能比较，当前代码先持久完整 intent 再标 blocked，不能表述为“collision 分支没有发生 draw”。
- durability 事实：独立 fsync trace 显示首次 entropy 前已 fsync 目录，但 empty regular intent fd 尚未 fsync；其 regular fd 在完整 JSON flush 后才 fsync。注入 claim-directory fsync 失败时，保留 empty claim 和 failure，重试被挡且未触 entropy。后续版本可在目录 fsync/entropy 前补 empty fd fsync，明确实际合同；本报告没有测试或宣称物理断电恢复。
- opening：无 RNG；依赖 typed locked main / DatasetProof、固定 holdout 路径和 native 主链。正常第二次 opening 调用拒绝，改 main independent-review bytes 在建 holdout folder 前拒绝。
- geometry：先 namespace/G4、typed dev catalog 和 holdout 的 typed main exclusions，再 exclusive claim `cases` / `generation`。共享生成入口逐 root 用 namespace 的 metadata/登记 seed；audit/case 先保存，随后检查 sensor/semantic/config collision。exhaustion、case 写入故障、collision、末尾 source drift 均保留 partial/failure，不写成功 manifest，不自动重跑。
- main Fake 完整输出为 **72 roots / 1296 actual episodes / 576 root means**；额外 holdout Fake 完整输出为 **12 / 216 / 96**，顺序及八条件重复数未漂移。数值是计划分母，不是已执行 episodes。
- holdout 补查：main 已知 root/sensor inventory 在新 geometry 前加入排除；登记 root seed 撞 main sensor 时未 claim cases/未调用 Fake。生成出的 sensor、semantic、config 分别撞 locked-main 时，首个 raw audit/case 已保存，随后 blocked，无 manifest。没有用弃样/replacement 处理 collision。
- CLI：独立测试 main() 的 path/SHA 成对参数、显式 phase/intent 路径一致性拒绝，以及 Fake native context 下的 namespace 正例。返回状态为 `artifact_written_pending_independent_native_binding`。没有测试真实安装后的 CLI transport；当前 staged ROOT/package 路由不被当成已安装 production。

writer 未生成 independent-review，也未自写 dataset native finding；这些仍须真实独立审查及 sole owner 应用。正常文件保留时的 one-shot/partial 行为通过，不关闭 W1 的 native-history 恢复窗口。

## DatasetProof 依赖与资格边界

依赖 source `inspection_dataset_admission.py` SHA256 `890ae2432ac6453c3f8c13ff95112703f858c43e35a50d24a6ab6abe65de1f93`。复用 `/root/control_review` 的 bounded P2 修复结论，未重复其完整审查：报告 `review/dataset-admission-component-review.v2.md` SHA `dc1289e16ccc48e28a2fa204859520e9189b6d6e62fe39364323d29a1c06a8a5`，binding `review/evidence/dataset-admission-component-v2/binding.v1.json` SHA `6b1106e0149f43608c660b0b2eeab847bcdcb9d4cb0db38d2104fd2483ec85c6`，本次已核实际文件 SHA。其原 P2 已有界关闭，不等于本 writer W1 关闭或完整 graph 批准。

实际 G4 formal 许可不存在，production 未安装，实际 R/datasets 在开始/归档时均不存在。本轮只写指定 review/evidence；临时 `.paper` 全为测试 setup 构造的 synthetic owner，未调用或改变真实 owner。未运行模型、physics、formal RNG、正式 seed/data；未扩大为 consumer/dispatcher/全图准入审查，不授 G4/main。W1 修复需另版独立有限复验，当前报告/证据冻结保留。
