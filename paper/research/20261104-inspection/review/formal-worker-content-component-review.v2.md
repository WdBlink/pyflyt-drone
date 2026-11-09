# Formal worker content component v2：闭包修复复验

结论：原 v1 的 P2「实际消费的 configuration_digest helper 未进入独立来源闭包」已关闭；本次限定修复未发现新的阻断项。此结论只覆盖未安装候选的来源闭包、配置身份和既有加载前门，不是 G4、正式数据或完整 formal 调用链许可。同模型家族、独立上下文工程复核，不替代人类领域审查。

输入绑定：`planning/formal-admission-candidate.v3/worker-content-component-freeze.v2/binding.json` SHA256 为 `87b7c1b8bd6679fe6394cffb484034c4093f733f565970240ef3fb1bfe40897b`。独立核验冻结 24 文件、对应 working 12 source + 10 tests 全部匹配。相对 v1，只有 `inspection_formal_admission.py` 和 `test_formal_case.py` 改变；新增 helper，其余 10 source、9 tests 字节相同。原 v1 报告与反例未改写。

`source/eval/inspection_formal_admission.py:32` 将 `eval/prepare_inspection_development_population.py` 纳入 REQUIRED_SOURCES；`:145–151` 的构造校验因此拒绝缺少它的 VerificationContext。新增 stage helper 与真实 production 文件逐字节相同，SHA256 均为 `9d564a45412165aab72795c9dca6cf61ad1d96f314066b0d0447e86947e5248a`。配置算法与 producer 未移动、复制重写或改变。消费链仍为 `inspection_formal_case.py:13,91` → `inspection_development_catalog.py` → 该 helper。

独立反例采用临时 synthetic native-shaped owner/catalog；在建立资格来源链之前，先把真实 helper 字节及其 SHA 绑定进去，随后实际 fresh-import helper、catalog 和 reader。没有用 fixture 的 placeholder 不匹配冒充修复生效。结果如下：

| 输入/变异 | 实际判定 |
| --- | --- |
| 正常配置、原 helper | 接受；configuration SHA `4790be2aa4a5a7065cc0438f48226d6b425b85a91d4b7cbaa64d7f8d3bc16634` |
| 将 manifest 配置 SHA 改为全 0、原 helper | 拒绝 semantic/configuration/noise identity |
| 把实际 helper 改成返回全 0，分别 fresh-import 两种输入 | 均拒绝 `admission artifact changed`；私有内容 decode 调用均为 0 |
| 恢复 helper 原字节后再次 fresh-import | 正常输入恢复接受；坏配置恢复身份拒绝 |
| 从可信 inventory 省略 helper | VerificationContext 构造即拒绝 |

这直接复验了原 v1 的实际 helper 漂移路径：漂移后的算法即使返回与坏 manifest 一致的值，也不能越过来源前门。证据脚本有当前 frozen working SHA guard，适配为新文件，未修改 v1 强 guard 或旧反例。

复跑命令为 `.venv/bin/python -m pytest -q -p no:cacheprovider <candidate>/tests/test_formal_case.py <candidate>/tests/test_formal_worker.py`，设置 `PYTHONDONTWRITEBYTECODE=1`，实际 **42 passed in 4.96s**。其中包含 `test_formal_case.py:101` 的内容后 fresh admission、`:196` 的同一有界 buffer 哈希/解码（读取后文件改变仍只认证返回 buffer），以及 `test_formal_worker.py:50` 的 admission/content 在 warmup constructor 前完成与坏输入不触达 constructor。物理构造器由 Fake 截断；未运行真实 world、模型、正式 RNG 或 owner 写入。

复现脚本、实际结果、测试日志、v1→v2 最小 diff、完整输入与产物 SHA 见 `review/evidence/formal-worker-content-component-v2/binding.v1.json`。未复审新 dispatcher/writer、真实 worker 进程集成、私有内容的科学/物理可解性或完整 graph；同 buffer 证据也不授予文件返回后永久不变能力。本轮已完成，报告与证据冻结后停止写 R。
