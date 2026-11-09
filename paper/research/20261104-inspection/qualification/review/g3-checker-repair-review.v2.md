# G3检查器v3因果seal有界封口复验

日期：2026-11-05。同模型家族独立上下文工程Verify。仅新增本报告，不改冻结的g3-checker-repair-review.v1.md、原run/world、旧receipt或production；未重跑600秒physics、重放或1×。

**在已发现反例和声明的入口范围内，F1–F3现均可关闭。** F1完整依赖集合/字节与F2强制22组实际replay绑定沿用前一轮已验证修复；v3进一步关闭F3的未来claim、缺同刻证据、引用inspection前survey包三项残余。原真实witness及11项检查保持通过。该结论不扩大为整个研究平台、任意输入的形式证明或G4。

## 版本身份

路径相对paper/research/20261104-inspection，源码相对repo。

| 对象 | SHA256 |
| --- | --- |
| 已冻结g3-checker-repair-review.v1.md | a5c9fa93b134be7fe3b03ba5bceff0d0c71a37913b07e3d9e7d720c2c2151cd6 |
| eval/qualify_inspection_reference.py（v3） | b59bf87d668557f293638f18e5d8dc2f0a1c3ac2fed8bbd70a7e2793b3451035 |
| eval/replay_inspection_reference.py（未再改） | 1c04a1340eca069398ca5b2e121e37df3701fd68b63fdad7413ea0e633159480 |
| qualification/reference/executor-realtime.v2.qualification.v3.json | 2739574c3d7b9f75e1f566ba37033962a87409744c0975c8671eb8db4de9c366 |
| qualification/runtime/reference-admission.v3.log | 4180455bfd3c95077fd59776dcae69c3a9a7cef18311fb2f922a413a55372c14 |

source/reference-analysis-v2归档5份源码逐SHA匹配binding，其中checker仍为70dff343…c2ec8；binding所指资格v2及22组replay.v2 SHA与原文件一致。旧发现没有被覆盖。当前checker相对归档v2的函数AST差异仅causal_chain_witness，F1/F2 helper及其调用未变。

## 极短独立复验

v3:90–105要求claim为native int/float、finite、0≤stamp≤report.t；至少一个引用measurement同stamp；该同刻引用还须属于所选成功inspect的实际qualified narrow视图集合，原两视图≥20m基线检查保留。

直接读取上一轮保存的`/tmp/g3-checker-repair-v2-control-evidence/chain-extra-inputs-and-results.json`三份**原始输入**，单独调用当前causal_chain_witness，全部返回None：

- report.t3.5但claim999：拒绝。
- claim3、引用只有2.5：拒绝。
- 引用inspection创建前的survey包@.5：拒绝。

此检查没有receipt、run/world SHA或qualify外围门，因此不能被“仅SHA gate挡住”混淆。另运行针对测试`pytest -q tests/test_inspection_reference_admission.py -k 'future_measurement or unbound_measurement or pre_inspection_survey' -p no:cacheprovider`，3 passed、7 deselected，耗时.07秒。保存的完整23guards/.11秒日志已读取，本轮未机械重跑全套。

对原run以归档14源和实际22组receipt直接调用qualify，passed=true、全部11checks与qualification.v3逐项相等，run/world/receipt SHA一致。实际witness仍是candidate-0-00005 / inspect job-0037 / report-candidate-0-00005 / observe job-0040 / local obs-0-289-0：report claim142.5对应成功inspect内的obs-0-285-0，原两视图最大基线25.495652m，新local在observe创建后144.5秒采集。原数据不需要改判或重跑。

运行闭包14份source与原run逐字节相同；原run SHA122e9f47…c5f871、world SHA8d60ea4b…4ad3c6及22组receipt SHA899bfadc…f0afeb保持不变。没有增加一次新物理、策略或科学样本。

reviewed by inherited model configuration, independent context, same model family

- 原F1/F2及F3残余在本轮明确反例范围关闭，原真实链和实际重放绑定可信；v1/v2分析历史分别保留。
- 22组receipt含完整raw error trace字段，但不是原完整physical-state或墙钟重放。72,000逐control tick lag数组和240Hz原位移未保存的限制不变；10Hz不能精确重建里程。
- 本报告只完成检查入口修复封口。完整G3需与既有物理/时钟/传感及业务审查组合；不授予G4、正式模型性能、真实vision或领域人工认证。human_review_pending。

报告完成后冻结。
