# staged G4Proof v3 两点修复独立复核

审查者 `/root/control_review`，同模型家族、独立工程上下文。仅复核v2新增P1的accepted evaluator入口与17项import成员绑定，不重审其他formal实现或独立native-owner候选。v2报告及其开放发现记录保留；本轮不改真实evaluator/owner、生产63源或候选，不运行模型/物理，不生成正式seed/order/case。

**v2新增的accepted-grader P1在v3关闭；本次限定范围无剩余必要缺陷。** 这是staged G4Proof软件组件修复结论，不授G4/main/holdout或namespace/dataset权限。

## 冻结版本与差异

输入binding SHA实核 `99c73827c94649f0c43cf708c32365eaac21e6909d2a3df7baf8dfdc3f9bc4c9`；源SHA `c981449425a835fad40eda9bbdc47dc76d2aa0e473a26bde8312743739a1779d`，测试SHA `191d7af54bee6062053897d40fbd5f6e81f2c639a1f455004951a073ee94dc53`。四项freeze文件核验匹配，working source/tests与freeze同字节后，从working tests路径独立运行 **42 passed/0.22s**。

相对冻结v2，source diff确实仅新增两处检查及解释注释：

- v3 **202–203行**对固定evaluator import manifest的每个成员，用其既定SHA执行Artifact检查；原exact file set/no-symlink约束保留。
- v3 **316–318行**按accepted design中的evaluator Artifact核实际入口，独立于后来G4 subject/source manifest的自述哈希。其余proof/owner/context逻辑未因这两点修复更改。

## 独立、隔离的闭合证据

1. **完整重绑不能覆盖accepted入口**：临时synthetic proof先正常通过；随后只改变其accepted design所指grader入口，并让qualification source manifest、exact VerificationContext、subject/review/acceptance及native request/control/checkpoint全部认可新source。design中的原入口SHA保持不变。v2这类自洽链可误收，v3现在明确以`admission artifact changed`拒绝。不是靠source/subject/native另一处旧SHA未更新来碰巧拒绝。
2. **17项成员逐一单独漂移**：每次复制原固定manifest原bytes及完整实际closure，仅给其中一个成员追加标记。17个成员分别执行一次，**17/17均在成员Artifact SHA处拒绝**；保留文件集合及manifest本体SHA相同。断言检查具体`artifact changed`，未把随后缺少PyFlyt目录等无关拒绝当作修复证据。
3. **正例保留**：正常synthetic G4Proof仍接受；对实际项目的factory只作metadata/source inventory读取，61份物理依赖仍可解析，没有调用真实owner_reader/load或打开任何数据。原42项回归全部通过。

这些探针只在`/tmp/inspection-g4-proof-review-v3/`创建合成输入；真实evaluator入口和17成员未修改/导入执行。当前`runtime/autoresearch-state-reader-v1/lib/runtime.py` SHA `5f887d7b188203a3beda96127b055d4e5009017eced0533f55afacc2a3c19f52`与v2审查绑定相同；本轮没有据此审查或批准另一native-owner候选。

## 交付与范围

根agent发checkpoint静止通知时，本轮立即暂停R写入，测试、探针、报告和binding先保存`/tmp`；恢复通知后才归档。旧v2报告没有改成“原来无缺陷”，旧误收证据继续有效。

精简[复验binding](staged-g4-proof-validator-v3-evidence/binding.v3.json)包含冻结/working/runtime SHA、测试日志、独立探针/结果和source diff的精确SHA。它们证明这两处组件修复，不证明独立科学review确已发生、当前硬件性能适用、全部phase触点完成或实际G4已闭合。G4Proof仍不是formal permission；未来namespace、dataset、opening及各层接线保持各自门控。
