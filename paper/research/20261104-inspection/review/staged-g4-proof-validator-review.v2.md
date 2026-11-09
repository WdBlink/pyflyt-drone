# staged G4Proof v2 修复独立复核

审查者 `/root/control_review`，同模型家族、独立工程上下文。只审未安装的G4Proof组件；没有改真实owner、生产63源或候选，没有模型/物理、正式seed/order/case生成。候选不提供namespace/dataset/holdout权限，本报告不将其他未接线触点当作此局部组件的既成承诺。

**v1五项具体反例已闭合；另发现一项accepted evaluator身份未交叉校验的P1，故当前组件仍有必要修复，不能据此授G4或formal GO。**

## 版本与独立检查

冻结binding SHA实核为 `16787754910e35b994d77b72825e6cf13ef931be21cb836d0c2a7f114c95d238`。源为 `91bd91d1ac5346508cecdc66584b67a72dbe421959cda276b8e6c533550fb512`；测试为 `d76673ed496febea615bc828ec460291b8482a3870800b4bf64f0c55fcb19c2f`。四项freeze文件均匹配，working source/tests也先验为同字节；按要求从working tests路径独立运行：**40 passed/0.22s**。未从改变parents定位的freeze测试副本误跑。

行号均指`planning/formal-admission-candidate.v2/proof-validator-freeze.v2/inspection_formal_admission.py`。

| v1发现 | v2复验结果 |
| --- | --- |
| P1：hash后重开JSON | 118–122行对同一bounded byte buffer验SHA与parse；独立在buffer读完后换closure JSON，旧source drift仍拒绝，注入实际触发1次。未再接受未绑定的新解析内容；关闭。 |
| P1：init/任意单依赖漏列 | 20–35、125–146、173–220、322–340行引入可信独立inventory及exact集合；改package init拒绝、增加原先absent eval initializer拒绝、即使重绑synthetic native/subject后用任意单依赖替换也拒绝；关闭原具体反例。 |
| P1：native pair/Brief/schema/pin | 149–170行实际复用Controller.lock/load。临时owner的checkpoint缺资格round、Brief正文漂移（pair仍一致）、旧schema、错误implementation分别由原生Refusal拒绝；关闭。未对真实owner调用此load。 |
| P2：后续revises撤回 | 238–271行显式revision使旧资格superseded；独立复验revision拒绝、无关later round正常接受；关闭。 |
| P3：1e999溢出 | 51–70行finite parse_float；正负1e999、NaN和duplicate key均拒绝；关闭。 |

正常synthetic proof仍接受。实际环境factory只做metadata/inventory读取：解析出的61份依赖与`qualification/eight-condition-pilot-intent.v2/dependency-hashes.json`恰为同一集合，两个envs package init已纳入，eval initializer absence明确记录。source_names为88项，含尚未安装的正式入口名称，**不是88份正式实现已落地或通过**。另用两个不同临时owner/SDK包验证lib与scripts清单union及owner_reader选用owner路径；此inventory fixture对owner-reader factory使用显式stub，没有冒称未来真实迁移/SDK组合已实际运行。

## 新P1：qualified source closure可覆盖accepted design的冻结grader

**位置：311–319行只验证design/population/registry引用，未跟随design.evaluator核实际入口；193–203行生产resolver只核固定evaluator manifest的SHA与文件集合，未核其17项成员bytes。** 后续322–340行只把实际source与G4 subject自述的source哈希比对，不能补足“accepted science的grader不可变”约束。

独立临时反例使用通过真实native Controller.lock/load校验的合成owner fixture及exact VerificationContext（状态由fixture写入，不是实际原生资格申请）：

1. accepted design声明`experiment/evaluator.v2.py`及原SHA `01fe9500…`，accepted plan/native/subject均正确绑定这份design。
2. 实际入口改为 `f4f1f31e…`，qualification source manifest认可新bytes，其subject/review/acceptance/native request均自洽，design中的原grader SHA保持不变。
3. `read_g4_proof`仍返回G4Proof。错误不是SHA链断裂，而是**一条完整新qualification链被允许改变已接受的科学grader**。

辅助库存反例保留原固定manifest原bytes/正确SHA和精确17个文件名，但17成员内容全部不匹配；factory仍返回context。factory本身只是resolver，单独返回库存不等于formal误收；它说明生产路径也尚未补足此交叉约束。该fixture同时验证了不同owner/SDK路径的union，不涉及模型或物理。

**最小修复**：在proof阶段按已接受design的evaluator Artifact核实际入口SHA；对它的固定import manifest逐项核成员SHA（以及既有exact集合/no-link约束），确保qualified source manifest不能覆盖accepted evaluator身份。可以在resolver或proof适当层验证成员，但必须形成不可绕过的accepted-design→entry→manifest→members链；不改grader、design或阈值。新增两个独立负例应分别定位入口漂移和manifest成员漂移，不能仅靠另一个SHA gate遮蔽。

反例/结果保留于 `/tmp/inspection-g4-proof-review-v2/probe_v2.py`、`independent-probe-results.json`；对应键为`accepted-evaluator-drift`和`factory_distinct_consumers_and_invalid_evaluator`。这不涉及正在运行的pilot结果，也不证明真实G4已经误授，只指出staged validator的误收路径。

## 适用边界与绑定

exact closure保证相对于**可信caller独立构造的VerificationContext**；不能从被审G4 JSON反喂inventory或自选owner_reader来宣称通过。package/current source检查也不等于OS、所有transitive imports、模型权重/硬件性能认证。原生完整性和已发生独立review的内容判断仍分别依赖原生协议与审查证据，G4Proof尤其不是dataset/opening capability。

本轮不实现修复，不授G4/main/holdout许可，不覆盖未来namespace/各层接线测试。原v1报告/证据保留。精简[复验binding](staged-g4-proof-validator-review.v2.binding.json)记录冻结/working/runtime SHA、40项日志和独立探针结果/脚本SHA；binding SHA为 `899f2d2f3865bab52a9d94f86e45e11537a86015b7432fbf01b470926c44be69`。临时结果中全部资格文件与native rounds均为synthetic软件fixture，非科学资格。
