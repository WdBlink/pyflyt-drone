# Development collision catalog producer：独立有限审查 v1

结论：12项纯测试独立通过，未安装producer对当前合格dev输入的artifact/semantic/config身份、seed类型和同artifact provenance合并有实证支持；但**历史输入覆盖被现行physical admission阻断，不能接受其作为完整碰撞库存**。旧 `dev-cases.v1/dev-L2-03.json` 的真实只读导入失败，不能因已退役/无效就从development曝光历史中删除。无实际catalog或G4 GO。

2026-11-06，`/root/statistics_design_review`，分离上下文、同模型家族。本轮只读候选/契约/明确既有来源，未改候选、owner或生产；未运行模型、物理、RNG、正式数据生成或实际catalog writer。按主Agent要求在发现明确缺口后停止扩大扫描。

## 冻结与实际验证

入口为 `planning/formal-admission-candidate.v3/catalog-component-freeze.v1/binding.json`，SHA256 `c2640eb60d22a9c257b21ea5453214ee68abadac8540ae117da459906e33c2a2`；4份冻结文件及contract指定5份helper/dependency均核SHA一致。producer SHA256 `d8b3f209f91aca933cb5965478b67d56a85ab845130c796ab6f0522b0f7ea002`；test SHA256 `83a23b155c76e60ff8562851e37652ae172278ee631f74982536ece5407cfa03`。working candidate与freeze副本相同；没有installed `eval/inspection_development_catalog.py`。

独立执行：

```text
PYTHONDONTWRITEBYTECODE=1 <repo>/.venv/bin/python -B -m pytest -p no:cacheprovider paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests/test_development_catalog.py --basetemp=/tmp/development-catalog-review-tests-v1 -q
```

结果 **12 passed in2.95s**。测试读取31份既有artifact：v2的24 case + dev-population的5 case + 当前event/clean两份extra，**不是31独立root或完整历史**。测试猴补default_rng/SeedSequence使调用即失败；producer本身仅用NumPy版本值，无RNG调用或writer。局部fixture只复制既有输入/变更合成元数据，不创建正式实例。

已核语义：

- Artifact按run-relative路径、精确SHA、strict JSON读取；拒绝逃逸、链接/多hardlink、改bytes。manifest必须非空records且split=dev，case也必须dev；NumPy必须1.26.4。
- seed仅native int、0≤seed<2¹²⁸，bool拒绝。收集sensor seed、已知root generation seed、manifest master、namespace intent的master/root seeds；不声称这是model采样seed或所有程序RNG历史。
- case semantic digest由加载的case重算对照manifest；configuration digest从实际CaseSpec字段重算，不直接信manifest宣称的configuration hash。它剔除identity/schema/domain/seed及entity identity，保留level、ROI、风、实体物理/运动及事件字段。
- clean的parent_case/parent_sha等扩展provenance不进入CaseSpec物理配置投影；因此嵌套parent身份不会污染configuration digest，而event与clean的真实干预仍区分。生产入口先校验clean sibling及parent，不把直接dataclass metadata替换测试误称无效clean可被加载。
- 同一路径重复case允许unknown root-generation provenance补为known、known被extra无metadata读取时保留；两个不同known seed拒绝。不同artifact路径即使共享semantic/config，也保留各自provenance，不伪造新的独立root。

契约指定依赖核对与实际导入的既有CaseSpec/population/configuration helper SHA均记录在binding；这不是对所有潜在import路径的完整attestation或最终G4 source admission。

## R1：历史碰撞库存错误复用现行物理准入

`case_record()` 无条件调用 `load_development_case()`，其CaseSpec路径会执行现行geometry/physical校验。独立只读调用：

```text
build_development_catalog(R, [bound qualification/dev-cases.v1/manifest.json])
→ ValueError: ground proxies cannot be certified separated between samples
```

v1/v2各24条，semantic交集23；唯一旧输入为 **dev-L2-03**：

- 旧file SHA：`12806f5d90bde3a4b8ac99d89d91c00c5ca7dc95d6d85029dafdb83243b5bc49`。
- 旧semantic SHA：`ed93254799d5b4634b9ed50c974f21f67a62b387f189f35a1d1f38c3e4d38947`，原JSON的独立semantic重算一致。
- v2对应semantic：`f56e7fd7c80b3ce1bcf4df77a94dcf6e8ee5724403e06845a880cb1119078cc9`。

v2 manifest的repair_rule/parent_validation明确保留旧版本、只修语义无效输入且不使用策略分数。这至少证明它属于已生成/被验证并影响修订的development历史；本轮不据此声称该特定旧root完成过真实flight。旧seed可能已被新版本覆盖，并不能替代旧semantic/config身份库存。

最小修复方向是独立的**只读归档identity路径**：仍严格检查来源hash、schema/类型/finite值、dev身份和native seed，保留historical-invalid/retired状态及原bytes/semantic/config；不要让当前可执行性决定是否能被库存记录。不得放宽实际 `load_development_case` 或mission guard、改旧输入或把无效输入重新称合格。修后须补旧v1 manifest/该case的回归和既有拒绝规则检查。本轮不实施。

## 已证实的最小覆盖清单

| 来源 | 本轮证据及覆盖要求 |
|---|---|
| `qualification/dev-cases.v1` **及** `.v2` | 两个manifest均应绑定，24+24 artifact版本的semantic联合为25，非25独立root。31-case测试只纳v2，漏掉上述旧semantic；当前producer无法完整读v1。 |
| `qualification/dev-population.v1` | 5个ID/OOD×level roots，连同namespace-intent及其master/root generation seeds；已在31-case测试内。 |
| `qualification/dev-clean-sibling.v1` | 1份退役clean及manifest/derivation须保留。derivation明确移除事件并把public level改L2；不能以当前same-level L3 clean替代。31-case测试未包括它。 |
| `qualification/dev-same-level-event-pair.v2/{event,clean}` | 当前两份explicit输入都要有source/provenance；event与5root中的OOD-L3同bytes可关联，clean是另一semantic/config，不能增加独立root数。31-case测试包含二者extra。 |
| 实际run的private-case origins | 需要与完整来源库存对账，不能只信case_id或run的suite标签。本轮仅查下述5个已知样本，不能宣称全部使用史已覆盖。 |

五份private-copy样本分别来自 classic dev-L3-00、失败八条件dev-L1-00、fixed-load clock dev-L1-00、当前same-level event、当前same-level clean；全部精确file SHA能匹配上述已知来源，路径/hash对应收录binding。同一bytes可对应v1/v2两个来源，不能臆造唯一版本origin。

以后实际used-input coverage需覆盖正常/失败/中断/退役/诊断尝试。若某private copy不匹配现有来源，应先保留精确原件及origin证明，进入只读归档identity审查；不可静默忽略，也不可从当前配置重造历史。本轮未全仓遍历run/receipt；退役clean运行族及其他未抽样private origins仍待调用方对账。这是覆盖要求，不声称发现了另一个已确认未知case。

## 本轮处置

小binding为 `review/evidence/development-catalog-component-v1/bindings.json`，含freeze/dependencies、六份明确manifest及其文件SHA、旧输入错误/parent_validation、5个private-copy映射及测试log hash。未写实际catalog、未draw seed或开formal namespace。

限定判断：valid-input组件测试通过；R1及使用史覆盖未闭合，不能据31-case pass冻结为完整development碰撞库存，不能G4 GO。保留现有报告/输入与源码，等待修复另版审查；本轮完成后停止R写入。
