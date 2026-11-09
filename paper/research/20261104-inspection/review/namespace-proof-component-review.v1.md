# namespace / phase 只读组件独立复核 v1

审查者 `/root/control_review`，同模型家族、独立工程上下文。对象仅为未安装的artifact/owner/phase/seed-intent链，不是完整formal系统。未修改候选、生产63源或真实owner，未运行模型、物理或科学RNG/geometry；合成fixture仅使用静态literal seed和已公开计划ID。未审实际native-owner迁移或pilot表现。

**本组件有两项P2需修复；其余已声明的namespace固定身份、撤回及holdout读取顺序检查在本轮通过。** writer/generator、实际dataset/raw/statistics完成认证、跨main/holdout collision及最终source inventory本来就在not_yet_implemented中，不作为本轮缺陷。无G4/main/holdout GO。

## 绑定与验证

冻结binding实核 `38ecf56c0dc841ab425c0206715e370b1cce24e843da6024bd723191c2de7911`，七项文件均匹配。namespace源 `2f4197da345c9c8a49ff877e26baff0848213ba5a18fca157106ba975986ab07`；phase源 `ddafaf5322f4b40d63925e0c2ad37b440b942ac7c1fd2c5ff6445f9ebb4fa818`。working四份源/测试先核为freeze同字节，G4Proof使用已复核的v3 `c9814494…`实际函数，Artifact类也确与namespace helper共享。

两组测试按要求分进程独立复跑：**namespace26 passed/1.17s；phase17 passed/0.03s**。以下行号指`namespace-proof-freeze.v1`内相应冻结文件。

## P2-1：后续owner快照只重核部分G4条件，可返回已不满足完整G4的proof

`inspection_namespace_admission.py:47–58`只重核active/task_stop、subject中的三个owner identity字段及finding绑定；`read_namespace_proof:130`完整G4验证后，139–140和191–192行选择的新owner快照没有再次核accepted plan packet/canonical frozen_plan关系，也没有复核qualified source bytes。

两个相互独立的临时反例均在首次完整G4通过之后、第二次owner_reader调用时注入：

- 改临时owner的plan packet，保留原frozen_plan/subject identity；更新control/checkpoint成为一致pair，Brief/schema/implementation仍由实际fixture NativeController.load通过。namespace继续读取intent并返回72-root NamespaceProof；立即对同一最新状态重跑原read_g4_proof则拒绝`accepted plan and live dispatch owner required`。
- 只改临时项目已qualified的mission source bytes。后两次owner读取及partial G4检查仍允许namespace返回；立即fresh G4检查拒绝`admission artifact changed`。

这不是要求返回对象永远抵抗之后的外部变化，而是**在本次函数显式重新选择的最新快照上，资格谓词已经比初次G4验证弱**。第一次校验结果不足以支撑此时返回值。

最小修复：复用完整read_g4_proof的谓词，对已经由原生reader核过的同一owner快照与qualified sources再次验证，放在声明的intent-read/最终返回边界；不要再维护一套只核三个identity字段的近似算法。通过可信VerificationContext传入该已核snapshot可避免额外跨快照读取。caller仍须在自身副作用边界调用，不把NamespaceProof当永久能力。

## P2-2：phase护栏未约束实际SDK的top-level conditions/seeds

`inspection_formal_namespace.py:43–69`核`recording_semantics.condition_repeats`及root/episode分母、metadata/split，但没有核实际design的`conditions`与`seeds`字段。

独立局部反例从已接受v3 design副本出发，只删除top-level一个condition，或只把top-level seeds改为`[0,1]`，`accepted_phase_layout(...,'main')`两次都接受并仍报告1296 episodes/576 outer means。其semantics.repeat字典虽保留八配置，真正SDK grid已不一致。当前真实accepted design没有被改动；本结果仅证明函数对其声称固定的full-grid合同保护不完整，不声称当前pilot少跑某一arm。

最小修复：同时核top-level conditions为原八项的确切列表、aggregate seeds为严格整数`[0]`（不接受bool替代），并保持原repeat/grid字段一致。无需改变方法数、实验数或阈值；科学合同变化应走既有显式adapter修订，而不能由此护栏静默接纳矛盾grid。

## 通过的反例与明确限制

- main/holdout合成正例分别返回72/12 roots，固定1296/216 episodes与576/96 means的实际phase测试通过；未生成这些真实实例。
- G4在第二次owner读取被revises撤回时，namespace在读取seed intent前拒绝；第三次读取才撤回时，在返回NamespaceProof前拒绝。未把后者写成“从未读过seed”，它在当时有效的前置检查后已经读过literal intent。
- 初始intent bytes重写即使给新SHA，也被旧native evidence绑定拒绝；fully rehashed foreign source ref、固定路径`./`别名均拒绝。原测试还覆盖namespacefinding别名/重复种子/布尔seed/开发seed碰撞和错误holdout父链。
- holdout读序先校G4、当前owner，再核main lock和opening，之后才读取holdout intent bytes；opening固定路径/ID、12ID roots和descriptive role受到检查。

`read_primary_analysis_lock:61–101`绑定typed lock→分离subject/review→design/source/四项原始证据Artifact→native应用。这里的哈希和独立review链**不等于任意raw JSON已被认证为真实1296集完成**；其producer/reviewer以后必须通过实际loader/SDK/raw/statistics完成认证才mint，正是本候选明示的未完成范围。本轮不把该边界误报为已承诺却没实现的raw grader。

另用合成round-list逆序、pair重哈希后，reader仍接受当前已存在的父链。这个事实仅说明v1检查了**读取前置关系**，未证明native应用/seed生成的历史先后；fixture不是实际原生apply轨迹，当前writer/producer尚未实现，故不另列违约P1。后续若声称G4先于namespace mint、main dataset先于lock、opening先于holdout intent发布，还须在对应writer/索引/应用链落实，不能引用本reader正例作为历史因果证明。

## 证据与范围

独立脚本/结果位于`/tmp/inspection-namespace-review-v1/probe_namespace.py`与`results.json`，两份独立测试日志同目录。精简[binding](namespace-proof-component-review.v1.binding.json)保存freeze/working/helper/design SHA、上述脚本/日志SHA及结果，SHA `b691caa98cee9f2b062dba12895acc5191a799a7c4aa1bea2909018b6b350928`。

这些都是隔离、fully rehashed native-shaped软件fixture，使用同v3 G4Proof与临时owner，不是对实际迁入d809版本的native-owner或活pilot作审计。没有真实formal namespace、数据开封、模型预约/dispatch或资源用量发生。本轮只报告两项当前组件缺陷与其余边界，不批准完整formal/G4。
