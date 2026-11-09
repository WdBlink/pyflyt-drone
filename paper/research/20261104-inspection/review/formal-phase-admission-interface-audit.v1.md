# G4 / formal phase admission 接口只读审计

审查者 `/root/control_review`，同模型家族、独立工程上下文。只读现有生成/reader/worker/模型服务/receipt/SDK接口与已接受合同；未导入执行这些模块，未运行模型、物理或生成器，未产生main/holdout种子、实例或排程，未修改冻结63源、owner、skill与既有证据。本报告是现状与必要集成边界定位，不实现或另设计scheduler，不重审已关闭的模型/统计正确性。

## 结论：正式阶段入口尚未贯通

**当前链是明确的development链，不能把case名称或purpose.phase改成main/holdout就当作正式准入。** 它在生成器、reader/ScenarioMission、LLM episode和service至少四层硬拒绝formal；另有经典入口、manifest和receipt接口只保证dev身份或调用方给出的绑定。现有SDK矩阵/排程可复用，但它们不授予G4或数据开封权限。

读取时最新已应用的是round-24，request SHA `e3a52fc4e538e8b80ec2dba55d34e7a672b26385b5a35474e13d41cc4dbefc59`，decision回指同一request，独立重算compact request digest相等；当前active round为round-25。G4仍needs_work的依据位于`.paper/control.json.rounds[-1].decision.finding.limitations`及该request的evidence映射：

- `qualification/g4-progress.v9.md` → `383c931f7965483d2180d5f4bfe0469cdfa49d58643f1d51eb426c755e5e8fe9`；3、15行仍声明needs_work、完整开发集和最终phase/source资格待闭合。
- `qualification/g4-next-integration-intent.v1.json` → `420dfcf13df10631bf18ff06dc347b8340fe30d9a567935bdb3a32422124ba46`。
- `qualification/eight-condition-pilot-intent.v2/binding.json` → `024e879176d6b6d591428d09864c82902dfd0e80cceb321402127776abb8a996`。

三条实际文件SHA均吻合已应用映射。这里没有一个已由本次读取证明有效的inspection G4-qualified/formal-open许可；native round记录的validation是`structure_and_binding_only`。runtime `apply_round`在1029–1113行核request/state/rules/evidence/documents，再把request/decision追加到`rounds`，不把finding文本自动升级为科学资格。**“round已apply”与“G4已过”必须分开。**

## 计划身份与split从哪里取得

唯一现有规范来源是accepted `experiment/design.ral.v3.json`（SHA `83f8f1a72f5261aaa5783c0d190f3788fafcd05bca1e7d16a6901c69c9abf0de`）：`cases`是72个main身份，`recording_semantics.root_metadata`给相同集合的domain/level；`split_contract`给分层计数、holdout身份、生成规则及开封条件。独立核对两套main集合完全相等，native plan receipt的design/population哈希也与当前文件吻合。

| 计划身份 | root数 | 固定实际episode / outer均值 |
| --- | ---: | ---: |
| `main-ID-L1-00..15`、`main-ID-L2-00..15`、`main-ID-L3-00..15` | 48 | 合并main总数见下 |
| `main-OOD-L2-00..11`、`main-OOD-L3-00..11` | 24 | main共1296 / 576 |
| `holdout-ID-L2-00..05`、`holdout-ID-L3-00..05` | 12 | holdout共216 / 96 |

每root仍为B0/B1/B2各1次，B3-M1/M2/M3与B4-M1/M2各3次，共18次、八条件；总1512是既有main+holdout分母，不含合同另列的secondary/development/有限infra retry。SDK外层`seeds=[0]`只是一次root mean，不是私有生成seed或额外flight。当前holdout没有OOD身份，不能自行补入其他strata。

上述均是**计划槽位**，不是已抽取的真实输入。design562–604明确该边界，population绑定为`planning/population.ral.v1.json`、SHA `307fbd5e77f841bf4a1b431a478527fa3aa85905694fd0242068d26c729a434c`。population5–9、28–38行要求每个dev-extension/main/holdout namespace使用独立fresh128bit master并在generation前持久登记，五个子流、跨namespace seed/semantic/config碰撞核验、固定whole-case拒绝/有限耗尽；不能把旧dev布局改名，也不能按方法表现换root或放宽几何/容量条件。共同仿真资产仍是固定条件，不是72条独立真实道路。

## 主调用链与最小门控触点

以下是需要补齐的边界，不要求新scheduler或重写已验证的控制/传感/评分器。phase-aware检查应从同一accepted design、正式dataset manifest和已应用的source-bound资格/开封证据取得输入，在最早的副作用之前执行；后续层还须比对同一绑定，不能只信CLI字符串。生成前使用计划槽位/G4/namespace intent，生成后读取或执行再核实际dataset manifest，不能形成“为取得manifest先绕门生成”的循环。holdout的密封生成与实际开封/读取也应分开：前者仍须在G4之后，后者另受main主分析已锁定和一次opening门约束。

| 现有文件 / 函数 | 当前硬限制或缺口 | 正式阶段的最小集成触点 |
| --- | --- | --- |
| `inspection_population.py:118–123,227–234,294–299`，`validate_population_case` / `generate_development_population_case`；`prepare_inspection_development_population.py:40–108` | 固定`split=dev`、`devext-`、输出dev；prepare只读dev prior suites、只造五个extension strata，并写`main_and_holdout_instantiated=false` | 在实际master/root抽取之前核G4 source/population/design资格；按已冻结72或12身份生成正确phase，持久namespace intent/完整generation audit。扩展现有跨namespace碰撞输入范围，保留同一joint law、5子流、有限耗尽和不替换规则。几何校验可复用，身份/phase检查须与几何分离处理；不能全局删掉dev检查。 |
| `inspection_cases.py:74–85,146–153`；`inspection_population.py:69–109`的reader/clean sibling | legacy CaseSpec固定dev；新PopulationCase与clean sibling也只接dev parent；`load_development_case`没有formal资格上下文 | 正式reader要严格核schema、phase、ID集合、lineage、case file/semantic SHA及dataset manifest；保留legacy dev入口。若使用既有clean intervention，仍绑定原parent/同level/同lineage，只移除事件，不能改成新独立root或从L3换成L2。 |
| `run_inspection_reference.py:33–53,130–158`的`worker/main`；`inspection_scenarios.py:12–16,98–100` | worker固定调用`load_development_case`，ScenarioMission再做dev validation；world/audit role也写dev。worker在读私有case前先做物理warmup | classic和LLM共享worker均需收到可信phase/input/admission绑定；runner启动前核一次，worker在warmup/构造mission之前再次核。然后使用对应reader；继续只让worker反序列化私有case，向actor只发原public stream。role、case/phase/manifest和world来源须真实记录，不能只改输出标签。 |
| `run_inspection_dev.py:18–59`；`inspection_episode_receipts.py:143–155`的`execute_local_episode` | classic CLI固定suite dev+未实例化formal；plan role为dev。底层经典episode函数本身只核条件/repeat/retry/clock，并不核G4/phase | classic正式入口在command/spawn前核phase与资格/开封证据，冻结到plan；底层重试/cache/验证逻辑可复用，但formal直接调用同样须有可信admission绑定，不能让外层检查被绕过。保留两次infra cap、unknown阻断与原Y。 |
| `run_inspection_llm_episode.py:67–75,93–95,144–156`；`run_inspection_llm_development.py:51–117,124–130` | episode固定dev-/devext-；reserve purpose固定`development`，归档路径为qualification/dev用途；runner没有formal phase/G4/open输入，manifest只有dev role，随后建client/启动worker | 在reserve之前核phase、case ID/manifest、accepted plan与G4/final closure；将phase及资格/开封证据绑定到episode plan、resource purpose、command、run与terminal/settlement。runner也须在client/model准备及worker.start前验证，避免只靠外层参数。复用原actor/client/worker及有限retry，不另启一套调度器。 |
| `inspection_model_service.py:27–60,97–146,163`的`admitted_limits/main` | held reservation和frozen plan有效仍只接受purpose.phase=`development`。其检查在MLX import/load之前并在请求前重复；service-intent134–140未保存phase/dataset/资格链 | 在现有admission中按明确phase分支：dev仍只dev；main须匹配已应用G4、formal manifest与episode identity；holdout再匹配同一次opening/锁定main分析。phase必须与已预约purpose和输入计划一致，并进入service-intent/离线meter期望。既有owner reservation不可改用途后冒充新phase。 |
| `inspection_measurement_loader.py:34–56`；`inspection_llm_episode_receipts.py:24–57,180–194`；`inspection_model_receipts.py:46–56` | 通用measurement loader**直接拒绝所有LLM条件**；classic formal门仅检查caller提供JSON的SHA和`status=qualified`，没有native应用/最终closure/split/open关系，plan对比字段也不含phase。LLM validator是实际请求/用量/score验证，但没有formal phase授权链 | 由同一正式expected manifest建立phase与case、plan、source/deps、evaluator、G4、opening关系，并验证native应用指针；classic沿用现有独立重计，LLM分支接现有LLM validator+完整settled archive证据，不能只给物理Y或把phase写进evidence。离线model meter需核其service-intent中的同一phase/许可绑定。 |
| `run_inspection_eight_condition_pilot.py:51–71,109,196–225`；`inspection_recording.py:21–49,78–176` | pilot入口固定原六dev根、`first_pilot=true`、全ID metadata、dev role/selection scope；SDK层本身不硬编码dev，也不核G4/open | 正式调用端从accepted72 metadata或12 holdout合同取身份，先过phase门再冻结order/sampler/input/source manifest，保持同一SDK/worker路径。main传原字节`frozen_outer_design`；holdout需要由已接受split合同派生并开封前冻结的12根outer设计，不能拿72根design改label或覆盖accepted v3。 |

这些触点共同防止两种问题：一是下层仍拒绝导致“上层写formal、实际跑dev”；二是只放开字符串后，dev记录/自建qualified JSON就可被形式上重标为formal。当前dev保护本身不是错误，正式phase也不是当前已实现功能。

## 可原样复用的SDK与执行能力

`episode_schedule`已有ID/OOD level metadata、固定root blocks、每root完整18槽与显式NumPy PCG64 schedule seed；可以接合同的72/12身份，无需为formal设计新排程算法。本审查没有调用它来生成正式顺序。sampler seed、schedule seed与私有root生成seed应分别登记，不能混称独立root。

`record_matrix`已有完整classic/LLM矩形、先建三个envelope、异常保留分母、缺失inner阻断outer、同root算术均值、raw/result/schedule SHA链和`frozen_outer_design`。`ExperimentRecorder`的exclusive产物和完整failure/stopped记录也可复用。它们的completion明确不是物理/科学接受，`primary_inference_ready=false`仍应保留；已有主统计适配继续接收完成的真实root means，本报告不改统计规则。

原有source/deps/evaluator字节检查、私有case copy、真实worker队列、actor/service计量、native reserve/settle、两次infra retry、unknown恢复和cache证据也应复用。新增的是phase/资格/split身份从前到后的强绑定，而不是重新实现已有模型计量或更换grader。新增正式门控/reader分支不在当前pilot已冻结63源的实际运行证明内；它们须前瞻版本化并独立验证，不向正在运行的pilot热改或把旧dev结果追认main。

## holdout一次开启的既有依据与尚缺的实际证据

1. **已有规范依据**：native accepted plan receipt、v3 design及population SHA；design584–600的12个holdout身份/216 episodes；同一八条件、固定模型/提示/工具/评分、原B1选择及根级分析规则。design600行明确它是主分析锁定后的单次描述性方向/量级复核，**不是第七项confirmatory test，也不能参与方法选择**。旧statistical-protocol文字中的“确认”不产生新增检验或α授权。
2. **仍须先实际闭合G4**：当前applied证据仍needs_work；完整开发pilot/余下开发集成、最终source/model/prompt/generator/split/retry/dependency/hardware资格及独立review，须形成与accepted contract相符并进入原生应用证据链的资格记录。单独一个任意JSON里的`qualified`不足；预算reservation也不足。
3. **main结束后的锁定证据**：真实main dataset/manifest、完整既定1296槽（含保留的政策失败及有限infra attempt）、固定输入/方法/source与正式root means、原定主分析输出及其锁定SHA/代码/选定classic关系。锁定后不再以holdout调整prompt、阈值、方法或population。合成SDK预检不是这些实际main证据。
4. **一次opening证据**：开封前固定holdout身份/派生outer设计、所用资格与main-analysis-lock SHA、同一方法/source/model、namespace intent/密封输入及一次opening身份；通过既有native evidence/operation链登记。phase admission须辨认已应用的同一次opening，重复进入只能恢复同一冻结输入与attempt链，不能换目录/operation ID重抽或再选方法。当前没有inspection专用typed holdout-open latch，不能因native operation可幂等就假定它自动强制全局一次开启。
5. **每次实际生成的资源前提**：继续使用同一owner的active round、accepted plan、真实预算确认和具体预约/结算。`runtime.reserve_budget:444–463`与`budget.admit:144–166`核这些通用资源条件；purpose是字符串，并不解析inspection G4/open语义。资源成功仅允许该预约范围内的动作，不会代替上述phase门。无需更改runtime/skill、放宽预算门或另开ledger。

以上列的是既有合同要求及接口尚需承接的证据，不是本报告自行批准这些阶段。当前不生成main或holdout，也不申请新方法、root数、阈值、付费或硬件范围。

## 绑定与未验证范围

精简[source-binding](formal-phase-admission-interface-audit.v1.source-binding.json)保存23份实际读取文件的SHA、无ledger/问答正文的native结构投影、round request重算及三个G4指针实核、计划ID分母核对。其SHA为 `d76cb8f197355a671f47ca4924450d2201d960ccfb4c5086c66ab92a1f85d40f`。accepted plan及population指针也已核对。control文件会因当前合法资源操作变化；绑定中的完整control快照SHA仅标识读取时刻，稳定引用是已应用round request/evidence，不要求live ledger停写。

本轮没有实现或测试formal gate、模拟绕过、运行任何科学条件，也没有审计所有历史目录是否存在其他候选开封材料。G4 needs_work判断限于实际读取的最新native应用链及其绑定进度文档；本报告不能授予G4/main/holdout执行、领域认证或完整平台资格。
