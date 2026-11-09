# G1 完整 grid 独立工程审查 v1

日期：2026-11-05。角色：独立上下文工程审查，继承主 Agent 的模型配置；不是跨模型审查，也不替代领域人工复核。只修改本报告，未修改执行器、runner、契约或已有诊断数据。未重跑100-slot grid，未审阅正在开发的protocol/scenario。

## 初审结论

**旧runner不应直接用于完整G1资格判定。** 已复现一项P1误收：heading_changes执行器始终直飞仍可被判通过。另有需要在100-slot运行前补齐的门槛冻结、失败留档和契约绑定问题。诊断10/10、full_grid=false、qualified=false的标签与实际记录一致，不是完整G1通过。

主 Agent 已根据中间审查消息开始修复。以下先保存旧版本发现及精确证据，后续修复结论单独追加，不能将新源码SHA回填旧诊断。

## 已审阅的初始绑定

| 内容 | SHA256 |
| --- | --- |
| envs/inspection/control.py | 611c0e8cf51bef2623c45d03047725e67f4a623b600e7e34b3dd18b04b7f721c |
| envs/inspection/flight.py | 02eb78ff829c0135aafd0110dc2c5b634a59eed5b2be6fdc5a33adc4fa7a73a0 |
| eval/qualify_inspection_g1.py | 597159e0766a80b33e000e27098b7712c689e4217574f4ed9a2d3d9e6c55b1b9 |
| planning/gate-contract.v1.json | 50d77ef7db12c8d35864594fd0b63b62c40b312ceaad6bd7ea276a5b6e432202 |

qualification/g1-diagnostic.v1/manifest.json中的上述源码快照均匹配初审读取时的文件；contract_sha256匹配gate-contract.v1。10个slot的实际文件SHA全部匹配manifest，各family的root=0/1均已completed，operational_error=null、passed=true。诊断记录的物理审计计数均与物理步相等：60/120/180/240秒分别为14400/28800/43200/57600次。没有将后续新文件或新门槛回填这10份历史记录。

## 初审必要问题

### R1 / P1：航向变化验收使用被测执行器自身目标，误收未执行航向请求

旧runner:33–35发出0°、+45°、−45°、0°请求，但52行把实测ground course与flight.last_targets[i].heading比较。last_targets又由flight.py:140–142的被测target生成路径写入。因此目标生成错误能同时污染执行与验收参考。

真实、有界故障注入：仅在内存将FlightWorld.target替换为“heading_changes模式始终返回FlightTarget(0, altitude)”，不改文件、不改位置速度；运行run_case("heading_changes",0,120)。旧runner在前后SHA均为597159…时得到：

- passed=true，operational_error=null，simulated_seconds=120；
- physics_audits=physics_steps=28800；
- 报告settled_max_course_error_rad=0.0046165752732036935；
- 从t=39.9→40的真实位置差独立算course=−0.004586974924952721rad；该时刻请求为+π/4，独立误差0.7899851383224012rad，约45.26°。

这直接证明该检查可把“从未执行+45°转向”当作该family成功。不是对真实诊断轨迹的失败判定，而是验证器错误产物反例。

运行100前必要修复：验收参考从独立冻结请求日历取得，不读被测target()/last_targets；将上述错误target反例纳入有界检查，确保拒绝。

### R2 / P2：门槛契约未完整驱动验收，航向family的窗口未冻结

旧runner:100–104只从契约读取families/duration/roots；安全、跟踪、退出与成功率均为源码硬编码。当前许多数值恰与gate-contract.v1相同，因此未观察到这些现值被放宽；但contract SHA不能证明实际执行了它声明的可配置门槛。

尤其旧runner:56–69使用15秒转向稳定窗口和10°最大course误差，而gate-contract.v1:30–35只有settle_seconds=20及距离/退出门槛，没有heading_schedule或10°阈值。15秒窗口实际比仅20秒后检查更严格，问题是未声明/无法由契约复算，不能把它说成门槛已完整冻结。heading_changes还把未计算的位置误差填为0，不能将这个0解释成真实位置跟踪精度。

运行100前必要修复：明确冻结family请求日历、settled course门槛、各phase稳定窗口、sortie退出切换时刻与family飞行边界，并让runner和FlightLimits读取同一契约。原诊断继续绑定旧契约/旧源码；新执行用新版本契约，不回写历史。明确跟踪RMSE是合并所有机体/时间的slot指标，并保留最大误差；若要声称逐机RMSE门槛，需单独计算，不能从合并RMSE推导。

### R3 / P2：已尝试slot可能仍标unstarted；断连失败丢失已有记录

旧runner:105–111预建完整grid是正确的，但112–124直到case结束、结果文件写完才保存slot状态。KeyboardInterrupt或进程终止落在run_case期间时，已经实际运行的slot仍留为unstarted，没有running/uncertain证据。合格判定不会因此通过，但失败尝试记录不完整。

旧run_case:78–80在finally直接flight.close；若断连抛异常，已有trace、summary、实际时长被外层117–119仅含operational_error的对象取代。外层确实保留了一个失败文件和slot，不是静默删除整个分母，但无法据该文件检查失败前的物理状态。

save():110还直接覆盖唯一manifest，写入中断可能截断grid文件。常规case错误的处理并不能抵御这个保存阶段中断。

运行100前必要修复：case开始前持久化running与时间，初始化/运行/断连错误分开保留已有记录；manifest采用同目录临时文件加原子replace，结果文件写完/hash核验后再提交slot终态。中断后保留running/uncertain，不把它视为从未尝试，不自动抹除失败重试。

### R4 / P2：契约字节和执行来源绑定不足

旧runner:100解析契约、106再次读取契约字节求SHA，未保存契约原始字节，也未检查其后变化。源码快照只覆盖glob选中的本地模块和runner；物理上游版本/源码/资产未在该manifest中直接绑定。全部inspection模块的glob还把未参与G1执行的评分代码纳入了冻结范围。

导入control/flight发生在17–18行、快照读取发生在101–102行；若这些文件在两者之间被编辑，快照可能不同于已加载代码。末尾sources_unchanged不能单独证明导入版本与快照一致。这是代码路径审查发现的竞态，未通过改写文件重现。

运行100前必要修复：一次读取并冻结契约字节及其SHA，保留实际契约快照；绑定明确G1执行闭包，而不是其他正在开发的模块；在新独立进程中验证导入前后G1源码字节稳定并禁止运行中变更，保留运行时版本或既有环境/物理来源证据的哈希。每slot仍通过manifest与结果SHA关联同一执行版本。当前10条诊断的旧快照保留，不用当前文件的SHA替换。

## 已验证且未发现坐标错误的范围

### 每个240Hz积分后审计

初始flight.py:25–31覆盖stepSimulation：调用pybullet.stepSimulation后立即调用after_physics(t)。_audit:86–121直接读取PyBullet新位置、姿态和世界速度，未依赖尚未刷新的PyFlyt缓存state，且每次查询contacts。因此不能把它描述成仅120Hz审计。

独立运行一个真实控制步（单机、wind=(1,1)m/s），包裹现有after_physics观察回调：

- 回调时间为1/240和2/240；
- 两次位置不同，physics_audits=physics_steps=2；
- 实测空气相对速度范围约19.99408–20.0m/s。

原诊断的长记录审计计数与此代码路径一致。完整安全结论仍受所实现检查集合及数值状态有效性约束；未用10Hz轨迹冒充240Hz全状态记录。

### 风下airspeed与course

初始flight.py:97用world velocity−wind求空气相对速度，139–148控制同样使用该测量；控制器通过measured_airspeed进入速度环。course取世界水平速度方向；air_heading_for_course按crosswind做crab correction，不把body速度直接当ground course。

对记录中的2m/s风向，独立数学核验0°/90°/180°/−90°四个desired course：20m/s空气速度沿计算出的air heading与风叠加后，ground course误差均小于1e−12rad。诊断偶数root无风、奇数root是不同方向的恒定2m/s风；只支持这个恒定风设计范围，不支持阵风/湍流或未知风估计质量。

### 五family现有诊断含义

- straight：20秒后横向误差及course；四机/单机分配按root固定。
- heading_changes：目前诊断表面通过，但旧验收有R1反例，不能用于最终资格。
- loiter：20秒后圆周径向误差，且全程安全审计。
- survey：capsule路径几何误差及全程安全；诊断没有独立记录phase/lap完成证明，不能从此扩展声称搜索或ROI覆盖完成。
- sortie_exit：先survey后exit，20秒phase稳定后跟踪误差；退出geometry/course连续dwell按物理步计数，现记录均超过20秒。任务只是物理退出圆控制，不证明完整任务工具/退出身份语义。

## 审查状态与限制

本报告记录的P1反例及逐积分hook/crab数学核验由审查Agent直接执行；10份诊断记录与manifest由程序汇总/哈希核对。没有运行100个slot，没有审阅新protocol/scenario，没有调用科研模型/API。一次针对旧签名的追加cleanup夹具遇到并行修复后的新contract参数而报TypeError，该调用没有进入运行，也未作为实现缺陷证据。

完整G1结果仍需修复完成、冻结契约与代码、实际100-slot执行及结果审查；human_review_pending。工程资格不等同于领域有效性复核，更不能自动升级为G2–G4或论文结果。

## Attention（初审）

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- R1：旧验收误收真实“恒定直飞”的错误目标执行器，必须修复后再运行100。
- R2–R4：门槛冻结、每次尝试留档和执行来源绑定须在长grid之前补齐。
- 此为旧版本初审；后续修复需要单独追加状态，不能覆盖旧诊断或用10/10替代完整资格。


## v3修复复验与运行前最终结论（追加）

**可以按当前冻结的planning/gate-contract.v3.json开始默认100-slot G1执行。** 本次范围内R1–R4已得到针对性修复，未发现剩余阻断运行的问题。这个结论是“允许开始收集资格证据”，不是“G1已通过”；最终仍须检查100个slot实际完成、逐family成功率、操作失败与版本不变状态。不得把历史v1的10/10诊断改标为新v3通过。

最终复核绑定：

| 文件 | SHA256 |
| --- | --- |
| envs/inspection/control.py | 611c0e8cf51bef2623c45d03047725e67f4a623b600e7e34b3dd18b04b7f721c |
| envs/inspection/flight.py | 272efaf8ad0d4aadbfefaeec25e8be64a75c601e5fd6d85407f4ccd9e1fd9686 |
| eval/qualify_inspection_g1.py | 057bf4dad4d01d83c2fb164bd00dde512e2c878a9f34fa780ab5841c64b0528b |
| tests/test_inspection_flight.py | 4b0141586eca08b4eaddd6f0841c6db2cd5a580684eeb22db71a1d500de98508 |
| planning/gate-contract.v3.json | 6723955a0efe68801af57fd3adf1a1b94c5c8cdaea0e7d0437fc84862a7b37de |

### 修复对应与直接检查

- R1关闭：runner:53–56从冻结heading_schedule提取独立requested_heading，74行直接比较真实ground course与该请求，不再读取target()/last_targets作验收参考；88行要求存在settled样本。新增test_wrong_heading_generator_fails_independent_schedule使用同一真实四机120秒错误target反例，确认operational_error=null但passed=false，独立course误差>0.7rad。
- R2关闭：FlightLimits.from_contract（flight.py:46–50）实际传递高度、空速、滚转、间距、风和退出几何门槛；runner:40–44、54–55、75–91、165–168读取风/半径/航线采样、独立请求日历、phase稳定窗口、跟踪、退出与逐family成功率。v3新增显式heading schedule、10°门槛、family边界、切换比例、采样范围和root_ids；与v1共有的G1字段值全部未改变。20秒settle是原契约值，现在真正用于验收，不将旧源码的15秒窗口继续冒充契约。实际频率仍为本次已检查的240/120Hz实现。
- R3关闭：runner:28–34采用同目录临时JSON加原子replace；146–148先持久化running与started_at，157–160在case文件原子写完并求SHA后保存终态。run_case:97–103单独捕获cleanup错误并保留trace、安全统计和模拟时长。中断原子写测试确认旧manifest保留running；不把中断slot记为unstarted或成功。运行中断后仍须人工/程序核对running状态，不应自动删除该尝试。
- R4关闭到本次运行所需范围：runner:21–25在导入本地控制/飞行模块前捕获三份源码，127–128检查导入后字节；123–124仅一次读取契约原始字节，137–138保留其SHA及解析快照，163行检查结束时原文件字节未变。133–140绑定PyFlyt源码/机型资产、PyBullet二进制哈希及PyFlyt/pybullet/numpy/numba版本；162–168把本地源码、契约和依赖不变性纳入qualified。冻结只包含G1本地执行闭包，不再把正在独立开发的评分/protocol/scenario文件混入。执行应使用新独立进程并在整个grid期间保持这些绑定输入不变。

本次按v3计算默认grid：5个family × root_ids 0…19，共100个slot，每family20个；成功率门槛0.95，未删除失败分母。普通已完成但passed=false的case仍在family分母；任何operational_failure使qualified=false，这比仅用0.95允许操作失败更保守。显式--roots/--families的诊断子集仍不能qualified=true。

### 本次实际复验

```sh
./.venv/bin/python -m pytest -q tests/test_inspection_flight.py
```

结果：**6 passed in 7.99s**。内容为真实两次积分后hook、契约限值实际生效、crosswind crab、错误target拒绝、cleanup保留物理/轨迹、原子manifest中断保留。未重跑长grid，也未据此宣称100个slot通过。

非阻断的解释边界：

- heading_changes中的tracking_rmse_m/tracking_max_m为未应用的位置指标占位0；其实际成功指标是独立course误差。其他family的settled_max_course_error_rad=0是未应用该指标的默认值，trace中的course_error_rad相对requested_heading（这些family保持0°），不能解释成圆周/航线航向误差为零。完整资格报告只能引用对应family实际使用的指标；后续如展示这些字段，宜标not_applicable。
- 合并RMSE按slot内机体/时间汇总；每个采样的最大误差仍被检查。不能从这个汇总推导“每机RMSE都≤15m”。
- survey几何控制资格不等同于ROI覆盖、感知或真实搜索任务成功；exit控制dwell不替代G2/G3任务终态。恒定0/2m/s风探针不替代阵风、湍流或估风误差资格。
- 当前独立上下文审查和工程fixtures不替代领域人工复核，human_review_pending。完整G1结果尚未由本审查见证；G2–G4、正式方法实验与科学有效性也未因此通过。

## 最终 Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- No remaining run-blocking findings for the reviewed frozen v3 G1 grid.
- 开始100-slot不等于G1已通过；按实际结果和同一契约另行判定。
- 保留旧诊断及R1错误产物反例，不用新SHA回填历史；领域人工复核仍待完成。


## G1-full.v1完整结果独立审查（追加）

**本次100-slot控制开发资格grid按冻结gate-contract.v3通过，manifest的qualified=true与独立核对结果一致。** 本节更新前文“尚未运行/尚未见证完整G1结果”的运行前状态；不改写旧诊断及旧错误产物证据。未重跑grid。

审查对象：qualification/g1-full.v1/manifest.json，SHA256为 b7820ec90cf487122d92896bb336fd7fd086410eb9472261b3b815b926ca14ca。审查前后该SHA一致。

### 全量完整性、时长与失败分母

- grid恰好是5个family与root_ids=0…19的笛卡尔积，100个唯一slot、100个唯一结果文件；每family20个，无缺失、重复或被删除的root。
- 目录恰有manifest及其引用的100个JSON，共101个文件，没有未列入manifest的其他case文件或残余临时文件。100个结果文件的实际SHA全部匹配各slot.sha256。
- 全部slot.status=completed、slot.passed=true，与结果内task/root/passed一致；100份operational_error均为null，所有JSON数值均有限。未发现running、unstarted、operational_failure或不通过slot被排除在分母外。
- 请求/实际模拟时长逐slot相同：straight60秒、heading_changes120秒、loiter180秒、survey180秒、sortie_exit240秒。累计15600模拟秒，即4小时20分。
- 每份记录均满足physics_audits=physics_steps=requested_seconds×240；总计3744000个world积分步及相同次数的审计。它们是同world物理积分审计计数，不是3744000个独立科学样本。
- 所有10Hz轨迹时间格与条数也完整，合计156000个时间点；每点位置/航迹角数组的机体数匹配。直飞偶数root的10个slot为单机，其余90个slot为四机，符合冻结采样规则。
- 每family成功率独立重算为20/20=1.0，分母始终为20，均高于0.95。全部操作正常完成，加上full_grid及三类不变检查，qualified条件成立；没有把只通过的slot重新归一。

### 冻结输入与当前字节核对

manifest内的三份源码快照与本报告上一节列出的冻结SHA及当前文件完全匹配，contract_sha256与gate-contract.v3原文件字节匹配，嵌入contract对象也完全一致。61项PyFlyt源码/机型资产及PyBullet二进制依赖SHA逐项重新计算，全部匹配；PyFlyt0.29.0、pybullet3.2.7、numpy1.26.4、numba0.68.1与当前安装元数据一致。

sources_unchanged、contract_unchanged、dependencies_unchanged均为true，且本审查对其对应当前字节进行了独立复核。此处不把“现在匹配”单独当成运行期间证明；运行前已审阅并绑定的起始捕获、导入检查、结束检查和本次manifest共同构成执行来源证据。

### 逐family验收重算

距离指标不是只相信case自报：从全部保存位置与冻结root采样重新生成radius/half_length及固定航线中心，独立计算直线横向距离、圆周径向距离和capsule边界距离；逐点与保存tracking_error_m一致，再按冻结settle/phase窗口重算合并RMSE及最大值，100份对应汇总全部匹配。航向family按冻结schedule及其phase窗口核对course误差汇总，不读取被测last_targets作参考。

下表中的RMSE是“20个slot中最大的slot内合并RMSE”，最大误差是所纳入稳定窗口的全族最大采样误差；未应用的指标明确列为—。

| family | 完成/通过 | 每slot秒数 | 最大slot RMSE m | 最大跟踪误差 m | 最大稳定course误差° | 最小退出连续dwell s |
| --- | --- | --- | --- | --- | --- | --- |
| straight | 20/20 | 60 | 9.271 | 12.473 | 0.296 | — |
| heading_changes | 20/20 | 120 | — | — | 0.306 | — |
| loiter | 20/20 | 180 | 4.457 | 10.526 | — | — |
| survey | 20/20 | 180 | 4.485 | 14.321 | — | — |
| sortie_exit | 20/20 | 240 | 4.285 | 22.398 | — | 94.617 |

这些值满足各自15m RMSE、40m最大误差、10°稳定course、20秒退出dwell门槛。sortie_exit的20份dwell数组均含4架飞机，逐机都满足门槛。

另从保存位置的相邻0.1秒差分独立求ground course，核对冻结schedule后的稳定窗口：straight全族最大约0.29562°，heading_changes约0.30568°，均远低于10°。这些是0.1秒平均位移航迹角的交叉检查，不冒充瞬时速度的精确重建。所有sortie_exit末20秒的10Hz样本均在退出圆半径误差和切向course门槛内，最大约7.635m/8.207°，与报告的长dwell一致。

### 硬安全记录核对

全部100份safety.violations为空；逐slot的高度、空气相对速度、滚转及适用的最小机间距摘要均在冻结门槛内。合计范围：

- AGL：64.904–96.177m，当前场景为平坦z=0地面；
- airspeed：19.124–20.868m/s，使用ground velocity−冻结恒定wind；
- 最大绝对bank：30.089°；
- 多机最小三维间距：192.250m。

审查确认的是冻结逐240Hz审计代码路径、此前真实hook拒绝检查、完整运行计数和保存的安全摘要/事件记录一致。本次没有重放物理，也没有把10Hz轨迹当成逐240Hz原始状态记录重新判定碰撞或每次积分的安全条件。

### 结论边界与后续版本

这100个slot是20个控制参数roots跨5个family的开发资格运行，重复使用相同root参数族；不是100个独立道路/地形/灾情场景roots，不形成主研究方法比较样本，不据此推出未来场景零失败或总体安全率。

通过范围只覆盖冻结v3所述控制family、初态/几何采样、平地和恒定0/2m/s风，以及记录中的1/4机配置。新增观察盘旋模式或修改control/flight/runner、物理依赖/契约后，应建立新的资格版本并按其影响复验，不回填这100个slot。此次结果不证明G2/G3/G4、感知/报告链、主研究任务可解性或科学结论。

领域人工复核仍为human_review_pending，本报告不能替代它。

## 全量结果最终 Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- No result-integrity or frozen-v3 acceptance discrepancies found across the 100 reviewed slots.
- G1-full.v1工程控制grid通过；控制参数roots与主研究场景roots、人类领域复核及其他gates保持分开。



