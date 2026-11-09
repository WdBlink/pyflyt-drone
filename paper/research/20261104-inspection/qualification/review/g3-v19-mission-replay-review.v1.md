# v19 B0 真实1×业务与因果重放独立复核 v1

审查日期：2026-11-05。审查者 `/root/mission_review` 为同模型家族独立上下文，不是人类领域专家或跨模型审查。工作树为 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。本次只新增本报告和 `/tmp/g3-v19-mission-review/` 下的独立重计脚本/小结果，没有修改生产、旧结果或旧报告，没有运行新 physics、replay、grid、测试或API。原600 s运行结束后才恢复CPU检查，最终replay完整落盘后才读取判定。

## 结论

**当前v19 B0的完整600 s真实1×记录支持G3参考业务链，独立分数重计和归档源码绑定的因果重放均通过。** 本次没有发现必须先修的业务G2/G3代码阻断。已完成的记录证明真实采样、报告、observe、优先级更新、断链交付、退休停采和物理退出可以在同一完整episode中发生；这不是所有G2场景穷尽通过，也不是G4或formal放行。

独立从原始覆盖包、隐藏实体、交付观测和报告审计重建账本及全部180个动态格，得到 C=.9372128637059725、S=1、D=.4、Q=Y=.7790709545686575、H=E=1、SR=0，与保存结果逐项完全一致。D未达到.5，不能因Y较高宣布完整任务成功。B0两个observe最终failed、四个cancelled；合法交付的测量仍按评分规则逐格计分，作业failed不等于所有历史质量清零。

最终重放116/116实际RPC、21字段组精确一致，无response discrepancy，且重新计算纯planner proposal并核对内部请求/响应。它是原执行历史的一次加速一致性审计，不是新策略样本、第二个1×或新的模型/规划器时延样本。未保存的full240Hz状态不能补造；raw240Hz误差trace没有列入重放比较keys。

时钟、raw tracking、传感器阈值和G1组合复用由control_review另审。本报告核对qualification.v1的输入与分析源码绑定及10项native bool真值，未用其passed代替下述独立业务检查。G3总体结论还须与该独立控制/时钟审查合并。human_review_pending，G4仍待。

## 输入、源码与完成状态

原run开始于2026-11-05T13:48:00.137480+00:00，policy=B0、layout=reference、seed=0、events=true、realtime=true。run/world均status=completed，世界simulated_seconds=600，physics_steps_executed=144000，原始wall_seconds=600.0084076250205；记录max lag=.03148783339032235 s。这里仅明确原运行身份，未从replay速度重新推导1×。

独立检查run中14个源码字符串的SHA、source_hashes、归档 `qualification/source/executor-v19-reference-v2/` 的实际字节和当前文件，全部一致。归档binding的run/world SHA与实际输入一致，file_count=14；envs/eval包脚手架为空。run的sources_unchanged/dependencies_unchanged均true，qualification的run/world SHA及四个analysis_sources_sha256均与对应v1归档匹配。61项依赖的完整复核留给control_review，本报告不将14源归档称为整个Python环境锁。

归档中的mission仍为退休guard修复版，protocol/scoring未变。所有以下结论绑定本报告末尾的原SHA；后续策略、控制或任务源变化不能回填到本次结果。

审查末尾发现当前replayer/checker正被其他任务升级。父任务已经在升级前把旧checker、分析工具及93dd版replayer共五源完整保存于 `qualification/source/reference-analysis-v1/`，其binding分别绑定本报告读取的v1 qualification/replay receipt；本次再次逐字节核对旧归档。**本报告冻结旧21字段组replay.v1和10checks qualification.v1**，不回填随后新22组replay.v2/11checks资格的结论；后者由另项审查覆盖。14个原真实run/world源码未变。

## 不调用生产scorer的完整重算

独立脚本只导入标准库。它从847条report audit按接收顺序重建10个报告ID：首次关联按20m门槛、距离/ID稳定排序并保留占用，随后不换实体；证据逐条核对delivered arrival、candidate、同测量时刻sensor身份、quality和原值。动态测量按稳定新证据ID封存，静态按最后合法接收顺序选择。该记录没有withdraw、非法provenance或动态幂等重复。

| 指标 | 独立重算与完整分母 |
| --- | --- |
| C | 3743个实际到达coverage packet，1224个唯一合法cell / 全部1306个required cell = .9372128637059725；所有包均满足0≤measured_at≤arrived_at≤600。未按发现数缩分母。 |
| S | 六个required静态实体全部保留，权重3+3+1+1+1+1=10，分子10，S=1；static_recall=6/6=1。 |
| D | 三个动态实体每个60格，共180格；完整权重100+120+120=340，分子70+14+52=136，D=.4。 |
| Q/Y | Q=(C+S+D)/3=.7790709545686575；H=E=1，故Y=Q。 |
| H | 从记录的全物理步safety输入重新聚合，violations={}，H=1。此处不宣称仅由误差/Z列独立重建bank/速度/碰撞。 |
| E | 四机记录的实际exit dwell均≥20 s；剩余航程审计值均≥1000 m，E=1。dwell=[78.24166666667767,78.95833333334369,84.20833333333891,203.83333333323012] s，remaining=[7803.321056971601,7788.045257803167,7787.880762105517,7802.624841826171] m。 |
| SR | H/E通过且C≥.8、S≥.6，但D=.4<.5，所以SR=0。 |

E重算使用保存的240Hz安全dwell与航程累计审计输入。world没有full240Hz位移状态，不能从10Hz位置精确重新积分实际航程；剩余航程数值及distance hook的源码/记录绑定需结合control_review。它是18000m航程预算与1000m退出余量，不是电池模型。

15个隐藏实体包括6个required静态、3个required动态和6个无rule的normal干扰物。干扰物不计任务分母，但可以产生假候选、消耗报告名额。所有2779条交付观测与sensor audit的evidence ID、原测量时刻、估计、类别、quality逐条一致；所有非background sensor truth均由对应实体在**原measured_at**的轨迹独立复算一致，没有用arrival或格端点truth替代。

六个静态最终所选证据为truth-000=`obs-2-180-0`、001=`obs-3-40-0`、002=`obs-3-158-0`、003=`obs-3-230-0`、004=`obs-2-9-0`、005=`obs-3-8-0`，质量均1；对应报告每个仅一个合法版本。完整静态实体/质量、每个动态entity的所有bins、selected ID、权重、质量与zero_reason均与world保存值精确一致，不只是四舍五入总分相同。

| 动态实体/需求窗口 | delivered分子/分母 | observed | 60格零值原因 |
| --- | --- | --- | --- |
| truth-006，120..300 s | 70/100=.7 | .7 | 17格无合法及时样本，1格quality不足，1格最新类别错误；41格质量1。 |
| truth-007，180..360 s | 14/120=.11666666666666667 | 同左 | 52格无合法及时样本，1格quality不足；7格质量1。 |
| truth-008，240..420 s | 52/120=.43333333333333335 | 同左 | 34格无合法及时样本；26格质量1。 |

每格先筛因果/交付时间，再选latest（同刻按ID），最后判identity/class/quality/localization。没有回退到旧正确样本。observed也保留最终600s审计上传要求；本次三个observed值恰与delivered相同，不代表两轨道一般等价。

### 180s实际调权跨格

本次补上此前单目标206s出现fixture未能实际覆盖的180s跨格：truth-006在120s出现，前20格权重1、后40格权重2，分母100。第20格[177,180]选择实际180s采集的 `obs-0-360-0`，权重仍1；第21格(180,183]选择 `obs-0-365-0`，权重2。180s样本没有重复归入下一格。调权前分子12、调权后分子58，共70。

180s priority_update事件实际发布为task_card_version=2；truth-007于180出现，全60格权重2；truth-008于240出现，同样全60格权重2。静态最终权重在600时为两项3、其余1。没有将事件后的新优先级追溯应用到此前格起点。

## 实际工具链、作业与本地缺格

116条top-level-v2 RPC由33 survey、33 cancel、31 inspect、10 submit/revise report、6 observe、3显式safe exit组成，全被接受；第四个exit来自360s外生事件。全部1201份public snapshot和116调用是原运行计数，不是额外策略样本。

74个作业、211个状态转移重新检查，均遵循queued→running/terminal的合法状态机，无同一机并存active job，最终全部terminal。33 survey均由实际cancel结束，不能称33次完整survey任务成功；它们仍可在实际轨迹上贡献曝光/coverage。31 inspect中29 succeeded、2 failed；六个observe中2 failed、4 cancelled；四个exit全部succeeded。

对29个成功inspect，独立从本机同candidate、创建后、终态前、quality≥.8、同object_type的真实观测重建qualified views，全部确有距首view≥20m的第二view，第二view测量时刻等于成功时刻；所有31个inspect的qualified样本数与job.params.measurements一致。不是把ACK或planned_service_until当作实际成功。

对六个observe同样重数本地qualified测量，数量分别220、113、107、190、135、4，与记录完全一致。它们的本地服务窗从service_started_at开始，与隐藏任务从appearance开始的计分窗不同；本地完成还不读取truth身份/类别正确性，也不等待6s上传grace。

| observe job | 实际终态 | 独立本地检查 |
| --- | --- | --- |
| job-0040，机0，start144.5/duration180 | 324.5 failed，local observation gaps retained | 220个qualified样本，60个服务格中缺15格。仍支持truth-006在原120..300窗内的合法质量.7。 |
| job-0067，机2，start287.5/duration180 | 467.5 failed，同上 | 107个qualified样本，60个服务格中缺39格；其报告首次关联为null，历史版本不会变成另一已占用truth的分数。 |
| job-0063，机3 | 360s外生退休cancelled | 113个qualified样本，服务未到期，不将之后未执行的格计作“已完成”。 |
| job-0068/0069/0071 | 470.0083333333333s显式safe exit preempt后cancelled | 样本190/135/4；取消不删除原合法报告版本，不把有限2s末次observe的4样本宣称完成原180s任务。 |

报告名额实际用满10格。六个静态各一格，三个动态true关联各一格；额外 `report-candidate-2-00015` 在truth-008已经被 `report-candidate-3-00015` 占用后首次关联为null，占第10格。它保留163个版本，既不退款，也不后续重绑。其他动态版本数为truth-006报告421、007报告140、008报告117，共同构成847个版本。这是B0的公开候选重复/分配限制，不能把额外测量误算成额外目标或声称高性能追踪已解决。

## 210–240s补传与曝光路由

机1在210≤t<240离线。完整记录有7条机1 survey观测在240到达，原measured_at为237、237.5、238、238.5、239（两条）、239.5；其中前6条超过基础.5s运输延迟，239.5→240属于正常基础延迟。所有7条quality<.8，不能因为补传完成就宣布它们有动态计分资格。ID和原值与曝光sensor audit逐项相同。

coverage有61包在240到达，原测量209.5..239.5，其中60包严格迟于基础到达时间。209.5包的ready_at=210，恰在断链开始时被缓冲。由完整arrival记录反推，239时尚未到达的观测6条、coverage60包，其中已经ready但断链等待的分别4条、59包。这里是从最终曝光/arrival重建的pending，不是冒称world保存了239时完整原pending快照。

**该1×没有断链期间机1的observe自动report route样本。** 七条均survey；它证明数据补传，不证明“已批准observe在该断链区间继续自动上传”的实际路径。对应强正例仍是此前store-forward.v2的34包fixture及其冻结独立报告，不能将其说成本次run又发生34包。

全run的847 report audit中，10条来自显式submit RPC，837条来自实际observe自动上传。逐条匹配曝光时的本机candidate、observe job区间、批准的report_id及原measured_at/estimate/class，均无错误；report接收t均等于该证据实际arrival。world没有逐包序列化原pending route对象，所以该关联由生产曝光规则、实际job区间及最终审计交叉核对，并由完整重放一致性支持。

本次真实覆盖了两种终态后原route交付：`obs-2-935-0`于467.5采集、job-0067同刻failed，468.0仍交到原report；`obs-2-940-0`于470.0采集，job-0071在470.0083333333333被safe exit取消，470.5仍交原report。它们不依赖交付时active job，也没有改测量时间。后者是实际cancel后路由延续，但不是交接给不同report/receiver后的路由冲突场景。

## 360s退休停采与物理退出

机3的observe job-0063在360事件被cancelled，原版本35后创建version36、job-0070的exogenous_capacity_exit。361.8666666666667实际采用退出程序，416.1666666666667因实际home-circle20s dwell成功；其余机退出成功时刻分别541.7583333333333、541.0416666666666、535.7916666666666。记录没有把360s ACK当成物理退出完成。

独立重计机3 t/measured_at≥360：target sensor_audit=0、coverage_evidence=0、sensor_opportunities=0、background_trials=0。新v19停止任务容量贡献的边界在完整1×中成立，未复现旧v18 F1。

本run没有机3退役前采集、退役后到达的Observation，不补造该正例；有一份旧coverage packet measured_at=359.5、arrived_at=360，原cells=`cell-30-90/cell-50-70/cell-50-90/cell-50-110`保留。旧Observation的359.5→360实证仍由retirement.v3物理fixture支持。本次没有“退休后全删旧包”的迹象。

## 精确重放审查

最终文件 `reference/executor-realtime.v2.replay.v1.json` 已完整出现且父任务确认进程退出。本次只读取并验证，没有再跑replay。它绑定实际run/world SHA、归档source_root和replayer SHA，simulated_seconds=600，tool_calls_replayed=tool_calls_expected=116，response_discrepancies=[]，passed=true。

审阅 `eval/replay_inspection_reference.py:23–105,109–162` 及 `inspection_artifacts.py`：先校验完整不可变输入、world hash、归档源码字节和实际import的Mission位置；逐原tick发送RPC；对内部plan_request完整输入/provenance逐项比对，重新调用归档纯plan计算proposal的program hash/reason/geometry；按原received_at释放结果并逐项比对validation audit。关闭时要求内部request/response全消费。不是仅重放外部工具日志后比较一个总分。

原内部事件为580 requests、580 responses、372 program commits、55 invalidations、5 capture rejections、48 physical seams。request/response serial集合精确相同；全部request OwnState时刻等于now且commit_at在now之后。536个含track请求的support全部来自同机本地证据，measured_at≤available_at≤request.now；未发现未来可用时间。实际跨机track seed请求为0，不能用本run证明跨机未送达信息边界已经被实际触发；其负例仍依靠独立fixture。

21个精确匹配字段组为：score、safety、trace、coverage_evidence、required_cells、sensor_audit、delivered_observations、jobs、job_transitions、tool_calls、reports、hidden_entities、public_events、execution_audit、development_envelope_id、execution_infrastructure_errors、execution_tracking_240hz、sensor_opportunities、background_trials、candidate_gate_multi_detection_frames、tool_journal_version。全部native bool true，keys与当前replayer明确的比较集合一致。

边界必须保留：`execution_tracking_trace`不在keys中；只比较tracking聚合，不能称raw240Hz误差字节或full240Hz物理状态精确重放。原始raw另由control_review重计。wall_seconds、clock lag、调度墙钟并非重放匹配字段；ReplayExecutor关闭新wall timeout并注入原planner_wall_seconds/received_at，所以98.6176985000493s的加速重放既没有再证明1×，也没有产生新的planner延迟或模型调用统计。

## G2/G3支持范围与未执行点

| 范围 | 本次结论 |
| --- | --- |
| 完整隐藏分母、固定关联、合法版本、同测量truth、dynamic筛选/最新/正确性 | 通过独立原始业务重算；静态6实体/权重10、动态180格/权重340均精确保留。 |
| 180s实际priority crossing | 本次完整1×新增真实覆盖，含180s端点只归一次及下一格加权。 |
| causal coverage/补传/终态后原route | 当前记录范围通过；实际断链survey与coverage补传、一次failed后和一次cancel后observe包交付成立。 |
| 外生退休停止新任务采样、物理退出及旧包 | 当前完整1×范围通过；四通道零、旧coverage保留；旧Observation正例由独立v3 fixture而非本run提供。 |
| ACK/作业失败/版本 | 实际合法状态机与全部终态通过。116 RPC全accepted，所以此run没有实际stale/offline拒绝反例；既有有界测试仍是这些负例来源。 |
| handoff/跨receiver seed/换report冲突 | 本run handoff=0，跨机track support=0；不能泛称实际完成所有交接/跨区分配场景。此前短真实handoff与mock正反例各保留其边界。 |
| G3 B0参考业务可解链 | Y≥.2、static recall≥.5，采样→确认→报告→observe→exit实际发生且基础设施错误为空；参考业务组件通过。时钟、sensor阈值及控制总体需合并control_review。 |
| 强动态策略/完整成功/科学结论 | D=.4、SR=0；B0两个observe本地缺格，报告重复耗满名额，不能称稳定180s高质量跟踪、B_classic*比较或正式统计。 |
| G4、模型适配、真实模型时延、人类领域审阅 | 本报告未授予资格；replay不增加样本量。 |

没有以“某场景未在这一个run发生”自动新增未规定的阻断条件，也没有以已有代码/10checks/replay通过消除实际未覆盖范围。是否满足完整门级发布条件，应按v19契约合并独立控制审查、已有G2正反fixtures与后续G4冻结材料。

## 版本与检查证据

下列qualification/planning路径相对 `paper/research/20261104-inspection/`，其余相对仓库。HEAD仍为 `270c6f0946b3be59527b2b77982f75a70da7ef77`；未提交工作树内容由实际SHA绑定。

| 文件 | SHA-256 |
| --- | --- |
| qualification/reference/executor-realtime.v2/run.json | `122e9f47ae63080ef0dac318a669e85b7e60a616d0e2f4ce3510fa4626c5f871` |
| qualification/reference/executor-realtime.v2/world.json | `8d60ea4bff2b4624835b4d9bd260ff72d644e76c85a873c6ad4b387f554ad3c6` |
| qualification/reference/executor-realtime.v2.replay.v1.json | `badd7268dd6921542168414e7bfa3901777fd7fb1289ac2a6e404dcd690b0e53` |
| qualification/reference/executor-realtime.v2.qualification.v1.json | `80785057b4e5c35877d5af40aa8b3533d6b1d69a0bafe1df280c0dd1caf58f67` |
| qualification/source/executor-v19-reference-v2/binding.json | `65a992d83c78889011645b16c9786a7c2ed4307a12278ac0885e4ecd69f30bdd` |
| planning/gate-contract.v19.json | `c0af9d0c6f49f2f879313d7bea767a917f8fa6a80e96701009d63ef83dab9b74` |
| envs/inspection/__init__.py | `cb0d38bbca6b531b261ca5a5f2f79664e352c9ddb6c19acc1777784c3115107a` |
| envs/inspection/control.py | `c676d237efe1b6227d238b4e5948c54a5725626e9d3f69e74ed9c1629a235270` |
| envs/inspection/execution.py | `dc301939ea1d55f08f01c69af5561c75f762ac45bdd0d144d677514387425ea4` |
| envs/inspection/flight.py | `ef9be915c92c207a3e8f43b9ddc1907010beea59be6d96136330e82a5d67cbdb` |
| envs/inspection/guidance.py | `9ab9c5290cbce60ad7f9fead635318f41561dedd1cfc2f0984b1cd45385bc0b9` |
| envs/inspection/mission.py | `715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7` |
| envs/inspection/paths.py | `dbd31dc94a7cdcbcb17b8d9f1847fa88cf4cd12e4609ec8ba57e2233a367855b` |
| envs/inspection/planning.py | `b87fcdedf3aa26a46560056376fb69f9ff969270149c9cae429b4a9e294df6ab` |
| envs/inspection/protocol.py | `c32bdabb67bf8493ff5125d0af4fc289d1d0210db762016a9c56a96dd2f174a4` |
| envs/inspection/reference.py | `85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3` |
| envs/inspection/scenario.py | `c4f56e2783bc21c087c3abdcbfae9a7472feae2bbbe1e007d86bb39b28c40af2` |
| envs/inspection/scoring.py | `bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a` |
| envs/__init__.py | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| eval/run_inspection_reference.py | `90509286b5e3d7bb849b2307f18336943bd1f2de66a0bf1b701099bb41b3f8b9` |
| qualification/source/reference-analysis-v1/eval/replay_inspection_reference.py | `93dd2d2b55148818edd6109d65d6c7054021b1647cc577466e92282bcc5ba69c` |
| qualification/source/reference-analysis-v1/eval/inspection_artifacts.py | `f2a95932dd715383b4e5007d5128ec57f453a26243a15a39d4eb8a25ab49064e` |

独立执行的三个标准库脚本为 `/tmp/g3-v19-mission-review/recount.py`（SHA `c73a2575ca3853c8b588f63aeefb716921389e42e47777f523431e7df62fb14a`）、`lifecycle.py`（`252d5d2793ad3ff5e6464e0a5fc2eca94720b702281f3c490b73730b4c0118a5`）、`provenance.py`（`d6f5a00e45164abd8350c16ccd661fcebc73bf986cc1c9e931a7fc15293fd389`），依次执行均退出0，最终聚合断言通过。重计结果 `recount.json` SHA=`f1e055e10c3c0d711e12f9a751d3bc8dd92c382377f975953068048613ae5f81`，包含全部重建版本/格选择、作业样本和原始输入绑定。末轮检查显式使用旧analysis-v1归档，避免并行升级被误作旧replay来源。未为本次只读审查重复运行协议套件或长物理任务。
