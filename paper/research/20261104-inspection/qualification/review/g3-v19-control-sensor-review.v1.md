# v19真实1×参考运行：G3物理/时钟/传感与检查入口审查 v1

日期：2026-11-05。同模型家族独立上下文工程Verify，不是跨模型或领域人工认证。仅写本报告，未改production、原run/world、旧资格JSON或其他报告；未重跑长physics/1×。完整业务评分、事件与重放由mission_review另审，本报告不重复其全部范围。

## 结论

**原始600秒记录在本审查的物理、时钟与传感点门上有一致的实际证据；但初版严格检查器有3项P2漏检，不能仅据其10个true宣称完整资格。** 144,000行误差独立重计、normal inspect点统计、已保存机会/随机draw及实际退出/range交叉核对均与原记录相符。原数据没有因下述/tmp反例被修改，也不需要因此重跑原600秒。

本报告首先冻结检查器SHA `09b19a6a…cbc21`的缺陷与原数据复算；主Agent已获通知可修复检查器/另立分析v2，后续修复不在下列初版结论中冒称已验证。当前仅确认所审G3组件的真实证据，完整G3判定仍须检查入口修复与业务/重放审查汇合。G4、正式性能、真实vision及人类认证均未授予。

## 身份与G1组件边界

路径相对paper/research/20261104-inspection；检查源码相对repo根。

| 对象 | SHA256 |
| --- | --- |
| qualification/reference/executor-realtime.v2/run.json | 122e9f47ae63080ef0dac318a669e85b7e60a616d0e2f4ce3510fa4626c5f871 |
| qualification/reference/executor-realtime.v2/world.json | 8d60ea4bff2b4624835b4d9bd260ff72d644e76c85a873c6ad4b387f554ad3c6 |
| qualification/reference/executor-realtime.v2.qualification.v1.json | 80785057b4e5c35877d5af40aa8b3533d6b1d69a0bafe1df280c0dd1caf58f67 |
| planning/gate-contract.v19.json | c0af9d0c6f49f2f879313d7bea767a917f8fa6a80e96701009d63ef83dab9b74 |
| qualification/source/executor-v19-reference-v2/binding.json | 65a992d83c78889011645b16c9786a7c2ed4307a12278ac0885e4ecd69f30bdd |
| eval/qualify_inspection_reference.py（初版） | 09b19a6a130810f29776faefb93c817f113d149ef7b7a38c3f0727e1d66cbc21 |
| eval/analyze_inspection_qualification.py | 04b1a63d3fb7f4f90451e198be146860b5f67f23e2a1d69afb0e5602062e2344 |
| eval/inspection_tracking.py | 19151659b4a969c51a6cbcb94b4f10aa1cb4489bebec6bd50f28b639a07884ba |
| eval/inspection_artifacts.py | f2a95932dd715383b4e5007d5128ec57f453a26243a15a39d4eb8a25ab49064e |

run→world SHA、资格JSON→run/world SHA及归档binding均一致。14份运行source逐一核对embedded hash、归档字节与读取时当前文件，全部匹配；61个实际PyFlyt源码/模型资产及PyBullet二进制重hash匹配，package_versions与本地元数据相同：PyFlyt0.29.0、pybullet3.2.7、numpy1.26.4、numba0.68.1。run记录Python3.11.15及四项数值线程环境为1；这些是保存范围，不是完整OS/所有数值二进制环境锁定。

运行源中的control/flight/planning/execution/paths与已审G1源相同，mission为v19 guard版 `715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7`，B0 reference.py为`85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3`，runner为`90509286b5e3d7bb849b2307f18336943bd1f2de66a0bf1b701099bb41b3f8b9`。基础100等价复用与v18 events=False40→v19 guard的限定组合证明已由g1-v18-full-review.v1.md及g1-v19-composition-review.v1.md完成；本次不是重做140个控制样本，也不把它们误称正式科学场景。当前运行是新events=True、实际异步worker的600秒证据。

## F1 [P2，初版未关闭] 依赖闭包非空不等于完整

位置：初版qualify_inspection_reference.py:92–99。仅检查dependency_hashes非空、unchanged=true及列出的文件逐项匹配，未要求完整预期集合。/tmp副本把原61项删到仅第一个真实PyFlyt文件，保留其正确hash，真实qualify仍passed=true、errors=[]。

必要修复：从实际支持的PyFlyt源码/资产及PyBullet范围构造完整expected key集合并精确比较，然后核对hash；不能用任意非空子集代表闭包。原run实际61项已独立全核一致，此缺陷是入口误接收能力，不是原数据依赖缺失。

## F2 [P2，初版未关闭] 发出证据对账无法发现未emit分母被删除

位置：:24–81、109–111。emitted ID一一对账能抓孤儿、重复、错frame及missed却emit，但没有独立的完整frame/机会锚。/tmp副本删去所有emitted_evidence_id=None的sensor_opportunities和background_trials，令raw misses/未触发背景分母消失，仍保留所有发出measurement，真实qualify仍passed=true。保存机会的miss比例会被改变，却不会触发拒绝。

必要修复：完整采集分母须由独立实际重放/完整frame账本验证，而非仅非空机会表及emit集合。外部receipt若用于该证明，应绑定实际run/world SHA、重放源身份、明确的完整字段组和成功比较结果；不得接受仅自填passed或删字段后重新算出的内部hash。本报告未独立穷举每个未检测frame的完整camera/ray几何，因此只确认**已保存机会表**的统计、RNG与measurement一致，不能据此宣称其全集已由初版checker证明。

## F3 [P2，初版未关闭] 工具名/作业种类存在不构成因果业务链

位置：:112–120。accepted工具名集合、任意inspect succeeded和任意observe service_started_at未连接到同candidate/report/evidence。/tmp副本把全部observe jobs的candidate_id、report_id改为从未inspect/报告的句柄，保留其他元数据，真实qualify仍passed=true。

必要修复：按candidate、机体、measurement及时间连接inspect succeeded→accepted report的同candidate/evidence→observe同report/candidate→其服务期实际新local sample；退出和coverage仍需实际证据。仅名字集合齐全不能代表整条链发生。实际原记录的完整业务链由mission_review另核，本报告没有从初版bool推定其成立。

三反例均仅写/tmp副本，并为副本重新绑定world SHA，避免只测试明显破坏文件hash的浅层故障。故它们显示“结构上自洽但证据不完整/不相连”可被误接收。冻结证据目录`/tmp/g3-checker-v1-control-review-evidence/`内有旧4份检查源码、真实qualify反例脚本、结果及本次独立原数据复算；binding.json列全部SHA。fault脚本SHA `56c76283a876e106ad88fd21997d50519412d5b55b3c6ad4a1f866e9cc161f17`，结果SHA `4a40c784c99a3bdcc5932c2ae6ad79ea79e466c89360725a37e76339beab403a`。未改原数据或“修复”其值。

## 原记录的物理审计独立复算

run/world均completed，请求/实际600秒，world.physics_steps_executed=flight.physics_steps=physics_audits=tracking.physics_samples=144,000。解压trace并核对little-endian float64、17列、字节数、raw SHA `c67c9e47a1b95aac6631facadaf645caed56ef43a017f4ac29c0ddafa8a2d3e2`、有限值和严格1/240秒格。显式xy_tube=10，Z interval全为94/98；每world步四机任一越界的独立重计与summary/资格JSON吻合：

| 指标 | 独立结果 |
| --- | --- |
| 最大同刻欧氏XY | 6.986841891m |
| 最大steady高度误差 | 1.053814458m |
| XY/Z越界world步 | 0/0 |
| seam失败 | 0 |
| 实际AGL范围 | 94.726616…97.097255m |
| 实际airspeed范围 | 16.015696…20.0m/s |
| 最大bank | 约30.62°，低于45° |
| 最小实际3D机距 | 38.622535m |

6000条10Hz位置与240Hz Z对齐、时间格完整，位置边界及10Hz机间距离与hard summary一致；无硬违例或基础设施错误。峰值见证actual/nominal距离及对应240Hz误差行吻合。

核对580个四机OwnState/range请求、372个commit、48个seam：snapshot有限且时间一致，airspeed由world_velocity−wind重算吻合，capture各域及seam.capture_valid均符合原10m、Z、角度/速度/率/积分器限制。55个proposal_invalidated和5个capture rejection保留，未算成功commit。这里使用独立复核算式，没有调用生产recount作“独立”结果。

四机最终实际exit dwell为78.241667、78.958333、84.208333、203.833333秒；剩余range为7803.321057、7788.045258、7787.880762、7802.624842m，均满足20秒/1000m。最后20秒位置独立检查R80 home circle，最大半径误差.105152m、差分course切向误差.614922°。

距离累计绑定冻结mission.py:69–76：每240Hz实际getBasePosition后累加3D位移，grade:436–441用同一distance_m导出score.remaining_range_m。请求range单调、10Hz折线行程不超过已记消耗，是旁证；world没有全240Hz位移，**不能从10Hz精确重构里程或称range已由全state重放验证**。hard速度/roll/接触也只有逐步源码采集链和摘要，原error trace不是完整physical-state replay。

## 实际1×时钟与5秒阻塞证据

冻结runner:29–35先用独立warmup world热身并关闭，再创建新seeded任务world和单次monotonic epoch；:40–79不重设任务epoch，worker独立步进，逐control tick记录`wall−epoch−sim_time`。记录真实wall600.008408秒，最大lag .031487833s、p95 .005689916s、>.1秒tick数0，最大见证在sim23.241667秒，与最大摘要一致。

progress.jsonl有完整600个1Hz递增进度点，sim恰1…600、wall单调，独立算出的这些点最大wall−sim为.026854792s<.1。**完整72,000个control-tick lag数组未保存**，故无法独立重新求全量max/p95；本报告核的是冻结采集实现、最大见证/摘要及全部已保存1Hz旁证，不把这些称为完整墙钟重放。

coordinator在sim100秒阻塞，实际sleep wall5.009128041秒，shared worker进度到105.008333333秒，即sim推进5.008333333秒，差−.000794708秒，远小于.1门槛；由before/after实际3D位置独立重算位移85.106338793m。它支持协调器等待期间物理world继续推进，不能声称sleep测得了真实模型推理延迟。此次使用实际异步planner，与G1 controlled nexttick证据不同。

## normal inspect点统计与保存机会分母

从原sensor_audit重新按mode及机2在[300,345)的degraded区间分组，不用quality_clock_calibration的汇总值计算：

| 组 | 实际measurement数 | 二维向量RMSE(m) | class正确率 |
| --- | --- | --- | --- |
| inspect_normal | 877 | 1.430834690 | 868/877=.989737742 |
| inspect_degraded | 45 | 3.740484876 | 24/45=.533333333 |
| survey_normal | 1789 | 5.705169453 | .774734489 |
| survey_degraded | 44 | 14.566025195 | .727272727 |

normal inspect满足合同RMSE≤3m、class≥.9。normal inspect中31个真实normal distractor有1个错类；normal survey中527个normal distractor有144个非normal假候选。另有24个真正background发出measurement，不能将其与normal实体混成同一分母，也不能把877个measurement当独立场景或总体可靠性保证。

| 保存机会mode | visible机会 | raw miss | 发出measurement | association suppressed |
| --- | --- | --- | --- | --- |
| survey | 2029 | 196（.096599310） | 1833 | 0 |
| inspect | 1025 | 103（.100487805） | 922 | 0 |

2546个wide-frame background trials中25次raw trigger（.009819324），24次最终发出；触发后还过实际可见性检查，所以25≠24不是检测miss或目标association suppression。目标raw miss和最终同帧关联抑制是不同阶段，本次inspect关联抑制恰为0，不能把缺measurement一概算raw miss。

逐机会检查(t,drone,entity)唯一、各emit与实际audit对应、无missed却emit、无孤儿/重复emit；背景同样对账。还按原root/tick/drone/entity/channel SeedSequence独立重做每个**已列机会**的miss draw和每个背景trial的trigger draw，全部吻合，合计2779个sensor audit ID全部被对应机会/背景引用。采集源码在actual camera/FOV/ray可见之后记录机会，随机通道不依赖方法调用次数；理想稳定云台、结构化合成测量的范围明确，不是real vision。完整未emit机会的全集证明仍受F2限制，待独立重放/账本完整性绑定补足。

原score记录H/E1、C.937212864、S1、D.4、Y.779070955、SR0。这里只检查G3要求Y≥.2、static_recall≥.5的接口与上述物理/传感条件；不自行重算全部C/S/D权重或把SR0说成任务成功，完整业务评分由另一报告负责。

## 实际验证与Attention

独立离线脚本`/tmp/audit_g3_v19_control_sensor.py`（SHA `dfb26c4dee47593e694ee2ce5fbe1c9c1147a3c0199744eb9fa547cce6b35050`）退出0；摘要/log SHA均`38b2591fa78a318e2bba4b92ba9542ad39f567a7112f9fa51dc9d54e14cf1b5b`，已复制到上述冻结/tmp证据目录。仅做字节/数组/公开审计记录的离线复核，没有启动物理或模型。完整业务重放的21组/116RPC证据由mission_review核查，本报告不将其冒称含原raw error trace或完整物理状态。

reviewed by inherited model configuration, independent context, same model family

- 原真实run的所审物理、时钟和normal sensor点门证据一致；三项初版checker P2必须修复后再将入口作为完整自动资格检查使用。
- 原记录14源/61依赖真实完整；反例仅位于/tmp，没有回填/删除历史。新检查源或receipt应另留版本，不修改原v1结论。
- 10Hz位置、1Hz进度、240Hz误差分别有各自边界；没有精确离线重算全240Hz里程、全部control lag或完整physics state。
- 单seed开发参考、sleep surrogate和synthetic sensor不等于模型时延、正式策略性能、真实视觉或人类领域认证；G4仍未放行，human_review_pending。

本版首先冻结初版检查入口与原记录复算；修复复验如有授权仅另行追加，不删旧发现。
