# v18完整执行器结果与G1组成资格独立审查 v1

日期：2026-11-05。同模型家族独立上下文工程Verify，不是跨模型或领域人工认证。只写本报告；未改源码、旧报告、契约或结果，未重跑physics/矩阵/全套测试。审核对象固定为v18 manifest+归档，后续当前mission的v19修改不回溯影响原v18的真实unchanged记录。

## 结论

**支持v18开发期G1控制资格的组成结论：原基础五族100份可以条件复用，新observe/transfer各20份全部满足v18合同。** 执行器manifest的qualified=true与独立复算一致；每族成功率均1.0，超过合同≥.95。没有发现slot SHA、冻结输入、完整时长、评分、计数或保存判定的矛盾。

| 组成证据 | 完成/通过 | 适用范围 |
| --- | --- | --- |
| g1-full.v4 | 100/100，五族各20/20 | 原基础控制族，输入等价复用，非新跑100次 |
| executor-full.v4 | 40/40，observe20/20、transfer20/20 | 冻结v18的360秒、四机、无风/2m/s风控制资格 |

结论只覆盖声明的开发包线：nominal17m/s、pitch_trim−.07、同刻欧氏XY≤10m、Z±2m、全部模式35m/540m名义准入和实际硬安全/退出要求。v16/v17的6m失败没有改判；新10m不是旧6m假设获证。20个observe**资格slot**通过，不代表20个observe job无本地缺格：其job终态仍全部failed，reason为local observation gaps retained；固定appearance0的D≥.8另行验收，详见下文。

Controlled next-control-tick规划仍只证G1控制/因果可解性。科学场景roots、正式模型比较、真实异步1×、G2/G3/G4及领域人工复核均未授予，human_review_pending。v19的新guard不在本报告资格身份内。

## 输入身份与冻结范围

路径相对paper/research/20261104-inspection，源码路径相对repo；旧源以qualification/source/finite-arcs-v18为根。

| 对象 | SHA256 |
| --- | --- |
| qualification/executor-full.v4/manifest.json | ef8a9efc045d661acffe1963b7c4eb25834ccb156c35b159eaf5e9c9df01a747 |
| planning/gate-contract.v18.json | f49a4439ed3fed4392d34ac69da5d087dea5c825ace49da39c1d2187163c8883 |
| qualification/source/finite-arcs-v18/binding.json | 1efd2687573965a812cd9c181708114f7a2692a432d4254a02a8e47880133ed5 |
| qualification/g1-full.v4/manifest.json | b07166e6415a9f8cfb6f7d003f53dc356938807537f9ad4a7b451fce108d1b97 |
| qualification/review/tracking-envelope-v18-review.v1.md | 195abbc611fd3cf103d6a2ed7a24d93258039fa18f1c33d81c41c56c42ec46aa |
| qualification/review/g1-v16-full-review.v1.md | 4b48cf4a58980d121278ac2f2486df501c4ec281ba45bd85a948bb401c9da928 |

manifest内contract与原v18文件整个对象相同，contract_sha256匹配原字节。15份source snapshot逐份UTF-8字节与归档相等；全部source SHA与binding.source_sha256一致，binding所指manifest SHA也匹配。关键执行源如下，其他9份同样逐份核对，未用后来工作树的mission替换原文件：

| 冻结源码 | SHA256 |
| --- | --- |
| planning.py | b87fcdedf3aa26a46560056376fb69f9ff969270149c9cae429b4a9e294df6ab |
| execution.py | dc301939ea1d55f08f01c69af5561c75f762ac45bdd0d144d677514387425ea4 |
| mission.py | 24bc2a332508640f4043398f649e180b9cae167b930d22e2108135651c855b75 |
| paths.py | dbd31dc94a7cdcbcb17b8d9f1847fa88cf4cd12e4609ec8ba57e2233a367855b |
| eval/inspection_tracking.py | 19151659b4a969c51a6cbcb94b4f10aa1cb4489bebec6bd50f28b639a07884ba |
| eval/qualify_inspection_executor.py | b76f6745e4f1cfed8c0b4055b683a93811f23685378ef2c84c3e723d8010a2cd |

61项外部依赖均实际重读重hash匹配（PyFlyt源码/列出的机型资产、PyBullet二进制）；四项版本元数据匹配：PyFlyt0.29.0、pybullet3.2.7、numpy1.26.4、numba0.68.1。sources_unchanged、dependencies_unchanged、contract_unchanged均true。该范围不等于整个OS/Python/所有数值库二进制环境的完整锁定。

## 全40分母、时长与逐步数据

完整网格恰为两族×roots800…819，40个唯一slot，目录JSON集合恰为manifest+40份记录。全部实际文件SHA匹配manifest，family/root/passed身份相符，status=completed、operational_error=null、infrastructure_errors为空；无删除失败、缺项、重复或分母过滤。

每份请求/实际360秒，expected_steps=world.physics_steps=flight.physics_audits=tracking.physics_samples=86,400；总14,400模拟秒、3,456,000个world步。四机exit/range数组均长4，10Hz trace各3600条，位置数组形状及时间格完整、所有JSON数值有限。风按偶数root无风、奇数root2m/s且方向root弧度；selected=(root//2)%4，fixture origin/appearance0与请求机匹配。这20个控制roots跨族复用，不是40个独立道路/地形科学场景。

40份actual_control_profile与合同完整G1.control_gains逐键相等。冻结Mission:33–34使用四机96m初态和inspection_gains；profile为pitch_p4.5、pitch_trim−.07、elevator_trim−.1、speed_p.12、期望vertical command limit1.2。nominal17和follower16…18.5是合同允许的不同层次；没有把初始20m/s误写为目标20，亦未把1.2称为实测爬升率上限。

逐份解压base64/zlib little-endian float64，核对raw SHA、17列、字节数、86,400行及严格1/240秒时间。按四机任一XY>10或Z超自身low/high每world步计一次，独立NumPy重计所有峰值及计数，完全匹配保存summary：**全部40份XY/Z/seam违规计数均0**。trace.xy_tube_m均显式10，低/高高度实际均94/98；t≥20秒max|Z−96|与steady-Z峰值相等；240Hz Z和10Hz位置逐点对齐。所有peak witness的actual/nominal水平距离、原始行XY和Z也一致。

## 动态D完整独立复算

从原始accepted report请求、delivered_observations及sensor_audit重建不可变版本，验证证据先到再报、measurement time和estimate不变、固定首次20m身份关联及candidate匹配。每root按原始appearance0的60个3秒格评分；首格含0、其余左开右闭；先过滤合法且report到地面≤格末+6秒，再按最新测量/同刻evidence ID选择，最后检查身份、class、quality≥.8并按**测量时刻**原fixture真值用3/5/10m分段质量计算。

没有调用冻结/当前共享grader。全部1200格的窗口、index、weight=1、evidence ID、quality与zero_reason逐项一致，20个D误差≤1e−12；无跳过空格、缩短分母、改用observed/forecast或选择正确旧证据。D均值.916666667、median.919444444、范围.838888889… .966666667。格诊断：positive_quality1105、no_legal_timely_measurement74、incorrect_class19、insufficient_quality2；标签不单独证明某种因果故障。

| root | D | 最大XY(m) | 最大steady Z误差(m) | 最小四机exit dwell(s) |
| --- | --- | --- | --- | --- |
| 800 | .933333333 | 3.691302 | .952097 | 78.666667 |
| 801 | .877777778 | 4.070919 | 1.349405 | 32.816667 |
| 802 | .877777778 | 7.317076 | 1.123639 | 82.091667 |
| 803 | .933333333 | 4.506079 | 1.442926 | 45.625000 |
| 804 | .894444444 | 7.276018 | .936013 | 48.141667 |
| 805 | .927777778 | 4.337078 | 1.240902 | 32.308333 |
| 806 | .911111111 | 3.443955 | 1.010542 | 48.141667 |
| 807 | .966666667 | 4.064060 | 1.309532 | 31.700000 |
| 808 | .911111111 | 5.773314 | 1.018667 | 48.141667 |
| 809 | .894444444 | 4.557263 | 1.419666 | 32.783333 |
| 810 | .927777778 | 3.382708 | 1.031245 | 79.658333 |
| 811 | .838888889 | 3.731631 | 1.248676 | 46.125000 |
| 812 | .911111111 | 4.075774 | .933878 | 48.141667 |
| 813 | .922222222 | 4.354419 | 1.383694 | 30.958333 |
| 814 | .927777778 | 4.041578 | 1.022075 | 48.141667 |
| 815 | .933333333 | 4.478723 | 1.357180 | 33.391667 |
| 816 | .966666667 | 5.813008 | 1.017335 | 48.141667 |
| 817 | .911111111 | 3.784906 | 1.252378 | 32.591667 |
| 818 | .916666667 | 4.284961 | 1.170195 | 34.691667 |
| 819 | .950000000 | 4.439917 | 1.430770 | 30.833333 |

20个observe请求duration均180，实际service start3.5…11.0秒、结束183.5…191.0秒，每个完整运行180秒；但job本地窗口“每格均有样本”的条件未满足，20个终态全部failed。资格D仍从hidden appearance0开始固定180秒，按合同允许部分格质量为0而总D≥.8。不得把两个窗口、两个成功定义混读。transfer则每份四个不同机的survey job全部succeeded。

## OwnState、capture/seam、nominal几何与硬安全

全量核对3205个plan_request的四机完整OwnState+实际remaining range，2382个commit、437个seam。snapshot时间与日志一致、向量形状和数值有限，airspeed由world_velocity−wind重算吻合；range非负单调，lead=.5秒，observed track满足measurement≤available≤now且age≤8。3205个response均在逻辑下一control tick，commit program SHA与对应response一致、时刻匹配请求commit_at。每份case都有四机tail_start。

对所有commit/seam独立复算门条件：XY≤10、Z interval、heading/course/bank error各≤10°、ground-speed error≤3、airspeed15…25、|pitch|≤15°、|vertical speed|≤3、各|body rate|≤1及积分器20/15。可直接由OwnState重算的字段与metrics相等；全部commit通过，全部seam.capture_valid=true，与seam_failure=0一致。11个capture_envelope_rejected及18个proposal_invalidated仍保留，未被当成成功commit。

全部已返回geometry的continuous nominal separation lower bound≥35.051245m、fence upper bound≤538.397469m；sample minimum减2.7/围栏sample maximum加1.35的补偿逐条一致。冻结planning:335–360对完整四机prefix+periodic tail共同筛选，所有mode使用35/540门。候选计数是整次request聚合：inspect最多1、observe6、exit4、survey13；“per work最多4个return options”不应误说成整次request总数≤4，更不代表异步时延已资格。

hard summary全部无violations；保存位置/原始240Hz Z、10Hz机间3D距离及fence与summary界限一致。归档flight:24–30、110–152与mission:69–76确认实际积分后取真实位置/速度完成hard、tracking和range审计。整体实际AGL94.395969…97.349405m，airspeed15.921498…20m/s，最大bank30.679159°，最小实际3D机距36.991996m；未记录碰撞、硬包线、分离或边界事故。

observe最小四机exit dwell30.833333s，transfer28.441667s，均≥20；最小remaining range分别11871.890678m、11874.362334m，均≥1000。全40份每机最后20秒位置独立检查R80 home circle，最大径向误差.566458m、位置差分course相对切向最大1.484692°；没有仅看exit job status。实际距离预算消耗≥10Hz折线下界且<27×360保守上界，各request实际range单调；未声称用10Hz可精确重算240Hz行程。

raw trace可独立重计已保存XY/Z/interval的每步计数和峰值，但不含每步完整actual XY/速度/roll/接触及全部nominal program。因此本审查**不是物理重放**，不能从原始full state重新生成所有误差、接触状态或空气速度。完整保存的计数、峰值见证、OwnState及源码采集链支持本次合同审查，范围不扩大为连续真实飞行保证。

## 基础100组成与边界

再次核对g1-full.v4全部100份实际SHA，manifest不变；与v18.base_control_qualification_basis直接SHA一致。v18.G1整个对象与原manifest.contract.G1相等；三份实际功能consumer的原snapshot、v18归档及现存文件逐字节相同：control.py `c676d237…a235270`、flight.py `ef9be915…d67cbdb`、qualify_inspection_g1.py `852ea8c0…794522`，完整SHA见前瞻审查及原manifest。依赖61项及四项版本元数据也与基础原绑定相同；envs初始化为空，inspection初始化只有同一docstring，无新增导入行为。

基础runner仅消费这三份功能源和G1子对象，未import变化的planner/paths/execution/mission。结合原g1-v16-full-review.v1.md对基础指标的独立复算，五族各20/20可作为本次组成证据，不因v18规划代码变化机械重跑100。复用不是新独立重复，不修改旧contract SHA/qualified，不替代新执行器证据，也不自动延伸到v19或未来不同输入。

## 验证记录与Attention

独立离线脚本`/tmp/audit_v18_full_control_review.py`，SHA `cb296b6a54cc3b082992f178b1badc9f043f41c907b99f5084d594753a4c5a2d`，命令`.venv/bin/python /tmp/audit_v18_full_control_review.py`退出0。日志`/tmp/audit_v18_full_control_review.log`，数值摘要`/tmp/audit_v18_full_control_review_results.json`；未import当前生产scorer/recount或执行physics。另做了base100绑定和各类日志全量计数检查。168 passed in21.15s日志已读取（SHA `3cb466e7d9720890816f0beb68bdbacbe4af139757755cd56a9f3d84b6bcdd07`），本审查未重跑全套，也未将其当40份物理证据替代物。

reviewed by inherited model configuration, independent context, same model family

- 无结果一致性缺陷；支持冻结v18的G1开发控制资格组成结论。20个observe资格slot通过与20个本地job缺格failed同时保留。
- 旧v16/v17按6m仍失败；本次是10m前瞻资格，不能回填或包装为原6m通过。
- 基础100为同输入历史组成证据，执行器40为新结果；控制roots不是独立正式场景roots，不能把两者当140个科学独立样本。
- 未授予真实1×、G2/G3/G4、正式实验或领域人工认证。后续mission guard/v19另作有界组合审查；不改变本v18结果身份。human_review_pending。

报告完成后冻结，不回填后续版本。
