# v16 全140 slot G1结果独立审查 v1

日期：2026-11-05。独立上下文工程复核，同模型家族；不是跨模型或领域人工复核。只写本报告，未改生产源码、合同、原结果或其他Agent产物；未重跑physics、矩阵、600秒任务或付费服务。审查绑定旧manifest及finite-arcs-v16归档，后续工作树变化不回溯污染本次attempt。

## 结论

**整体G1失败，原manifest判定正确。** 全140份文件的SHA、完整时长、审计覆盖及保存的判定没有发现矛盾。独立重算支持以下结果：

| 冻结矩阵 | 完成分母 | 分族通过 | qualified含义 |
| --- | --- | --- | --- |
| qualification/g1-full.v4 | 100/100 | straight、heading_changes、loiter、survey、sortie_exit各20/20 | true，仅基础五族 |
| qualification/executor-full.v2 | 40/40 | transfer20/20；observe15/20 | false |

合同要求每族≥.95，20个slot至少19个通过。observe=.75，因此不能以合并135/140或“基础+transfer通过”声称新G1或平台通过。失败根为403、413、418的XY管（前两者同时有seam失败），及415、419的D不足；全部高度管、硬安全、退出、range及基础设施检查通过。

**需保留的语义区别：20个observe job本地终态全部为failed，包含15个slot-pass。** reason均为`local observation gaps retained`；实际均完成180秒service interval。冻结验收按appearance=0的固定180秒/60格D≥.8及控制条件判slot，不要求observe job.status=succeeded（归档eval/qualify_inspection_executor.py:135–143）。job自己的“每个本地服务格均有样本”判据更严格，且服务窗口从3.5–4.5秒开始（execution.py:393–400）。这不推翻合同下15/20，但不能写成“15个observe job成功”“180秒无观测缺格”或“持续全覆盖”。这是解释风险，不能通过事后变更原合同或原结果消除。

基础五族证据可以在严格输入等价边界内作为以后资格的组成证据复用，详见后文；planner单独变化不自动要求重跑100基础slot。执行器observe/transfer的资格不随之自动复用。

## 冻结身份与范围

以下路径相对paper/research/20261104-inspection；归档源码路径从qualification/source/finite-arcs-v16起算。

| 对象 | SHA256 |
| --- | --- |
| qualification/g1-full.v4/manifest.json | b07166e6415a9f8cfb6f7d003f53dc356938807537f9ad4a7b451fce108d1b97 |
| qualification/executor-full.v2/manifest.json | 8abb0ab3af95b33037ce790e7a574dbab9ae25034dfcf08860bc63d351e60959 |
| planning/gate-contract.v16.json | 42ec6e4886463011ab3c00915aa7d91656e2ea043fb7989a6ff6d53f85d587af |
| qualification/source/finite-arcs-v16/binding.json | b4ca3d960b08e545c371261710bff33cf95cd07fd3106cd30891d073939addf0 |
| envs/inspection/control.py | c676d237efe1b6227d238b4e5948c54a5725626e9d3f69e74ed9c1629a235270 |
| envs/inspection/flight.py | ef9be915c92c207a3e8f43b9ddc1907010beea59be6d96136330e82a5d67cbdb |
| eval/qualify_inspection_g1.py | 852ea8c02cade269b48ea1bf99158913ddaef2038f6a368bd0987c31cc794522 |
| eval/qualify_inspection_executor.py | f5c58c340018933bfdd81a69d2a581846e9ba3046da9b567d79a34c0deaf7586 |
| envs/inspection/execution.py | df11567e1af3699f3bd9a99d3580ebba8062f806bbad8bd7e46c1b86f25bfdef |
| envs/inspection/planning.py | 69f2553852471bf0ec5c365190f2123633d6d0b42d8e9379fe15febd62af3a5d |
| eval/inspection_tracking.py | 0720d79beee9fdbaf873cafa477cfa04d5ecf76335dd01f30f9a86d0bcdb1e5c |

两份manifest的contract对象均与原v16 JSON相等，contract_sha256匹配原始文件。基础manifest保存3份执行源码，执行器manifest保存15份；全部逐份转回UTF-8字节与归档相应文件相等。基础3份与执行器中的相同源码逐字节一致。binding中的manifest SHA也匹配，且记录归档时15份当前源与snapshot相同；本审查未用可能已变化的工作树覆盖旧snapshot，未把新SHA回填旧记录。

两矩阵各61个外部依赖指针相同，实际逐项重新读取并计算SHA，全部匹配：PyFlyt源码与列出的模型/几何资产及PyBullet二进制。记录的版本PyFlyt0.29.0、pybullet3.2.7、numpy1.26.4、numba0.68.1均与安装元数据相符。manifest的sources_unchanged、dependencies_unchanged、contract_unchanged均为true；归档提供旧源码的独立可读绑定。61项加版本元数据是实际保存的范围，不把它夸称为整个OS/Python/NumPy二进制环境的可重建锁定。

## 140个slot、时长、失败分母

- 基础网格恰为五族×roots0…19；执行器恰为两族×roots400…419。每组family/root唯一，目录JSON集合分别恰为manifest+100/40份记录，无漏项、重复或未登记case文件。
- 全140个实际slot字节SHA均与manifest匹配；slot/record的task或family、root、passed一致。全部completed、operational_error=null，执行器infrastructure_errors均为空。不存在把平台错误剔出分母、只对成功记录归一或运行中记录当完成的情形。
- 基础请求/实际时长依次60、120、180、180、240秒，各20次，共15,600模拟秒；逐slotphysics_steps=physics_audits=duration×240，总3,744,000步。
- 执行器每slot请求/实际360秒，expected_steps=physics_steps=physics_audits=tracking.physics_samples=86,400，总14,400模拟秒、3,456,000步。合计30,000模拟秒及7,200,000个world积分步，不是独立科学样本或1×时钟证据。
- 10Hz时间格、记录数与位置数组形状全部完整，数值有限。基础straight偶数root为1机，其余基础和全部执行器为4机；所需exit/range基数全部正确。
- 风字段逐slot符合偶数root无风、奇数root为2m/s且方向root弧度；执行器selected=(root//2)%4。20个控制roots跨族复用，不是140个独立道路、地形或灾情场景roots。fixture固定appearance0，未删掉早期低分格。

### 实际profile与17m/s

归档control.py:103–109返回pitch_p4.5、pitch_trim−.07、elevator_trim−.1、speed_p.12、max_vertical_speed1.2；1.2限制期望垂直指令，不是实测爬升率。执行器全部40份actual_control_profile逐键与合同G1.control_gains相等；Mission初态实际固定四机96m（mission.py:33–34），原始高度/own snapshot交叉印证。

基础runner:45–47使用合同完整gain字典替换全部controller并读取nominal17；67–71对circle/survey/heading分支显式传入17，SurveyRoute转发该参数。基础slot没有另存actual_control_profile，其证据是被冻结且实读合同的执行分支，而非仅因manifest存在gain字典。初始20m/s与整个记录max_air20兼容，不能据此误称目标20。

执行器planning.py:20的AIRSPEED=17，合同validator:79–82同时绑定实际inspection_gains和AIRSPEED；follower的16…17.5m/s请求范围是合同显式允许项，不把“nominal17”说成每个物理步均以17飞行。

## 基础五族独立重算

直接从10Hz位置、root生成的radius/half_length、固定中心及实际切换时刻重新计算straight横向误差、circle径向误差、survey capsule距离和sortie切换后circle距离；每个误差值及settle后的RMSE/max与记录相符。heading验收按独立合同schedule检查，聚合course-error与保存值一致，另用相邻位置差分course交叉检查，settled最大约0.365078°。未调用runner的run_case或执行physics。

| 基础族 | 20个slot中最大RMSE(m) | 最大settled误差(m) | 最大settled course error |
| --- | --- | --- | --- |
| straight | 8.493361 | 11.493178 | 约0.363832° |
| heading_changes | 不适用 | 不适用 | 约0.365034° |
| loiter | 2.494243 | 3.926365 | 不适用 |
| survey | 1.767880 | 4.458880 | 不适用 |
| sortie_exit | 2.975685 | 24.226392 | 不适用 |

以上均满足合同RMSE≤15、max≤40及直飞/航向变更course≤10°；heading族保存的XY=0不当作实际位置追踪能力。sortie各机最终exit dwell最小94.058333秒，超过20秒。基础追踪阈值按原合同10Hz采样，硬安全则逐240Hz；没有把基础10Hz追踪误称为逐240Hz窄管证明。

## 观察D：固定180秒、完整60格独立复算

从原始reports accepted请求、delivered_observations和sensor_audit重建不可变测量版本，验证measurement timestamp/estimate与原证据一致、证据先到再报、首次20m身份关联固定且candidate匹配。对每个root的60格执行：原始appearance0的[0,180]秒窗口；首格含0边界、其余左开右闭；按report到地面的end+6秒过滤合法版本；先取最新measurement（同刻按evidence ID），再检查身份、class和quality≥.8；以测量时刻的实际fixture真值计算3/5/10m分段质量。没有用forecast、报告到达时真值或只选正确证据替代评分。

全部1,200格的index、窗口、weight=1、选中evidence ID、quality与zero_reason逐项与保存值相等，20个D误差≤1e−12。固定60格分母从未缩短；所有actor observe请求duration=180，job实际service start3.5…4.5秒、结束183.5…184.5秒，资格D窗口仍从0开始。

D均值0.908611111、median0.927777778、范围0.738888889…0.961111111。1,200格诊断为positive_quality1094、no_legal_timely_measurement88、incorrect_class13、insufficient_quality5；这类标签不能单独证明缺格全因通信或全因视觉。

| root | 原始D | 240Hz最大XY(m) | XY违规world步 | 最大steady Z误差(m) | seam失败 | slot |
| --- | --- | --- | --- | --- | --- | --- |
| 400 | 0.927777778 | 3.979188 | 0 | 1.070805 | 0 | pass |
| 401 | 0.927777778 | 4.832104 | 0 | 1.398133 | 0 | pass |
| 402 | 0.916666667 | 3.991320 | 0 | 1.034577 | 0 | pass |
| 403 | 0.916666667 | 7.078675 | 1794 | 1.198493 | 1 | fail |
| 404 | 0.927777778 | 3.368322 | 0 | 1.164759 | 0 | pass |
| 405 | 0.933333333 | 3.908916 | 0 | 1.289090 | 0 | pass |
| 406 | 0.961111111 | 4.167944 | 0 | 1.119991 | 0 | pass |
| 407 | 0.950000000 | 4.482058 | 0 | 1.380777 | 0 | pass |
| 408 | 0.933333333 | 4.219811 | 0 | 1.039165 | 0 | pass |
| 409 | 0.916666667 | 4.577040 | 0 | 1.249927 | 0 | pass |
| 410 | 0.933333333 | 3.838444 | 0 | 1.037047 | 0 | pass |
| 411 | 0.916666667 | 4.599813 | 0 | 1.337397 | 0 | pass |
| 412 | 0.950000000 | 3.816648 | 0 | 1.024377 | 0 | pass |
| 413 | 0.811111111 | 9.979866 | 2829 | 1.335710 | 2 | fail |
| 414 | 0.955555556 | 3.850636 | 0 | 1.246459 | 0 | pass |
| 415 | 0.738888889 | 5.141099 | 0 | 1.164206 | 0 | fail |
| 416 | 0.927777778 | 3.336588 | 0 | 1.044926 | 0 | pass |
| 417 | 0.911111111 | 4.576512 | 0 | 1.378802 | 0 | pass |
| 418 | 0.950000000 | 6.489497 | 411 | 0.983353 | 0 | fail |
| 419 | 0.766666667 | 5.026940 | 0 | 1.259481 | 0 | fail |

## 压缩240Hz误差与seam复算

审计直接base64解码、zlib解压，并核对每份raw_sha256、little-endian float64字节数、17列schema、86,400行和严格1/240秒时间格。每个world步对四机任一XY>6记一次；任一Z超各自low/high记一次。独立NumPy表达式重计数和最大值与40份summary完全相等；全部Z interval实际均[94,98]，t≥20秒的max|z−96|也与steady-Z峰值相等。所有10Hz Z都与相应240Hz行对齐；每个peak_witness的actual/nominal水平距离、原始行误差、Z与保存峰值一致。

observe共5034个违规world步，折合各case违规时长之和20.975秒：403为7.475秒，413为11.7875秒，418为1.7125秒。不是四机违规次数之和，也不应把seam失败再相加为另一段独立违规时间。transfer全部20份XY/Z/seam违规均0，最大XY5.347885m、steady Z1.414820m。

归档flight.py:24–30在实际pybullet积分后执行hook；mission.py:69–76先hard audit，再从实际getBasePosition抓位置、记录误差并累计range；execution.py:227–246逐步记录当前program误差/Z/interval及峰值见证。以上是实际积分后的记录路径，不是仅control tick调用的假240Hz计数。

共检查3262个plan_request完整四机own snapshot及实际remaining range，2325个program_committed和433个physical_program_seam。snapshot字段含position/attitude/body_rates/world_velocity/airspeed/controller_integrals，均有限且at等于日志时刻；空气相对速度由world_velocity−wind重算相符，request lead=.5秒，range非负、随时间单调下降，track测量/available均不超request now且age≤8。每份case均记录全部四机tail_start。3262个response均在逻辑下一control tick，commit对应response的program SHA一致。

对每个commit/seam用原始门条件独立判定：XY≤6；Z在interval；heading/course/bank误差各≤10°；ground-speed误差≤3m/s；airspeed15…25；|pitch|≤15°；|vertical speed|≤3；各|body rate|≤1；积分器≤20/15。own-state可直接重算字段与metrics逐一相符；所有commit满足门条件，seam capture_valid全部吻合。未把20个正常capture rejection当作成功commit，也没有忽略3个实际seam失败：

| root | 时刻(s) | seam | 实际XY(m) | 失败门 |
| --- | --- | --- | --- | --- |
| 403 | 13.000 | work_end | 6.138373 | XY>6 |
| 413 | 169.150 | work_end | 9.742177 | XY>6 |
| 413 | 171.150 | recovery_start | 8.988900 | XY>6 |

记录中的完整nominal prefix+periodic-tail几何screen lower-bound最小27.006972m、fence upper-bound最大538.760115m，符合预定27/544；这属于理想nominal几何，不能代替physical tube证据。本报告核验的是已记录误差、见证、快照和门条件；压缩行没有每步完整physical state/控制输入/nominal program，故**不是物理重放，也不能独立从每步actual XY重新生成全部误差**。已记录XY错误的计数/峰值现在可完全离线重计，不再是v11只有摘要而无法重计的状态；保留两者区别。

## 硬安全、退出及range

全部140份hard violation字典为空，无碰撞/分离/包线/边界事故记录，无operational/infra失败。硬安全的逐步采集见flight.py:110–152；复核summary上下限，并用全部保存位置/原始240Hz Z、10Hz机间距离与边界交叉检查，没有冲突。

| 范围 | altitude(m) | airspeed(m/s) | 最大bank | 最小实际3D机距(m) |
| --- | --- | --- | --- | --- |
| 基础100份 | 69.717955…97.240683 | 16.614182…20.0 | 30.487006° | 196.124155 |
| 执行器40份 | 94.360546…97.249927 | 15.919635…20.0 | 30.661778° | 30.070575 |

observe最小exit dwell45.183333秒、剩余range11874.717506m；transfer分别40.350000秒、11875.173570m，均满足四机20秒/1000m。对40份每机最后20秒原始位置独立检查home circle R80：最大半径误差0.456929m、差分course切向误差约0.809884°，支持已记录dwell，没有用job exit状态替代实际轨迹。18000−remaining range均大于等于10Hz折线行程下界且小于27×360的保守上界；真实range的累计在每240Hz actual位置执行。没有声称仅凭10Hz数据能完整重算接触状态、每步空气速度/roll或精确240Hz行程。

## 基础100份能否作为以后资格的组成证据复用

**可以有条件复用，无需因planner变动自动重跑。** AST/import及实际调用核对表明，基础runner:21–25、37–124的本地功能依赖是control.py、flight.py和runner自身；三者不import planning、execution、mission、guidance、paths或scoring。FlightWorld直接构造PyFlyt；基础只把contract.G1传入run_case（runner:164），不读取executor_qualification、G2/G3/G4或主任务600秒协议。

精确适用条件：

1. 三份功能源码的字节SHA仍等于上表，G1整个子对象与本v16完全相同（含gains、nominal17、根列表、五族时长、风/初态抽样、settle、包线和阈值）。本v16的G1按`json.dumps(G1,sort_keys=True,separators=(',',':'))`计算SHA为`74897851e47511dbf1fbcc36ca4a824263e53e38c7c97ccb6d3a65e7240e86d6`。
2. PyFlyt/机型资产/PyBullet的61项依赖及相关数值运行环境保持本次绑定的版本与行为；Python包初始化/导入路径仍无额外行为。归档envs/__init__.py及eval/__init__.py为空，envs/inspection/__init__.py只有docstring；这些非功能初始化条件一旦变化也应检查，不能只凭三份SHA忽略新导入副作用。
3. 原manifest/100份原记录/SHA与本报告保留为独立组成证据；新资格显式引用该证据及“仅基础五族、旧输入等价”的判断。即使新合同顶层版本/整体SHA因executor项变动而不同，也不得重写旧manifest的contract_sha256、sources_unchanged或把其qualified=true改称新整体G1=true。
4. 复用只覆盖原五族的控制包线、采样追踪和退出，不替代修改后planner/guidance/follower的360秒observe/transfer资格，也不覆盖新的模式、目标速度、geometry、gain、频率、硬边界或G1阈值。三份源或G1改变后，本次等价复用前提不成立，应为受影响资格产生新证据。

因此未来只改planner恢复候选/前馈且这些输入保持相同，不会使旧100份attempt失效，也没有工程依据机械重跑100份；但修改涉及control.py/flight.py、目标包线或G1设置时不能继续援用本条输入等价结论。未来实际改动尚未审查，本报告不预先认证其等价。

## 复验方法、限制与Attention

独立只读脚本`/tmp/audit_v16_control_review.py`，SHA `d5ed5872442ad73e380dce9f799380fa1122cb364f289f71ba3e7119eaaa05c5`；在repo下执行`.venv/bin/python /tmp/audit_v16_control_review.py`退出0，摘要`/tmp/audit_v16_control_review_results.json`、日志`/tmp/audit_v16_control_review.log`。脚本只读JSON/归档/依赖，用独立数组与评分表达式，不import当前运行器、不调用共享recount/scorer、不启动physics。临时文件不是永久资格输入；本报告记录方法和关键数值，原manifest/slot/归档才是证据身份。首次本地重算脚本把sortie切换边界的“积分前mode/积分后trace”错当同一时刻，已按冻结runner先选mode再step的顺序校正；这不是原记录矛盾。

reviewed by inherited model configuration, independent context, same model family

- 整体G1未过；观察族75%，不能按所有slot平均放过。20个observe job本地缺格失败与15个合同slot-pass必须分别陈述。
- 本审查完整核对了140份hash/分母/时长及40份240Hz误差行、60格D、完整own snapshots与门条件；无额外结果一致性缺陷。没有重新运行物理，也未声称原始error trace等同完整physical-state重放。
- 控制roots和固定目标fixture不是主研究道路场景roots；controlled nexttick纯规划是G1控制/因果可解性证据，不是1×、真实异步时延、LLM性能或G3证据。未授予G2/G3/G4或正式实验通过。
- 继承模型的独立上下文复核不能替代领域人工审查；human_review_pending。未提供完整聊天transcript，本报告以实际读取的冻结文件、记录和本次只读复验为界，不扫描其他聊天。

报告写完后冻结；后续新源/新矩阵另立资格与审查版本，不回填本次attempt。
