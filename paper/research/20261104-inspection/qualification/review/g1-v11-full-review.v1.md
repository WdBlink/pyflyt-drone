# v11新完整G1结果独立审查 v1

日期：2026-11-05。角色：独立上下文工程结果审查，继承主Agent模型配置；不是跨模型审查，也不是领域人工复核。唯一写入本报告，未修改源码、契约或结果，未回滚他人改动。没有重跑矩阵、600秒任务或付费服务；round2设计目录不在本次审查范围。

## 结论

**整体新G1未通过。** 独立核对支持两份manifest的原判定：

| 矩阵 | 完成slot | 通过 | 分族结果 | qualified |
| --- | --- | --- | --- | --- |
| g1-full.v2 | 100/100 | 100 | 五个基础family各20/20 | true，仅基础矩阵 |
| executor-full.v1 | 40/40 | 21 | observe1/20；transfer20/20 | false |

没有发现文件SHA、分母、完整时长、审计覆盖或保存判定的矛盾。基础控制与transfer成功不能替代observe资格；observe的17个D失败、4个XY管违规、7个Z envelope违规（有重叠）均保留。全部140份硬安全审计未记录事故/硬违例，但这不等于新G1或完整平台已通过。

本报告确认的是冻结v11结果与验收合同一致。Controlled next-control-tick规划只属于G1控制/因果可解性，不是1×、实际模型/规划时延或G3证据。human_review_pending。

## 审查对象与冻结身份

所有路径相对paper/research/20261104-inspection；源码相对仓库根目录。

| 对象 | SHA256 |
| --- | --- |
| qualification/g1-full.v2/manifest.json | 60ea1fb1b40787af80eb174cb3f8105bb6dbd55ea1075307f376c54f5c37236a |
| qualification/executor-full.v1/manifest.json | 21c6ee9864e3ec59ffcb445bc00bbd583b7ef0af70e8096a009db028c6d89a6a |
| planning/gate-contract.v11.json | 3af795fa1b00cce49d29fb1732ff3758fbc44476ed2d185ec7d681ed715a3752 |
| control.py | e6cdb38516dc172bd812dc7334862201f20912a1787ac800004953cca92dd4b5 |
| flight.py | ef9be915c92c207a3e8f43b9ddc1907010beea59be6d96136330e82a5d67cbdb |
| qualify_inspection_g1.py | 852ea8c02cade269b48ea1bf99158913ddaef2038f6a368bd0987c31cc794522 |
| qualify_inspection_executor.py | a8064b2eef9d68557006d8102b31b2c1ce9d823762b1be3b6cff67e02c32cbc8 |

两份manifest的contract对象相同，SHA匹配原始v11文件。基础矩阵保存3份执行闭包源码；执行器矩阵保存14份，包括共享控制/物理/规划/任务/评分及原子写helper。全部快照与审查时当前文件匹配；报告写入前再次确认manifest及这些源码未变，未发现round2目录工作污染冻结输入。

两矩阵各自列出的61项PyFlyt源码/机型资产及PyBullet二进制SHA均逐项重新计算并匹配。版本为PyFlyt0.29.0、pybullet3.2.7、numpy1.26.4、numba0.68.1，与当前安装元数据一致。sources_unchanged、dependencies_unchanged、contract_unchanged均为true，并核对了其对应字节；没有把当前文件SHA回填历史。

### 基础矩阵确实使用了新profile与18m/s

冻结qualify_inspection_g1.py:45–46从contract.control_gains构造ControlGains并替换全部controller；47读取nominal_airspeed_m_s=18。67–71对circle、survey和heading各分支都显式传入18，冻结SurveyRoute.target也确实将该参数传给FlightTarget/orbit_target，没有落回20默认值。

该profile包含pitch_p4.5、elevator_trim−.1、speed_p.12、desired vertical command limit1.2。基础slot未逐份另存actual_control_profile，因此应用证据是冻结的实际执行分支与输入绑定，并由空气相对速度向18收敛的记录交叉印证；不是仅看manifest里存在一个gain字典。20m/s是该基线的初始化速度，不能把初态max_air20误解为目标仍20。

执行器40份则逐slot保存actual_control_profile，全部与v11字典精确一致；初始保存位置与96m相符，冻结Mission初态为四机96m。

## 完整性、时长与失败分母

- 基础矩阵恰为5个family×roots0…19，100个唯一slot/文件；执行器矩阵恰为observe/transfer×roots0…19，40个唯一slot/文件。没有缺失、重复、删除或不在manifest中的case文件。
- 两目录分别仅有manifest+100/40个JSON。全部140个slot文件实际SHA匹配manifest，slot/record的family、root、passed一致。
- 全部status=completed，operational_error=null，execution infrastructure errors为空；没有未完成或平台错误被混入普通成功率。所有JSON数值有限。
- 基础各请求/实际时长为60/120/180/180/240秒，合计15600秒；执行器每slot请求/实际360秒，合计14400秒。总计30000模拟秒，不是30000秒1×证据。
- 基础逐slotphysics_steps=physics_audits=duration×240，合计3744000个world积分步。执行器逐slotworld/flight/tracking审计都为86400，与expected_physics_steps一致，合计3456000步。总计7200000个world步，不能视为独立科学样本。
- 10Hz轨迹时间格、条数、机体数组完整。基础直飞偶数root为1机，其余基础slot为4机；执行器全部4机，exit/range数组均长4。
- 每family成功率按完整20分母独立计算，得到基础五族1.0、observe.05、transfer1.0，与manifest完全一致。依冻结合同≥.95逐family检查，整体新G1失败；未只对通过的root归一。

wind字段逐slot符合偶数root冷静、奇数root为2m/s且方向root弧度。observe fixture的appearance全部为0，origin/selected drone符合(root//2)%4；actor inspect/observe命令使用相应selected机。以上是20个控制参数roots跨family复用，不能称140个独立道路/地形/灾情场景roots。

## 基础矩阵独立指标重算

从全部100份保存位置、冻结半径/half_length采样及路径中心独立计算直线、圆周和capsule距离，逐点与tracking_error_m一致；再按稳定窗口重算合并RMSE/max，全部与摘要一致。heading参考从冻结schedule取得，未用被测target自证。并以相邻0.1秒位置差分交叉检查ground course。

| family | 通过 | 最大slot合并RMSE m | 最大稳定窗口误差m | 最大稳定course误差° | 最小退出dwell s |
| --- | --- | --- | --- | --- | --- |
| straight | 20/20 | 8.664 | 11.698 | .333 | — |
| heading_changes | 20/20 | — | — | .334 | — |
| loiter | 20/20 | 2.594 | 4.054 | — | — |
| survey | 20/20 | 1.822 | 4.757 | — | — |
| sortie_exit | 20/20 | 2.972 | 18.593 | — | 93.662 |

均满足对应15m RMSE、40m max、10°course和20秒exit门槛。位置差分得到straight/heading最大稳定course约.33301°/.33407°，支持同一结论；它是0.1秒平均位移方向，不冒充瞬时速度。

heading_changes未应用的位置指标、其他family未应用的course字段不解释为真实零误差。基线安全范围为AGL69.914–96.842m、airspeed17.673–20.000m/s、最大bank30.412°、多机最小三维间距195.886m，均在对应硬门槛内。保存位置也全部在各family冻结边界内；直飞/航向family允许±5000m，不能将这些轨迹称为主任务±550m范围内搜索。

## 执行器独立完成与动态D重算

transfer逐slot有且只有四个survey job，drone集合为0…3，全部实际succeeded。所有40份硬安全/infra、四机最终20秒home exit及range reserve检查满足合同；各机末20秒位置的圆周半径与切向course也独立核对。全部执行器case的最小exit dwell≥49.233秒、最小剩余range≥11492.902m，显著超过20秒/1000m门槛。没有将ACK当任务成功。

对每个observe slot，本审查使用accepted report journal、实际delivered observation及sensor audit的同测量时刻truth，独立执行：

1. 检查报告证据已到达，原始estimate/stamp与证据一致；
2. 固定appearance=0的60个3秒格，候选按该格新测量和end+6秒到达筛选；
3. 只按最新measured_at、同刻evidence ID选样，再检查身份/类别/质量与3/5/10m误差；
4. 全部60格进入分母，不重置到首次发现或首次报告。

20份D、1200格的选中evidence ID和quality逐项与保存结果一致。D均值.5575、中位数.655556、范围.033333…至.85，仅作这些控制roots的描述，不是主研究总体估计。保存格级零原因：515格无合法及时报告、6格类别错误、8格质量不足；671格有正quality。不能只凭“无合法及时报告”把失败归因于纯通信或纯感知。

### observe全部root与窄管失败

XY/Z步数为保存的240Hz world审计步计数，不是机体数或独立统计样本。

| root | 独立D | XY>6m步数 | Z envelope步数 | 最终slot |
| --- | --- | --- | --- | --- |
| 0 | .794444 | 0 | 0 | false：D |
| 1 | .850000 | 0 | 0 | true |
| 2 | .583333 | 0 | 209 | false：D+Z |
| 3 | .700000 | 0 | 0 | false：D |
| 4 | .644444 | 0 | 491 | false：D+Z |
| 5 | .066667 | 0 | 0 | false：D |
| 6 | .683333 | 0 | 0 | false：D |
| 7 | .833333 | 661 | 0 | false：XY |
| 8 | .716667 | 0 | 0 | false：D |
| 9 | .450000 | 1001 | 0 | false：D+XY |
| 10 | .033333 | 0 | 0 | false：D |
| 11 | .283333 | 0 | 308 | false：D+Z |
| 12 | .666667 | 0 | 330 | false：D+Z |
| 13 | .400000 | 2057 | 1113 | false：D+XY+Z |
| 14 | .733333 | 0 | 0 | false：D |
| 15 | .561111 | 0 | 0 | false：D |
| 16 | .400000 | 0 | 535 | false：D+Z |
| 17 | .666667 | 0 | 1379 | false：D+Z |
| 18 | .255556 | 0 | 0 | false：D |
| 19 | .827778 | 1423 | 0 | false：XY |

按未舍入值比较.8，root0仍失败，不能先四舍五入放行。D达到.8的只有1/7/19，7和19仍因XY失败，所以observe最终仅root1通过。

- D失败17个：0、2、3、4、5、6、8、9、10、11、12、13、14、15、16、17、18。
- XY失败4个：7、9、13、19，共5142步（跨case合计21.425秒），最大误差8.231m。
- Z envelope失败7个：2、4、11、12、13、16、17，共4365步（跨case合计18.1875秒），最大settled误差2.464m。
- XY/Z共同失败root13；两种步数有时域重叠，不能把5142+4365冒充唯一违规总时长。
- 19个失败slot中：9个仅D、6个D+Z、1个D+XY、1个D+XY+Z、2个仅XY。没有退出/range或硬安全失败被掩盖。
- transfer20份XY/Z breach均0，最大XY5.669m、最大settled Z1.632m；全4机exit与任务完成通过。

四个XY失败均发生在奇数root的最大风条件，但这些root也改变噪声/初态，不能由此单独证明风是唯一原因。失败谱用于后续dev回归，保持原roots和阈值。

## 硬安全、退出及记录分辨率

全部140份safety.violations为空：collision、separation<15m、60–100m高度硬包线、15–25m/s空速、45°bank、flight boundary均未记录违规。执行器全体安全范围为AGL93.564–98.464m、airspeed15.929–22.308m/s、最大bank30.483°、最小三维间距25.376m。

窄XY6m和Z±2m/envelope失败不等同于硬事故。反过来，硬安全零事故也不能消除tube失败对规划跟踪假设的否定。基础20个sortie_exit与执行器40个case的末20秒采样位置均满足退出圆条件；执行器最坏径向/切向误差约.524m/.852°。range数值也与10Hz位置折线的距离下界一致。

**独立重算范围的实质限制：** 基础位置几何、稳定窗口、RMSE/max/course及observe完整D可以由现有JSON直接重算；执行器逐240Hz XY nominal/actual状态、逐tick误差或峰值witness没有完整保存。其精确XY峰值和5142/4365步数本次只能核对冻结audit实现、完整覆盖计数、阈值/峰值/保存判定一致性；不能从10Hz轨迹完全重新计算每一个240Hz物理tick。Z的7个受影响root已由原始10Hz位置相对96m独立交叉印证，sample max均不超过对应240Hz峰值；碰撞/瞬时bank/速度也未被本审查物理重放。

这项可复算性限制不推翻失败结论：独立可重算的D已有17/20不达标。后续结果若要求完全离线复核精确tube计数，应保留所需逐tick误差或可重放参考及峰值witness；本审查未修改现有结果填补缺失证据。

## 验收重算、范围与冻结

按独立task/D、完整coverage/profile、硬安全、tube breach、四机exit/range重新组合40个case verdict，全部与保存passed一致。按完整分母和冻结状态重算两个manifest的qualified，分别true和false，与原记录一致。未发现需要更改现有结果的错误，也未把失败root当作平台无效样本删除。

这次完成的是当前v11基础100slot与执行器40slot的结果审计。20个控制root跨族相关，不能把140slot或7200000物理步作为独立道路场景/方法样本；不存在由本结果支持的主研究显著性、LLM方法优劣或总体安全率结论。

所有执行器均为controlled nexttick transport：规划计算期间物理步可以暂停墙钟推进，不能替代主异步1×worker/G3。实际G2/G3/G4、600秒参考任务和领域人工有效性复核均未由本报告通过。后续round2设计使用新源码/契约/运行版本并保留本次失败，不回填本数据集。

本报告写入完成后冻结；未重跑物理、规划矩阵或付费API。human_review_pending。

## Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- 整体新G1失败：基础100/100和transfer20/20不能替代observe1/20。
- 全140slot完整性/冻结/覆盖一致；D和基础关键指标独立重算吻合，未发现改判或删分母。
- 精确240Hz tube计数缺少完整离线逐tick数据，只能完成所述审计与交叉印证；未冒称物理重放。
- 控制roots、1×/G3和领域人工review保持各自边界。

