# Guidance / paths 独立工程审查 v1

日期：2026-11-05。独立上下文工程审查，继承主 Agent 模型配置，未实施跨模型审查。唯一写入文件为本报告；未修改生产代码、已有探针记录或其他Agent改动，未重跑大矩阵。

## 结论

发现并复现的两项P2（不连续路径破坏采样补偿前提、等价角度接续被插成反向）已在本次审查期间修复，最终9项针对测试通过。当前所查moving-orbit前馈、ENU/crab符号、CSC端点及已有连续周期尾部记录未发现剩余确定性错误。

**已有6m/2m tracking tube结果仅是10Hz保存样本的通过/失败，不是连续时间tube资格。** 新增240Hz误差audit和运行中的transfer-fleet-wind.v1属于后续证据，本v1报告不审阅其结果，也不因代码已新增就升级历史记录。旧G1的参数与模式不覆盖pitch_p=4.5/elevator_trim=-0.1及这些新导航模式。这些探针均不是完整G1/G2/G3，human_review_pending。

## 范围与版本

已读取：envs/inspection/guidance.py、paths.py；eval/probe_inspection_observe.py、probe_inspection_transfers.py；tests/test_inspection_guidance_paths.py。核对初始指定记录，并根据主Agent补充纳入已完成transfer-fleet.v1/v2、transfer-single-wind.v1/v2。没有扩写调度架构，未审阅新的protocol/scenario实现。

历史记录中的paths.py SHA为 b0eae1d523f05ebf63bf5ce8f0d680a3deeac90dd8c2afe2e42539d0c5f258fc；两项P2可在该旧实现上重现。最后复核：

| 文件 | SHA256 |
| --- | --- |
| guidance.py | 9ab9c5290cbce60ad7f9fead635318f41561dedd1cfc2f0984b1cd45385bc0b9 |
| paths.py | fba70cbbf7ce74df30e9f76862cea5881d5024fc5b4f3af18c641ada58191c21 |
| test_inspection_guidance_paths.py | c3a30149bfc97669301595bafaf93aeb57496a050612a49590a8037cc1643d93 |
| probe_inspection_observe.py | de181fa3b194ee8c292c342c47fa1e1ed9c74bef8c12bfd7ca136aa9dd0f1628 |

各历史JSON的source_snapshots重新求SHA均匹配其source_hashes。不同记录绑定不同版本probe，未默认当前脚本等于旧哈希。审查完成时transfer probe另有新240Hz audit修改（当时SHA ebce7b069cdd20893facbf2e40f09c4457d8554eb16fefd72dbe07db86de79ee）；该新audit分支与其运行结果明确留待后续审查，不将这个SHA回填旧探针。

## 已修复缺陷

### P2-A：TimedPath未拒绝不连续primitive，导致连续距离下界假阳性

旧paths.py:142–150只检查路径非空和正dt/speed，没有验证相邻Primitive端点与切向。采样器把断裂端点在0.05秒内插成高速位置变化，fleet_clearance仍使用27m/s单机、54m/s相对速度的固定Lipschitz前提。

审查Agent独立构造了4个RouteProgram：airspeed全部20、wind全部0、尾周期全部0.2秒。其中A的primitive依次从x=-100、100、100、-101开始，B相反，每段长度1m；每个program首尾位置/heading闭合，但内部primitive断裂。旧构造器接受后：

- fleet_clearance(sample_dt=.05)的sampled_minimum约200m；
- continuous_lower_bound_horizontal_m约198.65m；
- 但programs[0/1].at(.025)的真实名义位置距离为0；
- 全部名义位置也在约±301m内，不能靠边界拒绝掩盖这个反例。

这是通用geometry screen的前提漏洞，不是当前make_program已有记录发生了瞬移。历史6份transfer program独立重建后，实际插值速度最大只有20或22m/s，连续几何有效，故其现有具体screen数值不因此作废。

同一旧构造器还接受dt=NaN，生成NaN duration/samples；接受零总长后at(0)抛IndexError。最小修复要求是构造期拒绝非finite时间/速度、零总长与几何不连续，并验证或推导clearance实际速度界。

修复复核：当前paths.py:143–149拒绝上述输入；276–283对每个program的prefix和tail逐段检查插值位置斜率有限且≤27m/s。from_samples也检查有限、严格递增时间和size界。新增test_discontinuous_primitives_cannot_bypass_clearance_speed_bound包含不连续路径、NaN、零长度及伪快闭环拒绝。本审查直接重放旧不连续/NaN/零长反例及伪快闭环，均已拒绝，P2-A关闭。

### P2-B：等价2π→0切向接续被线性插成反向目标

构造两段空间与切向均连续的东向直线：

```python
TimedPath((
    Primitive(Pose(0, 0, 2*math.pi), 1, 0),
    Primitive(Pose(1, 0, 0), 1, 0),
))
```

旧实现与第一次P2-A修复后的中间实现都允许该合法等价角度接续。但at(.025)返回位置(.5,0)、heading=π，处在名义位置上的target也指向西；期望应继续向东。这来自直接在2π和0间作数值插值。

最小修复是构造samples后先unwrap heading列再readonly，不改变位置或请求路径。当前paths.py:175已加入np.unwrap，与from_samples:188一致；新增等价角度反例位于test_inspection_guidance_paths.py:91。审查直接复验at(.025) heading=2π，target.heading≈−2.45e−16（东向），P2-B关闭。

## 数学与因果边界核对

### LocalTrack只重放收到的测量；预测不成为新证据

guidance.py:15–19把作用范围限定为调用方已收到/本地测量；first复制输入，update拒绝重复/旧stamp，predict拒绝向过去预测。它本身不实现访问控制或transport到达校验，调用方的可见性边界仍是前提。

observe probe:43–45提供一条明确标记initial_public_sample的初始候选测量；这是假设已获得初始候选，不是未知目标搜索成功。47–49的controller closure只访问自身导航、track.predict/velocity与本地wind。隐藏运动仅在56–75的传感/audit闭包中产生，通过实际detected后才更新track；没有把隐藏速度/未来点直接送入controller。

独立仅按measured_at≤trace.t的实际测量顺序重放LocalTrack，observe v1/v2/v3/v4每条保存estimated_center（及存在时estimated_velocity）的最大差均为0。除初始公共样本外，测量时间集合与detected帧时间集合完全相同。

v4的实际样本在44.5s后到48.0s才恢复，中间3.5秒没有新测量。预测继续参与导航，不进入measurements。local_3s_measurement_bins的40格都可能含实际样本，因为48.0属于(45,48]格端；这不等于连续测量gap≤3s，也不等于补回了45–48s失测。保存的max_gap=3.5和passed_physical_and_view_probe=false正确，两种诊断没有互相覆盖。该local bins字段也不是完整第6.2节交付评分。

### 移动圆与风下曲线前馈

令r为径向单位向量，t=Jr为ENU逆时针切向，d=v_center−wind。在圆上，相对中心切向速度

s=−d·t+sqrt(Va²−(d·r)²)

使air_velocity=d+s·t满足所需airspeed。guidance.py:85–93按均匀中心漂移对air-velocity方向求导，再以负roll符号映射PyFlyt；FlightTarget返回ground course，FlightWorld再作crab correction。径向反馈、中心加速度与滤波修正不包含在这个“精确均匀漂移”前馈证明内，由反馈处理，文档已限定。

现有moving-orbit有限差分测试在30个圆周相位核对air heading/径向相对速度/bank，均通过。另独立对course_kinematics在50个相位×正/零/负曲率共150点求wind-relative air heading差分，bank最大差约4.67e−9rad，未见ENU或crab符号错误。

风下物理失败的一个明确可行性边界：半径80m、恒定风2m/s、Va20m/s的全圆最大名义bank需求约31.663°，超过当前控制目标max_bank=30°；Va18时约27.007°。这说明“几何/airspeed约束可解”不自动保证当前bank限制下精确物理跟踪。该计算与wind.v1大phase误差、wind.v2降低到18后改善相容；未把它宣称为对全部瞬态误差的唯一因果归因。

### CSC端点、定时相位和无限尾部

CSC代码明确只枚举四种circle-straight-circle候选，不声称完整Dubins最短解。除已有3组端点fixture，本审查另抽150组有限pose，产生556个候选；最大端点误差约2.44e−12m、最大wrap航向误差约3.55e−15rad。每个候选primitive内部连接由前段end生成，未发现当前生成器端点推导错误。

TimedPath按wind-projected ground speed积分时间，follower同时用相位位置差与横向误差修正，不能把它理解为只跟几何横向距离。采样时间本身仍是数值预测，不是物理轨迹承诺。

RouteProgram要求prefix接到tail入口、tail首尾闭合且wind相同；tail以mod duration周期延续。fleet_clearance只接受相等tail周期，检查至max(prefix duration)+一个共同周期：此时覆盖所有未匹配prefix后缀与一个完整联合周期，不再仅截断到最短prefix。

在有效单机插值速度≤27m/s前提下，任意时刻距最近采样点≤sample_dt/2，pair距离变化≤54m/s；因此减27×sample_dt（dt=.05时1.35m）和边界加13.5×sample_dt成立。此结论仅针对理想插值nominal program，不包含实机tracking tube。

本审查从每份历史snapshot只提取make_program纯函数，用记录的stripe/delay/wind/airspeed重建，没有重跑物理或finite delay search。6份transfer记录中实际存在的planned_xy、prefix duration、tail period和fleet geometry摘要与重建值差均为0；geometry-only记录没有物理trace。实际插值速度最大20或22m/s，满足补偿前提。搜索只在声明的有限delay集合中最小化总声明delay再按词典序，未读取任务/目标分数。20×delay是公开的初始东向距离参数；有风/18m/s时不是实际等待秒数，新增字段initial_eastbound_distances_m明确了这点。

## 已有物理记录：独立重算与失败范围

全部已读记录的snapshot SHA与声明一致。有物理运行的记录均满足physics_audits=physics_steps=240×simulated_seconds，无隐藏operational_error；下表误差从实际保存位置重新求得，与摘要一致。

| observe记录 | 秒数 | ≥20s最大真中心径向误差m | ≥20s最大高度误差m | 最大测量gap s | 6m/2m采样管 | physical/view字段 |
| --- | --- | --- | --- | --- | --- | --- |
| v1 | 60 | 1.385 | 9.392 | 1.5 | 本审查重算失败；原版本未写tube字段 | true |
| v2 | 90 | 1.777 | 9.588 | 1.5 | false | true |
| v3 | 90 | 1.559 | 1.228 | 1.5 | true | true |
| v4 | 120 | 3.621 | 1.129 | 3.5 | true | false |

physical/view是硬包线与视场/gap检查，独立于较窄tracking tube。因此v1/v2的true不能解释为高度管成功，v3通过也不能抹掉之前高度管失败。v3/v4的pitch_p=4.5、elevator_trim=-0.1已记录；并非旧G1默认gains。

| transfer记录 | 风/名义Va | 最大水平相位+横向误差m | ≥20s最大高度误差m | 硬安全 | 暂拟采样管 |
| --- | --- | --- | --- | --- | --- |
| single.v1 | 0 / 20 | 3.425 | 1.332 | 无违规 | true |
| fleet.v1 | 0 / 20 | 10.322 | 1.385 | 无违规，最小距36.444m | false |
| fleet.v2 | 0 / 20 | 5.798 | 1.385 | 无违规，最小距38.403m | true |
| single-wind.v1 | (-2,0) / 20 | 20.831 | 2.947 | 无违规 | false |
| single-wind.v2 | (-2,0) / 18 | 3.404 | 1.450 | 无违规 | true |

transfer-fleet-geometry.v1不是物理通过：其名义采样最小距12.878m、补偿下界11.528m，小于27m暂拟筛选门槛，记录screen=false。fleet.v1通过几何筛选（下界33.499m），但实机第四架在恢复换bank处超过6m，正确保留物理采样管失败。fleet.v2新增40m切线runout并重新做有限delay选择，补偿下界34.663m，采样管通过；四机最终exit dwell至少35.133s。没有将这次几何修改或成功重跑回填v1。

这些tube峰值由10Hz trace计算，且某些运行终止时刻比最后trace晚不足0.1s。没有对遗漏的积分间隙或终止后缀作连续tube补偿。尤其fleet.v2距6m仅约0.202m，不能据此认证连续6m界；若用于15m+两个6m tracking tube的调度安全推导，必须用逐240Hz误差峰值/完整尾部或已证明的保守补偿重新资格。主Agent已接受该限制并另加新audit，本报告不提前确认它通过。

## 实际检查与停止边界

最终执行：

```sh
./.venv/bin/python -m pytest -q tests/test_inspection_guidance_paths.py
```

结果：9 passed in0.06s。本审查额外执行旧反例/修复重放、CSC150组端点检查、150点bank差分、四份滤波器因果重放、十份记录摘要/快照校验及六份历史program重建；没有运行新物理矩阵。

当前未发现需要继续改动本次已修数学/几何代码的确定性缺陷。新240Hz audit、transfer-fleet-wind.v1、生产路由接入和完整新G1模式/gains资格均为待后续证据；没有因此拖延本次审查交付。

## Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- 两项可复现P2已修复并复验；历史具体连续program的几何记录未因通用构造器漏洞作废。
- 10Hz采样管不是连续tracking tube资格；v4的3.5秒失测、fleet.v1和wind.v1失败均保留。
- 旧G1不覆盖新gains/modes；本报告不等同完整G1/G2/G3，也不替代领域人工复核。

