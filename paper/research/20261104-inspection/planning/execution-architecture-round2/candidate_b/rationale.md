# Candidate B：固定偏心观测弧与相位门移心

## Problem

最终 v11 基础控制100/100、transfer20/20，observe仅1/20；新G1失败。`observe-05.json`在4s已有实际合格观察，最终D=.0667、仅8个service样本；`observe-10.json`D=.0333、仅3个样本。两者随后仍不断采用“接近目标圆”的新程序。`observe-00.json`首个observe在3.5s已采样，但首次更新请求到56.57s才出现，第三至第七格已无及时证据。不能把这种损失归因于API延迟：这些均为next-tick受控G1。B保留单写者、异步规划、完整恢复和闭合尾环，改用**能看见目标的有限固定弧**，目标无需在圆心；不再每次追新的目标中心，也不等待两秒确认后才开始必要转弯。

## Usage (caller's view)

调用方仍为三处：mission用`submit(..., seed=已到达Observation)`受理inspect/observe；主钟用`advance(now, actual_navigation, actual_exit_dwell)`取得四机`FlightTarget`；四机曝光后用`local_measurement(obs, now)`推进实际后验条件。`cancel`只取消作业权限。inspect的两张20m分离实际图像完成后，下一observe沿用已有物理弧；没有第二套公开导航API，没有自动替Agent选择对象。调用示例先于类型写在`sketch.py`开头。

## Shape

`execution.py`单独拥有JobBook、逐机因果LocalTrack、四机RouteProgram及逐机`_Cell`；规划进程仅接收不可变`_EditRequest`，返回一次性`_PreparedEdit`。新增内容集中在这两个模块，直接复用`TimedPath/RouteProgram`、风修正、搜索/转场与home recovery。按`separate-before-serializing-shared-state`，没有第二个作业/飞行写者；按`encode-lessons-in-structure`，程序、cell和revision原子提交，不能先改圆心后补恢复。

每个圆弧的地面圆心固定。起始由当前保留路径未来gate的位姿/航向唯一确定左右切圆，actual navigation决定能否捕获，不能用目标中心反造飞机初态。初次捕获先看“最早合格视图机会”，在已有视野即将丢失时立即请求转弯；不等精确圆心到达才开始服务。仍处可用视野时优先续原弧。因果目标漂移造成光学余量不足时，在未来相位门插短切向直线，再接固定圆；圆心实际沿有限航迹平移。用有长度的曲率ramp衔接，端点重新计算，不能把roll变化压成瞬时接缝。候选≤16，只有最优两个做昂贵全机恢复检查；工作只承诺4–12s，丢更新即沿已预约恢复退出。按`boundary-discipline`，有限/因果/消费者校验只在请求/提交边界，内部纯几何构造。

不是“任何速度都可连续看见”。18m/s、R80、2m/s风的名义bank峰值约27°，接近30°控制上限；未知turn、实际roll、相位误差与跟踪余量必须实测。导航预测按测量年龄和创新增大误差余量；4m/s滤波限幅不是目标真速度上限，Gaussian噪声也没有严格有界支持。余量不足则缩短工作、实际宽镜头重获或明确拒绝，不能补造测量、延后隐藏出现窗、删格或降低D阈值。窄镜头同时检查50°off-nadir和10°half-angle；实际云台是ideal stabilized，不擅自把bank加成额外视轴偏转。roll仍影响实际姿态、0.3m光心杆臂和ray。宽镜头可见不等于质量≥.8（正常质量门限约43.30°）；低质实际宽帧可更新导航，不能算service。

每个提案保留旧前缀到future gate，接有限work、**目标无关已预约return**、原home周期尾环；全机检测直到全部进入同周期尾环后再完整一周期。若采用示例性10m水平管，安全准入必须由27m改为至少35m且围栏≤540m；3m高度误差在96m命令下仍须审计60..100硬限。这两个值均**未资格**，不是旧6m/2m失败的重命名。取消、handoff、worker死亡、断链都不释放任何飞机既定占用；改派必须再次与全部旧占用一起准入。新job token仅作业权限变化，不抹掉物理cell。handoff仍只凭接收机新真实合格样本取消精确旧source token。

## Synthesis decision

由arena填。B独立选择解析arc/ramp/runout库；没有读取候选A。它与“滚动预测闭环guidance轨迹”的区别是规划对象始终为显式几何primitive，目标轨迹只作有限光学约束/排序，不能直接移动飞行圆。

## Tradeoffs accepted

- 接受有限库不完备及不利初始航向的真实失测，换取不会每次重做长CSC、可完整预约的短时动作；不能由此降低原任务义务。
- 接受额外曲率ramp资格成本，换取不假设瞬时roll；当前解析探针未验证ramp，不能先宣称物理成功。
- 接受较大候选tracking管会挤压视野和四机可调度性，换取与现有失败证据一致的更保守准入；若新资格仍失败，停止扩大数值掩盖根因。

## Alternatives considered

- **随估计重建目标中心圆+CSC**：调用接口虽小，却把高代价捕获藏在每次refresh；已经有完整失败证据。
- **永久固定偏心圆**：接口最小，但1.5m/s目标60s理想距离达132.82m，超过可见范围；被本次同输入解析反例否证。
- **只在每整圈移心**：直线漂移且相位正确时很好；突发转向、错相位可能等待一整圈。采用每≤3s可选相位门与4–12s有限work，把整圈公式仅作为候选位移建议。

## Open questions and risks

以下均是待实验消除的问题，不需要用户代为决定常规参数：2s曲率ramp是否仍能在光学余量内转向且守住候选管？已有可见性但非可维持航向时能否及时捕获？未知转向/连续随机失测后的新真实帧是否在原3s bins内恢复？四机反复替换完整恢复时是否仍有足够可调度空间和真实1×CPU余量？如果其中任一失败，B尚不支持正式实验。

## Next implementation step

先独立写一个≤60s真实PyFlyt否证：从当前抖动20m/s初态、2m/s逆风、首个合法inspect时刻开始，执行含ramp的切弧→相位门直线→切弧并由真实相机2Hz喂因果filter，逐240Hz审计硬限/候选管，逐3s保留缺口；未通过之前不集成、不跑新大矩阵。

## 六项rubric证据（0–3，设计成熟度而非资格得分）

| 项 | B自评 | 已有证据与最小缺口 |
|---|---:|---|
| 物理几何/感知 | 2 | 唯一解析否证见下；永久固定圆和延迟初捕获被否证；ramp/真实roll/ray与未知turn仍未验证。 |
| 因果隔离 | 3 | sketch仅ObservedTrack、消费者局部history、实际own nav、已提交四机program；不读Entity/hidden_entities、未来或grader。预测类型无Observation出口。 |
| 任务能力/公平共享 | 2 | 完整保留survey/inspect/observe/handoff/exit、原180s/60格和原科学矩阵；新光学primitive所有B0–B4共享。持续采样可靠性尚未通过。 |
| 单owner/失败/取消 | 3 | 原子program+cell+revision；全机token、future capture、独立worker、旧完整恢复和尾环保留；实际handoff后置条件未弱化。 |
| 1×有界/资格成本 | 2 | tick仅lookup/一次nonblocking poll；≤16候选、≤2完整fleet验证；单worker。尚无实际耗时/1×及新ramp资格，不能报通过。 |
| 小接口/可维护 | 3 | 公开接口0增加；新内部请求/返回只包现有PlanRequest/Proposal，修改集中execution/planning；不建通用导航框架。 |

## 唯一小解析否证与边界

`analytic_falsifier.py`只读现有`paths`，没有world、generator、隐藏轨迹或旧run真值，0.073s完成。显式输入R80、air18、逆风2、目标导航速度1.5m/s、60s：东向phase gate直线46.65m（含直线期间目标继续移动），circle28.19s，完整cell31.10s。理想最大距离91.89m；暂拟10m机位+4m目标+0.3m水平光心杆臂+0.1075m遗漏余量后106.30m，低于按96−3−.3−.6m高度计算的109.76m reach。完整固定圆同输入132.82m失败。相同起始航向、目标在北侧的初捕获比较：1.5s future gate为106.84m尚可，3.5s gate为134.69m失败。**这仅排除了一个必要几何反例，不证明真实可持续感知。**

本探针没有ramp（故也没有物理roll）、实际capture、相机ray/10°aim误差、分类/随机失测/关联/上传、四机恢复或1×；目标允许误差是假设，不是已校准噪声支持。速度提高、预测转向或误差增长会耗尽余量，不能据此缩窄benchmark生成器或降低义务。最先否证项就是上节含ramp的真实首捕获；仅“先盘一圈几何好看”不足以救回动态前几格。
