# 有限固定弧执行器独立工程审查 v1

审查冻结时间：2026-11-05 10:13:46 UTC（工具时钟核验）。审查者：独立新上下文 `/root/finite_arc_review`。本报告只审查下表绑定的版本；主 Agent 已收到缺陷并准备另行修复，本报告不预先给未来版本背书。没有修改 production、实验状态、契约或旧结果，仅写本报告。autoresearch 仍是研究状态唯一拥有者。

## 结论

**当前版本不能称为完整的曲率 ramp 与完整入口域修复。发现三项具体缺陷，均已给主 Agent 可复现输入：**

1. 新 `reference_curvature` 在旧直线/弧接缝附近用中心差分混合两侧曲率，造成新 work 首行 bank 与旧 gate 命令不连续，仍被提交。
2. 新全导航快照中的世界速度只检查大小，未检查 course，方向错误的状态仍能通过 `_capture`。
3. 复用的恢复/转场几何仍有曲率跳变，包含提前 exit 的初始接缝及恢复内部、尾环接缝。该旧机制没有满足 round2 明确要求的全部实际接缝 ramp；不能用“复用了 recovery”代替新入口资格。

这些结论没有宣称某个反例已产生物理 hard 事故。前两项由纯状态/几何输入确定复现，第三项也有确定的名义命令跳变，另有当前小 pilot 的实际越管失败。主 Agent 已确认三项均修，不通过缩小文字范围规避。修复后的源码、测试和物理结果需要单独绑定并复核。

可确认的进展是：目标预测只用于有限导航候选排序；每个已接受 work 已预约完整目标无关返回和周期 home tail；获取中允许续约；实际世界速度、独立空速、控制器积分器和实飞剩余航程已进入请求。150 s 单 fixture 的 36000 行逐物理步误差已独立解码，未发现 6 m/高度区间违规，且结果记录四机均完成实际退出 dwell。它仍只是一个受控 transport fixture。

## 版本与来源

主要审查范围是相对 `qualification/source/executor-v11/` 的四个源码增量，行号均指本报告冻结前版本。

| 文件 | 本次实际审查 SHA-256 |
| --- | --- |
| `envs/inspection/planning.py` | `cedab6eebec595d17f0e776ea955419928896b3348083633694f816b1f9a93a1` |
| `envs/inspection/paths.py` | `26a57c8fddd4fb0898da4887f8ce8603b0e5bc312b8c608d76d0e8659cc53407` |
| `envs/inspection/execution.py` | `13cd11e725301794123dc078bb9388e6980ad44d67e730251277134c7b5fdb3d` |
| `envs/inspection/mission.py` | `491fe4c8123241e61a3d6392bfeea618604f9bb2717df035d095e0506ca550da` |

上述四个文件均与 `qualification/cell-recovery.v1/result.json` 保存的对应源码正文相同。没有对同时开发中的整个工作区作哈希污染判定。

阅读了 Codex 适配规则、`show-me-your-work`、原始 `inputs/BENCHMARK_RESEARCH_PLAN.original.md` 的业务/隔离/计分约束、round2 的 rationale/judgment、`planning/gate-contract.v12.json`。没有重复旧 140 slots 全审，也没有重开论文门槛或主截止时长。v11 的基础 100/100、transfer 20/20 与 observe 1/20 作为已裁决背景保留，不将它们转写成新模式通过。

## F1：旧路径接缝附近的新 work bank 不连续（P1，待修）

位置：`planning.py:56–58,217–222`；相关准入为 `execution.py:245–269,338–366`。

`reference_curvature` 使用 gate 前后各 0.025 s 的 course 差。当窗口跨过旧 straight→arc 接缝，估计得到两侧平均曲率；旧 `at(gate)` 的 bank 已经进入另一侧（或处于旧采样插值），新 `ramped_work` 却按这个平均曲率重新计算首行 bank。位置和 course 连续并不能消除这一命令跳变。

| 输入 | 旧 bank | 新 work 首行 bank | 跳变量 | `plan` |
| --- | ---: | ---: | ---: | --- |
| calm，首个接缝 gate=27.777777777778315 s | −22.432991° | −11.663294° | +10.769697° | 接受 |
| calm，实际 120 Hz gate=27.775 s | −20.189692° | −10.397334° | +9.792358° | 接受并提交，revision=1 |
| wind=(−2,0)，gate=31.25 s | −18.066210° | −9.263314° | +8.802896° | 接受 |

非零风在 gate±0.01 s 同样出现约 5.204°/8.864° 的 bank 差。现有 ramp 测试只在直线或恒曲率内部比较首行，未覆盖旧采样接缝。

最小复现（在仓库根目录，用 `.venv/bin/python`，只构造纯几何）：

```python
import math
import numpy as np
from envs.inspection.planning import initial_program, reference_curvature, ramped_work

p = initial_program(-100., (0., 0.))
j = np.flatnonzero(abs(p.prefix.samples[:, 5]) > 1e-9)[0]
t = p.prefix.samples[j, 0]
old = p.at(t)
k = reference_curvature(p, t)
new = ramped_work(old, k, k, 8., np.zeros(2)).at(0)
print(t, math.degrees(old[4]), math.degrees(new[4]))
```

另已构造四机 `PlanRequest`、本机位于旧 nominal 的完整 `OwnState`、合法 token 与 ready proposal，调用 `advance(27.775, ...)`；结果实际进入 `program_committed`，证明不是被最终 gate 自动拦下的候选。需要从旧 gate 已承诺命令求一致入口，并覆盖接缝两侧、非零风；修复名义命令连续性仍不等于获得完整物理管证书。

## F2：捕获门遗漏世界速度方向/course（P1，待修）

位置：`execution.py:257–268`；约束来源为 round2 judgment 的“完整自身入口域”。

新增的 `ground_speed_error_m_s` 只比较 `norm(own.world_velocity[:2])` 与名义速度；`heading_error_degrees` 比较飞机 yaw 与期望 air-heading。两项均不能替代世界地速的方向检查。即使本机正在横向运动，大小正确、yaw 正确，门仍可通过。

确定反例：calm、初始直线 `t=2 s`，本机 XYZ 恰等于 nominal，roll/pitch/rates/integrals 为零、yaw=0、世界速度 `(0,18,0)`、独立空速 18 m/s，并将 navigation 的机体速度一致地设成北向 18 m/s。实际 course=90°、nominal course=0°，`_capture(1,2,nav)` 返回 `True`；记录的 XY、heading、speed、bank 误差全部为零。

该输入不宣称是已经测得的正常飞行初态；它证明目前声明的完整入口域没有实施必要的方向条件。需要显式计算世界速度的 course（并处理低水平速度），对拒绝和合法有风 crab 状态分别验证。可接受阈值应与后续资格绑定，不能仅凭本次几何反例宣称全部状态域安全。

## F3：恢复内部与提前 exit 仍保留无 ramp 的曲率接缝（P1，待修）

位置：`planning.py:27–38,194–207`；`TimedPath.concatenate` 与 `RouteProgram` 只检查位置/course 的连续性。

work 的最后曲率确实先经 2 s runout 变成零，再进入恢复首段直线。但是 `recovery` 内部仍是 40 m 直线、CSC、40 m 直线、R80 tail：直线→首弧、弧→中间直线、中间直线→末弧、末弧→直线、直线→tail 中存在未平滑的 bank 变化。提前 `exit` 甚至不先从旧 work 的曲率 ramp 到零，直接进入恢复首段直线。`survey` 的旧几何接入也未统一平滑。

纯复现：四机 calm `initial_program`，选择 drone 1、`now=29.5,commit_at=30`，合法四 token、kind=`exit` 的请求得到被接受的 program；旧 gate bank=−22.432991°，新 gate bank=0°。任何当前恢复的末尾 40 m 直线→calm R80 tail 又是约 0°→−22.432991°。复用自 v11 的机制不属于新中心差分缺陷，但属于本次必须完成的恢复修复范围。

round2 judgment“实施接口”明确要求：“新增 ramp 必须用于所有实际发生的 bank/速度/高度接缝，包括 work→recovery→tail；复用 recovery 函数不自动继承新入口状态资格。”需要平滑实际空间/时序几何并重新检查风下速度、bank、围栏和全 fleet；只平滑 feedforward 数字而保留空间折率不能满足这一要求。

## 完整状态与恢复资格的证据边界

`mission.py:123–132` 的来源正确：位置/姿态/角速率从自己的导航读取，世界速度用 PyBullet `getBaseVelocity`，独立空速由该世界速度减世界风求范数，积分器只读。没有重用此前 A 候选“机体系速度减世界风”的混系做法。`distance_m` 在逐物理步以真实 XYZ 位移累计，remaining range 来自公开预算减实际已飞距离。

`plan` 用 `(combined.duration-now+20)*27+1000` 检查请求时的本机实际剩余航程，包含未来 gate 前的时间与完整返回/20 s dwell；27 m/s 是 25 m/s 空速与 2 m/s 风的保守地速上限。它不是把真实已飞距离误当名义路径长度。全机资源值被记录，本次选择机的完整返回成本接受检查；其他机既有完整程序仍保留。

然而成功 `program_committed` 不保存 `_capture` 的 metrics 或当前完整 `OwnState`；只有拒绝时保存 metrics。request snapshot 在 gate 前 0.5 s，不能独立重建成功 gate 的完整状态/PID。work→runout→return 和 return→tail 的实际入口也没有专门完整状态日志。已有逐 240 Hz XY/Z 记录能复算误差与区间计数，但没有完整姿态、速度、角速率和控制器记忆，**不能称为完整物理状态重放或完整恢复入口域证明**。下一版应记录这些实际入口，物理资格继续按原 6 m/2 m 与 hard 界失败即失败。

## 全 fleet、成本与因果/业务检查

- `paths.py:292–313` 仍检查四机全部有限前缀和一个共同周期 tail；保持每条插值轨迹的 27 m/s Lipschitz 前提、间距减 `27*dt` 与围栏加 `13.5*dt`。`plan` 实际检查连续 lower≥27 m、upper≤544 m；在已资格的 6 m 水平管前提下，分别对应 15 m 实际间距与 ±550 m 围栏。这里的管前提尚不能由新名义几何自行证明。
- 新 `xy_at` 与标量 `at(t)[:2]` 已在四条非零风 program 上分别比对 1004 个时间点，包含 prefix/tail 边界与多周期时刻，全部 `allclose(atol=1e-10)`。向量化没有把历史、返回或其他机占用删掉。共同周期论证只针对固定时序名义参考，不声称无限期物理相位管。
- 纯 planner 一组四机短观察请求均接受，各约 0.019 s；请求 pickle 约 1,793,479 bytes，回复约 80,277 bytes。该数据仅为此有限输入的隔离成本观察，不是最大时延或 1×证明。消息上限为回复 1.1 MB；当前 `Queue(maxsize=1)`、非阻塞 tick polling、wall timeout 与既有续飞策略仍在。没有新跑异步/1×时钟、四机同时 observe/交接、慢/晚/死 planner 的完整资格，不能声称这些通过。
- 目标预测只读取消费者本机已见/合法交付样本。covariance/mode/innovation 被保留，误差增长明确标成 heuristic；未导入目标实体、生成器未来、grader 或 seed。预测只排名候选，未新增预测 measurement 或服务得分。约束年龄/测量可用时刻仍检查。
- 删除 mandatory inspect delay 与 `planned_service_from` 对获取续约的阻挡，保留实际 qualified frame 才获取/计 service。stale camera 返回实际 wide sensing；没有让 LocalTrack clamp、导航预测或 6 s 上传宽限补足真实缺帧。
- `control.py`、`flight.py`、`guidance.py`、`protocol.py`、`reference.py`、`scoring.py` 与 v11 冻结源正文相同。`scenario.py` 仅更新公开 schema 版本、trial=0 和导航说明。`mission.py` 增量未改曝光时 report route、四机曝光后统一后置条件、报告名额/固定关联或评分分母。未发现本次增量偷换隐藏 appearance 起算的 180 s、60 个 3 s bins 或 6 s 实际到达宽限。

## 实际执行的检查与已有物理证据

本次只执行 AST 解析、纯几何/状态反例、向量化等价检查、已有 JSON/压缩 trace 解析，没有运行新的物理 episode、600 s、API、训练或正式实验。没有重复主 Agent 报告的 125 项测试；这些测试通过不是本独立审查新执行的结论。

### `cell-recovery.v1`

结果文件 SHA-256：`e7fc0e7a58a9d3220a152a8a644532eea905166b4769caf025cdf09dd1e0ada3`。

使用标准库 `base64/zlib/hashlib/struct.iter_unpack('<17d')` 独立解码全部 36000 行，不依赖 `eval/inspection_tracking.py` 的实现；也调用该 helper 比较结果。检查原始字节 hash、行宽、每行 `t=j/240`、所有值有限、每机 XY≥0、实际 Z 与记录区间关系并重新计数。

| 检查 | 实际结果 |
| --- | --- |
| 行数/时间/有限值 | 36000；错误 0 |
| raw trace SHA-256 | `85116d36322549249ae6580c5a52bc2ec814b40115f1177f05878a061559b2ea` |
| 最大 XY 误差 | 2.5327691060757247 m |
| XY>6 m 物理步 | 0 |
| Z 超记录区间物理步 | 0 |
| 所有 Z 相对 96 m 最大误差 | 1.4653173041850067 m |
| 请求快照 | 15 个 request 均有四机同刻全部 7 类字段及四机 remaining range |
| 结果保存的 hard 审计 | 36000/36000，无 violations |
| 结果保存的实际退出 dwell | 134.375、71.267、134.375、134.375 s |

标准库重计与保存 summary 一致。该 trace 没有全部原始 physical state，故没有独立重算 hard 空速/bank/separation；这里只准确引用该结果的 hard 审计输出。真实 inspect/report/observe 受理、18 s controlled permanent planner stop 和有限程序继续飞完返回的结论均保留。它不是完整 180 s 原窗 D、真实异步时延/1×或广泛入口域资格。

### 结束中的新小 pilot（只解析，不重跑）

审查后段读取 `qualification/executor-pilot.v7`。manifest 显示 `full_grid=false,qualified=false`，sources/contract/dependencies unchanged 为真；本报告没有重验其完整依赖 closure。

| root | passed | delivered D | XY peak | XY breach steps | hard violations |
| --- | --- | ---: | ---: | ---: | --- |
| 100 | true | 0.8388888889 | 3.004652 m | 0 | 无 |
| 101 | false | 0.6944444444 | 8.301632 m | 2378 | 无 |

两条均 360 s、86400 个物理/tracking 审计。root 101 峰值 witness 为 t=230.3166666667、drone 2；D 和水平管均未达到本次门槛。这里没有独立重建该峰值时的完整控制状态或对根因作物理归因。结果应原样保留，不能用单个 150 s fixture 或 root 100 pass 隐去；也不能把两根小 pilot 当完整 100..119 矩阵。

## 审查记录与交接

受本子任务“仅允许写本报告”的边界约束，未另建或修改主工作流 TSV。以下记录本次关键决定；证据均来自本报告所列源码、既有文件或本次有界工具输出。

| 检查点 | 决定 | 原因 | 证据 | 结果 |
| --- | --- | --- | --- | --- |
| 入口反例 | 将审查集中在新 ramp/入口和复用 recovery，而不重复旧全矩阵 | 三者是 round2 要补的因果接缝 | 四个源码 diff；round2 judgment | F1、F2 确定复现并先通知主 Agent |
| 数据检查 | 独立解码全部单 fixture 误差记录并检查向量化等价 | 分开可复算数据与尚无证明的状态域 | cell-recovery.v1；stdlib decode；4×1004 parity | 36000 行全部可复算；未扩张物理主张 |
| 恢复检查 | 将恢复无 ramp 单列为未闭合项 | round2 明确覆盖 recovery/tail，不能让旧实现替代要求 | planning.py:27–38,194–207；exit t30 反例 | F3 已通知，主 Agent 确认修复 |
| 报告冻结 | 冻结本报告，不等待修复覆写旧版本事实 | 当前 pilot 与缺陷必须可追溯 | 本报告源码 hashes；executor-pilot.v7 | 未来修复需单独 bounded followup |

与本次工具记录逐项核对，未把推测的物理风险写成实测事故，未把他人的测试运行说成本次执行，未声称 G1/G2–G4、正式实验或人工领域审查完成。本 fresh review 为同模型家族独立上下文；没有可用不同模型家族的审查执行，未声称跨模型复核。

## Attention

reviewed by GPT-6 family（独立新上下文，同家族）

- 优先处理 F1–F3；不以局部通过回填已失败版本。
- 现有请求快照和 scalar tracking trace 不足以独立重建成功 gate/恢复入口的完整状态与 PID。
- 4 机并发服务、未知转向/失测、事务边界及真实异步 1×仍待执行相应证据；本报告没有把它们判为已通过。
