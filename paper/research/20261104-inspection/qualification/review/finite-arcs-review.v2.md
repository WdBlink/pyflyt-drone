# 有限固定弧执行器修复后有界复核 v2

证据冻结时间：2026-11-05 10:22:37 UTC（工具时钟核验）。复核者：独立上下文 `/root/finite_arc_review`。本报告保留 v1，不修改 production、契约、旧结果或研究状态。只执行纯 Python 几何/状态检查和已完成文件解析，没有新物理 episode、API、600 s 或全套测试运行。

**结论：v1 的 F1/F2 反例已修复；F3 在本次可初始化的输入中已经实质改变空间/时序几何，恢复接缝不再仅靠 bank 数字平滑。但新增平滑器在允许风向下生成重复时间戳，导致初始化抛错（F4，P1 阻断）。当前版本仍不能放行完整风包线或 G1/G3。**

主 Agent 已收到 F4 的确定输入、行号与重复时间值，并确认保留正在运行的冻结证据，之后修复。本报告不预先批准未来改版。

## 审查绑定

下列源码均与已完成 `qualification/cell-recovery.v2/result.json` 保存的对应正文相同。本复核没有对并发开发中的其他文件作“源码污染”判定。

| 文件 | SHA-256 |
| --- | --- |
| `envs/inspection/planning.py` | `fc1530885269bd6e7a7e828736d12c37ca46dcbf9fdf211ea60c9c5fe9c22edc` |
| `envs/inspection/paths.py` | `0958cae1aa74be525f281c1f586a022c2acaaecd67405af78631ce7b01010458` |
| `envs/inspection/execution.py` | `72611b32fb01e3679fa616f2fd51969508ab4d263718a5810f70ea37ab4294e9` |
| `envs/inspection/mission.py` | `491fe4c8123241e61a3d6392bfeea618604f9bb2717df035d095e0506ca550da` |

范围包括以上修改、相关新增测试正文、`eval/inspection_tracking.py`、qualifier 的 seam 判定以及已完成的短恢复探针。主 Agent 报告的 134 项通过没有在此重跑，不能写成本次独立执行结果。

## F4：平滑窗口端点产生重复时间，合法风向初始化失败（P1，未修）

位置：`planning.py:109–123` 的窗口采样与 `paths.py:198–216` 的拼接。确定复现：

```python
from envs.inspection.planning import initial_program
initial_program(-300., (0., -2.))
```

或构造四机 `FlightExecutor(HOMES,(96.,)*4,(0.,-2.),start_planner=False)`。均在进入物理仿真前抛出：

```text
ValueError: bounded finite strictly timed samples required
initial_program -> smooth_own_joins -> TimedPath.concatenate -> TimedPath.from_samples
```

通过内存包装 `TimedPath.from_samples` 观察失败输入，没有修改源码。最终待拼接数组为 `(8423,6)`，零基索引 2566 与 2567 行时间都等于 `128.13285627347614`，`min(diff(t))=0`。两行 XY 仅差约 `2.5e-13 m`，bank 从约 `4.76e-15 rad` 到 0。不是实际路径倒飞，而是数值端点重复。

根因与 v1 已提过的 slice 边界类别一致：`np.r_[np.arange(0.,seconds,.05),seconds]` 可能保留与精确 endpoint 仅差浮点舍入量的内点；局部严格递增的极小间隔，加上全局 stamp 后会变成相同时间。需要先排除 endpoint 附近内点，保留精确 endpoint，再验证全局时间严格递增及速度斜率。不要只删除报错或忽略零间隔。

另一个允许方向 `wind=(-sqrt(2),-sqrt(2))` 在 `home_y=-100` 的初始化也失败。九个风向中七个方向完成了本次四机初始化，两个方向出现上述类错误。该失败发生在几何构造阶段，应保留为初始化/基础设施缺陷，不能误标普通物理不合格，更不能把未启动的 slot 当作通过。

## v1 三项修复复验

### F1：按旧 gate bank 求入口曲率，原反例关闭

`planning.py:62–66` 从旧 gate 的 nominal bank 逆解 `course_kinematics` 的线性曲率关系，避免跨 seam 的中心差分。

独立构造有旧 straight→R80 kink 的 raw program，避免新初始路线平滑后掩盖原反例；对 calm、±X 2 m/s、±Y 2 m/s 以及东北 2 m/s 六个风向，分别检查 gate−0.01 s、gate、gate+0.01 s 与最近的 120 Hz gate。24 个输入的新 work 首行位置/course 都与旧 gate 相同，bank 最大差 `1.0547118733938987e-15 rad`。原 v1 约 9–11° 的 bank 跳变没有再现。

首行 nominal ground speed 仍按常空速 18 m/s 的风下运动学重新求解，未逐字保留插值 sample 的速度。上述 raw 反例最大差为 `2.7777e-5 m/s`；另外一个已平滑旧路线中的 exit gate 观测到 `0.011516 m/s` 差。本次没有把这些小数值差认定为已发生的物理事故，也不把 bank 匹配夸大成任意切点的严格 C2 物理证书。

### F2：世界 course 已进入门，合法 crab 没有误拒

`execution.py:262–273` 使用世界速度计算 course，要求与 nominal course 相差≤10°。重建 v1 位置/yaw/速度大小正确而实际北向运动的输入，calm 与 +Y 风均被拒绝，记录 course error=90°。

另构造实际世界地速沿 nominal、机体 yaw 按 `air_heading_for_course` 正确 crab、独立空速 18 m/s 的状态：calm 与 +Y 2 m/s 均通过，heading/course error 都为零。−Y 状态门验证被 F4 的 executor 初始化错误阻断，因此不声称三风向 crab 都完成。低水平速度另有地速大小约束，本次没有扩展或另定阈值。

### F3：空间几何、周期尾环与退出首段均已实质修复，风包线仍受 F4 阻断

阅读确认 `smooth_own_joins` 用带端点位置/速度/加速度条件的有限 quintic 窗口，重新求 XY、course、ground speed 与 bank，并在窗口内检查空速 16..22 m/s、nominal bank≤30°。没有重置真实状态/PID，也没有恢复目标圆 acquire。新 exit 先从旧 bank 做 2 s rampdown；返回的最初 2 s home circle 纳入 finite prefix，周期 tail 旋转相位，保留完整原周期。

通过的七个风向均检查四机初始 program、提前 exit 与 survey 请求，检查完整数组有限、时间严格递增、样本数≤16000、全部 prefix 插值斜率、tail 接缝以及完整 fleet 计算：

| 有界几何检查 | 实际结果（通过方向） |
| --- | --- |
| 四机同风 tail 周期 | 每个方向内部四机相同，最大差<1e-9 s |
| tail 重复一周期 XY 误差 | ≤`1.086e-12 m` |
| `xy_at` 与逐次标量 `at` | 最大坐标差≤`5.685e-14 m` |
| prefix→tail 位置/course/speed/bank | 端点匹配（course 按 wrap 比较），bank 差 0 |
| 最大 prefix 插值速度斜率 | `20.026199 m/s`，小于 27 m/s 前提 |
| 最大相邻样本 bank 变化 | `1.554346°`，nominal bank 均≤30° |
| 四机初始完整 fleet 连续间距 lower | 约 `197.3 m` |
| 四机初始完整 fleet 围栏 upper | 约 `381.404 m` |
| exit gate 的 bank 匹配 | 数值精度内匹配，最大约`1.28e-15 rad` |

calm、−X 与 +Y 下还对单个直线→弧窗口做独立 XY 有限差分诊断。相对旧几何，窗口位置真实变化约 `0.0345..0.0453 m`；输出速度与 XY 差分速度的最大差约 `0.00150..0.00194 m/s`，由 XY 速度/加速度差分推导的 bank 与保存 bank 差约 `0.203..0.226°`。这支持“实际空间/时序几何已经改变”，排除了只改 feedforward 字段；有限差分和离散插值本身不能证明连续解析 C2 或真实飞机可捕获。F4 修复后需复验失败方向，不能把七个方向推及整个风包线。

## 导航预览与安全边界

32 s preview 仍仅使用 consumer-scoped、因果可用的 ObservedTrack 进行候选排序；实际承诺 work 仍≤8 s，后接 2 s runout、完整返回和 tail。preview 没有制造 evidence、推进服务时钟或改变 D 的原始窗口。该更长预览增加了纯 planner 计算量，本复核没有执行新的 1×或并发负载，不能据此给时延背书。

共享 nominal bank 上限 30°、实际 hard bank 45°、6 m XY/2 m 高度管、连续名义间距27 m、围栏544 m未放宽。新几何后依旧重算完整四机 finite prefix+共同周期 tail。拒绝候选仍属于 bounded-library refusal，不能宣称全局物理不可行。

## 已完成 `cell-recovery.v2` 的独立解析

结果文件 SHA-256：`c11dd6c22c529724b03ebce5f9c5bbdc35ba717f2a6d2f432c70ee62fc8152f2`。文件已完成，`sources_unchanged=true`，四个审查文件均与保存源码相同；没有读取正在写入的 partial 作为最终结果。

再次用标准库 base64/zlib/hashlib/struct 独立解码 36000 行 `<17d`，核对每行时间 `j/240`、finite 值和 raw hash，重计 XY>6 m 与实际 Z 超区间：

| 项目 | 结果 |
| --- | --- |
| raw trace SHA-256 | `39ce9b0ac58c3ef7d43e24b5ad8ffe5b83d29ad807967df9430005f7cf968bba` |
| 行数/时间/finite | 36000；无错误 |
| 最大 XY | `2.5327691060756683 m` |
| XY 越管/高度区间越界步 | 0 / 0 |
| 成功 commit 状态/metrics | 14 条，全部有同刻完整 OwnState 和积分器 |
| physical seam | 9 条，同刻 OwnState 完整，实际记录时间距 scheduled time<1/120 s |
| seam capture 复算 | 逐项从保存 metrics/OwnState 重算全部门条件，9/9 true；与 counter=0 failures 一致 |
| 保存的 hard 审计 | 36000/36000，无 violations |
| 保存的四机退出 dwell | 132.375、68.417、132.375、132.375 s |

九个 seam 包含四机 recovery_start、四机 tail_start，以及选择机 work_end。选择机的 work_end=25 s、recovery_start=27 s、tail_start≈81.5814 s，最后记录在 81.5833 s，验证了跨 120 Hz tick 的记录方式。成功 gate 与 fallback 的完整状态/PID证据缺口在这个实际 fixture 中已补齐。quintic 窗口内部每个小采样点不是单独命名 seam；本报告没有称每个曲率变化点都保存了完整物理状态。

这仍是 150 s、18 s 停止受控 planner 的一个 fixture。raw error trace 不包含全部动力学状态，不能独立重算 hard bank/空速/间距；这些字段只准确引用保存的审计输出。它不覆盖完整180 s原窗任务、多机同时服务/交接、整个风向域、完整 G1 或真实异步 G3。

## 交接与 Attention

reviewed by GPT-6 family（同家族独立上下文，未声称跨模型复核）

- 原 F1/F2 已用原类别反例复验关闭；F3 在通过输入中有实际几何修复与有限差分支持。
- **F4 必须先修复并有界复验合法风向初始化；当前完整风包线不可放行。**
- 只写本 v2，保留 v1 和旧失败。没有重跑主 Agent 的134项或产生新物理数据；没有授予 G1/G3、正式实验或人工领域审查完成状态。
