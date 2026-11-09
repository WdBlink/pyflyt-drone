# Pilot v2 两次 clock failure：独立源码与时间线审查

结论：这两次实际 attempt 都必须保留为 clock invalid；完成 600 s / 144000 物理步、最终 wall≈600 s 和较小 p95 均不能覆盖中途超过 0.1 s 的失败。现有记录证明落后是局部突发并随后追平，但不足以确定唯一的 CPU、调度、传感、IPC 或模型资源争用根因。尤其不能把峰值 tick 的 `sensor_tick=false`、很小的 `previous_publication_work_seconds` 或 12–15 ms 的 `current_step_work_seconds` 当作排除之前积压的证据。下一步应先做预登记、冻结新版本的分段诊断；本审查没有运行第三次尝试、重放、MLX 或 physics，没有修改阈值、重试上限、源、owner 或原始结果，也不授予 G4/main 许可。

审查者为 `/root/control_review`，同模型家族的独立上下文工程复核，不是跨模型或领域人工认证。路径以下相对 repository；`R=paper/research/20261104-inspection`。绑定为 `R/review/pilot-v2-clock-failure-source-audit.v1.binding.json`，SHA256 `9d97b2ee02caa48d564b9c34c147959649cbbeb889eb9ea302cab44cf6d2a6b6`。

## 1. 实际输入与可独立确认的失败

只读取 `logs/inspection-eight-condition-dev-flights.v2/dev-L1-00-B4-M1-2/{attempt-00,attempt-01}` 的 `flight/{run,world}.json`、600 行 `progress.jsonl`、actor 的 request/response/dispatch **时间元数据**以及 `started.json`、`terminal.json`，不分析模型文本优劣或任务分数。读取 `plan.json` 仅确认物理解释器；没有读取真实 owner 全文或重核资源结算。root 提供的 17 项 known-settled、353 calls / 2240091 tokens 不属于本报告独立核验结论。

| 原始值 | attempt-00 | attempt-01 |
|---|---:|---:|
| `run.started_at` UTC（不是 worker epoch） | 03:05:58.427279 | 03:16:22.106375 |
| sim / 物理步 / control Hz | 600 / 144000 / 120 | 600 / 144000 / 120 |
| 最终 wall seconds | 600.003913167 | 600.003694583 |
| 原始 max lag / 峰值 sim t | 1.768018833 / 316.541666667 | 0.352951125 / 531.15 |
| 原始 p95 lag | 0.009619208 | 0.008733292 |
| 原始 >0.1 s control ticks | 1629 | 588 |
| 峰值 `current_step_work_seconds` | 0.015310917 | 0.012125000 |
| 峰值前一 tick tail seconds | 0.000000834 | 0.000001041 |
| 独立从 1 Hz progress 重算的最大 `wall_elapsed-t` | 1.680803042 @317 | 0.271097000 @531 |

`lags` 每次 `mission.step()` 采一次，即 120 Hz、预期 72000 个 **control tick**；144000 是 240 Hz 物理积分分母，不能用于解释 `clock_lag_over_0p1_seconds_ticks`。原始 120 Hz lag 序列没有持久保存，因此本审查不能从结果独立重算精确 max/p95/1629/588；这些值已与原 world 和 terminal 交叉核对。独立重算的 1 Hz 墙钟残差也明显超过 0.1 s，支持两次时钟失败，不依赖 `passed` 或 terminal bool。

两个 world SHA256 分别是 `0c0cddc3cdff7a88985bd022cebce618fe418d7d76b3f348f508bcb1d2800d91`、`44cd9de740a8f46f7ef1bcbf3bba19b8dcda4041b4700bde16f46111112434f2`；两个 run 为 `a7760ec90dea6a85d35bb21821907b946389f41c78aeca28554f8a82bfc6a142`、`77aa41b30352e0899733a4f3cb2e183588db4c3b1250fe6680f2c0f9b635a3e5`。完整输入路径、字节数、SHA 和紧凑时间窗见 binding。

## 2. 计时边界：本轮最重要的确定问题

**P2（诊断可观测性缺口，不是已证明的唯一性能根因）：现有最大值 witness 不能归因累计 lag。** 证据为 `eval/run_inspection_reference.py:54–98`，尤其 `:61`、`:64`、`:81`、`:85` 和 `:98`。最小补足是保存下节所列分段时间，不能靠改名或只多写一个峰值 bool 解决。

| 阶段 | 实际位置与计时含义 |
|---|---|
| warmup / 新 mission | `run_inspection_reference.py:40–54`：旧 warmup 120 step 并 close、新 mission 构造完成后才设置 `epoch=time.monotonic()`。没有发现中途重设 epoch。 |
| 初态 publication | `:57`：在 epoch **之后**、首个 tick 之前。它会贡献初始墙钟，但不在 `current_step_work` / `previous_tail` 内。 |
| sleep / 唤醒 | `:60–64`：先按下一 tick 绝对 deadline sleep，返回后才开始 work 计时。实际 sleep 超时、唤醒后争不到 CPU/GIL 等均不在 `current_step_work` 中。落后时 remaining≤0，不再 sleep，允许追平；不会消除历史 max。 |
| RPC、executor、physics、sensor、共享进度 | `:65–80`：最多 16 条 RPC 的 get/tool/ACK、`mission.step()` 和共享进度锁全部在 work span。`mission.step` 内 2 Hz sample/deliver/public-cache 也在此 span（`mission.py:308–320`）。 |
| lag / 最大 witness | `:81–88`：在 mission step 和共享进度后采 `wall-epoch-sim`。峰值 work 只是这一 tick 的墙钟 span，包含其间被调度出去的时间，并非纯 CPU 时间。 |
| publication / 日志 tail | `:89–98`：2 Hz `public_state` + `updates.put`，1 Hz progress 构造、write/flush 在当次 lag 采样之后；其落后可在下一 tick 体现。`previous_tail` 是**紧邻的上一 control tick**，不是上一次真正 2 Hz publication 的总耗时。 |
| 异步 feeder | `multiprocessing.Queue.put` 本地 append 后返回，后台 feeder 另做 pickle/send；这些耗时没有被分别记录，也不等于 caller 的 `put` span。 |

以 Δt=1/120 s 近似反推，峰值 tick **进入 work 时已经落后** `lag + Δt - current_work`：attempt-00 约 1.761041 s，attempt-01 约 0.349159 s（witness 采样间有很小开销，不当作精确边界值）。所以 15.31/12.125 ms 不是把健康时钟突然拖到全部峰值的那一个工作量。两次 `sensor_tick=false` 只描述峰值这一 tick；316.5 / 531.0 的 sensor/publication 在此前，无法据此排除它们或更早阶段的贡献。

`run.started_at` 是 parent setup manifest 的 UTC（`run_inspection_llm_development.py:89–103`），随后才有 service load、process.start 和 worker warmup（`:124–130`）。没有保存它与 worker monotonic epoch 的精确 offset，不能用该 UTC 加 sim time 当作精确 OS/GPU 事件时间。actor 的共享四元组是 `[sim_t, drone0_actual_x, y, z]`（worker `:78–80`；parent `:121–123`），不是四架飞机或 wall timestamp。

## 3. 两段实际时间线，及能排除到哪里

**attempt-00：不是峰值附近新 RPC / route proposal 的直接同步处理。** 1 Hz 残差从 t313 的 0.021397 s 上升到 t314 的 0.581064、t315 的 0.654966、t316 的 1.422460、t317 的 1.680803；t320 又为 1.579501，至 t325 回到 0.013146。1 Hz >0.1 区间为 98、314–324、441、443；这不是从开场一直单调积累到终场的漂移。

峰值附近最近工具是 t287.358333 的 survey batch，下一条是 t345.141667 的 report；最近 executor 事件是 t289.75 的 serial9 commit，下一条 planning request 在 t497.216667。t309–328 没有 tool、plan response/request、commit 或 seam 事件。实际 request13 从共享进度 287.516667 等到 345.141667，模型 service wall57.168340 s / parent end-to-end57.622363 s，注入 wait=0。它与 lag 窗口重叠，**只证明并发存在，不证明模型争用是原因**。executor 没有记录 planner process failure/timeout；同时，记录本身不提供该进程当时的 CPU/调度轨迹。

**attempt-01：有持续后台规划负载，但峰值之前没有新的 RPC 或 proposal 消费。** t530 残差 0.008455 → t531 0.271097 → t532 0.126057 → t533 0.044560 → t534 0.007411。1 Hz >0.1 区间为 281、426–427、457、531–532。serial89 在 t530.208333 发出，t530.716667 capture expired；max witness 为 t531.15，response 到 t531.2 才被 worker `_poll` 读取（planner compute1.303632 s、fresh=false、program absent）。serial90 在531.2发出。t533.275 的 8 条 cancel/inspect RPC 在峰值之后。实际 request15 的共享进度范围为477.016667→533.266667，模型 service55.764427 s / parent56.256221 s，注入 wait=0；同样只构成时间重叠。

全次 attempt-00 有27 request/27 response、3 response fresh=false、2 compute>.5 s；attempt-01 有110 request/109 response、66 fresh=false、65 compute>.5 s，含大量 `continuous concatenation required` / `bounded finite strictly timed samples required` 拒绝。后者是可核实的额外规划负载和捕获过期现象，可能放大资源竞争；不能把 planner 的1.3 s compute直接记为 worker 被阻塞1.3 s，也不能用它解释 attempt-00 的主峰。

## 4. 相关源码机制：实证边界与待区分的可能性

- **route solver 已在独立进程。** `execution.py:36–52,95–100` 才调用 `plan(request)`；`mission.tool` 主要校验/登记 job（`mission.py:333–405`），没有同步执行该 solver。`advance` 每 tick `_poll` 并可 enqueue 下一 request（`execution.py:320–325,404–419`）。不能声称整个 solver 在 physics work span 内运行。
- **但 IPC 并未获得 wall-time 非阻塞保证。** `execution.py:277–308` 的 `results.get_nowait()` 会接收/解 pickle，随后还显式 `pickle.loads` 和 `program_hash`；后者复制完整 prefix/tail array（`:27–33`）。实际 `.venv/bin/python` 是 CPython3.11.15：`multiprocessing/queues.py:98–132` 在 poll 发现可读后调用 `_recv_bytes()`；`connection.py:429–437`、`:390–406` 会读满 frame，`get_nowait` 并不为部分到达的 frame 提供严格的 wall-time 上限。源码检查成立，但本次没有记录 receive/block/span；attempt-00 峰值窗口无 response，attempt-01 的serial89 response又在峰值后，故它不能成为这两个主峰的已证共同根因。
- **request/publication 的后台序列化与进程竞争可在主 work 之外发生。** request 含四个完整 programs（`execution.py:207–209`），`history_prefix` 在规划子进程可采样到当前时间（`planning.py:64–68,331–332`）。本地 feeder 会序列化这些对象；planner 的 `planner_wall_seconds` 在显式 `pickle.dumps` 之前形成（`execution.py:47`），不含 request transport、response pickle/send/queue delay。不能拿它做端到端预算。实际子进程及 feeder 的 CPU/等待时间均未保存。
- **每半秒的确定工作量应单独计时。** `_sample` 最多对4架逐一检查所有 coverage cells、实际 visibility ray、entity、随机噪声与关联（`mission.py:163–259`）；本例有1174 required cells。`_deliver` 扫 pending（`:265–286`）；`_refresh_public` 遍历所有 jobs（`:294–306`）。`public_state(since=...)` 虽然只输出增量 observations，仍扫描全部 delivered 并 deep-copy card/telemetry/jobs（`:322–328`）；inspect 工具也扫描 delivered（`:391–399`）。这是随着历史/工作模式变化的真实路径，但没有本次分段耗时，不能据数量断言哪项达到了0.3/1.7秒。
- **共享进度锁与发布队列没有单独等待记录。** parent 持 progress 锁仅 copy4数，未见故意把模型等待放在锁内（`run_inspection_llm_development.py:121–123`）；但 OS 在持锁期间暂停线程的影响没有测量。commands/updates 为无显式 maxsize 的 Queue（`:106`），因此不能直接把模型慢回复说成 `updates.put` 因固定容量满而阻塞；异步 feeder、内存、GIL/CPU压力仍需实测。

两次记录的 BLAS/OMP/MKL/VECLIB 线程环境均为1；这只是实际设置记录，不证明没有其他线程、OS调度、内存压力或 GPU 共享资源竞争。`mission.audit()` 的压缩240 Hz tracking、最终打分/大JSON落盘在循环结束后，不是316/531 s时已证的同步阶段。raw tracking error 不是全物理状态或 wall-clock 重放证据。

## 5. 下一次 bounded engineering diagnostic 的最小区分能力

应保持原600 s/1× clock定义、0.1门槛、原slot失败与二次上限，并把新诊断记成新工程版本。以下是可观测性触点，不是平行 scheduler、策略改进或加一次原slot免费重试：

1. 在 worker 每 control tick 仅写固定大小内存 ring-buffer：sim t、scheduled deadline、sleep requested/actual、wake overshoot、work入口 lag、RPC count/total span、executor.advance、physics、sensor/deliver/cache、shared-progress lock/writer、work后 lag、实际 publication/log tail。保留阈值触发前后窗口及 top-K spans，结束后落盘，避免逐tick JSON/flush 自己制造新负载。特别区分“上一control tick tail”与“上一次真实publication tail”。
2. 对 `executor._poll` 记录 serial、get/frame receive、外层/内层反序列化、program hash span与payload字节数；request enqueue/feeder serialize与planner compute/response pickle/send分别有单调时刻。只有 get span大才能支持 IPC 阻塞；只有 compute跨长窗口而 worker stage平稳，才支持后台负载相关而非同步求解。不要依赖不可靠的 `qsize()` 当因果证据。
3. 同一 worker span同时取 wall 与 thread/process CPU时间，低频采集各相关进程CPU/RSS及平台压力：wake overshoot大而各CPU span小，偏向调度/休眠/外部停顿；sensor或publication wall+CPU大，指向对应同步工作；wall大但CPU小，指向锁/IPC/被调度出去。process CPU还包含feeder线程，须与主线程CPU区别。仅凭两次都在模型请求内无法区分GPU、CPU或内存原因。
4. 给 worker epoch 一个可关联的UTC+monotonic同步锚，并给 parent/service/规划阶段保留对应 wall事件ID。actor已有sim进度足以做粗关联，不能替代共同时间轴。诊断版初态 publication 也要纳入记录，保持原epoch，不以重设epoch掩盖延迟。

本轮只做源读取、stdlib introspection和离线JSON计数/关联。没有长测试或新CPU负载实验，没有重新认证61依赖、SDK108/48失败分母、模型账本或科学性能；这些由各自审查/入口负责。上述归因缺口关闭前，不能把“主因已确定/修复已有效”写进G4证据；原两次clock invalid保持不变。

## 6. 源码字节绑定与复算方法

两份 run 的 embedded source 文本 UTF-8 SHA、recorded SHA 与本次读取的实际文件字节逐一相同：

| 文件 | SHA256 |
|---|---|
| `eval/run_inspection_reference.py` | `4c280adab488f9c81d68265dd9930d9775dbb6372b36d1d776d9350f9f36f3ca` |
| `envs/inspection/mission.py` | `715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7` |
| `envs/inspection/execution.py` | `dc301939ea1d55f08f01c69af5561c75f762ac45bdd0d144d677514387425ea4` |
| `envs/inspection/planning.py` | `b87fcdedf3aa26a46560056376fb69f9ff969270149c9cae429b4a9e294df6ab` |
| parent runner（仅上述相邻接口读取） | `c3bed0403ca4a8e7f5d79c0db3df5da0aeb27e46e37423e94474447fbd4da552` |

未来当前工作树改变不回填这两份历史 run。依赖无变化flag在本轮只作为原始值记录，没有扩称独立重验所有61项。复算只需stdlib读取两份 `progress.jsonl`，对每行算 `wall_elapsed-t`，按 t连续且残差>0.1分段；将 execution_audit 的 `now/received_at/at`、tool_calls 的`t`及actor的共享进度sim t投影到上述±窗口。绑定保留所有输入SHA及紧凑窗口，未复制真实owner或完整模型消息。分析临时脚本/中间文件在 `/tmp/inspection-clock-audit-v1/`，不是新实验结果。
