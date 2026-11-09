# Pilot v2 两次 clock failure：只读时间线审计 v1

结论：两次已完成600s积分的 attempt 都保留为 **known operational clock failure，Y=null**，不是策略Y0或合格飞行。已保存数据表明峰值是多轮累计 backlog 的一部分；最大 witness 那一轮的15.31ms/12.125ms工作不能解释已有的1.768s/0.353s迟滞。第一次峰期间无邻近工具/规划事件；第二次峰邻近持续规划活动，但早于相关 response 和新RPC。现有记录不足以认定模型、规划器、传感器、publication、辅助测试或系统调度中的某一项为 cause。不建议第三次策略重试、改变config或升阈值。

审查：2026-11-06，`/root/statistics_design_review`，分离上下文、同模型家族。只读两份实际 development attempt 的 world/run/progress/actor、相关 model transport、terminal 与辅助测试log。未运行模型/物理、formal seed/实例或新测试，未改生产源、live/owner。与 `/root/control_review` 的 clock-loop源码审查分工：本报告独立重算数值，代码计时scope采用其核实结果；不是独立复现完整动力学或正式方法成绩。

## 数据与计时scope

根：`logs/inspection-eight-condition-dev-flights.v2/dev-L1-00-B4-M1-2/attempt-{00,01}`。两次 run.policy 均 B4-M1、相同 sampler_base_seed=1500289886、injected_delay=0；各600s、144,000 physics steps，控制120Hz，对应72,000个control lag采样。`world` 的 legacy non_llm role label不用于识别实际策略，实际model/actor/receipt由run与transport给出。

完整72,000点 lag 序列未保存：world只有聚合、最大witness，progress为600行1Hz；144k tracking trace列是位置/误差等动力学量，没有lag列。因此本报告不能独立重算world p95、精确的>0.1连续区间或完整累计lag积分，只能引用聚合counter并重算1Hz残差。

源码scope：worker先warmup并关闭旧mission，再构造新mission后设 `epoch=time.monotonic()`；`run.started_at` 则来自更早的parent setup，包含后续service load/process startup等，不能当worker零点。共享四元组是 `[sim_time, drone0_x, drone0_y, drone0_z]`，后三项不是wall时间。模型service/parent字段是duration，也不是UTC时间戳。

witness的lag在mission.step及shared-progress写之后、publication tail之前采样；current_step_work含最多16条RPC drain、tool/step及共享锁等当前轮工作，非单独physics kernel时间。上一轮 `previous_publication_work_seconds` 是上一control tick的tail，不保证是最近一次2Hz实际publication；queue.put亦不含feeder线程的异步pickle/pipe完整耗时。因此两个很小的previous-tail值不能排除更早publication或异步队列成本。

原始run内嵌源码与其source_hashes逐项匹配；worker SHA256 `4c280adab488f9c81d68265dd9930d9775dbb6372b36d1d776d9350f9f36f3ca`，parent runner `c3bed0403ca4a8e7f5d79c0db3df5da0aeb27e46e37423e94474447fbd4da552`。其他读取哈希见binding。

## 峰值与累计backlog

| 项目 | attempt-00 | attempt-01 |
|---|---:|---:|
| run.started_at UTC（非worker zero） | 03:05:58.427279 | 03:16:22.106375 |
| 最大lag / sim t | 1.768018833s / 316.541666667 | 0.352951125s / 531.15 |
| witness对应worker wall_elapsed = t+lag | 318.309685500s | 531.502951125s |
| current_step_work | 15.310917ms | 12.125000ms |
| sensor_tick | false | false |
| previous tick tail | 0.834µs | 1.041µs |
| 估算进入当前work已有lag ≈ L+1/120−work | 1.761041250s | 0.349159458s |
| world记录p95 | 0.009619208s | 0.008733292s |
| world记录>0.1 counter / 72k | 1629 / 2.2625% | 588 / 0.8167% |
| counter对应control模拟时间长度 | 13.575s | 4.900s |

最后一行是超界tick数/120，不能称真实连续wall时长。work-entry估算有采样位置的微小差，不是额外计时器；它足以说明主要迟滞在本轮work之前已经存在。终点world wall_seconds分别600.003913s、600.003695s，表示后续追赶，并不能抹去中途clock admission失败。

独立由1Hz `wall_elapsed−t` 重算：

| attempt-00 sim t | 313 | 314 | 315 | 316 | 317 | 320 | 322 | 324 | 325 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 残差s | .021397 | .581064 | .654966 | 1.422460 | 1.680803 | 1.579501 | .771600 | .100567 | .013146 |

00的1Hz>0.1样本连续落在314–324（另98、441、443各一行），共14行；真正120Hz峰值在316.5417而非1Hz最大行317。01的530/531/532/533/534残差为 .008455/.271097/.126057/.044560/.007411s；>0.1样本为281、426–427、457、531–532，共6行。1Hz未见峰值不代表该秒内未超界，不能以这些行数替代1629/588 tick counter。

## 同sim轴的相邻事件

**00：** 最近前一工具是287.358333的assign_survey，后一工具是345.141667的report提交；峰值±10sim秒内无tool。最近前一execution事件为serial9 program_committed@289.75，后一事件为serial10 plan_request@497.216667，二者之间无已记录规划事件。峰值落在model call13在途区间：worker before=287.516667，reply=345.141667，parent compute=57.622363s、service=57.168340s；该reply后才enqueue两条report。因此不能把峰值归给这次reply/RPC处理，也不能仅凭“模型在途”认定推理抢占了physics。

**01：** 峰附近有持续规划request/response/expiry活动。serial89 request@530.208333、capture_expired@530.716667、response@531.2（planner_wall=1.303632s，fresh=false）；峰531.15早于其response约.05sim秒。随后serial90 request@531.2。最近新工具包直到533.275才进入（四cancel+四inspect），晚于峰约2.125sim秒，故它不可能是峰之前那次新增RPC。峰期间model call15仍在途477.016667→533.266667，service55.764427s、parent56.256221s；其后call16才从533.525开始。

planner_wall是规划任务的wall duration，不能当成worker同一段CPU独占时间；pending规划、共享机器上的model service及worker竞争仍只是候选关联，缺少每段thread/process CPU/调度记录。

峰±1sim秒的已记录sensor数据：00无entity sensor_audit/opportunities，背景trial有16条，t=316/316.5/317/317.5；01三类均无记录。无sensor记录不等于sensor函数未运行或耗时为零；`sensor_tick=false`只说明最大witness当前轮的标记，不排除更早sensor/publication阶段。

## UTC与辅助工作：只作相关定位

没有直接保存worker epoch的UTC/monotonic anchor。`started_at+sim` 得到约03:11:15/03:25:13会漏掉startup与lag，不能作为精确峰UTC。若假定progress原文件未retouch、UTC对monotonic偏移稳定、末次写入接近该行wall_elapsed，用 `final progress mtime−final wall_elapsed` 作**误差未知的相关定位估计**：

| 相关定位估计 | 00 | 01 |
|---|---|---|
| worker epoch UTC | 03:06:11.608381 | 03:16:35.096078 |
| 峰UTC | 03:11:29.918066 | 03:25:26.599029 |
| epoch相对run.started_at偏移 | +13.181102s | +12.989703s |

这不是新增真实timestamp或精确误差界。原transport文件mtime提供相邻锚点：00 call13 request/response最后写入03:10:59.569156/03:11:56.738912；01 call15为03:24:32.593827/03:25:28.360439。它们与sim在途区间一致，但也不是worker epoch字段。

实际查得 `/tmp/inspection-namespace-admission.v1.log` 为26pass/1.64s，最后写入03:09:41.145698；`inspection-development-catalog.v1.log` 为12pass/15.28s，最后写入03:24:13.323619。主Agent另给namespace测试在03:10:39之前完成，本轮未找到该时刻对应logmtime，保留为上界陈述。按上述近似对齐，两份短测试log均早于峰；不能据这些log断言其他root工作或系统竞争不存在，也不能据同一次飞行期间存在辅助工作就归因。没有完整进程CPU/资源时间线。

## 一个最小诊断建议与处置边界

建议单一改动：在后续另行规定的开发诊断版本中，预分配72k行数值clock-loop audit，记录每tick lag、RPC drain数量/耗时、mission.step耗时、publication-tail耗时与阶段标记，并保存一次UTC+monotonic worker-zero配对；结束后统一写盘，避免每tick新增JSON/flush。这样可以辨别“进入work前已有backlog”与某阶段的wall阻塞，并补齐相邻sensor/publication的计时。该诊断自身开销也须先测，不能默认为零。本次仅建议、不实现或运行，亦不据此第三次重试同一策略槽或改变任何clock阈值/config。

每次attempt的原run model usage分别20calls/142134tokens、17calls/121559tokens。资源known与clock有效性是不同维度；本报告没有重做主Agent所报17operations/353calls/2240091tokens或18/90/48 SDK分母审计，不将失败trajectory改写为可用方法成绩。两attempt与retry cap耗尽均保留，不做best-of-repeat。

精简binding：`review/evidence/pilot-v2-clock-failure-timeline-v1/bindings.json`，含原文件SHA、1Hz窗口、事件邻居、actor/model区间、mtime scope及实际脚本hash。缺少完整lag序列和phase资源时间线是此次归因的实质边界；本报告无G4/main GO。
