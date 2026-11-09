# Adaptive L2 clock 失败：独立工程证据复核 v1

结论：原始数组、时钟统计、计量及归档核验通过；本次新的adaptive B4-M1开发诊断**过程完成但clock条件失败**。完整序列呈29段累计超界，不能用主峰当前一轮工作或某个累计counter解释全部迟滞。supervisor观测确有可测成本且与峰相邻，但其456秒墙钟不等于456秒CPU，22秒父进程CPU也不覆盖整个进程树；现有证据不足以证明它是唯一原因，更不能回溯证明旧L1失败原因。

2026-11-06，`/root/statistics_design_review`，分离上下文、同模型家族。按autoresearch证据边界只读原始记录、source/freeze、归档和native结算回执，以离线NumPy/JSON/ZIP复算；未运行模型/物理/测试或正式seed生成，未改生产、owner、配置、阈值、失败槽或SDK。新诊断不是旧L1槽的重试/恢复，也不是G4或正式方法成绩。

## 完整性与数值重算

输入：`logs/inspection-clock-localization-adaptive-L2.v1/diagnostic.json` 及 `flight/`。clock与rusage各72,000×30 float64，均17,280,000 bytes；GC为250×11、22,000 bytes。按声明little-endian读取，全部finite，sim_after精确对应每1/120s至600s；144,000 physics steps。

- clock ticks SHA256 `bd385c1eeafa9cad2fdcafe46a74d7c30f244ff6b7fa3f4a8b5e0a7ce57a0236`。
- rusage SHA256 `44f46f7140a74aa0055ae024b5abb44135371e86699ffc01779136a53ec2f73f`；其metadata绑定的base clock-profile SHA一致。
- GC SHA256 `3615c2ff9f6c7b21f584603416c3682bb549c3b54a4ff23dae0fe8321b458ed1`。

独立重算max **1.8398782083609717s**，row54130（0-based）、sim451.091666667；p95 **1.1624444163714998s**，采用声明的sorted floor-order索引68399；>0.1为 **17,020/72,000 = 23.6389%**，对应141.8333 control模拟秒，不能称连续wall时长。各值与world/profile/diagnostic一致。

完整序列有29段连续超界。最长四段：427.466667–454.166667（3205tick，峰1.839878s）；473.658333–499.166667（3062tick，峰.707097s）；351.433333–374.116667（2723tick，峰1.733626s）；558.875–578.616667（2370tick，峰1.126302s）。终点world wall600.008597s说明之后追赶，不抹去过程中超界。

归档SHA256 `aaf252419a1fbb6ec38e5db1bcc644aa2a871941aec8f0750144fef46f82e816` 匹配；71成员无重复/路径逃逸、CRC通过，70个diagnostic/model-operation成员逐字节hash匹配原文件，另一个supervisor transport由归档绑定。freeze.v1的72份绑定文件核验一致。

## 主峰不是单轮制造

进入主峰tick时，`begin_wall−epoch−sim_before` 已为 **1.837646709s**；前一tick lag1.837598s，本tick lag1.839878s。当前rpc_count=1、sleep_requested=0、sensor=false、非1Hz logging tick；wake→rpc_end墙钟.529ms，mission.step边界9.953ms/main CPU6.622ms。当前短工作只能解释邻近增量，不能当成1.84秒的全部cause。

每阶段“墙钟最长的同一行”与同期CPU如下（ms）；不将不同位置的CPU最大值混入：

| 标记间隔 | sim_after | wall | thread CPU | process CPU |
|---|---:|---:|---:|---:|
| begin→wake（含有意sleep） | 88.133333 | 19.570833 | .056916 | .059000 |
| wake→rpc_end | 563.5 | 110.744584 | .240959 | .243000 |
| rpc_end→step_end | 564.916667 | 173.564125 | 7.662250 | 7.664000 |
| step_end→progress_end | 575.975 | 15.211292 | .142458 | .142000 |
| progress_end→stats_end | 413.016667 | 154.498250 | .057708 | .048000 |
| stats_end→pub_end | 337 | 74.718083 | 1.285417 | 5.811000 |
| pub_end→log_end | 337 | 9.343583 | .314125 | .560000 |

多个长wall区间只有很少main/process CPU，支持调查等待/脱调度；标记名称本身不能识别等待机制。begin→wake还包含主动sleep，不能全称stall。worker完整control窗口约600.0078wall秒、main-thread CPU264.9775秒、worker-process CPU267.1722秒；后者包括该进程其他线程，不包括model/planner或监督helper进程。

## Rusage范围与实际counter限制

getrusage使用 **RUSAGE_SELF**，不是physics主线程专属计数；begin/wake/log_end采样在各clock marker之后发生，返回值时间只被before/after包围。每阶段counter差因此属于其快照窗口，不能逐事件分派到某RPC、GC或kernel wait。

调用bracket最大2.362667ms（wake@362.625）；begin/log_end各最大2.131542/1.704959ms。clock log_end到下一begin的最大间隔5.103375ms（下一row sim521.1）。这些是观测到的调用/边界间隔，不是全部instrumentation CPU/分配开销的上界，更不证明额外遥测完全无扰动。

端点净变化：user CPU254.290280s、system CPU12.881884s、`ru_minflt` +14,754、`ru_majflt` **+5**、`ru_nswap` 0、`ru_nvcsw` +309,572、`ru_nivcsw` +964,302。不能把switch字段名直接等同唯一抢占原因。新增事实：216,000个按时间排列的快照中，**ru_nivcsw有229次下降1**，其他所查counter无下降；原始值保留，不clip0后冒称精确总切换事件数。该字段不满足理想单调计数假设，平台/聚合读数边界需保留。

major fault增量只落在sim1s的wake→log_end窗口+4、sim4s窗口+1。主峰±2sim秒的major增量为0，不能扩成全程无major faults。主峰当前wake→log_end窗口还记录minor +3、`ru_nivcsw` +53、major0；这是全worker进程的窗口差，不能把53指派给9.95ms step或宣称context switch是唯一cause。maxrss是累计高水位，端点差原始值81,920不能当当前RSS或内存压力证明。

开始前主线程requested QoS=33、relative0，仅是request值；没有修改调度，不保证effective priority，也不支持直接建议提高QoS。

## GC与supervisor对齐

GC250行配成125对，无未配对、overflow0，gen0/1/2为114/10/1。最大gen2在sim437.125附近、worker epoch后438.241816–438.371507s，wall129.691625ms/main CPU91.992500ms/process CPU91.993000ms；属于主峰所在长超界段里的真实局部开销，但早于主峰约14.56wall秒，不是同一个峰值tick。主峰前后±2wall秒仅一枚约.153ms的gen0；没有可据此认定1.84秒全部来自GC的证据。

supervisor观测序列1734条、overflow0、顺序非重叠。独立汇总：

| 保存的观测范围 | 数值 | 不能扩大为 |
|---|---:|---|
| observe函数wall间隔总和 | 456.280531s | 整个host或进程树的CPU |
| 单次最大wall | 1.270571s | 对worker造成同长阻塞的证明 |
| 同些区间父supervisor process CPU总和 | 22.082895s | helper子进程、coordinator、worker、model、planner全部CPU |
| clip到实际control窗口的观测wall union | 430.151840s | 这些时间都由observer占CPU |

源码计时包围 `super().observe()`；外部helper执行/等待、调度及其他工作均可能包含在wall，子进程CPU没有自动并入parent process_time。名义“.1s loop”不是固定10Hz采样保证：保存的start间距median .298152s、max1.371668s（少数初始化/关闭调用更近）。

主峰的monotonic时刻位于某次observe的worker相对epoch **452.443563–453.091198s** 内，该次wall.647636s、parent CPU.021710s。重叠是事实；在约430秒control时间都被观测区间覆盖的背景下，单次重叠既不独有，也不提供排他性因果证据。未保存helper/全host CPU、调度或等待栈，不能认定supervisor是唯一根因或把此新L2结果回溯成旧L1原因。

## 模型、source、退出与资源

18个actor request、18个response均与durable service request对应，seed按2026110616+ordinal−1递增、messages一致；18个response为completed。独立逐行journal校验得121,216 input tokens、6,535 output/journal行，总 **127,751 tokens/18calls**。M1下载receipt仍为 `d9b4b522e679b3fa84f25d46036b6f89642d2c6ea908c3524cfb12b8e5b14bdc`，原generation profile保留；这是新的adaptive请求流，不宣称与原L1或fixed tape输出相同。

23份run base source的内嵌内容与当前SHA、61份依赖及4份显式worker variant source均匹配；原pilot63 source也全部未变。实际通过 `coordinator.worker=partial(diagnostic_worker, profile_factory=RusageClockProfile)` 执行instrumented worker；run中原worker source map本身不是完整variant execution closure，额外4文件和freeze绑定必须同时保留。

diagnostic为completed_engineering_not_qualification、returncode0/errors=[]，world/run completed；它表示完整收集和退出，不是clock通过。保存的born-identity supervision为closed、remaining_owned/unclaimed_new_python_orphans空；只读ps也未见记录的监督/协调/service/worker等PID。仍限于记录的process-recovery范围，不作任意host/后代保证。

native保存结算为known，本次18calls/127751tokens；总391calls/2509976tokens，reserved calls/tokens为0、unknown/unresolved空。费用仍未计量，known不等于cost0。未读取或写入真实owner。原失败14份原始文件SHA与上次审查仍一致。

## 收束

证据支持“新的unscored adaptive开发实例再次出现clock失败，以及多处wall显著大于本worker CPU的区间”；不支持唯一cause、修复已完成、配置选择、恢复旧L1槽、生成SDK或G4 GO。root的未安装process-inventory候选未在本审查中测试或批准。

binding：`review/evidence/pilot-adaptive-clock-evidence-v1/bindings.json`，包含原始SHA、完整重算摘要、峰邻域、counter原始限制、GC/observation窗口及执行脚本SHA。独立脚本/中间结果留在 `/tmp`；完成后停止R写入。
