# G4开发进度v24

G4仍needs_work。G1–G3及已审同级B0参考对保持原范围。v3开发pilot在首槽两次clock失败后终止，0/108有效callback、0完整root；不重开旧槽、无第三attempt。正式seed/order/dataset均未生成，论文结果正文未开始。

独立v3终态审查核170成员SDK、32成员physical与两107成员resource archives精确；18/90/48失败/未启动分母完整，各次30calls165378tokens(159770input+5608generated)已known结算，source不变/cleanup closed。报告review/pilot-v3-terminal-evidence-review.v1.md保持原终态，不把raw Y或callback0解释成模型科学成绩。

新单次M2/L3工程遥测诊断已完整结束：600s/144000物理步/72000clock+72000rusage，maxlag .21028237460802757、P95 .0058332083125947065、32 ticks>.1；116完整GCpairs/overflow0。本run sim427.5的stats_end→pub_end为.210369624895975s，其中main-thread gen2 GC .2098897909745574s wall/.08993716599999857s CPU完全重叠，下一tick427.508333才记录最大滞后。该tick本身step只有2.384ms，不能把峰全部归给该step。另两个begin→wake .13081712496932596/.12463654193561524s而线程CPU31.667/29us，全部已记录GC无重叠。相关publication宽wake→log_end RUSAGESELF有5315minor/23majorfault等共现，不能唯一归因内核/IPC/GIL/磁盘；12次-1计数差原样保留。requested QoS33本来是USER_INTERACTIVE，不等于effective调度保证。

本诊断显式增加仪器和caffeinate，与无仪器pilot不是严格单因果A/B；它只定位自己的延迟，不证明旧两次故障唯一原因。106成员archive(25802040B)逐SHA验证；107source/61deps/11inputs事后不变，30calls165378tokens独立重计且native known结算。原exec70993在host中断后不可重新attach；保存的supervisor envelope给childreturn0、completed_engineering_not_qualification/errors[]/cleanup closed，不补造supervisor自身OS exit。实际科研账本现23操作全部known，504calls/3169634tokens，无held/unknown/unresolved；cost/host/Codex/capital/energy未计量，budget={}与既有本地资源授权不变。完整报告review/M2-L3-clock-evidence-review.v1.md、证据qualification/clock-localization-M2-L3.v1/与其独立binding。

primary consumer v2独立复验关闭PC1–PC4、原外层PC5及C1所测canonical路径，73附测/5独立cache正负通过；仍发现PC5-b P2嵌套false/0字典相等残余，5独立service通过/1实质red，v2仍request-changes。所有集中测试12:51:01.525454UTC前结束，无测试子进程残留，随后仅写报告；本次telemetry在其后启动。working已把service与model-intent的nested binding改canonical比较，新的18项类型回归通过，但该最后修补未独立复验。secondary额外arm SDK3项Fake检查通过，新增formal common-calendar纯适配仍未验证；全部formal候选未安装，不能称fullgraph合格。

针对已观察的GC长扫描与off-CPU唤醒，root新建未安装physics-runtime-window候选：仅physics主线程申请Mach soft-time-constraint(120Hz周期/4ms计算/8ms约束)，process cyclicGC在<=600s控制窗口延后，出窗口恢复原timeshare与GC并收集。原epoch/pacer/physics law/.1阈值/模型/actor均未改。Apple原始API文档与XNU头、Python3.11 gc文档支持机制但不提供硬实时保证；短native self-thread set/get/restore实际return0，8项生命周期/异常恢复/单行遥测metadata/hash检查通过。只是在planning/physics-runtime-window.v1/freeze.v1/binding.json中的候选，无真实600s载荷或独立源审查，不是已安装修复/安全认证。实际内存、GC absence、active/restored政策、清理与600s clock仍须新工程资格证据。

后续继续：独立检查最小runtime候选，单次同M2/L3载荷资格；按结果修根因，完成formal消费者/secondary接线与最后类型残余复验。完整合格新108槽pilot、clean+20s实际集成、最终source/model/prompt/split/retry/hardware冻结与独立G4接受仍待。19项义务、RA-L、分母/评分/选择器/统计没有更改，不能把clock失败或软件测试当论文方法结果。检查点后继续已授权工作。
