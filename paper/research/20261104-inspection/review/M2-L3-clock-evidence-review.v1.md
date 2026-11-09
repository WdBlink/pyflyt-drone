# M2/L3 实际 clock 遥测：独立原数据复算

结论：**本次单次工程诊断完整结束，但 clock 门仍失败**：max lag .21028237460802757s，p95 .0058332083125947065s，32个120Hz tick超过原.1s。完整分母、archive、冻结来源、实际计量和清理证据相符，未发现本轮证据完整性的阻断或矛盾。结果定位了这次带仪器运行中的长阶段，不能证明旧未带仪器pilot的唯一根因、修复成功、模型性能或G4；已终止v3仍0/108有效，不补槽。

审查者 `/root/control_review`，同模型家族独立工程上下文。只用独立stdlib脚本解析binary、JSON、ZIP与逐call journal；未运行模型、physics、RNG、native写动作或新的测试。脚本复用了本审查者上一份已冻结独立raw-reader的字段/时格/计量检查，新增当前阶段、GC交集和RUSAGESELF窗口重计；没有调用root的phase分析器或抄其摘要作为结果。独立脚本、全部复算和30call明细见 `review/evidence/M2-L3-clock-evidence-v1/`。

绑定：`actual-diagnostic-binding.v1.json` SHA `aff1ca1675c890427dd02e20b5da5947b28ebfd66323d37addf8e88e2240f2f6`；`complete-diagnostic-and-model-operation.zip` SHA `ce6f65637a989b01ce35eefa707f5fe009cc0af0209a41cab72a871c7b39e852`；archive-members SHA `00070f830cf9d6396e845d388c09ef1ed9bd86251da5a0cd7de75db675a8adb3`。106成员无重名，exact member集合、逐成员SHA与raw diagnostic+model-operation现场全集均吻合，未只选片段。usage receipt SHA `7561cf8cef500335090a70e5910e6275b62503bae4b56ff150d4dfc3790e24f1`、root phase-recount SHA `4f35f6ad453113f023426960e9b4ccd38afd4d3a557a88e98b2a63a53e802371`及其实际输入指针也核对。

freeze binding SHA `b6aa5715bf4855db70f5eff2fcc5f1b954951f5722ff27479b1b4d539e87e068`：111冻结文件、当前后验107来源、61指定flight依赖、11输入全部匹配；run内嵌23source全文、run/source hash、六variant来源、M2/B3/原dev-L3-00 case及sampler1096691202、profile、privatecopy→world→run链一致。source/private/dependency unchanged、variant后验和原case unchanged均true。当前后验匹配与保存的flight文本/variant证据不扩称全host环境或持续文件监控证明；后续若当前源变动，脚本拒绝当前字节不应回写历史attempt。

完整原始clock72000×30 float64与rusage72000×30实际解码，有限值、每行sim_before/after、阶段wall/thread/processCPU顺序、epoch与lag采样括号、RPC/sensor格和rusage syscall bracket均核验；clock→rusage metadata→world的SHA链接相符。另解压144000×17实际240Hz error/z/interval trace，有限值、SHA与t=(i+1)/240逐行正确，终点600s。world、safety、tracking的物理积分/审计/样本各144000；world wall600.0050802499754s。error trace不是全physical-state重放。

| 独立全量clock统计 | 值 |
| --- | ---: |
| control rows / sim终点 | 72000 / 600s |
| max lag / p95（原floor索引定义） | .21028237460802757s / .0058332083125947065s |
| >.1s ticks | 32 |
| peak tick（0-based）/ sim_after | 51300 / 427.5083333333333s |
| GC配对 / overflow / unmatched | 116 / 0 / 0 |
| GC gen0 / gen1 / gen2 | 106 / 9 / 1 |

binary SHA：clock `f1b05f0d7dead940288ca121791e05c04e7466d8f5cbd1a140a249b809f9f3c6`，GC `6d14e8fe45f6ad93567c4a012bb54ac5396952dc0339c34e66d9fea62da0e963`，rusage `5d1254e0fb376927e0e2ec5e36510c83e3c8bca4ab80fd2beeba3c8d4cda4406`。default GC enabled，threshold[700,10,10]，frozen_count0；116对start均control-active，115对在main thread、1对在其他thread。未关闭/冻结GC或改变threshold。

当前峰值的准确阶段关系：tick51299在sim_after427.5时，`stats_end→pub_end`墙钟 **.210369624895975s**、main-thread CPU .090576s、process CPU .090582s。唯一main-thread gen2 GC完整落在其中，wall **.2098897909745574s**、thread CPU **.08993716599999857s**、process CPU .089936s，起止均tick51299/stage5。下一tick51300才记录最大lag，其rpc_end→step_end本身仅.00238429196s；不能把该peak的全部lag归于当前物理step。

`inspection_diagnostic_worker.py:76–96` 在publication前采lag，`:87–90` 的publication包含public_state构造和Queue.put，因此前tick的publication/GC长间隔与下一ticklag相容；这个名称不等于纯Queue.put syscall或异步feeder序列化耗时。gen2不在`pub_end→log_end`，该更窄尾段确无记录GC重叠；root phase摘要此字段的空列表不能被误读成“整个publication无GC”。该tick不是1Hz写日志tick。这些是当前运行的时序定位，不能把GC区间wall全部叫GC纯计算：其wall显著大于CPU，仍含可能的off-CPU等待/OS活动。

| 两处最长 begin→wake | sim_after | 请求sleep | wall | thread CPU | 已记录GC交集 |
| --- | ---: | ---: | ---: | ---: | --- |
| tick21562 | 179.6916666667 | .0041045417s | .130817124969326s | 31.667µs | 空 |
| tick54562 | 454.6916666667 | .0043263336s | .124636541935615s | 29.000µs | 空 |

两处process CPU也仅32/29µs，RUSAGESELF begin→wake均ru_nivcsw+1、minor/majorfault0；低CPU和无GC交集支持“这两个区间不是长GC计算”，不能唯一判定OS调度、IPC、GIL、timer或内存事件。记录覆盖当前profile期间且GC无overflow；未变化的counter也不能排除其他活动。

publication附近的rusage只有更宽的 **wake→log_end** snapshot间隔，独立差分为ru_minflt+5315、ru_majflt+23、ru_nvcsw+1293、ru_nivcsw+25，ru_utime+.046027s/ru_stime+.049470s。它涵盖同tick物理工作、publication、GC和其他线程，不能把所有fault/切换精确分配给GC或声称fault导致了全部wall缺口。RUSAGESELF含physics进程全部线程，非全host或单一线程。完整216000 snapshot相邻差分保留ru_nivcsw **12次-1**，signed net/signed delta sum均449212，未clip；counter名不自动构成唯一wait归因。

getrusage本身的216000 syscall bracket中位1.917µs、p954.667µs、max3.701375ms、总.4978381334s，只括住调用，不是全部clock-marker、array、GC、publication、写入和supervision仪器税。stage wall/CPU也包括marker与可能的off-CPU时间；默认GC保留不代表仪器零影响。

requested QoS记录return0/class33/relative0/no settings changed；已从本机SDK原生 `sys/qos.h:130–132` 核 **QOS_CLASS_USER_INTERACTIVE=0x21=33**，header SHA在独立recount中绑定。这是epoch前的requested主线程属性，不是effective priority或调度保证；该记录不支持“原来低QoS、调高即可修复”的结论。未运行setter或其他调优。

计量：30个连续实际call目录，actor request的messages/hash与seed逐一接到durable request，原response与transport一致。每tokens.jsonl逐行核token ID、runtime_generation连续序号及prompt count，所有call status completed；输入159770、observed且journaled生成5608，合计 **30calls/165378tokens**，小于32/180000预约。逐call三文件SHA、service intent/completion、run.model_operation、diagnostic resource_recount与usage receipt全部一致。purpose明确development、新diagnostic ROLE、原case/sampler/六variant SHA、intent/freeze/launch-review精确，非旧SDK第三retry。

native settlement已独立只读核对q中回执，status known、reservation一致、receipt SHA及settle packet actualusage一致；ledger第23条为known/unreserved=false。23operations均known，累计 **504calls/3169634tokens**，held/reserved/unknown/unresolved为0；cost/host/capital/energy仍unmetered，不可写0或扩称完整成本计量。

supervisor envelope errors[]、returncode0明确属于coordinator子进程；born-root与coordinator PID一致，supervision closed、remaining_owned及unclaimed_new_python_orphans为空、overflow0。5825观察记录对应hot_native5824（另初始观察），hot_legacy0/fallback空；独立求和观察wall4.9719157715s、parent-processCPU4.56778s，wall中位.789250ms、max50.41425ms。这些不是纯syscall时长、全hostCPU或唯一clock原因；baseline/shutdown仍可全表读取。supervisor span621.623853667s含启动/收尾，不能称整个操作只用600s。

原exec70993在host interruption后无法attach；已有完整envelope、终止清理、childreturn0和最终核验支持上述结果，但**没有新的supervisor自身OS exit witness**。不从源码return语句、打印status或childreturn推造OS exit0。caffeinate -i-w和仪器/profile/supervision条件与失败pilot不同，case/seed/模型相同也不是严格isolated A/B；本run只能定位自己的延迟，不能倒推旧pilot的完整lag/GC或唯一根因。没有据raw Y选配置、填槽或授G4、formal/holdout、人类领域认证。

独立binding位于 `review/evidence/M2-L3-clock-evidence-v1/binding.json`。本轮有限结果已完成并冻结，停止写R；没有实施或新的诊断运行职责。
