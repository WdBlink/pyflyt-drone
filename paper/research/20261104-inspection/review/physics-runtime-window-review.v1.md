# Physics runtime window v1：独立有限源码审查

结论：**存在1项P2阻断，应先修复构造异常后的rollback，再审新版本；本v1不判可进入600s诊断。** 当前8项软件检查通过不覆盖该资源恢复路径，也不是600s clock、内存或G4资格。仅审未安装freeze.v1；已知root将另改working副本，本报告不审或预先关闭未测修复。

P2：`freeze.v1/source/timed_profile.py:14–20` 在 `self.window.start()` 已成功改变self-thread调度与process-GC状态后，还执行 `gc.isenabled/get_threshold/get_count/get_freeze_count` 和configuration字典构造。此处发生RuntimeError、MemoryError或KeyboardInterrupt时，except只调用 `super().stop()` 移除base GC callback，没有调用已有window.close。失败构造不返回profile对象，调用方无法靠通常的profile finally取得并关闭它。

独立Fake反例（未改变真实线程调度/GC）让实际PhysicsRuntimeWindow使用Fake native/collector，在start后注入get_count元数据错误。结果：原异常抛出、base callback确已移除，但 **window.closed=false、native RT仍active、collector仍disabled、audit.errors=[]**；反例自己的finally手动close才恢复。这不是仅由静态猜测提出的泄漏。最小修复是constructor的异常清理经过包含window.close的统一stop路径，保留原异常和恢复失败信息；增加“window.start成功后任一metadata采集失败”的回归。现有 `tests/test_timed_profile.py:36` 仅注入PhysicsRuntimeWindow构造本身失败，不能覆盖这一阶段。证据脚本及结果有完整v1强SHA guard，旧snapshot未改。

冻结输入SHA：binding `88cda8920ab03decbcdc09c891a54c50540b8dbefd3a9ed075b156b1ba694517`；physics_runtime `bf4e0b38f1d7da98ccd559091b2b722c74ed0e6fdfd0088336ccaa441bdfc21c`；timed_profile `7883d176b9defaf6a3e5f42ddc29483a3b28603afad3c4987f8f36c5cd50b527`。7个manifest成员全部逐SHA匹配。复跑仅冻结两test文件：`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider <freeze.v1>/tests`，实际 **8 passed in .03s**，含短self-thread native设置/恢复和单行numeric telemetry；没有模型、飞行、RNG或native owner写入。standalone反例首调用因缺repo PYTHONPATH在import前拒绝，随后用 `PYTHONPATH=.` 正确运行；不是候选产品失败。

其余已验证边界：

- Mach ABI与本机SDK `mach/thread_policy.h`及Apple XNU头一致：thread port/flavor/count为32bit natural，return/boolean/int vector为32bit；time constraint为三个uint32 absolute-time字段加preemptible，共4个word。代码将值限制在<2^31，使用c_int vector不改变本次正值bitpattern。ns→ticks为round(ns×denom/numer)，125/3 timebase对应period200000、computation96000、constraint192000、preemptible1；区间顺序有校验。提供的短native test确实比较当前native返回与requested vector，并恢复原timeshare模式。
- 精确语义上flavor1/count1是 **THREAD_EXTENDED_POLICY.timeshare**，值1表示standard timeshare；THREAD_STANDARD_POLICY自身COUNT为0。当前不是把零参数standard的伪结构当ABI使用，恢复实际上经过合法extended flavor。`get_default`传入false，返回default=true表示该flavor当前未生效，而非主动读取默认掩盖实际RT；fresh前门要求timeshare1且RT default，再核active_start与active_end。检测到demotion或恢复失败会记audit.errors。仅首尾读back不能证明中途从未临时demote。
- DarwinThread构造只取pthread_self→pthread_mach_thread_np，不接受外部PID或thread handle；该对象必须在可信physics线程上构造、启动和关闭。没有给模型/coordinator进程加RT或禁GC的代码。GC disable是**整个physics进程**的自动循环收集，不仅主线程；引用计数仍运行，手工collect仍可执行。前置full collect在epoch前，后置collect在loop外；GC原enabled状态和threshold快照/比较保留。600s循环中的循环对象增长、RAM/page pressure、其他线程/手工collect和实际GC事件仍需真实有界诊断证据。
- runtime窗口class本身依赖调用方保证<=600s/异常finally；profile duration有0<duration<=600验证。调度请求不是硬deadline或超时看门狗，未修改epoch/pacer/.1门槛/飞行控制/模型prompt。warmup和mission构造后的entry、loop结束及异常离开的真实顺序属于待审worker adapter，当前只有调用约定，不能把尚未读到的接线当已验证。
- 成功/失败set、demotion、standard恢复失败、初始GC enabled/disabled及重复close均有现有软件覆盖。restore失败被记录并继续尝试恢复GC；closed重复调用不再重放操作，不能声称一次restore失败后已恢复成功。后续worker须把任何audit.errors明确变成operationalfailure，不能仅因控制循环完成而qualify。
- 单行save实际核clock/runtime_policy/rusage三处audit一致、GC start=false且recorder_changes_gc_configuration=true；重写clock metadata后更新rusage的base-clock SHA，再重写rusage并更新world返回的rusage SHA，成功路径跨文件链接正确。这是该新variant拥有的完成metadata，不是修改旧原始记录。中途I/O失败仍应作为未完成/infra保留，不能接纳不一致metadata。本轮未发现另一项已复现的hash错误。

Apple说明Mach只提供soft real-time并可能demote；不支持“设置成功即可保证600s通过”。见[Apple scheduler文档](https://developer.apple.com/library/archive/documentation/Darwin/Conceptual/KernelProgramming/scheduler/scheduler.html)、[XNU policy ABI](https://raw.githubusercontent.com/apple-oss-distributions/xnu/main/osfmk/mach/thread_policy.h)。Python明确automatic GC补充reference counting；见[Python3.11 gc](https://docs.python.org/3.11/library/gc.html)。本机头文件SHA和上述具体支持范围保存在evidence/source-checks.json，未大量复制来源正文。原QoS33不需被“调高”来解释本候选；本次无QoS setter或有效priority保证。

范围是同模型家族、独立工程上下文对这一未安装component的有限review；没有重新审主任务/benchmark或新adapter，不授G4/main/holdout、人类领域认证或原失败根因结论。必须先闭合本P2并新freeze复验，再决定一次独立无分数runtime诊断；其真实600s需另核active/restored policy、GC、内存/清理和clock原门，不能由这8绿项替代。报告和独立binding冻结后停止写R。
