# Adaptive L2 clock 诊断：独立有界源码审查

结论：本次限定源码/单元范围未发现阻断项，仅支持按冻结intent和既有原生资源门执行**一次无分数的 dev-L2-00/B4-M1 工程诊断**。这不是原失败L1第三次重试、SDK/pilot补槽、配置采纳、修复有效证明或G4资格。clock超过0.1仍可作为完整的失败定位记录保存；低于0.1也不能据一次不同场景/负载的pass认定原故障已修复。

审查者 `/root/control_review`，同模型家族独立工程上下文。只读冻结源及必要共享调用点，不改源码、契约、旧证据或owner；没有模型forward、物理、重放、正式RNG/geometry或实际预算动作。另一个agent的parent实测证据审查只核其指针SHA，本轮不重复其结果重计。

## 绑定与检查

R=`paper/research/20261104-inspection`。`planning/clock-localization-adaptive-L2.v1.json` 与 freeze intent字节相同，SHA `771122d7dd7c6a430656ddee05a229add374ff5fce49364160e7dd86e3c4a94b`。freeze binding SHA `d9b07a0ffc64a687deef6404c546a5fb15cd2c2c2a27f353155b881b2f0bb328`；全部72项文件SHA核对通过，plan所列63项原科学源实际文件仍逐SHA相同。四个工程源和两测试working字节也先核与freeze一致。

| 工程源 | SHA256 |
|---|---|
| `eval/probe_inspection_adaptive_clock.py` | `38e0b5f43508e483b84c2d513728f6abbe0178e132d7b294e7c52966940aac99` |
| `eval/inspection_diagnostic_worker.py` | `eb1c8bc91f2bedeee159066f50ec5ada037ec40ac199c5b8cd3b176290f43207` |
| `eval/inspection_clock_profile.py` | `aa4ff3be49bd744e19ffa7e58b03c220b34a137c1fbfc13c4374f298c0b7dd7e` |
| `eval/inspection_clock_profile_rusage.py` | `43cc8ee2ddc196c01584fa0b2c079ed7871520f225dfff30a2b07e2ba3bbbe00` |

原reference worker仍为 `4c280adab488f9c81d68265dd9930d9775dbb6372b36d1d776d9350f9f36f3ca`，原LLM coordinator仍为 `c3bed0403ca4a8e7f5d79c0db3df5da0aeb27e46e37423e94474447fbd4da552`。dev case只核现有字节SHA `fb011f0b3960c0a730ba7e4f68c644b63aeb3c6cce0b01be2caf5c0f76ab68d3`，未生成新case。独立小binding `review/pilot-adaptive-clock-diagnostic-source-review.v1.binding.json` SHA `8fa89a71b9ec3848e42405cd42b7daf7da8943019958a627fff3d11454d111fb`。

实际执行纯单元：所附两模块 **20 passed/0.02s**；另3个独立Fake入口/清理单元 **3 passed/0.04s**。独立脚本与日志在 `/tmp/inspection-adaptive-clock-source-review-v1/`，SHA已收入binding。没有真正启动spawn子进程执行worker；下面明确区分pickle/构造检验和未来实际flight。

## Worker 注入与原语义

冻结 `probe_inspection_adaptive_clock.py:75–84` 在内部child入口先做相同reservation校验，再把原coordinator的worker符号显式设为 `partial(worker, profile_factory=RusageClockProfile)`，固定M1/B4、现有dev-L2-00、sampler2026110616和新flight输出；没有ablation或额外delay参数，沿用原默认zero delay。原coordinator在 `run_inspection_llm_development.py:105–109` 以spawn context创建target，读取的是当时已替换的worker符号。

独立Fake检验实际通过该child CLI，核固定参数，并使用真正 `multiprocessing.reduction.ForkingPickler` 对target roundtrip，恢复为可解析的 `eval.inspection_diagnostic_worker.worker` 及 `eval.inspection_clock_profile_rusage.RusageClockProfile`。另一单元进入原coordinator main，在Process构造处截获target/args，确认spawn及partial、600s/实时/原reference_layout=False，且在任何LocalModelClient或Process.start之前停止。这验证接线与序列化形状，不能替代未来真实spawn、MLX、物理整合执行。

与前次已审diagnostic worker相比，本次只多keyword-only `profile_factory=ClockProfile` 和以factory构造profile两处差异；原reference.worker未变。仍是warmup→新mission→构造profile→epoch→初态publication→相同绝对deadline `epoch+t+1/120` sleep→最多16 RPC→mission.step→共享进度→lag→publication/log。没有中途epoch重设、GC disable/freeze、QoS/nice/scheduler设置或0.1/重试上限改变。新增profile构造和QoS查询在epoch之前；counter/marker在循环内增加有限真实开销，不能称无扰动复现。

本次有意复用原adaptive B4 actor/coordinator/service，模型响应会按既有合法工具协议影响本次L2轨迹；它已不同于前次固定历史prompt+B0负载。这一变化已预先声明，不伪称精确复现旧L1。新程序不导入record_matrix/SDK入口，不读取Y作判据；底层world仍可能保存原audit评分，但本次没有采纳它或写科学槽位。

## Counter、CPU、GC与可解释范围

`inspection_clock_profile_rusage.py:12–14,32–48` 预分配每control tick三次snapshot，每次8 counter加before/after单调时刻，共30个float64；对应 `begin/wake/log_end`。实际每次取 `getrusage(RUSAGE_SELF)`，覆盖整个physics进程的全部线程，和GC/main-thread CPU不能混为同一范围。`ru_utime/stime`是累计CPU，fault/switch/swap为OS定义的累计counter，`ru_maxrss`是累计高水位，不能把差值当作逐tick RSS或其他进程状态。

`mark`先记原wall/thread/process marker，再采counter并保留syscall bracket；因此begin→wake span也包含begin counter及marker开销，不能直接减requested_sleep就把剩余全部叫纯sleep overshoot。可用原epoch+sim deadline和wake时刻区分调度落后，并用bracket估计本次采样的位置；不是单个kernel事件的精确时间。counter保持0/不变不能排除未支持的事件或其他进程的分页。既有微探针sleep后nivcsw+1和median542ns仅是该probe的counter/单次调用观察，不是所有等待的唯一归因标签或完整飞行仪器开销上界，本轮没有重跑该微探针。

`:17–29,38` 只查询当前物理主线程的requested QoS；本机SDK `pthread/qos.h` 的getter签名已对照，未执行新的实际QoS探针。返回requested class不代表effective priority、override效果或调度保证，非Darwin或错误状态也不能编成有效值；没有set调用。

原base recorder的30字段、wall/thread/process CPU、16384槽GC callback及overflow机制字节不变。两类numeric tick记录合计约34.56MB/72000行（另有GC/原lags等有限内存）；预分配不等于没有其他Python分配、仪器税或GC扰动。Rusage构造失败会移除已注册GC callback（所附单元已验）；`save`在控制循环后写binary，记录native endian/fields/字节数和SHA，rusage metadata链接base clock-profile SHA，world的返回profile再链接rusage metadata（`:50–66`）。只完成的control tick被计入rows；异常部分不能升级为72000完整行。

## 资源、监督与失败处置

`probe_inspection_adaptive_clock.py:33–51`要求特定development diagnostic purpose，绑定registry、receipt、case SHA、固定sampler及四个worker variant源SHA；entry须reserved且calls/tokens严格native int并足够。outer和内部child都调用它。它不是完整native计划认证器；原service还在模型import/load及每次真实生成之前核accepted frozen plan、active/stop、registry/profile/receipt、充分live reservation和独占operation，coordinator等service ready后才启动物理worker。实际意图目前明确not_yet_reserved_or_run，本报告没有替owner认证任何具体新预约；须先用既有原生入口真实reserve32calls/180000预算并通过前置门，不能unreserved生成或手改owner。

外层 `:94–104,118–137`保留SubprocessTree的0.1s观察cadence，wait界900s；ObservedTree仅加最多12000条parent观察wall/CPU记录，满槽记overflow，不扩成kernel subreaper。caffeinate只有声明的idle assertion，最终清理。normal child入口应由该outer监督启动；内部 `--instrumented-child` 不是可独立采纳的诊断结果入口。

`:105–115`完整完成条件是原run状态/来源/依赖/case不变、600s/144000物理步、72000 control和rusage rows，再用实际durable model-operation receipt重计。lag是否>.1单独保存，不作为“修好”门。`:119–136`任何cleanup unknown、错误、variant/case漂移均使状态operational_failure_or_unknown。独立Fake提供完整600/144000/72000元数据及已知非零计量，再让watch.close返回unknown：最终确实exit1/unknown，原非零resource_recount保留，未改成0；该Fake数值不是实际flight结果。

脚本不settle owner；成功路径以真实service intent/calls/tokens/completion为计量，失败缺证明时usage缺失/None而非0。失败仍可能有真实消耗，owner必须按durable journals实际重算或保留unknown，不能按返回码/客户端计数结零。既有SubprocessTree只跟踪已观测born identity，不能保证任意fork/host-loss；source检查证明失败闭锁，未在本轮实际杀进程/重测孤儿清理。硬件资本/能耗仍unmetered。

本报告仅条件支持冻结方案的一次无分数工程定位。完整飞行结束后仍需独立核原始clock/counter完整性、模型实际计量及监督证据；不能从本轮20+3单元pass、单次未来clock pass或L2轨迹结果反推出根因已确定/修复已通过。原失败L1记录、耗尽重试、SDK108/48失败分母和G4 pending边界保持原样。报告/绑定冻结后停止写R，不继续扩大审查。
