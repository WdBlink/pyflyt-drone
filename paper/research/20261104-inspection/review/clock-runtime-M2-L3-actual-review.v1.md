# Runtime candidate M2/L3：实际单次载荷独立复算

结论：**该candidate在这一个600s M2/B3/dev-L3-00工程载荷中通过原120Hz采样lag≤.1s门；证据完整性、恢复、计量和清理未发现本轮阻断。** 原始72000行重计max .008292792000247573s、p95 .003075666333472782s、>.1s为0。GC与Mach调度是组合干预，只有一次负载，不能据此授G4/完整pilot、硬实时或旧故障唯一根因结论，不能填补已永久失败的v3 SDK槽。

审查者 `/root/control_review`，同模型家族独立工程上下文；仅离线stdlib读取/解码/SHA/计量重算，未运行模型、physics、RNG、native写动作或新的测试。独立脚本直接解析clock/GC/rusage及逐call原文件，不调用root的phase分析器。脚本及结果在 `review/evidence/clock-runtime-M2-L3-actual-v1/`，脚本可从该目录直接运行；不依赖未来production仍维持旧字节。

输入：actual binding SHA `745f09383426d9d05ce07318c1249a990d6e0b64ddcd7aa3756d46ea011d552e`；完整archive SHA `75629ca2b694f8a763ea46fc3d330420c7cfcbf253e86746e27d7c39e5bf520b`；usage receipt SHA `f173e6db7a3eaabb9b23ea211781bcd04ab5af57348e4357edf6b0186022aa5d`；phase-recount SHA `cb182ddd113ccb6b112b35c60c15553649718664b539d7b5d61e4cf2f798f556`。106个ZIP成员无重名、exact集合、逐成员SHA与diagnostic+model-operation现场全集均相符。phase输入指针亦核SHA，再与独立raw统计交叉核对，不由其布尔声明直接通过。

freeze SHA `0ed2471d31e2814e487b62a4b3fc3544130da2ec16ed32ca651f889b6bdfb3ce`：113冻结文件完整匹配。当前109source/61deps/11inputs已于 **UTC2026-11-06T14:30:31.917024** 逐字节后验核为相同，观测SHA `4e94a7087b9e0c10f675a08a25c55647d64d0f71c4d9d259d416279c31a378e6`。通知root完成后，允许其继续有意的shared-worker接入；本审查此后只用immutable freeze/archive及该已捕获观测，不把后续变化误说成历史attempt漂移。也不能反过来把后续安装代码回填成本次实际来源。run内23source全文、source table、八variant、61依赖、privatecase→world→run、sampler1096691202、M2模型receipt/profile、B3/零注入/无ablation与原scope均相符；source/private/dependency与variant后验flags均true。

完整raw clock与rusage各72000×30 float64重新解析，所有有限值、逐行120Hz sim时格、阶段wall/thread/processCPU顺序、lag测量括号、epoch/sensor/RPC格与216000 syscall snapshot bracket正确。clock→rusage metadata→world跨文件hash一致，包括新runtime_policy。另解压144000×17个实际240Hz error/z/interval，SHA、有限值与t=(i+1)/240逐行正确，终点600s；world物理积分、safety audit与tracking samples各144000。该error trace不是完整physical-state重放。

| 本次完整分母与clock | 独立值 |
| --- | ---: |
| sim / physics / control / rusage | 600s / 144000 / 72000 / 72000 |
| max / p95 lag | .008292792000247573s / .003075666333472782s |
| >.1s ticks | 0 |
| max tick（0-based）/ sim_after | 24419 / 203.5s |
| raw control末次后的world wall | 600.039210375s |

当前clock binary SHA `6e3e343e9611a88431d5792e34dd05c485e1818e541c9d322b6d7103c9b77f6d`；GC `188b5583d72d9a35e129e6130deb98d5c62787dcde559b9ecd85b075a0aaeeea`；rusage `e70626743638796aee7e3ed4cc35c007cdaf6ea6060fcc7d458daacac8262313`。完整metadata与raw输入SHA在独立recount按原路径列出，未把前default同名文件SHA混成当前run。

runtime原始audit在clock/rusage/world/usage/actual binding各副本完全相同，errors=[]。timebase125/3，requested ticks[200000,96000,192000,1]对应period约8.333ms、computation4ms、constraint8ms、preemptible；active_start和active_end均相同且default=false。before/restored都是flavor1 extended-timeshare值1/default=false，prior_rt与rt_after_restore均default=true；GC原enabled=true/threshold[700,10,10]、epoch时false、close后true。记录window_wall600.003532958s。首尾读回证明此两点请求生效与退出恢复，不能证明中途每时刻有效priority或绝无临时demotion，也不是硬实时保证。QoS33是Mach设置前的requested USER_INTERACTIVE，metadata已明确pre-Mach scope；没有把普通QoS读数当新调度效果。

GC数组2事件正确配成唯一gen2，overflow0、unmatched0，tick/stage均-1、control_active=false；其结束wall2755.337336791小于原epoch2755.337394708。wall .021209958s、threadCPU .021017083s、collected32，与pre_epoch_collected32相符。**没有记录到control-active GC**，且metadata的GC-at-start=false、recorder_changes_gc_configuration=true。post_loop_collected14和enabled_after=true由source-bound runtime audit保存；callback在stop时已撤，因此后置collect不在原GC数组里，不能伪称raw GC向量也记录了post-loop14的时长。

这将GC工作明确移到控制窗外，并非消灭成本。world wall600.039210375s晚于runtime窗口记录；差值还含stop、后置collect和audit/收尾，不能把它全归于某一操作或从clock无超阈推成整个supervisor只耗600s。阈值未改，GC仅影响physics进程的自动循环收集；引用计数及其他进程默认GC仍存在。无观察GC不等于所有对象都被释放或其他root不会增长。

rusage的216000 snapshot保留全部signed原值。本run未出现负相邻delta，ru_nivcsw signed net222070；前default仍有12次-1/net449212，脚本没有clip或把负值修成0。本次minorfault净4973、majorfault净6、nvcsw净29；这些是RUSAGESELF全physics进程线程的OS-specific计数，共现不能唯一解释时钟改善，0也不证明某事件不存在。

**RSS限制**：本次原生ru_maxrss从243712000到250609664，增加6897664；前default为245743616→245760000，增加16384。它是该进程累计resident高水位，既不是瞬时RSS，也不涵盖coordinator/model/host总RAM。snapshot止于控制末次log_end，不能测出post-loop collect后的实时释放量；不能声称“无内存增长/无泄漏/其他负载内存已合格”。这里只证明这个有界600s run记录的峰值与增长，未来更大case或同一长寿命进程仍需各自证据。getrusage bracket累计.4528023381s、中位1.917µs、p954.250µs、max20.042µs，只是syscall括号，非全部仪器税。

与前次default诊断采用同一独立raw-reader重新计算，case/model/profile/sampler一致：

| 指标 | 前default | 当前组合candidate |
| --- | ---: | ---: |
| max lag | .210282374608s | .008292792000s |
| p95 lag | .005833208313s | .003075666333s |
| >.1 ticks | 32 | 0 |
| GC control-active pairs | 116 | 0 |
| actual model calls/tokens | 30 / 165378 | 30 / 165378 |

当前同时启用process-GC deferral与self-thread Mach约束，另有variant/factory/worker provenance差异；主while AST相同不使之成为单因素A/B。相同case/seed及相同总calls/tokens也不控制全部适应性回复wall、host状态、memory或调度背景。各run只在自身epoch内比较，不跨run直接相减absolute monotonic时刻。不据这张表区分哪项干预贡献或倒推旧未带仪器pilot唯一根因，也不据raw Y选配置/声称科学性能胜出。

资源逐call实证：30个连续durable call目录；每个actor request messages/hash、ordinal/seed接到原request；transport保留原response。tokens journal逐行runtime_generation/原prompt count/token ID正确：输入159770、runtime-observed且journaled生成5608，合计 **30calls/165378tokens**，全completed，未超32/180000。逐call SHA、service intent/completion、run model_operation、supervisor recount与usage receipt一致。native purpose绑定development新ROLE、原case/sampler、八variant、intent/freeze/source-review，无unreserved旁路或旧SDK补槽。

已实际读取 `q/native-settlement.v1.json`，SHA `1d67503fe7c8f9dd1f7706f46d3a8165d14bb5e5f7ae059117dc0c0676823c0e`，status=known、reservation/receiptSHA/settle packet actualusage一致，owner该项known/unreserved=false。第24次operation结算ACK记录累计 **534calls/3335012tokens**，reserved0、unresolved[]、unknown[]。这次结算不是仅据root意向推定；cost/host/capital/energy仍unmetered，不写0或扩称全硬件费用/功耗。

cleanup envelope errors[]、childreturn0、born-root与coordinator PID一致，supervision closed、remaining_owned及unclaimed_new_python_orphans为空；5952条观察对应hot_native5951，hot_legacy0/fallback空，overflow0。观察wall累计2.543535465s、parentCPU2.507929s，中位.4181455ms、max6.334541ms；这些不是kernel subreaper/host-loss保证，也非全hostCPU或唯一延迟原因。supervisor span619.40720125s。新receipt保存root的supervisor_exit_code0，与完整末态相容；reviewer没有另attach OS session，不冒称自己现场观察过该exit。它与前default host-interruption时缺失的OS witness是不同记录，未补造旧witness。

本轮无证据完整性阻断，结论只支持该冻结candidate、该单次载荷的工程clock证据。共享worker后续接入、最终source freeze/更多条件pilot与G4仍另门；不能将一次结果升级为全平台资格或人类领域认证。本报告/独立binding冻结后停止写R，没有新运行或实施职责。
