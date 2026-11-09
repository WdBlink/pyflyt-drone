# Post-inventory L2 实际时钟证据：独立全量复算

结论：本次无分数、带仪器的 dev-L2-00/B4-M1 运行，**通过已声明的120Hz采样lag≤0.1s门槛**；600s/144000物理积分/72000 control记录完整，archive、计量与清理证据未发现本轮阻断或矛盾。这只是该次工程clock资格，不是旧故障唯一因果证明、策略成绩、全部G4或新完整pilot许可。原pilot-v2永久失败，旧slot与重试分母不回填。

审查者 `/root/control_review`，同模型家族独立工程上下文。只读实际记录、冻结源和计量文件，未运行模型/飞行、改production、owner或旧证据。独立脚本仅用stdlib，不调用root的phase分析器或production validator；先从原始binary/逐call记录求值，再与原汇总交叉核对。

## 来源绑定与复算入口

R=`paper/research/20261104-inspection`；q=`qualification/clock-localization-adaptive-L2-postinventory.v1`；raw=`logs/inspection-clock-localization-adaptive-L2-postinventory.v1`。

- 完整archive SHA256：`f50c9eff10bdc1be30f09d89a884a921a15c73c850229bc1efb329d56b9f304b`。86个唯一成员逐一与archive-members manifest核SHA；raw14文件+model-operation71文件=85个现场文件全集均与archive相同，另一个`supervisor.stdout-stderr.log`只在archive中核验，其末行状态、maxlag和usage亦匹配。
- usage receipt SHA：`18ecfbaaef8251cc12ef5a7ea6d4e29926f8b3a1b650b2fef613d985ccc5b41c`。freeze binding SHA：`3d7f26115c117565bd475bafe14927567d04ef4191b056ba552cd12e1ac0f931`；111冻结文件全部匹配，107 source当前字节全部匹配。
- 原63源delta仍只有声明的cleanup与driver：`361ba11b…→ce210650…`、`ce911fdb…→c14c52b0…`。六variant与intent完全一致；run内嵌23源文本/recorded SHA/实际文件相同，61项flight依赖当前字节也逐一核对。不是重新审107份语义或MLX/OS完整依赖闭包，后续源码变化也不得回填本attempt。

独立交付为 [audit.py](evidence/postinventory-clock-evidence-v1/audit.py)、[完整复算](evidence/postinventory-clock-evidence-v1/recount.v1.json)、[23call明细](evidence/postinventory-clock-evidence-v1/model-calls.v1.json)及[binding](evidence/postinventory-clock-evidence-v1/binding.v1.json)。脚本SHA `9d43fc345cbce0ee51fa24b4266f10c4494a05fa1750fc251a7d61f71155f140`，binding SHA `a356a00d1db093cb883be4f958361be39fd9e29edd9471ed6ca16e9e44e0701f`。命令为 `.venv/bin/python <audit.py> --repo <repository> --out <本evidence目录>`；未来当前source/dependency漂移会使其当前字节检查拒绝，不能据此改写历史结果。

## 完整时钟、物理记录和仪器

直接解析17280000B的72000×30 float64 clock矩阵，核有限值、逐行sim_before/after、跨行/阶段wall与CPU时间顺序、RPC/sensor字段，并用progress_end/stats_end时间括住每行lag；逐行而非采样复算：

| 指标 | 独立值 |
|---|---:|
| control rows / sim终点 | 72000 / 600s |
| max lag | 0.055215666621620585s |
| p95（原floor索引定义） | 0.005578458247100571s |
| >0.1s ticks | 0 |
| peak tick（0-based）/ sim t | 50220 / 418.508333333s |
| 物理积分 / safety audit / tracking sample计数 | 各144000 |
| world wall duration | 600.0080292499624s |

clock-ticks SHA `eceaf5bd3b2784afb7d479c865b298544bd185c16b29a13c4d7db6c7e5de0b77`；GC binary SHA `e0d8eda3a7ac672a7dfec4d002fd360f82bf9404f19adc7ea7cabf4d7f80994c`；rusage binary SHA `940297a4fece052abc710fc12f7e1b1b76946684389d1e89e7413e974bbdc16c`，其metadata对base clock metadata的SHA链接亦核对。world内profile完整对应这些metadata，非独立孤立文件的数字凑齐。

另解压240Hz error/z/interval trace，独立核144000×17个有限float64，t从1/240到600、每行时间格正确，raw19584000B，SHA `3fa2412bf3bfebcf77680a51c4661a64d53734fe4a2e00c8eab18871e1631d4e`。这支持物理记录分母，**不是全physical-state重放**，本轮未从它重判全部安全/业务或Y。

GC276事件正确配对为138对、无未配对/overflow：gen0=126、gen1=11、gen2=1；137对在物理主线程、1对在其他线程，全部start处于control-active。GC默认开启、threshold[700,10,10]、frozen_count0，与前轮相同；已审冻结源没有关闭/冻结GC或改threshold。最大GC墙钟51.532542ms，主线程CPU51.484166ms，发生在peak同tick的rpc_end→step_end阶段；该阶段墙钟53.497291ms。这支持对**该局部峰值**的时间重叠解释，不是旧平台故障的唯一原因证明。

rusage72000×30矩阵即216000个带syscall wall bracket的snapshot，全部SHA、长度、chronology与stage位置核对；bracket中位1.125µs、p954.459µs、max360.167µs，累计0.347640s。它只括住getrusage，不包括完整marker/array/GC/observer仪器税。`ru_nivcsw`相邻原始snapshot有**59次负delta，均-1**；未clip，signed net与signed delta sum均280791。不能把OS-specific counter名或非单调差分当唯一wait归因，0/不变也不能排除事件。

requested QoS记录为33/relative0、getter return0，与前轮相同且只读，不能据此推断effective priority。每秒progress构造/write/flush仍在时钟循环内：本次pub_end→log_end最大10.665042ms，包含该阶段全部工作/调度，不能叫纯磁盘syscall开销。GC与这些开销均留在测得lag中。binary/最终audit/大JSON归档的post-loop开销不在600s控制tick定义内，不能说整个supervisor只消耗600s或仪器零成本。

## 逐call实际资源与清理

23个连续call目录，每个request/response/tokens三文件与archive、usage-receipt逐call SHA相同；23个actor public request的messages/hash、seed2026110616+ordinal−1与durable request一致，23个实际response与durable response一致。所有23项service status completed，输入字段和每token runtime_prompt_tokens、generation序号、token ID类型/计数逐行核对：

- 输入156574 tokens；runtime-observed生成6950 tokens，durable journal也正好6950行。
- 合计 **23 calls / 163524 tokens**，与逐call单独usage之和、service-completion、diagnostic及usage receipt全部相同，未超过32/180000预约。
- 每response的usage本来是**单call**，service-completion才是总和。首版独立脚本曾误用累计断言，已依据未变service字段定义修正后全量通过；不是原记录缺陷或改分。

native reservation ack与本operation ID和32/180000 packet一致，packet六源purpose/case/sampler匹配。后收到的 `native-settlement.v1.json` SHA `24844a3df9ccf8ffabe416da5e1410105f24912a1139ed3df3eee005390d826d` 已只读核status known及ID；本轮不重新审整个累计ledger或改其任何字段。计量是这一个本地service operation的提交输入+runtime观察输出，不是FLOPs、所有Agent总tokens或硬件能耗；资本/能耗仍unmetered，不能写0。

coordinator returncode0；supervision closed、remaining_owned及unclaimed_new_python_orphans为空、errors为空、观察overflow0。archive supervisor末行也为completed_engineering_not_qualification，实际hot_native5775/hot_legacy0/回退原因空。root session39054的OS exit0由root提供，本审查核的是保留的returncode/输出和清理证据，不冒称独立观察过该session；周期born-identity清理也不是kernel subreaper/host-loss保证。

## 与前轮的受限比较

同dev-L2-00字节/sampler、原actor/service/worker/profile/GC/QoS及门槛下，直接重算前轮原clock binary得到max1.839878208s、p951.162444416s、>0.1为17020 ticks；旧失败保留。本次模型请求却为23次，前轮18次（127751 tokens），反馈时间、公开状态、prompt及合法工具轨迹随之可能不同，不能当严格控制的策略或因果A/B。

| supervisor observation记录 | 前轮 | 本次 |
|---|---:|---:|
| 条数 | 1734 | 5776 |
| observation wall累计 | 456.280531s | 4.333403s |
| wall中位 / 最大 | 193.786812ms / 1.270571s | 0.616187ms / 15.042ms |
| parent process CPU累计 | 22.082895s | 4.267769s |

这些都是独立从每条observer记录求和/排序所得，parent process CPU**不包括旧PS子进程CPU**，wall累计不等于占用CPU或唯一阻塞原因。observation次数也不等于全表PS次数：attach用显式current snapshot也记录一次，因此本次5776记录对齐5775热native调用；旧1734不能逐条都叫fullPS。baseline/attach/shutdown直接全表读取不在hot backend计数中，hot_legacy0并非“整个运行零PS”。

结果与被审修复降低该热路径监控成本、这次L2采样clock门通过相容；不足以证明所有旧lag只由PS造成或未来其他模式已合格。没有据Y、任务成败或此单case选择配置。后续完整pilot/最终integration/source freeze以及其余G4要求另行完成；本报告和evidence冻结后停止写R，原pilot-v2不续跑、不填slot。
