# Pilot v3 终止实际证据：独立离线核验

结论：**新 v3 pilot 真实失败，0/108 有效 callback、0 个完整 root；不是 scientific model Y=0。** 两次同一 order0（dev-L3-00/B3-M2/repeat1）实际运行均完成 600s/144000 物理步，但 worker 记录的 max lag 分别为 .4054049999685958s、.2103543746906098s，均超过原 .1s 门；两次有限 cap 用尽，不能继续或补第三次。未发现本轮 archive、来源、SDK 分母或资源结算的矛盾；这一完整失败记录仍阻断 G4，不能用先前单 L2 clock 通过覆盖。

审查者 `/root/control_review`，同模型家族独立工程上下文；仅 stdlib 离线 JSON/ZIP/SHA/压缩 trace 重计，未导入执行源或运行模型、physics、RNG、native 写动作。独立脚本与60call明细见 `review/evidence/pilot-v3-terminal-v1/`。root 保存的 driver exit1 与 traceback/cap-exhausted、SDK failed 相容；本审查未独立观察已结束 OS session，故不把提供的 exit code 说成现场观测。

来源与 archive：terminal binding SHA `ea78536750d84959f8ecf9f6afa5a3745bb87268d4a6abb68e732355e9d0c011`；archive-members SHA `5f6272d679c7d19c94093c6c18d279a4c8815e5e70507af74bd6eec7166e90b8`。`complete-pilot-sdk.zip` SHA `478587c31568ba7faac950a1126d26e11dce046c47d1798215a117668c22378c` 的170成员，`complete-physical-attempts.zip` SHA `e69caefca9378a6d2b10e8b09eb4a0181b88d85ab00012f81f3a701f3248f5c9` 的32成员，均无重名、exact member set、逐成员SHA匹配，并逐字节等于完整现场树；没有只挑选成功片段。两份实际 resource archive 各107成员（含自身manifest）亦逐成员 SHA、原路径文件和 receipt/archive/terminal settlement链核验。coordinator在settle前封存的resource archive与含最终terminal/settle回执的physical archive互相补齐，不能把前者称为含全部settle后状态。

原 launch 的103source/61deps/10inputs及全部 embedded source 文本仍精确，source/dependency/input表与v3 preparation原字节一致，当前文件亦同SHA；schedule2026110621、sampler master2026110622与 preparation相同，未重新抽取。原B1 selector未变。本轮每flight嵌入的23source全文重新hash，等于预先plan和103总闭包中的对应SHA；61deps、privatecase、thread/condition/reservation/seed/profile、world→run SHA链和sources/dependencies/private_case unchanged均相符。完整源码集合与launch可复用边界不因这次失败改变。

| SDK envelope | 完整计划分母 | 独立重计状态 | attempted | value |
| --- | ---: | --- | ---: | --- |
| classic | 18 | 18 stopped | 0 | 全 null |
| llm | 90 | 1 failed、89 stopped | 1 | 全 null |
| root-means | 48 | 48 stopped | 0 | 全 null |

逐row/raw的身份、状态、attempted、reason和null一致，case×condition×repeat exact集合均完整。唯一start为order0；无order-receipt、pilot-completion或recording-completion-index。两次 physical attempts 只属于一个未返回有效值的SDK callback，不能写成2个有效模型样本；其他107集未启动，48均值未生成。

| 每次实际证据 | attempt00 | attempt01 |
| --- | ---: | ---: |
| simulated seconds / physics / safety audit / tracking samples | 600 / 144000 / 144000 / 144000 | 同左 |
| worker记录 max lag（s） | .4054049999685958 | .2103543746906098 |
| worker记录 p95（s） | .005993125339351479 | .005677832701849184 |
| worker记录 >.1 ticks | 129 | 92 |
| max witness sim t | 427.75 | 287.5083333333333 |
| cleanup | closed | closed |
| raw world Y（保留，未接纳） | .3313166234514549 | .3313166234514549 |
| terminal Y / status | null / known_operational_failure | 同左 |

两份240Hz tracking error/z/interval压缩数组均实际解压重计144000行，17列有限值、原始SHA和每行t=(i+1)/240正确，终点600s；raw SHA均为 `e1a4977aa11b56d2cd22f076175efcda3c40baa076a20cb95b49ef29a094f2d0`。它佐证逐物理步记录分母，**不是全physical-state、wallclock或完整lag重放**。本轮不重新判业务成绩或把rawY升级成被接纳值。

两次world/terminal/binding的max/witness/汇总SHA一致，因此失败按既有门槛明确；但这些普通worker飞行没有完整72000 lag/GC/rusage向量，表中max/p95/tick是认证到来源的worker聚合，非独立从逐tick wall原始序列重算。独立重计的各600条1Hz progress只得到 residual最大.2716026250/.2096895000s与2/3条>.1，不能取代120Hz129/92，不能据此恢复遗漏峰值、GC或唯一根因。witness current step分别.4034566660/.0858254581s，既不能把lag全归于该step，也不能证明RPC、调度或外部活动是哪一个原因。

每个attempt均有独立reserve→born/root start→return0→cleanup closed→archive/receipt→native known settle证据。root PID与started相同，remaining_owned与unclaimed_new_python_orphans为空；terminal14项attempt payload/92项model payload以及全部settlement_file_hashes逐一吻合。目录只有attempt00/01，无attempt02；两份retry_allowed仅表示故障类别，原coordinator的range(2)总cap仍适用。SDK的failure reason为finite symmetric infrastructure retry budget exhausted。

资源独立逐call复算：每次30个连续call目录；30个actor request的messages/hash与seed都逐一接到durable request，原response未被transport改变；每个tokens.jsonl逐行核runtime_generation序号、prompt count和token ID。每次输入159770、runtime-observed且durably journaled生成5608，合计 **30calls/165378tokens**；本轮两次合计 **60calls/330756tokens**。各自service completion、run.model_operation、receipt逐call SHA与总usage、settle packet及native ledger一致；没有把同样的总数误当成两次合计。各有32calls/180000tokens原生reserved，purpose为development且绑定相同plan与不同attempt ID；无unreserved旁路。owner只读快照22operations全known，累计474calls/3004256tokens、无held/unknown/unresolved，与上一414/2673500加本轮相符。cost/host/capital/energy未计量，不能因known calls/tokens写成0或完整成本已知。

并发限制必须保留：root/实际reviewer元数据给出首flight存在63个fixture tests（22.60s）的UTC窗口 `2026-11-06T12:04:48.869440Z–12:05:11.772747Z`；该事实不应隐去。retry期间该agent报告无集中测试，并不证明整台host无其他负载。本审查只绑定所提供的并发记录，不冒称独立重建过全主机进程时间线。run.started早于模型load/warmup，物理monotonic epoch没有UTC锚；禁止把started_at+sim time虚构为两峰UTC，也不能把首flight并发当作两次失败的唯一因果解释。后续单次带遥测诊断是另一个工程attempt，不是旧slot的第三retry，亦不改变本轮永久失败。

复算脚本最初曾误用compact message序列化计算episode plan SHA；对照未变的canonical_digest定义修正为默认分隔符后才形成上述结果，原数据未改。完整绑定见 `review/evidence/pilot-v3-terminal-v1/binding.json`。结论限于该失败包的完整性与分母/计量；不授G4/main/holdout、性能结论或领域人工认证。terminal报告冻结后不再改写；另行的新诊断launch准备单独审查。
