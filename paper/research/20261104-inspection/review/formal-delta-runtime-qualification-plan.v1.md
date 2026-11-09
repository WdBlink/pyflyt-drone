# Formal admission delta 的运行负载与最小资格证据计划

结论：没有源码依据机械要求重跑整个普通 development 108 集。那条路径仍以 formal_ref/formal_input=None 运行，即使重跑108，也不能自动覆盖 formal 每请求深验证的新负载。既有pilot可以继续作为其实际104source/61deps/10inputs和已跑工况的证据；最终G4仍按原dev-only校准/冻结合同判断；若声称新formal热路径的运行负载已覆盖，须有相符证据或明确的包络论证，不能由522软件green或普通dev clean/20s自动推出。提前取得真实main/holdout祖先的端到端验证不是新增G4门。

本报告只给 source-only 工程判断和未来有界验证条件，不实施、不运行、不授G4/main/human。当前order19在飞；本轮没有生产/NumPy import、tests/probe/native、大批hash、正式抽样/model/physics，未操作pilot或改104生产源。只定向读冻结文件、当前四个入口的小量文本差异及actual d809源码/说明。

## 精确范围

Subject `planning/formal-admission-candidate.v3/response-resource-integration.v1/binding.json` SHA256 `6c7456781f6dc985b570b5d5d463968ec8fe320b2a8f9a8d16d6fff1c826993d`。按其 frozen_copy读取；不重复整图预审和各组件真实性/统计审查。当前生产development路径和18/108闭合状态来自本次任务给定，不是本轮重计。没有bulk rehash44source或104source。

actual owner为d809 `runtime/autoresearch-document-snapshots-v2`，SDK为原 `runtime/autoresearch-state-reader-v1`；二者分别传入context factory。稳定旧CLI兼容exec不等于旧直接Controller可用。以下短文件名指subject对应 frozen source，准确路径/声明SHA见binding。

## 各阶段增加什么工作

| 阶段 | 新工作及源码定位 | 相对于物理epoch/在飞负载 |
| --- | --- | --- |
| coordinator/runner setup | `episode_plan`、admit_episode_plan/dispatch验证G4、dataset/execution/slot；frozen_sources重新读完整源，fingerprints读deps；LLM runner在首次CLI、client前、worker start前复验（run_llm_development:70–78,94–100,145–154） | 大部在spawn/epoch之前。多文件read/hash、JSON分配、owner lock可能拉长准备/ready时间；不能因在epoch前就忽略内存驻留、页缓存和后续fault影响 |
| worker预warmup | `run_inspection_reference:44–54` 调 load_formal_worker_input；case loader独立建context、验整dataset、同buffer私有copy SHA/JSON/语义/config，再fresh验证；secondary v2还前后admit_dispatch并做clean变换（formal_case:40–75,78–116） | 明确在warmup:55和runtime/epoch:74–77之前。没有把这些检查放进每个240Hz积分。会增加启动CPU/I/O、临时对象/峰值RSS；启动失败不能当飞行资格通过 |
| model service启动 | formal profile/weights前后多次admit，另为owner/intent取binding；可直接数到5次admit_dispatch入口（service:139,152,159,167,180） | ready前。模型weight树校验和MLX load本来就有；不是新增每请求weight重hash。新负载主要是深证明与原生owner验证 |
| **每个将生成的模型请求** | tokenizer/call/token预算通过后，service:198调用admit()→admitted_limits(formal)→admit_dispatch→execution/dataset/namespace/G4/owner全链。预算耗尽not_started在:194–197返回，不走这次生成前admit | **与独立physics在飞并发**。它不新增MLX forward，却新增CPU、磁盘/页缓存、owner锁与分配压力；不能只按“多一次dict查找”估计 |
| actor/coordinator在飞部分 | PublicLLMActor/协议、物理while/pacer没有由formal wrapper新增逐step证明；parent等待service结果/处理同类RPC。主要直接新等待来自service admit，及更晚reply对后续adaptive请求/命令时刻的影响 | physics会继续；更长E2E可能减少实际调用数、变late或触发原watchdog。即使raw generation一样，也不能推断策略轨迹或clock负载不变 |
| receipt/重验/分析 | 每attempt formal freshness、known native settlement、完成缓存read-only重验；archive/meter、SDK重计、closed snapshot、response/event/数值输出等（llm_episode:109,150–165,215–248；measurement_loader；analyzers） | 多在该flight结束或下一slot之前，不应计入该flight的物理epoch lag。它增加周转CPU/I/O/owner负载，并可能影响下一次冷/热缓存。不得在另一flight同时跑集中重计来“免费”隐藏其成本 |

一次admit_dispatch并非一次owner读取或一次61-dep遍历：fresh检查、main/secondary层叠以及holdout对locked main的递归会重读。同一字节可能在一次高层调用里多次解析/hash，准确次数只能用实际call trace确认，不能静态乘一个乐观常数。

深证明的字节规模由最终source/deps、**显式列入qualification_evidence的文件**、dataset case/audit/报告/树以及owner决定（formal_admission:340–366；dataset_admission:60–70,88–195）。若G4 evidence列表直接列大archive，就会hash其字节；若只是小已绑定摘要，不能反过来虚称每请求都读了大archive。应记录实际选择的包络，不能用小玩具文件替代后称覆盖真实包。

actual d809 `Controller.load:208–229`读control+checkpoint，校Brief、pair digest、implementation、live round refs和owner transport；implementation:231–236及transport:249起还hash所需代码/兼容入口。formal native_owner_reader:174–184使用原生lock/load；baseline dev service则直接json读control。不是同等成本的两个读法。main是72个case/audit；holdout12外加locked main72链。future owner/ledger/doc体积增长也是capacity边界，当前18callback状态不能自动代表研究尾部。

## 必须保留的计时归属

service `admit():198`以及request mkdir/write:200–205发生在`started:207`之前。response.wall_seconds在:221取值，因此不包含这段pre-generation admission，也不包含随后response.json/stdout传输；它仍只支持原reader所声明的generation/token-journaling区间。

client从submit前monotonic开始（model_client:71–75）到poll取到响应（:100–103），E2E包含admission、tokenization/请求落盘、service/transport/poll等待；原120s watchdog也包括这段。原actor/consume的60s软截止和delay20等待语义不能因此放宽。记录的client roundtrip+consume仍有已声明的采样缺口，不是精确source wall age/job adoption。此报告没有重新解释另审已限定的计时语义。

## 经典路径与科学解释边界

经典formal路径不承担等价的逐generation全面审计。`run_inspection_reference.py:248–250` 在process.start前复验；`:255–286` 在飞循环只消费public state/tool results、`policy.decide`并enqueue命令，没有admit_dispatch/native/source/dataset全审。源/依赖最后在`:301–303`核对。共同worker同样只在预warmup验formal_input。因此当前实现中的全面重审负载只在LLM每个实际生成前重复，不能把两类方法的在飞审计成本说成相等。

须分开三个来源层次：

- **用户/冻结科学合同的要求**：accepted `experiment/design.ral.v3.json` 的control_variables要求600s/1x、compute/link delays消耗飞行时间，development_and_freeze要求dev-only资格、最终source/model/prompt/budget/split/version/retry冻结，scope_and_cost.metering要求唯一owner预约/实际计量/unknown对账。本轮用户还明确要求每effect的native资源与撤销新鲜性保留。不能让旧reservation/已撤销资格继续授权一次新generation。
- **当前实现的具体手段**：每generation调用完整admit_dispatch，重复走全部dataset/source/evidence字节与原生pair核验，是staged源码为实现这些边界选择的保守深验证策略。上述科学合同并没有规定“每次必须重新读遍所有72/12 case/audit及所有source/archive字节”这一成本实现。不能把实现开销冒称科学协议天然的纯推理成本。
- **本次验证的责任**：策略既已在候选源中，就须忠实披露并计入实际E2E/资源负载。合同未指定具体读法，不表示现在可以删fresh验证、缓存旧permit或少读文件；本报告不建议优化或削弱任何信任边界。若未来改实现，应另作明确source revision及验证。

因此client E2E测到的是本实验系统中的推理/transport/poll等待加来源/授权验证等开销，不能不加限定称为纯LLM推理或可直接外推的部署时延。service.wall_seconds仍按上一节限定为generation/token-journaling区间。classic的policy_compute_seconds（`:281–283`）仅圈住policy.decide，也不能与LLM client E2E直接作为同一边界的纯算法速度比。

实际已发生的这些开销仍消耗600s，不能从飞行clock/Y或response中扣除、重定时后给LLM“免税”。可以并列报告各阶段和总管线等待，并声明这种实装成本不对称；不能把系统来源审计税误归因为模型本身。为工程压力演练而添加的shadow负载也须另标，不能悄悄变成正式classic的新方法成本。注入delay track仍是既定额外等待政策的总效果，不能假定自适应调用数相同或按常数减去审计就得到“更快模型”的反事实。

## 最小、诚实的分阶段证据

**A. 先固定delta与无副作用接口证据。** 完成既定pilot后，在任何新运行前登记final candidate/owner+SDK/source/deps/input/model/profile及工程intent。复用已闭合的合成gate/worker/service/receipt测试与本source diff：证明phase/refs/新鲜性调用保留、shared热循环/actor/模型profile未因delta改变。522可以支持软件回归，不能给出耗时、RSS或1x证明。实际费用仍先由唯一owner按既有上限预约，后按实计量结算；工程运行不进入科学N或旧SDK槽。

**B. 分清“mode integration”与“formal guard负载”。** 既定actual clean和20s-delay各600s仍有价值，需用final candidate真正的clean变换/实际额外等待、真实MLX/独立physics与完整receipt，而不是Fake outcome。若它们仍走development formal_ref=None，它们证明的是新增源安装下的dev模式、等待、清理、计量和clock。它们本身不证明formal_ref深链每请求负载；报告必须如此写明。

当前真正main/holdout namespace、dataset/review、execution及native祖先尚未产生；生产formal_ref要求这些真实祖先，因此此时无法提前诚实取得该分支的真实端到端成功证据。这不是要求在G4前先造这些祖先。

在不提前创建真实main/holdout、也不制造假G4 acceptance的前提下，若要补充运行负载包络，最小工程手段是**不授任何权限的只读负载演练**：按原fresh调用位置重现所需的读取/解析/hash/native-lock-load工作，用真实最终source/deps/候选证据字节与隔离的、明确synthetic的72/12+main-parent元数据规模；不产生正式case/seed，不向真实owner写qualified finding，也不以演练结果代替生产admission返回值。若当前没有这样的诊断入口，可以另登记一个source-bound工程adapter；本报告不实现，也不声称它已经存在或等价。

演练至少要披露实际调用次数、读取字节/文件集合、owner+checkpoint/live-ref规模、main/secondary/holdout递归包络、冷/热条件。不得删fresh检查、缓存一个旧G4Proof长期替代重验、缩小文件集，或用1-byte fixture代表真实archive成本。metadata-only软件路径正确性与真实负载并行证据应分别列出。

**C. 负载须与真实1x并发，不能只先测一遍再卸载。** 在既定两次actual集成中，若要支持formal热路径delta，工程演练需在相同进程/相同请求边界或经论证的保守并发包络下存在，并独立标明它是shadow workload而非真实formal admission。只先做standalone微基准，再跑无新负载dev flight，不能证明新增压力下仍1x。也不能把source-only估算的ms直接加减旧lag来宣布通过。

选择已冻结工况应依据既有CPU/RSS/模型体积证据而非Y。20s必须保持原B4-M1/无ablation合同；另一既定集成能覆盖哪种模型/资源共存工况就诚实写哪种。若两者仅覆盖M1而未给出其他冻结模型的资源包络，不宜自动外推M2/M3；最小补充应针对未覆盖资源条件，而非机械重跑六root×八条件。此建议不改变科学方法/阈值、模型选择或正式N，也不在本轮新增运行。

**D. 既定集成与补充负载演练的证据记录。** 以下对应已有门禁及新声明的可核性，不新增原计划的科学N、阈值或G4硬项；特定字段不可得时应收窄声明，不能虚构。若实施该负载验证，至少保留：

- 真正600s/144000物理步、原120Hz/240Hz审计与原100ms/两attempt合同；按现有G4/receipt区分policy结局与infra失败，不能挑高Y作为运行资格证据。
- 预warmup admission、epoch/worker启动、每次admission起止和对应decision/admitted-call/reservation的wall/CPU以及必要内存/I/O观测；service原始wall、client E2E、delay requested/measured和progress分开。观测自身source与开销需声明，不能再引入full-PS热轮询等未计负载。
- 实际clock原始存储及全部既有摘要/计数。若现有worker只保留摘要而没有72000原始lag行，必须保留这个限制，不能由max witness+1Hz progress重建72k；需要逐样本重计时另用预登记工程trace。
- worker/model/coordinator各自的CPU/RSS/I/O或可用原始counter及单位/高水位/负delta限制；不得clip不利counter或把共现直接当唯一原因。
- 原始run/world/actor/transport/model-call/usage/cleanup、全部known/unknown/retry分母、真实native reserve/settle ack、source/deps/input前后绑定；没有unknown/cleanup闭合证据就保持未通过，不零化或免费第三次重试。
- 独立重计branch覆盖、guard次数/包络、clock/完成/资源/因果链及原hard gates；有清楚的范围结论，不仅引用passed布尔值。

这些是预期证据要求，不是本轮已有结果。单次或两次成功只支持声明的机器/软件/模式/负载包络，不能提供硬实时或全体root保证。

**E. 原G4门与后续真实phase观察分开。** 原计划的最低要求是dev-only调整后冻结generator/threshold/prompt/budget/split/version/retry；它不要求逻辑上不可能的“G4前已取得真正main/holdout祖先的端到端证据”。本审查不把新意见扩为该硬门。既定dev clean/20s、最终source及可测额外负载包络可支持范围明确的组合论证；最终是否满足原G4由实际契约与独立接受判断，本报告不发条件GO。

只读演练不是actual production formal_ref成功。若包络保守/位置同构尚未证，未证的是这项formal运行性能声明；应如实留待有界工程补证或G4后的真实phase观察，不能自动把它混写成已证，也不能仅以“尚无真实holdout祖先”否定原dev-only G4。绝不能为取得证据先开真实main/holdout或造假qualified标签。

G4后每个原定formal slot仍须自己的fresh admission、原100ms clock/infra门与两attempt上限。后续真实phase可观察新分支，但不能倒填此前G4、不另加科学N、不免费第三次重试。若实际门失败，按已有失败/unknown保留与停止规则处理，不能因先前开发资格通过而改判成功。

## 为什么不是默认重跑108

新增重成本集中在formal条件分支，另有pre-epoch导入/绑定和postflight重验；源码未显示新逐240Hz证明或新的MLX生成算法。普通dev108重跑仍跳过formal_ref分支，不能填这条证据洞。已有完整pilot若最终成功，其实际冻结源、方法/模型/根与trajectory范围继续保留；source delta由上述接口、负载和实际模式集成补证，符合组合审查而不是把旧记录改成新source。

只有发现真实共用热路径/模型profile/prompt/控制/物理/事件语义改变，或有不能局部定位的广泛回归、尚未覆盖的资源工况，才有理由扩大所需验证。扩到多少应由实际影响决定；当前source-only信息没有给出“必须整个108重跑”的具体因果依据。相反，实际formal热路径未测/未被包络覆盖是具体缺口，但重跑未走它的108不是针对性补法。

## 本轮边界与完成

本轮没有测量任何新时间、字节总量、RSS、guard计数或clock，也没有实现adapter、预留资源或运行actual clean/20s。所有负载大小/干扰是源码可见工作与待测风险，不是已证故障或新根因结论。此计划只限定下一证据应覆盖什么，G4仍按实际未完成条件处理。保存报告/binding后停止R写入，不继续CPU工作。
