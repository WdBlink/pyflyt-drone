# 事件义务scope与实际配对：有界独立复验 v1

结论：kernel6ac的新scope、增益基线、创建时card及先前边界反例已按登记定义复验通过；33项event测试通过。已完成的event/clean开发配对、已知唯一retry、完整clean archive及共同日历全分数重算一致。**未发现本限定范围的新工程阻断；这仍只是B0、单个OOD-L3 root的开发证据，不是主效果、main或G4通过。** 原924报告、原candidate.v1与实际world均未改。

本审查为同模型家族独立上下文工程复核。未运行新physics、MLX、tokenizer或模型；未改源、owner、旧数据或他人文件。round22 checkpoint要求R静止时只写 `/tmp/inspection-event-scope-pair-review-v1/`；收到apply/begin恢复通知后才归档新报告。实际world重计顺序执行、单线程数值环境、审计子进程nice=10，未并行重计或重放物理。

## 来源绑定与开发解释

| 来源 | 实际SHA-256 |
|---|---|
| `eval/inspection_event_endpoints.py` | `6ac62f8d266966420bc3457d4b7992ae5755094bbfdd9b7928cf73dc1cd341b8` |
| `tests/test_inspection_event_endpoints.py` | `d3010361235f9f69b0eb6cfca874c300bc20857c5e28cfd0a430453b75781ea8` |
| `eval/inspection_llm_actor.py` | `056bdbce49bd9096a73df1d580d25b5f48987e228e742354805edaa5c863a497` |
| `qualification/event-obligation-scope.v1.md` | `b23ff2cd2aa847b6cb39611bdc246c1f9f0d13b533e91629005e57cf2f10b9d1` |
| accepted `experiment/design.ral.v3.json` | `83f8f1a72f5261aaa5783c0d190f3788fafcd05bca1e7d16a6901c69c9abf0de` |
| frozen `experiment/evaluator.v2.py` | `de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b` |

本次固定scope明确化了合同“affected obligation”，而未改变主任务score、原格、物理路径或总体：survey按已有四条中心线的水平Voronoi责任区与ROI相交，边界归低索引；inspect/observe仍只用事件前已交付candidate关联。该定义不依赖是否看到目标或实际FOV，所以不会把未发现/失败响应反向删成NA。文档记录在candidate.v2文件生成之前，源码和手算条件与文档一致。它是明确登记的pre-main开发口径，不把一条已见结果当作定义正确性的统计证明。

同一固定分区说明已进入所有模型共用SYSTEM，五种架构/消融context均保留相同段落。它只公开固定工具责任区，不公开hidden target位置/未来轨迹；offline truth定位仅用于事后标签，未反馈actor。v4 tokenizer元数据15项为3972–5442≤8192、actual_generation_calls=0，绑定同actor SHA；本轮只核元数据/消息内容，没有重新tokenize或把它当完整动态飞行context资格。

## 定向反例与scope手算

独立fixtures和33项event测试实际通过（pytest **33 passed in0.35s**）：

| 条件 | 复验结果 |
|---|---|
| y=-200/0/200及两侧邻域 | 精确边界归较低index，邻域分别归对应条带。源函数按连续event位置判定，不用未来检测结果或实际覆盖足迹。 |
| affected旧running survey、区内尚无任何packet的必需目标 | 仍eligible，缺响应则固定deadline删失；不是NA。 |
| 其他region的未满足目标 | no_affected_active_workload，不把其混入该survey义务。 |
| inspect candidate关联arrival<event / =event / >event | 仅前者赋资格；同钟/未来包不倒灌。 |
| same-ID先坏后恢复event前1/3基线 | 不算response；再提高到2/3才计。条件是高于max(0,event t⁻质量,前一sealed prefix)，关闭上一版基线歧义。 |
| 新job创建时合法card1，后来priority变2，随后产生新合格packet | 计响应；不会追溯使创建时合法job失效。priority自身的新job仍自然需card2。 |
| t⁻资格、同钟postevent job/report | 旧资格不借同钟交付；同钟合法新job可响应但measurement仍>event；same-ID同钟正增益与确有新报告的recovery可0elapsed。 |
| static同钟bad→good，后30s保持、迟到不足30s | 每个receipt prefix检查；bad不能被同钟后good抹去；不足完整30s仍按600删失。 |
| dynamic原格与短剩余窗口 | 完整原60格、十连续STARTS≥event、quality≥.5、sealed upload与末格end取max；不重开grid，短窗口不删eligible。 |

责任区源于固定公开几何，不是某方法表现导出的eligibility。该开发解释在所有方法上共用；条件曲线仍必须同时报告eligible/NA/censor分母，不能把条件样本差异当成因果优势。

## 原始配对、已知失败与唯一retry

三份已完成原world逐一实际调用原frozen evaluator.v2，均exit0、完整score字典精确相等；没有重跑物理。score一致与episode准入分开判断：

| 原始attempt | 状态 / clock max | 准入与用途 |
|---|---|---|
| event attempt-00 | completed / .0209994167 | 准入pair，1次event实际尝试。 |
| clean attempt-00 | known_operational_failure / .2283927090 | 超过固定.1；terminal.Y=null，只有clock_lag_exceeded，保留原score/全部记录但不用于pair。 |
| clean attempt-01 | completed / .0739885409 | 同一plan/配置下唯一允许retry，准入pair；剩余retry=0。 |

event目录恰attempt-00，clean恰attempt-00/01，没有未知、追加或被删除的attempt。每个intent/started/terminal的plan binding、ordinal、argv的case/policy/realtime/600s/out、stdout及六类record hashes均一致；spawn在intent之后，retry intent晚于第一次terminal的现存mtime。fresh validate_flight的Y/evidence与terminal精确一致。第一次失败不是策略Y0，也没有被成功重试抹掉或拿来择优取分。

三个world都是600s/144000steps，recorded source/dependency/private-case guards通过；run内嵌17源逐条SHA匹配plan，三attempt的17源及61dependency哈希表相同。这里核对的是固定指纹/原guard与完整score，不新做物理/硬件attestation。

episode的plan_sha256是adapter定义的sorted JSON默认分隔符规范，event为`ba3a4ba9b4110db44d7d4003513678e70cf9b43f4b3a26e6674fd13f05177c50`，clean为`c84756b4c832dfef9da5dd2d4403363dcfa03aec0c75adfb4a671ed8abaacf28`；它不同于plan文件字节SHA，也不同于owner compact canonical。审计初稿误套compact规范导致三个plan比较为false，已按实际adapter规范独立纠正，原回执无需修复；前后审计记录均保留。三份frozen score结果没有因此重复运行或改写。

## 几何、共同日历与archive

两条准入飞行的case_id/lineage、seed、ROI、wind、完整entities/motion、schema/domain一致；clean仅移除九个私有干预字段，并按已登记派生将level标为L2。clean仍属于原OOD-L3 root的同一lineage，没有增加或转移一个独立root。

当前B0源码不读取level字段；该公开L3→L2标签差异在派生记录中明确存在，不把它隐去。这里只验收当前B0开发配对，不据此认证其他会读取level的LLM条件或更广义纯机制因果结论。

| 描述量 | 独立重算 |
|---|---:|
| event原Y | .7809919967814705 |
| clean原Y | .7812312312312312 |
| clean按event demand calendar重价Y | .7985358564305932 |
| event减clean共同日历Y | **-.01754385964912275** |

common-calendar首先精确复现clean原完整score，再以event demand重算；与pair.json的整个common_calendar_recount（含分母/原格/实体结果）完全一致。原clean/world score未被覆盖，重价不是新rollout，也不是用clean原Y直接相减。event primary score保持原值。

完整clean archive SHA为 `402fff69e5b9d03b81c0033daa6a17001af3224aed1ea3d791144c99c037e38e`。独立流式检查成员集合、去重/路径/链接、内部manifest、每成员SHA以及对应原文件SHA全部通过；两个attempt、失败terminal、stdout、progress、run/world/private case与派生记录均保留。pair的event/clean terminal SHA及语义plan hash也与原件一致，cache_reused不代表额外飞行。

## candidate.v2与有限结论

candidate.v2绑定kernel6ac和原event world `044cfe5547d0ffefc172a75740427731f27a3f02ab0c374e9b4081e688bfdadc`，30行逐字段精确复现：6events×5required targets；6eligible、24NA、4response删失、5recovery删失。4s/9s/17s等短窗口保留原deadline，未通过NA筛除。原v1与924报告不覆盖，Y不因endpoint解释而离线改写。

该实际pair补齐了这一开发例子的已准入clean/event材料；实际eligible仍全为dynamic，静态恢复/区域未发现等边界主要由手算覆盖。不能把一个B0 root、30行或重复尝试当作30个独立样本，不能产生总体效应/显著性或资格覆盖所有root/method的结论。其他静态实际场景、全部方法/正式输入、完整统计与G4仍须其各自证据。

## 实际执行与证据

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-event-scope-pair-review-v1/scope_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_event_endpoints.py -q -p no:cacheprovider --basetemp /tmp/inspection-event-scope-pair-review-v1/pytest
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B /tmp/inspection-event-scope-pair-review-v1/pair_audit.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-event-scope-pair-review-v1/correct_hash_spec.py
```

pair_audit内部仅顺序启动3个原 `experiment/evaluator.v2.py --world ... --expected-world-sha256 ...`；cwd=/tmp、single-thread数值环境、nice10，stdout/stderr全部保存。另执行只读inline receipt-chain核对，保存receipt-chain-check.json。未重做大型grid、生成、物理或tokenization。

source/scope、pair与candidate快照、手算/回执/全score重计输出和完整输入SHA保存在 `review/evidence/event-endpoint-scope-pair-v1/`，`/tmp`原件保留。原大型world/archive不改。结论限于本次固定语义、原始回执与n=1开发配对；**不盖main/G4，也不把描述性差值称总体事件效应。**
