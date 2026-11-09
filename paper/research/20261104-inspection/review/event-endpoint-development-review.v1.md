# 开发事件端点：独立有界审查 v1

结论：初读版本发现了**静态30s保持的确定错误**及同钟边界不一致；修补后的924dd版本已关闭列明反例。动态原60格、十连续格、实际sealed arrival和固定deadline删失在已查边界正确。**survey责任范围、same-ID增益基线及correctcard时间锚仍未由本报告认定为冻结合同的唯一含义，G4事件端点资格不能据此通过。** 当前只有一条event trace的候选分析；未验收完整实际clean/event配对，更未进入main图表统计。

这是同模型家族独立上下文工程审查。全程只读原数据与源码，只写本报告/evidence，新增物理/模型生成均为0。未改score、旧candidate、owner、主试总体或其他作者文件。没有把clean的首次clock失格或未验收的retry当作完成pair。

## 契约、版本与实际数据绑定

规范为accepted `experiment/design.ral.v3.json` 的 `secondary_contract.response_recovery`，实际SHA `83f8f1a72f5261aaa5783c0d190f3788fafcd05bca1e7d16a6901c69c9abf0de`。工作树为 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。

| 来源 | 实际SHA-256 |
|---|---|
| 初读endpoint source，原candidate的实现 | `6e1939858590805705e6dc81a7a2038c91ac866d0954bbe14349d71309dbd934` |
| 本轮修补后endpoint source | `924dd68026e1e5bb646b7297ba10724ac8e6ebdf45fc224bef86acf70e4e7d63` |
| 修补后event tests | `178eb3c052568ecb7f3cad5b66ecb8f139a0273347213428a70bcb129cae10b8` |
| 完成的OOD-L3-B0原world | `044cfe5547d0ffefc172a75740427731f27a3f02ab0c374e9b4081e688bfdadc` |

原source/旧tests/设计/原30row candidate最先已快照。修补由主 Agent完成；本审查没有改生产。旧candidate仍绑定6e193，未把其source SHA或rows回填为924dd；本轮新源码下的离线比较仅保存在审查evidence。

## 确定反例及修补复验

| ID | 初读问题 | 924dd定向结果 |
|---|---|---|
| F1，P1：同刻坏版本被后续好版本掩盖 | static_recovery按arrival timestamp对全versions算quality。event5、good10/order1、bad20/order2、good20/order3，返回10；合同要求后30s **所有versions**≥.6，bad20破坏10..40，最早应20。 | 逐接收prefix检查，返回20/elapsed15。完整初读反例保留，不因实际trace无static eligible而忽略。 |
| F2，P2：声称t⁻但使用事件同刻交付关联 | 原partial reports及candidate↔truth join用≤event；单位fixture只有一条arrival=event的candidate观测，资格由NA变eligible，与仅“事件前已交付”说明不符。 | 改为reports.t<event、obs.arrived_at<event；同刻观测不再赋资格。 |
| F3，P2：排除合法同钟postevent新job | worker在event已生效后，可于相同sim clock创建新card合法job；原created_at≤event拒绝了created_at=event、measurement>event的合法响应。 | created_at=event可，正确card/rule且后续新≥.8 packet计响应；旧card拒绝；measurement仍必须严格>event。 |

F1不是不可达格式：真实worker每个控制tick最多处理16条RPC，实际event trace也有大量同钟报告journal项。当前实际候选的6个eligible目标全为dynamic，因此其静态NA行不能实证覆盖这个静态保持缺陷。

## 同钟相位与t⁻口径

实际worker先在当前mission.t处理工具，再调用mission.step。step先 `_events→advance→integrate`，到达新t后再次 `_events`，然后才sample、deliver、refresh_public。下一loop在该t处理的工具，以及该t的自动交付/报告，都位于event生效之后。online/camera采用[start,end)半开区间，退休在事件时刻已生效且停止新采样。

因此，将**资格**的已有job/已收信息取t⁻是因果一致的确定化：job.created/started<event、finished≥event；reports/obs仅取<event。无需把事件同刻新delivery反灌到资格。它是对合同“at eventtime”的开发期边界约定，应明确记入实现版本，而非暗称原合同已经逐个指定了比较符号。

同时，**响应/恢复**可接受该时刻新收到的版本：同钟postevent的新合法job仍须产生measurement>event的packet；已有ID.created<event的静态revision若产生正增益，可在arrival=event计0elapsed；静态recovery若确有该时刻新received version≥.6且全后30s保持，也可为0elapsed。不能仅因事件前已有good而自动计恢复。旧ID限制只属于same-ID response例外；冻结static recovery本身没有旧ID限制，新ID的合格received report也可恢复。

独立fixture确认：sameclock revision=10、static recovery=10/elapsed0；只有旧good且没有新version仍删失600；新ID的同钟合格报告可恢复；坏card与同刻measurement仍被拒。这些0是sim clock量化下的端点，不是硬件零延迟证明。

## 与冻结合同的逐项判断

| 契约部分 | 已查实现 / 状态 |
|---|---|
| 资格 | priority只看变更rule、active且未满足static .6/dynamic .5；hardware事件要旧active inspect/observe/survey；无postevent新job不直接变NA。t⁻接收/作业边界修补通过。survey→target范围仍需下述澄清。 |
| packet响应 | 按实际新measurement时间，不把ACK/queued/running或无新包的job完成当响应；质量≥.8、connected/nonretired、deadline、candidate/rule/mode等限制保留。job失败但有合法新包仍可响应。correctcard锚当前取measurement时刻，非合同明示唯一含义。 |
| dynamic recovery | 完整60个原index/连续3s grid，十连续格STARTS≥event且各quality≥.5；只用原score选择的sealed evidence。端点=max(末格end,所有所选sealed arrival)，不重开grid、不用预测。当前event world的原score先精确重构，因此这些bins与该event demand一致。 |
| static recovery | 每个received version按prefix≥.6，逐后续version保持满30s；候选最迟570。590才合格而没有满30s，仍保留eligible并在600右删失，不以短随访例外删掉目标。 |
| censoring / NA | static deadline600；dynamic min(600,appearance+180+6)。未达标在固定deadline删失，存elapsed/censorflag/horizon；短剩余dynamic窗口仍eligible并可能结构性删失。NA行有reason且endpoint为None，不能变为成功或从总分母消失。 |
| 单位与资格guard | 输出event×target明细，不是独立样本；本模块没有完成rootcluster曲线/受限均值/正式统计。合同明确需要手算及完整实际clean/event开发兄弟验收，当前不具备整体通过证据。 |

dynamic helper的原grid与sealed版本真实性依赖上游完整score/receipt绑定；不能把任意传入的60行数组当作已认证原grid。当前接口绑定actual case/world并核原score，不应通过替换private_case/score假装生成一个clean/common-demand新rollout。后续clean/event分析须明确保存实际geometry、event demand及原score重构/公共日历结果的分离来源。

## 尚未关闭的解释项，不按当前30row挑口径

**D1：survey义务scope。** 当前924dd把affected running survey扩展到所有active未满足target，而job本身带region_id。“affected obligation”没有在冻结合同中明示为整个任务域；也不能由一次trace的分母反向证明。应明确survey的固定责任区域/目标映射或明确的任务级范围，作为所有方法共同规则，不能按实际FOV、是否成功发现或未来包决定资格。当前全域候选规则只可保留为development解释，本报告不替其盖合同已实现。

**D2：same-ID正增益的基线。** 当前924dd是高于紧前sealed prefix且>0，没有要求高于event前quality。独立fixture：event前quality1/3，event后先降0再回1/3，当前返回一次response。严格正增益可以排除幂等重复，但“new positivequality”是否要求超越event基线尚未钉死；本报告不因actual trace全static NA而把这个选择当作已验证。需要先明确基线及合法新version要求，再增加相应fixtures。

**D3：correctcard时间锚。** 当前按packet measurement时刻选择version。fixture：hardware event10，合法card1 job创建12，后续priority20，packet measured22；当前拒绝。若合同意为job创建时合法，后来priority不应追溯令该job失效；若意为measurement时必须仍匹配当前card，则需明示。冻结文字没有把此锚写细，不能静默当已决定。

主 Agent已提出下一开发版本的scope/baseline/card澄清方向；本报告按其要求先冻结924dd，**未把尚未实施的新口径代入原30行或称其通过**。它们不改变主任务分数/物理/总体，但会改变次要端点分母/时间，必须在main之前固定来源与版本，并按合同ambiguity_guard完成下一轮手算和actual sibling核验。

## 完成的event trace只读重算

原world来自 `logs/inspection-population-dev-flights.v1/devext-OOD-L3-00-B0-0/attempt-00/flight/`，run/world completed，600s/144000steps、wallclock1x、lag记录≤.1；原source/dependency/private-case guards均记录True，world/private case SHA与candidate/run绑定。原score被重构为完全相同后，6e193原candidate的30行逐字段精确复现。

| event | target总行 | eligible | NA | response删失 | recovery删失 |
|---|---:|---:|---:|---:|---:|
| priority | 5 | 2 | 3 | 0 | 1 |
| link_start | 5 | 2 | 3 | 2 | 2 |
| link_end | 5 | 1 | 4 | 1 | 1 |
| camera_start | 5 | 1 | 4 | 1 | 1 |
| camera_end | 5 | 0 | 5 | 0 | 0 |
| retirement | 5 | 0 | 5 | 0 | 0 |
| 合计 | 30 | 6 | 24 | 4 | 5 |

六个eligible全为dynamic。独立按其60原格与sealed arrival重算recovery一致：priority/truth-001于298.5恢复，其他5行按各自308或352 deadline删失；4s/9s/17s等短随访没有被改为NA。没有把link恢复或camera恢复本身当business recovery。

924dd离线复算仍为30/6eligible/4response删失/5recovery删失；相对原v1只变了link_start/truth-001的quality_received_at_event字段（t⁻接收边界），不是Y变化或新策略样本。原v1文件不覆盖。该一致性不证明D1–D3口径正确，也不完成clean/event pair；本审查未读取进行中clean retry作为终态或声称事件因果效应已算出。

## 实际执行与证据

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-event-endpoint-review-v1/counterexamples.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-event-endpoint-review-v1/audit.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-event-endpoint-review-v1/repaired_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-event-endpoint-review-v1/interpretation_probes.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_event_endpoints.py tests/test_inspection_episode_receipts.py tests/test_inspection_process_cleanup.py -q -p no:cacheprovider --basetemp /tmp/inspection-event-endpoint-review-v1/pytest
```

实际组合 **37 passed in1.00s**；修补后event文件单独collect为20项，其他所选episode/cleanup共17项。测试与probe均无新物理/模型；import链打印pybullet build信息不代表构造或推进了world。首次资格fixture缺case_id，补全审查fixture后复验，不把它当生产问题。

初读与924dd源码/tests、accepted design、原candidate、手算反例、完整重算摘要和命令日志在 `review/evidence/event-endpoint-development-v1/`，`/tmp/inspection-event-endpoint-review-v1/`原件保留。端点实现缺陷已按此版本定向关闭，语义与完整pair资格仍pending；不得用于main图表统计、独立样本扩增或G4通过声明。

收尾时检测到共享工作树已进入主 Agent的下一轮语义修订：endpoint实际SHA变为 `6ac62f8d266966420bc3457d4b7992ae5755094bbfdd9b7928cf73dc1cd341b8`，tests为 `d3010361235f9f69b0eb6cfca874c300bc20857c5e28cfd0a430453b75781ea8`。本报告按要求冻结已审924dd，不追认这两个后续字节；bindings明确保存该漂移，旧candidate/设计/world输入未变。后续scope/baseline/card版本需另轮审查，不能把本轮结论回填新源码。
