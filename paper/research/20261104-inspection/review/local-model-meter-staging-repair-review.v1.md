# Local model meter / staging：独立定向修复复审 v1

结论：上一版五项 meter/staging 缺口已按原反例闭合；本轮新增 smoke probe 的两处绑定缺陷也已在修补后定向闭合。相关68项测试通过，正常、较小实际预算、撤销和拒绝覆盖回归通过。本结论限于已执行的纯CPU/fake边界，**真实模型 generation=0、GPU forward=0、PyFlyt物理=0**；不是G4、真实smoke、episode真实性或外部unknown恢复批准。

本报告由同模型家族独立上下文执行工程复审。仅写新报告和evidence，不改生产源码、owner/live/真实 `.paper`、权重、经典cohort或旧报告/旧结果。主 Agent在本轮修补了两处probe绑定，前后源码和反例分别保存，没有回填成同一版本。

## 实际版本绑定

工作树 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`；HEAD `270c6f0946b3be59527b2b77982f75a70da7ef77`。以下为实际文件SHA-256，不能仅靠HEAD代表这些新增源码。

| 文件 | 最终审查 SHA-256 |
|---|---|
| `eval/inspection_model_receipts.py` | `922372548ceb972e9f95d1c52923cfe7d9dd6ce12dba2feec3b2d44a99167683` |
| `eval/inspection_model_service.py` | `64e8307ece55b1d6928e91607a269dfb082926265f1d98a2bde90a5931eb2e8f` |
| `eval/run_inspection_llm_development.py` | `582eb76c2420b7831dd1cb434d80ec8f8d3f04ad89fb5a5a11e90e73b6b24cdc` |
| `eval/probe_inspection_model_generation.py`（最终） | `f5f6e2193d65c599857274f6ac6c9650e590218e44f6883e2f1e49fde54f0c1f` |
| `eval/probe_inspection_model_generation.py`（本轮初读） | `8a693da5618542a11182088a01123a5b5692bb2627ec46584f049db7b9ee9716` |
| `tests/test_inspection_model_receipts.py` | `4adf0e0358d8d49514f142d4f288d57d17626357058d7a3971456a67da3d638f` |

初读完整源码在 `review/evidence/local-model-meter-staging-repair-v1/source/`；最终probe另存 `source-probe-final/`。client、wire维持前次精确版本，完整哈希见binding。没有重新加载真实模型或重审完整G4。

## 原五项闭合结果

| 原发现 | 定向实测与结论 |
|---|---|
| M1：completion数字类型宽松 | 原calls=True和calls=1.0变体现均ValueError；meter122–127显式要求usage恰有calls/tokens且每值为native int，随后按目录/日志重算的int累计值核对。原不合法completion不再被接受。 |
| M2：token行重复key | 原一行含token999/token7变体现为UnresolvedModelOperation。meter20–39的strict object parser复用于88行JSONL，重复key/非有限常量不能静默取末值。 |
| M3：只看最终tokens，未看准入headroom | 原input3/maxout1024/max_total4/observed1变体，expected和intent实际预算同步设为4，现明确报“call could not have passed…headroom guard”。因此不是靠expected不匹配提前掩盖原缺陷。meter85–86逐call使用此前used+input+configuredMaxOutput核对实际MaxTotal。 |
| M4：未知operation树忽略 | 原unexpected-call/request.json现ValueError。meter67–77只允许声明的root条目与每call三文件；根 `.DS_Store` 普通文件正例通过，call内 `.DS_Store` 被拒。原未知目录不再被当作无工作忽略。 |
| M5：actual较小配置未绑定 | service129–139把实际CLI calls/total/input/output/seconds与temp/top_p/thinking共8字段写入intent.request_budget；meter55–64与预派发expected一致并检查reserved覆盖配置。独立检查wrong actual out32 vs expected64被拒，reserved.calls1而配置maxcalls32被拒。实际较小1call/16384tokens/64out正例通过。 |

原normal fixture仍重算1call/4tokens。runtime-observed2/journaled1的已知runtime差额测试仍通过，不把未journal部分免费删除。上述accepted→rejected结果仅是receipt信任边界复验，不是实际GPU漏计/补计实证。

68项测试实际 **68 passed in0.67s**。本轮未改任何测试或生产实现；独立probe从当前fixture构造输入并修改相应字段。旧v1的源码、接受反例和报告保持原样。

## staging 与实际较小预算的 FakeCLI 回归

沿上一版隔离FakeMLX CLI，再加入真实准备使用的较小配置。它只安装显式ModuleType假模块，load/stream是本地纯Python函数；小blob为11字节 `not a model`，不是权重。service源码副本字节相同，`__file__`仅在当前隔离Python进程映射到 `/tmp`，故生产operation_directory计算出的写入位置也在该隔离repo/logs下。

| 场景 | 实际结果 |
|---|---|
| registry上限配置 normal | fake load1/stream1，meter1call/3tokens；第二次相同reservation在load前FileExistsError，既有全部operation文件SHA不变。 |
| 较小actual1call/16384tokens/64out | 同样meter1call/3tokens，intent与expected actual配置一致；第二次调用同样拒绝覆盖、全部SHA不变。 |
| 无frozen plan | load0/stream0，准入拒绝，无operation。 |
| blob核验后撤销 | 再准入拒绝，load0/stream0，无operation。 |
| fake load后、请求前撤销 | load1但stream0，请求前拒绝；intent保留、completion缺失，不能当免费成功。 |

无真实owner修改、预约、结算或真实logs下的新operation。runner本轮字段由generation_calls_are_actual改为admitted_local_request_count_is_receipted，未从calls>0推出输出token已发生；本轮zero-output smoke分支还独立核实不会称生成成功。

## 新 smoke probe 两处发现与修补闭合

初读版本 `8a693d…` 的两个具体反例均已发送主 Agent，随后才读取和执行最终 `f5f6e2…`：

| 反例 | 初读结果 | 最终结果 |
|---|---|---|
| P1：所选alias的receipt未校验 | 双模型冻结registry fixture中，M1 receipt路径被换成完整有效M2 receipt；reservation purpose合法绑定M2。真实admitted_limits及verify_model_files均选择/通过M2，probe却记录model_alias=M1并exit0成功。这里是模型标签/来源绑定缺陷，不是绕过owner获得未授权模型。 | probe43–44先将实际receipt SHA与所选alias的registry SHA比较，在FakeClient创建/submit前ValueError拒绝；submit=0、无operation、无真实模型。 |
| P2：transport与durable call未join | fake transport.status=completed，但唯一落盘response.status=runtime_failure；meter明确回传该call为runtime_failure且1call/4tokens。旧probe仍按transport+observed>0标成功。 | 唯一call数量、response文件SHA与meter binding、所有durable字段与transport均被核对；现operational_failure_or_unknown/exit1，保留meter1call/4tokens和差异错误，不标成功。 |

修补前JSON/log独立保存为 `smoke-probe-before-binding-repair.*`，初读源码副本未覆盖。最终alias早拒绝发生在probe intent保存前，故无probe.json；审查harness仅增加“若无receipt则记录早拒绝/实际raised”的处理，而未修改输入、admission或fake service。这一harness标签不是生产保存的终态，也不冒称未知调用已结清；实际submit=0/无operation表明该特定拒绝先于任何模型调度。

最终probe的正常/故障回归（均显式fake client）：

| 分支 | 最终状态、计费与证据 |
|---|---|
| normal | receipted_generation_smoke_not_capability_qualification/exit0；合成observed1，重算1call/4tokens，durable status=completed。不是实际生成样本。 |
| known runtime failure | receipted_request_failure_not_generation_qualification/exit1；1call/4tokens仍保留，不免费丢弃。 |
| completed但generated0 | 同上request_failure/exit1；1call/3input tokens仍收费，不声称有模型输出。 |
| transport unknown | operational_failure_or_unknown/exit1；probe异常、transport日志和operation intent保留；无completion/recount，不填零usage，不自动重试。 |
| transport/durable mismatch | operational_failure_or_unknown/exit1；完整meter仍保留并报告source-binding错误。 |

FakeClient初始化还实际调用纯CPU的admitted_limits和完整小blob核验，避免靠完全跳过准入制造alias反例。它没有启动MLX或子进程后端；fake input token3、单个合成chunk只是为有效meter形状构造，不能解释为真实tokenizer结果。消息/seed/完整episode的外部因果真实性仍不由这个纯fixture或request meter证明。

## 实际执行与范围

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-model-meter-staging-repair-review-v1/meter_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-model-meter-staging-repair-review-v1/budget_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-model-meter-staging-repair-review-v1/service_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-model-meter-staging-repair-review-v1/smoke_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_model_receipts.py tests/test_inspection_llm_actor.py tests/test_inspection_model_client.py tests/test_inspection_process_cleanup.py tests/test_inspection_llm_protocol.py tests/test_inspection_model_admission.py -q -p no:cacheprovider --basetemp /tmp/inspection-model-meter-staging-repair-review-v1/pytest
```

probe源、JSON/log与精确binding保存于 `review/evidence/local-model-meter-staging-repair-v1/`，原件保留在 `/tmp/inspection-model-meter-staging-repair-review-v1/`。service/smoke脚本各在本轮配置或源码变化后定向重跑，未运行长物理、GPU或付费API。68项测试不包含真实新probe生成，其新增两条绑定由上述独立fake main验证。

本轮已关闭列明的局部反例，没有剩余被本轮实证确认的同类阻断；不能扩大成完整平台资格。计费仍仅admitted local requests + submitted input + runtime-observed generated tokens，durable journal另列，GPU内部隐藏工作/资本/能耗/FLOPs未独立计量。完整LLM episode adapter、SDK入账、硬件/真实模型时延、full context和外部unknown恢复仍pending；真实smoke仍需既有有效owner预约并由另一次实际运行给出证据，本报告不代为批准、预留、结算或重试。**G4继续pending。**
