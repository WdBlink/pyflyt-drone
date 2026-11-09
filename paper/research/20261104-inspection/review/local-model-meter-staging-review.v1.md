# Local model meter / staging：独立窄范围审查 v1

结论：staging 路径迁移、已有 operation 拒绝覆盖、已有准入/撤销和正常 fake main 回归通过；**meter 仍有四类接受缺口，以及实际较小 request budget 未绑定的问题，需修复后另文复验**。61项相关单元测试通过，不能覆盖本轮新增反例。真实模型 generation/GPU forward/PyFlyt物理均为0；本报告不批准 G4、真实 episode、外部 unknown 恢复或实际 GPU 计量资格。

本审查为同模型家族独立上下文工程审查。只写本报告及 evidence，未改生产、owner/live/真实 `.paper`、权重、经典 cohort、旧回执或旧报告。所有 owner fixture、operation、blob 都在 `/tmp`。FakeMLX 的1个合成 chunk不算真实模型生成。

## 源码绑定

工作树 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`，Git HEAD `270c6f0946b3be59527b2b77982f75a70da7ef77`；新增文件以实际内容SHA-256为准。

| 文件 | 实际 SHA-256 |
|---|---|
| `eval/inspection_model_receipts.py` | `e267d191cc1c718cb02de09e02d4c8d1d30e714b1fa312c0006cef705d91e41e` |
| `eval/inspection_model_service.py` | `ea0e9f5c7c14fb7201561ebd54c1cfe29093d0bbdd0354586ec2de149613cd87` |
| `eval/run_inspection_llm_development.py` | `0f1de6896c4b1682584923b851e56cc271f84a9edd99510fc1e02b458fbd678f` |
| `tests/test_inspection_model_receipts.py` | `cb002dac8af47b1d53f0bfa75221ccdcddc45880dfd693e8426b72caa2d5ebb1` |

actor/client/process_cleanup/wire 保持上一修复复审版本；完整哈希和源码副本在 `review/evidence/local-model-meter-staging-v1/source.binding.json` 与 `source/`。对上一冻结 service/runner 逐字节 diff 也保存：service只新增 operation_directory、迁移路径和 owner/PID/PGID intent metadata；runner只新增同 helper import 并更换 operation lookup。没有把新 meter 的检查倒灌到旧资格。

## 接受缺口与开放项

| ID | 定位、可复现证据及影响 |
|---|---|
| M1，P2：completion 数字类型不严格 | meter100–104 只用字典 `!=` 比较 totals，Python 的 True/1.0均等于1。将正常 fixture 的 completion usage.calls 改为 True 或1.0，仍返回已重算 usage={calls:1,tokens:4}。返回值虽是重算 int，不会凭这两个反例证明实际漏计，但不合法 completion 已被当成一致回执接受；需严格 native int、键和范围校验。 |
| M2，P2：token JSONL 不使用严格 JSON reader | meter67 用普通 json.loads，区别于其他 receipt 的 duplicate/nonfinite 拒绝。将一行 token 改为 `"token":999,"token":7` 后仍接受并以末值7处理。必须拒绝歧义行，而非静默挑字段；本次未据此声称 GPU token 消失。 |
| M3，P2：未复算每次准入 headroom | meter只检查最终累计actual tokens≤caps，未执行 service154–155 的 `prior_used + input + max_output <= max_total_tokens`。合成预冻结 max_total/reserved tokens=4、input3、max_output1024、observed1、completion4，meter接受；实际 service 对该请求应 not_started，不可能产生这个 admitted call。需按call序列重算当时的实际 request budget/剩余预算门槛。 |
| M4，P2：未知 operation 树内容被忽略 | meter52 只glob `call-*`；新增 `unexpected-call/request.json`，包含一份额外请求形状数据，meter仍只计1call/4tokens并成功返回。这不是额外实际GPU工作实证，但严格已完成 operation 验收不能静默略过未知请求/目录。应按已声明树验收，主 Agent已确认仅允许忽略.DS_Store。 |
| M5：较小实际 request budget 无独立绑定 | service115–118本来允许CLI calls/output/tokens等≤registry上限；当前intent130–135没有实际CLI budget，meter62又要求request.max_output_tokens等于expected generation_profile.max_output_tokens。若expected用registry上限，合法较小输出配置被拒；若expected另填有效配置，intent没有calls/tokens等实际运行配置可与它核对。主 Agent已明确第一次真实smoke将用较小上限，应在intent记录实际request_budget，并和调用者**预先冻结**的实际配置相等，再用registry约束其上界。此项是源码接口缺口，不冒称已完成一趟真实小预算运行。 |

主 Agent已接受上述修复方向。此报告冻结当前源码和接受反例，未等待修补后回填。M1–M4结果均在 meter-probe.json；用保存的独立脚本重复一次得到相同接受结果（meter-probe-repeat.json）。expected是隔离的、事先构造的可信fixture，不是从待验证intent自动生成；M3同时调整该合成expected和相应intent，以检验完整预算规则，而非假冒真实owner修改。

## 已通过的 meter 与 staging 边界

meter的正常逻辑会核对预期 reservation、owner_frozen_plan、model receipt SHA、reserved和service SHA；要求call目录连续命名；逐call重算submitted input + runtime-observed generated，检查token行计数/文本、response counters/status/deadline、最终completion和reservation上界。正常合成fixture重算1call/4tokens；runtime_failure 的observed2/journaled1可保留差额，而不是把未journal部分免费删除。它信任已固定service的本地运行计数，不是独立测量GPU内部工作。

13项meter测试覆盖缺response、截断journal、文本/input/source/model/owner/totals不符、late completed、token bool、call gap与正常/已知runtime差额；并入既有测试实际 **61 passed in0.66s**。既有测试通过与上表新增接受缺口并不冲突。

`operation_directory` 的实际只读返回值为：

```text
/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/logs/inspection-model-resource-operations.v1/<reservation>
```

空串、`../x`、`a/b`、`a.b`、bool均被拒绝。固定位置在research根R之外；此次只读取实际helper的返回值，**未在真实logs下创建任何operation**。服务probe将模块 `__file__` 映射到 `/tmp` 中字节相同的源码副本，因此实际调用该helper得到隔离repo/logs位置；runnerprobe则把共享helper映射到隔离staging路径。不是改写真实helper/磁盘源码。

隔离FakeMLX CLI实证：

| 场景 | 结果 |
|---|---|
| 正常owner/registry/完整小blob + fake请求 | Fake load1、fake stream1，真实generation0；intent含owner frozen digest/reserved/PID/PGID，completion正常，meter重算1call/3tokens。 |
| 同reservation第二次CLI | mkdir exist_ok=False在第二次load前抛FileExistsError；既有operation全部文件SHA逐项完全不变。无覆盖或免费新调用。 |
| 无frozen plan | 准入ValueError，fake load0/stream0，无operation。 |
| 完整blob检查后撤销 | 再准入ValueError，fake load0/stream0，无operation。 |
| fake load后、请求前撤销 | load1但stream0，请求前准入拒绝；保留intent，无completion，属于未完成operation，不能计免费成功。 |

probe只安装显式ModuleType FakeMLX后调用main，未导入真实MLX。所有blob只是11字节`not a model`，逐SHA检查的对象不是实际权重。第一次调试CLI fixture漏传必填max_calls/max_total_tokens，被argparse拒绝；补齐明确参数后才执行上表结果，这不属于真实模型attempt。

新staging下还回归上一轮主循环fixture：合法600s/144000-step fake world和usage shape正常completed；normal reply入队1条，late/physical600回复0条；known failure/unknown transport为operational_failure，not_started不发命令；private/source/dependency变化或缺失、world缺失/599s/[]/null、坏/缺completion和bool usage仍被拒。19个main分支及三个CLI消融接线结果均保存。它们是分支测试，不是真实600秒、1×、模型时延或业务样本。

## 实际执行与资格边界

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-model-meter-staging-review-v1/meter_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-model-meter-staging-review-v1/service_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-model-meter-staging-review-v1/main_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_model_receipts.py tests/test_inspection_llm_actor.py tests/test_inspection_model_client.py tests/test_inspection_process_cleanup.py tests/test_inspection_llm_protocol.py tests/test_inspection_model_admission.py -q -p no:cacheprovider --basetemp /tmp/inspection-model-meter-staging-review-v1/pytest
```

meter反例最初以inline纯CPU脚本运行，随后保存为meter_probe.py并重复验证，前后JSON均保留。证据交付副本在 `review/evidence/local-model-meter-staging-v1/`；`/tmp/inspection-model-meter-staging-review-v1/` 原件保留。Fake OS进程由既有测试清理完毕，本轮无真实service进程或物理进程。

本模块是read-only recount，不写owner usage ledger、不批准预约、不结算、不自动重试。caller仍须在dispatch前从可信owner/frozen plan固定expected，并在未知/部分日志时保留资源债务；不能用已完成函数的失败代替零使用。当前函数不是runner中完整LLM episode adapter的替代品。实际模型软件/硬件、full-context、request↔actor↔command↔world因果绑定、episode/SDK真实性、外部unknown恢复仍未通过。计费定义仅 admitted local requests + submitted input + runtime-observed generated tokens；durably journaled另列，资本、能耗、GPU/FLOPs均未测量。**G4继续pending。**
