# 本机模型上下文与解释器修复独立复核

审查者 `/root/control_review`，同模型家族、独立工程上下文。窄审两个开发修复及必要 caller 行为；未启动 GPU、MLX generation、PyFlyt、真实 worker、owner 操作或新飞行。

## 结论

**本轮两项修复在所审范围通过，未发现仍需阻断这两个开发修复的实质开放缺陷。** literal venv 的 metadata 探针确认 `.absolute()` 保留正确环境；固定 public/tool/memory 分区阻止消融把空余 memory 换成更多公开观察；fake runner 实际处理32次 context failure 后仍完成、未派发模型请求或把原评分补成零。

这是代码/控制流复验，不是三模型的完整上下文、实际1×资源竞争、OOD 飞行或 G4 资格。14,000 UTF8 user payload 不是 8,192 tokenizer tokens 的数学保证，system/template 也在实际输入中；service 的精确 tokenizer/context/token gate仍不可省略。实际完整状态是否经常触发裁剪/拒绝，需要开发飞行证据，不能由小 prompt smoke 或本报告推断性能。

## 解释器路径：实际 metadata 探针通过

[run_inspection_llm_development.py:99](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/run_inspection_llm_development.py:99) 现将 `args.llm_python.absolute()` 放入实际 service argv，而不 resolve Python launcher 的 symlink。对本机两个实际路径只执行 `sys.prefix/base_prefix/executable` 和 `importlib.util.find_spec('mlx_lm')`，没有导入该包：

| 启动路径 | 实际 prefix | mlx_lm spec | MLX/模型导入 |
| --- | --- | --- | --- |
| 仓库 `.venv-inspection-llm/bin/python` literal absolute | 仓库 `.venv-inspection-llm` | venv `lib/python3.14/site-packages/mlx_lm/__init__.py` | false |
| 同一路径 `.resolve()` | `/opt/homebrew/opt/python@3.14/Frameworks/Python.framework/Versions/3.14` | null | false |

这独立复现原 runner 的环境故障原因，且 fake runner 捕获的实际 `LocalModelClient` argv[0] 仍为完整 literal venv 路径。不是仅根据 Path API 名称推断。此修改针对 runner 的 service 启动；原服务 smoke 若本来使用 literal venv，并不受该故障影响，但它们未导入 actor，不能被升级成 actor/真实飞行验证。本轮没有重新生成或重核其实际使用量。

## 固定分区与消融：全量压力投影一致

[inspection_llm_actor.py:18](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_llm_actor.py:18) 固定 public 10,000、tool 1,500、memory 2,000、whole-user 14,000 UTF8 bytes。第132–153行统一保留 task card、telemetry、全部 queued/running jobs；仅整条删除较旧 terminal jobs/观察，不截数值或伪造新观测。第176行先 deepcopy 完整投影，再分别裁 tool/memory；memory 空余从不增加 public/tool 限额。第185–198行裁剪整个旧 exchange/alias/note/plan条目，不改原持久 state。

独立构造同一公开输入：48条多字节大 observation、14条 terminal+4条 active job、最大CJK plan/note及两轮 history、大 tool history。对 B3、B4、periodic、no-cross-aircraft-memory、no-progressive-confirmation-guidance 五种 actor，逐字段比较 public_state/recent_tool_results，全部相同；所有 active job 与 card 保留，原 plan/note/history/tool_results/state 全部未被裁剪过程修改。

| actor | public bytes | tool bytes | memory bytes | whole-user bytes |
| --- | ---: | ---: | ---: | ---: |
| B3 | 9472 | 520 | 23 | 10065 |
| B4 | 9472 | 520 | 1648 | 11690 |
| B4 periodic | 9472 | 520 | 1648 | 11690 |
| B4 no-cross-aircraft-memory | 9472 | 520 | 55 | 10097 |
| B4 no-progressive-confirmation-guidance | 9472 | 520 | 1648 | 11690 |

五者保留相同的最后两条 observation（从 e-46 起）；memory 消融没有因减少约1.6KB memory而额外获得感知。periodic 只改变调度机制；memory 消融移除计划/note/alias summary，仍使用相同原始公开输入；no-progressive 只移除 policy 的 two-view 指导，不改变公开工具或共享 grader。这里验证软件投影/具名因素，不证明模型实际采用这些机制或消融已取得 G4。

## Context failure 与预算/结束行为：fake runner 实际验证

[actor.start_request:159](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_llm_actor.py:159) 在投影前消费 decision attempt、记录 last_started 并清 triggers；失败不会回退额度，也不会留下 inflight。runner [第168行](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/run_inspection_llm_development.py:168) 只记录 `policy_context_failure`、真实 attempt 和 `model_request_dispatched=false`，随后继续 worker updates。

本轮用真实 runner main/actor，替换 worker、mp context、client、operation目录和依赖 metadata 为显式软件 fixture。未启动子物理/模型进程、未调用真实 service/预算准入；fixture world 的 Y=.37 只是用于检查是否被重写的 sentinel，不是实测成绩。模拟61条 t0..600 public updates，每次提供超额 essential card，并持续新增公开 candidate 触发：

- 实际 journal 恰好32条 context failure，attempts 顺序1..32，之后不再尝试；61条状态继续被处理。
- 模型 submit=0、mutating dispatch=0，fake service completion明确 calls/tokens=0；不是将32次策略尝试误报成32次模型生成或资源消耗。
- runner return=0、manifest.status=completed，原 fake world 的 Y仍为 .37，没有 exception-abort、免费 retry 或补成 Y=0。
- fake client 捕获 argv 证实 literal venv 参数；检查 sys.modules 无 PyFlyt/pybullet/MLX 导入。

纯 actor 另核 service `not_started`：attempt保留、inflight清除、无命令，max_calls=1后不再 due；所以精确 tokenizer/总token gate 拒绝也不会产生免费再问。已读 service/client caller：service [第159行](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_model_service.py:159) 在 forward 前核精确token与预留输出，失败返回 not_started；client加实际端到端时间，actor将其作为 noncompleted reply忽略。该检查不等于本轮重新审计模型下载、准入/settlement或原smoke账本。

对 JSON 中 `note="\\ud800"` 的实际解码反例，返回 `policy_format_failure`、无命令、history为空、inflight清除、attempt=1且不再 due。说明第225–228行的 UTF8 检查是在污染持久 memory 前生效，而非仅由日志默认 ASCII escaping掩盖。

真实 runtime_failure/transport unknown 仍是另一类需保留使用量并对账的操作故障；本修复没有将它们改成合法低分、零消耗或无限 retry。decision attempts、actual admitted model calls和worker评分必须继续分列。

## 证据与未证项

读取主 Agent `/tmp/inspection-context-segmented.v1.log` 的73 passed/.68s；本 Agent未冒称重跑全部73项。独立实际检查为上述metadata双路径、五种压力投影、真实 main 的 fake worker/client、not_started、Unicode反例。临时源快照、metadata JSON、fake_runner_check.py/结果、projection-result.json、unicode-result.json保存在：

`/var/folders/n_/2cw9gf5n52z_w6kllyb44ttw0000gn/T/inspection-context-interpreter-review-dv8f9b8n/`

| 来源 | SHA-256 |
| --- | --- |
| repo `eval/inspection_llm_actor.py` | `b22b93afcaeb7b8ec24b630e935591e0dcae98523f37e75ad9923d23e2382b80` |
| repo `eval/run_inspection_llm_development.py` | `859c5edab7085c72740b7bada5d5b98b193a4ecc67892b443c14854dcb6a58a3` |
| repo `eval/inspection_llm_protocol.py` | `c323f708fd2c2eb54a440758bd162785f3ee2fcebd95b9bca21eaf7ed6ef44d4` |
| repo `eval/inspection_model_client.py` | `40ccd5f5f4100e4ed42a6ac22ecf35f3bc24a9db4e2b57f1826625b78afcb158` |
| repo `eval/inspection_model_service.py` | `64e8307ece55b1d6928e91607a269dfb082926265f1d98a2bde90a5931eb2e8f` |
| repo `tests/test_inspection_llm_actor.py` | `efcdb93474eed3b4798683887da618f5c15e24311eeda5c5f9c9cf6c695e92fd` |
| `/tmp/inspection-context-segmented.v1.log` | `d1c1a3a81b4da3f4f01eabf307fa9bdfab8a4f10e6569114a70740c41f6c0f52` |

仅新增本报告和自有临时软件证据，未改 production、owner、live logs、frozen core/classic、原模型使用量或旧报告。未验证真实600s LLM飞行、实际GPU/physics争用、三模型max-context token分布或完整G4；没有正式test/holdout、真实vision、领域人工或跨模型认证结论。
