# Shared population law / formal pre-draw component：独立有限审查 v1

结论：**本次限定组件审查未发现实质阻断。** 共享规律拆分保留原有 joint proposal/rejection、5 个随机流、1000 次有限外层尝试与全部内容校验；正式 wrapper 在生成入口前调用已审 namespace predicate，并使用该 namespace 的精确 root/seed。结论仅针对未安装 source 候选，不是正式数据/完整 writer graph、G4 或 dispatcher 准入。

## 绑定及独立执行

reviewer `/root/statistics_design_review`：独立上下文、同模型家族；不是人工或跨模型审查。本报告依 autoresearch 证据边界作限定工程复核，不执行主实验。输入 freeze SHA256 `a098c81c91a11c0120d9d519c7860c965cacc89bc41d2f33019165a7442b8f1e`；其全部 files 和 working_sources SHA 重新核对一致。

| source | SHA256 |
|---|---|
| 原 production `eval/inspection_population.py`（字节未变） | `bb7098554475a65d4717077b7704bcf5902cd0fc674c1a9a95a51a7e066a5ff7` |
| candidate `inspection_population.py` | `afa5f70650b230fe8eff826b409e28d8d9e41a96b82435b2c92aed7fd38797fd` |
| candidate `inspection_formal_population.py` | `5e231f1856c586fdfc2a3b8dca32b2599a71d9f009ea5a3d6daca33231a543eb` |

逐行阅读 source diff、candidate content/generation 正文与 35 行 wrapper；定向读 namespace predicate 的 root metadata、native owner、source、seed/collision、holdout-opening 链及测试。namespace/admission/helper SHA 与 freeze 内列明的已审版本相同；没有另行扩大成其全图复审。

实际运行 Python 3.11.15 / NumPy 1.26.4：

```sh
PYTHONPATH=/tmp:$PWD PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider -p population_review_capture paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests/test_population_shared_law.py paper/research/20261104-inspection/planning/formal-admission-candidate.v3/tests/test_formal_population_gate.py --basetemp=/tmp/population-component-review-tests-v1 -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider /tmp/test_population_independent_namespace_v1.py --basetemp=/tmp/population-independent-namespace-fixtures-v1 -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/population-shared-law-static-review-v1.py
```

结果：**provided 22 passed in 6.54s；新增 independent 8 passed in 0.38s；静态 AST probe exit 0。** 作者 combined140 日志与绑定一致，本人仅重跑与本组件有关的 22 项，不把其余 118 项计为独立复跑。

## 共享规律：代码与五个既有 root 的证据

AST 核对：原 `validate_population_case` 的身份 guard 完全相同，原其余正文逐节点等于新 `_validate_population_contents`。原所有其他非拆分 definitions（含 legacy readers、PopulationCase/EventCleanSibling、clean validator、exhaustion class）逐节点相同。原 dev generator 的前置身份 guard 也相同。

将新 generator 仅作三项明确归一化——case split `phase`→`dev`、选择 dev/formal 身份 validator、dev/formal audit role——其余正文 AST 与原 generator 完全相同。这保留 `SeedSequence(seed).spawn(5)`、四个 geometry/objects/events/wind RNG 及第五 sensor stream；完整候选联合拒绝，没有增加局部 entity 重采样或按已见结果改 seed。外层仍唯一 `range(1000)`，耗尽抛原 `PopulationGenerationExhausted`，未增加 replacement/range relaxation。为避免生成额外候选，没有实际强制跑 1000 次耗尽；这是静态边界核验。

同一次 22 项重跑中，本人额外拦截每次新旧 validation，保存**接受与拒绝候选的完整内容 semantic SHA、结果/原因和成功 validation**，逐候选比较，而非仅比较最终成功 case：

| 已存在 dev root | 每版实际候选数 |
|---|---:|
| `devext-ID-L1-00` | 4 |
| `devext-ID-L2-00` | 12 |
| `devext-ID-L3-00` | 4 |
| `devext-OOD-L2-00` | 59 |
| `devext-OOD-L3-00` | 68 |
| 合计 | 147 |

新旧各 147 个候选全部逐项一致（各含 142 rejected、5 accepted）；最终 case JSON 及完整 generation audits 同原存档一致。独立 SeedSequence guard 仅放行这 5 个已登记 root seeds：实际调用 10 次，恰为 old/new 各重放一次，没有 new root 或 actual main/holdout draw。这是对 5 个既有 root 的运行证据，不宣称穷举所有 seed；代码的 AST 等价另支持规律拆分未引入其他分支变化。

冻结的初始失败日志保留，已读其失败断言/差异/summary：5 个失败均停在 dataclass tuple 与 JSON list 的直接链式比较；新测试先保留新旧 dataclass 直接比较，再 JSON normalize 比原存档。独立逐候选和 audit 比较均通过，没有通过丢掉 rejection/audit 检查来掩盖差异。

## 正式入口：先 proof，后随机；内容类型不构成准入

wrapper 首先检查 NumPy 版本，然后调用 `read_namespace_proof`，此前没有 `SeedSequence`、geometry 或 generator import/call。predicate 的输入仍是 acceptance、intent、verification 及可选 holdout opening。返回后使用其 hash-bound intent metadata 和精确 `root_generation_seeds` 映射；case_id 必须 native str、唯一 metadata match 且属于 namespace，然后才 import/call私有共享 generator，传入 `(case_id, proof.phase, registered_domain, registered_level, registered_seed)`。audit 绑定 intent path/SHA 和 native request SHA。

原 7 个 Fake 边界测试本次重跑通过：namespace reject 在生成前、main/holdout 精确 seed 传递、unknown root、missing seed、duplicate metadata、错误 NumPy 前置拒绝。另 8 个独立探针使用**真实已审 predicate 代码及完全位于 `/tmp` 的 synthetic owner/source/intent fixture**，将所有真正 `SeedSequence` / `default_rng` 设置为抛错，并仅以 Fake generator 返回字面结果：

- main / holdout 正例只把临时 namespace 中的精确注册 root/seed 交给 Fake；没有产生几何。
- unknown root、dev label、缺 root seed、boolean seed、与 synthetic dev catalog 的 seed collision、缺 holdout opening，均抛错且未触达 Fake generator。

这里的 synthetic native records / 字面 seed 不是实际 owner 批准、正式 seed 抽样或正式数据。测试不能替代真实 fresh gate。

legacy `load_development_case` / `PopulationCase.from_dict` 仍执行 dev 身份限制，正式 label 的内存 serialization fixture 被拒绝。新增 `decode_formal_population_case` 仅验证 schema、身份一致和同一内容约束；它没有 proof、namespace membership、root-seed provenance 或生成能力。provided 测试把旧 dev case 在内存改名后通过 content decoder，只说明内容检查可复用；这种改名不构成实际 formal 实例或 admission。独立 wrapper 的 dev-label 反例在已审 namespace 上仍被挡住。

## 待集成边界；无扩大授权

本候选没有 writer、dataset 全集碰撞核验、dispatcher、formal clean 派生处理或最终 REQUIRED_SOURCES 闭包安装；这些是明确待办，本轮不要求组件冒充全 graph。本次正式正例均 Fake 截获；没有验证当前 production 包路由或真实 CLI 能直接运行新 helper。未来接线必须使 wrapper 导入同一冻结 shared module，并保留完整身份/碰撞/provenance/错误证据；私有 `_generate_population_case` 和 content decoder 都不是可独立授予准入的 capability。

生产源字节未变，R/datasets 不存在。只重放 5 个既有 dev roots；所有其他临时输入均为 `/tmp` synthetic fixture。未调用 actual main/holdout RNG、模型、物理或 paid API，未写真实 owner、source、运行日志或正式数据，不安装、不授 G4/main/novelty approval。
