# Development population / worker：独立有界接线审查 v1

结论：已完成的 `qualification/dev-population.v1` 五个固定开发输入、seed子流、与旧dev24的碰撞检查及新schema接线均通过本轮独立复验；未发现会阻止这些固定输入读入/dispatch的具体P1。五case及全部147次joint候选审计可精确CPU重放。**这不是物理可解性、主试/holdout总体、模型或G4通过。**

本审查为同模型家族独立上下文工程审查。只写本报告及证据，不改源码、科学冻结总体、owner、旧dev身份/pilot、原namespace或他人文件。本轮新增物理步/模型生成均为0；没有另抽master、生成新科学root或运行600s。下述固定seed重放、AST/fake接线及受控失败fixture均明确区别于实际飞行。

## 版本和最终输入

工作树 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。审查前先读源码；看到最终manifest及5条records、无failure.json后才按完成输入验收。最终namespace manifest SHA-256：

```text
90d7136465542cb36530ff806ffc92f3527b0e7026d881229f450e94066e4731
```

namespace-intent SHA：`3557d42fa75cd20ed85ac1f629097914c140069d042c5d9b863f5b52dd356dbb`；旧dev24 manifest SHA：`238d5a321e01ca54adc0507e5035c37e271b3a3eeb3ad20d4d81e89211df5a72`。

| 当前source | 实际SHA-256 |
|---|---|
| `eval/inspection_population.py` | `5fcea10119f01077aecf59dda73cbbd3f5b08a1c4c3c6daa93a77a08e73244eb` |
| `eval/prepare_inspection_development_population.py` | `9d564a45412165aab72795c9dca6cf61ad1d96f314066b0d0447e86947e5248a` |
| `eval/inspection_scenarios.py` | `9159ba44790d6d315074a824e52adeebd96468ab3032c6f0f07fdb755ce2753f` |
| `eval/run_inspection_reference.py` | `4c280adab488f9c81d68265dd9930d9775dbb6372b36d1d776d9350f9f36f3ca` |
| `eval/inspection_common_calendar.py` | `777b099d041e302b613f9de485b72f793bce7b9fa59901b236f21ce623376c79` |
| `eval/run_inspection_dev.py` | `470b639ad8c41876490156755449a9c82ca0ccecf73099d5d62dbab366add586` |
| `eval/run_inspection_classic_pilot.py` | `4cce1140d144ff719ff7a7f01b35204e1a4525868b490b7f4915fd78aa5581dc` |
| `eval/run_inspection_llm_development.py` | `e3b6f836aa686455bca363a251e17b680ed29e661ad5f1d26b20d67a3fdcf44b` |

与已审 `qualification/source/model-orchestrator-v2/inspection_population.py`（`32788a37a6d1f6d182c6635f8dd3b8a26680be316d68ee828f0e0592c7cf8ab9`）对比，变化为module说明、新classmethod及development reader/validator分派。`generate_development_population_case` 与 `validate_population_case` 两函数的AST完全相同，joint采样与finite1000主体未变；没有以接线修复替换科学冻结law。

## 新namespace独立复算

NumPy实际版本为1.26.4。对intent中的master重新执行SeedSequence.spawn(5)及声明的little-endian 4×uint32编码，5个root seed全部一致；各root再spawn(5)的第五支重新生成sensor seed，全部一致。master+5roots+5sensor共11个seed互异，并与旧24输入seed及旧manifest master无交集。

| 固定case | entity数 | 完整joint候选次数 | 重放 |
|---|---:|---:|---|
| devext-ID-L1-00 | 9 | 4 | case与全部audit精确一致 |
| devext-ID-L2-00 | 11 | 12 | 同上 |
| devext-ID-L3-00 | 14 | 4 | 同上 |
| devext-OOD-L2-00 | 9 | 59 | 同上 |
| devext-OOD-L3-00 | 11 | 68 | 同上 |

本轮没有调用prepare抽新master；只以已存5个root seed重执行原纯CPU generator。约3.4s内完成case、每条candidate index/status/reason/placement与validation的完整精确对比；没有跳过先前拒绝条件或仅强行重放最后一次。147次均为对同一批固定输入的重算，不是147个独立root或新的实验样本。

每case的file SHA、case digest、audit SHA、root/sensor metadata和domain/level均匹配manifest。五个新case经新reader load→as_dict→load保持类型、domain、OOD曲线motion和digest；旧24 case经同一reader也精确roundtrip。configuration hash另外从原始JSON独立去schema/身份/seed/domain与entity key并排序重算，全部一致；改名、改sensor seed、改entity handles并反转entity顺序后，configuration hash保持不变。新5与旧24在所声明的seed、case digest、configuration定义上无碰撞，新5之间也无此碰撞。

这类精确hash检查不是近重复、所有数值/几何等价变换或统计独立性的证明。它没有把同一root的重命名/噪声变化计成新科学几何，也不证明五个dev strata有总体代表性。

manifest明确split=dev、main_and_holdout_instantiated=False、first_pilot=False；五case均为新的devext身份。旧pilot仍恰为dev-L1/2/3各00/01共六项。classic pilot的选择函数仍显式要求该六项集合；本namespace不是其替代输入，没有重标旧root或触碰main/holdout。

## reader、worker与物理输入接线

源码路径为：worker先校验private case文件字节SHA，再经 `load_development_case` 分派旧CaseSpec或PopulationCase，再进入ScenarioMission；后者按真实类选择validate_development_case，保留case.seed（sensor子流）、wind、实体motion和日历。reference/classic/LLM/dev入口的相关source清单已包含inspection_population；普通dev runner继续绑定suite manifest、case bytes、源与依赖，并要求development角色。

独立probe提取并执行实际source AST，对底层父类/world/executor使用显式fake，**不import或构造PyFlyt**：

| 检查 | 五case实际结果 |
|---|---|
| ScenarioMission父构造输入 | seed=case.seed，wind_xy=case.wind_xy，events=False，指定start_planner值，逐项相同。 |
| 初始代理同步 | 静态位置与case.position(0)一致；未来dynamic放在不可见z=-10；entity keys完整保留，register新body一次。 |
| coverage网格 | 与case自身ROI的20m中心格完全一致，没有沿旧固定ROI裁剪。 |
| L3事件边界 | link/camera在start包含、end排除；priority version更新，retire恰在对应时刻提交一次exit并标retired；重复调用无第二次退休。 |
| worker真实函数体 | 五case均经实际private-file guard及新reader完整读入；fake结果0physics steps。错误expected SHA在构造case mission前拒绝。 |
| common-calendar入口 | 同一新reader/validator接受五case，geometry digest匹配。只stub `_score` 验证schema路由，不把sentinel当真实分数。 |

main/holdout身份、未知field、bool seed、NaN wind五个reader负例均被拒。首次fake worker fixture缺少当前world.step_period接口，修正fake接口后完成检查；没有改生产或用fixture错误当工程缺陷。

评分边界保持：已审measurement recount从private entity/cells构造SimpleNamespace，不以CaseSpec schema白名单决定分数；新domain字段不需要修改冻结evaluator.v2。这里没有重做或改动冻结evaluator，也没有用stub common-calendar声称分数已重算或合格。

## intent、有限失败与已有短smoke

prepare在任何case draw前调用durable_once保存master、5roots、strata、来源/旧manifest hash；实际helper是exclusive create、flush/fsync文件及父目录。本轮受控fixture让首个generator明确抛出带1000条synthetic拒绝的PopulationGenerationExhausted：实际持久化顺序为intent→exhaustion→failure，failure状态namespace_blocked_no_automatic_reseed、completed_cases=0，无最终manifest。相同out第二次调用被拒，entropy mock总共只调用一次，没有自动换root、放宽范围或覆盖。

该fixture是失败记录流程测试，不是又生成一个真实namespace或实际观察1000次几何失败。真实本namespace已有完整成功manifest，没有failure文件；所有原文件不改。

另只读检查已完成的 `logs/inspection-population-ood-read-smoke.v1`：run/world completed，新schema的devext-OOD-L3-00，5.0s、1200物理步、非实时、无hard violation。17个source当前字节与其record独立匹配，private-case字节匹配；61 dependency的unchanged是原run报告值，本审查未重新认证全部依赖。它只是已存在的5s接线证据，不是本轮新物理、600s可解或G1/G3/G4通过。

## 执行与交付

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-development-population-review-v1/audit.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-development-population-review-v1/wiring_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-development-population-review-v1/failure_probe.py
```

另执行只读inline JSON/source检查保存existing-5s-smoke-check.json。没有复跑主 Agent报告的整套73测试；本轮以固定真实输入完整重放、29输入reader、负例、实际函数AST接线和失败流程作独立验证。证据、namespace/source快照及SHA在 `review/evidence/development-population-worker-integration-v1/`，`/tmp`原件保留。

本轮无尚未关闭的实证P1；结论只限开发输入工程一致性和接线。连续几何/容量条件是必要检查，不是飞行和持续动态追踪的可解性证明；后续任何600s、策略比较或新资格须按其实际source/case/clock/receipt独立验收，不能把本报告扩大为科学主试总体或G4通过。
