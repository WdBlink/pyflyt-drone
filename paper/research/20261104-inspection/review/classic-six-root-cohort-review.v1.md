# Classic 六根 dev cohort 独立结果审计 v1

日期：2026-11-06（Asia/Shanghai）。**完整 cohort 的 dev selection 身份验证通过：6 个预选 root × B0/B1/B2 各 1 个有效 repeat，共 18 个 SDK completed 槽；固定 B1/B2 选择规则得到 B1。** B0 的 dev 宏均值更高，仍单列，未被偷换为候选选择结果。本结论不属于 main、holdout、正式八配置成绩或 G4 admission。

## 观察时点、版本与边界

最初只读观察时 selection/results 尚不存在，未对 17/18 选择。准备来源审计期间，driver 正常写出：
- SDK `finished_at=2026-11-05T20:27:19.847818+00:00`（UTC，即 Asia/Shanghai 2026-11-06 04:27:19.847818）。
- `execution_status=completed`、`reported_denominator=18`、`attempted_count=18`、`outcome=inconclusive`。
- selection 的 denominator/completed_slots 均为 18、root_count 为 6，并绑定最终 results SHA。

完整只读审计检查通过时点：`2026-11-05T20:30:21.112594+00:00`。运行环境 Python 3.11.15 / NumPy 1.26.4。只新增本报告；未改 live logs、ownerstate、core/production、源码、输入、SDK 或既有报告，未 dispatch/重跑任何 flight，未启动 GPU、LLM 或 paid API。使用 `measure_completed_episode` 和纯读校验，**没有调用会写锁或可能 dispatch 的 `execute_local_episode`**。临时空 bytecode 缓存目录已清理。

关键不可变绑定如下；文末附全部 139 个文件的 SHA 清单。

| 对象（相对于仓库根） | SHA-256 |
|---|---|
| `logs/inspection-classic-dev-recording.v3/design.json` | `3fd0b4c1be631ffd8f8bb538b656930900f0fd6da2aa2bd0f67d772d26c438b3` |
| `logs/inspection-classic-dev-recording.v3/schedule.json` | `413cb00aef766135d634959a9deed7d8823e2b5d6e7433efa9d088ffbb02331b` |
| `logs/inspection-classic-dev-recording.v3/source-hashes.json` | `712615c31abcc659b25d84e0230f1876ef652e41c29a026dec36eb79dcd48c2e` |
| `logs/inspection-classic-dev-recording.v3/source-files.json` | `2fb81bdc618d9fab300f24f963b3a4199234185606d19a4cf603282839de1cb4` |
| `logs/inspection-classic-dev-recording.v3/dependency-hashes.json` | `d90279e0d9a3c26fb2fe5070c40c495cea0145f277808c32e01f7957f69e7d20` |
| `logs/inspection-classic-dev-recording.v3/results.json` | `63274c6f88b42de993e46c29e930e7425d274766b9c9d2e167867e7d0ff47c49` |
| `logs/inspection-classic-dev-recording.v3/selection.json` | `75f57580f0f5371d91a741fd4f9573e52192d019e9730335123a7ae67897a32e` |
| `paper/research/20261104-inspection/qualification/dev-cases.v2/manifest.json` | `238d5a321e01ca54adc0507e5035c37e271b3a3eeb3ad20d4d81e89211df5a72` |
| `eval/inspection_measurement_loader.py` | `fa93cdc64a12271908bafeeea7fd099a5d9c70d0d7bab57492d0db26daec109b` |
| `eval/inspection_common_calendar.py` | `c5246ab15bbeaf5ccb9baf6d9037e74a830072723b6f6421f2288ea8d50240ba` |
| `eval/inspection_recording.py` | `a2fbdddcf7546b3f69f1cc8d2fc1d565620be3ce6582c6f61f8e93923decffcb` |
| `paper/research/20261104-inspection/experiment/evaluator.v2.py` | `de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b` |

v3 driver 精确 SHA 为 `b59263b31b413caaee0924600e1e8974b104ee47a15199ca1d6e89ad36267464`；local supervisor 为 `e63d69e75251752330da2806915bae711ad4be2920e9017c2ffea726d64386dc`；receipt adapter 为 `59f88d5a3ec90497e19ec91ac9b914f09fe6a2303d029e6aecbb6fb2f958387b`；B1/B2 classic policy 源为 `763b14320f95e5d534154c4d4886cbd40e12cf13182f64d5eb572cd7d5e4fd00`。

## 初始计划、排程与完整分母

对照 `source-hashes.json` 的全部 **33** 项与 `source-files.json` 的对应文本重新求 SHA，集合和值均匹配；对实际源文件及初始 **61** 个 PyFlyt/pybullet 依赖文件重新求 SHA，均与初始记录匹配。所用 provided SDK 的 `experiments.py / production.py / science.py` 与 `/Users/wdblink/.codex/skills/autoresearch/paper/lib/` 原文件逐字节 SHA 相同；未替换 recorder。

冻结 suite 共有 24 个 dev 记录，其中原 `first_pilot=true` 恰好为六个 `dev-L{1,2,3}-{00,01}`，六个不同 lineage_group；无 main/holdout 实例化。六个 case 文件实际 SHA、canonical semantic SHA、case_id、lineage_group、split=dev 与 level 均逐一核验。各方法使用同 root 的同一个冻结 case。

从固定 PCG64 seed **2026110601** 独立重建排程（未调用 driver 的 schedule 函数），全部 18 项精确一致。root 块顺序：
`L3-00 → L2-00 → L1-00 → L2-01 → L3-01 → L1-01`。
每块的 B0/B1/B2 顺序见下表。18 个 invocation 的 order_index/argv/condition/case/repeat=0、返回码 0、stdout receipt 指针均核验；原 raw 的时间戳按冻结 schedule 顺序首尾不重叠。最终 SDK canonical grid 的排序与执行顺序分开处理，未把 canonical 存储顺序误作 callback 顺序。

跨 v1/v2/v3：design 除 planned_at 外精确一致，schedule 全字段精确一致，初始 source map 只有 parent driver SHA 变化。v1 planned_at `2026-11-05T17:05:05.781139Z` 早于第一条 native started `17:05:05.817593Z`；候选、六根、数量、选择规则不是在两条旧 flight 已知之后新增的 v3 决策。

| 原执行序号 | root | condition | 有效 Y | v3 cache | native attempts |
|---:|---|---|---:|---|---:|
| 0 | dev-L3-00 | B0 | 0.8245752535103837 | 是 | 1 |
| 1 | dev-L3-00 | B2 | 0.6271768508436419 | 是 | 1 |
| 2 | dev-L3-00 | B1 | 0.65594260176561 | 否 | 1 |
| 3 | dev-L2-00 | B2 | 0.5582428561233956 | 否 | 1 |
| 4 | dev-L2-00 | B1 | 0.5789292595650977 | 否 | 1 |
| 5 | dev-L2-00 | B0 | 0.7845597455424045 | 否 | 1 |
| 6 | dev-L1-00 | B0 | 0.9970187393526406 | 否 | 2 |
| 7 | dev-L1-00 | B2 | 0.803449744463373 | 否 | 1 |
| 8 | dev-L1-00 | B1 | 0.9373935264054514 | 否 | 1 |
| 9 | dev-L2-01 | B2 | 0.7086379041861317 | 否 | 1 |
| 10 | dev-L2-01 | B1 | 0.7022877164056059 | 否 | 1 |
| 11 | dev-L2-01 | B0 | 0.9242649079417422 | 否 | 1 |
| 12 | dev-L3-01 | B1 | 0.7308584657197564 | 否 | 1 |
| 13 | dev-L3-01 | B2 | 0.7560108237745705 | 否 | 1 |
| 14 | dev-L3-01 | B0 | 0.9089803504638944 | 否 | 1 |
| 15 | dev-L1-01 | B0 | 0.9971381847914963 | 否 | 1 |
| 16 | dev-L1-01 | B1 | 0.9901880621422732 | 否 | 1 |
| 17 | dev-L1-01 | B2 | 0.9754701553556828 | 否 | 1 |

这是 **6 个 root 的 18 个配对 root×method 单元**，不是 18 个独立 root；SDK seed=0 是单次重复标识，不替代 case 文件中冻结的实际场景 seed。未重新证明 generator 抽样独立性或物理可解性。

## expected 的独立构造与真实回执链

未把每个 plan 自称的值照抄为 expected。对每个槽，独立从初始清单构造：

- worker source map：初始 envs/inspection 的 12 个源 + envs/__init__.py + run_inspection_reference/cases/scenarios，共 16 项；B1/B2 再加初始 classic 源，17 项。
- suite SHA、对应 case 文件及 semantic SHA、61 个 dependency hashes、初始 supervisor/adapter SHA。
- 原固定线程环境四项均为 `"1"`，clock lag 门限 **0.1s**，repeat=0，phase=development。
- 另核 plan lineage_group、max_attempts=2，避免把 loader 未逐字段比较的 metadata 当作已保证。

以上 expected 调用只读 `measure_completed_episode` **18 次全通过**，其 Y/完整 score/terminal/world/plan 绑定与原 SDK raw、invocation stdout、results 全一致。随后用全新空 `PYTHONPYCACHEPREFIX`、`-B`、`PYTHONDONTWRITEBYTECODE=1` 的 fresh Python 再完成同样 **18 次**，值全部一致；记录实际加载模块路径/SHA，临时缓存目录运行后仍为空。此追加验证避免把已有项目 bytecode 当作校验后的源执行证据。

全部 episode 与 attempt 目录实际盘点：仅上述 18 个 episode，恰好 **19 个 attempt**，无额外未列 episode/attempt、无缺 terminal/未知 attempt、无 attempt-02。每个 attempt 的 intent/start/stdout/run/world/copied-case 六文件链重算 SHA，terminal 绑定匹配；intent 的真实 CLI argv 指向同初始 case、正确 policy、`--realtime --duration 600` 及本 attempt 的 out。所有 embedded worker source 文本重新求 SHA，精确等于独立 expected；run 的 source/dependency 不变标记、runtime threads、world case SHA、输入复制 SHA、时钟/步数条件及 score 约束均通过对应校验。

### 保留的时钟失效与有限重试

唯一 native 失效记录：
- episode `dev-L1-00-B0-0/attempt-00`；
- terminal status `known_operational_failure`，terminal Y 为 **null**，failure_reasons 精确为 `["clock_lag_exceeded"]`；
- max lag **0.23917641642813692s > 0.1s**，retry_allowed=true、remaining budget=1；
- terminal SHA `76647ff49de82812b2f91421ab1ead05a5f0e4ee17ed7b4d67d707bae7e53ebb`；
- world SHA `4d17cbcd1a449007540ecfea273267cad9966a5d002d68a98ce88afc21ed9b4a`，run SHA `b27149efe8b09fbfa6012c4d87892d68512722e28de0df3e7a47483734db2a7e`。

该记录保存完整 144000-step/600s 输出，但因预声明时钟标准不准入。其 world score.Y 的离线诊断值为 0.9970187393526406，**未当作一个方法观测、未改为 policy Y=0，也未丢弃文件**。同一不可变 plan 的 attempt-01 随后是第一个有效 completed；总尝试恰为 2，预算及线程/源/依赖/输入保持相同。其他 17 个 episode 均仅 1 attempt。未出现对有效政策低分重飞或 best-of-repeat 取优。

### 缓存与原失败 SDK envelope

v3 仅 order 0/1 为 cache：
- L3-00/B0 native started `2026-11-05T17:05:05.817593+00:00`，v3 supervisor 复用耗时 0.15656012506224215s；
- L3-00/B2 native started `2026-11-05T17:18:26.796309+00:00`，复用耗时 0.40301324997562915s。

二者都早于 v3 planned_at `2026-11-05T17:36:27.552228Z`，且复用同一 attempt-00/原 terminal SHA。它们不增加 native flight 数。全 store 是 19 个 native attempt、18 个有效完成；其中 v3 新 dispatch 为 **17 个 attempt = 16 个有效完成 + 1 个时钟失效**，另 2 个有效完成来自早期 cache。

旧 envelope 原样保留：

| 原 SDK envelope | 完整 rows / raw 文件 | completed | failed | stopped 且 attempted=false | selection | results SHA-256 |
|---|---:|---:|---:|---:|---|
| v1 | 18 / 18 | 0 | 1 | 17 | 无 | `c92f92c5fa68dd207accd27c2f92ee9ab5a04f0a9ab90b9b477bc9e7b1605fba` |
| v2 | 18 / 18 | 1 | 1 | 16 | 无 | `cda2e3332b30909ffe55afc8ad4167c810efb5616da716ed64def5baaa359a11` |

旧 v1/v2 live results SHA 与对应 `qualification/classic-pilot-generation{1,2}.v1/results.json` 快照一致。其 18 个 raw 文件各自都存在、字段逐行与 results 一致；`stopped` 是 SDK 原枚举且 attempted=false，不能称作已飞失败。v1 已尝试槽是 B0 的 parent operation failure；v2 B0 为有效 cache、B2 是 parent source-change operation failure。旧 envelopes 没有 selection。v3 将保留且合格的 native receipt 严格重包，不把旧失败/未执行槽删除后宣称原 envelope 完成，也不把三个 envelopes 的 54 个 planned rows 合并成新增独立样本。

## 完整 score 与固定实体/分母

18 个 loader 有效结果均重建 report ledger/覆盖/静态/动态/bin/分母并与原完整 score dict 精确一致。另以已独立审查的冻结 `evaluator.v2.py` 对 **全部 19 个 attempt world** 各启动一次 fresh CLI，传来自 terminal 链的实际 world SHA：19/19 exit 0，完整 score dict 相同；唯一时钟失效的结果只用于核验留存输出，不进入 selection。

再用不 import 评分函数的独立算术，从 private case 生成所有目标 entity key、60×3s 的 bin 起止/index/原优先级权重，验证缺失 bin 仍为零且保留；按原 coverage packets/cell 集合、实体/bin 表和 safety/range primitives 独立算 C/S/D/Q/H/E/Y/SR/recall，两分母及所有标量在 19 个 world 中均精确相等。ROI→完整 cell 键和值映射也逐个核对。

| root | static/dynamic 目标数 | static 权重分母 | dynamic 权重分母 | 每动态实体 bin 数 |
|---|---:|---:|---:|---:|
| dev-L1-00 | 4 / 0 | 4 | 0 | 0 |
| dev-L1-01 | 4 / 0 | 4 | 0 | 0 |
| dev-L2-00 | 5 / 3 | 5 | 180 | 60 |
| dev-L2-01 | 4 / 2 | 4 | 120 | 60 |
| dev-L3-00 | 5 / 3 | 9 | 331 | 60 |
| dev-L3-01 | 5 / 2 | 9 | 226 | 60 |

同 root 的三个方法分母一致；L2-01/L3-01 的目标数与动态分母允许按原 case 不同，并未硬套 L2-00/L3-00 的数量。上述重算继承 recorded coverage/sensor/safety/range primitives：不是 240Hz 原始 trace、真实里程或完整物理因果链的独立重放；没有把数值一致扩大为这些未执行验证。

## 固定选择规则的独立复算

先取每 root×condition 的唯一合格重复；每 level 两根等权，再对 L1/L2/L3 三等级等权：
`macro(c) = (mean_root_L1(Yc) + mean_root_L2(Yc) + mean_root_L3(Yc))/3`。
这里各级根数相同，数值恰等于每方法六根均值，但实现仍按原等级宏平均复算。

| 统计范围 | B0 | B1 | B2 |
|---|---:|---:|---:|
| L1：2 root 均值 | 0.9970784620720685 | 0.9637907942738623 | 0.8894599499095279 |
| L2：2 root 均值 | 0.8544123267420733 | 0.6406084879853517 | 0.6334403801547637 |
| L3：2 root 均值 | 0.866777801987139 | 0.6934005337426832 | 0.6915938373091062 |
| 等等级宏均值 | 0.9060895302670936 | 0.7659332720006325 | 0.7381647224577993 |

原规则是 `B1 if mean_B1 >= mean_B2 - 1e-12 else B2`。独立宏均值差 B1−B2 = **0.027768549542833254**，远大于 tolerance，结果 **B1**，与原 selection 精确一致。另用固定标量核对 equal、差 0.5e-12、差 2e-12 三个 selector 边界，分别得到 B1/B1/B2；没有据 dev 结果更换 tolerance 或候选集合。

B0=0.9060895302670936 的描述性宏均值确实更高；按预声明它保持 baseline/正式矩阵中的 B0，不被拿来替代 B1/B2 之间的 classic_best。这里不对 B1/B2 或 B0 宣称显著性/泛化优势，不算正式检验或 holdout 效果。

## 执行记录与限制

实际执行的是纯读 Python 脚本（首轮工作目录为仓库根）：
`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -`。
脚本只 import loader/validators，读取源/清单/JSON，调用 `measure_completed_episode`、`validate_flight`，并通过 subprocess 调用**离线 evaluator**，没有调用实际 worker/supervisor CLI。PyFlyt import 输出过 pybullet build banner；未创建 mission/aircraft/simulation 或提交生成调用。

一次附加旧 envelope 探针起初将未尝试状态假定为 `unstarted`、误以为只写 attempted raw，断言失败；读取实际 SDK 后改按原 `stopped/attempted=false` 及 18 raw 全量重核。最终全部实测断言通过，未把该 harness 假设错误计为实现缺陷。最终检查 hash inventory 及初始33源/61依赖未变，未追写 live 结果。

判定：**固定六根 classic dev cohort 的完整分母、来源/输入绑定、原回执留存、有限重试、缓存身份及 selector 结果可审计通过；本次未发现需阻断该 dev selection 身份的残留不一致。** 这不是 G4 GO，也不认证真实 LLM/paid 路径、主八条件矩阵、独立统计推断、trace 物理充分性、未来 hard-crash recovery 或任何后续源版本；不重新执行 G1–G3。六根是固定 dev 选择集，不能把这批 selection 用过的数据当独立最终 test。

Attention / reviewed by inherited-parent model：同模型家族独立上下文的代码、原文件与离线实测审计，非跨模型盲审或人类专家认证。覆盖固定 v3/全部19 native receipt 与原 v1/v2留存，不覆盖后续模型 smoke 或上游尚未准入的事项。

## 最终文件 SHA 清单

以下键相对于 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。139 项包含 v3 7个顶层文件、6个case/manifest与审计用模块、18组 raw/invocation/plan、19个 terminal，以及旧 v1/v2完整raw与清单和原SDK核心。每个 terminal SHA 进一步绑定 intent/start/stdout/run/world/copied-case 的记录哈希；本次已实际重新计算这些子文件，不只是接受 terminal 自称。清单是审查证据，不是新实验输入或结果。

```json
{
  "eval/inspection_common_calendar.py": "c5246ab15bbeaf5ccb9baf6d9037e74a830072723b6f6421f2288ea8d50240ba",
  "eval/inspection_measurement_loader.py": "fa93cdc64a12271908bafeeea7fd099a5d9c70d0d7bab57492d0db26daec109b",
  "eval/inspection_recording.py": "a2fbdddcf7546b3f69f1cc8d2fc1d565620be3ce6582c6f61f8e93923decffcb",
  "logs/inspection-classic-dev-flights.v1/dev-L1-00-B0-0/attempt-00/terminal.json": "76647ff49de82812b2f91421ab1ead05a5f0e4ee17ed7b4d67d707bae7e53ebb",
  "logs/inspection-classic-dev-flights.v1/dev-L1-00-B0-0/attempt-01/terminal.json": "0eb3ed2a31c00d04f9cb1486f12bae0f12858d439d84e4ca50fb3a2d95a5acac",
  "logs/inspection-classic-dev-flights.v1/dev-L1-00-B0-0/plan.json": "399c98aca5a60604614b56e39c12321339e0efbde745dc61ade99ce0d7942bc3",
  "logs/inspection-classic-dev-flights.v1/dev-L1-00-B1-0/attempt-00/terminal.json": "890730ed9dcd9c472ccb5d7c4ca08757b5e3ba8beb21249139a4c2f9764b45b1",
  "logs/inspection-classic-dev-flights.v1/dev-L1-00-B1-0/plan.json": "1e17b559bf7db271f654624f4cca9b1719c3ea2f9cef12237b16000722c25406",
  "logs/inspection-classic-dev-flights.v1/dev-L1-00-B2-0/attempt-00/terminal.json": "51fa53ae0c3d80865b5f77192b5705792f234bdb8c7731c694a8cd384206d991",
  "logs/inspection-classic-dev-flights.v1/dev-L1-00-B2-0/plan.json": "83c0b4a04d18bf38ab88d81ef00cc2075643941dda0c909fded3939c71757a24",
  "logs/inspection-classic-dev-flights.v1/dev-L1-01-B0-0/attempt-00/terminal.json": "c3b67073ddafba2a9c1deedb8f62a3c030a8a4954d75d889572cbb98fa79233b",
  "logs/inspection-classic-dev-flights.v1/dev-L1-01-B0-0/plan.json": "385398f86ad911c65c4b90161a8ce41aa437d630ff1be5cc7f3daf43c00a6348",
  "logs/inspection-classic-dev-flights.v1/dev-L1-01-B1-0/attempt-00/terminal.json": "c7441a1fae412e896c2c6eae406132af7213bfb70745df54a25ca72e940f69ac",
  "logs/inspection-classic-dev-flights.v1/dev-L1-01-B1-0/plan.json": "0b372a8cdc422c410b6a991a613a1abe9c4d7e12a1520c00ff062f3e2bb8e962",
  "logs/inspection-classic-dev-flights.v1/dev-L1-01-B2-0/attempt-00/terminal.json": "08a46b05305afe101de2465beb29874b250fa94fbb17244d399a22694d7d6259",
  "logs/inspection-classic-dev-flights.v1/dev-L1-01-B2-0/plan.json": "db01daa56158eb04b781a185443aca41226995c1f69334afedd9e59905b2767c",
  "logs/inspection-classic-dev-flights.v1/dev-L2-00-B0-0/attempt-00/terminal.json": "923a767a32198fbec545ef491b38bbe6dc2fd314e0aea315ceac66a3873d786d",
  "logs/inspection-classic-dev-flights.v1/dev-L2-00-B0-0/plan.json": "0b352b081cfa215b066e3dfa1db7cf0076c758786e9fda22f7dd1fea7c9318fd",
  "logs/inspection-classic-dev-flights.v1/dev-L2-00-B1-0/attempt-00/terminal.json": "51692f6f1160690736675f7652578dd85029275d1ce4f9c6bbc9e43eb7ab1b78",
  "logs/inspection-classic-dev-flights.v1/dev-L2-00-B1-0/plan.json": "75753834e6c482ca37235b6ee98e430f8725fb02f46066cfbe243a6e962d5feb",
  "logs/inspection-classic-dev-flights.v1/dev-L2-00-B2-0/attempt-00/terminal.json": "7ff5e7f78cfcf7408ddda43f74ba32af36fd6ffbeaceecaf11a5abba4f4a993f",
  "logs/inspection-classic-dev-flights.v1/dev-L2-00-B2-0/plan.json": "d10293870e39919633244a5b778ea64377e8faacc27b2fd02268fb154482081b",
  "logs/inspection-classic-dev-flights.v1/dev-L2-01-B0-0/attempt-00/terminal.json": "77812f8d599707aa7afcdcaeed11428a5308035acf0ba92638400fe98718f629",
  "logs/inspection-classic-dev-flights.v1/dev-L2-01-B0-0/plan.json": "f47eedfcc2db0bda66ff75f1f60a1ee8fc6eff9709719c4c5e4fe57e2a69c915",
  "logs/inspection-classic-dev-flights.v1/dev-L2-01-B1-0/attempt-00/terminal.json": "d5dd52dc5e3fb1eaf7c937603d107e6246ac5ca809a4e1ea51706380ac593cc2",
  "logs/inspection-classic-dev-flights.v1/dev-L2-01-B1-0/plan.json": "3ab9738155b58c613b31eeaaa96a363a4d66d0d42135185a2225ec40298fca83",
  "logs/inspection-classic-dev-flights.v1/dev-L2-01-B2-0/attempt-00/terminal.json": "85acfa9a144d94d3f5abbbc1ac315a9c353e892fe9bf56a2ae4a838655a7cd1f",
  "logs/inspection-classic-dev-flights.v1/dev-L2-01-B2-0/plan.json": "f77c30aee43614a4da76e39f94a4c2611b1ffac019095c8dabd8424ab78ab9b6",
  "logs/inspection-classic-dev-flights.v1/dev-L3-00-B0-0/attempt-00/terminal.json": "35ed4a32fff671d3812d8aaffabd6eb8f6185cfca1619e0d6c7f4978f0703ac7",
  "logs/inspection-classic-dev-flights.v1/dev-L3-00-B0-0/plan.json": "ea9f9294ae92ab41e3ee865397d9a76749e160c88b9a0b5a1115143409e53a54",
  "logs/inspection-classic-dev-flights.v1/dev-L3-00-B1-0/attempt-00/terminal.json": "cff9f3047290efa83b64874804cbbd4732db0f723e088a02cb226aeb61dc5c22",
  "logs/inspection-classic-dev-flights.v1/dev-L3-00-B1-0/plan.json": "b6b9620673ee65ea289af6c712a80e167449089192cec0cf5d0fa3324674e1cd",
  "logs/inspection-classic-dev-flights.v1/dev-L3-00-B2-0/attempt-00/terminal.json": "d9d255ed9ef6279b16907f8bc09ffe817fc5f583e8721b97a979dd8794549c5c",
  "logs/inspection-classic-dev-flights.v1/dev-L3-00-B2-0/plan.json": "76234c85e0afb8bcccba2477eaf416a6b5e098d8269ddb5bcec456f3decc6586",
  "logs/inspection-classic-dev-flights.v1/dev-L3-01-B0-0/attempt-00/terminal.json": "85aa1da85599d2466b84b664944493eb59ae0ef2b86dd765a71e028cb4e96707",
  "logs/inspection-classic-dev-flights.v1/dev-L3-01-B0-0/plan.json": "2887ecdd9d1a003852a764b7490d6f98af4ef62a92eac715fcd42bc0b35bcb8f",
  "logs/inspection-classic-dev-flights.v1/dev-L3-01-B1-0/attempt-00/terminal.json": "de79635ecb4845e485cab7d081cf98b0a8918f602250aca2fcde74f239884331",
  "logs/inspection-classic-dev-flights.v1/dev-L3-01-B1-0/plan.json": "59dbcf2ec4550d50efe15b70befe42ddd759e018006123e943740f5a521eb698",
  "logs/inspection-classic-dev-flights.v1/dev-L3-01-B2-0/attempt-00/terminal.json": "75a53a0d97168570ef6d5f6d6f46f6245bea3231cc040268828d53cc6ad8d67e",
  "logs/inspection-classic-dev-flights.v1/dev-L3-01-B2-0/plan.json": "e5ca2f3e1287a789ed018cf23106ae5135b37d32cd23fd33383370478c126084",
  "logs/inspection-classic-dev-recording.v1/design.json": "d70e2c1244597702f8e17a28a4dc0d6ec121eed4daf134f80195b651814b001b",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L1-00-B0-0.json": "7530050c81e4ae0ddf98829b41733df886989b02f9ba303d0e33868a7aa979dc",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L1-00-B1-0.json": "6774a0c603b7365b1f72301cb31b813bb596c9b85c1a3353a6b30ec99750a575",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L1-00-B2-0.json": "9bf04da5d1e74739619181196d2e686efaa598756845e0a586d1ac944418440d",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L1-01-B0-0.json": "0f3fbf03245ef60b01d3a6332d7352163d7cc40f295af99396e6731c447d15c1",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L1-01-B1-0.json": "8ec77468aa0fa7e0484271140fc937ae0c7f2bf9b1068c5d8a26175537c9cfd9",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L1-01-B2-0.json": "11ec4c3d2b2540e9679e33cf7bc26b5f72ffac332e17c5d823ac12cd6fa53614",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L2-00-B0-0.json": "bb9b3485a207614ee5f2e3fdfded23d8e690fd6328ba0b1aaf00d5fb2cdab288",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L2-00-B1-0.json": "f7214661accde7472c0efafff34d875b71b7c8f57477a9225b3f5b143da421b6",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L2-00-B2-0.json": "332ea4b890f01ee54854e2739d11fe32bbbd5e4e2f6b4b6ae47a37300ea714cb",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L2-01-B0-0.json": "bcbb086adee42dcfd4299d245a005fcd2eaf3c791928d0e877ded06c2c7e3377",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L2-01-B1-0.json": "df2f80883b5af794c24e9c28f41f0daccf1d823ad5b695a94a6bcbd44981ec93",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L2-01-B2-0.json": "5ce50c7eb10027e715b77e24079e436939059e0c27c457a36921f3a67e0e67a4",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L3-00-B0-0.json": "f47a1e647526b1221d345fd5fc987123a2485745d8ec560fa0ca04c2f80f1613",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L3-00-B1-0.json": "635295aeddfbff24cff4b405c28bbb179afec4a64f6cd223b34460f61039587f",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L3-00-B2-0.json": "06d7c309c757f7dd88cd282ccbd172b0b035a6ee75cdf7dd45ec32bd7aee810d",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L3-01-B0-0.json": "c6dcf83d6758c528f7075dadbf86f548e1bbe338870c56404ddbf1d91856f7e4",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L3-01-B1-0.json": "8d5c958faff757705c540527faf74744746dd0af3662270e42abcc5e773a2eaa",
  "logs/inspection-classic-dev-recording.v1/raw/dev-L3-01-B2-0.json": "63d295f5390bcc573e53c722abe6035fe25720c5b17fad57424196d6edcfa415",
  "logs/inspection-classic-dev-recording.v1/results.json": "c92f92c5fa68dd207accd27c2f92ee9ab5a04f0a9ab90b9b477bc9e7b1605fba",
  "logs/inspection-classic-dev-recording.v1/schedule.json": "413cb00aef766135d634959a9deed7d8823e2b5d6e7433efa9d088ffbb02331b",
  "logs/inspection-classic-dev-recording.v1/source-hashes.json": "7abd5c66c77c5f1403f9d1bc6d32aa270f352db3dd9a700c2ae2c1cad988e8e5",
  "logs/inspection-classic-dev-recording.v2/design.json": "2013a1a8855b8c0e8896c37cc596237b9b62cdb3ed21a6bd78380aee1e93729e",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L1-00-B0-0.json": "a5cfd970bd838eb290e8fae56db823f5d2bbce6bbd93ccc3327df3da454c88d9",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L1-00-B1-0.json": "ad863f09858c073b233823c4bc86b2e5ef5a54fd8755f1422085ef8d9e4ddf72",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L1-00-B2-0.json": "85cba14ceb7ea37ba5f4df2be5b6b4d815a96c68ff7ef862446413753c87edbf",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L1-01-B0-0.json": "58320742d56297a1caf86602e763f0a88d2be82efd02bbdfc00d2c428737b3ea",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L1-01-B1-0.json": "bf714e12c9d8665fae4310a2a38e8cd9d1b924b4af29dfa67f11f82ce5df31d0",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L1-01-B2-0.json": "c6ba7e51319687da4932756c19bd3aa45437edc9548c62fd5db4bff762c2fa54",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L2-00-B0-0.json": "4d981bc36017d99d8c451e069d9d6cf006cbf1c4a6f175c5ff0a4056c6ab2638",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L2-00-B1-0.json": "5cfb423528d7ddec0561a0d2dee59360469704cacd9037c950246ada4865feca",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L2-00-B2-0.json": "0effa60fa06cc6f5824115e1217c635402a724c848563b29ed274d0bda9e69ef",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L2-01-B0-0.json": "8f507ddcd4ca9abbfcbcb96d58581f2fdf583bc73a938392ff2e18d1fb1a658e",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L2-01-B1-0.json": "74df2906a74e728194b181cd55aed5f747dc9023d8bd7de8dcdc3fc177c052db",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L2-01-B2-0.json": "34c81a57282aa7202ed548e5baea2d9f53c7aadbfbbe50a08cfd0bbcdbaea09b",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L3-00-B0-0.json": "87020d2e3a1430d2e21d47ac790f4a8883da15b50c7ad77cf017dce9cc00f641",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L3-00-B1-0.json": "65aae3be1fc2e3bd1b2c5dbbfe86be19276eef97b7bdfea9c775d3372e0a3af7",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L3-00-B2-0.json": "ea7eeb30d71f879cbcbe9c61155666b5f2f08ee15c81b8229b24e4451b83d9a4",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L3-01-B0-0.json": "af65a47225b76858439dc5793b3bf0816366e2d3afc2e7fa1689ca09fe4ebd7e",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L3-01-B1-0.json": "cbdc8b1ac66595938adfa4526b8968729abff5f3546baa7259ab6c031ac8b96f",
  "logs/inspection-classic-dev-recording.v2/raw/dev-L3-01-B2-0.json": "37408b6b5b95be298657c2e5b03fef40329f84e540c0dfd384e742afc36dad43",
  "logs/inspection-classic-dev-recording.v2/results.json": "cda2e3332b30909ffe55afc8ad4167c810efb5616da716ed64def5baaa359a11",
  "logs/inspection-classic-dev-recording.v2/schedule.json": "413cb00aef766135d634959a9deed7d8823e2b5d6e7433efa9d088ffbb02331b",
  "logs/inspection-classic-dev-recording.v2/source-hashes.json": "9eedc25472c9696b85771b69125a416b8aeee6adeb1fd825e35416d760aa084b",
  "logs/inspection-classic-dev-recording.v3/dependency-hashes.json": "d90279e0d9a3c26fb2fe5070c40c495cea0145f277808c32e01f7957f69e7d20",
  "logs/inspection-classic-dev-recording.v3/design.json": "3fd0b4c1be631ffd8f8bb538b656930900f0fd6da2aa2bd0f67d772d26c438b3",
  "logs/inspection-classic-dev-recording.v3/invocations/00.json": "0c6e074bd133e0afa930ea53ac361fd5619ec79734139e51fde84333002a2bda",
  "logs/inspection-classic-dev-recording.v3/invocations/01.json": "f41bf812413ed60bc2cc0ea43758f10ff0482aa5596c4c969484927cf58e74fc",
  "logs/inspection-classic-dev-recording.v3/invocations/02.json": "e709e79f4fecf50e87c683d062ef83f1578bf69311fca3335bc4be5e03577956",
  "logs/inspection-classic-dev-recording.v3/invocations/03.json": "5cfe0d4b60e9a94e398506a7640dc249ecfca4b58d01cb3b15d3c80f86c6a0d9",
  "logs/inspection-classic-dev-recording.v3/invocations/04.json": "f2eca441d026c0733385fcc37e8a6b0bc851510433f4c9b3e52fee406bc6c870",
  "logs/inspection-classic-dev-recording.v3/invocations/05.json": "37798d102d8af2c7cc354b76b377fd4bc80612b951c15be27be6f7dadcf3f532",
  "logs/inspection-classic-dev-recording.v3/invocations/06.json": "28ebf8f92b8620e24c90ad60d432954731bb6acc40d66f861e90985777311634",
  "logs/inspection-classic-dev-recording.v3/invocations/07.json": "20fb96b9f1e0ef7598e306d90b60c3bd12dc7065c2bd11f4735c4467fe764e0f",
  "logs/inspection-classic-dev-recording.v3/invocations/08.json": "f347040431546a19858e69d98bbaf9b896bed2527d155b831c520ebecb3876fa",
  "logs/inspection-classic-dev-recording.v3/invocations/09.json": "710c40289335edccc3293fe9b813e8dceef564d4f6dfb917a3e6dce14eb47598",
  "logs/inspection-classic-dev-recording.v3/invocations/10.json": "e957389662c67ab093ffe533a1eeaac26656bbb0b65f55274a591dcfd9ddb5e1",
  "logs/inspection-classic-dev-recording.v3/invocations/11.json": "4202f6f445c2ed8aa21219c21b1ccbcd61583b6110e4c6fd978c18bb392e3ce3",
  "logs/inspection-classic-dev-recording.v3/invocations/12.json": "ee6574a9c5930c7f9937b26f26310420d7765c707cae0258665c8e73653b7e6e",
  "logs/inspection-classic-dev-recording.v3/invocations/13.json": "e58e5e4bbf1c05556b8d4e29ad4ea2466d05a18ce02f84e5b9fd3961ba8a6437",
  "logs/inspection-classic-dev-recording.v3/invocations/14.json": "973472e938469ae0a49a1c76475a042bc8a0f04f15cd1c14398d787ef7640fbd",
  "logs/inspection-classic-dev-recording.v3/invocations/15.json": "da21e8aade1543ead9af719ed9c5e3174fe98912f17066c13676e78159c37030",
  "logs/inspection-classic-dev-recording.v3/invocations/16.json": "dd5ad02aa792f3ada48f99ff8f3fd0881fa1e3e906091829b089c51ffa448fa6",
  "logs/inspection-classic-dev-recording.v3/invocations/17.json": "1174f736bb6ab69e12fba810fbda1aabae898982e2b65ec85e541ba3573931ff",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L1-00-B0-0.json": "42a659888b304de4824f0cfa0d8e137a40b98c7f257203578ec0b5e2c3deecec",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L1-00-B1-0.json": "14e2f34bd2637f1f38f11b8f4a7ba8295be970418b27173cfbd449557fb2c0d5",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L1-00-B2-0.json": "dac33f67621297e2f9edbb359f3c03d6013aefe07738c455291eef606475c9de",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L1-01-B0-0.json": "f5d0b172eaf1258ba27ff8c3ae15dbe31abb71e70e55dae9dd348abb6b5b7d89",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L1-01-B1-0.json": "d2a14f4cc5fd2db81b1507bd206878285d5edccfe65777aca724c6effe55745c",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L1-01-B2-0.json": "9822130eb91c00eddc6213d172952ba479e63227e1ee322cc6cdc30b7ff62bd6",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L2-00-B0-0.json": "73a6e0772dd8869df0fb52a24876fb36921e15872d7fe6ed6ffa8b9afadfed3f",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L2-00-B1-0.json": "c157d98bd7be759e583f00e8adb2c94cf0c7cbf7bc21fdd2d6eac0e5d998b0a6",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L2-00-B2-0.json": "bc0688f5629008c1f7c9778ce921621413ab358859ac71211bd923a820b7a299",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L2-01-B0-0.json": "c90451a5fe0475b1a01621f87ba5aabab2e28148d5b280d6e11a6798faeaa222",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L2-01-B1-0.json": "0aca3ffa0f8721c27bf617bd55d9474352545971e74421fa53dd7544a3158804",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L2-01-B2-0.json": "ef2132ae7cdbeaea9585cdb14ce589b3cb31f11ad943c6f4b61718c930fe3dd0",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L3-00-B0-0.json": "3b4726e0165140789b996908fb3bc667c3b6e753d2534f6190f5f9322a812628",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L3-00-B1-0.json": "f821e38fc469c5eafd320fa326f6cb5c4d8567be3f8ced0e299bb295e1fee23a",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L3-00-B2-0.json": "ac29afb5423010a84d5f897dec90008fc8f949a4634eff65c60244f398129343",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L3-01-B0-0.json": "7454dd04372df5bcdc3866afeeb879412f1af45a842623f12c5af1007d8cfd95",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L3-01-B1-0.json": "0c0bd1d058086d29b8dbd2ad0138c72365bb6b9fbd97f073e365774b27662438",
  "logs/inspection-classic-dev-recording.v3/raw/dev-L3-01-B2-0.json": "5b65b816681dfed2a5e7c6dd55601d7fe4c26eb5742d98b3e28ef451994493df",
  "logs/inspection-classic-dev-recording.v3/results.json": "63274c6f88b42de993e46c29e930e7425d274766b9c9d2e167867e7d0ff47c49",
  "logs/inspection-classic-dev-recording.v3/schedule.json": "413cb00aef766135d634959a9deed7d8823e2b5d6e7433efa9d088ffbb02331b",
  "logs/inspection-classic-dev-recording.v3/selection.json": "75f57580f0f5371d91a741fd4f9573e52192d019e9730335123a7ae67897a32e",
  "logs/inspection-classic-dev-recording.v3/source-files.json": "2fb81bdc618d9fab300f24f963b3a4199234185606d19a4cf603282839de1cb4",
  "logs/inspection-classic-dev-recording.v3/source-hashes.json": "712615c31abcc659b25d84e0230f1876ef652e41c29a026dec36eb79dcd48c2e",
  "paper/research/20261104-inspection/experiment/evaluator-source-manifest.v2.json": "ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80",
  "paper/research/20261104-inspection/experiment/evaluator.v2.py": "de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b",
  "paper/research/20261104-inspection/qualification/dev-cases.v2/dev-L1-00.json": "5600888c8cc583edadbb1e1d43f8f0dcd928b9b4e44c2dc93296a1788c62b422",
  "paper/research/20261104-inspection/qualification/dev-cases.v2/dev-L1-01.json": "eb5e10b7a7bb3727853d0b5d7868200819f646585cf3b6082a01a58039e95f40",
  "paper/research/20261104-inspection/qualification/dev-cases.v2/dev-L2-00.json": "fb011f0b3960c0a730ba7e4f68c644b63aeb3c6cce0b01be2caf5c0f76ab68d3",
  "paper/research/20261104-inspection/qualification/dev-cases.v2/dev-L2-01.json": "cc72a42df0879bb33c3c39a91318059c10e51bef5663246b58e5f417d4c4f30c",
  "paper/research/20261104-inspection/qualification/dev-cases.v2/dev-L3-00.json": "ae583cf527b4db9e0768231bdeaf595a20e6a344db07aabe31f0858a1a5275f3",
  "paper/research/20261104-inspection/qualification/dev-cases.v2/dev-L3-01.json": "8ccea56f7d8cc1e562ad86a9640b7bed55446aff396fa93a9942d50bd91ba9e3",
  "paper/research/20261104-inspection/qualification/dev-cases.v2/manifest.json": "238d5a321e01ca54adc0507e5035c37e271b3a3eeb3ad20d4d81e89211df5a72",
  "paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/experiments.py": "4dffd762966ef27d78fd521341aefb2145895b26ad9d382d16f04f1bab6eec02",
  "paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/production.py": "8e8b41849893980267378fa2843af69b0f65555cac5523bfc33f7a3e37275770",
  "paper/research/20261104-inspection/runtime/autoresearch-state-reader-v1/lib/science.py": "9f7a8eecb2741f18afaccfb3da1ba33951d88c2bd4fd1983dfc1b5953649cfaf"
}
```

