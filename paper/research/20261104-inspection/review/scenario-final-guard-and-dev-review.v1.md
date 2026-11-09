# 最终输入guard与L3开发证据独立复核 v1

日期：2026-11-06。审查者 `/root/mission_review`，同模型家族独立上下文，不是人类领域或跨模型审查。只新增本报告及 `scenario-final-guard-and-dev-review.v1.evidence/`，没有改源码、旧报告、旧manifest、输入或实际结果。没有启动physics、长episode、API或重放；只执行零物理步guard/分析探针和已存数据复算。B1进行中的运行不在本报告结果范围内。

## 结论与当前未闭合项

**上一报告R1的既定反例已关闭。** 最终guard在copy被删除而world仍在时，main正常退出1，manifest保存operational_failure、private_case_unchanged=false和verification_error，不再停在running。正常副本退出0，改字节副本退出1，读入前改字节仍不创建ScenarioMission。

**实际dev-L3-00 B0的600s/1×记录、原始源/输入绑定、完整分数和25组精确重放证据在本报告范围内通过。** 独立重算得到C=.9844425237683665、S=1、D=92/331=.27794561933534745、Y=.7541293810345713、H=E=1、SR=0，与原保存值完全一致。它是一个预选dev单例，不能称强动态追踪、正式方法结论或G4 GO。

**仍有两个具体P2，均已由父任务接受后续修复，但当前未关闭：**

1. copy和world同时缺失时，input检查已经false，但失败状态赋值位于world.exists分支内，CLI错误地退出0并保存completed。
2. common-calendar入口未validate传入的demand_calendar，接受当前CaseSpec校验拒绝的priority_at=0或非法schema，仍返回重计分结果。

这两项没有污染本次已绑定的有效L1/L3记录。原报告继续保留，拟议修复不回填当前源码或正在运行的B1。clean兄弟尚未真实执行，common-calendar只完成离线报告重建/换权重，**事件因果结果尚未完成**。human_review_pending。

## 最终guard实测与新的组合缺失反例

当前 `eval/run_inspection_reference.py:24–30` 的check_private_copy捕获OSError并返回False/error，finally在248–250行保存这两个值。用上一轮同一个真实main/worker入口加全flight/process双替身，0实际物理步重验：

| 情况 | 当前实测 |
| --- | --- |
| 原副本不变、world存在 | exit0，manifest=completed，unchanged=true。 |
| worker读取前改变seed字节 | worker=operational_failure，ScenarioMission未创建，错误明确为private case changed before worker read。 |
| 结束时改变copy字节，world存在 | exit1，manifest=operational_failure，unchanged=false。 |
| 结束时删除copy，world存在 | exit1，无未捕获异常；manifest=operational_failure，unchanged=false，verification_error含缺失路径。原R1关闭。 |
| **结束时copy与world同时删除** | **exit0，manifest=completed，unchanged=false。新P2仍存在。** |

新问题位置为 `run_inspection_reference.py:251–257,262–263`。false输入检查没有在world条件之外强制失败；缺world也没有独立使CLI失败。探针在worker已发done的结束阶段删除这两个文件，不是伪造业务分数，也没有执行真实world。外层receipt校验器会拒绝缺world，因而不构成当前receipt层的合格样本；CLI本身仍违反其已知输入故障应保存终态失败的约定。

父任务已接受统一world读取/解析/状态异常处理、将input false失败赋值移到条件外、缺world显式失败。本报告冻结f357版的真实行为，不把该修复提案称已通过。后续独立小probe应同时保留正常、改单文件、删copy、组合缺失及坏world终态分支。

## 原实际L3与当前guard的版本分离

实际路径为 `qualification/dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight/`。run/world均completed，requested/simulated=600，wallclock_1x，实际physics_steps=144000，wall_seconds=600.0089673750335，max lag=.03050283269647025 s。sources_unchanged、dependencies_unchanged、private_case_unchanged在原运行记录中均true。

16个run嵌入源码的SHA与 `qualification/source/actual-dev-L3-v2/` 对应字节全部一致，archive binding正确指向原run SHA；61项记录依赖当前逐项匹配。copied private-case的file SHA在run/world一致，与dev-cases.v2/dev-L3-00原文件逐字节相同；world.private_case也等于该copy，canonical digest为 `54322e6e9117abe9d3b4c42d9b37d8f62eeb5edfa0b007890a9fce8a94b7154f`。

实际L3使用runner `1f4ef28d...`；当前唯一不同的运行源是 `f357415b...`，新增check_private_copy及finally错误字段，worker/正常业务路径无变化。附属runner.diff保留唯一差异。这个有界工程判断支持有效输入的原业务证据继续按**原版本**解释，不能将新guard倒灌旧manifest。其余env/core与原运行字节相同。

## 完整物理覆盖与独立评分

从保存raw误差记录独立base64/zlib解码、按17列little-endian float64重计：144000行，每行t=i/240（1e-9容差），全部有限，raw SHA正确，另有6000个10Hz位置点。最大XY误差7.554937514984477m，明确10m tube下XY breach=0，记录高度区间breach=0。没有用新默认解释旧门槛。

这些原始列是逐步误差/Z/区间，不能补造full240Hz状态、完整动力学重放或精确里程。H从全步safety.violations={}聚合；E从原全步hook的实际dwell与剩余航程审计输入聚合，本报告没有从10Hz位置重新发明精确odometry。四机实际exit job均succeeded，剩余航程约7898..7908m，高于1000m余量；预算是18000m航程，不是电池模型。

独立标准库业务重建没有调用生产scorer/common-calendar：按原接收顺序恢复ReportLedger语义，固定首次20m关联、名额、原证据、同测量时truth、静态最后合法版本和动态先筛/最新/再判正确性。544个版本占8个报告ID，无无效provenance；全部1668条观测与曝光audit原值相同，非background truth由CaseEntity解析轨迹在原measured_at独立复算一致。

| 分量 | 完整分母及独立结果 |
| --- | --- |
| C | 3055个实际交付coverage包，1139/1157 required cells=.9844425237683665。 |
| S | 5个required static，权重9，分子9，S=1；4个normal不计任务分母但保留为干扰物。 |
| D，truth-005 | 32/120=.26666666666666666；observed相同，60格保留。 |
| D，truth-006 | delivered60/95=.631578947368421；observed70/95=.7368421052631579。 |
| D，truth-007 | 0/116=0；observed也0，全部60格仍在分母。 |
| D总体 | (32+60+0)/(120+95+116)=92/331；3实体、180格未按发现数量缩小。 |
| Q/Y/SR | Q=Y=.7541293810345713，H=E=1；D<.5，SR=0。 |

每个实体的所有selected ID、格权重、quality/zero_reason与保存结果精确相同。8个报告为5个静态、2个真关联动态和1个null关联动态；null报告不抢救为遗漏truth-007，也没有退款。高Y不证明三个动态目标均有合格报告。

105个实际RPC均accepted：34 survey、31 cancel、25 inspect、8 submit/revise、4 observe、3显式exit；第四个exit来自外生退休。67个作业/192个转移合法，无active重叠，最终全terminal。inspect为22 succeeded/3 failed，observe为2 failed/2 cancelled，survey为33 cancelled/1 failed，exit为4 succeeded。ACK数不能当成功作业数。本次未把两个observe的local gap failed解释为历史合法报告全部归零。

## 随机日历与本次实际覆盖

私有日历为priority196；机1断链[244,279)；机0降质[303,343)；机3退休384。实际public events恰为196 priority_update/card2、384 exogenous_capacity_exit，没有误用旧180/360事件。

- **调权**：truth-006的193..196格仍权1，实际196s证据 `obs-1-392-0` 只归该格；196..199格权2。truth-007的195..198格跨过196但起点尚早，仍权1，198..201才权2。缺观测的格也保留正确权重。
- **断链**：机1在[244,279)没有观测arrival。279恢复时交付64条观测，原measured_at=243.5..278.5，其中63条严格超过基础.5s延迟；coverage交付71包，其中70包严格迟到。243.5包ready=244，恰落断链起点被缓冲。
- **实际observe route**：断链采集中有39条自动report证据可join曝光时机1原observe job/report_id。全run 536条自动report无路由/原时间戳/原值join错误，另8条为显式submit。`obs-1-617-0`在308.5采集、job-0038同刻failed，309仍按原route到达。不是假设所有补传包都及时得分。
- **交付与观察质量区别**：truth-006 observed比delivered多10个权重分，反映本条trace的时效门诊断，不是完整策略的clean兄弟因果效应。observed也要求最终审计交付，不把pending字段当已到达。
- **降质**：机0的[303,343)共有8条sensor audit，quality全部<.8。此处核对实际窗口与门槛，不用8样本宣称校准了总体退化分布。
- **退休**：机3在384被取消observe并创建exit；384之后target、coverage、opportunity、background新采集四通道均0。旧coverage383.5→384合法保留；本run没有旧target Observation跨退休到达，不补造该正例。exit最终448.51666666666665实际成功，其余机也有实际dwell。

本run **handoff=0**，不能称随机跨机交接、每种cancel后新receiver路由或所有在线/退役组合均已实际覆盖。外生退休后旧Observation的正例仍来自此前独立退休fixture；这里只证明实际发生的旧coverage路径。

## 25组重放与cache复用

`L3-00-replay.v1.json`绑定本次实际run/world SHA和当前c3b4609e版replayer；105/105调用、response_discrepancies=[]，25个字段组均native true，包含score/safety/业务、执行内部journal、raw误差trace及case_sha/private_case/adapter。它比旧21组重放多了raw误差和新场景字段，仍没有full240Hz状态。receipt.source_root=null，表示当时用匹配current源执行；源码随后已归档，但没有据此虚称又从归档source-root运行一次。

本次只核验最终重放产物与代码/输入绑定，没有重新执行。110.83828445791733s加速重放不是新策略样本、第二个1×或新planner/model时延；按原tick注入内部响应并重算proposal不产生新的墙钟延迟实验。

对真实完成的receipt树复制到/tmp，在command_factory及Popen均禁止dispatch的条件下调用execute_local_episode，实际返回cache_reused=true、actual_attempts=1，Y与原结果相同，只存在attempt-00，未调用工厂。原receipt/world没有写入；本次验证的是已绑定缓存复用途径，**没有新flight或新增样本**。

## common-calendar报告重建与新P2

位置：`eval/inspection_common_calendar.py:21–81`。原报告按receipt/source另行绑定后，该模块在arrival≤report接收时注册证据，按原report接收顺序重新提交，并要求原response逐项相同；先复原整个原score完全一致，再在同geometry/motion/seed下换需求日历。它不重新跑policy、sensor、通信、退休或physics，H/E和剩余航程保留原实际值。

本次对实际L1 v1-full、实际L3均复现原score exact；L3产物与现有 `L3-00-common-calendar.v1.json` 完全相同。改变geometry或篡改原保存D的反例被拒。共用需求日历必须有原geometry/seed/实体；不是把不同root直接称作事件兄弟。

**P2：demand_calendar没有被validate_case。** 现行64–77行只校验world内geometry及部分字段相等，未验证新日历schema/事件范围。实测：

```text
replace(case, priority_at=0)  -> validate_case拒绝；rescore_common_calendar接受。
replace(case, schema='unsupported-calendar-schema') -> rescorer接受。
```

非法priority日历仍导出dynamic分母360及D=.32222222222222224。它没有改变当前已核验的原L3数据，但使“已支持的同日历”输入范围不明确。父任务已接受入口validate修复，当前852068a4版仍未修。若将来有意支持超出CaseSpec的纯分析日历，应另定义显式类型/契约，不能静默绕过现有场景校验。

另对同一实际L3 trace使用合法clean L2需求日历，离线结果D=.32222222222222224、Y=.7688882486635297、静态分母5、动态180；C/H/E保持原值。这个变化只是**同一已发生trace换权重**。不能据此写“无事件时策略表现更好”、事件损失值或已完成clean兄弟运行；clean时policy观测/计划/采集/交付可能不同，真实兄弟尚未跑。

模块明确把run/source/receipt绑定交给调用者，返回值本身只有case digest。本报告通过附属binding将当前derived output与实际world/source SHA连在一起；后续分析也须保留外层绑定，不能只凭original_score_exact布尔值接纳任意记录。

## 命令、证据与SHA

实际执行 `/tmp/scenario-final-review/` 中guard.py、guard-combined.py、business.py、calendar-and-cache.py、bind-and-events.py，均以 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B <script>` 运行，退出0。零退出表示探针的预期断言成立，**不表示其中存证的两个P2行为正确**。没有重跑测试套件、physics或replay。归档按source snapshot核验；本次没有据源码字节一致宣称已从归档环境执行。

长期证据在同目录 `scenario-final-guard-and-dev-review.v1.evidence/`，含五脚本/JSON、两个当前受审源码副本、replayer副本和runner.diff。binding.json SHA=`305e6cfcd8dfa7f70f3742d701dcc88b13d402000e4b58b4f7af10a938559880`，绑定所有附属文件及原run/world、replay、common输出和terminal。临时cache-copy及fake final目录不是科学结果，不能并入数据集。

以下qualification路径相对 `paper/research/20261104-inspection/`，其余相对仓库。当前runner与实际L3 runner分别绑定，不互相回填。

| 文件 | SHA-256 |
| --- | --- |
| eval/run_inspection_reference.py（当前guard） | `f357415bbc5d569c57013d721b236a593024f837a0374651b9f49eb6c1f0e8cd` |
| qualification/source/actual-dev-L3-v2/eval/run_inspection_reference.py（实际运行） | `1f4ef28d9935a6293f9ce6ea97d3472b2781f9aee3dc166ba20b8804c2e719b1` |
| eval/inspection_common_calendar.py | `852068a4d461be077b34f8e5bf0c45f03960ffed3cbbc60ba5f5ba6f272bcf5b` |
| eval/replay_inspection_reference.py | `c3b4609e880ca6d95385e1a2ea5f9adae7367cd47d3aaa4db3568f5a40dd5387` |
| eval/inspection_episode_receipts.py | `59f88d5a3ec90497e19ec91ac9b914f09fe6a2303d029e6aecbb6fb2f958387b` |
| qualification/dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight/run.json | `1b54682cb3a90739a0c185e78d5334120fb1c54ed2ee3a16281f360bdd0387b8` |
| qualification/dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight/world.json | `faaf04929e7ccd4558f96a4b346b30e241fcccd902c2224d3a0d6b2112b41bca` |
| qualification/dev-receipts.v2/dev-L3-00-B0-0/attempt-00/flight/private-case.json | `ae583cf527b4db9e0768231bdeaf595a20e6a344db07aabe31f0858a1a5275f3` |
| qualification/dev-receipts.v2/dev-L3-00-B0-0/attempt-00/terminal.json | `6d3bdd5e4abbaedf298f9f6eab9d68636043dae32b86397155191e55a1bfdbd6` |
| qualification/dev-receipts.v2/L3-00-replay.v1.json | `2bb18b57eec36175ccc3f30cd65a9d1a03a0d29ad39e58141e032e62de8d69a1` |
| qualification/dev-receipts.v2/L3-00-common-calendar.v1.json | `31bade10c1ec322c6118ec577cef84913185408c13363d9ebde2fcf1305cfdf1` |

原raw误差SHA为 `ca740d3c2dc300cfab0ef223c202f8889d700c70de4ff1ff76022226837b471e`。HEAD仍为 `270c6f0946b3be59527b2b77982f75a70da7ef77`；本轮所有资格判断截止于明确绑定的版本，不预先放行后续两个P2修复、B1结果、clean兄弟或G4。
