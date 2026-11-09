# scenario v2修复独立复审 v1

日期：2026-11-06。审查者 `/root/mission_review`，同模型家族独立上下文，不是人类领域或跨模型审查。工作树 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。只新增本报告和附属 `scenario-adapter-repair-review.v1.evidence/`；未改生产、控制器、原输入、旧报告或旧结果。没有启动physics、replay、长episode或API，也没有读取进行中的L3完整运行作为最终证据。

## 结论

**原F1–F5的既定反例均已在当前修补源码上得到正确处理，可关闭这些原缺陷。** 新连续余量对当前声明的CaseEntity轨迹族成立；24份dev-cases.v2全部通过重验，只有L2-03改变，其余23份字节相同。L2-03由原seed在新几何规则下精确重建，仍属原lineage，不增加独立root数。

**还有一个新P2残留R1：结束校验时冻结副本被删除，会使finally抛异常，run.json停在running。** 它没有误标成功，但未保存已知终态失败。父任务已接受在当前完整运行结束后修复，本报告不等待中途改源，也不将拟议修复写成通过。

这次完成的是输入/几何/绑定修复复审，不是G4 GO。没有以23个小测试、完美证据fixture、旧L1单例或正在执行的L3结果替代场景可解性、方法比较、完整资格和人类领域审阅。human_review_pending。

## 原F1–F5复验

原报告、反例及 `qualification/source/scenario-adapter-v1/` 保留。本次直接读取原 `scenario-adapter-review.v1.evidence/admission.json` 中的case，经当前from_dict/validate_case重验，并对数字字段及worker/main补充独立检查。

| 原缺陷 | 当前实测与结论 |
| --- | --- |
| F1，.191m半径与601检查同相导致采样间越ROI | 原case被 `unsupported trajectory` 拒绝。新增radius=60、heading=0、bend=0的包线内反例，601点仍全部inside，但182.83185307179585时真实峰值outside，现被连续boundary guard拒绝。**关闭。** |
| F2，10个完全共轨代理通过容量上界 | 原case被 `ground proxies cannot be certified separated between samples` 拒绝。所有有dynamic的pair均有连续中心距离下界，static/static原45m规则保留。**关闭。** |
| F3，bool数值与非string标识被接受 | 原复合反例被身份检查拒绝；另分别测试5个实体scalar bool、bool origin、三维origin、数字key、数字case_id/lineage，全部ValueError，避免只靠首个错误掩盖其他字段。**关闭。** |
| F4，分离共线边误判相交 | 原九顶点合法polygon完整通过，1480个coverage cell；当前bbox排除分离段，现有测试也保留真实重叠拒绝。**关闭。** |
| F5，worker可使用与初始copy SHA不同的字节仍completed | 同一原seed改写反例在worker读入时operational_failure，ScenarioMission未创建；结束时再改copy字节则main退出1、manifest=operational_failure/private_case_unchanged=false；不变副本退出0且unchanged=true。**原错误字节获准问题关闭，另见R1。** |

F5用真实worker/main入口配全flight/process双替身检查，0实际物理步，没有把假world的completed作为任务成绩。worker now接收expected_case_sha256，在一次read_bytes后即比较，再解析同一份字节；世界记录private_case_file_sha256，business audit保留canonical case digest/private case。main传入初始hash并结束再核验copy，避免原路径读取与初始绑定脱节。

## 连续界为何成立，以及它没有证明什么

位置：`eval/inspection_cases.py:186–225`。当前开发族限制为radius60..85、|heading|≤.12、bend=0、0≤speed≤1.6。bend=0时

```text
p(t) = p0 + R_heading * [radius*sin(speed*(t-appearance)/radius), 0]
|p'(t)| <= speed
```

单实体601个均匀点覆盖[appearance,600]，步长Δ=(600-appearance)/600。任意时刻距最近采样点≤Δ/2，所以位移≤speed*Δ/2。代码不是把“采样多”当证明，而是对这个位移上界留出余量：

- ROI：每个采样点inside，且到全部有限polygon边段的最小距离严格大于margin+1e-8。若中间出界，连续路径必先穿边，与距离界矛盾。bbox相交检查与非零边/面积等原规则仍约束简单polygon。
- building：对封闭轴对齐矩形计算真实欧氏距离，要求每个采样点距矩形大于2+margin。到固定集合的距离是1-Lipschitz，因此连续中心clearance>2m。
- pair：从两者均出现的max(appearance)到600均匀采样。相对位置的速度上界为speed_a+speed_b；采样距离要求>3+(speed_a+speed_b)*Δ/2，因此连续中心距离>3m。static/static使用45m原规则。

本次新增包线内ROI反例避免“仅限制半径就称连续guard已测”：radius=60、speed=1.5、heading=0、bend=0，峰值位于两个采样点之间，601点inside而峰值outside，当前明确报 `trajectory cannot be certified inside ROI between samples`。该例峰值position=(311.70803349802577,0)，证明当前guard实际挡住这类轨迹，而非仅靠原小半径被拒。

这些界只证明声明解析轨迹的**中心**在ROI内、距building>2、proxy中心间距>3，不是完整球体到ROI边界的半径余量、目标身份可辨识性、25m candidate关联门正确性、传感器始终可见或实际固定翼安全轨迹。不是C≥.8、D≥.5、调度/返航或完整物理可解证明。若以后允许bend非0、另一运动函数、瞬变速度或非冻结实体，须重新证明真实速度上界，不能继续无条件复用本式。

严格数字检查拒绝bool，from_dict仍是JSON输入到本代码原生dataclass的入口。本次没有把这个边界称为任意Python子类/恶意position回调的沙箱；也没有以构造一个能覆盖私有方法的Python对象来否定有限JSON轨迹族的界。

## dev-cases.v2不是新增24根

本次读取原24文件与新24文件，分别验证v2 file SHA、canonical digest、重算validation与manifest完全一致，case_id/seed/lineage_group/level全部与原对应项相同，master seed相同。first_pilot仍仅L1/L2/L3各00/01。

| 项目 | 独立结果 |
| --- | --- |
| 原文件按新规则重验 | 23通过；仅dev-L2-03因pair separation失败。 |
| v2字节变化 | 仅dev-L2-03；其余23个文件逐字节等原输入。 |
| v2重验 | 全24通过，manifest parent SHA精确指向v1。 |
| 同seed重建 | 调用当前generate_dev_case(原case_id,原level,原seed)，结果dataclass与新L2-03完全相同。没有另取新seed。 |
| 旧L2-03实际几何缺陷 | 独立0.1s时间网格发现445.0s时truth-008(dynamic inspection_vehicle)与truth-011(static bank_anomaly)距离.960690637884352m。这是实际轨迹几何距离，不是仅由保守余量造成的误拒。 |

重建函数只读固定seed与几何/规则/容量校验，不接收或读取策略成绩；本次没有运行L2-03 policy。父任务声明它此前未跑policy，本审查不将这个历史声明扩为扫描所有外部会话的证明。manifest显式写同seed语义修复、保留原输入/结果且不增加独立根。没有发现按方法成绩选择新布局的代码路径。

旧L2-03的canonical digest为 `ed93254799d5b4634b9ed50c974f21f67a62b387f189f35a1d1f38c3e4d38947`，新为 `f56e7fd7c80b3ce1bcf4df77a94dcf6e8ee5724403e06845a880cb1119078cc9`。对应file SHA为 `12806f5d90bde3a4b8ac99d89d91c00c5ca7dc95d6d85029dafdb83243b5bc49` → `be180b059870870b6e55e616df59ce14680e62df6d42dc85d42a76b771a95bca`。

v1/v2是同一24 lineage的输入版本，不可合计48根、将新L2-03当额外独立样本，或把不同版本混作相同场景配对。公平比较需按canonical case digest、source、预算和版本配对；必要capacity bound继续只说明完美报告数量未先天超10，不代表实际可解。

## R1 [P2，未关闭] 结束校验副本缺失时不落盘终态

位置：`eval/run_inspection_reference.py:239–240`。finally直接调用private_case_path.read_bytes，没有捕获OSError。独立main双替身在worker已完成、主进程结束校验前删除冻结副本，得到：

```text
exception = FileNotFoundError
main exit result = raised, not normal return
persisted run.json status = running
private_case_unchanged = absent
```

因此没有错误的completed/pass，但已知输入验证失败没有保存为operational_failure，运行恢复端只能看到悬空状态。原F5的字节更改已安全失败，不能因本残留将其说成完全没修；同样不能把新删除分支称为已完成failure记录。

父任务已明确接受后续在当前600s结束后catch OSError、记录private_case_unchanged=false/verification_error并继续保存known operational_failure；本次按其要求冻结当前报告，不改源、不等待拟议修复。修复后需要另一次独立小probe检查deleted/unreadable copy与正常分支，而非回填本报告或正在运行的源码绑定。

## 复用与输入隔离范围

独立将当前 `envs/inspection/` 的12个.py与 `qualification/source/executor-v19-reference-v2/` 对应文件比较，全部字节相同；ScenarioMission也仍为d2b0fbfc版，没有评分/公开信息/控制器改动。可以复用上一报告对L1 Q=(C+S)/2、D=null、完整静态分母、withdraw不缩分母、H/E门控和L2/L3权重虚方法的检查。本次23个cases tests也通过，但没有重做整套旧平台资格或把原G1/参考数据升级为新场景总体结论。

新hash传递与失败校验只在worker/main侧，未传给policy构造函数；CaseSpec/seed/真值/未来事件仍属私有worker与最终评测审计。新master/seed没有因同seed修复而成为公开actor输入。持续保留“有权读取本机输出文件的任意恶意Python代码”不在RPC隔离保证中的边界。

原L1 v1-full完整文件/manifest保持原版本，本报告没有改它们、没有为其添加新private_case_unchanged或新校验标签。即使L1-00输入在v2字节相同，也不能把本次新runner guard验证倒灌旧运行。原适配器报告指出的新case replay/自定义sensor日历资格工具适配仍是独立集成事项，本次没有凭五项修复自动关闭这些事项。

## 实际检查及证据

下列命令在仓库根执行，全部没有创建实际物理世界。verify中的0.1s扫描是解析地面轨迹函数计算，不是积分/新episode。没有复跑长physics或他人的全库测试。

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/scenario-adapter-repair-review/verify.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/scenario-adapter-repair-review/freeze.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/scenario-adapter-repair-review/numeric-continuity.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_cases.py -q -p no:cacheprovider --basetemp /tmp/scenario-adapter-repair-review/pytest
```

前三者退出0，意味着各自预期断言成立，**不意味删除副本分支正确**；该分支的异常被探针捕获并存证。测试退出0，23 passed in .97s。本次没有独立重跑父任务所述的receipt guards/全库suite，不将其在跑或已报告的总计冒作本审查执行数。

长期复现材料位于同目录 `scenario-adapter-repair-review.v1.evidence/`，含verify/freeze/numeric-continuity三个脚本与JSON以及tests.log；临时输出位于 `/tmp/scenario-adapter-repair-review/`。脚本的fake process/world仅用于检查真实runner的输入/终态分支，不可当真实run或分数证据。附属binding.json保存七文件、当前source及输入manifest的SHA，其SHA为 `60faf5857f294471678f0e4ae1b672ae8059fa6dfd723d34663fc60dadf426a0`；复现需先创建临时目录，已存在的fake final目录应另用新临时路径，不能覆盖真实结果。

## 当前版本绑定

下表是本轮实际读取/执行SHA，不延用原v1的case/runner SHA。inspection core保持旧值是已核对字节相同。HEAD=`270c6f0946b3be59527b2b77982f75a70da7ef77`，工作树新增文件未提交，内容按SHA绑定。qualification路径相对 `paper/research/20261104-inspection/`，其余相对仓库。

| 文件 | SHA-256 |
| --- | --- |
| eval/inspection_cases.py | `7537b17aa0cbcfd4cffdb00c1d35e45e56217f477e055db77ba80724c0868cef` |
| eval/inspection_scenarios.py | `d2b0fbfc77ee40e06115393624a3ac2f36d11fa21981590c71f3bc0beea5dcec` |
| eval/run_inspection_reference.py | `1f4ef28d9935a6293f9ce6ea97d3472b2781f9aee3dc166ba20b8804c2e719b1` |
| tests/test_inspection_cases.py | `09af63773c6e0d30283c46f7c0fba16e69fa53babd86bd7961e5807289d69e66` |
| envs/inspection/mission.py | `715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7` |
| envs/inspection/scoring.py | `bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a` |
| qualification/dev-cases.v1/manifest.json | `b5a00d2cf88549dcabce17667825fa8e1f171052516b853cb3095f3f218ee6b9` |
| qualification/dev-cases.v2/manifest.json | `238d5a321e01ca54adc0507e5035c37e271b3a3eeb3ad20d4d81e89211df5a72` |

后续R1修复与实际L3运行应分别绑定其真实source/输入/完整结果；本轮结论截止于上述源码和0物理步检查，不为未来结果预先授予资格。
