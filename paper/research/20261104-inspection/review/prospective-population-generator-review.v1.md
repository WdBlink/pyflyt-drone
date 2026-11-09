# 前瞻总体生成器独立审查 v1

审查者 `/root/control_review`；同模型家族、独立工程上下文。只审查新增的纯几何生成器与冻结总体的一致性，未运行 PyFlyt、LLM、owner 操作或真实科学 suite。旧 frozen classic/core/CaseSpec 未改。

## 结论及版本边界

**当前已复验的 `32788a37…` 关闭本轮发现的两项 P2。** `999b5a97…` 已修正 joint proposal，但 validator 尚有四个实际漏项；`32788a37…` 同时拒绝四反例、保持合法 joint case/audit，并实际捕获一项“601 点都超过 45m、点间却不足 45m”的连续混叠反例。未发现还需阻断当前纯开发几何 kernel 的实质开放缺陷。

这仅支持修复后的开发几何 kernel，不授 G4、物理 OOD 可解性、完整总体来源鉴定或真实独立 suite 资格。全命名空间 fresh master/去重/lineage 登记及新 schema worker 接线仍待实现；legacy worker 不支持此 schema 的限制正确保留。不得把诊断 case、同 seed 跨层测试或几何通过升级为正式 roots/飞行。

版本协调曾有时间窗口：首次源码文本展示局部重抽，之后取得 SHA/保存时已是 joint 修复；此前消息一度误称 `999b…` 为旧版，现明确更正。证据目录 `inspection_population.initial.py` 是 **joint 修复版 999b**，不是原始局部重抽快照。历史局部重抽源码由其唯一 place 逆补丁重建，SHA 精确为主 Agent 提供的 `516ef909…`，并已重新运行同输入对照；这是**重建历史对照，非当时原子捕获**。报告不把旧问题绑定到修复后 bytes。

## F1 / P2：顺序局部重抽改变冻结 joint law——999b 已关闭

冻结 [population.ral.v1.json:36](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/planning/population.ral.v1.json:36) 的总体为规定 joint draws 仅按固定几何/容量谓词条件化。历史 `516ef909…` 的 place 先对每个静态实体局部重抽直到 ROI/45m 合格，再最终全 case 验证。先前对象位置会改变后续可行区域的归一化，因而不是所声明的 joint 条件总体；随机打乱 hidden keys 不消除带 class 标签的位置分布差异。

独立离散精确反例：两个带标签位置独立均匀取自 `{0,1,3}`，要求距离≥2。整体条件化的四个允许有序对各为 1/4。局部最多 K=1000 次重抽时，记 A1=1−(2/3)^1000、A2=1−(1/3)^1000，接受后的 `(0,3)/(1,3)` 各为 A1/(2A1+A2)≈1/3，`(3,0)/(3,1)` 各为 A2/[2(2A1+A2)]≈1/6；使用 Fraction 精确确认两类均不等于 1/4，不是仅用无限极限或有限连续样本拟合下结论。

修复后的 [inspection_population.py:170](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_population.py:170) 每实体仅一次无条件 proposal，完整 case（含事件/风）抽完后统一验证/拒绝。inner 1≤1000，未修改 frozen 范围或 guard。每 whole candidate 保留索引、全部 proposal 及最终拒绝 reason；1000 whole 后抛带 audit 的 exhaustion，不改 root 或按成绩补样。

同一显式软件 fixture 的重建旧版为 1 个 whole candidate、max local candidate_index=1；999b joint 为 3 个 whole candidates、local index 全 0。两份 `.as_dict()`/digest 确实不同（不是跨模块 dataclass 类型比较的假差异）；该单例只展示代码路径改变，不被当作真实总体频率/效能证据。

## F2 / P2：validator 不是完整 schema/support gate——32788 已关闭已列反例

在精确保存的 joint 版 `999b…`，一个正常生成的 devext ID/L1 case 经以下四次单项变异，`validate_population_case` 都实际接受：

| 变异 | 违反内容 | 32788 实际复验 |
| --- | --- | --- |
| 一个 hidden key 改为空串 | 非空稳定实体身份 | ValueError 拒绝 |
| hidden key 改 True | 原生字符串身份 | ValueError 拒绝 |
| wind 改 `(True,0.0)` | 原生有限非 bool 数值 | ValueError 拒绝 |
| road_obstruction 改到 `(300,-200)` | ID target x 支持域 `[-240,240]`；几何本身仍合格 | ValueError 拒绝 |

这不表示 sampler 正常生成过坏字段；它证明旧 validator 不能作为后续新 schema admission 的充分检查。修复版 `_finite`、key/numeric guard、静态/normal/ID 动态条带及 distinct、OOD 圆心可判定支持域、128bit seed 上界现已加入。原四反例在同一个合法旧 joint case 上重放，分别命中身份、风、support gate，没有只由一个 SHA gate 掩盖。

合法同 seed ID/L1 的完整 case 字典和 audit 在 999b/32788 逐字段相同，仍 3 个 whole candidates，确认这些 guard 没有重新定义合法 joint proposal。ROI 是否来自特定参数族、root 是否独立、是否重用了隐藏配置，仍须 generator/source/seed/audit 来源证明；几何 validator 本身不宣称完整生成法鉴定。

## 连续几何界、容量与抽样语义

当前解析 `CaseEntity.position` 是旋转后的 `(r sin(phi), b r(1−cos(phi)))`，其速度范数≤`abs(speed)*max(1,abs(bend))`；ID bend0 与 OOD bend1 都满足。每条从 appearance 至600的601点含端点，最近样点时间距≤Δt/2：边界/建筑使用单体速度×Δt/2，pair 使用两体速度界之和×Δt/2。样点 inside+边界余量对 simple polygon 的点间 containment 保守；building2m 余量成立；pair≥45m 同时强于 groundproxy3m。不是把601采样本身当连续证明。

独立实际混叠 fixture：从合法 ID/L3 解析轨迹构造一个静态 normal，601 点最小距离 **45.00010645068812m**，某个点间真实时刻距离 **44.99996451638133m**。速度时间 margin **0.5377488462027722m**，当前 validator 报 `continuous 45m identity separation guard`。逐项核对全部 pair 后，只有预期的 truth-000/truth-005 pair 违反 margin，未被其他 pair 的错误遮盖。这直接覆盖原测试仅在 appearance 时刻放置30m near miss 没覆盖的点间情形；未调用物理引擎。

ID 仿射/独立 jitter、OOD 有序径向角/独立 radius、新圆心/heading 曲线、数量/出现/事件/风范围与冻结文字对应。OOD origin 的构造等于 `center−R(heading)*(0,radius)`，中心的反算 support guard 与解析式一致；新重叠链路/相机日历仍只属于 L3。五个 SeedSequence 子流分别为 geometry/entity/events/wind/sensor；sensor seed 不随 placement/rejection 的调用次数更新。这里核对的是声明 PRNG 下的实现，不证明真实 root 独立性定理。

必要容量 bound 正确按最终静态权重、完整60个动态格权重，累积最高 perfect-report 权重到 S≥.6/D≥.5 后求最少报告数；两类相加与10名额比较。它不证明 coverage、实际相机视图、航迹/交接、四机安全/退出、实际时间或策略能达到门槛。OOD 几何通过不能继承旧 G1–G3 包线之外的物理资格。

## 测试、范围与尚未实现的边界

读取主 Agent 的日志：初始10 tests/4.56s，以及 joint/schema 修复14 tests/16.46s；本 Agent 未重跑全套，不把主 Agent 自检冒称独立复跑。独立实际执行为：保存版本的旧/joint 同输入对照、有限 K 精确概率反例、999b 四项接受、32788 四项拒绝、合法 case/audit 不变及一个点间 alias 反例；均为纯 CPU，检查 sys.modules 无 PyFlyt/pybullet 导入。

原参数化测试覆盖 ID/L1、ID/L3、OOD/L2、OOD/L3，但同 seed 是显式软件测试，不是四个独立科学 roots。主 Agent 报告 joint 诊断出现 3/8/207 个候选，仍只是有限几何生成可行的几个样例，本轮没有独立重跑 OOD207 或据其证明总体成功率。

当前是单 case kernel，未实现完整 namespace/suite driver。冻结要求的 fresh128bit master 先落盘、全 namespace seed/语义配置去重、lineage、源/版本绑定、候选拒绝持久化与 exhaustion 阻断必须由后续 driver 落实；改变 case_id 不增加独立性，完整 CaseSpec digest 含身份字段也不能单独识别几何克隆。此限制已由主 Agent 明确承认，不作为“当前诊断伪称独立”的指控。没有 formal/holdout 实例，新 schema 仍不得静默路由到不支持它的 legacy worker。

## 精确绑定与证据

| 来源/版本 | SHA-256 |
| --- | --- |
| 历史局部重抽（逆 place 重建） | `516ef909bac05145a397d86135da31eb78b9cae5f006f63fa646c4bc25239add` |
| joint 修复、schema 反例仍开放 | `999b5a97ae49d55f9e58136c05f631be4a720429249dcbdde00300e351c467fc` |
| 当前 joint+schema 修复 | `32788a37a6d1f6d182c6635f8dd3b8a26680be316d68ee828f0e0592c7cf8ab9` |
| 当前 `tests/test_inspection_population.py` | `a776ba91a34f403b8cde3cc62102622cf49795a27960e3e782845dce8ffdf0fd` |
| 不变 `eval/inspection_cases.py` | `7537b17aa0cbcfd4cffdb00c1d35e45e56217f477e055db77ba80724c0868cef` |
| 不变 `envs/inspection/scenario.py` | `c4f56e2783bc21c087c3abdcbfae9a7472feae2bbbe1e007d86bb39b28c40af2` |
| 冻结 `planning/population.ral.v1.json` | `307fbd5e77f841bf4a1b431a478527fa3aa85905694fd0242068d26c729a434c` |
| 冻结 `experiment/design.ral.v3.json` | `83f8f1a72f5261aaa5783c0d190f3788fafcd05bca1e7d16a6901c69c9abf0de` |
| `/tmp/inspection-population-joint-repairs.v1.log` | `30dab8c427a13d7e0c5a459ae74ddb14e972b19f395b01b5ddcb2f657c718058` |

R 下自有证据目录为 `review/prospective-population-generator-review.v1-evidence/`，包括版本绑定说明、999b joint snapshot、516e 重建 snapshot、327 修复快照、反例脚本/结果与全 pair 证明。`counterexamples.reproduce.py` 使用跨模块 `.as_dict()` 比较；实际独立字段比较在 `case-comparison.json`，原脚本/结果保留。所有证据另由 `verification-manifest.v1.json` 逐字节绑定；后续修复快照/结果来自 `/var/folders/n_/2cw9gf5n52z_w6kllyb44ttw0000gn/T/inspection-population-repair-review-hgl7paei/`。报告/证据只在主 Agent 解除 round19 checkpoint 暂停写入后落盘，暂停期间所有新增检查材料只写临时目录。

本报告保留全部旧发现及修复版本边界，不授 G4/物理 OOD/独立 suite/正式性能、人类领域或跨模型认证。未改 production、任何 frozen classic/core/legacy schema、owner、live logs，也没有模型或新物理执行。
