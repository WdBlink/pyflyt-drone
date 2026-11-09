# G4 记录与统计规格独立审查 v1

审查日期：2026-11-05。审查身份：与主 Agent 分离的独立上下文，继承同一模型家族；不是跨模型家族验证。只读代码、原计划和既有合成记录，仅写本报告。未运行物理仿真、真实策略、模型/API、test 或真实 holdout，未产生正式统计成绩。预算 `{}` 不构成付费授权。

**结论：v4 的不等重复矩形记录与 root 均值绑定得到支持；它不是主分析适配器，不能据此放行 G4。** 现有主分析缺失是确定阻断。下文给出可在真实 test 数据前冻结的候选：ID/OOD 分开、等级宏平均、root 内先均值、分层配对 percentile bootstrap 95% CI、null-centered bootstrap-t 双侧基础 p、三对比 × 两域的 global Holm6。其推断保证是有条件的、名义/渐近的，不是小样本精确保证。必须先实现并通过明确的合成预检，再作独立审查。

## 1. 审查绑定与证据范围

本报告绑定 `qualification/adapter-preflight.v4/source-snapshot.json` 及该快照所对应的旧版实现。即使当前工作树稍后改动，也不能用本报告批准新版。

| 证据 | 核对结果或作用 |
| --- | --- |
| `inputs/BENCHMARK_RESEARCH_PLAN.original.md:221-242`，另见 `2455-2461` | root、拆分、重复、宏平均、三对比、失败、SR、事件和删失的规范来源 |
| `planning/recording-adapter.ral.v3.md:3-20` | 正确声明合成、描述性、不代表 G4 放行及未完成项 |
| `eval/inspection_recording.py`，v4 SHA `ef63e3958920f3f21b80864d20f428fd5f3e1c26d204f595a0156902777d5f3c` | 此次记录语义审查对象 |
| `eval/preflight_inspection_recording.py`，v4 SHA `9408109cd712bd7d8ac580afbc2fe3bd200964fc4581361f21ffe691180929b5` | 此次预检覆盖审查对象 |
| SDK `lib/experiments.py`，SHA `4dffd762966ef27d78fd521341aefb2145895b26ad9d382d16f04f1bab6eec02` | attempt、异常、补格和终态的真实实现 |
| SDK `lib/science.py`，SHA `9f7a8eecb2741f18afaccfb3da1ba33951d88c2bd4fd1983dfc1b5953649cfaf` | SDK 只提供描述性 case 汇总，不提供本研究推断 |
| `qualification/adapter-preflight.v4/report.json` | 合成预检的声明和分母 |
| v4 source snapshot SHA `2b0c70b30f8f06d6cb1a4d200ecd31bbf56e9359a3e08149830de36761510511` | 独立只读重算匹配；审查时报告列出的 15 份源文件 SHA 全匹配 |

使用的 pstack 规则为 `/Users/wdblink/.codex/pstack/docs/harness/codex.md` 与 `/Users/wdblink/.codex/pstack/skills/show-me-your-work/SKILL.md`。受限于“只写本报告”，决策轨迹嵌入文末，不额外建立 TSV。没有访问无关会话，也没有另派跨模型审查者；本报告本身是主任务的独立审查输入。

## 2. 已核实的分母、聚合与失败语义

对既有 JSON 进行了只读遍历，未重新运行预检。核对包括所有 results 行、内层身份、start 身份集合、每条外层绑定的 raw SHA、results SHA、均值和 episode 数。

| 既有显式合成形状 | classic 内层 | LLM 内层 | 合计内层槽 / start 文件 | 外层均值行 |
| --- | ---: | ---: | ---: | ---: |
| 主形状 72 root | 216 | 1,080 | 1,296 / 1,296 | 576 |
| holdout 形状 12 root | 36 | 180 | 216 / 216 | 96 |

- 两种形状 start 的 `(case, repeat_ordinal, condition)` 集合分别恰等于内层身份集合，无缺项或重复。672 条外层行的已完成内层绑定数、哈希、算术均值和 `not_an_additional_episode=true` 全一致。
- 所有 classic 外层值为 0，所有 LLM 外层值为 .25，符合夹具 `repeat/4` 的 1 次与 3 次均值。此结果完全由合成函数构造，不能解释为方法差异。
- 每 root 的 planned episode 槽为 `3×1 + 5×3 = 18`；主形状 `72×18=1296`，holdout 形状 `12×18=216`。外层是 `root×8` 聚合记录，不能再加进飞行数。SDK 外层的 `attempted=true` 表示聚合发生，不表示新增飞行。
- `operational-failure` 双 root 夹具：classic 6 槽中 1 条 `failed/attempted=true/operation`、5 条 `stopped/attempted=false/operation`；LLM 30 条及 outer 16 条均为后者。完整计划格得到保留；所有三个 results 的 `execution_status=failed`。
- `nonfinite`、`out-of-range`、`boolean` 三种既有夹具也都保留 6/30/16 格，拒绝不合法 Y，没有把不可分析空值当 0 分。

这支持的是该正常异常路径上的**完整计划分母保留**。未启动、操作失败与真实策略失败是不同状态。不能将 `stopped/attempted=false` 解释为已飞且失败，也不能将 SDK `completed` 解释为任务成功。

`inspection_recording.py:70-75` 要求 measure 返回有限、非布尔的 Python int/float，Y 在 [0,1]，并给 dict evidence。策略无效命令、预算耗尽、timeout、过期报告、安全/退出失败等，应由冻结 grader 完成该 episode 评分，再返回合法 Y 和失败类别；Y 是否为 0 取决于冻结评分，不能一律擦成 0。合法 Y=0 仍是 SDK `completed` 的测量。

SDK `MeasurementFailure` 是另一条路径：`experiments.py:89-92` 会吞下该异常并记录 measurement failure；SDK 自己可按 penalty 做描述性统计。但当前外层 `inspection_recording.py:85-87` 要求每个内层身份在 measurements 中且 `status=completed`，故不能将这种异常作为策略失败的通常接线方式，否则 outer 会阻断。这个限制在文档要求“策略失败应返回评分”下是可接受的，应加入接口预检。

## 3. 确定缺口与准确覆盖边界

### R1：旧 v4 的实际执行顺序不符合原计划；正式 runner 阻断

`inspection_recording.py:62-64` 固定先遍历全部 classic、再全部 LLM；SDK `experiments.py:32` 的 grid 固定 case → repeat → condition。它没有执行原计划 `:232` 要求的分层随机方法顺序，可能将方法与运行时段、资源积压或服务漂移混淆。SDK canonical grid 可继续用作存储顺序，但不能自动充当实际调度顺序。

主 Agent 已告知拟增加预冻结完整 episode-order、所有 recorder 先开、按 schedule 路由到相应 recorder；这是合理修复方向，**本审查未见其新版证据，不批准该修复已完成**。新 schedule 必须精确覆盖 planned `(root, repeat, condition)` 各一次，跨 classic/LLM 内层随机，root 在 domain/level 间按冻结规则交错，实际 order receipt 与计划比较。写 JSON 的顺序不能被误当实际测量顺序。

### R2：start 是尝试意图，不是真实飞行/模型调用开始的凭证

`inspection_recording.py:65-70` 在进入 SDK attempt 后、调用 measure 前写 start。因此可证明函数调用前尝试意图文件已写入；不能证明 sim 已建立、第一物理步已推进、provider 已接收请求，或 job 已进入真实执行。raw 的 SDK `started_at` 也早于 marker。`status=running` 是历史事件字段，当前终态应由 raw/results/恢复索引联合判定。

SDK `production.py:28-38` 使用 `open('xb')`、write、close，无 fsync 或原子替换。它提供常规文件写入与不覆盖不同内容的约束，不保证断电持久，也不消除进程被杀时的部分 JSON。不能以当前 marker 声称已具 crash-safe exactly-once 执行。

最小正式接线需要区分：`attempt_intent_written`、实际 runner/物理开始 receipt、provider request receipt（如有）、终态 receipt。硬中断留下意图而无终态时为 `unknown`，需要按冻结恢复政策审计；不能自动当未启动重跑或计为策略 0 分。当前 SDK 没有该完整恢复索引。

### R3：完整终态补格只覆盖上下文正常展开的异常，不覆盖所有灾难中断

SDK `__exit__/abort` 会在正常 Python 栈展开时补齐未启动格；SIGKILL、断电和解释器崩溃不经过 `__exit__`。设计文件仍可保留计划分母，但 results/raw 可能不完整。正式恢复器必须据计划、intent、实际开始 receipt、raw 识别缺项，保留未知状态并阻止统计，不通过“没有结果文件=未执行”推断。

另外，若错误发生在 LLM 阶段，已 `finish` 的 classic results 保持 `completed`，outer 会失败；不是所有已打开 envelope 在任何错误位置都变成 failed。整次研究必须有全局完成索引，不能单看某份内层 results 放行。v4 只注入第一槽操作错误，没有覆盖中途、最后槽后、聚合中、写 start 失败、写 raw 失败等位置。

### R4：SDK 只校验矩形及文本身份，不证明独立 root、split 或主分析正确

当前 designs 不校验 domain/level 数量、root 谱系、事件兄弟 group、拆分泄漏、生成器版本、实际 episode/provider 证据或经典确定性。72 个不同字符串不等于 72 个独立 root。正式统计输入需独立冻结 manifest，不能靠解析字符串前缀建立科学身份。

SDK `science.py:258-325` 只先按 seed 平均再按 case 作均值和预声明差值；`interval=none`，explicit contrasts 强制 `outcome=inconclusive`，并明确 `descriptive_case_summary` 与 `multiple_contrasts_descriptive_no_multiplicity_adjustment`。本适配器正确保留这些限制。主分析绝不能把其单一 72-root 平均、SD/SE 或可选 normal CI 直接改名为两域宏平均/主置信区间。

## 4. 可预注册的主分析候选

以下是供实现和预注册审议的完整候选，未实装、未验收。父任务在审查过程中明确选择“bootstrap 近似主分析、global Holm6”的方向，因此 Hoeffding 留作预声明保守旁证；不根据未来结果在二者间选择。

### 4.1 输入冻结、分析总体与估计量

冻结清单至少包括：root ID 与独立谱系 group、split/domain、level、生成器及采样版本/seed、主 episode variant、事件兄弟映射、8 配置及各自 repeat 数、方法/模型版本、提示/预算、grader/rule calendar、顺序、invalid_attempt 与恢复政策、统计代码/hash、bootstrap 参数和比较表。

主分析只有 test-ID 48 root（L1/L2/L3 各16）和 test-OOD 24 root（L2/L3 各12），分别计算。真实 holdout 仍关闭；合成同形状夹具不构成接触真实 holdout。日后 holdout 的确认假设、alpha/多重比较与解释需另在打开前固定，不能将本轮 alpha 自动重复使用并声称整个研究错误率不变。

假设：每个 domain-level 内的 root 是从冻结 generator 分布独立同分布抽取，不同 root 相互独立；root 内可以任意相关。模型版本与执行协议固定，无跨 root 记忆/缓存造成的结果依赖；资源竞争、共享服务时段冲击不能被“随机了顺序”自动消除。如果 root 为目的抽取的固定集合或存在跨 root 共同冲击，下述总体推断不能照搬；可继续公开该集合的描述性成绩，并如实标明边界。

记域为 g、等级为 l、root 为 r、方法为 m，R_m 为 1 或3。对预先指定的主 episode variant：

```
x[g,l,r,m] = (1/R_m) * sum_j Y[g,l,r,m,j]
mu_hat[g,m] = (1/K_g) * sum_l mean_r x[g,l,r,m]
d[g,l,r,c] = x[g,l,r,minuend(c)] - x[g,l,r,subtrahend(c)]
delta_hat[g,c] = (1/K_g) * sum_l mean_r d[g,l,r,c]
```

ID 的 K=3，OOD 的 K=2；每个等级等权，每等级内 root 等权，每 root 内该方法的规定 repeats 等权。classic 不复制成三份，也不按 episode 数给 LLM 三倍权重。三个比较固定为 `architecture-M1: B4-M1−B3-M1`、`architecture-M2: B4-M2−B3-M2`、`classic-M1: B4-M1−B_classic*`。classic* 仅由预注册 dev 选择规则在 B1/B2 选出，夹具的 B1 不能当选择结果。

这里“配对”是共享独立 root；不是默认跨模型的 repeat0 与 repeat0 接收相同 provider seed。一次 classic 与三次 LLM 的平均有不同方差和分布，但 root 级均值差仍是定义明确的单个观测。固定 R 时，root 分布包含该协议的有限重复噪声；root bootstrap 不需要再加一层 episode bootstrap，也不把重复视为独立 root。

事件 clean/event 兄弟是附加配对诊断：它们不自动计进1296主槽，也不能将二者的均值悄悄替代主 Y。若未来要用固定 variant 混合定义主 endpoint，必须在 test 前重定义权重及完整 episode 分母。所有兄弟、重复和8方法共同留在同一 root 向量中重采样。

### 4.2 分层配对 percentile bootstrap 95% CI

固定 `B=99999`、NumPy `Generator(PCG64)`、`seed_ID=2026110401`、`seed_OOD=2026110402`，冻结 NumPy 版本和算法；这些数是候选参数，不是已运行参数。

1. 依冻结 manifest 排序（domain 固定 ID/OOD；level 升序；root ID 固定字典序；方法与比较用显式固定表），不能依文件发现顺序或输入行顺序。
2. 每次 b，在每个 domain-level 内，从该层 n_l 个 root 索引中有放回抽 n_l 个。不同层独立抽取；同一个被抽 root 的所有方法、repeats、事件兄弟和诊断同时搬运。不能按方法独立抽样，也不能跨等级池化。
3. 用原始 x/d 在这组索引上重算宏平均及三个差值，得到 `delta_star[b,g,c]`。相同域的所有比较共享该次 root 索引，保留其相关结构。
4. 区间为 delta_star 的 .025/.975 分位数，固定 `method="linear"`：j=(B−1)q，在相邻排序值间线性插值。均值类指标保持原有限范围；不得自动改成 BCa、basic、正态或选择更窄区间。

这些是 pointwise、未多重校正的近似95%区间；不是 simultaneous95%，也不是 finite-sample exact95%。CI排除0不能代替下述 Holm 判定；percentile CI 与 bootstrap-t p 本就不是同一种检验反演，允许二者有表面不一致。SciPy 官方文档给出了 paired index resampling 的语义，也提醒 BCa 在退化分布下可产生 NaN；本候选显式选择 percentile，不能沿用库的默认 BCa。[SciPy bootstrap 官方文档](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

### 4.3 弱均值零假设的 null-centered bootstrap-t 双侧基础 p

对每个 g,c，检验 `H0: delta[g,c]=0`；不要求各等级的效应都为0，不要求差值关于0对称，也不要求方法分布相同。

观测标准误：

```
bar_d[l] = mean_r d[l,r]
s2[l] = sum_r (d[l,r]-bar_d[l])**2 / (n_l-1)
se_hat = sqrt(sum_l s2[l] / (K**2 * n_l))
T_obs = delta_hat / se_hat
```

各层至少2个 root，正式形状更严格要求16/12。数值计算用 float64 和稳定求和/两遍方差，不用不同方法的独立 SE 相加丢掉配对协方差。

对每层构造 `e[l,r] = d[l,r]−bar_d[l]`。用4.2的同一分层 root 索引取 e，计算：

```
delta0_star[b] = (1/K) * sum_l mean(e[l, indices_b_l])
se_star[b] = sqrt(sum_l sample_variance(e[l, indices_b_l], ddof=1)
                         / (K**2 * n_l))
T0_star[b] = delta0_star[b] / se_star[b]
p_raw = (1 + count_b(abs(T0_star[b]) >= abs(T_obs))) / (B+1)
```

等价地，非退化 draw 的分子可用 `delta_star[b]−delta_hat`。须做这条代数等价的合成核对，防止忘记中心化，或把总体宏均值误当各层均值减去。每层去自身样本均值是在模拟“估计误差”的分布：即使真实不同层效应非零且在宏平均中抵消，仍可用于宏均值零假设。它不是宣称各层真实效应皆为0，也没有将 method 标签交换。

Hall–Wilson 明确建议以中心化、并用每个 resample 自己的尺度标准化的 bootstrap pivot 构造检验；其讨论不是本设计的小样本保证。这里有限K、层内iid且各层样本量增长、抽样比例不退化、总方差非零时，可通过层内均值 bootstrap 的一致性与独立层线性组合，解释上述一阶渐近近似。这个“分层配对 root”适配推理属于本报告的推导，不能写成文献直接验收了当前72root实验。[Hall & Wilson, 1991, pp.758–759](https://ruc-econometrics.weebly.com/uploads/2/3/3/8/23387606/two_guidelines_for_bootstrap_hypothesis_testing.pdf)，[Bickel & Freedman, 1981, §2](https://www.math.ntu.edu.tw/~hchen/teaching/LargeSample/references/BickelFreedman81.pdf)

`+1` 防止 Monte Carlo p=0，保证分辨率至少1/(B+1)；它不把近似 bootstrap null 变成精确随机化检验。B有限还有模拟误差，固定一个 seed/批次并公开 exceedance count，不能看到接近阈值便换 seed 或追加抽样。可报告基于抽样尾计数的 Monte Carlo 区间作为数值诊断，但不得把它当基础推断有效性的证明。

### 4.4 零、近零标准误与离散退化的固定保守规则

候选冻结 `SE_EPS=1e-12`，单位与 Y 差值相同；这是数值保护阈值，不是从 test 校准的科学效应阈值。

- 若观测 se_hat≤SE_EPS：不除法，不产生无穷 T；该项 `p_raw=1`、`T_obs=null`、状态 `degenerate_or_near_degenerate`，不能作显著性结论。仍输出效应、root差值和经验bootstrap区间，并标明退化。常量非零差值不自动给 p=0；全零差值则效应0、区间[0,0]、p=1。
- 若某 draw 的 se_star≤SE_EPS，保守将该 draw 计为尾部 exceedance（相当于 abs(T0_star)=∞），无论其分子是否也为0。不删除该 draw，不重抽到有方差为止，不使用随意小数替换分母制造大 T。记录退化 draw 数和比例。
- 候选再固定质量规则：任一项退化 draw 比例 `>0.01`，该项最终 `p_raw=1` 并标 `bootstrap_reference_unstable`；保留本来计算出的尾计数为诊断，不能改用另一个检验寻找显著性。这是预先保守拒绝作结论的工程规则，不是1%处存在某个统计定理。阈值只能在真实 test 前审议/冻结。
- 任一非法数、unexpected NaN/inf、负方差、B不符属于实现/输入错误，阻断该域分析；不能静默改成0。只有上述明确的退化分支可正常产生保守 p=1。

这些规则不修复小样本、零值堆积、稀有巨大损失造成的 bootstrap 偏差。12/16 roots一层的情形仍需预声明合成压力检验；即使压力检验通过，也不能声称对所有有界分布达到精确5%错误率。固定正的近零阈值会对非常小方差情形损失功效，这是保守性的代价。

### 4.5 global Holm6，明确家庭与结论

把 `ID/OOD × architecture-M1/architecture-M2/classic-M1` 的六个基础 p 作为一个家庭，α=.05。三个科学比较仍不变，统计上是六个域特定零假设；不将两域各做Holm3@.05后声称全局FWER=.05。

固定以 `(p_raw, frozen_hypothesis_id)` 排序，得到 p_(1)≤…≤p_(6)。

```
p_adjusted_(i) = min(1, max_{j<=i} ((6-j+1)*p_(j)))
reject_i = p_adjusted_i <= .05
```

逆置换回冻结比较表，公开 raw p、adjusted p、效应方向、全部6个分母与状态。不能因一个比较退化/不显著就缩小家庭；保守退化项p=1仍占名额。真正缺失或操作状态未解决时，不用p=1伪造一个完整成绩表，按第5节阻断。正效应且reject才可说“名义/渐近检验支持改善”；负效应且reject是退化表现，不是改善。未reject不能证明等效或无效。

Holm本身不要求这六个有效基础p相互独立；它也不能修好偏差的基础p。使用近似 bootstrap p 时，整体只能主张相同前提下的名义/渐近FWER控制，不得宣称小样本精确FWER≤.05。官方R文档说明Holm对假设间依赖的适用边界，并给出原论文链接。[R stats `p.adjust` 官方文档](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)

### 4.6 为什么不把朴素置换当主检验；保守旁证

root 内交换两方法标签，等价于差值符号翻转。其有限样本精确性需要相应的随机化设计/尖锐零假设，或联合分布在这些交换下不变；常用充分条件是独立 root 差值在零假设下关于0对称。仅 `E[d]=0` 不给出这种不变性。方法运行先后被随机化，不等于方法输出值标签在均值零假设下可交换。

classic单次确定值与LLM三次均值通常有不同方差和形状；相同期望也不证明可交换。即使对比两LLM方法且repeat数都为3，编排不同仍不保证差值对称。学生化置换可在某些条件下渐近有效，但不能将独立两样本的定理直接搬到此处的分层配对 root，或改名为精确检验。Chung–Romano 的论文明确区分相同分布下精确性和更弱参数相等下的渐近有效性；该论文不直接验证此处算法。[Chung & Romano, 2013](https://arxiv.org/abs/1304.5939)

可额外预注册有界独立root的保守旁证：`d∈[-1,1]`，`w_lr=1/(K n_l)`，`W=sum_l 1/(K²n_l)`，

```
p_Hoeffding = min(1, 2*exp(-delta_hat**2/(2*W)))
```

推导：给各独立 `w_lr*d_lr` 应用Hoeffding两侧界；其范围宽度是2w，故在弱宏均值零假设下 `P(|delta_hat|>=t) <= 2 exp(-t²/(2W))`。这不要求标签交换、对称性或相等repeat，只要求有界与root独立。需用原planned权重和完整分母。该旁证同样可用global Holm6，但不与bootstrap结果二选一。它很保守：平衡ID48/OOD24在Holm6首阈值下，效应绝对值约需超过 .478/.676 才能过界，不适合把 .05 候选效应包装为“已有功效”。这是代数阈值，不是已跑的研究结果。[Hoeffding, 1963 原论文](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)，[出版方元数据与摘要](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830)

## 5. 缺失、策略失败、基础设施失效与顺序

主表实行全计划分析门槛：所有预声明root、配置、repeat和评分来源必须可核验。策略失败是评分数据，保留在原分母；没有结果、未启动、unknown、invalid score、基础设施失效都不是策略0分。不得 available-case/pairwise deletion、只算成功repeat、以2次均值替代规定3次、按低分补跑，或从同族根中挑替代root。

基础设施 invalid_attempt 最小可实施政策候选：

1. 保存原attempt及可核实的外部证据，区分服务故障与按协议属策略失败的model timeout/预算耗尽；定义不能看Y。
2. 冻结一个对称重试单元。最简单审计选择是同root完整8配置×规定repeat的18槽 block；已启动但invalid的整个block保留，若资源授权允许，只追加一次全新block，按事先规定顺序与随机流执行，明确supersedes绑定。
3. 使用按时间序“第一个完整且有效block”；不能在有效attempt中挑高分。所有有效旧低分也留档；不能混合先后block中各方法最佳episode。
4. 仍未完整则保留root、标分析未完成，不删除root或减n后照常给确认性结论。当前没有付费/正式运行授权，本条只是待G4冻结的候选政策。

若成本决定采用更窄的对称单元，可以另行提出并在test前审查，但“对关联方法对称”必须有具体表，不能只写这句话。真实runner/恢复索引尚未接好是阻断，不是SDK已经解决的事项。

完整性校验必须同时核对design、results、raw、意图/开始/终态receipt、global completion index的哈希和身份。统计输入不得把fixture角色、外层行或provider seed字段错误认作真实episode；所有正式条目都需要真实trace、grader版本与成本/调用receipt的绑定。经典确定性、分层抽样与顺序公平性由资格/manifest证据支持，合成记录不能证明。

## 6. SR 聚类与恢复/响应的最小诊断方案

### SR

先对每episode按冻结阈值求二元 `SR=I(H=1,E=1,C≥.8,S≥.6,D≥.5)`，L1不含D；若最终开发冻结改变阈值则绑定其版本。然后每root对规定R求均值，再按等级宏平均。不能先平均C/S/D再过阈值，因为该量与episode成功概率不同。

用4.2同一root向量重采样产生root-cluster percentile区间，保留嵌套repeat与事件兄弟。全0/全1时区间可经验退化，必须标记，不解释为总体成功率被精确确定。只有每root一个独立二元结果且目标与分层权重兼容，才可直接用Wilson；LLM root的0/1/3分数不是单次Bernoulli，跨等级宏平均也不能把所有repeat塞进普通Wilson。SR为描述诊断，不加入主六项显著性家庭。

### 响应、恢复与删失

必须先冻结可自动判定的endpoint：通知时刻t0；合法受影响rule/version；实际executor采用新/改job的首次trace时刻；恢复则用同权重日历下的受影响交付/观察缺口指标与持续要求。ACK、计划文本或SDK attempt意图均不是响应。无需改派由冻结公开要求与离线裁判确认，单列为N/A及计数，不填0秒与无需恢复者混合。

每个eligible episode保留 `(root, method, repeat, event_id, t0, deadline, observed_duration, event_observed, no_action_needed, failure_reason)`。未发生响应/恢复者在deadline记右删失并报告删失率；不能只平均成功者。基础设施中断是invalid/unknown，不冒充行政右删失；策略导致提前退出/事故保留“未达成”及其类别，不能以提前退出时刻作独立无信息删失从而美化曲线。

最小方案先不拟合Cox或作log-rank：公开每episode时间与删失标记、各时限响应/恢复比例、未达成比例。若计划事件时间允许，应在dev冻结共同观察窗τ且每个eligible episode都有至少τ的行政随访，报告 `U=min(T,τ)` 与 `I(T≤τ)`，未达成者U=τ；这是明确的“截至τ的截断等待”诊断，不是仅成功者平均恢复时间。root内先平均，按等级宏平均，CI仍以root同组重采样。策略失败后恢复不可能者对该截止endpoint视为未达成，并单列失败类别，不能隐去。

若无法给所有eligible episode共同τ，只报告各自deadline与删失，不把不等截止的min(T,τ_i)平均后叫同一个RMST。要用KM/RMST估计潜在恢复分布，另需预注册可支持的删失/竞争风险假设与适当聚类分析；本报告不提前批准。clean/event损失按原计划:238：需求权重不变时用Y_event−Y_clean；优先级变化时离线按同event日历重评分clean，使用Y_event−Y_clean|event，兄弟仍同root重采样。

## 7. 实现后的最低合成预检与审查要求

本节都是**待执行要求**，本审查没有运行这些新测试。只使用显式合成数据，禁止用真实test/holdout调算法。

| 预检 | 必须验证的行为 |
| --- | --- |
| 完整形状与映射 | 72root/1296槽/576外层；合成12root/216槽/96外层；8方法与1/3重复精确。缺/多root、错误level、泄漏谱系、重复episode身份、真假holdout混用应拒绝 |
| root均值与宏平均 | 独立手算不同repeat值；classic不复制；不平衡**小合成层**能区分宏平均与pool均值；正式形状仍强制16/12 |
| 配对、兄弟与顺序不变 | 同root完整向量共同抽样；人为高相关方法应保持差值相关性；复制事件兄弟不增加独立n；重排输入文件/行后给同样输出与抽样序列；调度receipt精确覆盖计划且跨两个内层 |
| 弱宏零假设 | 层效应分别非零但宏均值为0，检查每层中心化与 `delta_star−delta_hat` 等价；不能错误检验所有层效应都为0 |
| p与SE独立算例 | 固定小root数据及显式resample索引，手算ddof=1、宏SE、中心化分子、每draw SE、绝对双侧尾、>= ties与+1分母；证明没有除以观测SE替代每draw SE |
| 全零/常量/近零 | 全零p=1；非零常量观测SE0亦p=1；各层常量但层间不同不会虚造层内SE；SE_EPS边界和1%退化draw阈值按冻结比较符处理；draw退化不删/不重抽；非法NaN/inf阻断 |
| Holm6 | 用预先固定的raw p向量手算，例如 .001,.01,.03,.04,.2,1 应得 .006,.05,.12,.12,.4,1；含ties、全1、输入重排、保守p=1占满6格；missing不缩家庭 |
| 失败与恢复 | 合法评分的策略失败仍参加均值；MeasurementFailure接口边界；未启动、invalid、缺repeat、损坏/错hash、unknown阻断；第一/中间/最后测量及outer写入错误；start writer失败时measure未被调用；强杀恢复不制造已飞或0分 |
| SR与时间 | episode先阈值再平均；root-cluster不将3repeat变3root；全0/1退化标记；deadline恰达成边界；全删失、无需改派、事故提前退出、优先级同日历重评分及不同τ_i不冒充共同RMST |
| 保守旁证 | 用固定有界差值/权重手算Hoeffding，层不平衡、全0、正负反转，确保无SE参与该界 |

此外，在真实数据前做**预声明的纯合成统计压力检查**：层内对称连续、离散/安全门控零值堆积、极偏有界且均值为0、稀有大负差、classic确定性+LLM三次随机、两个非零层效应抵消、部分真零假设的六项家庭。模拟root向量须保持共享方法的相关结构与原重复不等，而不只是独立生成六条差值。固定生成参数、seed、重复次数及二项MC区间并公开所有情形；不能只展示一个校准良好的分布。此检查能发现实现和明显失准，不能证明全体分布上的小样本保证，也不是正式方法成绩。

预检不应以“合成p显著”作通过标准；应核对已知算法、完整分母、保守分支、估计误差和失败处理。若某压力分布明显失准，可在test打开前修订候选、记录理由并重新独立审查；不能先看真实test结果再选推断法。

## 8. 放行边界与决策轨迹

G4记录/统计部分仍至少阻断于：正式schedule与receipt未验收、真实episode/模型成本接线缺失、generator/split/root身份未冻结验证、主统计候选未实现/预检/独立复核、经典dev选择与失败恢复政策未冻结。G1仍在运行以及G3未放行是父任务提供的状态，本审查没有核验其结果，不能用本报告越过这些门槛。

只要承认抽样假设与名义/渐近范围，4.1–4.5不存在“必须改用label/sign置换才有效”的硬性理由；它提供可实施的近似基础p，满足Holm需要，而不是仅写一个校正名词。反过来，如需宣称有限样本精确错误率，或真实root不独立/方法随时间变化，则当前bootstrap方案有硬性范围阻断；合成通过不能解除该数学或数据前提。

以下为本独立审查的决策轨迹；时间为本报告形成时登记，指向上述可审计证据，不冒充精确工具执行时间。

```tsv
ts	phase	decision	why	evidence	result
2026-11-05T11:29:45Z	start	只读原计划和v4记录，唯一写本报告	保持独立审查与父任务写入范围	/root/statistics_design_review; 本报告第1节	完成范围约束
2026-11-05T11:29:45Z	recording	认可v4矩形分母及root均值绑定，限定为显式合成	遍历既有JSON与绑定哈希一致	qualification/adapter-preflight.v4/report.json; source-snapshot.json	1296与216内层身份及672聚合一致
2026-11-05T11:29:45Z	blocker	指出固定classic后LLM顺序与start/硬中断边界	代码不能证明计划要求的随机顺序或真实启动恢复	v4快照inspection_recording.py:62-70; SDKexperiments.py:32,46-55,109-115; production.py:28-38	阻断正式runner接线
2026-11-05T11:29:45Z	statistics	按父任务补充采用bootstrap-t主候选及globalHolm6，Hoeffding作旁证	需要可执行的弱均值基础p并接受有限样本近似边界	本报告4.3-4.6; Hall-Wilson1991pp758-759	候选未实装未验收
2026-11-05T11:29:45Z	verification	只读复核现有产物，不运行新推断或真实数据	避免把设计审查变成正式统计成绩或事后选择	本报告第2,7节	新预检仍待主Agent执行
```

Attention

reviewed by inherited-parent model（同模型家族的独立上下文；本执行环境未提供可核验的具体模型ID）

- 此审查验证旧v4合成记录和设计，不批准父任务正在修的新版，也没有检验bootstrap实现。
- 主要待注意点是root抽样与跨root独立性、12/16每层的小样本离散/偏态、退化SE保守分支与真实恢复索引。未获得跨模型家族的二次审查。
- 源文件已有多人改动，所有代码行号以第1节绑定的v4快照为准；只读证据核对未将当前脏工作区认作已提交/冻结版本。
