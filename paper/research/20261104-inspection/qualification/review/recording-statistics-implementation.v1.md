# G4 记录与统计实现独立审查 v1

状态：**旧 v4 的调度问题在 v5 合成 callback 层得到修复；新统计模块的算术符合候选规格，但 bootstrap 确认性主检验在当前有界独立 root 弱均值假设下被压力试验实质否证。G4 仍阻断。**

最重要结果：六个零假设都为真的、合法有界 iid root 稀有损失场景中，当前 `bootstrap-t + global Holm6` 在 200 组独立合成数据上误报 **189/200=94.5%**，外层 Monte Carlo Wilson95%区间 **[90.42%,96.90%]**。这不是零标准误退化，也不是把 LLM repeats 当独立 root：该场景全部 1,200 项状态都是 `nominal_asymptotic`，没有退化 draw。两个事先固定编号的数据集增至99,999draws后仍误报。

建议在没有任何正式方法/test数据前，预注册 **Hoeffding 有限界基础 p + global Holm6** 作为确认性分析；保留原计划的 root bootstrap 区间和 bootstrap p 作为明确标记的名义/描述诊断，同时补 pointwise 有限界95%区间和 Bonferroni-global6 同时95%区间，披露低功效。本报告只提出并独立核对该替代，不修改生产源码，也不批准尚未实现的修订。

## 1. 范围、来源与冻结

审查日期：2026-11-05。本报告复用上一次独立设计审查上下文，与主 Agent 分离，继承同一模型家族，不代表跨模型家族验证。适用规则为已读 Codex pstack 适配和 `show-me-your-work`。本次只写本报告及 `qualification/review/statistics-implementation-v1/` 内的自有脚本/证据；未修改 production、旧报告、旧预检或正式数据。没有物理仿真、模型/API、真实 test/holdout 调用。

本次审查绑定以下字节，主 Agent 收到冻结通知后才另行修订：

| 文件 | SHA256 |
| --- | --- |
| `eval/inspection_statistics.py` | `a42bd777f783b5f9fc6702944117663a0e564566fa77f1a3ebf1e534c43c70a8` |
| `eval/inspection_recording.py` | `9613a8e800d470110dbf86890837ba853c3f50834ae29f0722994927d27cb352` |
| `eval/preflight_inspection_statistics.py` | `744dedf043e9e6ca980ef14ce522d3e9ce49f2a8f97fce0d443e26565de733a8` |
| `eval/preflight_inspection_recording.py` | `ee6e2481d0b75d907e12100fed69e6636422610067ae4ea1a5bdf3530d4002d1` |
| `tests/test_inspection_statistics.py` | `9ca80bc6428155a3434e2ecfa26791d79f1f05e58c1593e9b499a2bbb86c67be` |

源快照、SHA和所有独立结果存于 `review/statistics-implementation-v1/results.v1/`。两次审查脚本执行前后均校验上述来源未变。本报告中的行号以这些快照为准，不能用于批准以后同名文件的新字节。

独立压力与标量复算使用系统 Python3.14.7 / NumPy2.5.2，单线程BLAS，固定bootstrap seed。原v5预检运行的NumPy为1.26.4；在独立runtime中重现的schedule slots完全一致，仅版本元数据字段不同。另用项目 `.venv/bin/python` 的现成环境执行统计测试，4项通过。没有为此安装依赖。

## 2. 已完成的实现核验

### 2.1 v5 实际 callback 调度与记录

对既有 `adapter-preflight.v5/{full-shape,holdout-shape}` 全量读取并核对：

| 既有显式合成形状 | episode槽/receipt | root块 | 外层行 |
| --- | ---: | ---: | ---: |
| main-shape | 1,296 | 72 | 576 |
| holdout-shape | 216 | 12 | 96 |

每个18槽块恰有一个root、classic3槽和LLM15槽；schedule可按冻结seed重现，并对metadata输入倒序不变。每条order receipt、start与raw的身份、order_index、schedule SHA一致，start/完成时刻按顺序单调对应；global completion index绑定三份results；672条外层行均绑定正确1/3个内层SHA、results SHA及算术均值。

源码 `record_matrix:105-118` 先打开三个SDK recorder，再遍历外部冻结schedule路由到classic/LLM attempt，不再由SDK canonical grid决定实际callback顺序。第一主root的方法顺序已交错：`B4-M2,B2,B3-M3,B3-M1,B1,…`；不是先全部classic后全部LLM。`episode_schedule:36-47` 使用层内root排序后随机化、归一化jitter排列root块，再随机每块18槽，满足本次合成层面的预冻结分层调度要求。

这验证的是**合成测量callback层**。`order-receipts`明确是callback返回，不能替代真实飞行/provider receipt。start仍只是尝试意图；写一次、哈希与completion index没有实现fsync断电恢复、unknown消解或exactly-once真实执行。全局完成索引缺失仍须阻止正式统计。真实root谱系、split、费用和classic dev选择仍未由此验证。

### 2.2 分层均值、SE、向量重采样与中心化

`inspection_statistics.py:55-62` 的估计量与SE为：

```
delta = (1/K) sum_l mean_r d_lr
SE² = sum_l sample_variance(d_l, ddof=1) / (K² n_l)
```

它用root配对差的方差，保留方法协方差；没有按repeat数给LLM三倍权重。独立小算例 `d1=[0,.2]`、`d2=[.4,.6,.8,.2]` 得宏差 .3、SE=`sqrt(1/150)`，不同于pool均值 .366667，与实现一致。

`analyze_roots:119-128` 每次以同一索引抽取完整8列root向量，再计算方法差和sample variance；同方法彼此独立抽样的错误未出现。完全相同的两方法在root间都可变时，独立检查得到差0、SE0、区间[0,0]、p=1，支持保留配对结构。当前RootTable只有主root均值，没有真实事件/兄弟trace接口；“同root兄弟同组”只能作为未来输入接线要求，不能声称已实现SR/事件适配。

`studentized_reference:82` 使用 `(delta_star−delta_hat)/SE_star`，与每层去自身均值后重采样的宏误差在代数上等价；不是未中心化的delta_star，也不是除以固定观测SE。不同层非零效应的宏零假设可以保持该解释。对97个固定draws，审查脚本另以root差、Python `fsum`/`statistics.variance`、手写分位数与Holm逐项标量复算，六项效应/SE/p/区间与实现一致，标量最大绝对误差 `3.47e-18`，尾计数相同。

### 2.3 seed、batch、输入顺序与退化规则

每层独立 `PCG64(SeedSequence([domain_seed,level]))` 避免batch切换改变不同层的随机流交错；本次独立比较batch13/31与metadata/方法字典倒序，完整输出相同。正式冻结应记录此SeedSequence规则，不仅记录“PCG64”；它与旧设计报告中未细分level stream的简写有差异，此差异在真实数据前明确即可。

独立边界检查确认：

- 观测SE为0、1e-13、1e-12时，p=1、T=null；不因常量正效应给p=0。
- 100draws中1个退化draw（恰1%）仍计尾部，尾数1、p=2/101；2个（2%）触发不稳定并置p=1。退化draw没有丢掉或补抽。
- 双侧使用绝对值和`>=`保留ties。独立三draw算例尾数2，+1后p=3/4。
- 六项Holm与独立排序/累积max一致；保守p=1仍占家庭。生产`holm`是通用长度函数，但`analyze_roots`构造固定两域×三对比后才调用。

`RootTable.parse`默认要求完整48ID+24OOD、合法8列与唯一lineage，数值角色清楚；其公开构造和`require_main_shape=False`是可绕过正式shape的接口边界。真正正式adapter应唯一走默认严格parse并另验证trace/lineage来源，而不能把数值对象类型当证据认证。

## 3. 独立有界 iid null 压力试验

四种分布在运行前已写入独立脚本，各200个独立数据集，每个数据集按原48ID/24OOD形状、classic1/LLM3协议生成root均值，调用999draws的被审查实现。每个数据集固定独立数据seed `[20261105,8101,scenario_index,replication]`。bootstrap seed保持生产常量，因此外层Wilson区间估计的是**该固定seed、999draw实现**在独立数据下的误报/覆盖率；不是对所有bootstrap随机流平均的保证。

每个场景六项宏均值差的真值都严格为0，所有Y∈[0,1]，root之间独立；允许root内多方法相关。原始每项输出、seed、尾计数和退化计数完整留在 `stress-replicates.jsonl`，不能仅凭下表替代审计。

### 3.1 冻结的生成分布

1. **symmetric_uniform_null**：每root共享base~Uniform(.35,.65)，classic=base；每LLM方法3次独立Uniform(−.2,.2)噪声后求均值并加base。所有期望相同，差对称，方法向量有共享root成分。
2. **skew_episode_null**：classic、B3为.95；两个B4各为3次独立Bernoulli(.95)的均值。均值均.95，差为强偏态离散量，经典确定，重复数不等。
3. **rare_root_loss_null**：classic、B3为.95；对每个B4和每root独立，以概率.02令3次成绩皆为0；否则成绩为 `.95 + a + U_j`，`a=.95×.02/.98=.019387755102…`，U_j独立Uniform(−.002,.002)，再对3次求均值。于是 `E[Y_B4]=.02×0+.98×(.95+a)=.95`；每root d的分布为2%大损失−.95、98%约+.01939小增益。root内共享稀有条件是环境root特征，不是把三个repeat当三个独立场景；常见分支微小连续噪声使“没有看见稀有损失”的样本SE仍显著大于1e-12。
4. **shared_zero_gate_null**：对称场景再乘各方法共享的root级Bernoulli(.65)门控，模拟共同安全/退出零值堆积。所有方法期望仍相同，保留整向量相关。

### 3.2 global Holm6 的全家庭误报

| 合成null场景 | bootstrap-t + Holm6误报 | 外层Wilson95% MC区间 | Hoeffding p + Holm6误报 |
| --- | ---: | ---: | ---: |
| 对称连续 | 6/200 = 3.0% | [1.38%,6.39%] | 0/200 |
| episode强偏态离散 | 0/200 = 0% | [0%,1.88%] | 0/200 |
| root稀有大损失 | **189/200 = 94.5%** | **[90.42%,96.90%]** | 0/200 |
| shared zero gating | 4/200 = 2.0% | [0.78%,5.03%] | 0/200 |

所有零计数的上界约1.88%，不代表真实误报率是0。前三个看起来保守/接近5%的数字不能证明一般校准；rare反例已在现有假设内部推翻该规模下的通用确认性适用性。Hoeffding四场景0/200也不是它有限有效性的证明，证明来自第5节的独立有界不等式。

偏态episode场景中，OOD的两个B4比较分别有128/200和127/200项被保护为`bootstrap_reference_unstable`，另有7/6项观测退化；其低误报有保守分支贡献，不能当高功效证据。rare场景却全部正常状态，零SE保护不覆盖“样本根本未见稀有损失”这个问题。

### 3.3 percentile nominal95% CI 的真零覆盖率

每格分母均为200，三个比较依次为M1编排、M2编排、对classic；完整逐项Wilson区间见独立JSON。

| 场景 | ID-M1 | ID-M2 | ID-classic | OOD-M1 | OOD-M2 | OOD-classic |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 对称连续 | 93.5% | 93.5% | 92.5% | 89.0% | 92.5% | 91.5% |
| episode强偏态 | 92.5% | 91.5% | 92.5% | 88.0% | 85.5% | 88.0% |
| root稀有损失 | **65.0%** | **59.5%** | **65.0%** | **38.5%** | **33.0%** | **38.5%** |
| shared zero gate | 93.0% | 92.5% | 92.5% | 91.5% | 91.5% | 90.0% |

例如rare的ID-M1为130/200，MC95%区间[58.16%,71.27%]；OOD-M1为77/200，[32.03%,45.40%]；OOD-M2为66/200，[26.86%,39.78%]。因此保留bootstrap95区间时应明确nominal/descriptive，不能借其排除0作可靠的当前样本确认性主张。

200rep的外层精度有限：若真实率约5%，标准误约1.54个百分点；估计10/200时Wilson95%约[2.74%,8.96%]。单数据集999draws的尾率约.05时Monte Carlo标准误约.0069，约.00833时约.0029，最小p=.001。这些误差不足以解释94.5%的误报。不能将800个不同场景合并成一个校准数，也不能把6个相关hypothesis×200当1,200个独立误报试验。

## 4. 稀有损失反例的机制与高draw复核

在rare分布中，一个B4的48个ID roots全未见损失的概率为 `.98^48=.3791854`；24个OOD roots为 `.98^24=.6157803`。两个B4×两个域这四个独立子样本至少一个未见损失的概率为

```
1 − (1−.98^48)^2 (1−.98^24)^2 = .9431038。
```

它与观测到的全家庭误报.945接近。这个概率是**未见损失事件**的精确计算，不被本报告宣称与bootstrap拒绝事件严格等价。未见损失时，经验分布只有小增益及微小噪声；重抽样没有机会生成未见的−.95，SE却非零，中心化bootstrap-reference非常窄，所以两侧p很小。Holm只能校正有效p的多重性，不能修复这种经验分布遗漏。

对固定rep0和rep1，另外用99,999draws复核；数据编号预先按0/1选取，没有挑最高误报样本。rep0示例：

- `OOD.architecture-M1`：effect=.01956243，SE=.00010362，退化draw=0；bootstrap p=1e-5、global Holm=6e-5，nominal95%区间[.01936859,.01975894]，真值0被漏掉。
- `ID.architecture-M2`：effect=.01928196，SE=.00009022，退化draw=0；p=1e-5、Holm=6e-5，区间[.01910901,.01945139]。
- rep1同样出现4项Holm拒绝；并非提高B就恢复校准。

此结果不否定固定非退化分布下随每层root数增长的bootstrap一阶渐近理论；它否定“12/16每层、有界iid、弱均值为0”已足够支持当前确认性主检验的推断。只因dev未见稀有损失不能证明它不存在，不能靠把SE_EPS或1%阈值调高一点消除此机制。本次更换主检验在真实test之前、根据公开合成反例完成，属于预注册设计修订；不能在正式结果后按显著性选择。

## 5. 独立核验的有限界确认性替代

### 5.1 假设与基础 p 推导

域g中有K等级，每层n_l个独立root。先按冻结规则求1/3重复均值，再求同root方法差 `d_lr∈[-1,1]`。宏均值估计为

```
w_lr = 1/(K n_l)
delta_hat = sum_lr w_lr d_lr
mu = E[delta_hat]
W = sum_lr w_lr² = sum_l 1/(K² n_l)
```

每个加权独立项范围为[−w,+w]，宽度2w。Hoeffding两侧界给出

```
P(|delta_hat−mu| ≥ t) ≤ min(1, 2 exp(−t²/(2W)))。
```

故对弱零假设 `H0: mu=0`，可采用保守super-uniform基础p

```
p_bound = min(1, 2 exp(−delta_hat²/(2W)))。
```

推导只需要各root差独立、给定界与固定权重，不要求正态、对称、方法标签可交换、classic/LLM方差相同或repeat相同。root内不同方法/事件/重复可以相关；不同假设之间也可相关。当前源码`:131,140`的旁证公式与此独立推导一致，并在固定高draw数据上与审查端直接算式匹配。[Hoeffding, 1963 原论文](https://www.cs.rpi.edu/academics/courses/spring06/random/hoefding.pdf)，[出版方摘要与DOI](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500830)

把六项p_bound送入同一global Holm6，在这些基础p有效前提下，可得**有限样本FWER≤.05的保守控制**，不是“错误率恰为5%的精确检验”。Holm不要求六项相互独立，但要求每个真零假设的基础p有效；这正是将bootstrap p降为诊断的理由。[R stats `p.adjust` 官方说明](https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html)

前提仍有实质内容：真实generator来源需支持root独立；配置/模型/评分与权重需先冻结；不能跨root共享学习状态或遭受未建模共同冲击；不能可选停机、删低分root或挑retry；完整分母、invalid/unknown处理和经典dev独立选择应满足先前契约。IID并不是Hoeffding不等式必需条件，但层内冻结generator IID为本研究总体解释提供目标。若真实数据缺失或root不独立，换检验不能解除阻断。

### 5.2 pointwise95%和global6同时95%区间

对每项比较，有限界pointwise95%区间为

```
r_point = sqrt(2 W log(2/.05))
CI_point = [max(−1,delta_hat−r_point), min(1,delta_hat+r_point)]。
```

对六个域特定比较，使用Bonferroni给每项α/6：

```
r_sim = sqrt(2 W log(12/.05))
CI_sim = [max(−1,delta_hat−r_sim), min(1,delta_hat+r_sim)]。
```

union bound使六项区间同时覆盖概率至少.95，不要求假设间独立。它们应命名为`Bonferroni global6 simultaneous95`；不能称为Holm反演区间。主p用Holm比Bonferroni可能更有力，区间可与Holm判定不完全对应，需如实标注，不能暗示pointwise区间已控制六项。

| 完整域形状 | W | pointwise95半宽 | global6同时95半宽 |
| --- | ---: | ---: | ---: |
| ID，3×16 root | 1/48 | .39205014 | .47787023 |
| OOD，2×12 root | 1/24 | .55444262 | .67581056 |

界很宽，反映只假设有界独立时的小样本信息限制。Holm6首阈值下效应需约>.478(ID)/.676(OOD)才可能拒绝；这不足以承诺检出原计划的.05候选效应。不得因此在见到正式结果后扩N追显著，也不能将未拒绝解释为等效。不同对比若经开发合法证明更窄的**先验确定**范围可另提规格，但不能从test观测min/max收紧界。

确认性界不使用SE_EPS、不依赖bootstrap RNG，不继承“SE0时一律p=1”保护；常量非零差值仍按有界公式分析。原repeat数、等级权重和n不变，不把1,296episodes当独立分母。父任务拟新增的主字段必须指向bound p及其Holm，避免沿用`reject_nominal_alpha05`的bootstrap布尔值作为主判定。

### 5.3 本次对界的数值检查与覆盖边界

审查脚本独立计算上述半宽与已有旁证p；另枚举两层n=2/3的32种独立±1结果，第一层P(+1)=.75、第二层=.25，层均值+.5/−.5宏平均为0。对α=.01/.05/.1/.2/.5逐项精确加总，检验错误率均不超α，相应pointwise覆盖不低于1−α；此小枚举用于检查权重/指数/方向，不充当普遍证明。

未来修订还应核验：全零和常量差、接近边界±1、clip后的区间、非平衡小层权重、global6并列p和家庭固定、源码/hash变化、完整形状与缺失拒绝。四压力场景的Hoeffding-Holm均0/200只说明这些诊断中保守，没有测出主效应功效，也没有验证真实root来源。

## 6. 运行证据、限制与后续门槛

实际执行：

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 .../review_statistics.py --replications 200 --draws 999
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 .../review_bounded.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_inspection_statistics.py
```

结果：两份自有检查脚本通过；4项既有统计测试在项目环境通过（0.08s）。第一次在系统Python尝试pytest因未安装pytest退出，之后使用父任务提供的现成`.venv`完成，没有安装依赖；第一次schedule复现将NumPy版本元数据也要求相等而停止，纠正为分别记录版本、核对实际slots后继续，未改生产实现。初次停止只产生源快照，未计入200rep结果。

关键证据：`results.v1/report.json`（源/runtime/完整MC汇总）、`stress-replicates.jsonl`（800数据集的六项输出）、`independent-numeric-checks.json`（手例/97draw标量复算/边界）、`existing-v5-audit.json`（完整schedule/raw/receipt检查）、`bounded-and-high-draw-checks.json`（界的独立算式/32点枚举/两个99,999draw样本）。所有均为显式合成或只读核查，不能进入真实成绩表。

本报告接受的范围为：v5 callback调度修复及当前数值公式实现基本正确；拒绝将当前bootstrap-t主检验作为当前弱假设/样本量下可靠的确认性依据。有限界修订必须另存版本、更新预注册/测试与输出角色，再独立复核。本报告不提前批准该改版。

SR聚类、事件兄弟/同权重日历、恢复删失适配仍未实现；真实split/trace/费用/provider、classic选择、硬中断resume仍未完成。本次数值工作不能解除这些G4门槛，也不评价正在进行的物理qualification。

决策轨迹：

| 决策 | 证据 | 结果 |
| --- | --- | --- |
| 将v5原始callback修复与真实飞行恢复分开判断 | existing-v5-audit.json；生产recording快照 | callback修复支持；真实接线保留阻断 |
| 先独立复算算法，再压力检查适用性 | independent-numeric-checks.json；stress脚本固定四分布 | 算术符合规格不等于检验有效 |
| 保留rare反例全部200rep，不用零SE解释 | stress-replicates.jsonl；189/200及零退化计数 | 确认性主bootstrap阻断，已及时通知主Agent |
| 固定rep0/1增加draw核对数值误差解释 | bounded-and-high-draw-checks.json | 99,999draws仍误报，非999draw精度造成 |
| 提出有限界主检验及两类区间，不改生产 | 本报告第5节；独立界检查 | 待预注册修订与复验 |

Attention

reviewed by inherited-parent model（同模型家族独立上下文，具体模型ID未由执行环境提供）

- **必须保留189/200反例及有限样本边界；不能以实现或普通null测试通过批准当前主bootstrap确认性推断。**
- 审查NumPy2.5.2与原1.26.4不同；少数独立复算与项目测试均通过，但不声称跨版本所有随机算法永远一致。
- 有限界的有效性来自明确的独立有界前提；本报告没有认证真实root生成、正式数据或未来修订，也没有跨模型家族的二次审查。
