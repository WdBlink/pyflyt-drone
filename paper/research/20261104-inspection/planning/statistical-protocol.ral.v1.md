# 主评测数值推断规格：正式数据前修订

状态：候选实现与合成预检已执行，等待独立修订复核；不授予G4，不表示真实root来源已合格。没有运行正式方法、主test或holdout。原19项交付、八配置、72主root与12holdout计划不变。

## 单位、估计量与固定家庭

每个独立root先分别计算各方法实际重复的Y均值：B0/B1/B2各1，B3-M1/M2/M3和B4-M1/M2各3；重复、飞机、时间格和事件兄弟都不是新增root。ID有L1/L2/L3各16root，OOD有L2/L3各12root。两域分开，各域等级等权宏平均，不按域内root池另作等级加权。

三个预声明方法对比为B4-M1−B3-M1、B4-M2−B3-M2、B4-M1−B_classic*。B_classic*只能在独立dev按冻结规则从B1/B2选择；当前合成预检中的B1是占位，不是选择结果。每个对比在ID/OOD分别有一个假设，构成预先固定global6家庭，双侧α=.05；不得按方向、等级或结果事后缩小家庭。

域内K等级，每层n_l独立root，root差值d_lr∈[-1,1]，固定权w_lr=1/(K n_l)。估计量delta=sum(w_lr d_lr)，W=sum(w_lr²)。公开每root差值、宏均值、root配对SE及未加权差值中位数；不得把中位数或pool均值冒充主估计量。

## 确认性主检验及区间

主基础p固定为Hoeffding保守界：min(1,2exp(−delta²/(2W)))，六项共同做Holm调整。有限样本FWER≤.05以真实root差独立、固定权重、有界完整分母、无可选停止/挑选/遗漏为前提；root内不同方法、重复和事件可相关，六个假设之间无需独立。数值代码不认证这些真实来源条件，仍须generator、split、trace、运行日历及漂移/共享学习状态的证据。

同时报告两类裁剪到[-1,1]的有限界区间：pointwise95半宽sqrt(2W log40)；Bonferroni-global6同时95半宽sqrt(2W log240)。后者不是Holm反演，可能与Holm结论不完全对应，名称和覆盖角色须明确。ID半宽分别约.39205/.47787，OOD约.55444/.67581。该规格很保守，不承诺检出原计划的.05候选效应；未拒绝不能解释为等效，不能在看过正式结果后扩N追显著。

## 保留原计划的root bootstrap，但限定其角色

各域每等级内按root重采样同一完整方法向量，99999次、NumPy PCG64、SeedSequence([domain_seed,level])，ID seed2026110401、OOD seed2026110402；quantile=linear。先求root内重复均值，再重采样root。报告pointwise nominal95% percentile区间；它不具当前弱假设/小样本下的有限样本覆盖保证，不用排除0作为确认性结论。中心化studentized bootstrap p及其globalHolm6仅留在`bootstrap_reference_diagnostic`，不得替代主p字段。

零/近零SE、退化draw的原保守规则继续仅作用于该诊断：SE_EPS=1e−12；零SE p=1；每个零draw作为尾部保留，退化比例>.01则诊断p=1。主有界p不使用SE或这些保护；常量非零差仍按有界公式分析。

## 修订的实际证据及版本边界

独立`qualification/review/recording-statistics-implementation.v1.md`确认旧实现算术，但否证其确认性适用性：合法有界iid、弱均值零假设的2%稀有root大损失分布，在200组×999draws中全家庭误报189/200，MC95区间[90.42%,96.90%]；固定rep0/1改99999draw仍误报，且没有零SE退化。原代码snapshot、800组四分布压力结果、失败p和报告全部保留。该公开合成反例发生在正式数据之前，不是根据方法胜负选择检验。

修订实现`eval/inspection_statistics.py`将主p、主Holm及主拒绝字段明确指向有界检验；原bootstrap放入具名诊断。7项单测覆盖手算权重/SE、Holm、诊断退化、输入/batch不变、有限界半宽、32种弱null精确枚举及窄bootstrap不误用为主结论。`qualification/statistics-preflight.v2`为72根显式合成预检，检查完整形状和手算效应；不是正式数据或总体功效证据。修订后实现还须独立复核，不能引用旧报告作为新版批准。

## 尚未冻结的相邻事项

真实输入loader、独立lineage/split/生成器、费用与实际trace、硬中断unknown恢复、有限对称infra重试和dev经典选择尚未完成。SR按root聚类；事件兄弟用同root重采样、统一需求权重日历；恢复时间保留右删失与无动作NA。这些适配仍需实现/核验，不能用本主Y数值模块替代。holdout在主结果后只作一次独立确认，其开封/假设/α规则须开封前另冻结；当前12根合成形状不会暴露真实holdout。

该文件是研究适配规格，不覆盖或重写提供的ExperimentRecorder。当前SDK全分母/调度预检与数学验收分别保存，均不能把缺失的真实实验变成完成。
