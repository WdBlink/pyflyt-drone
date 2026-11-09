# 八配置、不等重复的可执行记录适配

状态：提供SDK的合成预检已执行，尚未作科学验收或正式冻结。主分析统计、真实episode接线、生成器与模型等仍未完成。本文件补充v2设计，保留原记录；不表示G4或论文已放行。

实现为`eval/inspection_recording.py`。直接使用autoresearch提供的`ExperimentRecorder`、`write_once`与其设计校验，没有复制记录算法。内层按重复次数分为两个独立矩形：B0/B1/B2为3配置×1实际episode重复，B3-M1/M2/M3与B4-M1/M2为5配置×3实际重复。三份设计先全部持久化、三个SDK envelope在任何测量前全部打开；其中两份记录真实episode，第三份记录root内均值。每个内层测量调用前另用提供的write_once保存不可变start事件，status=running表示该时刻已开始，当前终态由SDK完成/失败record决定，不能单读start事件推断仍在飞行。出现操作错误时各已打开SDK保留失败及未启动的完整格，外层不产生可分析均值。真实策略失败应按原冻结评分得到合法Y（可能为0），不能抛成基础设施错误后删除。

这是对v2“自定义非矩形内层日记”的最小化替代。两份内层仍由提供SDK拥有原始尝试，外层保持原独立root单位，不复制经典成绩为三次。SDK的内层seed字段只是实际重复序号，provider实际是否接收随机seed须在evidence单独记录；外层唯一pass0是一个聚合，绝非一次新飞行或provider随机种子。每条外层均值绑定恰好1或3个已完成内层原始record SHA及对应results SHA。

`qualification/adapter-preflight.v4/report.json`是已执行的合成证据，实际源码与提供SDK的15份Python闭包另有source-snapshot及SHA，启动结束源码一致。其值全为显式合成fixture，不代表策略效果、真实飞行或费用。

| 输入形状 | 内层classic episode槽 | 内层LLM episode槽 | 实际episode候选合计 | 外层均值行 |
| --- | ---: | ---: | ---: | ---: |
| 主评测72 root，ID各等级16、OOD L2/L3各12 | 216 | 1080 | 1296 | 576 |
| holdout12 root，L2/L3各6 | 36 | 180 | 216 | 96 |

这两种合成形状共1512 episode槽，不是新增1512次已飞样本。外层672行不能作为飞行数。合成fixture指定B1只是使第三对比名称可校验，不是已在dev选出B_classic*；正式选择仍只能由冻结dev规则决定。

独立手算的合成结果：三个classic均只有repeat0值0，五个LLM为repeat0/1/2值0/.25/.5，root均值分别0/.25。所有1296主槽与576主均值均与此逐项一致。故意基础设施异常后，双root夹具保留classic6、LLM30、outer16的完整分母，已失败与未启动均保留且禁止统计。非有限、超[0,1]与布尔score也分别拒绝并保留原外层分母。

SDK每层仅设interval=none、outcome=inconclusive；其描述性汇总不替代本研究root成对推断。本次尚未实现/验收：ID/OOD与等级宏平均、root成对bootstrap95%区间、三个预声明对比及Holm、兄弟事件trace同root重采样/同权重日历、SR root聚类、恢复右删失、实际episode及模型/费用receipt绑定。这些仍阻断G4与plan阶段科学接受。代码/预检还需要独立语义审查；本文件没有将自检称为独立审查。
