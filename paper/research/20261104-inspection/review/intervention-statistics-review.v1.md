# Intervention statistics：纯描述性数值实现独立复核 v1

结论：在本次冻结的纯描述性数值范围内，**未发现实质实现阻断**。完整12个固定L3 parents、每域6个、repeat→root、配对重采样、D3常量、对比方向、等域宏平均和nominal标识均与v3设计相符。本结论不认证真实parent身份、score/receipt准入、eventhelper、dispatcher/完整graph或实际结果，不授G4/main、human认证或整阶段批准。审查者 `/root/statistics_design_review`，独立上下文、同模型家族，无实现职责。

Subject `planning/formal-admission-candidate.v3/intervention-statistics-freeze.v1/binding.json` SHA256 `1fc5217da10fe745406d2aa20a098fe1eca200a53c150874f660704edbcd2be4`。两成员的实际hash/大小均与binding相同：

- `source/eval/inspection_intervention_statistics.py` SHA `b26803c5ed4ee82927703660b8b63c949963f8aa6f60503a820487905bc5b3ce`；
- `tests/test_intervention_statistics.py` SHA `278f5feba0b660b5ccf4a83b7e2f49292f30231077ab3c3b5b6c9ca9ec076395`。

根契约v3 SHA `d4b3738a659a51736d2e38bacdea2f08a6e222f992744b8480d18a274c5ee69a`相符。没有加载后来working数值实现。作者429/11项软件检查仅为提供的背景，本轮没有复跑或将其充作独立检查。

## 实际核验

源码形成精确parent×method×repeat集合，拒缺失、重复、额外root/arm/repeat、非completed状态、错误domain以及非法8分量；经典1repeat、模型3repeat，先各自root均值再按域6root平均。clean为额外216+引用primary216=432输入行；ablation为108+36=144；latency为180+36=216。引用同一primary不增加独立root或新flight；policy Y=0输入保留，非terminal不是任意补0的成绩。

每域仅生成一份索引矩阵，整行携带所有方法、arms、original/common及8个分量；batch只切既成索引矩阵。combined点值和bootstrap draw均为两域各1/2，而非先平均两个CI端点。clean original只列各方法，差值只在common_calendar做event-minus-clean；ablation为arm-minus-primary；latency四个非零delay减本轨fresh-zero，并另列fresh-zero-minus-primary-stability。未引入p、Holm新家庭、仅成功样本筛选或独立repeat样本。完整原始bins本来不进入该scalar模块，不能由该模块替代上游读取/复核。

独立检查使用实际NumPy1.26.4，**复制冻结源码到自己的/tmp**，启用dont_write_bytecode；只用AST读取staged两个依赖的固定CONDITION_REPEATS/SEEDS并比对协议，再以仅含常量的stub导入，避免执行生产或其他候选。实际两依赖source哈希也记录在binding。freeze本身仅列两成员，并未冻结完整import closure；本轮不据纯数值结果认证安装后的来源闭包。该模块只检查已admit数值的形状/类型/范围，不承担score业务一致性、真实condition/lineage与资源来源认证。

独立合成fixture使用非平行的parent/方法/重复/分量变化、H/E零门及两域不同值，避免仅靠全部root斜率抵消的例子。测试实例将DRAWS临时设为37（原冻结文件不变），三轨的全部root均值、8分量、两域/combined点值、所有作用域、完整contrast及nominal CI，与另行按root计数频率矩阵乘法、Python排序手算linear分位数的结果一致；最大root均值误差1.1102230246251565e-16，整体比较容差2e-14。逐轨反转输入行顺序，输出完全相同；输入内容未被修改。

另独立生成clean/ablation/latency×ID/OOD六份**完整99999×6 bootstrap索引矩阵**，按照固定domain seeds、salt101/102/103、PCG64/SeedSequence、int64、endpoint=False；其little-endian字节SHA、shape与范围均存binding。37-draw实际模块输出的index SHA与独立生成完全一致，冻结源码DRAWS=99999及所有常量已核。没有独立重跑完整99999-draw的全部输出；完整索引计算不是正式population几何/seed抽样，亦不是正式统计结果。

15个额外边界反例全部拒绝：缺行、重复、额外repeat、False或float repeat、错domain、范围外parent、错arm、failed状态、NaN、越界分量、NumPy bool、浮点二元SR、缺D、未投影bins。另全零合法分量正控保留12parent并输出零宽nominal区间，不把零宽解释成有限样本覆盖保证。上述均为明确合成软件数据；没有实际方法/飞行成绩或经验功效/覆盖率认证。

## 边界与完成

源码与输出清楚标为prespecified_secondary_descriptive、p_values=not_computed、unchanged_main_global6；每域只有6parent，区间仅nominal/descriptive，计算流分离不证明真实root独立。不存在从本次合成通过推导G4、正式数据准入或显著性保证的授权。实际receipt/graph/event见证接线尚不在本模块范围，未来仍须独立验收。

44项独立检查（包括15非法输入组及全零正控）通过；数值脚本实际UTC `2026-11-06T18:27:32.525694Z`–`18:27:32.676313Z`，内部0.1505秒。精确命令、脚本、环境、所有检查/反例及SHA在binding。没有pytest/全候选套件、生产import、formal population抽样、model/physics/native/effects；无遗留子进程。报告/绑定归档后清理自己的/tmp代码副本，停止全部CPU检查及R写入。
