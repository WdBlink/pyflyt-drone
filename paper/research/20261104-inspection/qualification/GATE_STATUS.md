# 当前资格状态

本页记录开发工程证据，不表示正式方法结果、实机认证、领域人工审阅或论文完成。原19项义务、八配置和正式分母不变。

最新继续点：`g4-progress.v163.md`。G1–G4 仿真资格通过。正式主回调 152/1296，完整场景 8/72；主阶段继续逐槽核验。

| 门 | 当前状态 | 实际证据与边界 |
| --- | --- | --- |
| G1 | 开发控制资格通过 | 基础五族100份条件复用；v18新observe/transfer各20全部合格。v19唯一退休采样guard在原events=False域不可达，独立审查支持复用同40，不计新重复。 |
| G2 | 开发协议/fixture与实际参考集成通过 | 十项在声明范围有证据；新退休v3为0→500s/120000步。实际1×参考另覆盖固定分母、180s调权端点、补传、failed/cancel后原route交付、退休停采/旧coverage。未覆盖的长handoff等情形不扩称已运行。 |
| G3 | 开发期600s参考资格通过 | 新B0实际1×、144000步，max lag=.031487833s<.1；Y=.779071、静态recall1、H/E1。14源/61依赖不变、22字段/116RPC精确重放；新checker三项P2经原反例修复复验关闭。SR仍0。 |
| G4 | 仿真资格通过 | v4完整108/108、970测试、实际clean/20秒和无损容量实测均闭合；150源/61依赖/21证据精确 subject 获独立接受并原生登记。实机和领域人工审查仍待。 |

G1依据：`g1-full.v4/manifest.json`、`executor-full.v4/manifest.json`、`review/g1-v18-full-review.v1.md`、`review/g1-v19-composition-review.v1.md`。新包线是17m/s、XY同刻欧氏10m、稳定Z±2m、全模式名义连续35m/540m；硬15m间距/60–100mAGL/15–25m/s/45°约束不变。旧v16/v17按原6m失败，不重分类。

20个observe资格slot通过指固定完整180s/60格D≥.8及物理/退出等合同条件，不表示20个本地observe job均成功；它们仍因本地缺格保留failed终态。控制roots不是独立科学场景；受控nexttick不是真实规划/模型时延。完整v18源已冻结于`source/finite-arcs-v18/`。

G2缺陷与新证据：`review/g2-v18-retirement-review.v1.md`、`protocol/retirement.v2.json`、`protocol/retirement.v3.json`。新v3不能回改v2的124覆盖包/101新观测缺陷；两次全部5000组物理位置及240Hz误差bytes相同。

G3依据：`reference/executor-realtime.v2/{run,world}.json`、`.replay.v2.json`与`.qualification.v3.json`（后两者位于reference下、同前缀），以及`review/g3-v19-control-sensor-review.v1.md`、`review/g3-v19-mission-replay-review.v1.md`、`review/g3-checker-repair-review.v2.md`。独立重算C=.937213/S=1/D=.4/Y=.779071/SR=0全部吻合；normal inspect877次，位置向量RMSE1.430835m、类别868/877。raw miss与关联抑制/背景帧分母分别记录，不用detected-only倒推漏检。

本次max/p95来自原worker逐control tick时钟测量、保存最大见证和1Hz进度旁证；完整72k lag数组未保存，不能说已离线精确重计全数组max/p95。里程由逐240Hz原hook累积并绑定实际剩余量；10Hz位置不用于冒造精确里程。误差trace及22组execution replay不等于保存了完整240Hz位姿/速度状态。单次dev参考不证明总体可靠性；`human_review_pending`保持。

G4推断：`planning/statistical-protocol.ral.v1.md`（路径相对研究根）及`review/statistics-bounded*-review.v1.md`。原bootstrap确认性方案在稀有损失合成null下误报189/200，完整保留；新主p为有界独立root Hoeffding+globalHolm6，bootstrap仅名义/诊断，低功效和独立性条件必须披露。合成形状1296+216槽不是1512条已飞数据。

B1/B2的旧重复handoff与类别tie永久等待已修复并经独立复验；六root开发比较现已完成。固定B1/B2选择器选B1，B0仍有更高宏平均Y；这不是所有经典方法的最优声明。主test/holdout未开封，没有付费API或新增算力采购。

最新开发进度见 `g4-progress.v7.md`；v1–v6作为历史材料保持原字节。本页G1–G3范围和v19原证据不回改；新场景与后续分析的修复版本须各自绑定。

当前继续点：`qualification/g4-progress.v8.md`；v1开发pilot已结束并保留完整分母，下一步补齐公开合同与同级clean干预后冻结v2，G4仍needs_work。

最新继续点：`qualification/g4-progress.v9.md`；同级clean参考对/双执行重放已完成，完整v2八配置pilot待checkpoint后启动，G4仍needs_work。

最新继续点：`qualification/g4-progress.v10.md`；v2已完成2/108有效callback，clock失败与唯一retry均保留，当前集间checkpoint/诊断，G4仍needs_work。

最新继续点：`qualification/g4-progress.v11.md`；v2已完成8/108有效callback，fresh order8 gate，G4仍needs_work。原owner存储修复与formal入口均为未安装候选。

最新继续点：`qualification/g4-progress.v12.md`；v2已完成9/108有效callback，fresh order9 gate，源候选表示修复获独立批准但尚未迁移；G4仍needs_work。

最新继续点：`qualification/g4-progress.v13.md`；同owner表示迁移已完成并验证旧记录精确保留，v2 pilot9/108，G4仍needs_work。当前owner d809…，旧CLI为exec兼容入口。

最新继续点：`qualification/g4-progress.v14.md`；v2已失败终止，10/108有效callback、0完成root，17资源操作全known。固定负载时钟诊断尚未运行；G4仍needs_work，正式数据与写作均未开始。

最新继续点：`qualification/g4-progress.v15.md`；固定负载诊断完成并known结算，18资源操作373calls/2382225tokens、无held/unknown；新L2无分数rusage诊断尚未运行。namespace祖先撤销组件P2闭合；G4仍needs_work，正式数据与写作未开始。

最新继续点：`qualification/g4-progress.v16.md`；L2工程诊断复现超限并完整known结算，19资源操作391calls/2509976tokens，无held/unknown。进程监督修复候选未安装；G4仍needs_work，正式数据和写作未开始。

最新继续点：`qualification/g4-progress.v17.md`；监督修复已安装、46项正常import检查通过，post-repair载荷资格尚未reserve/run；历史archive identity R1有限闭合、完整对账仍待。G4仍needs_work。

最新继续点：`qualification/g4-progress.v18.md`；修复后L2完整载荷clock通过该次100ms门槛，实际20资源操作414calls/2673500tokens全known且无held/unknown。当前持久化开发输入覆盖已独立核对；共享生成规律/预随机门组件140项检查通过但未安装，完整pilot及正式phase接线仍待。G4仍needs_work，正式数据和写作未开始。

最新继续点：`qualification/g4-progress.v19.md`；shared-law/预随机组件有限独立通过。新DatasetProof原173合并检查通过但独立发现最终snapshot后字节/tree闭合P2（含locked-main依赖），保持未安装并待v2修复；one-shot writer只是14项Fake检查的未审prototype，禁止实际formal抽样。资源不变，G4仍needs_work。

最新继续点：`qualification/g4-progress.v20.md`；DatasetProof最终窗口P2和writer已登记但丢目录后重抽W1均有限独立复验关闭。组件仍未安装；worker内容reader只是未测draft，实际formal consumer全链、完整开发pilot与集成/最终freeze仍待。资源不变，实际formal数据和论文正文未开始，G4仍needs_work。

最新继续点：`qualification/g4-progress.v21.md`；worker内容/预warmup接口完成有限检查；实际配置hash helper遗漏闭包的P2经v2独立复验关闭，42项通过。执行清单/recording抽取是未测draft，runner/service/receipt/完整轨道仍待接线。全部staged未安装，G4仍needs_work；资源不变。

最新继续点：`qualification/g4-progress.v22.md`；execution有限独立26+17通过，新增dispatch/service接线40软件项、全staged323通过但尚未完整集成/独立审查；全部未安装，无实际正式数据或新flight/model。完整pilot/clean+delay/finalfreeze仍待，G4仍needs_work。

最新继续点：`qualification/g4-progress.v23.md`；新v3完整开发pilot首槽两clock失败(.4054/.2104s)耗尽cap，0/108有效callback，全部失败/未启动分母和两attempt归档；22资源全known474calls3004256tokens，无held/unknown。primary-v1独立5项request-changes，working修复+secondary原型362检查通过但未独立接受/未安装；M2/L3完整遥测诊断待。G4仍needs_work，无实际正式数据或论文结果。

最新继续点：`qualification/g4-progress.v24.md`；M2/L3单次完整遥测复现.210282s/32超限，独立GC/等待阶段重计及23资源known504calls3169634tokens闭合。未安装runtime候选8检查通过但无真实载荷/源审；primary v2仍有PC5-b，working类型修补18检查待独立复验。G4仍needs_work，无正式数据或论文结果。

最新继续点：`qualification/g4-progress.v25.md`；新 runtime 单次 M2/L3 工程负载600s实际 maxlag8.293ms/0超100ms，24资源全known534calls3335012tokens无held/unknown。共享worker接线115检查、17+5独立有限复核、真实20s smoke通过；原始load独立重计闭合。新v4完整108槽仅prepare、尚未dispatch；PC5-b最小修补已独立关闭，formal graph未安装。G4仍needs_work，无正式数据或论文结果。

最新继续点：`qualification/g4-progress.v26.md`；v4首槽真实600s生产验收通过，1/108 callbacks、0完整root，max47.3ms/0超100ms；25资源known555calls3472962tokens无held/unknown。已在order1预约前barrier；staged393软件检查通过且未安装，新增近邻仅source acquisition，独立完整G4接受仍待，无正式数据或论文结果。

最新继续点：`qualification/g4-progress.v27.md`；v4现3/108 callback、0完整root，三槽真实600s/max47.3、85.7、10.0ms均合格，order3前barrier。26资源known584calls3633384tokens无held/unknown；staged393检查的secondary/runtime delta独立29+8有限复核完成，仍未安装。新三近邻独立核对完成，14引用机械v1失败/v2 pass分别保留；统计原文仅有限阅读、待独立审。G4仍needs_work，正式数据和论文结果未开始。

最新继续点：`qualification/g4-progress.v28.md`；v4现4/108 callback、0完整root，新增B4-M1真实600s/max12.9ms合格，order4前barrier。27资源known597calls3726671tokens无held/unknown。两统计来源独立23项通过、16引用机械pass及限定literature阶段实际recorded，native进入experiment；G4仍needs_work，formal候选未安装、分析锁实际producer待补，正式数据和写作仍未开始。

最新继续点：`qualification/g4-progress.v29.md`；v4现6/108 callback、0完整root，新M2/B0真实600s/max9.21、9.54ms合格，order6前barrier。28资源known627calls3890237tokens无held/unknown。隔离分析入口最终15/408软件检查通过；验证后重绑定与空call目录两轮独立P2均保留反例并限定复验关闭，仍未安装/未用实际formal数据。G4仍needs_work；完整pilot、clean/delay实际集成及finalfreeze/独立接受待，写作仍未开始。

最新继续点：`qualification/g4-progress.v30.md`；v4现9/108 callback、0完整root，新M3/M1/M3真实600s/max10.41、14.95、14.68ms合格，order9前barrier。native known686calls4307817tokens无held/unknown。继承L1 D=None误拒P2经真实grade合成例独立复验关闭；隔离候选429软件检查及compile通过。事件phase协议文字修正与纯配对统计分别有限复核；后者尚未接线/注册，全部formal代码未安装。G4仍needs_work，无实际正式数据或论文结果。

最新继续点：`qualification/g4-progress.v31.md`；v4现12/108 callback、0完整root，新M1/M2/B4M1真实600s/max13.70、9.70、14.02ms合格，order12前barrier。native known762calls4805143tokens，无held/unknown。新增secondary三轨SDK/主锁/真实回执重核与common-calendar/事件双路径候选；490/499阶段绿灯未掩盖独立3P2及8身份残余，最终v5 targeted78和compile通过、22项有限复验闭合。精确43source26tests仍未安装；response/resource诊断、完整pilot与clean/delay/finalfreeze/独立G4仍待，无实际正式数据或论文结果。

最新继续点：`qualification/g4-progress.v32.md`；v4现15/108 callback、0完整root，B1/M2/M3新增真实600s/max9.93、12.55、14.64ms合格，order15前barrier。native known806calls5111469tokens，无held/unknown。response/resource候选接入main/secondary并绑定明细ZIP/摘要，48green遗漏的多response snapshot覆盖经两独立IO探针复现，v2修后全522/compile过并有限复验。精确44source27tests仍未安装；完整pilot、actualclean/delay、finalfreeze/容量/完整图预检及独立G4仍待，无实际正式数据或论文结果。

最新继续点：`qualification/g4-progress.v33.md`；v4首root18/108callback完整闭合，三新增max11.81/12.33/13.78ms均合格，order18前barrier。native known883calls5634314tokens，无held/unknown。整图source-only预审未见确定死路；M2输出诊断保留架构依赖及量化/实现归因未定。首root部分容量投影低于128MiB/16MiB门，但L3/失败、最终归档/导出和磁盘余量待fresh核算，不能授capacity或G4通过。生产冻结及正式分母不变，继续原pilot；正式数据和论文结果未开始。

最新继续点：`qualification/g4-progress.v34.md`；v4现21/108callback、1完整root，新三L3 max9.24/8.57/8.47ms全部合格，order21前barrier。native956calls6141839tokens，无held/unknown。容量v1两过强解释由v2澄清且独立确认；report audit次数与占用名额已全21槽核对，原score不改。formal逐请求深审计的负载/非对称成本/计时范围及既定clean20组合证据计划已source-only审阅，原G4硬门不扩大。全部formal未安装，完整pilot/实际集成/最终冻结容量和独立G4仍待，无正式数据或论文结果。

最新继续点：`qualification/g4-progress.v35.md`；v4现24/108callback、1完整root，新三L3 max9.86/8.88/10.05ms全部合格，order24前barrier。native1020calls6612111tokens，无held/unknown。三原Y为0.3051/0.3437/0.5165，全部重复保留；报告audit与实际占用名额分列并和原600s进度核对。生产冻结不变，无新诊断/安装/协议改动，formal仍未安装；完整pilot及后续G4条件继续，正式数据/论文结果未开始。
