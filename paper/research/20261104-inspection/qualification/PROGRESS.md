# 参考任务与平台资格进度

当前摘要见 [GATE_STATUS.md](GATE_STATUS.md) 与 [g4-progress.v1.md](g4-progress.v1.md)。G1–G3已在声明开发范围通过；G4新增真实dev L1/L3、回执/cache/重放、24根输入和本机模型结构预检，仍未冻结。下文各版本为历史记录，旧失败与未完成判断不回填。

本页接续 `RESULTS.md` 的早期开发探针。没有进入正式方法实验、test/holdout 或论文正文写作。

## 已完成并复核的控制资格

G1 的冻结 v3 矩阵已实际完成：直飞60 s、独立航向变化120 s、盘旋180 s、搜索航线180 s、出航/退出切换240 s，每族20个控制参数root，总计100个slot。五族均20/20通过；累计15,600仿真秒、3,744,000次world物理步审计。每一slot的哈希、时长、240 Hz安全审计计数与对应任务指标均已独立核对。

范围是指定平地、恒定0/2 m/s风、共享模式−1执行器及所登记轨迹/包线，不是现实安全认证。20个控制参数root跨五族不是100个主研究独立道路/地形场景。结果和源码、61项依赖及契约绑定见 `g1-full.v1/manifest.json`；独立复核见 `review/g1-review.v1.md`。新增的控制模式需要另做资格，不能回填这100条记录。

## 参考链与工具/裁判

已保存真实PyFlyt动力学下的完整加速B0开发链，涵盖搜索、实际位姿视锥/ray观测、两视角确认、静态/动态报告和四机空中退出。历史 `reference/diagnostic.v4` 为600 s，H=E=1、六个静态代理均有有效报告；动态及时交付很弱。它不是1×资格或正式方法成绩，其旧SR实现被独立审查确认错误，不能采用该SR。

G2新增正反例与实际短时四机交接检查。独立审查发现的动态封存重排、CG射线自遮挡、SR组件判据、事件发布顺序和异常审计丢失问题已按反例修复并复验。传感器使用每机本地因果tracker，覆盖账须等原始测量包实际到达，断链补传保留原时戳，自动报告路由绑定采集时的已批准job。对应50项协议/评分测试通过，包含实际短时四机交接；当时全仓库93项通过。

## v5真实1×参考及重放

`reference/realtime.v1` 已实际运行600 s，冻结源码未变，C=0.9280245、S=1、D=0.00294118、H=E=1、Y=0.643655、SR=0、静态召回=1。剩余航程约5999 m/机，满足明确的1000 m退出余量，未将该资源模型称为真实电池。单条B0参考不能支持一般可靠性或方法优劣结论。

最大时钟滞后0.033861 s；协调器阻塞5.000447 s时，物理世界推进4.991667 s，机0位移95.267 m。该等待是sleep替代探针，不是模型/API时延。正常inspect观测101条，位置向量RMSE=1.293 m、类别正确100/101；正常survey观测416条、RMSE=5.711 m；背景假候选18条。传感点阈值与单条参考收益/时钟门槛有证据，未由detected-only记录估计漏检概率。

按110次实际工具执行时刻重放，13组分数/安全/完整轨迹/原始观测/覆盖/job/报告等产物逐项相同。重放不是新策略重复或新独立场景。产物为 `reference/realtime.v1.{replay,analysis}.json`，完整冻结本地源码另存 `source/reference-v5/`；独立结果复核见 `review/mission-review.v1.md`。

动态180格仅1格非零；本次没有observe_track/handoff调用，SR按原计划应为0。断链期间本次没有目标Observation样本，因此不能声称目标证据补传已端到端实证；有60份覆盖包保留原测量时刻并在恢复后补传。回放另发现B0会在确认包尚未到达时重复确认，当前已修复该状态竞争，但新版本需重新验证，不能回填旧产物。

## 尚待通过的门槛

- G2：持续观察/跨区执行的完整语义与物理可行性、真实目标断链补传等剩余场景；已有参考工具/裁判子集的修复和复核不冒称全平台通过。
- G3：已取得v5单条真实1×收益/时钟/传感点阈值证据；仍需持续动态观察的可解性与更多dev根校准。新执行模式须重新资格；旧参考不能代替更新后平台验证。
- G4：开发冻结、完整工具能力、强经典策略、模型与prompt/预算、完整记录和统计适配及pilot资源依据。当前固定stripe搜索工具不支持一般跨区重新分配，动态观察仍需更充分的物理跟踪能力，不能冻结为正式平台。

两套architect候选及独立交叉评判已完成，已开始小物理probe；完整执行器尚未集成。期刊RA-L和默认任务预算 `{}` 的实际答复持续有效；没有新增付费API、额外算力或投稿授权。正式八配置与原19项义务保持不变。`human_review_pending`；独立工程审查不是人类领域审查。

## 新执行模式的物理否证与修复

采用A的单写者/异步规划边界，吸收B的18m/s移动观察环与短S连接。解析推导不是资格；生产mission尚未迁移到新执行器。

单机因果观察probe只收2Hz本地带噪、10%随机缺失测量，隐藏点仅由sensor/auditor使用。`observe-guidance.v1/v2`的原控制器安全但稳态高度误差约9m，否证暂拟2m高度管。仅在probe中把pitch_p调至4.5、elevator_trim调至−.1后，v3从80m/20m/s实际爬升减速至96m/18m/s，逆风2m/s、移动点1.5m/s，90s无硬违规，稳定段径向误差1.559m、高度误差1.228m。v4加45°转向与45..48s强制失测，120s仍安全、视锥可见率1，最大测量缺口3.5s，因此缺口门槛明确失败；没有把预测或上传6s宽限补成新测量。尚不包含遮挡、分类、传输、多机任务。

`transfer-single.v1`实际完成最南→最北转场、目标区搜索圈、回原始退出圆。四机初始8s安排的完整前缀/返回/周期尾部解析间距不足而被拒。有限搜索后`transfer-fleet.v1`虽安全，退出反向换bank使第四机水平误差10.322m，暂拟管失败。加入40m退出切线runout并重算时序，`transfer-fleet.v2`四机都进入另一区完成圈并回原退出圆：45370次实际240Hz审计，无硬违规，实际最小间距38.403m，10Hz采样最大水平误差5.798m、稳定高度误差1.385m，各机退出dwell均≥35s。

逆风`transfer-single-wind.v1`在20m/s/R80时安全但误差20.831m/2.947m，暂拟管失败；18m/s的v2得到水平3.405m/高度1.450m，无硬违规、退出38.18s。上述误差峰值来自10Hz trace，**只证明采样管检查，尚非连续6m/2m管资格**；正式新模式资格须逐240Hz检查误差。硬安全本身始终逐240Hz审计。

独立审查复现了几何输入不连续可破坏固定速度补偿的漏洞；已加构造连续性/有限性与screen插值斜率校验，并修复2π↔0等价切向的heading插值。9项guidance/path测试通过；本轮新模块前全套99项通过。命令须用`.venv/bin/python -m pytest`保持项目import路径；直接pytest的collection失败日志也保留。当前仍推进共享执行器集成、连续误差审计及新模式资格，G2–G4与正式实验未放行。

`transfer-fleet-wind.v1`进一步实际执行18m/s、2m/s逆风四机换区/搜索/返回，新增逐240Hz误差审计50986步：无硬违规，最小距37.057m，连续物理步采样最大水平3.419m、20s后高度1.437m，6m/2m管违规步均0，退出dwell均≥38s。这仍是一个开发配置，不是20roots资格矩阵或整体G1放行。新审计待独立后续复核。

独立`review/guidance-path-review.v1.md`已完成两项P2修复复验、历史因果滤波与几何/摘要重算；9项局部测试通过，新基础模块阶段全套108项通过、compileall通过。该全套结果发生在flight API迁移前。当前正把flight改为只消费显式FlightTarget与退出标记，并迁移mission/作业写者；此工作中的当前树需集成后重新跑检查，不把早先108通过当作迁移后验证。

## 统一作业执行器集成

flight现只消费显式FlightTarget/退出标记；作业写者迁入executor，搜索与全前缀/周期退出尾环几何检查在独立进程。采用18m/s和probe验证的俯仰增益，但包线仍标development-unqualified，不能引用旧100/100作为新模式资格。新增future2s提交、四机token/revision、逐物理步XY与开发高度区间审计、无回应5wall秒分类和不中断既定回退、旧/重复本地包隔离，以及同批报告去重/拒绝回滚。高度切换前20s是明确的开发过渡区间，之后±2m；不称全程2m管。

独立`review/executor-integration-review.v1.md`复现并关闭了queued-source交接、Z捕获、规划无回应占队列、同批静态报告重复四项问题。49项边界/协议/path检查与局部compileall通过；全套及新资格矩阵仍待。`executor-diagnostic.v1`加速600s安全退出但B0重复静态报告耗满名额、无observe调用；v2加速600s静态全交付、H/E1、D=.038235、SR0，已有4次observe但大量规划结果在加速时钟下过期，不能当主轨时延或动态能力结论。两份原attempt及源码保留。

新的`executor-realtime.v1`600s真实1×正在运行（含5s协调器阻塞），未在完成前判门槛；当前树与其启动源码保持一致。正在补内部规划事件的精确重放；这类重放只做审计，不计新策略样本或新时延数据。G2/G3/G4与新G1待完成，正式八配置、19项义务未变化。

`executor-realtime.v1`已实际完成600s：C=.955590、S=1、D=.052941、H=E=1、Y=.669510、SR=0，8份报告、3次observe、144000次实际物理步审计。最大时钟滞后.101875s，严格超过原.1s门槛，不能四舍五入判通过。新逐步tracking记录最大XY误差13.895m、5699个world步超6m，稳态Z误差1.533m、4175步超开发高度区间。硬飞行约束仍无违规，最小距34.576m；硬安全通过不抵消资格误差管与时钟失败。源码启动/结束未变，完整源另存source/executor-v1/。后续精确内部计划重放和根因修复继续；未进入正式实验或正文。

## 精确内部规划重放与控制根因

`executor-realtime.v1.replay.v2.json`已通过：142次实际工具调用，13项原组加execution audit/envelope/infra/tracking，共17组逐项一致。每个内部plan输入从相同本地观测历史重建，纯planner结果哈希复算，响应只按原接收tick释放；原planner墙钟耗时被保留。此审计不是新策略重复或时延实验。第一次归档漏映射绝对runner路径的失败日志保留，已通过source_locations同等映射补齐archive并完成复验，没有修改原结果。

`executor-realtime.v1.tracking-analysis.json`重建291个内部响应，10Hz最大误差点在t480机0退出下降：水平13.895m，沿航迹领先11.814m、横向−7.314m；机1的6.788m峰值也在切回较低搜索高度后。不是由这个单点推断所有误差原因。新的独立高度切换probe确认原控制在下降时会出现16.640m相位误差；原始与加油门敏感性两次失败均保留。

限制desired vertical command至0.8m/s并对该饱和环做积分抗饱和后，`altitude-energy.v5`在四机2m/s逆风、180s/43200实际物理步中，无硬违规、XY/高度区间违规步均0；最大XY3.630m、稳定Z0.909m、退出均≥79s。平静/逆风早期v3/v4结果也保留；v4尚未验证全过渡区间，不以其旧pass代替v5新增检查。此0.8是内部垂直指令上限，不是已认证的实际爬升率界。按实际高度差/0.8+20s登记过渡区间，再查稳定±2m，硬AGL60..100与水平间距条件未变；新模式仍须冻结资格矩阵。

已把稳定控制profile接入mission；持续观察不再反复替换尚未完成的接近段。3s本地失测后改为真实宽视场重获，先形成原始帧再以可观测位置/类型选择最多一个匹配（含背景假候选），四机同帧曝光后才执行交接后置条件。上传6s宽限保持独立。新增visible-before-miss机会账与背景frame trial，后续G3可以核对漏检分母，而不从detected-only行推断。顶层工具日志去掉内部handoff转发重复，旧日志仍按旧版本处理。119项在最新日志字段加入前通过；当前完整来源需再次检查。尚未把新profile跑成完整G1/G3通过，正式实验和论文仍未放行。

## 新G1完整矩阵前的开发冻结

已保留executor-pilot.v1..v6全部slot。v1/v2低初始高度与较长capture lead下，观察D=.411/.628，虽无硬/管违规但不足.8门槛；v3统一96m仍出现观察与过渡失败。v4把共享计划捕获lead由2s降为0.5s，root0得到固定180s窗D=.977778、observe成功、四机退出，86400实际物理步中XY最大3.432m/稳定Z1.389m且管违规0；这是子集，qualified=false。v5使用实际高度probe验证的desired VZ1.2，两个transfer root均通过，observe root1 D=.733333仍失败。v6新增先在既定安全航迹尝试2s真实相机确认后再请求转场，root0 D=.794444仍按原.8门槛失败、root1 D=.85通过。没有删root或放宽D/硬约束；当前完整分布必须由全20根检验。

共享初态统一96m是开发观测几何校准：72m配R90的可见余量不足；原始60..100m硬限、原始评分/出现时窗/全部格分母不变。新1.2m/s是垂直指令限制，实际物理率无此认证；0.5s捕获lead仍须真实异步1×验证，controlled nexttick只用于G1。

`gate-contract.v11.json`绑定实际profile、96m初态、.5s lead、2s相机尝试、实际Hz、20个不同roots、各家族成功率.95。新G1矩阵为原五类基础控制各20（100 slot）及observe/transfer各20（40 slot）；新G1低层控制和原始G3真实时间证据分开。所有4机必须实际退出20s、保留1000m资源余量，追踪/hard审计均须等于时长×240且非零。缺审计或profile不符是operational_failure，不能被95%均值抵消。

独立`review/g1-executor-qualification-design.v1.md:163`补验已关闭setup异常漏close、契约未绑定与审计计数不全三项；仅独立复跑3项guard，不冒称独立重跑124项。主Agent全套124项在最终qual结果helper微调前通过，最新9项qualification/analysis guard通过；完整冻结矩阵尚未运行，G2–G4、新完整1×及正式实验仍待。

## v11完整G1结果与第二次结构否证

`g1-full.v2`已完成原五族100/100，全部通过；`executor-full.v1`完成40/40，transfer20/20通过，observe仅1/20通过，整体新G1失败。两矩阵合计30000模拟秒、7200000个实际240Hz world积分步；它们是相关控制参数roots，不是140个独立科学场景或1×证据。原源码/依赖/契约启动结束一致，所有失败完整保留，无operational_failure/未启动slot被删除。v11执行闭包14文件另存`source/executor-v11/`，绑定原manifest SHA。

独立`review/g1-v11-full-review.v1.md`核对140个slot SHA、各61项依赖、v11字节/运行profile、固定笛卡尔积、时长和完整审计覆盖。基础几何/稳定指标与observe完整1200个隐藏固定格的D均重新计算一致。observe仅root1合格，17个D不足.8；root7/19虽D≥.8仍XY越界，不能改判。XY四个root合计5142个world步、Z七个root合计4365步，有重叠。全部执行器硬安全、四机退出与资源余量通过，不能抵消持续感知和暂拟管的失败。

当前产物没有保存全部240Hz nominal/actual逐tick数据。独立审查能重算D和10Hz几何，精确tube峰值/计数只能核对冻结审计实现、覆盖和聚合，并交叉印证。后续新运行将补充可复算误差记录及峰值witness；不修改旧数据补造证据。G1仍未通过，G2–G4、正式实验和正文尚未放行。

第二轮architect的A为同律因果向量场加闭环预测，B为固定偏心弧/有限曲率ramp/相位直线；各自独立小probe与限制保存在`planning/execution-architecture-round2/`。A的一秒预测初始化有body/world速度坐标缺陷，3.139m不能作tube校准。B解析试验没有ramp，4m目标误差也未涵盖survey σ4二维95%半径，不能凭其3.46m视野余量资格。

主Agent的`cell-capture.v1`仅做一次58模拟秒真实Mission否证，逆风2m/s、四机、有限2s曲率ramp、固定40m测试直线与完整预留返回。13920实际物理步，XY2.533m、稳定Z1.499m，无现有6m/2m或硬违规；inspect在1..3s以两张真实视图完成，报告与observe在3s受理。前28s的机1实际目标距离峰值105.412m，45张真实目标帧、最大新测量间隔1.5s。非自适应固定直线后32..40s距离增至129.323m，约36s开始失视，42.722s后已执行预留无目标返回。不能把故意回退后的失视计为180s任务成功，也不能把初捕获成功扩成持续资格。该结果支持继续验证短弧几何，但否证任意固定runout；下一步从旧未来gate构造有限弧/直线并以消费者因果观测排序，保持完整fleet/返回准入。

## 有限弧执行器、完整返回与独立缺陷修复

已删除生产规划中的精确目标中心圆/C2捕获分支，改为从未来旧gate生成有限固定弧/直线及2s曲率ramp；8s work承诺包含完整无目标return与周期home。目标预测只排序，原实际帧/固定180s全部格/交付规则不变。获取中可续约，inspect固定2s等待删除。每机独立传感模式/协方差/创新、实际world地速/独立空速/完整导航与PI快照、逐步实飞剩余资源进入请求。全fleet的固定时序前缀/尾环检查向量化，与标量插值的等价性有独立复验；不宣称无限期物理相位管。

`cell-recovery.v1`实际150s/36000步，18s永久停止受控规划，四机实际退出dwell最短71.267s；XY2.533m/Z1.465m、硬/6m/高度区间越界均0。新增每个240Hz tick的XY误差、实际Z及区间压缩记录与peak witness，可离线重计；不是完整原始状态物理重放。独立`review/finite-arcs-review.v1.md`用stdlib逐36000行复算一致，并发现三项P1：跨曲率接缝的中心差分导致gate bank跳变、capture漏检world course、复用return/尾环仍有bank跳变。没有用单fixture通过掩盖缺陷。

v12的`executor-pilot.v7`两个新开发控制root完整360s各86400步：100 D=.838889通过；101 D=.694444、XY8.302m/2378步越界失败，硬约束均无违规。峰值t230.317位于observe结束后的无目标return，支持继续检验恢复入口，不能单凭该点证明全部根因。完整源码另存`source/finite-arcs-v12/`，原失败不重标。

三项已实现修复：从旧gate的nominal bank逆解曲率、world course≤10°捕获；对初始capsule、转场、return内部和return→tail的自有空间/时序几何做有限C2窗口平滑，exit先2s降曲率。C2不用于获取目标圆。尾环前2s进入有限prefix、旋转原tail保持原精确周期；nominal bank≤30°，实际hard仍45°。成功gate及work/recovery/tail seam新增实际完整状态/PI与capture metrics；seam域失败单独计数并拒绝资格。8s短窗先见损失后，增32s因果CV预览仅作导航排序，不承诺32s工作或生成新证据。

修复后的`cell-recovery.v2`同样150s/36000实际步，18s停止时机1roll=-.456125rad、角速率非零；14次成功commit和9个work/recovery/tail seam均有实际状态/metrics，capture均通过。四机实际退出最短68.417s，XY2.533m、稳定Z1.467m、hard/管/seam失败均0，source unchanged。仍只是一个受控fixture，不是完整G1或真实1×。

v13的`executor-pilot.v8`两个原开发root完整保留：100 D=.855556、XY3.292m通过；101 D=.816667已达D门槛，但XY6.689m/955步越界仍失败（峰值t278.492在机0退出tail），Z/seam/hard无违规。不能把D通过改成整体通过。源码另存`source/finite-arcs-v13/`。

独立后续复验又发现允许风向(0,-2)及(-sqrt2,-sqrt2)会因C2窗口末端浮点近重复行在全局平移后成为相同timestamp，初始化失败。已保留审查事实，按既有slice的1e-8端点间隔修复，未把非法初始化当普通任务失败。最新全套136项通过`runtime/tests-finite-arcs.all.v4.log`，包含负风向初始化、严格时间/斜率、gate bank边界与错误course反例。独立修复版复核仍按对应报告范围判定。

`gate-contract.v14.json`冻结下个开发候选：nominal18m/s不变，沿航迹追赶requested airspeed从16..24限制为16..18.5，以否证追赶速度扩大转弯半径的可能反馈。它只是待验证假设，不是已证实根因。保持6m/2m、hard、原D/时窗/分母；新root200..219在该版本参数修改前登记，pilot/full共享时不多计独立样本。新候选尚未物理通过；G1、G2–G4和正式实验/论文仍未放行。

独立`review/finite-arcs-review.v2.md`已复验原F1/F2与实际C2几何/入口日志，并保留F4原初始化阻断。`review/finite-arcs-review.v3.md`绑定当前四源，九代表风向共36个初始program和18个完整exit/survey请求通过严格时间、有限性、≤27m/s斜率、≤30°nominal bank与全fleet检查，原F4反例关闭。此为有界解析复验，新18.5候选尚无物理资格，旧probe不迁移背书。

## 17m/s几何校准、启动配平与下个完整矩阵

v14的`executor-pilot.v9`root200/201均保留失败：D=.777778/.716667，XY5.979/6.031m，后者120步越6m，硬/Z/seam均无违规。真实帧分别在约66..78/65..82s及随后每圈出现失视，限制phase速度本身未解决视野余量。源码保存在`source/finite-arcs-v14/`。没有将新roots较弱的分数单独归因为profile，噪声/初态也同时改变。

开发校准改为共享nominal17m/s、观察半径70/80m、follower16..17.5。它增加固定偏心弧的光学余量，并减小home圈的名义bank负担；不是改变hard空速15..25m/s、45°bank或原评分。v15的`executor-pilot.v10`重用已暴露root200/201作诊断，D提高至.877778/.933333且XY5.718/4.589m、水平违规0；但Z区间916/1230步违规，root201稳定Z2.003865m也严格失败，不四舍五入。root200启动t7.146s下降至93.6035m。源码保存在`source/finite-arcs-v15/`，原slot未改判。

`g1-full.v3`已经按17m/s、旧pitch_trim=-.04完整执行五族100/100、均通过。它只属于该旧控制profile，不能背书新配平。所有该轮源码/依赖/契约启动结束一致，仍不抵消executor高度失败。

配平probe明确覆盖actual control override，不伪装成SDK正式成绩：`startup-trim.v1`(-.055,calm)150s通过，但`startup-trim.v2`(-.055,max-wind)仍325个Z区间违规步，均保留。随后`startup-trim.v3/v4`使用-.07，在max-wind/calm各150s、各36000实际步，确认/报告/observe与18s永久受控planner停止后的四机退出实际完成，hard、6m/2m、seam违规均0。最大XY3.923/3.312m、稳定Z.860/.861m；整个高度范围分别94.634..97.112m与95.141..97.107m。最大风向为显式近2m/s输入，非完整连续风域证明；两单例不是20roots资格或180s D/G3。

已把实际probe的pitch_trim=-.07接入共享`inspection_gains()`，冻结`gate-contract.v16.json`；其余控制profile、17m/s、70/80m观察弧、16..17.5follower不再改动。下一轮重跑原基础五族各20，并对先前未评估的控制roots400..419运行observe/transfer各20，共140slot。此为控制开发资格，不是科学道路/地形root。所有4机实际退出/1000m余量、完整240Hz误差记录、成功gate与四机tail seam的OwnState/PI日志必须存在；缺失或不可重计归operational_failure，不能被95%成功率掩盖。当前全套136项通过`runtime/tests-17m_s-trim.all.v2.log`；新完整矩阵尚未执行，G1仍未放行。

## 八配置记录SDK预检

`eval/inspection_recording.py`使用提供的autoresearch `ExperimentRecorder`，按classic3配置×1重复、LLM5配置×3重复分成两个内层矩形，再用同SDK形成root-condition外层均值；每次测量前另用提供write_once保存不可变start事件，非额外飞行。内层完整原始记录和SHA绑定到外层，操作故障保留失败/未启动全分母并阻断汇总，策略失败则应保留合法原评分。没有复制经典成绩充三次，也没有缩成两臂。

`adapter-preflight.v1..v4`均为合成夹具；最新v4实际通过72主root形状1296内层槽/576外层均值、12holdout形状216槽/96均值。主形状经典值0、LLM重复0/.25/.5，逐项手算均值0/.25匹配。双root故意操作错误保留classic6/LLM30/outer16全分母；非有限、超范围和布尔Y分别拒绝。SDK及adapter15份源在v4启动结束一致，完整source-snapshot与SHA保留。它们不是1512条已飞数据、方法结果或费用receipt，fixture里的B1不是dev选出的B_classic*。

详见`planning/recording-adapter.ral.v3.md`。主分析的等级宏平均、ID/OOD分层root配对bootstrap/Holm、SR聚类、事件同日历/兄弟、恢复删失以及真实trace/费用接线和独立审查仍未完成，故该预检不放行G4。G2旧跳epoch补传脚本已改写为从0实际飞行至206s再加入地面proxy的方案，尚未实际运行新版本，不能引用旧33s探针为新平台证据。正式实验和论文仍未启动。

## v16完整140slot与新G2补传结果

`g1-full.v4`已完成100/100、五族均通过；`executor-full.v2`已完成40/40，transfer20/20、observe15/20，**整体G1仍失败**。两manifest的source/dependencies/contract启动结束一致，无operation failure或未启动slot被删除。完整v16执行闭包15文件另存`source/finite-arcs-v16/`，归档当刻与当前源逐字节相同并有binding receipt。新结果待独立全量复核；不把先前进行中计数或单例通过作为验收。

observe失败为403/413/415/418/419：403 D=.916667但XY7.079m/1794步且一个seam失败；413 D=.811111但XY9.980m/2829步且两个seam失败；418 D=.95但XY6.489m/411步；415/419分别D=.738889/.766667，XY均未越6m。全部高度区间、hard、安全退出/资源与infra均通过。三个XY峰值沿/横分解分别约(-3.387,-6.216)、(-8.425,-5.350)、(-3.131,-5.684)m，不能把全部误差说成纯相位或纯侧偏。

415/419的内部日志各有26/34次traffic occupation拒绝和14/15次no finite candidate；419另有一次bank capture error11.807°。源码的home恢复当前只返回首个fenced CSC，而两个最佳work候选之后没有再搜索其他home CSC；这是已确认的有限搜索限制，尚未证明替代路线就能提升D。当前follower的曲率前馈按nominal空速生成，实际空速/phase修正可能不同；下轮将针对两处做有界改进并重新前瞻验证。原6m/2m/hard/评分窗口仍保持，所有v16失败不改判，正式实验不启动。

上一节记录“尚未运行”的新补传已随后实际完成：`protocol/store-forward.v2.json`从0实际飞至241s，57840实际物理步，无跳epoch。206s才加入地面proxy，不重置飞机；207s inspect受理、209s真实完成、209.5s报告/observe受理。34条原目标数据全部在240s恢复通信时到达；239s的32条待传原始数值/时刻与已达版本一致。完整60格有限trace诊断delivered4/60、observed8/60，未飞的未来格仍在分母，不能称180s观察成功。

独立`review/g2-v16-review.v1.md`确认16源snapshot/hash/current一致、37条report provenance、34补传/32pending、57840行逐步误差重计及不调用生产scorer的60格重算一致；另独立执行51项协议/评分、2项单步物理检查通过。protocol/scoring与v11冻结源逐字节相同，允许复用不变纯core的审查。G2仍pending，明确缺当前实际360s外生退休→任务释放→安全盘旋/终态链；0步mock退休语义不替代该实际场景，新600s真实1×尚未运行属G3。

## 记录顺序与主推断规格审查

独立`review/recording-statistics-design.v1.md`核对v4的1296+216内层身份、start与672条聚合绑定，同时发现旧adapter固定classic后LLM顺序、不具硬中断恢复、尚无主推断等范围阻断。已由主Agent把执行改为显式预冻结episode-order：domain/level根块贯穿运行时段，每root的18个method/repeat跨两个SDK内层随机交错；SDK canonical grid仅作存储。每次callback与冻结order一致并有index/hash receipt，全部正常完成后才写全局记录完成索引。v5合成预检已执行通过1296+216全形状及同一seed输入重排不变；尚未独立批准改版。

start仅表示调用measure前已写尝试意图，不证明真实物理/provider已启动，也不保证断电fsync。全局索引只证明callback记录完成；真实runner/终态receipt、hard-kill unknown恢复和有限对称重试仍未接线，不用“无results=未执行”重跑。最新预检source-snapshot保留，不把外层聚合当新飞行。

统计独立审查提供了待实现、待预检的候选：ID/OOD分开、等级宏平均、root内先均值、同root完整向量分层bootstrap pointwise95%区间；null-centered studentized bootstrap给弱宏均值零假设的近似双侧p，三比较×两域global Holm6；零/近零SE与退化draw保守处理，并公开有限样本/独立root假设。它不是已运行统计、有限样本精确保证或G4验收；真实generator/split、强classic、模型/预算、费用接线仍未具备。19项义务与八配置不变。

## v17全量否证、v18前瞻包络与统计实现

`executor-full.v3`完整40/40已结束：transfer20/20，observe17/20，整体qualified=false。source/dependencies/contract启动结束一致。20个observe的完整180s/60格D均≥.888889；604/612/619仅因原6m同时间点XY界失败，分别535/741/839个240Hz world步，峰值6.868807/7.630975/6.966670m。其余高度区间、seam、hard、安全退出、资源和infra检查均通过。原v17源已冻结在`source/finite-arcs-v17/`，旧失败不改判；资格slot与严格本地observe作业终态仍分开。

`planning/tracking-envelope-calibration.v1.md`是开发包线重校准提案，撰写时transfer尚未结束；这里补记其后完整结果与实施事实。当前v18候选已实现10m同一时刻欧氏水平误差界，所有任务的全fleet完整前缀/返回/尾环名义连续间距同时收紧至35m、围栏至540m。垂直±2m和原科学评分/180s全部格/实际15m硬间距不变。显式trace门槛与合同必须一致，历史缺字段固定按原6m解释，不能用新默认重计旧成功。`gate-contract.v18.json`在运行前冻结新控制roots800..819；独立几何/绑定审查与全40前瞻实飞尚待，不能判G1通过。

新149项完整测试通过，包含四任务统一35m准入拒绝、10m capture边界、legacy6m/expected10混用拒绝；项目`compileall envs train eval scripts utils tests`通过。基础100的复用仍受三源、完整G1子对象、依赖/导入初始化不变和独立组合审查约束，不能替代新40。

候选`eval/inspection_statistics.py`已实现ID/OOD等级宏平均、root配对分层bootstrap、中心化studentized参考、globalHolm6和SE退化保守处理。4项单测及`statistics-preflight.v1`72根合成手算/顺序/batch/退化预检通过，不是实际方法数据或G4验收。独立实现审查与小样本偏态、稀有损失、零门控压力诊断已派发；候选渐近推断可能在n12/16下失准，未据此冻结正式推断。真实split/费用/硬中断恢复、强classic与模型接线仍待。

## 退休链语义缺陷与数值推断前置修订

`protocol/retirement.v2.json`从0实际飞行500s/120000步，物理退出链16项检查通过；第一次v1因numpy.bool_结果序列化失败未保存轨迹，failure.json/日志保留，不计资格。独立`review/g2-v18-retirement-review.v1.md`重计120000行、17源码/61依赖、360s版本1→2、旧航迹保留、一步真实位移、360.5未来捕获与441.883s实际退出终态，并独立跑53项bounded检查。它还找到P1：退休后至421.5s仍有124份非空覆盖包和101条新公开观测。故这条物理子链的passed不能等于G2；完整G2仍失败。

明确保守容量语义：360s起停止该机新的任务target/coverage/background采样，360前已采集包仍按原时戳合法迟到交付，飞机继续物理返航/独立盘旋。已给probe加入两项新验收，但尚未修改冻结v18核心；正运行的40份G1均events=False，将完整结束并保留后另立修复/实际验证，不改旧manifest。

独立统计实现审查否证原bootstrap-t确认性方案：2%root稀有损失、真实弱均值null的200组合成数据中globalHolm6误报189/200，MC95%区间[90.42%,96.90%]，99999draw复核仍失败且无零SE退化。原800组四分布/全部source/错误p保留。已在正式数据前修订为有界独立root的Hoeffding基础p+globalHolm6，补pointwise95与Bonferroni-global6同时95有限界CI；保留原root bootstrap95区间与studentized p为明确nominal/diagnostic。ID/OOD的同时半宽约.478/.676，低功效和独立性条件必须披露，不因不显著事后扩N。

`planning/statistical-protocol.ral.v1.md`给出完整候选规格。`statistics-preflight.v2/v3`均为显式72根合成预检。独立`review/statistics-bounded-review.v1.md`核验公式/输出角色/28边界与旧800效应；各分布新主误报0/200仅是合成诊断，理论保证来自有界独立前提。另发现的np.bool_ helper拒绝漏洞已由唯一guard改动修复，`statistics-bounded-guard-review.v1.md`独立核验4反例、36合法输入不变、11项测试通过。真实root/费用/恢复/事件/SR与G4仍未验收。

新增`eval/qualify_inspection_reference.py`核对新reference的完整实际Hz/步数、240Hz误差、sensor-visible机会/原始miss/关联抑制/background frame分母与任务链；源/依赖绑定和原质量/时钟校准保留。新增6项及后续合并19项guards通过，尚无新600s真实1x结果。runner现记录实际依赖hash和可选B0/B1/B2，但不进入正式方法评测。

`eval/inspection_classic.py`为开发候选：B1集中式滚动greedy utility auction（不称CBBA复现），B2明确SEARCH/CONFIRM/OBSERVE/HANDOFF/EXIT事件状态；只读公开包/当前任务卡，复用报告响应与安全工具。6项公共输入、唯一派发、同批确认去重/名额、健康交接/version、迟包不回退和失败cooldown检查通过；尚无真实dev成绩、独立算法审查或B_classic*选择。代码存在不代表已具备强经典性能，仍需实际同协议开发试验。

## v18完整40通过、退休采样定向修复

`executor-full.v4`在冻结v18源/合同下完整40/40结束，observe20/20、transfer20/20、qualified=true，source/dependencies/contract全部unchanged，无operational_failure；D范围.838888889..966666667、最大XY7.317076330m、四机退出dwell最短28.441667s。15文件闭包已实际归档`source/finite-arcs-v18/`并有manifest SHA binding。它与五族基础100的组成结论待独立全量结果审查，仍不是40个独立研究场景、真实延迟或G4。

冻结完整v18后，核心仅一行修改：`mission._sample`在exit_active之外也跳过retired aircraft。新v19合同明确capacity退出停止新任务target/coverage/background采样，但旧包可迟到。`protocol/retirement.v3.json`从0重跑500s/120000实际步，新增两项后18checks均通过；360后target observations、coverage packets、visible opportunities、background trials全部0，1条359.5s采集/360s到达观测仍保留。v2/v3全部5000组实际positions相同，240Hz error rawSHA相同；新源差只有mission guard和probe新验收。旧v2的实际G2缺陷不改判，新G2关闭须独立复验。

v19拟条件复用v18 events=False的40份控制证据：该域retired初始空且唯一赋值受events=True保护，新guard分支从未执行；控制/规划/评分、G1子对象和其余源/依赖不变。此处是待独立组合证明，不把旧40算成新40、不修改原manifest。新的G2事件执行和G3真实1x必须使用修复后源。完整169项复验正在运行，正式试验与写作仍未放行。

## G1组成/退休修复与经典状态机的独立验收

`review/g1-v18-full-review.v1.md`完整重计40份/每份86400行、1200个D格、3205请求、2382commit与437seam，支持基础100+新40的开发G1控制组成。`review/g1-v19-composition-review.v1.md`证明只有一个在events=False域不可达的退休guard差异，可复用相同40而不计新重复。20个observe job仍因本地缺格failed，独立报告与当前总览均明确，不能把资格slot通过说成作业全部成功。

`review/g2-retirement-repair-review.v1.md`独立核验v3的17源/61依赖、120000行/5000位置、四通道0和359.5→360旧包完整值，并跑56项检查。旧F1在v19/v3范围关闭；G2十项均在声明的fixture/语义范围内有证据，无已知需要先修的代码阻断。新600s真实1x、完整内部plan重放和综合事件证据仍未执行，不能借该报告判G3/G4。

退休核心修复后的全套169项测试21.23s通过；这是经典策略后续两项补丁前的完整检查。经典旧P1（已受理handoff仍重复第三机）、P2（物理inspect成功但类别tie永久等待）已由独立公开快照反例确证，old源码按独立f3da4081…SHA核实保存在`review/classic-candidates-v1-evidence/`，原反例也复制绑定。新版本按candidate/job/drone保留接收任务到terminal，ACK先于telemetry窗口也保护receiver；物理完成与类别决议分开，等包最多2s后可再确认。新10项纯策略tests及`review/classic-repair-review.v1.md`独立完整旧/新反例复验通过，两缺陷限定关闭；未跑实际classic开发，更未选择B_classic*或放行G4。第一次新生命周期test未模拟失败通知/cooldown的2fail日志v2保留，补真实通知时序并修ACK窗口后v3通过。

当前用户原文授权另存`inputs/qualification-authorization.v1.json`，说明后来指令授权本地G1–G4/非LLM参考任务；不新增付费API、算力采购、投稿或消息发送授权。默认budget={}、19项承诺和完整八配置不变。当前全部长physics与CPU检查均已结束，下一步保存controller检查点并独占本机运行新B0实际600s/1x。

## 新600s真实1×、精确重放与G3检查器修复

`reference/executor-realtime.v2`已实际完成600s/1×，144000实际physics/flight/tracking步；14运行源/61指定依赖启动结束不变。max clock lag=.03148783339032235s，低于原.1s门槛；5s协调器阻塞实际wait5.009128s、飞机推进5.008333s并位移85.106339m。C=.937212864、S=1、D=.4、H=E=1、Y=.779070955、static recall1、SR0；最后四机dwell最短78.241667s，余量最短7787.880762m。硬/XY10/Z区间/seam均0，XY峰6.986842m，稳定Z1.053814m。B0不是正式方法成绩，更不称动态任务全部成功；6个observe job有两个strict本地缺15/39格。

14文件运行闭包已归档`source/executor-v19-reference-v2/`并绑定run/world SHA。加速精确execution replay v1重放116/116真实RPC、21字段相等，v2只在replayer字段表新增raw error trace后另跑加速600s，22字段全部精确相等，无response差异。它们是同一episode的审计，不是新策略重复或时延数据。两版replayer/checker原文件已分别与receipt SHA核实归档`source/reference-analysis-v1/v2/`，不把新源码回填旧receipt。

独立`review/g3-v19-control-sensor-review.v1.md`重计144000行及传感分母/误差，normal inspect877、向量RMSE1.430835m、类别868/877；survey机会2029/196raw miss，inspect1025/103，背景2546帧/25触发/24发出。原max/p95完整72k lag数组未存；审查核源码、最大见证与600个1Hz进度点，不宣称离线精确重计全数组。实际里程核240Hz源hook与保存余量、10Hz仅旁证，raw error不等于全state。

独立业务报告`review/g3-v19-mission-replay-review.v1.md`重建847报告版本/180动态格/340权重，全部score精确相等；实际180s样本属于177–180旧权重格，180–183新格权重2。断链本run为7条survey/61coverage在240恢复，observe-route断链样本0；837条auto-report均关联原曝光作业/route，实际467.5→468 failed-job迟到、470→470.5 cancel后迟到均存在。本run没有handoff；退休旧Observation0但有359.5→360旧coverage原样交付。对应其他正例仍只属于其实际fixture，不挪来填本run。

初版strict checker虽对原数据pass，但独立3变异发现P2：依赖可删为子集、未emit机会/背景分母可删、无关联inspect/report/observe仍pass。修复增加完整实际61依赖集合、外部真实replay绑定及22组完整分母、明确因果join。进一步3个helper反例揭示future/unbound claim stamp与inspection前旧survey仍能形成witness；v3补native/finite/非未来stamp、同刻引用及与有效inspect视图交集。`review/g3-checker-repair-review.v1/v2.md`分别保留中间残余并独立封口，三个原helper反例不依赖replay/SHA gate直接被拒绝。原实际witness不变，qualification.v3实际11checks通过。23目标guards、完整183项20.73s通过；env14源/实际run/world从未更改。

据实际G1组成、G2十项范围fixture/新参考集成、G3质量/时钟/完整重放及缺陷关闭报告，开发期G1–G3资格已具备；这些是有限平地/结构化sensor/声明风与几何范围，非实机、人类或总体认证。G4仍needs_work，正式test/holdout与结果写作未启动。

并行准备G4时只做了公共模型文档/价格与开放repo元数据获取，使用提供的source-fetch留原字节/URL/SHA；并用既有凭据执行一次只读models metadata GET（HTTP200，66chat IDs），未做付费推理、未显示/保存凭据值。现有机器M5 Max/64GiB、可用磁盘约302GiB，为后续本地固定开放权重可行性提供资源事实，未下载模型权重或新增算力。候选API别名不能冒充HF精确权重commit；docs明确max_tokens不含CoT和服务可能变更，正式费用/版本/思考预算仍待解决，尚未选定M1/M2/M3或请求付费授权。

## 控制器自有checkpoint读限故障与受审迁移

原round15 prepare已产生d8ce9…请求，但apply返回`JSON exceeds the 16 MiB input limit`，没有记录round完成。独立只读核实原control16,754,137B尚小于16MiB，checkpoint因额外嵌套缩进为17,326,544B；两者canonical state digest一致，数据未损坏。原implementation bfd85afe…匹配全局安装源。没有删除/重初始化历史来规避问题。

在本run内保存原runtime与修改副本（全局Skill未改），新副本仅lib/runtime.py/CLI两既有文件有改动，新增软件测试；自有control/checkpoint读取128MiB，外部JSON默认16MiB与重复key/nonfinite保护不变。新增明确source-bound的migrate-state-reader入口要求原快照/新实现/真实独立review及完整pair/Brief校验，保留所有研究/资源状态并追加迁移历史，仅使旧rules pending request失效以重新prepare。66项测试OK（1跳过），独立额外17MiB/破损pair/审批守卫/中断恢复fixture通过；初期测试路径/import失败日志一并保留。

实际review `recovery/state-reader-v1/independent-review.v1.json`由control_review出具并绑定SHA，随后唯一owner执行迁移bfd85afe…→a112e7…；迁后逐字段断言研究/预算/activeRound全同、旧history前缀及全部旧operations全同。原pair备份、失败apply、新migration result与验证均保留。原d8ce9…请求/判断文件未改，源在迁移history保留；当前仍round15，须新rules重新prepare/apply，不能把失败apply说成完成。后续唯一入口见`ACTIVE_CONTROLLER.md`，使用R/runtime/autoresearch-state-reader-v1/scripts/research.py继续原`.paper`，不是第二研究账本。迁移不授予G4、付费或论文完成。

并行已写`planning/scenario-adapter.ral.v1.md`候选合同，明确新增CaseSpec/ScenarioMission置于eval适配，保留已资格物理核心；24dev/6pilot与原72主/12holdout分开，L1聚合/未来信息/时窗/容量必要上界/事件语义及真实dev要求。尚未据此生成或运行正式test/holdout。
