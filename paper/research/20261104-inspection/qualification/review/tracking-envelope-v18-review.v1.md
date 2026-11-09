# v18 tracking envelope前瞻独立工程审查 v1

日期：2026-11-05。角色：同模型家族的独立上下文工程Verify；不是跨模型或领域人工复核。只写本报告，未修改源码、契约、既有结果，也未撤销他人改动。未运行新40任务、长physics、600秒任务或付费服务。

## 结论

**未发现阻止按当前冻结输入前瞻执行v18完整40-slot资格的实质代码漏洞。** 10m仍是同一时刻的欧氏水平位置误差，配套共同准入确实提高为35m连续名义间距和540m连续名义围栏；capture/seam/逐240Hz计数及trace元数据使用同一包线。历史缺字段固定6m，新合同expected10拒绝混用。允许运行不等于G1已通过，更不授予G2/G3/G4。

基础100份的三份实际功能consumer、完整G1子对象、61项依赖、版本及导入初始化条件目前与g1-full.v4相同，原100份SHA未变，**可作为v18基础五族组成证据复用**。新的执行器40份仍需自己完成、保留全部失败并独立审查。

初读发现的文档一致性P3已通过追加历史/当前状态澄清关闭，见下文；没有未关闭的运行阻断项。实际10m跟踪能力、全40成功率、真实异步1×及领域人工判断仍未证。

## 发现

### [P3，已关闭] 校准提案的状态叙述落后于实现和完整v17记录

证据：planning/tracking-envelope-calibration.v1.md:3仍为“未实施”，:5仍说v17 transfer运行中。当前实现已采用10m/35m/540m，v17 executor-full.v3实际40个slot全部completed，observe17/20、transfer20/20、qualified=false。提案的科学/工程边界要求本身没有矛盾，但若把该文档当当前执行状态，会误读完成范围。

最小修复：保留提案形成时的历史文字，明确其历史时点，追加“已实施但未资格；v17完整40结束且按6m失败；v18待前瞻执行”的状态和直接证据指针。无需改判旧slot，也不需为这项文字修复重跑physics。

后续复验：主Agent已在同文档:22–24追加“后续实施记录”，明确原段落为历史提案、v17全40结束且仍失败、v18已实现但待资格及新根执行边界。本审查读取追加文本后关闭此项；源码/合同未因文档澄清变化。未把追加段所称compileall作为本审查自行执行的检查。

## 读取和身份

路径相对repo根；planning及qualification路径相对paper/research/20261104-inspection。审查期间下列实现/合同字节保持不变，最后再次核对。

| 对象 | SHA256 |
| --- | --- |
| envs/inspection/planning.py | b87fcdedf3aa26a46560056376fb69f9ff969270149c9cae429b4a9e294df6ab |
| envs/inspection/execution.py | dc301939ea1d55f08f01c69af5561c75f762ac45bdd0d144d677514387425ea4 |
| envs/inspection/mission.py | 24bc2a332508640f4043398f649e180b9cae167b930d22e2108135651c855b75 |
| envs/inspection/paths.py | dbd31dc94a7cdcbcb17b8d9f1847fa88cf4cd12e4609ec8ba57e2233a367855b |
| eval/inspection_tracking.py | 19151659b4a969c51a6cbcb94b4f10aa1cb4489bebec6bd50f28b639a07884ba |
| eval/qualify_inspection_executor.py | b76f6745e4f1cfed8c0b4055b683a93811f23685378ef2c84c3e723d8010a2cd |
| tests/test_inspection_execution.py | aa30b7a93925c946d578cd726db3ff19e005e101d6da1a0841b488a6461abd9d |
| tests/test_inspection_qualification_analysis.py | 413a1f71f39298d96c8281ff6b1c7fb9ef3b35b9cf478ff840aba0633f3146ae |
| planning/gate-contract.v18.json | f49a4439ed3fed4392d34ac69da5d087dea5c825ace49da39c1d2187163c8883 |
| planning/tracking-envelope-calibration.v1.md（追加澄清后） | 3ede976f11e61f7656818ddfd9c5d7c808bf5db1b191b434603d3e97e900b048 |
| qualification/runtime/tests-envelope10.all.v2.log | 606712381e14d965f17c4e349607ba375a935613e7b92b8483d33ee09b35f6fe |
| qualification/executor-full.v3/manifest.json（v17） | 56a968cc6bf307f87edce3b0ead69ec29ffcc235aa434d3dc1d03628d5ce33d4 |
| qualification/g1-full.v4/manifest.json | b07166e6415a9f8cfb6f7d003f53dc356938807537f9ad4a7b451fce108d1b97 |

paths.py只扩展读取了必要的RouteProgram、TimedPath输入与fleet_clearance连续界实现，未将本次范围扩成follower物理性能全面审计。149passed日志实际位于qualification/runtime，而非research run根目录的runtime；已读取正确文件。日志记录149 passed in21.90s，本审查没有声称自行重跑149项。

## 误差定义与安全余量

execution.py:228–247在实际stamp计算`norm(actual_xy−program.at(stamp).xy)`。未搜索最近路径点、重配相位、忽略along-track误差或使用未来时间来缩小误差。对任意时刻，若两机各有水平误差≤10，则实际水平距离至少为名义距离−20；35m名义间距保留15m实际水平间距，且3D距离不会小于水平距离。若名义|x|、|y|≤540，欧氏水平误差≤10保证实际各轴≤550。

这是**以实际误差界成立为前提**的余量关系，不能反过来证明controller已达到10m。合同仍要求逐240Hz实际hard audit和tube audit；本次没有用纸面35/540替代真实执行结果。

单一共享常量planning.py:21 `TRACKING_TUBE_M=10`：

- planning.py:354–355对完整fleet准入使用`15+2*TRACKING_TUBE_M`及`550−TRACKING_TUBE_M`。
- execution.py:23、70导入并保存同一常量；:232逐step违例以`>xy_tube_m`计数；:268 capture采用`≤xy_tube_m`。
- mission.py:453–458将同一executor值写入trace.xy_tube_m，保留原17列误差/Z/interval、raw SHA和peak witness。
- qualifier:83–93核对合同tube、35/540配套值；:111把合同tube显式传入recount。短检查确认当前v18合同被接受，把tube单独改回6即拒绝。

optical_rank在planning.py:177同样加10m自身误差余量，仍只是导航排序启发式；真实frame/ray、measurement及评分不因此成为保证或被补造。工程tube从6放宽到10是明确的前瞻开发包线重校准，不应表述为工程门槛完全没变；D≥.8、180秒60格、质量与hard envelope没有随之放宽。

## ALL modes、完整占用与连续界

planning.py:269–320分别产生exit、survey、inspect/observe的work+完整return+periodic tail。各分支均经过同一个plan循环:335–360；candidate替换单机后与其他三机**已承诺的完整program**一起传给fleet_clearance。没有某模式只检查work、退出路径不检查或漏掉旧占用的特殊放行。前面的538m/544m离散预筛不会绕过最终35/540完整共同门。

paths.py:312–333逐条验证所有prefix与tail的实际线性插值位置斜率≤27m/s（含浮点容差），并拒绝不同period的尾环；从t=0查到max(prefix duration)+共同period，因此覆盖有限前缀及进入共同周期后的所有相位。实际nominal查询与vectorized xy_at使用相同时间插值/周期映射，RouteProgram要求prefix/tail及周期端点闭合。

sample_dt=.1时最近采样点距离不超过.05秒。两轨迹相对速度上界54m/s，所以距离下界从sample minimum扣2.7m；单轴速度上界27m/s，所以围栏上界加1.35m。最终35/540比较的是已经补偿的continuous字段，并非直接比较sample minimum/maximum。数学上是连续名义插值轨迹的保守界（数值容差范围内），不声称物理连续时间误差已获认证；实际物理执行按合同240Hz检查。

初始化program先于任何job proposal。除检查共同代码外，额外纯几何计算了roots800…819对应全部11种不同风（一次无风、10个奇数root的2m/s风），每种四机完整初始prefix+tail均满足35/540：最小continuous separation lower bound197.3m，最大continuous fence upper bound381.398808m。另检查2m/s东西/南北轴向风；没有运行physics。此检查只覆盖拟运行根的初始几何，不是40个任务执行替代品。

## capture、seam及240Hz记录

execution.py:249–275只改变声明的XY门；Z interval、heading/bank/course误差、ground speed、实测airspeed、pitch、vertical speed、body rates和积分器限制仍在。commit使用该_capture，已执行seam也使用同一个_capture并增加seam_capture_failures（:422–428）；qualifier:143–144要求XY/Z与seam违规计数均为0，没有事后按新摘要略去seam失败。

mission.py:69–76保留实际physics积分后的hard audit→getBasePosition→tracking audit→实际距离累计链。240Hz行中的XY、Z和interval来自同一个物理stamp，trace与summary计数没有改成10Hz或控制频率。recount检查raw SHA/字节数/schema、有限值和严格1/240时间格，逐world步对四机任一越界计一次，非把单机次数混加。

本地短检查使用实际audit_tracking：同刻XY偏移(6,8)恰为10时不越界，再增加.001m后记一次越界；没有运行积分。现有capture test对9m接受、10.001m拒绝，并仍走其他capture域。压缩记录仍不是完整physical-state replay；本次只审其定义、绑定与短检查，新v18逐步数据尚不存在，不能预先重计全40。

## legacy固定6m，旧失败保持失败

inspection_tracking.py:20–25对缺xy_tube_m的历史trace固定使用6；新显式阈值必须为有限正number，并与调用方expected_xy_tube_m完全相同。不会因为当前常量变10而把旧计数按10重算；qualifier.coverage_valid:111以新合同10调用，旧记录拒绝进入新资格。

v17 manifest完整40 completed，source/deps/contract unchanged均true，observe=.85、transfer=1、qualified=false。source/finite-arcs-v17中15份实际snapshot与manifest逐字节相符，40份slot实际SHA重新核对全部匹配；没有用当前v18源替换旧snapshot。

对三份真实v17 observe失败用当前legacy兼容recount重算，保持以下原6m失败：

| root | 原最大XY(m) | 原6m违规world步 | expected10 |
| --- | --- | --- | --- |
| 604 | 6.868807388 | 535 | 拒绝 |
| 612 | 7.630974876 | 741 | 拒绝 |
| 619 | 6.966670457 | 839 | 拒绝 |

三份新合同coverage_valid均false，未修改其内容或passed。即使某旧记录的XY峰值小于10，也不能据此升级：旧路径准入/合同按6m余量，不能自动满足新的35m名义分离前提。v16/v17失败仍按其各自合同保留，v18只能用前瞻新网格取证。

## 基础100组成证据复用复验

已重新核对g1-full.v4全部100个实际文件SHA、manifest直接指针与v18.base_control_qualification_basis的SHA一致。三份实际功能consumer仍逐字节匹配原snapshot：

| consumer | SHA256 |
| --- | --- |
| envs/inspection/control.py | c676d237efe1b6227d238b4e5948c54a5725626e9d3f69e74ed9c1629a235270 |
| envs/inspection/flight.py | ef9be915c92c207a3e8f43b9ddc1907010beea59be6d96136330e82a5d67cbdb |
| eval/qualify_inspection_g1.py | 852ea8c02cade269b48ea1bf99158913ddaef2038f6a368bd0987c31cc794522 |

v18.G1与原manifest.contract.G1整个对象相等；gains含pitch_trim−.07、nominal17、所有五族时长/roots/包线/追踪/退出阈值相同。61项外部依赖实际重hash匹配；PyFlyt0.29.0、pybullet3.2.7、numpy1.26.4、numba0.68.1元数据匹配。envs/__init__.py仍为空、inspection/__init__.py仍仅docstring；基础runner没有import planner/execution/mission或eval package initializer（当前eval为namespace，无__init__.py）。

因此当前可以引用原100基础slot及g1-v16-full-review.v1.md的已执行审查，planner/paths/execution改变不自动要求重跑100。复用只覆盖原五族同输入行为；保留旧manifest/原合同身份，在新结果显式标为组成证据。不得把旧qualified=true回填成v18整体qualified，也不替代新执行器40个任务。若以后三consumer、完整G1、依赖/初始化行为变化，应重新判断复用前提，不预先批准未来改动。

## 实际验证与未证范围

现有针对测试命令：

```text
.venv/bin/python -m pytest -q tests/test_inspection_execution.py::test_every_job_kind_requires_the_enlarged_complete_fleet_clearance tests/test_inspection_execution.py::test_capture_uses_declared10m_boundary_without_changing_other_domains tests/test_inspection_qualification_analysis.py::test_trace_threshold_is_explicit_and_legacy_data_stays_under6m
```

结果6 passed in0.74s（四mode参数化+capture+legacy）。四mode测试实际用31m完整geometry反例，能捕获回退旧27m门；额外短检查对共同gate的35/540精确边界与34.999/540.001反例逐模式测试。后者通过隔离candidate builder验证共享准入分支，不能替代各builder的物理运行。

独立只读短检查脚本`/tmp/check_tracking_envelope_v18.py`，SHA `516be0ae64c69613e9969a8ce04a42975e0d5ed4b868f0723dbe80e924af4e45`，退出0；日志`/tmp/check_tracking_envelope_v18.log`。另根列表初始几何检查日志`/tmp/check_v18_initial_roots.log`。临时脚本/日志不是新资格结果，未写入原source/结果目录。

reviewed by inherited model configuration, independent context, same model family

- 可前瞻执行v18 roots800…819的两族完整40任务；全部分母、异常、逐步审计及失败需保留，之后仍需独立结果审查。当前没有v18 full结果，不宣布G1或平台通过。
- 10m是工程假设的前瞻重校准，35/540仅在实际误差界成立时保护硬几何；没有把旧6m失败重分类。校准文档已追加澄清历史和当前状态，P3关闭。
- 没有重跑149项或physics矩阵；已执行6项目标guards及上述短数值/字节检查。61依赖+版本是当前保存的闭包范围，不是完整机器环境锁定。
- 控制root与固定fixture不是独立主研究场景；controlled nexttick仍只涉及G1控制/因果可解性，不证明真实异步1×、模型延迟或G3。G2/G3/G4与人类领域复核均未授予，human_review_pending。

本报告完成后冻结；下一资格结果另立记录，不回填本报告的前瞻执行判断。
