# 新场景适配器独立工程审查 v1

日期：2026-11-05。审查者 `/root/mission_review`，同模型家族独立上下文，不是人类领域或跨模型审查。工作树 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。本次按 poteto-mode 的只读调查与 good-benchmarks 的纯审计范围工作，仅新增本报告及附属 `scenario-adapter-review.v1.evidence/`，临时文件在 `/tmp/scenario-adapter-review/`。没有修改实现、控制器、现有输入或旧结果，没有运行新物理世界、长episode、API或重负载检查。父任务的L1真实1×期间，只执行了获准的低负载输入/评分探针。

## 结论

**当前版本不应据此宣布G4 GO。** L1评分分母/门控与L2/L3继承路径的有界正例通过，未发现公开任务卡泄露未来日历或隐藏真值的直接路径；24个冻结dev输入也都与manifest绑定一致。但五类可复现缺口仍需修复后另版本复核：采样轨迹检查可被混叠绕过、重叠动态需求获准、实体数值/身份类型不严、合法共线分离边误拒，以及冻结case副本没有传递/结束核验。

这些反例针对公开的 `CaseSpec.from_dict/validate_case` 与 `--case` 信任边界，不表示当前24个自产dev case已经受污染或全部不可解。报告保留旧SHA和原输入，修复应重新验收这24份原文件，不能因新校验失败而按成绩静默重抽或回填旧manifest。

父任务已将本次三个实现文件和tests原样归档至 `qualification/source/scenario-adapter-v1/`，本次核对binding中的四个SHA与原文件一致。下列命令实际在修复前执行；修复后若需重现旧行为，应把旧snapshot复制到临时独立eval包并优先加入sys.path，复用未变envs core和原dev输入，禁止覆盖仓库源码。不能直接运行新实现，再把不同结果解释成原v1行为。

父任务后来完成L1-00-full真实600s运行。本次只核对其完成状态、输入副本/world哈希及保存分数，不重新审计全步控制/分数。它支持该旧版L1单例实际执行，不消除以下输入验证反例，也不替代G4的数据/方法/可解性与人类审阅要求。human_review_pending。

## F1 [P1] 601时刻检查可以接受实际出ROI的轨迹

位置：`eval/inspection_cases.py:34–42,164–179`。只约束radius>0、speed≤1.6和|bend|≤1，随后每entity取601个等距时刻检查中心点。半径没有开发包线下界；601个采样没有连续边界余量，也没有以速度上界排除采样间越界。

反例 `admission.json -> roi_aliasing` 从冻结dev-L2-00构造。令appearance=120、speed=1.5、radius=`1.5*.8/(2*pi)`≈.1909859、bend=0，把origin放在ROI第一条边内侧.05m，heading朝外法向。601检查的间隔恰为.8s，所有采样相位都是整圈，中心全部inside，`validate_case`返回valid=true。实际t=120.2时，position=(-21.728652875441597,-364.6112587956407)，独立in_polygon判定outside。

这不是“601样本不足以证明解析分离”的抽象提醒，而是被当前入参范围接受的具体越界。生成器自产radius60..85/heading±.12/bend0不生成这条反例，因此不将其算入冻结24集失败。修复可以先严格限制为声明的实际dev轨迹族，再使用解析包络或带速度/曲率上界的保守连续余量；只增加采样数而无连续界仍不能一般保证。

## F2 [P1] 动态代理完全共轨仍被当作有效需求

位置：`eval/inspection_cases.py:154–187`，尤其180行只对static两两检查45m间距。动态与动态、动态与static没有对应退化/重合检查。

反例 `admission.json -> coincident_ten_dynamic` 保留dev-L2-00的static实体，把一条已合法动态轨迹复制成10个不同key的完全同轨、同位置、同类代理。校验通过，capacity_bound输出static=3、dynamic=5、combined=8≤budget10。

这10个需求的几何/可见语义退化。生产 `_visible` 的一条ray只返回一个hit body，完全重叠的代理不是10个独立可辨识观测；固定首次关联与隐藏实体身份不能靠“完美报告数足够”恢复。该探针没有运行physics或量化它的最终D上界，不把未测的最高得分写成实测，也不声称当前24例存在这种完全重合。

`capacity_bound` 自身已明确标注necessary bound，不应因此给它“物理可解”责任；缺口在于整体validator仍将退化需求标valid。至少应拒绝永久同轨/同点目标和不允许的proxy重合，明确动态交叉的身份/间距规范；正常短暂交叉与永久不可辨识重复不能混为一类。任意新增最小距离门槛都须按同一规则重新核验原24输入，不能按策略表现删case。

## F3 [P2] JSON finite检查不等于实体数值与身份类型校验

位置：`eval/inspection_cases.py:127–132,154–174`。wind/ROI/seed有显式type检查，实体origin、radius、speed、heading、bend和case/lineage标识没有同等检查。`json.dumps(...allow_nan=False)`能拒绝NaN/Inf，不能拒绝布尔值或错误标识类型。

反例 `admission.json -> boolean_numeric_and_nonstring_identity` 使用case_id=true、lineage_group=123，把一个static实体origin设为(true,false)、radius=true、heading=false、bend=true。当前仍返回valid=true，canonical digest也正常生成。Python把这些布尔值按1/0计算，掩盖了格式错误。

此处应按schema拒绝bool数值与非string标识，检查每个实体向量维度、标量有限性、范围，以及字典/序列结构；不能仅依赖后续算术异常。此反例不是当前生成器输出，也不是已观察到分数错误，但它直接否定“from_dict拒绝非法case”的完整保证。

## F4 [P2] 相交判断把分离共线边误判为相交

位置：`eval/inspection_cases.py:141–149`。orientation乘积≤0的判断缺少segment bounding-box/on-segment条件。当两段共线但不重叠时，两组orientation均0，因而误拒。

反例 `admission.json -> valid_disjoint_collinear_polygon` 使用九个实际顶点：

```text
(-400,-400),(400,-400),(400,400),(100,400),(100,200),
(-100,200),(-100,400),(-400,400),(-380,0)
```

这是有顶端凹口的简单多边形；非相邻顶边两段共线但分离。本次独立带bbox的segment检查得到非相邻交点0，grid有1480格，原case所有实体均在该polygon内。当前validator在任何实体检查之前就报 `ROI edges intersect`。不是用自交ROI要求放行，也不是把多个冗余共线点伪装成九个拐角。

修复应保留真实交叉、重叠、接触退化的拒绝，用完整段相交判定接受这一合法替代。仅删intersection检查会使既有crossed_roi反例失效。

## F5 [P1] 冻结副本的初始hash没有约束worker实际读入内容

位置：`eval/run_inspection_reference.py:38–41,139–143,162–171,228–244`。主进程把原文件复制为out/private-case.json，记录该副本file SHA，然后只把路径传给worker。worker稍后重新按路径读取；没有expected hash参数或读入后的原始字节核验，结束只校验source/dependency不变，没有case副本不变项。

反例 `score-and-freeze.json -> freeze_mutation` 调用实际worker入口，但将所有flight/Mission对象替换为内存双替身，**0实际物理步**。复制dev-L1-00并记初始SHA后，把副本seed加1，再进入worker。worker仍status=completed，实际使用的新case digest与原值不同：

| 项目 | SHA-256 |
| --- | --- |
| 初始副本file SHA | `5600888c8cc583edadbb1e1d43f8f0dcd928b9b4e44c2dc93296a1788c62b422` |
| worker实际读入副本file SHA | `faf4711b8cbfc202e7f4b9e63906cf9ae60a8e770b190b623c2790a71e50eb10` |
| 原canonical case digest | `7dea477c5234ef3e22bf576a9f610c3941c3b09efe4c1db4bb6a7d9574db0e95` |
| 实际使用canonical digest | `c5e78cb2a56c25a05f9b76de09f26b119760c950de95936b89d8573d8e677229` |

这是输入反序列化/绑定的语义反例，不是一次真实CLI运行或科学样本。它表明现有记录会暴露事后的case/hash不一致，但执行入口没有把不一致判失败；仅凭sources_unchanged/dependencies_unchanged不足以保证方法使用同一输入。也可能发生worker读完后副本被改、结果case正确但归档副本错误的另一方向。

应将初始期望字节hash传给worker并在读入时比对，记录验证后的canonical digest，并在结束验证冻结副本未变；不一致应明确记录输入绑定失败。原始 `--case` 来源在复制完成后改变不应污染已冻结副本，不必要求它之后永远不变。还应在资格/配对分析中join manifest case_id/digest/file SHA、private copy和world.private_case，而非只按文件名配对。

本次实际L1/L3短诊断与L1完整诊断的copied file SHA均与run记录一致，world文件SHA也一致，没有据此声称它们已经受到该问题影响。

## 已通过与有界复用

### 信息边界与同步

代码审查没有发现新未来信息进入policy的直接路径。CaseSpec只在worker中反序列化；主进程复制原始字节但不给policy构造函数传case。继承的public_state按显式白名单组装任务卡、当前/已收到telemetry、job cache、delivered observations等，adapter没有把 `self.case`、private_case、seed、hidden entities或未来calendar放进它。最终business_audit包含private_case/case_sha属于评测侧产物，不是actor update。这里是受控policy/RPC边界，不是对任意能读本机文件的恶意Python代码的沙箱保证。

public card只替换level/ROI/grid；L1删dynamic rule并将dynamic_requirement=None，事件priority只在 `_events` 触发后升级卡版本。online/camera_quality的半开区间正确，非L3字段全None时按drone整数短路，不产生默认事件。已有测试名字 `test_calendar_boundaries_and_no_future_event_in_card` 实际只断言link/health边界，没有真正检查card/public_state；不能用该测试名代替上述源码边界核验，后续宜补实际公开快照断言。

`ScenarioMission.__init__` 在父构造未积分前重置/复用地面proxy，不重置飞机；多余旧proxy移走并关闭碰撞，新proxy注册。父before_physics仍调用当前entities的position，因而CaseEntity的heading/radius/bend运动进入实际同刻proxy更新。wind_xy传入FlightWorld，事件通过virtual online/camera_quality/_events/weight接入；父积分后再次执行_events，再进入_sample，所以v19 retired guard仍阻止退休机新增mission采样，旧pending交付逻辑未被adapter覆盖。

此判断是调用链检查，不是所有随机风/新轨迹/随机退休机的实际控制资格。尤其L1没有退休，L3的link/camera/retire对象和时刻已改变，旧固定机号180/210/300/360参考不能代替这些新场景的完整事件实证。

### 评分继承

执行现有14项cases tests，退出0，14 passed in .37s。另用 `ScenarioMission.__new__`、真实ReportLedger和生产grade做0物理步手算fixture，三等级均填入合法完美证据，C/S/Q/Y=1、H/E/SR=1，与独立分母计算一致：

| 等级00 fixture | required static | required dynamic | static权重分母 | dynamic权重分母 | D |
| --- | ---: | ---: | ---: | ---: | --- |
| L1 | 4 | 0 | 4 | 0 | null |
| L2 | 5 | 3 | 5 | 180 | 1 |
| L3 | 5 | 3 | 9 | 331 | 1 |

L1 withdraw一个报告后S=.75、静态分母仍4；normals不加入任务分母，未观测目标没有被删。L1 H违规会使Y=0，缺退出dwell或不足航程余量使E=0。L2/L3经super.grade继承原full60格动态选择，并通过override weight正确读取私有calendar；即使父events=False也不会误用旧固定180s权重。L1 Q=(C+S)/2、SR不含D，与原计划等级约定一致；不能跨等级拿Q/Y直接排名。

这些是裁判/聚合正反例，完美证据由评测侧构造，不是策略获得了100%质量或可解性实证。原protocol/scoring core未因本adapter改变，按对应SHA复用旧账本/评分审查；新sampler几何、风、事件组合与策略实效仍需单独验证。

## 冻结dev集合与比较范围

`qualification/dev-cases.v1/manifest.json` SHA=`b5a00d2cf88549dcabce17667825fa8e1f171052516b853cb3095f3f218ee6b9`。24条记录均有文件，file SHA、canonical digest和重算validation与记录一致；level各8根，case/lineage命名不同；first_pilot正好L1/L2/L3各00/01六例。source_sha与当前旧版generator相同；main_and_holdout_instantiated=false。没有把main/holdout当作已形成，也没有按观察成绩挑first_pilot。

24例的necessary combined perfect reports bound为2..6，均≤10；其意义仅是名额未在完美报告假设下先天超限。它没有检查C可达性、传感器遮挡/身份、交通/返回余量、动态窗口、完整任务调度和具体方法可解。所有case仍来自同一基础九边ROI的较小尺度/旋转/顶点抖动、同一固定stripe执行器与自生成对象族；seed/digest不同不支持新的道路/地形家族或OOD覆盖主张。L1/L2/L3为分别生成的dev实例，不是每等级共享同一布局的配对消融。

实际文件检查确认L1-00-short和L3-00-short均completed/12s、E=0/Y=0；只是适配器短物理诊断。后来L1-00-full已completed/600s，保存C=.9940374787052811、S=1、D=null、H=E=SR=1、Y=.9970187393526406，输入副本与world hash匹配。本报告未独立重算该完整物理run，因此仅把它作为已存在的旧版本单例，不作为五项修复后的结果或G4 GO。

`--case`的B0/B1/B2走同一个ScenarioMission和预算，策略构造参数没有私有case。公平比较仍须保证同一case digest、环境/策略source、预算、延迟规则、配对/重试和完整分母；目前只有dev预选，无至少两方法的完整分数差异或B_classic*选择证据。

另有明确G4集成待办：当前 `replay_inspection_reference.py:125` 仍无case分支，按run.seed/layout/events构造基类InspectionMission；新case run将seed/events记为None、layout=private_frozen_dev_case，不能直接当旧reference重放。旧analyze的退化sensor分类也仍按固定机2/300..345与run.events判断，不能直接用于新私有calendar。它们是旧参考资格工具的使用边界，本报告未修改或重做控制审查；在宣称新case可冻结复算之前须有匹配适配、输入绑定和相应证据。

## 命令、反例与源码绑定

实际命令均在仓库根执行，没有启动新physics：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/scenario-adapter-review/admission.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_cases.py -q -p no:cacheprovider --basetemp /tmp/scenario-adapter-review/pytest
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/scenario-adapter-review/score-and-freeze.py
```

三者退出0，耗时约.49/.37/.16s，分别为输入反例/24集绑定、14现有tests、评分与worker双替身检查。反例“接受”是被审查缺陷的实测，不是预期正确行为。长期证据保存在本报告同目录 `scenario-adapter-review.v1.evidence/`。未来复现先创建 `/tmp/scenario-adapter-review/`，以仓库根为cwd运行附属两脚本；默认仍把小结果写到该临时目录，不改现有输入。

| 证据文件 | SHA-256 |
| --- | --- |
| scenario-adapter-review.v1.evidence/admission.py | `f4b6089c4b00e3d8ca81ff2052c46b05b5bc2fb1667a281220e9367d76140dda` |
| scenario-adapter-review.v1.evidence/admission.json | `85f6ccd574be47137ef28c1b5c52ba4aff9bc1372cf57f0e3ebec459cc5ff88c` |
| scenario-adapter-review.v1.evidence/score-and-freeze.py | `28431d32b642d995be5e304d34221d191972a64541a64745088be26bfec7ece2` |
| scenario-adapter-review.v1.evidence/score-and-freeze.json | `0c7be19b93bbe08c2570e7f787fda3959d4804b5dd2192339e899c2b4c3053b4` |
| scenario-adapter-review.v1.evidence/tests.log | `35bd51c20293720137c63a616139b2b056a80d5c3a397cfc6525fa67feaab8bb` |

以下为本次读取/执行的旧源码SHA，不因主任务后续修复而回填。HEAD=`270c6f0946b3be59527b2b77982f75a70da7ef77`，新增inspection代码处于未提交工作树，实际内容由SHA决定。

| 源文件 | SHA-256 |
| --- | --- |
| eval/inspection_cases.py | `836f9504c131fedcc00c8199b07db2a89bb1380efe4717768d854024a29c552d` |
| eval/inspection_scenarios.py | `d2b0fbfc77ee40e06115393624a3ac2f36d11fa21981590c71f3bc0beea5dcec` |
| eval/run_inspection_reference.py | `8b2aa155445d15ad736d972d42b09c210b07dacbd439e52627161837c43e1f75` |
| tests/test_inspection_cases.py | `6b30772120c3f1038c22ea9d52dd7a89803126b15fb5abcbfcf3e00e00ea8015` |
| envs/inspection/mission.py | `715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7` |
| envs/inspection/scoring.py | `bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a` |

本审查完成的是旧版本的缺陷定位、正例边界和可复现证据。主任务接受修复后，须另版本核对差异、重放本反例与正常替代、重新验收原24输入，并区分未变core可复用范围和新场景实际qualification；本v1不能被改写成修复后通过。
