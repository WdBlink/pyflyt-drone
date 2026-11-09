# PyFlyt Drone：民用机群巡检Agent Benchmark实验与论文计划 · 项目交接

生成时间：2026-10-02T12:38:35.340632+08:00  
版本：HEAD 270c6f0946b3be59527b2b77982f75a70da7ef77; branch codex/paper-privileged-vision; original dirty .gitignore, untracked eval/paperstudio\_rta\_experiment.py, ideaspark\_run/, paper/protocol/rta-experiment-runtime.json, research-brief.md, tests/test\_paperstudio\_rta\_experiment.py all preserved; new .project-handoff/ only for current research. Existing ctx symlink external. 本次新增 .benchmark-design/20261002-competition/，PROJECT\_HANDOFF.md保留不改；最终另存BENCHMARK\_RESEARCH\_PLAN.md。

这份规则很适合作为任务机制的参考，但要形成有研究意义的 benchmark，必须把“比赛得分”转译为“巡检信息是否按时、可信、安全地交付”。建议以**灾后道路与河道快速核查**为主场景：中央任务 Agent 调度固定翼机群完成区域观测、疑似异常递进确认、授权巡检车辆或测量船的短窗口观察与交接，并在任务变化后改派。比较方法共享合格飞控、因果感知和执行工具；模型推理、通信等待与工具执行期间，飞机和目标继续运动。该设计可以检验高层规划、编排和动态重规划，而不会把舵面控制、感知器差异或暂停仿真产生的优势混入主要结论。[S32][S33]

本次交付的是完整研究与论文方案，尚不是已运行 benchmark 或实验结果。规则、现有资产和近邻文献共同限定了方案：赛事动态计分和结束条款需要独立重写；MultiUAV-Plat、MissionBench、ESARBench 已覆盖相邻问题，不能宣称首次或预写 LLM 优势；当前 PyFlyt 资产尚未通过任务级控制、信息隔离和裁判资格。下面的任务参数、阈值、样本数和时间估计均为**待开发校准并冻结的建议**，不能当成现有平台能力、用户已批准规范或已取得成绩。[S17][S18][S19][S33]

## 1. 先确定业务交付，才能确定 Agent 应该决策什么

灾后核查人员面对的问题不是“飞机飞了多远”，而是在截止时间前能否知道：哪些指定路段与河段已经看清；哪些疑似阻断、漂浮障碍或设施异常值得现场复查；授权巡检车辆或测量船在指定时间窗口经过哪些通行段、何时失去观测。合成任务用异常代理物表达这些信息需求，不声称通过看见代理物就完成真实灾害诊断。这样既有具体民用意义，也允许建立可核验的隐藏真值。[S33]

一轮任务的交付包含观测覆盖账、带定位与类别证据的静态报告、带原始观测时戳的动态位置序列、失去观测与交接缺口记录，以及任务结束时的安全退出状态。任务卡只给公开区域、对象类别本体、优先级、名额和截止时间，不给未知实体位置、对象总表、未来轨迹或事件脚本。优先级由公开、版本化的类别/区域规则决定效用权重；规则如何作用于隐藏实体、更新何时生效和评分分母，分别在第 4 节与第 6 节确定。机群先搜索，发现不确定候选后决定继续覆盖、换视角确认或换机核查；对移动候选决定由谁观察、何时交接；在优先级或可用能力变化后重新分配。有限报告名额和固定动态窗口使“多看一眼”与“及时交付”形成真实取舍。[S33]

Agent 是中央 mission-level 编排器。每架飞机的航迹控制、局部感知跟踪和安全保护持续运行；Agent 只调用搜索、检查、观察、交接、取消和报告工具，不输出油门、舵面或 30 Hz 控制动作。共享工具可用经典覆盖规划、可飞的 Dubins 航迹与有限状态执行器，所有方法得到同样能力。工具负责执行已指定任务，不能替 Agent 决定全局优先级、未知目标搜索分工或跨机报告取舍；否则评价的是工具内置调度器。主实验采用一个中央 Agent 管理多机，分布式多 Agent 留作后续扩展。[S33]

区域三维建模原始数据采集可以作为同一业务的扩展轨道，但主实验先评价观测覆盖与证据质量。采集轨道至少要检查 GSD、前向/旁向重叠、视角、清晰度和多视图支持；只有加入复建管线与几何真值，并验证模型完整度、精度等，才能称为三维重建评测。QUARTZ 已用 mesh 完整度、精度和 F-score 检验质量驱动协同，建筑建模采集也已有重叠、相机姿态、作业时间等要求；地面覆盖率不能替代这些验收。[S25][S26][S33]

本研究据此提出三个问题：共享执行条件下，Agent 能否完成空间分工与可信信息交付；事件发生后，编排方式能否减少任务损失；真实推理和通信延迟如何改变质量、安全与成本。这些问题允许答案为“LLM 不如强经典策略”，而不是把成功预设为设计前提。[S33]

## 2. 赛事提供压力结构，benchmark 必须消除计分与终止歧义

科目一与本场景最接近：6 机在不超过 1 km² 的不规则区域内，于 25 min 内识别定位任务卡指定的 10 个静态与 6 个动态目标；离地高度为 40–120 m，允许遮挡、移动路线变化与间歇通信干扰。任务期间通过 JSON 上报，系统只读取前 16 个目标信息，同一物理对象重复条目仅取最优定位一次。裁判使用不向参赛系统发送的 RTK 真值，并保存飞控、影像与姿态时间戳审计。这些机制可直接启发区域搜索、有限信息预算、部分观测、变化响应和隐藏裁判设计。[S32]

| 原规则机制或数值 | 借鉴的研究作用 | 本方案的转译 |
| --- | --- | --- |
| 6 机、25 min、不规则区域 | 多机分工与截止时间压力 | 主实验 4 机，2/6 机作规模轨道；区域与时长按固定翼资格校准 |
| 静态与移动对象、遮挡、路线调整 | 递进核查、观察交接与重规划 | 合成异常代理、授权巡检代理、相似正常对象；未来状态对 Agent 隐藏 |
| 前 16 条、重复对象取最优 | 有限报告预算 | 首次创建消耗名额，同 ID 因果修订；重复新 ID 不获得“挑最好一次”的奖励 |
| 静态定位误差档 3/5/10 m | 精度与类别联合交付 | 保留档位作为候选，按传感器资格校准；同时报告连续误差与 precision/recall |
| 动态从首次上报起算、每至少 3 s 选点 | 连续观察要求 | 从环境出现时刻起算固定窗口，未发现、缺测和迟交格均为零 |
| 通信干扰、不暂停、不重启 | 在线决策与交付时效 | 断链机继续已批准 job，地面只接收已到达包，实际等待推进世界 |
| 自主返回及日志审计 | 完成条件与安全责任 | 本阶段独立定义空中安全退出，不宣称已实现返回落地；保留可复查 trace |

原动态定位分按 `8 × C × min(N/40,1)` 计算，误差档系数为 1、0.6、0.3、0，类别正确最多再加 2 分，总分 60；静态每对象定位为 3/2/1/0 分，取得定位分后类别正确再加 1 分，总分 40。它适用于比赛裁决，但不宜直接作为研究协议：180 s 按 3 s 划分是 60 格，40 点饱和无法表达全部缺测；从首次报告起算会把迟发现前的损失隐藏；重复取最优会鼓励试探多个版本；“首次目标条目”与后续动态点怎样占名额也需明确。原平均误差公式中 `Pi` 与 `RPi` 印成逗号，正文支持距离理解，但本研究不能替赛事声称符号已经严格定义。第 6 节给出独立、唯一的裁判语义。[S32][S33]

结束规则也不能照搬。原通则称结束返回少于 3 机整体为零；科目三表中 1/2 返回机又列有 5/10 完成分，适用优先级未明。任务中落地退出与正常终点降落的关系同样需要解释。原规则还有高度或边界累计 3 s 扣 5 分、晚返不超过 3 min 每机扣 3 分、超过 3 min 整体零分、飞出区域连续超过 10 s 或应急干预后退出等条款。benchmark 应独立区分硬安全事故、工具拒绝、外生能力退出和正常终态，不把上述歧义包装成赛事官方解释。[S32]

AGL 是飞机相对地形的高度，必须由当前位置高度扣地形高程计算，不能直接等于 `world-z`，也不能套用比赛场地的固定基准。赛事 7.9 m/s 风速是裁判可中断依据，不能成为当前固定翼已验证风包线。科目二仅借鉴对象价值与稀缺资源优先级；科目三仅借鉴已知区域中的未知标识、能力匹配与完成后退出。本方案不执行投送、低空密林穿越，也不把它们作为主论文的附加任务。[S32][S33]

材料使用边界来自已读原文第五部分第七条：未公开赛务信息的互联网传播受限，赛事采集影像、遥测、日志、通信协议等仅用于挑战赛。本次在本地分析泛化任务机制；未来公开论文和复现包只使用自生成地图、对象、日志及重画示意。原文对外发布状态和主办方解释未核验，不能把“文件在本地可读”理解为可以公开原规则、截图或真实赛务数据。[S32]

## 3. 近邻已经存在，差异应落在可核验的评测条件

“无人机 + LLM + 工具调用”不足以构成新颖性。最接近的三项工作已经覆盖多机任务规划、视觉任务与报告、递进搜索，论文必须逐机制比较，而不是以一个不同场景名称宣称首次。[S17][S18][S19]

| 工作与已读范围 | 已有任务/信息机制 | 动作、完成与指标 | 对本方案的具体约束与尚未知项 |
| --- | --- | --- | --- |
| MultiUAV-Plat：平台、任务、Agent4Drone 方法段落深读，结果浏览 | 75 会话、1,500 任务、9,396 隐藏检查；区域搜索、分配巡逻、目标分配，REST 权限与局部感知 | Agent4Drone 对比 ReAct；任务顺序执行并保留上一任务位置 | 多机规划与工具交互已有直接先例；固定翼气动约束、持续物理时钟、长 job 后验完成的细节未充分核验 |
| MissionBench：任务/评测与局限深读 | 120 任务、5 环境、4 任务族；RGB 输入；22 模型主评测使用 30 个 Test 任务 | UE5/Cosys-AirSim；方向加幅度航点，含倒退/侧移；观察、巡逻、操作、报告计分，部分报告采用 LLM judge | 报告交付已有评价；此类动作不能直接当固定翼可执行航迹；同步机群及连续等待语义在本次摘录中未充分核验 |
| ESARBench：设置、指标深读，baseline 部分阅读 | 4 类 GIS 地形、600 搜救任务；线索发现至受困者定位 | UE5/AirSim-Colosseum；成功、时间与报告质量，同视觉接口检测基线 | 递进搜索与信息获取已有近邻；天气/时段快照不能自动证明飞行中事件持续变化；多机交接与真实推理时钟仍需逐机制复核 |

这些数字和机制是文献阅读记录，不是本次复现实验。表中的“未充分核验”不等于“原工作没有”，也不能直接成为论文缺口。[S17][S18][S19]

补充工作说明为什么执行层与规划层必须分开：BEDI 的六种子能力形成动态具身任务链，但当前依据仅摘要与官方 README，全文 HTML 和期刊页未成功访问；APEX 让空间语义记忆、RL 决策与 grounding 异步运行，以避免 VLM 成为控制瓶颈；Web-of-Drones 将 LLM、MCP/W3C WoT 与 ArduPilot 结合，已有覆盖、编队等工具任务，并提示长工具“受理”与“完成”需要分别核验；TACOS 已有多机搜索和故障重分配，但其仿真实验采用全局实体信息、没有实现感知，不能拿来作为部分观测条件下的直接成绩对照；PEACE 的 typed 计划、确定执行器、有限重规划与时间戳记忆值得借鉴，其正文明确是初步验证而非定量 benchmark。[S20][S21][S22][S23][S24]

强经典对手同样有依据。QUARTZ 以重建质量驱动多机 next-best-view 和稀疏工作量重分配；固定翼协同覆盖已经考虑相机 FOV、重叠、转弯半径、高度与分工；遥感覆盖平台已有 DARP 等分区与路径优化；动态不确定环境中的异构机群任务分配也有经典先例。后二类及建筑建模工作的部分依据仅摘要或 metadata，足以说明方向存在，不足以声称已复现其详细算法或同条件性能。[S25][S26][S27][S28][S29]

因此，候选贡献是一个具体组合：民用信息交付任务、因果局部观测、固定翼同步持续执行、带期限的证据报告、强基线与可审计统计，以及由实验显示的能力边界。固定翼本身、异步工具或“Dynamic Workflow”名称均不作为单独新贡献。正式写稿前，近邻矩阵还应按任务、信息权限、同步机群、动作可执行性、job 完成语义、推理时钟、报告裁判和强基线逐格补证；仍无法区分时，论文收窄为连续固定翼巡检协议扩展或跨平台复核研究，不声称 first/SOTA。[S33]

## 4. 用一条参考 episode 把能力压力落到真实决策链

参考实例设为 4 架同型固定翼、约 800 × 800 m 的合成不规则区域、可转弯的外缘缓冲航带、总时长 600 s、AGL 60–100 m、标称巡航 20 m/s。飞机从互有安全间距的稳定空中状态出发，开始时已有安全盘旋或搜索指令；终点是各机进入独立安全退出走廊并通过预定义盘旋验收。这是空中任务交接结束，不是自动降落或真实续航证明。[S33]

上述尺寸和速度均需资格校准。最小转弯半径可先用 `R ≈ V²/(g tan φmax)` 估算，随后用带风实际航迹跟踪、转弯误差和机翼清距验证；不能用瞬移、原地悬停、倒退侧移或无限转弯率补足能力。若 600 s 或候选区域不足以形成可完成任务，应在开发阶段调整几何、时限或包线，保留真实物理约束。[S33]

参考生成器布置 6 个静态异常代理（3 类、每类 2 个）、3 个移动授权巡检代理，以及 6 个正常或无关相似对象；这些数量只供生成器与裁判使用。公开任务卡要求覆盖道路/河段 ROI，并按类别/区域规则搜索和交付，不公布实体总数或清单。第一报告名额 `B=10`，同 ID 合法修订和后续动态点不新增名额。已知的是类别本体与交付要求，未知的是具体实体坐标、出现时刻和未来轨迹。正式生成器必须改变图形、布局、对象数量和事件组合，不能让每轮都是九个相关对象的固定答案。[S33]

为补齐优先级到评分的连接，本方案新增如下独立协议候选：每条公开规则含 `rule_id`、静态/动态任务类型、类别集合、固定区域、多级优先级 `p`、交付要求与截止，以及 `task_card_version/effective_from`；候选 `p=1/2/3`、`w(p)=p`，数值在 dev 校准后冻结。生成器以类别和归属区域把每个相关隐藏实体唯一分配给一条规则；静态实体按固定位置，动态实体按出现位置确定归属，之后穿越区域不暗改其规则。类别/区域规则须互斥且覆盖全部相关实体，正常相似对象不继承目标权重；生成器校验唯一归属，Actor 只见规则和测量，不见这个映射、隐藏真值坐标或实体总数。[S33]

更新通过任务通道在 `τ` 发布给地面 mission 服务，同时公开递增版本与 `effective_from=τ`，所有方法在同一时刻可读；生效不等待 Agent 轮询或 ACK。主轨优先级事件只改变所声明规则的 `p`，不改变 ROI、本体、报告名额、静态截止 `T`、动态出现窗口/交付宽限或硬安全约束。优先级是软效用，不是强制“立刻取消全部旧任务”的命令；第 6 节按最终静态需求和逐格动态需求计分，第 9 节另区分改派是否落地与是否改善交付。未来版本与更新时刻仍对 Actor 隐藏；改变期限或任务范围属于另外的预注册轨道。[S33]

| 环境/裁判掌握的参考时间 | 任务世界中的变化 | Agent 的合法压力 |
| --- | --- | --- |
| 120、180、240 s | 三个移动代理依次出现，各有 180 s 要求窗口 | 及时发现并安排观察；首次上报不能延后窗口 |
| 180 s | 发布带版本与生效时刻的类别/区域紧急优先级更新 | 按同一公开权重规则平衡交付、已有确认、覆盖与退出余量 |
| 210–240 s | 一个局部链路中断 | 地面信息变旧；本机继续旧 job，恢复后处理积压包 |
| 300–345 s | 一机相机质量下降 | 从合法质量/健康信息识别能力变化，决定确认或交接 |
| 360 s | 一机退出任务执行，底层保持独立安全盘旋 | 释放其任务，改派剩余机，避免把外生退出当坠毁 |

这张时间表用于设计说明，不向 Agent 提供未来事件。正式生成时在开发范围随机事件时刻并留出 OOD 时序组合；任务也包含无扰动和单扰动对照，不能让模型只记住固定脚本。[S33]

三个任务等级沿同一业务递进：L1 为区域覆盖与静态搜索确认；L2 加入移动短窗口观察与交接；L3 再加入优先级、通信、感知退化和单机能力退出等事件。L0 纯覆盖只作控制资格与诊断，不包装为第四个业务创新。各等级分别汇总，不用不同难度的混合总分决定排名。[S33]

| 能力 | 受控操纵 | 可观察结果与归因边界 |
| --- | --- | --- |
| 任务理解 | 语义等价卡片、明确优先级更新 | 硬约束保真与按第 6 节权重的交付；软优先级不用命令服从率替代；另设自然语言到结构化卡片子任务，固定解析器能力不计给 Agent |
| 空间与资源规划 | 区域形状、机群规模、初态与负载变化 | 合格覆盖、负载、航程、退出余量；工具只提供可飞路线，不自动做全局分工 |
| 递进核查 | 遮挡、假候选、证据质量与名额变化 | 确认/暂缓取舍、precision、覆盖损失、报告名额浪费 |
| 时间与交接 | 固定窗口、包年龄、长转弯恢复 | 首发现延迟、有效时间格、最大交接缺口、身份混淆 |
| 动态重规划 | 同 root 无事件/有事件兄弟任务 | 按第 9 节判定改派落地、统一需求权重下的事件差值及更新规则的交付；未响应/未恢复与无需改派分开 |
| 工具和证据纪律 | version 冲突、过期包、拒绝、错误 ACK/过期 job fixture | 无证据声明、非法参数、无效重试、僵死等待、预算耗尽；命令已发出不等于任务恢复 |

非 LLM 方法得到同一个公开结构化任务卡；主实验把任务编排与语言解析分开。自然语言轨道另行检查解析后约束是否保真，避免通过只给经典方法人工完美任务、或把固定解析器成果算给 LLM 而改变比较对象。[S33]

## 5. 合法观测、异步 job 与连续时钟共同决定可用策略

权限分为三层。world/truth 层只服务物理和隐藏裁判；机载控制与传感器层只获得各机因果可见测量；地面 mission Agent 通过链路收到包，字段包含观测时间、到达时间、质量或协方差、`candidate_id` 与证据来源。Agent 不读取真值目标向量、仿真对象总表、`truth target_id`、测试种子、生成器状态、未来轨迹或裁判正确性。`candidate_id` 是观测系统产生的可引用句柄，不是隐藏物理对象 ID。旧目标环境中存在真值 `target_vector`，Flatten wrapper 仍拼接它，既有因果感知契约也明确要求 truth 与 actor 隔离；本方案须将该边界落实为接口检查，不能仅靠提示词禁止使用。[S3][S5][S6][S10][S33]

主实验采用共享结构化传感器：在实际位姿下通过视锥、ray 遮挡、距离及成像质量门槛产生不确定位置、速度、类别和误报，并保留 provenance。内部用 truth 生成测量是合成传感器实现，不称真实视觉。噪声、误报、误差、时延和质量门槛在 dev 校准后冻结，随机流以 `root/seed + time + sensor` 定义，不能按方法调用次数重新抽噪声。不同策略飞到不同位置自然看到不同内容，但同一物理测量条件下的扰动保持可比。无噪声理想几何包仅作上界诊断；RGB/固定检测器或 VLM 是独立视觉轨道，与结构化主表分开。当前 FastSAM 原型不自动等于合格多类别语义检测器或多机相机隔离。[S6][S12][S13][S33]

最小接口足以闭合业务，不需先建复杂 Agent 框架：[S33]

| 工具 | 合法职责与返回要求 |
| --- | --- |
| `get_mission_state()`、`list_observations(since)` | 只返回本方公开任务、健康、观测覆盖账及已到达包；已核查覆盖不得由隐藏 truth 代写 |
| `assign_survey(drone_id, region_id, priority, expected_version)` | 使用公开地图生成可飞搜索航迹，受理后返回 `job_id` |
| `inspect_candidate(drone_id, candidate_id, view_policy)` | 对已观察候选申请具名视角/确认；不能输入未发现的真值坐标 |
| `observe_track(drone_id, candidate_id, duration)`、`handoff_observation(from, to, candidate_id)` | 调度本机合法连续采样或交接，保留观察缺口与证据来源 |
| `cancel_job(job_id, expected_version)`、`enter_safe_exit(drone_id)` | 版本一致地取消互斥任务或申请安全退出；实际达到后才完成 |
| `submit_or_revise_report(report_id, type, estimate, evidence_ids)` | 接受 schema 与证据引用；只反馈结构/受理状态，不反馈裁判身份匹配、误差或类别正确性 |

返回 `accepted` 只是 ACK；job 的 `queued/running/succeeded/failed/cancelled` 由真实执行状态推进。`succeeded` 必须满足工具公开后验条件，例如所选视角已取得符合门槛的观测或飞机已到达退出状态，而不是命令已写入飞控。互斥任务由 version/状态保护，过期 `expected_version` 拒绝；拒绝不消耗报告名额，却消耗调用、计算与时间预算。任务/改派日志保留所服务的 `rule_id`、当时可见的 `task_card_version` 与实际生效证据，执行器 job 版本和任务卡版本分别记录。失败、取消、断链和任务重派均进入同一日志，避免模型以无限重试覆盖错误。[S22][S24][S33]

观察 job 可根据 Agent 选择的策略在机载自动采样和上传，无需每 3 s 再问一次 LLM。断链期间，机载继续旧航线和已批准 job，地面不能看到新包；恢复后补传保留真实观测时刻、到达时刻和年龄，不能把旧包标成新观察。任务 Agent 的得分来自选谁、何时开始、确认与交接、何时报告和改派，而不是底层自动采样的频率。[S33]

**主模式为真实连续 1× 时钟**：`sim_time` 对应单调墙钟，模型推理、API/网络等待和工具运行期间控制器与目标继续推进。物理积分、飞控、传感器、Agent 轮询分别记录频率；现有 base 的 120 分步基数与 Aviary 的 `physics_hz` 不能混称一个物理 Hz。若宿主无法维持 1×，按预先冻结的漂移/积压容限定位为基础设施问题，不能悄悄暂停目标、慢放世界后仍报告实时模型性能。正式运行尽量隔离资源，随机化方法顺序；并行 world 必须先测资源竞争。缓存模型回复只能用于带回放标记的审计与诊断。[S2][S4][S33]

另设受控延迟注入轨道，候选 0/2/5/10/20 s：仿真可以加速，但逻辑时钟增加等价等待，控制与目标同样推进。这个轨道回答固定延迟对策略的因果影响；主实时轨道另报告实际 p50/p95 推理与工具延迟、timeout、token 和成本，二者不合并为一个“模型速度”结论。候选每 episode 60 次调用与 60,000 token 先经 pilot 校准；工具调用预算一致，经典规划计算也计时。外部故障保留原 attempt，并依冻结策略作有限重试或失败，不能免费从头运行到成功。[S33]

## 6. 把真实交付唯一映射成分数，保留失败原因

评分先写成可测试裁判，再看模型结果。以下数值是设计候选，dev 校准后整体冻结；不得在 test 上选择对 LLM 有利的定位、窗口、成功或权重阈值。[S33]

### 6.1 覆盖、名额与身份关联

覆盖 `C ∈ [0,1]` 为指定 ROI 中合格观测 cell 的面积比例。合格采样必须有无遮挡、分辨率/俯视角/清晰度达标及真实时间戳证据，飞过不等于看见。额外记录超过任务要求的重复采样占比 `R`，用于解释覆盖效率；合理多视图确认不无条件扣分。该 `C` 不表示三维模型完整度或真实灾害诊断质量。[S33]

首次通过 schema 的新报告创建即消耗不可退还的一格，撤回不释放名额；同 `report_id` 合法修订不再消耗，但每次估计必须绑定真实证据，不能修改证据采集时间。重复新 ID 继续占名额并计入重复/虚警。格式拒绝不占名额，格式正确却无有效证据的声明仍可能零分；裁判不给真伪反馈。[S33]

隐藏对象关联在首次创建时，用声明所绑定的观测时刻与位置估计进行一对一空间匹配，候选门限 20 m；按接收顺序处理，最近距离优先，等距按隐藏裁判 ID 稳定决胜，类别不参与匹配，truth ID 不给 Agent。已关联对象不再分给新报告 ID；关联普通对象不会成为相关目标；未匹配报告不能以后用未来 truth 补救；同 ID 修订可以改估计，却不能换成另一实体。静态 dev 布局先保证对象间距大于两倍门限，动态交叉轨迹另作关联资格检查。若裁判匹配存在歧义，报告歧义率并在 dev 修订，不把裁判身份不可靠静默归因于 Agent。[S33]

### 6.2 静态与动态证据交付

静态目标 `j` 使用截止 `T` 前最后一个已到达合法版本，平面定位误差为 `e_j`。候选精度函数为：`q(e)=1`（`e≤3 m`）、`2/3`（`3<e≤5 m`）、`1/3`（`5<e≤10 m`）、`0`（`e>10 m`）。目标得分 `s_j=q(e_j)×I(类别正确且证据有效)`；未报告或不可匹配为零。令 `J_S` 为生成器定义的全部相关静态实体，`w_j(T)` 为该实体所归属规则在截止时的最终公开权重，则 `S=Σ_{j∈J_S}w_j(T)s_j / Σ_{j∈J_S}w_j(T)`。分母包含全部未发现、未报告和零分实体，不能只对已发现目标归一。档位借鉴赛事而非假定当前传感器可达；另报连续误差、召回、precision 与类别准确率，防止分档掩盖误差结构。[S32][S33]

静态 `S` 评价截止时的最终业务需求，不是历时积分：优先级在 `τ` 更新后，最终权重也作用于 `τ` 之前已交付的静态报告；已有合格证据可以满足新需求，不强制重复观测。这里明确采用最终需求重加权，不将之前的报告另计一次，也不改其内容、观测时戳或到达时刻。主轨截止 `T` 不随优先级缩短；生成器须留出已资格通过的响应余量，不能在截止瞬间发布不可响应的更新。静态提前交付与更新后新增质量另作时间诊断，不声称主 `S` 测量了反应速度。[S33]

动态目标由环境定义出现时刻 `a_j`，候选窗口 180 s，固定 60 个 3 s 时间格，格端点 `t_b=a_j+3b`，`b=1…60`。第 1 格为 `[a_j,a_j+3]`，其余为 `(t_b−3,t_b]`；格端采样归前一格，因而一个原始新测量最多支持一格。主生成器须保证完整窗口和最后 6 s 交付宽限位于 episode 截止 `T` 前；主轨不截断分母，不在运行后删除困难格。若开发确需短窗口，须在另一协议中显式冻结格数与分母后再测。[S33]

每个新测量 `m` 保留原始证据 ID、真实观测时刻 `τ_m`、针对该时刻的平面位置估计 `p̂_m(τ_m)`、类别及身份声明、provenance 和完整报告/数据可供地面裁判审计的到达时刻 `r_m`。估计使用的证据必须在 `τ_m` 或之前采集；动态点在生成时封存估计与类别，之后上传不改时戳或值，合法修订只影响后续新测量。对选中的测量，`e_jb=‖p̂_m^{xy}(τ_m)−p_j^{truth,xy}(τ_m)‖₂`，不与格端 `t_b` 的真值比较。移动造成的包年龄、交付延迟和格端预测误差分别诊断，不混入观测时刻定位误差；预测/插值不算新观察。这是参考赛事“同时间真值”并独立补全的研究协议，不声称其余筛选规则来自赛事原文。[S32][S33]

主轨 `D_delivered` 对每个 `j,b` **先筛选，再排序**：候选须为该格的新测量，schema/provenance 和因果证据合法，关联沿用第 6.1 节的固定报告身份，且完整报告与所需证据到达满足 `r_m≤t_b+6 s`。在此集合中取观测时刻最大的一个；同观测时刻按预先固定的证据 ID 升序决胜。真值身份正确性、类别正确性与误差不用于挑样；选定后才检验，身份/类别错误或证据不满足质量门槛则该格 `z_jb=0`，不退回较旧正确样本；否则 `z_jb=q(e_jb)`。集合为空也为零。较新样本迟到而较旧样本及时，就选及时旧样本；较新样本后来到达不能覆盖已经结算的格。工具可自动上传，无需每格一次模型调用。[S33]

令 `J_D` 为全部相关动态实体，`w_jb=w_j(t_b−3)` 取格起点已生效的最新任务卡版本。更新恰在格起点时作用于该格，发生在格内部时从下一个格生效；已开始/已结束格不追溯重加权，也不延长出现窗口。主 `D=D_delivered=Σ_{j∈J_D}Σ_{b=1}^{60}w_jb z_jb / Σ_{j∈J_D}Σ_{b=1}^{60}w_jb`。分母包含全部实体和全部要求格，未发现、缺测、迟交或错误身份的零格仍带原定权重。无优先级变化且所有目标等权时，才退化为各目标 `d_j=(1/60)Σ_b z_jb` 的平均；首次报告不改变起点。[S33]

`D_observed` 使用同一时间格、固定身份、误差时刻、封存估计和权重分母，只将候选到达条件改为任务结束前 `r_m≤T` 完成审计上传，再在自己的候选集合中选最新、同刻按证据 ID 决胜，并对选中样本作同样的身份/类别/质量检查。它表示机载取得且最终可审计的报告证据，不表示未留证据的全部视觉可见性。两轨可能选中不同测量：最终上传的较新测量可能更不准，故 `D_observed−D_delivered` 未必非负，也不能单独称纯通信损失。另报机载轨选中证据的及时到达比例、迟报率和各轨零格原因，区分采集缺失、错误证据与交付问题；H3 以 `D_delivered` 的受控变化为依据。6 s 宽限在 dev 依业务校准，迟到不倒填、预测不冒充采样的原则保持不变。[S33]

### 6.3 安全、终态与主 endpoint

`H` 是 episode 硬安全指标：发生碰撞、地形侵入/坠毁或预注册硬包线违规则为 0，否则为 1。检查在每个物理步进行并保留违规持续时间；数值抖动容限须冻结，不以比赛累计 3 s 的扣分规则允许危险。安全保护介入或命令被拒不自动算事故，单列介入次数、原因与依赖保护完成任务的比例。[S33]

`E` 为任务终态：截止前，所有非外生退出机已进入预定义安全退出状态，剩余航程预算满足退出余量；外生退出机需独立安全盘旋并释放任务。Agent 导致的失联或失败不获豁免。资源先用明确航程/时间预算，不能冒充真实电池；退出状态不表示完成落地。[S33]

L1 任务质量 `Q=(C+S)/2`；L2/L3 为 `Q=(C+S+D)/3`。主连续 endpoint 为 `Y=H×E×Q`。等权只是透明汇总约定，不是已校准真实业务效用；不跨等级直接比较一个混合排行榜。候选安全任务成功为 `SR=I(H=1,E=1,C≥0.8,S≥0.6,D≥0.5)`，L1 不含 `D`。阈值在资格与 dev 中冻结。由于 `H×E` 会产生大量零值，主表同时展示 `C/S/D/Q/H/E/Y/SR` 与分母，不用安全样本积分代替全程安全率。[S33]

诊断保留首有效报告时间、虚警、重复、有效观察占比、最大交接缺口、航程、任务退出比例、硬越界秒数、最小机间距、job/工具错误、保护器介入、token 与 API 成本。优先级更新单列受影响规则下的静态最终交付和更新后动态格质量，使用同一权重/分母；静态在更新时已满足的成绩与之后新增成绩分开，未发现实体仍在分母。主 `Y` 据此评价新需求下的加权交付，不能仅由高 `Y` 声称迅速响应或每次都应改派。主结论还需权重敏感性和成本—质量 Pareto，避免某个汇总权重掩盖低覆盖、高风险或昂贵频繁重规划。这些诊断主要用于解释，不全部升级为显著性发现。[S33]

## 7. 当前资产支持底座复用，还没有支持任务级资格

冻结快照为 `HEAD 270c6f0946b3be59527b2b77982f75a70da7ef77`、分支 `codex/paper-privileged-vision`。原有 `.gitignore` 改动及未跟踪 RTA runner、runtime 协议、相关测试、research brief、ideaspark 目录均保留；已有外置 ctx 与 `PROJECT_HANDOFF.md` 不在本次修改范围。资产和历史主要继承上一阶段 2026-10-02 取证，本阶段新增赛事全文审读与研究设计，不重新运行训练、episode 或项目测试。[S1][S33]

| 可复用资产 | 证据支持的范围 | 新 benchmark 尚需补齐 |
| --- | --- | --- |
| Aviary 多机初始化、setpoints 与步进 | 存在共享多机物理底座，默认 `physics_hz=240` | 同步任务世界、多机航迹执行、碰撞/间距/AGL/频率资格 |
| V-tail、ObjLock、Flatten 等任务环境 | 单机动作与观测链存在，历史有 reset/step、vector wrapper 与回调 smoke | 任务接口以单机为主；truth `target_vector` 仍存在，不能直接作为合法 Agent 观测 |
| 名为 YOLO 的感知封装 | 使用 FastSAM、视觉特征与相机 patch 原型 | 语义检测、质量/误报校准、EGL 条件与类级相机闭包的多环境隔离未验证 |
| 高层 PPO 命令原型 | 有 heading/altitude/airspeed 意图 | 构造器把 base env 传给 render 参数、写入字段与 `self.target` 不一致、调用缺失 `_get_obs`、6/4 维动作衔接问题，不能直接当合格航迹执行器 |
| 训练回调与并行 wrapper | 既有 smoke 支持当时回调/归一化/临时保存路径 | `SubprocVecEnv` 是独立训练环境，不是同步机群；快照未发现本地 models/logs，合格 checkpoint 未知 |

这些边界由静态调用链与既有记录支持，不是本轮 runtime 验证。[S2][S3][S4][S5][S6][S7][S8][S9][S11]

重要旧尝试必须据证据收窄。历史 OPC 记录确有 `compileall`、11 个 import、兼容 shim、A-to-B reset、vector wrapper、回调完整路径等成功；其中回调检查在临时小环境产生模型与评估文件。由此可保留“当时工程路径通过局部 smoke”，但不能升级为“低层训练已完成、机群 Agent 已验证”。本地没有 checkpoint 也不证明其他设备从未训练成功，只说明本次快照无法据此采用合格控制模型。[S1][S9][S11]

旧 RTA 材料则进一步限制“实验已经证明新方向”的主张：当前 runner 没有 LLM，枚举 27 个三段动作序列并使用手写 bearing 评分；观测是 evaluator truth 生成的理想几何包，不是 renderer/FastSAM 输出。协议是 12 s、10 Hz 新鲜包、30 Hz 环境步、15% 包删除，4 个独立物理 roots × 2 个嵌套删除 draws × 8 条件，共 64 trials。结果表的均值 6.4–6.775 等单位是安全且有用样本积分秒，而不是 episode 全程安全成功率；每条件 `n=8` 也不等于 8 个独立世界。[S12][S13][S14]

因此，若原宽泛完成主张被理解为“已有 Agent 实验”或“第二/三轮独立复现”，必须修订：runner 和协议将它收窄为短时几何包/调度实验，既有文件核验又显示第二/三轮 trials、statistics、calibration、replay 四文件 hash 完全相同，只能确认数据相同，不能证明独立重跑。这个局部结果值得保留为时钟、观测新鲜度与调度接口的经验线索；它没有检验本方案的多机任务交付、LLM 编排、真实视觉或 600 s 硬安全。未试变体仍未知，不能据此否定整个 Agent 方向，也不能把旧图用作新 benchmark 数据。[S12][S13][S14][S15]

取证覆盖也有边界：宿主扫描 4,997 个本地记录首行元数据，确认此前 119 文件对应 118 个唯一历史聊天（7 主、111 子，1 个主聊天有两段文件），结构浏览 8,495 个公开事件；关键主线和资产深读，其余工具正文未逐字全文复审，3 个子记录无公开事件。其他主机/云历史总量未知，10 个旧 PaperStudio 变体未全部全文重审。文献未复现，若干摘要级来源及访问失败明确限制相关工作判断。因此本方案不宣称“全历史没有做过某实验”，也不把历史转述当新增独立验证。[S16]

从设计到可评测平台须通过四个门槛：[S33]

| 门槛 | 最小交付与检查 | 未通过时如何处理 |
| --- | --- | --- |
| G1 控制资格 | 固定依赖/机型/有效频率；优先选择合格经典 autopilot；单/多机直飞、转弯、盘旋、出航、退出、风下跟踪与硬安全 | 收窄包线、几何或扰动，不靠不合格低层制造 Agent 失败 |
| G2 世界/工具/裁判 | 同步多机与对象运动；坐标/AGL；信息隔离；job 生命周期；边界点、名额、重复、迟报、补传、终态豁免，以及同刻真值/先筛后选/优先级生效与归一分母正负 fixture | 修接口或裁判根因，尚不接主模型跑分 |
| G3 感知/时间公平 | 结构化传感器误差与虚警校准；同条件扰动；1× 时间与资源争用；合法非 LLM 参考策略获得明显非零任务收益 | 分清不可行任务、裁判歧义、传感器及基础设施问题 |
| G4 开发冻结 | dev-only 调整生成器、阈值、prompt、预算、split、版本与重试规则 | 通过后只运行与分析 test，不按测试分数改协议 |

G1 的候选规模为每任务族至少 20 个开发 roots、总计至少 100 个开发 episode，正常可行任务至少 95% 无硬违规且完成；这些是待校准资格建议，不是已通过结果，也不是证明绝对安全的认证。失败 root 保留，不能因后续模型低分事后剔除。完整 benchmark 资格还需要 G2–G4，G1 一个数字不能代替全部门槛。[S33]

## 8. 用强经典基线与同模型对比检验可反驳假设

所有主方法共享飞控、结构化传感器、工具协议、任务卡与截止，必须都能确认、报告和安全退出。不能只用无报告的割草覆盖策略作对手，也不能让 LLM 多得隐藏对象信息或更强执行器。[S33]

| 方法 | 实现与公平性要求 | 对应解释 |
| --- | --- | --- |
| B0 固定分区 + 规则确认 | 按公开地图分区，发现候选后规则确认/报告，保留退出余量 | 提供能闭合业务的简单下界 |
| B1 在线 utility/auction 分配 | 基于合法局部包和可飞成本滚动重规划；若仅 CBBA-inspired，就不称完整 CBBA 复现 | 检验 LLM 是否超过在线经典分工 |
| B2 有限状态/行为树 | 显式处理确认、观察、交接、优先级、故障与退出；dev 调参 | 工程可解释事件响应强对手 |
| B3 统一 ReAct | 多个模型同工具、同提示结构与预算 | 主要模型横向比较 |
| B4 计划—执行—监控—事件改派 | 同一模型、同感知/工具，改变编排方式 | 隔离编排增益与换模型增益 |
| B5 混合方案，可选 | LLM 解释任务卡，经典调度执行；单独列为方法 | 分析语义与调度分工，不混入纯编排归因 |
| 特权 oracle，仅参照 | 可用完整未来信息检验场景可行性 | 只作上界/资格诊断，不参加公平胜负 |

经典策略应在 dev 同样认真调参并覆盖整条交付链。覆盖/任务分配基线可参考固定翼、遥感与动态分配先例，但需要实现本平台可飞航迹和合法传感器闭环，不能把抽象栅格路线称为同条件复现。[S25][S27][S28][S29][S33]

预注册三条可反驳假设，并将可用替代解释写入结果分析：[S33]

- H1：L3 事件后，同模型 B4 相比 B3 提高同一任务卡日历下的 `Y`，或改善第 9 节按统一需求权重计算的配对事件差值；它可能仍不如 B1/B2。优先级更新另看受影响规则的交付与改派落地，不能由命令数量判断能力。用无事件兄弟任务检验额外监控是否只增加抖动与成本。
- H2：假候选多且名额紧时，递进核查提高 precision 与 `Y`；但确认过量可能降低 `C/D`。核查并非越多越好，需同时展示收益与机会成本。
- H3：连续时钟使高延迟方法损失更多及时动态交付，监控可减少而不保证消除下降。依据固定观测时刻误差、先筛及时再选最新的 `D_delivered` 与迟报 trace 检验；不将两条独立选样轨道的分数差直接当纯通信损失。实际延迟与固定注入分别检验，不能把零延迟优势当部署优势。

候选五个 LLM 配置可具体化为 `B3-M1/M2/M3` 与 `B4-M1/M2`，三经典配置为 B0/B1/B2。M1/M2/M3 来自至少三个供应商或模型家族，并含一个可固定版本的开放权重模型；具体型号、版本和价格在执行日核对官方信息后冻结。本次不凭未核验的“最新型号”选模型。temperature、reasoning 等级、上下文、工具 JSON、prompt、重试、缓存、版本和调用限额进入 manifest；无法固定随机 seed 的 API 使用独立 repeat 并披露。B5、视觉轨道、语言解析、多 Agent 均属附加实验，不改变主矩阵。[S33]

消融只在预选 L3 的 12 roots 上，以同模型 B4 三次重复，每次去掉一个因素。优先三项为事件触发改固定周期、跨机候选去重/记忆、递进确认；age/provenance 检查可作额外诊断，grader 保持相同，因此旧包仍会被拒。关闭安全保护只能进入隔离仿真的 unsafe 诊断，不能列主排行推荐。机群 2/4/6、名额 6/10/14、延迟和噪声敏感性选小而清晰的子集，不做无法支付或解释的全因子乘积。[S33]

## 9. 以独立 root 拆分和配对估计，先核实规模与成本

`root` 指独立道路/地形图、对象布局及运动随机族。相似布局变换、语义等价任务卡、同布局纹理变体、无/有事件兄弟 episode 绑定同一 split；同 root 的 dropout、噪声抽样或 LLM 重复不是新增独立场景。OOD 应来自新道路图族、遮挡/通信模式或事件组合，而不只是换一个 seed；物理风条件仍限制在已经资格通过的包线内。[S33]

| 拆分候选 | 独立 roots | 用途与冻结边界 |
| --- | --- | --- |
| dev | 24 | 控制/任务调参、阈值与预算校准；6-root pilot 从这里抽取，不占 test |
| test-ID | 48，L1/L2/L3 各 16 | 同生成家族内未见布局；正式协议不再改 |
| test-OOD | 24，L2/L3 各 12 | 新图族或模式/事件组合；与 ID 分开报告 |
| 确认 holdout | 12，L2/L3 各 6 | 在方法和主分析冻结前不接触；只作确认，不再调 prompt |

主条件为 4 机、共享结构化传感器、实际连续时钟。LLM 每 root 每配置 3 次独立重复；经典环境/算法若确定则运行 1 次，不复制三份制造样本。先在 root 内聚合重复，再做跨方法配对。ID 与 OOD 分开，等级先各自平均再宏平均，防止任务数量比例决定一个混合总分。[S33]

五个 LLM 配置在 72 个 ID/OOD roots 上共 `5×72×3=1,080` episodes；三经典共 `3×72=216`，主矩阵合计 1,296。按每轮 600 s，仅飞行墙钟约 216 h；holdout 另有 `5×12×3+3×12=216` episodes，约 36 h。该估算不含资格、dev、消融、无事件兄弟对照、规模/噪声轨道、排队、故障重试和分析，不能描述为“半天低成本完成”。多 world 并行只在资源公平检查后采用，并分层随机方法顺序。若实际可支付规模较小，在打开 test 前整体调整并披露，不从运行后挑容易 roots。[S33]

6-root pilot 至少覆盖任务等级与事件代表条件，用于实测 tokens、API/工具成本、实际推理延迟、RTF/积压和 root 差值方差；它只能粗估，不能证明正式研究已有统计功效。预算按每方法实测 token 用量乘执行日价格，再加本地工具计算、存储、重试和完整飞行时间。最低有意义效应可暂以 `Y` 差 0.05 讨论，但须由业务损失与 pilot 方差校准；不迁移旧 RTA 的秒级阈值。是否承担正式规模是 pilot 后的资源决定。[S33]

主要 endpoint 为 root 平均 `Y`。主对比是同模型 `B4-M1 − B3-M1`；另预声明 `B4-M2 − B3-M2` 和 `B4-M1 − B_classic*`，其中 `B_classic*` 只在 dev 按冻结规则从 B1/B2 选择，不看 test。三个主要比较采用 Holm 校正，按 root 配对 bootstrap 给 95% CI，配对置换可作预注册替代或稳健检查。公开每 root 差值、均值/中位数、效应与 CI；重复、时间格、不同飞机均不视为独立样本。事件兄弟保持同组重采样。[S33]

`SR` 的嵌套重复采用 root-cluster 区间；只有每 root 一个独立二元结果时才直接采用 Wilson，不能将所有 repeats 当独立 Wilson 样本。需求权重不变的通信/感知/能力事件，任务损失为配对 `Y_event−Y_clean`。优先级事件改变需求权重，直接相减会混入评分口径变化；其配对诊断将 clean trace 离线按 event 的同一规则版本日历重评分为 `Y_clean|event`，再算 `Δ_priority=Y_event−Y_clean|event`，两边都用第 6 节的最终静态权重和逐格动态权重及完整分母。Actor 在 clean rollout 中仍只见当时合法的未更新卡片；离线重评分不回传信息，不改主表各轮实际任务的 `Y`。`Δ_priority` 表示获知需求变化后的交付差值，不能解释为同任务物理扰动的纯损失；同 root、同方法配对及 bootstrap 分组保持不变。[S33]

改派响应从更新通知到一项服务受影响公开规则、参数与安全约束合法的新派或修改 job 首次进入真实执行：job 已为 `running`，对应航迹/观察配置已被持续执行器实际采用，以 `rule_id/task_card_version` 和控制/采样 trace 判定；不要求该 job 已经 `succeeded`，ACK 或仅生成计划不算。它只证明改派落地，不证明优先级取舍最优，后者由该规则的加权交付与 `Y` 判断。若通知时既有任务与已交付证据已满足更新需求，单列“无需改派”，不以没有新命令判失败；其成立由冻结的公开后验要求与离线裁判核验。其余未响应/未恢复在 deadline 右删失并报删失率。恢复用预注册的受影响规则交付/观察缺口指标定义，按同一权重日历比较，不要求回到已不可能的旧得分，也不只平均成功案例。[S33]

所有策略失败，包括无效命令、预算耗尽、过期报告、模型 timeout 后依冻结规则失败和安全退出未达成，都保留在分母。确有证据的基础设施 crash/服务故障以 `invalid_attempt` 标记保留，并按预注册规则对关联方法对称处理；不得因为低分删除或无限重跑。C/S/D、成本与失败类别主要作诊断，不事后挑显著指标。统计必须体现 root 独立性；旧 4 roots × 2 draws 的表格口径不能直接移植。[S13][S14][S33]

## 10. 论文按研究问题组织，图表对应主张与原始 trace

暂定题名为 *Evaluating LLM Mission Agents for Multi-UAV Inspection under Continuous Flight and Delayed Feedback*。论文的中心论证是：民用交付需求产生有限报告、因果观测、持续固定翼和时间窗；共享低层与感知隔离高层决策；公平实验揭示任务质量、安全与时延的边界。benchmark 的贡献是可审计任务、协议、生成器、资格与强基线，以及真实结果支持的诊断，不能先承诺某个 Agent 方法获胜。[S33]

建议按以下证据依赖写作，而不是按模型名字逐一讲成绩：[S33]

| 论文部分 | 应回答的问题与必备内容 |
| --- | --- |
| Introduction | 四段：具体巡检信息交付痛点；成熟飞控之上仍需编排与变化响应；最近三项工作已覆盖什么及尚需复核的协议差别；有限贡献与 RQ |
| Related Work | 按 mission Agent、在线分配/覆盖、连续执行与延迟、证据报告评测分组，区分全文机制证据与摘要级未知 |
| Benchmark | 先场景/实体/episode，再权限、工具与时钟、生成器/split、评分和资格；明确设计数值和冻结版本 |
| Baselines and Experiments | 共享条件、强经典实现、模型配置、同模型编排、主矩阵与消融、配对统计、成本和失败分母 |
| Results | 按 RQ 解释主质量/安全、事件与交接、延迟和报告取舍；每项主张有对应图表与可复查 trace |
| Limitations | 合成异常代理/感知，非落地终态，控制资格有限，无实机/真实电池，版本漂移，小样本与场景外推，近邻尚未复现 |
| Conclusion | 只总结已观察结果与适用条件；不把完整协议发布等同现实救灾有效性 |

| 图表 | 计划内容 | 承载的论证与数据要求 |
| --- | --- | --- |
| Fig. 1 | 场景—机载观察—地面决策—持续执行—隐藏 grader 权限图 | 说明测什么、共享什么、truth 如何隔离；区分现有实现与新接口 |
| Fig. 2 | 一轮任务的合法事件、观测到达、确认、交接、退出时间线 | 展示因果任务链与连续时钟，不让未来 truth 进入 actor |
| Fig. 3 / 主表 | 分等级 ID/OOD 的 Q/Y、C/S/D、H/E/SR 与 root 配对 CI | 判断方法实际交付与安全，避免零分门控掩盖原因 |
| Fig. 4 | 事件前后覆盖、按同一权重日历的交付/动态格、观察交接缺口 | 检验 H1/H2，包含无事件兄弟、优先级诊断、无需改派与未恢复/删失情况 |
| Fig. 5 | 实际延迟与固定注入的质量—成本 Pareto | 检验 H3，区分规划逻辑、部署速度和预算代价 |
| Fig. 6 | 两类代表失败：合法包→tool/job→物理/报告后果 | 解释误确认、过期交付或交接失败，样例选择规则预先说明 |
| Table 1 | 最近工作逐机制阅读矩阵 | 约束贡献边界，未知格保留未知 |
| Table 2 | 任务生成参数、传感器/控制资格与冻结阈值 | 说明可复现范围与难度，不把建议写成已实现 |
| Table 3 | 主基线、模型、ID/OOD、质量/安全/成本 | 提供完整胜负与实用取舍 |
| Table 4 | 单因素消融与失败类型 | 支撑机制解释，unsafe 诊断单列 |

每条结果主张建立 `claim → figure/table → raw run → version/root/method` 映射，图表只来自对应正式 trace；旧 64 trials 只能解释研发前提，不能进入新 benchmark 主结果。可复现包日后包含 schema、生成器与 split、资格与 grader fixture、工具协议、基线、版本 manifest、合法观测/动作日志、回放摘要和许可证；模型回应的保存遵循实际服务条件。论文图不能通过删去低分、失败或高成本案例美化方法。[S12][S13][S14][S33]

预先保留阴性结果分支，保证实验回答问题而不是追求胜利：[S33]

- LLM 仅胜 B0、不能胜 B1/B2：写清语义变化或恢复条件下的局部作用与代价，不宣称整体优于经典规划。
- 全部 LLM 不胜但诊断稳定：论文定位为能力/失败/时延 benchmark，避免包装为新 Agent 方法成功。
- 差异由低层或基础设施事故驱动：先修资格；未经隔离的事故不能归因为模型规划能力。
- 优势只在暂停世界或零延迟存在：主要结论是部署时延边界，主实时结果不得被注入上界替代。
- 各方法在 test 饱和：如实报告分辨率不足；只在未来新协议用新 split 调难度，不用本 test 调后再称泛化。
- 最近工作不能被协议矩阵实质区分：收窄成固定翼连续巡检扩展或跨平台复核，保留研究价值但撤回首创主张。

## 11. 先闭合非 LLM 参考 episode，再冻结正式评测

赛事规则作为设计参考的价值已经明确：它提供搜索、有限报告、观察时间窗、通信变化和审计结构。成为有意义的 Agent benchmark，还必须由民用交付、唯一裁判、合格平台和公平实验共同成立。当前完成的是这些要素的研究方案与取证边界；控制器、同步任务世界、job、传感器与新 grader 都仍待实现和验证。[S32][S33]

| 阶段 | 交付与进入下一阶段的门槛 | 粗略时间建议 |
| --- | --- | --- |
| P0 本次设计 | 规则转译、近邻/资产核对、场景/协议/实验/论文完整方案；保留旧交接 | 本次产物；不是实验通过 |
| P1 控制与同步 world | G1/G2：4 机合法航迹、AGL/间距、连续目标、job 与信息隔离 | 1–2 周，取决于实际人力和低层资格 |
| P2 感知、裁判与 pilot | G3/G4：正负 fixture、合法参考收益、6-root 成本/时钟 pilot，冻结协议和主规模 | 1–2 周；未通过则修根因或收窄包线 |
| P3 正式研究 | 按冻结 ID/OOD、主矩阵、小消融与预注册分析执行 | 1–2 周仅为估计，以实测算力/API预算重排 |
| P4 确认与论文 | untouched holdout、主张—数据审计、结果图/正文与复现包 | 1–2 周估计，不保证发布日期 |

首个下一阶段交付应是一条**非 LLM 的完整参考 episode**：4 机合法航线→因果候选包→确认/报告→安全退出，以及 grader 正负边界 fixture。它先证明业务链、裁判和共享执行条件成立，随后才接模型、跑 pilot 并冻结正式评测。该 episode 本次没有执行，表中实现、测试、训练均是下一阶段工作。[S33]

下一阶段真正需要决定的是：资格实测是否支持候选区域、时限、传感器和评分阈值；pilot 成本和方差是否支持正式 ID/OOD 与 holdout 规模。通过则冻结版本和投入；未通过则有依据地缩小包线、任务难度或论文主张，而不是先跑大量 LLM 再调协议找优势。三维复建、视觉感知、多 Agent 和真实落地不先混入主验收，只有主链成立且各自验收明确后才扩展。[S33]

本次只读原有代码、文档、ctx 与历史，不执行来源中的训练/实验命令，不改旧 `PROJECT_HANDOFF.md`，不上传或发布赛事材料。最终计划与运行目录独立保存；研究方案的完整交付不意味着平台验收、外部工作复现或历史覆盖已经完整。[S1][S16][S32][S33]


## 来源与取证范围

以下为宿主提供并冻结的证据快照；程序仅检查格式、一致性和引用，不验证来源真伪。

### 项目根目录

```text
/Users/wdblink/Code/my_repo/pyflyt-drone
```

### 原始请求与授权约束

```text
用户原始请求：先查看项目路径里之前执行过的记录和代码/工具资产，深度调研在无人机仿真环境评估大模型Agent高层规划、编排及动态重规划的先例；参考多机建模采集和目标发现递进观察，形成严谨详实的实验设计和论文写作方案。上一阶段已交付PROJECT_HANDOFF.md，不覆盖它。用户最新原话：“/Users/wdblink/Downloads/“智信—2026”无人智能挑战赛竞技类比赛规则【定稿】 2.doc 这个文档里的规则，是不是很适合作为benchmark的设计参考材料，可以借鉴一下，最终还是要形成一份完整的从具体有意义的场景出发，评测agent相关能力的benchmark论文实验和写作计划，这样才有意义。” 本次必须从具体业务场景组织完整研究计划，不仅回答适合/复述规则，不仅列选择。只读原有项目、文档和历史；新建独立BENCHMARK_RESEARCH_PLAN.md及运行目录，不训练/执行实验、不上传比赛材料、不发布、不修改旧代码/ctx/文档/脏文件。高层任务Agent，不做实时舵面控制；机群区域建模原始数据可作为明确验收边界的扩展。技术文献已读第一手来源，未充分核验的项写未知；不得宣称first/SOTA。文献事实引用来源，宿主设计数值明确是待校准建议。最终正文要完整覆盖规则借鉴与改写、相关工作差异、现实场景与episode、任务/能力、合法信息和工具与时钟、确定评分语义、平台资格与实现缺口、强基线/实验矩阵/数据拆分/统计/成本、论文结构/图表/主张与阴性结果分支/实施阶段。历史覆盖及旧64trials结果只解释前提，不复述全历史。
```

### 检索覆盖与限制

```text
截至2026-10-02。归属为项目cwd及父任务链，Git HEAD270c6f0946b3be59527b2b77982f75a70da7ef77，branch codex/paper-privileged-vision。宿主list_projects识别project local-1a9a7957352394f5cd2babcddd42ad9b；list_threads(max50)仅当前聊天；list_archived_threads首50无本项目，下一游标2026-08-16T12:15:52.093Z未继续分页。补以本地sessions/archived_sessions共4997个rollout首行metadata扫描，120匹配文件包含当前1；此前119文件对应118唯一聊天ID（7主/111子），主聊天01a0563f有两段续接文件，3个子记录无公开事件。逐一浏览全部119条序列/请求/末回复，8495公开事件；关键7主聊天和相关工具输出/资产深读，其余工具参数输出未逐字全文阅读。排除隐藏分析、system/developer/turn_context/compaction和无归属外部正文。原始184.8MB历史不能称全文人工审完，其他主机/云总量未知。资产静态AST清单46Python文件已解析（不是runtime验证），关键env/train/eval调用链分段或全文读、有效配置/协议/ctx/旧工程smoke与新RTA结果核对。models/logs本地不存在。10个paperstudio_run变体目录已发现，重点external-v1文件深读，其他变体/旧spec计划/约60篇cached idea-spark文献未逐项全文复审。已读原始paper/官方项目约20项；最接近MultiUAV-Plat/MissionBench/ESARBench等方法和评测段落深读，若干仅摘要/metadata，全文403/HTML失败和GitHub API403明确记录，未复现任何外部系统。只读代码无本次compileall/training/episode；仅既有JSON解析、数据hash比较和资产/历史归属检查。 本阶段继承上次历史/资产/文献快照并新增用户指定赛事规则：完整正文提取与18页修复字体预览全读；未扩大历史检索、未执行项目实验。本地规则用途限制以原文条款为准，未核验其对外发布状态或主办方解释。设计候选为待资格校准方案。
```

### [S1] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/.project-handoff/20261002-benchmark/workspace-snapshot.json>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/.project-handoff/20261002-benchmark/workspace-snapshot.json
```

位置：

```text
Git HEAD/branch/status 与文件存在性/摘要检查
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次宿主只读命令结果；sha256仅用于交付时确认旧文件未变，不是运行验证。
```

证据摘录（纯文本）：

```text
.
{
  "generated_at": "2026-10-02T10:21:55.652628+08:00",
  "head": "270c6f0946b3be59527b2b77982f75a70da7ef77",
  "branch": "codex/paper-privileged-vision",
  "status": " M .gitignore\n?? .project-handoff/\n?? eval/paperstudio_rta_experiment.py\n?? ideaspark_run/\n?? paper/protocol/rta-experiment-runtime.json\n?? research-brief.md\n?? tests/test_paperstudio_rta_experiment.py\n",
  "ctx_resolved": "/Users/wdblink/Knowledge/Obsidian/LLM-Wiki/raw/CTX/pyflyt-drone-ctx",
  "models_exists": false,
  "logs_exists": false
}
```

### [S2] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/envs/aviary.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/envs/aviary.py
```

位置：

```text
L50–L84；L92–L115；L346–L392
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。静态读取多机初始化与setpoints/步进链，未本次加载依赖或运行；task-level单机封装另见S6/S7。
```

证据摘录（纯文本）：

```text
原文第50–84行：

    The `aviary` is a handler for physics stepping, setpoint handling, collisions tracking, and much more.
    It provides a common endpoint from where users may control drones or define tasks.

    Args:
        start_pos (np.ndarray): an `(n, 3)` array for the starting X, Y, Z positions for each drone.
        start_orn (np.ndarray): an `(n, 3)` array for the starting orientations for each drone, in terms of Euler angles.
        drone_type (str | Sequence[str]): a _lowercase_ string representing what type of drone to spawn.
        drone_type_mappings (None | dict(str, Type[DroneClass])): a dictionary mapping of `{str: DroneClass}` for spawning custom drones.
        drone_options (dict[str, Any] | Sequence[dict[str, Any]] | None): dictionary mapping of custom parameters for each drone.
        wind_type (None | str | Type[WindField]): a wind field model that will be used throughout the simulation.
        wind_options (dict[str, Any] | Sequence[dict[str, Any]]): dictionary mapping of custom parameters for the wind field.
        render (bool): a boolean whether to render the simulation.
        physics_hz (int): physics looprate (not recommended to be changed).
        world_scale (float): how big to spawn the floor.
        seed (None | int): optional int for seeding the simulation RNG.
        np_random (None | np.random.Generator): a numpy random number generator to be used for RNG.

    """

    def __init__(
        self,
        start_pos: np.ndarray,
        start_orn: np.ndarray,
        drone_type: str | Sequence[str],
        drone_type_mappings: None | dict[str, type[DroneClass]] = None,
        drone_options: dict[str, Any] | Sequence[dict[str, Any]] | None = None,
        wind_type: None | str | type[WindFieldClass] = None,
        wind_options: dict[str, Any] = {},
        render: bool = False,
        physics_hz: int = 240,
        world_scale: float = 1.0,
        seed: None | int = None,
        np_random: None | np.random.Generator = None,
    ):

原文第92–115行：
            start_orn (np.ndarray): an `(n, 3)` array for the starting orientations for each drone, in terms of Euler angles.
            drone_type (str | Sequence[str]): a _lowercase_ string representing what type of drone to spawn.
            drone_type_mappings (None | dict(str, Type[DroneClass])): a dictionary mapping of `{str: DroneClass}` for spawning custom drones.
            drone_options (dict[str, Any] | Sequence[dict[str, Any]] | None): dictionary mapping of custom parameters for each drone.
            wind_type (None | str | Type[WindField]): a wind field model that will be used throughout the simulation.
            wind_options (dict[str, Any] | Sequence[dict[str, Any]]): dictionary mapping of custom parameters for the wind field.
            render (bool): a boolean whether to render the simulation.
            physics_hz (int): physics looprate (not recommended to be changed).
            world_scale (float): how big to spawn the floor.
            seed (None | int): optional int for seeding the simulation RNG.
            np_random (None | np.random.Generator): a numpy random number generator to be used for RNG.

        """
        super().__init__(p.GUI if render else p.DIRECT)
        print("\033[A                             \033[A")

        # set random state
        if seed and np_random:
            raise AviaryInitException(
                "Cannot set both `seed` and `np_random` arguments together."
            )
    
（其余原文不纳入摘录；阅读范围见scope）
```

### [S3] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/fixedwing_objlock_env.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/fixedwing_objlock_env.py
```

位置：

```text
L148–L181；L471–L495；L557–L615；L704–L750
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。当前783行核心单机环境已分段静态阅读；excerpt选择观测与成功条件相关段落；不能当Agent接口。
```

证据摘录（纯文本）：

```text
原文第148–181行：
        self.spawn_exclude_center = bool(spawn_exclude_center)
 
        # --- State Variables ---
        self._lock_steps = 0
        self._prev_est_dist_m: Optional[float] = None
        self._last_cx = 0.5
        self._last_cy = 0.5
        self._last_area = 0.0
        self._last_depth_m = 0.0
        self._steps_since_seen = 60  # Default max
        self._prev_area: Optional[float] = None
        self._vision_history = np.zeros(
            (self.duck_vision_history_len, 9), dtype=np.float32
        )
        self._vision_history_filled = 0
        self.duck_pos: Optional[np.ndarray] = None # Duck position in global frame

        # --- Observation Space ---
        # "attitude": [ang_vel, ang_pos, lin_vel, lin_pos, action, aux_state]
        # "target_vector": [dx, dy, dz] (Body frame vector to duck)
        # "duck_vision": [visible, cx, cy, area, depth, steps_norm, d_left, d_center, d_right]
        duck_vision_dim = 9 * self.duck_vision_history_len
        if self.duck_vision_use_deltas:
            duck_vision_dim += 4
        self.observation_space = spaces.Dict(
            {
                "attitude": self.combined_space,
                "target_vector": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64
                ),
                "duck_vision": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(duck_vision_dim,), dtype=np.float32
                ),
            }

原文第471–495行：
            deltas[1] = float(base[2] - prev[2])
            deltas[2] = float(base[3] - prev[3])
            deltas[3] = float(base[4] - prev[4])

        return np.concatenate([history_flat, deltas], axis=0).astype(np.float32)

    def _spawn_duck(self):
        """Spawns duck with manual contact array resize to avoid IndexError."""
        if self.duck_body_id is not None:
            try:
                existing = {self.env.getBodyUniqueId(i) for i in range(self.env.getNumBodies())}
                if self.duck_body_id in existing:
                    self.env.removeBody(self.duck_body_id)
            except Exception:
                pass
            self.duck_body_id = None

        # Random Position Logic
        rng = self.np_random
        
        # Spawn within flight dome, but on ground
        r = self.flight_dome_size / 2.0
        x = rng.uniform(-r, r)
        y = rng.uniform(-r, r)
        z = 0.05 # Ground

原文第557–615行：
            # Avoid spawning near center (start pos)
            if self.spawn_exclude_center and x * x + y * y < 100.0:
                continue

            try:
                col_id = self.env.createCollisionShape(
                    self.env.GEOM_CYLINDER,
                    radius=float(self.obstacle_radius),
                    height=h,
                )
                vis_id = self.env.createVisualShape(
                    self.env.GEOM_CYLINDER,
                    radius=float(self.obstacle_radius),
                    length=h,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                )
                body_id = self.env.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=col_id,
                    baseVisualShapeIndex=vis_id,
                    basePosition=[x, y, z],
                )
                self.obstacle_ids.append(body_id)
            except Exception:
                pass

        try:
            max_id = -1
            for i in range(self.env.getNumBodies()):
                uid = self.env.getBodyUniqueId(i)
                if uid > max_id:
                    max_id = uid
            required = max_id + 1
            if self.env.contact_array.shape[0] < required:
                new_size = required + 50
                self.env.contact_array = np.zeros((new_size, new_size), dtype=bool)
        except Exception:
            pass

    def _try_enable_egl(self):
        # EGL Loading Logic
        try:
            if self._egl_plugin_id is not None:
                return
            
            egl = pkgutil.get_loader("eglRenderer")
            plugin = egl.get_filename() if egl else "eglRendererPlugin"
            self._egl_plugin_id = self.env.loadPlugin(plugin, "_eglRendererPlugin")
        except Exception:
            self._egl_plugin_id = None

    def _try_patch_camera_capture(self):
        if self._camera_capture_patched:
            return
        
        try:
            import pybullet as p
            original_capture = Camera.capture_image
            

原文第704–750行：
        
        return self._build_vision_vector(0.0)

    def _depth_buffer_to_meters(self, depth_buffer: float) -> float:
        near, far = 0.1, 255.0
        denom = (far - (far - near) * depth_buffer)
        if abs(denom) < 1e-9:
            return float(far)
        return float((far * near) / denom)

    def _estimate_obstacle_zone_distances_m(
        self, depth: np.ndarray, seg: np.ndarray
    ) -> tuple[float, float, float]:
        duc
（其余原文不纳入摘录；阅读范围见scope）
```

### [S4] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/base/fixedwing_vtail_base_env.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/base/fixedwing_vtail_base_env.py
```

位置：

```text
L42–L59；L94–L107；L211–L233；L314–L348
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。单机Vtail base和120分步约定已读；physics_hz须由Aviary实际参数另核，120不是自动的真实物理频率。
```

证据摘录（纯文本）：

```text
原文第42–59行：
            angle_representation (Literal["euler", "quaternion"]): angle_representation
            agent_hz (int): agent_hz
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution

        """
        if 120 % agent_hz != 0:
            lowest = int(120 / (int(120 / agent_hz) + 1))
            highest = int(120 / int(120 / agent_hz))
            raise ValueError(
                f"`agent_hz` must be round denominator of 120, try {lowest} or {highest}."
            )

        if render_mode and render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Invalid render mode {render_mode}, only {self.metadata['render_modes']} allowed."
            )
        self.render_mode = render_mode

原文第94–107行：
        )

        """ ENVIRONMENT CONSTANTS """
        self.start_pos = start_pos
        self.start_orn = start_orn
        self.flight_mode = flight_mode
        self.flight_dome_size = flight_dome_size
        self.max_steps = int(agent_hz * max_duration_seconds)
        self.env_step_ratio = int(120 / agent_hz)
        if angle_representation == "euler":
            self.angle_representation = 0
        elif angle_representation == "quaternion":
            self.angle_representation = 1


原文第211–233行：
        self.reward = 0.0
        self.info = {}
        self.info["out_of_bounds"] = False
        self.info["collision"] = False
        self.info["env_complete"] = False

        # need to handle Nones
        if options is None:
            options = dict()
        if drone_options is None:
            drone_options = dict()

        # camera handling
        drone_options["use_camera"] = drone_options.get("use_camera", False) or bool(
            self.render_mode
        )
        drone_options["camera_fps"] = int(120 / self.env_step_ratio)

        # init env
        self.env = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="fixedwing_vtail",

原文第314–348行：
    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict]:
        """Steps the environment.

        Args:
            action (np.ndarray): action

        Returns:
            state, reward, termination, truncation, info

        """
        # reset the reward
        self.reward = -0.1

        # pass the action, but clip the throttle
        self.action = action.copy()
        aviary_action = action.copy()
        aviary_action[..., -1] = (aviary_action[..., -1] / 2.0) + 0.5
        self.env.set_setpoint(0, aviary_action)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break

            self.env.step()

            # compute state and done
            self.compute_state()
 
（其余原文不纳入摘录；阅读范围见scope）
```

### [S5] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/flatten_objlock_env.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/flatten_objlock_env.py
```

位置：

```text
全文
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。当前wrapper全文阅读，拼接观测，不负责禁止真值输入。
```

证据摘录（纯文本）：

```text
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from envs.fixedwing_envs.fixedwing_objlock_env import FixedwingObjLockEnv


class FlattenObjLockEnv(gym.Wrapper):
    """
    Wrapper for FixedwingObjLockEnv that flattens the observation space.
    
    New Observation:
    [attitude (23), target_vector (3), duck_vision (9)]
    Total size: 35 (typically)
    """

    def __init__(self, env: FixedwingObjLockEnv):
        super().__init__(env)
        self.env = env
        
        # Compute flattened size
        obs_space = env.observation_space
        self.attitude_shape = obs_space["attitude"].shape[0]
        self.target_shape = obs_space["target_vector"].shape[0]
        self.vision_shape = obs_space["duck_vision"].shape[0]
        
        total_size = self.attitude_shape + self.target_shape + self.vision_shape
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_size,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._flatten_obs(obs), reward, term, trunc, info

    def _flatten_obs(self, obs: dict) -> np.ndarray:
        return np.concatenate([
            obs["attitude"],
            obs["target_vector"],
            obs["duck_vision"]
        ], axis=0).astype(np.float32)

（其余原文不纳入摘录；阅读范围见scope）
```

### [S6] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/objlock_yolo_env.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/objlock_yolo_env.py
```

位置：

```text
L18–L38；L80–L108；L306–L343；L443–L467；L560–L598
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。872行FastSAM封装分段静态阅读；名字YOLO不代表使用YOLO，patch需use_egl触发，类级camera monkeypatch闭包、多环境隔离未验证；不能等同仅更换感知源。
```

证据摘录（纯文本）：

```text
原文第18–38行：

class FixedwingObjLockEnv(FixedwingVtailBaseEnv):
    """
    Fixedwing Object Lock Environment (No Waypoints).

    Actions are roll, pitch, yaw, thrust commands.
    The task is to locate and strike a yellow duck on the ground using visual cues.

    Args:
        sparse_reward (bool): whether to use sparse rewards.
        flight_mode (int): UAV flight mode.
        flight_dome_size (float): allowable flying area size.
        max_duration_seconds (float): max simulation time.
        angle_representation (Literal["euler", "quaternion"]): "euler" or "quaternion".
        agent_hz (int): looprate.
        render_mode (None | Literal["human", "rgb_array"]): render mode.
        render_resolution (tuple[int, int]): render resolution.
        duck_urdf_path (str | None): path to duck URDF. If None, uses default from pybullet_data.
        use_egl (bool): whether to attempt EGL hardware rendering (for headless GPU).
    """


原文第80–108行：
        duck_lock_lost_penalty: float = 0.5,
        duck_approach_reward_clip_m: float = 2.0,
        wind_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 100.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
            wind_config=wind_config,
        )

        self.sparse_reward = sparse_reward

        # --- Duck Phase Configs ---
        self.duck_urdf_path = duck_urdf_path or os.path.join(
            pybullet_data.getDataPath(), "duck_vhacd.urdf"
        )
        self.use_egl = use_egl
        
        # FastSAM Seg Configs
        self.fastsam_model = None
        self.fastsam_weights_path = self._resolve_default_fastsam_weights_path()
        self.fastsam_device = "cuda"
        self.fastsam_retina_masks = True

原文第306–343行：
        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state of the current timestep."""
        # --- Base Attitude ---
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()

        new_state: dict = dict()
        if self.angle_representation == 0:
            new_state["attitude"] = np.concatenate(
                [ang_vel, ang_pos, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )
        elif self.angle_representation == 1:
            new_state["attitude"] = np.concatenate(
                [ang_vel, quaternion, lin_vel, lin_pos, self.action, aux_state], axis=-1
            )

        # --- Target Vector (Duck Relative Pos in Body Frame) ---
        if self.duck_body_id is not None and self.duck_pos is not None:
             # Global Delta
             diff = self.duck_pos - lin_pos
             # Rotate to Body Frame
             # Get Rotation Matrix (3x3) from Quaternion
             rot = np.array(self.env.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
             # Body = R.T * Global
             target_vector = rot.T @ diff
        else:
             target_vector = np.zeros(3, dtype=np.float64)
        
        new_state["target_vector"] = target_vector

        # --- Duck Vision Features ---
        base_feature = self._compute_vision_features()
        new_state["duck_vision"] = self._build_duck_vision_observation(base_feature)

        self.state = new_state


原文第443–467行：
        d_left = float(vis[6])
        d_center = float(vis[7])
        d_right = float(vis[8])
        depths = [d for d in (d_left, d_center, d_right) if d > 0.0 and np.isfinite(d)]
        if not depths:
            return

        d_obs = min(depths)
        d_safe = float(self.obstacle_safe_distance_m)
        if d_safe <= 0.0:
            return

        if d_obs >= d_safe:
            return

        scale = float(self.obstacle_avoid_reward_scale) * 0.5 # Reduced scale for duck phase (always on)

        penalty = scale * (d_safe - d_obs) / d_safe
        penalty = min(penalty, float(self.obstacle_avoid_max_penalty))
        self.reward -= penalty

    def _reset_duck_state(self):
   
（其余原文不纳入摘录；阅读范围见scope）
```

### [S7] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/fixedwing_lowlevel_env.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/envs/fixedwing_envs/fixedwing_lowlevel_env.py
```

位置：

```text
L16–L52；L95–L123；L130–L163
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。低级环境全文静态读；6执行器和self.target，不是HighLevelCmdEnv所期望的wrapper接口。
```

证据摘录（纯文本）：

```text
原文第16–52行：
    Reward: Based on tracking error of high-level targets, with safety and smoothness constraints.
    """
    
    metadata = {"render_modes": ["human"], "name": "fixedwing_lowlevel_env"}

    def __init__(
        self, render_mode: Optional[str] = None, wind_config: Optional[dict[str, Any]] = None
    ):
        super().__init__()
        self.render_mode = render_mode
        self._wind_config = wind_config or {}
        
        # Define simulation parameters
        self.start_height_m = 10.0
        self.start_speed_mps = 15.0
        self.min_speed_mps = 5.0
        self.target_speed_range = (10.0, 20.0)
        self.target_height_range = (5.0, 20.0)
        
        # Initialize Aviary
        start_pos = np.array([[0.0, 0.0, self.start_height_m]])
        start_orn = np.array([[0.0, 0.0, 0.0]])
        drone_options = dict(starting_velocity=np.array([self.start_speed_mps, 0.0, 0.0]))
        
        self.env = Aviary(
            start_pos=start_pos,
            start_orn=start_orn,
            drone_type="fixedwing",
            drone_options=drone_options,
            render=self.render_mode == "human",
            physics_hz=240,
            world_scale=1.0,
        )

        if bool(self._wind_config.get("enabled", False)) and not bool(
            self._wind_config.get("randomize_on_reset", False)
        ):

原文第95–123行：
        return obs, info

    def step(self, action):
        self._episode_steps += 1
        self.prev_action = action.copy()
        
        # Step physics
        self.env.set_all_setpoints(action.reshape(1, -1))
        self.env.step()
        
        obs = self._compute_obs()
        
        # Extract state for reward calculation
        # state: [ang_vel(3), ang_pos(3), lin_vel(3), lin_pos(3)]
        state = self.env.state(0)
        if isinstance(state, np.ndarray):
            state = state.flatten()
        elif isinstance(state, (list, tuple)):
            state = np.concatenate(state).flatten()

        current_psi = state[5] # Yaw
        current_vel = np.linalg.norm(state[6:9])
        current_alt = state[11]
        
        target_psi, target_h, target_v = self.target
        
        # Calculate errors
        psi_err = abs(self._wrap_pi(target_psi - current_psi))
        h_err = abs(target_h - current_alt)

原文第130–163行：
        # Termination conditions
        terminated = False
        if current_alt < 1.0 or current_alt > 100.0:
            terminated = True
            reward -= 100.0
        
        truncated = False
        if self._episode_steps >= 2000: # Max steps
            truncated = True
            
        info = {"target": self.target.copy()}
        
        return obs, reward, terminated, truncated, info

    def _compute_obs(self):
        state = self.env.state(0)
        if isinstance(state, np.ndarray):
            state = state.flatten()
        elif isinstance(state, (list, tuple)):
            state = np.concatenate(state).flatten()
            
        #
（其余原文不纳入摘录；阅读范围见scope）
```

### [S8] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/train/train_highlevel_cmd.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/train/train_highlevel_cmd.py
```

位置：

```text
L86–L117；L145–L185
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。原型静态阅读；传base_env给render_mode、写target_heading却lowlevel读取self.target、调用不存在_get_obs，以及低级6维与base4维动作衔接，说明不能直接当合格共享航迹执行器；未本次运行。
```

证据摘录（纯文本）：

```text
原文第86–117行：
        if wind_config and bool(wind_config.get("enabled", False)):
            base_env = WindOnResetWrapper(base_env, wind_config)
        self._base_env = FlattenWaypointEnv(base_env, context_length=context_length)

        # 保存范围参数用于动作空间定义
        self._flight_dome_size = flight_dome_size

        # 高层动作空间: [heading, altitude, airspeed]
        # - heading_cmd: [-pi, pi]
        # - altitude_cmd: [0, flight_dome_size]
        # - airspeed_cmd: [0.0, 30.0]（简化选择，后续可调整/标定）
        self.action_space = spaces.Box(
            low=np.array([-np.pi, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.pi, flight_dome_size, 30.0], dtype=np.float32),
            dtype=np.float32,
        )

        # 高层观测空间沿用 FlattenWaypointEnv
        self.observation_space = self._base_env.observation_space

        # 构造一个底层包装器用于生成底层模型的观测（不用于 step）
        self._low_wrapper = FixedwingLowLevelEnv(self._base_env)

        # 加载底层 PPO 模型
        if not os.path.exists(low_model_path):
            raise FileNotFoundError(
                f"未找到底层模型文件: {low_model_path}，请先完成底层训练"
            )
        self._low_model = PPO.load(low_model_path)
        self._low_deterministic = deterministic_low

        # 加载 VecNormalize 统计量用于手动归一化底层观测

原文第145–185行：
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """重置环境并返回扁平化观测。"""
        obs, info = self._base_env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action: np.ndarray):
        """执行一步：高层动作经底层控制器转换为环境动作，再前进一步。

        Args:
            action: 高层指令 [heading_cmd, altitude_cmd, airspeed_cmd]
        Returns:
            (obs, reward, terminated, truncated, info)
        """
        # 解析高层动作
        heading_cmd = float(action[0])
        altitude_cmd = float(action[1])
        airspeed_cmd = float(action[2])

        # 写入底层目标
        self._low_wrapper.target_heading = self._normalize_angle(heading_cmd)
        self._low_wrapper.target_altitude = float(np.clip(altitude_cmd, 0.0, self._flight_dome_size))
        self._low_wrapper.target_airspeed = float(np.clip(airspeed_cmd, 0.0, 100.0))

        # 生成底层模型观测（未归一化）
        low_obs_raw = self._low_wrapper._get_obs(None).astype(np.float32)
        # 归一化到底层模型训练分布
        low_obs_norm = self._normalize_low_obs(low_obs_raw)

        # 底层模型推理得到环境动作（roll, pitch, yaw, thrust）
        low_action, _ = self._low_model.predict(
            low_obs_norm, deterministic=self._low_deterministic
        )

        # 执行基础环境一步
        obs, reward, terminated, truncated, info = self._base_env.step(low_action)

        return obs, reward, terminated, truncated, info


# 训练参数
TRAIN_CFG = {
（其余原文不纳入摘录；阅读范围见scope）
```

### [S9] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/train/train_objlock.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/train/train_objlock.py
```

位置：

```text
L55–L98；L239–L268；L574–L614
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。训练入口/回调和配置静态读；SubprocVecEnv是独立训练环境，不是共享物理机群。未找到models/logs或checkpoint；模型训练完成未知。
```

证据摘录（纯文本）：

```text
原文第55–98行：
    "duck_lock_step_reward": 0.2,
    "duck_approach_reward_scale": 0.1,
    "duck_global_scaling": 60.0,
    
    # 障碍物参数
    "num_obstacles": 0,
    "obstacle_radius": 2.0,
    "obstacle_height_range": (10.0, 30.0),
    "obstacle_safe_distance_m": 10.0,
    "obstacle_avoid_reward_scale": 1.0,
    "obstacle_avoid_max_penalty": 5.0,

    # 相机参数
    "camera_profile": "cockpit_fpv",

    # 视觉历史观测参数
    "duck_vision_history_len": 3,
    "duck_vision_use_deltas": True,
    "drone_model": "fixedwing_vtail",
    "drone_model_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "my_models")),

    "wind": {
        "enabled": True,
        "mode": "gust_sine",
        "wind_enu_mps": [0.0, 0.0, 0.0],
        "wind_enu_mps_range": [[-10.0, 10.0], [-10.0, 10.0], [-0.10, 0.10]],
        "gust_amp_enu_mps": [0.0, 0.0, 0.0],
        "gust_amp_enu_mps_range": [[0.0, 3.0], [0.0, 3.0], [0.0, 0.3]],
        "gust_freq_hz": 0.2,
        "gust_phase_rad": 0.0,
        "randomize_on_reset": True,
        "randomize_gust_phase": True,
    },
}

TRAIN_PHASES = {
    "phase1": {
        "num_obstacles": 3,
        "obstacle_safe_distance_m": 20.0,
        "obstacle_avoid_reward_scale": 2.0,
        "duck_strike_reward": 0.0,
        "duck_visible_step_reward": 0.5,  # 25% of normal
        "duck_area_reward_scale": 1.0,   # 20% of normal
        "spawn_exclude_center": True,

原文第239–268行：

        # Hack detection
        if timeout_rate > 0.9 and collision_rate < 0.01:
            print("[WARNING] Possible orbit/freeze hack detected")
            self.logger.record("eval/hack_detected", 1.0)

    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                from stable_baselines3.common.vec_env import sync_envs_normalization

                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            from stable_baselines3.common.evaluation import evaluate_policy

            self._is_success_buffer = []
            self._reset_eval_metric_buffers()

            episode_rewards, episode_lengths = evaluate_policy(
      
（其余原文不纳入摘录；阅读范围见scope）
```

### [S10] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/ctx/decisions/0003-causal-perception-and-evidence-gates.md>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/ctx/decisions/0003-causal-perception-and-evidence-gates.md
```

位置：

```text
全文
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。全文阅读；actor信息边界和common perception要求是旧论文目标契约，而非现有代码已实现。
```

证据摘录（纯文本）：

```text
# 0003. 采用因果感知与可检索证据门

* Status: accepted
* Date: 2026-08-05 · Deciders: project owner

## Context and problem statement

ADR-0002 已选择“训练期特权信息到感知部署”为论文主线，但原始契约仍允许 evaluator failure metadata 与 simulator depth 进入 Actor，且把多项机制、确认性估计与证据生命周期留给实现期解释。R1.2 独立审查因此无法判定信息边界、机制归因、统计完整性和 OPC protected transition 是否会 fail closed。

## Decision drivers

- 部署输入必须来自明确命名且因果可实现的机载 sensor suite。
- 训练期 privilege 必须可通过 action-path invariance 证明不进入 deployment export。
- 每个承载机制结论的因素必须拥有独立开关与匹配对照。
- 确认性统计必须在 execution 前具备完整 denominator、estimator 与 missingness rule。
- 被接纳的 evidence 必须可检索、可重哈希，并在缺失或损坏时产生 NONPASS。
- OPC heartbeat 必须机械阻止未通过 review、未闭合 finding 或缺失 evidence 的 protected successor。

## Considered options

- 保留原始三机制 prose contract，在各实现单元补充细节。
- 将真实 Actor 限制为 RGB-only，并删除所有 metric depth 输入。
- 采用同步 onboard RGB-D 加 proprioception，拆分 Actor measurement 与 evaluator telemetry，并以 tracked JSON 作为执行 authority。
- 继续依赖通用 OPC harness 的文字计划与最终人工审查。

## Decision outcome

采用同步 onboard RGB-D 加 proprioception 作为 deployment sensor claim；Actor 只接收因果 sensor product，typed adapter failure 与 simulator truth 保持 evaluator/training-only。方法、评估和 loop guard 分别由 `paper/protocol/method-matrix.json`、`paper/protocol/evaluation-protocol.json` 与 `paper/protocol/loop-guard.json` 定义，并由 standard-library validator 执行。确认性 scope 限于 `P` 对冻结 internal baseline；更强 claim 在独立 gate 前保持禁止。

本决策补充并收紧 ADR-0002，不取代或改写 ADR-0002 的主线选择。

### Consequences

- 优点：Actor/Evaluator/Critic 边界、association、corruption、五机制归因、paired statistics 与 evidence lifecycle 可由 binary tests 审计。
- 优点：missing/corrupt evidence、incomplete records 与 non-clean protected reviews 无法通过缩小 denominator 或文字说明转写为成功。
- 优点：未来 heartbeat tick 可直接调用 `loop-complete`，不再依赖 aspirational prose。
- 代价：现有 observation、checkpoint、normalizer、FastSAM prototype 与训练入口不兼容，必须在 fresh contract PASS 后逐单元重构。
- 代价：template、calibration、implementation hashes 与 episode count 在 freeze 前为空，因而 confirmation 必须保持不可执行。
- 代价：未完成 matched external reproduction 时，只能形成 internal-method claim 或 NONPASS dossier。

## Confirmation

- 当前规范：[`../spec/02-privileged-perceptual-guidance.md`](../spec/02-privileged-perceptual-guidance.md)
- 执行协议（repository-relative）：`paper/protocol/README.md`
- 机械 validator（repository-relative）：`scripts/validate_paper_protocol.py`
- disposable literature evidence：[`../scratch/02-systematic-literature-screen.md`](../scratch/02-systematic-literature-screen.md)

（其余原文不纳入摘录；阅读范围见scope）
```

### [S11] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/.harness-fix-opc/nodes/test-execute/test-results.md>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/.harness-fix-opc/nodes/test-execute/test-results.md
```

位置：

```text
全文
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。历史OPC验证记录全文读；真实回调、vectorwrapper、ObjLock reset/step在当时条件成功，但没有任务级Agent/机群能力验证；本次不重跑。
```

证据摘录（纯文本）：

```text
# Test Execution Results

## Interpreter
- Runtime checks used `/opt/homebrew/Caskroom/miniforge/base/envs/pyflyt/bin/python`.
- The default system `python3` does not include the project simulation dependencies.
- Production code does not embed this interpreter path.

## Results
1. `compileall`
   - Command: `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/Caskroom/miniforge/base/envs/pyflyt/bin/python -m compileall envs train eval scripts utils`
   - Result: PASS.

2. Import smoke
   - Modules imported: `envs.ab_fixedwing_env`, all compatibility shims, `train.train_ppo_ab`, `train.train_objlock`, `eval.eval_objlock`, and `eval.eval_waypoint_objlock`.
   - Result: PASS, output `import-ok 11`.

3. Shim identity smoke
   - Asserted all compatibility exports are identical to the maintained classes.
   - Result: PASS, output `identity-ok`.

4. A-to-B runtime reset
   - Constructed `make_fixedwing_ab_env(...)`, ran `reset(seed=123)`, asserted `obs.shape == (26,)`, and closed the env.
   - Result: PASS, output `ab-reset-ok (26,) True`.

5. A-to-B vector wrapper
   - Wrapped the factory in `DummyVecEnv` and `VecNormalize`, ran reset and one sampled step.
   - Result: PASS, output `ab-vec-ok (1, 26) (1,) (1,) dict`.

6. Callback structure
   - Asserted `ObjLockAvoidanceEvalCallback` has `_log_success_callback` and `_on_step`, and does not define `_on_eval_end`.
   - Asserted installed SB3 `EvalCallback` also lacks `_on_eval_end`.
   - Result: PASS, output `callback-structure-ok`.

7. Callback full-path smoke
   - Ran a tiny temporary Gymnasium env through `PPO.learn(...)` with `ObjLockAvoidanceEvalCallback`.
   - Wrapped train/eval envs in `VecNormalize` so the normalization sync path executed.
   - Asserted temp `best_model.zip` and `evaluations.npz` were produced.
   - Asserted callback success, duck-strike, collision, out-of-bounds, and timeout buffers matched expected terminal info.
   - Result: PASS, output `callback-fullpath-ok best_model.zip evaluations.npz [True, True]`.

8. Artifact isolation
   - Checked the repository tree for `best_model.zip` and `evaluations.npz`.
   - Result: PASS, no generated model or evaluation artifact was found under the repository.

## Not Run
- Long PPO training was not run.
- GUI rendering was not run.
- No Git staging or commit was performed.

（其余原文不纳入摘录；阅读范围见scope）
```

### [S12] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/eval/paperstudio_rta_experiment.py>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/eval/paperstudio_rta_experiment.py
```

位置：

```text
L58–L117；L155–L191；L238–L280；L347–L397；L478–L516
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。当前未跟踪575行external runner全文静态阅读。无LLM：select_sequence枚举27个3×4s动作序列/手写bearing评分；geometric_packet源于真值理想视锥，不是renderer/FastSAM。safe/useful按新鲜sample帧积分，不是全程安全任务成功；净空近似和地速也不能当formal安全包线。
```

证据摘录（纯文本）：

```text
原文第58–117行：
# frozen in the runtime configuration, so keep the namespace join explicit.
REGISTERED_ROOT_MAP = {
    1: 150001,
    2: 150002,
    3: 150003,
    4: 150004,
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("claim_id") != EXPECTED_CLAIM:
        raise ValueError("claim binding mismatch")
    condition_ids = {item.get("id") for item in plan.get("conditions", [])}
    if condition_ids != EXPECTED_CONDITIONS:
        raise ValueError("condition set differs from the registered RTA assay")
    seeds = plan.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 8:
        raise ValueError("the registered assay requires eight seeds")
    if len(set(seeds)) != len(seeds) or not all(isinstance(seed, int) for seed in seeds):
        raise ValueError("seeds must be eight distinct integers")
    if {seed // 100 for seed in seeds} != set(REGISTERED_ROOT_MAP):
        raise ValueError("seeds must map to the four frozen physical roots")
    if any(seed % 100 not in (1, 2) for seed in seeds):
        raise ValueError("each physical root requires deletion draws 01 and 02")


def deletion_draw(seed: int, frame_index: int) -> float:
    payload = f"{seed}:{frame_index}:packet-delete-v1".encode("ascii")
    raw = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(raw, "big") / float(1 << 64)


def packet_after_deletion(
    packet: dict[str, float], seed: int, frame_index: int, deletion_rate: float
) -> dict[str, float]:
    if packet["visible"] > 0.5 and deletion_draw(seed, frame_index) < deletion_rate:
        return {
            "visible": 0.0,
            "center_x": 0.0,

原文第155–191行：
    if packet["visible"] <= 0.5:
        return ("left", "straight", "right")
    initial_bearing = (packet["center_x"] - 0.5) * 100.0
    yaw_delta = {"straight": 0.0, "left": 20.0, "right": -20.0}
    best: tuple[float, tuple[str, str, str]] | None = None
    for sequence in itertools.product(primitives, repeat=3):
        bearing = initial_bearing
        integrated = 0.0
        changes = 0
        for index, primitive in enumerate(sequence):
            bearing -= yaw_delta[primitive]
            integrated += abs(bearing)
            if index and primitive != sequence[index - 1]:
                changes += 1
        score = integrated + 2.0 * changes
        candidate = (score, sequence)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    return best[1]


def root_for_seed(config: dict[str, Any], trial_seed: int) -> dict[str, Any]:
    registered_root = trial_seed // 100
    try:
        root_seed = REGISTERED_ROOT_MAP[registered_root]
    except KeyError as exc:
        raise ValueError(f"no registered root for trial seed {trial_seed}") from exc
    for root in config["scenario_roots"]:
        if int(root["physics_seed"]) == root_seed:
            return root
    raise ValueError(f"no frozen root for trial seed {trial_seed}")


def make_env(config: dict[str, Any], physics_seed: int) -> FixedwingObjLockEnv:
    runtime = config["runtime"]
    env = FixedwingObjLockEnv(

原文第238–280行：

def safety_state(
    env: FixedwingObjLockEnv,
    truth_packet: dict[str, float],
    config: dict[str, Any],
) -> tuple[bool, dict[str, float]]:
    metric = config["metric"]
    state = env.env.state(0)
    attitude = state[1]
    velocity = state[2]
    position = state[3]
    speed = float(np.linalg.norm(velocity))
    clearance = obstacle_clearance(env, position)
    safe = (
        not bool(env.info.get("collision"))
        and not bool(env.info.get("out_of_bounds"))
        and float(position[2]) >= float(metric["minimum_altitude_m"])
        and float(metric["speed_range_mps"][0]) <= speed <= float(metric["speed_range_mps"][1])
        and abs(math.degrees(float(attitude[0]))) <= float(metric["maximum_abs_bank_degrees"])
        and abs(math.degrees(float(attitude[1]))) <= float(metric["maximum_abs_pitch_degrees"])
        and clearance >= float(metric["minimum_obstacle_clearance_m"])
        and truth_packet["depth_m"] >= float(metric["minimum_target_standoff_m"])
    )
    useful = (
        truth_packet["visible"] > 0.5
        and abs(truth_packet["center_x"] - 0.5) <= float(metric["useful_center_half_width"])
  
（其余原文不纳入摘录；阅读范围见scope）
```

### [S13] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/paper/protocol/rta-experiment-runtime.json>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/paper/protocol/rta-experiment-runtime.json
```

位置：

```text
全文
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。当前未跟踪runtime协议全文读；4独立physics roots×2 dropout_draws×8conditions=64trial，dropout .15、12s短时域、10Hz packet/30Hzstep；这些是旧调度实验，不是Agent评测规范。
```

证据摘录（纯文本）：

```text
{
  "schema_version": "rta-experiment-runtime/v1",
  "status": "frozen-preconfirmatory",
  "runtime": {
    "python": "/tmp/pyflyt-paperstudio-venv311/bin/python",
    "pyflyt": "0.29.0",
    "pybullet": "3.2.7",
    "agent_hz": 30,
    "fresh_frame_hz": 10,
    "horizon_seconds": 12.0,
    "flight_dome_m": 500.0
  },
  "airframe": {
    "model": "fixedwing_vtail",
    "model_dir": "my_models",
    "start_position_m": [0.0, 0.0, 100.0],
    "start_speed_mps": 20.0
  },
  "camera_model": {
    "source": "evaluator-truth geometric ideal packet",
    "horizontal_fov_degrees": 100.0,
    "vertical_fov_degrees": 100.0,
    "downtilt_degrees": 40.0,
    "packet_fields": ["visible", "center_x", "center_y", "area_ratio", "depth_m"],
    "actor_forbidden_fields": ["target_vector", "duck_pos", "simulator_body_id", "future_packet"]
  },
  "scenario_roots": [
    {"root_id": "root-150001", "physics_seed": 150001, "target_position_m": [120.0, -24.0, 0.05]},
    {"root_id": "root-150002", "physics_seed": 150002, "target_position_m": [125.0, -8.0, 0.05]},
    {"root_id": "root-150003", "physics_seed": 150003, "target_position_m": [125.0, 8.0, 0.05]},
    {"root_id": "root-150004", "physics_seed": 150004, "target_position_m": [120.0, 24.0, 0.05]}
  ],
  "trial_seed_registration": {
    "encoding": "100 * registered_root + deletion_draw",
    "registered_to_physics_seed": {
      "1": 150001,
      "2": 150002,
      "3": 150003,
      "4": 150004
    },
    "deletion_draws": [1, 2],
    "frozen_before_confirmatory_execution": true
  },
  "degradation": {
    "deletion_rate": 0.15,
    "algorithm": "sha256-counter-v1",
    "key_fields": ["trial_seed", "frame_index", "packet-delete-v1"],
    "method_independent": true
  },
  "executor": {
    "chunk_seconds": 4.0,
    "primitives": {
      "straight": [0.0, -0.4, 0.0, -0.5],
      "left": [-0.02, -0.4, 0.0, -0.5],
      "right": [0.02, -0.4, 0.0, -0.5]
    },
    "candidate_sequences": "cartesian product of three primitives over three chunks (27 sequences)",
    "trigger": "two consecutive fresh missing packets",
    "no_launch_sentinel_seconds": 12.1
  },
  "metric": {
    "name": "safe_useful_viewing_seconds",
    "bounds": [0.0, 12.0],
    "frame_seconds": 0.1,
    "minimum_target_standoff_m": 20.0,
    "minimum_obstacle_clearance_m": 5.0,
    "minimum_altitude_m": 10.0,
    "speed_range_mps": [8.0, 80.0],
    "maximum_abs_bank_degrees": 60.0,
    "maximum_abs_pitch_degrees": 45.0,
    "useful_center_half_width": 0.35,
    "useful_center_half_height": 0.35
  },
  "calibration": {
    "seeds": [910001, 910002, 910003],
    "straight_requirements": {"seconds": 12.0, "minimum_altitude_m": 50.0, "maximum_abs_bank_degrees": 20.0},
    "turn_requirements": {"seconds": 4.0, "minimum_abs_bank_degrees": 10.0, "maximum_abs_bank_degrees": 40.0}
  }
}

（其余原文不纳入摘录；阅读范围见scope）
```

### [S14] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0003/experiment_run/A000031/statistics.json>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0003/experiment_run/A000031/statistics.json
```

位置：

```text
全文
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。全文读取既有8条件结果表；单位为safe-and-useful样本积分秒，不能声称全程安全成功率；n=8含4roots×2嵌套draws，表内bootstrap口径不能当8独立场景。
```

证据摘录（纯文本）：

```text
[
  {
    "condition": "crossed_clean_clean_schedule",
    "n": 8,
    "mean": 6.775,
    "std": 0.3058944729337692,
    "ci_low": 6.575,
    "ci_high": 6.975
  },
  {
    "condition": "crossed_degraded_degraded_schedule",
    "n": 8,
    "mean": 6.4,
    "std": 0.6071008388821652,
    "ci_low": 6.0,
    "ci_high": 6.775
  },
  {
    "condition": "crossed_degraded_clean_schedule",
    "n": 8,
    "mean": 6.775,
    "std": 0.3058944729337692,
    "ci_low": 6.575,
    "ci_high": 6.975
  },
  {
    "condition": "crossed_clean_degraded_schedule",
    "n": 8,
    "mean": 6.4,
    "std": 0.6071008388821652,
    "ci_low": 6.0,
    "ci_high": 6.775
  },
  {
    "condition": "natural_clean_closed_loop",
    "n": 8,
    "mean": 6.775,
    "std": 0.3058944729337692,
    "ci_low": 6.575,
    "ci_high": 6.975
  },
  {
    "condition": "natural_degraded_closed_loop",
    "n": 8,
    "mean": 6.5,
    "std": 0.6654751256486924,
    "ci_low": 6.05,
    "ci_high": 6.9
  },
  {
    "condition": "proactive_clean_packets",
    "n": 8,
    "mean": 6.625,
    "std": 0.5922113522335464,
    "ci_low": 6.2,
    "ci_high": 6.975
  },
  {
    "condition": "proactive_degraded_packets",
    "n": 8,
    "mean": 6.625,
    "std": 0.5922113522335464,
    "ci_low": 6.2,
    "ci_high": 6.975
  }
]
（其余原文不纳入摘录；阅读范围见scope）
```

### [S15] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/.project-handoff/20261002-benchmark/result-file-check.json>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/.project-handoff/20261002-benchmark/result-file-check.json
```

位置：

```text
全文
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次sha256比较第二/三轮既有trials/statistics/calibration/replay四文件完全相同。独立文件核验，不构成独立运行复现。
```

证据摘录（纯文本）：

```text
{
  "checked_at": "2026-10-02T10:18:30.747284+08:00",
  "scope": "本次仅解析和比较既有文件；未重新运行任何项目实验。相同数据不构成独立实验复现。",
  "checks": [
    {
      "file": "trials.json",
      "round2_path": "/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0002/experiment_run/A000016/trials.json",
      "round3_path": "/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0003/experiment_run/A000031/trials.json",
      "round2_sha256": "22b5ff351a38857f17b3e018f652aa35d8a9449496b581515adb1a28e6714d88",
      "round3_sha256": "22b5ff351a38857f17b3e018f652aa35d8a9449496b581515adb1a28e6714d88",
      "exact_equal": true
    },
    {
      "file": "statistics.json",
      "round2_path": "/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0002/experiment_run/A000016/statistics.json",
      "round3_path": "/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0003/experiment_run/A000031/statistics.json",
      "round2_sha256": "de1ca1a0da6b9e62cb5ea6509b967c9821b266b22aef5703b650ddf5fb85ab50",
      "round3_sha256": "de1ca1a0da6b9e62cb5ea6509b967c9821b266b22aef5703b650ddf5fb85ab50",
      "exact_equal": true
    },
    {
      "file": "calibration.json",
      "round2_path": "/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0002/experiment_run/A000016/calibration.json",
      "round3_path": "/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0003/experiment_run/A000031/calibration.json",
      "round2_sha256": "73c5253cb42d4590696cacce150d1baf98d37416f81612f9dee914712f3e2ad1",
      "round3_sha256": "73c5253cb42d4590696cacce150d1baf98d37416f81612f9dee914712f3e2ad1",
      "exact_equal": true
    },
    {
      "file": "replay-qualification.json",
      "round2_path": "/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0002/experiment_run/A000016/replay-qualification.json",
      "round3_path": "/Users/wdblink/Code/my_repo/pyflyt-drone/paperstudio_runs/rta-20260901-external-v1/outputs/0003/experiment_run/A000031/replay-qualification.json",
      "round2_sha256": "0cd79c3572395d6724e5b872d1141548e66391876290793932977ebf70ccb2af",
      "round3_sha256": "0cd79c3572395d6724e5b872d1141548e66391876290793932977ebf70ccb2af",
      "exact_equal": true
    }
  ]
}

（其余原文不纳入摘录；阅读范围见scope）
```

### [S16] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/.project-handoff/20261002-benchmark/history-summary.json>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/.project-handoff/20261002-benchmark/history-summary.json
```

位置：

```text
119条已确认归属历史记录的全序列结构/请求与末条对外回复浏览摘要
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。宿主扫描4997个本地rollout首行元数据，无解析失败；120匹配文件含当前1；此前119文件、118唯一历史ID（7主聊天/111child），一个主聊天有2文件。3子记录无公开事件。8,495公开事件已结构浏览，不声称逐字读184MB原始日志。
```

证据摘录（纯文本）：

```text
宿主生成的覆盖摘要（非历史独立实验）：7主聊天涉及贡献指南、代码修复与ctx、旧RL论文gate、research brief、idea/PaperStudio、PaperFlow；111子任务含探索、写作、审查、审批，不能均视作实验。关键主线与实际资产深读；其余工具全文未逐项阅读。未读取分析、私有指令或宿主恢复状态。
```

### [S17] · web · read

[打开来源](<https://arxiv.org/html/2606.31073v1>)

定位：

```text
https://arxiv.org/html/2606.31073v1
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。MultiUAV-Plat (2026-06-30), §3平台/任务与§4Agent4Drone，§5结果浏览；官方README与API文档部分读，未执行。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：75会话、1500任务、9396隐藏检查；区域搜索、分配巡逻、目标分配；限制REST权限和局部感知。Agent4Drone对比ReAct。任务顺序执行，保留上一任务位置。它已直接覆盖多机任务规划与工具交互；不能再笼统宣称首次。固定翼连续物理时钟和气动能力在本次读取范围未充分核验。
```

### [S18] · web · read

[打开来源](<https://arxiv.org/html/2607.22014v1>)

定位：

```text
https://arxiv.org/html/2607.22014v1
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。MissionBench (2026-07), §1/3/4.1与结果、局限；official project page/code README读；未运行。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：120任务、5环境、4任务族；22模型主评测使用30任务Test。UE5/Cosys-AirSim，RGB输入，方向+幅度航点动作；含倒退/侧移。任务以观察、巡逻、操作、报告评分，非仅距离。部分报告用LLM judge；起点目标部分可见，不是无约束搜索。不能等同固定翼可执行性。
```

### [S19] · web · read

[打开来源](<https://arxiv.org/html/2605.01371v1>)

定位：

```text
https://arxiv.org/html/2605.01371v1
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。ESARBench (2026-05), §2/3设置与§3.3metrics读、baseline部分读；未运行。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：UE5/AirSim-Colosseum、4类GIS地形、600搜救任务，线索发现到受困者定位；含成功、时间和报告质量评价。同视觉接口检测基线可供设计参照。天气/时段任务快照不自动证明飞行中连续变化。目标搜索与递进信息获取已有近邻。
```

### [S20] · web · read

[打开来源](<https://arxiv.org/abs/2505.18229>)

定位：

```text
https://arxiv.org/abs/2505.18229
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。BEDI (2025预印本；ISPRS2026 DOI 10.1016/j.isprsjprs.2026.01.013检索元数据), abstract和官方GitHub README read；全文HTML失败、期刊页403。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：六种子能力构成动态具身任务链；同时使用真实UAV图像静态任务与UE4.27/AirSim动态场景。工具调用/任务链已有先例。摘要与README不能支持未核验的全部量化结果或固定翼实时耦合。
```

### [S21] · web · read

[打开来源](<https://arxiv.org/html/2602.00551v1>)

定位：

```text
https://arxiv.org/html/2602.00551v1
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。APEX (2026-02), §3与§4.2/附录§9部分读；未核验会议信息，未复现。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：空间语义记忆、RL动作决策和目标grounding异步运行，以避免VLM成为控制瓶颈；已支持高低层解耦思路。不是本次新方法，也不证明我方现有分层控制原型合格。
```

### [S22] · web · read

[打开来源](<https://arxiv.org/html/2605.03788v1>)

定位：

```text
https://arxiv.org/html/2605.03788v1
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。Web-of-Drones (2026-05), abstract/§1–4主要正文与实验设计读，结果部分浏览；作者稿说明WoWMoM2026。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：LLM Agent+MCP/W3C WoT+ArduPilot，覆盖、编队、灌溉任务；6LLM、每设置10次。几何工具与guardrail有益；重复相同初态主要反映推理随机性。长运行工具受理和实际完成要分开核验。
```

### [S23] · web · read

[打开来源](<https://arxiv.org/html/2510.01869v2>)

定位：

```text
https://arxiv.org/html/2510.01869v2
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。TACOS (2025/2026版本), §IV/V任务/架构/限制与摘要读；未复現实机。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：Coordinator/Supervisor与原生执行API，多无人机搜索、故障后任务重分配及仿真/实机示例。仿真实验假定全局实体信息而未实现感知，因而不能当部分可观测同条件成绩；动态任务编排已存在。
```

### [S24] · web · read

[打开来源](<https://arxiv.org/html/2606.00104v1>)

定位：

```text
https://arxiv.org/html/2606.00104v1
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。PEACE (2026-06), §III/IV与限制读；未运行ROS2/PX4。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：LLM一次生成typed工具计划，确定性执行器和失败时有限重规划，带时间戳记忆、ROS2/PX4 SITL/Gazebo；108提示8类任务为初步验证，正文明确非定量benchmark。高度/围栏约束不等于全部安全证明。
```

### [S25] · web · read

[打开来源](<https://arxiv.org/html/2607.24233v1>)

定位：

```text
https://arxiv.org/html/2607.24233v1
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。Quality-Adaptive Multi-UAV 3D Reconstruction / QUARTZ (2026-07), §I–IV部分、§Vsetup/metrics/ablation已读；official repo发现未运行。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：TSDF质量驱动多机NBV、稀疏工作量重分配；ROS/Gazebo四旋翼+LIDAR，假设完美定位/通信；3机、10次重复，并用mesh完整度、精度和F-score验证。经典方法已做质量感知与动态分工，不能当LLM独有优势。
```

### [S26] · web · read

[打开来源](<https://www.sciencedirect.com/science/article/pii/S0924271618303009>)

定位：

```text
https://www.sciencedirect.com/science/article/pii/S0924271618303009
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。A multi-UAV cooperative route planning methodology for 3D fine-resolution building model reconstruction (ISPRS2018), publisher abstract检索结果read；页面全文/DOI访问403。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：多机建筑建模采集考虑相机姿态、足够重叠、续航和作业时间等，以真实采集/建模质量验证。支持必要重叠与数据质量要求；摘要不足以支持详细算法复现。
```

### [S27] · web · read

[打开来源](<https://doi.org/10.1016/j.dt.2025.01.008>)

定位：

```text
https://doi.org/10.1016/j.dt.2025.01.008
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。Multiple fixed-wing UAVs collaborative coverage 3D path planning (Defence Technology2025), publisher abstract检索read；未读全文。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：固定翼协同覆盖规划明确考虑相机FOV/重叠、转弯半径、高度与分工。说明固定翼质量/可行路线不是空白，也支持强经典规划基线需求。
```

### [S28] · web · read

[打开来源](<https://arxiv.org/abs/2201.07030>)

定位：

```text
https://arxiv.org/abs/2201.07030
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。Cooperative Multi-UAV Coverage Mission Planning Platform for Remote Sensing Applications (2022), abstract read；未执行。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：遥感多机覆盖平台采用DARP等分区/路径优化，考虑初始位置和能力；可借鉴任务分配基线，不能直接把栅格路径当固定翼可飞航迹。
```

### [S29] · web · read

[打开来源](<https://dspace.mit.edu/entities/publication/92949acd-9009-4f4e-ae76-a730b473cef9>)

定位：

```text
https://dspace.mit.edu/entities/publication/92949acd-9009-4f4e-ae76-a730b473cef9
```

位置：

```text
标题/摘要/章节阅读范围见scope
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次2026-10-02 primary来源检索。Real-Time Multi-UAV Task Assignment in Dynamic and Uncertain Environments (AIAA2009), MIT作者机构摘要/metadata read。；未下载、安装或运行。
```

证据摘录（纯文本）：

```text
宿主阅读摘要（非逐字引文）：动态不确定环境、异构机群和障碍任务分配的经典先例。只支持研究方向/方法存在，不是本次已实现或同条件性能。
```

### [S30] · file · not_found

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/models>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/models
```

位置：

```text
当前路径存在性检查
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。本次路径不存在；无法证明本项目从未产生checkpoint或远端无模型。
```

证据摘录（纯文本）：

```text
（未读取或不可得；无证据摘录）
```

### [S31] · unknown · not_found

定位：

```text
其他设备/外部云会话/未给出的历史导出
```

位置：

```text
未查询外部主机或未知账号
```

归属依据、适用范围与读取深度：

```text
继承上一阶段2026-10-02取证；本次未重扫/重跑。这些渠道真实总量和存在性未知；本次不访问不相关项目正文。
```

证据摘录（纯文本）：

```text
（未读取或不可得；无证据摘录）
```

### [S32] · file · read

[打开来源](</Users/wdblink/Downloads/%E2%80%9C%E6%99%BA%E4%BF%A1%E2%80%942026%E2%80%9D%E6%97%A0%E4%BA%BA%E6%99%BA%E8%83%BD%E6%8C%91%E6%88%98%E8%B5%9B%E7%AB%9E%E6%8A%80%E7%B1%BB%E6%AF%94%E8%B5%9B%E8%A7%84%E5%88%99%E3%80%90%E5%AE%9A%E7%A8%BF%E3%80%91%202.doc>)

定位：

```text
/Users/wdblink/Downloads/“智信—2026”无人智能挑战赛竞技类比赛规则【定稿】 2.doc
```

位置：

```text
全文；第一部分第一/二/三条；第二部分第二条(2)(4)(5)；第四部分第二条(1)(5)；第五部分第七条。页码取本次修复字体后的18页转换预览，关键p1–7、p11–14、p18。
```

归属依据、适用范围与读取深度：

```text
用户指定的本地Word二进制.doc；原文只读SHA256 e35c8c8fe60e2ba94c90b730704c4e966269eafc07d6dae66fffff9f20b1c18a。使用bundled LibreOffice转换，初次14页预览缺中文字体不可用；任务独立fontconfig后18页预览全部目视核对，pdftotext全文读取；原文件未修改。转换版分页不是Word原生分页保证，以章节条款为准。仅私有研究参考，不在本次对外上传/发布。
```

证据摘录（纯文本）：

```text
宿主根据已读原文自写摘要，非逐字引文：赛事三个科目。科目一6机，在不超过1km²不规则区域，于25min内识别定位任务卡指定10静态+6动态目标，40–120m离地高度；对象可遮挡，移动窗口3min且路线调整，间歇通信干扰。裁判RTK真值不向参赛系统发送，无电讯接口。任务期间上报JSON，系统只读取前16个目标信息；同物理对象重复条目仅最优定位计一次。静态定位平面误差≤3m/≤5m/≤10m/>10m为3/2/1/0分，有定位分后类别正确再1分，总40。动态从首次上报开始每≥3s选点，用同时间真值平均距离误差，8×C×min(N/40,1)，C各误差档1/.6/.3/0，类别正确最多再2分，总60；原公式印成Pi与RPi逗号而非明确差向量，正文支持距离理解但不能补写成原始严谨符号。需自主返回降落，记录飞控/影像/姿态时间戳作审计；任务不暂停不重启。高度/边界累计3s扣5，晚返≤3min每机扣3，>3min整体0。通则结束返回<3机整体0、飞出区域连续>10s或落地退出，有应急人工干预则该机退出。科目二6机携20模拟投掷物按不同对象价值投送，30min、AGL30–120m；不作为本设计执行任务。科目三6机携颜色补给箱，≤5m密林穿越配对补给，25min；1/2返回机表列5/10完成分，与通则<3整体0的优先级未明，正常终点降落与中途落地退出也需独立定义。赛事风速7.9m/s为裁判可中断条件，非平台已验证风包线。第五部分第七条限制未公开赛务信息互联网传播，赛事采集全部影像/遥测/日志/通信协议等仅用于挑战赛，不能直接作为开放benchmark数据。只借鉴泛化结构，公开产物应使用自生成地图/对象/日志和重画示意。
```

### [S33] · file · read

[打开来源](</Users/wdblink/Code/my_repo/pyflyt-drone/.benchmark-design/20261002-competition/scenario-candidate.md>)

定位：

```text
/Users/wdblink/Code/my_repo/pyflyt-drone/.benchmark-design/20261002-competition/scenario-candidate.md
```

位置：

```text
全文
```

归属依据、适用范围与读取深度：

```text
宿主2026-10-02提出的待实施场景、协议、统计和论文设计候选。原文全部为建议、需开发资格校准冻结；没有训练/episode执行，也不是用户已批准具体参数。必须保留可反驳假设、阴性分支、现实性/模型公平性局限。
```

证据摘录（纯文本）：

```text
# 宿主提出的研究候选：灾后道路与河道快速核查

本文件是 2026-10-02 的待实施设计输入，全部任务参数、资格门槛、实验规模和假设均为建议；不是用户已批准的规范，不是现有代码能力或实验结果。作者应审议、解释并完善，不能原样称为事实。最终交付要从一个有意义的场景连贯推出 Agent 能力、协议、实验和论文，不停留于若干选题。

## 主场景与研究范围

建议主线：灾后道路与河道快速核查。地方巡检人员需要在固定时间内获得区域观测覆盖、静态疑似异常的定位与核查证据、以及授权巡检车辆或测量船在指定短窗口内的位置与通行状态。机群先分区搜索，再确认候选、交接短时观察、整理有限名额的可信报告，最后进入安全退出航线。使用自行生成的合成区域、异常代理物和巡检车辆，不使用比赛真实场地、影像、日志或协议。任务不包含投送、军事目标选择、攻击或真实人员持续跟踪。

意义来自信息交付：哪些路段完成核查、哪些异常值得现场复查、巡检车辆经过哪些通行段以及何时失去观测。任务卡只给类别、区域、交付要求、优先级和截止时间，不给未知对象坐标、全局对象列表或未来事件。区域三维建模原始数据采集是同一场景的扩展：检查视角/GSD/重叠/清晰度/多视图支持；只有复建管线和几何真值验收通过才可称三维重建评测。正文主实验只称观测覆盖与证据质量，避免把地面覆盖率当三维质量。

LLM Agent 是中央任务编排器，读合法局部观测及其记忆，调用航线/观察/报告工具。每架无人机的固定低层控制器、局部感知跟踪和安全保护连续运行，Agent 不输出舵面、油门、30Hz动作或每3秒的飞控命令。高级工具可以用经典覆盖规划、Dubins 路径与有限状态机执行；这些能力给所有比较方法相同。先做单中央 Agent 多飞机，分布式多Agent为可选扩展，不混入主贡献。

研究问题：在共享飞控、共享感知和工具条件下，模型/Agent 是否能把任务语义与因果可见的信息变成有效的机群分工、递进观察、证据交付和事件后重规划？加入实际推理/工具延迟后，质量、风险和成本怎样改变？不能预设 LLM 更好，也不能声称首个无人机 Agent benchmark；MultiUAV-Plat、MissionBench、ESARBench 等已经存在。

## 从规则借鉴与改写

借鉴科目一的区域搜索、静/动态对象、遮挡、有限上报、通信扰动、截止时间、结束约束及隐蔽裁判真值；借鉴科目二的一般资源分配与优先级取舍，不复现投送；科目三只借鉴已知区域/未知具体标识、能力匹配与完成后安全退出，不做5m密林穿越。

逐项区分：比赛6机、10静态+6动态、25min、40–120m AGL、前三档定位误差3/5/10m、前16个报告、动态3min每3s取样N/40，均是比赛规则而非现有平台配置或拟benchmark默认值。benchmark机群2/4为主、6为压力测试，任务时长和几何尺度由固定翼资格校准。AGL必须扣地形高程，不能当world-z或套71m场地基准；7.9m/s是赛事中止依据，不是当前飞机已验证风包线。

改写歧义/漏洞：固定观察窗口从环境对象出现开始，不从Agent首次上报开始；未发现、晚发现和缺测的时间格计0。动态180s/3s有60格，不直接用40点饱和掩盖缺测。重复报告不由裁判挑最优结果；保留因果修订、报告身份与名额消耗。明确首次报告名额与动态后续点的关系。来源平均误差公式印成Pi与RPi逗号，依正文理解距离，但不能声称原公式已经严格定义差向量。通则返回<3机记0，与科目三1/2机完成分表存在适用优先级未明；自主终点降落与任务中落地退出也须区分。benchmark独立确定唯一终止语义，不冒充赛事官方解释。

源文第五部分第七条限制未公开赛事信息传播及赛事采集数据用途。本次仅在本地私有计划中分析泛化任务结构；公开论文/开源包应仅使用自行生成数据和重画示意，不附原规则、截图、实际赛场或比赛数据。若将来要公开引用未公开规则或使用真实赛务数据，须先解决来源授权；本次不上传、不发布，也无需因此中止通用方案设计。

## 一个可执行的参考episode（全部待资格校准）

候选：4架同型固定翼，约800×800m合成不规则区域，允许转弯的外缘缓冲航带，总时长600s，AGL60–100m，标称巡航20m/s。以有间距的稳定空中初态开始，已有初始安全盘旋/搜索指令；终点是进入各机独立安全退出走廊并完成预定义盘旋验收，不等于自动降落。资格若不支持这些尺寸，应调整区域/时限/速度而不牺牲物理约束。最小转弯半径可用V²/(g tan φmax)作初估，必须用带风实际跟踪误差/机翼清距校准；不允许瞬移、原地悬停、无限转弯率。

任务卡：覆盖指定道路/河段ROI；查找并核查6个静态异常代理（3类，每类2个）和3个移动授权巡检代理；场景还含6个正常/无关相似对象；第一报告总名额B=10，后续合法修订或动态点不新增名额。这些只是参考实例，正式开发/测试需要生成变化实例，不能所有任务固定九对象答案。对象样式和目标类型是已知本体，但实体位置/出现时刻/未来轨迹不可见。

环境时间表仅供生成器和裁判：三个移动对象分别出现于120/180/240s，各180s；180s发出已知类别/区域的紧急优先级更新，210–240s一个局部通信链路中断，300–345s一机相机质量下降，360s一机任务执行能力退出但由底层保持独立安全盘旋，不模拟无可控动力坠毁。正常episode可以无扰动；单扰动与复合扰动成配对变化，正式事件时刻在开发范围随机并分出OOD时序组合，不能让所有方法预知固定timeline。

三个嵌套任务等级同一业务：L1区域覆盖+静态搜索确认；L2加入短窗口移动观察及交接；L3加入任务更新/局部通信丢失/感知退化/安全退出一机等事件。L0仅覆盖用于资格与诊断，不计为独立业务创新。2/4/6机规模按资源规范声明，测试不按比赛“少于3机0分”裁决。

## 能力与可测操纵

1. 任务理解：语义等价任务卡改写、明确优先级变更；看硬约束遵从和对应交付，不用LLM给LLM主评分。给非LLM方法同一个公开结构化任务卡；另设“自然语言→结构化”的解析子任务，不能把固定解析器的能力算给Agent。
2. 空间/资源规划：独立覆盖、任务负载平衡、重复观测的必要性、航程和退出余量；按区域形状/机群数量变化测。
3. 递进核查：不确定候选→另一视角/另一飞机确认→提交/暂缓；改变遮挡、假候选和报告名额测校准与选择。
4. 时间推理与观察交接：有限目标窗口、传感器时效、长转弯恢复；记录首发现延迟、固定窗口有效观测比例、交接缺口及身份混淆。
5. 动态重规划：相同root的无事件/有事件配对；事件通知到第一项正确生效的改派动作、最终任务损失、恢复时间。改发命令≠恢复，等待ACK≠完成；不要定义“恢复”为恢复到已不可能的原得分。
6. 工具编排与证据纪律：job状态、版本冲突、过期包、重试、取消、报告provenance；非法参数率、无证据声明率、僵死等待、预算耗尽分别归因。

## 合法信息、执行工具与连续时钟

三个边界：world/truth只给物理与裁判；onboard controller/sensor只可得到各机可见测量；missionAgent通过链路得到带观测时间、到达时间、质量/协方差、candidate_id和来源的包。actor不读target_vector、simulator对象表、未来轨迹、种子、测试生成器、评分真值或真值target_id。observer几何proxy可以内部用truth作FOV/遮挡与噪声生成，但明确为合成传感器，不叫真实视觉。

规划主实验使用共享结构化传感器：视锥+ray遮挡+距离/成像质量门槛，在实际因果可见对象上生成不确定类别、位置/速度测量和误报；噪声、误检率、定位误差、时延均要校准并预先冻结。理想可见无噪声感知仅作上界诊断。RGB/固定检测器或VLM输入是独立视觉子轨，使用相同任务root/控制器，不和结构化主表混在一个排行榜，也不把FastSAM当自动多类别语义检测器。观测流按seed+time+sensor固定，避免按方法调用次数抽随机数破坏配对。

建议最小工具：get_mission_state()仅本方公开任务/健康/观测覆盖账；list_observations(since)仅已到达包；assign_survey(drone_id,region_id,priority,expected_version)；inspect_candidate(drone_id,candidate_id,view_policy)；observe_track(drone_id,candidate_id,duration)；handoff_observation(from,to,candidate_id)；cancel_job(job_id,expected_version)；enter_safe_exit(drone_id)；submit_or_revise_report(report_id,type,estimate,evidence_ids)。航线工具读公共地图与本方测量，不知道隐藏对象。任务Agent可以申请具名候选观察，不能直接输入尚未观测的truth坐标。

工具返回accepted/rejected与job_id，真实执行状态queued/running/succeeded/failed/cancelled由低层观测推进；succeeded必须满足任务后验条件，而非动作已发出。事务防护限制互斥任务，过时expected_version拒绝；失败不耗报告名额但耗调用/时间预算。异步观察job可以在本机合法测量上自动采样并按Agent选择的报告policy上传，不要求LLM每3s调用。缓存只能保存因果包，断链时地面Agent看不到新包；本机继续旧航线/已批准观察job，恢复后队列包保留真实观测时间，不能伪造为新观测。

连续飞行主模式：sim_time按1×单调墙钟映射，实际LLM请求、工具执行、网络等待都推进任务世界；控制器线程持续按仿真步进，Agent等待时继续原job或安全盘旋。物理积分频率和Agent轮询分别记录，不把120的env分步基数直接宣称物理Hz。框架一旦达不到1×物理步进，应判为基础设施问题并定位，不偷偷暂停目标或慢放任务。物理/感知/Agent预算统一，单方法串行请求避免外部资源竞争；缓存的LLM回复只能用于审计/诊断并标回放，不能宣称真实实时模型性能。

另设可复现的延迟注入模式：固定延迟0/2/5/10/20s，仿真可加速但计算时钟必须加等价延迟、控制/目标推进；用来分离规划逻辑和速度代价。主实时结果包括各模型实际p50/p95延迟、APItimeout和cost；固定注入表只能支持受控延迟结论。每episode调用上限候选60次和token预算候选60000先开发校准；所有方法工具同预算，非LLM规划compute也计时；API故障不能免费从头跑，只记录原attempt并按冻结重试规则执行一次有限重试或失败。

## 可审计评分候选

先冻结实现可判定定义，再看模型结果。保留分量，避免单一分数隐藏低覆盖/高风险。以下仅待开发校准的协议建议。

覆盖C：指定ROI中合格观测cell的面积比例[0,1]。合格须有视锥内无遮挡、符合分辨率/俯视角/清晰度及时间戳的真实采样证据；飞过cell不等于看见。重复R：超过任务要求的重复合格采样/全部合格采样；用于消融解释，合理多视图确认不视为无条件负分。C不等于三维重建质量。

报告：create首次合法schema接受消耗不可退还一格；同report_id的修订不新消耗，但每次估计绑定当时真实观测证据，不能改观测时间。可撤回但不释放名额。重复新ID消耗格，计入重复/虚警；裁判不给正确性反馈。隐藏匹配在首次创建时按观测时刻的位置与有效对象作空间门限的一对一关联，优先接收序，最近距离、平距按裁判ID稳定决胜，阈值候选20m；真值ID不给actor，类别不用于匹配。先前已关联的对象不再给新报告ID分配；误报关联到普通对象不变成相关目标，未匹配报告不后来用未来真值补救；合法同ID修订可改善静态估计和以后动态点，不能将对象身份换成另一个实体。若对象距离小于门限且真值关联不可靠，单独报告关联歧义率并修订开发匹配方案后冻结，不能静默当Agent失败。静态开发实例先保证最小间距大于2倍关联门限；动态交叉轨迹要资格测试，主任务不以视觉re-ID创新作为前提。

静态S：每个相关目标用截止前最后一个合法已到达报告版本计算平面位置误差e；q(e)=1(e≤3m), 2/3(3<e≤5), 1/3(5<e≤10),0(e>10)。s_j=q(e_j)*I(类别正确且证据有效)，未提交/不可匹配为0，S按任务卡目标权重归一平均。3/5/10来自比赛参考，是否可达须根据合成测量/相机校准，正式阈值可变但冻结且不看测试模型成绩选阈值；同时报告定位连续误差、召回/precision与类别准确率。区分“看到异常代理”与“确认真实灾害”的外部效度。

动态D：从环境定义的出现时刻起，180s窗口固定60个3s格t_b；格中用截至t_b已采集最新合法测量，观测年龄≤3s、类别正确、证据有效，且数据/报告到达地面mission服务≤t_b+6s才有效。每个合法新测量最多支持一个格（固定时间格，不能将一条旧包反复当新观测）；平面误差q(e)同静态。d_j=(1/60) sum_b q(e_jb)*I(valid_jb)，D为移动目标任务权重归一平均。缺测/未发现/迟报/身份错误为0；插值与运动模型预测另列预测质量，不计为新观测。上传可自动job执行。同时报告D_observed（机载采集有效，允许任务结束前上传）与D_delivered（上述及时到达，主D）区分感知丢失与通信交付损失；及时性阈值6s要开发业务校准。初始发现延迟不会因首次报告改变起点被隐藏。观察任务调度与3s传感器频率必须分离。

安全H：碰撞、地形侵入/坠毁、超出预注册硬包线等任一发生则0，否则1；硬包线每物理步检查并按duration记录，数值抖动阈值须开发校准冻结，不能用累计3s就允许任意危险。命令拒绝/安全保护介入不自动当事故，单列拦截频率和原因。终态E：所有非外生退出的机在截止前进入预定义安全退出状态、剩余航程预算满足退出余量；外生退出机需在独立安全盘旋且已释放任务，Agent导致失联/失败不豁免。terminal E是飞行中任务交接结束，不是落地或真实续航证明。代理资源先用明确航程/时间预算，不冒充真实电池能耗。

主连续指标Y=H*E*Q，L2/L3中Q=(C+S+D)/3，L1 Q=(C+S)/2；等权是透明的benchmark汇总约定而非真实业务已校准效用，不跨等级直接排名。主表必须同时给C,S,D,Q,H,E,Y，提供权重敏感性和Pareto图。安全任务成功SR=I(H=1,E=1,C≥.8,S≥.6,D≥.5)（L1不含D）；阈值待开发资格冻结。H*E门控会形成0值集中，因此必须同时显示原始Q和失败原因，不能拿安全sample积分当episode安全率。报告precision、虚警、重复、首有效报告、有效观察占比、最大交接缺口、距离、任务退出比例、硬越界秒数、最小机间距、工具错误、被保护器救援比例和token/APIcost是诊断，避免全部做显著性主张。

## 平台资格与最小实施缺口

当前项目是低层研究资产，不是已就绪benchmark。共享Aviary多机setpoints基础可用；task wrappers主要单机，旧ObjLock奖励是2m接近目标且obs有truth target_vector；高层PPO原型存在环境共享/目标字段/观测方法/动作维度衔接问题；没有本地合格控制checkpoint。旧RTA runner无LLM、12s理想几何局部包、4roots×2draws×8conditions64trials，不能支撑Agent结论。

阶段G1：固定版本/依赖清单与有效频率；选最小合格航迹控制器（先经典autopilot而非必须重新RL训练）；在开发root单机、多机、盘旋/转弯/出航/退出/风下验证跟踪与安全。候选每族≥20roots，总≥100开发episode，正常可行任务≥95%无硬违规且完成；资格数值需冻结且失败的root不能按模型事后删。低层难以通过则缩小包线/几何区域、暂不做对应扰动，不制造必失败任务。

G2：同步机群、对象运动、AGL/坐标、相机ray、因果信息隔离、工具job生命周期；裁判手工小fixture验证边界点、重复、名额、迟报、通信恢复、终态豁免等最小正负例；地面和机载权限分离。G3：结构化传感器误差与虚警校准、时间/物理公平性检查、参考策略在合法观测下获得明显非零任务收益；另外带特权完整未来信息的方法仅作可行性参照，不给主baseline。G4：dev-only调整并冻结生成器/指标/提示/预算，之后正式test只运行与分析，不改阈值以提高LLM分数。

用户这次只授权调研与方案；上述实现/测试/训练均待下一阶段执行，本次不能写成已跑。根目录只新增计划、运行目录，保留PROJECT_HANDOFF与旧代码/ctx/dirty文件。公开可复现包日后需task schema、生成器、seed split、qualifier、工具协议、grader、baseline、environment/version manifest、合法观测与动作日志、回放摘要和license（模型回应保存按各服务可用条件）。

## 比较方法、实验矩阵和统计

最小强基线：B0固定分区覆盖+规则候选确认/报告；B1在线utility/auction或CBBA-inspired分配+滚动重规划（若不实现原论文算法就不叫完整CBBA复现），使用同公共地图、局部包和工具；B2工程有限状态/行为树+事件重分配，同样预算。三者通过dev调参，并具备目标确认/安全退出/报告机制，不能把无报告的割草策略当唯一定量对手。B3统一ReAct（多个模型同工具/同prompt）；B4计划→执行→监控→事件重规划Agent，仅改变Agent编排；B5可选LLM解释任务卡+经典调度混合。特权oracle参照不参与公平胜负。固定低层控制器；共享structured sensor为主。自然语言解析、RGB/VLM、多Agent分别附加研究，不互混主结论。

模型至少三个供应商/模型家族、含一个可固定版本开放权重模型；具体名称/版本/价格在执行日期读取官方信息冻结，本次不选未经核验最新型号。主横比全部跑统一ReAct，编排对比用同一固定模型，混合模型算独立方法不偷换模型。temperature、reasoning等级、上下文长度、工具JSON格式、retry及缓存写入manifest；无法给deterministic seed的API用独立repeat并报告。

设计三个问题/预注册可反驳假设：H1事件发生后B4相对同模型B3提高Y，但可能仍不及B1/B2；H2对假候选与名额紧缺，递进核查能提高报告precision/Y，但过量确认降低C/D；H3连续时钟降低高延迟方法有效动态交付，事件监控能减少但不保证消除下降。效应最小阈值例如Y差+.05仅作power候选，先开发pilot估方差和最低有意义业务变化，不复制旧+.5秒阈值。阴性结果同样支持benchmark识别能力边界。

数据拆分候选：dev24独立root（任务调参/协议校准），test-ID48root（按L1/L2/L3各16），test-OOD24root（L2/L3各12），另保留完全不触碰的确认holdout12root（L2/L3各6）；root=独立地形道路图/对象布局及运动随机族，不是同root不同dropout或LLM重复。相似布局的变换与无/有事件兄弟episode同root同split；资产/纹理/任务卡等价改写组不得跨split泄漏。OOD按新道路图族/遮挡或通信模式/事件组合定义，避免只换seed；只在开发范围内做已资格通过的物理风扰动。测试支持精度有限，不能称全世界灾后场景代表性。

主评测：4机+structured sensor+实际连续时钟；每root每Agent方法3repeat，classic方法如环境/算法确定只跑1但不可重复拷贝3当独立数据，统计先root内汇总再方法配对。主要统计在ID/OOD分别报告不合并掩盖；按等级宏平均再总合成，避免L1/L3数量决定结论。候选5个LLM方法×(48+24)roots×3=1080episodes，3classic×72=216，共1296；10min每ep约216个飞行墙钟小时，未计LLM排队/复跑/资格（可并行world但资源竞争要测量并分层/随机顺序），不能称廉价半天完成。holdout还需5×12×3+3×12=216episodes约36小时。先6root pilot测每ep token/cost/RTF，预算=各方法实测tokens×模型单价+工具计算+无免费重跑；执行前再冻结规模，不承诺当前机器可满足。

受控消融在test专门选定L3的12roots、同模型B4三个repeat，每次仅移除一个因素：事件触发改为固定周期；跨机candidate去重/记忆；递进确认；age/provenance检查；安全保护关闭只能在隔离仿真诊断并标unsafe、不得主排行推荐。做remove-age可能产生被grader拒绝的stale报告以示协议价值，grader不随消融改变。主要消融3项就可，其他扩展不无限堆砌。4/6规模、B=6/10/14、0/2/5/10/20s延迟、RGB/传感器噪声敏感性选清晰小子集，不全因子乘积。同root负例“无变化就不该大量重规划”衡量任务抖动及额外成本；构造一次错误ACK/过期job让系统检查executor而非模型词语。

统计：主endpoint root平均Y，主对比B4-B3（同模型），按root配对bootstrap 95%CI（或root配对置换），重复/缺失bin/各drone不是独立样本。事件兄弟root保持同组bootstrap。预声明三个主要对比用Holm校正；C/S/D/成本及failure类别为描述诊断，不事后挑p显著指标。公开每root差值、均值/中位、CI、效应、safe-task SR及Wilson/cluster CI；SR有嵌套repeat采用rootclusterCI而非把所有repeat独立Wilson。事件损失用Y_event-Y_clean，响应/恢复时间未达成按deadline右删失并报告删失率，不能只平均成功案例。所有非基础设施策略失败计入；经证明平台crash/服务故障按预注册invalid_attempt标签保留且所有关联方法对称处理，不按低分删除。提供固定截止的成本-质量Pareto与权重敏感性，不能只报最高得分的一次。

## 论文写作、图表、交付及结果分支

暂定英文题名建议：Evaluating LLM Mission Agents for Multi-UAV Inspection under Continuous Flight and Delayed Feedback。不凭未核验缺口宣称first或SOTA。中心论证：真实业务信息需求引出部分观测/有限报告/同步固定翼/连续时钟；协议用共享低层隔离高层决策；结果给任务质量-安全-延迟边界。固定翼本身、异步调用、Dynamic Workflow名字都不是单独新贡献；benchmark贡献是可审计任务/协议/生成器/强基线及有证据的能力诊断。

引言4段：具体巡检交付问题；为何成熟飞控之外仍有任务编排与变化；最近MultiUAV-Plat/MissionBench/ESARBench等已做什么、此协议组合的差别与尚需复核项；本文有限贡献与RQ。Related Work按任务与评测边界而非论文列表（UAV missionAgents、多机在线分配/coverage、连续物理与延迟、证据交付/benchmark设计）。Benchmark节先业务、实体任务、信息/工具/时间，再generator与split、指标和资格。Baselines与实验节共享条件、主对比及统计。Results按RQ解释主表、事件、延迟、误报/交接，不按模型名逐个赞评。Limitations明确合成灾害代理/感知、终态非落地、控制器资格、无实机/真实电池、模型版本漂移和小样本外推。Conclusion仅已观测结论，不预写LLM优势。

计划图表：Fig1场景-观察-决策-执行-隐藏grader架构；Fig2单episode可见事件/任务变更/确认/交接/退出时间线（无隐藏truth泄露到actor）；Fig3主Q/Y与C/S/D分量及CI；Fig4事件前后覆盖/动态有效采样与handoff缺口曲线；Fig5实际/注入延迟质量成本Pareto；Fig6两类失败case合法包→tool/job→物理/报告后果。Table1最近工作逐机制阅读矩阵（未知就未知），Table2任务生成与资格/阈值，Table3主baselines/模型/ID-OOD/安全/成本，Table4消融与失败类型。图不用美化负结果，100%有trace来源。实际写稿前保存claim→figure→rawrun mapping，不把现有旧64trials图当新benchmark数据。

结果分支：若LLM只胜简单策略不胜在线经典，论文写清在哪些semantic/恢复条件接近或失败；若一律不胜但诊断稳定，可做failure/latency benchmark报告，不包装新agent方法；若Agent与平台事故相关，应先修资格，不能把低层失败都称模型能力；若只在paused/零延迟有优势，主结论是部署时延边界；若任务易而各法饱和，dev期调整难度，test饱和不得再调测试后宣称泛化。若创新矩阵不能区分最近平台，缩窄为固定翼连续巡检协议扩展或跨平台复核研究，而非首创。

阶段交付建议按门槛而非保证日期：P0本次规则审读+文献/资产定位+完整计划；P1 G1/G2合格控制与同步world（估1–2周需实际人力）；P2 G3/G4 sensor/grader工具fixture与pilot冻结（估1–2周）；P3正式ID/OOD/main+小消融（按实测计算预算，估1–2周）；P4holdout、claim审计、结果图/正文/复现包（估1–2周）。日期是估计，任何资格失败顺延且禁先大规模LLM跑分。每阶段必须写实际通过证据和未通过项。首个下一阶段交付应是4机合法航线→因果候选包→确认/报告→安全退出的一条非LLM参考episode和graderfixture；之后才接LLM。这次交付完整方案，不声称该episode已实现。

```
