# 机制近邻边界记录 v2（研究材料，非论文正文）

已定向阅读任务/接口/时间语义段；不宣称全部论文已复现。前七篇的身份与有限主张见 `packet.first-neighbors.v7.json` 和机械citation-check；该检查不负责外部真实性或新颖性判断。SMART/GRaD-Nav++使用已校准PDF的明确版本，本轮补读方法段，正式citation身份记录仍待补全。

| 来源 | 已有机制与证据范围 | 对本研究贡献主张的约束 |
| --- | --- | --- |
| MultiUAV-Plat v1 | 多机公共工具、角色权限、隐藏验证、编排与ReAct对照；既有六文件代码抽查还见后台目标状态更新。 | 不能声称首次多机LLM平台/编排或断言其世界全程暂停；其推理等待期间的完整动力学时钟仍未核实。 |
| MissionBench v1 | 单指令多任务族、连续幅值动作、闭环观察与报告，按任务验收。 | 不能声称首次闭环航空任务或显式报告；连续幅值不自动等于推理期间持续的物理时钟。 |
| ESARBench v1 | SAR图像环境与Event-Snapshot-Task；每snapshot内受害者/线索位置固定。 | 仅据方法区分目标变化协议；不能把目标静止说成整个AirSim暂停。 |
| Retriever v2 | 具显式时钟、同步/陈旧策略和记录式重放的异步因果流运行时，真实机器人案例。 | 通用异步、因果接口、多速率和日志重放均有明确先例；本研究不能将这些作为首创。 |
| α³-Bench v1 | 对话式UAV工具与网络状态；A2A在对话层异步且逻辑即时。 | 网络/工具约束已有基准；对话状态语义不能当成持续飞行动力学实证。 |
| PhysAI-Bench v1 | 从任务trace抽取决策前缀，评估多选行动选择。 | 因果前缀输入已有先例；该离线选择任务与本研究在线策略交付不同，不横比模型分数。 |
| AERIAL v1 | SITL实测场景—模型映射与任务代码生成；生成时延指标排除SITL任务执行。 | 不声称首次UAV时延/模型成本评测；区分固定模型下的编排比较与按场景选模型。不能由该时延定义推出SITL等待时暂停。 |
| SMART arXiv2503.04798v3 | 方法III-B/C：ADG状态staged/enqueued/finished、RPC进度回执、机器人独立队列与控制器、并行执行。 | 执行监测、ACK/完成区分、延迟安全与物理执行已有先例；其任务是MAPF路径执行。本条不把它误叫LLM多机任务论文。 |
| GRaD-Nav++ arXiv2506.14009v2 | 方法III-B1：CLIP与策略异步，每10步更新特征；III-D实际飞行onboard profile，IV-C含途中指令切换。 | 异步视觉语言与低层飞控并行、实际推理开销和途中切换已有先例；本研究仅可能贡献限定的多机工具/证据交付与受控时延评测协议。未复现其系统或沿用其数字作本研究结果。 |

后两行新增阅读定位来自 `text/smart-calibration-version.v1.txt`（III-B/C）与 `text/gradnav-calibration-version.v1.txt`（III-B1、III-D、IV-C），PDF/转换hash见acquisition/mechanism-comparator-text-conversion.v1.json。它们的公开稿年份、v2/v3更新时间和RA-L最终出版版本须分别核验，不以PDF页眉的模板或preprint页数替代正式记录。

仍待工作：补齐后两项citation记录及来源审查；逐项判断T1中的未知机制格；不能在G4或完整正式结果前用本表宣称新的benchmark有效、LLM获胜或审稿已通过。
