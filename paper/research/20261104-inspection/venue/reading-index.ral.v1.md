# Representative-paper reading index

8 papers, 47 figures, 26 tables, 130 facets or semantic blocks.

Counts describe the supplied records. Visual reading and semantic judgments remain attributed declarations.

## S01: Aerial Gym Simulator

7 figures; 2 tables; 33 facets or semantic blocks.

[Source PDF](samples/S01-aerial-gym.pdf)

### S01-F1 (figure)

平台提供哪些可验证能力？

场景、传感器和控制部署形成一条功能链。

Evidence: 首图的机型、传感器输出与实机走廊拼图

Design: 首图同时交代输入、执行和验证

Limitations: 功能展示不是任务成功率

[PDF page 1](samples/S01-aerial-gym.pdf#page=1)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S01-F1 / top

多种机体与环境揭示什么？

该示例中可直接定位：多种机体与环境。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：多种机体与环境。

Design: 首图同时交代输入、执行和验证

Limitations: 功能展示不是任务成功率

#### S01-F1 / bottom-left

自定义几何/语义传感器输出揭示什么？

该示例中可直接定位：自定义几何/语义传感器输出。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：自定义几何/语义传感器输出。

Design: 首图同时交代输入、执行和验证

Limitations: 功能展示不是任务成功率

#### S01-F1 / bottom-right

深度导航实机展示揭示什么？

该示例中可直接定位：深度导航实机展示。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：深度导航实机展示。

Design: 首图同时交代输入、执行和验证

Limitations: 功能展示不是任务成功率

### S01-F2 (figure)

仿真、控制、任务与学习如何衔接？

共享张量字典连接各管理模块与任务接口。

Evidence: 横跨两栏的模块图及双向箭头

Design: 把共享执行器与Agent接口分层

Limitations: 该全局训练数据通路不能照搬为Actor权限

[PDF page 3](samples/S01-aerial-gym.pdf#page=3)

### S01-F3 (figure)

机体支持范围是什么？

图示包含刚性、柔性与可重构多旋翼。

Evidence: a–e五种机体渲染

Design: 明确已验证机型集合

Limitations: 不支持固定翼资格结论

[PDF page 3](samples/S01-aerial-gym.pdf#page=3)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S01-F3 / a

刚性四旋翼揭示什么？

该示例中可直接定位：刚性四旋翼。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第3页：刚性四旋翼。

Design: 明确已验证机型集合

Limitations: 不支持固定翼资格结论

#### S01-F3 / b

平面八旋翼揭示什么？

该示例中可直接定位：平面八旋翼。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第3页：平面八旋翼。

Design: 明确已验证机型集合

Limitations: 不支持固定翼资格结论

#### S01-F3 / c

非平面八旋翼揭示什么？

该示例中可直接定位：非平面八旋翼。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第3页：非平面八旋翼。

Design: 明确已验证机型集合

Limitations: 不支持固定翼资格结论

#### S01-F3 / d

柔性Morphy揭示什么？

该示例中可直接定位：柔性Morphy。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第3页：柔性Morphy。

Design: 明确已验证机型集合

Limitations: 不支持固定翼资格结论

#### S01-F3 / e

可重构机体揭示什么？

该示例中可直接定位：可重构机体。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第3页：可重构机体。

Design: 明确已验证机型集合

Limitations: 不支持固定翼资格结论

### S01-F4 (figure)

合成传感器实际输出哪些量？

相机与两类激光雷达输出不同几何/语义通道。

Evidence: a、b及各四个子视图

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

[PDF page 5](samples/S01-aerial-gym.pdf#page=5)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S01-F4 / a

相机四种通道揭示什么？

该示例中可直接定位：相机四种通道。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：相机四种通道。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / a.i

深度图揭示什么？

该示例中可直接定位：深度图。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：深度图。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / a.ii

表面法向图揭示什么？

该示例中可直接定位：表面法向图。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：表面法向图。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / a.iii

分割图揭示什么？

该示例中可直接定位：分割图。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：分割图。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / a.iv

面片索引揭示什么？

该示例中可直接定位：面片索引。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：面片索引。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / b

两类LiDAR通道揭示什么？

该示例中可直接定位：两类LiDAR通道。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：两类LiDAR通道。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / b.i

OS0分割揭示什么？

该示例中可直接定位：OS0分割。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：OS0分割。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / b.ii

OS0距离揭示什么？

该示例中可直接定位：OS0距离。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：OS0距离。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / b.iii

Dome分割揭示什么？

该示例中可直接定位：Dome分割。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：Dome分割。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

#### S01-F4 / b.iv

Dome距离揭示什么？

该示例中可直接定位：Dome距离。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：Dome距离。

Design: 示例输出须与接口字段对应

Limitations: 可视化不等于感知误差校准

### S01-F5 (figure)

不同控制抽象的跟踪能否与仿真相符？

位置误差、指令与速度被联合比较。

Evidence: 速度/加速度两列，实线实机与仿真统计带

Design: 控制资格图并列误差与控制行为

Limitations: 两列任务不是独立场景样本

[PDF page 7](samples/S01-aerial-gym.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S01-F5 / velocity.error

速度控制位置误差揭示什么？

位置误差在展示轨迹中逐步趋近零。

Evidence: 实际查看PDF第7页：速度控制位置误差。

Design: 控制资格图并列误差与控制行为

Limitations: 两列任务不是独立场景样本

#### S01-F5 / velocity.command

速度控制策略指令揭示什么？

速度/偏航命令经历瞬态后回落。

Evidence: 实际查看PDF第7页：速度控制策略指令。

Design: 控制资格图并列误差与控制行为

Limitations: 两列任务不是独立场景样本

#### S01-F5 / velocity.state

速度控制实际速度揭示什么？

测得速度随闭环指令趋于稳定。

Evidence: 实际查看PDF第7页：速度控制实际速度。

Design: 控制资格图并列误差与控制行为

Limitations: 两列任务不是独立场景样本

#### S01-F5 / acceleration.error

加速度控制位置误差揭示什么？

加速度抽象也表现出误差收敛。

Evidence: 实际查看PDF第7页：加速度控制位置误差。

Design: 控制资格图并列误差与控制行为

Limitations: 两列任务不是独立场景样本

#### S01-F5 / acceleration.command

加速度控制策略指令揭示什么？

加速度命令的瞬态形式与左列不同。

Evidence: 实际查看PDF第7页：加速度控制策略指令。

Design: 控制资格图并列误差与控制行为

Limitations: 两列任务不是独立场景样本

#### S01-F5 / acceleration.state

加速度控制实际速度揭示什么？

该示例最终速度回落，不能由此证明全部包线合格。

Evidence: 实际查看PDF第7页：加速度控制实际速度。

Design: 控制资格图并列误差与控制行为

Limitations: 两列任务不是独立场景样本

### S01-F6 (figure)

深度策略能否在走廊执行？

例子连接机载深度、指令与实机轨迹。

Evidence: 左侧深度嵌图及右侧轨迹叠图

Design: 合法感知到实际轨迹须可追溯

Limitations: 单次展示不能给出安全率

[PDF page 7](samples/S01-aerial-gym.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S01-F6 / depth

传感器深度嵌图揭示什么？

该示例中可直接定位：传感器深度嵌图。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：传感器深度嵌图。

Design: 合法感知到实际轨迹须可追溯

Limitations: 单次展示不能给出安全率

#### S01-F6 / onboard

目标及当前控制方向揭示什么？

该示例中可直接定位：目标及当前控制方向。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：目标及当前控制方向。

Design: 合法感知到实际轨迹须可追溯

Limitations: 单次展示不能给出安全率

#### S01-F6 / trajectory

完整走廊航迹揭示什么？

该示例中可直接定位：完整走廊航迹。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：完整走廊航迹。

Design: 合法感知到实际轨迹须可追溯

Limitations: 单次展示不能给出安全率

### S01-F7 (figure)

仿真和实机状态/动作分布哪里不同？

误差和速度较接近，推力仍存在域差异。

Evidence: 实机/仿真两列各三行

Design: 同时保留接近和偏离的状态量

Limitations: 五训练seed不等于五独立任务root

[PDF page 7](samples/S01-aerial-gym.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S01-F7 / real.error

实机位置误差揭示什么？

真实跟踪存在瞬态，但整体贴近所示仿真范围。

Evidence: 实际查看PDF第7页：实机位置误差。

Design: 同时保留接近和偏离的状态量

Limitations: 五训练seed不等于五独立任务root

#### S01-F7 / sim.error

仿真位置误差揭示什么？

仿真对设定点变化产生规律瞬态。

Evidence: 实际查看PDF第7页：仿真位置误差。

Design: 同时保留接近和偏离的状态量

Limitations: 五训练seed不等于五独立任务root

#### S01-F7 / real.thrust

实机各电机推力揭示什么？

实机各电机推力存在持续不均衡。

Evidence: 实际查看PDF第7页：实机各电机推力。

Design: 同时保留接近和偏离的状态量

Limitations: 五训练seed不等于五独立任务root

#### S01-F7 / sim.thrust

仿真各电机推力揭示什么？

仿真电机推力较对称，显示未完全覆盖硬件差异。

Evidence: 实际查看PDF第7页：仿真各电机推力。

Design: 同时保留接近和偏离的状态量

Limitations: 五训练seed不等于五独立任务root

#### S01-F7 / real.velocity

实机速度揭示什么？

真实速度响应包含可见峰值与恢复。

Evidence: 实际查看PDF第7页：实机速度。

Design: 同时保留接近和偏离的状态量

Limitations: 五训练seed不等于五独立任务root

#### S01-F7 / sim.velocity

仿真速度揭示什么？

仿真速度峰值与实机相近但并非逐点一致。

Evidence: 实际查看PDF第7页：仿真速度。

Design: 同时保留接近和偏离的状态量

Limitations: 五训练seed不等于五独立任务root

### S01-T1 (table)

吞吐比较是否同时保留功能差别？

高吞吐伴随不同传感器和碰撞建模能力。

Evidence: 按实例规模和传感器功能分列

Design: 吞吐必须与任务能力共同报告

Limitations: 硬件与仿真语义不同限制直接排名

[PDF page 6](samples/S01-aerial-gym.pdf#page=6)

### S01-T2 (table)

分辨率和并行规模如何影响渲染？

速度随分辨率与环境数变化，保留崩溃格。

Evidence: 两套渲染器的分辨率×实例矩阵

Design: 失败格不删；报告RTF与积压

Limitations: 该FPS不等于端到端Agent实时性

[PDF page 6](samples/S01-aerial-gym.pdf#page=6)

Reading declaration: {"method": "rendered_pdf", "notes": "Codex primary于2026-11-05实际查看120dpi渲染页。每个测得视觉均有独立判断；未识别表格另列manual-visuals。公开PDF版本与最终出版版差异另见sample-version-audit。", "pages": [1, 3, 5, 6, 7]}

## S02: LiP-LLM

2 figures; 5 tables; 6 facets or semantic blocks.

[Source PDF](samples/S02-lip-preprint.pdf)

### S02-F1 (figure)

语言推理与优化各承担什么？

技能、依赖图与线性分配被显式分开。

Evidence: 三列技能生成→依赖→任务分配

Design: 画清LLM与经典调度职责

Limitations: 此图没有持续飞行时钟

[PDF page 2](samples/S02-lip-preprint.pdf#page=2)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S02-F1 / skills

LLM选择可用技能揭示什么？

该示例中可直接定位：LLM选择可用技能。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第2页：LLM选择可用技能。

Design: 画清LLM与经典调度职责

Limitations: 此图没有持续飞行时钟

#### S02-F1 / dependencies

显式先后依赖图揭示什么？

该示例中可直接定位：显式先后依赖图。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第2页：显式先后依赖图。

Design: 画清LLM与经典调度职责

Limitations: 此图没有持续飞行时钟

#### S02-F1 / allocation

优化分配与执行揭示什么？

该示例中可直接定位：优化分配与执行。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第2页：优化分配与执行。

Design: 画清LLM与经典调度职责

Limitations: 此图没有持续飞行时钟

### S02-F2 (figure)

实验复杂度如何改变？

三个场景改变机械臂、移动机器人与房间配置。

Evidence: a双臂、b双臂加移动、c多房间

Design: 场景变化必须对应受控因素

Limitations: 室内操控不等于机群巡检

[PDF page 5](samples/S02-lip-preprint.pdf#page=5)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S02-F2 / a

双臂单桌揭示什么？

该示例中可直接定位：双臂单桌。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：双臂单桌。

Design: 场景变化必须对应受控因素

Limitations: 室内操控不等于机群巡检

#### S02-F2 / b

双臂加移动机器人揭示什么？

该示例中可直接定位：双臂加移动机器人。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：双臂加移动机器人。

Design: 场景变化必须对应受控因素

Limitations: 室内操控不等于机群巡检

#### S02-F2 / c

多房间、多臂、多移动机器人揭示什么？

该示例中可直接定位：多房间、多臂、多移动机器人。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：多房间、多臂、多移动机器人。

Design: 场景变化必须对应受控因素

Limitations: 室内操控不等于机群巡检

### S02-T1 (table)

指令扰动的内容与分母是什么？

四组包含基本、命名、歧义与新增对象。

Evidence: 组别/细节/例子/trial列

Design: 预先定义任务变化与数量

Limitations: 语言改写不是独立物理root

[PDF page 5](samples/S02-lip-preprint.pdf#page=5)

### S02-T2 (table)

不同指令组的成功、效率、时间如何权衡？

方法优势依赖组别，指标并列。

Evidence: G1–G4与A/B环境的SR/SPL/Time

Design: 展示质量与时间而非单个优点

Limitations: 文中时间只对成功实例统计，不用于本研究主分母

[PDF page 6](samples/S02-lip-preprint.pdf#page=6)

### S02-T3 (table)

机器人约束是否改变结论？

约束分层后保留各方法三个指标。

Evidence: C1–C3与A/B环境的结果块

Design: 对硬约束分层汇报

Limitations: 不能把SPL当真实飞行航程

[PDF page 6](samples/S02-lip-preprint.pdf#page=6)

### S02-T4 (table)

更复杂指令下是否仍全面占优？

成功率与过程时间的优势并不一致。

Evidence: C场景L1–L3中SMART-LLM有更短过程时间

Design: 明确保留速度与成功的相反排序

Limitations: 不能只用成功案例时间支持部署结论

[PDF page 7](samples/S02-lip-preprint.pdf#page=7)

### S02-T5 (table)

复杂场景约束增加后表现如何？

成功和效率提升与更高过程时间并存。

Evidence: C场景C1–C3表

Design: 主表应同时给质量和资源代价

Limitations: 预印本9页，不能视为RA-L最终页数惯例

[PDF page 7](samples/S02-lip-preprint.pdf#page=7)

Reading declaration: {"method": "rendered_pdf", "notes": "Codex primary于2026-11-05实际查看120dpi渲染页。每个测得视觉均有独立判断；未识别表格另列manual-visuals。公开PDF版本与最终出版版差异另见sample-version-audit。", "pages": [2, 5, 6, 7]}

## S03: NL2HLTL2Plan

6 figures; 4 tables; 25 facets or semantic blocks.

[Source PDF](samples/S03-nl2hltl-preprint.pdf)

### S03-F1 (figure)

多机器人时序依赖如何可视化？

离散时间快照对应多对象并行执行。

Evidence: 九帧地图、对象颜色和t标签

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

[PDF page 1](samples/S03-nl2hltl-preprint.pdf#page=1)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S03-F1 / t1

初态揭示什么？

该示例中可直接定位：初态。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：初态。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

#### S03-F1 / t16

机器人3拾取钥匙揭示什么？

该示例中可直接定位：机器人3拾取钥匙。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：机器人3拾取钥匙。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

#### S03-F1 / t18

机器人3放置钥匙揭示什么？

该示例中可直接定位：机器人3放置钥匙。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：机器人3放置钥匙。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

#### S03-F1 / t31

机器人1拾取棒球棒、机器人4接近铅笔揭示什么？

该示例中可直接定位：机器人1拾取棒球棒、机器人4接近铅笔。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：机器人1拾取棒球棒、机器人4接近铅笔。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

#### S03-F1 / t33

机器人1接近床、机器人4拾取铅笔揭示什么？

该示例中可直接定位：机器人1接近床、机器人4拾取铅笔。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：机器人1接近床、机器人4拾取铅笔。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

#### S03-F1 / t37

棒球棒放床、机器人4接近边桌揭示什么？

该示例中可直接定位：棒球棒放床、机器人4接近边桌。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：棒球棒放床、机器人4接近边桌。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

#### S03-F1 / t47

机器人1拾取手机、机器人4接近边桌揭示什么？

该示例中可直接定位：机器人1拾取手机、机器人4接近边桌。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：机器人1拾取手机、机器人4接近边桌。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

#### S03-F1 / t50

机器人1接近扶手椅、机器人4放置铅笔揭示什么？

该示例中可直接定位：机器人1接近扶手椅、机器人4放置铅笔。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：机器人1接近扶手椅、机器人4放置铅笔。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

#### S03-F1 / t57

手机放扶手椅、机器人4接近边桌揭示什么？

该示例中可直接定位：手机放扶手椅、机器人4接近边桌。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：手机放扶手椅、机器人4接近边桌。

Design: 用事件/观察时间串联可审计轨迹

Limitations: 离散步数不是墙钟推理时延

### S03-F2 (figure)

层次任务表示如何连接语言和规划？

任务树、逐层指令与LTL一一对应。

Evidence: 提示列与输出树/层次公式列

Design: 表示转换须标出可核验中间产物

Limitations: 形式规格不自动保证物理执行

[PDF page 2](samples/S03-nl2hltl-preprint.pdf#page=2)

### S03-F3 (figure)

对比方法的结构差别是什么？

直接代码生成与翻译器加规划器被并列。

Evidence: 上下两条共享状态/技能输入的通路

Design: 比较架构必须明确共享输入

Limitations: 不能由结构图证明性能增益

[PDF page 5](samples/S03-nl2hltl-preprint.pdf#page=5)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S03-F3 / top

SMART-LLM直接代码计划揭示什么？

该示例中可直接定位：SMART-LLM直接代码计划。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：SMART-LLM直接代码计划。

Design: 比较架构必须明确共享输入

Limitations: 不能由结构图证明性能增益

#### S03-F3 / bottom

翻译器/LTL/规划器分离揭示什么？

该示例中可直接定位：翻译器/LTL/规划器分离。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：翻译器/LTL/规划器分离。

Design: 比较架构必须明确共享输入

Limitations: 不能由结构图证明性能增益

### S03-F4 (figure)

同一任务的动作顺序如何不同？

两个行序列显示不同取放顺序。

Evidence: 上行本文方法、下行LLM的四帧

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

[PDF page 6](samples/S03-nl2hltl-preprint.pdf#page=6)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S03-F4 / ours-1

先处理绿苹果揭示什么？

该示例中可直接定位：先处理绿苹果。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：先处理绿苹果。

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

#### S03-F4 / ours-2

接续放置绿苹果揭示什么？

该示例中可直接定位：接续放置绿苹果。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：接续放置绿苹果。

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

#### S03-F4 / ours-3

处理红苹果揭示什么？

该示例中可直接定位：处理红苹果。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：处理红苹果。

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

#### S03-F4 / ours-4

处理橙子揭示什么？

该示例中可直接定位：处理橙子。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：处理橙子。

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

#### S03-F4 / llm-1

先处理绿苹果揭示什么？

该示例中可直接定位：先处理绿苹果。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：先处理绿苹果。

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

#### S03-F4 / llm-2

随后处理橙子揭示什么？

该示例中可直接定位：随后处理橙子。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：随后处理橙子。

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

#### S03-F4 / llm-3

最后处理红苹果揭示什么？

该示例中可直接定位：最后处理红苹果。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：最后处理红苹果。

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

#### S03-F4 / llm-4

完成红苹果放置揭示什么？

该示例中可直接定位：完成红苹果放置。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：完成红苹果放置。

Design: 同输入并列展示取舍

Limitations: 选择案例不能代表平均效应

### S03-F5 (figure)

交接任务的几何配置有哪些？

直线和方形布局有不同可交接关系。

Evidence: a直线、b方形及源/目标标识

Design: 交接拓扑需要显式展示

Limitations: 臂间交接不等同空中观测交接

[PDF page 6](samples/S03-nl2hltl-preprint.pdf#page=6)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S03-F5 / a

直线协作臂揭示什么？

该示例中可直接定位：直线协作臂。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：直线协作臂。

Design: 交接拓扑需要显式展示

Limitations: 臂间交接不等同空中观测交接

#### S03-F5 / b

方形协作臂揭示什么？

该示例中可直接定位：方形协作臂。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：方形协作臂。

Design: 交接拓扑需要显式展示

Limitations: 臂间交接不等同空中观测交接

### S03-F6 (figure)

时序指令如何落到真实交接？

四幅真实机械臂图连接对象、目标区和轨迹。

Evidence: a–d带彩色轨迹与头对头箭头

Design: 用源对象身份追踪实际后果

Limitations: 不是统一长期任务成功证据

[PDF page 7](samples/S03-nl2hltl-preprint.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S03-F6 / a

首轮对象和目标区揭示什么？

该示例中可直接定位：首轮对象和目标区。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：首轮对象和目标区。

Design: 用源对象身份追踪实际后果

Limitations: 不是统一长期任务成功证据

#### S03-F6 / b

绿色轨迹及臂间交接揭示什么？

该示例中可直接定位：绿色轨迹及臂间交接。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：绿色轨迹及臂间交接。

Design: 用源对象身份追踪实际后果

Limitations: 不是统一长期任务成功证据

#### S03-F6 / c

后续对象移动揭示什么？

该示例中可直接定位：后续对象移动。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：后续对象移动。

Design: 用源对象身份追踪实际后果

Limitations: 不是统一长期任务成功证据

#### S03-F6 / d

蓝色交接轨迹与终点揭示什么？

该示例中可直接定位：蓝色交接轨迹与终点。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：蓝色交接轨迹与终点。

Design: 用源对象身份追踪实际后果

Limitations: 不是统一长期任务成功证据

### S03-T1 (table)

任务/机器人数量变化如何影响规划？

成功、路程和完成步数分开呈现。

Evidence: 任务数×机器人数组合；括号拆解转换/规划成功

Design: 主表应拆解失败阶段

Limitations: 本文只记录成功实例路程/时间，不能照搬为全分母主分析

[PDF page 5](samples/S03-nl2hltl-preprint.pdf#page=5)

### S03-T2 (table)

真实用户改写下质量和运行时如何权衡？

双方均能成功，优化质量与运行时有取舍。

Evidence: 四任务的成功、成本和Runtime

Design: 对相反排序保持解释

Limitations: 小样本台面不能证明真实灾害外推

[PDF page 6](samples/S03-nl2hltl-preprint.pdf#page=6)

### S03-T3 (table)

协作交接约束是否导致失败？

逐任务成功率揭示协作规划差别。

Evidence: 八任务及具体约束说明

Design: 约束与结果同表便于复查

Limitations: 任务数不等于独立用户/场景数

[PDF page 6](samples/S03-nl2hltl-preprint.pdf#page=6)

### S03-T4 (table)

失败发生在哪个表示转换阶段？

任务分解、时序提取、LTL和动作补全分别计数。

Evidence: 四类失败率及节点分母说明

Design: 错误率必须连同各自分母报告

Limitations: 不同节点分母不能直接相加成episode失败率

[PDF page 7](samples/S03-nl2hltl-preprint.pdf#page=7)

Reading declaration: {"method": "rendered_pdf", "notes": "Codex primary于2026-11-05实际查看120dpi渲染页。每个测得视觉均有独立判断；未识别表格另列manual-visuals。公开PDF版本与最终出版版差异另见sample-version-audit。", "pages": [1, 2, 5, 6, 7]}

## S04: SMART

8 figures; 0 tables; 24 facets or semantic blocks.

[Source PDF](samples/S04-smart.pdf)

### S04-F1 (figure)

栅格计划如何落到物理多机器人场景？

同一地图连接二维基准、两机型与三维仿真。

Evidence: a–e地图/机型/3D场景

Design: 首图交代抽象与执行的差距

Limitations: 不是航空动力学验证

[PDF page 1](samples/S04-smart.pdf#page=1)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S04-F1 / a

物理引擎地图揭示什么？

该示例中可直接定位：物理引擎地图。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：物理引擎地图。

Design: 首图交代抽象与执行的差距

Limitations: 不是航空动力学验证

#### S04-F1 / b

原始2D栅格揭示什么？

该示例中可直接定位：原始2D栅格。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：原始2D栅格。

Design: 首图交代抽象与执行的差距

Limitations: 不是航空动力学验证

#### S04-F1 / c

foot-bot模型揭示什么？

该示例中可直接定位：foot-bot模型。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：foot-bot模型。

Design: 首图交代抽象与执行的差距

Limitations: 不是航空动力学验证

#### S04-F1 / d

JetBot模型揭示什么？

该示例中可直接定位：JetBot模型。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：JetBot模型。

Design: 首图交代抽象与执行的差距

Limitations: 不是航空动力学验证

#### S04-F1 / e

三维环境揭示什么？

该示例中可直接定位：三维环境。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：三维环境。

Design: 首图交代抽象与执行的差距

Limitations: 不是航空动力学验证

### S04-F2 (figure)

时序计划怎样转成执行依赖？

动作图明确机器人内顺序与跨机约束。

Evidence: a问题、b计划、c依赖图

Design: 用状态变迁解释异步执行

Limitations: 依赖安全仍受模型假设约束

[PDF page 2](samples/S04-smart.pdf#page=2)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S04-F2 / a

两机器人网格任务揭示什么？

该示例中可直接定位：两机器人网格任务。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第2页：两机器人网格任务。

Design: 用状态变迁解释异步执行

Limitations: 依赖安全仍受模型假设约束

#### S04-F2 / b

离散计划揭示什么？

该示例中可直接定位：离散计划。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第2页：离散计划。

Design: 用状态变迁解释异步执行

Limitations: 依赖安全仍受模型假设约束

#### S04-F2 / c

动作依赖图揭示什么？

该示例中可直接定位：动作依赖图。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第2页：动作依赖图。

Design: 用状态变迁解释异步执行

Limitations: 依赖安全仍受模型假设约束

### S04-F3 (figure)

规划、监控、执行、仿真如何解耦？

中央监控和逐机器人执行器共同推进状态。

Evidence: planner→plan→server→queue/PID→simulator

Design: Agent与持久执行器分层；延迟已有研究先例

Limitations: 架构不能单独证明时钟公平

[PDF page 4](samples/S04-smart.pdf#page=4)

### S04-F4 (figure)

环依赖何时导致停滞，怎样修复？

加入满足几何限制的safe stop可打断图中环。

Evidence: a环式等待、b中间停点

Design: 失败例应连到可检查后验条件

Limitations: 网格停点几何不能直接移植固定翼

[PDF page 4](samples/S04-smart.pdf#page=4)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S04-F4 / a

环形等待死锁揭示什么？

该示例中可直接定位：环形等待死锁。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第4页：环形等待死锁。

Design: 失败例应连到可检查后验条件

Limitations: 网格停点几何不能直接移植固定翼

#### S04-F4 / b

加入安全中间停点揭示什么？

该示例中可直接定位：加入安全中间停点。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第4页：加入安全中间停点。

Design: 失败例应连到可检查后验条件

Limitations: 网格停点几何不能直接移植固定翼

### S04-F5 (figure)

用户能否观察任务与约束？

界面同时显示路径、机器人状态和依赖。

Evidence: 3D视口与控制/状态侧栏

Design: 为本研究保留可审计trace视图

Limitations: 截图不是功能测试

[PDF page 5](samples/S04-smart.pdf#page=5)

### S04-F6 (figure)

规模测试使用哪些地图族？

六类地图覆盖不同障碍与通道结构。

Evidence: a–f六地图及原始栅格插图

Design: OOD需按生成族定义

Limitations: 地图渲染不是独立实验分母

[PDF page 6](samples/S04-smart.pdf#page=6)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S04-F6 / a

empty揭示什么？

该示例中可直接定位：empty。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：empty。

Design: OOD需按生成族定义

Limitations: 地图渲染不是独立实验分母

#### S04-F6 / b

maze揭示什么？

该示例中可直接定位：maze。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：maze。

Design: OOD需按生成族定义

Limitations: 地图渲染不是独立实验分母

#### S04-F6 / c

random揭示什么？

该示例中可直接定位：random。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：random。

Design: OOD需按生成族定义

Limitations: 地图渲染不是独立实验分母

#### S04-F6 / d

game揭示什么？

该示例中可直接定位：game。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：game。

Design: OOD需按生成族定义

Limitations: 地图渲染不是独立实验分母

#### S04-F6 / e

warehouse揭示什么？

该示例中可直接定位：warehouse。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：warehouse。

Design: OOD需按生成族定义

Limitations: 地图渲染不是独立实验分母

#### S04-F6 / f

sortation-center揭示什么？

该示例中可直接定位：sortation-center。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：sortation-center。

Design: OOD需按生成族定义

Limitations: 地图渲染不是独立实验分母

### S04-F7 (figure)

不同地图/机器人数量如何影响实时因子？

各场景随规模增大出现不同吞吐下降。

Evidence: 六子图比较两物理引擎，纵轴对数

Design: 实时因子、硬件和规模同时报告

Limitations: 吞吐图不能替代模型真实API等待实验

[PDF page 6](samples/S04-smart.pdf#page=6)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S04-F7 / random

随机障碍的规模/速度揭示什么？

在该地图和所示硬件上，增加机器人数量降低仿真速度；两引擎的可实时运行范围不同。

Evidence: 实际查看PDF第6页：随机障碍的规模/速度。

Design: 实时因子、硬件和规模同时报告

Limitations: 吞吐图不能替代模型真实API等待实验

#### S04-F7 / game

游戏地图的规模/速度揭示什么？

在该地图和所示硬件上，增加机器人数量降低仿真速度；两引擎的可实时运行范围不同。

Evidence: 实际查看PDF第6页：游戏地图的规模/速度。

Design: 实时因子、硬件和规模同时报告

Limitations: 吞吐图不能替代模型真实API等待实验

#### S04-F7 / sortation

分拣中心的规模/速度揭示什么？

在该地图和所示硬件上，增加机器人数量降低仿真速度；两引擎的可实时运行范围不同。

Evidence: 实际查看PDF第6页：分拣中心的规模/速度。

Design: 实时因子、硬件和规模同时报告

Limitations: 吞吐图不能替代模型真实API等待实验

#### S04-F7 / empty

空地的规模/速度揭示什么？

在该地图和所示硬件上，增加机器人数量降低仿真速度；两引擎的可实时运行范围不同。

Evidence: 实际查看PDF第6页：空地的规模/速度。

Design: 实时因子、硬件和规模同时报告

Limitations: 吞吐图不能替代模型真实API等待实验

#### S04-F7 / warehouse

仓库的规模/速度揭示什么？

在该地图和所示硬件上，增加机器人数量降低仿真速度；两引擎的可实时运行范围不同。

Evidence: 实际查看PDF第6页：仓库的规模/速度。

Design: 实时因子、硬件和规模同时报告

Limitations: 吞吐图不能替代模型真实API等待实验

#### S04-F7 / maze

迷宫的规模/速度揭示什么？

在该地图和所示硬件上，增加机器人数量降低仿真速度；两引擎的可实时运行范围不同。

Evidence: 实际查看PDF第6页：迷宫的规模/速度。

Design: 实时因子、硬件和规模同时报告

Limitations: 吞吐图不能替代模型真实API等待实验

### S04-F8 (figure)

仿真与实机验证规模如何区分？

相似布局分别用于仿真和小规模物理机器人。

Evidence: a Isaac Sim、b实机

Design: 分清仿真规模与实机规模

Limitations: 不能把数千仿真机声称为实机数量

[PDF page 7](samples/S04-smart.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S04-F8 / a

仿真验证布置揭示什么？

该示例中可直接定位：仿真验证布置。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：仿真验证布置。

Design: 分清仿真规模与实机规模

Limitations: 不能把数千仿真机声称为实机数量

#### S04-F8 / b

七台实机布置揭示什么？

该示例中可直接定位：七台实机布置。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第7页：七台实机布置。

Design: 分清仿真规模与实机规模

Limitations: 不能把数千仿真机声称为实机数量

Reading declaration: {"method": "rendered_pdf", "notes": "Codex primary于2026-11-05实际查看120dpi渲染页。每个测得视觉均有独立判断；未识别表格另列manual-visuals。公开PDF版本与最终出版版差异另见sample-version-audit。", "pages": [1, 2, 4, 5, 6, 7]}

## S05: Scalable MRIPP

5 figures; 4 tables; 3 facets or semantic blocks.

[Source PDF](samples/S05-mripp.pdf)

### S05-F1 (figure)

实际信息采集任务是什么？

无人机与标记目标在物理场景中被标出。

Evidence: Tello与Aruco标记照片

Design: 业务对象和感知代理应可辨认

Limitations: 标记发现不等于自然灾害诊断

[PDF page 1](samples/S05-mripp.pdf#page=1)

### S05-F2 (figure)

采集、协调和策略训练如何闭合？

占据/目标/已探索信息构成协调图并驱动策略。

Evidence: 黑色执行回路与绿色训练缓存通路

Design: 画清部署信息与训练特权

Limitations: 不可把全局训练量泄露Actor

[PDF page 3](samples/S05-mripp.pdf#page=3)

### S05-F3 (figure)

给定资源预算下发现收益如何变化？

曲线同时比较多个策略和通信条件。

Evidence: 预算横轴、发现比例及均值/SD带

Design: 展示资源—收益整条轨迹

Limitations: 250次含重复，不可直接当250独立地图

[PDF page 6](samples/S05-mripp.pdf#page=6)

### S05-F4 (figure)

规模外推是否始终优于基线？

最大环境中CAtNIPP超过本文方法；图注概括过强。

Evidence: 3×/8×/16×三子图与机器人数量

Design: 结论须服从实际曲线，保留逆转

Limitations: 不同通信设定影响可比性；不照抄图注胜负

[PDF page 7](samples/S05-mripp.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S05-F4 / 3x

三倍场景的机器人规模曲线揭示什么？

三倍场景中本方法曲线高于所列对照。

Evidence: 实际查看PDF第7页：三倍场景的机器人规模曲线。

Design: 结论须服从实际曲线，保留逆转

Limitations: 不同通信设定影响可比性；不照抄图注胜负

#### S05-F4 / 8x

八倍场景的机器人规模曲线揭示什么？

八倍场景中本方法与CAtNIPP差距较小。

Evidence: 实际查看PDF第7页：八倍场景的机器人规模曲线。

Design: 结论须服从实际曲线，保留逆转

Limitations: 不同通信设定影响可比性；不照抄图注胜负

#### S05-F4 / 16x

十六倍场景中CAtNIPP领先揭示什么？

十六倍场景中CAtNIPP曲线高于本方法，限制图注的泛化表述。

Evidence: 实际查看PDF第7页：十六倍场景中CAtNIPP领先。

Design: 结论须服从实际曲线，保留逆转

Limitations: 不同通信设定影响可比性；不照抄图注胜负

### S05-F5 (figure)

任务路径的空间结果是什么？

三机轨迹覆盖城市模型的不同部分。

Evidence: 彩色航迹与城市对象

Design: 典型轨迹解释主指标

Limitations: 单条轨迹不证明安全/泛化

[PDF page 7](samples/S05-mripp.pdf#page=7)

### S05-T1 (table)

发现比例与规划时间如何权衡？

多种学习/经典方法给出不同收益与时间。

Evidence: 发现比例±波动及Time(s)

Design: 强经典与简单策略都保留

Limitations: 此对比设置暂忽略机间碰撞；不能支持本研究安全结论

[PDF page 6](samples/S05-mripp.pdf#page=6)

### S05-T2 (table)

已探索区域记忆是否有作用？

含/不含记忆的发现比例有差异。

Evidence: 两行相同任务指标

Design: 记忆消融保持其他条件一致

Limitations: 文中普通t检验不能移植为root配对分析

[PDF page 6](samples/S05-mripp.pdf#page=6)

### S05-T3 (table)

通信奖励是否有作用？

是否使用奖励改变采集收益。

Evidence: 两行通信奖励消融

Design: 区分机制与模型更换

Limitations: 同一环境重复不创造独立root

[PDF page 7](samples/S05-mripp.pdf#page=7)

### S05-T4 (table)

带物理场景的参考收益是多少？

模拟城市场景中与随机下界比较。

Evidence: 本方法与随机策略两行

Design: 平台可行性参照与强主比较应区分

Limitations: 弱下界不能替代本研究B1/B2

[PDF page 7](samples/S05-mripp.pdf#page=7)

Reading declaration: {"method": "rendered_pdf", "notes": "Codex primary于2026-11-05实际查看120dpi渲染页。每个测得视觉均有独立判断；未识别表格另列manual-visuals。公开PDF版本与最终出版版差异另见sample-version-audit。", "pages": [1, 3, 6, 7]}

## S06: Heterogeneous MRTA

4 figures; 0 tables; 2 facets or semantic blocks.

[Source PDF](samples/S06-hetmrta.pdf)

### S06-F1 (figure)

任务为什么需要异构协作？

救援对象的能力需求与机器人类型对应。

Evidence: 能力图标和当前/未来路线

Design: 业务任务与资源匹配共同画出

Limitations: 概念图不证明真实救灾效用

[PDF page 1](samples/S06-hetmrta.pdf#page=1)

### S06-F2 (figure)

任务与机器人依赖如何进入调度？

双编码与交互注意力用于下一任务选择。

Evidence: 代理/任务图到策略解码的结构

Design: 共享数据与调度决策分开

Limitations: 本论文不需复现该网络作为唯一强基线

[PDF page 5](samples/S06-hetmrta.pdf#page=5)

### S06-F3 (figure)

优化耗时与最终质量是什么关系？

快速策略与长时间优化的优势随时间变化。

Evidence: 20/30任务两子图的makespan—time曲线

Design: 固定截止比较，避免只报无限时最优

Limitations: 优化可最终更优，不能宣称学习策略处处占优

[PDF page 7](samples/S06-hetmrta.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S06-F3 / left

20任务的优化质量/耗时揭示什么？

在20任务例子中，长时间优化最终可超过快速策略的解质量。

Evidence: 实际查看PDF第7页：20任务的优化质量/耗时。

Design: 固定截止比较，避免只报无限时最优

Limitations: 优化可最终更优，不能宣称学习策略处处占优

#### S06-F3 / right

30任务的优化质量/耗时揭示什么？

在30任务例子中也必须区分固定截止质量与最终收敛质量。

Evidence: 实际查看PDF第7页：30任务的优化质量/耗时。

Design: 固定截止比较，避免只报无限时最优

Limitations: 优化可最终更优，不能宣称学习策略处处占优

### S06-F4 (figure)

能否实际形成/解散任务协作组？

不同高度标识机型，锥体标识任务。

Evidence: 六机室内搜索照片

Design: 物理能力演示与主统计分开

Limitations: 照片不等于完整episode安全率

[PDF page 8](samples/S06-hetmrta.pdf#page=8)

Reading declaration: {"method": "rendered_pdf", "notes": "Codex primary于2026-11-05实际查看120dpi渲染页。每个测得视觉均有独立判断；未识别表格另列manual-visuals。公开PDF版本与最终出版版差异另见sample-version-audit。", "pages": [1, 5, 6, 7, 8]}

## S07: Online multi-UAV planning

8 figures; 4 tables; 14 facets or semantic blocks.

[Source PDF](samples/S07-pursuit.pdf)

### S07-F1 (figure)

部分观测与任务几何如何定义？

遮挡示意与仿真空间被对应。

Evidence: a遮挡/检测示意，b仿真场景

Design: 明确视域与隐藏量

Limitations: 追逃任务不作为本研究民用业务目标

[PDF page 2](samples/S07-pursuit.pdf#page=2)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S07-F1 / a

遮挡、检测与捕获区域示意揭示什么？

该示例中可直接定位：遮挡、检测与捕获区域示意。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第2页：遮挡、检测与捕获区域示意。

Design: 明确视域与隐藏量

Limitations: 追逃任务不作为本研究民用业务目标

#### S07-F1 / b

物理仿真环境揭示什么？

该示例中可直接定位：物理仿真环境。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第2页：物理仿真环境。

Design: 明确视域与隐藏量

Limitations: 追逃任务不作为本研究民用业务目标

### S07-F2 (figure)

动力学校准与训练部署如何连接？

系统辨识、任务生成和策略部署分层。

Evidence: 现实侧、仿真侧与预测网络路径

Design: 控制资格与Agent比较分开

Limitations: CTBR动作不是mission tool接口

[PDF page 3](samples/S07-pursuit.pdf#page=3)

### S07-F3 (figure)

可见状态如何编码？

本机、其他机、障碍与预测经过注意力融合。

Evidence: 三路输入及自注意力结构

Design: 列清Actor观测来源

Limitations: 该状态接口不等于真实视觉

[PDF page 4](samples/S07-pursuit.pdf#page=4)

### S07-F4 (figure)

训练任务如何覆盖难例？

局部扩展与全局探索维护活动任务档案。

Evidence: 双路径生成器

Design: dev难度调整与test隔离

Limitations: 训练难例选择不能用于筛除低分test

[PDF page 5](samples/S07-pursuit.pdf#page=5)

### S07-F5 (figure)

ID/OOD测试场景如何区分？

四种几何产生不同遮挡与通道。

Evidence: Wall/NarrowGap/Random/Passage

Design: 用场景族解释OOD

Limitations: 更换随机seed不足以构成OOD

[PDF page 5](samples/S07-pursuit.pdf#page=5)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S07-F5 / wall

横向阻挡揭示什么？

该示例中可直接定位：横向阻挡。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：横向阻挡。

Design: 用场景族解释OOD

Limitations: 更换随机seed不足以构成OOD

#### S07-F5 / gap

窄通道揭示什么？

该示例中可直接定位：窄通道。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：窄通道。

Design: 用场景族解释OOD

Limitations: 更换随机seed不足以构成OOD

#### S07-F5 / random

随机遮挡揭示什么？

该示例中可直接定位：随机遮挡。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：随机遮挡。

Design: 用场景族解释OOD

Limitations: 更换随机seed不足以构成OOD

#### S07-F5 / passage

逃离通道揭示什么？

该示例中可直接定位：逃离通道。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：逃离通道。

Design: 用场景族解释OOD

Limitations: 更换随机seed不足以构成OOD

### S07-F6 (figure)

难度参数变化时质量/耗时怎样变化？

捕获率和捕获步数对半径的响应不同。

Evidence: a成功率，b步数

Design: 质量与时间曲线并列

Limitations: 此任务指标不能转作巡检交付成绩

[PDF page 6](samples/S07-pursuit.pdf#page=6)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S07-F6 / a

捕获半径与成功率揭示什么？

缩小捕获半径时，不同方法成功率下降幅度明显不同。

Evidence: 实际查看PDF第6页：捕获半径与成功率。

Design: 质量与时间曲线并列

Limitations: 此任务指标不能转作巡检交付成绩

#### S07-F6 / b

捕获半径与步数揭示什么？

步数曲线保留失败造成的上限惩罚，不能只平均成功例。

Evidence: 实际查看PDF第6页：捕获半径与步数。

Design: 质量与时间曲线并列

Limitations: 此任务指标不能转作巡检交付成绩

### S07-F7 (figure)

成功策略在不同地图如何执行？

四种几何产生不同队形/运动路径。

Evidence: 四场景路径快照

Design: 从轨迹解释失败或收益

Limitations: 案例不代替总体统计

[PDF page 6](samples/S07-pursuit.pdf#page=6)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S07-F7 / wall

墙体两侧协作揭示什么？

该示例中可直接定位：墙体两侧协作。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：墙体两侧协作。

Design: 从轨迹解释失败或收益

Limitations: 案例不代替总体统计

#### S07-F7 / gap

窄口拦截揭示什么？

该示例中可直接定位：窄口拦截。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：窄口拦截。

Design: 从轨迹解释失败或收益

Limitations: 案例不代替总体统计

#### S07-F7 / random

遮挡目标搜索揭示什么？

该示例中可直接定位：遮挡目标搜索。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：遮挡目标搜索。

Design: 从轨迹解释失败或收益

Limitations: 案例不代替总体统计

#### S07-F7 / passage

出口分工揭示什么？

该示例中可直接定位：出口分工。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：出口分工。

Design: 从轨迹解释失败或收益

Limitations: 案例不代替总体统计

### S07-F8 (figure)

模块与速度外推分别贡献什么？

学习曲线与速度敏感性回答不同问题。

Evidence: a训练曲线，b不同目标速度

Design: 消融与敏感性各自固定比较单位

Limitations: 训练样本数不是独立测试root

[PDF page 7](samples/S07-pursuit.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S07-F8 / a

训练样本效率及模块消融揭示什么？

所示训练轨迹中完整方法更早取得较高捕获率。

Evidence: 实际查看PDF第7页：训练样本效率及模块消融。

Design: 消融与敏感性各自固定比较单位

Limitations: 训练样本数不是独立测试root

#### S07-F8 / b

目标速度外推揭示什么？

完整方法在展示的速度范围内较稳定；这不证明范围外泛化。

Evidence: 实际查看PDF第7页：目标速度外推。

Design: 消融与敏感性各自固定比较单位

Limitations: 训练样本数不是独立测试root

### S07-T1 (table)

主成绩是否同时保留效率和碰撞？

四场景同时报告成功、步数和碰撞，排序并非全部一致。

Evidence: 四场景×三指标×多方法

Design: 不以一个汇总遮盖安全退步

Limitations: 碰撞计数/步归一不等于episode硬安全

[PDF page 6](samples/S07-pursuit.pdf#page=6)

### S07-T2 (table)

预测误差在哪些场景更大？

四场景误差不同。

Evidence: 预测误差一行四列

Design: 感知/预测诊断与交付主分开

Limitations: 预测不是本研究新采样证据

[PDF page 7](samples/S07-pursuit.pdf#page=7)

### S07-T3 (table)

OOD下移除模块如何影响质量？

不同模块对成功、步数和碰撞影响不同。

Evidence: Random/Passage的模块对照

Design: 同模型单因素消融并列分量

Limitations: 不能把模块交互全部归因单机制

[PDF page 7](samples/S07-pursuit.pdf#page=7)

### S07-T4 (table)

仿真与实机可行性何时分离？

高仿真表现的无平滑策略在实机失败。

Evidence: Sim/Real及fail、未执行标记

Design: 保留失败和未启动状态

Limitations: 不能把未执行斜线当成功或零耗时

[PDF page 7](samples/S07-pursuit.pdf#page=7)

Reading declaration: {"method": "rendered_pdf", "notes": "Codex primary于2026-11-05实际查看120dpi渲染页。每个测得视觉均有独立判断；未识别表格另列manual-visuals。公开PDF版本与最终出版版差异另见sample-version-audit。", "pages": [2, 3, 4, 5, 6, 7, 8]}

## S08: GRaD-Nav++

7 figures; 7 tables; 23 facets or semantic blocks.

[Source PDF](samples/S08-gradnav.pdf)

### S08-F1 (figure)

训练特权与部署输入如何区分？

训练critic使用特权量，实际部署仅保留规定感知输入。

Evidence: 上训练下部署的两部分

Design: 权限图需显式列出训练/评测差异

Limitations: VLA低层动作不是本研究任务级编排

[PDF page 1](samples/S08-gradnav.pdf#page=1)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S08-F1 / training

训练特权与可微动力学揭示什么？

该示例中可直接定位：训练特权与可微动力学。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：训练特权与可微动力学。

Design: 权限图需显式列出训练/评测差异

Limitations: VLA低层动作不是本研究任务级编排

#### S08-F1 / deployment

实际机载视觉/指令到动作揭示什么？

该示例中可直接定位：实际机载视觉/指令到动作。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第1页：实际机载视觉/指令到动作。

Design: 权限图需显式列出训练/评测差异

Limitations: VLA低层动作不是本研究任务级编排

### S08-F2 (figure)

组合指令如何产生不同轨迹？

两条未训练组合任务具有不同路线和终点对象。

Evidence: 左穿门到车，右绕门到显示器

Design: 任务卡与轨迹共同展示

Limitations: 只为成功例，不能证明总体泛化

[PDF page 5](samples/S08-gradnav.pdf#page=5)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S08-F2 / left

穿门后到CART揭示什么？

该示例中可直接定位：穿门后到CART。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：穿门后到CART。

Design: 任务卡与轨迹共同展示

Limitations: 只为成功例，不能证明总体泛化

#### S08-F2 / right

从右侧绕门后到MONITOR揭示什么？

该示例中可直接定位：从右侧绕门后到MONITOR。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第5页：从右侧绕门后到MONITOR。

Design: 任务卡与轨迹共同展示

Limitations: 只为成功例，不能证明总体泛化

### S08-F3 (figure)

低质量视觉重建是什么样？

左右渲染对比提供退化机制的可见依据。

Evidence: 左模糊/缺损，右较清晰

Design: 退化消融要展示实际输入

Limitations: 渲染质量与结构化噪声不是同一指标

[PDF page 5](samples/S08-gradnav.pdf#page=5)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S08-F3 / left

退化3DGS渲染揭示什么？

低质量模型呈现更模糊和缺失的视觉线索。

Evidence: 实际查看PDF第5页：退化3DGS渲染。

Design: 退化消融要展示实际输入

Limitations: 渲染质量与结构化噪声不是同一指标

#### S08-F3 / right

高质量3DGS渲染揭示什么？

较高质量模型保留更清晰的门与物体细节。

Evidence: 实际查看PDF第5页：高质量3DGS渲染。

Design: 退化消融要展示实际输入

Limitations: 渲染质量与结构化噪声不是同一指标

### S08-F4 (figure)

不同环境和指令的实际行为如何变化？

四指令×两门位置用轨迹叠图展示。

Evidence: 两行四列，左/穿/上/右

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

[PDF page 6](samples/S08-gradnav.pdf#page=6)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S08-F4 / mid-left

中间门左侧揭示什么？

该示例中可直接定位：中间门左侧。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：中间门左侧。

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

#### S08-F4 / mid-through

中间门穿越揭示什么？

该示例中可直接定位：中间门穿越。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：中间门穿越。

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

#### S08-F4 / mid-above

中间门上方揭示什么？

该示例中可直接定位：中间门上方。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：中间门上方。

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

#### S08-F4 / mid-right

中间门右侧揭示什么？

该示例中可直接定位：中间门右侧。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：中间门右侧。

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

#### S08-F4 / left-left

左侧门左绕揭示什么？

该示例中可直接定位：左侧门左绕。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：左侧门左绕。

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

#### S08-F4 / left-through

左侧门穿越揭示什么？

该示例中可直接定位：左侧门穿越。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：左侧门穿越。

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

#### S08-F4 / left-above

左侧门上方揭示什么？

该示例中可直接定位：左侧门上方。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：左侧门上方。

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

#### S08-F4 / left-right

左侧门右绕揭示什么？

该示例中可直接定位：左侧门右绕。 仅支持这一个面板的机制/场景说明。

Evidence: 实际查看PDF第6页：左侧门右绕。

Design: 案例布局与受控因素一一对应

Limitations: 帧叠图不是独立试验样本

### S08-F5 (figure)

环境变化时专家路由如何变化？

同任务在两环境具有不同门控时序。

Evidence: 上/下两环境两专家曲线

Design: 机制诊断与主效果区分

Limitations: 门控变化本身不能证明性能因果

[PDF page 7](samples/S08-gradnav.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S08-F5 / top

左门环境专家路由揭示什么？

左门环境两专家激活强度随任务阶段改变。

Evidence: 实际查看PDF第7页：左门环境专家路由。

Design: 机制诊断与主效果区分

Limitations: 门控变化本身不能证明性能因果

#### S08-F5 / bottom

中间门环境专家路由揭示什么？

中间门环境的激活主次与上方面板不同。

Evidence: 实际查看PDF第7页：中间门环境专家路由。

Design: 机制诊断与主效果区分

Limitations: 门控变化本身不能证明性能因果

### S08-F6 (figure)

飞行中指令变化是否引发响应？

示例把换指令轨迹和相似度变化并列。

Evidence: 左场景，右步100切换后的相似度

Design: 响应须同时有状态和效果证据

Limitations: 已有中途指令切换先例；单例不证明群体恢复优势

[PDF page 7](samples/S08-gradnav.pdf#page=7)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S08-F6 / left

更换指令后的实际航迹揭示什么？

示例航迹显示指令更新后转向新的任务方向。

Evidence: 实际查看PDF第7页：更换指令后的实际航迹。

Design: 响应须同时有状态和效果证据

Limitations: 已有中途指令切换先例；单例不证明群体恢复优势

#### S08-F6 / right

相似度在切换点后的变化揭示什么？

切换附近的匹配度下降后恢复并波动，不能直接当任务成功率。

Evidence: 实际查看PDF第7页：相似度在切换点后的变化。

Design: 响应须同时有状态和效果证据

Limitations: 已有中途指令切换先例；单例不证明群体恢复优势

### S08-F7 (figure)

实际硬件推理资源是否稳定？

运行时存在周期峰值，资源量随时间变化。

Evidence: RAM/CPU/GPU/功率/单次推理五行

Design: 记录真实p50/p95与异常等待

Limitations: 已有飞行推理代价先例；不据此推断本研究机群交付损失

[PDF page 8](samples/S08-gradnav.pdf#page=8)

Panel measurement correction: 已按实际渲染页逐块检查并补齐面板；文字提取器漏字母、将单位(s)误当标签或未识别无字母子图。这里使用可定位的实际字母/行列/时间/坐标量名称，保留原机械测量供对照。

#### S08-F7 / ram

内存轨迹揭示什么？

展示时段内内存用量相对稳定。

Evidence: 实际查看PDF第8页：内存轨迹。

Design: 记录真实p50/p95与异常等待

Limitations: 已有飞行推理代价先例；不据此推断本研究机群交付损失

#### S08-F7 / cpu

CPU负载揭示什么？

CPU负载随执行时间出现峰值。

Evidence: 实际查看PDF第8页：CPU负载。

Design: 记录真实p50/p95与异常等待

Limitations: 已有飞行推理代价先例；不据此推断本研究机群交付损失

#### S08-F7 / gpu

GPU负载揭示什么？

GPU负载在该实际飞行记录中持续波动。

Evidence: 实际查看PDF第8页：GPU负载。

Design: 记录真实p50/p95与异常等待

Limitations: 已有飞行推理代价先例；不据此推断本研究机群交付损失

#### S08-F7 / power

功率轨迹揭示什么？

功率轨迹随时间变化，需作为实测资源量保留。

Evidence: 实际查看PDF第8页：功率轨迹。

Design: 记录真实p50/p95与异常等待

Limitations: 已有飞行推理代价先例；不据此推断本研究机群交付损失

#### S08-F7 / runtime

单次推理延迟与周期峰值揭示什么？

推理存在周期性高延迟峰；仅报平均值会隐藏这一结构。

Evidence: 实际查看PDF第8页：单次推理延迟与周期峰值。

Design: 记录真实p50/p95与异常等待

Limitations: 已有飞行推理代价先例；不据此推断本研究机群交付损失

### S08-T1 (table)

安全与任务奖励如何规定？

奖励与早停条件被明确列出。

Evidence: Safe Control / Navigation两组

Design: 公开硬包线和阈值来源

Limitations: 奖励权重不等于合格安全保证

[PDF page 4](samples/S08-gradnav.pdf#page=4)

### S08-T2 (table)

仿真中已见/未见任务怎样失败？

用stage与overall绝对成功计数保留零值。

Evidence: 训练/未见列和PPO/VLM/退化对照

Design: 总体分母必须保留；条件阶段率另报

Limitations: 少量重复不能证明广泛泛化

[PDF page 5](samples/S08-gradnav.pdf#page=5)

### S08-T3 (table)

实机结果是否复制仿真排序？

阶段与总体成绩存在sim-to-real差别。

Evidence: 与Table II并列的实机计数

Design: 主表区分模拟和真实轨道

Limitations: 本研究没有实机数据，不能借其成绩代替

[PDF page 5](samples/S08-gradnav.pdf#page=5)

### S08-T4 (table)

多环境适应与模型容量如何权衡？

两环境下小模型对照表现不同。

Evidence: 任务方向×门位置的仿真计数

Design: 保持相同模型/场景因子

Limitations: 不能把训练适应当未知环境zero-shot

[PDF page 6](samples/S08-gradnav.pdf#page=6)

### S08-T5 (table)

同一适应比较在实机如何变化？

成功计数暴露模拟之外的退化。

Evidence: 两环境四方向的实机计数

Design: 不要只展示仿真有利行

Limitations: 每格三次是极有限精度

[PDF page 6](samples/S08-gradnav.pdf#page=6)

### S08-T6 (table)

增加专家数是否总是更好？

模拟收益提高而实机成功可能下降。

Evidence: 2/4/8专家的Sim/Real对照

Design: 质量—计算—部署时延边界应一并报告

Limitations: 正文把退化联系到较低控制频率；仍非本研究根因证据

[PDF page 7](samples/S08-gradnav.pdf#page=7)

### S08-T7 (table)

任务组合空间是什么？

四种穿越指令与三个终点对象构成任务表。

Evidence: 附录stage1与stage2表

Design: 协议关键定义不能隐藏在外链

Limitations: 附录计入八页上限

[PDF page 8](samples/S08-gradnav.pdf#page=8)

Reading declaration: {"method": "rendered_pdf", "notes": "Codex primary于2026-11-05实际查看120dpi渲染页。每个测得视觉均有独立判断；未识别表格另列manual-visuals。公开PDF版本与最终出版版差异另见sample-version-audit。", "pages": [1, 4, 5, 6, 7, 8]}

