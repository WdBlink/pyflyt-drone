# 控制执行器与评分组件独立工程审查 v1

日期：2026-11-05。角色：独立上下文的工程只读审查；继承主 Agent 的模型配置，未实施跨模型审查。仅本报告由审查 Agent 写入，未修改控制器、runner、评分器、测试或已有资格记录。审查依据为项目规则、pstack Codex 适配规则和 good-benchmarks 纯审计流程。

## 结论与验证边界

原始 runner 的两项 P2 已有部分修复：控制循环异常会返回 operational_error；非有限状态现在保存带 null 和故障索引的 terminal_state。但后续版本仍有必要修复：四机首步失败的 min_separation=inf 使严格 JSON 保存再次崩溃；首步失败摘要仍显示未标明来源的初始默认值。未发现控制坐标符号错误，未发现新增评分组件在本次要求的第 6.2 节选择与时间语义上的偏差。

本报告仅支持无风开发控制烟测及评分工程 fixtures 的结论，不证明完整 G1/G2、G1–G4、正式 benchmark、论文实验或人工领域复核通过。human_review_pending。评分器自己声明不实现传感、报告 admission 或身份匹配；本审查未将其接受的 evaluator-only 标志视为这些机制已资格通过。

## 版本区分

初审针对三条 v1 飞行记录绑定的原始 runner：

- eval/qualify_inspection_control.py 的原始 SHA256 为 76fcbb001e432c83b04abd7681c354ffda44be88bdb90032da354d77a2afdbd8。
- 原始字节现在位于 qualification/source/v1/qualify_inspection_control.py，本次重新计算哈希一致。
- qualification/source/v1/control.py 与 probe_inspection_surfaces.py 也已保留。三条飞行记录所绑定的控制器与 runner 哈希均与该目录的对应文件一致。
- 控制器 SHA256 为 3fe0a5721b755f1f7fc2f64754f9c0b26d45356f5709c506472e66177b6146eb。

后续修复复核时读取的版本：

| 文件 | SHA256 |
| --- | --- |
| eval/qualify_inspection_control.py | f6f90d7e5366d42c9e53bfed04518049eb80a5148e1cecb71fc0805de6ff3807 |
| tests/test_inspection_control.py | f459abc2798b72a5705ff65a95b29b82d680a1814ceaf8c99aa65987e7158888 |
| envs/inspection/scoring.py | 7909877c61bbc143ad73f454572342cc801375cf4b82bdd98d807f9e8e9c43e9 |
| tests/test_inspection_scoring.py | 0f9c4f34c5c0f6a92dcaca6fe51b1c6831b826e221eae7ca710499bc41049403 |

以下行号属于上述后续修复版本；初审问题的行号明确引用归档副本。并行修改后的文件需按哈希区分，不能把旧飞行记录解释为新 runner 的重新运行结果。

## 初审发现及修复复核

### 初审 P2-A：运行异常不留下失败记录

原始 qualification/source/v1/qualify_inspection_control.py:37–78 的 try/finally 只断连；控制器或 world.step 抛异常后，109–111 行的结果保存不会执行。真实异常注入重现了 RuntimeError 上抛且没有返回记录。工作单 planning/control-qualification.v1.md:7 要求保留失败原因。

后续 runner:86–92 已捕获控制循环异常并保存 traceback、实际模拟时长和 physics_steps；CLI:124–130 还捕获 fly 自身异常。新增 tests/test_inspection_control.py:88–98 在首个控制命令执行前注入控制器异常，确实验证 count=1 时 simulated_seconds=0、envelope_pass=false 和错误字符串。该原始缺陷的单机返回路径已修复，但四机落盘仍受下述 P2-C 阻断。

### 初审 P2-B：非有限状态提前退出，丢失故障现场

原始 runner:49–51 在保存摘要和每 12 步轨迹前 break。首个物理控制步后注入 NaN，原结果为 trace=[]、仅 nonfinite_state 违规名，且摘要仍为 80/20 初始值。

后续 runner:52–62 已新增终止时刻、sanitized states 和 nonfinite_indices。新增 tests/test_inspection_control.py:101–117 在真实 Aviary 第一步后的状态读取注入 NaN，验证故障坐标与 null。这部分现场保存已修复；摘要语义和四机序列化还未完全解决。

## 后续复核仍需修复

### P2-C：四机首步失败结果仍不能写成严格 JSON

证据：eval/qualify_inspection_control.py:34 初始化 min_separation=float("inf")，只有 68–71 行完成一次有限状态统计才更新，109 行在 count>1 时直接返回该值。CLI:132 的 json.dumps(..., allow_nan=False) 在 fly 的异常捕获范围之外。

复现一：把 FixedwingAutopilot.step monkeypatch 为抛 RuntimeError，再调用 fly(1, 0, "straight", 4)。结果 envelope_pass=false、simulated_seconds=0、summary.min_separation=inf；json.dumps(result, allow_nan=False) 抛 ValueError: Out of range float values are not JSON compliant。

复现二：沿新增测试的 Aviary.all_states property 注入方式，在首步把 states[0][2,1] 改为 np.nan，再调用 fly(1, 0, "straight", 4)。虽然 terminal_state 正确保留 t=1/120、nonfinite_indices=[[0,2,1]] 和 null，summary.min_separation 仍为 inf，严格 JSON 仍抛相同异常。两种对应 count=1 的结果都可序列化。

影响：四机正是候选共享执行器的目标范围。此路径仍会在保存失败记录时丢失整个文件，不能把初审 P2-A 标为完整解决。当前新增测试只使用 count=1，且没有执行实际保存所用的严格 JSON 序列化，因此未捕获回归。

必要修复：未观测的间距使用 null 或从可用初态实际计算；确保整个返回对象的所有数值可严格序列化。将首步异常、首步非有限两项用例参数化覆盖 1/4 机，并断言 json.dumps(..., allow_nan=False) 可成功。不要仅删除 allow_nan=False 来允许非标准 JSON。

### P2-D：首步失败摘要仍把初始默认值呈现为无来源范围

证据：runner:35–36 将高度/速度范围写死为 80/20、最大滚转写死为 0；52–63 行的 nonfinite 分支先退出，异常分支同样不更新；107–109 行没有 finite_state_samples、summary_valid、范围截止时间或初态范围说明。

复现：上述 count=1 首步 nonfinite 结果可序列化，但 terminal_state.t=0.008333333333333333、trace=[]，summary 仍是 min_altitude=max_altitude=80、min_speed=max_speed=20、max_abs_roll=0。即使仅一个速度分量失效，终止快照中仍有实际有限高度，该摘要也未涵盖终止时刻。首个控制命令前抛异常时，摘要相同，而其实际含义最多只是初态。

影响：envelope_pass=false 和 terminal_state 使失败状态本身诚实，但摘要字段没有说明是初态、已统计有限步范围还是空观测默认值。下游聚合或人工阅读可能将其当成失败前真实范围。

必要修复：从实际 world 初态初始化范围并明确统计时间/样本数，或在没有完成有限状态统计时返回 null 与 samples=0；终止快照与摘要采用什么统计范围应明确。测试应同时检查来源/样本数或空值语义，不能仅检查故障索引。

## 新增评分组件与原计划第 6.2 节对应

原计划路径：inputs/BENCHMARK_RESEARCH_PLAN.original.md:134–148。评分代码是单个已固定关联实体的纯函数；全实体静态/动态聚合、身份匹配、schema admission 与因果 provenance 的实际判定由调用方负责，当前不视为已实现。

| 要求 | 代码与测试证据 | 结论 |
| --- | --- | --- |
| q(e) 的 3/5/10 m 含等号边界 | scoring.py:15–24；test_inspection_scoring.py:20–23 | 与计划136行一致 |
| 第一格闭区间，其余左开右闭；一个新测量最多归一格 | scoring.py:93–101；测试33–37 | 与计划140行一致 |
| 完整窗口及最终宽限必须位于 deadline 之前 | scoring.py:82–87；测试92–94 | 拒绝截断窗口，不静默缩分母 |
| 先筛 schema/provenance/到达合法候选，再选观测时刻最新及证据ID升序 | scoring.py:98–102；测试49–55、79–82 | 与计划144行一致；provenance 的真实性仍是调用方条件 |
| 身份、类别、质量、误差只在选定后判断，不退回旧正确样本 | scoring.py:104–117；测试40–46 | 与计划144行一致 |
| 误差比较测量时刻真值，不比较格端或到达时刻真值 | scoring.py:112–116；测试64–68 | 与计划142行一致 |
| 每格权重取格起点；全部格进入分母 | scoring.py:93–97、118–121；测试26–30、71–76 | 与计划146行一致；当前只返回单实体加权得分与格明细 |
| observed 仅把到达条件改为最终审计截止，独立选最新 | scoring.py:98–102；测试58–61 | 与计划148行一致，差值可负，未被强制当通信损失 |
| 静态按最终截止前最后合法到达版本，不按正确性挑选 | scoring.py:124–136；测试85–89 | 与计划136行一致；最终权重聚合未在此函数中实现 |

独立于现有测试，额外手算并执行 13 项边界检查：出现时刻包含、出现前排除、内部端点归前格、最后端点包含、窗口后排除、宽限恰好包含、宽限后排除、最新错误覆盖旧正确、非法 provenance 在挑样前排除、observed 最终截止恰好包含及超出排除、格起点恰好生效、静态同接收时刻证据ID决胜、测量/格端/到达真值三者不同的定位检查。结果全部符合独立预期。此结果是评分工程校验，不是方法对比或完整 G2 资格。

## 实际执行证据与未知范围

初审命令：

```sh
./.venv/bin/python -m pytest -q tests/test_inspection_control.py
```

结果：8 passed。初审额外在显式 mode=-1 下运行初始航向误差 ±90°/180° 的各20秒有界物理检查，均维持当前包线；这是补充方向烟测。

后续修复复核命令：

```sh
./.venv/bin/python -m pytest -q tests/test_inspection_control.py tests/test_inspection_scoring.py
```

结果：30 passed in 1.11s。另执行本报告 P2-C/P2-D 所列四种首步失败注入和严格 JSON 序列化，count=4 两种情形失败、count=1 两种情形成功。

v1 长 trace 通过程序汇总，未手工逐行审读：

- straight.v1.json：30秒，300个10Hz样本；摘要高度79.196–80.000m，速度19.974–20.279m/s。
- orbit.v1.json：120秒，1200个10Hz样本；约4.25圈，最大绝对径向误差3.016m。
- four-orbits.v1.json：600秒，6000个10Hz样本；各约21.27圈，最大绝对径向误差3.023m；记录的最小间距279.941m，最低高度75.318m。
- 包线摘要按控制步统计，保存轨迹为10Hz；后者的极值不应冒充240Hz物理步安全审计。
- envelope_pass 没有路径误差或最小机间距门槛；注入“orbit任务恒定朝东”的错误 guidance 后仍可通过短时包线检查。因此该字段只支持包线结论。现有文档与 role 标签对此没有夸大。

未验证：风、任意路线/目标切换、退出圆交接、危险接近与碰撞场景、1×连续时钟、Actor权限边界、真实传感/报告数据链、全实体评分聚合、身份交叉关联资格、≥20 dev roots / ≥100 episodes的统计资格、至少两个现有方法的正式区分实测，以及人类专家有效性复核。

## Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- P2-C：四机首步失败仍无法严格JSON落盘；现有测试未覆盖 count=4 与序列化。
- P2-D：首步失败摘要的统计来源/覆盖范围仍未明示。
- 当前30项测试及13项边界检查只证明所覆盖工程契约；不能升级为完整资格、正式科学结果或人工复核通过。


## 最终修复复验（追加；此前发现保留为历史）

本节取代前文“仍需修复”的当前状态判断，不删除或回写中间版本的缺陷证据。当前已复核修复关闭 P2-A/B/C/D；本次范围内没有剩余必须修改的工程缺陷。此结论仍不升级任何完整 G1/G2/G1–G4、正式科学实验或人工领域复核状态。

最终复核版本：

| 文件 | SHA256 |
| --- | --- |
| eval/qualify_inspection_control.py | 9880f15f4c50e35c096d38fbc77777f8a49224ce5b4a446305f04eb0d61c2eb0 |
| tests/test_inspection_control.py | f459abc2798b72a5705ff65a95b29b82d680a1814ceaf8c99aa65987e7158888 |
| eval/probe_inspection_clock.py | 943a0e808f738718872157ada236f239b5f3e637fa122893aa18eb2149bed502 |

版本校正：前文记录的 tests/test_inspection_control.py 哈希 f459… 在并行修改期间采集；它实际对应本节的32项测试版本。前次30项测试运行结果及当时读取的单机测试代码确实存在，但未取得其独立冻结字节副本，因此不能把 f459… 回填为30项测试时的源码版本。原始三条飞行记录的 runner SHA 76fc…、第一次修复 runner SHA f6f9…、clock.v1 的 runner SHA 8c2f…、本节 runner SHA 9880… 保持分开。

### 已关闭缺陷的直接复验

- runner:141 将未观测的 min_separation 转为 null。重新注入 count=1/4 首个控制命令异常，以及 count=1/4 首个完成步的速度 NaN，共四种组合，均得到 envelope_pass=false 且 json.dumps(result, allow_nan=False) 成功。四机不再因 inf 丢失记录。P2-C关闭。
- runner:142–143 新增 summary_scope 和 valid_completed_control_steps。四种首步失败的 valid_completed_control_steps 均为0，范围明确为“initial state plus finite completed steps; excludes any invalid terminal state”。在当前固定初态80m/20m/s条件下，摘要初值现在有明确范围，不再冒充终止失效状态的统计。P2-D关闭。
- runner:101–112 在控制循环异常时保存终止快照或 snapshot_error。首命令失败的快照 t=0；非有限现场保留 nonfinite_indices=[[0,2,1]] 和对应 null。P2-A/B关闭。
- runner:117–120 将断连异常追加到 operational_error。额外只在断连调用时注入 RuntimeError，四机先真实完成2个有限控制步、simulated_seconds=1/60，结果失败且严格JSON序列化成功。初版注入夹具提前读取 PyBullet disconnect 时触发 client ID 副作用，未用于控制器判定；改为调用时断连后重新验证。
- tests/test_inspection_control.py:89–125 已把两项失败测试参数化为1/4机，并实际执行 allow_nan=False 严格序列化，检查 valid_completed_control_steps 和异常快照。

重新运行：

```sh
./.venv/bin/python -m pytest -q tests/test_inspection_control.py tests/test_inspection_scoring.py
```

实际结果：32 passed in 1.29s。评分组件及其测试SHA未变，本报告前述第6.2节审查和13项独立边界检查的结论保持其原有范围。

### clock.v1/v2 的结果与绑定复核

本节审阅现有时钟记录、当前源码和哈希绑定，未另行重跑20秒时钟探针。两轮使用5秒 blocking_sleep_surrogate_no_model_call，不是实际模型/API延迟测量，记录role也明确不是G3资格。

| 项目 | clock.v1 | clock.v2 |
| --- | --- | --- |
| 记录判定 | passed=false，历史真实失败 | passed=true，开发判据通过 |
| 实际协调器等待 | 5.0050782909384s | 5.0050370410317555s |
| 等待期间模拟推进 | 5.916666666666666s | 5.016666666666667s |
| 推进与等待之差 | +0.9115883757282663s | +0.0116296256349111s |
| worker最大滞后 | 0.9144536247166495s | 0.010748333351997038s |
| worker P95滞后 | 0.004708967078478593s | 0.0037974434660279495s |
| worker模拟总时长 | 20s/4800物理步 | 20s/4800物理步 |
| worker包线/运行错误 | true / null | true / null |
| probe源码SHA | 66bbdc439285f3e3f64876d67b74749966c1eee5e97039ad460c8edf42d3f049 | 943a0e808f738718872157ada236f239b5f3e637fa122893aa18eb2149bed502 |
| runner源码SHA | 8c2f9583a9930f291f694eec2f73ce73ed1f3eab9d65a4bc9ae93c4eda7d411b | 9880f15f4c50e35c096d38fbc77777f8a49224ce5b4a446305f04eb0d61c2eb0 |

绑定检查：

- clock.v1.json:22–24 的 worker_result_sha256 与 clock.v1.world.json 实际字节完全相符；v1 max lag见 world记录12–14行，超出0.1s开发容差。全局RTF约1不消除该瞬时滞后，因此仍保留失败，不因v2通过改写。
- clock.v2.json:22–24 的 worker_result_sha256 与 clock.v2.world.json 实际字节完全相符；source_sha256 与当前 probe 源码完全相符。
- clock.v2.world.json:59及64起保存 runner/control/PyFlyt源码哈希与嵌入快照。四个快照重新计算SHA全部与记录匹配，也全部匹配本次当前文件。warmup中的源码绑定与正式时钟段相同。
- probe_inspection_clock.py:30–37 在独立1秒加速预热飞行完成且包线通过后，再调用 fly 新建 seeded world，开始20秒实时段。预热的 requested_seconds=simulated_seconds=1、envelope_pass=true、operational_error=null 已保存于 clock.v2.world.json:19887 起。未见在任务中途重新设置wall_start或跳过积压的代码。
- runner:42 定义任务实时epoch，49–52行按同一epoch调度，59行逐控制步记录滞后。当前 probe:72–74 同时检查等待段推进误差和整段最大滞后，v2记录满足0.1s开发阈值。仅平均RTF未被当作通过依据。

历史可复现性限制：clock.v1 没有嵌入旧runner/probe源码快照，当前文件也不再匹配其旧SHA。本次能确认其失败数值、worker文件绑定以及版本与v2不同；未在限定目录找到旧SHA对应的源码字节，因此不能独立重建或重新执行clock.v1的精确旧实现。没有使用当前sourcehash回填历史v1。其失败原因中的“首次JIT积压”与v2预热改善相容，但精确归因未由本次审查的独立性能剖析证明。

### 最终 Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- 本报告识别的P2-A/B/C/D均已在上述最终版本关闭；旧发现保留为历史，不表示当前仍未修复。
- clock.v1失败、clock.v2带独立初始化预热的开发通过分别保留；v1旧源码精确重建仍未核实。
- 当前工程测试、单次20秒sleep替代等待探针和无风飞行烟测不证明完整时钟/权限/评分资格、真实模型集成、正式方法区分或人工复核通过。


## 决策轨迹审计（追加；7条决策）

范围：planning/decisions.tsv:2–8 的7条决策。按约定将 qualification/...、planning/...、inputs/... 解析为 research run 相对路径，tests/... 解析为仓库相对路径；全部证据文件均存在。本轮只读核对这些指针、qualification/RESULTS.md、qualification/validation.v1.json 和相关保留日志，没有重跑测试，没有扩大代码审查，也没有读取其他聊天。

限制：当前聊天完整transcript未提供。因此，本审计无法完成 show-me-your-work 对“完整本轮对话逐段对应”的核查，不能从本地文件确认所有决策的发生顺序、精确动作时间、未记入日志的转向或controller实际调用过程。依据为本审查实际读取/复验的范围与保存产物；下表明确区分直接复验、已有运行输出和未核实状态。

| TSV行 | 决策 | 证据与RESULTS/validation核对 | 结论 |
| --- | --- | --- | --- |
| 2 | 保留PyFlyt并用fdopen guard编译PyBullet | install-failed.log:9544–9551显示fdopen宏与SDK声明冲突；install-fdopen.log:2–5及16–17显示PyBullet3.2.7成功构建、PyFlyt0.29.0安装。planning/control-qualification.v1.md:28–35保存命令，environment.v2.json:4–12声明依赖可用；本审查此前实际导入并执行了PyFlyt物理测试 | 构建修复有证据；行内result“imports passed”比这两个安装日志直接证明的范围更强，存在指针精度问题，见下方A |
| 3 | 显式mode=-1共享传统控制器 | 工作单17–18行、surfaces.v1.json的独立舵面探针与本次已完成的源码坐标/真实物理复核一致；RESULTS.md:11、18、22保持开发范围 | 一致；不把控制符号探针升级为任务执行资格。“six-surface”实为五个舵面加电机的六个command通道，可更精确命名，但不改变已验证行为 |
| 4 | 保存直飞、圆周、四机600秒开发轨迹 | 三份JSON均envelope_pass=true；原始runner字节已归档且哈希匹配，validation:6–33分别保留30/120/600秒与原版绑定检查；RESULTS.md:12–18与本报告程序汇总一致 | 一致；all envelope checks仅指这些运行中的控制步包线检查，记录没有声称逐物理步或完整任务资格 |
| 5 | 记录默认无隐含任务总量上限的真实预算答复 | user-budget-answer-20261105.json确实含明确答复，budget-answer.v1.json:7–9引用及SHA与原始答复字节一致；RESULTS.md:5继续保留不新增付费API/算力/投稿授权的边界 | 用户答复及来源绑定已核验；result“confirmed by controller”无法仅由两个输入文件证明，见下方B |
| 6 | 保留首次时钟失败并在独立world预热 | clock.v1/v2.json的false/true、worker文件SHA绑定及v2源码快照已在上节直接核对；RESULTS.md:15–16和validation:36–58分别保留历史失败与短时开发通过 | 判定与证据一致；why中的首次JIT归因仍须作为推断，见下方C |
| 7 | 保存异常/非有限现场及四机严格JSON | 测试文件与本报告最终复验直接支持1/4机故障、严格JSON及32项focused通过；runtime/tests.all.v1.log:1–2为57通过，validation:61–63和RESULTS.md:24一致；本审查没有重跑57项 | 结果有保留原始输出支持；该行证据未直接指向57项日志，存在指针精度问题，见下方D |
| 8 | 实现固定身份的证据挑选与手算fixtures | 测试文件及本审查此前实际运行的combined suite支持20项评分case通过；本报告13项独立边界检查覆盖指定语义；RESULTS.md:23–25、validation:80–89明确全实体聚合/真实有效性判定/完整grader与人工复核未完成 | 一致；未把fixture数量当正式数据或方法区分证据 |

### Attention：弱证据与措辞

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- A，TSV第2行：两个直接指针证明“安装构建失败→成功”，没有独立 import smoke 输出，也不显示调用时完整CFLAGS环境。当前可运行状态已由本审查实际测试支持，故不判“imports passed”为虚假；建议将result写成其指针直接证明的built/installed，或追加planning命令、environment与实际测试日志指针。
- B，TSV第5行：已核验用户原始答复及输入文件SHA；未得到controller接收/应用该答复的回执或状态指针。建议追加实际controller回执/状态证据，或将result限定为“actual user answer recorded and hash-bound”。这是证据不足，不是判定controller未接受，也不是否定用户已有授权。
- C，TSV第6行及RESULTS.md:15：“首次JIT积压”是与首次滞后及预热后改善相容的解释。现有保存结果没有独立profiler证据区分JIT、初始化和调度开销，宜标“疑似/推断”；v1失败与v2短时通过本身已有直接数值支持，不受此归因限制影响。
- D，TSV第7行：32focused由本审查直接复验，57total由runtime/tests.all.v1.log:2与validation:61–63保留，数值一致。建议把该日志与validation直接加到evidence，避免读者误以为所指单个测试文件或本报告独立重跑过全57项。
- 所有7条证据文件存在，未发现RESULTS/validation把历史clock.v1改成通过、把32当57或把fixture声明成完整资格的矛盾。validation.v1.json列出的7个当前源码SHA也全部与文件实际字节一致。validation的clock.v1“pass”是绑定/历史判定保留检查通过，具体check名与detail已明确，不表示clock.v1运行通过。
- 完整聊天transcript未提供；无法据此宣布整段运行轨迹审计完整通过。上述4项均为证据链精度或归因措辞限制，本次没有发现会推翻已直接复核开发数值的新增工程缺陷。


### 决策轨迹补充：第8条澄清（TSV第9行）

已读取新增 clarification 行及更新后的 RESULTS.md:15。该行以追加记录明确取代旧clock行中确定性的JIT原因措辞；RESULTS.md现在写“启动积压后追赶……JIT归因与预热结果相容，但尚无独立性能剖析证明”。其两个证据指针 RESULTS.md 与本报告均存在，且本报告前述clock绑定审查已记载相同的归因限制。

本条 result“Observed failure and v2 result unchanged”与已核验clock.v1=false、clock.v2=true及既有数值一致；未把失败改为通过，未改变0.1s开发阈值或升级结论。前节Attention C的措辞问题已由这条追加澄清关闭。A/B/D仍是证据指针精度或controller回执尚未直接核实的限制，不推翻已核对的开发结果。

最终轨迹审计范围为8条决策（TSV第2–9行）。本轮未重跑测试、未改代码、未读取其他聊天；完整当前聊天transcript未提供的限制仍然成立。


### 决策轨迹补充：第9条直接证据（TSV第10行）

本次只读取新增 evidence 行所指的5项直接证据，未重跑测试或扩展代码审查：

- qualification/four-orbits.v1.json 记录真实PyFlyt0.29.0/PyBullet3.2.7运行600秒、144000物理步、envelope_pass=true；qualification/runtime/environment.v2.json:4–12记录无缺失包及available状态。结合此前已核验的原版源码/运行绑定，足以支持“imports passed”及实际运行可用性，前节A的直接指针缺口关闭。
- .paper/control.json 的精确JSON路径 operations.confirm-default-budget-20261105 现已直接读取。request_sha256为 b536471f25915a119377be3da93b5e4dcb73e301b4cd8b557764bffeb3a3c3c6；result.status=budget_confirmed，result.budget.limits={}，new_local_work_allowed=true。该controller应用回执与先前核验的原始用户答复及SHA共同补足证据，前节B关闭。本次确认的是该已登记operation的实际状态，不据此扩大付费/API/算力或投稿授权。
- qualification/runtime/tests.all.v1.log:2 实际保留“57 passed in 1.80s”；qualification/validation.v1.json:61–63直接对应repository tests pass及相同输出。前节D的全量测试日志指针缺口关闭；本审查未重新运行57项测试。

新增行采用追加补证而未改写旧行。其result“Pointers supplemented; recorded verdicts unchanged”与已读取证据一致。最终审计覆盖9条决策（TSV第2–10行）；先前A/B/C/D四项轨迹注意项均已通过追加澄清/直接指针补足。本次未发现剩余证据指针缺口或会误导开发结果的表述。

最终保留的审查限制：完整当前聊天transcript未提供，未扫描其他聊天，因此不声称完成对整个运行对话、所有动作先后及未入日志转向的完整审计。工程结果与科学资格的边界、独立上下文而非跨模型审查、human_review_pending均保持不变。

### 最终轨迹 Attention

reviewed by inherited parent model configuration (independent context; cross-model review not performed)

- No remaining evidence-pointer flags within the nine reviewed decision rows.
- Full-transcript audit not performed because the current chat transcript was not provided.





