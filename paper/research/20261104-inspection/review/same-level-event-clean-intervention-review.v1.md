# 同公开级别 clean 事件干预：独立定向审查 v1

结论：冻结 v2 在本次限定的输入验证、同级别 L3 接线及 JSON 往返范围通过定向检查；未发现阻止这对固定开发输入 dispatch 的 P1。v1 的真实往返缺陷已独立复现，v2 修复闭合。此结论不授予飞行、精确 replay、科学因果效果、main/holdout 或 G4 资格。正在进行的实际飞行不作为本报告结果。

审查人为同模型家族的独立上下文工程 Agent；只写本报告和对应 evidence，不改实现、其他作者文件或旧证据。未构造真实 PyFlyt world，未运行模型、600s 飞行、物理重放或总体生成。检查使用既有 OOD-L3-00 根输入、冻结模块与显式 fake 后端；没有重新抽取根输入。

| 版本/来源 | 实际 SHA256 |
| --- | --- |
| v1 source-binding.json | cb05161b9986403e81e1776ee067be9efa81fca534cf9e7f3f9a8ada12650c0d |
| v1 inspection_population.py | 8780356e3281dd18f6ebd2b75b69df038f48470f7d7443c1c8438da6d10705c4 |
| v2 source-binding.json | 1c96bd5343759d5bc142dc619d0b2ca6afd74c9d58dc3653fb4a04be42b0f3f1 |
| v2 inspection_population.py | bb7098554475a65d4717077b7704bcf5902cd0fc674c1a9a95a51a7e066a5ff7 |
| inspection_cases.py，两版不变 | 7537b17aa0cbcfd4cffdb00c1d35e45e56217f477e055db77ba80724c0868cef |
| inspection_scenarios.py，两版不变 | 9159ba44790d6d315074a824e52adeebd96468ab3032c6f0f07fdb755ce2753f |
| test_inspection_population.py，两版不变 | 5d44cfbfbe4ca38aaa2f399eb0c620d0d41bdbe2ccfe7df7ecec22c480570753 |

逐项重算 v2 binding 的全部 17 个 snapshot 文件；均匹配声明 SHA，检查时 live 对应 17 文件也均匹配。完整清单见 `evidence/same-level-event-clean-intervention-v1/results.json`，结论绑定冻结快照，不跟随未来 live 修改。

| 检查 | 实际结果与边界 |
| --- | --- |
| 完整合法父输入 | `EventCleanSibling.from_dict` 先经 `PopulationCase.from_dict` 校验内嵌父输入（population.py:46–67），再要求 L3 及父语义 digest。父 split、namespace、几何、运动、容量、日历仍走普通 validator。`event_clean_sibling` 也先完整验证父输入。没有以 clean 类型绕过普通 L3 校验。 |
| 限定干预 | population.py:70–79 用父 `as_dict()` 构建唯一期望值，九个 `EVENT_FIELDS` 置 None，外层 schema 改为显式 sibling schema；其余 base 字段 JSON 表示必须完全相同。实际 base 差异恰为 schema 加九事件字段。seed、domain、level、身份、ROI、实体、wind 保留，六个时刻逐实体 position 一致。新增 parent_case/parent_sha256 是证明元数据，不是额外任务因素。 |
| 普通 L3 不被放宽 | 直接把普通 PopulationCase L3 的九字段清空仍拒绝；L2 父、无日历父、非法持续时间父、main/holdout 父、非法几何父均拒绝。原 validator 的 L3 日历区间未修改。 |
| 类型及未知字段 | 生成和 JSON 读回均为 `EventCleanSibling`，不是普通 PopulationCase；外层、parent 和 entity 未声明字段都拒绝。root level/seed/domain/identity/wind/geometry/motion/parentSHA 等篡改拒绝；逐一恢复九个事件字段中的任一个也拒绝。最终共 33 个拒绝用例通过。 |
| 同级别公开初始输入 | 实际 ScenarioMission 类 AST 配显式 fake 父实现：两侧同 seed/wind、初始代理位置、coverage cells、初始卡逐字段相等，level 均 L3。真实 prepare 保存的 t0 卡、输入 manifest/binding 另行核对字节/语义 hash；没有独立重新构造真实 world。原 adapter 仅从外层 case 字段执行，不读取 parent_case 日历。 |
| 事件执行边界 | 真实 adapter 方法在 fake 对象上：clean 对全部机/所测边界 online=True、camera=1；t=600 `_events` 无退休、卡版本更新或事件提交。父对象仍在 link_start 断链、link_end 恢复，camera_start 降质，最终 priority/retirement 各一次。此为方法边界检查，不是物理正例。 |
| 实际 reader 路径 | 冻结 worker AST 配 fake mission，读取 pair 的冻结文件并验证 expected fileSHA，event 成为 PopulationCase，clean 成为 EventCleanSibling；两者 duration=0、fake physics counter=0、completed。warmup 也是假实现，未运行真实物理。common-calendar 实际入口 AST 接新 reader 接受 clean 几何/父事件日历；评分函数为显式 stub，因此只证 schema 接线。 |

v1 的保留缺陷：生成 clean 的 parent_case 内 ROI 是 tuple，JSON 读回后为 list，`old_loaded != old_clean`，但两者 `digest()` 相同；独立原版复现与主 Agent 保存的 `1 failed, 107 passed` 一致。v2 population.py:55 用 `PopulationCase.from_dict(...).as_dict()` 归一化嵌入父输入，修复后 dataclass 和 digest 均相等。本报告不把 v2 结果倒灌 v1。主 Agent 保存的 v2 日志是 `108 passed in 36.89s`，日志 hash `d916617d0f3e1307e586ba773f105aeac89d68708a2f1402a9d2c96d6ccd8d2f` 与 checks-binding 一致；这是读取并绑定的外部测试，不冒称本 Agent 重跑 108 项。

本次实际命令（cwd 为仓库根）：

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python -B paper/research/20261104-inspection/review/evidence/same-level-event-clean-intervention-v1/probe.py > paper/research/20261104-inspection/review/evidence/same-level-event-clean-intervention-v1/probe.stdout.json
```

最终 exit=0，`passed=true`；完整可执行 probe、结果及标准输出保留。probe 使用 importlib 从冻结 v1/v2 加载纯 scenario/cases/population 模块，AST 执行冻结 adapter/worker/common-calendar，避免导入真实 mission/flight。定向探针先运行 25 个反例，随后补逐事件字段检查到 33 个；最终 stdout/results 对应后一次。

固定输入重新核对：event 文件 SHA `fc992f5e83c8614e8594af381ef26725450324e845339532043aff98f41a740a`、语义 digest `be9e0b050f0192e4bfdeddfbb054d5c5c99625b904313555bdaba9eac21b8983`；clean 文件 SHA `a1556317b555b8f201596a6705192dc21784d3444863df6446734dca958e7fa6`、语义 digest `99b367178b3ad36b65eb037e035b83fd1e4ae4682b57ac7123982454787fb2fc`。均与两个 manifest 及 pair binding 匹配，parentSHA 是 CaseSpec 的语义 digest，不是文件字节 SHA。该内嵌 SHA 只证明内部一致；外部预先冻结的 manifest/fileSHA 才约束使用哪一个根，不能将自洽重写的整个父/子输入冒称原根。

保留的元数据与工具边界：worker `run_inspection_reference.py:102` 的 `world.events` 由 `case.level == 'L3'` 得到，新 clean 也为 true。它在此是 level 派生元数据，不是激活开关。执行由九字段控制；`ScenarioMission` 给基类明确传 `events=False`，覆盖 online/camera/_events/weight 后按 case 日历运行。不能把这个布尔值当事件开启证据，需联合 schema、parentSHA、冻结九字段、干预计划及实际 journal/状态审计认定。

已用 `rg` 检索当前 `eval/`、`envs/`、`tests/`、`experiment/` 的布尔 events 下标/get 消费者，原始命中保存 `events-consumers.txt`。未发现 `world.events` 驱动此开发运行、grade、common-calendar 或 endpoint 判分。两个旧 `run.events` 消费者有明确边界：`analyze_inspection_qualification.py:65` 用固定机2/300–345分组；private-case 的 run manifest 把 events 置 None，不能拿此旧分析器证明随机开发日历传感器分组。`replay_inspection_reference.py:138` 仅非 private 分支用 run.events 激活旧 InspectionMission，private 分支读冻结 case；审查初读时仍用 CaseSpec reader，不能加载新 population/sibling schema。主 Agent 随后提供附加 `public-contract-clean-candidate.v2/replay-reader-repair.py`，SHA `72a6898fc9280215b9a60fd74b680dc092f601124cad9f403bd8f420c270486f`，改为同一 `load_development_case`；已读到修复接线，未执行完整 replay，不给回放资格。此附件不属于当前飞行 17 源清单，不回填原飞行 manifest。

剩余待证是这对同公开级别干预的实际完成、完整原始/共日历评分链、实际事件/停采/迟到包、及新 schema 的真实精确 replay；本报告没有用旧 L3→L2 clean pair 充当该新输入的结果。静态校验给出输入一致性与必要几何/容量条件，不能推出策略可解性或强性能，更不能推出总体事件因果效果。
