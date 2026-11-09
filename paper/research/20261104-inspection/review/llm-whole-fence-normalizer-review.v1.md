# Whole-fence JSON normalizer：独立有界审查 v1

建议：**可以作为统一development framing修复采用，但安装时必须先保留现有raw类型/24,000字符上限，再unwrap并复用原strict checker。** 当前v2纯wrapper与旧checker直接组合会绕过raw长度上限，已有两个具体反例；本报告不批准省略该guard的接法。其他已查whole-wrapper语义与JSON/schema拒绝边界符合预期。没有修改active协议或原飞行，不能离线给旧失败飞行换Y。

本轮为同模型家族独立上下文工程审查，只有纯CPU解析与只读历史completed文本核对。新增真实模型调用=0、物理步=0、模型重试=0；未改生产源码、控制器、owner或旧结果。第一诊断开始时active strict协议和本审计收尾仍为 `c323f708fd2c2eb54a440758bd162785f3ee2fcebd95b9bca21eaf7ed6ef44d4`，不把候选或离线分析冒称已安装/已执行策略。

## 精确版本与最初发现的错绑

工作树 `/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。

| 文件 / 角色 | 实际SHA-256 |
|---|---|
| 拟采用 `qualification/llm-wire-normalizer-candidate.v2.py` | `d27f8e7aa0116afc042c9f02fda9e1d26b5e6c253b92788df581e5666841b59f` |
| 历史 `qualification/llm-wire-normalizer-candidate.v1.py` | `57b93dc87ef32283f7d289b8c2e82b777aad2d2e48a3cf495987c12ddfb4fe81` |
| prefix.v1，历史19条 | `84172f644d0b6d4ae4807693a39ea00b5cb8e61cd75d31566b067d64cf376588` |
| prefix.v2，完整已存26条 | `7d61c17af05b84ddda874d0861e061f3066c7711a5c816d53abf6569e4fdf4c1` |

最初实际读到的同名candidate.v1已经是d27f，但prefix.v1仍绑定57b9，存在不一致；初读源码/JSON与SHA原样保存在initial-source。主 Agent确认修U+2028后未另存版本，随后把历史v1通过逆唯一edit恢复，精确SHA匹配原prefix；d27f另存v2并建立prefix.v2。`llm-wire-normalizer-version-history.v1.md`明确这次恢复**不是早先atomic copy**。本审计没有抹去初读错绑，也没有把两版混成同一来源；当前两版artifact binding均已实核匹配。

## wrapper语义与旧checker复用

v2只在去除外部空白后，首行恰为允许的opening fence（裸或json）、末行恰为closing fence，并且中间没有另一个以fence开始的行时，移除一层完整边框。它不搜寻prose中的JSON、不选择多个块、不修JSON/参数、不替换工具名、不重试模型。去边框后仍调用原 `parse_reply`，因此字段、command数量、重复JSON key、非有限数、note/plan界限及工具schema校验继续由同一个checker决定。

独立24个纯CPU例子的实际结果：

| 例子 | 结果 |
|---|---|
| 原始JSON、LF json fence、裸fence、CRLF、fence外空白、JSON字符串内U+2028、字符串内backticks | 均可解析；字符串值保留，U+2028没有被替换为换行或删除。 |
| prose前置/后置、多块、嵌套fence、python language、opening与JSON同一行 | 不unwrap，旧checker拒绝。 |
| wrapper内部duplicate key、NaN、未知tool、多JSON object、9commands、工具缺字段、坏JSON、note513字符、非string | 即使识别边框也由旧checker拒绝，没有参数或schema修补。 |

v2按ASCII LF拆行，而非splitlines；这是U+2028在合法JSON字符串里保持原值的关键。CRLF留下的JSON边界CR属于合法空白；没有改JSON字符串内容。工具参数的数值/版本/证据合法性仍归共享worker，本轮的“parser接受”不是ACK、作业成功或评分正确。

## 安装前需保留的raw长度guard

两个实证反例，均使用同一个旧strict checker：

| raw输入 | 旧checker直接检查 | d27f unwrap后再检查 |
|---|---|---|
| JSON payload补空白到24,000字符，再加整层fence，raw=24,012 | reply length invalid | payload=24,000，被接受 |
| 24,000个外部空格+普通整层fence，raw=24,042 | reply length invalid | payload=30，被接受 |

不需要改变纯wrapper的目的；共享 `parse_reply` 接入顺序应为：先执行当前raw `str`/非空/长度≤24,000检查，再调用unwrap，再执行原strict JSON/schema流程。不要先strip/unwrap再把raw长度检查迁到缩短后的文本上。按这一raw guard组合，本轮24例为7个合法接受、17个非法拒绝，两超长反例不再通过；这是离线guard组合验证，**不是已安装生产实现的通过记录**。

边框容忍本身会改变格式成功率，应作为明确的dev协议版本冻结，统一给所有已声明B3/B4模型路径使用同一个函数和guard；不按某模型/架构选择性提取或开更宽语法。无需额外模型调用或参数修复。raw model response、token journal和原始SHA应继续保存，新的run绑定新的协议来源，便于区分旧strict诊断与新接口。

## 两版离线prefix的独立核对

只读 `logs/inspection-model-resource-operations.v1/dev-model-realtime-M1-B3-L3-00-v1-20261106/` 中artifact指定的已完成response.json文本；没有读取进行中world/score作为终态，也没有重跑模型。

| artifact与对应候选 | 已完成文本数 | whole fence | 旧strict接受 | 候选+strict接受 |
|---|---:|---:|---:|---:|
| prefix.v1 / 历史57b9 | 19 | 15 | 3 | 17 |
| prefix.v2 / 拟采用d27f | 26 | 22 | 3 | 24 |

两版每条raw_text SHA、ordinal、wrapper判定、旧/候选verdict均与原已完成response逐项完全一致；原response.status均completed。最大raw长度2911，不涉及上面的超长反例。两版prefix均明确models_rerun=0、no_counterfactual_Y=True；历史prefix.v1文件SHA未改变。

这些计数只回答“同一段已有输出去掉whole framing后，strict parser会不会接受”。它们不能证明新rollout能发出相同请求/命令、能及时到达、被worker接受、提高覆盖或Y。若更早的命令获得执行，后续公共状态、prompt和模型输出会改变；因此无法从24/26构造旧飞行的反事实成绩。原strict诊断的Y、失败和工具/作业结果必须继续按原run协议保留，本审查没有重算或修改任何Y。

## 实际命令与交付

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-whole-fence-review-v1/probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-whole-fence-review-v1/prefix_audit.py
```

另用同一24行probe结果按原raw类型/长度条件作组合核对，保存在guard-composition-check.json。精确probe源、初读/版本化源码、原prefix/historical说明和JSON/log保存在 `review/evidence/llm-whole-fence-normalizer-v1/`；`/tmp/inspection-llm-whole-fence-review-v1/` 原件保留。全部是有界解析，无新科学样本、速度/能力结论或G4放行。
