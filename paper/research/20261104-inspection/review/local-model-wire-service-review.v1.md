# 本地模型wire/service独立工程审查 v1

日期：2026-11-06。审查者 `/root/mission_review`，同模型家族独立上下文，不是人类领域或跨模型审查。仅新增本报告及附属 `local-model-wire-service-review.v1.evidence/`，临时结果在 `/tmp/local-model-wire-review/`。未修改生产文件、真实.paper、预算ledger、模型文件或已有结果；未启动physics、下载、真实MLX load/forward/generate或GPU操作。经典1×期间仅做低负载小测试。

## 结论

**真实frozen_plan=null的负向准入成立，真实model_generation=0；当前仍不是正向硬件、时延、episode或G4资格。** 本次真实main负向调用在预算准入处拒绝，尚未读取模型receipt、更未import MLX，真实control字节不变且没有创建operation目录。

19项已有单元测试通过。另通过sys.modules注入纯FakeMLX，验证了完整service正常路径、同reservation重复启动拒绝、三个具体故障和模型清单缺失，不调用真实后端。结果揭示五类仍需解决或由上层正式契约证明的缺口：owner状态准入过浅/非有限caps获准、模型pin缺完整清单绑定、journal IO错误下计量不实、超时空流被标completed、嵌套grader字段通过serializer。

没有观察到真实预算绕过、真实超支或真实隐藏数据流入模型。这些是候选实现的有界反例，不能以它们虚构一次真实生成；也不能以当前负向安全和19个小测试宣称正向service已合格。

## F1 [P1] 服务自身不足以证明“完整计划+当前round+足够有限reservation”

位置：`eval/inspection_model_service.py:13–24,66,97`。当前只检查frozen_plan truthy、历史存在某个stage=plan receipt、active_round不是None、entry.status=reserved及两个比较。没有验证完整plan/receipt hash的对应、active_round结构、reservation与当前round/模型/方法的绑定，reserved caps也没有native finite integer检查。

独立反例只写/tmp伪owner state：

- frozen_plan={fixture:true}、receipts=[{stage:plan}]、active_round={}，reserved calls=1/tokens=50，admitted_limits仍接受。这不是完整研究/执行计划。
- reserved calls/tokens为JSON NaN，申请calls=100/tokens=100000仍接受，因为NaN<requested为false。不是把CLI请求的整数上限成功转换成有效owner额度。

真实controller由唯一owner维护，当前frozen_plan为null且无该reservation，所以以上并非已经突破真实controller。父任务确认owner的有效frozen_plan实际是digest字符串，正确修复应核对plan receipt的canonical digest与activeRound关联，不能把它误当另一份plan对象再造预算schema。若上层能保证并冻结这些状态不变量，应给出同owner权威验证及完整正例；服务当前这段浅检查本身不能承担其错误消息所声称的完整计划保证。建议调用owner现有validator或严格验证其实际schema/关联，拒绝非有限/非整数额度，并把round/plan/reservation及关键CLI配置绑定到intent。不要另建一份独立预算账本。

每个请求会重新读取owner状态，这是正向防线；service也不写owner ledger。但模型SHA检查发生在第一次准入与load之间，加载前没有再次检查reservation。对“加载时仍有效”的声明，需要owner lease/撤销约定或相应重新校验；本次未注入该并发变化，不把静态窗口描述成已观察到的真实消耗。

## F2 [P1] “校验清单中每个文件”不等于完整固定模型pin

位置：`inspection_model_service.py:35–47,67–80`。verify_model_files能检查已列文件的大小、路径不逃逸和完整SHA，现有同大小改blob反例也正确拒绝。但它接受空files，未验证清单完整性/重复项、模型revision/身份或与frozen plan中批准receipt的对应；receipt自身SHA只事后写intent，没有与预批准值比对。

FakeMLX反例使用/tmp小目录及files=[]的receipt，service跳过所有blob校验，仍到达Fake load及response。没有真实加载任意权重，也没有扫描真实十余GB文件。这个结果说明截断/错误receipt可以静默取消“全文件SHA重核”，不是说现有下载receipt已被篡改。

应至少拒绝空/重复/结构错误清单，并把批准的完整下载receipt digest、revision/模型身份及必要文件集合绑定到owner冻结输入。若调用方负责完整pin，需将该调用方验证与service放进同一可审查入口，不能仅凭此函数成功宣称固定模型已核验。

## F3 [P2] IO失败时journaled token计数与文件事实不一致

位置：`inspection_model_service.py:109–123`。count和text在write_once(token...)前递增；token写异常虽被捕获并标runtime_failure，count仍被用作emitted_tokens_journaled和usage，汇总也沿用该数。

Fake流输出一个synthetic token，注入ENOSPC使token文件在写入前失败，结果：

```text
token files = 0
response.status = runtime_failure
response.emitted_tokens_journaled = 1
input_tokens_submitted = 2
response/service-completion usage = calls1, tokens3
```

这是“已观察到一个输出”与“已成功journal一个输出”的混淆。不能简单把已消耗的生成免费归零；应分开observed emission、成功落盘journal及未知/失败状态，并明确owner按何种证据settle。至少不能用journaled字段声称不存在的文件。request/response/completion写入本身及load/template/tokenize阶段也可能异常退出，当前未统一终态；已存在intent/partial目录需按unknown/需核对处理，不能自动以0消耗或重启服务覆盖。

本次没有要求service写预算状态。唯一owner应从保留的request/token/response/partial证据结算；它们是计量输入，非新的预算owner。

## F4 [P2] request秒数上限不是当前代码的完整超时保证

位置：`inspection_model_service.py:91–115`。计时从template/encode后才开始，elapsed只在一个chunk产出并journal之后检查。首次yield阻塞、load、tokenize或无chunk结束没有等价deadline处理。

Fake时钟反例：max-request-seconds=1，Fake stream耗时10s后无任何yield结束，返回status=completed、text空、wall_seconds=10，而不是policy_timeout。没有真实sleep、MLX forward或10秒硬件等待。真实first-token/流停顿也无法靠循环体内的检查强制中断，这是代码推论，不是本轮硬件实测。

应至少在无chunk/循环完成时正确分类elapsed，并定义请求总时延、生成时延和outer wall deadline。若要求硬上限，需要可终止的外层进程/监督与保守部分用量核对，不能将协作式chunk检查称作硬timeout。仍须保留policy_timeout、模型格式失败与runtime/IO失败的不同原因。

## F5 [P2] 8个顶层字段加有限黑名单未覆盖所有grader数据

位置：`eval/inspection_llm_protocol.py:19–21,70–86`。顶层必须恰为8个public字段，递归阻止当前列出的私有key及下划线前缀，这些已有反例通过。但nested结构不受public section schema约束，FORBIDDEN_FIELDS缺少实际评测score及其他grader/truth名称。

构造恰有8个合法顶层key的frame，在task_card下加入score={Y:1}、truth={target_xy:[1,2]}、grader_output，serialize_public_frame成功输出全部字段。此fixture没有把真实隐藏信息发给模型。当前Mission.public_state本身按明确白名单生成安全内容，本次未发现它实际加入这些字段；反例说明serializer不应独自被称为完整隐藏真值/评分隔离保证。

建议按实际public section schema拒绝未知内部字段，或至少从当前评测审计结构补齐私有key并保存producer→serializer的真实快照检查。service只校验messages的角色/文本形状，不证明调用方确实调用这个serializer；prompt adapter/episode接线仍需独立检查。不能用字符串黑名单证明任意prompt文本都无隐藏信息。

## 已成立的边界及重复消耗范围

- parse_reply拒绝重复JSON键、NaN/Inf及溢出非有限值、尾随文本、markdown围栏、未知tool、缺/多字段、过深输入等，统一抛ModelReplyError；这个类明确是policy-format失败，不是免费基础设施重试理由。
- 7种tool字段集合与当前共享worker一致，批量最多8命令；note/plan及总长度有限。参数数值、version、证据及任务语义明确留给共享worker，wire接受不意味着作业成功或报告得分。本报告不把这个有意分工误报为decoder缺陷。
- 公共frame时钟有限且0..600，顶层集合精确，既有seed/entity_key等递归私有key确实被拒；F5限定未覆盖的内部字段。
- 正常FakeMLX路径使用enable_thinking=False、tokenizer_config.local_files_only=true、trust_remote_code=false；2输入token加1个成功journal输出，response与completion均calls1/tokens3，token文件1。
- R/resource-operations/resID的mkdir exclusive阻止同reservation第二次启动，实测FileExistsError；计数不会因同ID进程重启自动归零继续用。同一reservation的owner ledger在正常Fake服务前后相同，service没有创建/settle预算项。
- 同一服务内used_calls/used_tokens及保守input+max_output检查限制本地请求；预算不足发not_started。资本、加载/散热、能耗和FLOPs未计，completion明确标capital_energy_cost_unmetered，不能拿token账本称总计算成本。
- Mistral仅在显式--fix-mistral-regex时传fix_mistral_regex=true，服务不根据receipt主动强制该flag。冻结调用设置必须包含此项；本次没有验证真实Mistral/Qwen/GLM的tokenizer或无thinking行为。

CLI/model/template/请求设置目前不全部写入service-intent；后续可复现资格应保留max calls/total/input/output/seconds、fix flag、温度/top-p、seed与完整模型pin。空/异常stdin、模板/加载失败和owner撤销的终态不是由这19个单元测试穷尽覆盖。

## 真实负向调用与Fake边界

本次读取真实control发现frozen_plan=null，使用不存在的review-negative-no-reservation进行真实main负向调用。它在admitted_limits抛ValueError；传入的model-receipt路径不存在但未被读取，MLX import在更后位置不可达。control字节前后相同，真实resource-operations对应目录不存在。真实generation、GPU操作均0，未改真实.paper。

所有正向service路径的mlx、mlx.core、mlx_lm、sample_utils都在隔离审查进程sys.modules中替换为带fixture标识的纯Python模块；Fake load返回普通object，Fake stream只yield手写SimpleNamespace。没有接触真实MLX模块、权重或GPU。Fake token只是检验计数/文件分支，不是模型能力、硬件吞吐、首token时延、MLX版本兼容、ctx上限或episode证据。

当前单元测试主要覆盖wire及admission/blob函数；本次补充才到完整main后半段。真实硬件前还需关闭/界定上述缺口，并用真正完整的owner plan/reservation和固定模型配置取得受控正向证据。现阶段不宣称完成LLM episode adapter、端到端格式分类、真实模型时延或G4 GO。

## 命令、版本及证据

实际命令在仓库根执行，均退出0：

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/local-model-wire-review/probe.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. .venv/bin/python -B /tmp/local-model-wire-review/negative.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_llm_protocol.py tests/test_inspection_model_admission.py -q -p no:cacheprovider --basetemp /tmp/local-model-wire-review/pytest
```

测试为19 passed in .01s（12 wire、7 admission/blob）。独立probe约.07s，真实负向调用仅标准库控制文件读取，没有重负载。probe退出0表示成功捕获并断言上述行为，不表示其中fail-open/计量/timeout行为正确。

长期证据为本报告同目录 `local-model-wire-service-review.v1.evidence/`，含两个探针/JSON、测试日志和受审源码副本。binding.json SHA=`0aa213536b7722593b38e71871af8d83786a97db5f634069e038f26ade43e84b`，绑定全部七个证据文件及当前四源；复现使用新的/tmp目录，禁止把fixture control当真实owner状态或把Fake backend替换成实际MLX运行。

| 文件 | SHA-256 |
| --- | --- |
| eval/inspection_llm_protocol.py | `88c166ae80ed95be68c13058bbfe3b056789f011aba8b8d6ab4d180c4c6b5ef1` |
| eval/inspection_model_service.py | `b88720e9fd7937235f5c90b116cc3fad688efa92b7477e699af78711370beff5` |
| tests/test_inspection_llm_protocol.py | `520a72aa37293e5add08506f095810ad2ddcba8a6d6456276541651018bcb402` |
| tests/test_inspection_model_admission.py | `ea3ed5e4f923485d4fba3f6e75bf813ff7c08c2d0dad5777ad0f43944341417e` |

父任务已接受上述五类反例，将另版本修复owner/receipt绑定、private key集合、runtime/journaled计量及soft result deadline；硬GPU取消留给未来client监督。这些是已接受的修复方向，不是本轮已执行或已通过内容。

HEAD=`270c6f0946b3be59527b2b77982f75a70da7ef77`。工作树候选代码未提交，报告仅绑定本轮实际SHA，不为后续修复、模型调用或平台资格预先背书。
