# 三模型实际文本 smoke：独立只读审计 v1

结论：三项已有真实文本smoke的原始operation、token meter、归档、来源和native结算链均通过本次独立重计。共 **3 calls、167 tokens =139 submitted input +28 runtime-observed output**；output全部有对应journal。三个owner ledger条目均为known，回执SHA精确匹配，审读快照中没有calls/tokens预留或unknown遗留，也没有重复计费证据。

它们仅证明三个固定本地模型各完成一次同一短文本请求并留下可核计费记录，不是飞行、方法性能、速度比较、能力评测或G4资格。本审计**未重跑任何生成、tokenizer、物理或大权重哈希**，未修改owner、源码、真实logs、旧回执或旧结果；新增生成调用为0。

## 重算结果与模型身份

| alias / 冻结repo | revision | seed | input | observed / journaled output | native known usage |
|---|---|---:|---:|---:|---:|
| M1 / `mlx-community/Qwen3.8-27B-4bit` | `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | 2026110605 | 53 | 9 /9 | 1call /62tokens |
| M2 / `mlx-community/GLM-4.7-Flash-4bit` | `1454cffb1a21737e162f508e5bc70be9def89276` | 2026110606 | 42 | 9 /9 | 1call /51tokens |
| M3 / `mlx-community/Mistral-Small-24B-Instruct-2501-4bit` | `51ed0c6d2f98a25d8a60f19994f0e86c944995e0` | 2026110607 | 44 | 10 /10 | 1call /54tokens |

三者request messages与各自probe逐项相同，均为固定system及“返回commands=[]、note=ready”的文本smoke。durable request的seed是native-int并与probe精确相等；没有drone工具执行。各token行的runtime_prompt_tokens分别为53/42/44，runtime_generation_tokens逐行从1连续到9/9/10；token/text和response完整文本一致，usage按input+observed重新求和。三者严格JSON解析均为 `{"commands":[],"note":"ready"}`。这些token数不用于跨模型速度、推理能力或效率排名。

registry原文件SHA为 `6d8547cb0597113d5411a771e1c75f807e43d769ecdfe4d92d32d41633e0fda6`；alias、repo、revision、小download receipt的实际SHA与registry固定值均匹配，receipt中分别列15/12/10个文件。未重哈希实际大权重；完整blob核验是已审服务在load前的执行路径，当前审计只核其固定源码/本地回执链，不能把它改称本轮独立完成了49GB权重验真或硬件attestation。

## operation → archive → usage receipt

每项实际原路径为：

```text
logs/inspection-model-resource-operations.v1/dev-model-smoke-M{1,2,3}-v1-20261106/
logs/inspection-model-smoke-M{1,2,3}.v1/transport/
paper/research/20261104-inspection/qualification/local-model-generation/M{1,2,3}-smoke.v1/
```

本审计独立逐行解析/计数，并用**冻结快照**的纯meter/wire模块只读复算；没有调用service/model代码。结果与probe.resource_recount及usage-receipt.resource_recount逐字段完全一致。

| 检查 | 实际结果 |
|---|---|
| 预派发expected、purpose、intent | reservation、owner frozen plan、registry/model/service SHA、reserved与actual8字段预算一致；实际配置1call/16384total/8192input/64output/60s、temp.2/top_p.95/thinkingFalse。三者intent PID为正整数且PID=process_group。 |
| 作业/响应数量 | 每项恰有call-0001；request、tokens、response、completion齐全。transport恰2个JSON envelope：同reservation ready与逐字段完全相同的response。 |
| request / response bindings | seed/messages回到probe；meter保存的request/response/token SHA精确匹配原文件；response全durable字段与probe transport响应一致，parent附加ordinal/end-to-end时间保留。 |
| 计数和状态 | response与completion均1call、input+observed tokens；output observed=journaled，无未journal差额；status completed且deadline_exceeded=False。实际保存的墙钟字段合法，但本报告不比较其快慢。 |
| 完整zip | 每个zip恰7个成员：operation三call文件+intent/completion和transport stdout/stderr。无重复成员/符号链接；所有成员与原文件**逐字节相同**，zip SHA等于usage receipt所记SHA。 |
| probe / receipt | qualification probe与原logs probe字节相同；usage receipt保存的probe SHA、archive SHA、模型alias/repo、计费、解析结果都匹配。 |

M1/M2 stderr为空，M3含保存的transformers TokenizersBackend提示；它没有导致本次请求/receipt失败。此记录不构成额外tokenizer或硬件证明。

## native reserve / settle 与无重复计费

三项reservation依次为 `dev-model-smoke-M1-v1-20261106`、`…M2…`、`…M3…`。本审计只读 `.paper/control.json`，将对应reserve/settle原packet按owner相同规范独立canonical digest，再比operations与history的request_sha256：六项全部一致。每项history恰有一次reserved与一次known settle，operation ID与packet没有复用成不同参数。

| alias | native已知usage receipt SHA-256 |
|---|---|
| M1 | `25ea9e28dae2ce8121c62d2e1a2accf0a4d1765e4014de217c3d02f171d1824e` |
| M2 | `9bbc1ebc0c8618502054a496ee717d3f88ca55d8b7477e1f3fb50fe9e48707ab` |
| M3 | `b84a75ad766e13d8e55a200e913edf5b9d576091dbbc54dfcb064e983434e62d` |

每项ledger.status=known、unreserved=False，reported_usage与上表精确一致，receipt路径/实际SHA/receipts唯一项一致。三receipt路径与SHA互异，三个原operation各恰1call。native结算历史known累积为1call/62→2calls/113→3calls/167，未额外累计第二次调用。记录中的reserved={1,16384}保留历史预约额度；它不是尚未释放的活跃预留。最后一次native budget摘要为reserved.calls=0、reserved.tokens=0、unknown_usage=[]、unresolved_usage=[]，审读快照ledger总共3条且全known。

成本维度须单列：native摘要仍把三项cost记为unreported_fields，receipt亦声明capital_energy_cost_unmetered=True。这里的“known/无unknown遗留”限定于所声明的本地calls/tokens，不表示能耗、资本成本或完整宿主账单已经计清。

以下原始时间均为2026-11-05 UTC（北京时间为11月6日）；微秒原值保存在audit.json：

| alias | reserve | probe started | durable request文件mtime | durable response文件mtime | settle |
|---|---|---|---|---|---|
| M1 | 20:28:49 | 20:29:11 | 20:29:22 | 20:29:27 | 20:31:59 |
| M2 | 20:32:26 | 20:33:23 | 20:33:34 | 20:33:37 | 20:36:38 |
| M3 | 20:37:05 | 20:40:06 | 20:40:14 | 20:40:16 | 20:46:23 |

各项intent→request→response→completion的现存mtime也按序位于probe开始和settle之间。native记录、probe时间、文件mtime及固定服务的“先检查live reservation再load/生成”路径共同支持先reserve→生成→settle的本机证据链；文件mtime不是不可篡改时钟或独立硬件attestation。

## 真实来源快照与执行边界

三probe声明的全部5个Python源，其完整快照均在 `qualification/source/model-orchestrator-v2/`，并与probe所记SHA及当前源码逐项一致：

| source | SHA-256 |
|---|---|
| `probe_inspection_model_generation.py` | `53dff69f11ff8065ffabf28ed301940b0e91ecc9e665b085b0cfb1a609ca8190` |
| `inspection_model_service.py` | `64e8307ece55b1d6928e91607a269dfb082926265f1d98a2bde90a5931eb2e8f` |
| `inspection_model_client.py` | `40ccd5f5f4100e4ed42a6ac22ecf35f3bc24a9db4e2b57f1826625b78afcb158` |
| `inspection_model_receipts.py` | `922372548ceb972e9f95d1c52923cfe7d9dd6ce12dba2feec3b2d44a99167683` |
| `inspection_llm_protocol.py` | `c323f708fd2c2eb54a440758bd162785f3ee2fcebd95b9bca21eaf7ed6ef44d4` |

这里的完整快照指这5个声明的源文件，不是Python解释器、MLX/transformers全部依赖、内存映像、权重实际硬件装载或GPU执行的签名证明。现有脚本/metadata不能提供这类attestation，本报告亦未补造。

本次实际执行唯一重计命令：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-actual-three-model-smokes-review-v1/audit.py
```

audit.py独立核token/text/totals、严格JSON、每个archive与原件、native packet digest/去重/顺序，并调用冻结的纯reader作交叉核对。未导入MLX/PyFlyt或调用owner运行时load/save/settle。源码/摘要/ledger定向摘录与所有输入SHA保存于 `review/evidence/actual-three-model-smokes-v1/`；原实际日志及archive不改。完整control在读取时的SHA也保存，但仅复制相关三ledger/六history/六operation的摘录，没有把整个可变owner状态复制为一份新的权威来源。

本轮没有发现这三项已完成文本smoke的计费/结算绑定缺口。它们不是新策略样本、真实任务episode、飞行延迟实验或能力比较；无多重复统计、完整context、LLM episode/SDK/G4、外部unknown恢复或硬件计量资格由此自动成立。
