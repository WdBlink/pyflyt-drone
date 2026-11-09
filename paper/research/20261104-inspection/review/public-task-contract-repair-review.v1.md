# 公共任务契约三项说明修复复核

审查者 `/root/control_review`，同模型家族、独立工程上下文。本轮仅按冻结候选做source-only复核，关闭[原报告](public-task-contract-sufficiency-review.v1.md)的三项P2说明缺口；不读取pilot成绩/actor输出，不运行模型、物理、候选模块、测试或反例fork，不修改执行源与旧证据。

## 结论与绑定

**原P2-1、P2-2、P2-3均已关闭，本范围无剩余必要说明缺陷。** 结论只代表所列公共说明与既有任务语义一致，不授予G4、main、事件干预/生成器设计或实际运行许可。

实际输入是 `qualification/public-contract-clean-candidate.v1/source/`，其source-binding SHA为 `cb05161b9986403e81e1776ee067be9efa81fca534cf9e7f3f9a8ada12650c0d`。独立核对归档精确13文件集合和全部SHA，关键候选为：

| 源 | 实际SHA256 |
| --- | --- |
| actor | `cd1bf019c637d447946cba30e1e2d8572c4ff90594de4b5b2d90f49cd65ccc7b` |
| LLM protocol | `b1c2f73c3894058105cea8b3b6088128fbe44fee29bda9854423ce3c0c3d19ed` |
| scenario | `6e7517b7b515d9f1a9cb86001112f4abd101939976dd3e044f0889a4665a5d22` |

ReportLedger、scoring、mission、execution、reference与classic六份任务语义源，与原审查归档**逐字节相同**。绑定中另四份population/cases/scenarios/test文件只核SHA与集合，未审其实现、总体/事件或科学设计；候选binding中涉及它们的role不随本报告获得批准。以下行号均对应本报告独立归档。

## 三项闭合对应

| 原发现 | 新公开说明 | 原任务权威及闭合判断 |
| --- | --- | --- |
| P2-1：首次ID与“copy”冲突，修订生命周期不完整 | `inspection_llm_actor.py:55–57`明确首次自建、之后复制accepted ID，并追加PUBLIC_TOOL_CONTRACT；`inspection_llm_protocol.py:21–25`给1..64字符、持久槽、unsupported claim也可能占槽、固定object_type/首关联和storage-only ACK | 对应`envs/inspection/protocol.py:73–96,105–126,138–140`。没有把revision说成换目标或重置关联，亦未返回实际关联key/正确性；关闭。 |
| P2-2：有效证据对应条件不全 | 新LLM protocol26–33行给非空unique收到的IDs、同candidate、已到达、有限stamp且[0,now]、全部测量<=stamp且至少一项相等、全部quality>=.8；明确schema-valid低质量/unsupported仍可能ACK并占槽 | 对应ReportLedger89–104、117–126行。正确区分有效交付与schema拒绝；未凭quality数字承诺class/location正确，也没有新增多证据/inspect-only/数值逐字节复制硬门；关闭。 |
| P2-3：自动上传、seal及两种时间窗不全 | 新LLM protocol37–52行说明observe/handoff已有report、采样时绑定route、到地面自动append、无须每格手工submit、离线保留、job失败不抹数据；dynamic首次measurement封存/重复幂等；全局appearance180s/3s格+6s交付与job首个新qualified本地测量起算分开 | 对应mission247–276、289–292；ReportLedger123–135；scoring74–77、94–103。没有用job窗口重置全局需求，也没把晚上传说成新测量；关闭。 |

静态检查确认PUBLIC_TOOL_CONTRACT是字符串常量，actor的SYSTEM表达式直接追加它；不会只定义一段未发送的说明。所有actor函数AST（10项）与LLM protocol函数AST（6项）均与旧版本相同：新增的是公共说明与SYSTEM连接，没有改解析器、公开投影、工具白名单或执行/评分行为。检查脚本只解析源码，不import/execute候选模块。

## truth、策略与确认消融边界

新增内容只陈述公开输入/输出和交付规则；没有实际truth identity/class/position、隐藏appearance数值、未来事件、定位误差、正确性反馈、隐藏分母明细或root/seed。LLM protocol54–57行的公开字段/禁入字段集合和serializer函数均保持原值/原AST，原ReportLedger响应也未改成correctness oracle。新增31–33、51–52行还明确离线正确性判定和隐藏信息不可用。

没有加入经典majority vote、15m去重、30s年龄、470s退出、竞价cost或调度顺序答案。新34–36行给出inspect的两个实际qualified views/20m job完成事实，并明确它**不是报告硬门，报告无需等待inspect成功ACK**；对应execution151–165行和Ledger实际准入，未改变grader。

确认消融逻辑仍为actor204–208行只移除原来的“two high-quality distinct-view confirmations”优先指导；新公共job事实在baseline和ablation中都保留。这区分了共享工具含义与被移除的确认优先指导，未把“先完成两视图再允许报告”的新策略偷偷补进所有条件。符合原已接受设计625–640行“相同证据/工具/评分准入、移除确认指导”的边界。此处只证明本次说明改动未改变该单因素接口；不证明消融的实际效果、充分机制强度或完整G4资格。

## qualification_status与限制

scenario相对原版唯一差异是删除 `qualification_status=development_unqualified_new_modes`；对原AST移除该键后与候选AST完全一致。没有增加qualified/G4通过标签，也没有改航程、安全、传感、执行参数或权限。它关闭了原无scope资格元数据的说明风险，不把未运行事项升级为通过。

本轮没有验证新增SYSTEM在所有真实tokenizer上的上下文长度、实际模型理解、因果执行、时钟、统计/事件设计或效果；没有查看旧pilot成绩来选择规则。原报告和原prompt/结果证据保持不变，新的说明不能追改旧请求、旧Y或既有attempt。后续源码组合、完整G4及资源/运行门仍须各自正常验收。

完整规则对应见 [closure-map.v1.json](public-task-contract-repair-v1-evidence/closure-map.v1.json)，静态一致性结果见 [source-check-results.json](public-task-contract-repair-v1-evidence/source-check-results.json)。独立证据binding见 [binding.v1.json](public-task-contract-repair-v1-evidence/binding.v1.json)，SHA `79367304967f2e46f20de5f93ee5c70939096fb5e55e5281203dc18e912bd5bc`。
