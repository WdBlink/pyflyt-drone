# Formal event endpoints：有限修复复核 v2

结论：**request changes；F1/F2及F3创建见证部分已关闭，但F3同根因仍残留于program链。**正常commit/local/handoff未在本轮正控中被收紧误拒；不能将作者全套通过或25项修复正负控通过写成v4无阻断。原v1报告与反例保留，不覆盖。审查者 `/root/statistics_design_review`，独立上下文、同模型家族，无实现职责，不作human/G4/main批准。

Subject `planning/formal-admission-candidate.v3/secondary-analysis-freeze.v4/binding.json` SHA256 `80bef7386ac2aa1ee64307f4c6eae237d4ddcd5712df947b597602d34e54e1c2`；event源 SHA `ea5e207c720060c195ab603cc7dec9049b61e2058eb0b2a43366e4c377e0a949`；formal event测试 SHA `4a05d4b3374b20a682050f035d086761c4cae9f75fe749fc8a9ea95fb130fd99`。八个明确成员实际字节与binding相同，协议v4仍为 `f4be91e78f692698fda07373e7e6d0a693d2dd17f2b0f80c3d57d1f91830aa72`。consumer/REQ/purestats与v1相同；本轮只复核event helper delta，不重审其他模块/fixture导入方案。

## 有界关闭项

F1：原duplicate same-serial commit从identified改为adoption_not_identifiable_from_trace，endpoint=None；新逻辑在完整audit按serial计数。新增正常“同job、不同唯一serial”的刷新链仍identified且保留第一次time11，未将所有多commit都误作重复。

F2：原12.25的off-2Hz local transition/service/frame从identified改为trace unknown；合法12.0/12.5本地采集仍identified，无需commit或上传。该exact半秒判断与实际依赖可相容：定向读取PyFlyt `Aviary.step`，elapsed_time由整数physics_steps/physics_hz赋值；在240Hz且sensor条件physics_steps%120==0时，1200个原半秒时间均精确落在二进制可表示的n/2，独立字面公式检查没有浮点误拒。没有运行Aviary或物理；这不推广到改过计时源的其他版本。

F3-creation：原args.drone_id=False→job.drone_id=0误匹配已转trace unknown。expected_version=False、response.version=True、card float、job drone/version bool、复制参数/创建时刻类型被改也转unknown。正常native-int版本、合法复制int duration、float duration、未上传local及合法handoff正控仍identified。handoff源drone bool/float/越界/同目的、源version float/负值、目的drone/version bool均拒作采用见证。此为有限schema/type正负控，不是所有handoff源状态/工具执行重放的认证。

## F3-program 残余（P2，未关闭）

根因仍是 `_adoption_response` program链的Python等值/算术：request/commit drone比较、request/response/commit的serial匹配、revision加1及自机token version比较没有先要求native int。在保持正常fixture其他字段不变时，下列8个单字段变体**全部仍identified/time11**：

| 单字段修改 | 结果 |
|---|---|
| request.drone=False | identified |
| commit.drone=False | identified |
| request.serial=True | identified |
| response.serial=True | identified |
| commit.serial=True | identified（witness.serial也为true） |
| request.revision=False | identified |
| commit.revision=True | identified |
| request.tokens[own_drone][0]=True | identified |

这些值依Python的False==0、True==1及bool算术通过了数值比较，却不是冻结执行器产生的native整数身份。它们是明确非法trace的软件反例，不是实际飞行事故或已完成receipt准入的数据；本轮未重审上游收集/receipt是否另有这组拒绝。但当前selector自身声称的严格program链绑定尚不能成立，创建见证变strict未消除该根因。

最小修复范围：仅对参与该链关联的request/response/commit serial、request/commit revision和drone、自机token version先做native-int与对应范围校验，再比较/算术；保留job/hash/有序与唯一性约束。不能将bool/float强转成int后接受。保留正常唯一serial刷新、正常版本递增与自机token正控。无需借此扩成任意hostile全字段或全graph证明。

## 实际检查与边界

重用v1封存的exact selector AST/literal fixture，在v4副本重放原3项并作上述typed/handoff/正常刷新delta正负控；25项修复断言通过。随后按父Agent明确追加的同根因小范围核验，8个program身份变体实测误收，完整输入构造与输出保存在binding。原missing-request/hash/future-seed/ambiguous-frame、同advance terminal及错误region负控行为保留；没有重复原wrapper/summary/恢复全套或bootstrap/SDK/ledger/闭包。

最后probe实际UTC `2026-11-06T19:17:03.717692Z`–`19:17:03.721284Z`，内部0.00345秒；原修复复放于19:16:22.113304Z结束。四个线程变量均为1，无pytest、全候选、生产包import、actualformal抽样、模型、物理、native/effects或遗留子进程。父Agent另提供v4 whole499pass/124.28s/exit0，本轮未复跑；该结果并未覆盖并关闭上述8反例。所有counterexamples均为literal，不混称真实策略成绩。

F4/F5/G4及response/resource诊断仍未完成；本报告不授实现整体/正式数据/阶段准入。全部软件probe已结束，报告/binding归档后停止CPU检查与R写入。旧版报告和证据原样保留。
