# Response/resource reader：独立有限计时/计量语义审查 v1

结论：**request changes，存在一项P1多响应处理阻断。**现有字段的主要语义与实际service/client/coordinator定义相符，但reader覆盖了原快照变量，常见多generated response会崩溃；单响应正控及作者48focused通过不能批准本冻结实现。审查者 `/root/statistics_design_review`，独立上下文、同模型家族，无实现职责；不授完整formal、G4/main、全graph或human认证。

Subject `planning/formal-admission-candidate.v3/response-resource-freeze.v1/binding.json` SHA256 `2cbc108e4e7550aa0abbb4b904f07b00d98256c561f3434497e5c21aa89b31a3`；reader SHA `0ae4487c8dcf9904afbb83d6860ae9c961ea97a745d18753e7f26eeec0114ee2`；契约v1 SHA `65f11b1d8119bb0c832338d05ee08b813c5a1be9b6d0b0e3c5607f26d42c5e47`。七个冻结成员实际hash/大小与binding一致；只审reader/契约与必要来源定义，primary/secondary接线不在本结论内。源/测试/实际与staged依赖精确hash见binding，不使用working后续修补字节。

## R1（P1）：before快照被进度数值覆盖

`episode_response_diagnostics` 第41行将第三项快照命名before，`_bytes`第19行必须读取snapshot['files']。但第141行又执行before=_seconds(received['worker_before_request'][0])，把原dict改成数值。下一条generated join在第129行重新读取原service response时，将这个float作为snapshot传给_bytes，抛TypeError: 'float' object is not subscriptable。

两个独立/tmp IO反例均实测复现同一129→19调用栈：两个generated responses（known usage calls2/tokens205）；以及not_started后接一个generated（calls1/tokens103）。两链均包含完整本reader所需的服务文件、actor形状、fresh-join形状和正确SHA；它们是明确软件fixture，不是实际fresh receipt admission或正式数据。故不能归因于错误token总数或孤立字段篡改。变量覆盖不依赖fixture省略的上游字段；实际多response处理同样会走该路径。

最小修复是让原snapshot变量保持dict不变，对request/reply/after进度使用不同名字；不得通过丢第二条响应、跳过原service bytes复核或吞异常绕过。回归至少保留上述两链，核全部response输出、calls/tokens/ordinal及原文件SHA，另保留context洞与retry正控。本轮不验证父Agent已开始的working修补或未来v2。

## 计时来源与诚实边界

定向读取实际production及staged service/client/coordinator/actor/validator。四个关键计时代码块（service started→result、client submit/poll、inject_reply_wait、consume_response）两者字节相同；receipt的causal_request_joins字段也相同。这里只比语义相关块，不认证完整来源闭包。

- **service_generation_and_journaling_wall_seconds**：timer在chat-template、tokenizer.encode、request文件写入和随机seed设置之后，围绕stream_generate与token journal open/write/flush/fsync和关闭、异常处理；可包含stream内部prefill/生成，不是纯GPU kernel。wall在response文件写入/打印之前取值，不含此前prompt编码、冷加载或之后response持久化。
- **client_end_to_end_before_injection_wall_seconds**：submit在JSON/pipe发送前取monotonic，到poll真正消费响应取差，包含传输、服务、轮询等待；是注入前区间。它不同于service区间，不能相加当“两个独立阶段成本”。
- **injected_wait_measured_wall_seconds**：协调器收到reply后另计真实sleep区间；requested arm只是设置值。not_started、runtime_failure、final或reply时flight已到600不再注入。client与service原时长不被注入值改写，soft deadline判定使用原service/client耗时，不把20秒注入误当模型60秒超时。
- worker_before_request是提交前捕获的物理progress；worker_before_injected_wait是consume内、实际wait之前的reply-progress快照；worker_after_response是在wait后但decode/commands.put前捕获。三者都不是直接测得的实际service开始/结束或job采用时刻。
- **recorded_dispatch_age_sim_seconds**严格是这个after快照减source公开sim时间。producer把同一快照复用于worker_progress_at_enqueue，但真实解码和入队可以更晚；不能称exact enqueue-age或job adoption。
- **client_roundtrip_plus_consume_wall_seconds**为client elapsed加consume_started到构造decoded audit字段时的墙钟；含注入/解码/入队等该区间的开销，但漏掉poll取样到consume_started之间未采样gap，也不含随后decoded日志写入/函数返回尾部。它不是源公开信息wall-age，更不是实际物理生效端到端延迟。commands计数明确只表示enqueued_not_accepted，不是接受或采用。

这些改名与契约对旧dispatch_age字段的限制准确；不需要新reader重造原validator/FSM，亦不以1×sim差倒推出未采样wall时间。

## 计量、ordinal与有限正负控

单response实际软件正控分别保留service5、client6、实际wait20.05、recorded sim-age26.05、wall-sum26.1。context decision1被拒后decision2/transport1/admitted call1正确连接；not_started的generation时长为None、实际wait0且不计生成call，client控制响应时长保留。final-after-flight真实call/tokens仍计量、实际wait0、commands0，decode disposition保留，未倒改原质量。

known retry正控分开valid calls1/tokens102与all-known calls2/tokens110；失败attempt仍保留service2秒，不混入valid-flight actor时长。reader的context_rejections/non_generation_responses及相应summary计数来自**末次valid-flight actor**；失败attempt本reader仅保留已知usage/service，原actor原件由上游闭包保留，不把这些计数误称all-attempt actor事件总数。

资源summary按完整原repeat平均usage，known_attempt_count另按全部attempt数累计；3repeat例valid平均102、all-known平均110、known attempts6。缺repeat拒绝，retry不是额外repeat/root。经典无model服务的calls/tokens=0正控成立；money/energy为None，不代表飞行/主机/资本成本为0。没有request层推断、CI/p或货币Pareto支配结论。

unknown terminal、改actor bytes、bool admitted call ordinal、错settlement token总数均拒绝。fresh causal joins仍由原receipt validator负责；本轮仅对读取/映射的输出形状与字段对照，不重新认证真实messages/seed/decoder或native结算。

## 实际执行与范围

最后独立probe于 `2026-11-06T20:04:50.357133Z`–`20:04:50.390954Z` 完成，内部0.03367秒；14项断言含“两个反例必须复现TypeError”的缺陷确认，**不是14项实现全部通过**。四线程变量均为1。冻结reader函数AST加明确Artifact/parser适配器，仅写自己的自动清理/tmp；没有生产包import、MLX、模型、物理、native、formal population或真实回执操作。第一次自建classic fixture因/tmp的/var与/private/var别名未resolve而中止；只修自有fixture的canonical path后完成上述运行，没有改subject，也不把该harness错误当生产缺陷。

父Agent提供作者48focused实际通过70.77秒，但仅单response形状遗漏R1；该绿灯及旧499/78不能作为本模块接受。working修复与新增回归尚不在此v1 subject内。未重复原数值/事件F1–F3或完整接线审查；这些不由此报告升级为G4。全部probe已结束，无遗留子进程；归档后停止CPU检查及R写入。
