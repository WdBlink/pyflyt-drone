# 固定模型负载 clock 工程诊断：冻结 v1 独立审查

结论：v1 有一项需先修复的 P2，暂不作为可执行诊断版本。已提交请求在 poll 异常时从 supervisor 的请求分母/时间线消失。原 freeze.v1 保留；后续修复另版，不回填本版结论。其余已读源码符合“固定历史 M1 负载 + B0、一次无分数工程定位”的窄用途，不是原 B4 第三次重试、配置资格或 G4/main 许可。

审查者 `/root/control_review`，同模型家族独立工程上下文。R=`paper/research/20261104-inspection`。输入 `qualification/clock-localization-fixed-load.v1/freeze.v1/binding.json` SHA256 `8b6449535a54f3b00cbf4a8e11a7ce6d6e3a2f5fe0c18004158a7b4e080b3b4e`，其7项文件逐字节核对。独立绑定 `review/pilot-clock-diagnostic-source-review.v1.binding.json` SHA256 `24e50f0b207e93f573e96d18db3be460bfa4e351c986d4969aa7bcad1a5955bf`。下述行号均指 freeze.v1/source 内文件，非随后修改的 working copy。

**P2：成功 submit 后的 poll 异常丢掉实际 dispatch 记录。** `eval/probe_inspection_clock_load.py:76–89` 只在本地创建 record；`:79` 已调用 `client.submit`，但直到得到 response 后的`:87`才 append。`:88–89`只保存 traceback。`:211–213`又从已 append 的 records 数 submitted，因此一次实际已发请求可能被汇总成0，并丢失 seed/target/submit UTC+monotonic 锚。该信息正是将负载与物理 clock 窗口关联所需，不能把缺 response 当作未提交。

独立单元反例在 `/tmp/inspection-clock-diagnostic-source-review-v1/test_retained_submit.py`（绑定记录其 SHA）：直接加载冻结 v1 模块，FakeClient.submit 成功计数1，poll 注入异常；最终 errors1、records=[]、汇总 submitted0。1项反例测试通过，证明原缺陷，未调用服务或物理。最小修复：submit 前持久在内存 records 中登记 intent，分开记录 dispatch attempt、client提交计数/返回、response 与异常；未知 service admission 保持未知，不能将 pipe attempt 叫作 admitted/completed。失败仍保留原错误状态、服务 durable journal 和真实 owner 结算路径。

这不是已发现“native 账本被 settle 成0”：该脚本不调用 owner settle；`resource_recount` 只在正常完成时调用实际 `validate_model_operation`（`:193`），其他路径输出缺失 usage/None，错误状态为 operational_failure_or_unknown。失败后需要 owner 对实际服务 intent/calls/completion做真实重算或保持未知，不能依据上述错误的 workload_counts结算。硬件资本/电耗在 intent 和 receipt validator 中明确 unmetered，不能写成0。

已验证的正向范围如下：

- 三源码/测试 working 字节在运行所附单元测试前与冻结一致；`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider tests/test_inspection_clock_profile.py` 独立得到11 passed/0.02s。它们覆盖数值记录、fake wall/CPU区分、GC上限、无效duration、初态publication未完成、模型输出不转命令和已结束时不新提交。没有执行 worker/MLX/PyFlyt，也没有证明真实计时开销或清理在现场一定成功。
- `inspection_diagnostic_worker.py:21–41,43–97` 保留 warmup→新mission→epoch、相同绝对 deadline `epoch+t+1/120`、sleep 在 work 前、最多16 RPC、mission.step→共享进度→lag→publication/log的顺序；未重设中途epoch、改0.1阈值或重试处置。`ClockProfile`在epoch前分配容量；增加的markers/callback和原有lags list仍有有限开销，不能称无扰动重放。诊断 supervisor 用B0，故也不能将“不复现”解释为排除原B4故障。
- `inspection_clock_profile.py:16–40,56–101,108–138` 预分配30个float64字段/120Hz tick和16384个GC event槽，72000完整tick约17.28MB；索引越界抛错，GC满槽记overflow。wall/thread/process CPU分开保存；process含feeder线程，GC记录实际执行线程和最近marker后的stage。mark的三种时钟顺序读取、不是原子采样，阶段差值含仪器开销；不能据极小CPU差值作精确细分归因。GC callback未disable/freeze/改threshold，结束后移除；overflow>0不得宣称完整GC事件链。UTC有monotonic前后括号且不重置epoch。
- `probe_inspection_clock_load.py:64–89,155–179`：workload函数没有commands队列参数；响应只存record，不进入parse_reply/actor；只有 `ReferencePolicy.decide(public_state)`写commands。恶意commands文本的所附Fake单元也确认没有执行通路。固定tape20条messages/seed/target与原actor请求逐条相同，22个source指针SHA均核对，case SHA `5600888c8cc583edadbb1e1d43f8f0dcd928b9b4e44c2dc93296a1788c62b422`；原输入+最大输出bound156815<180000。请求串行，迟到或未提交应作为差异记录，不能称精确复现原适应策略。
- 预算路径已读到实际共享service：supervisor`:108–127`核registry/receipt、native reservation的development purpose和diagnostic tape；`inspection_model_service.py:27–60,112–143`在模型import/load之前核accepted plan digest、active round、stop、reserved状态及calls/tokens足够、冻结registry/receipt/profile，并独占operation目录。M1实际profile为8192/1024、soft60s、temperature.2/top_p.95/thinkingFalse，20 calls/180000；M1 tokenizer无需额外regex flag。没有新paid或main phase旁路。当前freeze只含proposal，未给具体已预留reservation artifact；本报告不替owner确认实际预约或授权，实际启动仍须通过既有service前置门。
- supervisor`:166–185,197–218`有780s等待界、在途模型等待/finish、物理进程join、服务/物理/caffeinate清理和结束后来源检查；worker`:119–135`关闭mission并保存部分clock记录。异常状态保留，不作qualification。`stop_worker_tree`只提供已观测直接子树的尽力清理，不是kernel subreaper/host-loss或任意已脱离孤儿保证；这些共享清理边界未在本轮重测。若cleanup或计量未知，不能写已完成或资源0。实际诊断前应使用已审冻结字节、无并行活动模型/物理、独占新输出，并保留服务journal以便原生owner结算。

本轮未改生产实现、冻结输入、owner或任何旧结果，仅新写本报告/绑定及/tmp单元证据。没有审查科学G4完成、正式种子或生成，不选择配置，不减108/48原失败分母。原P2需用新freeze和异常单元有界复验；其关闭前本版不准入该次工程诊断。
