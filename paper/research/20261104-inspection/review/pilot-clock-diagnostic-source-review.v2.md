# 固定模型负载 clock 诊断：v2 有界修复复验

结论：原 v1 P2“submit已发生而poll异常导致请求记录/分母消失”已关闭；本次差异范围未发现其他阻断项。仅支持按冻结 intent 执行**一次无分数工程诊断**，实际启动仍须通过现有原生充分预约、development purpose、模型/registry/profile及service独占operation前置验证。这不是新的预算批准、原B4第三次重试、平台修复有效证明或G4/main许可。两个原clock invalid和耗尽重试分母不变。

审查者 `/root/control_review`，同模型家族独立工程上下文。R=`paper/research/20261104-inspection`。原v1报告及反例不修改。v2输入 `qualification/clock-localization-fixed-load.v1/freeze.v2/binding.json` SHA256 `7acbab10b21b2644543efcaae9e6de7fdf4d0f32ef9021f4a567d8acc12ecdf9`；intent SHA `fb1acb459e604595e8b7583ed06fff676b3e9a97bbf1285c2ec34e984d0b21cd`。独立绑定 `review/pilot-clock-diagnostic-source-review.v2.binding.json` SHA `fc6c5dd12a127f50395836ac3c02fd9465cba4324ba26a939e7f59c9d76847a3`。

逐字节核对v2全部7项冻结输入；working三源/测试与freeze完全相同后复跑所附单元，13 passed/0.02s。与v1相比，`inspection_clock_profile.py`（SHA `aa4ff3be49bd744e19ffa7e58b03c220b34a137c1fbfc13c4374f298c0b7dd7e`）、`inspection_diagnostic_worker.py`（`304fd96297f19a958f189bac03f7628894645e9b8107de5fb309039884884cb1`）及固定20请求tape（`1ce4b730a025317d03eb139b3a7e092e6b4b89786128c8d351a77af84499f6c4`）未变。intent只更新时间、修复源/测试SHA与版本说明；未变更物理、B0、时钟、0.1门槛、模型profile或重试处置。

修复源 `freeze.v2/source/eval/probe_inspection_clock_load.py` SHA `2d41a68e11bb2bcb191bed4be808b5d31415c579c3f741d4140de5d36e1a1a4b`：`:76–84` 在任何submit I/O之前append带ordinal、target、seed、message SHA及UTC/mono submit锚的intent；`:85–94`分别记录submit返回等待和明确 `dispatch_or_response_unknown`、失败时刻/traceback；`:95–97`才补实际response/结束sim。异常继续进入errors，不伪造response或completed。

`:221–226`的计数区分 `dispatch_attempted`、`client_submitted_requests`、`responses_received`、completed；明确client计数只是pipe dispatch attempts，不能称服务admission。真实账本用量仍以durable service intent/call/token/completion和既有 `validate_model_operation` 为准。缺服务完成证明/cleanup时保留unknown，不用客户端0推断实际用量0，也不由本脚本自行settle owner。

独立复用原反例并扩大到submit异常：直接加载冻结v2模块，FakeClient成功submit后poll抛错，以及submit尚未返回即抛错，两种均保留1条intent、seed7、message SHA、target和submit/failed锚，状态unknown、response缺失、errors1。前者client计数1、后者0，但两者dispatch_attempted均1。2项独立单元通过/0.01s；脚本与日志SHA在binding（`/tmp/inspection-clock-diagnostic-source-review-v1/test_retained_submit_v2.py`）。这正面检验原缺陷，未由其他门掩盖。

沿用v1源码审查的窄边界：原epoch/pacer/完整120Hz分母保留，预分配阶段/CPU/GC记录有有限扰动；模型输出没有到B0 commands的通路；service在模型import/load前验证native reservation；异常/清理和资源未知不得结算为0。没有通过单元测试证明实际开销、OS/IPC/GC根因、现场清理一定成功或原故障可复现；GC overflow时不能说完整事件链，process CPU含feeder而thread CPU仅实际调用线程。worker阶段仍把mission.step内部合并，若峰值落在step需下一次证据支持的进一步定位，不能先猜测修复。

本次只运行13+2个纯CPU/Fake单元；没有模型、physics、重放、正式种子或owner写入。尚未独立核具体新reservation receipt，不替代实际原生前置门与事后真实结算。诊断应使用本冻结源/固定tape、独占新输出和既定一次operation；与原B4不同的B0负载下不复现只能作inconclusive，不转成配置择优或qualification。报告/绑定写入后本轮停止写R，不继续扩大审查。
