# G4开发进度v14

2026-11-06：固定八配置开发pilot v2已失败终止（driver exit 1），不是暂停待续。完整108槽分母中10个有效callback、0个完成的独立root；实际13次物理运行全部保留。classic18槽为2completed/16stopped，LLM90槽为8completed/1failed/81stopped，outer48槽全部stopped。order10 B4-M1 repeat2两次600s/144000步运行均clock invalid，max lag分别1.768018833s与0.352951125s，已耗尽对称的一次基础设施重试，不执行第三次。见 `eight-condition-pilot-terminated.v2/binding.json` 及两个完整ZIP。有效策略结果、先前无效运行和未启动槽均不删除或改标。

独立source与timeline审计见 `review/pilot-v2-clock-failure-{source,timeline}-audit.v1.md`。峰值主要为已积累的backlog，现有逐秒进度和最大见证不足以定位唯一根因；旧worker没有保存完整72000 lag序列或准确worker UTC epoch，run.started不是该epoch。不得将模型、RPC、GC、系统调度或同时进行的开发活动断言为已证实原因。

下一步为一次无分数工程诊断：固定20个原M1公开prompt/seed只形成硬件负载，输出只保留、不解析或执行；物理侧由原B0参考策略独立控制。新增worker衍生版本保留原epoch/pacer/GC配置/600s分母和0.1s阈值，预分配120Hz wall/thread/process CPU分段记录、完整lag序列、有限GC事件与UTC/monotonic bracket，循环结束后写盘。首版意图、tape、源和11项单元测试冻结在 `clock-localization-fixed-load.v1/freeze.v1/`。独立审查复现“submit后poll异常漏记已发请求”的P2；新freeze.v2保留发送前意图/时间/seed/unknown状态，13项单元测试通过，独立v2复核确认P2关闭，13项所附单元和2项独立异常反例通过，未发现本轮其他阻断；原v1不覆盖。见 `review/pilot-clock-diagnostic-source-review.v2.md`，仅准入一次无分数工程诊断。该诊断尚未预留或运行；需独立source审查后以唯一native资源操作预留至多20calls/180000tokens。它不是原B4失败槽重试、策略成绩或G4接受；未复现同样峰值也不能证明修复。

当前唯一owner仍为d809…（见ACTIVE_CONTROLLER.md），表示迁移已经实际完成，旧记录精确保持；旧稳定CLI为exec兼容入口。实际模型资源累计17个操作全known、353calls/2240091tokens，无held/unknown/未结清操作；cost、Codex host、资本和能耗未计量，不能写0。旧pilot进程已退出，不再使用order9旧ack充当新gate；退出后未获ack的请求已单独存档并移除。

G1–G3及同级非LLM event/clean参考对保持原声明范围。G4仍needs_work：还需证据支持的时钟修复与资格、完整新前瞻pilot、clean+20s实际工程集成、最终source/model/prompt/generator/split/retry/hardware冻结与独立接受。正式入口候选组件不构成全管线资格；namespace v2的main namespace后继revocation检查仍有独立P2待修。没有抽取actual main/holdout seed/order/instance，没有正式实验或论文结果正文。budget={}、原19项义务和实验分母保持。
