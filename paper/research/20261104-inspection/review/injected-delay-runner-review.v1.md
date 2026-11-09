# 开发 runner 注入延迟独立复核

审查者 `/root/control_review`，同模型家族、独立工程上下文。本轮只读新增runner延迟与driver零值交接，不重复另一审查者负责的ledger/whole-episode字段、预约、结算或完整causal admission。不运行MLX forward、PyFlyt、真实600s flight或owner操作；只写本报告和独占evidence。

## 结论及版本

**本轮未发现会阻止该开发延迟机制使用的必要缺陷。** 实际审查runner SHA为 `c3bed0403ca4a8e7f5d79c0db3df5da0aeb27e46e37423e94474447fbd4da552`，新增测试SHA为 `b9048096f5e25211f43b2c00921840a0bba60aed6d60ba882b152abad7ebfb96`。driver SHA `c908ca0818e2a5c66d7cd0b221999c9e1b573c939eec339640c73b4ceeea196a` 与上一独立通过的0922原字节精确相比，**仅新增一行plan字段 `injected_delay_seconds: 0`**，没有暗中更改已复验的grader冻结或classic清理。差异证据见 [driver.diff](injected-delay-runner-v1-evidence/driver.diff)。

这只是runner实现/短测试证据，不授予真实model latency、物理延迟鲁棒性、完整G4、正式main/holdout或人类领域认证。

## 已核语义

以下行号均指本报告归档的[c3bed runner](injected-delay-runner-v1-evidence/run_inspection_llm_development.py)：

- **CLI 59、64–67行**：延迟只能为0/2/5/10/20整数；非零仅允许B4-M1且ablation为空。零值保留既有五种LLM条件及B4单因素ablation规则，不允许B4-M3。
- **helper 40–48行**：只在真实`kind=response`且status为completed/policy_timeout、非final、采样物理进度小于600时额外等待；not_started/runtime_failure/final/进度600均不等。helper不改response，不将额外等待写进原service/parent耗时。零值没有sleep，明确记录额外等待0。
- **消费137–169行**：journal原样保存response，另记worker_at_model_reply与释放后worker_after_response、requested/actual wait。传给actor的是新字典；raw service wall和parent end-to-end仍保留在原response中。soft60按原raw/parent耗时及deadline_exceeded判断，注入时间单独累计到dispatch_age_wall_seconds；这遵循所声明的“真实模型软截止+额外墙钟延迟”语义，不是把总dispatch年龄限制为60s。
- **158、189、205–207行**：即使actor只见较旧frame，只要当前共享进度已经600，消费就不会向commands入队；跨600的注入同样不派发。final consume不额外等待也不派发。新request还受当前progress小于590约束，故不会在600后因积压public update另开请求。
- **worker边界105–130、170–188行**：物理worker是独立spawn进程，parent的sleep不停止它；等待期间parent不读update队列，释放后按队列因果消费。这里审的是控制流，未据此声称任何实际1x物理/队列容量或延迟成绩已通过。

## 独立短验证

读取了作者日志 `/tmp/inspection-injected-delay-tests.v2.log`，确为43 passed/2.53s；本结论另有独立执行，不只引用该自检数字。

1. **120组CLI边界**：调用真实main/argparse，组合两架构×三模型×四ablation状态×五delay。通过guard的组合在首次registry读取前由fixture主动截断，其余均SystemExit2；与上述允许集合完全一致。没有输出目录、worker、模型client、owner预约或实际资源动作。
2. **13个原消费函数fixture**：从c3bed源码抽取未经修改的`consume_response` AST，放入最小lexical fixture，使用真实PublicLLMActor、fake clock/progress/queue。覆盖0/2/5/10/20、completed恰好60s、parent耗时60+ε、policy_timeout/not_started/runtime_failure、物理600但actor frame仍旧、599→600、final。所有断言通过：raw response及journal response逐字段不变；恰好60s可解码入队，60+ε不能；注入不伪增raw服务耗时；额外wait与dispatch年龄分开；达到600或final均无入队；runtime_failure按原运营异常路径抛出。
3. **新增测试独立复跑5 passed/2.11s**：其中真实独立timer子进程在parent实际等待2s期间继续推进超过1.5s，response保持不变；其余四项确认不应等待时没有调用sleep。该timer不代表PyFlyt或模型服务。

复现脚本及全部断言结果见 [check_runner.py](injected-delay-runner-v1-evidence/check_runner.py)、[independent-results.json](injected-delay-runner-v1-evidence/independent-results.json)、[independent-pytest.log](injected-delay-runner-v1-evidence/independent-pytest.log)。全部源与证据SHA见 [binding.v1.json](injected-delay-runner-v1-evidence/binding.v1.json)，binding SHA `63a6f27c9fa1586c3a6cfb3dd0dc9480ae9da935b220eeca885ea2474a52556f`。

## 未证范围

未运行新的八条件pilot、延迟180-slot矩阵、MLX或物理；未实测额外wait下的1x lag、worker队列长度、实际传感/任务表现及统计恢复阈值。新增test文件本身主要测helper；本报告的AST消费fixture补充60/600/零值与raw保全边界，但不冒充完整进程/协议重放。ledger、整集plan字段、重试和结算仍须另一独立审查闭合；原live ledger与历史Y、失败attempt、原报告均未改写。后续执行仍须为当前源码/字段重新冻结合法envelope及正常原生资源预约。
