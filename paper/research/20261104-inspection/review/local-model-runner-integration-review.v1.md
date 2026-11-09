# Local model runner integration：独立有界审查 v1

结论：当前三个文件仍是 development prototype，尚不宜启动依赖其终态回执的真实模型实验。发现两个终态阻断，以及 stdin 故障下的有界关闭、worker 子进程清理和方法开关覆盖缺口。现有 43 项测试通过，不能覆盖这些 runner 主循环/终态反例。真实 generation、真实 GPU forward、PyFlyt 物理和新科学样本均为 **0**；本报告不放行 G4。

本审查由同模型家族的独立上下文执行，属于工程审查，不是人类领域审查或跨模型验证。生产源码、owner/live、冻结经典组及其 runner、真实回执均未修改。先在 `/tmp/inspection-llm-integration-review-v1/` 运行并保存证据，收到 round18 checkpoint 已保存的通知后才复制到本报告 evidence 目录。旧审查均保持原样。

## 版本与证据绑定

工作树：`/Users/wdblink/.codex/worktrees/538e/pyflyt-drone`。Git HEAD：`270c6f0946b3be59527b2b77982f75a70da7ef77`；这些新增 inspection 文件未受该提交单独覆盖，以下实际内容 SHA 才是审查版本。

| 文件 | 实际 SHA-256 |
|---|---|
| `eval/inspection_llm_actor.py` | `d33db4c6519f93a4971cb8e5e4fd1b07ef8adfbda14e01e3dfdb2274b8421b79` |
| `eval/inspection_model_client.py` | `cb04f02d47dd5f3478aad3909357fae0a845396aa7f0eee168b376e691995e71` |
| `eval/run_inspection_llm_development.py` | `5a8c420e601a3576fdef91fd5c46634f410b40bfa3d9dca4de8cf4a69c6a3ee9` |
| `eval/inspection_llm_protocol.py` | `c323f708fd2c2eb54a440758bd162785f3ee2fcebd95b9bca21eaf7ed6ef44d4` |
| `eval/inspection_model_service.py` | `cfbb8e908677d439c56c31ebcc3ca196a72ccecd3d7d6829b07d6aead6c11810` |
| `eval/run_inspection_reference.py` | `1afeab04ba55b8311fe0c2f093c7174627fb9226918886f4066161ff62e04068` |
| `envs/inspection/execution.py` | `dc301939ea1d55f08f01c69af5561c75f762ac45bdd0d144d677514387425ea4` |

wire/service 与此前 repair 审查的已冻结版本相同，可复用其纯 admission、完整 blob 检查及 FakeMLX 边界结论；不能因此推导本次新 runner 集成已经通过。相关测试、cleanup helper 的完整 SHA 和源码副本见 `review/evidence/local-model-runner-integration-v1/binding.json` 与 `source/`。

`qualification/model-context-tokenization.v1.json` 实际 SHA：`624fa525fee9b508b7a55a7aa635418a382476699161b9df73f0c48e825bbece`。其角色明确为 synthetic public context tokenizer-only，generation_calls=0；5 个 actor/ablation 请求 × 3 模型共 15 个计数，4830–6480 tokens，均小于 8192。本次只核对记录和消息绑定，未重新加载 tokenizer。它既不是 full actual context 上界，也不是 forward/时延/业务资格。M3 的 B4 tokenizer 计数也不代表 runner 允许 B4-M3（runner49 明确拒绝）。

## 必须修复及明确的覆盖缺口

| ID | 等级 / 影响 | 定位及实际证据 |
|---|---|---|
| F1 | P1：校验失败仍宣告 completed | runner196–204 只保存 unchanged 布尔；没有把 False 合入 error/status。world 仅存在时记 SHA，没有要求存在或读取验证。全 fake 主循环中，删除 private 副本后 `private_case_unchanged=False`、改变冻结源码后 `sources_unchanged=False`、缺 world，三种情况均返回 0/status=completed。依赖失败也在同一未判错分支，未额外运行该变体。 |
| F2 | P1：损坏 completion 使终态永久残留 running | runner171–173 的 completion 解析错误被 except 捕获，但 finally192–195 再次无保护解析同文件；错误覆盖原错误，196–204 不再执行。写入 `{` 的 fake completion 实际产生 JSONDecodeError，main 无返回，磁盘 run.json 仍为 running。source/dependency 的 finally 读操作也未隔离 OSError，属同类终态风险，但本次未删除生产文件复现。 |
| F3 | P2：同步 stdin 写没有外层截止，故障时无法回到清理 | client64–73 的 submit 在调用线程同步 write/flush；runner164 直接调用，无独立 watchdog。fake child 发送 ready 后不读 stdin，70,000 字符 user 消息（低于 service 每消息 100,000 字符限制）使 submit 在 150ms 观察点仍阻塞。只有探针外部 stop_unknown 杀 child 后才以 BrokenPipeError 解锁；submitted=1/inflight 保留。该反例是管道阻塞实证，不是 600s 实飞。独立物理进程仍能推进，但主协调器不能保证自己按时走到终态/清理。 |
| F4 | P2：异常时只杀 physical worker 父进程 | runner177–182 对 mp worker terminate/kill，没有清理其 planner 子进程树；executor99 创建 daemon planner。隔离 OS 进程组的纯 CPU 模拟实际显示父 SIGTERM 后 daemon 孙进程仍存活，随后探针明确杀掉隔离组。这里验证的是 Python/OS 机制及源码组合，未运行真实 planner/physics。模型 client 的 stop_unknown 已使用 process-group cleanup，不代表 physical worker 也有此保证。 |
| F5 | P2：三种消融尚无 runner 执行入口 | runner39–47 没有 `--ablation`，actor90 也没有传该参数，manifest 没有记录 ablation。直接构造 actor 的开关并不等于可以用当前 runner 执行或复现消融。应在真实消融数据前接入、校验并绑定具体开关。 |
| F6 | P2：B4 实际监控范围窄于提示中的 job/report outcomes | actor83–87 的 signature 只看 card version、retired/link/camera 和 failed/cancelled jobs；result94 只对 rejection 加 trigger。已知 candidate 的 inspect succeeded、accepted report 不会触发即时监控。纯输入反例：t=0 完成一次请求，t=10 仅新增 succeeded job，B4 triggers=[]/due=False，须等周期20s。没有把“所有成功/报告结果均触发事件重规划”实现出来；其实际监控能力应按这个范围声明或补齐。 |

F1/F2 必须在可信终态/实际用量实验前关闭。F3/F4 应在真实模型故障可能发生前补上有界关闭，并把无法核清的用量保留为 unknown。F5 是消融执行缺失，不能以直接 actor/tokenizer fixture 替代；F6 是方法定义覆盖缺口，不是从低分推断出的性能问题。

F1 反例没有伪造真实 owner admission：探针把 client、mp context/process、依赖查询替换为显式 fake，并在 `/tmp` 构造源码/输入/operation；真实 worker、service、权重、owner 状态完全没有运行。fake done 只是驱动 runner 的终态分支，不代表有合法 world 或科学成绩。恰恰需要 runner 拒绝这个缺失或变化的落盘证据。

## 已实际核实的正确边界

| 检查 | 当前可支持结论与边界 |
|---|---|
| 公共因果入口 | actor71–89 先调用 exact8-field serializer，再复制 state 和已收到 observations；没有读 case、seed、truth、future events、grader。worker 的 public_state 是主信息流边界，递归私有 key 拒绝只是额外防御。result 是可信 worker 的工具返回通路，不是任意隐藏输入过滤器。 |
| 共享模型/工具配置 | B3/B4 从同一个选定 registry generation_profile 读取 max calls/input/output/tokens/seconds，使用相同 service 与七种 worker RPC。每次至多8命令、actor至多32请求；runner另有2048入队上限。实际最大32×8=256，2048不会增加模型工具权限。工具参数、版本和证据仍由同一 worker 决定；模型解码不是 ACK。未声称经典组有相同模型调用预算。 |
| 单 in-flight | actor.due/start 和 client.submit 都禁止未解请求重入。真实 fake 子进程测试通过；请求发送前即记 submitted，写中断不会反推为零使用。 |
| 等待与物理推进结构 | worker 是独立 spawn process，模型回复由 reader thread + 非阻塞 poll 获取；主循环等 updates 仅20ms。模型 load/ready 在物理 t=0 前，实际 forward 请求在 worker 启动后。fake 主循环记录请求 progress0→回复5，验证数据通路；这不是新的真实物理/真实模型等待实证。F3 的协调器阻塞与 worker 独立推进可以同时成立。 |
| late / t≥600 | actor169–185 丢弃非 completed、wall 非有限/超限、deadline_exceeded 或 state≥600；runner125–129 取 service wall 和父端 end-to-end 的最大值，并另看实际 shared physical time<600；final 响应不发命令。fake 主循环正常 reply 入队1条；end-to-end 2s 超过 fixture 1s cap 时0条；physical progress600 时0条。actor独立599/61s与600反例也均0条。不是每个 enqueue 瞬间的原子墙钟证明，实际执行截止仍由 worker 物理循环约束。 |
| format / budget exhaustion | malformed model JSON 是 actor policy_format_failure，无命令；service not_started 在 actor 被忽略，无命令，物理时钟仍继续。not_started 不能冒充一次已生成调用；完成 usage 以 service operation 实际表计为准。当前真实 full 上下文未验证，frame17,000字符裁剪不等于整个消息包/模型 tokens 上界。 |
| known runtime / unknown | runner 对明确 runtime_failure 抛已知运行故障，对 client 协议/EOF/transport 异常保留异常；fake 两类均 operational_failure/exit1/0命令，没有变成Y0，也没有自动重跑。manifest目前依赖异常文本区分两类，尚无完整 SDK/结构化失效回执协议。不能把缺 completion 或未知 partial use 计成0。 |
| 预约与用量 | 沿已审 service：同 owner frozen plan/active round/reservation、registry/receipt 完整绑定，exclusive operation mkdir，service 内 calls/tokens 累计及每 token journal；runner不新建 owner预算、不同步结算，也没有自动重试。exclusive目录阻止同 reservation 重启；换reservation/新输出不意味着可免费重跑。真实使用必须连同部分日志和 unknown 交给 owner 核清后结算，不能用 actor attempts 或 fake usage 替代。 |

timeout 的当前真实语义是软结果截止：client.poll 没有 request-wall watchdog；未返回的请求可占住整个 flight，结束后 finish 默认再等120s，才在超时路径停止模型组。没有证明可在60s硬取消 GPU。迟到结果不下命令这一边界已核实，不能改写成已完成硬实时模型取消资格。

## B4 与消融到底改变什么

直接 actor fixture 的 `event_due_at10`：B3=False、B4=True、B4-periodic=False，证实 periodic 开关确实把10s事件触发改为20s固定周期。B4基线携带短 plan/note、monitor_triggers 和只由已送达静态 estimate_xy 形成的15m alias hints；没有隐藏 ID。alias 是启发式提示，不保证真实物体唯一性，也不自动取消/重写模型发出的重复报告。

`no-cross-aircraft-memory` 确实删除 plan/note 和显式 alias_groups，但 actor76/124 的跨帧观测缓存照常保留。反例把 drone0 的 ev-a、drone1 的 ev-b 在t=0送达，t=20新state不含 observations，消融后的 prompt 仍含 ev-a/ev-b。因而它不能声称“移除所有跨机记忆/关联能力”；应按实际移除的派生别名/持久计划来命名，或实现预声明机制。保留相同公共原始信息本身不构成真值泄漏。

`no-progressive-confirmation-guidance` 仅删 SYSTEM 中一段 two-high-quality-distinct-view 指导；它没有独立的确定性递进确认模块可关闭，也不改变 grader/任务约束。可称提示指导消融，不能称已证明因果移除了所有递进确认行为。上述差异均不构成真实性能强弱证据，且三者目前都缺 F5 的 CLI 接线。

## 实际执行与交付边界

执行命令（均由仓库 `.venv/bin/python -B`，禁止字节码写入）：

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-integration-review-v1/probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-integration-review-v1/pipe_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B /tmp/inspection-llm-integration-review-v1/orphan_probe.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_llm_actor.py tests/test_inspection_model_client.py tests/test_inspection_llm_protocol.py tests/test_inspection_model_admission.py -q -p no:cacheprovider --basetemp /tmp/inspection-llm-integration-review-v1/pytest
```

结果：43 passed in0.25s；独立 probe 主循环11个分支、actor监控/因果/截止分支、阻塞 stdin 和进程组清理反例均已保存。原43测试没有调用新 runner main，故其通过与 F1/F2 并不冲突。所有 fake 子进程均已清理。`/tmp` 原证据保留，交付副本在 `review/evidence/local-model-runner-integration-v1/`，包括本次读取的 source snapshots、probe源/JSON/日志和 binding。

本轮未运行真实 service main/admission/load/generation；未触碰当前真实预算/预约或真实 `.paper`。先前 frozen_plan=null 的负调用记录不能据此报告成当前状态；本报告不对 checkpoint 后 owner 状态作未经执行的推断。真实 MLX/MLX-LM/tokenizer 等软件依赖的冻结、端到端硬件时延、完整 prompt 演化、LLM episode 回执/SDK接入和业务质量仍待后续证据，当前 manifest 的 dependency 指纹主要覆盖 PyFlyt/pybullet。不存在本轮新增 G4 或 formal GO。
