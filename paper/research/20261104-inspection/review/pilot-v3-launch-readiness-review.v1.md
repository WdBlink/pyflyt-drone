# 独立新开发 pilot v3：launch readiness

结论：在本次冻结来源和原六个 dev root 范围内，**未发现启动独立新 v3 全 108 槽开发 pilot 的明确阻断**。启动仍须使用新 SDK out、专属新 flight receipt store 和未用 reservation prefix，四个 thread 环境变量均为 1，并保持下面的来源、输入及 schedule/sampler 参数。它是修复后的新完整开发矩阵，不是 v2 的第三次重试、补槽、正式实验或 G4 GO。本审查不执行 launch，也不预先保证 108 槽的时钟/模型/资源结果。

审查者 `/root/control_review`，同模型家族独立工程上下文。只读源码、已有 preparation/失败终态及已审 clock 证据；独立 stdlib 脚本不导入执行源，不调用 RNG、模型、物理或 native owner 写操作。未重跑软件测试；root 的新 staged/legacy 软件检查不冒称本审查实测。

绑定与完整分母：

| 项目 | 独立核验 |
| --- | --- |
| preparation binding | `aeb15ca38f736b1aa4fc31c05d6e71a61c8b690b89824c8ef25af8401f2e7413` |
| 7 成员 preparation archive | `0587ebda0ae141173254b6dd0486304c58acfd9370875cc7865b0e89d614ec0b`；唯一成员集合、成员 SHA、现场文件逐字节相同 |
| 当前 source / flight dependencies / inputs | 103 / 61 / 10 全部 SHA 匹配；103 份 embedded source 文本同 SHA 且等于当前字节 |
| 实际 production driver | `c14c52b0e9a7aa00336b6e69b3274d5ba0284077cf9b04c1f85fd34c2dc87b86` |
| schedule | seed `2026110621`；存储 SHA `27f3c27b9aace5c706ec701c749b737ffebaf06254f3abb4a170eaad23048ce6` |
| sampler | master `2026110622`；90 个 LLM order index 恰好各有一个 uint32 seed；本轮没有重新生成它们 |
| 矩阵 | 6 原 dev root，每 root 一个 18 槽块；B0/B1/B2 各 6，五 LLM 条件各 18；18+90=108，root×condition=48 个均值 |
| 原 selector | SHA `75f57580f0f5371d91a741fd4f9573e52192d019e9730335123a7ae67897a32e`；18 旧经典槽、6 root、固定 B1，未重选 |

从实际 driver `:95–125` 重建 exact 来源集合，包括 13 份既有 recording SDK lib、37 份 native owner source、1 份稳定 compatibility entry、绑定 transport，以及 accepted evaluator entry/固定 manifest/全部 17 成员；同现场 103 key 集合完全相同。61 依赖集合独立枚举为全部 60 个指定 PyFlyt 源/资产及实际 PyBullet 二进制，字节均匹配。accepted design 与 registry 对 native accepted plan SHA 均匹配；evaluator 固定 manifest 锚 `ecc2069f…`、exact file set、成员 SHA 与无 symlink 条件也成立。这里只核声明和实际消费的该闭包，不扩称完整 Python/MLX/OS 二进制环境证明。

`run_inspection_eight_condition_pilot.py:59–65` 只接受原六个 dev ID、独立 lineage 和 dev manifest，读取已有 case 文件；未调用 population writer 或 formal namespace。实际 production `run_inspection_formal.py` 尚不存在。prepare-only 的 `:136–142` 在 reserve/measure/record_matrix 之前返回；本次 7 份 preparation 是 source/order intent，其 0 calls/0 flights 字段不被当作全主机资源监控证据。本审查逐槽集合、block、repeat、seed key 和 source 算法，但按禁止 RNG 的边界没有重新抽取 schedule 或 sampler 序列。

实际调用链与失败边界：driver `:153–169` 的 checkpoint 等待位于 reserve/spawn 前，移除 request 后再核冻结字节；`:171–207` 的 classic 路径用 SubprocessTree/finally，nonclosed 或未知即阻断。LLM `:213–230` 把 source/deps/case/profile、零 injected delay、`.1` lag 与 `max_attempts=2` 交给 `execute_development_llm_episode`。后者 `:70–75,89` 限 dev/devext、两次和 100ms；`:145–168` 先核 source/input，再以唯一 native CLI budget-reserve 预约单 attempt 后 spawn。service `admitted_limits:27–59` 在 MLX load 前及后续请求重查 accepted plan、active round、live reserved、registry/model receipt 和 purpose.phase=development；不借 staged formal purpose 开放 main。

coordinator `:184–212` 从 durable operation 重计实际 usage，写完整 archive/receipt，再走 native settle；缺计量用 unknown 与空 usage 表示未知，未把它当 measured zero。`:217–240` 要完成 model/causal/source/evaluator 检查且 cleanup closed 才能接纳，未知、源错、输入错、grader/因果错不能自动 retry。`inspection_llm_episode_receipts.py:39–57` 另核嵌入 source 文本、实际 parser、reservation/seed/profile；`:70–179` 把原始请求、durable call、response、decode/enqueue 和预算相接。`validate_flight:100–107` 要求完整 600s/144000 步/240Hz/120Hz/1×及 lag≤.1，不改变 policy failure 的原 Y。

原 v2 故障槽两份 terminal 仍为 known_operational_failure/clock_lag_exceeded，max 分别 1.768018833322742s、0.3529511250322912s；无 attempt-02。两份 terminal 的 retry_allowed 是局部失败类别，不能覆盖 coordinator `range(2)` 的总上限。旧 plan 与 v3 实际 LLM 来源中的 cleanup SHA 已不同，原 slot plan equality 会拒绝混用；新完整 v3 须独立命名和保留全部 108 分母，不能覆盖/续旧 receipt。SDK `inspection_recording.py:99–123` 先校 exact108 并打开全部三类 envelope，`:128–168` 只有全部内层实际完成才生成 48 均值，异常保留计划分母而不写 completion。

与实际 clock 证据的关系：当前全部 103 来源均能在已审 postinventory freeze 中找到同 SHA。既有独立报告 SHA `a7c69c051f95887b915c5a684601b3b1b9a3f89916b950a4b6c03f79754b4c1e` 已重计单次 L2 600s、72000 control 行，max .055215666621620585s、>100ms 为 0、23 calls/163524 tokens 与清理闭合；本次只复用该已审结论和匹配来源，不重算/运行它。该工程单次结果支持进行新矩阵，不能保证其他模式、根和模型通过，也不证明旧 lag 的唯一原因。cleanup 的 hot native/provider 是该次已审安装字节；baseline/shutdown 全表和未知孤儿阻断仍在，非 host-loss/subreaper 保证。

阶段顺序：accepted `design.ral.v3.json:546–558` 明列“开发边界与实际全八 SDK/clock/accounting → 完整六 root pilot → 最终 G4 source/model/prompt/generator/split/retry freeze 与独立审查 → main”。`:658–688` 保持两次基础设施 cap、policy 原 Y、sole-owner 实际计量与 G4/未知阻断 formal。原 WorkOrder W-development-G4 也把完整 pilot 列为依赖。故未发现必须先安装全 formal graph 才能运行本开发 pilot 的合同条件；root 先前的自定工程顺序不是额外硬门。同期只在 staged 完善 formal IO、不改 production，可以保留清楚的证据边界。后续安装任何 delta 时，必须对最终消费者/来源闭包与本 pilot 的可复用部分做独立绑定和必要集成验证；不能声称本 pilot 用过尚未安装源码，也不能仅凭本 pilot 完成授 G4 或 main/holdout 访问。

本次 owner 只读快照为 active round-36、frozen plan `f17d4111…`、implementation `d809362c…`、20 条 usage 状态均 known，无 held/unknown；这不是持久不变或实际 native lock/load 的再认证。最近已应用 finding 仍明确 G4 needs_work。启动前由既有 native 资源门重查实际状态；不手改状态或放宽预算。无本轮新阻断，尚未执行的完整 v3、最终 G4/integration/source freeze、正式 main/holdout 及领域人工审核继续 pending。

独立脚本、复算、精确 SHA 和审查绑定位于 `review/evidence/pilot-v3-launch-v1/binding.json`。本报告与证据归档后停止写 R。
