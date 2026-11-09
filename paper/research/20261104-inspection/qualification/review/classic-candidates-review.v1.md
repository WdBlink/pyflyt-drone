# 新经典候选 B1/B2 独立工程审查 v1

审查者为同模型家族独立上下文 `/root/mission_review`，日期为 2026-11-05。本报告冻结下列旧候选 hash 的反例，不修改生产代码、tests、env 物理闭包或旧运行结果。没有运行长 physics、600 s、grid 或模型 API。小检查与合成公开状态结果保存在 `/tmp/classic-candidates-review.jyEEaT/`。

## 结论

B1/B2 的公开输入与 runner 路由形状成立，现有 **6 passed in 0.06s**。B1 的实现是集中滚动 greedy utility auction，不是 CBBA；B2 是事件/状态优先级与距离成本分派。当前没有实际 dev 成绩，不能称强经典基线，也没有选出 B_classic*。

**当前实现有两个未关闭的共享状态机缺陷。** P1 是 handoff 已受理、receiver 仍 queued 时，旧 source 仍会触发第三机重复交接；P2 是物理 inspect 完成但类别未决时，候选会永久被 confirmed 等待分支压住。两者均用 B1/B2 的纯公开快照反例复现，属于方法实现问题，不是对实际任务性能的估计。应先修复并核对完整生命周期，再消耗实际开发运行资源。G4 与 formal 未放行。

## F1 [P1，未关闭] 受理后的 handoff 缺少候选级接收任务保留

位置为 `eval/inspection_classic.py:113–120,165–176,178–202,220–225`。`_tasks` 对同 candidate 的 active 作业只取第一个 source。旧 observe source 因相机退化需要交接时，即使另一个健康 receiver 已 queued，它仍只看到旧 source，并再次构建 handoff。result 在 ACK 时清 destination 的 pending，并把 candidate 加 observed；但 source 非空时，observed 不阻止这条重复路径。

实际纯状态机反例保存在 `duplicate-handoff.json`，两个 policy 都得到同一序列。初态包含低质量 source1 正在 observe(moving)，两个已经超过 minimum_commit_seconds 的成熟 survey2/3，以及健康空闲机0；moving 已有报告。操作序列如下。

1. 第一次 decide 产生 handoff1→0，source version4、receiver version0，公开 args 正确。
2. 接收 accepted ACK 后，下一公开快照明确包含 receiver0 的 observe(moving) queued，source1 仍 running。这正是先获取再释放 source 的正常等待期。
3. 第二次 decide 却发出 `cancel_job(search2,expected_version=1)`，为同一 moving 再准备接收机。
4. 对 cancel 的 accepted/version2 ACK 和新鲜快照处理后，第三次 decide 发出 **handoff1→2**，candidate/report 仍相同，receiver0 仍未终态。

这不是 ACK 没清 pending 导致的简单挂起，恰好是 ACK 清了 RPC pending 后没有保留 candidate/job 的接收生命周期。它会取消其他搜索并占用额外飞机；本报告不据此断言实际 episode 分数降低了多少。现有单次 handoff 测试只验证正确版本和 ACK 清 pending，没有覆盖 receiver 终态前的后续快照。

修复应维护已接受 handoff 的 candidate→receiver job 保留，并检查该 candidate 的全部 active 作业。健康接收作业即使 queued 也应抑制第三机重复接收；receiver failed/cancelled 后按规则释放和 cooldown，不能用一个 observed 布尔值代替整个生命周期。source 仍须保留到真实 receiver acquisition，且不得取消 source 的 successor token。需要两 policy 的 ACK→queued→后续快照→终态恢复完整反例。

## F2 [P2，未关闭] 类别没有严格多数时永久等待已经到齐的确认包

位置为 `eval/inspection_classic.py:47–55,64–71,122–124`。`_ingest` 根据 inspect succeeded 无条件加入 confirmed，`_reports` 则要求严格类别多数。若物理两视图完成但类别投票打平，报告不会提交，`_tasks` 又因为 confirmed 而始终跳过，仅解释为“完成遥测先于最后确认包”。

`boundaries.json` 中的两个 policy 反例都使用两份已经到达、quality=.95、机位相隔25 m的 inspect 包，类别分别为 road_obstruction 和 bank_anomaly，并带有该 candidate 的 succeeded inspect job。20 s 时没有 inspect/report，仅安排 survey；到29 s，telemetry 新鲜、survey 已超过8 s、两包明确均已到达且候选仍在30 s有效期内，仍没有 inspect 或 report。confirmed 含 ambiguous，reports 与 pending_reports 均为空。

这是“物理核查完成”和“类别确认完成”两个事实被同一 confirmed 状态混用。没有等待中的新确认 job，就没有保证会出现第三份 narrow 证据来解除状态。它不能仅按无限等待最后一个包来处理。应有限等待补传后，进入明确的未决/再确认或 cooldown 路径；类别已决、报告 pending 或容量已满时则不要无意义重复确认。该修复仍需新版本检查，不回填此反例。

## 其他核查及实际边界

| 核查面 | 本次证据与结论 |
| --- | --- |
| 公开输入 | policy 只从 decide(state)/result(name,args,response) 取得公开数据；classic 模块不导入 Mission、生成器、truth、seed、未来轨迹或 grader。现有 poison 字段测试在额外 seed/hidden_entities/score 改变后动作不变。它证明相应代码路径，不冒称 OS 文件权限沙箱。 |
| Base 复用 | 两者覆写 decide，复用 ReferencePolicy 的结果处理、report_request 与 metadata。没有调用旧 B0 decide 作为隐藏解题通道。 |
| runner | `run_inspection_reference.py:115,125–158` 明确 B0/B1/B2；B0 默认保持。B1/B2 把 classic 源、policy ID 和 asdict(settings) 写入 manifest，settings 在 process.start 前记录。world 仍由同一 worker 推进；策略计算计时存在，但没有实际本候选时延/性能数据。 |
| 异步取消与版本重发 | 本次对两者独立验证 mature survey→cancel(version1)→accepted(version2)→新鲜 cancelled 快照→inspect(expected_version2)，均通过。super.result 会释放 pending_cancellations 与对应飞机 pending。没有靠同一批直接发取消后新任务来假装原子替换。 |
| Handoff ACK pending | override result 明确处理 to_drone，因此没有遗留“没有 drone_id 就永不清 pending”的简单 bug；但 F1 表明这还不足以保留作业。source/receiver 真实获取与释放仍由公共工具结果/后续 job 状态决定。 |
| 报告与名额 | `_reports` 用实际 narrow evidence、空间分离和最后8份严格多数；same-batch static 公开空间去重包含 pending_reports，借 base result 管理 aliases/拒绝回滚。现有 budget9/10 的去重测试通过。未使用裁判真值去重，不保证不同 candidate_id 的动态物理实体已被全局识别。 |
| 冷却 | failed inspect/observe 按 candidate 做30 s cooldown，terminal_seen 避免每快照延长同一失败；乱序包排序且每句柄仅留64份，已有迟包/失败恢复测试通过。RPC rejection 与真实 job failure 不同，未声称有完整故障重试资格。 |
| 不可用 receiver | available 排除 pending、retired、link unavailable、超过3 s的飞机 telemetry 与 camera_quality<.8；先做退出时限/余量判断，必要时发 enter_safe_exit。对这些规则的实现检查不是实际退出能力证明。 |
| Source 边界 | inspect source 被保留；observe source 健康时不抢占，退化时生成 handoff，发 RPC 前检查非同机且 source link available。source 最终合法性/精确版本仍由工具检查。F1 的问题在“全部现有接收作业没有被纳入”，不是 source版本参数拼错。 |
| 单次唯一性 | 两 allocator 的一批分派均对飞机与 candidate 唯一，现有测试通过；不能扩大解释成跨 ACK/queued 生命周期唯一，F1 已否证后者。 |
| 软优先级 | B1 用 priority×type value/cost，B2 按 handoff、priority、observe 等规则排序；没有把优先级更新变成强制清空所有工作。运行中的 inspect/observe 通常不进入重拍，survey 超过8 s才可被重新分派。它们不是全局最优、持续任意抢占或已验证迅速响应。 |
| 搜索与资源启发式 | survey 按请求次数、距离选择 region；次数在发请求时增加，不代表实际 coverage/pass。成本用距离/17、测量年龄及 active 惩罚，退出使用130 s lead和剩余航程启发式；参数尚未由 dev 结果校准或择优。 |

当前没有实际 B1/B2 dev episode、root 配对比较、经典候选选择、统一真实成本或统计证据。规则更多、名字带 auction/FSM 或这些小测试通过，都不能替代“强经典”结果。也没有从旧 B0 分数推断本候选收益。

## 执行与版本证据

有界命令为 `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_classic.py -q -p no:cacheprovider --basetemp /tmp/classic-candidates-review.jyEEaT/pytest`，退出0，6 passed in .06s。没有写 pytest cache 或修改生产 tests。额外反例使用合成公开快照和实际 policy 方法，不是物理运行/新科学样本。

初次简化 handoff 探针只 ACK receiver，漏处理其他新 survey 的 ACK，因此其他飞机仍 pending，没有触发第二次分派。随后改用明确已有的成熟 survey2/3、真实 ACK 与后续 job/version 快照，得到上文两个 policy 均成立的重复序列。报告采用后者，不把未复现的初次探针写成发现。

| 文件 | 审查 hash SHA-256 |
| --- | --- |
| eval/inspection_classic.py | `f3da40811a356ba6a7b39e53d968d7e24d9bbcf430abb4446075cfbf5be50cc2` |
| tests/test_inspection_classic.py | `9064df55e31ea32523e7c2bf46b866e8bb2f139d7b3cbb6d1cf5c38259ec6252` |
| eval/run_inspection_reference.py | `90509286b5e3d7bb849b2307f18336943bd1f2de66a0bf1b701099bb41b3f8b9` |
| envs/inspection/reference.py | `85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3` |

复现结果为 `/tmp/classic-candidates-review.jyEEaT/duplicate-handoff.json`、`boundaries.json`，现有测试日志为同目录 tests.log。文件路径均相对仓库，报告只约束上述 hash；后续修复另行复验。本报告不授予经典方法 dev 选优、B_classic*、G4 或 formal 放行。
