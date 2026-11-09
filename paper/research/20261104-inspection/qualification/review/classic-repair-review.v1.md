# B1/B2经典候选状态机修复独立复验 v1

日期：2026-11-05。独立上下文工程Verify，同模型家族；不冒称跨模型或人工领域审查。只写本报告，未修改production、tests、契约或旧证据。所有纯策略CPU检查已先完成，未运行physics、600秒任务、实际B1/B2 episode、付费服务或经典选优；B0真实1×开始后的CPU测试不属于本次工作。

## 结论

**旧报告的P1重复handoff、P2物理完成后class tie永久confirmed，在本次限定范围均可关闭。** 当前10项测试通过；独立使用归档旧源重新复现两类完整反例，再对修复源运行相同生命周期，确认重复交接被抑制且tie可在有限等待后重新inspect。没有发现本次修复引入的必要回归修复项。

这是纯公开输入状态机修复证据，不是B1/B2强性能、方法择优、B_classic*或G4资格。旧classic-candidates-review.v1.md和全部旧失败/反例保留，不把修复后的结果回填成旧源本来正确。

## 版本与证据身份

源码路径相对repo；qualification路径相对paper/research/20261104-inspection。

| 对象 | SHA256 |
| --- | --- |
| eval/inspection_classic.py（修复后） | 763b14320f95e5d534154c4d4886cbd40e12cf13182f64d5eb572cd7d5e4fd00 |
| tests/test_inspection_classic.py | 9857ffb1a383f97d74f2842c2fdce472415337591be7e61ab5ce9d9f61613b3d |
| envs/inspection/reference.py（复用父类） | 85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3 |
| qualification/review/classic-candidates-review.v1.md | ae2188ff64d56a6198711f280ef49e4470b73ca984e60281bccfc4d04bb40c6b |
| qualification/review/classic-candidates-v1-evidence/inspection_classic.before-repair.py | f3da40811a356ba6a7b39e53d968d7e24d9bbcf430abb4446075cfbf5be50cc2 |
| qualification/review/classic-candidates-v1-evidence/binding.json | 6e1ca39b4a7ee88734a1c93dff2100654627ff942c2edcc0ddc705e8389bd3a2 |

binding列出的旧source、duplicate-handoff.json、boundaries.json、tests.log均实际重hash匹配；旧source SHA与独立旧报告记录精确相等。本次据该已绑定旧字节复现，不仅相信“逆patch恢复”的叙述；也不声称能从SHA推断全部编辑历史。

## P1：接受ACK到receiver终态的候选/飞机预留

修复点：classic.py:39–40新增inspect_finished与handoff_reservations；:237–244在accepted handoff ACK中保存candidate→receiver job_id/drone_id，RPC pending清除后不再丢失接收生命周期。_ingest:50–63仅在**对应receiver job**真实terminal时释放reservation，不因旧source终态误释放。decide:179–181也排除被reservation占用的receiver，保护ACK先于job telemetry的窗口，避免给它塞survey。_tasks:121–125检查同candidate的全部active inspect/observe，已有inspect或健康receiver（含queued）会抑制第三机分派。

独立完整反例使用两种policy分别执行：source1正在observe但camera=.55，空闲健康机0，成熟survey2/3，有moving报告及两份新鲜公开measurement。

| 阶段 | 归档旧源 | 修复源 |
| --- | --- | --- |
| 初次decide及accepted ACK | handoff1→0 | handoff1→0，保留candidate/receiver reservation |
| ACK后receiver尚未出现在遥测 | 可能重发/再分派，无法表达receiver生命周期 | 不发重复handoff，也不给receiver0新survey |
| receiver0 queued，source1仍running，survey2/3成熟 | cancel_job(search2,version1) | 不取消第三机、不重复handoff |
| old cancel受理后新鲜version2，receiver0仍queued | 再发handoff1→2，复现旧F1完整序列 | 不进入该取消/第三机序列 |

还独立清除修复policy的reservation，仅保留公开queued receiver，直接验证_tasks的“全部active同candidate”保护也成立，避免测试只证明字典短路而漏掉第二条防线。

新增参数化测试:88–125对两种policy覆盖ACK→receiver不可见→queued→第三机survey未被cancel→真实failed通知→30秒cooldown→新鲜measurement后恢复。失败通知在22秒被消费，cooldown到52秒；terminal_seen避免同一旧失败每次快照重新延长。不得把“跳到52秒才首次看见failed”当作已过cooldown——该情形会从52秒重新开始等待，旧v2失败日志正是这个无效测试时序。

reservation在accepted ACK后持续到receiver succeeded/failed/cancelled；拒绝ACK不新增reservation。正常source保留/实际receiver获取后释放仍由既有生产handoff协议负责，本次没有用policy布尔值代替物理交接，也未给它做新physics资格。

## P2：物理inspect结束与类别已决分开

_ingest:55–56只记录真实inspect.finished_at，不再因物理job succeeded把candidate放入confirmed。_reports:72–80仍要求实际qualified narrow evidence、20m机位分离及最后8份严格多数；只有类别已决才confirmed，normal继续rejected。_tasks:131–138区分“已决，等report/容量”和“未决，物理结束后的有限补包等待”。默认等待2秒，从物理finished_at计算，不是无限等待最后包；期满且仍unresolved时可重新inspect。

独立复用旧完整边界：两份已到达、quality=.95、机位相距25m的narrow包，类别分别road_obstruction/bank_anomaly，物理inspect已结束。20秒初次decide先派survey；其ACK被真实处理，29秒survey成熟且飞机telemetry新鲜，candidate仍未过30秒有效期。

- 旧源confirmed永久包含ambiguous，29秒仍无inspect/report/cancel，重现旧F2。
- 修复源不将tie加入confirmed；有限等待后29秒发cancel(search0,version2)，accepted ACK后看到真实cancelled及fresh version3，30秒发inspect_candidate(ambiguous,expected_version3)。没有靠手动清空pending假装取消是原子替换。
- 当前参数化测试:128–144另覆盖20秒处尚在等待、23秒可以再次inspect。已决类别、pending报告或容量耗尽仍不会因此无意义重复核查；现有同批static去重与budget9/10测试继续通过。

2秒是开发策略的有限等待设置，不是对所有网络包最大时延的新保证；晚包仍由原history去重、排序和64条保留规则处理。本次不把再次inspect必然成功、类别必然正确或得分提高当作已验证结论。

## 执行结果、历史失败与检查边界

实际运行命令：

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest tests/test_inspection_classic.py -q -p no:cacheprovider --basetemp /tmp/classic-repair-review-pytest
```

结果10 passed in0.08s。未写项目pytest cache；没有运行physics。已有runtime/classic-guards.v2.log的2failed/8passed保留；v3.log记录10passed，与本次实测相符。修复测试中的真实失败通知/cooldown时序和ACK-window availability保护均在当前源/测试中可见，本报告没有把v2失败删掉或改称通过。

独立反例脚本`/tmp/classic_repair_independent.py`，SHA `bcd593a5b1b7d31480890fe51c573d581b03ddcd748a68895c9e33d37a96fe30`，退出0；输出`/tmp/classic_repair_independent_results.json`，SHA `dcd7f617ae4f57d18eec2ae26f86eb1e77427d30f4eca3f2bb2971f5047d36a6`。脚本加载已绑定旧源和当前真实policy，使用合成公开状态/真实result方法走完整ACK和版本序列，未读取truth/seed/grader或私有物理状态。

这轮保留的检查还包括原有单次唯一性、公开poison字段不影响动作、handoff版本参数、same-batch report去重、迟包排序及失败恢复；这些是代码路径边界，不是OS权限沙箱、任意RPC故障恢复或所有异步交错的完整证明。没有重审allocator策略合理性、runner时钟、实际B1/B2任务结果或B0性能。

reviewed by inherited model configuration, independent context, same model family

- 原P1/P2在当前明确source hash及复验范围关闭；旧反例先复现，再验证修复，未只抄10passed。
- 当前无新增必要修复项。检查仅约束公开快照/ACK状态机和所列生命周期；不授予实际强经典性能、经典候选选择、B_classic*、G4或formal。
- CPU检查已在B0真实1×开始通知前结束；其后只整理报告。领域人工复核不由本次代替。

报告写完后冻结；后续源码另版复验。
