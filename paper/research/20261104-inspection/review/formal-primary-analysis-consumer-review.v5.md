# Formal primary analysis consumer v6 L1 修复有限复验

结论：前报告的 P2（合法 L1 `D=None` 被拒）已在本 immutable v6 关闭，本轮未发现此小 delta 的新阻断。结论只限 L1 NA 评分准入与投影修复，不授 whole graph、G4、正式 main/holdout 或人类领域认证。

审查者 `/root/control_review`，同模型家族独立工程上下文。仅写本报告及独立证据，未改实现、测试、旧报告、生产来源或 native state；没有实际模型、物理、正式 RNG 或资源动作。

## 精确 subject 与差异

Subject `planning/formal-admission-candidate.v3/primary-analysis-freeze.v6/binding.json`，SHA256 `22affb24af636e16ef77cc0da756255c4717070e025b424a2bbe023248109808`。

分析 entry SHA256 `edeaeb6e76243f90d013bbd0691608e226b1e2d3d1b227ae09fdb6a869fc795b`；测试 SHA256 `a2c3eed7d5c09c1e33f25ddb4159357d728a7118647707fcf27248050777c302`。五冻结成员的完整 SHA/size 与声明均核对，探针前后五 working 对应成员与冻结字节相同。精确 v5→v6 patch、五成员清单及探针都在独立 evidence 中。

差异仅为 entry 增加 `scalar_score` 并在原位置调用，及相应测试：Fake 1296 fixture 改按已绑定 slot level 对 L1 给 `D=None`，检查 admitted 输出 null 恰对应 L1，再增加六项直连 NA/value case。准入模块和两数值模块保持 v5 原字节。未审本轮之外的新统计模块或其他工作文件。

关闭依据的前报告为 `formal-primary-analysis-consumer-review.v4.md`（SHA256 `323aaa7dad55968b5f694aa53347d38ce75e3fe2ffd97160b8cbd9491f93acdc`）；其 v5 缺陷与原证据保持不变。本轮不回填旧版本，不把此前软件 green 当作旧 L1 已通过。

## 原反例复用与正反结果

自有 `probe_v6_L1_repair.py` 在 `.venv/bin/python` 下最终 exit 0，约 0.17s；线程变量均设 1，关闭 pyc。仅导入模块、调用 helper/无构造的评分对象及原代码窄块，没有构造或运行 physics/model，没有完整 SDK、数值重采样或 full suite。

复用原评分来源 `planning/formal-admission-candidate.v3/source/eval/inspection_scenarios.py:81`–`:104`（SHA256 `9048fb7b55dde1f5f262c8c60d12357259cc81c0cf9b2b22bec620cc71812638`），再次只以 `ScenarioMission.__new__` 建合成 L1 对象并直接运行实际 `grade()`；`:99` 返回的真实评分格式 `D=None` 现能通过实际 `scalar_score`。这仍是合成评分对象，不是实际 L1 飞行证据。

| 有限检查 | 实际结果 |
| --- | --- |
| 原 L1 完整评分输入 helper | 接受；恰返回八字段，D 原样为 None，未填 0、未删除字段 |
| 原收集校验/投影代码窄块 | 原样编译 frozen AST 后接受同一 score 与同一 raw evidence；admitted score 的 D 为 None；full_score_sha256 等于原完整 score 的 canonical SHA |
| 原 full score 与 raw | 深度内容与 canonical SHA 保持不变；所有嵌套详情仍在原 score/raw 中 |
| L1 数值/非有限/其他类型 D | 拒绝；L1 必须 None |
| L2/3 D | 正常 0、.5、1 接受；None、NaN、Inf、越界、bool、string 拒绝 |
| 其他字段与类型 | 150 个 finite/binary/缺字段/Y 不一致/非法 level 变异均拒绝；包含 C/S/Q/Y 非有限、越界、bool、None，以及 H/E/SR 的 bool、float、越界、None |
| raw evidence nested/scalar 漂移 | 仍在投影前拒绝 `SDK value/evidence differs from independently revalidated physical receipt` |

原代码定位：`source/eval/analyze_inspection_formal.py:35`–`:44` 的 helper 严格要求 native int level 属 1/2/3，L1 D 必须 None；其余实际参与端点保持 native 有限 0..1，H/E/SR 保持整数二值且 Y 必须等于验证值。收集器 `:229` 在原完整 raw/evidence 一致性检查之后调用 helper，`:233` 仍对原 `score` 取完整 canonical SHA。它没有将投影覆盖原 score。原 component 汇总的 L1 D NA、semantic refs、预约前缀、文件/目录闭包与 CLI 逻辑均无 delta，沿用已审边界。

注意：窄投影探针复制并原样编译 frozen collect 的证据核对至 row append AST，实际调用本版 helper；它没有重跑整个 collector 或实际 `measure_completed_episode`。完整物理验证与 SDK admission 的可信性沿用此前审查范围，不凭本次 Fake 输入额外认证。

## 测试与未证范围

本轮未重复 1296 Fake SDK 集成或全 suite。root 告知 session68393 全候选检查在跑，另含不属本 subject 的新 pure 统计 fixture；本报告不等待或据未复验的全套状态扩大结论。新增 fixture 的 null 断言从源码可见，但它是否通过全 suite 需以 root 最终日志另行绑定。

原 source world/SDK raw 复制逻辑没有变化；本次只验证原输入 score/raw 内容守恒，不宣称重验全部 SDK envelope、真实数据/统计、全 physical state 重放或完整 graph。metadata ZIP 仍是哈希索引而非 raw flight。

全部 checks 已结束；报告与 binding 冻结后停止 R 写入。本修复关闭 P2 的事实不等同 G4 或正式执行许可。
