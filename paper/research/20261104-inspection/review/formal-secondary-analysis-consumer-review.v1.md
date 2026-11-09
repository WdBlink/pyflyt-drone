# Formal secondary analysis consumer 有限独立复核

结论：在本轮限定 SDK/receipt 连接器范围内未发现新的阻断项。三个完整合成 SDK 分母及锁定 primary 对照 join、发布前闭包、原字节复制和诊断 ZIP 布局均有实际正反检查。该结论不接受尚未完成的整图、不授 G4/main/holdout，也不认证事件采纳/恢复数值、统计推断或人类领域正确性。response/resource 诊断仍 pending。

审查者 `/root/control_review`，同模型家族独立工程上下文。没有修改实现、测试、旧证据、生产来源或 native state；全部执行限纯软件和 `/tmp` 合成 native-shaped fixture，无实际 population draw/model/physics/资源动作。

## Subject、版本和边界

Subject：`planning/formal-admission-candidate.v3/secondary-analysis-freeze.v2/binding.json`，SHA256 `c50e74768028f747d07e6f3ee20805f6b996dfe75e79e6bc63ffa5a6e74f7bb3`。

新 entry `source/eval/analyze_inspection_formal_secondary.py` SHA256 `3756da21c9d338d415dcff4034121e81c736a6dc3966d83e4d5661d6ddde54d6`；REQ 模块 SHA256 `508026b4d3ac69f5bd7de8cfc7e53f3343dca88e64a4cfc79c00335104f37c20`。八个 subject 成员的完整 SHA/size 均核对，见独立 binding。纯统计模块与已指定的 `b26803c5ed4ee82927703660b8b63c949963f8aa6f60503a820487905bc5b3ce` 相符，本轮不重复数值审查。

复用已关闭的 primary v6 `_receipt_snapshot`/`scalar_score`：冻结及探针所用 working entry SHA256 `edeaeb6e76243f90d013bbd0691608e226b1e2d3d1b227ae09fdb6a869fc795b`，body 未改；其评审 `formal-primary-analysis-consumer-review.v5.md` SHA256 `86d92635bfd90e9d3930f089e0e020e352b93458a41a2566b4b5f8946ac595f4`。不重验此前完整 namespace/dataset/worker/native 证明。

测试 bootstrap 版本单独记录：root 的 v2 focused 告知为 61 passed / 28.89s；随后的整库 collection 因生产 common_calendar 缓存没有 formal 函数而 ImportError，尚不等于执行测试失败。v3 freeze（SHA256 `0d1fadf64cd4a02d6fc7c0d80ebbca015fbd7924646aa13a9cecf096f16e934a`）只给 connector test 加一行显式载入 staged common_calendar，所有 source 与 v2 相同；自有探针使用该 bootstrap（test SHA256 `bc31fec84d014b733b89c16e12171dbbf93339f62120b4fed54bd0192c75f68c`）。该 fixture 修补不是 production 实现变化，本报告未声称之后的 full suite 已通过。

另一个审查流程期间，working event helper 和其测试有意变为 `86b872af…` / `e15fad5e…`。本轮记录并排除这项外部变更；consumer/REQ/纯统计/primary helper 保持探针前后原字节。最终探针显式把 event summary 函数绑定本 subject 的旧 frozen event 文件；event analyzer 与 common-calendar 评分本来即 Fake。本报告绝不认证新 event 实现，或把它冒称 immutable v2 原字节。

## 源码信任链核对

- `analyze_inspection_formal_secondary.py:36`–`:53` 从实际 primary/secondary admission 读取新鲜 proof，要求同一 primary，再经现有 `read_primary_analysis_lock` 验 main lock；锁定 subject 的 dataset 必须与 base request 相同。`:51`–`:52` 的三新 semantic refs 与 recording_index 使用 Artifact SHA，未以同名裸 JSON 代替。
- `:56`–`:77` 在 measure 前记录闭合树与绑定 plan，核 main phase、execution、case/condition/repeat/sampler/order，secondary 另核 track/arm/parent；调用实际 readonly measurement 后比较原快照，再输出八端点及原完整 score SHA。已有 loader `inspection_measurement_loader.py:135`–`:150` 用 `read_only=True`，reserve/settle/dispatch hooks 均为 forbidden；`:156` 要求完整原 score 重计一致。本轮 Fake measurement 不重新认证真实物理计量。
- `:110`–`:181` 验完整请求、order/arm/seed、accepted main design 派生的固定 secondary specs 和完成分母；clean 三 envelope、ablation/latency 两 envelope。先 capture 同 buffer SHA，再把 captured bytes 写到私有 TemporaryDirectory，原 SDK `_experiment` 校验该副本，未重新信任晚读原件。
- `:183`–`:225` 将实际 receipt 的 value/full evidence 与每个 SDK raw 核对，并重算每个原 root 的完整 repeat 均值；`:226`–`:240` 按 bound admitted-score artifact 的 1296 个唯一身份，重验被引用 primary 原 episode，并逐行 canonical 比较，未把旧 scalar 摘要当真实 receipt。
- `:243`–`:257` 从已绑定 world buffer 与 dataset case 做诊断连接；common 全 score 留入 detail，scalar 和 full SHA 并列。此处的 D1、双路径、恢复/删失/NA 意义由另审负责，本轮只查输入/输出 join 与布局。
- `:258`–`:264` 及 `:290`–`:294` 在收集尾和发布前复核最新 proof/main lock/subject、semantic refs 与全部 primary/secondary 原闭合快照。`:295`–`:305` 复制 captured 原 bytes，outer projection 仅改 design 引用位置和 raw 相对路径，未改 value/evidence 或复制的 design 内容。`:308`–`:329` 给两 ZIP 与每成员明确 SHA。
- REQ 相对 primary-v6 仅增四个实际消费者：intervention_statistics、analyze_inspection_formal_secondary、event_endpoints、inspection_tracking；diff 已保存。没有创建 main/holdout lock 或 opening 的调用，subject `:347` 明确保留 response/resource pending。

## 实际软件证据

自有 `probe_secondary_consumer.py` 最终实际 exit 0，线程限制设 1、禁 pyc。使用原 SDK、已提供的 fully synthetic native-shaped fixtures，真实执行连接器；admission 上游、物理 measurement、common-calendar/event 跟踪和数值结果使用 Fake。没有真实正式 namespace/geometry/data，也没有把 Fake owner 当真实 controller。全套或其他流水线未重跑。

| track | 新 episode | 重验 primary 对照 | 输出 rows | outer 均值 |
| --- | ---: | ---: | ---: | ---: |
| clean | 216 | 216 | 432 | 96 |
| ablation | 108 | 36 | 144 | 36 |
| latency | 180 | 36 | 216 | 60 |

三个完整矩阵均通过原 SDK 对 copied projection 的校验；所有捕获文件与原记录逐字节相同。projection 各 run 的 value/evidence/其余字段保持原值，只有 raw 路径搬移；复制 design 未变。两 ZIP 的成员数、成员 SHA、ZIP SHA 均逐一检查。clean 的完整 common score 经 detail ZIP 与 row 的 `common_full_score_sha256` 精确 join，八端点投影亦一致；这里只证明布局，无 common-calendar 科学语义结论。

在 logs 放入同名但未绑定的 admitted-episodes.json 不影响真正锁定输入。analysis 期间 Fake owner 保持不变；执行/dispatch hooks 换成 forbidden，实际未调用。positive 还保留原 pending 声明。

12 个独立实际拒绝分支（逐项错误及检查结果见 results.json）：

1. SDK 丢一槽，原 SDK 拒绝 `all planned runs, including failures and stops, must remain recorded`。
2. SDK order 错 arm；3. order 错 sampler seed。
4. 锁定 admitted-score 文件字节改变；5. 同一 primary identity 的重计 score 与锁定 row 不同。
6. Fake physical measurement 抛 unknown operational error，异常向上传播，未填零。此项不冒称新验了真实 meter 的 unknown 逻辑。
7–9. admitted binding 分别错 arm、sampler seed、parent index，均在 measurement 调用之前拒绝。
10. 全部 receipt 校验后，在统计阶段新增 primary operation 空 call 目录，发布前原树 guard 拒绝。
11. 同一阶段改 locked main semantic artifact，fresh proof 拒绝。
12. SDK 私有副本校验时修改原 SDK results bytes，原 SDK 验的仍是 captured 副本，收集尾对原件 SHA 复核拒绝。

所有 12 次均确认没有新 analysis 目录、copied recording 或 projected results。为在同一 Fake recording 上做时序反例，脚本仅移除自己刚创建的合成 positive 输出，未触及真实 run 或旧证据。

保留两次探针装配事实：首次自有 tempfile 未 resolve，遇到 macOS `/var`↔`/private/var` 路径差异而失败，归一化后修复；第二次所有业务断言走完后，强 guard 检测到上述外部 event working 变更而中止结构结果写入。最终只显式绑定旧 frozen summary、收窄 working 稳定性声明至本 consumer 范围后重跑通过。未把这两项作为产品缺陷或初次即通过，也未掩盖外部版本差异。

## 裁定与未完成项

未发现本 subject 连接器内可复现的新阻断，可继续完成 staged 集成和其他独立审查。事件 helper 的新旧版科学意义、pure 统计数值/interval、真实 physical/native receipt、response/resource 诊断及完整安装/执行图仍各有独立门。软件结构阳性、metadata/detail ZIP 或合成 owner 均不能替代真实正式数据、G4 或领域人工验收。

报告与 binding 冻结后停止全部 checks 和 R 写入，不扩展本轮范围。
