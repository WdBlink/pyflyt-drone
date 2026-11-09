# G4开发进度v19

G1–G3及同级非LLM参考对保持原范围；修复后单次L2/B4-M1实际600s clock门已独立复算通过（max55.216ms、72000步无100ms超界），原失败pilot-v2及其全部失败/未启动分母不回填。资源仍20操作allknown、414calls/2673500tokens，held/unknown/unresolved0，cost/host/capital/energy未计量；本轮没有模型、物理或实际formal RNG动作。

共享population law/正式预随机门组件已获有限独立通过，见 `review/population-shared-law-component-review.v1.md`、`review/evidence/population-shared-law-component-v1/bindings.json`。独立22+8项通过；对原5个dev种子的147个新旧候选逐一比对内容hash、接受/拒绝及原因完全一致，AST归一后仅phase身份/validator/audit角色三处改变。只重放既有dev seeds，不生成formal geometry；组件未安装，未授权writer/fullgraph。随后staged namespace补main/holdout master/root碰撞4例，并将5个已staged消费者加入G4 REQUIRED_SOURCES；这些属于新组件版本，旧freeze/review按原范围保留。

新DatasetProof候选只hash全case/audit字节，不反序列化私有mission；绑定canonical72/12 manifest、单独review/report、namespace/G4/source/population/catalog、全部分母/metadata/seed/semantic/config标签与native应用。with/dev/main↔holdout跨集合检查和locked-main递归结构已实现。root合并173项通过；独立实际29dataset+41namespace+7预随机=77通过，但发现 **1项P2未修复**：最后raw owner读取/fresh namespace之后，没有再核此前已验证的dataset bytes/tree。在该窗口，main的manifest/review/report/case/audit追加空白或新增文件的6反例，以及holdout的locked-main case/review、额外文件3同类反例仍被接受。task_stop及main/holdout dataset finding撤销控制例能正确拒绝，遗漏限定为最终文件闭包。

该阻断已冻结于 `review/dataset-admission-component-review.v1.md` 和 `review/evidence/dataset-admission-component-v1/binding.v1.json`；v1源码/原反例完整保留在 `planning/formal-admission-candidate.v3/dataset-component-freeze.v1/`。下一步v2必须在最后snapshot/fresh namespace之后，对同一已绑定版本的manifest/review/report、所有case/audit和精确tree再闭合，并覆盖holdout依赖的locked-main闭包；不能通过换SHA/版本接受变化，也不宣称原子文件系统或无限返回后保证。实际worker在副作用前仍须单buffer哈希/内容校验。该候选保持未安装，不能当G4。

新增one-shot opening/namespace/geometry writer只是未审prototype，冻结于 `planning/formal-admission-candidate.v3/writer-component-freeze.v1/`，依赖上述未修P2，不能实际启用。14个/tmp synthetic检查通过：exclusive canonical intent文件在literal entropy之前claim；完整intent持久化后才容许geometry；collision/entropy failure/partial geometry/finite exhaustion/source drift保留且禁止自动reseed或重跑。测试只用literal/Fake，没有真实formal随机抽样。holdout typed main lock/opening及main sensor seeds排除在prototype检查内。CLI、真实内容reader/worker warmup前门、classic/LLM/service每call、receipt/measurement/SDK与分析全graph仍待集成，完整开发pilot与clean+20s实际集成也未运行。

当前持久化dev身份覆盖按v18独立结果保持：56 artifact版本、32semantic/32physical、120排除seed；此后没有新增实际private-case输入。所有main/holdout实际seed/排程/实例、正式结果和论文正文均未开始。原19项义务、budget={}、统计/评分/选择器和正式分母不变。保存本轮真实缺陷与进度后，继续内部修复，不将检查点当停止或完成。
