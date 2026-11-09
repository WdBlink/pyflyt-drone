# Formal primary analysis consumer v5 有界增量复核

结论：仍有一项 P2 阻断。合法 L1 完整评分含 `D=None`，分析收集器却要求八项均为有限数值，真实 L1 不能进入分析。此校验从 v4 继承，本报告不把它说成 v5 新引入。v5 的其余投影、语义输出引用及 typed secondary 预约前缀增量，在下述限定正反检查中通过。不得以软件全套绿色宣布 L1、G4 或正式分析可运行。

审查者 `/root/control_review`，同模型家族独立工程上下文，仅负责本报告及独立证据。未修改生产/候选源、测试、旧证据或 native state；未运行模型、物理、正式 RNG、资源预约或正式数据生成。

## 精确输入与版本边界

Subject：`planning/formal-admission-candidate.v3/primary-analysis-freeze.v5/binding.json`，SHA256 `a2d521d166f523b04ebede33eb19ec6bd53f2207ea4d5a7e1c0ec445a082dfb0`。

分析 entry SHA256 `26a464ab995e02d83b29374cc40deb24db670ec21b310ba891b8262d79cff1e7`，测试 SHA256 `98a53872335c1575ff322eb5b5ee5c5a6771847c5f16b0aed4b5ff26cd919f27`。五成员冻结 SHA/size 全部核对，探针前后 working 对应成员逐字节相同。随后 root 开始的有意 working v6 修复不属本 subject，不当作 v5 漂移或本报告通过证据。

逐字节比较 v4→v5，仅 entry 和新测试改变；三个未变成员为 `inspection_formal_admission.py`、`inspection_statistics.py`、`inspection_secondary_statistics.py`。完整五 SHA、精确 patch 和探针均绑定在独立 evidence。v4 已闭合的三时序窗口、CLI、空调用目录及非 regular 条目保护沿用前报告 `formal-primary-analysis-consumer-review.v3.md`（SHA256 `b1812b126019743171ed8a6e62b4abf0b4195cde0c6df4b6b5a7f1fe26c232c4`）范围；未重复全 1296 SDK 矩阵或统计重采样。

## P2：合法 L1 的 `D=None` 在投影前被拒绝

证据：

- v5 `source/eval/analyze_inspection_formal.py:31` 的 `SCORE_FIELDS` 含 D；`:217`–`:221` 对所有八字段无条件要求 native int/float、finite 且 0..1，未按 `slot.level` 区分 L1。
- 评分来源 `planning/formal-admission-candidate.v3/source/eval/inspection_scenarios.py:81`–`:104`：`:82` 区分 L1；`:99` 明确返回 `D=None`，Q 为 C/S 平均。该旁证源 SHA256 `9048fb7b55dde1f5f262c8c60d12357259cc81c0cf9b2b22bec620cc71812638`，不是本轮改动源。
- v5 `tests/test_formal_analysis.py:55`–`:59` 的 Fake measure 为各级 D 都给数值；因此投影测试不能覆盖真实 L1 null 语义。`:270` 的 component 汇总虽然排除 L1 D，但收集器已经先拒绝，不能据此称 L1 可用。

独立重现：只用 `ScenarioMission.__new__` 创建无构造副作用的合成 L1 对象，给定一个静态目标、空报告、覆盖与四机退出/里程数组，直接运行现有 `grade()`，实得 `D is None`。随后将该完整分数及同值 SDK evidence 送入从冻结 `collect_recording` 提取、逐节点原样编译的 `:211`–`:225` 校验/投影块，实际抛出 `ValueError: finite bounded revalidated score components required`。这是原代码谓词的窄反例；未调用整个收集器、实际 flight loader 或完整 SDK admission，更未伪称该 fixture 是真实 L1 飞行。相同负例在 L2/3 应拒绝，已实际确认。

必要最小修复：按已验证 slot level，仅 L1 允许并原样保留 `D=None`，其余必需端点仍要求有限有界数值，H/E/SR 仍为整数二值；L2/3 D 仍须有限有界。`full_score_sha256` 继续针对原完整 score，包括 null 与所有嵌套细节；不得把 L1 D 填成零或从原 score 删除。保留 component 的 L1 D NA 规则，并增加源评分产生 `None` 的回归与 L2/3 null 负例。此修复尚不在本冻结版本内。

## 本轮其他增量实证

自有探针 `probe_v5_delta.py` 最终 exit 0，约 0.22s，四个线程环境变量设为 1、关闭 pyc 写入；全部使用 `/tmp` 合成 fixture，未跑全 suite。检查范围和替身明确如下：

1. 原收集代码 `:207` 从 `measure_completed_episode` 取得 full score，`:214` 对同 Y 和完整 evidence 的 canonical SHA 做一致性比较，`:224` 才投影。窄探针执行冻结原始校验/投影 AST：正常 L2 输出恰为 C/S/D/Q/H/E/Y/SR，`full_score_sha256` 等于完整 score 的 canonical SHA，嵌套详情未被改动。仅修改 SDK nested detail 或 scalar，均在投影前拒绝。缺 C、bool C、NaN/out-of-range D、float H、不同 Y 均拒绝；NaN 实际由 strict canonical JSON 更早拒绝。
2. 实际 `component_summary` 在字面两级、完整八条件和既定 repeat 形状的合成数据上，保持 L1 根/层无 D 且 `L1_D=not_applicable`，L2 D 均值正常保留。此项通过不抵消上述 collect 阻断。
3. 实际 `analyze_primary` 发布函数用 Fake admission/collector/numerical results 和小型 L2 字面 grid运行，仅检查新增 subject 引用：`:385`–`:387` 的 `admitted_episode_scores`、`component_summary`、`sr_statistics` 均指向实际输出字节的 SHA；分别改一字节后，原 Artifact reader 均拒绝 `admission artifact changed`。合成 captured raw 在复制后逐字节相同，独立合成 world 文件未变。此项不证明 collector→publisher 全链或真实统计/SDK 已重验；metadata closure ZIP 仍是路径/哈希索引，不是 raw flight。
4. 实际 `_receipt_snapshot` `:51`–`:60` 对 LLM slot 的 track 只接受 clean/ablation/latency；没有 track 时仍派生 main/holdout。五种合法路径各自得到正确 `formal-{kind}-...` operation/resource/reserve/settle 前缀，原 main/holdout 不变。`unadmitted`、空字符串和布尔 track 均在 operation lookup 前拒绝。各合法前缀下新增空目录或 FIFO 均被原 guard 拒绝，恢复后通过。这里的 slot/admission 是 Fake；该内部 helper 不替代先前已审 native/typed phase admission，未授予调用者正式权限。

## 边界与交付

本报告只检验 immutable v5 小增量与暴露的 L1 继承缺口。原 `measure_completed_episode` 全量物理评分重计、原 SDK 三 envelope、72/12 正式数据、真实资源与完整 graph 沿用既有门禁，均未在本轮重新认证。root 最初 15 focused 及随后 412 green 的告知不覆盖本独立 None 反例；本审查未重跑或据未读取的全套日志作通过依据。无需为软件 green 改写评分语义。

原问题与本轮证据保持冻结，后续 v6 须另作小 delta 复验。全部 checks 已结束；保存报告与 binding 后停止 R 写入。不授 G4/main/holdout 或人类领域认证。
