# Formal whole-graph 软件接线 source-only 预审

结论：沿本 immutable 集成映射与已审组件，未发现确定的 phase/schema/path/producer 组合死路，亦未找到必须另建 scientific scheduler 或修改科学协议才能抵达下游的代码缺口。这是 source-only 预审，不是整图软件接受、运行资格或 G4 决定。实际 G4 仍缺完整 108 pilot、actual clean/20s 集成、最终 source/capacity qualification 和最终独立接受；522 项作者软件 green 不能填这些证据。

审查者 `/root/control_review`，同模型家族独立工程上下文。当前 pilot 在途，本轮仅定向读源码/小文档并做文字推理：没有导入/执行生产模块或 NumPy，没有 pytest/probe、全树 hash、native 操作、formal RNG/model/physics，也没有触碰在飞进程或104项生产来源。没有创建临时试验树；仅最终写本报告与少量 evidence 文档。

## Subject 和证据层次

Subject `planning/formal-admission-candidate.v3/response-resource-integration.v1/binding.json`，实际核对其单文件 SHA256 为 `6c7456781f6dc985b570b5d5d463968ec8fe320b2a8f9a8d16d6fff1c826993d`。该映射声明 44 source / 27 tests，逐项指向既有 immutable frozen_copy；本轮按这些路径定向读主要接口，不重新逐行或 bulk rehash 全44源。`source-refs.json` 保留所读 frozen 路径及 manifest 声明 SHA，不能误读为新做了44源完整环境 attestation。

复用边界：原 namespace/dataset/worker/dispatch/runtime/receipt 与 SDK/分析/统计组件各有限评审继续各自有效；本轮只查跨组件衔接。最近闭合的 response v2 评审（report SHA `385bf6f82812b090f4468f0fba04c16a99d20f78e2f7316cfaa6a40b162bb078`）已实际复放 multiresponse 原反例，但那是合成 reader 接口，不是本整图真实正式执行。secondary consumer 评审（`95ba358cf839651141a84a617c4cfe4ef5367da670fdb329bb479cdfa502ea0d`）有 Fake/native-shaped+真实 SDK 连接器正反结果；本轮未重跑，亦不由此接受后来 event v5 的科学语义。

Subject 的 author logs/522 passed137.58s/compile0 是已提供的作者检查记录。本轮没有执行或独立重计这些日志，没有将其包装成新整图实证。本轮新增源码阻断发现为零；新增执行实证也为零，只有上述 binding 单文件身份核对。

## 完整接口路径与必要产物

下列短文件名均指 subject 对应 `source/eval/...` 的 frozen_copy，完整位置与声明 SHA 见 source-refs.json。

| 边界 | 已有 producer / consumer | 必需产物与衔接 |
| --- | --- | --- |
| G4/source proof | `inspection_formal_admission.py:188,291` | 用 actual owner 包与独立 SDK 包建立 context；typed G4 acceptance→subject+independent review→实际 applied native finding；design/population/registry/evaluator/source/deps/硬件/证据均需真实绑定。这里不生成 namespace |
| pre-entropy namespace | `prepare_inspection_formal_population.py:110` / `inspection_namespace_admission.py:177` | G4 与现行 owner 验完，独占 canonical intent 后才取 master/root/schedule/sampler entropy；intent 文件和 native namespace finding 必须先于 geometry。existing/partial 不自动替换 |
| dataset/review | `write_geometry:184` / `inspection_dataset_admission.py:88` | 每个 root case+generation audit+完整 manifest；独立 `datasets/{phase}.v1/independent-review.json` 引用 `review/...` report；native dataset finding 后才有 DatasetProof |
| primary execution | `prepare_inspection_formal_execution.py:16` / `inspection_formal_execution.py:76` | 已验 dataset/review→固定 order+sampler map+execution manifest；另以 native finding 应用。producer 是可调用 API，缺 CLI 不构成缺算法 |
| secondary execution | `inspection_formal_secondary.py:68,101` | 从同一 admitted main primary plan 确定12个 L3 parents、固定 arms/parent indices/原 sampler；manifest+order 和 native finding。没有新 roots 或新采样器选择 |
| dispatch→worker | `inspection_formal_dispatch.py:171,193,127` / `inspection_formal_case.py:40` | 固定 packet→typed exact slot→private copy，同 buffer SHA+JSON+content identities；secondary v2 payload含 dispatch，clean 在私有已验 parent 上变换；`run_inspection_reference.py:44–77` 在 warmup前验 formal 内容，epoch/pacer随后才启动 |
| model/native meter | `run_inspection_formal.py:51,88`、`run_inspection_llm_episode.py:73`、`inspection_model_service.py:30` | plan/argv传实际 phase+dispatch；每 attempt 原生 reserve→service→计量→settle，`admit_model_settlement:99` 比较实际 known ledger/purpose/receipt；finite2/100ms、unknown拒绝与原清理链继续存在 |
| provided SDK | `run_primary:120`、`run_secondary:35`、`inspection_recording.py:78,94` | 固定下一 callback与计划 slot一致；SDK envelope先打开，complete矩阵/原重复不减；completed/read-only receipt后填 value/evidence。main保留 accepted outer 原字节，holdout仅既定12根描述设计；secondary接固定extra矩阵 |
| main analysis→lock | `analyze_inspection_formal.py:327–414` / `read_primary_analysis_lock:101` | fresh原 SDK/每slot/闭合文件/native计量重验→actual subject；独立 main review后才能制作 canonical lock并 native应用。analysis entry 明确不自授 lock |
| once holdout→descriptive | `write_holdout_opening:87`→namespace→geometry→execution→同 primary runner/analyzer | 检查锁定 main及其 dataset，先产生/应用 opening，再允许 holdout intent entropy。main/holdout master/root/sensor/语义/config碰撞链保留；holdout只描述 L2/L3匹配方向/大小，family size=0，不加第7检验 |
| secondary baseline join | `analyze_inspection_formal_secondary.py:36,226–240` | 必须有实际 native main lock；重新验收 selected main episodes并与锁定 admitted-score Artifact逐行比较，再与新 secondary records连接。不是读同名未绑定分数或用重复响应充当独立 roots |

固定分母路径相容：main 72×18=1296（outer576），holdout12×18=216（outer96），extras clean216/ablation108/latency180；secondary referenced primary 为216/36/36。公开 IDs 来自 accepted design 与 `accepted_phase_layout:35`：main ID各L1/L2/L3 16、OOD各L2/L3 12；holdout仅ID L2/L3各6。这里没有生成、枚举新种子或 materialize任何正式 case。

`analyze_inspection_formal.py:398–412` 的实际 main subject schema、72/1296/576/global6、design/source/data/分析引用，与 `read_primary_analysis_lock:112–144` 所需形状相容；新的 scalar/response refs是额外绑定输出，不改变旧 subject形状。holdout reader必须经该真正独立 lock 链；结构 reader 不单独认证任意 JSON 为真实完成，这是旧信任边界，不在本轮扩大。

## 不是缺代码的手工/原生适配环节

这些产物现在尚未存在或尚未获真实接受时，下游应拒绝。它们属于已有协议的操作前提，不是本次找到的不可执行死路：

1. **最终 G4 pack**：`build_development_catalog`（`:87`）已有可调用 producer，最终 source manifest由 context实际消费集合确定，另需 hardware/evidence/原18slot classic selection/actual review。root可用有记录的一次性适配器写 typed subject/review/acceptance与SHA，再经唯一 native prepare/apply 流程；只有实际独立审查成立才可给 qualified标签。此预审不替它产生 GO。
2. **逐阶段 native findings**：namespace、dataset、primary/secondary execution、main lock、holdout once-opening 的 finding ID/evidence必须用现行 Controller正常应用；磁盘上写出 producer artifact 本身不授权限。`require_native_binding:255` 仍检查真实 request digest、applied decision及后续 revises。
3. **dataset review/main review 与 main lock**：必须实际完成内容/碰撞或 raw SDK/统计复核，由独立身份写 review；合法一次性 JSON emitter可做序列化和提交，不能替代独立工作或制造假 review。main lock固定 `analysis/main-primary-lock.v1.json`；dataset review固定 canonical path，report在review/。现有 reader形状明确，无需另写平行 grader或scheduler。
4. **run-request JSON**：primary 10字段、secondary多一个 secondary_execution并换typed schema，可按 `read_request`/`read_secondary_request` 用一次性 emitter填写已验 Artifact refs。`prepare_primary_execution`/`prepare_secondary_execution` 可直接调用现有 API；不是因为无专门 CLI就必须扩写生产算法。
5. **runtime 与 SDK不可混淆**：现行 `ACTIVE_CONTROLLER.md` 指 actual d809 owner为 `runtime/autoresearch-document-snapshots-v2`；原 reader-v1的 scripts/research.py 是稳定exec兼容入口，旧直接Controller已失效。run-request.runtime必须指 actual owner package，sdk保留原 recording SDK package `runtime/autoresearch-state-reader-v1`。factory显式分别接受两者并 union消费者；`owner_action` 的旧CLI路径可走已装兼容exec，不代表可把旧直接Controller当actual owner。该配置有合法路径，不是缺实现。

相应 native 顺序仍是 G4→main namespace→main dataset→primary execution；实际 SDK和analysis producer依该execution后产生 subject；review/native main lock→once opening→holdout namespace→holdout dataset→holdout execution。源码显式 index guards和实际 producer数据流分别提供这些约束；本报告没有声称一个结构 predicate单独证明了真实 raw/stat完成。

## 当前仍阻止正式开展的实际前提

- G4 真实开发证据尚不完整，不能凭这份代码预审或作者522green发 main/holdout permission。当前108 pilot未完成、actual clean/20s与最终capacity/source独立接受缺口按实际任务继续保留。
- 44source integration map是 staged模块身份清单，不是完整可运行env或最终G4 source_manifest。factory实际集合还包括envs、actual owner+SDK libs/scripts/refs、固定evaluator closure、包init/absence及61依赖；需在合法安装和最终freeze后形成实际 closure。现有pilot只证明它实际104source及当时inputs/deps，安装后的组合证据范围需最终独立判定。
- 只有正确的 qualified physics interpreter/四线程1和literal MLX venv配合固定registry/模型receipt，现有CLI链才具备运行前提；本轮没有执行任何解释器import/ABI/负载capacity检查，不能称实际环境已经整图验证。
- 完整真实 raw receipts、project logs、模型operation archives、SDK复制和分析输出必须保留供实际 review；metadata ZIP/Artifact SHA不等同完整physical-state重放。main lock的独立工作不能由结构标签取代。

## 未验证与完成状态

本轮没有新增执行实证，因此未对所有阶段串行运行一次、新schema在真实MLX解释器import、最终source/dependency/capacity、实际formal生成可解性或全部分布的600s/1x作通过判断。未发现具体需要本轮CPU探针才能裁定的源码矛盾，故没有为泛化confidence申请/启动测试。后续如出现具体接口疑点，应先在实际between-episode barrier安排有界Fake或导入检查，不能干扰在飞pilot。

本 source-only预审已完成，结论为“所审组合未见确定死路，真实门禁/产物仍需按顺序完成”，不是整体接受。保存本报告和binding后停止R写入，不继续扩展。
