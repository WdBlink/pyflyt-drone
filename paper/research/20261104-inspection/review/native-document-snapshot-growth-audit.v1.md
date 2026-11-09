# Native documents snapshot 增长与兼容入口只读审计 v1

结论：主要重复源已定位；未来轮次改用同一 owner 拥有的不可变 documents snapshot + SHA 引用，在保留旧内嵌格式的条件下可设计实现。它可以消除当前每完成轮次约6MiB的 documents map 增量，不需要重跑或重标既有 episode。不过当前尚无实现、迁移、容量/恢复验证或 GO；旧 CLI 路径不在 pilot63 和 `implementation()` 的闭包内，这是必须显式补齐的接口绑定缺口，不能作为静默替换入口的理由。

审查：2026-11-06，同模型家族、分离上下文的独立 Agent。仅静态读取12个相关源码/契约/测试文件，读取已有单个 round request 的 documents 元数据并计算体积，核对 source manifests；未反序列化或披露约70MiB controller 大状态。未调用 owner CLI、执行迁移、改实现、创建第二 ledger/scheduler、生成 main 实例/模型/物理数据。精简证据见 `review/evidence/native-document-snapshot-growth-audit-v1/bindings.json`，不复制 documents 全 map 或 owner state。

## 已测事实与增长路径

点时只读测量为 control **75,046,701 bytes / 71.5701MiB**，checkpoint **77,554,596 bytes / 73.9618MiB**。两文件均仅读取字节统计/SHA，未对 canonical state 相等性作本轮认证；live owner 仍可能正常更新。round24 前59.56MiB、两个 checkpoint 增加约12MiB为主 Agent 提供的历史观察，本次未重建该历史大状态。

最新已存在的 round25 issued request 有25,921项 documents。单张 map compact JSON 为4,701,843 bytes，独立 indent2 为5,194,343 bytes；放在实际 `rounds[].request.round.documents` 的缩进深度，纯 JSON 复算增量为 **6,231,191 bytes / 5.9425MiB**；加 checkpoint 的 `state` 包装后是 **6,438,561 bytes / 6.1403MiB**。这是精确序列化体积计算，不是性能 benchmark。33份外置 issued request 合计80,000,981 bytes，当前最大5,626,863 bytes；只记录计数/体积/摘要，没有复制其内容。

| 路径 | 确切存储行为 |
|---|---|
| `rounds.documents`（14–31） | 递归构造 `relative_path -> {sha256, lines}` 元数据，不是嵌入文件全文。仅排除 `.paper`、delivery、.git 和 `control/round-decisions` 等路径；源码/研究 snapshots 等其他文档仍计入。 |
| `runtime.begin_round`（965–990） | `active_round.documents` 保存整个当时 map；返回完整 active round。`rewind`（937–944）也有独立的同型创建点，不能漏改。 |
| `prepare_round`（992–1023） | `copy.deepcopy(active)` 进入 issued request，`pending_round=issued`，而 active 仍在；序列化状态此时有两份 baseline map。另写完整 `control/round-decisions/<digest>-request.json`。`documents_sha256` 只是当前 documents map 的 canonical digest，不是又一张完整当前 map。 |
| `apply_round`（1029–1115） | 清掉 active/pending，但将完整 request 永久追加到 `rounds[]`。每完成一轮留下一个 baseline map，下一次 begin 又增加新 active。 |
| `record`（898–901） | `operations` 和 `history` 存 request **digest** 及 result；该 round result 是 summary，没有直接再嵌完整 request/documents。因此不能把这条增长链说成 history 又复制一整张 map。其他历史记录类型不在这条断言内。 |
| `save`（227–231） | 同一 writer 先写 latest checkpoint、后写 live control，两份逻辑状态；checkpoint wrapper 额外缩进导致更大。不是第二 ledger，但磁盘写入与保存副本会成倍。CLI `trace` 还在 run 外写完整 begin/prepare 返回值，属于额外传输档案。 |

`read_json`（49–61）有16MiB默认外部输入限制及128MiB可选硬上界；`load`、recover 和 reader migration 对 controller/checkpoint 均用128MiB。checkpoint 比 control 大，会更早触界。`save` 本身没有按该上限先验拒绝，可能成功持久化之后下一次 load 才拒绝。现有 migrate-state-reader 也通过同一个128MiB reader，不能假定越界后原迁移命令仍可用。

按当前 map 不再增长、其他字段不变的条件计算，checkpoint 剩余约54MiB，新增9张同规模 completed-round map 就会越界；prepare 的 transient duplicate、active 以及其他 ledger 增长可使其更早发生。这个条件示例不是剩余轮次或小时保证。30分钟仅是 advisory，不应用减少必要 checkpoint、丢历史或设置内部任务总上限来掩盖表示问题。

## 最小 v2 设计边界（建议，尚未实现/验证）

保持 `.paper/control.json` 与 latest checkpoint 的同一 owner、同一 lock、同一 operation ID/ledger 语义。新增文件只表示 documents map 的不可变存储对象，不赋予调度、计费、状态转换或独立“最新状态”的权力。

建议把未来 round baseline 写为明确 tagged ref，例如 `documents_ref={format,path,sha256,documents_sha256,count}`：path 固定在 owner-only `.paper/round-documents/`，filename 由内容 hash 推导；`sha256` 绑定原始文件字节，`documents_sha256` 保留旧 canonical map digest 语义，两者不可混淆。新 round 只允许 `documents` 内嵌旧格式和 `documents_ref` 二选一，未知格式、双格式和额外逃逸路径必须拒绝。resolver 读取时返回独立局部值，不能将 map 回填进 request/state，从而重新制造重复或改变历史 request digest。

- **原子持久与安全：** owner lock 下生成 strict JSON（重复键、非有限值、错误 hash/lines/count 均拒绝）；逐级目录创建后 fsync parent，临时文件 flush/fsync，create-once/no-clobber 安装并 fsync 目录。已存在的同名对象要核验字节相同，不能覆盖“修好”。路径限定、无 symlink、单 hard link、普通文件检查须覆盖每级路径及打开对象；所有权是运行约定，不是仅凭 chmod 获得硬件 attestation。
- **顺序与崩溃：** 先使 snapshot durable，再将 ref 写入 checkpoint，最后更新 control。snapshot 已写但无引用的孤儿可保留/审计，不自动挑它恢复；checkpoint 引用缺失、篡改或未落盘对象必须 fail closed。最新 checkpoint 是唯一恢复来源，不能退回较旧 export、从当前研究目录重建历史 baseline，或以同一路径当前 bytes 替代原 map。
- **读取与完整性：** 用旧内嵌 map 原样读取历史；新 ref 在实际使用前验证路径、字节 SHA、strict schema、canonical digest/count。recover/export 必须验证全部传递 ref 闭包。若正常 load 选择 lazy 验证而非每次全闭包扫描，要明确它不认证所有历史对象完好，并确保任何使用/恢复/导出不会跳过验证；不能缓存完整 map 后又写回 owner state。
- **digest 与重放：** 原 `digest`、legacy issued request bytes/digest、decision/op ID 和已记录 result 原样保留。对旧 apply 的 idempotent replay 先以原 wire request 计算 request digest，不先转换/展开/重标历史。新的 ref request 自有新的 digest，不给旧 request 追认新格式。
- **导出：** 当前 finalize 仅打包 state、assets 和 software，不会自动带出 `.paper` 下新对象。必须把 ref 闭包加入交付 manifest/ZIP 并提供明确可移植 resolver base；verify-export 校验全部 snapshot bytes、集合及 ZIP 等价，防 dangling refs。现有 export 内含 scripts 只能证明导出时 bytes，不能反向补齐先前执行的 CLI 闭包。

只对未来 begin/rewind 创建 ref 可以保持现有所有 `rounds`、history、usage_ledger 和证据字节不动。它不会压缩现有71MiB，也不证明其他字段永不增长：operations/history 仍追加，budget result 还可携带 unresolved/unreported 列表。需用计划剩余 round/episode/reservation 数量及余量做明确 synthetic capacity 测试，同时检查 prepare transient 与 checkpoint 大小；不以提高 reader bound 一次就宣称无表示上限。snapshot 单文件若将来超过默认16MiB，也需 owner 专用、显式审查的读取/分片策略，不能悄悄放宽外部 packet 防线。

## 迁移与最少函数/测试触点

源码触点最少包括：

1. `lib.rounds` 的 snapshot writer/strict resolver（可小模块复用）与 `workload` 的 baseline 适配；保持 `documents` 当前研究范围语义，`.paper` 排除保证新 owner snapshot 不反过来改变研究 workload。
2. `runtime.begin_round`、`rewind` 的新 baseline 创建；`prepare_round` 的小 ref 请求；`apply_round` 的 ref 验证及保留原 request 的存储。不要只改 begin 而漏掉 deepcopy/rewind。
3. `load/recover` 的新格式/closure 语义与一个显式的新 migration command；不要把新表示变更伪装成现有 `kind=controller-json-reader-v1` 仅 reader 修复。
4. `finalize/verify_export` 的 snapshot 闭包；v2 `implementation()` 与入口 source manifest 的 scripts 绑定；v2 CLI parser/dispatch 和旧路径兼容入口。

已有 `tests/test_state_reader_migration.py`、`tests/test_runtime.py` 的 round binding/staleness/idempotence/CLI evidence 是最直接扩展点。新 synthetic 检查至少覆盖：

- legacy-only、new-ref-only 与混合历史读入；同 baseline 内容复用 snapshot；旧 request/op replay 返回原 result，changed input 同 ID 拒绝。
- create/prepare/apply 多轮后状态只增加 ref 与真实其他记录，不再按 document-count 成倍增加；最大外部 packet 仍16MiB；计划 horizon 的 controller/checkpoint/transient 容量与时间开销分别测量。
- missing/wrong SHA、schema/count/hash 错、重复 JSON key、非有限/非法 lines、双格式、路径穿越、symlink/hardlink 及拒绝后的 state 不变。
- fsync/安装/两次 state 写之间故障注入；孤儿无权成为 latest；latest 指向坏 ref 时不回退；恢复后不重复 operation/收费。
- migration 前后研究字段完整相等，旧 history **精确前缀**、旧 operations **所有原 key/value**、旧 rounds、ledger/reservation、receipt、frozen plan/model generation 原样保留；仅明确允许实现标识、格式标识及追加迁移记录/operation 的差异。不能只把 history/operations 整体排除后宣称保留。
- relocated export 与 ZIP 的完整 ref 集合、坏/缺 ref、旧格式 export 回归；兼容入口原 argv/stdout JSON/exit code/重复 op ID 行为。

本次没有运行这些新测试；这是一份触点与必要验证规格。

## Pilot63 与唯一 owner CLI 的缺口

只读核对当前 live `logs/inspection-eight-condition-pilot.v2/source-hashes.json`：63个 source 均与当前 bytes 相等；其中13个是 v1 `lib/*.py`，`scripts/research.py` **不在63项中**。manifest SHA256 `0fa9d8dfcd54b855a6500cf6723721fbccde63a528a9f63c235d62a21b5b75c1`。`Controller.implementation()`（223–225）也只选择 lib、references、vendor scripts，漏掉顶层 scripts/research.py；独立按其公式计算35文件 implementation digest 为 `a112e7aaff3ab6c99f983a2030fc190d7ccd1af940a943604d94ad318e8fef20`。旧 CLI 当前 SHA256 `5ca105c6a26cd631ca6c5cd70a1698677b23b924a0f26a0579a349607121d7a0`。

`run_inspection_llm_episode.owner_action`（27–38）硬编码 `runtime/autoresearch-state-reader-v1/scripts/research.py`，timeout120s，期待同 stdout JSON/exit code；`lib.client.invoke`（11–23）也由自身 package 拼该稳定路径，timeout180s。旧 pure SDK bytes 可以保留；旧脚本可设计为只校验绑定并 `exec` 到 v2 唯一 owner CLI 的兼容入口，argv 原样传递，用原 interpreter，且 **在 exec 前不 import/加载旧 Controller**。旧直接 Controller 若实现 pin 不符应继续拒绝，不能留下两个有效 writer 或 fallback ledger。

这在接口层面可行，但不是原冻结自动批准的变更。原 driver 的 `check()` 只检查已有63，无法发现旧 CLI 被换成 wrapper。需要事先保留旧 CLI source 快照，显式登记旧/新入口 SHA、v2完整 source 集合与唯一目标，独立审查 migration/transport amendment；未来 dispatch 前对 wrapper 与 target closure 有实际执行的校验。仅写一个未被执行检查的新 manifest 不能关闭缺口。G4 最终 source closure 必须纳入实际 owner CLI/compat wrapper、目标 runtime 和相关 import/配置，旧 envelope 的缺失只能披露，不能追认它此前已被绑定。

## Fresh gate 的条件性答案

可以设计在 **下一次 between-episode gate** 转换 transport/owner 表示，同时维持原 case、schedule、sampler seed、controller/模型 source、费用和回执身份，从而无需任何 flight/model 重跑或 episode 重标。当前 gate 逻辑（pilot driver 139–151）在下一 slot 的 reserve/spawn 前等待，并在释放后重做旧63检查；它本身不能证明 ledger/no-inflight，也不会验证未列入的 CLI。

因此迁移前须在同一 owner lock 和实际 gate 证据下确认：无运行中的 flight/model/request/service、无 pending reserve/settle transport、无 running/uncertain attempt；usage summary 无 reserved/unknown/held 或待补关键 meter。`ready()` 只检查 attempts，**不等于** no unresolved budget reservations。最简单边界是关闭当前 semantic round 且 pending_round 为 null，再 migrate，再用 v2 begin；若保留旧 active baseline，必须原值保留并读旧格式，不用迁移时当前文档替代它。已 issued request 若因 rules/implementation 变化失效，要明确记录旧 request 摘要/归档并重新 prepare，不能丢弃它后声称所有值未变。

迁移只追加 source-bound migration 事件并转换允许的实现标识，不动旧 ledger/history/results；迁移 idempotent，错误来源/review/target/gate 时不写状态。完成后独立核对 budget totals、reservation/operation IDs、历史前缀、完整回执路径及新 owner 独占，再决定释放下一 dispatch。既有 known failure、unknown、cache 和实际 episode 都不由表示迁移重新分类。当前仍有飞行，本次未核验 fresh gate，也未批准现在切换。

最终判断：设计方向可行，现有增长及 CLI closure 缺口均为实质工程问题；本报告是只读审计和候选规格，无实现通过、迁移通过、G4 或 main GO。
