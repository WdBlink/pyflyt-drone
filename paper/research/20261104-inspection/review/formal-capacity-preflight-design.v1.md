# Formal容量预检：source-only设计 v1

结论：**当前不能宣布容量可承载，也没有本轮实测“已撞上限”的证据。** d809…唯一owner已经用future round-document/result refs消除了旧live-state的重复内嵌，但仍有128MiB owned-pair、16MiB单对象/普通JSON、归档存储和传输时间门槛。必须在已确认的between-episode barrier取fresh基线做下面的离线probe；首个完整开发root后可先做部分预检，pilot结束和最终freeze前必须刷新。不能以旧模板余量、文件数或估算代替通过，也不能缩研究N、删旧记录/未计量cost或扩预算来“解决容量”。

审查者 `/root/statistics_design_review`，独立上下文、同模型家族。仅定向读取源码/既有小不可变记录；未读取实际.paper/control/checkpoint/ledger、未查询运行产物大小/free-space、未du/全树hash、未导入生产/跑pytest/NumPy/CLI、未构造大型树或formal实例，未暂停在飞任务。此报告不是迁移重审、容量通过、formal/main/G4或human批准。

## 1. 权威链与固定负载

本轮实际读取唯一owner入口 `runtime/autoresearch-document-snapshots-v2/scripts/research.py` 与必要lib；owner implementation pin **d809362cdb06563d520b197f9974af39c32c435332f6c13b31421b6f02753847**取自已独立审查的冻结身份/父Agent当前声明，本轮未重算整个implementation闭包。旧v1入口目前只有os.execv转发，不import旧Controller；actual immutable transport SHA核为 **2f698501f2d0504d3a7cb24dcff34cecb94aca2bce69b4db17b240ad1c609832**。当前关键source SHA见binding，不能据此追认所有未读来源/实际state。

| 轨道 | 新episode | classic | LLM |
|---|---:|---:|---:|
| main | 1296 | 216 | 1080 |
| holdout | 216 | 36 | 180 |
| clean | 216 | 36 | 180 |
| ablation | 108 | 0 | 108 |
| latency | 180 | 0 | 180 |
| 合计 | 2016 | 288 | 1728 |

每槽最多2attempt给出4032总attempt，其中3456 LLM attempts、576 classic attempts。LLM每attempt一次reserve与一次settle，故最多新增3456 resource reservations及6912 native operation records；还须加实际基线与分析/检查点等非episode操作。known retry不能漏，cache无新dispatch不能重复算attempt。单个模型请求不是新native reservation：它在该attempt的既有service operation内生成call-NNNN。

当前actor源码硬限每attempt最多32 decision attempts，context失败也推进attempt计数，因此最多110592次decision/transport请求的结构上界（真实admitted calls可更少）。这不是已授权预算或预计用量；实际C/T必须按最终registry/profile逐条件冻结。若同一model共用profile，最坏attempt数按M1/M2/M3为1728/1152/576；总request/token上界应使用这些计数乘各自已批准C/T，不用pilot平均值替换或新增预算。名义600s flight合计336h（主+holdout252h，secondary84h），双attempt最坏672h；不含加载、归档和分析，亦不是墙钟/预算保证。

## 2. 源码确定的限制与尚未知的量

| 路径 | 确定限制/行为 | 当前不能据此推出 |
|---|---|---|
| owner load/save | control与checkpoint各128×1024² bytes；compact序列化+换行；save先检查两者再写checkpoint→control | 当前余量、Python RSS、读写/锁/恢复耗时 |
| round-doc/result objects | 每个16×1024² bytes，content-addressed、单链接regular file、严格schema/SHA；旧inline保留 | 所有future map/result都一定小于16MiB或累计disk有界 |
| CLI input_json、普通runtime read_json、admission Artifact.json | 通常16MiB；owner读取单独显式128MiB | 128MiB owner能让普通packet/index/round-request也超过16MiB |
| formal native owner reader | 使用唯一Controller现有lock/load协议，走128MiB pair，不新建状态 | 任意callback读取都可放宽上限或省略transport检查 |
| SDK JSON捕获 | primary/secondary collector对单份SDK JSON用16MiB | scalar投影后所有design/raw/index/metadata自然都合格 |
| secondary world读取 | 显式512MiB bounded read | world一定不会超、内存只需512MiB或整个collector有512MiB总上限 |
| response/resource读取 | 单个所读actor/service/terminal等原bytes显式128MiB | 32次请求一定能约束JSON字节长度，或该cap等于全reader内存 |
| LLM owner_action / SDK client | owner subprocess120s；SDK invoke180s；stdout JSON整体捕获/解析，无另一个小输出限额 | refs会让该耗时不随历史增长，或旧CLI是第二个owner |
| runtime/admission锁与export | fcntl锁无此处独立超时；历史closure在recover/export完整核验 | 日常load历史lazy就代表最终closure便宜 |
| other temporal gates | 当前LLM supervisor900s；model ready/request watchdog120s；score evaluator subprocess120s；actor软响应上限≤60s | 慢native/admission/归档可以被忽略，或可自行提高门限 |

native identifier为1–96 ASCII字符。当前formal reservation固定prefix的最长track为ablation，在4位order下reserve身份36字符、settle-前缀后43字符，结构上未撞96；2016/1296等也不使order字段超过4位。`:04d`是最小宽度而非通用“9999记录总上限”；call序列每operation最多当前32，原lexical排序不会跨到10000。没有在所读预算/ref/SDK函数中发现按2016总episode数直接拒绝的计数上限；这不是所有组件无限制的证明。relative path、无symlink/hardlink、ref恰好三段/content-address均仍须满足；OS实际最长路径/磁盘/inode未测，不宣称已满足。

特别注意：prepare_round还内含workload.changed_files/large_files、evidence和plan_diff，apply后的round request/decision仍进state；**不是所有round内容都被ref替代**。prepare输出/部分JSON writer并不统一预检普通16MiB reader限额，所以不能只检查documents_ref而漏掉issued request。最终应验证每个产物经过的最窄reader，而不是只看writer能写出文件。

## 3. 增长仍在哪里

usage_ledger本身、operation/history中的小refs、round request/decision随真实工作增长，旧inline历史不减。每known、未计量cost的receipt仍进入budget.summary.unreported_fields；必须保持cost未知。future result refs只让live pair不再复制完整result；完整result对象、CLI trace以及attempt的reserve-owner/settle-owner wire回执仍保留完整budget summary。内容随计量变化，不能假定content-address去重消掉它们；累计磁盘仍可有ΣK_i的平方增长，不能把live state线性化说成总存储线性。

archive在settle前打包attempt/model-operation/plan，通常已含reserve-owner输出；complete-attempt.zip存于R/qualification/local-model-episodes。ZIP manifest/CRC、闭合元数据和后续export/recovery仍要遍历/读实际内容。此处writer没给应用级archive总字节/成员数承载证明，不可从ZIP能打开推出整个流程可承载。

model raw/service/flight多数在project/logs之外于R，故不直接进rounds.documents；R内resource-requests的reserve/settle JSON与usage receipt会进入。最坏LLM部分至少要表示6912请求packet、3456usage receipt及3456 archive；每operation另有service-intent/completion与每admitted call3份request/response/token journal。按当前32cap，仅call文件结构上可到331776份，但它们大多在外部logs，不能用此数直接当R documents count。

rounds.documents遍历R中指定文本后缀，排除.paper/delivery/.git/control/round-decisions；ZIP本身不入map，native refs也不反向膨胀该map。每个map对象仍保存所有匹配文件的path/hash/lines；对每个新baseline重做扫描/line计数，累计工作与snapshot存储取决于文档数、路径/JSON转义字节和semantic round次数R_future。R_future不由episode数唯一决定，也不能为容量事后挑一个很小值。owner load还读/parse pair、求digest、验live refs/source transport；reserve/settle的budget_evidence重验既有usage receipts。故JSON字节合格仍可能先触120s传输/内存/实际并发物理公平性问题。

已定向读取历史 `recovery/owner-document-snapshots-v2/capacity-projection.v2.json`（SHA `c4b039f3b95f8d53ffe486954158de1bbea85160bbfed27269d428980098aa6f`）：其base_compact_bytes=61,992,508，13,000 future ops/5,000 LLM attempts/2,000 round envelopes的projected control=131,054,508、checkpoint=131,054,594，界为134,217,728，约3.02MiB余量。两种数值分别对应control与checkpoint，不能混同。这是旧特定baseline与形状的合成结果，不是fresh事实；本轮未复跑它，更不将within_bound=true外推当前。也不重复旧lossless迁移或扩大owner reader上限。

## 4. 最小离线probe接口（待实际pause acknowledgement，不执行于本轮）

建议输入是一个小、source-bound的probe manifest，语义接口如下；这是设计，不是已存在或已运行命令：

```text
capacity_probe(baseline_snapshot_ref, source_pins, closed_pilot_inventory_ref,
               episode_horizon, final_profile_refs, round_policy, output_tmp)
```

1. **基线/clone来源。** root确认order17完成后、order18前的actual checkpoint_ready，无在飞/在途model/service及未决native transport，再由现有唯一owner的授权只读路径/已有锁获取一致control+checkpoint及它们引用对象的immutable快照，或提供其已核副本。记录原bytes SHA和时间，check两者语义一致；不运行init/migrate/reserve/settle/finalize，不重标旧episode。原.paper保持只读，不把whole state复制到R报告。golden副本放隔离/tmp，只复制普通文件、不用共享hardlink/symlink；probe只变内存副本/明确softwarefixture，不产生第二个真实ledger或scheduler。
2. **只读事实inventory。** 只列该gate已封存的具名pilot/root/model-operation/usage/archives与必要metadata，取当前counts/bytes/最长路径、已计量profile C/T、CLI stdout及artifact大小；不扫模型权重/全R树来制造负载。free bytes/inodes也是独立只读事实，需绑定其文件系统/时间。第一完整开发root的数据不自动是L3/长回复/全部事件的最大上界。
3. **精确表示投影。** 从golden state逐值保留旧history/ledger/rounds/inline；按原source序列化格式，在内存追加至最大3456 LLM attempts/6912两步ops，另列明确O_misc和登记R_future。cost仍缺失/unknown，timestamp/ID/path按最终最长真实形状，不用短placeholder或cost0。按每一边界取峰值：reserve后、settle后、active+pending request、apply后、最新checkpoint wrapper、单个result/doc对象、request/subject/index与完整closure；digest/换行/ensure_ascii/pretty transport各按真实encoder计字节。对象可流式累计，只物化需要验限的峰值，不构造十万文件大型树。
4. **输出与界定。** 输出source pins、baseline SHA、workload假设，control/checkpoint峰值、最大单ref对象与普通JSON、docs/ops/receipt/closure计数、每consumer上限/余量、wire日志与归档累计量，以及哪些仅按实际closed样本投影。archive不要把极易压缩的虚拟零内容压缩率当真实上界；若只有观测倍率，应明确不构成worst-case保证。
5. **耗时/RSS。** barrier内先量只读parse/hash、峰值compact编码、budget summary/既有receipt核验、docs扫描、closure/export读取的时间与峰值RSS；不因serialized128MiB就假定内存128MiB。完整owner/compat CLI120/180秒或保存/fsync/恢复耗时若需实测，另用source-pinned隔离softwarefixture、具名dummy receipts，无真实资源动作；不能在actual.paper上以“测试reserve”测性能。无法构造有效隔离路径/source绑定时保留未测，不把fixture路径错误当实际source故障。

### State gate与transport/归档空间必须分别实测

status不是无磁盘副作用的测量命令：该CLI也trace原result，预算summary包含随K增长的unreported_fields；它虽不新增reserve operation，也会增加CLI输出/保留文件。probe输入须显式包含计划status/其他summary调用次数，不从2016推定它为0或无限轮询。不得在飞行中调用status“量大小”。

在barrier的封存inventory中分别记录每次status/reserve/settle的实际wire stdout bytes、CLI trace bytes；按实际encoder构造峰值K的summary/三类完整result，不能只算小result_ref。记录unique result对象、原始usage/request/actor/model files与每个complete-attempt.zip的实际compressed bytes。ZIP内复制和外部原件是两份存储；统计路径集合须互斥，不能把同一stdout路径在raw与CLI类别重复相加，也不能只统计其中一份而漏archive。

建议明确实测式：`B_transport = Σ_status,reserve,settle (B_retained_stdout + B_CLI_trace)`；`B_persistent = Σ_unique_persistent_paths B_file`（含live pair、unique refs、原件、requests、transport、archives、analysis outputs，各路径只一次）；`free_needed = ΔB_future_persistent + max_stage(B_additional_temp_and_export_copies) + declared_headroom`。将free_needed与实际目标文件系统free bytes/inodes并列，不把serialized state余量当磁盘余量。对owner原子pair保存、result安装、archive构建及最终export的真实并存阶段取峰值，不随意把永不同时存在的临时副本相加；APFS/COW或快照造成逻辑bytes与实际free变化不同，须披露计量口径和外部变化。

`Σ_i summary_bytes(K_i)`的transport/result累积可以平方增长，`max_i summary_bytes(K_i)`才是单对象/单JSON门槛；两者必须分开。archive压缩与真实metadata尺寸无法由旧live-state投影证明，必须用具名closed文件和保守且披露的未来形状刷新。没有这些实测，state gate通过不能写成storage/nativeledger整项通过。

最小验收是所有计划峰值按真实最窄reader/encoder不过限，并公开量化余量、累计disk与所需I/O/时间/RSS；未解释的增长项不能只留“似乎够用”。若失败是工程阻断，保留证据并提同owner最小表示/布局候选、独立复核与fresh gate，不删历史、不降N、不填cost0、不提高门限或扩预算。容量probe通过也不能证明与240Hz真实worker并行时100ms门/公平性通过；本轮不启动任何飞行/模型压力来测它。

## 5. 必须刷新的最后边界

本次计划只允许在root明确的pause acknowledgement后进入实证；本审查者没有自行暂停或执行。order18前的首次probe是部分pilot基线，不是最终baseline。pilot完成、所有known retry/settle闭合后，以及最终source/registry/round-policy/dataset/execution/metadata布局冻结后，须再取fresh owner pair/refs、actual ledger/missing-cost项、R文档与各reader载荷/磁盘/路径/耗时，重做剩余horizon投影（已完成dev不冒充formal剩余，formal2016不因cache虚减）。任何新的review/source快照/semantic rounds/修复均可能改变基线，旧通过不得静默继承。

本轮没有容量数值实测、没有probe/test执行或运行产物大小读取。只在最后少量归档此报告/绑定；完成后停止R写入，不授formal/main/G4或容量通过。
