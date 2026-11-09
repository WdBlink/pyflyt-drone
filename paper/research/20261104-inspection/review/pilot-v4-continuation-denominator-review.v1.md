# v4 pilot continuation：SDK分母/缓存重放契约独立审查 v1

结论：**新SDK envelope可以在明确、单槽、source-bound恢复授权与完整重放核验后，成为同一108槽队列的后续视图；不能拼成216槽，也不能修改旧unknown/abort记录。**本轮只读核对旧SDK/源码与必要小回执，不批准late-cleanup实体判断、尚未提供的wrapper实现或实际恢复，不运行cache API、模型、物理、资源/native动作，也不授G4/main/human认证。审查者 `/root/statistics_design_review`，独立上下文、同模型家族。

## 1. 实际旧视图与身份

`logs/inspection-eight-condition-pilot.v4` 的原order SHA256为 `4e9fe5e86c0621bc7cdc2876391498432c594beddc99f26a2fbc7cecd64636cb`，schedule seed2026110641、固定6个dev roots/108 slots；原sampler表master2026110642。独立逐项读取24个order receipts和其SDK completed raw，身份/顺序/schedule SHA/value与原results一致，order receipts精确为000000–000023。没有读取巨大world或重算真实score。

| 旧SDK组 | 固定分母 | completed | operation-failed | stopped | attempted_count |
|---|---:|---:|---:|---:|---:|
| classic | 18 | 3 | 0 | 15 | 3 |
| llm | 90 | 21 | 1 | 68 | 22 |
| root-means | 48 | 0 | 0 | 48 | 0 |

三组execution_status均failed/outcome inconclusive。实际episode合计108=24completed+1failed+83stopped，SDK attempted25不是25个有效callback。`record_matrix`只在全部108个measure返回后才构造outer均值，所以即使首root18槽已返回，旧48均值仍全部stopped；不能从旧包虚构已完成root-mean。

order24为 **dev-L3-01 / B3-M1 / repeat_ordinal2**，实际sampler3179516317；后继order25为dev-L3-01/B4-M2/repeat2。旧slot24 SDK raw是operation-failed/value null；原plan文件SHA `2548b2670207ee623bcd38af39e121b95b2dec0045c861e90cfe9b623ab97c19`，terminal SHA `86f0189e000815abc250df4092bc264bbb7fa79dcca65a032edec88d68c3f6cd`。terminal保留unknown_process_cleanup、Y=null、retry_allowed=false、return0；tree保留status unknown、remaining_owned=[]、unclaimed_new_python_orphans=[26337]。小usage receipt报告25calls/166000tokens，其SHA与terminal settlement绑定一致。physical_validation.failure_reasons=[]与meter字段可读，但本轮未重验全flight/归档/native settlement；全局1045/6778111、无held/unknown是父Agent提供状态，不冒称独立查询。

## 2. SDK行为为何要求一个全新完整视图

已读旧 `record_matrix` 与 provided `ExperimentRecorder`：三个recorder先开；异常使当前槽写operation failed，其余未启动槽写stopped/attempted=false，全grid都保留。SDK明确拒绝覆盖旧results或同raw目录的身份。恢复必须使用新out/新raw/新results，旧abort包永远保留。

新envelope仍走完整原108顺序：orders0–23是24个strict cached callbacks，order24只能是经过单独接受的reconciled cached callback，25–107共83槽才可新flight（15classic+68LLM）。前25个callback写入新SDK raw是一次容器重记/缓存入账，不是新增25次飞行、生成或planned repeat；newrootmeans仍48，独立roots仍6。LLM SDK的seed0/1/2是repeat ordinal，不是模型sampler；不能把缓存改成repeat3或重抽3179516317。

只有新包完整且通过原adapter核验后，才有一个完整的108/18/90/48结果视图。中途再次abort仍是失败/部分视图，不能把前后若干成功行自行拼成完成。root内原repeat全部纳入算术平均，Y=0也保留；不best-of、不按高分选包、不重选B1。

## 3. 必要的one-slot恢复条件（当前尚未逐项验收）

1. **预先声明succession规则。** 不可变continuation manifest绑定原queue/order、6roots、完整condition/repeat grid、sampler表、reservation prefix、suite/input/selection、原scientific/evaluator及104 source/61dep/10input清单，同时指出新out和旧abort hashes。新包是同queue的successor，不是新研究cohort；authority按恢复登记/完成核验决定，不按数值优劣或完成行数挑选。
2. **保留unknown历史。** 原terminal/tree/process-return、SDK failed raw和native usage/archive/settle bytes均不覆盖、不改status、不填旧Y。当前PID消失只支持当前缺席，不证明原归属/出生身份或原时刻已closed。追加disposition须明确这是开发期单槽恢复判定，绑定原unknown及现时证据/局限、唯一native finding和独立review；不是声称旧API原本认可它，更不能推广到正式数据/所有unknown。
3. **精确API例外。** 普通旧LLM API确实拒绝unknown_process_cleanup，不能直接靠cache_reused=true越过它。wrapper只对已绑定slot024、精确directory、canonical plan+原plan文件、case/condition/repeat/sampler、reservation `dev-all8-pilot-v4-024-a0`、terminal/archive/usage/settle与处置finding等全匹配时返回已独立重验的原score。未匹配或第二个unknown继续按旧API失败；不得宽泛catch UnresolvedEpisode、只看PID不在、只读world.score或SDK旧value来放行。
4. **wrapper是显式新执行来源。** 新wrapper源SHA和执行入口/API hook应被新增continuation binding固定并在cache/后续effect/结束前核验；旧104没有自动覆盖wrapper行为。旧pilot/module、SDK13和全部child生产来源保持原字节；新源码作用仅是前瞻host恢复适配，不伪装成旧源码没有变化的普通重跑。新SDK设计/日志时间可表示容器重放时间，但须链接原prospective design/order时间，不把早先实际flight重新标成此后生成。
5. **first25零新增资源/飞行。** 普通cache逐项走旧真实receipt/source/input/evaluator/native-meter复核；late slot需专门的有效追加证明。cache失败即阻断，不fallback spawn/reserve/retry。wrapper/driver应在order25首次新effect前完成prefix审计：25个缓存对应原24+1、原native reservation/usage未新增未重复settle、原physical/model attempt目录数未增加；记录明确零新flight/model/reservation，而不是声称SDK重写也零I/O。原classic CLI缓存验证可有软件进程/日志，不等于新物理飞行。
6. **全部identity不漂移。** 顺序、case/suite、actor/model/profile、source/deps、plan/evaluator、sampler和旧reservation prefix不变；order25之后继续原两attempt cap，attempt编号按原receipt store，不因换envelope重新给已有槽两次机会。若后来还有未决故障，原规则阻断，不能复用本单槽豁免。
7. **下游明确认识追加证明。** 新slot24 completed SDK raw须绑定原unknown terminal SHA、disposition/native/独立review与真实原score/资源，标reconciled-cache。只懂旧generic completed receipt的consumer仍会拒绝该unknown；没有显式one-slot桥接验收前，完整SDK本身不证明该槽资格，不得偷偷把原receipt当completed。

原unknown仍为unknown历史状态，“effective admission via late disposition”是另一条有来源/时间的判断。过程unknown与native usage unknown是不同维度：meter known/held0不能自行关闭cleanup。此例外不能改25calls/166000tokens，也不能因Y好坏决定是否恢复。

## 4. 两视图的权威索引与防双计

建议保留一个不可变queue-level continuation index：original_failed_envelope_ref、original order/sampler/input/source hashes、continuation source/disposition/finding/review refs、new envelope ref与状态、每canonical slot的原episode/attempt/reservation和新SDK raw映射。原24 cached值须与旧callback一致；slot24单独注明旧SDK value=null、追加admission后给原物理score；后83提供新closed receipt。每个canonical(slot, actual attempt, reservation)在资源/飞行汇总只出现一次。

最终报告把旧包称“同queue的失败执行视图”，新完成包称“同queue的权威后续视图”，引用二者而不相加。不能写216新episode/12新root/96独立outer样本；24+25两个缓存前缀也不是49个独立测量。新包完成时pilot分母仍108、重复矩阵不变，实际attempt/资源另按唯一原回执计数；原中断与特殊恢复在failure/operational记录中保留。

## 5. 检查与限制

本轮只做源码与小JSON/原SDK行的标准库读取/字段核对；逐项24callback对应成立，原分母与slot24小回执如上。focused核对旧pilot、recording adapter、provided experiments及LLM supervisor四份当前源码SHA均等于旧source-hashes里的值，清单有104项；没有重hash全104/61/10，也没有执行这些API。其余冻结清单必须由未来实际guard核验。完整Y重算、late PID/cleanup认定、新wrapper、独立finding是否真的已接受、actual resume均未在本轮认证。

没有新model/physics/resource/native动作、无生产/.paper/原SDK编辑，也无临时新SDK实验。全部旧字节保留，报告/binding归档后停止R写入。本结论只是使同队列续包不双计的必要条件，不是已恢复成功或G4/main批准。
