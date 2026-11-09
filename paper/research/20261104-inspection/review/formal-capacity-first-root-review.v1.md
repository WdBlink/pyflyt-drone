# 首root部分容量测量：独立只读审查 v1

结论：**v1的模板算术与分母可核对，但两项“conservative/actual unique object”解释不成立；root追加的v2 clarification已正确撤回它们。**这是有限解释更正，不是新增实测或最终容量通过。其余L1库存、pair模板与free-space仍是明确部分场景，不能授G4/main、完整formal或human认证。审查者 `/root/statistics_design_review`，独立上下文、同模型家族。

Subject `qualification/formal-capacity-first-root.v1/binding.json` SHA256 `edd4e53deecc01e4b8c5a99152a4b11f8c043be94a66d283d702acf67d645a6d`。只读取两JSON、两probe source、g4-progress.v33及后续两份小clarification与必要owner source。四个小成员SHA/大小与binding相符。**没有读取、hash或解压offline-golden-pair.zip，没有读取actual.paper大state，没有运行原/新probe、生产import、tests/native/du/模型/物理或大树扫描。**因此ZIP CRC/解压SHA、真实owner pair和status耗时仅是所提供证据的自述，未冒充本次独立复验。

## 1. 分母与场景口径

2016是main1296+holdout216+clean216+ablation108+latency180的新episode；其中1728LLM、288classic。最多2attempt时4032是所有attempt，而3456仅LLM resource attempts；每个LLM attempt一次reserve+settle，故6912 native operations。单个request/call不是新reservation。

另一projection的5000是人为追加的LLM ledger-entry数量，不是新的研究N，也不是把4032 total attempts重命名为LLM。13000由10000 reserve/settle+2000 round_apply+1000 other构成，与5000×2相符。2000 round envelopes是这个旧表示模板的假设，不能保证未来实际round数量或每round载荷；inventory中的2000 status/other wire又是**另一个**次数假设，不能因数值相同就当相同命令集合。

两个产物的base file/semantic SHA一致。projection保存base compact62,542,961、control128,056,961、checkpoint128,057,047、界134,217,728，最大pair余量6,160,681 bytes，算术正确；但只是最大已观测usage-entry模板3386bytes和近期round-envelope模板18469bytes被复制。script没有构造未来真实doc objects、有效unique receipt链或pending-round峰值，也未save/load/export该合成状态；共享Python envelope仍会被JSON重复编码，不能借此称实际state承载已验证。within_bound只对当前声明的serialized形状成立。

## 2. C1：settle模板不是status/other transport上界

inventory-probe-source.py:67–88仅构造peak `{'status':'known','reservation':...,'budget':budget}`。251916-byte pretty wire与174970-byte trace相加，乘6912+2000=8912，确得3,804,408,032；乘法正确，**上界解释没有依据**。

实际CLI status还包含stage/stale/approval/attempts/delivery/checkpoint/user_questions等字段；其他命令各有不同result，并非settle模板。没有逐命令编码这些字段，也没有证明它们被settle payload支配；因此不能用`conservative_transport_stdout_and_trace_bytes`命名为计划transport上界。最小更正是保留原数值并改称“settle-template repeated8912 scenario”，或在最后preflight分别编码真实reserve/settle/status/round/other完整result及各trace wrapper。不能用一次status的耗时替代峰值status的字节证明。

## 3. C2：unwrapped payload不是持久化对象/累计unique footprint

167885-byte compact计的是裸settle result；owner `write_result`还编码 `{'schema':RESULT_FORMAT,'result':result}` 加换行。两者不是同一单对象量，当前below_single_object_limit只验证该裸payload远小于16MiB，不能直接宣称测到了wrapped对象。它也不证明round-doc map、issued request、SDK JSON或index都小于各自边界。

1,496,191,120=167885×8912的乘法也正确，但漏wrapper，并将不经record的status算作operation-result对象；“other”命令是否record及结果是否唯一也未逐项定义。它既不是实际累计unique对象大小，也不具已证明的上界方向。应分开payload、wrapped object、stdout、trace及真正recorded operation数，unique去重按真实object identity统计；不把status transport当新增ledger/result对象。

root追加 `planning/formal-capacity-first-root-clarification.v2.json` SHA `b600f3b5c27bfa2babc63fd60c6d68f4cfb52b4ec7f79f844fdb323bc6e28904` 与对应MD SHA `1e86489977d862e89510ad2c909b2ee3a1dce664e703ec22f3b7b3cbd484293a`，准确区分上述三种单模板字节，明确8912纯场景、撤回status/other上界与wrapped/unique解释、保持needs_work。**C1/C2的过强解释已由追加更正解决，v1原字段仍作为历史保留；实际新量尚未补测。**本轮不要求在飞期间重跑或改原测量字节。

## 4. Disk与峰值边界

原件最大27,819,190、model原件3,098,662、ZIP10,567,091合计41,484,943；乘2016/4032分别83,633,645,088/167,267,290,176，算术一致。它是把同一L1最大整套组合复制到所有attempt的情景，2016/4032包括classic，model/settlement ZIP本来只有LLM；不能称逐轨、逐类的实际清单，更不是L2/L3、失败或长回复的worst-case。

当前五组logical inventory路径（episode/model/settlement archive+receipt/.paper/CLI trace）作为所选集合彼此分列，但不代表整个run/所有未来产物。尤其episode originals内已含reserve-owner/settle-owner stdout，ZIP又含其中副本；若把future raw+model+ZIP情景再加“全部未来stdout+trace”估算，要明确替换/扣除已包含stdout，否则存在重叠计数。v33目前分开披露且明确未闭合，因此这些数不能直接相加作free-space结论。

free187,914,485,760bytes是该时刻/文件系统事实自述，不是未来储存额度。logical st_size与APFS物理占用不同，后续golden归档、round-doc/result对象、resource packets、analysis、pending request、export与临时副本/外部增长仍需刷新。state/checkpoint余量、16MiB单JSON/object、累计disk与120s时序是不同gate；任一不能替其他通过。inventory进程RSS505,577,472及1.23s只覆盖该probe；不证明representation脚本或实际save/fsync/recover/export/完整CLI峰值。

## 5. Cost、来源及完成边界

inventory源码明确从模板移交复制，新增reported_usage只填calls/tokens，并assert cost缺失、unreported_fields增加3456、known_usage.cost为空、unknown为空；未把money/energy/capital补0。representation source不调用record/save/reserve/settle；合成5000/13000记录只留在进程中，不能并入实际known883calls/5634314tokens。本轮没有实际state，故只核源码和记录的一致解释，不声称重算真实ledger totals或逐值证明原pair保持。

g4-progress.v33保留首L1/dev/18槽、无main/holdout、非worst-case和final freeze后refresh等限制，overall needs_work边界正确。其“额外peak wire/trace与unique-result对象模板估算”须按已追加v2解释，不可用于最终capacity GO。当前修正不影响、重跑或重标pilot，也不授G4/main。

14项小artifact字段/整数代数核对一致；不是执行容量probe。精确输入、所读源与更正、代数值见binding。完成后停止R写入；未改任何v1产物、生产、owner或pilot。
