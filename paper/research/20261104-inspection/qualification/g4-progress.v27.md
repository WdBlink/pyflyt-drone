# G4 开发继续点 v27

G4仍needs_work。原19义务、RA-L、budget={}、100ms时钟门、600s/144000步、两attempt上限、B1原选择、评分与统计不变。正式main/holdout的实际namespace、seed/order及dataset尚未生成；没有正式结果或论文结果正文。既有v2/v3失败与未启动分母继续保留。

v4完整开发pilot现有3/108个实际有效callback、0完整root、尚无完整48 root means。三槽都只有attempt00、return0/terminal completed、600s/144000步，maxlag分别为0.047306791666528625s（B4-M2/repeat1）、0.08571254166625408s（B3-M2/repeat0）、0.010033665999799268s（B2/repeat0），均为dev-L1-01；P95分别0.0028497083332013062、0.0031797090003351514、0.003515749667229784s，超过100ms的ticks均0，physics_runtime.errors=[]。这是局部生产验收，不是完整独立root、正式方法比较或硬实时保证；普通world没有保存完整72000 clock-phase向量。

UTC2026-11-06T16:01:22.587671已在order3/B4-M1预约与spawn前确认between-episode barrier。新prefix绑定 `qualification/pilot-v4-first-three.v1/binding.json`，SHA256 `a7baea6198ed68a31062947fc35be42355b6b4bcc58ca7e8f757764b508c1e69`；24个原始计划/terminal/run/world/private/progress成员的33,142,692字节ZIP逐成员比对，并保存immutable的当时stdout/pause/native-status副本。两次LLM的原始resource archive与settlement hashes也核对；不会用后续append的pause日志作immutable证据。归档初查曾因LLM/classic终端字段名不同而停止；最终按各自physical_validation/evidence核run/world SHA，未改变实际记录。

新B3-M2使用29calls160422tokens已native known；当前26已知资源操作累计584calls3633384tokens，reserved0、unknown/unresolved=[]。B2没有模型用量。cost/host/Codex/capital/energy仍未计量，不能称费用0或免费。完整status观测已封存在上述prefix；本检查点后保持同一owner，不新建账本或scheduler。

staged formal候选仍未安装。最终393项合并软件检查通过（63.95s）的原证据沿用；新的限定独立审查 `review/formal-secondary-runtime-consumer-review.v1.md`（SHA256 `5686330f523c34356a6e9c1d6a6220b3592a71b37347bff1eddf120d14f51ac9`），绑定 `review/evidence/formal-secondary-runtime-consumer-v1/binding.json`（`e335077026fd5055664e38895e64abf03dc34ae325ea2634246eecb58e2bdcd0`），未发现该组件的新实质阻断。独立29测试20.74s与8个实际CLI/Fake参数探针均在order2 barrier内完成；这些是软件/合成fixture结果，不是真实formal flight、namespace、独立roots或G4接受。实际clean216/ablation108/latency180消费者、原parent/seed、common-calendar与共享runtime门的有限审查范围见报告，不扩大为whole-graph或human认证。

十四条文献的机械检查现须明确区分：`citation-check.core-and-backward.v1.json`实际fail，不能因CLI exit0说pass；其缺少新三条candidate的evidence_spans list。仅补既有支持片段的schema后，v2实际pass/issues=[]（SHA256 `92e141ed7b508ee6159edbca630e0338b6ac1faddd3eeb75c269e2df4a9b7740`），仍只管record integrity。新增TypeFly/CognitiveDrone/RaceVLA的独立有限核对已冻结在 `review/backward-three-literature-review.v1.md`（`ec62d2fb1f1eecdbc69b4006248699cd9b9df82e38ba282ec4437bdf227618f5`），绑定 `review/evidence/backward-three-literature-v1/binding.json`（`62b83d37def54269ded9693d38ecc5262acc672d7cf5a2c8584c4f78fb969330`）。身份、comma作者、版本、bib可逆修补和有限机制支持无本范围阻断；TypeFly/ChatFly名称差异、work-level引用边和不穷尽性保留。原packet/catalog v3不改字节，以免失去精确审查绑定；其中await-review文字是当时历史状态，由本新审查补充。

统计依据新取得Hoeffding1963原扫描与Crossref并实际视觉读印刷页14–16；Holm1979只定向读期刊原文转写镜像的§2/印刷页66–67，原PDF未取得，OCR排版限制明确。`literature/statistical-primary-reading.v1.md`记录按独立root有界配对差自行推导、六项Holm与Bonferroni区间区别、低功效及跨root相关假设限制；尚未独立审查，也未进入已核验14引用。两次provided fetch实际中途404/403，前两成功源保留，失败stdout/实际读PNG绑定于 `literature/acquisition/statistical-primary-reading.v1/binding.json`（`c2188f72e853cd19138cbef47ba2238d6387dcace01be4ad15ed1652d4dc9da1`）。没有改冻结公式/样本数或将seed独立等同输出独立。

本检查点所有其他R写者及集中checks已停；主exec4162等待已有checkpoint.request。记录native round后立即恢复v4，不改其104sources/61deps/10inputs。剩余仍是完整108开发槽、clean+20s delay真实集成、最终source/model/prompt/generator/split/retry/hardware冻结及独立G4接受，再进入正式实验与论文写作。
