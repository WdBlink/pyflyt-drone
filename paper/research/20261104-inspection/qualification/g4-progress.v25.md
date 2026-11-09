# G4 开发继续点 v25

G4 仍为 needs_work。G1–G3 和既有同级 B0 事件/clean 参考对保持原证据范围；旧 v2/v3 pilot 永久保留失败终态，不增加旧槽 attempt。正式 main/holdout 的 namespace、seed、order、数据仍未生成，未开始论文结果正文。

新 runtime 候选单次 M2/L3 工程负载已实际完成：600s、144000 物理步、72000 clock/rusage 行，最大滞后 0.008292792000247573s，P95 0.003075666333472782s，0 ticks 超过原 0.1s 门槛，supervisor OS exit0。runtime 开始与结束均读回请求的 self-thread Mach constraint；结束恢复原 timeshare/默认 RT 状态、GC enabled；audit.errors=[]。窗口 wall600.003532958s；pre-epoch collect32、post-loop collect14。该候选同时改变物理进程的 cyclic-GC 时机和当前线程软实时调度，不是严格单因果 A/B、不提供硬实时保证，也不能替代全 pilot/G4。

完整 106 成员 archive 为 `qualification/clock-runtime-M2-L3.v1/complete-diagnostic-and-model-operation.zip`（SHA256 75629ca2b694f8a763ea46fc3d330420c7cfcbf253e86746e27d7c39e5bf520b，26674021B）；原始回执、阶段重计、freeze 与 actual binding 同目录。独立审阅者已在 UTC2026-11-06T14:30:31.917024 核109 live sources/61 deps/11 inputs及113冻结文件完全匹配，之后仅审阅封存数据；后续有意接线不回写这次 load 的来源身份。该次30calls/165378tokens已获原生 known settlement；当前24资源操作、534calls/3335012tokens，reserved0、unknown/unresolved=[]。cost/host/Codex/capital/energy 未计量；用户 budget={} 和既有本地资源授权不变。

已将候选 helper 逐字接入 `eval/inspection_worker_runtime.py`，共享 reference worker 在 warmup/mission 构建后、原 epoch 前进入窗口，并在原 loop 后及异常 finally 恢复；原 while AST 保持一致。reference/dev/classic pilot/all-eight pilot/LLM runner 的实际 source inventories 同步包含 helper。共享 flight 验收只对包含新 helper 的来源版本要求完整 runtime audit；缺失、demotion、GC 状态错误、恢复错误及 bool/int 混淆反例均拒绝，历史来源维持原范围。旧工程 diagnostic derivatives 留作其封存版本，未改写。本次接线尚待独立审查、真实短烟测和新的完整108槽 pilot。

针对接线与既有 receipt/case/protocol 的115项检查通过（16.13s），compileall通过；初次 Fake mission 缺少真实 loop 会访问的 fixture 属性导致1 fail已保留，修复只调整 Fake 后复验通过。软件检查不是物理 flight。源 delta、6份 before 原文、新 helper 和测试保存在 `planning/physics-runtime-integration.v1/freeze.v1/`，binding SHA256 b57679da7beea68d611ed7eb96c8ff5b5bbc793f055cad302f959f6dd59da9b8。

上一轮 primary consumer 的 PC5-b 嵌套类型残余已在独立最小 delta 复验关闭：18附测+8独立检查，报告 `review/formal-primary-consumer-review.v3.md` SHA256 b278e6a03f2b66c75681701df8b2cd0ee8d63d93c7144da9d9da66d98012c520，绑定 b8926191d4a024856c7bf33effedc34aff6f71166227f34276c0305f37404141。该审查仅覆盖 canonical 比较修补，不能升级全 formal/secondary graph 或 G4。新 formal common-calendar 适配8项 literal fixture 检查通过（2.87s），含手算 S/Y 权重、封存历史不变和身份/分数错误拒绝；它们没有生成真实 formal roots，也不是实际 flight。证据在 `qualification/formal-common-calendar-software.v1/`。secondary 全链及最终全来源接线仍待验证/独立审查，staged graph 未安装。

本轮封存前复核已结束：`review/clock-runtime-M2-L3-actual-review.v1.md`（2f3806255701ecf03894a45acf54165baccf3351141beca883cac53980279f41）和其 binding（dbade59ab594437663d7871c480b9ea482d0bf50f9a876a9b1cc28ef35d2a649）独立核原始72000/144000记录、仅pre-epoch GC、native known与清理；未发现该次载荷证据阻断。ru_maxrss 为 physics-process 累计高水位243712000→250609664，增长6897664B；不声称即时RSS、post-loop释放量、无泄漏或全host内存通过。

`review/runtime-production-integration-review.v1.md`（bcd6bd0369a1ff1d404c8c90c95541068a1baa49d73b4c5c0d23900ee3af0d75）与 bindings（d3aa00cfcf392aac96c65dc1da8cc924fa2395d5e5b134aed1d918c9d7ce72bb）对7源最小接线给有限无明确阻断：17附测+5独立检查，保留未重复native单元的范围；accelerated不构造runtime、失败审计阻断与旧scope不变。集中CPU检查UTC14:42:27.870372前结束，其后才启动真实短烟测。

20s非LLM生产烟测实际CLI exit0、4800物理步、maxlag0.008421249999628344s/P950.0034560409996631236s/0over；runtime.errors=[]、申请/恢复和source/deps/private binding通过。7成员archive与 binding（211edd621ae879702d46d165f1d603cb6f37356957fd111e88359b1b9e8e37d2）在 `qualification/runtime-production-smoke.v1/`；正确只因incomplete_or_wrong_clock拒绝为600s成绩。该max/P95是普通world报告，未保存/重计完整相位向量。新v4 prepare-only已封存104sources/61deps/10inputs、108=18classic+90LLM/48rootmeans，binding59122e64c8dbeb812655ed1792859ca79cd8846f10a8048b91e5365f0c771317；schedule2026110641/sampler2026110642，首槽dev-L1-01/B4-M2/repeat1，未实际dispatch。原selector未重选。磁盘前检191GiB可用，另有host CPU任务正常运行，未终止无关任务。

后续：原生检查点后在新目录/新资源前缀运行完整108槽开发 pilot，保留全部失败/未启动分母。不得修改 pilot 消费的 source/deps/input。使用现有 between-episode barrier 做原生检查点；飞行期间不集中跑本任务 CPU 测试。clean+20s delay 实际集成、最终 G4 freeze/独立接受仍是正式数据的前置门。
