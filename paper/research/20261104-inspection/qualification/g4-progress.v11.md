# G4开发进度v11

2026-11-06 UTC。G1–G3保持已声明模拟包线内的资格；G4仍needs_work，未生成main/holdout seed、order或instance，未写结果正文。

固定v2开发pilot当前8/108个有效callback，均为dev-L1-00，0个完整独立root均值；另有原order0一次clock超限尝试，不纳入有效比较，原日志和唯一infra retry保留。order0/1/2/3/4/6/7的有效Y均.49701873935264057，order5 B2为.9829642248722317。这只是同一L1开发root的前缀，不能推断总体方法排名、选方法或调prompt。完整SDK只在finish/exit输出，不造部分SDK结果。原case/schedule/sampler/B1选择不变。

新fresh between-episode ack为order8 B1前2026-11-06T02:04:42.994247Z，晚于本轮checkpoint.request。root-owned driver PID74542/birth08:22:04仍在，限定Python进程清单只见driver，无在途模型/flight；原生lock/load检查无running/uncertain attempt，14个资源操作全known，286calls/1813213tokens，reserved=0、unresolved/unknown为空。cost和native host/资本/能耗未计量，不填0。来源prefix.v3包含8个真实raw、8个callback receipt、原先验表、fresh ack和点时usage summary。63源/61依赖/10输入全部现SHA匹配，未热改。

非LLM同级event/clean参考对的600s/144k、两次冻结评分、25字段与98/125RPC实际执行重放已完成并经独立审查，范围与局限沿用v10/pair-final；不把n1当总体资格。first-three-v2-model-toolchain-review.v1独立连接73requests/194commands，未发现合法命令被隐藏改写/丢弃。M1多次source-time合法duration在真实异步推理后过期，是实际时间预算失配，不是固定duration600；M2含捏造ID，M3曾inspect成功但report证据不合评分条件。此为具体开发轨迹诊断，不给整体方法判断。

一次GC instrumented非实时replay已结束：72000 control steps/82RPC/25字段exact，最大in-step GC .015774333s；未复现原118ms停顿。t0最大step .894931542s包含replay冷启动。输入堆freeze和callbacks改变条件，原因仍inconclusive，不改GC或选配置，不按低分追加策略重跑。证据qualification/gc-localization.v1。

正式阶段8触点候选仍未安装。G4Proof v1独立审查发现3P1/P2/P3；v2的40测试和独立反例复核已关闭旧五项，但新增accepted evaluator链P1保持：不能只让后来qualified manifest自洽地认可新grader。v3增加accepted design.evaluator实际SHA和固定17成员逐hash检查，42测试通过，独立复核待完成；G4Proof本身仍不是namespace/dataset/holdout权限，其他生成/执行/service/receipt接线尚未完成。

唯一owner的增长问题已独立定位：每轮永久再嵌一份约6MiB documents map，当前pair约71.6/74MiB接近128MiB reader长期容量。review/native-document-snapshot-growth-audit.v1.md给出future-only不可变ref、旧inline兼容、fresh gate、恢复/export和CLI闭包规格。root在隔离runtime/autoresearch-document-snapshots-v2开发候选，没有迁移或修改pilot冻结13个runtime lib。新增容量疑点是record双存逐渐增长的budget missing-cost map，正在纯合成审计；不能用cost0掩盖。候选future operation result refs仍未经独立审查，当前原owner/CLI继续唯一有效。现有snapshot新测试尚有失败需修复、完整集成测试失败需定位，不给修复完成或迁移GO。

后续仍须固定108pilot完成、正式phase/generator/admission源码完善并独立审查、已预登记B4-M1同级clean+20s实际工程集成、最终source/model/prompt/split/retry/hardware冻结和G4独立接受，之后才开始正式实验。预算{}与全部19项交付义务、72main/12ID-holdout及1296/216分母、附加事件/消融/时延实验保持。
