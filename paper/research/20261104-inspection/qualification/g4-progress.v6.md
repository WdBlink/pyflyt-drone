# G4开发进度v6

记录日2026-11-05 UTC。G1–G3仍仅在已声明开发包线内通过，G4为needs_work。未生成main/holdout、未写论文结果。原生实际用量保持29calls/162210tokens；本轮后续都是既有本地物理、tokenizer或CPU软件验证，没有新增模型生成。

已完成同一新OOD根B0的事件/clean实际配对，资料见`qualification/event-clean-pair-OOD-L3-B0.v1/`。事件版一次尝试Y=0.7809919967814705；clean首600s最大clock lag=0.22839270900003328，按已知基础设施失败保留，不准入；唯一一次同源同输入重试最大lag=0.07398854093628415，通过并得实际Y=0.7812312312312312。两侧17源/61依赖相同，root/seed/geometry/motion相同；三个原world均经冻结evaluator完整score复算一致。clean按事件原需求日历重评分Y=0.7985358564305932，配对差为-0.01754385964912275。仅n=1开发诊断，不是正式事件效果、独立新root或成功率结论。全部attempt及失败保留，新ID-L3的B0连续飞行资格正在运行。

全八条件外围driver的grader漂移与classic detached timeout缺口已由`review/eight-condition-development-driver-repair-review.v1.md`独立关闭；原失败报告保留。新增延迟runner的CLI、原始消费函数边界和真实独立timer测试经`review/injected-delay-runner-review.v1.md`复核，无本范围未关闭项。注入只允许B4-M1、无消融的0/2/5/10/20秒单因素；原模型soft60仅计算service/原parent compute，额外等待、模型回复/释放物理见证和dispatch年龄分开记录。timer不是PyFlyt或模型性能证据。

内层整集适配的原2P1/3P2及后续两项cache残余均保留在三个版本review中。最新已独立通过29项组合测试并证明exact settled三文件集合、start/return/cleanup语义、source bytes、seed和enqueue守卫；仍发现public clock可比worker request witness超前的合成反例。修复候选再约束0<=public source<=worker_before<=reply<=after及非负有限dispatch sim age，31项episode/cleanup/delay测试通过，最终窄复验待归档。未知进程/用量阻断，不补Y0；known fee与未reconciled episode分开。SubprocessTree仅为出生身份轮询、登记后代清理及未知Python孤儿的保守阻断，不宣称任意setsid/断电证明；本地并发合成测试曾触发未归属新Python孤儿导致一次正例被保守挡住，原失败保留，未绕过守卫，PIDs随后不存在；来源未确认。

长pilot新增`checkpoint.request`集间关卡候选：只在下一集reserve/spawn前等待并写ack，不暂停物理或活模型；用于唯一owner每轮检查点后继续。该新增分支尚待窄复核，新prepare将重新冻结当前源，旧prepare不能充作当前实现通过。完整pilot仍固定原六root、八条件、18classic+90LLM、48outer，B1选择不重选，没有真实108-slot结果。

事件kernel原静态同钟bad→good遮蔽与t^-边界反例已在`review/event-endpoint-development-review.v1.md`列明和定向复验，原6e/924候选保留。新明确化见`event-obligation-scope.v1.md`与`event-endpoint-ties.v1.md`：survey按已有四中心线的固定公开Voronoi责任区关联受影响未满足目标，不以是否发现/FOV缩分母；inspect/observe仅用事件前已交付identity；新job正确card锚创建时；same-ID正增益超过事件前及上一prefix，防坏→恢复旧值造响应。静态逐接收prefix保持完整30s，动态用原十连续格，不足窗口仍删失。33项手算event tests通过，新候选v2仍30event×target行、6eligible，未回改原v1或Y。新scope及实际配对的独立审查尚待，不能据这些候选值放行G4。

固定责任区说明已加入所有模型共享的工具SYSTEM，没有增加私有信息。新版tokenizer-only v4三模型×五种投影15项为3972–5442输入tokens，全部<=8192，actual generation=0；旧v3保留。模型预算、主score/endpoint定义、实际航线、总体及19项交付义务不变，具体操作化在G4前须受审查。

下一步：关闭上述限定修复/语义缺口，保存当前检查点，完成新ID/OOD开发资格与native全链实际模型episode，然后固定八配置pilot，独立验收最终源码/prompt/model/generator/split/retry/资源清单。G4未通过前不运行正式集或holdout，不写论文结果正文。无付费API、算力购买、投稿或对外消息；native host与硬件资本/能耗仍未计量。
