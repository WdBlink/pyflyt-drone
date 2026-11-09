# G4开发进度v23

G4仍needs_work。G1–G3及同级B0参考对保持原已审范围，单次L2/B4-M1修复后clock成功不能扩称整个模型矩阵合格。实际formal namespace/seed/order/dataset仍未生成，正式实验和论文结果正文未开始。

新独立v3开发pilot实际启动后于首槽dev-L3-00/B3-M2/repeat1耗尽两次clock基础设施尝试而终止，exit1，0/108有效callback，0完成独立root；没有第三次旧槽尝试。两次均完整600s/144000物理步、程序return0、source/deps不变、cleanup closed，但maxlag分别.4054049999685958/.2103543746906098s，大于.1。P95分别.005993125339351479/.005677832701849184s，超限tick129/92；最大见证为sim427.75、当前work.4034566660411656s（sensorFalse），与sim287.5083333333333、当前work.08582545805256814s（sensorFalse）。这两次没有保存完整72k lag/GC/rusage数组，不能唯一归因GC、host、MLX或工具。原raw Y=.3313166234514549均保留，但未接纳为有效测量。

SDK完整classic18、LLM90、rootmeans48分母均failed；尝试计数0/1/0（内层LLM该槽包含两次实际attempt），其余stopped保留。全部170成员SDK和32成员physical archive逐成员SHA与原文件验证，见qualification/eight-condition-pilot-terminated.v3/binding.json；两份原settled model archives/usage/owner ack也各自保留。每次30calls/165378tokens，总60calls/330756tokens。唯一科研账本累计22操作allknown、474calls/3004256tokens，无held/unknown/unresolved；cost/host/Codex/capital/energy未计量。budget={}、本地既有模型/算力授权不变，无付费API或新增采购。

独立launch核验事先确认103source/61deps/10inputs、108slots/6root/48means与原B1 selector精确；所有103在已有postinventory freeze同SHA。当前production全程未改。正式接线仅staged并行研发，后续安装任何delta须独立决定证据复用，不能声称pilot用了新候选。独立primary-v1的63项软件测试日志UTC窗口12:04:48.869440Z–12:05:11.772747Z，确有早期firstflight并发；得知flight后暂停集中测试，retry期间无该类测试。run.started早于epoch，旧flight未保存UTC物理epoch，不伪造精准峰UTC或根因。

primary consumer immutable-v1独立request-changes：PC1 P1缓存忽略更晚attempt并混用Y/world；PC2 P2 readonly创建.lock；PC3 P2 formal可返回development角色；PC4 P2 formal service在gate后取未验证raw control；PC5 P2 bool repeat被当0。63附测+7反例为纯software/Fake，负控另模型receipt被正确拒绝。C1只观察canonical main路径无flags时先建out，不声称实际physics绕过。报告review/formal-primary-consumer-review.v1.md和对应bindings原样保留；其结论不涵盖working修复。

working v2新增read_only不创建目录/lock/plan/attempt，缓存要求canonical<=2且completed为末次，角色来自stored plan拒formal降格；ExecutionPlan携带最后native验证的owner snapshot，正式service与settlement只取该snapshot，purpose canonical/typed匹配；CLI对已知canonical formal路径缺flags在setup前拒绝。相关新增缓存检查为实际synthetic subprocess，绝不是真模型/飞行。新增secondary纯内容/计划适配保留固定12main L3父root，clean保持L3/几何/运动/noise，仅删除四事件；消融108/clean216/latency180额外槽，sampler按父primary root/condition/repeat原值，零delay选择独立新飞行而不作可选复用。全部仅未安装prototype，尤其完整secondary worker/receipt/SDK效果和common-calendar正式读回仍待验证。

root targeted42项通过；fullstaged362项通过（58.21s），compileall通过；候选overlay的79项legacy adapter/计量/延迟软件回归通过（11.90s），先前launcher/fixture失败也保存。快照planning/formal-admission-candidate.v3/secondary-and-primary-repair-checkpoint.v1/binding.json精确限定这些范围。working修复还未独立接受，formal图未完成，不能以362项代替G4或真实数据。

下一步先完成负证据/检查点，然后以单独未计分M2/L3工程诊断保存完整GC/rusage/clock，定位两个实际clock失败；不重开v3旧slot或根据其Y调整方法。继续修复/复验formal消费者与secondary轨道。完整合格新pilot、clean+20s实际集成、最终source/model/prompt/split/retry/hardware冻结及独立G4接受仍待；原19项义务/RA-L/分母/评分/选择器/统计没有更改。
