# Response/resource reader：R1最窄修复复核 v2

结论：**R1/P1原两条多响应反例已关闭；本次局部delta未见新阻断。**仅认证此变量覆盖修复及必要正控，不重审/放行完整F5、primary/secondary接线、receipt/FSM、formal数据或G4/main，不作human认证。审查者 `/root/statistics_design_review`，独立上下文、同模型家族，无实现职责。v1报告及原异常栈原样保留。

Subject `planning/formal-admission-candidate.v3/response-resource-freeze.v2/binding.json` SHA256 `8441e871ceb1c2b95602eaa45d1b430a6b0bcb5dceeec03dbb360374c13356e0`；reader SHA `d4cf1af0aef3a085ceffe2014cfd088e2ba836b4d80c53b6c6c3793f622b7a8f`；response测试 SHA `a8f7df595ca6dee0a9d5ea4df53982f9ac709f9b2b0e999d9317d23ff162331c`。七成员实际hash/大小相符，除reader与其测试外均与v1相同；协议v1 SHA仍为 `65f11b1d8119bb0c832338d05ee08b813c5a1be9b6d0b0e3c5607f26d42c5e47`。

完整reader AST经request_progress→before、reply_progress→reply、response_progress→after的反向名称映射后与v1精确一致；新函数的before仅在初始receipt_snapshot解构处写入一次，后续始终是闭合snapshot字典。实际读到的变化只将三个进度局部变量独立命名，原文件读取、SHA核验、时钟关系、输出字段及计量算法保留，没有跳过第二条响应或吞异常。v1明确记录的timing/source语义范围沿用，未重新扩大来源审查。

原literal IO fixture的write/fixture两个构造函数AST在两版测试中相同。复放结果：

- **2generated**：返回2条valid-flight响应，admitted ordinals=[1,2]，service时长=[5,4]，usage calls2/tokens205；
- **not_started→generated**：返回2条响应，ordinals=[None,1]，generation时长=[None,4]，usage calls1/tokens103；
- 两者第二条before/reply/after进度均为30/35/55.1，原service文件SHA保留，未再发生129→19的snapshot TypeError。

必要单响应、context decision2/transport1/call1、known retry、declined、final、经典0调用和完整repeat平均正控保持原结果；money/energy仍None，final/not_started不伪造额外wait或生成时长，retry成本不当新增repeat。没有重复原数值、事件或全graph软件套件。

12项独立限定断言通过。实际probe `2026-11-06T20:09:38.391801Z`–`20:09:38.413706Z`，内部0.02174秒，四个线程变量均为1；使用冻结函数AST和明确标准库Artifact/parser适配器、自己的自动清理/tmp，不是实际fresh receipt admission。无pytest/whole-suite、生产包import、formal抽样、model、physics、native/effects或遗留子进程。父Agent提供whole522pass/compile0仅作背景，本轮未复跑，不能替代该反例关闭证据。

全部probe结束；报告/binding归档后停止CPU检查及R写入。未安装候选与完整formal/G4状态没有因本报告自动改变。
