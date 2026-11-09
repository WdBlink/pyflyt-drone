# G4开发进度v8

v1 pilot已于2026-11-05T23:45Z在fresh集间ack后、order2 reserve/spawn之前主动结束，SDK exit130保存全分母。经典18槽未开始；LLM90槽中2完成/1回调操作中断/87未开始；outer48未开始。SDK attempted=3是进入callback数量，实际模型/物理episode仅2。完整SDK归档与原因见`qualification/eight-condition-pilot-terminated.v1/termination.json`。这是开发协议修复，任务保持active；不是用户停止，不把策略低分作为基础设施重试或清除旧结果。

source-only独立审查确认3项P2公开I/O说明缺口：首次report ID及持久关联、有效证据对应规则、observe自动上传/封存和全局窗口。修复仅公开既有合同，不改变隐藏truth、grader或策略，不断言缺口导致当前低分。旧2集score/receipt不可变；累计82calls/474862tokens均known，无held/unknown，native host/资本/能耗未计量。

独立事件审查还限定旧B0 clean对照的公开L3→L2标签差异；正式干预必须保留父根L3初始公开标签，只移除9个事件字段，并显式验证父子关系。此修复与公共说明补齐在新pilotv2源冻结前执行，旧对照保留，不扩称LLM纯事件资格。

G1–G3在既有声明包线内qualified；G4仍needs_work。新ID-L3与OOD-L3 B0各完成600s/144000步，完整旧六root经典18槽及前述两模型飞行均保留，但不能替代新版本108槽八配置pilot。最近cache时钟、event scope/pair、checkpoint boundary限定独立审查已完成；新公共合同和clean干预需定向验证及独立审查。未生成main/holdout或撰写结果正文。19项图表/章节义务、八配置、样本和评分协议不变。
