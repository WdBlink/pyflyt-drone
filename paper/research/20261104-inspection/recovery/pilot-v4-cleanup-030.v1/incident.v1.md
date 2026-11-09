# 原v4 order030第二次cleanup unknown

continuation supervisor exec99187已实见exit1。原order30/dev-L3-01/B4-M1/repeat1/attempt00物理与模型返回0，但原process-tree为unknown、terminal为unknown_process_cleanup/Y=null/retry=false，未返回有效callback。该视图保留30completed/1failed/77stopped、31attempted；48outer均stopped。无order31预约或attempt01，原024限定恢复不外推。

原plan canonical SHA d30e4a1b92f86b3c91c09741776eee3b541d7e1ecc1d872d0c257fc3a7ef3f3e；terminal SHA3cf398b2674d81c2217b1b823fb765886aacf3bf9426634c74568ae836774831；tree SHA8821c585f5c2fb6477f0cb3391b2bcfe4276160b9dcf9dec096c294511e69f58。owned五PID为81848/81850/81851/81968/81979，remaining_owned=[]，最终unclaimed_new_python_orphans=[86715]。原记录未保存86715完整row/birth/扫描时刻；首次后续ps查询已不存在，不能据PID数值或消失认定其来源。

原实际600s/144000步、0超100ms、maxlag0.009984750002331566s、runtime failure_reasons=[]；冻结rawY=0.3436623024638006、C=0.9939498703543648、S=0.037037037037037035、D=0、H/E=1、SR=0。这是cleanup覆盖前的证据值，不是已接纳SDK Y。原24calls/173756tokens已known结算。

新增只读inspect_original.py实际重核原payload/model/settlement SHA、ZIP CRC、104source/61deps/10inputs、原actor/clock/冻结完整grader/meter/24causal joins，原evidence逐值相同；actual sole-owner known entry及purpose仍精确属于本plan和repeat1。evidence.v1/binding.json SHA71c073c93af9ef4c0cdb25345380b4373e13730800a2687e60b66677dcc83312。后续观察UTC2026-11-07T00:28:48.553240Z，原owned live={}、同uid Python orphans={}、86715不存在；仅为后来安静事实，不倒填原closed，不发送信号。

全任务native累计1153calls/7468127tokens、reserved0、unknown/unresolved/stale=[]；计量known与episode cleanup unknown分开。费用/Codex/主机/能源未计量。本次无新模型/物理/预约/结算或额外attempt；未修改任何原terminal/tree/ZIP/SDK/评分。

独立source-only归属观测审查未确认内部漏清理根因，确认当前归档只保留unknown PID、丢失用于判断的完整row。所提被动close包装仅保留原扫描结果，需单独实现、测试、freeze和审查；不新增扫描、不改变原unknown规则、不授030合格。当前没有030恢复proposal/acceptance、没有observer实现，也没有新continuation或G4权限。继续内部修复；正式实验、图表与论文结果仍待G4。
