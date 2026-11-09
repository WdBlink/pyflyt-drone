# G4 开发继续点 v47

G4仍needs_work。同一v4队列现在41/108有效callback，L1-01、L3-01两个完整root，L2-00五条。真实continuation.v3 exec46816已在2026-11-07T04:28:00.666490Z确认原order41预约/spawn前暂停；下一条是dev-L2-00/B0/repeat0。原38缓存和三个精确历史处置保留，另有三条原新attempt00，非缓存、无重试。

| order | 配置 | 最大时钟滞后 | Y | 已接受不同报告ID | 实际本地生成 |
| --- | --- | ---: | ---: | ---: | --- |
| 38 | B4-M2 repeat2 | 9.953ms | 0.3314065510597302 | 0 | 27calls / 173109tokens |
| 39 | B4-M1 repeat1 | 9.681ms | 0.37585099550417467 | 3 | 24calls / 170079tokens |
| 40 | B1 repeat0 | 10.560ms | 0.5777548398935682 | 9 | 无 |

每条600s/144000步，0个超100ms tick、runtime errors=[]，原return0、completed、清理closed，H/E=1、SR=0。两个LLM实际新sidecar与各自原tree逐值相同且无观察错误；classic CLI原closed。原评分/causal/meter及B1冻结评分再算通过。B1有710条report audit，但不同accepted ID仅9，与最终600s progress相同；其余两条audit为0和3。中途32条模型响应日志不等同第38号的27次实际生成。普通clock摘要不扩称完整逐tick向量。

封存 `qualification/pilot-v4-orders-038-040.v1/binding.json` SHA8aac11cfaf09d80efe68c84ab307f8702ce4dadbeb2fdc17b0e86a653b014cc5，含原三计划/SDK raw/终态/物理/评分、两模型结算和被动侧录、B1离线完整重算、实际pause与native状态。唯一owner累计known1365calls/8823252tokens，reserved0、unknown/unresolved/stale=[]。新增51calls/343188tokens均原LLM槽；费用、主机、能耗与Codex未计量。

新三条均原清理closed，前瞻精确外部作业处置未触发，故此处仅验证了真实同队列运行与正常分支。原三份unknown与旧SDK中止保留，生产104source/61deps/10inputs、原order/sampler/19义务及实验分母不变。两条LLM及一条B1是同一L2开发root的未完整片段，不作总体或方法胜负推断。

本轮证据后恢复同一supervisor，从原order41继续67未启动槽（56LLM、11classic）。formal44source/27tests未安装；原108、实际clean/20s、最终来源/容量/完整图与独立G4仍待。RA-L、budget={}和G1–G3仿真/human_review_pending保持。之后才进入正式数据、图表和论文。
