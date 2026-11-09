# G4 开发继续点 v48

G4仍needs_work。同一v4队列44/108有效callback，L1-01、L3-01两个完整root，L2-00已完成8/18。实际continuation.v3 exec46816 于2026-11-07T05:01:09.816515Z确认order44/dev-L2-00/B3-M1/repeat0预约前暂停。新三条为原attempt00，均真实新飞行，无重试。

| order | 配置 | 最大时钟滞后 | Y | 已接受不同报告ID | 实际本地生成 |
| --- | --- | ---: | ---: | ---: | --- |
| 41 | B0 repeat0 | 10.288ms | 0.7932017208510463 | 9 | 无 |
| 42 | B4-M2 repeat1 | 8.943ms | 0.3314065510597302 | 0 | 28calls / 174316tokens |
| 43 | B3-M2 repeat2 | 8.906ms | 0.3314065510597302 | 0 | 30calls / 163797tokens |

新三条均600s/144000步、0个超100ms tick、runtime errors=[]、原return0/completed/cleanup closed，H/E=1、SR=0。两LLM实际sidecar与原tree逐值相同，无观察错误；B0原CLI闭合。全部原评分重核一致，B0新离线冻结评分含C=.9925681255161024/S=1/D=.387037037037037；其report audit 899条，实际不同accepted ID 9，与最终进度一致。M2两槽保留零报告与各自计量，不重跑或改变模型参数。普通clock记录不等于完整逐tick向量。三个同L2片段不能支持总体或配置优劣推断。

封存 `qualification/pilot-v4-orders-041-043.v1/binding.json` SHA62bd2130ea7ccdfcc4f338396bad847bc60682f076edd9246cd8db72117d41bb。实际owner累计known1423calls/9161365tokens，新增58calls/338113tokens均原槽，reserved0、unknown/unresolved/stale=[]；费用/主机/能源/Codex用量未计量。原104source/61deps/10inputs与原分母/顺序/种子不变；原三份unknown及旧SDK中止保留。前瞻外部归属规则本批无实际新命中。

记录本轮后恢复同一supervisor从order44运行剩余64原槽（54LLM、10classic）。formal44source/27tests未安装；完整108、实际clean/20s、最终冻结/容量/组合图与独立G4仍待。RA-L、budget={}、19义务和G1–G3仿真/human_review_pending不变，仍无正式数据、图表或论文结果。
