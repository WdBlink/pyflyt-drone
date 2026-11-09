# G4 开发继续点 v49

G4仍needs_work。同一v4队列47/108有效callback，L1-01、L3-01两个完整root，L2-00已完成11/18。实际continuation.v3 exec46816于2026-11-07T05:34:37.547953Z确认order47/dev-L2-00/B4-M2/repeat0预约前暂停。新增三条均真实原attempt00，未重试。

| order | 配置 | 最大时钟滞后 | Y | 已接受不同报告ID | 实际本地生成 |
| --- | --- | ---: | ---: | ---: | --- |
| 44 | B3-M1 repeat0 | 9.390ms | 0.3314065510597302 | 2 | 24calls / 160022tokens |
| 45 | B4-M1 repeat0 | 9.036ms | 0.35362877328195247 | 2 | 22calls / 158281tokens |
| 46 | B3-M1 repeat1 | 14.788ms | 0.3314065510597302 | 3 | 18calls / 121641tokens |

三条均600s/144000步、0个超100ms tick、runtime errors=[]、原return0/completed/cleanup closed、H/E=1、SR=0。三份真实sidecar与原tree逐值一致，观察无错误。冻结全评分、原因果和计量复核一致；C均.9942196531791907/D均0，S分别0/.06666666666666667/0。接受ID与最终进度相符，工具接受不等于隐藏评分命中；原报告audit分别2/2/3。普通clock记录不等于完整逐tick向量。无总体/配置胜负推断。

封存 `qualification/pilot-v4-orders-044-046.v1/binding.json` SHA28eb8d79162b683b04c20af124e92c1336c39a963d82cbe7c4205243a449bbcd。实际owner累计known1487calls/9601309tokens，本批新增64calls/439944tokens，reserved0、unknown/unresolved/stale=[]。费用、主机、能源和Codex未计量。原104source/61deps/10inputs、顺序/种子/分母未改；原三个unknown和旧SDK中止保持。前瞻规则本批无真实新命中。

记录后继续同一supervisor，order47开始剩余61原槽（51LLM、10classic）。formal44source/27tests未安装；原108、实际clean/20s、最终冻结/容量/完整图和独立G4仍待。RA-L、budget={}、原19义务和G1–G3仿真/human_review_pending保持，无正式数据、图表或论文结果。
