# G4 开发继续点 v65

G4仍needs_work。同一v4队列93/108有效回调，五个完整root，第六个dev-L2-01已3/18。continuation.v3实际于2026-11-07T14:08:34.810554+00:00在order93/B3-M1预约前暂停。新增90–92均原attempt00、600s/144000步、无超100ms tick、runtime errors=[]。最新三条原清理closed；前瞻规则在本批未新命中。

- 90 B2/repeat0：max lag 10.366ms，Y=0.39086516442246033，C=0.9142621599340478、S=0.25、D=0.008333333333333333、H/E=1/1、SR=0；工具审计/不同accepted ID=38/3；无LLM操作。
- 91 B3-M3/repeat2：max lag 9.845ms，Y=0.3314097279472383，C=0.9942291838417148、S=0、D=0、H/E=1/1、SR=0；工具审计/不同accepted ID=9/6；15calls/107863tokens。
- 92 B4-M1/repeat2：max lag 9.637ms，Y=0.3314097279472383，C=0.9942291838417148、S=0、D=0、H/E=1/1、SR=0；工具审计/不同accepted ID=5/5；16calls/114192tokens。

封存 `qualification/pilot-v4-orders-090-092.v1/binding.json` SHA 36651fa0274f040cb56b0f89c4ba55d3a4f7373d4b752a1118e23accbe410bcb。唯一owner累计known2312calls/14957090tokens，新增31calls/222055tokens，reserved=0、unknown/unresolved/stale=[]。费用、主机、能源和Codex未计量。原104source/61deps/10inputs、顺序/种子/108分母不变；旧SDK中止和历史原unknown保留。

记录后从order93继续剩余15原槽。部分第六root不作整体比较，SDK未正式结束。formal44source/27tests未安装；实际clean/20s、最终冻结/容量/完整图与独立G4仍待。RA-L、budget={}、19义务与G1–G3仿真/human_review_pending保持，无正式数据、图表或论文结果。
