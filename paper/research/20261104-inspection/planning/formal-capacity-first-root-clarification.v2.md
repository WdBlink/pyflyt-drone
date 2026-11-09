# 首root容量估算更正 v2

独立只读复核指出 v1 `inventory-and-shape.json` / `inventory-probe-source.py` 的两个估算标签不能成立。保留v1字节与此前round46原记录；本文件更正解释，不重跑probe、不修改实际owner或飞行。

1. `conservative_transport_stdout_and_trace_bytes=3804408032` 将一个peak budget-settle模板的wire+trace尺寸乘以6912+2000。status还有stage/stale/approval/attempts/delivery/checkpoint/user_questions，其他命令也不是该模板。因此这只是“同一结算模板重复8912份”的算术场景，不能称计划transport的保守上界，也不能声称已经逐命令算过总量。
2. `compact_result_bytes=167885` 是未封装的settlement result payload，不是实际`write_result`持久化对象。实际writer还封装schema/result。`conservative_unique_result_object_bytes=1496191120` 既漏封装、又把不经record生成operation-result对象的status计入，不是实际unique object累计量或其上界。

单模板pretty wire251916、CLI trace174970及compact payload167885bytes仍是v1合成形状的测量；本更正不将这些数升级成actual wrapped object大小。旧13000ops/5000attempt/2000round的live-pair模板投影是另一项、仍只在其声明形状下成立。closed L1原件/ZIP inventory、实际free-space和sample-based2016/4032乘积未被这两项问题推翻，但它们仍不是L3/失败或最终导出峰值的上界。

费用缺失仍保留，不填0；没有真实resource operation、usage、namespace或实验分母变动。v33已保留needs_work/非worst-case限制，但其中“额外peak wire/trace与unique-result对象模板估算”必须按本更正理解，不能用于最终capacity GO。

最终容量检查须分别构造真实reserve/settle/status/round/其他所用命令的encoder载荷，区分unwrapped payload、wrapped unique result对象、stdout和trace；status不生成result对象。还须连同future doc maps、实际L3/失败档案、分析与导出并存峰值给出余量。当前这项更正只撤回过强标签，不补发容量通过。独立发现的原报告与binding另行保留于 `review/formal-capacity-first-root-review.v1.md` 及其evidence目录。
