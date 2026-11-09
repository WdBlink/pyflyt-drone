# 精确037追加处置 v2 独立复验

结论为 **accept_exact037_only**。v1两个已实证的引用窗口均已关闭。当前接受绑定精确037历史处置与以下实现，原037仍为 `unknown_process_cleanup`、Y=null、retry=false。新native finding和实际38-cache接续仍须独立核对。

审查人为 `/root/control_review`，同模型家族独立工程上下文。proposal.v2 SHA为 `4f94dbf6ed55fe748368b9d58dd74e7de2cc89c2bbcdc0cb36fa0a104143f238`，disposition源SHA为 `ebfc8a8f2efecf06362f01f078fef4c144aa5dcaecb925351c41d2bcb65cd2b3`，事实包SHA为 `1d69331ce39b41a92d73d39808449439c5761c21b3fa48b9010eb70110379406`。原v1拒绝报告SHA为 `d24d1c057308f12685a7c446449daef1df60e0d0542d9ae554fb9e1319cbf09d`，其源码、提案和实际反例保留。v2提案与v1逐字段比较，唯一变化是绑定的disposition源SHA。原episode、归属、闭合树和原结果字段完全相同。

## 修复复验

原窗口脚本仅迁移候选SHA、提案路径和预期结果。真实 `verify`、真实 `require_native_binding`、真实/tmp文件操作，合成已应用native row与原recount替身用于隔离窗口。两项原反例均拒绝，即owner期间换FACTS并改其原成员，以及单独改写helper源码。额外的隔离prepare负例在相同成员漂移下拒绝，目标提案没有发布。恢复字节后prepare正常发布同一个初始闭合树，verify正常对照通过。缺native applied finding与增加空call目录仍被拒。实际细节在 `window-results.json`。

源码把解析的facts保留在owner窗口之前。闭合树重验后，对初始facts文件、成员、原recount refs、helper及原proposal/review/acceptance检查固定SHA。prepare也保留初始helper引用，并在发布前校验同一组固定ref。修复覆盖本轮可观察窗口。文件系统在返回后的变化仍由后续消费者核验。

## 实际只读检查

本轮实际调用原 `exact037` 重核封存flight的冻结grader、actor、meter和当前只读native owner用途。校验后的Y为 `0.3314065510597302`，22calls/160664tokens、22个causal joins，原证据canonical SHA `19fd7d30bd96189f2516fc0dde1e600c1afcc7549abf04cc0239bcf448c15c00`。原archive CRC、known receipt、case/repeat/attempt和冻结plan匹配，原return为0。独立重算的归属proof与事实包、提案相同，匹配037的两条原始扫描行。该检查复用已封存过滤launchd记录，没有重新调查外部服务。先前归属审查关于boot与过滤日志的边界继续适用。

全部216个事实引用重新核SHA，28个目录/87个文件的原attempt/model/resource树在重核前后相同，精确匹配proposal.original_inputs。actual verify在接受文件尚不存在时拒绝。真实native finding的必要性另由使用真实checker的隔离负例确认。本轮独立JSON为提交给该后续正常native步骤的精确审查证据，不改现行ledger或旧回执。

只审037追加处置。024/030既有处置、原SDK失败分母、前瞻归属规则和runner接线均按各自证据处理。未执行capture、实际prepare、接续、飞行、模型、信号或native写动作。完成后停止R写入。
