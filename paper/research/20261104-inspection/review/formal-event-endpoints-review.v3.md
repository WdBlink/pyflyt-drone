# Formal event endpoints：F3-program限定修复复核 v3

结论：**F3-program八个已登记反例在v5均关闭，本轮限定delta未见新阻断。**正常commit/local/refresh/handoff正控保留；沿用未改的F1/F2/creation修复边界，不把本结论升级为完整event协议、receipt/SDK/graph、实际formal数据或G4/main批准。审查者 `/root/statistics_design_review`，独立上下文、同模型家族，无实现职责，不作human认证。前两版报告与所有原反例原样保留。

Subject `planning/formal-admission-candidate.v3/secondary-analysis-freeze.v5/binding.json` SHA256 `2f71d6fee26641c868f54b681d3a35bc705e879150f0e6640844c1857052b4ed`；event源 SHA `d12ca95432645aa05da534303d17dde69e4fd0620b06ead4d31176860f819f59`；formal event测试 SHA `6a6d6362514de67d8625dce2539b266e444d9c12568b19b67ec3a6d83efa4d35`。八成员实际hash/大小与binding一致；与v4相比仅event helper/formal event测试成员改变，其余consumer/REQ/purestats/测试成员字节不变。协议v4 SHA仍为 `f4be91e78f692698fda07373e7e6d0a693d2dd17f2b0f80c3d57d1f91830aa72`。

实际读取完整helper delta，函数AST比较只有 `_adoption_response` 改变：commit.serial/revision/drone以及request.serial/revision/drone、response.serial先要求 `type is int`，不作cast；四对tokens先校验list/tuple结构、native-int非负version和str/None身份，再进入原值/顺序/job绑定。request/response检索改安全get。该修复直接消除了bool/int等值进入链关联和revision算术的原根因；这里不声称已证明任意取值或全部工具协议。

原八个独立单字段反例全部从identified/time11转为 `adoption_not_identifiable_from_trace`、endpoint=None：request/commit.drone=False；request/response/commit.serial=True；request.revision=False；commit.revision=True；自机token version=True。输入JSON值与布尔类型未被修改，不存在强制转换后继续接受。

必要正控：正常commit time11、未上传local time12、合法半秒local12.5、合法handoff、同job不同唯一serial刷新仍保持FIRST=time11、合法typed tuple tokens均identified。代表性request.drone=0.0、自机token version=1.0、空闲机token version=False或job_id为整数均转unknown。原重复commit、off2Hz frame和creation bool drone反例仍unknown，未回退成policy删失。没有重新扩大wrapper/恢复/summary、数值bootstrap、SDK/ledger或收集闭包审查。

全部22项限定检查通过，使用冻结AST及旧版已封存literal fixture，不导入生产包。实际最后probe为 `2026-11-06T19:21:11.287646Z`–`19:21:11.297388Z`，内部0.00960秒，四线程环境变量均为1；无pytest/whole-suite、formal population抽样、模型、物理、native/effects或遗留子进程。完整构造、输出与精确输入SHA保存于binding。

作者v4 whole499通过不是v5 whole结果；本轮没有重跑whole，也不将尚未取得的v5 targeted结果记为通过。compile/作者测试状态不能替代上述独立反例。F4/F5/G4、resource/response诊断及实际接线资格仍pending。全部CPU/probe结束，报告/binding归档后停止全部R写入。
