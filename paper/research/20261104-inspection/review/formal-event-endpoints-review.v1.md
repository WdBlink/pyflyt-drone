# Formal event endpoints：有限独立实现语义审查 v1

结论：**request changes，三项局部P2未关闭，不能批准该冻结实现。**双路径、输出时刻区分、NA/删失和恢复的主结构与前瞻设计相符，但selector对下述明确非法/不唯一trace仍报告identified/FIRST采用。反例都是小型literal软件输入，不是实际飞行异常或已通过完整receipt准入的数据；本轮不重审SDK/ledger/收集闭包，也未证明上游会排除这些模式。若拟依赖上游排除，须有明确覆盖证明，不能由full tracking计数或原score一致性替代。

Subject `planning/formal-admission-candidate.v3/secondary-analysis-freeze.v1/binding.json` SHA256 `2c224a7c7d8d37e7b083fd8be771895a2385d9dd33a99fd1962ef1ee4d3cec00`；冻结event源 SHA `988c1b25c2c792a8e184fa21f34ac40246d1b1e7805e94efc4dd692ceb9a0af3`；v4契约 SHA `f4be91e78f692698fda07373e7e6d0a693d2dd17f2b0f80c3d57d1f91830aa72`。本轮只使用该v1副本，不拿working修订或后来的fixture修订当输入。纯bootstrap源仍为已审b26803…，只核字节不重复方法审查。精确源、测试和实际调用链hash、脚本及完整结果在binding。

## F1（P2）：重复commit被错误认作唯一采用链

位置：冻结event源 `_adoption_response` 289–331行。实现对每个commit分别查询此前唯一request和response，却没有要求同serial/job/program的commit唯一。将正常fixture的program_committed整条复制一次（serial=1、revision=1、at=11、job=j、相同SHA），结果仍为identified、time=11，未产生trace gap；两个重复marker都成为候选。v4明确要求唯一且有序的request/response/commit，这个义务未落实。

最小修复：先对相关完整audit的serial/commit身份查重并核有序状态，再允许形成采用候选；重复/冲突commit应阻断FIRST结论（trace gap或明确validation error均可），不能只用min掩盖重复。保留正常单commit、缺request、hash mismatch及正常local路径正负控。

## F2（P2）：来源不可能的2Hz本地帧仍identified

位置：332–354行。把正常local fixture的running transition、service_started_at、唯一inspect sensor_audit.t一致改为12.25，其他创建/seed条件保持合法，结果仍identified并输出“actual 2Hz local sampling transition”。实际mission._sample只在physics_steps%120==0、240Hz物理下运行，即原0.5s格；12.25没有该帧。仅有相同自报时间与唯一帧数量不足以支持来源链推断。

最小修复：将采样时间与原相机/physics tick和完整tracking覆盖关联，以预声明的微小浮点容差核对原格点及0–600范围；不将off-grid时间四舍五入改成合法响应，也不重置grid。无法证明时记trace gap，不是policy右删失。保留12.0/12.5等合法格及未上传但可唯一关联的本地采集正控。该反例不表示真实worker产生过off-grid帧。

## F3（P2）：非法bool drone被等值当成合法工具创建见证

位置：`_creation_witness` 240–254行。合法job.drone_id=0、工具其余参数不变，只把call.args.drone_id改False，selector仍identified。实际 `InspectionMission._tool` 371–373行明确拒绝bool drone；对相同args执行实际_tool AST的独立纯cut-point（dummy对象、无mission/flight构造）返回rejected/unknown aircraft。因此Python的False==0将真实工具不可能接受的字段绑定成合法创建见证。

最小修复：对drone/job/version/expected-version等native整数身份先严格拒绝bool/非int，再作值/结构一致性检查；相应token/serial等身份核验不应依赖宽松bool/int等值。合法数值duration的允许类型应依原工具规则，勿扩大为未声明的通用“所有字段必须同Python类型”。本轮仅证明上述明确bool反例，不声称已完成任意hostile全字段审计。

## 已实际核实的正确范围

- 正常program commit可给出control-tick推断、首后续1/240 hook和1/120控制步末端；后来succeeded不抹去早先采用。缺request/hash mismatch保留trace gap；同advance terminal不算采用；错误责任stripe不算服务对象。
- 正常local-acquisition在旧continuation上无需新commit或地面上传；未来seed、非唯一inspect帧保留gap。参数/规则/candidate/region的主要关联结构已读，但F3证明其合法性绑定尚有类型漏洞；没有将accepted字符串等同于独立重放全部工具协议。
- `_event_apply_time`保留nominal与推断apply tick，priority10.001推断到1201/120；camera不虚构mutation。formal wrapper明确actor exact known为unknown；不把系统τ或producer输出说成actor已收到。
- 原dev纯函数AST与生产旧版一致，仅将原analyze函数内的dev validation移到保留的公开dev入口。没有放开legacy dev reader。wrapper先检查parent digest/固定L3/600s/144000/120/240与tracking覆盖，再组装formal记录；本轮wrapper的admission/tracking/score重建是显式Fake，不能替代真实准入证明。
- 实际纯评分函数验证静态good→bad→good必须从后一个good起再完整稳定30s；临近deadline不足30s仍删失。wrapper分开recovery_start与start+30确认，information response保持原观测/正静态修订定义。
- 实际纯grade_dynamic_bins+dynamic_recovery手例保留60个原3s格；event101/appear100的十格恢复见证为原indices2–11、start103、到达确认133.5，不重置grid。rich wrapper据原bins/arrival提取同样见证。
- 原表的pre_event_threshold_met/NA、policy censor、trace unknown和U/H/indicator区分成立；摘要正控同时保留三行全分母、两eligible、一NA以及一gap/一censor，observed-only时间分布无新增p/CI。方法状态依赖eligibility的限制在输出明示，不把target/repeat当独立样本。

这些正确结果不能消除F1–F3，也不是正式job采用时刻或真实任务恢复的测量结果。冻结单写者源+关联audit仍只是条件推断，240Hz tracking不含逐行job/setpoint；程序合法采用不等于任务成功、actor意图或优先级最优。

## 实际执行、版本与边界

selector脚本执行11个literal采用场景（含三项意外identified）、1个实际_tool非法参数cut-point和2个时刻函数，最后运行 `2026-11-06T19:09:54.140242Z`–`19:09:54.143638Z`、内部0.00324s。wrapper/summary/旧AST的12项检查于 `19:11:30.077407Z`–`19:11:30.104070Z` 完成、内部0.02654s，全部通过。四个线程环境变量均为1。脚本使用冻结函数AST、纯评分源码以及明确Fake依赖，无完整formal case、population抽样、生产包import、pytest、模型、物理、native或effects。

父Agent提供的focused v1状态是60pass/1fail，并说明最后dynamic fixture有模块别名问题；后续v2只修fixture加载、source未变。本轮未复跑整套，也未把未获得的v2结果写作通过。新代码尚不能因设计说明或部分正控转为批准。审查者 `/root/statistics_design_review`，独立上下文、同模型家族，无实现职责，不授G4/main、整阶段或human认证。

全部CPU/probe已于19:11:30.104070Z结束，无遗留子进程；报告/binding归档后停止全部R写入。只新增本review两产物，保留旧证据和其他作者编辑。
