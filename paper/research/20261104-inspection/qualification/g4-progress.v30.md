# G4 开发继续点 v30

G4仍needs_work。v4开发pilot已9/108个有效callback、0完整root；没有实际main/holdout namespace、seed/order、dataset或论文结果。原19义务、RA-L、budget={}、600s/144000步、100ms门、两attempt上限、B1原选择、17源grader及主global6均不变。旧失败/未启动分母保留。

新增order6/7/8均为dev-L1-01，分别B3-M3/repeat0、B3-M1/repeat2、B3-M3/repeat1。三者都仅attempt00、completed/return0、600s/144000步、0超100ms、physics_runtime.errors=[]；maxlag依次0.010406791001514648、0.014954500000385451、0.014680457998110796s，P95依次0.004563583333379029、0.004564125332876756、0.004746500000692322s。新增资源19/138779、23/152340、17/126461 calls/tokens均已known；native实际status总686calls4307817tokens，reserved0、unknown/unresolved=[]。cost/host/Codex/capital/energy未计量，不称0费用。

UTC2026-11-06T18:06:36.928071Z已确认order9/B3-M1预约/spawn前barrier，同一exec4162继续等待原checkpoint.request。全部集中软件检查在此后执行，104生产source/61deps/10inputs未改。三槽计划、terminal/run/world/private及resource settlement SHA、当时stdout/pause/native-status的immutable副本见 `qualification/pilot-v4-slots-seven-nine.v1/binding.json`（a6c9cb8f7f6f05deea285740b938ceb6aae079ca6602e2581f75d6ac5793888b）。首个只读整理脚本误以为clock嵌套，KeyError发生在binding发布前；按实际顶层schema修正，原副本未覆写，不是flight/资源重试。普通clock摘要仍不是完整72000-phase向量。

未安装的primary分析候选v5增加八分量投影、full-score canonical SHA以及subject对scores/components/SR的实际artifact引用，原SDK/raw/world内容保留；typed secondary track派生真实resource前缀。v5原15 focused51.02s、whole412110.50s通过，但独立复核确认v4继承的P2：真实L1 grade的D=None会被数值检查误拒，而原Fake fixture错误给了数值D。该绿灯不作为问题闭合证据。v5报告及原反例保存在 `review/formal-primary-analysis-consumer-review.v4.md`。

v6仅修此边界：L1严格保留D=None，L2/3仍要求finite[0,1]，其余binary/finite/Y一致检查不放宽。Fake SDK1296集成现在按真实typed level给出NA，另加六个直接边界例。实际全候选429 passed114.38s/exit0、compileall exit0；其中25个primary分析检查、11个新纯统计检查，都是软件fixture而非正式政策测量。有限独立v5复验用实际ScenarioMission.grade的合成L1对象确认修复，L2/3和150个其他非法边界仍拒绝；原score/raw/完整score SHA保留，未授wholegraph/G4。报告 `review/formal-primary-analysis-consumer-review.v5.md`，binding SHA a550fb9c7119eebd4c6fb6d5a97ac583f127d6ca98ea2cea4fc3cb7bf3332100。

v6五成员freeze SHA22affb24af636e16ef77cc0da756255c4717070e025b424a2bbe023248109808。精确41source/23tests重建映射、旧baseline、v5/v6实际日志与退出证据见 `planning/formal-admission-candidate.v3/primary-analysis-integration.v2/binding.json`（b5fd1b40ff901807860547b36dfd4705a45b3a4c053c7b930edb379855985622）。该快照制作时最后两份独立review尚pending；其后实际完成的限定结论另由对应review/binding及本记录绑定，不能改写快照时间范围。

secondary分析前瞻契约v3修正同刻phase：每轮commands/job在当前mission.step head_events之前；上一轮tail_events与相机帧可早于下一轮同t命令。分别保存τ_nominal、可证明/推断的t_apply_tick、实际因果见证的t_actor_known（缺失unknown）；priority public_events.t是登记时间，不能冒称self.t应用测量。program-commit和local-acquisition双路径、无后续step/同advance terminal拒绝、NA/右删失/trace不可识别分开。D2/D3常数和解释不变。有限独立v3文字复核关闭原phase残余，不是实现批准。

新增未接线纯模块 `inspection_intervention_statistics.py`：固定12个L3 parents，每域6；原repeat先平均，同一root向量携带所有方法/arm/组件与原/common score。NumPy1.26.4、99999、linear、PCG64 SeedSequence([domain_seed,3,salt101/102/103])、canonical顺序和little-endian int64索引SHA按前瞻协议实现；两域等权。clean仅common-calendar event-minus-clean，原始两轨各方法值保留；三消融minus-primary；时延minus本轨fresh-zero，fresh-zero-minus-primary另作稳定性描述。无新p/确认家庭。两成员freeze SHA1fc5217da10fe745406d2aa20a098fe1eca200a53c150874f660704edbcd2be4。独立数值复算以隔离37draw实例检查全部三轨非平行root/组件，并另核六个完整99999×6索引SHA；44项检查（含15非法边界例）通过，不冒称独立复跑全99999输出。其报告为 `review/intervention-statistics-review.v1.md`，binding SHA7e449ff430f78c8e1013b2609fca4f355681168d52bd8471cc0edc7d7b9d3a7e；实际root429检查包含本模块原11项完整配置软件fixture。

该模块尚未接入secondary实际分析consumer或REQUIRED_SOURCES，formal graph全部仍未安装。事件采用/恢复与资源/实际响应时延分析接线仍待，不因设计或纯数值绿灯宣称已完成。native仍experiment（16条限定文献已接受）。在本轮证据记录、所有其他写者结束后恢复同一v4 pilot order9。剩余完整108开发槽、actual clean/20s-delay集成、最终source/model/prompt/generator/split/retry/hardware freeze及独立G4决定；此后才生成正式数据和论文结果。统计区间独立性与小样本限制、仿真/待人类复核边界继续保留。
