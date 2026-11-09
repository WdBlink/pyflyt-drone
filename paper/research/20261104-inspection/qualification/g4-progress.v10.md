# G4开发进度v10

2026-11-06 UTC：v2完整开发pilot已经实际运行，G4仍needs_work，无main/holdout实例或结果正文。完整设计为原六root108内层（18classic/90LLM）及48outer；当前完成order0 B3-M3/repeat0和order1 B3-M2/repeat2，均属dev-L1-00，不能计作两个独立root或已完成root mean。SDK results仍只在finish/exit写，当前只保存真实raw与prefix，不造部分SDK结果。证据`qualification/eight-condition-pilot-prefix.v2/binding.json`。

order0第一次600s/144000步的M3尝试maxlag=.11925233295187354>.1，terminal=known_operational_failure，其Y不纳入有效比较。最大见证为t406采样步，step work=.11834158294368535；尚无已证根因。原轨迹及23calls/172552tokens保留并正常结算。唯一允许的基础设施重试完成600s/144000步、maxlag=.055235750041902065；C=.9940374787052811/S=0/D=None/H=E=1/Y=.49701873935264057/SR=0，22calls/163426tokens。不按策略低分再次重跑。

order1 M2一次完成600s/144000步、maxlag=.021025541673111547；同一C/S/D/H/E/Y/SR，30calls/165651tokens。三次实际尝试对应两个有效callback。连同旧诊断和smoke，唯一owner累计157calls/976491tokens、9operation全known，无held/unknown。保留全部旧协议代、原B1选择和新v2冻结63源/61依赖及原order/sampler，不做基于分数的source/prompt修订。native host、资本与能耗仍未计量，无付费API或新增算力购买。

00:53:01Z收到order2 B3-M1前fresh ack；该槽尚未reserve/spawn。当前flag仅控制集间门，不暂停任何活模型或飞行。因当前实际clock见证，已按benchmark-checklist登记`planning/gc-pause-localization.v1.json`并在无在途操作时启动一次instrumented non-realtime execution replay。仅定位GC与slow step重叠；预存原始JSON堆冻结、加计时会改变profiling条件，不是生产配置或实际1x资格，不给speedup/过去因果归因/配置选择结论。helper为新诊断文件，冻结pilot63源未改，日志暂在logs之外的临时输出/新logs诊断目录，checkpoint后再绑定实际结果。

长跑新增按driver PID74542生命周期自动释放的caffeinate -i -w空闲休眠断言，记录于pilot/awake-assertion.json；不把此前118ms问题归因于休眠。当前CPU18cores、profiling前load约3.8，常规四thread变量保持1。硬件/执行条件将在最终G4清单披露。

同级clean参考对的实际证据已独立闭合：`review/same-level-reference-pair-actual-evidence-review.v1.md`核37项binding、20个archive成员/17源61依赖、两次frozen fullscore及common-calendar精确重算。两次真实执行replay各25字段和98+125RPC精确相同。保留事件SR0、clean SR1及n1边界；这对没有handoff调用。退休同钟取消后旧已采包送达只支持具体371.5→372s旧report路由正例，不能扩为任意actor cancel或handoff保障。原始pair和旧clean L3→L2代都保留，不回改。

文献记录已扩为11条固定primary版本并完成7+2+2独立有限审查，`citation-check.core-and-mechanisms.v1.json`机械pass；三份review分别first-neighbor-citation-records、smart-gradnav-mechanism-citation、core-method-citation。明确原ReAct与bounded B3适配、PyFlyt祖先论文与当前包线、arXiv首版/更新年及OpenReview403不升级conference。此非穷尽文献或新颖性批准，未开始正文。

`review/formal-phase-admission-interface-audit.v1.md`定位正式阶段8组必要触点：当前生成、reader、ScenarioMission、worker、经典/LLM episode、service及离线receipt均需同一原生source-bound G4/namespace/holdout opening链。不能仅改dev标签；现有SDK matrix/schedule可复用。accepted v3的72main/12ID-holdout身份及1296/216分母不变，未抽正式seed/order/instance；实现将前瞻开发候选且不热改正在运行的63源。仍须完成完整108pilot、已预登记实际B4-M1/20s等待与clean reader组合工程集成、最终source/phase/generator/split/retry/model/prompt/hardware冻结和独立G4审查，之后才可正式生成。
