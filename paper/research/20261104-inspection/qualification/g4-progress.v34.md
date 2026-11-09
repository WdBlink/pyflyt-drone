# G4 开发继续点 v34

G4仍needs_work。v4现21/108有效callback、1个完整L1物理root block；第二个dev-L3-01仅完成3/18槽，不产生方法均值或总体比较。provided SDK最终outer recording尚未完成。原19义务、RA-L、budget={}、八配置/科学分母、600s/144000步、100ms/两attempt、B1选择、17源grader/global6保持；无实际main/holdout namespace/seed/order/dataset/分析锁/论文结果。

UTC2026-11-06T21:25:08.746035Z，同一supervisor exec4162确认order21前barrier。新增order18 B4-M1 repeat0、19 B4-M2 repeat0、20 B3-M3 repeat2均仅attempt00、completed/return0、600s144000步、0超100ms、runtime errors=[]。maxlag依次0.009242707998055266、0.008568666999053676、0.008474333000776824s；P95依次0.002809874667832446、0.003043750332039963、0.0023813330008124467s。原plan/terminal/run/world/private/SDK callback/raw/settlement及当时stdout/pause/status见 `pilot-v4-slots-nineteen-twentyone.v1/binding.json` SHA8250e65dff77e6f67e513bcbb7aeece035518c7fb3b36a5d29bf65a641318862。普通摘要/最大见证不是完整72000-phase向量。

三槽实际模型资源分别24calls172374tokens、27calls175907tokens、22calls159244tokens。唯一owner实际known956calls6141839tokens、reserved0、unknown/unresolved=[]、stale=[]；cost/host/Codex/capital/energy仍未计量。三槽冻结C均0.9939498703543648、S/D均0、H/E均1、Y均0.3313166234514549/SR0，保留策略结局和原分母，不按低分改协议或改为infra retry。

报告计数的单位已定向核对：world.reports是ledger.audit，包含提交/修订/幂等重复；progress.reports是len(ledger.reports)，即占用名额的不同报告。首root和新三槽binding中的legacy `reports_count` 应解释为audit条目数。新增三槽的audit条目为1/0/12，实际accepted不同ID为1/0/7；M3最后一次报告提交为561.3s，没有截止后补报。全21份原world SHA吻合、accepted不同ID数与各自最后600s进度名额一致，所有audit提交在[0,600]；澄清 `pilot-v4-report-count-clarification.v1.json` SHA059fc8ec11776af25db043c8d76d2318ba0d709ffbcbac4f22d8bcf752bd217c。原world/score/SDK/旧binding未改，不把12次调用当12份新报告。没有新评分器或新flight。

首root容量v1的独立只读复核核对14项小artifact字段/整数算术，发现C1/C2解释过强：peak settle模板不能支配status/other wire；裸payload167885bytes未含持久化schema/result封装，status也不生成operation-result对象。追加 `planning/formal-capacity-first-root-clarification.v2.md/.json`（JSON SHAb600f3b5c27bfa2babc63fd60c6d68f4cfb52b4ec7f79f844fdb323bc6e28904）保留原测量字节并撤回计划transport保守上界、actual wrapped object和unique累计量三项解释。8912份单模板乘法仍只是算术场景。原live-pair模板与L1库存/ZIP/free-space未被此更正推翻，仍非最终容量证明。

独立报告 `review/formal-capacity-first-root-review.v1.md` SHA655db749f6fd78343400860826afe1c54c2e166a7e285ec5ffdb10d2b79bcf3a，binding SHA984d4d3fc50d960eca9c6bbe1b61c120b8bbeae5aa665f548660c7409cd75813，确认过强解释已在追加更正范围关闭，没有补测capacity。最终清单须分classic/LLM和互斥路径；原件已含reserve/settle stdout，不能再把全wire无扣除加到原件+ZIP上。最终pending request/doc maps/真实各command wrapper、L3/失败/分析/导出并存峰值与磁盘余量仍待fresh核算。仅在barrier确认offline ZIP解压SHA与临时golden原pair相同后，删除约125MB本次冗余/tmp副本；R内原ZIP、measurement/source/binding完整保留，不删研究历史。

新增SDK容量定向只读检查见 `planning/sdk-recording-capacity-readthrough.v1.md/.json`（JSON SHAe8ed2f9c253f0f539531e1ece6fae6b35718788d32fa70cf91e51f42cf067f5a）：provided recorder把完整evidence逐raw存储，results只汇总短行，因此没有发现main1080份大trace必然串成单个超16MiB summary的故障。当前983byte dev raw不是formal raw上界；候选formal raw另有完整score，最终逐raw/index/输出仍须检查。round47 baseline map实际5068311bytes/27926条，旧30000模板count不是未来文档数上界。未导入生产/NumPy或运行新load test。

运行delta source-only报告定位formal guard的额外负载：worker私有内容/来源验证在预warmup；model service每次真实生成前还重新admit_dispatch，递归执行source/deps/qualification evidence、dataset/audit/tree及native owner lock/load，holdout再依赖locked main。普通dev formal_ref=None不走这条分支，重跑其108也不能代替该负载覆盖。classic在飞没有等价逐generation全面审计；service.wall_seconds不含pre-generation admit，client E2E/watchdog包含它。因此后者是本实验实装管线等待，不是纯LLM推理/可直接外推的部署时延；不事后从clock/Y扣除审计开销。

报告 `review/formal-delta-runtime-qualification-plan.v1.md` SHA7c58e61ac68fe9da1cbf58287676def4ac1fb1b7f8844d919c6d0418ac0d393b、binding SHAe433cfa643d30055dfb033ef68be3bae5499ddbb1e1b218fdd321b9efceda451。其一处将root工程约束误归终端用户的句子，已由追加v2来源勘误纠正：report SHAe2beaf4a459b2361c1c5f4f36055ff16b6b344b2c605d704bc14db67aa6b46f3、binding SHAff746f8bdbbd906da7e0dc56bf8c2758d66e744173f675b48cf6438d1544aec0。旧v1保留；v2只在来源归属上优先，其他结论未重新实验。终端用户要求按计划G1–G4后正式实验，没有新增“G4前先取得真实main/holdout祖先E2E”的硬门。

最终组合证据沿原dev-only G4冻结合同：先完成当前pilot，安装已审delta后完成既定actual clean/20s-delay集成，冻结source/model/prompt/generator/split/retry/hardware，按必要范围记录额外guard负载包络。dev clean模式可复用既有EventCleanSibling，20s入口已在候选；源码可连接不代表已实际运行。没有真实phase祖先时，只读/合成负载演练须明确非授权/非formal，不能造G4或提前draw；真实phase分支留在G4后逐槽观察并遵守原fresh admission/100ms/infra/cap2。报告不给条件GO，也不机械要求无针对性的108重跑。

本轮生产104source/61deps/10inputs未改，formal44source/27tests仍隔离未安装；除原pilot新三槽外，没有新增pytest、诊断模型调用或其他物理运行，既有522检查不扩称新增实证。两独立审查者均已停止R写入。本轮owner记录后恢复同一pilot order21；完整pilot、actualclean/20s、最终容量/来源/组合范围与独立G4决定仍待，之后才进入正式实验、图表与论文。G1–G3既有仿真资格/human_review_pending及小样本/源独立性边界保留。
