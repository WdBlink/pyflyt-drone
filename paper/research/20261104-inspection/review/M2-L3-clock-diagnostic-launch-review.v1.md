# M2/L3 单次 clock 遥测诊断：launch source review

结论：**未发现一次无分数工程定位诊断的明确 launch 阻断**，限新角色/输出/预约，先走 sole native reserve 再启动。它不填补、不继续或重试已终止 v3 SDK 槽，不创建新科学样本，也不授 G4。此审查未启动 probe、导入实际执行模块、调用模型、physics、RNG 或 native 写动作；只运行 stdlib 文件/SHA/AST/diff 检查，未重跑单元测试。

输入绑定：freeze `qualification/clock-localization-M2-L3.v1/freeze.v1/binding.json` SHA `b6aa5715bf4855db70f5eff2fcc5f1b954951f5722ff27479b1b4d539e87e068`；intent SHA `394d416179cd77c1188762e4875653f23127e837224af29a45f898c8e320a902`；working `planning/clock-localization-M2-L3.v1/probe.py` SHA `d9c26adc44fbcd7cb1e119499fe93568228c64128482497d58e9fb0079b89e3a`。独立核111冻结文件、107 live source、61依赖、11输入全部字节/SHA匹配；107精确等于原pilot103来源与intent的六extra来源之并集（其中cleanup/provider已在103，净增4）。依赖与原pilot相同；11输入精确为原10输入加既存sampler文件，没有draw。新freeze不是生产安装。

相对当前、已审并实际用于前次postinventory harness的 `eval/probe_inspection_adaptive_clock.py`，逐字替换恰好八处后等于新probe完整正文：docstring、从working路径定位ROOT、ROLE、CASE、SAMPLER_SEED、self-source路径、registry M1→M2、child M1/B4→M2/B3。无其他逻辑差异，完整diff已保存。working probe的祖先定位实际得到repo根；应运行intent绑定的working probe，不把仅供证据保留的freeze/source镜像当新runtime。父/子命令与actor/coordinator/service仍是现有入口。

具体绑定为原有 `dev-L3-00.json`（SHA `ae583cf5…`，split=dev）、B3/M2，以及已冻结v3 order0 sampler `1096691202`（master2026110622的既有存储值）。未重标root、未生成formal/holdout实体或seed。M2 registry/download receipt与原冻结SHA相符，generation profile精确等于intent：32calls/180000tokens、8192输入/1024输出、soft60s、temperature.2/top_p.95/thinking=false。新输出 `logs/inspection-clock-localization-M2-L3.v1`、新reservation `dev-clock-adaptive-M2-L3-v1-20261106`；审查时输出、service operation、reserve packet与ledger ID均尚不存在。

资源与运行顺序：`probe.py:34–52,76` 在父/子前门要求live reserved、足够calls/tokens和精确development purpose，包括ROLE、model receipt、case、sampler及六项variant SHA；`:77–85` 显式用 `partial(worker,profile_factory=RusageClockProfile)` 注入现有diagnostic worker，并把M2/B3/原case/seed传给未改的coordinator。共享service仍独立要求accepted plan、active round、registry/profile/reservation，在MLX load和真实request前复查，probe不提供绕过资源门的方法。当前没有实际预约；后续必须用上述精确新purpose登记，不能unreserved settle或手改owner。

`eval/inspection_diagnostic_worker.py`、ClockProfile/RusageClockProfile、原reference worker、原coordinator、control、cleanup/provider的当前SHA均与已审postinventory freeze相同。新增factory只用于instrumented worker；原普通worker未换epoch、pacer、GC、.1s门或控制逻辑。该组源码的原独立审查和单次实际遥测证据可复用为仪器/接口基础，不等于本次新M2/L3已通过。107闭包本轮核的是冻结字节和已声明差异，没有重新通审所有来源或完整MLX/OS依赖。

supervisor `:55–66,95–104` 保留现有SubprocessTree的.1s观察节奏、12000条有界wall/processCPU记录和900s wait界；`:119–139` 保留终止、born-identity清理、inventory diagnostics、overflow、variant/case后验与错误状态。异常不生成成功评分或免费重试。`:106–116` 检查完整600s/144000物理步/72000 clock+rusage，独立实际model meter并记录lag阈值是否超过；状态名仍 `completed_engineering_not_qualification`，即便exit0也不能读成clock门已过或故障已修。若异常导致缺计量，必须保留未知/held直到按实际service journal重计与native reconcile；本harness本身不负责settle，不能把缺resource_recount当0。

`probe.py:97–98` 保留caffeinate `-i -w`，intent已准确声明失败pilot没有此assertion；还有instrumented worker/profile、bounded supervision timing和新诊断source/output差异。因此这是定位用复现条件，不是严格isolated A/B，不可仅凭pass证明某个唯一根因或修复；lag不合格也必须保留。profile完整记录后还需按实际72000行、GC/rusage、计量和清理另审，macOS计数负delta不能clip，counter名不唯一归因。采集与写出本身有开销，默认GC保留不等于零仪器税。

本次无自动retry、SDK record_matrix或receipt admission；新reservation/service目录的exclusive admission和fresh输出约束保留。旧v3两次失败、0/108有效和0root永久维持；已冻结terminal独立报告 SHA `5a4bade47d278a91628215fe9a920a6b42be51b2cc618ddc13c11a4a3732f257`。intent明确等全部自身CPU集中检查结束后再飞行，并禁止在途自身集中测试/checkpoint/生产改源/并行模型物理；该安排是预防混淆，不是过去首flight并发解释的反证，也不证明主机绝对空闲。飞行开始后本审查不再追加本机probe或集中CPU重计。

同模型家族独立工程上下文，只判断这个单次新准备的来源/接线/前门；未知包括实际spawn/forward、完整遥测、600s clock、计量与清理结果。G4、完整新pilot、最终formal安装集成及人类领域审核均未因此通过。证据见 `review/evidence/M2-L3-clock-launch-v1/binding.json`；本报告与terminal报告均冻结，本轮到此停止写R。
