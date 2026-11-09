# G4 开发继续点 v31

G4仍needs_work。v4开发pilot现12/108有效callback、0完整root；没有实际main/holdout namespace、seed/order或dataset，也未开始论文结果。原19义务、RA-L、budget={}、600s/144000步、100ms门、两attempt上限、B1原选择、17源grader与主global6不变。所有旧失败/未启动分母保留。

新增order9/10/11同为dev-L1-01，分别B3-M1/repeat0、B3-M2/repeat1、B4-M1/repeat0；均仅attempt00、completed/return0、600s/144000步、0超100ms、physics_runtime.errors=[]。maxlag为0.013702042000659276、0.00970483400305966、0.014017124998645158s；P95为0.0039033333353586386、0.003065750000700973、0.003874541665823017s。新增22/146917、30/178072、24/172337 calls/tokens已known；native实际status总762calls4805143tokens，reserved0、unknown/unresolved=[]。cost/host/Codex/capital/energy未计量，不写0费用。十二槽仍只是同一dev root的部分18槽，不构成完整root或正式方法比较。

UTC2026-11-06T19:02:04.932678Z已在order12/B1预约/spawn前确认barrier，同一exec4162继续等待现有checkpoint.request。本轮集中检查只在该barrier后执行，104生产source/61deps/10inputs未改；回合中的新代码均在隔离候选。三槽计划、terminal/run/world/private及settlement SHA、当时stdout/pause/native-status的immutable副本见 `qualification/pilot-v4-slots-ten-twelve.v1/binding.json`（881e445227c87a44b8d666dbc37f1efe7753f40efbf10214ca0d386af9402976）。普通clock max/P95/witness不是完整72000-phase重计。

隔离新增secondary分析入口，尚未安装或用于真实formal数据。它要求完整且独立锁定的main分析，复核实际主锁和scores/components/SR/index引用，重新admit本轨全部216/108/180槽及所需原primary216/36槽，与锁定scalar/full-score SHA一致；不因同名case或任意score JSON接受对照。三/两原SDK envelopes以同一捕获bytes在私有临时目录验证，完整order/arm/sampler/repeat/root mean和callback marker复核；物理/model/resource/reserve/settle的全部文件及递归目录在读入前固定，验证/发布前比较原集合和SHA。复用已修primary snapshot/scalar helper，typed track限定reservation前缀，不生成main锁、holdout opening或任何native资源操作。

common-calendar先重现各自原score，再按原event parent同一需求日历重计；完整重评分与原动态bins进入逐episode ZIP，scalar八字段保留canonical full-score SHA。SDK原design/raw/value/evidence字节保留，outer results仅投影design/raw相对路径。闭合文件树metadata与每episode详细事件表分别ZIP/index绑定，防止大单JSON超限；metadata不冒充raw flight或全host原子性保证。pure bootstrap仍原已审字节，主global6未改。新consumer/event/numeric及实际tracking依赖已加入候选REQUIRED_SOURCES，生产仍未安装。

事件helper保留原dev公开验证入口与手算逻辑，新增显式fixed12-parent formal入口。job落地、原合格信息响应、原窗口质量恢复分别报告；commit与local-acquisition双路径仅在规则/卡片/因果候选/固定survey责任区域及实际step/采样关联充分时给条件推断，不冒称逐控制job/setpoint直接记录。系统τ、推断应用tick、actor实际接收时刻分列，后者在world中未记录则unknown；clean没有真实事件通知，不虚造其响应行。静态start与start+30确认、动态十个原3s格及到达见证保留；NA（含pre_event_threshold_met）、policy右删失、trace gap分别计数，所有U/H/indicator留原表。按root/domain/method/event列全分母与H/observed-only分布，无新增时间检验或成功样本筛选。

软件失败与独立反例全部保留。v1 focused60pass/1fail27.59s：最后dynamic-witness fixture定位旧package属性；v2改当前staged模块后61pass28.89s。随后whole收集因旧common_calendar cache缺formal helper失败；v3仅显式加载staged依赖，whole490pass132.95s。两次都是fixture/import闭包错误，不是新flight或资源重试。绿色仍未闭合事件实现：独立v1实际发现重复commit、off-2Hz局部帧、False/int创建绑定三P2（报告af9c50e41bdcbf518a1f9e3421782da5b623d417d0ef4effed4e2eb24031fe10）；v4修后whole499pass124.28s，但独立v2仍复现program链8个bool/int身份残余（报告56bd70b01bd7ee578e3436ad00a82bd0681bed5efc40528acf11b0f84cfc3425）。不能用任一whole绿灯覆盖当时反例。

最终v5只补严格program/request/response integer identities和四个(version,job_id)token类型，保留原值、不cast；最终三相关文件78pass20.76s/exit0及compileall exit0。没有运行/声称v5 whole507；v4 whole499与v5 targeted78边界分开。独立最后22检查确认八原反例转unknown/endpoint=None、正常commit/local12及12.5/handoff/不同serial刷新-FIRST/typed tuple tokens保留，未见该delta新阻断；binding `review/evidence/formal-event-endpoints-v3/binding.json` SHA756700281fad492007fafdfcf8fa88a7cb975b9fb8fbc974347bced2799b4ee3。原F1/F2/creation修复按未改范围沿用。该语义检查为literal/AST及显式Fake wrapper，非真实formal证据。

独立consumer复核用3组实际提供SDK+Fake physical/native-shaped阳性得到216+216/108+36/180+36及outer96/36/60，12项针对性反例拒绝且无新输出，未见consumer/REQ范围新阻断；binding `review/evidence/formal-secondary-analysis-consumer-v1/binding.json` SHA7277c9e2a3a2b9453b769b59ca83650e7db1e8bb63a74db156b827b826f647d9。它使用精确冻结consumer及旧event summary stub，不认证后续event/numeric语义；首个/tmp路径别名错误、后续外部event working变化与修正guard范围均明示。两类独立检查不合并为wholegraph/G4/human批准。

最终八成员freeze SHA2f71d6fee26641c868f54b681d3a35bc705e879150f0e6640844c1857052b4ed。精确43source/26tests的旧baseline+最终delta重建、全部版本测试/失败与退出证据见 `planning/formal-admission-candidate.v3/secondary-analysis-integration.v1/binding.json`（6e72718a1c6fb7c529dd54a7de065300bc02e037d51e995b0c65841a7e844e82）；其制作时最后review尚未由root观测为冻结完成，最终限定结论由对应review/binding及本记录另绑。软件数据均为明确fixture，没有实际formal抽样、模型/物理或native资源effects。

native仍experiment（16条限定文献已接受）。response/resource实际时延及成本诊断consumer仍pending，不能宣称F4/F5/T4或G4已整体完成。所有检查及其他R写者结束后，本轮owner记录并立即恢复同一pilot order12；继续完整108开发槽、actual clean/20s-delay集成、最终source/model/prompt/generator/split/retry/hardware冻结、容量预检及独立G4决定，再进入真实正式实验、图表与论文写作。仿真/待人类复核、小样本与真实root独立性限制仍保留。
