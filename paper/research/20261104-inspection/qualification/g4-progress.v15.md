# G4开发进度v15

一次独立审查准入的无分数B0+固定M1负载诊断已实际完成。600s、144000物理步、72000控制步、20个模型请求全部完成；full-array重算max lag=.08569320901295896s，p95（floor-order）=.005028416697541616s，>.1为0。完整ZIP、源/输入/依赖不变、原始二进制和实际资源计量绑定于 `clock-localization-fixed-load.v1/actual-diagnostic-binding.v1.json`；可复算脚本与phase-recount另存。独立证据审查 `review/pilot-clock-localization-evidence-review.v1.md` 核了数组、GC、归档和原失败原件，未把本次未复现称作修复。

最大lag继承前一tick的pub_end→log_end 86.357333ms间隔，同区间thread CPU .102458ms、process CPU .104ms；该tick的%120及%240条件均false，没有执行publication或1Hz logging。区间名不能被解释为logging原因。141对GC完整、overflow0；唯一gen2为394s的28.324541ms，不与主峰重叠。低CPU墙钟间隔支持继续调查等待/off-CPU方向，不能唯一认定调度、GIL、页故障或其他机制。20个请求bytes/seed/messages、输出text及token journal bytes均与原失败attempt00相同，但B0策略、实际请求迟到/耗时和host时序不同，不能叫原B4端到端复现。

该诊断20calls/142134tokens已由真实durable journal独立重算并native known结算；累计18个操作全known、373calls/2382225tokens、held/unknown/unresolved均0。cost仍未报告，host/资本/能耗仍未计量，不写成0。driver与模型/物理进程实际退出；旧pilot v2仍永久failed，108/48分母和两次clock invalid不回改，不进行第三次原槽重试。

下一项候选为另一条既有开发场景dev-L2-00的单次无分数adaptive B4-M1诊断，sampler2026110616，至多32calls/180000tokens。复用原actor/coordinator/service，仅显式worker factory增加begin/wake/log_end进程rusage计数及调用bracket、epoch前只读requested QoS，并记录既有外层进程观测的耗时。原63项科学源逐SHA不变，新增instrumentation/factory另版冻结；原GC/epoch/pacer/.1门槛/4thread1和模型profile不改，不调整调度。20项纯单元通过；只读微探针显示计数可变化，但sleep使nivcsw+1，因此不能把字段名当唯一机制；getrusage本身median542ns不构成整个仪器开销上界。意图与72文件freeze位于 `clock-localization-adaptive-L2.v1/`；独立source审查已完成，20所附+3独立Fake通过，真实spawn Process构造的partial目标及pickle可解析，未知cleanup拒绝completed；尚未实际预留或运行。见 `review/pilot-adaptive-clock-diagnostic-source-review.v1.md`。该新L2诊断不是独立科学root、原L1重试或pilot样本，pass也不能证明旧根因修复。

未安装的namespace proof v3已关闭main namespace祖先撤销P2，37项原/新增fixture与7项独立检查通过。复用同一native-valid snapshot及原完整main reader，先于holdout seed读取且在final snapshot再次检查；旧phase源/测试不变，历史23pass仅条件复用，未冒称重跑。见 `review/namespace-proof-component-review.v3.md`。这不是正式writer/dataset/raw/statistics完成证据；catalog来源覆盖、生成/接线及最终完整source inventory仍待。

G1–G3及同级非LLM event/clean参考对保持。G4仍needs_work，未抽取actual main/holdout seed/order/instance，未运行正式实验或开始论文结果正文；原19项义务、budget={}与完整实验分母保持。继续可用的已授权内部诊断和接线工作，不将时钟未复现或组件单元通过升级为G4。

历史catalog producer全12项测试重跑通过，仅覆盖当前显式合格输入；独立审查发现旧dev-cases.v1的dev-L2-03为已保留的唯一旧语义输入，现行load_development_case因continuous separation失败而拒收，故历史碰撞库存覆盖尚被阻断。后续需在catalog专用只读归档identity路径保留无效/退役输入，不放宽实际mission/现行加载的物理门，也不把31case测试称完整历史覆盖。当前未生成或资格化最终catalog。

该历史库存R1已由 `review/development-catalog-component-review.v1.md` 冻结：旧/新manifest的语义联合25（不是25独立root），另需5root dev population+intent、退役L2 clean、当前L3 event/clean及已核private-copy来源；这只是最小覆盖清单，不宣称全部使用史。R1待修，独立审查已停止写R。
