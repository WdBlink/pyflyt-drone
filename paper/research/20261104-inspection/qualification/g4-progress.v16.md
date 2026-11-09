# G4开发进度v16

单次dev-L2-00 adaptive B4-M1无分数诊断已实际完成且clock invalid：600s/144000步/72000 clock及rusage rows，maxlag1.8398782083609717s、floor-order p951.1624444163714998s、17020/72000步>.1（29连续段）。真实18calls/127751tokens已native known结算；当前19资源操作全known，累计391calls/2509976tokens，held/unknown/unresolved0，cost/host/资本/能耗仍未计量。完整归档与source/supervision/计量绑定 `clock-localization-adaptive-L2.v1/actual-diagnostic-binding.v1.json`。此为完整工程故障记录，不是有效pilot样本，旧v2失败SDK仍永久保留且未恢复。

独立审查 `review/pilot-adaptive-clock-evidence-review.v1.md` 重算全部数组/GC/归档/来源/模型计量。主峰begin已经落后1.8376s，不能以该tick约10ms工作解释全部backlog。major fault全程5、峰附近0；ru_nivcsw原始216000快照有229次-1回退，不clip为0或伪称精确切换数。GC125pairs/overflow0，唯一gen2约129.692ms在sim437.125，不是峰附近唯一根因。大量长wall低CPU间隔仍只支持off-CPU/等待方向，未唯一归因。

另有直接可测基础设施开销：原监督1734次full-table ps观察累计wall456.28053s、parent CPU22.082895s；控制窗口覆盖union430.15s，既非总CPU，也不能据重叠证明旧L1或新L2根因。已准备未安装的process-inventory候选：只在当前root已绑定时，用SDK libproc查询已观测root/子孙，保留原完整PS baseline/attach、未绑定root恢复、shutdown/未知孤儿检查及.1s wait cadence，native不确定回退原fullPS。原_inventory/stop_worker_tree/init/attach/wait/close AST逻辑相同。13个新纯/真实自建睡眠进程测试与39个合并兼容/receipt故障测试通过；实际新生child发现、PID/parent变化拒绝、失败attach恢复与清理均覆盖。四进程空闲20pair微测身份/完整forest一致，中位wall fullPS86.175ms、native.436ms，非载荷clock资格。SDK C sizeof136/offset与实际字段核对；早期全表native权限失败/中间hybrid方案和当前选择均保留，没有省略不可读owned进程。见 `planning/inspection-process-inventory-candidate.v1/freeze.v1/`，尚待独立审查和安装/来源闭包修订；尚未新预留或执行post-repair资格。

历史catalog R1仍待修：旧invalid dev-L2-03必须进入只读identity库存，不能由现行可执行性筛掉；实际mission/load guard不放宽。namespace祖先撤销组件P2已有限闭合（37+7），完整formal writer/generator/consumer/raw/analysis接线和最终source清单仍待。G1–G3及同级非LLM参考对保持原范围，G4仍needs_work。未生成actual main/holdout种子/顺序/实例，未开始正式实验或论文结果正文；原19项义务、budget={}、全部失败/未启动分母不变。
