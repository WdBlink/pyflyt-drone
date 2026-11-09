# 下一候选的水平误差界校准提案

状态：开发提案，未实施、未资格，不改判任何既有slot。原始计划的科学评分、6s交付宽限、180s全部格、15m物理间距、60..100m高度、15..25m/s空速与45°bank均不变。

v16完整observe15/20；新v17的20个observe已全部完成，D均≥.888889，但604/612/619仍因原点时刻6m水平管失败，峰值6.869/7.631/6.967m。v17 transfer尚在执行，不能提前给完整判定。v16另有9.980m峰值，证明原6m是被数据否证的工程假设。新前馈与完整返回搜索并不使原6m假设自动成立。

下一候选拟采用10m**同一时刻的欧氏水平误差界**，不改成phase-free/最近路径距离，不改变任何已录数据的误差定义。它是开发包线重校准，不是把旧数据改成通过。稳定高度仍±2m，完整OwnState/PID/capture和逐240Hz误差记录保留。

更大物理误差界必须与更保守的全机准入同步：

- 所有survey/inspect/observe/exit、旧有限占用、完整返回及共同周期尾环，连续名义水平间距由27m提高到至少35m，即15+2×10。
- 连续名义围栏由544m收紧到540m，即550−10。继续使用fleet_clearance已经扣除的时间采样Lipschitz间距补偿/围栏补偿，不能只改离散sample阈值。
- 光学导航预览同样使用10m自身误差余量，仍仅作排序，不能替代实际frame/ray或补造evidence。
- 单一共享常量供planning/execution使用；trace显式写自己的xy_tube_m。新recount必须核对合同与trace相同，历史缺字段只按原6m解释，不能用新默认重计成旧成功。

新source/contract在运行前冻结，使用先前未评估的控制roots800..819做完整observe/transfer各20，新失败全部保留。只有完整分母、来源/依赖/审计与每族≥.95条件满足且独立复核后，才判新的G1。既有v16/v17仍按各自6m合同失败。

G1基础100可仅在control.py/flight.py/qualify_inspection_g1.py三份消费源、G1整个子对象、依赖和导入初始化均不变时引用g1-full.v4；新规划器必须有自己的40slot证据，不能由基础100替代。此复用边界已在review/g1-v16-full-review.v1.md独立说明。

没有新增付费API、算力采购、正式test/holdout曝光或投稿。当前所有科学方法成绩与G2–G4仍待验收。

## 后续实施记录（2026-11-05）

以上为保留的历史提案，开头“未实施”及transfer进行中描述只对应撰写时刻。随后v17全部40slot完成，transfer20/20、observe17/20，整体失败；原6m三项失败和原manifest均未改判。v18候选已实施共享10m、ALL modes 35m/540m完整连续准入及显式trace/contract一致性，149项完整测试和项目compileall通过。新根800..819契约已前瞻冻结，独立审查仅判断允许启动40slot；尚无v18完整实飞资格结论。参见`qualification/PROGRESS.md`与独立`qualification/review/tracking-envelope-v18-review.v1.md`（审查报告落盘后才是实际报告）。
