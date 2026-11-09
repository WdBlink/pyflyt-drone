# 共享执行器的实施选择

采用 A 的单写者执行器与独立规划进程，吸收 B 的18m/s、R90、96m观察构造及R100短S连接候选。主Agent已逐项阅读两套rationale、sketch与geometry probe；独立评判同样完整阅读。根评分为A(2,2,2,3,2,2)=13/18、B(2,3,2,2,1,2)=12/18，与judgment.md一致。选择主要依据是物理worker不承担搜索和完整航迹认证、作业终态/版本/物理续段有同一写者，而非1分总分差。

这是开发实施选择，不是新平台资格。先物理否证，再实现必要的最小接口；不先建立通用搜索框架。沿用FlightTarget与FixedwingAutopilot；JobBook作为executor私有helper，不另建一套状态；mission保留隐藏实体、传感、不可变证据、传输、报告与评分。观察空间关联先保留唯一现有实现，向executor提供不含私有entity_key的本机数据。executor不能读取其他飞机未传输的估计；跨机种子只来自实际已送达的公开观测。

公共形状限制为submit(task)、advance(own_navigation,local_measurements,events)、public_jobs()。proposal绑定输入时刻、四机精确job token、reservation版本、资格包线ID、有限primitive列表、未来捕获/提交截止。planner只收不可变快照，不导入mission、Aviary、generator或grader。120Hz循环不等待planner；取消/退休先失效版本，迟到proposal与晚包不能复活终态。所有接纳路线含已预约、目标无关的物理恢复及原始退出圆；取消不释放正在飞行的空域。初始路线同样须在任务截止前有实际退出续段。

服务计时语义须升版本：survey从实测目标路线捕获起计duration，observe从首条新合格本机观测起计服务区间，另设明确获取截止。失测不补时，预测不填证据。动态评分仍从隐藏appearance开始的180s/3s全分母；6s只表示上传到达宽限，不是传感丢失容忍。handoff须接收机取得实际新合格样本，并已提交双机可行续段，才取消精确旧source token。

交通使用有限解析连接和离线时空占用检查，不实现Hybrid A*/全图lattice/NMPC。先测R100短S及A连接族的实际roll reversal、围栏、相位和恢复，再按有界FIFO候选相位试接纳。搜索/认证进程慢或停止时，飞机继续已提交程序。必须有跨stripe完成pass和双机观察/交接正例，也保留冲突拒绝例；全拒绝不能验收。

首个60s单机因果probe已执行，记录observe-guidance.v1.json：逆风2m/s、目标合成1.5m/s、2Hz噪声/漏测，240Hz审计安全且视锥持续可见，稳态径向误差≤1.385m，但高度误差达9.392m，**否证暂拟2m高度管**。该小probe没有遮挡、分类、上传或多机，不能替代mission资格。先定位减速/爬升/转弯高度控制，再进入交通实现；旧100/100 G1不覆盖此模式。

独立评判补算了相机水平偏移及目标高度，18m/s候选的条件视锥余量约9.652m；20m/s角点仅约.228m。两者仍依赖未资格的误差管，禁止当作证书。后续新控制族另冻结资格契约；原19项论文义务、八配置和正式研究目标不变。
