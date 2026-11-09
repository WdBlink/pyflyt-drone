# 开发期受影响义务的固定操作化

此文件在正式输入生成之前明确实现冻结合同中的“affected obligation”，待独立复验，不声称G4或正式效果。主任务、评分、总体、counts、原始格、物理路线及模型预算不变；旧候选与反例保留。当前单根的结果不用于挑选这些定义。

survey的region_id采用已有executor中心线y=(-300,-100,100,300)的固定水平Voronoi责任分区，并与实际ROI相交。最近中心线定义stripe，等距归较低索引：stripe0为y<=-200，stripe1为-200<y<=0，stripe2为0<y<=200，stripe3为y>200。这不是相机FOV或某次实际覆盖足迹。该静态工具语义须在模型pilot前公开给所有方法，不能给actor隐藏真值/未来轨迹。

硬件事件t^-的running survey仅关联该job.region_id内、事件时刻位置属于该责任区的当前active且仍需交付目标；offline grader可读真值位置，但其分区或目标映射不回传actor。无论是否已经发现该目标、是否成功感知，都按固定责任区判断，因此不能用“没观测到”把应有响应变成NA，也不会把其他stripe全部未满足目标算入受影响任务。inspect/observe有明确candidate，则仅用事件前已交付candidate与hidden identity的固定关联，不借用未来包。结束于事件时刻的旧业务job视事件前active，事件同钟新job不倒灌资格。

正确task_card_version锚在新job创建时：创建时priority尚未更新为1，更新后为2；rule_id须对应目标义务。后来priority更新不追溯使先前合法创建的job在别的事件响应分析里无效。priority自身响应仍须postevent新job，自然要求新卡。测量必须严格晚于事件，job可以在已生效的同钟创建。

same-ID static revision需ID.created_at<event，收到的新版本质量严格高于0、事件t^-已交付质量基线及前一sealed prefix质量。这样幂等重报、坏版本后恢复原值不能制造新的业务质量响应。新版本可以在事件同钟收到，按实际worker相位属于postevent。static recovery没有旧ID限制，但必须确有postevent新received version>=.6，且逐个接收prefix保持完整30s；同钟坏→好版本不能被时间聚合抹去。详见event-endpoint-ties.v1.md；其中旧“全area目标/仅前prefix增益候选”由本文件明确化，旧候选不回改。

所有eligible/NA/censored分母按event×target逐项保存；同一root的目标、事件、方法和重复不增加独立样本。方法间条件曲线必须同时报告这些分母，不能只画成功响应或宣称条件样本的因果优势。
