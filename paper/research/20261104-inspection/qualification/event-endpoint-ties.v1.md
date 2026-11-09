# 开发事件端点的同钟顺序

这是冻结endpoint的实现约定候选，待独立复验；不改主score/原始需求格或formal总体。原始6e193候选与其反例保留，不回改旧结果。

依据实际worker和`InspectionMission.step`，每个物理tick在积分后再次执行事件，再sample/deliver/publication，随后同一时钟值的工具请求可能执行。故资格采用严格t^-：只用report.t<event、observation.arrived_at<event的已交付状态；活跃业务job须created/started<event，finished>=event视事件前仍在执行。不得以事件后刚到的包制造“事件前”资格。

响应可以来自created_at==event、正确新card/rule的postevent新job，但packet.measured_at仍须严格>event，且实际有新合格packet、飞机在线未退休，ACK/queued不是响应。静态同ID修订须该ID.created_at<event，事件时刻收到的新正质量增益可以记录0elapsed。静态恢复同样可由event时刻新收到的合格版本开始，且必须完整保持随后30秒；恢复不额外要求旧report ID。仅旧good版本持续存在不能自动生成postevent恢复。

静态保持逐个接收prefix检查，包括同一timestamp的多个版本。good10→bad20→good20不能用最后good20遮住bad20来把恢复追溯到10；最早候选为20。t+30必须<=600，晚到版本不能凭缩短保持期获得恢复；未恢复仍按600右删失。动态恢复只用完整原60格中的十连续合格格、各STARTS>=event，终点取末格结束与所选sealed报告到达时刻之最大值；剩余不足十格仍eligible并可能结构性删失，不转NA。

当前survey area-work全当前active未满足目标及static同ID严格正增益解释仍待语义审查；当前候选统计不能据此称最终事件效果。所有event×target记录与同根重复须成组，不视为新增独立样本。
