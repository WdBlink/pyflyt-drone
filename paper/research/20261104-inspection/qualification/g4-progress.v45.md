# G4 开发继续点 v45

G4仍needs_work。同一v4队列37/108 callback、两个完整root（L1-01、L3-01）。第三份SDK视图因order37的清理unknown停止，保留37 completed、1 failed、70 stopped及48 outer stopped。38次物理attempt均为各原slot的attempt00。order38（dev-L2-00/B4-M2/repeat2）未预约、未启动。

order36 L2-00/B3-M1/repeat2正常完成、清理closed；order37 B3-M3/repeat1原return0、tree unknown、terminal Y=null、retry=false。两条600s/144000步、0个超100ms tick、runtime errors=[]；max lag分别9.677/9.354ms。原完整冻结评分重算均为Y=.3314065510597302、C=.9942196531791907、S/D=0、H/E=1、SR=0；order37评分尚未纳入有效callback。对应25calls/165249 tokens与22calls/160664 tokens结算known。order37 report audit 38条，accepted不同ID 7；最终progress与后者一致。普通clock证据不等同完整逐tick数组。

第三次原扫描中的65758（PPID1/PGID65758/UID501，原出生秒及comm一致）与同boot的PID1 launchd确切job、成功spawn和移除事件相符；双扫描时窗夹在后两事件之间，五个已登记原进程均已清理。晚时点观察原owned与该PID均未见。024/030同一job及各自精确PID的OS记录与原tree文件时刻相符，但缺少原扫描行/时间，强佐证不等同037直接交叉匹配。未修改外部job、未发送信号、未读取语音或应用日志。

复核 `recovery/pilot-v4-cleanup-037.v1/evidence.v1/binding.json` SHA1d69331ce39b41a92d73d39808449439c5761c21b3fa48b9010eb70110379406，含两原分数、原终态/模型/归档/结算绑定、原sidecar、只读OS记录、native状态与SDK中止。独立归属审查 `review/pilot-v4-foreign-cleanup-evidence-review.v1.md`，SHA003206f7f3aa555b4ac7406833566395e3336aaba8c583bac16ef541ba4d2645；其结论不代替精确037处置接线验收。原024/030 native接受和三份unknown均保留。

唯一owner累计known1314 calls/8480064 tokens、reserved=0、unknown/unresolved/stale=[]。费用、主机、能耗和Codex原生用量未计量。原104 source/61 deps/10 inputs、科学分母和原19项义务保持。下一步提交精确037处置，随后fresh38-cache重建并在原order38前核验零新effect；同类外部进程的前瞻处置需要单独冻结、验证并显式使用。正式44source/27tests仍未安装，无新模型调用、飞行或正式数据；完成原pilot、实际clean/20s、最终冻结/容量及独立G4后才进入正式实验与论文。
