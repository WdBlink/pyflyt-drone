# v4 cleanup 外部进程归属证据独立审查 v1

037的两次原扫描与同一个外部 launchd job 实例吻合。024和030也有精确PID、job和时间的强佐证，缺少当时的进程行和扫描边界，证据强度低于037。037当前最小可审路径是在原attempt结束并完成known结算后，单独提交绑定原记录的处置。其 `unknown_process_cleanup`、Y=null和retry=false仍以原terminal为准，既有024/030接受仅适用于各自slot。

## 三次归属证据

审查人为 `/root/control_review`，同模型家族独立工程上下文。复算使用提供的三份过滤 `/usr/bin/log show` JSON、过滤launchctl元数据、原tree/terminal和037原sidecar。脚本、逐字段结果、输入SHA与过滤原件在同名evidence目录。检查的是存量记录和文件时间，原grade、clock和meter结果沿用原记录。

所有提供的事件都来自PID1，`processImagePath`和`senderImagePath`均为 `/sbin/launchd`，共享 `bootUUID=AAE87917-4736-4BAA-99B9-F7C816816E6A`。每组先出现精确 `gui/501/local.home-jarvis.xiaomi-voice [PID]` 的xpcproxy和python成功spawn，再出现同boot、同PID的系统removal。去除秒级近似后，成功spawn、原tree文件时间和removal的顺序如下。表中均为2026-11-07 +0800。

| order / PID | Python成功spawn | 原tree birth / mtime | removal |
| --- | --- | --- | --- |
| 024 / 26337 | 06:18:36.892904 | 06:18:37.187908 / 37.187993 | 06:18:37.200347 |
| 030 / 86715 | 08:23:25.318551 | 08:23:25.577844 / 25.577934 | 08:23:25.601664 |
| 037 / 65758 | 10:53:04.896116 | 10:53:05.210531 / 05.210606 | 10:53:05.401655 |

037侧录 `6edd60553b11cf9032bf67e7a5f05f22e05550769c9856c624c3e2b5fa25cee8` 的original_result与tree逐值相等。context匹配原plan SHA与037前缀，observer SHA83ff匹配，原异常为空且无观察错误。两次扫描均见65758，PPID1、PGID65758、UID501、born为 `Sat Nov 7 10:53:04 2026`，状态先R后S，comm与过滤job program字段精确相同。已登记组为60623和60626，remaining_owned=[]，PID也不在原baseline中。两次扫描在02:53:05.064001–.132385Z及02:53:05.132930–.206915Z，结束时间由记录的monotonic耗时换算。整个扫描窗均夹在对应spawn与removal之间。该实例的归属依据是原扫描身份与精确OS job事件的交叉匹配，路径和PGID各自只提供佐证。记录的removal表示日志中的移除事件，其时间精度不可当成精确退出时刻。

024和030的原tree birth分别早于对应removal 12.439ms和23.820ms，mtime也在窗内。这强烈支持两次命中同一外部job的解释。原tree没有unknown row或确切扫描时间，文件birth/mtime是观察到的文件系统元数据，无法约束较早扫描到哪一个同PID实例。当前证据不能恢复其当时的comm、PGID和born，也不能形式化排除扫描到写文件之间发生的PID复用。三个OS片段共享bootUUID，037侧录自身未存独立boot ID。过滤材料仅能核查所提供事件的唯一匹配，不能证明完整主机日志无其他事件。不得把这些限制换成单靠PID、名称、当前launchctl配置或路径匹配的豁免。

037原tree SHA为 `a4f78cb975844c88ad90e7fb65f92b151110d92b20e7bf90b03b534ceff36ba2`，terminal SHA为 `2183b93170d7362349bcfe33e452809d989ab8f0a212d9848436e0e4c4a6aaf0`。原terminal内的maxlag为0.009353917004773393，冻结rawY为0.3314065510597302，22calls/160664tokens。原settle ack确认为同037-a0的known。现有材料支持精确处置送审，仍需该slot自己的独立接受和native finding。

## 接入点的取舍

原close用全局扫描识别未登记的新同uid非Z Python、PPID1的进程。`root未登记`、`owned仍live`和`unknown列表非空`均能独立触发unknown。保留这三项判定、原扫描与信号行为。归属证明至多处理第三项中的具体PID。即使全部待认领PID都匹配外部job，另两项存在时仍应阻断。 [原close](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/inspection_process_cleanup.py:150)

1. **已结算后追加处置。** 原supervisor先保存tree/return，生成archive与meter/settle，再由原完整validator出结果。cleanup unknown仍落原terminal并停止。外层针对精确plan、attempt、原tree/terminal和fixed payload/settlement refs，绑定完整原grade/meter、sidecar、OS记录、归属断言与独立/native接受。处置只向新视图提供经验证的原结果及额外proof，原SDK中止视图和旧记录保留。此点复用024/030已审过的追加模式，成本是单次失败后需重验和明确接续。037的removal日志晚于close，事后点可以按完整证据判断。当前推荐此点处理037。 [原supervisor写入顺序](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/run_inspection_llm_episode.py:174) [原unknown门](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/run_inspection_llm_episode.py:224)
2. **原close已封存后，给supervisor单独的有效判定。** 可在未来独立冻结的接线中保留raw close、sidecar、OS证据，再让只针对该attempt的proof控制有效判定。原supervisor没有读 `effective_status` 的入口，运行时只认 `cleanup_evidence.status`，缓存也硬查tree.status为closed。若要求未来canonical process-tree仍存原unknown，单加字段无法同时通过两道原门。维持104磁盘源时，必须显式区分原unknown与提供给旧supervisor的有效视图，确保证据随新attempt归档，并在重用有效缓存前再核固定proof。不得将derived closed写成原close通过。完整grade/meter可沿用旧入口，admission语义和缓存消费者仍是新增资格面。这里还存在时序限制，037记录的removal晚于close结束约193ms，实际读取日志还可能晚于事件时间。即时proof缺失时继续按原unknown，不能靠放宽等待或跳过removal使该分支通过。 [旧缓存门](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/eval/run_inspection_llm_episode.py:107) [既有observer封存](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/recovery/pilot-v4-cleanup-030.v1/observer-freeze.v2/observe_cleanup.py:125)

两点都可保持104磁盘源和原close算法，②需要更广的接线、可审的raw/effective双状态和缓存重验。①足以处理现存037，②应作为新前瞻候选单独审查；不能回改已存回执。任何适配器及其证据生成、消费源码都需单独冻结与provenance，保留原风险门和科学分母。当前既有双处置只提供精确024/030分支。 [现行接续入口](/Users/wdblink/.codex/worktrees/538e/pyflyt-drone/paper/research/20261104-inspection/recovery/pilot-v4-cleanup-030.v1/source/continue_pilot.py:323)

## 最小前瞻证明与验证

- 每次处置绑定同一attempt的root身份、原close结果、最终扫描中全部待认领row、原observer固定ref及错误状态。原root需已经登记且剩余owned为空，其他错误继续照旧处理。逐PID绑定UID/PPID/PGID/comm/born，保留秒级born的实际精度。对一个未获证明的PID即保留该attempt的原阻断。
- 只读采集OS证据并封存实际查询边界、boot来源、退出状态、完整性和原bytes SHA。匹配同一boot内精确job label/PID的launchd xpcproxy、成功spawn、对应removal。核扫描时窗与born，检查时钟基准和事件歧义。以037原JSON/sidecar做真实文件I/O阳性，写出每步匹配，保留原unknown。采集不可用、缺事件或截断时记录缺证。程序配置路径只佐证，任何路径/PGID/PID单项匹配均不足以通过。
- 至少做真实I/O负例覆盖删改sidecar或OS日志、别名与读取窗口漂移、相同路径不同job、PID前缀误匹配、不同UID/PGID/born/boot、越出spawn/removal窗、重用PID的重复或矛盾事件。再验证mixed unknown列表仍有一个未证明、任何owned live、root缺失、observer错误、原异常、known结算或原validator不满足时保持原阻断。原本closed的正常路径逐值保留。②还须以隔离caller证明raw/effective及原异常各自保存，缓存不接受失绑proof，meter/grade和分母一致。实际前瞻归属若未遇到新外部命中，应如实标为该分支尚未实测。

本轮执行的是只读文件匹配与时间重计，使用已过滤的OS记录，未访问外部job的语音或配置内容。实际fresh v3、38-cache barrier与037独立/native处置留给后续核验。本报告不授运行，已停止R写入。
