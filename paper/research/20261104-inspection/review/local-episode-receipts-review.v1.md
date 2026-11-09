# 本地episode回执独立审查 v1

日期：2026-11-06（Asia/Shanghai）。**当前版本不能验收为完整G4回执链。** 8项已有测试独立通过，但新增轻量反例确认缓存链未完整核验、spawn后写start失败遗留子进程、可见源缺陷被缺文件降级为可重试故障；目录创建的持久化也有覆盖缺口。unknown恢复保持保守，未被自动改成Y=0或重新启动。

本审查与主Agent分离、同模型家族独立上下文，不是跨模型家族验证。按已读poteto-mode的只读调查路径及show-me-your-work记录证据。只新增本报告及`review/local-episode-receipts-v1/`独立证据，不改实现/控制器/旧结果。所有子进程均为合成JSON或sleep/锁测试，不是物理飞行。测试进程全部已回收。

## 1. 绑定版本与实际检查

| 文件 | SHA256 |
| --- | --- |
| `eval/inspection_episode_receipts.py` | `adc951711839238bc2d8f023571c95d18da11e37263d487d11672f6ebf4391a2` |
| `tests/test_inspection_episode_receipts.py` | `4f1e826b318ce4324c88529620f369a28a2148a5002d76672edecb80716b24d1` |
| 新入口`eval/run_inspection_dev.py` | `e63d69e75251752330da2806915bae711ad4be2920e9017c2ffea726d64386dc` |
| SDK适配`eval/inspection_recording.py` | `9613a8e800d470110dbf86890837ba853c3f50834ae29f0722994927d27cb352` |
| worker入口`eval/run_inspection_reference.py` | `8b2aa155445d15ad736d972d42b09c210b07dacbd439e52627161837c43e1f75` |

主Agent另保留旧实现/入口/tests于`qualification/source/local-receipts-v1/`；本审查自己的两份原始源码快照与运行结果在`review/local-episode-receipts-v1/results/{source-snapshot,report}.json`。探针启动与结束的两份受审源码SHA一致。以下行号绑定上述旧字节，不批准后续同名修订。

实际命令：

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider --basetemp=paper/research/20261104-inspection/review/local-episode-receipts-v1/pytest-temp tests/test_inspection_episode_receipts.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python paper/research/20261104-inspection/review/local-episode-receipts-v1/probe.py
```

结果为`8 passed in 0.24s`及独立探针通过执行。这里“探针通过”指反例成功复现和证据保存，不指受审代码无缺陷。没有运行新物理、API/模型或重负载；已经完成的L1-00-full不是由本模块产生，不作actual-receipt flight证据。

## 2. 支持的状态与正常路径

数据形状为冻结plan下的attempt-00/01。intent只证明持久化尝试意图；started只证明父进程观察到Popen成功，不证明第一物理步或模型/provider调用；terminal是父进程已观察退出后的评分/操作结果。

- `attempt`存在但terminal缺失，或terminal JSON不完整，会阻断为unknown，不启动第二次。这覆盖意图前后、spawn/start之间、等待期间和terminal写入前的常见进程崩溃窗口，前提是目录/文件在恢复后仍存在。
- 合成worker正常完成、E=0导致Y=0时记录`completed`，缓存重用且不补跑；已知退出7且无结果时记录`known_operational_failure`，Y不被填0，最多两个attempt。
- 独立注入terminal写入ENOSPC后，world已完整存在但无terminal；下一次仍unknown阻断，worker只调用1次。不能通过“有world看起来完成”自动补认成功。
- 独立对B0/B1/B2分别使用同plan的first-fail夹具，均恰两次attempt，第二次Y=0完成，两个terminal保留。现有exhaustion测试证明第二次失败后不再启动第三次。
- 独立实际进程持有flock时，另一监督调用被`another supervisor`阻断，未启动worker。锁保护同一目录中的协作监督者；不保护外部改写/替换锁文件，也不是跨主机租约。

## 3. 需要修复的问题

### R1 / P1：缓存未验证完整intent/start/terminal链，回执可自相矛盾

位置：`inspection_episode_receipts.py:121-133`。completed缓存只重算flight evidence、与terminal.evidence比较，未验证intent/start存在或其内容绑定、stdout哈希，也未比较重算Y与terminal.Y。

四个独立复现均使用先真实执行一个轻量Python合成worker产生的缓存，再修改一处文件，随后传入必失败command工厂：

| 探针 | 修改 | 当前行为 |
| --- | --- | --- |
| `cache-missing-start` | 删除started.json | 仍cache_reused=true，worker计数1 |
| `cache-intent-argv` | intent.argv改为不同命令 | 仍cache_reused=true，worker计数1 |
| `cache-stdout` | 改stdout.log，terminal保留旧stdout_sha256 | 哈希失配仍复用 |
| `cache-terminal-Y` | terminal.Y由0改为1 | 返回重算Y=0，却把terminal.Y=1的文件作为新回执引用 |

最后一例没有把返回数值变成1，问题是新SDK包可绑定一条与自身Y不一致的终态文件。其余例破坏了审计链但不触发拒绝。文件可由外部写者同时全部伪造不是本报告声称能解决的密码学威胁；这里是不完整/不一致历史在正常缓存校验中未fail-closed。

建议terminal绑定并复核intent、started、stdout的SHA和身份字段，比较Y；重用前核对plan/ordinal/argv/实际输出目录及三阶段角色。损坏或缺链应阻断，不新增physical attempt。SDK只绑定本次计算的terminal SHA无法发现该终态内部未验证的矛盾。

### R2 / P1：spawn之后start写入错误绕过子进程清理

位置：`:147-165`。Popen之后的`durable_once(started.json)`位于try/wait异常清理之外。独立将该写入注入ENOSPC，Popen启动一个纯sleep进程后函数立即抛异常，但子进程仍存活。探针确认`child_alive_after_supervisor_return=true`，随后由审查脚本立刻SIGKILL并wait回收。

恢复调用仍被unknown阻断，未重复启动，这是正确部分；但未知进程继续运行可与后续world争用资源，也不能靠函数异常就认为物理已停止。应把所有spawn后写入、等待、输出flush等路径纳入统一try/finally，安全停止并回收本次进程组，保留unknown语义。真正SIGKILL监督者不能执行finally，仍需记录pid/进程身份并由显式reconcile处理，不得自动重放。

### R3 / P2：缺文件兜底掩盖可见源/输入缺陷，给其一次免费重试

位置：`:58-89,168-174`。validate已开始收集`source_or_dependency_mismatch`，但读取缺失case文件会抛OSError；外层把所有这类异常覆盖成唯一的`worker_operational_failure`。

独立`bad-source-missing-case`夹具让run完整、`sources_unchanged=False`，然后删除private-case。两次都退出0。当前结果为worker调用2次、两个terminal的retry_allowed=true、原因均只剩worker_operational_failure，最终报预算耗尽；没有按源缺陷第一次就停止。

该反例没有获得合法Y，也没有突破2次上限，故范围是错误分类与无效额外运行，不是坏分被接受。建议先收集所有可读的源/输入/绑定缺陷，读缺项不得抹去已有非重试原因；正常非零本地退出与损坏的已完成回执分别定义。已有单独bad-source测试不足以覆盖组合故障。

### R4 / P1耐久契约缺口：新目录的父目录未在spawn前持久化

位置：`:104-117,138-143`。durable_once会fsync文件及该文件所在目录，但新episode根目录的父目录、attempt目录在episode目录中的新条目没有对应的创建后父目录fsync。

独立fsync轨迹显示episode目录只在写plan时被刷一次，此时attempt尚不存在；创建attempt之后只刷attempt自身，没有再刷其父episode目录。因此“意图文件已fsync”不足以证明承载它的新目录条目也持久化。若恢复后整个新attempt条目丢失，当前逻辑会把它当作从未启动；这是耐久性覆盖缺口，**未作真实断电复现，也不声称已观察该机器的文件系统丢条目**。

应在创建receipt根目录及attempt后持久化其父目录，并定义新建祖先目录链的责任，或要求调用方提供已持久化base。Linux官方fsync说明明确区分文件与父目录条目的持久化；Mac官方文档还说明普通fsync不等于严格物理掉电保证，需按平台明确F_FULLFSYNC/硬件边界。[Linux fsync说明](https://man7.org/linux/man-pages/man2/fsync.2.html)，[Apple fsync说明](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/fsync.2.html)

## 4. 新dev入口与缺失的SDK集成

新`run_inspection_dev.py`在读取manifest时限制dev命名空间，唯一匹配case，检查直接子文件及其文件SHA；构建包含suite/lineage、源/依赖、thread environment、监督器及adapter SHA的plan，并用同一固定argv工厂启动600s/realtime/B0或B1/B2。其源集合与worker在`--case`条件下的源集合静态吻合，较任意command_factory调用有更明确的设置边界。

尚需注意：

- 本审查只读该入口，没有执行物理集成。没有由它产生的真实flight receipt，不能把旧L1-00运行倒认作模块验收。
- receipt模块只比较run自报的private_case_sha与复制文件SHA，未直接验证worker反序列化后实际读入的case canonical SHA/身份；plan的case_sha、lineage和suite SHA尚未逐项与worker实际读取证据闭环。
- plan声明线程全为1，worker会记录runtime_thread_environment，但validate_flight未比较二者；Python/package_versions也未绑定到plan。重试同设置在新固定command闭包下有静态支持，不能仅凭相同plan哈希就证明任意command_factory执行相同设置。
- manifest的“冻结”仍依赖外部已登记的预期版本。入口记录当前manifest SHA，并检查其自报dev标签，不等于由数值代码认证独立root来源或此前已批准的版本。

当前调用链只到`run_inspection_dev → execute_local_episode`。`inspection_recording.py`还没有真实measure/receipt loader；现有预检measure仍是合成。要复用真实flight进入新SDK包，应在measure回调里按冻结身份取得/执行本地receipt，先验证完整来源链，再返回该已验证Y和receipt SHA；policyY0正常调用SDK complete，UnresolvedEpisode应传播并让SDK保留失败与未启动全分母。

缓存命中时，新SDK的attempt/start只代表本次记录/导入回调，不能再算一次飞行。应保留`cache_reused`、真实physical attempt数、原receipt身份、原Y及全部trace哈希，外层root均值继续绑定该内层raw；跨包复用同episode不能增加独立root/episode样本数。当前receipt返回绝对本机路径，导出可复现包还需要受控相对路径/打包清单绑定，不能只保存一个可能失效的绝对地址。

现有record_matrix固定全部8配置，而本模块只支持3个classic，LLM5配置/3repeat尚无真实measure/provider/费用路径。不能以三个本地receipt补足完整1296槽或放行G4，也不能自动将未知LLM任务当本地可重试失败。paid请求恢复仍明确不在本模块范围。

## 5. 验收边界与决策轨迹

| 检查/决策 | 证据 | 结果 |
| --- | --- | --- |
| 先跑既有8测试，再主动组合故障 | pytest结果、独立probe.py | 8通过不覆盖R1-R4 |
| 对缓存分别改单一文件 | results/report.json的4个cache条目 | R1成功复现 |
| 在start和terminal写入分别注入ENOSPC | start-write-failure、terminal-write-failure | R2遗留进程；terminal unknown阻断正确 |
| 用bad-source叠加missing-case | bad-source-missing-case | R3分类丢失，2次额外运行 |
| 追踪真实fsync调用，不作掉电宣传 | directory-fsync-trace及官方文档 | R4范围明确，未实测断电 |
| 同条件cap与真实进程锁 | same-cap-by-condition、interprocess-lock | B0/B1/B2均2次cap，竞争者未启动 |
| 将新入口接线与真实实验区分 | 新入口及SDK调用方只读检索 | 无真实receipt flight/SDK完整矩阵接线 |

主Agent已收到R1-R4及源快照指针并接受另版修复。修订后须用新SHA复验，不以本报告认可未来实现。未知恢复的显式reconcile、真实输入来源、模型/provider/费用、对称重试在全8配置的公平性与正式矩阵完整性仍未认证。

Attention

reviewed by inherited-parent model（同模型家族独立上下文，非跨模型家族审查）

本次只读审查支持unknown阻断与正常本地故障/策略Y0分离的已测范围；R1-R4及缺失集成阻断完整回执验收。全部JSON为显式合成，不是物理或方法成绩。
