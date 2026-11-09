# v4 第二次 cleanup unknown：归属观测独立审查 v1

结论：**030仍为 unknown，024限定接受不得外推。** 两次触发都符合现有保守门；本轮没有确认能解释86715的内部生命周期缺陷，也没有证据把它认定为外部进程。确定存在的是归属观测缺口：最终命中的未知进程只保存PID，未保存其出生/父进程row和扫描时间，离线无法恢复这些事实。最小 close 元数据包装可改善下一次留证，但不能替030授予合格或追认归属。

## 对象与已读记录

审查人为 `/root/control_review`，同模型家族独立工程上下文。定向阅读四指定模块及必要的 `inspection_model_client.py` 生命周期调用；五文件当前SHA均精确匹配原pilot的104source envelope（manifest SHA `abd4b12aba595746e9cc97eb44ff359fc64da1fa79c64eb160e720b0dc873560`）。完整源SHA和原记录SHA保存在 `source-binding.json`，源码及两份tree的副本在本轮 evidence。未扫描完整R、未重跑validator/physics/model或执行系统进程调查。

| 原order | 已登记owned | remaining_owned | 最终未归属PID | 持久化结果 |
| --- | --- | --- | --- | --- |
| 024/B3-M1/repeat2 | 21710/21712/21713/21757/21768 | [] | 26337 | unknown_process_cleanup / Y=null / retry=false |
| 030/B4-M1/repeat1 | 81848/81850/81851/81968/81979 | [] | 86715 | unknown_process_cleanup / Y=null / retry=false |

原030 tree SHA=`8821c585f5c2fb6477f0cb3391b2bcfe4276160b9dcf9dec096c294511e69f58`，terminal SHA=`3cf398b2674d81c2217b1b823fb765886aacf3bf9426634c74568ae836774831`。其原记录显示return0/run completed、clock最大0.009984750002331566、物理failure_reasons=[]，冻结Y=0.3436623024638006、H/E=1、S=0.037037037037037035、D=0；meter证据24calls/173756tokens。这里只核对已存字段，不声称重新离线计算144000步clock、分数或native用量。原rawY不是已接纳SDK Y。

Root报告发现时86715已不存在，并确认supervisor exec99187 exit1；这些后续说明不提供86715过去的parent/birth身份。本轮未独立取得该PID存活时的系统记录，未把root活动方式、没有Python进度读者或此后安静当作归属证明。

## gate实际范围与确定的观测边界

`inspection_process_cleanup.py:106–118` 在Popen前取得全局baseline，再把直接child和同次扫描可见后代登记。120–148行的循环通过已登记且born相同的parent扩展owned；每次库存读取后才等待至多0.1s，所以不是内核事件流，也不保证严格100ms一次。

Darwin热路径只遍历全部已知owned及其当前children（`inspection_darwin_inventory.py:94–135`）。已登记后reparent的节点仍可被查询；从未被采到、已失去可见祖先链的进程不会凭空成为owned。native检查失败走原全局ps fallback（cleanup:21–36），退出阶段仍走全局扫描（168–175），并未因为native只查森林而取消未知孤儿门。native内部比较微秒出生字段（101行），输出到旧接口/归档的born字符串仍只有秒（122–124行）；本报告不虚构更高精度历史身份。

close对匹配born的已登记live进程执行原SIGTERM/SIGKILL清理，最后把“未owned、相对baseline新identity、同uid、PPID1、comm包含python、非Z”的行列为unknown（150–179）。该谓词不要求已证明属于当前任务，注释101–103行明确可能保守阻断无关daemon；PGID也不作归属豁免。remaining_owned=[]证明的是本门采到并登记的集合，不能证明全主机无其他后代。

**可证的留证不足**：172–178行已有用于判断的完整row，却只归档PID列表；baseline、各次扫描边界、unknown row、首次/末次观察与parent变化均未持久化。两份tree实际也只有该PID列表。因此现在无法知道86715的born/comm/原父链，无法区别外部新orphan与扫描间脱离森林的内部短生命周期child。这是确定的采样/存档能力边界，**不是已复现的内部泄漏根因**。没有完整row也不能从PID数值顺序推断准确出生时间。

## 内部生命周期的有限检查

runner使用spawn multiprocessing队列/worker，并另开model service（`run_inspection_llm_development.py:105–130`）；正常路径先join物理worker并检查退出，再finish service（202–207）。finally保留worker/service失败（210–236）。`inspection_model_client.py:35–47,106–129` 将service置独立session，但仍是子进程；正常finish关闭stdin、wait、join reader并拒绝非零退出或未收尾请求。线程daemon不是独立Python PID，不能仅凭两个PID模式把未知PID称作reader、resource_tracker或MLX child。相同五节点拓扑不能证明其余未知PID的来源。

外层在tree.wait后必经close，并先归档tree/return再计量和评分（`run_inspection_llm_episode.py:160–185`）；故后面的grader/native-settle进程不可能倒过来导致这次已经采到的86715。这个时序排除仅指**本次之后**启动的操作，不排除同机同时存在的其他来源。224–239行即使物理/模型证据可重核，也将cleanup unknown强制设Y=None、retry=false并抛UnresolvedEpisode；缓存路径107–120行继续拒绝。未发现可据现记录确定的漏join/漏close因果链；也不能由无manifest错误证明没有短命未登记child。

## 最小前瞻方案：只记录原close的既有扫描

Root提出的最小方案在边界上可行：同一supervisor进程内、仅原close调用期间临时包装同模块 `_inventory`，每次原函数只调用一次，保存它本来返回的row；不新增ps/libproc扫描、不改baseline/owned、原信号/等待/unknown谓词及原result。close退出后再写独立logs sidecar，避免进入已哈希attempt、model-operation或resource目录。这样不增加observer Python进程，也不把新的全表轮询带回在飞热路径。**本轮只审方案，尚无适配器实现或测试可认证。**

需要保留的最小契约：

1. 保存episode/plan/reservation、原root identity、原baseline同uid Python元数据、owned_births、原始result或原异常；各捕获扫描记录顺序、UTC和monotonic起止、原row字段及精度，不采集命令参数/凭据。最终unknown完整row必须来自原判断使用的同一返回库存，先保留该row，不能稍后只查询已消失PID来替代。
2. 不能硬编码恰好“两阶段两张表”。close初始observe通常走native而不经此hook，也可能fallback；owned-live循环可产生额外ps读取。按实际scan序号留证；若给SIG阶段标签，须有真实可观测调用点依据，不能仅猜序号。该最小hook不声称记录全部native hot观察。
3. 串行/非重入与线程归属检查；只截取本次close所触发的扫描。无论原close正常或抛错，finally先恢复原函数；不调用第二次close，不修改任何原dict/result，不吞掉原异常。sidecar失败独立标为观察不完整，不能覆盖原tree/result或成为成功依据，后续effect不得把缺失观察冒称闭合。
4. 常规本次路径close发生在runner退出后；timeout/attach异常路径可能尚在清理在飞worker，不能笼统声称所有情况下都已结束物理。额外sidecar I/O放在原close结束后，仍记录观察开销/错误并保留该限制。
5. 该sidecar能解决“命中row没留下”，不能恢复扫描间fork/reparent的父链。若后续确需完整生命周期归属，则需要另行验证可用的进程出生/退出事件来源或明确登记producer；轮询更频繁也不能声称无遗漏。该升级不是本轮必须实施或已可用的能力。

原104磁盘源码可以保持不动，但临时包装改变实际执行消费者并增加元数据开销，因此**仍须将新adapter、入口和sidecar格式单独source-freeze、独立测试并声明provenance**，不能称旧104已经包含该仪器。直接修改cleanup/provider/runner，或改变采样、身份、等待/宽限、信号和unknown规则，则必须冻结新的相关源码版本并评估资格delta。外部纯观察也不能用来无证据排除进程或让030合格；024 reviewer JSON仅绑定其精确proposal。

本轮未实现observer/continuation，未导入运行模型或物理、未运行测试/新资源/native动作，未发送任何信号（包括signal0探测）。所有原unknown、回执、ZIP、SDK与104源保持原样；完成后停止本轮R写入。
