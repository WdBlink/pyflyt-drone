# Pilot v4 cleanup unknown：现有恢复契约与最小候选审查

结论：现有生产 API 没有可直接解锁 slot024 的 cleanup-reconciliation 入口。原 unknown 应保留，不能盲跑 attempt01、改terminal/tree、换前缀绕开cap或再跑整108。root提出的“唯一slot追加late-cleanup disposition + 独立review/native finding + 显式source-bound development continuation wrapper”可作为最小候选，未见与原完整分母、两attempt、sole ledger的必然矛盾；实现与接受条件如下，尚不能据本报告直接运行。

本审查只读源码/实际小记录及root已保存的观察，不改104生产源、原plan/terminal/tree/ZIP/SDK结果或.paper，不调用native/kill、不启动model/physics/resource。无测试/飞行或完整actor/meter离线重计；root正在做的PID/native调查不在本报告中冒充已独立完成。

## 已读事实与边界

order24 `dev-L3-01-B3-M1-2/attempt-00`：

- 原terminal SHA `86f0189e000815abc250df4092bc264bbb7fa79dcca65a032edec88d68c3f6cd`；status unknown_process_cleanup，return0，Y=None，retry_allowed=false。物理验证记录failure_reasons=[]、maxlag .01017320800019661；world header实际为600s/144000步。
- tree SHA `ae19d0f443f3093bcee565e45214e23ff63e3f366b71507fc588c7ff64fa6291`：root21710，owned五身份21710/21712/21713/21757/21768，remaining_owned=[]，unclaimed_new_python_orphans=[26337]。原tree只给未认领PID列表，没有其born/uid/command/父链记录；不能把它推断成自家或无关。
- terminal仍保留非空LLM evidence，model meter usage25calls/166000tokens、25 causal joins；这是原记录，不是本轮重新验收。原settle-owner ACK SHA `a2913d8ee138f1d924516755f7941548900295785ca28837113203d394f23575` 为known；当前native1045calls/6778111tokens、reserved0/unknown[]是root的最新核对，本轮未读整ledger。
- later observation SHA `1749652f84ae310724768e564a447c8ea51f63ddf2e809009562b030eba33e98`，UTC22:23:59.316745：原owned live为空、当前同uid Python orphan为空、26337当前不存在。其role明确“当前观察，非原ownership或episode admission”，本报告沿用这个正确限制。
- 旧SDK完整保留18/90/48：classic3completed+15stopped；LLM21completed+1failed+68stopped；outer48全stopped，三个envelope execution_status均failed/inconclusive。24个inner callback完成，不等于outer已聚合或108已完成；未知失败不是科学policy Y=0。

plan的文件bytes SHA为 `2548b2670207ee623bcd38af39e121b95b2dec0045c861e90cfe9b623ab97c19`，原canonical plan digest为 `0ab1224d422a4bee4e3934b22f34a2a36d8bfa21f1886004b8ba54be8a38b4ae`；恢复绑定不得混用两种摘要算法。全部小记录SHA与源SHA见独立observed-originals/source-refs。原archive/usage/settle的SHA只按terminal原绑定列出，本轮未重hash整个ZIP或重验全部tokens。

## 当前源码为什么阻断

`inspection_process_cleanup.py:98–104` 明确未登记的同uid新Python orphan是不确定性，不是随意kill的权限；也可能保守阻断无关daemon。`:172–179`在最后全局inventory中发现它就输出unknown，即使known-owned已空。当前quiet只能支持后来的关闭观察，不能让原close历史变成closed，更不能证明26337原来无关。

生产 `run_inspection_llm_episode.py:224–225` 在cleanup非closed时强制Y=None、retry=false；`:227–239`写unknown terminal并raise。缓存入口`:107–109`直接拒绝unknown/非retryable；后面`:118`仍要求原process-tree.status=closed，`:123–126`比较原payload/model bindings。只改terminal.status会在树证明处拒绝，再改tree会破坏payload/ZIP/native receipt绑定。原测试 `test_normal_exit_cannot_admit_a_live_new_python_orphan` 也专门验证手改terminal不能绕过。

当前生产API没有recovery_ref或read_only参数，staged formal reader尚未安装，不能把两者混作已有能力。已有model/flight validators可以作为新resolver的只读重验组件，但不能由旧execute函数自动越过其unknown门。

## 现有 native/recording 入口能做什么

- actual d809 `Controller.reconcile:1342`只认state.attempts里running/uncertain的原生WorkOrder action，绑定order_sha256/pending_action；此LLM目录attempt不是该对象。不能拿slot路径当native attempt ID来让它改外部terminal。
- `budget-settle:581–602`管理资源，不接受scored admission；已known用量不能重写/再次扣费。cleanup unknown与model资源known可以同时真实成立。不能refund、改成not_started、重复reserve25calls或造unreserved结算旁路。
- 当前driver要求新的out目录；provided record_matrix已把旧abort envelope持久化成完整failed/stopped分母，没有安全“接着写原SDK”的接口。新SDK视图可以按原计划重读缓存，但必须保留原abort，而非覆盖原results。
- 普通native round可以把追加recovery证据/源码/review绑定为实际finding；它记录新的工程判断，不代替模型/物理重验，也不使旧缓存API自动认识恢复。

## root最小候选：兼容性及必要条件

候选可限定为本**唯一slot024、原attempt00**，不改共享formal readers、旧104源或科学方法。它新增的是后续知识的disposition与显式consumer，不是历史terminal修补。

1. **追加不可变receipt。** `late_cleanup_disposition`需绑定pilot/slot/condition/repeat/sampler/reservation、canonical+bytes plan摘要、原terminal/tree/intent/started/return/stdout、原run/world/actor/operation、原archive/usage/native ACK与原SDK/preparation/schedule身份，以及新process observation、resolver/continuation source和独立review。原status/Y=None保留；新判断明确叫late cleanup reconciled，不叫original cleanup closed或unrelated process proven。
2. **把当前quiet用作晚到关闭条件，不单独当admission。** resolver用fresh born-safe观察复核原owned与26337现状、同uid活Python orphan范围；静态observation只能作为证据锚，不能永久permit。原26337身份缺口继续声明。当前无活进程、全链字节仍闭合、完整actor/meter/score均重验和native known相容，可支持此受限late disposition；quiet本身不足以产生Y。不能仅凭PID现在消失就回写历史ownership，也不按裸PID发信号或通用白名单。
3. **完整重验而非只看物理clock。** 必须调用原source-bound physical/grader与LLM source/model/seed/causal/parser/meter验证，确认原Y及完整evidence未变，原25/166000与当前sole-owner known记录相同、无未计量/unknown/额外effect。原源码在cleanup失败时会覆盖admission_error文字，不能从这个字符串推定没有其他错；本例非空25join evidence是有利记录，仍需fresh验证。offline冻结evaluator子进程可作为只读重计，flight/service/native预算命令必须禁止；不能把“禁止新flight”误实现成跳过grader。
4. **先独立review，再实际native应用。** 新receipt、actual evidence、独立review及reader源码一起绑定唯一native finding，检查现行/未撤销。记录应针对恢复契约，而不是借native reconcile或budget状态替换外部episode状态。无法完成任一重验时保留unknown，无新reservation/attempt。
5. **只读resolver最窄返回面。** 输入精确old plan/terminal/attempt与显式reconciliation Artifact；没有spawn/command_factory/资源写入接口。匹配**整套**身份和原SHA，而非只看case_id/order_index；错误hash、别的attempt/slot、新unknown仍委托旧API并拒绝。它可返回“原Y，经reconciliation缓存复用”及原unknown terminal SHA+新disposition/review/native finding引用，不能只返回浮点Y使旧terminal看似completed。独立后验loader也必须显式理解这条新证据链；旧loader仍拒绝是正确行为。
6. **continuation是新SDK视图，数据来源透明。** wrapper仅替换原pilot模块中该execute调用的委派入口，源码必须真实冻结/核对并在每次effect前后维持；不是未登记的交互式monkeypatch。原104+61+10、原selection、case/root/repeat、schedule和sampler、prefix及原plan保持；新wrapper/resolver额外source单独manifest和SDK伴随文件，不能冒称原104已包含它。先重验24普通缓存、再重验1个reconciled缓存；只从25开始原定新effect。新的raw evidence需区分cache_reused与reconciled来源、旧physics/source与新录制器source，不虚称25个新飞行。
7. **不减少分母、不扩大重试。** 新envelope仍18classic/90LLM/48outer和108同序；原abort完整保留。预检025以后无未知已发生effect；如果有，先按同样规则对账，不能blindly从25再起。未知attempt00当前没有自动attempt01权利，清理恢复也不应把完整600s再跑一遍；更不能换prefix/output当“新第一次”绕两attempt/第三retry。future slots仍全委托原API及原100ms/两attempt/unknown门。

在这些条件下，该方案没有改模型prompt、physics、score、科学N或资源事实；它是明确标注的工程恢复，不需要重跑已闭合24+原024或整108。当前fresh observation并不等于这些条件已全部完成；尤其fresh grader/causal/meter/native匹配、actual review/native finding及新consumer实现尚需实际证据。

## 应由候选软件证明的最小负例

本报告不写/运行测试。实施前后至少检查：错old plan/terminal/source、漏掉26337或原owned身份、live/新unknown、伪review/native撤销、model计量/actor/原ZIP漂移、其他slot/attempt误匹配，全部fail-closed且不spawn/reserve。成功复用原Y不得改原文件；repeat调用幂等；同一个旧slot不能被接纳两次；wrapper额外source改变应拒绝。新SDK必须完整保留分母并清楚标示reconciliation，原SDK字节不变。这是候选实现接受条件，不是本轮已过测试。

## 裁定

现有API：没有可直接用来恢复此slot的合法既有入口。root提出的新development-only追加式契约：结构上可行，需上述精确source-bound条件、独立实现复核与native证据，不能直接以当前quiet快照批准continuation。本审查没有实现、修改或执行恢复，不签发G4/main/human接受。

报告和binding保存后停止R写入，不继续CPU调查。
