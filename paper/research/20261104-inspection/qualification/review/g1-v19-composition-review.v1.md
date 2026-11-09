# v19退休采样guard与G1证据复用独立审查 v1

日期：2026-11-05。同模型家族独立上下文工程Verify；不是跨模型或领域人工审查。仅写本报告，不改v18报告、manifest、source归档或既有结果。不重跑40任务、500/600秒physics或全套测试。

## 结论

**支持当前v19继承v18的G1控制资格组成证据，无需为这一处guard重新执行原40个events=False任务。** 这是对同一组已执行40份记录的行为等价复用，不是40个新重复，也不把旧v18的source/contract SHA改为v19。

基础100份的三consumer/G1对象/依赖/初始化等价条件继续满足，可与上述40份组成v19开发期G1控制证据。新retired分支在真实events=True域确实有行为变化；其单个500秒退休fixture已执行并有界复核，但不能由此授予完整G2、真实异步1×/G3、G4、正式实验或人工领域认证。human_review_pending。

未发现本次唯一guard及上述限定复用的实质缺陷；未来改变retired写入途径、harness事件设置或其他执行输入时需重新审查，不提前认证。

## 绑定对象

路径相对paper/research/20261104-inspection；源码相对repo根。

| 对象 | SHA256 |
| --- | --- |
| planning/gate-contract.v19.json | c0af9d0c6f49f2f879313d7bea767a917f8fa6a80e96701009d63ef83dab9b74 |
| qualification/executor-full.v4/manifest.json（原v18） | ef8a9efc045d661acffe1963b7c4eb25834ccb156c35b159eaf5e9c9df01a747 |
| qualification/review/g1-v18-full-review.v1.md（已冻结） | b37fef5fbdce03a87b39a8097a48cc4bcae4277aba142645f07f9acf6de5e194 |
| qualification/g1-full.v4/manifest.json | b07166e6415a9f8cfb6f7d003f53dc356938807537f9ad4a7b451fce108d1b97 |
| envs/inspection/mission.py（v18） | 24bc2a332508640f4043398f649e180b9cae167b930d22e2108135651c855b75 |
| envs/inspection/mission.py（v19） | 715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7 |
| eval/probe_inspection_retirement.py（v3 fixture源） | d60efec443723c4dfa61e6e9821aae333845ffe47c7d655c99394ae980bbc61d |
| qualification/protocol/retirement.v2.json | 99f1f0408dd48f950710d053c3f1d5cd99575dfc62400fd8f26164a90f393086 |
| qualification/protocol/retirement.v3.json | 73c0f9da170f3870a2ffb3a88e4daa553d5a9d9182f942ffc3b1d0936ce36452 |

v3的17份source_snapshots逐一验证其source_hashes并与实际当前文件字节相符（包含新mission、probe及envs初始化）；61项依赖逐项重hash匹配，且依赖集合/值与原v18相同。sources_unchanged、dependencies_unchanged均true。本报告使用这些明确快照与已冻结v18作比较，不把当前文件名相同视作历史身份相同。

## 唯一代码差异及events=False域等价性

对原v18 manifest全部15个source逐份比较，14份完全不变，只有mission.py:168的以下一行：

```diff
- if self.flight.exit_active[drone_id]:
+ if self.flight.exit_active[drone_id] or drone_id in self.retired:
```

并用整个旧source精确替换这一行后与当前source全字符串相等验证，没有其他隐含生产差异。合同v19.G1、executor_qualification、reference_spec三个整个对象与原v18相等；新增G2.retirement_sensing描述及资格复用指针不改变原G1运行输入。v19.executor_control_qualification_basis准确指向原manifest SHA，并声明same40而非new40。

因果证明边界如下：

1. 同一冻结qualifier:159始终以`InspectionMission(...,events=False,...)`构造G1 case；纯计算规划在physics步之间完成、逻辑nexttick回包，未引入新的时钟输入。
2. Mission:49把retired初始化为空。完整源码中唯一增员路径是_events:113–114的`if self.events ...: self.retired.add(3)`；工具调用、actor和qualification fixture均不写该集合。events=False下集合始终为空。
3. 因此每次_sample中新增的成员测试恒false。原本exit_active为true仍立即跳过；为false时仍执行完全相同的采样、coverage、局部tracker、上传、报告及随机通道。没有新增/删除G1测量或修改其时刻，规划器和物理控制序列也不变。

这是限定G1 harness域内的语义等价证明，不是声称events=True行为相同、任意外部代码修改mission.retired仍等价，或set membership的墙钟开销为0。后者属于真实1×后续测量，不能从本次复用推断。

基础100 runner没有import Mission，三功能源control.py/flight.py/qualify_inspection_g1.py与原snapshot仍相同；其完整G1对象、61项外部依赖、数值库版本及无行为导入初始化条件已在紧接此前的v18结果审查重新核对。此次唯一mission差异不进入基础执行闭包。故100+40沿用同一组已执行证据，不额外增加科学样本数，不修改旧qualified或unchanged值。

## events=True的实际退休边界：短证据复核

_events在step入口及物理步之后、_sample之前调用（mission.py:308–315），因此事件恰在360秒生效后，本tick感知就能看到retired。新guard位于目标、coverage、visible-opportunity及background采样前（:163–187及其后循环），所以不仅停止目标检测，也停止其他新任务采样。_deliver:265–286没有新增retired丢弃条件：退出前已获取的不可变包仍按原ready/online条件补传，未因退休倒扣历史数据。

本次没有重跑500秒。对已有retirement.v3做独立离线复核：

- 初始epoch0、实际结束500秒，world/flight/tracking均120,000步；解压17列误差行并核对raw SHA、字节数、有限值和严格240Hz时间格，独立重计XY最大4.090187918m、10m违规0、Z interval违规0，与保存值完全相同。
- 18项checks从before/after/job/audit/raw记录独立重新构造，全部与保存值一致且true。旧survey在359.991666667秒仍running，360秒被cancelled；当tick program hash不变，位置位移约.142m且均介于.01与27/120m，支持逻辑抢占未瞬移。exit360.5秒开始物理采用，441.883333333秒完成真实20秒dwell。
- 从原始business审计重计机3在t≥360的新target observations、coverage(measured_at)、visible opportunities、background trials，四类均0。并非仅抄摘要字段。
- 原始delivered中仍有且仅有一条退休前后跨界旧包`obs-3-719-0`，measured_at359.5、arrived_at360；estimate及测量时间与sensor_audit相等。该包质量不因补传而升级，补传也不等于新任务采样。
- v2/v3全部5000组四机位置数组逐元素**精确相同**，两份解压后的240Hz误差bytes也精确相等，raw SHA均`f91bf44aa607cdecb7598cc60e29341dabd3cbb52ae215db5978c669f0e8b6a3`。这个seed0控制fixture显示guard未改变其飞行，不能扩大为所有事件场景/真实时延的结果。
- hard无违例、seam失败0、基础设施错误空，四机最终dwell均≥41.333333秒，剩余range均≥9493.939960m。退休后新work和旧version cancel均拒绝；这些只覆盖本fixture的外生退休/补传边界，未宣布G2其他项目完成。

## 本次执行的短检查及限制

运行当前针对测试：`.venv/bin/python -m pytest -q tests/test_inspection_protocol.py::test_retired_sensor_stops_new_acquisitions_but_preserves_pre_event_upload`，结果1 passed in0.76s。测试:408–419明确是边界fixture，直接设置retired后检查四类采样并补传359.5秒旧包；没有把它冒称实际飞到360秒。真实时间连续性来自上面的已执行500秒v3记录。

离线检查脚本`/tmp/check_v19_control_composition.py`，SHA `9799265894be85e2a221124f16bd4d926e5c39d3b810becb6bc8253243c8a5b4`，退出0；摘要`/tmp/check_v19_control_composition_results.json`。脚本不启动physics、不调用生产recount，直接解压和重计，并核对合同、source和18项记录条件。未重跑全套测试；原168项日志属于v18前述验证，不替代新guard或实际退休证据。

reviewed by inherited model configuration, independent context, same model family

- v19可条件复用已审核的基础100和v18的events=False40份控制证据，保持原数据、合同和source身份，不重复计数、不新建伪重复矩阵。
- 对新退休分支，有一个真实从0到500秒的受控fixture和本次短guard检查；仍需完整G2及真实异步1×/G3的各自证据。此复核不声称整个事件任务、科学评分或正式研究通过。
- 原始误差trace可离线重计且v2/v3误差bytes相等，但不是完整physical-state replay。代码或retired可达性域再变更时本次等价结论不自动延伸。
- 领域人工复核仍pending；没有付费、正式test/holdout或投稿授权从此推断。

报告完成后冻结；后续版本另立证据。
