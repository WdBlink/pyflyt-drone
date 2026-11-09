# 同公开级别参考 pair：实际证据独立复核 v1

结论：这一个固定开发根、B0、同初始公开 L3 的 event/clean pair，其完成回执、输入/源码绑定、完整原评分、共需求日历重算及两份执行重放回执均通过本次只读复核。未发现需要否定这对已存证据的具体绑定/算分错误。事件侧较低成绩和 SR=0 必须原样保留；这不是总体事件因果效果、主试结果、模型能力比较或 G4/main 放行。

本 Agent 为同模型家族独立上下文工程审查，只写本报告与 `review/evidence/same-level-reference-pair-actual-v1/`；没有修改实现、原始 operation、历史输入、回执或其他作者改动。没有运行物理、replay、模型、main 或新总体生成。只顺序读取已完成记录，低优先级执行两次纯冻结评分和一次纯共日历评分；该三个评分进程及最终两探针均 exit=0，证据保留可复核。

| 核心输入 | 实际核对的 SHA256 |
| --- | --- |
| qualification/dev-same-level-event-pair.v2/pair.json | 6ff386e880379d9b88e3382bb6af9714e1365c0eb30e187f5bc7a6819253ac84 |
| binding-final.json | 4a240b353b796fa0433edb3c04f6c0cf5f0b0b6b923b34129f6c86506c295508 |
| complete-reference-pair.zip | 266fcdc253c5f7e6177109951e95d3c77103f6c302122d966a77f59a3a59c1b5 |
| execution-replayer.py | 72a6898fc9280215b9a60fd74b680dc092f601124cad9f403bd8f420c270486f |
| 冻结 experiment/evaluator.v2.py | de826d2488b5a8ad366ccec9ead8b81bebf6b794a533a38ac55338916848611b |
| 冻结 evaluator-source-manifest.v2.json | ecc2069fc8795adb9096173ab2d859b6540a94309ecf99310755cb7b3177bf80 |

最终 binding 的 37 项文件逐项重算一致。zip 的 20 个唯一成员共 55,449,082 未压缩字节，无绝对路径、`..`、symlink 或重复成员，成员集合等于两侧原 operation 的全部实际文件；流式重算每成员 SHA 后，逐项与原文件一致。该 zip 保存原 operation，replayer/两份 replay 输出/共日历结果则由 zip 外 final binding 分别绑定，不能说它们都包含在 zip 内。

两侧均只有 `attempt-00`，不存在第二次实际尝试；terminal 均 completed、returncode=0、failure_reasons=[]、retry_allowed=false，未消耗唯一重试。plan 的语义 hash（`json.dumps(sort_keys=True, allow_nan=False)`，默认空格）与 intent、started、terminal 一致；intent 的 CLI 明确指定对应冻结 case、B0、realtime、duration600。started 记录在 intent 之后；stdout 和 terminal.record_hashes 的六个文件逐项匹配。PID/时间戳是本机回执，不能外推为独立硬件 attestation。

17 个飞行源、61 个依赖的两侧字典相同。逐条重算 run 中嵌入的 17 源文本，与 plan 和 `flight-source/` 原字节一致；run 的 source/dependency/private guard 均 true，并重算当前可见 61 依赖均匹配保存 hash。两侧 supervisor/adapter 的计划绑定仍分别是 `470b639a…` / `59f88d5a…`，不把它们混入飞行 17 源计数。重要飞行源：population `bb7098554475a65d4717077b7704bcf5902cd0fc674c1a9a95a51a7e066a5ff7`、scenarios `9159ba44790d6d315074a824e52adeebd96468ab3032c6f0f07fdb755ce2753f`、runner `4c280adab488f9c81d68265dd9930d9775dbb6372b36d1d776d9350f9f36f3ca`；其余源码均由已逐项校验的 final binding 固定，不跟随 live 后续修改。

| 实际量 | event | clean |
| --- | --- | --- |
| 原始 world SHA | e12cf0d6c5aca4f34966e92e9f7c6361542f20af8c9bceff0a054770c5c61523 | 2573d6ce28c5be0703b7a0d992a7baea763fc9f513bf78258c8f3ac1bfe3c7d4 |
| 仿真/物理步 | 600s / 144000 | 600s / 144000 |
| max clock lag | 0.04121920765999221s | 0.0456188750686124s |
| C | 1103/1110 = 0.9936936936936936 | 1101/1110 = 0.9918918918918919 |
| S | 5/5 = 1 | 3/3 = 1 |
| D | 45/209 = 0.215311004784689 | 80/120 = 0.6666666666666666 |
| Q=Y（H=E=1） | 0.7363348994927942 | 0.886186186186186 |
| SR | 0 | 1 |
| 实际顶层 RPC / report journal 条目 | 98 / 559 | 125 / 981 |
| 封存 report 名额 | 7/10 | 6/10 |
| job 终态 cancelled / succeeded / failed | 31 / 30 / 4 | 46 / 29 / 2 |

两侧 clock 超过 .1s 的 tick 数为 0；均为 240Hz 物理、120Hz 控制，10Hz position trace 各 6000 行。job 列表均为终态，没有把 failed/cancelled 作业改称成功。真实 observe_track 顶层调用 event 4 次、clean 5 次；作业局部终态、取得观测和评分 D 是不同判据，不能据一个 job succeeded 宣称持续180s合格，也不能把低 D 改成操作故障 Y=0 或删除该样本。

完整分母复核：11 个实体为 3 个必需静态、2 个必需动态、6 个 normal 干扰物。覆盖网格由冻结 evaluator 按原 ROI/20m lattice 重建为全部 1110 格；未发现的目标及无证据格仍留在分母。两动态出现时刻为 166s、122s，各固定 60×3s；共 120 格均保留，无重开窗口或删去低质格。事件 priority_at=189，需求权重按既有冻结规则取 bin STARTS：truth-001 的187–190格仍权重1、下一190–193格权重2；truth-009 的188–191格仍1、下一191–194格为2。两个目标总权重112、97，合计209；不是把跨格部分积分另换规则。clean 无 priority，两个分母各60，总120。静态最终权重 event 为1+1+3=5，clean 为1+1+1=3。独立按原 score 中所有实体/格重算 S、D、Q、H、E、Y、SR 与原值逐项一致，再由新进程冻结 evaluator 重建全部报告/证据链，完整 score dict 精确相等（不限于八个摘要数值）。

H 对应原144000次安全审计 violations={}。E 由原实际 range hook 的四机剩余航程及实际 home dwell 判定：event 最小剩余7796.452270764825m、最小 dwell64.95833333335642s；clean 最小7799.485770148249m、最小63.80000000002381s，均满足1000m reserve及20s。离线 evaluator 复用已存实际剩余里程；本审查没有从不存在的完整240Hz physical-state轨迹补造里程，也未重新做控制资格。

输入根仍为已审的同级显式 sibling：parent 语义 SHA `be9e0b050f0192e4bfdeddfbb054d5c5c99625b904313555bdaba9eac21b8983`；clean `99b367178b3ad36b65eb037e035b83fd1e4ae4682b57ac7123982454787fb2fc`。clean 完整 parent_case 与 event 原 private_case 相等，parentSHA一致；只有外层显式 schema、九事件字段及证明元数据不同，level 均 L3，seed/身份/实体轨迹/ROI/wind保持。private 文件字节、套件 manifest 和 plan/run/world 的对应 SHA 全部匹配。t0卡 proof 已在前轮独立输入审查验证并由本次 final binding 固定，两侧原始 t=.1 trace 也完全相等；本轮没有再构造真实 world。旧 L3→L2 pair 未替代此输入，旧结果也未覆盖。

实际事件边界保留原日历：priority189；机2断链[299,335)；机3相机降质[304,348)；机1退休372。public_events 明确记录 priority 和退休两项，link/camera 属日历驱动状态而不是额外四条 public_events 记录，不能将该数组长度当全部事件数。

断链证据：机2在[299,335)采集的63条观测，原 ID/测量时刻均能 join 原 sensor_audit，实际 arrival 全为335；其中62条是非背景实体（含 normal 干扰物），46条是必需动态目标，不把干扰物叫作任务目标。机2没有一条 arrival 落在断链半开区间。335时刻26条 report journal 路由逐条 join acquisition 的 observe job/report_id/candidate/原测量位置与时刻；其中25条来自机2补传，另一条是机3正常上传。相关 observe job 在这些上传时仍 active，不把此恢复段冒称“任意 cancel 后路由仍合法”的实证。

退休证据：机1在372后 sensor_audit、sensor_opportunities、background_trials、coverage 新测量四通道均为0。371.5采集的两旧包仍在372合法到达，其中一个动态包 `obs-1-743-0` 沿 `report-candidate-1-00009` 在372 accepted，原 measured_at=371.5保留。旧 observe `job-0057` version19 在372因 safe exit cancelled；外生 exit `job-0059` version20 在372创建、372.5启动、455.7416666666667因 actual home-circle20s dwell succeeded。冻结 mission.step 先执行当刻事件、再采样/交付，因此这一证据可支持“同钟外生退休取消后，先前已采包原路由仍合法”的窄正例。两侧均没有 handoff_observation 顶层调用；任意 actor cancel、handoff成功/失败/跨离线接收机等并未因此全部验证。

共需求日历：独立用已校验的冻结 evaluator closure、原 clean 观测/报告及 event 的 priority 日历重建，不运行物理，也不改原 score。原 clean score 先精确重建，再换需求日历，得到 C=.9918918918918919、S=1、D=.6746411483253588、Y=.8888443467390835、H=E=SR=1，与保存的完整共日历 score 相等。event Y 减该值为 `-0.15250944724628934`。它是 n=1、一个根、一个策略下的描述性配对差，不是总体因果效应、显著性或额外策略样本；原 clean Y=.886186…仍保留，不能用共日历数替写主 score。

两份实际执行重放回执已完整读到并按原输入重新绑定：event replay SHA `fba9a5d9793e865fe14a22ea0fd63c2bdb1b41c4128fff28d858141a388dc77a`，clean `4ffcac51ffd14f82088e134932de4486c2853fe2feb7afbae5fe0327768114be`。run/world SHA精确指向上述原物理记录；98/98及125/125顶层RPC，response_discrepancies=[]，全部25字段逐项 exact=true，包括完整score、安全/10Hz trace、覆盖格/证据、sensor/arrival、作业与transition、tool/report、隐藏实体/public events、内部execution_audit、tracking聚合与error trace、机会/背景记录、case/private_case/adapter。replayer 对内部 planner 按已录逻辑tick重新计算纯计划并释放原响应，不重测原wall延迟。receipt 的 source_root=null，不能说当次使用了`--source-root`归档隔离启动；producer 源码在运行时调用 verify_source_bundle，且本轮已将嵌入源与归档17源字节独立对照。shared reader 修复不在原飞行17源中，不回填飞行 manifest。

这25组包含 `execution_tracking_trace` 的原编码误差/z/允许区间数据，但不是完整240Hz飞机位置/速度状态。两次 replay 是对已执行工具/计划历史的确定性复核，既不是新策略rollout，也不是新的真实1×时延、硬件或统计样本。本 Agent 未再次执行 replay。

`world.events` 两边仍为 true，`run.events` 则为 None：这是 case.level=L3 的旧元数据，不是激活开关。运行、评分与干预判定必须使用固定 schema/parent/九字段、原日历和实际审计。前轮已指出的旧固定机2/固定300–345分析分组不能直接用于此随机日历；不据该布尔值给 clean 错贴事件激活标签。

实际命令（cwd 为仓库根）：

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B paper/research/20261104-inspection/review/evidence/same-level-reference-pair-actual-v1/audit.py > paper/research/20261104-inspection/review/evidence/same-level-reference-pair-actual-v1/audit.stdout.log 2> paper/research/20261104-inspection/review/evidence/same-level-reference-pair-actual-v1/audit.stderr.log
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 nice -n 10 .venv/bin/python -B paper/research/20261104-inspection/review/evidence/same-level-reference-pair-actual-v1/link_route_probe.py > paper/research/20261104-inspection/review/evidence/same-level-reference-pair-actual-v1/link-route.stdout.log 2> paper/research/20261104-inspection/review/evidence/same-level-reference-pair-actual-v1/link-route.stderr.log
```

audit 内部设置 nice10并顺序调用两个 frozen evaluator CLI及一个 pure common_recount 子进程，精确命令向量保存 results.json；全部 exit=0。三个纯评分进程在导入阶段打印相同的42字节 pybullet build-time banner，未构造 world/飞机或调用 step；原 stderr 保留。末尾审查 binding helper 最初错误假设 stderr 必为空，因此触发一次断言；核对 banner 后仅修正该审查假设，没有重跑原物理/评分或改动被审源。原记录未写入，探针和完整结果均在本报告 evidence 目录。没有运行108集pilot、测试新方法、追加模型调用或将该pair升格为主试/G4。
