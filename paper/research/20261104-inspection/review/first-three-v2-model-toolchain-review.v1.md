# 首三个 v2 模型工具链：限定实际因果审查 v1

结论：这三次有效 callback 的公开输入→真实服务回复→原 parser→入队→worker→报告/评分链，在已存日志范围语义一致，未发现合法命令被实现丢弃、改写，或本次失败所需的公开调用前提未告知的具体反例。相同的低分不足以判定平台故障；这里存在不同的标识/版本选择错误、异步时效失效、格式失败和报告证据失效，不能统一归因为一种“模板”问题。本报告不优化策略/提示、不建议 few-shot/预算/阈值，不授予 G4/main、总体能力结论或人类领域认证。

独立上下文、同模型家族工程复核。只写本报告和 `review/evidence/first-three-v2-model-toolchain-v1/`。未改任一 pilot 冻结执行源、原日志或 owner；未调用模型、MLX、物理、policy replay。执行的仅是纯 JSON/hash 分析、原纯 wire parser 和 ReportLedger/静态质量函数；不实例化 mission/world。没有重跑失效尝试，也未把其费用或物理历史删除。

| 实际有效输入 | 请求数 | wire 解码/格式失败 | 入队=worker RPC | 接受的 inspect / report / exit | 工具拒绝 |
| --- | ---: | ---: | ---: | --- | --- |
| dev-L1-00-B3-M3-0/attempt-01 | 22 | 22 / 0 | 90 | 2 / 7 / 12 | stale version 60；existing report缺失2；duration超剩余7 |
| dev-L1-00-B3-M2-2/attempt-00 | 30 | 28 / 2 | 28 | 0 / 0 / 0 | stale card或unknown rule 5；unknown aircraft 23 |
| dev-L1-00-B3-M1-2/attempt-00 | 21 | 20 / 1 | 76 | 0 / 0 / 4 | duration超剩余72 |

三者有效 terminal 均 completed，C=`0.9940374787052811`、S=0、H=E=1、Y=`0.49701873935264057`；均600s/144000步且时钟在门内。完整静态分母均为4个必需目标，各最终质量为0；M3报告7次接受但仅4个不同永久名额，M2/M1均无报告，不把 ACK 或 inspect 成功等同 S>0。M3原 attempt-00 是完整物理但 clock lag=`0.11925233295187354`，terminal 正确保留 known_operational_failure、Y=null、唯一原因clock_lag_exceeded及允许固定重试；不能与有效 attempt-01 混算，也不能将其降为策略 Y=0。其原 world SHA及 clock 已另行核对。

实际源码绑定：三次 run/episode plan 的直接 source_hashes 为相同23项（是本地 run 的直接清单，不冒称此次重审全部 cohort63源）；逐条嵌入文本 SHA均匹配，run source/dependency/private guard 均 true。pilot63源冻结保持原样。本次保存并核对的关键原始源码快照如下：

| 原源码 | SHA256 |
| --- | --- |
| inspection_llm_actor.py | cd1bf019c637d447946cba30e1e2d8572c4ff90594de4b5b2d90f49cd65ccc7b |
| inspection_llm_protocol.py | b1c2f73c3894058105cea8b3b6088128fbe44fee29bda9854423ce3c0c3d19ed |
| run_inspection_llm_development.py | c3bed0403ca4a8e7f5d79c0db3df5da0aeb27e46e37423e94474447fbd4da552 |
| envs/inspection/mission.py | 715fce4a99325b6d26fbb1e9ffc20896603bfbd4d48396f0513a46f5ad14aec7 |
| envs/inspection/protocol.py | c32bdabb67bf8493ff5125d0af4fc289d1d0210db762016a9c56a96dd2f174a4 |
| envs/inspection/scoring.py | bc8a21947d340fce28c83c3d2313ad9a2e7711e6af07c1bd275f774009b5631a |
| envs/inspection/reference.py | 85f97b6cb632c7f178c20ad9f53097510cab1dca0cd56aec15a5fe222b9365c3 |

输入、服务与公开性逐段检查：

- 重算 plan→terminal 语义 hash、terminal中actor SHA、全部 payload_hashes 和 model_payload_hashes；对应 run/world/actor/service-completion 均匹配。完整具体文件 SHA保存在 results.json，不复制大段原始模型推理。
- 73次 request/response/decoded 三元日志全部按 ordinal 连接。记录的 messages hash按原 compact/ensure_ascii=False 算法相等；每次服务 durable request 的 messages、seed 与 actor请求完全相同；服务 durable response 的全部字段与 actor收到的 response相同，parent额外时延/ordinal除外。
- 对每个实际 user公共帧运行原 `serialize_public_frame`，均通过。逐条出现于帧的观察与原 delivered_observations 完整字段相等，测量/到达均不晚于该帧sim_time，没有发现私有truth/seed/未来证据倒灌。跨请求重复展示的观察项分别275/407/278条，不能当唯一新观测数。
- 公开 recent_tool_results 共120/176/120项，逐条找到其请求之前已发生的原 worker 工具调用/响应；不是伪造反馈，也没有把尚未执行结果提前提供。记忆/上下文是明确有界投影，不宣称每个历史 observation 都始终保留。
- 对每条真实 raw response 重运行**绑定的原 parser**，reply对象及错误原因与日志一致；whole fence移除计数 M3=22、M2=28、M1=18，不能把这三次表现归因于旧 strict parser 的 fence拒绝。有效decoded命令按原顺序与 `commands_enqueued_not_accepted` 精确相等，平铺后又与原 world.tool_calls 的 name/args逐项相等：总194条，没有改参数、补命令、补报告或丢合法命令。执行时刻均不早于入队共享进度、且小于600。

最少代表性时戳与分类：

| 运行及时间 | 已公开条件/原命令 | 实际响应与解释 |
| --- | --- | --- |
| M3，source0→worker16.958333 | 公共四机 job_version均0；原raw命令 expected_version=1 | stale version，返回version0。wire/runner没有把0改成1；这是原模型参数，不是平台off-by-one。后续拒绝反馈确实出现在公开上下文。 |
| M2，source0→4.783333；source40→44.35 | 卡version1、机0..3和stripe0..3可见；前者命令card9/rule static-5/stripe-5，后者drone5 | 分别 stale task card or unknown rule、unknown aircraft。原parser只校验工具字段形状，语义拒绝由worker执行，未把可用合法ID改坏。 |
| M1，source0→17.091667；source20→40.85 | duration分别600、580，其余原参数保留 | 两者在各自**输入快照**恰能容纳，物理等待后执行时已超过剩余时间，worker拒绝。全部72次duration拒绝在source时尚可容纳；不是72次固定600，也不能说它们初读时全非法。这是基于旧公开时钟用满剩余时间而在异步执行时失效。 |
| M3，source378→401.808333 | observe_track使用report-1，但当时没有先前accepted report-1 | existing report缺失。每次实际SYSTEM包含完整公共合同，明确要求已有accepted report_id，未发现调用前提隐藏。 |
| M3，source402→446.791667 | observe duration180 | source时尚可容纳，执行时失效；7次duration拒绝中2次属于此类、另5次在输入source时已不容纳。拒绝不是统一的模型格式错误。 |
| M3，377.616667 ACK→415 /434终态 | 两个inspect_candidate进入原worker | job-0002于415、job-0001于434真实succeeded，理由均为两次实际合格视图相隔20m。平台确实执行了inspect，未因最终S0抹去成功。 |

公开SYSTEM明确说明四机、600s、物理在推理期间继续推进、当前job/card version及工具拒绝需读取反馈；task_card又给出deadline和duration实际起算语义。duration的当前剩余时间约束在 `mission.py:384–388` 执行。本报告将 M1 归为实际时效失效，不伪称输入快照能告知未知未来耗时，也没有证据表明实现违反其已声明的异步时钟行为。M1后续duration数值随输入剩余时间变化，不能将这一观察压成“模型一直复读600”。

格式失败也保持原分类：M2 source20/580的回复在worker36.958333/595.058333处被原parser判为 `reply must be one strict JSON object`；M1 source577、worker590.541667为 `commands schema is invalid`。均独立复现相同错误，无命令入队、不免费自动重试，不把这些格式失败冒称基础设施故障。其余回复照常解码、分派，未因含code fence丢失合法命令。

M3报告链说明了“ACK成功但S=0”是可核对的协议语义。按原协议纯函数重建全部到达证据及7次report提交，逐条 response 与保存值完全相等，固定首次关联/永久名额保持；再用原静态质量函数对全部4个必需目标重算，与 world.score.static_entities 全字典相等且S=0。以下只引必要证据与时戳，不粘贴模型说明文字：

- 446.791667的report-1引用3个不在实际到地证据集中的ID，并包含quality<.8的已到达证据：provenance_valid=false、quality_valid=false。report-2声明measured_at400，却引用400.5/401的证据，也有低质项。这里“晚于400”是相对**模型自己申报时间**，不意味着 actor 获得未来信息；这些真实证据可以早于请求source402合法到达。
- 485.291667复用report-1/2并改成别的candidate，首次关联仍固定；另有混candidate、缺失ID和申报stamp之后的证据，不能因改报ID就重新关联。
- 548.108333的report-4所引质量均合格，但声明517而最大被引测量仅516.5，没有一条正好517，故provenance=false。579.416667再把report-4改为别的candidate，虽然时间/质量条件合格，identity_correct=false，不能消除固定首次关联。

这些条件均在每次实际SYSTEM中的 `PUBLIC_TOOL_CONTRACT`（wire source:21–43）明确给出：证据必须已到达、属于请求candidate、测量时刻不晚于申报stamp且至少一条恰等于stamp、全部quality>=.8；新ID永久占名额，首次association固定；ACK是存储响应而非隐藏正确性oracle。`ReportLedger.submit` source:101–126有对应的 causal/identity/quality 分开标志。7次ACK中并无一条同时满足完整有效得分条件；只占4/10名额，S0也不是容量耗尽。没有观察到一个合法有效report被实现丢弃的反例。

M2没有inspect/submit命令或接受作业，M1也没有inspect/submit命令；M3则有真实inspect成功但报告失效。这三种实际链不同。共享物理执行与传感器可以在缺少有效任务作业时产生coverage；高C不是LLM成功报告的证据。相同Y在此只是同C且四目标质量均0的算术结果，不能据三次表面相似表现认定唯一心理/架构原因，更不能改写原成绩。

实际轻量命令（cwd为仓库根）：

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B paper/research/20261104-inspection/review/evidence/first-three-v2-model-toolchain-v1/audit.py > paper/research/20261104-inspection/review/evidence/first-three-v2-model-toolchain-v1/audit.stdout.log 2> paper/research/20261104-inspection/review/evidence/first-three-v2-model-toolchain-v1/audit.stderr.log
```

脚本内部nice10，最终exit0、stderr空、passed=true；使用从原run提取的源码，不导入真实model/flight模块。另有只读JSON摘要/`rg`源码定位，没有运行测试集、调用生成或重飞。这里只审三条有效实际链及一个已知clock失效分类；未重新审全部模型/owner计费、未知外部恢复、全部工具场景或整个108集pilot。该局部通过不能扩大为G4/main通过，也不能作为方法改善或统计结论。
